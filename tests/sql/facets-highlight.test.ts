import { describe, expect, it, beforeEach } from "vitest";
import { Engine } from "../../src/engine.js";
import { extractQueryTerms, highlight } from "../../src/search/highlight.js";
import { standardAnalyzer } from "../../src/analysis/analyzer.js";

// ==================================================================
// Fixtures
// ==================================================================

async function createEngine(): Promise<Engine> {
  const e = new Engine();

  await e.sql(`
    CREATE TABLE articles (
      id SERIAL PRIMARY KEY,
      title TEXT NOT NULL,
      body TEXT,
      category TEXT,
      author TEXT,
      year INTEGER
    )
  `);
  await e.sql("CREATE INDEX idx_articles_gin ON articles USING gin (title, body)");

  await e.sql(`INSERT INTO articles (title, body, category, author, year) VALUES
    ('Introduction to Database Systems',
     'A database system provides efficient storage and retrieval of structured data. Modern database engines support SQL queries for data manipulation.',
     'databases', 'Alice', 2020),
    ('Information Retrieval Fundamentals',
     'Information retrieval is the science of searching for information in documents. Full-text search engines use inverted indexes for fast retrieval.',
     'search', 'Bob', 2021),
    ('Advanced Query Optimization',
     'Query optimization transforms SQL queries into efficient execution plans. The database optimizer uses cost-based methods to find the best plan.',
     'databases', 'Alice', 2022),
    ('Machine Learning for Search',
     'Machine learning techniques improve search relevance. Neural retrieval models learn to rank documents using deep learning architectures.',
     'search', 'Carol', 2023),
    ('Graph Database Design',
     'Graph databases store data as vertices and edges. They excel at traversal queries and relationship-heavy workloads.',
     'databases', 'Bob', 2021),
    ('Natural Language Processing',
     'NLP enables computers to understand human language. Text analysis and search applications benefit from NLP techniques.',
     'nlp', 'Carol', 2022)
  `);

  return e;
}

// ==================================================================
// Highlight: core utility
// ==================================================================

describe("Highlight utility", () => {
  it("highlights a single term", () => {
    const text = "The quick brown fox jumps over the lazy dog";
    const result = highlight(text, ["fox"]);
    expect(result).toBe("The quick brown <b>fox</b> jumps over the lazy dog");
  });

  it("highlights multiple terms", () => {
    const text = "The quick brown fox jumps over the lazy dog";
    const result = highlight(text, ["fox", "dog"]);
    expect(result).toBe("The quick brown <b>fox</b> jumps over the lazy <b>dog</b>");
  });

  it("highlights case-insensitively", () => {
    const text = "The Quick Brown Fox";
    const result = highlight(text, ["quick", "fox"]);
    expect(result).toBe("The <b>Quick</b> Brown <b>Fox</b>");
  });

  it("supports custom tags", () => {
    const text = "hello world";
    const result = highlight(text, ["world"], { startTag: "<em>", endTag: "</em>" });
    expect(result).toBe("hello <em>world</em>");
  });

  it("returns text unchanged when no match", () => {
    const text = "The quick brown fox";
    const result = highlight(text, ["zebra"]);
    expect(result).toBe("The quick brown fox");
  });

  it("returns empty string for empty text", () => {
    expect(highlight("", ["fox"])).toBe("");
  });

  it("returns text when terms are empty", () => {
    expect(highlight("hello world", [])).toBe("hello world");
  });

  it("handles null text", () => {
    expect(highlight(null, ["fox"])).toBe("");
  });

  it("highlights with analyzer", () => {
    const analyzer = standardAnalyzer();
    // "running" stems to "run", and "run" also stems to "run"
    const text = "She was running quickly through the park";
    const result = highlight(text, ["run"], { analyzer });
    expect(result).toContain("<b>running</b>");
  });

  it("extracts fragments", () => {
    const text = "A ".repeat(100) + "important keyword here" + " B".repeat(100);
    const result = highlight(text, ["keyword"], { maxFragments: 1, fragmentSize: 60 });
    expect(result).toContain("<b>keyword</b>");
    expect(result).toContain("...");
    expect(result.length).toBeLessThan(text.length);
  });

  it("extracts multiple fragments", () => {
    const text =
      "First match here. " +
      "X ".repeat(50) +
      "Second match here. " +
      "Y ".repeat(50) +
      "Third match here.";
    const result = highlight(text, ["match"], { maxFragments: 2, fragmentSize: 40 });
    expect(result.split("<b>match</b>").length).toBeGreaterThanOrEqual(2);
  });

  it("highlights all words", () => {
    const text = "foo bar baz";
    const result = highlight(text, ["foo", "bar", "baz"]);
    expect(result).toBe("<b>foo</b> <b>bar</b> <b>baz</b>");
  });
});

// ==================================================================
// Highlight: extract_query_terms
// ==================================================================

describe("extractQueryTerms", () => {
  it("extracts simple terms", () => {
    const terms = extractQueryTerms("database query");
    expect(terms).toContain("database");
    expect(terms).toContain("query");
  });

  it("extracts terms from boolean AND", () => {
    const terms = extractQueryTerms("database AND query");
    expect(terms).toContain("database");
    expect(terms).toContain("query");
    expect(terms).toHaveLength(2);
  });

  it("extracts terms from boolean OR", () => {
    const terms = extractQueryTerms("database OR query");
    expect(terms).toContain("database");
    expect(terms).toContain("query");
  });

  it("extracts terms from phrase", () => {
    const terms = extractQueryTerms('"information retrieval"');
    expect(terms).toContain("information");
    expect(terms).toContain("retrieval");
  });

  it("extracts terms with field prefix", () => {
    const terms = extractQueryTerms("title:database");
    expect(terms).toContain("database");
  });

  it("extracts terms from NOT operator", () => {
    const terms = extractQueryTerms("database NOT query");
    expect(terms).toContain("database");
    expect(terms).toContain("query");
  });

  it("extracts terms from complex query", () => {
    const terms = extractQueryTerms(
      'title:database AND "query optimization" OR search',
    );
    expect(terms).toContain("database");
    expect(terms).toContain("query");
    expect(terms).toContain("optimization");
    expect(terms).toContain("search");
  });
});

// ==================================================================
// uqa_highlight() in SQL
// ==================================================================

describe("SQL uqa_highlight()", () => {
  let engine: Engine;

  beforeEach(async () => {
    engine = await createEngine();
  });

  it("highlights basic search results", async () => {
    const result = await engine.sql(
      "SELECT title, uqa_highlight(body, 'database') AS snippet " +
        "FROM articles WHERE body @@ 'database'",
    );
    expect(result!.rows.length).toBeGreaterThan(0);
    for (const row of result!.rows) {
      expect(row["snippet"]).toContain("<b>");
      expect(row["snippet"]).toContain("</b>");
    }
  });

  it("highlights with custom tags", async () => {
    const result = await engine.sql(
      "SELECT title, uqa_highlight(body, 'search', '<em>', '</em>') AS snippet " +
        "FROM articles WHERE body @@ 'search'",
    );
    expect(result!.rows.length).toBeGreaterThan(0);
    for (const row of result!.rows) {
      expect(row["snippet"]).toContain("<em>");
      expect(row["snippet"]).toContain("</em>");
    }
  });

  it("highlights multi-term queries", async () => {
    const result = await engine.sql(
      "SELECT title, uqa_highlight(body, 'database query') AS snippet " +
        "FROM articles WHERE body @@ 'database query'",
    );
    expect(result!.rows.length).toBeGreaterThan(0);
    for (const row of result!.rows) {
      expect(row["snippet"]).toContain("<b>");
    }
  });

  it("preserves original text content", async () => {
    const result = await engine.sql(
      "SELECT title, uqa_highlight(title, 'database') AS hl_title " +
        "FROM articles WHERE title @@ 'database'",
    );
    expect(result!.rows.length).toBeGreaterThan(0);
    for (const row of result!.rows) {
      const clean = (row["hl_title"] as string)
        .replace(/<b>/g, "")
        .replace(/<\/b>/g, "");
      expect(clean).toBe(row["title"]);
    }
  });

  it("works without WHERE @@ clause", async () => {
    const result = await engine.sql(
      "SELECT title, uqa_highlight(title, 'graph') AS snippet FROM articles",
    );
    expect(result!.rows).toHaveLength(6);
    const highlightedCount = result!.rows.filter((r) =>
      (r["snippet"] as string).includes("<b>"),
    ).length;
    expect(highlightedCount).toBeGreaterThanOrEqual(1);
  });

  it("works with LIMIT", async () => {
    const result = await engine.sql(
      "SELECT title, uqa_highlight(body, 'search') AS snippet " +
        "FROM articles WHERE body @@ 'search' " +
        "ORDER BY _score DESC LIMIT 2",
    );
    expect(result!.rows.length).toBeLessThanOrEqual(2);
    for (const row of result!.rows) {
      expect(row["snippet"]).toContain("<b>");
    }
  });

  it("handles NULL body", async () => {
    await engine.sql(
      "INSERT INTO articles (title, body, category, author, year) " +
        "VALUES ('Empty Article', NULL, 'misc', 'Dave', 2024)",
    );
    const result = await engine.sql(
      "SELECT title, uqa_highlight(body, 'test') AS snippet " +
        "FROM articles WHERE title @@ 'empty'",
    );
    expect(result!.rows.length).toBeGreaterThanOrEqual(1);
    expect(result!.rows[0]!["snippet"]).toBeNull();
  });

  it("extracts fragments with size limit", async () => {
    const result = await engine.sql(
      "SELECT title, uqa_highlight(body, 'database', '<b>', '</b>', 1, 60) AS snippet " +
        "FROM articles WHERE body @@ 'database'",
    );
    expect(result!.rows.length).toBeGreaterThan(0);
    for (const row of result!.rows) {
      expect(row["snippet"]).toContain("<b>");
    }
  });
});

// ==================================================================
// uqa_facets() in SQL
// ==================================================================

describe("SQL uqa_facets()", () => {
  let engine: Engine;

  beforeEach(async () => {
    engine = await createEngine();
  });

  it("computes facets for a single field", async () => {
    const result = await engine.sql("SELECT uqa_facets(category) FROM articles");
    expect(result!.columns).toContain("facet_value");
    expect(result!.columns).toContain("facet_count");
    expect(result!.columns).not.toContain("facet_field");
    const counts: Record<string, number> = {};
    for (const r of result!.rows) {
      counts[r["facet_value"] as string] = r["facet_count"] as number;
    }
    expect(counts["databases"]).toBe(3);
    expect(counts["search"]).toBe(2);
    expect(counts["nlp"]).toBe(1);
  });

  it("computes facets with text search filter", async () => {
    const result = await engine.sql(
      "SELECT uqa_facets(category) FROM articles WHERE body @@ 'search'",
    );
    const counts: Record<string, number> = {};
    for (const r of result!.rows) {
      counts[r["facet_value"] as string] = r["facet_count"] as number;
    }
    const total = Object.values(counts).reduce((s, c) => s + c, 0);
    expect(total).toBeGreaterThan(0);
  });

  it("computes multi-field facets", async () => {
    const result = await engine.sql(
      "SELECT uqa_facets(category, author) FROM articles",
    );
    expect(result!.columns).toContain("facet_field");
    expect(result!.columns).toContain("facet_value");
    expect(result!.columns).toContain("facet_count");
    const fields = new Set(result!.rows.map((r) => r["facet_field"]));
    expect(fields.has("category")).toBe(true);
    expect(fields.has("author")).toBe(true);
  });

  it("computes facets for year", async () => {
    const result = await engine.sql("SELECT uqa_facets(year) FROM articles");
    const counts: Record<string, number> = {};
    for (const r of result!.rows) {
      counts[r["facet_value"] as string] = r["facet_count"] as number;
    }
    expect(Object.keys(counts).length).toBeGreaterThanOrEqual(4);
  });

  it("respects WHERE clause filtering", async () => {
    const result = await engine.sql(
      "SELECT uqa_facets(category) FROM articles WHERE body @@ 'database'",
    );
    const counts: Record<string, number> = {};
    for (const r of result!.rows) {
      counts[r["facet_value"] as string] = r["facet_count"] as number;
    }
    const total = Object.values(counts).reduce((s, c) => s + c, 0);
    expect(total).toBeGreaterThan(0);
    expect(total).toBeLessThanOrEqual(6);
  });

  it("computes author facets", async () => {
    const result = await engine.sql("SELECT uqa_facets(author) FROM articles");
    const counts: Record<string, number> = {};
    for (const r of result!.rows) {
      counts[r["facet_value"] as string] = r["facet_count"] as number;
    }
    expect(counts["Alice"]).toBe(2);
    expect(counts["Bob"]).toBe(2);
    expect(counts["Carol"]).toBe(2);
  });

  it("sorts facet values alphabetically", async () => {
    const result = await engine.sql("SELECT uqa_facets(category) FROM articles");
    const values = result!.rows.map((r) => r["facet_value"] as string);
    expect(values).toEqual([...values].sort());
  });

  it("returns empty result for no matching docs", async () => {
    const result = await engine.sql(
      "SELECT uqa_facets(category) FROM articles WHERE body @@ 'xyznonexistent'",
    );
    expect(result!.rows).toHaveLength(0);
  });
});
