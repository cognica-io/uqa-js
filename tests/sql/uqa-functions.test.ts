import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

async function makeEngine(): Promise<Engine> {
  const e = new Engine();
  await e.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT, body TEXT)");
  await e.sql("CREATE INDEX idx_docs_fts ON docs USING gin (title, body)");
  e.insert("docs", 1, { title: "machine learning basics", body: "algorithms for ML research" });
  e.insert("docs", 2, { title: "deep learning neural networks", body: "CNN and RNN models" });
  e.insert("docs", 3, { title: "cooking recipes", body: "pasta and pizza" });
  return e;
}

describe("UQA SQL functions", () => {
  it("text_match", async () => {
    const e = await makeEngine();
    const r = await e.sql("SELECT id, _score FROM docs WHERE text_match(title, 'learning') ORDER BY _score DESC");
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
  });

  it("multi_field_match", async () => {
    const e = await makeEngine();
    const r = await e.sql("SELECT id, _score FROM docs WHERE multi_field_match(title, body, 'learning') ORDER BY _score DESC");
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
  });

  it("fuse_log_odds", async () => {
    const e = await makeEngine();
    const r = await e.sql("SELECT id, _score FROM docs WHERE fuse_log_odds(text_match(title, 'learning'), text_match(body, 'algorithms')) ORDER BY _score DESC");
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
  });

  it("fuse_prob_and", async () => {
    const e = await makeEngine();
    const r = await e.sql("SELECT id, _score FROM docs WHERE fuse_prob_and(text_match(title, 'learning'), text_match(body, 'algorithms')) ORDER BY _score DESC");
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
  });

  it("fuse_prob_or", async () => {
    const e = await makeEngine();
    const r = await e.sql("SELECT id, _score FROM docs WHERE fuse_prob_or(text_match(title, 'learning'), text_match(body, 'pasta')) ORDER BY _score DESC");
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
  });

  it("fuse_prob_not", async () => {
    const e = await makeEngine();
    const r = await e.sql("SELECT id, _score FROM docs WHERE fuse_prob_not(text_match(title, 'cooking')) ORDER BY _score DESC");
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
  });

  it("sparse_threshold", async () => {
    const e = await makeEngine();
    const r = await e.sql("SELECT id, _score FROM docs WHERE sparse_threshold(text_match(title, 'learning'), 0.1) ORDER BY _score DESC");
    expect(r).not.toBeNull();
  });

  it("create_analyzer and list_analyzers", async () => {
    const e = new Engine();
    const r1 = await e.sql("SELECT * FROM create_analyzer('my_ws', '{\"tokenizer\":{\"type\":\"whitespace\"},\"token_filters\":[{\"type\":\"lowercase\"}],\"char_filters\":[]}')");
    expect(r1).not.toBeNull();
    const r2 = await e.sql("SELECT * FROM list_analyzers()");
    expect(r2).not.toBeNull();
    expect(r2!.rows.some((r: Record<string, unknown>) => r["analyzer_name"] === "my_ws")).toBe(true);
  });

  it("create_graph and drop_graph", async () => {
    const e = new Engine();
    const r1 = await e.sql("SELECT * FROM create_graph('test_g')");
    expect(r1).not.toBeNull();
    expect(e.graphStore.hasGraph("test_g")).toBe(true);
    await e.sql("SELECT * FROM drop_graph('test_g')");
    expect(e.graphStore.hasGraph("test_g")).toBe(false);
  });
});

describe("UQA functions with JOIN", () => {
  async function makeJoinEngine(): Promise<Engine> {
    const e = new Engine();
    await e.sql("CREATE TABLE hotels (id INTEGER PRIMARY KEY, name TEXT)");
    await e.sql(
      "CREATE TABLE rooms (id INTEGER PRIMARY KEY, hotel_id INTEGER NOT NULL, name TEXT, description TEXT)",
    );
    await e.sql("INSERT INTO hotels (id, name) VALUES (1, 'Grand Hotel')");
    await e.sql("INSERT INTO hotels (id, name) VALUES (2, 'Beach Resort')");
    await e.sql(
      "INSERT INTO rooms (id, hotel_id, name, description) VALUES (1, 1, 'Deluxe Room', 'ocean view with pool access')",
    );
    await e.sql(
      "INSERT INTO rooms (id, hotel_id, name, description) VALUES (2, 1, 'Standard Room', 'city view')",
    );
    await e.sql(
      "INSERT INTO rooms (id, hotel_id, name, description) VALUES (3, 2, 'Beach Suite', 'ocean front with private beach')",
    );
    await e.sql("CREATE INDEX rooms_name_idx ON rooms USING gin (name)");
    await e.sql("CREATE INDEX rooms_desc_idx ON rooms USING gin (description)");
    return e;
  }

  it("fuse_log_odds with INNER JOIN", async () => {
    const e = await makeJoinEngine();
    const r = await e.sql(`
      SELECT r.id, r.name, h.name AS hotel_name, _score
      FROM rooms r
      JOIN hotels h ON r.hotel_id = h.id
      WHERE fuse_log_odds(
        bayesian_match(r.name, 'ocean'),
        bayesian_match(r.description, 'ocean')
      )
      ORDER BY _score DESC
    `);
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
    for (const row of r!.rows) {
      expect(row["_score"]).toBeTypeOf("number");
      expect(row["hotel_name"]).toBeTypeOf("string");
    }
  });

  it("bayesian_match with INNER JOIN", async () => {
    const e = await makeJoinEngine();
    const r = await e.sql(`
      SELECT r.id, r.name, h.name AS hotel_name, _score
      FROM rooms r
      JOIN hotels h ON r.hotel_id = h.id
      WHERE bayesian_match(r.description, 'ocean')
      ORDER BY _score DESC
    `);
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
    for (const row of r!.rows) {
      expect(row["_score"]).toBeTypeOf("number");
      expect(row["hotel_name"]).toBeTypeOf("string");
    }
  });

  it("fuse_prob_or with LEFT JOIN", async () => {
    const e = await makeJoinEngine();
    const r = await e.sql(`
      SELECT r.id, r.name, h.name AS hotel_name, _score
      FROM rooms r
      LEFT JOIN hotels h ON r.hotel_id = h.id
      WHERE fuse_prob_or(
        bayesian_match(r.name, 'beach'),
        bayesian_match(r.description, 'beach')
      )
      ORDER BY _score DESC
    `);
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
    for (const row of r!.rows) {
      expect(row["_score"]).toBeTypeOf("number");
    }
  });

  it("single-table UQA still works (regression check)", async () => {
    const e = await makeJoinEngine();
    const r = await e.sql(`
      SELECT id, name, _score FROM rooms
      WHERE fuse_log_odds(
        bayesian_match(name, 'ocean'),
        bayesian_match(description, 'ocean')
      )
      ORDER BY _score DESC
    `);
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
    for (const row of r!.rows) {
      expect(row["_score"]).toBeTypeOf("number");
    }
  });
});

describe("bayesian_match_with_prior as calibrated signal in fusion", () => {
  async function makePriorEngine(): Promise<Engine> {
    const e = new Engine();
    await e.sql(
      "CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, body TEXT, updated_at TEXT, authority TEXT)",
    );
    await e.sql("CREATE INDEX idx_articles_fts ON articles USING gin (title, body)");
    const now = new Date().toISOString();
    const old = new Date(Date.now() - 365 * 86400000).toISOString();
    e.insert("articles", 1, {
      title: "machine learning basics",
      body: "algorithms for ML research",
      updated_at: now,
      authority: "high",
    });
    e.insert("articles", 2, {
      title: "deep learning neural networks",
      body: "CNN and RNN models for learning",
      updated_at: old,
      authority: "low",
    });
    e.insert("articles", 3, {
      title: "cooking recipes",
      body: "pasta and pizza",
      updated_at: now,
      authority: "medium",
    });
    return e;
  }

  it("fuse_attention with bayesian_match_with_prior (recency)", async () => {
    const e = await makePriorEngine();
    const r = await e.sql(`
      SELECT id, _score FROM articles
      WHERE fuse_attention(
        bayesian_match_with_prior(title, 'learning', 'updated_at', 'recency'),
        bayesian_match(body, 'learning')
      )
      ORDER BY _score DESC
    `);
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
    for (const row of r!.rows) {
      expect(row["_score"]).toBeTypeOf("number");
      expect(row["_score"] as number).toBeGreaterThan(0);
      expect(row["_score"] as number).toBeLessThanOrEqual(1);
    }
  });

  it("fuse_attention with bayesian_match_with_prior (authority)", async () => {
    const e = await makePriorEngine();
    const r = await e.sql(`
      SELECT id, _score FROM articles
      WHERE fuse_attention(
        bayesian_match_with_prior(title, 'learning', 'authority', 'authority'),
        bayesian_match(body, 'algorithms')
      )
      ORDER BY _score DESC
    `);
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
    for (const row of r!.rows) {
      expect(row["_score"]).toBeTypeOf("number");
      expect(row["_score"] as number).toBeGreaterThan(0);
      expect(row["_score"] as number).toBeLessThanOrEqual(1);
    }
  });

  it("fuse_multihead with bayesian_match_with_prior", async () => {
    const e = await makePriorEngine();
    const r = await e.sql(`
      SELECT id, _score FROM articles
      WHERE fuse_multihead(
        bayesian_match_with_prior(title, 'learning', 'updated_at', 'recency'),
        bayesian_match(body, 'learning')
      )
      ORDER BY _score DESC
    `);
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
    for (const row of r!.rows) {
      expect(row["_score"]).toBeTypeOf("number");
      expect(row["_score"] as number).toBeGreaterThan(0);
      expect(row["_score"] as number).toBeLessThanOrEqual(1);
    }
  });

  it("fuse_learned with bayesian_match_with_prior", async () => {
    const e = await makePriorEngine();
    const r = await e.sql(`
      SELECT id, _score FROM articles
      WHERE fuse_learned(
        bayesian_match_with_prior(title, 'learning', 'authority', 'authority'),
        bayesian_match(body, 'learning')
      )
      ORDER BY _score DESC
    `);
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
    for (const row of r!.rows) {
      expect(row["_score"]).toBeTypeOf("number");
      expect(row["_score"] as number).toBeGreaterThan(0);
      expect(row["_score"] as number).toBeLessThanOrEqual(1);
    }
  });

  it("fuse_attention with two bayesian_match_with_prior signals", async () => {
    const e = await makePriorEngine();
    const r = await e.sql(`
      SELECT id, _score FROM articles
      WHERE fuse_attention(
        bayesian_match_with_prior(title, 'learning', 'updated_at', 'recency'),
        bayesian_match_with_prior(body, 'learning', 'authority', 'authority')
      )
      ORDER BY _score DESC
    `);
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
    for (const row of r!.rows) {
      expect(row["_score"]).toBeTypeOf("number");
      expect(row["_score"] as number).toBeGreaterThan(0);
      expect(row["_score"] as number).toBeLessThanOrEqual(1);
    }
  });
});

describe("set_table_analyzer via SQL", () => {
  it("applies CJK analyzer and enables prefix search", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE pages (id INTEGER PRIMARY KEY, title TEXT)");
    await e.sql("CREATE INDEX idx_pages_fts ON pages USING gin (title)");
    await e.sql("SELECT * FROM set_table_analyzer('pages', 'title', 'standard_cjk', 'both')");
    e.insert("pages", 1, { title: "bayesian inference" });
    const r = await e.sql("SELECT id, _score FROM pages WHERE text_match(title, 'bayesi')");
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
  });
});
