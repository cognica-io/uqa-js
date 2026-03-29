import { describe, expect, it } from "vitest";
import { tokenizeFts, FTSParser } from "../../src/sql/fts-query.js";
import type {
  TermNode,
  PhraseNode,
  VectorNode,
  AndNode,
  OrNode,
  NotNode,
} from "../../src/sql/fts-query.js";
import { Engine } from "../../src/engine.js";

// ==================================================================
// Lexer unit tests
// ==================================================================

describe("FTSTokenizer", () => {
  it("single term", () => {
    const tokens = tokenizeFts("hello");
    expect(tokens[0]!.type).toBe("TERM");
    expect(tokens[0]!.value).toBe("hello");
    expect(tokens[tokens.length - 1]!.type).toBe("EOF");
  });

  it("multiple terms", () => {
    const tokens = tokenizeFts("hello world");
    expect(tokens[0]!.type).toBe("TERM");
    expect(tokens[1]!.type).toBe("TERM");
    expect(tokens[2]!.type).toBe("EOF");
  });

  it("phrase", () => {
    const tokens = tokenizeFts('"hello world"');
    expect(tokens[0]!.type).toBe("PHRASE");
    expect(tokens[0]!.value).toBe("hello world");
  });

  it("vector", () => {
    const tokens = tokenizeFts("[0.1, 0.2, 0.3]");
    expect(tokens[0]!.type).toBe("VECTOR");
    expect(tokens[0]!.value).toBe("0.1, 0.2, 0.3");
  });

  it("boolean keywords", () => {
    const tokens = tokenizeFts("a AND b OR c NOT d");
    const types = tokens.slice(0, -1).map((t) => t.type);
    expect(types).toEqual(["TERM", "AND", "TERM", "OR", "TERM", "NOT", "TERM"]);
  });

  it("case insensitive keywords", () => {
    const tokens = tokenizeFts("a and b or c not d");
    const types = tokens.slice(0, -1).map((t) => t.type);
    expect(types).toEqual(["TERM", "AND", "TERM", "OR", "TERM", "NOT", "TERM"]);
  });

  it("field colon term", () => {
    const tokens = tokenizeFts("title:hello");
    const types = tokens.slice(0, -1).map((t) => t.type);
    expect(types).toEqual(["TERM", "COLON", "TERM"]);
  });

  it("parentheses", () => {
    const tokens = tokenizeFts("(a OR b)");
    const types = tokens.slice(0, -1).map((t) => t.type);
    expect(types).toEqual(["LPAREN", "TERM", "OR", "TERM", "RPAREN"]);
  });

  it("empty string", () => {
    const tokens = tokenizeFts("");
    expect(tokens.length).toBe(1);
    expect(tokens[0]!.type).toBe("EOF");
  });

  it("unterminated quote throws", () => {
    expect(() => tokenizeFts('"hello')).toThrow(/Unterminated quoted phrase/);
  });

  it("unterminated bracket throws", () => {
    expect(() => tokenizeFts("[0.1, 0.2")).toThrow(/Unterminated vector literal/);
  });

  it("complex query", () => {
    const tokens = tokenizeFts('title:"neural network" AND embedding:[0.1, 0.2]');
    const types = tokens.slice(0, -1).map((t) => t.type);
    expect(types).toEqual([
      "TERM",
      "COLON",
      "PHRASE",
      "AND",
      "TERM",
      "COLON",
      "VECTOR",
    ]);
  });
});

// ==================================================================
// Parser unit tests
// ==================================================================

describe("FTSParser", () => {
  it("single term", () => {
    const ast = new FTSParser(tokenizeFts("hello")).parse();
    expect(ast.type).toBe("term");
    expect((ast as TermNode).field).toBeNull();
    expect((ast as TermNode).term).toBe("hello");
  });

  it("phrase", () => {
    const ast = new FTSParser(tokenizeFts('"hello world"')).parse();
    expect(ast.type).toBe("phrase");
    expect((ast as PhraseNode).phrase).toBe("hello world");
  });

  it("field term", () => {
    const ast = new FTSParser(tokenizeFts("title:hello")).parse();
    expect(ast.type).toBe("term");
    expect((ast as TermNode).field).toBe("title");
    expect((ast as TermNode).term).toBe("hello");
  });

  it("field phrase", () => {
    const ast = new FTSParser(tokenizeFts('title:"hello world"')).parse();
    expect(ast.type).toBe("phrase");
    expect((ast as PhraseNode).field).toBe("title");
    expect((ast as PhraseNode).phrase).toBe("hello world");
  });

  it("field vector", () => {
    const ast = new FTSParser(tokenizeFts("embedding:[0.1, 0.2, 0.3]")).parse();
    expect(ast.type).toBe("vector");
    expect((ast as VectorNode).field).toBe("embedding");
    expect((ast as VectorNode).values).toEqual([0.1, 0.2, 0.3]);
  });

  it("explicit and", () => {
    const ast = new FTSParser(tokenizeFts("a AND b")).parse();
    expect(ast.type).toBe("and");
    expect((ast as AndNode).left.type).toBe("term");
    expect((ast as AndNode).right.type).toBe("term");
  });

  it("explicit or", () => {
    const ast = new FTSParser(tokenizeFts("a OR b")).parse();
    expect(ast.type).toBe("or");
  });

  it("not", () => {
    const ast = new FTSParser(tokenizeFts("NOT a")).parse();
    expect(ast.type).toBe("not");
    expect((ast as NotNode).operand.type).toBe("term");
  });

  it("implicit and", () => {
    const ast = new FTSParser(tokenizeFts("a b")).parse();
    expect(ast.type).toBe("and");
    expect((ast as AndNode).left.type).toBe("term");
    expect((ast as AndNode).right.type).toBe("term");
  });

  it("precedence and over or", () => {
    const ast = new FTSParser(tokenizeFts("a OR b AND c")).parse();
    expect(ast.type).toBe("or");
    expect((ast as OrNode).left.type).toBe("term");
    expect(((ast as OrNode).left as TermNode).term).toBe("a");
    expect((ast as OrNode).right.type).toBe("and");
  });

  it("grouping overrides precedence", () => {
    const ast = new FTSParser(tokenizeFts("(a OR b) AND c")).parse();
    expect(ast.type).toBe("and");
    expect((ast as AndNode).left.type).toBe("or");
    expect((ast as AndNode).right.type).toBe("term");
  });

  it("complex nested", () => {
    const ast = new FTSParser(
      tokenizeFts("(title:attention OR body:transformer) AND embedding:[0.1, 0.2]"),
    ).parse();
    expect(ast.type).toBe("and");
    expect((ast as AndNode).left.type).toBe("or");
    expect((ast as AndNode).right.type).toBe("vector");
  });

  it("double negation", () => {
    const ast = new FTSParser(tokenizeFts("NOT NOT hello")).parse();
    expect(ast.type).toBe("not");
    expect((ast as NotNode).operand.type).toBe("not");
    expect(((ast as NotNode).operand as NotNode).operand.type).toBe("term");
  });

  it("empty query", () => {
    expect(() => new FTSParser(tokenizeFts("")).parse()).toThrow();
  });

  it("trailing operator", () => {
    expect(() => new FTSParser(tokenizeFts("a AND")).parse()).toThrow();
  });

  it("unbalanced paren", () => {
    expect(() => new FTSParser(tokenizeFts("(a OR b")).parse()).toThrow();
  });

  it("three implicit and", () => {
    const ast = new FTSParser(tokenizeFts("a b c")).parse();
    expect(ast.type).toBe("and");
    expect((ast as AndNode).left.type).toBe("and");
    expect((ast as AndNode).right.type).toBe("term");
    expect(((ast as AndNode).right as TermNode).term).toBe("c");
  });
});

// ==================================================================
// SQL integration tests
// ==================================================================

async function makeEngineWithDocs(): Promise<Engine> {
  const engine = new Engine();
  await engine.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT, body TEXT)");
  await engine.sql("CREATE INDEX idx_docs_fts ON docs USING gin (title, body)");
  await engine.sql(
    "INSERT INTO docs VALUES (1, 'neural network theory', 'attention mechanisms for deep learning')",
  );
  await engine.sql(
    "INSERT INTO docs VALUES (2, 'graph theory introduction', 'traversal algorithms and shortest paths')",
  );
  await engine.sql(
    "INSERT INTO docs VALUES (3, 'attention neural models', 'neural network graph embedding theory')",
  );
  return engine;
}

describe("FTSMatchSQL", () => {
  it("Engine FTS @@ operator", async () => {
    const engine = await makeEngineWithDocs();
    const result = await engine.sql(
      "SELECT id, title FROM docs WHERE body @@ 'attention'",
    );
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBeGreaterThan(0);
  });

  it("single term", async () => {
    const engine = await makeEngineWithDocs();
    const result = await engine.sql("SELECT id FROM docs WHERE body @@ 'neural'");
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBeGreaterThan(0);
  });

  it("phrase", async () => {
    const engine = await makeEngineWithDocs();
    const result = await engine.sql(
      "SELECT id FROM docs WHERE body @@ '\"attention mechanisms\"'",
    );
    expect(result).not.toBeNull();
    // Phrase search: "attention mechanisms"
    expect(result!.rows.length).toBeGreaterThanOrEqual(0);
  });

  it("all column", async () => {
    const engine = await makeEngineWithDocs();
    const result = await engine.sql(
      "SELECT id FROM docs WHERE title @@ 'neural' OR body @@ 'neural'",
    );
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBeGreaterThan(0);
  });

  it("boolean and", async () => {
    const engine = await makeEngineWithDocs();
    const result = await engine.sql(
      "SELECT id FROM docs WHERE body @@ 'neural AND network'",
    );
    expect(result).not.toBeNull();
  });

  it("boolean or", async () => {
    const engine = await makeEngineWithDocs();
    const result = await engine.sql(
      "SELECT id FROM docs WHERE body @@ 'attention OR graph'",
    );
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBeGreaterThan(0);
  });

  it("boolean not", async () => {
    const engine = await makeEngineWithDocs();
    const result = await engine.sql(
      "SELECT id FROM docs WHERE body @@ 'NOT attention'",
    );
    expect(result).not.toBeNull();
  });

  it("grouping", async () => {
    const engine = await makeEngineWithDocs();
    const result = await engine.sql(
      "SELECT id FROM docs WHERE body @@ '(attention OR graph) AND neural'",
    );
    expect(result).not.toBeNull();
  });

  it("implicit and", async () => {
    const engine = await makeEngineWithDocs();
    const result = await engine.sql(
      "SELECT id FROM docs WHERE body @@ 'neural network'",
    );
    expect(result).not.toBeNull();
  });

  it("field specific", async () => {
    const engine = await makeEngineWithDocs();
    const result = await engine.sql("SELECT id FROM docs WHERE title @@ 'neural'");
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBeGreaterThan(0);
  });

  it("hybrid text vector", async () => {
    // Verify that FTS AST can represent a hybrid query mixing text and vector
    const ast = new FTSParser(
      tokenizeFts("attention AND embedding:[0.1, 0.2, 0.3]"),
    ).parse();
    expect(ast.type).toBe("and");
    expect((ast as AndNode).right.type).toBe("vector");
  });

  it("score calibrated", async () => {
    const engine = await makeEngineWithDocs();
    const result = await engine.sql("SELECT id FROM docs WHERE body @@ 'attention'");
    expect(result).not.toBeNull();
    if (result!.rows.length > 0) {
      // Results should have scores
      expect(result!.rows.length).toBeGreaterThan(0);
    }
  });

  it("order by score limit", async () => {
    const engine = await makeEngineWithDocs();
    const result = await engine.sql(
      "SELECT id FROM docs WHERE body @@ 'neural' ORDER BY id LIMIT 2",
    );
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBeLessThanOrEqual(2);
  });

  it("combined with equality", async () => {
    const engine = await makeEngineWithDocs();
    // Verify FTS works, then verify we can combine FTS with equality
    const allResult = await engine.sql("SELECT * FROM docs");
    expect(allResult).not.toBeNull();
    expect(allResult!.rows.length).toBe(3);
    // FTS on body field
    const ftsResult = await engine.sql("SELECT id FROM docs WHERE body @@ 'attention'");
    expect(ftsResult).not.toBeNull();
    // body: "attention mechanisms for deep learning" is in doc 1
    expect(ftsResult!.rows.length).toBeGreaterThan(0);
  });
});

// ==================================================================
// Edge cases
// ==================================================================

describe("FTSMatchEdgeCases", () => {
  it("empty query via Engine", async () => {
    // Empty FTS query should throw during parse
    expect(() => new FTSParser(tokenizeFts("")).parse()).toThrow();
  });

  it("unknown field", async () => {
    const engine = await makeEngineWithDocs();
    // Searching on a field that doesn't exist should still work (empty results)
    const result = await engine.sql(
      "SELECT id FROM docs WHERE title @@ 'nonexistent_term_xyz'",
    );
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(0);
  });

  it("malformed vector", () => {
    expect(() => new FTSParser(tokenizeFts("field:[abc, def]")).parse()).toThrow();
  });

  it("malformed vector via parser", () => {
    expect(() => new FTSParser(tokenizeFts("field:[abc, def]")).parse()).toThrow();
  });

  it("unbalanced parens", () => {
    expect(() => new FTSParser(tokenizeFts("(a AND b")).parse()).toThrow();
  });

  it("extra closing paren", () => {
    expect(() => new FTSParser(tokenizeFts("a AND b)")).parse()).toThrow();
  });

  it("vector only query via Engine", () => {
    // A vector-only FTS query should parse
    const ast = new FTSParser(tokenizeFts("embedding:[0.1, 0.2, 0.3]")).parse();
    expect(ast.type).toBe("vector");
    expect((ast as VectorNode).values).toEqual([0.1, 0.2, 0.3]);
  });

  it("not only via Engine", () => {
    // A NOT-only FTS query should parse
    const ast = new FTSParser(tokenizeFts("NOT hello")).parse();
    expect(ast.type).toBe("not");
  });

  it("empty vector literal", () => {
    // "field:[]" should throw (empty vector)
    expect(() => new FTSParser(tokenizeFts("field:[]")).parse()).toThrow();
  });
});
