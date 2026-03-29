import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// The Python QueryBuilder tests for execute_arrow() and execute_parquet()
// rely on pyarrow. In TS we verify the equivalent functionality through
// the execute() method which returns a PostingList.

async function makeEngine(): Promise<Engine> {
  const engine = new Engine();
  await engine.sql(`
    CREATE TABLE docs (
      id SERIAL PRIMARY KEY,
      title TEXT,
      body TEXT,
      year INTEGER,
      score REAL
    )
  `);
  await engine.sql(
    "CREATE INDEX idx_docs_fts ON docs USING gin (title, body)",
  );
  await engine.sql(`INSERT INTO docs (title, body, year, score) VALUES
    ('attention is all you need', 'transformer model uses self attention', 2017, 9.5),
    ('bert pre-training', 'bidirectional encoder representations', 2019, 8.0),
    ('graph attention networks', 'attention on graph structured data', 2018, 7.5),
    ('vision transformer', 'image recognition with patches', 2021, 6.0),
    ('scaling language models', 'scaling laws for neural language models', 2020, 8.5)
  `);
  // Ensure the table is in the engine's _tables map (required for _contextForTable)
  engine.getTable("docs");
  return engine;
}

describe("ExecuteArrow", () => {
  it("execute returns result with doc ids and scores", async () => {
    const engine = await makeEngine();
    const result = engine.query("docs").term("graph", "title").execute();
    expect(result.length).toBeGreaterThanOrEqual(1);
    for (const entry of result) {
      expect(entry.docId).toBeGreaterThan(0);
      expect(entry.payload).toBeDefined();
    }
    engine.close();
  });

  it("basic term query returns matching docs", async () => {
    const engine = await makeEngine();
    const result = engine.query("docs").term("attention", "title").execute();
    expect(result.length).toBeGreaterThanOrEqual(1);
    // "attention" appears in "attention is all you need" and "graph attention networks"
    const docIds = result.entries.map((e) => e.docId);
    expect(docIds.length).toBeGreaterThanOrEqual(2);
    engine.close();
  });

  it("empty result for nonexistent term", async () => {
    const engine = await makeEngine();
    const result = engine.query("docs").term("xyznonexistent", "title").execute();
    expect(result.length).toBe(0);
    engine.close();
  });

  it("with scoring returns positive scores", async () => {
    const engine = await makeEngine();
    const result = engine
      .query("docs")
      .term("graph", "title")
      .scoreBm25("graph", "title")
      .execute();
    expect(result.length).toBeGreaterThanOrEqual(1);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0);
    }
    engine.close();
  });

  it("doc ids are positive integers", async () => {
    const engine = await makeEngine();
    const result = engine.query("docs").term("graph", "title").execute();
    for (const entry of result) {
      expect(Number.isInteger(entry.docId)).toBe(true);
      expect(entry.docId).toBeGreaterThan(0);
    }
    engine.close();
  });
});

describe("ExecuteParquet", () => {
  it("execute returns structured result", async () => {
    const engine = await makeEngine();
    const result = engine.query("docs").term("graph", "title").execute();
    expect(result.length).toBeGreaterThanOrEqual(1);
    expect(result.entries).toBeDefined();
    engine.close();
  });

  it("basic term query consistency", async () => {
    const engine = await makeEngine();
    const result1 = engine.query("docs").term("graph", "title").execute();
    const result2 = engine.query("docs").term("graph", "title").execute();
    expect(result1.length).toBe(result2.length);
    const ids1 = result1.entries.map((e) => e.docId).sort((a, b) => a - b);
    const ids2 = result2.entries.map((e) => e.docId).sort((a, b) => a - b);
    expect(ids1).toEqual(ids2);
    engine.close();
  });

  it("roundtrip consistency between executions", async () => {
    const engine = await makeEngine();
    const qb = engine.query("docs").term("graph", "title");
    const result1 = qb.execute();
    const result2 = qb.execute();
    expect(result1.length).toBe(result2.length);
    expect(result1.entries.map((e) => e.docId)).toEqual(
      result2.entries.map((e) => e.docId),
    );
    engine.close();
  });

  it("empty result for nonexistent term", async () => {
    const engine = await makeEngine();
    const result = engine.query("docs").term("xyznonexistent", "title").execute();
    expect(result.length).toBe(0);
    engine.close();
  });
});
