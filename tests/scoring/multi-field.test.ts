import { describe, expect, it } from "vitest";
import { IndexStats } from "../../src/core/types.js";
import { createBayesianBM25Params } from "../../src/scoring/bayesian-bm25.js";
import { MultiFieldBayesianScorer } from "../../src/scoring/multi-field.js";
import { MultiFieldSearchOperator } from "../../src/operators/multi-field.js";
import { Engine } from "../../src/engine.js";

// -- TestMultiFieldBayesianScorer --

describe("TestMultiFieldBayesianScorer", () => {
  it("single field", () => {
    const stats = new IndexStats(100, 10.0);
    const scorer = new MultiFieldBayesianScorer(
      [["title", createBayesianBM25Params(), 1.0]],
      stats,
    );
    const score = scorer.scoreDocument(1, { title: 3 }, { title: 10 }, { title: 10 });
    expect(score).toBeGreaterThan(0.0);
    expect(score).toBeLessThan(1.0);
  });

  it("two fields higher than one", () => {
    const stats = new IndexStats(100, 10.0);
    const scorer = new MultiFieldBayesianScorer(
      [
        ["title", createBayesianBM25Params(), 1.0],
        ["body", createBayesianBM25Params(), 0.5],
      ],
      stats,
    );
    const oneField = scorer.scoreDocument(
      1,
      { title: 3, body: 0 },
      { title: 10, body: 100 },
      { title: 10, body: 50 },
    );
    const twoFields = scorer.scoreDocument(
      1,
      { title: 3, body: 5 },
      { title: 10, body: 100 },
      { title: 10, body: 50 },
    );
    expect(twoFields).toBeGreaterThan(oneField);
  });

  it("zero tf gives neutral", () => {
    const stats = new IndexStats(100, 10.0);
    const scorer = new MultiFieldBayesianScorer(
      [["title", createBayesianBM25Params(), 1.0]],
      stats,
    );
    const score = scorer.scoreDocument(1, { title: 0 }, { title: 10 }, { title: 10 });
    expect(score).toBeCloseTo(0.5, 1);
  });
});

// -- TestMultiFieldSearchOperator --

describe("TestMultiFieldSearchOperator", () => {
  async function makeEngine(): Promise<Engine> {
    const e = new Engine();
    await e.sql("CREATE TABLE docs (id SERIAL PRIMARY KEY, title TEXT, body TEXT)");
    await e.sql("CREATE INDEX idx_docs_fts ON docs USING gin (title, body)");
    await e.sql(
      "INSERT INTO docs (title, body) VALUES ('machine learning guide', 'intro to ML algorithms')",
    );
    await e.sql(
      "INSERT INTO docs (title, body) VALUES ('database systems', 'indexing and learning structures')",
    );
    await e.sql(
      "INSERT INTO docs (title, body) VALUES ('cooking recipes', 'delicious pasta dishes')",
    );
    return e;
  }

  it("multi field search returns results", async () => {
    const e = await makeEngine();
    const table = e.getTable("docs");
    const ctx = { invertedIndex: table.invertedIndex, graphStore: e._graphStore };
    const op = new MultiFieldSearchOperator(["title", "body"], "learning");
    const result = op.execute(ctx);
    expect(result.length).toBeGreaterThan(0);
  });

  it("multi field scores are probabilities", async () => {
    const e = await makeEngine();
    const table = e.getTable("docs");
    const ctx = { invertedIndex: table.invertedIndex, graphStore: e._graphStore };
    const op = new MultiFieldSearchOperator(["title", "body"], "learning");
    const result = op.execute(ctx);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });

  it("multi field with weights", async () => {
    const e = await makeEngine();
    const table = e.getTable("docs");
    const ctx = { invertedIndex: table.invertedIndex, graphStore: e._graphStore };
    const op = new MultiFieldSearchOperator(["title", "body"], "learning", [2.0, 0.5]);
    const result = op.execute(ctx);
    expect(result.length).toBeGreaterThan(0);
  });

  it("cost estimate", () => {
    const op = new MultiFieldSearchOperator(["title", "body"], "test");
    const stats = new IndexStats(100);
    const cost = op.costEstimate(stats);
    expect(cost).toBe(200.0);
  });
});

// -- TestMultiFieldSQL --

describe("TestMultiFieldSQL", () => {
  async function makeEngine(): Promise<Engine> {
    const e = new Engine();
    await e.sql("CREATE TABLE docs (id SERIAL PRIMARY KEY, title TEXT, body TEXT)");
    await e.sql("CREATE INDEX idx_docs_fts ON docs USING gin (title, body)");
    await e.sql(
      "INSERT INTO docs (title, body) VALUES ('machine learning', 'algorithms for ML')",
    );
    await e.sql(
      "INSERT INTO docs (title, body) VALUES ('cooking recipes', 'pasta and pizza')",
    );
    return e;
  }

  it("multi field match via operator", async () => {
    const e = await makeEngine();
    const table = e.getTable("docs");
    const ctx = { invertedIndex: table.invertedIndex, graphStore: e._graphStore };
    const op = new MultiFieldSearchOperator(["title", "body"], "learning");
    const result = op.execute(ctx);
    expect(result.length).toBeGreaterThan(0);
  });

  it("multi field match with weights via operator", async () => {
    const e = await makeEngine();
    const table = e.getTable("docs");
    const ctx = { invertedIndex: table.invertedIndex, graphStore: e._graphStore };
    const op = new MultiFieldSearchOperator(["title", "body"], "learning", [2.0, 0.5]);
    const result = op.execute(ctx);
    expect(result.length).toBeGreaterThan(0);
  });

  it("multi field cost scales with field count", () => {
    const op2 = new MultiFieldSearchOperator(["title", "body"], "learning");
    const op3 = new MultiFieldSearchOperator(["title", "body", "abstract"], "learning");
    const stats = new IndexStats(100);
    expect(op3.costEstimate(stats)).toBeGreaterThan(op2.costEstimate(stats));
  });
});

// -- TestMultiFieldQueryBuilder --

describe("TestMultiFieldQueryBuilder", () => {
  async function makeEngine(): Promise<Engine> {
    const e = new Engine();
    await e.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT, body TEXT)");
    await e.sql("CREATE INDEX idx_docs_fts ON docs USING gin (title, body)");
    e.insert("docs", 1, { title: "machine learning", body: "algorithms for ML" });
    return e;
  }

  it("query builder multi field", async () => {
    const e = await makeEngine();
    const result = e
      .query("docs")
      .scoreMultiFieldBayesian("learning", ["title", "body"])
      .execute();
    expect(result.length).toBeGreaterThan(0);
  });

  it("query builder multi field with weights", async () => {
    const e = await makeEngine();
    const result = e
      .query("docs")
      .scoreMultiFieldBayesian("learning", ["title", "body"], [2.0, 0.5])
      .execute();
    expect(result.length).toBeGreaterThan(0);
  });
});
