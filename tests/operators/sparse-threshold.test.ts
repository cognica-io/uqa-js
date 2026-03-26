import { describe, expect, it } from "vitest";
import { createPayload, IndexStats } from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import { Operator } from "../../src/operators/base.js";
import { SparseThresholdOperator } from "../../src/operators/sparse.js";

// -- Helpers --

class FixedScoreOperator extends Operator {
  private readonly _entries: {
    docId: number;
    payload: ReturnType<typeof createPayload>;
  }[];

  constructor(entries: { docId: number; score: number }[]) {
    super();
    this._entries = entries.map((e) => ({
      docId: e.docId,
      payload: createPayload({ score: e.score }),
    }));
  }

  execute(_context: ExecutionContext): PostingList {
    return PostingList.fromSorted(this._entries);
  }
}

// -- TestSparseThresholdOperator --

describe("TestSparseThresholdOperator", () => {
  it("filters below threshold", () => {
    const source = new FixedScoreOperator([
      { docId: 1, score: 0.3 },
      { docId: 2, score: 0.7 },
      { docId: 3, score: 0.5 },
    ]);
    const op = new SparseThresholdOperator(source, 0.5);
    const result = op.execute({});
    expect(result.length).toBe(1);
    expect(result.entries[0]!.docId).toBe(2);
    expect(result.entries[0]!.payload.score).toBeCloseTo(0.2);
  });

  it("zero threshold keeps all positive", () => {
    const source = new FixedScoreOperator([
      { docId: 1, score: 0.1 },
      { docId: 2, score: 0.5 },
    ]);
    const op = new SparseThresholdOperator(source, 0.0);
    const result = op.execute({});
    expect(result.length).toBe(2);
  });

  it("high threshold excludes all", () => {
    const source = new FixedScoreOperator([
      { docId: 1, score: 0.3 },
      { docId: 2, score: 0.5 },
    ]);
    const op = new SparseThresholdOperator(source, 1.0);
    const result = op.execute({});
    expect(result.length).toBe(0);
  });

  it("exact threshold excluded", () => {
    const source = new FixedScoreOperator([{ docId: 1, score: 0.5 }]);
    const op = new SparseThresholdOperator(source, 0.5);
    const result = op.execute({});
    expect(result.length).toBe(0);
  });

  it("adjusted scores", () => {
    const source = new FixedScoreOperator([
      { docId: 1, score: 0.8 },
      { docId: 2, score: 0.6 },
    ]);
    const op = new SparseThresholdOperator(source, 0.3);
    const result = op.execute({});
    expect(result.length).toBe(2);
    const scores = new Map<number, number>();
    for (const e of result) {
      scores.set(e.docId, e.payload.score);
    }
    expect(scores.get(1)).toBeCloseTo(0.5);
    expect(scores.get(2)).toBeCloseTo(0.3);
  });

  it("preserves doc id order", () => {
    const source = new FixedScoreOperator([
      { docId: 1, score: 0.9 },
      { docId: 5, score: 0.8 },
      { docId: 10, score: 0.7 },
    ]);
    const op = new SparseThresholdOperator(source, 0.1);
    const result = op.execute({});
    const docIds = result.entries.map((e) => e.docId);
    expect(docIds).toEqual([1, 5, 10]);
  });

  it("cost estimate", () => {
    const source = new FixedScoreOperator([{ docId: 1, score: 0.5 }]);
    const op = new SparseThresholdOperator(source, 0.3);
    const stats = new IndexStats(100);
    const cost = op.costEstimate(stats);
    expect(cost).toBe(source.costEstimate(stats));
  });

  it("empty source", () => {
    const source = new FixedScoreOperator([]);
    const op = new SparseThresholdOperator(source, 0.5);
    const result = op.execute({});
    expect(result.length).toBe(0);
  });
});

// -- TestSparseThresholdSQL --

describe("TestSparseThresholdSQL", () => {
  it("sparse threshold via query builder from engine", async () => {
    const { Engine } = await import("../../src/engine.js");
    const engine = new Engine();
    await engine.sql("CREATE TABLE docs (id SERIAL PRIMARY KEY, content TEXT)");
    await engine.sql(
      "INSERT INTO docs (content) VALUES ('machine learning algorithms')",
    );
    await engine.sql(
      "INSERT INTO docs (content) VALUES ('deep learning neural networks')",
    );
    await engine.sql(
      "INSERT INTO docs (content) VALUES ('database indexing structures')",
    );
    engine.getTable("docs");

    // sparse_threshold is an IR-level function available through QueryBuilder
    const result = engine
      .query("docs")
      .term("learning", "content")
      .scoreBayesianBm25("learning", "content")
      .sparseThreshold(0.3)
      .execute();
    expect(result).toBeDefined();
    engine.close();
  });

  it("sparse threshold invalid args via query builder", async () => {
    const { Engine } = await import("../../src/engine.js");
    const engine = new Engine();
    await engine.sql("CREATE TABLE docs (id SERIAL PRIMARY KEY, content TEXT)");
    await engine.sql(
      "INSERT INTO docs (content) VALUES ('machine learning algorithms')",
    );
    engine.getTable("docs");

    // sparseThreshold without a source query should throw
    expect(() => {
      engine.query("docs").sparseThreshold(0.3);
    }).toThrow();
    engine.close();
  });
});

// -- TestSparseThresholdQueryBuilder --

describe("TestSparseThresholdQueryBuilder", () => {
  it("query builder sparse threshold", async () => {
    const { Engine } = await import("../../src/engine.js");
    const engine = new Engine();
    await engine.sql("CREATE TABLE docs (id SERIAL PRIMARY KEY, content TEXT)");
    await engine.sql(
      "INSERT INTO docs (content) VALUES ('machine learning algorithms')",
    );
    await engine.sql(
      "INSERT INTO docs (content) VALUES ('deep learning neural networks')",
    );
    engine.getTable("docs");

    const result = engine
      .query("docs")
      .term("learning", "content")
      .scoreBayesianBm25("learning", "content")
      .sparseThreshold(0.3)
      .execute();
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
    }
    engine.close();
  });

  it("query builder requires source", async () => {
    const { Engine } = await import("../../src/engine.js");
    const engine = new Engine();
    await engine.sql("CREATE TABLE docs (id SERIAL PRIMARY KEY, content TEXT)");
    await engine.sql(
      "INSERT INTO docs (content) VALUES ('machine learning algorithms')",
    );
    engine.getTable("docs");

    expect(() => {
      engine.query("docs").sparseThreshold(0.3);
    }).toThrow();
    engine.close();
  });
});
