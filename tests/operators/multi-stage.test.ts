import { describe, expect, it } from "vitest";
import { createPayload, IndexStats } from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import { Operator } from "../../src/operators/base.js";
import { MultiStageOperator } from "../../src/operators/multi-stage.js";

// -- Helpers --

class FixedScoreOperator extends Operator {
  private readonly _entries: [number, number][];

  constructor(entries: [number, number][]) {
    super();
    this._entries = entries;
  }

  execute(_context: ExecutionContext): PostingList {
    return PostingList.fromSorted(
      this._entries.map(([docId, score]) => ({
        docId,
        payload: createPayload({ score }),
      })),
    );
  }
}

// -- TestMultiStageOperator --

describe("TestMultiStageOperator", () => {
  it("single stage top k", () => {
    const op = new FixedScoreOperator([
      [1, 0.9],
      [2, 0.5],
      [3, 0.3],
    ]);
    const ms = new MultiStageOperator([[op, 2]]);
    const result = ms.execute({});
    expect(result.length).toBe(2);
  });

  it("single stage threshold", () => {
    const op = new FixedScoreOperator([
      [1, 0.9],
      [2, 0.5],
      [3, 0.3],
    ]);
    const ms = new MultiStageOperator([[op, 0.4]]);
    const result = ms.execute({});
    expect(result.length).toBe(2);
  });

  it("two stage pipeline", () => {
    const stage1 = new FixedScoreOperator([
      [1, 0.9],
      [2, 0.7],
      [3, 0.5],
      [4, 0.3],
    ]);
    const stage2 = new FixedScoreOperator([
      [1, 0.95],
      [2, 0.6],
      [3, 0.4],
    ]);
    const ms = new MultiStageOperator([
      [stage1, 3],
      [stage2, 2],
    ]);
    const result = ms.execute({});
    expect(result.length).toBe(2);
  });

  it("stage rescoring", () => {
    const stage1 = new FixedScoreOperator([
      [1, 0.5],
      [2, 0.9],
    ]);
    const stage2 = new FixedScoreOperator([
      [1, 0.95],
      [2, 0.3],
    ]);
    const ms = new MultiStageOperator([
      [stage1, 2],
      [stage2, 1],
    ]);
    const result = ms.execute({});
    expect(result.length).toBe(1);
    // Doc 1 should be the top after stage2 rescoring
    expect(result.entries[0]!.docId).toBe(1);
  });

  it("threshold stage", () => {
    const stage1 = new FixedScoreOperator([
      [1, 0.9],
      [2, 0.5],
      [3, 0.1],
    ]);
    const stage2 = new FixedScoreOperator([
      [1, 0.8],
      [2, 0.6],
      [3, 0.2],
    ]);
    const ms = new MultiStageOperator([
      [stage1, 0.3],
      [stage2, 0.5],
    ]);
    const result = ms.execute({});
    // Stage1: keeps 1 (0.9) and 2 (0.5)
    // Stage2: re-scores to 1 (0.8) and 2 (0.6), both >= 0.5
    expect(result.length).toBe(2);
  });

  it("empty after cutoff", () => {
    const stage1 = new FixedScoreOperator([
      [1, 0.3],
      [2, 0.2],
    ]);
    const ms = new MultiStageOperator([[stage1, 0.5]]);
    const result = ms.execute({});
    expect(result.length).toBe(0);
  });

  it("requires at least one stage", () => {
    expect(() => new MultiStageOperator([])).toThrow("at least one stage");
  });

  it("cost estimate cascading", () => {
    const stage1 = new FixedScoreOperator([[1, 0.9]]);
    const stage2 = new FixedScoreOperator([[1, 0.8]]);
    const ms = new MultiStageOperator([
      [stage1, 10],
      [stage2, 5],
    ]);
    const stats = new IndexStats(100);
    const cost = ms.costEstimate(stats);
    expect(cost).toBeGreaterThan(0);
  });

  it("three stages", () => {
    const s1 = new FixedScoreOperator(
      Array.from(
        { length: 7 },
        (_, i) => [i + 1, 0.9 - (i + 1) * 0.1] as [number, number],
      ),
    );
    const s2 = new FixedScoreOperator(
      Array.from(
        { length: 7 },
        (_, i) => [i + 1, 0.8 - (i + 1) * 0.05] as [number, number],
      ),
    );
    const s3 = new FixedScoreOperator([
      [1, 0.99],
      [2, 0.5],
    ]);
    const ms = new MultiStageOperator([
      [s1, 5],
      [s2, 3],
      [s3, 1],
    ]);
    const result = ms.execute({});
    expect(result.length).toBe(1);
  });
});

// -- TestMultiStageSQL --

describe("TestMultiStageSQL", () => {
  it("staged retrieval via query builder", async () => {
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
    await engine.sql(
      "INSERT INTO docs (content) VALUES ('search engine optimization')",
    );
    engine.getTable("docs");

    // staged_retrieval is available through QueryBuilder multiStage
    const s1 = engine
      .query("docs")
      .term("learning", "content")
      .scoreBayesianBm25("learning", "content");
    const s2 = engine
      .query("docs")
      .term("algorithms", "content")
      .scoreBayesianBm25("algorithms", "content");
    const result = engine
      .query("docs")
      .multiStage([
        [s1, 3],
        [s2, 1],
      ])
      .execute();
    expect(result).toBeDefined();
    engine.close();
  });

  it("staged retrieval single stage via query builder", async () => {
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

    const s1 = engine
      .query("docs")
      .term("learning", "content")
      .scoreBayesianBm25("learning", "content");
    const result = engine
      .query("docs")
      .multiStage([[s1, 2]])
      .execute();
    expect(result).toBeDefined();
    engine.close();
  });
});

// -- TestMultiStageQueryBuilder --

describe("TestMultiStageQueryBuilder", () => {
  it("query builder multi stage", async () => {
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

    const s1 = engine
      .query("docs")
      .term("learning", "content")
      .scoreBayesianBm25("learning", "content");
    const s2 = engine
      .query("docs")
      .term("algorithms", "content")
      .scoreBayesianBm25("algorithms", "content");
    const result = engine
      .query("docs")
      .multiStage([
        [s1, 2],
        [s2, 1],
      ])
      .execute();
    expect(result.length).toBeLessThanOrEqual(1);
    engine.close();
  });

  it("query builder stage requires operator", async () => {
    const { Engine } = await import("../../src/engine.js");
    const engine = new Engine();
    await engine.sql("CREATE TABLE docs (id SERIAL PRIMARY KEY, content TEXT)");
    await engine.sql(
      "INSERT INTO docs (content) VALUES ('machine learning algorithms')",
    );
    engine.getTable("docs");

    const empty = engine.query("docs");
    const s1 = engine
      .query("docs")
      .term("learning", "content")
      .scoreBayesianBm25("learning", "content");

    expect(() => {
      engine.query("docs").multiStage([
        [empty, 2],
        [s1, 1],
      ]);
    }).toThrow();
    engine.close();
  });
});
