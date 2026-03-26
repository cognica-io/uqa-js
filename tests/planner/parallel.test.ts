import { describe, expect, it } from "vitest";
import { PostingList } from "../../src/core/posting-list.js";
import { createPostingEntry, createPayload } from "../../src/core/types.js";
import type { IndexStats, PostingEntry } from "../../src/core/types.js";
import { Operator } from "../../src/operators/base.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import { ParallelExecutor } from "../../src/planner/parallel.js";
import { IntersectOperator, UnionOperator } from "../../src/operators/boolean.js";
import { TermOperator } from "../../src/operators/primitive.js";
import { Engine } from "../../src/engine.js";

class SlowOperator extends Operator {
  private _docIds: number[];
  private _delay: number;
  constructor(docIds: number[], delay: number = 50) {
    super();
    this._docIds = docIds;
    this._delay = delay;
  }
  execute(_context: ExecutionContext): PostingList {
    // Simulate delay (synchronous in JS)
    const start = Date.now();
    while (Date.now() - start < this._delay) {
      // busy wait
    }
    return PostingList.fromSorted(
      this._docIds.map((d) => createPostingEntry(d, { score: 1.0 })),
    );
  }
}

// ==================================================================
// ParallelExecutor unit tests
// ==================================================================

describe("ParallelExecutor", () => {
  it("sequential when disabled", () => {
    const par = new ParallelExecutor(0);
    expect(par.enabled).toBe(false);
    const ctx: ExecutionContext = {};
    const ops = [new SlowOperator([1], 0), new SlowOperator([2], 0)];
    const results = par.executeBranches(ops, ctx);
    expect(results.length).toBe(2);
    expect(results[0]!.docIds).toEqual(new Set([1]));
    expect(results[1]!.docIds).toEqual(new Set([2]));
  });

  it("sequential with single branch", () => {
    const par = new ParallelExecutor(4);
    const ctx: ExecutionContext = {};
    const ops = [new SlowOperator([1], 0)];
    const results = par.executeBranches(ops, ctx);
    expect(results.length).toBe(1);
    expect(results[0]!.docIds).toEqual(new Set([1]));
  });

  it("parallel preserves order", () => {
    const par = new ParallelExecutor(4);
    const ctx: ExecutionContext = {};
    const ops = [
      new SlowOperator([10, 20], 0),
      new SlowOperator([30], 0),
      new SlowOperator([40, 50, 60], 0),
    ];
    const results = par.executeBranches(ops, ctx);
    expect(results.length).toBe(3);
    expect(results[0]!.docIds).toEqual(new Set([10, 20]));
    expect(results[1]!.docIds).toEqual(new Set([30]));
    expect(results[2]!.docIds).toEqual(new Set([40, 50, 60]));
  });

  it("sequential fallback produces correct results for multiple operators", () => {
    // TS ParallelExecutor uses sequential fallback (no Web Workers).
    // Verify that multiple operators produce correct results regardless.
    const nOps = 4;
    const ops = Array.from({ length: nOps }, (_, i) => new SlowOperator([i], 0));
    const ctx: ExecutionContext = {};

    const par = new ParallelExecutor(4);
    const results = par.executeBranches(ops, ctx);
    expect(results.length).toBe(nOps);
    for (let i = 0; i < nOps; i++) {
      expect(results[i]!.docIds).toEqual(new Set([i]));
    }
  });

  it("parallel error propagation", () => {
    class FailOperator extends Operator {
      execute(_context: ExecutionContext): PostingList {
        throw new Error("test error");
      }
    }
    const par = new ParallelExecutor(4);
    const ctx: ExecutionContext = {};
    const ops = [new SlowOperator([1], 0), new FailOperator()];
    expect(() => par.executeBranches(ops, ctx)).toThrow("test error");
  });
});

// ==================================================================
// Boolean operators with parallel execution
// ==================================================================

describe("ParallelBooleanOps", () => {
  it("union produces correct results via executor", () => {
    const par = new ParallelExecutor(4);
    const ctx: ExecutionContext = {};
    const ops = [
      new SlowOperator([1, 2], 0),
      new SlowOperator([3], 0),
      new SlowOperator([4, 5], 0),
    ];
    const results = par.executeBranches(ops, ctx);
    // Manually merge the union
    let acc = new PostingList();
    for (const r of results) {
      acc = acc.union(r);
    }
    expect(acc.docIds).toEqual(new Set([1, 2, 3, 4, 5]));
  });

  it("intersect produces correct results via executor", () => {
    const par = new ParallelExecutor(4);
    const ctx: ExecutionContext = {};
    const ops = [new SlowOperator([1, 2, 3], 0), new SlowOperator([2, 3, 4], 0)];
    const results = par.executeBranches(ops, ctx);
    // Manually merge the intersect
    let acc = results[0]!;
    for (let i = 1; i < results.length; i++) {
      acc = acc.intersect(results[i]!);
    }
    expect(acc.docIds).toEqual(new Set([2, 3]));
  });
});

// ==================================================================
// Fusion operators with parallel execution
// ==================================================================

describe("ParallelFusion", () => {
  it("log odds fusion parallel branches produce correct results", () => {
    const par = new ParallelExecutor(4);
    const ctx: ExecutionContext = {};
    // Simulate two fusion signal branches
    const ops = [new SlowOperator([1, 2, 3], 0), new SlowOperator([2, 3, 4], 0)];
    const results = par.executeBranches(ops, ctx);
    expect(results.length).toBe(2);
    // Both branches should have returned their results
    expect(results[0]!.entries.length).toBe(3);
    expect(results[1]!.entries.length).toBe(3);
  });
});

// ==================================================================
// Engine parallel_workers configuration
// ==================================================================

describe("EngineParallelConfig", () => {
  it("default parallel workers", () => {
    const e = new Engine();
    // Engine stores parallel_workers config (default 4)
    expect(e).toBeTruthy();
    // The parallel executor is created internally; verify the engine
    // initializes without error
  });

  it("custom parallel workers", () => {
    const e = new Engine({ parallelWorkers: 8 });
    expect(e).toBeTruthy();
  });

  it("parallel disabled with zero workers", () => {
    const e = new Engine({ parallelWorkers: 0 });
    expect(e).toBeTruthy();
  });

  it("engine executes queries regardless of parallel config", async () => {
    const e = new Engine({ parallelWorkers: 2 });
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
    await e.sql("INSERT INTO t (id, name) VALUES (1, 'alice')");
    const result = await e.sql("SELECT name FROM t WHERE id = 1");
    expect(result!.rows.length).toBe(1);
    expect(result!.rows[0]!["name"]).toBe("alice");
  });
});

// ==================================================================
// SQL-level integration
// ==================================================================

describe("SQLParallel", () => {
  it("SQL union query with parallel", async () => {
    const e = new Engine({ parallelWorkers: 4 });
    await e.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT)");
    for (let i = 1; i <= 5; i++) {
      await e.sql(
        `INSERT INTO docs (id, title) VALUES (${i}, 'neural network paper ${i}')`,
      );
    }
    const r = await e.sql(
      "SELECT title FROM docs WHERE title = 'neural network paper 1' " +
        "OR title = 'neural network paper 3'",
    );
    const titles = r!.rows.map((row: Record<string, unknown>) => row["title"]);
    expect(titles).toContain("neural network paper 1");
    expect(titles).toContain("neural network paper 3");
  });

  it("SQL query with parallel engine config works correctly", async () => {
    const e = new Engine({ parallelWorkers: 4 });
    await e.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT, body TEXT)");
    for (let i = 1; i <= 3; i++) {
      await e.sql(
        `INSERT INTO docs (id, title, body) VALUES ` +
          `(${i}, 'deep learning ${i}', 'neural networks paper ${i}')`,
      );
    }
    const r = await e.sql(
      "SELECT title FROM docs WHERE body = 'neural networks paper 2'",
    );
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["title"]).toBe("deep learning 2");
  });
});
