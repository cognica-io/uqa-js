import { describe, expect, it } from "vitest";
import { SpillManager, mergeSortedRuns } from "../../src/execution/spill.js";
import { Batch } from "../../src/execution/batch.js";
import { PhysicalOperator } from "../../src/execution/physical.js";
import { SortOp, HashAggOp, DistinctOp } from "../../src/execution/relational.js";
import type { SortKey, AggregateSpec } from "../../src/execution/relational.js";

// -- Helpers ---------------------------------------------------------------

class RowSourceOp extends PhysicalOperator {
  private _rows: Record<string, unknown>[];
  private _batchSize: number;
  private _offset: number;

  constructor(rows: Record<string, unknown>[], batchSize = 1024) {
    super();
    this._rows = rows;
    this._batchSize = batchSize;
    this._offset = 0;
  }

  open(): void {
    this._offset = 0;
  }

  next(): Batch | null {
    if (this._offset >= this._rows.length) {
      return null;
    }
    const end = Math.min(this._offset + this._batchSize, this._rows.length);
    const batchRows = this._rows.slice(this._offset, end);
    this._offset = end;
    return Batch.fromRows(batchRows);
  }

  close(): void {
    this._offset = 0;
  }
}

function drain(op: PhysicalOperator): Record<string, unknown>[] {
  op.open();
  const rows: Record<string, unknown>[] = [];
  for (;;) {
    const batch = op.next();
    if (batch === null) break;
    rows.push(...batch.toRows());
  }
  op.close();
  return rows;
}

// ======================================================================
// Spill infrastructure
// ======================================================================

describe("SpillManager", () => {
  it("lifecycle", () => {
    const mgr = new SpillManager();
    const r1 = mgr.newRun();
    const r2 = mgr.newRun();
    expect(r1).not.toBe(r2);

    mgr.writeRows(r1, [{ a: 1 }]);
    mgr.writeRows(r2, [{ a: 2 }]);

    expect(mgr.readRows(r1)).toEqual([{ a: 1 }]);
    expect(mgr.readRows(r2)).toEqual([{ a: 2 }]);

    mgr.cleanup();
  });

  it("write and read", () => {
    const mgr = new SpillManager();
    const r = mgr.newRun();
    mgr.writeRows(r, [
      { x: 1, y: "a" },
      { x: 2, y: "b" },
    ]);
    mgr.writeRows(r, [{ x: 3, y: "c" }]);

    const rows = mgr.readRows(r);
    expect(rows.length).toBe(3);
    expect(rows[0]!["x"]).toBe(1);
    expect(rows[2]!["y"]).toBe("c");
    mgr.cleanup();
  });
});

describe("MergeSortedRuns", () => {
  it("single run", () => {
    const run = [{ v: 1 }, { v: 3 }, { v: 5 }];
    const merged = mergeSortedRuns([run], [["v", true]]);
    expect(merged.map((r) => r["v"])).toEqual([1, 3, 5]);
  });

  it("two runs asc", () => {
    const r1 = [{ v: 1 }, { v: 4 }, { v: 7 }];
    const r2 = [{ v: 2 }, { v: 3 }, { v: 8 }];
    const merged = mergeSortedRuns([r1, r2], [["v", true]]);
    expect(merged.map((r) => r["v"])).toEqual([1, 2, 3, 4, 7, 8]);
  });

  it("two runs desc", () => {
    const r1 = [{ v: 7 }, { v: 4 }, { v: 1 }];
    const r2 = [{ v: 8 }, { v: 3 }, { v: 2 }];
    const merged = mergeSortedRuns([r1, r2], [["v", false]]);
    expect(merged.map((r) => r["v"])).toEqual([8, 7, 4, 3, 2, 1]);
  });

  it("nulls asc", () => {
    const r1 = [{ v: 1 }, { v: null }];
    const r2 = [{ v: 2 }];
    const merged = mergeSortedRuns([r1, r2], [["v", true]]);
    const vals = merged.map((r) => r["v"]);
    expect(vals).toEqual([1, 2, null]);
  });

  it("nulls desc", () => {
    const r1 = [{ v: null }, { v: 1 }];
    const r2 = [{ v: 2 }];
    const merged = mergeSortedRuns([r1, r2], [["v", false]]);
    const vals = merged.map((r) => r["v"]);
    expect(vals).toEqual([null, 2, 1]);
  });

  it("empty runs", () => {
    const merged = mergeSortedRuns([], [["v", true]]);
    expect(merged).toEqual([]);
  });
});

// ======================================================================
// SortOp
// ======================================================================

describe("SortOp", () => {
  it("small input sorted ascending", () => {
    const rows = [{ v: 3 }, { v: 1 }, { v: 2 }];
    const op = new SortOp(new RowSourceOp(rows), [
      { column: "v", ascending: true, nullsFirst: false },
    ]);
    const result = drain(op);
    expect(result.map((r) => r["v"])).toEqual([1, 2, 3]);
  });

  it("larger input sorted ascending", () => {
    const rows = Array.from({ length: 20 }, (_, i) => ({ v: 20 - i }));
    const op = new SortOp(new RowSourceOp(rows), [
      { column: "v", ascending: true, nullsFirst: false },
    ]);
    const result = drain(op);
    expect(result.map((r) => r["v"])).toEqual(
      Array.from({ length: 20 }, (_, i) => i + 1),
    );
  });

  it("descending", () => {
    const rows = Array.from({ length: 50 }, (_, i) => ({ v: i + 1 }));
    const op = new SortOp(new RowSourceOp(rows), [
      { column: "v", ascending: false, nullsFirst: false },
    ]);
    const result = drain(op);
    expect(result.map((r) => r["v"])).toEqual(
      Array.from({ length: 50 }, (_, i) => 50 - i),
    );
  });

  it("multi key sort", () => {
    const rows = [
      { a: 2, b: 1 },
      { a: 1, b: 2 },
      { a: 1, b: 1 },
      { a: 2, b: 2 },
    ];
    const op = new SortOp(new RowSourceOp(rows), [
      { column: "a", ascending: true, nullsFirst: false },
      { column: "b", ascending: true, nullsFirst: false },
    ]);
    const result = drain(op);
    expect(result.map((r) => [r["a"], r["b"]])).toEqual([
      [1, 1],
      [1, 2],
      [2, 1],
      [2, 2],
    ]);
  });

  it("with nulls", () => {
    const rows = [{ v: 3 }, { v: null }, { v: 1 }];
    const op = new SortOp(new RowSourceOp(rows), [
      { column: "v", ascending: true, nullsFirst: false },
    ]);
    const result = drain(op);
    // Nulls last by default
    expect(result.map((r) => r["v"])).toEqual([1, 3, null]);
  });

  it("empty input", () => {
    const op = new SortOp(new RowSourceOp([]), [
      { column: "v", ascending: true, nullsFirst: false },
    ]);
    const result = drain(op);
    expect(result).toEqual([]);
  });
});

// ======================================================================
// HashAggOp
// ======================================================================

describe("HashAggOp", () => {
  it("small input", () => {
    const rows = [
      { g: "a", v: 1 },
      { g: "b", v: 2 },
      { g: "a", v: 3 },
    ];
    const op = new HashAggOp(
      new RowSourceOp(rows),
      ["g"],
      [{ outputCol: "total", funcName: "SUM", inputCol: "v" }],
    );
    const result = drain(op);
    const byG: Record<string, number> = {};
    for (const r of result) {
      byG[r["g"] as string] = r["total"] as number;
    }
    expect(byG).toEqual({ a: 4, b: 2 });
  });

  it("group by with count and sum", () => {
    const rows = Array.from({ length: 100 }, (_, i) => ({
      g: i % 5,
      v: i,
    }));
    const op = new HashAggOp(
      new RowSourceOp(rows),
      ["g"],
      [
        { outputCol: "cnt", funcName: "COUNT" },
        { outputCol: "total", funcName: "SUM", inputCol: "v" },
      ],
    );
    const result = drain(op);
    expect(result.length).toBe(5);
    const byG: Record<number, Record<string, unknown>> = {};
    for (const r of result) {
      byG[r["g"] as number] = r;
    }
    for (let g = 0; g < 5; g++) {
      expect(byG[g]!["cnt"]).toBe(20);
      const expectedSum = Array.from({ length: 100 }, (_, i) => i)
        .filter((i) => i % 5 === g)
        .reduce((a, b) => a + b, 0);
      expect(byG[g]!["total"]).toBe(expectedSum);
    }
  });

  it("aggregate only (no group by)", () => {
    const rows = Array.from({ length: 50 }, (_, i) => ({ v: i }));
    const op = new HashAggOp(
      new RowSourceOp(rows),
      [],
      [{ outputCol: "cnt", funcName: "COUNT" }],
    );
    const result = drain(op);
    expect(result.length).toBe(1);
    expect(result[0]!["cnt"]).toBe(50);
  });

  it("empty input", () => {
    const op = new HashAggOp(
      new RowSourceOp([]),
      [],
      [{ outputCol: "cnt", funcName: "COUNT" }],
    );
    const result = drain(op);
    expect(result.length).toBe(1);
    expect(result[0]!["cnt"]).toBe(0);
  });
});

// ======================================================================
// DistinctOp
// ======================================================================

describe("DistinctOp", () => {
  it("small input", () => {
    const rows = [{ v: 1 }, { v: 2 }, { v: 1 }, { v: 3 }];
    const op = new DistinctOp(new RowSourceOp(rows), ["v"]);
    const result = drain(op);
    const vals = result.map((r) => r["v"] as number).sort((a, b) => a - b);
    expect(vals).toEqual([1, 2, 3]);
  });

  it("dedup larger input", () => {
    const rows = Array.from({ length: 200 }, (_, i) => ({ v: i % 10 }));
    const op = new DistinctOp(new RowSourceOp(rows), ["v"]);
    const result = drain(op);
    const vals = result.map((r) => r["v"] as number).sort((a, b) => a - b);
    expect(vals).toEqual(Array.from({ length: 10 }, (_, i) => i));
  });

  it("multi column", () => {
    const rows = Array.from({ length: 100 }, (_, i) => ({
      a: i % 3,
      b: i % 5,
    }));
    const op = new DistinctOp(new RowSourceOp(rows), ["a", "b"]);
    const result = drain(op);
    const keys = result
      .map((r) => [r["a"] as number, r["b"] as number] as [number, number])
      .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const expected = Array.from(
      new Set(
        Array.from({ length: 15 }, (_, i) => `${String(i % 3)},${String(i % 5)}`),
      ),
    )
      .map((s) => s.split(",").map(Number) as [number, number])
      .sort((a, b) => a[0]! - b[0]! || a[1]! - b[1]!);
    expect(keys).toEqual(expected);
  });

  it("empty input", () => {
    const op = new DistinctOp(new RowSourceOp([]), ["v"]);
    const result = drain(op);
    expect(result).toEqual([]);
  });
});

// ======================================================================
// SQL integration with spill_threshold (Engine not yet ported)
// ======================================================================

describe("SQLSpillIntegration", () => {
  it("sort via SQL", async () => {
    const { Engine } = await import("../../src/engine.js");
    const engine = new Engine({ spillThreshold: 10 });
    await engine.sql("CREATE TABLE t (id SERIAL PRIMARY KEY, v INTEGER)");
    for (let i = 30; i >= 1; i--) {
      await engine.sql(`INSERT INTO t (v) VALUES (${String(i)})`);
    }
    const result = await engine.sql("SELECT v FROM t ORDER BY v ASC");
    const rows = (result as { rows: Record<string, unknown>[] }).rows;
    expect(rows.map((r) => r["v"])).toEqual(
      Array.from({ length: 30 }, (_, i) => i + 1),
    );
    engine.close();
  });

  it("groupby via SQL", async () => {
    const { Engine } = await import("../../src/engine.js");
    const engine = new Engine({ spillThreshold: 10 });
    await engine.sql("CREATE TABLE t (id SERIAL PRIMARY KEY, g INTEGER, v INTEGER)");
    for (let i = 0; i < 50; i++) {
      await engine.sql(`INSERT INTO t (g, v) VALUES (${String(i % 5)}, ${String(i)})`);
    }
    const result = await engine.sql(
      "SELECT g, COUNT(*) AS cnt FROM t GROUP BY g ORDER BY g",
    );
    const rows = (result as { rows: Record<string, unknown>[] }).rows;
    expect(rows.length).toBe(5);
    for (const r of rows) {
      expect(r["cnt"]).toBe(10);
    }
    engine.close();
  });

  it("distinct via SQL", async () => {
    const { Engine } = await import("../../src/engine.js");
    const engine = new Engine({ spillThreshold: 10 });
    await engine.sql("CREATE TABLE t (id SERIAL PRIMARY KEY, v INTEGER)");
    for (let i = 0; i < 40; i++) {
      await engine.sql(`INSERT INTO t (v) VALUES (${String(i % 8)})`);
    }
    const result = await engine.sql("SELECT DISTINCT v FROM t ORDER BY v");
    const rows = (result as { rows: Record<string, unknown>[] }).rows;
    expect(rows.map((r) => r["v"])).toEqual(Array.from({ length: 8 }, (_, i) => i));
    engine.close();
  });
});
