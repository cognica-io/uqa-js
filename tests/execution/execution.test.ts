import { describe, expect, it } from "vitest";
import { Batch } from "../../src/execution/batch.js";
import { SpillManager, mergeSortedRuns } from "../../src/execution/spill.js";

describe("Batch", () => {
  it("creates from rows", () => {
    const b = Batch.fromRows([
      { id: 1, name: "Alice" },
      { id: 2, name: "Bob" },
    ]);
    expect(b.length).toBe(2);
    expect(b.columnNames).toContain("id");
    expect(b.columnNames).toContain("name");
  });

  it("converts back to rows", () => {
    const rows = [
      { id: 1, name: "Alice" },
      { id: 2, name: "Bob" },
    ];
    const b = Batch.fromRows(rows);
    expect(b.toRows()).toEqual(rows);
  });

  it("column access", () => {
    const b = Batch.fromRows([{ x: 10 }, { x: 20 }]);
    expect(b.column("x")).toEqual([10, 20]);
  });

  it("getColumn returns null for missing", () => {
    const b = Batch.fromRows([{ x: 1 }]);
    expect(b.getColumn("missing")).toBeNull();
  });

  it("slice works", () => {
    const b = Batch.fromRows([{ x: 1 }, { x: 2 }, { x: 3 }]);
    const s = b.slice(1, 2);
    expect(s.length).toBe(2);
    expect(s.column("x")).toEqual([2, 3]);
  });

  it("selectColumns with aliases", () => {
    const b = Batch.fromRows([{ id: 1, name: "Alice", age: 30 }]);
    const s = b.selectColumns(["id", "name"], new Map([["name", "n"]]));
    expect(s.columnNames).toContain("id");
    expect(s.columnNames).toContain("n");
    expect(s.column("n")).toEqual(["Alice"]);
  });

  it("take by indices", () => {
    const b = Batch.fromRows([{ x: 10 }, { x: 20 }, { x: 30 }]);
    const t = b.take([2, 0]);
    expect(t.length).toBe(2);
    expect(t.column("x")).toEqual([30, 10]);
  });
});

describe("SpillManager", () => {
  it("stores and retrieves runs", () => {
    const sm = new SpillManager();
    const idx = sm.newRun();
    sm.writeRows(idx, [{ a: 1 }, { a: 2 }]);
    expect(sm.readRows(idx)).toEqual([{ a: 1 }, { a: 2 }]);
    sm.cleanup();
  });
});

describe("mergeSortedRuns", () => {
  it("merges sorted runs", () => {
    const runs = [
      [{ x: 1 }, { x: 3 }, { x: 5 }],
      [{ x: 2 }, { x: 4 }, { x: 6 }],
    ];
    const merged = mergeSortedRuns(runs, [["x", true]]);
    expect(merged.map((r) => r["x"])).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("handles descending sort", () => {
    const runs = [
      [{ x: 5 }, { x: 3 }, { x: 1 }],
      [{ x: 6 }, { x: 4 }, { x: 2 }],
    ];
    const merged = mergeSortedRuns(runs, [["x", false]]);
    expect(merged.map((r) => r["x"])).toEqual([6, 5, 4, 3, 2, 1]);
  });
});
