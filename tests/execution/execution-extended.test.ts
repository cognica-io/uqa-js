import { describe, expect, it } from "vitest";
import { Batch } from "../../src/execution/batch.js";
import { SeqScanOp, PostingListScanOp } from "../../src/execution/scan.js";
import {
  FilterOp,
  ProjectOp,
  SortOp,
  LimitOp,
  HashAggOp,
  DistinctOp,
} from "../../src/execution/relational.js";
import type { SortKey, AggregateSpec } from "../../src/execution/relational.js";
import type { PhysicalOperator } from "../../src/execution/physical.js";
import { PostingList } from "../../src/core/posting-list.js";
import { MemoryDocumentStore } from "../../src/storage/document-store.js";
import {
  Equals,
  GreaterThan,
  LessThan,
  Between,
  InSet,
  createPayload,
} from "../../src/core/types.js";
import { Table, createColumnDef } from "../../src/sql/table.js";

// ======================================================================
// Helpers
// ======================================================================

function makeTable(
  rows: Record<string, unknown>[],
  name = "t",
): { table: Table; store: MemoryDocumentStore } {
  const columns = [
    createColumnDef("id", "integer", {
      pythonType: "number",
      primaryKey: true,
      autoIncrement: true,
    }),
    createColumnDef("name", "text", { pythonType: "string" }),
    createColumnDef("age", "integer", { pythonType: "number" }),
    createColumnDef("score", "float", { pythonType: "number" }),
  ];
  const table = new Table(name, columns);
  for (const row of rows) {
    table.insert(row);
  }
  return { table, store: table.documentStore as MemoryDocumentStore };
}

function collectRows(op: PhysicalOperator): Record<string, unknown>[] {
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
// SeqScanOp
// ======================================================================

describe("SeqScanOp", () => {
  it("scans empty table", () => {
    const { store } = makeTable([]);
    const rows = collectRows(new SeqScanOp(store));
    expect(rows).toEqual([]);
  });

  it("scans all rows", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
      { name: "Carol", age: 35, score: 9.0 },
    ]);
    const rows = collectRows(new SeqScanOp(store));
    expect(rows.length).toBe(3);
    expect(rows[0]!["name"]).toBe("Alice");
    expect(rows[1]!["name"]).toBe("Bob");
    expect(rows[2]!["name"]).toBe("Carol");
  });

  it("includes _docId", () => {
    const { store } = makeTable([{ name: "Alice", age: 30, score: 9.5 }]);
    const rows = collectRows(new SeqScanOp(store));
    expect(rows[0]!["_docId"]).toBeDefined();
  });

  it("respects batch size", () => {
    const { store } = makeTable(
      Array.from({ length: 5 }, (_, i) => ({
        name: `user${String(i)}`,
        age: 20 + i,
        score: i,
      })),
    );
    const op = new SeqScanOp(store, null, 2);
    op.open();

    const batch1 = op.next();
    expect(batch1).not.toBeNull();
    expect(batch1!.length).toBe(2);

    const batch2 = op.next();
    expect(batch2).not.toBeNull();
    expect(batch2!.length).toBe(2);

    const batch3 = op.next();
    expect(batch3).not.toBeNull();
    expect(batch3!.length).toBe(1);

    expect(op.next()).toBeNull();
    op.close();
  });

  it("supports reopen", () => {
    const { store } = makeTable([{ name: "Alice", age: 30, score: 9.5 }]);
    const op = new SeqScanOp(store);
    const rows1 = collectRows(op);
    const rows2 = collectRows(op);
    expect(rows1.length).toBe(rows2.length);
    expect(rows1[0]!["name"]).toBe(rows2[0]!["name"]);
  });
});

// ======================================================================
// PostingListScanOp
// ======================================================================

describe("PostingListScanOp", () => {
  it("basic scan with document store enrichment", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { name: "Alice", age: 30 });
    store.put(2, { name: "Bob", age: 25 });

    const pl = new PostingList([
      { docId: 1, payload: createPayload({ score: 0.9 }) },
      { docId: 2, payload: createPayload({ score: 0.8 }) },
    ]);

    const rows = collectRows(new PostingListScanOp(pl, store));
    expect(rows.length).toBe(2);
    expect(rows[0]!["_docId"]).toBe(1);
    expect(rows[0]!["_score"]).toBe(0.9);
    expect(rows[0]!["name"]).toBe("Alice");
    expect(rows[1]!["_docId"]).toBe(2);
    expect(rows[1]!["_score"]).toBe(0.8);
  });

  it("preserves payload fields", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { name: "Alice" });

    const pl = new PostingList([
      {
        docId: 1,
        payload: createPayload({ score: 0.5, fields: { _extra: "data" } }),
      },
    ]);

    const rows = collectRows(new PostingListScanOp(pl, store));
    expect(rows[0]!["_extra"]).toBe("data");
  });

  it("handles empty posting list", () => {
    const store = new MemoryDocumentStore();
    const pl = new PostingList([]);
    const rows = collectRows(new PostingListScanOp(pl, store));
    expect(rows).toEqual([]);
  });

  it("respects batch size", () => {
    const store = new MemoryDocumentStore();
    const entries: { docId: number; payload: ReturnType<typeof createPayload> }[] = [];
    for (let i = 0; i < 5; i++) {
      store.put(i, { x: i });
      entries.push({ docId: i, payload: createPayload({ score: i }) });
    }
    const pl = new PostingList(entries);

    const op = new PostingListScanOp(pl, store, null, 2);
    op.open();

    const batch1 = op.next();
    expect(batch1!.length).toBe(2);
    const batch2 = op.next();
    expect(batch2!.length).toBe(2);
    const batch3 = op.next();
    expect(batch3!.length).toBe(1);
    expect(op.next()).toBeNull();
    op.close();
  });
});

// ======================================================================
// FilterOp
// ======================================================================

describe("FilterOp", () => {
  it("filters with Equals", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
      { name: "Carol", age: 30, score: 9.0 },
    ]);
    const op = new FilterOp(new SeqScanOp(store), "age", new Equals(30));
    const rows = collectRows(op);
    expect(rows.length).toBe(2);
    expect(rows.every((r) => r["age"] === 30)).toBe(true);
  });

  it("filters with GreaterThan", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
      { name: "Carol", age: 35, score: 9.0 },
    ]);
    const op = new FilterOp(new SeqScanOp(store), "age", new GreaterThan(28));
    const rows = collectRows(op);
    expect(rows.length).toBe(2);
    const names = new Set(rows.map((r) => r["name"]));
    expect(names.has("Alice")).toBe(true);
    expect(names.has("Carol")).toBe(true);
  });

  it("filters with Between", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
      { name: "Carol", age: 35, score: 9.0 },
    ]);
    const op = new FilterOp(new SeqScanOp(store), "age", new Between(26, 34));
    const rows = collectRows(op);
    expect(rows.length).toBe(1);
    expect(rows[0]!["name"]).toBe("Alice");
  });

  it("filters with InSet", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
      { name: "Carol", age: 35, score: 9.0 },
    ]);
    const op = new FilterOp(
      new SeqScanOp(store),
      "name",
      new InSet(["Alice", "Carol"]),
    );
    const rows = collectRows(op);
    expect(rows.length).toBe(2);
  });

  it("returns empty when no match", () => {
    const { store } = makeTable([{ name: "Alice", age: 30, score: 9.5 }]);
    const op = new FilterOp(new SeqScanOp(store), "age", new Equals(99));
    const rows = collectRows(op);
    expect(rows).toEqual([]);
  });

  it("returns empty for missing column", () => {
    const { store } = makeTable([{ name: "Alice", age: 30, score: 9.5 }]);
    const op = new FilterOp(new SeqScanOp(store), "nonexistent", new Equals(1));
    const rows = collectRows(op);
    expect(rows).toEqual([]);
  });
});

// ======================================================================
// ProjectOp
// ======================================================================

describe("ProjectOp", () => {
  it("basic projection", () => {
    const { store } = makeTable([{ name: "Alice", age: 30, score: 9.5 }]);
    const op = new ProjectOp(new SeqScanOp(store), ["name", "age"]);
    const rows = collectRows(op);
    expect(Object.keys(rows[0]!)).toContain("name");
    expect(Object.keys(rows[0]!)).toContain("age");
    expect(rows[0]!["name"]).toBe("Alice");
    expect(rows[0]!["age"]).toBe(30);
  });

  it("projection with alias", () => {
    const { store } = makeTable([{ name: "Alice", age: 30, score: 9.5 }]);
    const op = new ProjectOp(
      new SeqScanOp(store),
      ["name"],
      new Map([["name", "user_name"]]),
    );
    const rows = collectRows(op);
    expect(rows[0]!["user_name"]).toBe("Alice");
  });

  it("single column projection", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
    ]);
    const op = new ProjectOp(new SeqScanOp(store), ["name"]);
    const rows = collectRows(op);
    expect(rows.length).toBe(2);
    expect(Object.keys(rows[0]!)).toEqual(["name"]);
  });
});

// ======================================================================
// SortOp
// ======================================================================

describe("SortOp", () => {
  it("sorts ascending", () => {
    const { store } = makeTable([
      { name: "Carol", age: 35, score: 9.0 },
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
    ]);
    const keys: SortKey[] = [{ column: "age", ascending: true, nullsFirst: false }];
    const op = new SortOp(new SeqScanOp(store), keys);
    const rows = collectRows(op);
    expect(rows.map((r) => r["age"])).toEqual([25, 30, 35]);
  });

  it("sorts descending", () => {
    const { store } = makeTable([
      { name: "Carol", age: 35, score: 9.0 },
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
    ]);
    const keys: SortKey[] = [{ column: "age", ascending: false, nullsFirst: false }];
    const op = new SortOp(new SeqScanOp(store), keys);
    const rows = collectRows(op);
    expect(rows.map((r) => r["age"])).toEqual([35, 30, 25]);
  });

  it("sorts with multiple keys", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 30, score: 8.0 },
      { name: "Carol", age: 25, score: 9.0 },
    ]);
    const keys: SortKey[] = [
      { column: "age", ascending: true, nullsFirst: false },
      { column: "score", ascending: false, nullsFirst: false },
    ];
    const op = new SortOp(new SeqScanOp(store), keys);
    const rows = collectRows(op);
    expect(rows[0]!["name"]).toBe("Carol");
    expect(rows[1]!["name"]).toBe("Alice");
    expect(rows[2]!["name"]).toBe("Bob");
  });

  it("sorts with nulls last by default", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", score: 8.0 },
      { name: "Carol", age: 25, score: 9.0 },
    ]);
    const keys: SortKey[] = [{ column: "age", ascending: true, nullsFirst: false }];
    const op = new SortOp(new SeqScanOp(store), keys);
    const rows = collectRows(op);
    expect(rows[0]!["name"]).toBe("Carol");
    expect(rows[1]!["name"]).toBe("Alice");
    // Null age sorts last
    expect(rows[2]!["age"]).toBeNull();
  });

  it("emits batches of correct size", () => {
    const { store } = makeTable(
      Array.from({ length: 5 }, (_, i) => ({
        name: `user${String(i)}`,
        age: 100 - i,
        score: i,
      })),
    );
    const keys: SortKey[] = [{ column: "age", ascending: true, nullsFirst: false }];
    const op = new SortOp(new SeqScanOp(store), keys, 2);
    op.open();

    const batch1 = op.next();
    expect(batch1!.length).toBe(2);
    const batch2 = op.next();
    expect(batch2!.length).toBe(2);
    const batch3 = op.next();
    expect(batch3!.length).toBe(1);
    expect(op.next()).toBeNull();
    op.close();
  });
});

// ======================================================================
// LimitOp
// ======================================================================

describe("LimitOp", () => {
  it("basic limit", () => {
    const { store } = makeTable(
      Array.from({ length: 10 }, (_, i) => ({
        name: `user${String(i)}`,
        age: 20 + i,
        score: i,
      })),
    );
    const op = new LimitOp(new SeqScanOp(store), 3);
    const rows = collectRows(op);
    expect(rows.length).toBe(3);
  });

  it("limit larger than data", () => {
    const { store } = makeTable([{ name: "Alice", age: 30, score: 9.5 }]);
    const op = new LimitOp(new SeqScanOp(store), 100);
    const rows = collectRows(op);
    expect(rows.length).toBe(1);
  });

  it("limit zero", () => {
    const { store } = makeTable([{ name: "Alice", age: 30, score: 9.5 }]);
    const op = new LimitOp(new SeqScanOp(store), 0);
    const rows = collectRows(op);
    expect(rows).toEqual([]);
  });

  it("limit truncates batch", () => {
    const { store } = makeTable(
      Array.from({ length: 10 }, (_, i) => ({
        name: `user${String(i)}`,
        age: 20 + i,
        score: i,
      })),
    );
    const op = new LimitOp(new SeqScanOp(store, null, 5), 3);
    op.open();
    const batch = op.next();
    expect(batch).not.toBeNull();
    expect(batch!.length).toBe(3);
    expect(op.next()).toBeNull();
    op.close();
  });
});

// ======================================================================
// HashAggOp
// ======================================================================

describe("HashAggOp", () => {
  it("group by count", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
      { name: "Carol", age: 30, score: 9.0 },
    ]);
    const aggSpecs: AggregateSpec[] = [{ outputCol: "cnt", funcName: "COUNT" }];
    const op = new HashAggOp(new SeqScanOp(store), ["age"], aggSpecs);
    const rows = collectRows(op);
    const result = Object.fromEntries(rows.map((r) => [r["age"], r["cnt"]]));
    expect(result[30]).toBe(2);
    expect(result[25]).toBe(1);
  });

  it("group by sum", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
      { name: "Carol", age: 30, score: 9.0 },
    ]);
    const aggSpecs: AggregateSpec[] = [
      { outputCol: "total", funcName: "SUM", inputCol: "score" },
    ];
    const op = new HashAggOp(new SeqScanOp(store), ["age"], aggSpecs);
    const rows = collectRows(op);
    const result = Object.fromEntries(rows.map((r) => [r["age"], r["total"]]));
    expect(result[30]).toBeCloseTo(18.5);
    expect(result[25]).toBeCloseTo(8.0);
  });

  it("group by avg", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 10.0 },
      { name: "Bob", age: 30, score: 8.0 },
    ]);
    const aggSpecs: AggregateSpec[] = [
      { outputCol: "avg_score", funcName: "AVG", inputCol: "score" },
    ];
    const op = new HashAggOp(new SeqScanOp(store), ["age"], aggSpecs);
    const rows = collectRows(op);
    expect(rows[0]!["avg_score"]).toBeCloseTo(9.0);
  });

  it("group by min max", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 10.0 },
      { name: "Bob", age: 30, score: 8.0 },
      { name: "Carol", age: 30, score: 9.0 },
    ]);
    const aggSpecs: AggregateSpec[] = [
      { outputCol: "min_s", funcName: "MIN", inputCol: "score" },
      { outputCol: "max_s", funcName: "MAX", inputCol: "score" },
    ];
    const op = new HashAggOp(new SeqScanOp(store), ["age"], aggSpecs);
    const rows = collectRows(op);
    expect(rows[0]!["min_s"]).toBeCloseTo(8.0);
    expect(rows[0]!["max_s"]).toBeCloseTo(10.0);
  });

  it("aggregate only (no group by)", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
    ]);
    const aggSpecs: AggregateSpec[] = [{ outputCol: "cnt", funcName: "COUNT" }];
    const op = new HashAggOp(new SeqScanOp(store), [], aggSpecs);
    const rows = collectRows(op);
    expect(rows.length).toBe(1);
    expect(rows[0]!["cnt"]).toBe(2);
  });

  it("aggregate on empty table", () => {
    const { store } = makeTable([]);
    const aggSpecs: AggregateSpec[] = [{ outputCol: "cnt", funcName: "COUNT" }];
    const op = new HashAggOp(new SeqScanOp(store), [], aggSpecs);
    const rows = collectRows(op);
    expect(rows.length).toBe(1);
    expect(rows[0]!["cnt"]).toBe(0);
  });

  it("count with specific column skips nulls", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", score: 8.0 },
    ]);
    const aggSpecs: AggregateSpec[] = [
      { outputCol: "cnt", funcName: "COUNT", inputCol: "age" },
    ];
    const op = new HashAggOp(new SeqScanOp(store), [], aggSpecs);
    const rows = collectRows(op);
    // Only Alice has age, Bob's age is null
    expect(rows[0]!["cnt"]).toBe(1);
  });
});

// ======================================================================
// DistinctOp
// ======================================================================

describe("DistinctOp", () => {
  it("basic distinct", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
      { name: "Alice", age: 30, score: 7.0 },
    ]);
    const op = new DistinctOp(new ProjectOp(new SeqScanOp(store), ["name", "age"]), [
      "name",
      "age",
    ]);
    const rows = collectRows(op);
    expect(rows.length).toBe(2);
  });

  it("all unique", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
    ]);
    const op = new DistinctOp(new ProjectOp(new SeqScanOp(store), ["name"]), ["name"]);
    const rows = collectRows(op);
    expect(rows.length).toBe(2);
  });

  it("all same", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Alice", age: 30, score: 8.0 },
      { name: "Alice", age: 30, score: 7.0 },
    ]);
    const op = new DistinctOp(new ProjectOp(new SeqScanOp(store), ["name"]), ["name"]);
    const rows = collectRows(op);
    expect(rows.length).toBe(1);
  });
});

// ======================================================================
// Operator composition
// ======================================================================

describe("Composition", () => {
  it("scan -> filter -> project", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
      { name: "Carol", age: 35, score: 9.0 },
    ]);
    const op = new ProjectOp(
      new FilterOp(new SeqScanOp(store), "age", new GreaterThan(28)),
      ["name", "age"],
    );
    const rows = collectRows(op);
    expect(rows.length).toBe(2);
    expect(rows.every((r) => !("score" in r))).toBe(true);
    const names = new Set(rows.map((r) => r["name"]));
    expect(names).toEqual(new Set(["Alice", "Carol"]));
  });

  it("scan -> filter -> sort -> limit", () => {
    const { store } = makeTable(
      Array.from({ length: 20 }, (_, i) => ({
        name: `user${String(i)}`,
        age: 20 + i,
        score: i,
      })),
    );
    const keys: SortKey[] = [{ column: "age", ascending: false, nullsFirst: false }];
    const op = new LimitOp(
      new SortOp(new FilterOp(new SeqScanOp(store), "age", new GreaterThan(30)), keys),
      3,
    );
    const rows = collectRows(op);
    expect(rows.length).toBe(3);
    expect(rows[0]!["age"]).toBe(39);
    expect(rows[1]!["age"]).toBe(38);
    expect(rows[2]!["age"]).toBe(37);
  });

  it("scan -> group -> sort", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 10.0 },
      { name: "Bob", age: 25, score: 8.0 },
      { name: "Carol", age: 30, score: 6.0 },
      { name: "Dave", age: 25, score: 9.0 },
    ]);
    const aggSpecs: AggregateSpec[] = [
      { outputCol: "cnt", funcName: "COUNT" },
      { outputCol: "avg_score", funcName: "AVG", inputCol: "score" },
    ];
    const keys: SortKey[] = [{ column: "age", ascending: true, nullsFirst: false }];
    const op = new SortOp(new HashAggOp(new SeqScanOp(store), ["age"], aggSpecs), keys);
    const rows = collectRows(op);
    expect(rows.length).toBe(2);
    expect(rows[0]!["age"]).toBe(25);
    expect(rows[0]!["cnt"]).toBe(2);
    expect(rows[0]!["avg_score"]).toBeCloseTo(8.5);
    expect(rows[1]!["age"]).toBe(30);
  });

  it("posting list -> filter -> sort -> limit", () => {
    const store = new MemoryDocumentStore();
    for (let i = 0; i < 10; i++) {
      store.put(i, { x: i, label: i % 2 === 0 ? "even" : "odd" });
    }

    const entries = Array.from({ length: 10 }, (_, i) => ({
      docId: i,
      payload: createPayload({ score: i / 10 }),
    }));
    const pl = new PostingList(entries);

    const keys: SortKey[] = [{ column: "x", ascending: false, nullsFirst: false }];
    const op = new LimitOp(
      new SortOp(
        new FilterOp(new PostingListScanOp(pl, store), "label", new Equals("even")),
        keys,
      ),
      3,
    );
    const rows = collectRows(op);
    expect(rows.length).toBe(3);
    expect(rows.map((r) => r["x"])).toEqual([8, 6, 4]);
  });

  it("full pipeline: scan -> filter -> group -> sort -> limit", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 10.0 },
      { name: "Bob", age: 25, score: 8.0 },
      { name: "Carol", age: 30, score: 6.0 },
      { name: "Dave", age: 25, score: 9.0 },
      { name: "Eve", age: 35, score: 7.0 },
    ]);
    const aggSpecs: AggregateSpec[] = [
      { outputCol: "total", funcName: "SUM", inputCol: "score" },
    ];
    const sortKeys: SortKey[] = [
      { column: "total", ascending: false, nullsFirst: false },
    ];
    const op = new LimitOp(
      new SortOp(
        new HashAggOp(
          new FilterOp(new SeqScanOp(store), "age", new LessThan(35)),
          ["age"],
          aggSpecs,
        ),
        sortKeys,
      ),
      1,
    );
    const rows = collectRows(op);
    expect(rows.length).toBe(1);
    // age 25: 8+9=17, age 30: 10+6=16 -> top is 17
    expect(rows[0]!["total"]).toBeCloseTo(17.0);
  });

  it("distinct -> sort -> limit", () => {
    const { store } = makeTable([
      { name: "Alice", age: 30, score: 9.5 },
      { name: "Bob", age: 25, score: 8.0 },
      { name: "Alice", age: 30, score: 7.0 },
      { name: "Carol", age: 35, score: 9.0 },
      { name: "Bob", age: 25, score: 6.0 },
    ]);
    const sortKeys: SortKey[] = [{ column: "age", ascending: true, nullsFirst: false }];
    const op = new LimitOp(
      new SortOp(
        new DistinctOp(new ProjectOp(new SeqScanOp(store), ["name", "age"]), ["name"]),
        sortKeys,
      ),
      2,
    );
    const rows = collectRows(op);
    expect(rows.length).toBe(2);
    expect(rows[0]!["name"]).toBe("Bob");
    expect(rows[1]!["name"]).toBe("Alice");
  });
});

// ======================================================================
// Batch additional tests
// ======================================================================

describe("Batch extended", () => {
  it("fromRows basic", () => {
    const batch = Batch.fromRows([
      { x: 1, y: "a" },
      { x: 2, y: "b" },
    ]);
    expect(batch.length).toBe(2);
    expect(batch.columnNames).toContain("x");
    expect(batch.columnNames).toContain("y");
  });

  it("roundtrip", () => {
    const original = [
      { x: 1, y: "hello" },
      { x: 2, y: "world" },
    ];
    const batch = Batch.fromRows(original);
    expect(batch.toRows()).toEqual(original);
  });

  it("empty batch", () => {
    const batch = Batch.fromRows([]);
    expect(batch.length).toBe(0);
    expect(batch.toRows()).toEqual([]);
  });

  it("null handling", () => {
    const batch = Batch.fromRows([
      { x: 1, y: null },
      { x: null, y: "hi" },
    ]);
    const result = batch.toRows();
    expect(result[0]!["y"]).toBeNull();
    expect(result[1]!["x"]).toBeNull();
  });
});
