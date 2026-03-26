import { describe, expect, it } from "vitest";
import initSqlJs from "sql.js";
import { SQLiteDocumentStore } from "../../src/storage/sqlite-document-store.js";
import { ManagedConnection } from "../../src/storage/managed-connection.js";

async function makeStore(
  tableName = "t1",
  columns: [string, string][] = [
    ["id", "INTEGER"],
    ["name", "TEXT"],
    ["score", "REAL"],
    ["active", "INTEGER"],
  ],
): Promise<{ conn: ManagedConnection; store: SQLiteDocumentStore }> {
  const SQL = await initSqlJs();
  const db = new SQL.Database();
  const conn = new ManagedConnection(db);
  const store = new SQLiteDocumentStore(conn, tableName, columns);
  return { conn, store };
}

// ======================================================================
// Basic CRUD
// ======================================================================

describe("CRUD", () => {
  it("put and get", async () => {
    const { store } = await makeStore();
    store.put(1, { id: 1, name: "alice", score: 9.5, active: 1 });
    const doc = store.get(1);
    expect(doc).toEqual({ id: 1, name: "alice", score: 9.5, active: 1 });
  });

  it("get missing returns null", async () => {
    const { store } = await makeStore();
    expect(store.get(999)).toBeNull();
  });

  it("delete", async () => {
    const { store } = await makeStore();
    store.put(1, { id: 1, name: "alice" });
    store.delete(1);
    expect(store.get(1)).toBeNull();
  });

  it("delete nonexistent is noop", async () => {
    const { store } = await makeStore();
    // Should not throw
    store.delete(42);
  });

  it("overwrite same doc id", async () => {
    const { store } = await makeStore();
    store.put(1, { id: 1, name: "alice", score: 5.0 });
    store.put(1, { id: 1, name: "bob", score: 8.0 });
    const doc = store.get(1);
    expect(doc).not.toBeNull();
    expect(doc!["name"]).toBe("bob");
    expect(doc!["score"]).toBe(8.0);
  });
});

// ======================================================================
// Type handling
// ======================================================================

describe("Types", () => {
  it("integer column", async () => {
    const { store } = await makeStore("t1", [["val", "INTEGER"]]);
    store.put(1, { val: 42 });
    expect(store.getField(1, "val")).toBe(42);
  });

  it("text column", async () => {
    const { store } = await makeStore("t1", [["val", "TEXT"]]);
    store.put(1, { val: "hello" });
    expect(store.getField(1, "val")).toBe("hello");
  });

  it("real column", async () => {
    const { store } = await makeStore("t1", [["val", "REAL"]]);
    store.put(1, { val: 3.14 });
    expect(store.getField(1, "val")).toBeCloseTo(3.14);
  });

  it("boolean stored as integer", async () => {
    const { store } = await makeStore("t1", [["val", "INTEGER"]]);
    store.put(1, { val: 1 });
    store.put(2, { val: 0 });
    expect(store.getField(1, "val")).toBe(1);
    expect(store.getField(2, "val")).toBe(0);
  });
});

// ======================================================================
// NULL handling
// ======================================================================

describe("Nulls", () => {
  it("missing field stored as null", async () => {
    const { store } = await makeStore();
    store.put(1, { id: 1, name: "alice" });
    const doc = store.get(1);
    expect(doc).not.toBeNull();
    // score and active not provided -> NULL -> excluded from dict
    expect(doc!["score"]).toBeUndefined();
    expect(doc!["active"]).toBeUndefined();
  });

  it("explicit null stored as null", async () => {
    const { store } = await makeStore();
    store.put(1, { id: 1, name: null, score: null, active: null });
    const doc = store.get(1);
    expect(doc).not.toBeNull();
    // Null fields excluded from dict -- may be undefined or null depending on impl
    expect(doc!["name"] == null).toBe(true);
    expect(doc!["score"] == null).toBe(true);
  });

  it("get field null returns null or undefined", async () => {
    const { store } = await makeStore();
    store.put(1, { id: 1 });
    expect(store.getField(1, "name") == null).toBe(true);
  });
});

// ======================================================================
// docIds property
// ======================================================================

describe("DocIds", () => {
  it("empty store", async () => {
    const { store } = await makeStore();
    expect(store.docIds.size).toBe(0);
  });

  it("after inserts", async () => {
    const { store } = await makeStore();
    store.put(10, { id: 10, name: "a" });
    store.put(20, { id: 20, name: "b" });
    store.put(30, { id: 30, name: "c" });
    expect(store.docIds).toEqual(new Set([10, 20, 30]));
  });

  it("after delete", async () => {
    const { store } = await makeStore();
    store.put(1, { id: 1 });
    store.put(2, { id: 2 });
    store.delete(1);
    expect(store.docIds).toEqual(new Set([2]));
  });
});

// ======================================================================
// length
// ======================================================================

describe("Length", () => {
  it("empty", async () => {
    const { store } = await makeStore();
    expect(store.length).toBe(0);
  });

  it("after inserts", async () => {
    const { store } = await makeStore();
    store.put(1, { id: 1 });
    store.put(2, { id: 2 });
    expect(store.length).toBe(2);
  });

  it("after delete", async () => {
    const { store } = await makeStore();
    store.put(1, { id: 1 });
    store.put(2, { id: 2 });
    store.delete(1);
    expect(store.length).toBe(1);
  });
});

// ======================================================================
// maxDocId
// ======================================================================

describe("MaxDocId", () => {
  it("empty returns -1", async () => {
    const { store } = await makeStore();
    // TS implementation returns -1 for empty (Python returns 0)
    expect(store.maxDocId()).toBe(-1);
  });

  it("single row", async () => {
    const { store } = await makeStore();
    store.put(7, { id: 7 });
    expect(store.maxDocId()).toBe(7);
  });

  it("multiple rows", async () => {
    const { store } = await makeStore();
    store.put(3, { id: 3 });
    store.put(10, { id: 10 });
    store.put(5, { id: 5 });
    expect(store.maxDocId()).toBe(10);
  });

  it("after deleting max", async () => {
    const { store } = await makeStore();
    store.put(1, { id: 1 });
    store.put(5, { id: 5 });
    store.delete(5);
    expect(store.maxDocId()).toBe(1);
  });
});

// ======================================================================
// getField
// ======================================================================

describe("GetField", () => {
  it("existing field", async () => {
    const { store } = await makeStore();
    store.put(1, { id: 1, name: "alice", score: 8.0 });
    expect(store.getField(1, "name")).toBe("alice");
    expect(store.getField(1, "score")).toBe(8.0);
  });

  it("missing doc", async () => {
    const { store } = await makeStore();
    expect(store.getField(99, "name") == null).toBe(true);
  });

  it("unknown column returns null or undefined", async () => {
    const { store } = await makeStore();
    store.put(1, { id: 1 });
    expect(store.getField(1, "nonexistent") == null).toBe(true);
  });
});

// ======================================================================
// evalPath
// ======================================================================

describe("EvalPath", () => {
  it("flat single key", async () => {
    const { store } = await makeStore();
    store.put(1, { id: 1, name: "alice" });
    expect(store.evalPath(1, ["name"])).toBe("alice");
  });

  it("flat missing doc", async () => {
    const { store } = await makeStore();
    expect(store.evalPath(99, ["name"])).toBeUndefined();
  });

  it("nested path falls back to dict traversal", async () => {
    const { store } = await makeStore("t1", [["name", "TEXT"]]);
    store.put(1, { name: "alice" });
    const result = store.evalPath(1, ["name", "first"]);
    expect(result).toBeUndefined();
  });

  it("single element path uses getField", async () => {
    const { store } = await makeStore();
    store.put(1, { id: 1, score: 7.5 });
    expect(store.evalPath(1, ["score"])).toBe(7.5);
  });
});

// ======================================================================
// Table isolation (multiple tables on same connection)
// ======================================================================

describe("TableIsolation", () => {
  it("two tables independent", async () => {
    const SQL = await initSqlJs();
    const db = new SQL.Database();
    const conn = new ManagedConnection(db);
    const storeA = new SQLiteDocumentStore(conn, "alpha", [
      ["x", "INTEGER"],
      ["y", "TEXT"],
    ]);
    const storeB = new SQLiteDocumentStore(conn, "beta", [
      ["p", "REAL"],
      ["q", "TEXT"],
    ]);

    storeA.put(1, { x: 10, y: "a" });
    storeB.put(1, { p: 3.14, q: "pi" });

    expect(storeA.length).toBe(1);
    expect(storeB.length).toBe(1);

    expect(storeA.get(1)).toEqual({ x: 10, y: "a" });
    expect(storeB.get(1)).toEqual({ p: 3.14, q: "pi" });

    storeA.delete(1);
    expect(storeA.length).toBe(0);
    expect(storeB.length).toBe(1);
  });

  it("different schemas", async () => {
    const SQL = await initSqlJs();
    const db = new SQL.Database();
    const conn = new ManagedConnection(db);
    const store1 = new SQLiteDocumentStore(conn, "narrow", [["val", "INTEGER"]]);
    const store2 = new SQLiteDocumentStore(conn, "wide", [
      ["a", "TEXT"],
      ["b", "TEXT"],
      ["c", "REAL"],
      ["d", "INTEGER"],
    ]);

    store1.put(1, { val: 100 });
    store2.put(1, { a: "x", b: "y", c: 1.0, d: 0 });

    expect(store1.getField(1, "val")).toBe(100);
    expect(store2.getField(1, "c")).toBe(1.0);

    expect(store1.docIds).toEqual(new Set([1]));
    expect(store2.docIds).toEqual(new Set([1]));
  });
});

// ======================================================================
// Edge cases
// ======================================================================

describe("EdgeCases", () => {
  it("empty document", async () => {
    const { store } = await makeStore();
    store.put(1, {});
    const doc = store.get(1);
    // All columns NULL -> empty dict
    expect(doc).toEqual({});
  });

  it("large doc id", async () => {
    const { store } = await makeStore();
    const bigId = 2 ** 40;
    store.put(bigId, { id: bigId, name: "big" });
    expect(store.get(bigId)).toEqual({ id: bigId, name: "big" });
  });

  it("idempotent table creation", async () => {
    const SQL = await initSqlJs();
    const db = new SQL.Database();
    const conn = new ManagedConnection(db);
    const cols: [string, string][] = [["val", "INTEGER"]];
    const store1 = new SQLiteDocumentStore(conn, "dup", cols);
    store1.put(1, { val: 42 });
    // Re-create -- should reuse existing table
    const store2 = new SQLiteDocumentStore(conn, "dup", cols);
    expect(store2.get(1)).toEqual({ val: 42 });
  });
});
