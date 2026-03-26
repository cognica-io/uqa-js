import { describe, expect, it } from "vitest";
import initSqlJs from "sql.js";
import { SQLiteGraphStore } from "../../src/storage/sqlite-graph-store.js";
import { ManagedConnection } from "../../src/storage/managed-connection.js";
import { createVertex, createEdge } from "../../src/core/types.js";

async function makeStore(
  tableName?: string,
): Promise<{ conn: ManagedConnection; store: SQLiteGraphStore }> {
  const SQL = await initSqlJs();
  const db = new SQL.Database();
  const conn = new ManagedConnection(db);
  const store = new SQLiteGraphStore(conn, tableName);
  return { conn, store };
}

function sampleVertices() {
  return [
    createVertex(1, "", { name: "Alice", age: 30 }),
    createVertex(2, "", { name: "Bob", age: 25 }),
    createVertex(3, "", { name: "Charlie", age: 35 }),
  ];
}

function sampleEdges() {
  return [
    createEdge(1, 1, 2, "knows", { since: 2020 }),
    createEdge(2, 1, 3, "knows", { since: 2019 }),
    createEdge(3, 2, 3, "works_with", { project: "alpha" }),
  ];
}

function populate(store: SQLiteGraphStore): void {
  store.createGraph("test");
  for (const v of sampleVertices()) {
    store.addVertex(v, "test");
  }
  for (const e of sampleEdges()) {
    store.addEdge(e, "test");
  }
}

// ======================================================================
// Write-Through Persistence
// ======================================================================

describe("WriteThrough", () => {
  it("add vertex persisted", async () => {
    const { conn, store } = await makeStore();
    store.createGraph("test");
    store.addVertex(createVertex(1, "", { name: "Alice" }), "test");

    const rows = conn.query(
      `SELECT vertex_id FROM "${store["_tableName"]}_vertices" WHERE vertex_id = 1`,
    );
    expect(rows.length).toBe(1);
    expect(rows[0]!["vertex_id"]).toBe(1);
  });

  it("add edge persisted", async () => {
    const { conn, store } = await makeStore();
    store.createGraph("test");
    store.addVertex(createVertex(1, "", { name: "Alice" }), "test");
    store.addVertex(createVertex(2, "", { name: "Bob" }), "test");
    store.addEdge(createEdge(1, 1, 2, "knows", { since: 2020 }), "test");

    const rows = conn.query(
      `SELECT source_id, target_id, label FROM "${store["_tableName"]}_edges" WHERE edge_id = 1`,
    );
    expect(rows.length).toBe(1);
    expect(rows[0]!["source_id"]).toBe(1);
    expect(rows[0]!["target_id"]).toBe(2);
    expect(rows[0]!["label"]).toBe("knows");
  });

  it("in memory and sqlite consistent", async () => {
    const { conn, store } = await makeStore();
    populate(store);

    // In-memory
    expect(store.verticesInGraph("test").length).toBe(3);
    expect(store.edgesInGraph("test").length).toBe(3);

    // SQLite
    const vCount = conn.query(
      `SELECT COUNT(*) as cnt FROM "${store["_tableName"]}_vertices"`,
    );
    const eCount = conn.query(
      `SELECT COUNT(*) as cnt FROM "${store["_tableName"]}_edges"`,
    );
    expect(vCount[0]!["cnt"]).toBe(3);
    expect(eCount[0]!["cnt"]).toBe(3);
  });
});

// ======================================================================
// SQLite-backed neighbors() Queries
// ======================================================================

describe("NeighborsSQL", () => {
  it("neighbors out", async () => {
    const { store } = await makeStore();
    populate(store);

    const neighbors = store.neighbors(1, "test", null, "out");
    expect(new Set(neighbors)).toEqual(new Set([2, 3]));
  });

  it("neighbors out with label", async () => {
    const { store } = await makeStore();
    populate(store);

    const neighbors = store.neighbors(2, "test", "works_with", "out");
    expect(new Set(neighbors)).toEqual(new Set([3]));
  });

  it("neighbors in", async () => {
    const { store } = await makeStore();
    populate(store);

    const neighbors = store.neighbors(3, "test", null, "in");
    expect(new Set(neighbors)).toEqual(new Set([1, 2]));
  });

  it("neighbors in with label", async () => {
    const { store } = await makeStore();
    populate(store);

    const neighbors = store.neighbors(3, "test", "knows", "in");
    expect(new Set(neighbors)).toEqual(new Set([1]));
  });

  it("neighbors nonexistent vertex", async () => {
    const { store } = await makeStore();
    store.createGraph("test");
    expect(store.neighbors(999, "test")).toEqual([]);
  });
});

// ======================================================================
// edges_by_label()
// ======================================================================

describe("EdgesByLabel", () => {
  it("edges by label", async () => {
    const { store } = await makeStore();
    populate(store);

    const knowsEdges = store.edgesByLabel("knows", "test");
    expect(knowsEdges.length).toBe(2);
    for (const e of knowsEdges) {
      expect(e.label).toBe("knows");
    }
  });

  it("edges by label none", async () => {
    const { store } = await makeStore();
    populate(store);

    const missing = store.edgesByLabel("nonexistent", "test");
    expect(missing).toEqual([]);
  });
});

// ======================================================================
// Persistence Across Reconnection
// ======================================================================

describe("Persistence", () => {
  it("vertices survive export/import", async () => {
    const SQL = await initSqlJs();
    const db1 = new SQL.Database();
    const conn1 = new ManagedConnection(db1);
    const store1 = new SQLiteGraphStore(conn1);
    populate(store1);

    const data = db1.export();
    const db2 = new SQL.Database(data);
    const conn2 = new ManagedConnection(db2);
    const store2 = new SQLiteGraphStore(conn2);

    expect(store2.getVertex(1)).not.toBeNull();
    expect(store2.getVertex(1)!.properties["name"]).toBe("Alice");
    expect(store2.getVertex(2)).not.toBeNull();
    expect(store2.getVertex(3)).not.toBeNull();
  });

  it("edges survive export/import", async () => {
    const SQL = await initSqlJs();
    const db1 = new SQL.Database();
    const conn1 = new ManagedConnection(db1);
    const store1 = new SQLiteGraphStore(conn1);
    populate(store1);

    const data = db1.export();
    const db2 = new SQL.Database(data);
    const conn2 = new ManagedConnection(db2);
    const store2 = new SQLiteGraphStore(conn2);

    expect(store2.getEdge(1)).not.toBeNull();
    expect(store2.getEdge(1)!.label).toBe("knows");
    expect(store2.getEdge(3)).not.toBeNull();
    expect(store2.getEdge(3)!.label).toBe("works_with");
  });

  it("adjacency survives export/import", async () => {
    const SQL = await initSqlJs();
    const db1 = new SQL.Database();
    const conn1 = new ManagedConnection(db1);
    const store1 = new SQLiteGraphStore(conn1);
    populate(store1);

    const data = db1.export();
    const db2 = new SQL.Database(data);
    const conn2 = new ManagedConnection(db2);
    const store2 = new SQLiteGraphStore(conn2);

    expect(new Set(store2.neighbors(1, "test", null, "out"))).toEqual(new Set([2, 3]));
    expect(new Set(store2.neighbors(3, "test", null, "in"))).toEqual(new Set([1, 2]));
  });
});

// ======================================================================
// GraphStore Interface Compatibility
// ======================================================================

describe("InterfaceCompat", () => {
  it("get vertex", async () => {
    const { store } = await makeStore();
    populate(store);

    const v = store.getVertex(1);
    expect(v).not.toBeNull();
    expect(v!.properties["name"]).toBe("Alice");
    expect(store.getVertex(999)).toBeNull();
  });

  it("get edge", async () => {
    const { store } = await makeStore();
    populate(store);

    const e = store.getEdge(1);
    expect(e).not.toBeNull();
    expect(e!.label).toBe("knows");
    expect(store.getEdge(999)).toBeNull();
  });

  it("vertices in graph", async () => {
    const { store } = await makeStore();
    populate(store);

    const verts = store.verticesInGraph("test");
    expect(verts.length).toBe(3);
  });

  it("edges in graph", async () => {
    const { store } = await makeStore();
    populate(store);

    const edges = store.edgesInGraph("test");
    expect(edges.length).toBe(3);
  });
});
