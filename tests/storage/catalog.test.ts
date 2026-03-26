import { describe, expect, it } from "vitest";
import initSqlJs from "sql.js";
import { Catalog } from "../../src/storage/catalog.js";
import { ManagedConnection } from "../../src/storage/managed-connection.js";

// Helper: create an in-memory sql.js database + ManagedConnection + Catalog
async function makeCatalog(): Promise<{
  conn: ManagedConnection;
  catalog: Catalog;
}> {
  const SQL = await initSqlJs();
  const db = new SQL.Database();
  const conn = new ManagedConnection(db);
  const catalog = new Catalog(conn);
  return { conn, catalog };
}

// ======================================================================
// Catalog metadata
// ======================================================================

describe("CatalogMetadata", () => {
  it("set and get", async () => {
    const { catalog } = await makeCatalog();
    catalog.setMetadata("key1", "value1");
    expect(catalog.getMetadata("key1")).toBe("value1");
  });

  it("get missing returns null", async () => {
    const { catalog } = await makeCatalog();
    expect(catalog.getMetadata("nonexistent")).toBeNull();
  });

  it("overwrite", async () => {
    const { catalog } = await makeCatalog();
    catalog.setMetadata("k", "v1");
    catalog.setMetadata("k", "v2");
    expect(catalog.getMetadata("k")).toBe("v2");
  });
});

// ======================================================================
// Table schemas
// ======================================================================

describe("CatalogTableSchemas", () => {
  it("save and load round trip", async () => {
    const { catalog } = await makeCatalog();
    const cols = [
      {
        name: "id",
        type_name: "serial",
        primary_key: true,
        not_null: true,
        auto_increment: true,
        default: null,
      },
      {
        name: "title",
        type_name: "text",
        primary_key: false,
        not_null: true,
        auto_increment: false,
        default: null,
      },
    ];
    catalog.saveTableSchema("papers", cols);
    const schemas = catalog.loadTableSchemas();
    expect(schemas.length).toBe(1);
    const [name, loadedCols] = schemas[0]!;
    expect(name).toBe("papers");
    expect(loadedCols).toEqual(cols);
  });

  it("drop removes schema", async () => {
    const { catalog } = await makeCatalog();
    catalog.saveTableSchema("t1", [
      {
        name: "a",
        type_name: "int",
        primary_key: false,
        not_null: false,
        auto_increment: false,
        default: null,
      },
    ]);
    catalog.dropTableSchema("t1");
    expect(catalog.loadTableSchemas()).toEqual([]);
  });

  it("multiple tables", async () => {
    const { catalog } = await makeCatalog();
    for (const name of ["t1", "t2", "t3"]) {
      catalog.saveTableSchema(name, [
        {
          name: "x",
          type_name: "int",
          primary_key: false,
          not_null: false,
          auto_increment: false,
          default: null,
        },
      ]);
    }
    const schemas = catalog.loadTableSchemas();
    const names = new Set(schemas.map((s) => s[0]));
    expect(names).toEqual(new Set(["t1", "t2", "t3"]));
  });
});

// ======================================================================
// Named graphs
// ======================================================================

describe("CatalogNamedGraphs", () => {
  it("save and load named graph", async () => {
    const { catalog } = await makeCatalog();
    catalog.saveNamedGraph("social");
    const graphs = catalog.loadNamedGraphs();
    expect(graphs).toContain("social");
  });

  it("drop named graph", async () => {
    const { catalog } = await makeCatalog();
    catalog.saveNamedGraph("tmp");
    catalog.dropNamedGraph("tmp");
    const graphs = catalog.loadNamedGraphs();
    expect(graphs).not.toContain("tmp");
  });

  it("multiple named graphs", async () => {
    const { catalog } = await makeCatalog();
    catalog.saveNamedGraph("g1");
    catalog.saveNamedGraph("g2");
    catalog.saveNamedGraph("g3");
    const graphs = catalog.loadNamedGraphs();
    expect(new Set(graphs)).toEqual(new Set(["g1", "g2", "g3"]));
  });
});

// ======================================================================
// Index persistence
// ======================================================================

describe("CatalogIndexes", () => {
  it("save and load index", async () => {
    const { catalog } = await makeCatalog();
    catalog.saveIndex({
      name: "idx_name",
      indexType: "btree",
      tableName: "users",
      columns: ["name"],
      parameters: {},
    });
    const indexes = catalog.loadIndexes();
    expect(indexes.length).toBe(1);
    expect(indexes[0]![0]).toBe("idx_name");
    expect(indexes[0]![1]).toBe("btree");
    expect(indexes[0]![2]).toBe("users");
    expect(indexes[0]![3]).toEqual(["name"]);
  });

  it("drop index", async () => {
    const { catalog } = await makeCatalog();
    catalog.saveIndex({
      name: "idx1",
      indexType: "btree",
      tableName: "t",
      columns: ["x"],
      parameters: {},
    });
    catalog.dropIndex("idx1");
    expect(catalog.loadIndexes()).toEqual([]);
  });
});

// ======================================================================
// Analyzers
// ======================================================================

describe("CatalogAnalyzers", () => {
  it("save and load analyzer", async () => {
    const { catalog } = await makeCatalog();
    catalog.saveAnalyzer("ws", { type: "whitespace", lowercase: true });
    const analyzers = catalog.loadAnalyzers();
    expect(analyzers.length).toBe(1);
    expect(analyzers[0]![0]).toBe("ws");
    expect(analyzers[0]![1]).toEqual({ type: "whitespace", lowercase: true });
  });

  it("drop analyzer", async () => {
    const { catalog } = await makeCatalog();
    catalog.saveAnalyzer("ws", { type: "whitespace" });
    catalog.dropAnalyzer("ws");
    expect(catalog.loadAnalyzers()).toEqual([]);
  });
});

// ======================================================================
// Persistence across close/reopen (same connection, schema survives)
// ======================================================================

describe("CatalogPersistence", () => {
  it("schemas survive re-instantiation on same connection", async () => {
    const SQL = await initSqlJs();
    const db = new SQL.Database();
    const conn = new ManagedConnection(db);
    const cat1 = new Catalog(conn);
    cat1.saveTableSchema("t", [
      {
        name: "x",
        type_name: "int",
        primary_key: false,
        not_null: false,
        auto_increment: false,
        default: null,
      },
    ]);
    cat1.setMetadata("ver", "1");
    cat1.saveNamedGraph("g1");

    // Re-create Catalog on same connection
    const cat2 = new Catalog(conn);
    expect(cat2.loadTableSchemas().length).toBe(1);
    expect(cat2.getMetadata("ver")).toBe("1");
    expect(cat2.loadNamedGraphs()).toContain("g1");
  });

  it("data survives binary export/import", async () => {
    const SQL = await initSqlJs();
    const db1 = new SQL.Database();
    const conn1 = new ManagedConnection(db1);
    const cat1 = new Catalog(conn1);
    cat1.saveTableSchema("papers", [{ name: "title", type_name: "text" }]);
    cat1.setMetadata("version", "42");

    // Export and reimport
    const data = db1.export();
    const db2 = new SQL.Database(data);
    const conn2 = new ManagedConnection(db2);
    const cat2 = new Catalog(conn2);
    expect(cat2.loadTableSchemas().length).toBe(1);
    expect(cat2.getMetadata("version")).toBe("42");
  });
});
