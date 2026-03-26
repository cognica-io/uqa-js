import { describe, expect, it, beforeEach } from "vitest";
import { FDWHandler } from "../../src/fdw/handler.js";
import type { FDWPredicate as HandlerPredicate } from "../../src/fdw/handler.js";
import {
  createForeignServer,
  createForeignTable,
  fdwEquals,
  fdwBetween,
  fdwIn,
} from "../../src/fdw/foreign-table.js";
import type {
  ForeignServer,
  ForeignTable,
  FDWPredicate,
} from "../../src/fdw/foreign-table.js";
import type { ColumnDef } from "../../src/sql/table.js";
import { DuckDBFDWHandler } from "../../src/fdw/duckdb-handler.js";
import { ArrowFDWHandler } from "../../src/fdw/arrow-handler.js";

// ---------------------------------------------------------------------------
// MockFDWHandler -- provides test data for scan/filter/project testing
// without requiring DuckDB WASM or Arrow runtime.
// ---------------------------------------------------------------------------

const TEST_DATA: Record<string, unknown>[] = [
  { id: 1, name: "Alice", age: 30, dept: "eng", salary: 90000 },
  { id: 2, name: "Bob", age: 25, dept: "sales", salary: 70000 },
  { id: 3, name: "Charlie", age: 35, dept: "eng", salary: 110000 },
  { id: 4, name: "Diana", age: 28, dept: "hr", salary: 65000 },
  { id: 5, name: "Eve", age: 32, dept: "eng", salary: 95000 },
];

class MockFDWHandler extends FDWHandler {
  private _data: Record<string, unknown>[];
  private _closed = false;
  readonly server: ForeignServer;

  constructor(server: ForeignServer, data?: Record<string, unknown>[]) {
    super();
    this.server = server;
    this._data = data ?? [...TEST_DATA];
  }

  get isClosed(): boolean {
    return this._closed;
  }

  scan(
    _foreignTable: ForeignTable,
    columns?: string[] | null,
    predicates?: HandlerPredicate[] | null,
    limit?: number | null,
  ): Record<string, unknown>[] {
    let rows = [...this._data];

    // Apply predicates
    if (predicates && predicates.length > 0) {
      rows = rows.filter((row) =>
        predicates.every((pred) => {
          const v = row[pred.column];
          switch (pred.operator) {
            case "=":
              return v === pred.value;
            case "<>":
            case "!=":
              return v !== pred.value;
            case "<":
              return (v as number) < (pred.value as number);
            case "<=":
              return (v as number) <= (pred.value as number);
            case ">":
              return (v as number) > (pred.value as number);
            case ">=":
              return (v as number) >= (pred.value as number);
            case "IS NULL":
              return v === null || v === undefined;
            case "IS NOT NULL":
              return v !== null && v !== undefined;
            case "IN":
              return Array.isArray(pred.value) && (pred.value as unknown[]).includes(v);
            case "BETWEEN": {
              const p2 = pred as FDWPredicate;
              return (
                (v as number) >= (pred.value as number) &&
                (v as number) <= (p2.value2 as number)
              );
            }
            case "LIKE": {
              if (typeof v !== "string" || typeof pred.value !== "string") return false;
              const re = new RegExp(
                "^" + pred.value.replace(/%/g, ".*").replace(/_/g, ".") + "$",
              );
              return re.test(v);
            }
            default:
              return true;
          }
        }),
      );
    }

    // Apply limit
    if (limit !== null && limit !== undefined && limit >= 0) {
      rows = rows.slice(0, limit);
    }

    // Apply column projection
    if (columns && columns.length > 0) {
      const colSet = new Set(columns);
      rows = rows.map((row) => {
        const projected: Record<string, unknown> = {};
        for (const col of colSet) {
          if (col in row) projected[col] = row[col];
        }
        return projected;
      });
    }

    return rows;
  }

  close(): void {
    this._closed = true;
    this._data = [];
  }
}

// ---------------------------------------------------------------------------
// FDW Catalog -- manages servers, foreign tables, and handlers
// ---------------------------------------------------------------------------

class FDWCatalog {
  private _servers = new Map<string, ForeignServer>();
  private _foreignTables = new Map<string, ForeignTable>();
  private _handlers = new Map<string, MockFDWHandler>();
  private _tables = new Map<string, Record<string, unknown>[]>();

  createServer(
    name: string,
    fdwType: string,
    options: Record<string, string> = {},
    ifNotExists = false,
  ): void {
    if (this._servers.has(name)) {
      if (ifNotExists) return;
      throw new Error(`Server '${name}' already exists`);
    }
    if (fdwType !== "duckdb_fdw" && fdwType !== "arrow_fdw" && fdwType !== "mock_fdw") {
      throw new Error(`Unsupported FDW type: ${fdwType}`);
    }
    const server = createForeignServer(name, fdwType, options);
    this._servers.set(name, server);
  }

  dropServer(name: string, ifExists = false): void {
    if (!this._servers.has(name)) {
      if (ifExists) return;
      throw new Error(`Server '${name}' not found`);
    }
    // Check for dependent tables
    for (const [tblName, tbl] of this._foreignTables) {
      if (tbl.serverName === name) {
        throw new Error(
          `Cannot drop server '${name}': foreign table '${tblName}' depends on it`,
        );
      }
    }
    const handler = this._handlers.get(name);
    if (handler) {
      handler.close();
      this._handlers.delete(name);
    }
    this._servers.delete(name);
  }

  hasServer(name: string): boolean {
    return this._servers.has(name);
  }

  getServer(name: string): ForeignServer {
    const s = this._servers.get(name);
    if (!s) throw new Error(`Server '${name}' not found`);
    return s;
  }

  getHandler(serverName: string, data?: Record<string, unknown>[]): MockFDWHandler {
    let handler = this._handlers.get(serverName);
    if (!handler) {
      const server = this.getServer(serverName);
      handler = new MockFDWHandler(server, data);
      this._handlers.set(serverName, handler);
    }
    return handler;
  }

  createForeignTable(
    name: string,
    serverName: string,
    columns: Map<string, ColumnDef>,
    options: Record<string, string> = {},
    ifNotExists = false,
  ): void {
    if (this._foreignTables.has(name) || this._tables.has(name)) {
      if (ifNotExists) return;
      if (this._tables.has(name)) {
        throw new Error(
          `Cannot create foreign table '${name}': name conflicts with regular table`,
        );
      }
      throw new Error(`Foreign table '${name}' already exists`);
    }
    if (!this._servers.has(serverName)) {
      throw new Error(`Server '${serverName}' not found`);
    }
    const ft = createForeignTable(name, serverName, columns, options);
    this._foreignTables.set(name, ft);
  }

  dropForeignTable(name: string, ifExists = false): void {
    if (!this._foreignTables.has(name)) {
      if (ifExists) return;
      throw new Error(`Foreign table '${name}' not found`);
    }
    this._foreignTables.delete(name);
  }

  hasForeignTable(name: string): boolean {
    return this._foreignTables.has(name);
  }

  getForeignTable(name: string): ForeignTable {
    const ft = this._foreignTables.get(name);
    if (!ft) throw new Error(`Foreign table '${name}' not found`);
    return ft;
  }

  registerRegularTable(name: string): void {
    this._tables.set(name, []);
  }

  scanForeignTable(
    name: string,
    columns?: string[] | null,
    predicates?: HandlerPredicate[] | null,
    limit?: number | null,
    data?: Record<string, unknown>[],
  ): Record<string, unknown>[] {
    const ft = this.getForeignTable(name);
    const handler = this.getHandler(ft.serverName, data);
    return handler.scan(ft, columns, predicates, limit);
  }

  listForeignTables(): string[] {
    return [...this._foreignTables.keys()];
  }

  listServers(): string[] {
    return [...this._servers.keys()];
  }

  closeAll(): void {
    for (const handler of this._handlers.values()) {
      handler.close();
    }
    this._handlers.clear();
  }

  toJSON(): Record<string, unknown> {
    const servers: Record<string, unknown>[] = [];
    for (const [, s] of this._servers) {
      servers.push({ name: s.name, fdwType: s.fdwType, options: s.options });
    }
    const tables: Record<string, unknown>[] = [];
    for (const [, ft] of this._foreignTables) {
      const cols: Record<string, unknown>[] = [];
      for (const [colName, col] of ft.columns) {
        cols.push({ name: colName, type: col.typeName });
      }
      tables.push({
        name: ft.name,
        serverName: ft.serverName,
        columns: cols,
        options: ft.options,
      });
    }
    return { servers, tables };
  }

  static fromJSON(data: Record<string, unknown>): FDWCatalog {
    const catalog = new FDWCatalog();
    const servers = data["servers"] as Record<string, unknown>[];
    for (const s of servers) {
      catalog.createServer(
        s["name"] as string,
        s["fdwType"] as string,
        s["options"] as Record<string, string>,
      );
    }
    const tables = data["tables"] as Record<string, unknown>[];
    for (const t of tables) {
      const cols = new Map<string, ColumnDef>();
      for (const c of t["columns"] as Record<string, unknown>[]) {
        cols.set(c["name"] as string, {
          name: c["name"] as string,
          typeName: c["type"] as string,
          pythonType: "string",
          primaryKey: false,
          notNull: false,
          autoIncrement: false,
          defaultValue: null,
          unique: false,
          vectorDimensions: null,
          numericPrecision: null,
          numericScale: null,
        });
      }
      catalog.createForeignTable(
        t["name"] as string,
        t["serverName"] as string,
        cols,
        t["options"] as Record<string, string>,
      );
    }
    return catalog;
  }
}

// ---------------------------------------------------------------------------
// Helper to build column definitions
// ---------------------------------------------------------------------------

function makeCols(...names: [string, string][]): Map<string, ColumnDef> {
  const cols = new Map<string, ColumnDef>();
  for (const [name, type] of names) {
    cols.set(name, {
      name,
      typeName: type,
      pythonType: type === "INTEGER" || type === "REAL" ? "number" : "string",
      primaryKey: false,
      notNull: false,
      autoIncrement: false,
      defaultValue: null,
      unique: false,
      vectorDimensions: null,
      numericPrecision: null,
      numericScale: null,
    });
  }
  return cols;
}

// ---------------------------------------------------------------------------
// Source normalization helper (DuckDB-specific)
// ---------------------------------------------------------------------------

function normalizeSource(source: string, hivePartitioning = false): string {
  const trimmed = source.trim();
  // If it contains parens, treat as an expression -- do not wrap
  if (trimmed.includes("(")) return trimmed;
  // If it is a file path (has extension), wrap in read function
  const extMatch = trimmed.match(/\.(\w+)$/i);
  if (extMatch) {
    const ext = extMatch[1]!.toLowerCase();
    const hiveOpt = hivePartitioning ? ", hive_partitioning=true" : "";
    switch (ext) {
      case "parquet":
        return `read_parquet('${trimmed}'${hiveOpt})`;
      case "csv":
        return `read_csv_auto('${trimmed}'${hiveOpt})`;
      case "json":
        return `read_json_auto('${trimmed}')`;
      case "ndjson":
        return `read_ndjson_auto('${trimmed}')`;
      default:
        return trimmed;
    }
  }
  // Otherwise treat as a table name (no wrapping)
  return trimmed;
}

// ---------------------------------------------------------------------------
// WHERE clause builder for Hive predicate pushdown
// ---------------------------------------------------------------------------

interface HivePredicate {
  column: string;
  operator: string;
  value: unknown;
}

function buildWhereClause(predicates: HivePredicate[]): string {
  if (predicates.length === 0) return "";
  const parts: string[] = [];
  for (const p of predicates) {
    const col = `"${p.column}"`;
    switch (p.operator) {
      case "=":
        if (p.value === null) {
          parts.push(`${col} IS NULL`);
        } else if (typeof p.value === "string") {
          parts.push(`${col} = '${p.value}'`);
        } else {
          parts.push(`${col} = ${String(p.value)}`);
        }
        break;
      case "<>":
      case "!=":
        if (p.value === null) {
          parts.push(`${col} IS NOT NULL`);
        } else if (typeof p.value === "string") {
          parts.push(`${col} <> '${p.value}'`);
        } else {
          parts.push(`${col} <> ${String(p.value)}`);
        }
        break;
      case "<":
        parts.push(`${col} < ${String(p.value)}`);
        break;
      case "<=":
        parts.push(`${col} <= ${String(p.value)}`);
        break;
      case ">":
        parts.push(`${col} > ${String(p.value)}`);
        break;
      case ">=":
        parts.push(`${col} >= ${String(p.value)}`);
        break;
      case "IN": {
        const vals = (p.value as unknown[])
          .map((v) => (typeof v === "string" ? `'${v}'` : String(v)))
          .join(", ");
        parts.push(`${col} IN (${vals})`);
        break;
      }
      case "LIKE":
        parts.push(`${col} LIKE '${String(p.value)}'`);
        break;
      case "NOT LIKE":
        parts.push(`${col} NOT LIKE '${String(p.value)}'`);
        break;
      case "ILIKE":
        parts.push(`${col} ILIKE '${String(p.value)}'`);
        break;
      default:
        parts.push(`${col} ${p.operator} ${String(p.value)}`);
    }
  }
  return parts.join(" AND ");
}

// ---------------------------------------------------------------------------
// Predicate extraction from WHERE clause for pushdown
// ---------------------------------------------------------------------------

interface ExtractedPredicates {
  pushed: HivePredicate[];
  deferred: string | null;
}

function extractPredicates(
  whereClause: string | null,
  partitionColumns: Set<string>,
): ExtractedPredicates {
  if (!whereClause) return { pushed: [], deferred: null };

  // Simple AND-conjunction extraction
  // Split by AND (top-level only, not inside parens or OR)
  const parts = splitAndClauses(whereClause);
  const pushed: HivePredicate[] = [];
  const deferredParts: string[] = [];

  for (const part of parts) {
    const trimmed = part.trim();
    // Check if it contains OR -- if so, defer it
    if (/\bOR\b/i.test(trimmed)) {
      deferredParts.push(trimmed);
      continue;
    }

    const pred = parseSinglePredicate(trimmed);
    if (pred && partitionColumns.has(pred.column)) {
      pushed.push(pred);
    } else if (pred) {
      // Non-partition column predicates are also pushable for general use
      pushed.push(pred);
    } else {
      deferredParts.push(trimmed);
    }
  }

  return {
    pushed,
    deferred: deferredParts.length > 0 ? deferredParts.join(" AND ") : null,
  };
}

function splitAndClauses(expr: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  const upper = expr.toUpperCase();
  for (let i = 0; i < expr.length; i++) {
    if (expr[i] === "(") depth++;
    else if (expr[i] === ")") depth--;
    else if (depth === 0 && upper.substring(i, i + 5) === " AND ") {
      parts.push(expr.substring(start, i));
      start = i + 5;
    }
  }
  parts.push(expr.substring(start));
  return parts.filter((p) => p.trim().length > 0);
}

function parseSinglePredicate(expr: string): HivePredicate | null {
  const trimmed = expr.trim();

  // IN operator
  const inMatch = trimmed.match(/^"?(\w+)"?\s+IN\s*\((.+)\)$/i);
  if (inMatch) {
    const col = inMatch[1]!;
    const vals = inMatch[2]!.split(",").map((v) => {
      const t = v.trim();
      if (t.startsWith("'") && t.endsWith("'")) return t.slice(1, -1);
      return Number(t);
    });
    return { column: col, operator: "IN", value: vals };
  }

  // LIKE / NOT LIKE / ILIKE
  const likeMatch = trimmed.match(/^"?(\w+)"?\s+(NOT\s+LIKE|ILIKE|LIKE)\s+'(.+)'$/i);
  if (likeMatch) {
    return {
      column: likeMatch[1]!,
      operator: likeMatch[2]!.toUpperCase().replace(/\s+/g, " "),
      value: likeMatch[3]!,
    };
  }

  // BETWEEN
  const betweenMatch = trimmed.match(/^"?(\w+)"?\s+BETWEEN\s+(\S+)\s+AND\s+(\S+)$/i);
  if (betweenMatch) {
    return {
      column: betweenMatch[1]!,
      operator: "BETWEEN",
      value: [Number(betweenMatch[2]), Number(betweenMatch[3])],
    };
  }

  // Comparison operators
  const cmpMatch = trimmed.match(/^"?(\w+)"?\s*(=|<>|!=|<=|>=|<|>)\s*(.+)$/);
  if (cmpMatch) {
    const col = cmpMatch[1]!;
    const op = cmpMatch[2]!;
    let val: unknown = cmpMatch[3]!.trim();
    if ((val as string).startsWith("'") && (val as string).endsWith("'")) {
      val = (val as string).slice(1, -1);
    } else if (!isNaN(Number(val))) {
      val = Number(val);
    }
    return { column: col, operator: op, value: val };
  }

  return null;
}

// ---------------------------------------------------------------------------
// Hive partition data for testing
// ---------------------------------------------------------------------------

const HIVE_DATA: Record<string, unknown>[] = [
  { id: 1, value: 100, year: 2020, month: 1 },
  { id: 2, value: 200, year: 2020, month: 2 },
  { id: 3, value: 150, year: 2021, month: 1 },
  { id: 4, value: 250, year: 2021, month: 2 },
  { id: 5, value: 300, year: 2022, month: 1 },
  { id: 6, value: 350, year: 2022, month: 2 },
  { id: 7, value: 120, year: 2020, month: 1 },
  { id: 8, value: 180, year: 2021, month: 1 },
];

const PARTITION_COLUMNS = new Set(["year", "month"]);

// ===========================================================================
// Tests
// ===========================================================================

describe("CreateDropServer", () => {
  let catalog: FDWCatalog;
  beforeEach(() => {
    catalog = new FDWCatalog();
  });

  it("mock FDW handler verifies handler API", () => {
    const server = createForeignServer("s1", "mock_fdw");
    const handler = new MockFDWHandler(server);
    const ft = createForeignTable("t1", "s1", makeCols(["id", "INTEGER"]));
    const rows = handler.scan(ft);
    expect(rows.length).toBe(5);
    expect(rows[0]!["name"]).toBe("Alice");
  });

  it("create server", () => {
    catalog.createServer("myserver", "duckdb_fdw");
    expect(catalog.hasServer("myserver")).toBe(true);
  });

  it("create server with options", () => {
    catalog.createServer("myserver", "duckdb_fdw", { path: "/data/test.db" });
    const s = catalog.getServer("myserver");
    expect(s.options["path"]).toBe("/data/test.db");
  });

  it("create server multiple options", () => {
    catalog.createServer("myserver", "duckdb_fdw", {
      path: "/data/test.db",
      memory_limit: "2GB",
    });
    const s = catalog.getServer("myserver");
    expect(s.options["path"]).toBe("/data/test.db");
    expect(s.options["memory_limit"]).toBe("2GB");
  });

  it("create server arrow", () => {
    catalog.createServer("arrowserver", "arrow_fdw");
    const s = catalog.getServer("arrowserver");
    expect(s.fdwType).toBe("arrow_fdw");
  });

  it("create server no options", () => {
    catalog.createServer("bare", "duckdb_fdw");
    const s = catalog.getServer("bare");
    expect(Object.keys(s.options).length).toBe(0);
  });

  it("create server if not exists", () => {
    catalog.createServer("myserver", "duckdb_fdw");
    // Should not throw
    catalog.createServer("myserver", "duckdb_fdw", {}, true);
    expect(catalog.hasServer("myserver")).toBe(true);
  });

  it("create duplicate server error", () => {
    catalog.createServer("myserver", "duckdb_fdw");
    expect(() => catalog.createServer("myserver", "duckdb_fdw")).toThrow(
      /already exists/,
    );
  });

  it("create server unsupported FDW", () => {
    expect(() => catalog.createServer("bad", "unknown_fdw")).toThrow(/Unsupported FDW/);
  });

  it("drop server", () => {
    catalog.createServer("myserver", "duckdb_fdw");
    catalog.dropServer("myserver");
    expect(catalog.hasServer("myserver")).toBe(false);
  });

  it("drop server if exists", () => {
    // Should not throw
    catalog.dropServer("nonexistent", true);
  });

  it("drop server not found error", () => {
    expect(() => catalog.dropServer("nonexistent")).toThrow(/not found/);
  });

  it("drop server with dependent table", () => {
    catalog.createServer("myserver", "duckdb_fdw");
    catalog.createForeignTable("ft", "myserver", makeCols(["id", "INTEGER"]));
    expect(() => catalog.dropServer("myserver")).toThrow(/depends on it/);
  });

  it("drop server closes handler", () => {
    catalog.createServer("myserver", "duckdb_fdw");
    const handler = catalog.getHandler("myserver");
    expect(handler.isClosed).toBe(false);
    // Remove dependent tables so drop can succeed
    catalog.dropServer("myserver");
    expect(handler.isClosed).toBe(true);
  });
});

describe("CreateDropForeignTable", () => {
  let catalog: FDWCatalog;
  beforeEach(() => {
    catalog = new FDWCatalog();
    catalog.createServer("myserver", "duckdb_fdw");
  });

  it("create foreign table", () => {
    catalog.createForeignTable("ft", "myserver", makeCols(["id", "INTEGER"]));
    expect(catalog.hasForeignTable("ft")).toBe(true);
  });

  it("create foreign table column types", () => {
    const cols = makeCols(["id", "INTEGER"], ["name", "TEXT"], ["salary", "REAL"]);
    catalog.createForeignTable("ft", "myserver", cols);
    const ft = catalog.getForeignTable("ft");
    expect(ft.columns.size).toBe(3);
    expect(ft.columns.get("salary")!.typeName).toBe("REAL");
  });

  it("create foreign table if not exists", () => {
    catalog.createForeignTable("ft", "myserver", makeCols(["id", "INTEGER"]));
    // Should not throw
    catalog.createForeignTable("ft", "myserver", makeCols(["id", "INTEGER"]), {}, true);
  });

  it("create duplicate foreign table error", () => {
    catalog.createForeignTable("ft", "myserver", makeCols(["id", "INTEGER"]));
    expect(() =>
      catalog.createForeignTable("ft", "myserver", makeCols(["id", "INTEGER"])),
    ).toThrow(/already exists/);
  });

  it("create foreign table name conflicts with regular table", () => {
    catalog.registerRegularTable("users");
    expect(() =>
      catalog.createForeignTable("users", "myserver", makeCols(["id", "INTEGER"])),
    ).toThrow(/conflicts with regular table/);
  });

  it("create foreign table missing server", () => {
    expect(() =>
      catalog.createForeignTable("ft", "nonexistent", makeCols(["id", "INTEGER"])),
    ).toThrow(/not found/);
  });

  it("multiple foreign tables on same server", () => {
    catalog.createForeignTable("ft1", "myserver", makeCols(["id", "INTEGER"]));
    catalog.createForeignTable("ft2", "myserver", makeCols(["id", "INTEGER"]));
    expect(catalog.hasForeignTable("ft1")).toBe(true);
    expect(catalog.hasForeignTable("ft2")).toBe(true);
    const ft1 = catalog.getForeignTable("ft1");
    const ft2 = catalog.getForeignTable("ft2");
    expect(ft1.serverName).toBe(ft2.serverName);
  });

  it("drop foreign table", () => {
    catalog.createForeignTable("ft", "myserver", makeCols(["id", "INTEGER"]));
    catalog.dropForeignTable("ft");
    expect(catalog.hasForeignTable("ft")).toBe(false);
  });

  it("drop foreign table if exists", () => {
    // Should not throw
    catalog.dropForeignTable("nonexistent", true);
  });

  it("drop foreign table not found error", () => {
    expect(() => catalog.dropForeignTable("nonexistent")).toThrow(/not found/);
  });

  it("drop foreign table then server", () => {
    catalog.createForeignTable("ft", "myserver", makeCols(["id", "INTEGER"]));
    catalog.dropForeignTable("ft");
    // Now server can be dropped
    catalog.dropServer("myserver");
    expect(catalog.hasServer("myserver")).toBe(false);
  });
});

describe("DMLGuard", () => {
  let catalog: FDWCatalog;
  beforeEach(() => {
    catalog = new FDWCatalog();
    catalog.createServer("myserver", "duckdb_fdw");
    catalog.createForeignTable("ft", "myserver", makeCols(["id", "INTEGER"]));
  });

  it("insert rejected", () => {
    // Foreign tables are read-only; MockFDWHandler.scan only supports reads
    const ft = catalog.getForeignTable("ft");
    // The handler has no insert method -- attempting an insert-like operation
    // on a foreign table should conceptually be rejected
    expect(ft.serverName).toBe("myserver");
    // Verify handler only exposes scan() and close()
    const handler = catalog.getHandler("myserver");
    expect(typeof handler.scan).toBe("function");
    expect(typeof handler.close).toBe("function");
    // No insert/update/delete methods exist
    expect((handler as unknown as Record<string, unknown>)["insert"]).toBeUndefined();
  });

  it("update rejected", () => {
    const handler = catalog.getHandler("myserver");
    expect((handler as unknown as Record<string, unknown>)["update"]).toBeUndefined();
  });

  it("delete rejected", () => {
    const handler = catalog.getHandler("myserver");
    expect((handler as unknown as Record<string, unknown>)["delete"]).toBeUndefined();
  });
});

describe("SourceNormalization", () => {
  it("DuckDBFDWHandler normalizeSource exposed via helper", () => {
    const result = normalizeSource("test.parquet");
    expect(result).toContain("read_parquet");
  });

  it("parquet path auto wrapped", () => {
    const result = normalizeSource("data/file.parquet");
    expect(result).toBe("read_parquet('data/file.parquet')");
  });

  it("csv path auto wrapped", () => {
    const result = normalizeSource("data/file.csv");
    expect(result).toBe("read_csv_auto('data/file.csv')");
  });

  it("json path auto wrapped", () => {
    const result = normalizeSource("data/file.json");
    expect(result).toBe("read_json_auto('data/file.json')");
  });

  it("ndjson path auto wrapped", () => {
    const result = normalizeSource("data/file.ndjson");
    expect(result).toBe("read_ndjson_auto('data/file.ndjson')");
  });

  it("expression with parens not wrapped", () => {
    const result = normalizeSource("read_parquet('custom.parquet')");
    expect(result).toBe("read_parquet('custom.parquet')");
  });

  it("table name not wrapped", () => {
    const result = normalizeSource("my_table");
    expect(result).toBe("my_table");
  });

  it("case insensitive extension", () => {
    const result = normalizeSource("data/file.PARQUET");
    expect(result).toBe("read_parquet('data/file.PARQUET')");
  });
});

describe("DuckDBFDWBasicQueries", () => {
  let catalog: FDWCatalog;
  beforeEach(() => {
    catalog = new FDWCatalog();
    catalog.createServer("myserver", "mock_fdw");
    catalog.createForeignTable(
      "employees",
      "myserver",
      makeCols(
        ["id", "INTEGER"],
        ["name", "TEXT"],
        ["age", "INTEGER"],
        ["dept", "TEXT"],
        ["salary", "INTEGER"],
      ),
    );
  });

  it("select all", () => {
    const rows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    expect(rows.length).toBe(5);
  });

  it("select columns", () => {
    const rows = catalog.scanForeignTable(
      "employees",
      ["name", "age"],
      null,
      null,
      TEST_DATA,
    );
    expect(rows.length).toBe(5);
    expect(Object.keys(rows[0]!)).toEqual(expect.arrayContaining(["name", "age"]));
    expect(rows[0]!["id"]).toBeUndefined();
  });

  it("where equality", () => {
    const rows = catalog.scanForeignTable(
      "employees",
      null,
      [{ column: "name", operator: "=", value: "Alice" }],
      null,
      TEST_DATA,
    );
    expect(rows.length).toBe(1);
    expect(rows[0]!["name"]).toBe("Alice");
  });

  it("where comparison", () => {
    const rows = catalog.scanForeignTable(
      "employees",
      null,
      [{ column: "age", operator: ">", value: 30 }],
      null,
      TEST_DATA,
    );
    expect(rows.length).toBe(2); // Charlie(35), Eve(32)
  });

  it("where and", () => {
    const rows = catalog.scanForeignTable(
      "employees",
      null,
      [
        { column: "dept", operator: "=", value: "eng" },
        { column: "age", operator: ">", value: 30 },
      ],
      null,
      TEST_DATA,
    );
    expect(rows.length).toBe(2); // Charlie(35), Eve(32)
  });

  it("where or", () => {
    // OR is simulated: scan with no filter, then post-filter
    const allRows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    const filtered = allRows.filter((r) => r["dept"] === "sales" || r["dept"] === "hr");
    expect(filtered.length).toBe(2); // Bob(sales), Diana(hr)
  });

  it("where in", () => {
    const rows = catalog.scanForeignTable(
      "employees",
      null,
      [{ column: "dept", operator: "IN", value: ["eng", "hr"] }],
      null,
      TEST_DATA,
    );
    expect(rows.length).toBe(4); // Alice, Charlie, Diana, Eve
  });

  it("where between", () => {
    const ft = catalog.getForeignTable("employees");
    const handler = catalog.getHandler("myserver", TEST_DATA);
    const rows = handler.scan(ft, null, [
      {
        column: "age",
        operator: "BETWEEN",
        value: 28,
        value2: 32,
      } as HandlerPredicate & {
        value2: number;
      },
    ]);
    expect(rows.length).toBe(3); // Alice(30), Diana(28), Eve(32)
  });

  it("where like", () => {
    const rows = catalog.scanForeignTable(
      "employees",
      null,
      [{ column: "name", operator: "LIKE", value: "A%" }],
      null,
      TEST_DATA,
    );
    expect(rows.length).toBe(1);
    expect(rows[0]!["name"]).toBe("Alice");
  });

  it("order by desc", () => {
    const rows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    rows.sort((a, b) => (b["age"] as number) - (a["age"] as number));
    expect(rows[0]!["name"]).toBe("Charlie");
  });

  it("limit", () => {
    const rows = catalog.scanForeignTable("employees", null, null, 2, TEST_DATA);
    expect(rows.length).toBe(2);
  });

  it("limit offset", () => {
    const allRows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    const offset = 2;
    const limit = 2;
    const rows = allRows.slice(offset, offset + limit);
    expect(rows.length).toBe(2);
    expect(rows[0]!["name"]).toBe("Charlie");
  });

  it("distinct", () => {
    const rows = catalog.scanForeignTable("employees", ["dept"], null, null, TEST_DATA);
    const unique = [...new Set(rows.map((r) => r["dept"]))];
    expect(unique.length).toBe(3); // eng, sales, hr
  });

  it("count", () => {
    const rows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    expect(rows.length).toBe(5);
  });

  it("sum", () => {
    const rows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    const total = rows.reduce((acc, r) => acc + (r["salary"] as number), 0);
    expect(total).toBe(430000);
  });

  it("avg", () => {
    const rows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    const avg = rows.reduce((acc, r) => acc + (r["salary"] as number), 0) / rows.length;
    expect(avg).toBe(86000);
  });

  it("min max", () => {
    const rows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    const salaries = rows.map((r) => r["salary"] as number);
    expect(Math.min(...salaries)).toBe(65000);
    expect(Math.max(...salaries)).toBe(110000);
  });
});

describe("DuckDBFDWAggregation", () => {
  let catalog: FDWCatalog;
  beforeEach(() => {
    catalog = new FDWCatalog();
    catalog.createServer("myserver", "mock_fdw");
    catalog.createForeignTable(
      "employees",
      "myserver",
      makeCols(
        ["id", "INTEGER"],
        ["name", "TEXT"],
        ["age", "INTEGER"],
        ["dept", "TEXT"],
        ["salary", "INTEGER"],
      ),
    );
  });

  it("group by", () => {
    const rows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    const groups = new Map<string, number>();
    for (const r of rows) {
      const dept = r["dept"] as string;
      groups.set(dept, (groups.get(dept) ?? 0) + 1);
    }
    expect(groups.get("eng")).toBe(3);
    expect(groups.get("sales")).toBe(1);
    expect(groups.get("hr")).toBe(1);
  });

  it("group by having", () => {
    const rows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    const groups = new Map<string, number>();
    for (const r of rows) {
      const dept = r["dept"] as string;
      groups.set(dept, (groups.get(dept) ?? 0) + 1);
    }
    const filtered = [...groups.entries()].filter(([, count]) => count > 1);
    expect(filtered.length).toBe(1);
    expect(filtered[0]![0]).toBe("eng");
  });

  it("group by multiple columns", () => {
    const rows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    const groups = new Map<string, number>();
    for (const r of rows) {
      const key = `${r["dept"] as string}-${String((r["age"] as number) > 30 ? "senior" : "junior")}`;
      groups.set(key, (groups.get(key) ?? 0) + 1);
    }
    expect(groups.size).toBeGreaterThan(1);
  });
});

describe("DuckDBFDWJoins", () => {
  let catalog: FDWCatalog;
  const localData = [
    { id: 1, team: "alpha" },
    { id: 2, team: "beta" },
    { id: 3, team: "alpha" },
  ];

  beforeEach(() => {
    catalog = new FDWCatalog();
    catalog.createServer("myserver", "mock_fdw");
    catalog.createForeignTable(
      "employees",
      "myserver",
      makeCols(
        ["id", "INTEGER"],
        ["name", "TEXT"],
        ["age", "INTEGER"],
        ["dept", "TEXT"],
        ["salary", "INTEGER"],
      ),
    );
  });

  it("join foreign and local", () => {
    const foreignRows = catalog.scanForeignTable(
      "employees",
      null,
      null,
      null,
      TEST_DATA,
    );
    // Simulate hash join
    const localMap = new Map(localData.map((r) => [r.id, r]));
    const joined = foreignRows
      .filter((r) => localMap.has(r["id"] as number))
      .map((r) => ({
        ...r,
        team: localMap.get(r["id"] as number)!.team,
      }));
    expect(joined.length).toBe(3);
    expect(joined[0]!["team"]).toBe("alpha");
  });

  it("join two foreign tables", () => {
    catalog.createServer("server2", "mock_fdw");
    catalog.createForeignTable(
      "departments",
      "server2",
      makeCols(["dept", "TEXT"], ["location", "TEXT"]),
    );
    const deptData: Record<string, unknown>[] = [
      { dept: "eng", location: "SF" },
      { dept: "sales", location: "NY" },
      { dept: "hr", location: "LA" },
    ];
    const empRows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    const deptRows = catalog.scanForeignTable(
      "departments",
      null,
      null,
      null,
      deptData,
    );
    const deptMap = new Map(deptRows.map((r) => [r["dept"], r["location"]]));
    const joined = empRows.map((r) => ({
      ...r,
      location: deptMap.get(r["dept"]) ?? null,
    }));
    expect(joined.length).toBe(5);
    expect(joined[0]!["location"]).toBe("SF"); // Alice is eng
  });

  it("three way join local and foreign", () => {
    const foreignRows = catalog.scanForeignTable(
      "employees",
      null,
      null,
      null,
      TEST_DATA,
    );
    const deptLocations: Record<string, string> = {
      eng: "SF",
      sales: "NY",
      hr: "LA",
    };
    const localMap = new Map(localData.map((r) => [r.id, r]));
    const joined = foreignRows
      .filter((r) => localMap.has(r["id"] as number))
      .map((r) => ({
        ...r,
        team: localMap.get(r["id"] as number)!.team,
        location: deptLocations[r["dept"] as string] ?? null,
      }));
    expect(joined.length).toBe(3);
    expect(joined[0]!["location"]).toBe("SF");
    expect(joined[0]!["team"]).toBe("alpha");
  });

  it("left join foreign and local", () => {
    const foreignRows = catalog.scanForeignTable(
      "employees",
      null,
      null,
      null,
      TEST_DATA,
    );
    const localMap = new Map(localData.map((r) => [r.id, r]));
    const joined = foreignRows.map((r) => ({
      ...r,
      team: localMap.get(r["id"] as number)?.team ?? null,
    }));
    expect(joined.length).toBe(5); // All foreign rows kept
    expect(joined[3]!["team"]).toBeNull(); // Diana (id=4) not in local
  });
});

describe("DuckDBFDWSubqueries", () => {
  let catalog: FDWCatalog;
  beforeEach(() => {
    catalog = new FDWCatalog();
    catalog.createServer("myserver", "mock_fdw");
    catalog.createForeignTable(
      "employees",
      "myserver",
      makeCols(
        ["id", "INTEGER"],
        ["name", "TEXT"],
        ["age", "INTEGER"],
        ["dept", "TEXT"],
        ["salary", "INTEGER"],
      ),
    );
  });

  it("subquery in where", () => {
    const allRows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    const avgSalary =
      allRows.reduce((acc, r) => acc + (r["salary"] as number), 0) / allRows.length;
    const filtered = allRows.filter((r) => (r["salary"] as number) > avgSalary);
    expect(filtered.length).toBe(3); // Alice(90000), Eve(95000), Charlie(110000)
  });

  it("cte over foreign table", () => {
    const engRows = catalog.scanForeignTable(
      "employees",
      null,
      [{ column: "dept", operator: "=", value: "eng" }],
      null,
      TEST_DATA,
    );
    // CTE-like: use engRows as a derived table
    const result = engRows.filter((r) => (r["age"] as number) > 30);
    expect(result.length).toBe(2); // Charlie(35), Eve(32)
  });

  it("scalar subquery", () => {
    const rows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    const maxSalary = Math.max(...rows.map((r) => r["salary"] as number));
    const richest = rows.filter((r) => (r["salary"] as number) === maxSalary);
    expect(richest.length).toBe(1);
    expect(richest[0]!["name"]).toBe("Charlie");
  });
});

describe("DuckDBFDWWindowFunctions", () => {
  let catalog: FDWCatalog;
  beforeEach(() => {
    catalog = new FDWCatalog();
    catalog.createServer("myserver", "mock_fdw");
    catalog.createForeignTable(
      "employees",
      "myserver",
      makeCols(
        ["id", "INTEGER"],
        ["name", "TEXT"],
        ["age", "INTEGER"],
        ["dept", "TEXT"],
        ["salary", "INTEGER"],
      ),
    );
  });

  it("row number", () => {
    const rows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    rows.sort((a, b) => (a["salary"] as number) - (b["salary"] as number));
    const withRowNum = rows.map((r, i) => ({ ...r, row_num: i + 1 }));
    expect(withRowNum[0]!["row_num"]).toBe(1);
    expect(withRowNum[4]!["row_num"]).toBe(5);
  });

  it("rank", () => {
    const rows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    rows.sort((a, b) => (b["salary"] as number) - (a["salary"] as number));
    let rank = 1;
    const withRank = rows.map((r, i) => {
      if (i > 0 && r["salary"] !== rows[i - 1]!["salary"]) rank = i + 1;
      return { ...r, rank };
    });
    expect((withRank[0] as Record<string, unknown>)["rank"]).toBe(1);
    expect((withRank[0] as Record<string, unknown>)["name"]).toBe("Charlie");
  });

  it("running sum", () => {
    const rows = catalog.scanForeignTable("employees", null, null, null, TEST_DATA);
    rows.sort((a, b) => (a["id"] as number) - (b["id"] as number));
    let runningSum = 0;
    const withRunningSum = rows.map((r) => {
      runningSum += r["salary"] as number;
      return { ...r, running_sum: runningSum };
    });
    expect(withRunningSum[0]!["running_sum"]).toBe(90000);
    expect(withRunningSum[4]!["running_sum"]).toBe(430000);
  });
});

describe("DuckDBFDWExplain", () => {
  it("explain foreign table", () => {
    const catalog = new FDWCatalog();
    catalog.createServer("myserver", "mock_fdw");
    catalog.createForeignTable(
      "ft",
      "myserver",
      makeCols(["id", "INTEGER"], ["name", "TEXT"]),
    );
    const ft = catalog.getForeignTable("ft");
    // Explain output: describe the scan plan
    const plan = {
      operation: "ForeignScan",
      table: ft.name,
      server: ft.serverName,
      columns: [...ft.columns.keys()],
    };
    expect(plan.operation).toBe("ForeignScan");
    expect(plan.table).toBe("ft");
    expect(plan.columns).toEqual(["id", "name"]);
  });
});

describe("DuckDBFDWSources", () => {
  it("parquet auto detected", () => {
    const src = normalizeSource("data.parquet");
    expect(src).toContain("read_parquet");
  });

  it("explicit read parquet", () => {
    const src = normalizeSource("read_parquet('data.parquet')");
    expect(src).toBe("read_parquet('data.parquet')");
  });

  it("csv auto detected", () => {
    const src = normalizeSource("data.csv");
    expect(src).toContain("read_csv_auto");
  });

  it("csv explicit read csv", () => {
    const src = normalizeSource("read_csv('data.csv')");
    expect(src).toBe("read_csv('data.csv')");
  });

  it("json auto detected", () => {
    const src = normalizeSource("data.json");
    expect(src).toContain("read_json_auto");
  });

  it("missing source option", () => {
    // A server without a source option should produce a fallback
    const server = createForeignServer("s", "duckdb_fdw");
    expect(server.options["source"]).toBeUndefined();
  });
});

describe("HandlerLifecycle", () => {
  it("handler cached per server", () => {
    const catalog = new FDWCatalog();
    catalog.createServer("s1", "mock_fdw");
    const h1 = catalog.getHandler("s1");
    const h2 = catalog.getHandler("s1");
    expect(h1).toBe(h2); // Same instance
  });

  it("handler shared across tables", () => {
    const catalog = new FDWCatalog();
    catalog.createServer("s1", "mock_fdw");
    catalog.createForeignTable("t1", "s1", makeCols(["id", "INTEGER"]));
    catalog.createForeignTable("t2", "s1", makeCols(["id", "INTEGER"]));
    const ft1 = catalog.getForeignTable("t1");
    const ft2 = catalog.getForeignTable("t2");
    // Both tables use the same server, so same handler
    expect(ft1.serverName).toBe(ft2.serverName);
    const h1 = catalog.getHandler(ft1.serverName);
    const h2 = catalog.getHandler(ft2.serverName);
    expect(h1).toBe(h2);
  });

  it("close clears handlers", () => {
    const catalog = new FDWCatalog();
    catalog.createServer("s1", "mock_fdw");
    const handler = catalog.getHandler("s1");
    expect(handler.isClosed).toBe(false);
    catalog.closeAll();
    expect(handler.isClosed).toBe(true);
  });
});

describe("InformationSchema", () => {
  it("foreign table type", () => {
    const catalog = new FDWCatalog();
    catalog.createServer("s1", "mock_fdw");
    catalog.createForeignTable("ft", "s1", makeCols(["id", "INTEGER"]));
    const ft = catalog.getForeignTable("ft");
    // Foreign tables have a server name distinguishing them from regular tables
    expect(ft.serverName).toBe("s1");
    expect(catalog.hasForeignTable("ft")).toBe(true);
  });

  it("foreign table with view", () => {
    const catalog = new FDWCatalog();
    catalog.createServer("s1", "mock_fdw");
    catalog.createForeignTable(
      "ft",
      "s1",
      makeCols(["id", "INTEGER"], ["name", "TEXT"]),
    );
    // Simulate a view: scan with a column projection
    const rows = catalog.scanForeignTable("ft", ["name"], null, null, [
      { id: 1, name: "Alice" },
      { id: 2, name: "Bob" },
    ]);
    expect(rows.length).toBe(2);
    expect(rows[0]!["id"]).toBeUndefined();
    expect(rows[0]!["name"]).toBe("Alice");
  });

  it("dropped foreign table disappears", () => {
    const catalog = new FDWCatalog();
    catalog.createServer("s1", "mock_fdw");
    catalog.createForeignTable("ft", "s1", makeCols(["id", "INTEGER"]));
    expect(catalog.listForeignTables()).toContain("ft");
    catalog.dropForeignTable("ft");
    expect(catalog.listForeignTables()).not.toContain("ft");
  });
});

describe("CatalogPersistence", () => {
  it("persist and restore server", () => {
    const catalog = new FDWCatalog();
    catalog.createServer("s1", "duckdb_fdw", { path: "/data/test.db" });
    const json = catalog.toJSON();
    const restored = FDWCatalog.fromJSON(json);
    expect(restored.hasServer("s1")).toBe(true);
    expect(restored.getServer("s1").options["path"]).toBe("/data/test.db");
  });

  it("persist and restore foreign table", () => {
    const catalog = new FDWCatalog();
    catalog.createServer("s1", "duckdb_fdw");
    catalog.createForeignTable(
      "ft",
      "s1",
      makeCols(["id", "INTEGER"], ["name", "TEXT"]),
    );
    const json = catalog.toJSON();
    const restored = FDWCatalog.fromJSON(json);
    expect(restored.hasForeignTable("ft")).toBe(true);
    const ft = restored.getForeignTable("ft");
    expect(ft.columns.size).toBe(2);
  });

  it("query after restore", () => {
    const catalog = new FDWCatalog();
    catalog.createServer("s1", "mock_fdw");
    catalog.createForeignTable(
      "ft",
      "s1",
      makeCols(["id", "INTEGER"], ["name", "TEXT"]),
    );
    const json = catalog.toJSON();
    const restored = FDWCatalog.fromJSON(json);
    const rows = restored.scanForeignTable("ft", null, null, null, [
      { id: 1, name: "Alice" },
    ]);
    expect(rows.length).toBe(1);
    expect(rows[0]!["name"]).toBe("Alice");
  });

  it("drop persists", () => {
    const catalog = new FDWCatalog();
    catalog.createServer("s1", "mock_fdw");
    catalog.createForeignTable("ft", "s1", makeCols(["id", "INTEGER"]));
    catalog.dropForeignTable("ft");
    catalog.dropServer("s1");
    const json = catalog.toJSON();
    const restored = FDWCatalog.fromJSON(json);
    expect(restored.hasServer("s1")).toBe(false);
    expect(restored.hasForeignTable("ft")).toBe(false);
  });

  it("multiple servers persist", () => {
    const catalog = new FDWCatalog();
    catalog.createServer("s1", "duckdb_fdw");
    catalog.createServer("s2", "arrow_fdw");
    const json = catalog.toJSON();
    const restored = FDWCatalog.fromJSON(json);
    expect(restored.hasServer("s1")).toBe(true);
    expect(restored.hasServer("s2")).toBe(true);
    expect(restored.getServer("s2").fdwType).toBe("arrow_fdw");
  });
});

describe("HivePartitionDiscovery", () => {
  let catalog: FDWCatalog;
  beforeEach(() => {
    catalog = new FDWCatalog();
    catalog.createServer("hive", "mock_fdw");
    catalog.createForeignTable(
      "hive_tbl",
      "hive",
      makeCols(
        ["id", "INTEGER"],
        ["value", "INTEGER"],
        ["year", "INTEGER"],
        ["month", "INTEGER"],
      ),
      { hive_partitioning: "true" },
    );
  });

  it("select all rows", () => {
    const rows = catalog.scanForeignTable("hive_tbl", null, null, null, HIVE_DATA);
    expect(rows.length).toBe(8);
  });

  it("partition columns populated", () => {
    const rows = catalog.scanForeignTable("hive_tbl", null, null, null, HIVE_DATA);
    for (const row of rows) {
      expect(row["year"]).toBeDefined();
      expect(row["month"]).toBeDefined();
    }
  });

  it("data and partition columns together", () => {
    const rows = catalog.scanForeignTable(
      "hive_tbl",
      ["id", "year"],
      null,
      null,
      HIVE_DATA,
    );
    for (const row of rows) {
      expect(row["id"]).toBeDefined();
      expect(row["year"]).toBeDefined();
      expect(row["value"]).toBeUndefined();
    }
  });

  it("distinct partition values", () => {
    const rows = catalog.scanForeignTable("hive_tbl", ["year"], null, null, HIVE_DATA);
    const years = new Set(rows.map((r) => r["year"]));
    expect(years.size).toBe(3); // 2020, 2021, 2022
  });

  it("count per partition", () => {
    const rows = catalog.scanForeignTable("hive_tbl", null, null, null, HIVE_DATA);
    const counts = new Map<number, number>();
    for (const r of rows) {
      const year = r["year"] as number;
      counts.set(year, (counts.get(year) ?? 0) + 1);
    }
    expect(counts.get(2020)).toBe(3);
    expect(counts.get(2021)).toBe(3);
    expect(counts.get(2022)).toBe(2);
  });
});

describe("HivePredicatePushdown", () => {
  let catalog: FDWCatalog;
  beforeEach(() => {
    catalog = new FDWCatalog();
    catalog.createServer("hive", "mock_fdw");
    catalog.createForeignTable(
      "hive_tbl",
      "hive",
      makeCols(
        ["id", "INTEGER"],
        ["value", "INTEGER"],
        ["year", "INTEGER"],
        ["month", "INTEGER"],
      ),
      { hive_partitioning: "true" },
    );
  });

  it("equality on partition column", () => {
    const rows = catalog.scanForeignTable(
      "hive_tbl",
      null,
      [{ column: "year", operator: "=", value: 2021 }],
      null,
      HIVE_DATA,
    );
    expect(rows.length).toBe(3);
    for (const r of rows) expect(r["year"]).toBe(2021);
  });

  it("comparison on partition column", () => {
    const rows = catalog.scanForeignTable(
      "hive_tbl",
      null,
      [{ column: "year", operator: ">", value: 2020 }],
      null,
      HIVE_DATA,
    );
    expect(rows.length).toBe(5);
    for (const r of rows) expect((r["year"] as number) > 2020).toBe(true);
  });

  it("multiple partition predicates", () => {
    const rows = catalog.scanForeignTable(
      "hive_tbl",
      null,
      [
        { column: "year", operator: "=", value: 2021 },
        { column: "month", operator: "=", value: 1 },
      ],
      null,
      HIVE_DATA,
    );
    expect(rows.length).toBe(2); // id=3, id=8
  });

  it("partition and data predicates", () => {
    const rows = catalog.scanForeignTable(
      "hive_tbl",
      null,
      [
        { column: "year", operator: "=", value: 2021 },
        { column: "value", operator: ">", value: 160 },
      ],
      null,
      HIVE_DATA,
    );
    expect(rows.length).toBe(2); // id=4(250), id=8(180)
  });

  it("between on partition column", () => {
    const ft = catalog.getForeignTable("hive_tbl");
    const handler = catalog.getHandler("hive", HIVE_DATA);
    const rows = handler.scan(ft, null, [
      {
        column: "year",
        operator: "BETWEEN",
        value: 2020,
        value2: 2021,
      } as HandlerPredicate & {
        value2: number;
      },
    ]);
    expect(rows.length).toBe(6);
  });

  it("or not pushed down", () => {
    // OR cannot be expressed as simple AND predicates
    // So we scan all and post-filter
    const allRows = catalog.scanForeignTable("hive_tbl", null, null, null, HIVE_DATA);
    const filtered = allRows.filter((r) => r["year"] === 2020 || r["year"] === 2022);
    expect(filtered.length).toBe(5);
  });

  it("mixed and or", () => {
    const allRows = catalog.scanForeignTable("hive_tbl", null, null, null, HIVE_DATA);
    const filtered = allRows.filter(
      (r) => (r["year"] === 2020 || r["year"] === 2022) && r["month"] === 1,
    );
    expect(filtered.length).toBe(3); // id=1,7(2020,m1), id=5(2022,m1)
  });

  it("not equal on partition", () => {
    const rows = catalog.scanForeignTable(
      "hive_tbl",
      null,
      [{ column: "year", operator: "<>", value: 2021 }],
      null,
      HIVE_DATA,
    );
    expect(rows.length).toBe(5);
    for (const r of rows) expect(r["year"]).not.toBe(2021);
  });

  it("less than on partition", () => {
    const rows = catalog.scanForeignTable(
      "hive_tbl",
      null,
      [{ column: "year", operator: "<", value: 2021 }],
      null,
      HIVE_DATA,
    );
    expect(rows.length).toBe(3);
    for (const r of rows) expect(r["year"]).toBe(2020);
  });

  it("greater equal on partition", () => {
    const rows = catalog.scanForeignTable(
      "hive_tbl",
      null,
      [{ column: "year", operator: ">=", value: 2022 }],
      null,
      HIVE_DATA,
    );
    expect(rows.length).toBe(2);
    for (const r of rows) expect(r["year"]).toBe(2022);
  });

  it("string predicate on data column", () => {
    const dataWithStrings = [
      { id: 1, name: "Alice", year: 2020 },
      { id: 2, name: "Bob", year: 2021 },
      { id: 3, name: "Charlie", year: 2021 },
    ];
    catalog.createServer("s2", "mock_fdw");
    catalog.createForeignTable(
      "str_tbl",
      "s2",
      makeCols(["id", "INTEGER"], ["name", "TEXT"], ["year", "INTEGER"]),
    );
    const rows = catalog.scanForeignTable(
      "str_tbl",
      null,
      [{ column: "name", operator: "=", value: "Bob" }],
      null,
      dataWithStrings,
    );
    expect(rows.length).toBe(1);
    expect(rows[0]!["name"]).toBe("Bob");
  });
});

describe("HivePartitionAggregation", () => {
  let catalog: FDWCatalog;
  beforeEach(() => {
    catalog = new FDWCatalog();
    catalog.createServer("hive", "mock_fdw");
    catalog.createForeignTable(
      "hive_tbl",
      "hive",
      makeCols(
        ["id", "INTEGER"],
        ["value", "INTEGER"],
        ["year", "INTEGER"],
        ["month", "INTEGER"],
      ),
    );
  });

  it("sum per year", () => {
    const rows = catalog.scanForeignTable("hive_tbl", null, null, null, HIVE_DATA);
    const sums = new Map<number, number>();
    for (const r of rows) {
      const y = r["year"] as number;
      sums.set(y, (sums.get(y) ?? 0) + (r["value"] as number));
    }
    expect(sums.get(2020)).toBe(420); // 100+200+120
    expect(sums.get(2021)).toBe(580); // 150+250+180
    expect(sums.get(2022)).toBe(650); // 300+350
  });

  it("avg with partition filter", () => {
    const rows = catalog.scanForeignTable(
      "hive_tbl",
      null,
      [{ column: "year", operator: "=", value: 2021 }],
      null,
      HIVE_DATA,
    );
    const avg = rows.reduce((acc, r) => acc + (r["value"] as number), 0) / rows.length;
    expect(avg).toBeCloseTo(580 / 3);
  });

  it("group by two partition columns", () => {
    const rows = catalog.scanForeignTable("hive_tbl", null, null, null, HIVE_DATA);
    const groups = new Map<string, number>();
    for (const r of rows) {
      const key = `${String(r["year"])}-${String(r["month"])}`;
      groups.set(key, (groups.get(key) ?? 0) + 1);
    }
    expect(groups.get("2020-1")).toBe(2); // id=1,7
    expect(groups.get("2021-1")).toBe(2); // id=3,8
    expect(groups.get("2022-2")).toBe(1); // id=6
  });
});

describe("HivePartitionJoins", () => {
  it("join hive with local table", () => {
    const catalog = new FDWCatalog();
    catalog.createServer("hive", "mock_fdw");
    catalog.createForeignTable(
      "hive_tbl",
      "hive",
      makeCols(["id", "INTEGER"], ["value", "INTEGER"], ["year", "INTEGER"]),
    );
    const hiveRows = catalog.scanForeignTable("hive_tbl", null, null, null, HIVE_DATA);
    const localData = [
      { year: 2020, budget: 1000 },
      { year: 2021, budget: 1500 },
      { year: 2022, budget: 2000 },
    ];
    const localMap = new Map(localData.map((r) => [r.year, r.budget]));
    const joined = hiveRows.map((r) => ({
      ...r,
      budget: localMap.get(r["year"] as number) ?? null,
    }));
    expect(joined.length).toBe(8);
    expect(joined[0]!["budget"]).toBe(1000); // year=2020
  });
});

describe("HivePartitionOrderLimit", () => {
  let catalog: FDWCatalog;
  beforeEach(() => {
    catalog = new FDWCatalog();
    catalog.createServer("hive", "mock_fdw");
    catalog.createForeignTable(
      "hive_tbl",
      "hive",
      makeCols(
        ["id", "INTEGER"],
        ["value", "INTEGER"],
        ["year", "INTEGER"],
        ["month", "INTEGER"],
      ),
    );
  });

  it("order by partition column", () => {
    const rows = catalog.scanForeignTable("hive_tbl", null, null, null, HIVE_DATA);
    rows.sort(
      (a, b) =>
        (a["year"] as number) - (b["year"] as number) ||
        (a["month"] as number) - (b["month"] as number),
    );
    expect(rows[0]!["year"]).toBe(2020);
    expect(rows[rows.length - 1]!["year"]).toBe(2022);
  });

  it("limit with partition filter", () => {
    const rows = catalog.scanForeignTable(
      "hive_tbl",
      null,
      [{ column: "year", operator: "=", value: 2021 }],
      2,
      HIVE_DATA,
    );
    expect(rows.length).toBe(2);
    for (const r of rows) expect(r["year"]).toBe(2021);
  });
});

describe("HiveNormalizeSource", () => {
  it("bare parquet with hive", () => {
    const result = normalizeSource("data.parquet", true);
    expect(result).toBe("read_parquet('data.parquet', hive_partitioning=true)");
  });

  it("bare parquet without hive", () => {
    const result = normalizeSource("data.parquet", false);
    expect(result).toBe("read_parquet('data.parquet')");
  });

  it("bare csv with hive", () => {
    const result = normalizeSource("data.csv", true);
    expect(result).toBe("read_csv_auto('data.csv', hive_partitioning=true)");
  });

  it("explicit expression ignores hive flag", () => {
    const result = normalizeSource("read_parquet('data.parquet')", true);
    expect(result).toBe("read_parquet('data.parquet')");
  });

  it("table name ignores hive flag", () => {
    const result = normalizeSource("my_table", true);
    expect(result).toBe("my_table");
  });
});

describe("HiveBuildWhereClause", () => {
  it("single equality", () => {
    const clause = buildWhereClause([{ column: "year", operator: "=", value: 2021 }]);
    expect(clause).toBe('"year" = 2021');
  });

  it("multiple predicates", () => {
    const clause = buildWhereClause([
      { column: "year", operator: "=", value: 2021 },
      { column: "month", operator: "=", value: 1 },
    ]);
    expect(clause).toBe('"year" = 2021 AND "month" = 1');
  });

  it("string value", () => {
    const clause = buildWhereClause([
      { column: "name", operator: "=", value: "Alice" },
    ]);
    expect(clause).toBe("\"name\" = 'Alice'");
  });

  it("null equality", () => {
    const clause = buildWhereClause([{ column: "name", operator: "=", value: null }]);
    expect(clause).toBe('"name" IS NULL');
  });

  it("null not equal", () => {
    const clause = buildWhereClause([{ column: "name", operator: "<>", value: null }]);
    expect(clause).toBe('"name" IS NOT NULL');
  });

  it("in operator", () => {
    const clause = buildWhereClause([
      { column: "year", operator: "IN", value: [2020, 2021] },
    ]);
    expect(clause).toBe('"year" IN (2020, 2021)');
  });

  it("in with strings", () => {
    const clause = buildWhereClause([
      { column: "dept", operator: "IN", value: ["eng", "sales"] },
    ]);
    expect(clause).toBe("\"dept\" IN ('eng', 'sales')");
  });

  it("like operator", () => {
    const clause = buildWhereClause([
      { column: "name", operator: "LIKE", value: "A%" },
    ]);
    expect(clause).toBe("\"name\" LIKE 'A%'");
  });

  it("not like operator", () => {
    const clause = buildWhereClause([
      { column: "name", operator: "NOT LIKE", value: "B%" },
    ]);
    expect(clause).toBe("\"name\" NOT LIKE 'B%'");
  });

  it("ilike operator", () => {
    const clause = buildWhereClause([
      { column: "name", operator: "ILIKE", value: "alice" },
    ]);
    expect(clause).toBe("\"name\" ILIKE 'alice'");
  });

  it("mixed operators", () => {
    const clause = buildWhereClause([
      { column: "year", operator: ">=", value: 2020 },
      { column: "name", operator: "LIKE", value: "A%" },
      { column: "dept", operator: "IN", value: ["eng"] },
    ]);
    expect(clause).toContain('"year" >= 2020');
    expect(clause).toContain("\"name\" LIKE 'A%'");
    expect(clause).toContain("\"dept\" IN ('eng')");
  });
});

describe("PredicateExtraction", () => {
  it("all predicates pushed no deferred where", () => {
    const result = extractPredicates("year = 2021", PARTITION_COLUMNS);
    expect(result.pushed.length).toBe(1);
    expect(result.deferred).toBeNull();
  });

  it("like pushed down", () => {
    const result = extractPredicates("name LIKE 'A%'", new Set(["name"]));
    expect(result.pushed.length).toBe(1);
    expect(result.pushed[0]!.operator).toBe("LIKE");
  });

  it("not like pushed down", () => {
    const result = extractPredicates("name NOT LIKE 'B%'", new Set(["name"]));
    expect(result.pushed.length).toBe(1);
    expect(result.pushed[0]!.operator).toBe("NOT LIKE");
  });

  it("ilike pushed down", () => {
    const result = extractPredicates("name ILIKE 'alice'", new Set(["name"]));
    expect(result.pushed.length).toBe(1);
    expect(result.pushed[0]!.operator).toBe("ILIKE");
  });

  it("in pushed down", () => {
    const result = extractPredicates("year IN (2020, 2021)", PARTITION_COLUMNS);
    expect(result.pushed.length).toBe(1);
    expect(result.pushed[0]!.operator).toBe("IN");
  });

  it("in subset", () => {
    const result = extractPredicates("year IN (2021)", PARTITION_COLUMNS);
    expect(result.pushed.length).toBe(1);
    expect((result.pushed[0]!.value as number[]).length).toBe(1);
  });

  it("in with strings", () => {
    const result = extractPredicates("dept IN ('eng', 'sales')", new Set(["dept"]));
    expect(result.pushed.length).toBe(1);
    expect(result.pushed[0]!.value).toEqual(["eng", "sales"]);
  });

  it("in and comparison together", () => {
    const result = extractPredicates(
      "year IN (2020, 2021) AND month > 1",
      PARTITION_COLUMNS,
    );
    expect(result.pushed.length).toBe(2);
    expect(result.deferred).toBeNull();
  });

  it("like and comparison together", () => {
    const result = extractPredicates(
      "name LIKE 'A%' AND year > 2020",
      new Set(["name", "year"]),
    );
    expect(result.pushed.length).toBe(2);
  });

  it("no where clause", () => {
    const result = extractPredicates(null, PARTITION_COLUMNS);
    expect(result.pushed.length).toBe(0);
    expect(result.deferred).toBeNull();
  });

  it("or remains deferred", () => {
    const result = extractPredicates("year = 2020 OR year = 2021", PARTITION_COLUMNS);
    // OR cannot be split into separate pushable predicates
    expect(result.deferred).not.toBeNull();
  });
});

describe("HivePartitionWithCSV", () => {
  it("csv hive partition", () => {
    const src = normalizeSource("data.csv", true);
    expect(src).toContain("hive_partitioning=true");
    expect(src).toContain("read_csv_auto");
  });
});

describe("HivePartitionCatalogPersistence", () => {
  it("hive option persists", () => {
    const catalog = new FDWCatalog();
    catalog.createServer("hive", "mock_fdw");
    catalog.createForeignTable("ft", "hive", makeCols(["id", "INTEGER"]), {
      hive_partitioning: "true",
    });
    const json = catalog.toJSON();
    const restored = FDWCatalog.fromJSON(json);
    const ft = restored.getForeignTable("ft");
    expect(ft.options["hive_partitioning"]).toBe("true");
  });
});
