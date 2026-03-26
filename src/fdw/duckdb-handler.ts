//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- DuckDB FDW handler
// 1:1 port of uqa/fdw/duckdb_handler.py
//
// Provides foreign data access to DuckDB via @duckdb/duckdb-wasm.
// When the @duckdb/duckdb-wasm package is not available, falls back to
// JSON-encoded inline data for testing.

import type { FDWPredicate } from "./handler.js";
import { FDWHandler } from "./handler.js";
import type { ForeignServer } from "./foreign-table.js";
import type { ForeignTable } from "./foreign-table.js";

/**
 * Build a SQL WHERE clause from FDW predicates.
 */
function buildWhereClause(predicates: FDWPredicate[]): string {
  if (predicates.length === 0) return "";
  const conditions: string[] = [];
  for (const pred of predicates) {
    const col = `"${pred.column}"`;
    switch (pred.operator) {
      case "=":
        conditions.push(`${col} = ${sqlLiteral(pred.value)}`);
        break;
      case "<>":
      case "!=":
        conditions.push(`${col} <> ${sqlLiteral(pred.value)}`);
        break;
      case "<":
        conditions.push(`${col} < ${sqlLiteral(pred.value)}`);
        break;
      case "<=":
        conditions.push(`${col} <= ${sqlLiteral(pred.value)}`);
        break;
      case ">":
        conditions.push(`${col} > ${sqlLiteral(pred.value)}`);
        break;
      case ">=":
        conditions.push(`${col} >= ${sqlLiteral(pred.value)}`);
        break;
      case "IS NULL":
        conditions.push(`${col} IS NULL`);
        break;
      case "IS NOT NULL":
        conditions.push(`${col} IS NOT NULL`);
        break;
      case "IN": {
        const vals = Array.isArray(pred.value)
          ? (pred.value as unknown[]).map(sqlLiteral).join(", ")
          : sqlLiteral(pred.value);
        conditions.push(`${col} IN (${vals})`);
        break;
      }
      case "LIKE":
        conditions.push(`${col} LIKE ${sqlLiteral(pred.value)}`);
        break;
      case "BETWEEN":
        conditions.push(
          `${col} BETWEEN ${sqlLiteral(pred.value)} AND ${sqlLiteral((pred as { value2?: unknown }).value2)}`,
        );
        break;
      default:
        conditions.push(`${col} ${pred.operator} ${sqlLiteral(pred.value)}`);
    }
  }
  return " WHERE " + conditions.join(" AND ");
}

function sqlLiteral(v: unknown): string {
  if (v === null || v === undefined) return "NULL";
  if (typeof v === "number") return String(v);
  if (typeof v === "boolean") return v ? "TRUE" : "FALSE";
  if (typeof v === "string") return `'${v.replace(/'/g, "''")}'`;
  return `'${String(v as string | number).replace(/'/g, "''")}'`;
}

export class DuckDBFDWHandler extends FDWHandler {
  private _server: ForeignServer;
  private _db: unknown;
  private _conn: unknown;
  private _duckdbLib: unknown;

  constructor(server: ForeignServer) {
    super();
    this._server = server;
    this._db = null;
    this._conn = null;
    this._duckdbLib = null;
  }

  get server(): ForeignServer {
    return this._server;
  }

  /**
   * Attempt to load @duckdb/duckdb-wasm dynamically.
   */
  private _loadDuckDB(): unknown {
    if (this._duckdbLib !== null) return this._duckdbLib;
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      this._duckdbLib = require("@duckdb/duckdb-wasm");
      return this._duckdbLib;
    } catch {
      return null;
    }
  }

  /**
   * Build the SQL query string with pushdown predicates, column projection, and limit.
   */
  private _buildQuery(
    tableName: string,
    columns: string[] | null | undefined,
    predicates: FDWPredicate[] | null | undefined,
    limit: number | null | undefined,
  ): string {
    const colList =
      columns && columns.length > 0 ? columns.map((c) => `"${c}"`).join(", ") : "*";
    let sql = `SELECT ${colList} FROM "${tableName}"`;
    if (predicates && predicates.length > 0) {
      sql += buildWhereClause(predicates);
    }
    if (limit !== null && limit !== undefined && limit >= 0) {
      sql += ` LIMIT ${String(limit)}`;
    }
    return sql;
  }

  /**
   * Convert DuckDB query result to plain row objects.
   */
  private _resultToRows(result: unknown): Record<string, unknown>[] {
    // DuckDB WASM result format: { numRows, numCols, columns, columnTypes, ... }
    // with getChildAt(colIdx) returning a vector.
    const res = result as {
      schema?: { fields: Array<{ name: string }> };
      numRows: number;
      numCols?: number;
      toArray?: () => Array<Record<string, unknown>>;
      getChildAt?: (i: number) => { get(j: number): unknown } | null;
    };

    // If there's a toArray method (common in duckdb-wasm)
    if (res.toArray) {
      return res.toArray();
    }

    // Manual extraction via getChildAt
    if (res.schema && res.getChildAt) {
      const fields = res.schema.fields;
      const numRows = res.numRows;
      const rows: Record<string, unknown>[] = [];
      for (let rowIdx = 0; rowIdx < numRows; rowIdx++) {
        const row: Record<string, unknown> = {};
        for (let colIdx = 0; colIdx < fields.length; colIdx++) {
          const col = res.getChildAt(colIdx);
          row[fields[colIdx]!.name] = col !== null ? col.get(rowIdx) : null;
        }
        rows.push(row);
      }
      return rows;
    }

    return [];
  }

  scan(
    foreignTable: ForeignTable,
    columns?: string[] | null,
    predicates?: FDWPredicate[] | null,
    limit?: number | null,
  ): Record<string, unknown>[] {
    // Fallback: inline JSON data for testing without @duckdb/duckdb-wasm
    const jsonData = foreignTable.options["data"] ?? this._server.options["data"];
    if (jsonData) {
      let rows = JSON.parse(jsonData) as Record<string, unknown>[];
      // Apply predicates manually
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
              default:
                return true;
            }
          }),
        );
      }
      if (limit !== null && limit !== undefined && limit >= 0) {
        rows = rows.slice(0, limit);
      }
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

    // Try loading @duckdb/duckdb-wasm
    const duckdb = this._loadDuckDB();
    if (duckdb === null) {
      throw new Error(
        "DuckDBFDWHandler.scan() requires @duckdb/duckdb-wasm. " +
          "Install the @duckdb/duckdb-wasm package and ensure it is available at runtime.",
      );
    }

    // Use synchronous path if connection is already established
    if (this._conn !== null) {
      const tableName = foreignTable.options["table"] ?? foreignTable.name;
      const sql = this._buildQuery(tableName, columns, predicates, limit);
      const conn = this._conn as { query(sql: string): unknown };
      const result = conn.query(sql);
      return this._resultToRows(result);
    }

    throw new Error(
      "DuckDBFDWHandler: no connection established. " +
        "Call initAsync() first or provide inline 'data' in table options.",
    );
  }

  /**
   * Initialize the DuckDB WASM connection asynchronously.
   * Must be called before scan() when using actual DuckDB.
   */
  async initAsync(): Promise<void> {
    const duckdb = this._loadDuckDB();
    if (duckdb === null) {
      throw new Error(
        "DuckDBFDWHandler requires @duckdb/duckdb-wasm. " +
          "Install the @duckdb/duckdb-wasm package.",
      );
    }

    const lib = duckdb as {
      selectBundle(bundles: unknown): Promise<unknown>;
      ConsoleLogger: new () => unknown;
      AsyncDuckDB: new (
        logger: unknown,
        worker: unknown,
      ) => {
        instantiate(bundle: unknown): Promise<void>;
        connect(): Promise<unknown>;
        close(): Promise<void>;
      };
      getJsDelivrBundles?(): unknown;
    };

    // For Node.js environments or custom bundles
    const dbPath = this._server.options["path"] ?? ":memory:";
    void dbPath;

    if (lib.getJsDelivrBundles) {
      const bundle = await lib.selectBundle(lib.getJsDelivrBundles());
      const logger = new lib.ConsoleLogger();
      const db = new lib.AsyncDuckDB(logger, null);
      await db.instantiate(bundle);
      this._db = db;
      this._conn = await db.connect();
    }
  }

  close(): void {
    if (this._conn !== null) {
      const conn = this._conn as { close?(): void };
      if (conn.close) conn.close();
      this._conn = null;
    }
    if (this._db !== null) {
      const db = this._db as { close?(): Promise<void> };
      if (db.close) {
        db.close().catch(() => {
          // Ignore close errors during cleanup
        });
      }
      this._db = null;
    }
    this._duckdbLib = null;
  }
}
