//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Apache Arrow FDW handler
// 1:1 port of uqa/fdw/arrow_handler.py
//
// Scans Apache Arrow IPC files (Feather / Arrow) or RecordBatch data.
// When the `apache-arrow` package is available, uses it to decode Arrow IPC
// data from a URL or buffer specified in the foreign server options.

import type { FDWPredicate } from "./handler.js";
import { FDWHandler } from "./handler.js";
import type { ForeignServer } from "./foreign-table.js";
import type { ForeignTable } from "./foreign-table.js";

/**
 * Evaluate an FDW predicate against a row value.
 */
function evaluatePredicate(pred: FDWPredicate, value: unknown): boolean {
  const v = value;
  const target = pred.value;
  switch (pred.operator) {
    case "=":
      return v === target;
    case "<>":
    case "!=":
      return v !== target;
    case "<":
      return (v as number) < (target as number);
    case "<=":
      return (v as number) <= (target as number);
    case ">":
      return (v as number) > (target as number);
    case ">=":
      return (v as number) >= (target as number);
    case "IS NULL":
      return v === null || v === undefined;
    case "IS NOT NULL":
      return v !== null && v !== undefined;
    case "IN":
      return Array.isArray(target) && (target as unknown[]).includes(v);
    case "LIKE": {
      if (typeof v !== "string" || typeof target !== "string") return false;
      const re = new RegExp("^" + target.replace(/%/g, ".*").replace(/_/g, ".") + "$");
      return re.test(v);
    }
    default:
      return true;
  }
}

/**
 * Apply column projection and predicate pushdown to raw rows.
 */
function applyProjectionAndFilters(
  rows: Record<string, unknown>[],
  columns: string[] | null | undefined,
  predicates: FDWPredicate[] | null | undefined,
  limit: number | null | undefined,
): Record<string, unknown>[] {
  let result = rows;

  // Apply predicates
  if (predicates && predicates.length > 0) {
    result = result.filter((row) =>
      predicates.every((pred) => evaluatePredicate(pred, row[pred.column])),
    );
  }

  // Apply limit
  if (limit !== null && limit !== undefined && limit >= 0) {
    result = result.slice(0, limit);
  }

  // Apply column projection
  if (columns && columns.length > 0) {
    const colSet = new Set(columns);
    result = result.map((row) => {
      const projected: Record<string, unknown> = {};
      for (const col of colSet) {
        if (col in row) projected[col] = row[col];
      }
      return projected;
    });
  }

  return result;
}

export class ArrowFDWHandler extends FDWHandler {
  private _server: ForeignServer;
  private _cachedData: Record<string, unknown>[] | null;
  private _arrowLib: unknown;

  constructor(server: ForeignServer) {
    super();
    this._server = server;
    this._cachedData = null;
    this._arrowLib = null;
  }

  get server(): ForeignServer {
    return this._server;
  }

  /**
   * Attempt to load the apache-arrow library dynamically.
   */
  private _loadArrowLib(): unknown {
    if (this._arrowLib !== null) return this._arrowLib;
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      this._arrowLib = require("apache-arrow");
      return this._arrowLib;
    } catch {
      return null;
    }
  }

  /**
   * Convert Arrow table/RecordBatch to plain row objects.
   */
  private _arrowTableToRows(table: unknown): Record<string, unknown>[] {
    const tbl = table as {
      schema: { fields: Array<{ name: string }> };
      numRows: number;
      getChildAt(i: number): { get(j: number): unknown; length: number } | null;
    };

    const fields = tbl.schema.fields;
    const numRows = tbl.numRows;
    const rows: Record<string, unknown>[] = [];

    for (let rowIdx = 0; rowIdx < numRows; rowIdx++) {
      const row: Record<string, unknown> = {};
      for (let colIdx = 0; colIdx < fields.length; colIdx++) {
        const col = tbl.getChildAt(colIdx);
        row[fields[colIdx]!.name] = col !== null ? col.get(rowIdx) : null;
      }
      rows.push(row);
    }

    return rows;
  }

  /**
   * Load data from the Arrow source specified in server options.
   * Supports 'data' option (JSON-encoded rows for testing) and
   * 'buffer' option (base64-encoded Arrow IPC buffer).
   */
  private _loadData(foreignTable: ForeignTable): Record<string, unknown>[] {
    if (this._cachedData !== null) return this._cachedData;

    // Option 1: inline JSON data (for testing without apache-arrow)
    const jsonData = foreignTable.options["data"] ?? this._server.options["data"];
    if (jsonData) {
      this._cachedData = JSON.parse(jsonData) as Record<string, unknown>[];
      return this._cachedData;
    }

    // Option 2: base64-encoded Arrow IPC buffer
    const arrowLib = this._loadArrowLib();
    if (arrowLib === null) {
      throw new Error(
        "ArrowFDWHandler.scan() requires apache-arrow. " +
          "Install the apache-arrow package and ensure it is available at runtime.",
      );
    }

    const bufferData = foreignTable.options["buffer"] ?? this._server.options["buffer"];
    if (bufferData) {
      const tableFromIPC = (arrowLib as { tableFromIPC: (buf: Uint8Array) => unknown })
        .tableFromIPC;
      const bytes = Uint8Array.from(atob(bufferData), (c) => c.charCodeAt(0));
      const table = tableFromIPC(bytes);
      this._cachedData = this._arrowTableToRows(table);
      return this._cachedData;
    }

    throw new Error(
      "ArrowFDWHandler: no data source specified. " +
        "Set 'data' (JSON) or 'buffer' (base64 Arrow IPC) in server/table options.",
    );
  }

  scan(
    foreignTable: ForeignTable,
    columns?: string[] | null,
    predicates?: FDWPredicate[] | null,
    limit?: number | null,
  ): Record<string, unknown>[] {
    const rows = this._loadData(foreignTable);
    return applyProjectionAndFilters(rows, columns, predicates, limit);
  }

  close(): void {
    this._cachedData = null;
    this._arrowLib = null;
  }
}
