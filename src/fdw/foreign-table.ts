//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Foreign table and server definitions
// 1:1 port of uqa/fdw/foreign_table.py

import type { ColumnDef } from "../sql/table.js";

// ---------------------------------------------------------------------------
// FDWPredicate -- pushdown predicate for foreign scans
// ---------------------------------------------------------------------------

export interface FDWColumnDef {
  readonly name: string;
  readonly type: string;
  readonly nullable: boolean;
  readonly defaultValue?: unknown;
}

export interface FDWPredicate {
  readonly column: string;
  readonly operator: string; // "=", "<>", "<", "<=", ">", ">=", "LIKE", "IN", "IS NULL", "IS NOT NULL", "BETWEEN"
  readonly value: unknown;
  readonly value2?: unknown; // second value for BETWEEN
}

// ---------------------------------------------------------------------------
// ForeignServer
// ---------------------------------------------------------------------------

export interface ForeignServer {
  readonly name: string;
  readonly fdwType: string;
  readonly options: Record<string, string>;
}

// ---------------------------------------------------------------------------
// ForeignTable
// ---------------------------------------------------------------------------

export interface ForeignTable {
  name: string;
  serverName: string;
  columns: Map<string, ColumnDef>;
  options: Record<string, string>;
}

// ---------------------------------------------------------------------------
// ForeignTable factory helpers
// ---------------------------------------------------------------------------

export function createForeignServer(
  name: string,
  fdwType: string,
  options?: Record<string, string>,
): ForeignServer {
  return { name, fdwType, options: options ?? {} };
}

export function createForeignTable(
  name: string,
  serverName: string,
  columns: Map<string, ColumnDef>,
  options?: Record<string, string>,
): ForeignTable {
  return { name, serverName, columns, options: options ?? {} };
}

/**
 * Build a FDWPredicate for equality comparison.
 */
export function fdwEquals(column: string, value: unknown): FDWPredicate {
  return { column, operator: "=", value };
}

/**
 * Build a FDWPredicate for a range (BETWEEN) comparison.
 */
export function fdwBetween(column: string, low: unknown, high: unknown): FDWPredicate {
  return { column, operator: "BETWEEN", value: low, value2: high };
}

/**
 * Build a FDWPredicate for IS NULL.
 */
export function fdwIsNull(column: string): FDWPredicate {
  return { column, operator: "IS NULL", value: null };
}

/**
 * Build a FDWPredicate for IS NOT NULL.
 */
export function fdwIsNotNull(column: string): FDWPredicate {
  return { column, operator: "IS NOT NULL", value: null };
}

/**
 * Build a FDWPredicate for IN list.
 */
export function fdwIn(column: string, values: unknown[]): FDWPredicate {
  return { column, operator: "IN", value: values };
}
