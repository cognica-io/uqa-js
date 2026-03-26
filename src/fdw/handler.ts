//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Foreign Data Wrapper handler base
// 1:1 port of uqa/fdw/handler.py
//
// Abstract base class for Foreign Data Wrapper handlers.
//
// Each handler knows how to connect to a specific kind of external data
// source (e.g. a DuckDB in-process database or an Arrow Flight SQL endpoint)
// and return query results as rows.
//
// Implementations must provide scan() (fetch data) and close() (release resources).

import type { ForeignTable } from "./foreign-table.js";

/**
 * A single pushdown predicate for FDW handlers.
 *
 * Represents "column operator value" comparisons that can be pushed
 * down to the data source for server-side filtering (e.g. Hive
 * partition pruning, remote WHERE clauses).
 *
 * Supported operators:
 *   - Comparison: =, !=, <>, <, <=, >, >=
 *   - Set membership: IN
 *   - Pattern matching: LIKE, NOT LIKE, ILIKE, NOT ILIKE
 */
export interface FDWPredicate {
  /** Column name. */
  readonly column: string;
  /** Comparison operator (e.g. "=", "<", "IN", "LIKE"). */
  readonly operator: string;
  /**
   * Literal value for the comparison.
   * Scalar (number, string, boolean, null) for comparisons and pattern operators;
   * array of scalars for IN.
   */
  readonly value: unknown;
}

/**
 * Abstract interface for scanning external data sources.
 *
 * Concrete implementations (DuckDBFDWHandler, ArrowFlightSQLFDWHandler)
 * translate the scan request into native queries against the external
 * data source and return results as rows of key-value objects.
 */
export abstract class FDWHandler {
  /**
   * Scan the foreign table and return rows.
   *
   * @param foreignTable - The foreign table metadata.
   * @param columns - Optional column projection (all columns if null).
   * @param predicates - Pushdown predicates for server-side filtering.
   *     Each FDWPredicate carries a column name, a comparison operator,
   *     and a literal value. Handlers translate these into native filter
   *     expressions (e.g. SQL WHERE clauses) so the data source can prune
   *     data before transmission -- critical for Hive partition pruning
   *     and remote query efficiency.
   * @param limit - Optional row limit pushed down to the data source.
   * @returns Array of row objects containing the requested data.
   */
  abstract scan(
    foreignTable: ForeignTable,
    columns?: string[] | null,
    predicates?: FDWPredicate[] | null,
    limit?: number | null,
  ): Record<string, unknown>[];

  /** Release any resources held by this handler. */
  abstract close(): void;
}
