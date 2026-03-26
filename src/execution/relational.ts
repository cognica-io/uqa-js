//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- relational physical operators
// 1:1 port of uqa/execution/relational.py

import type { Predicate } from "../core/types.js";
import { ExprEvaluator } from "../sql/expr-evaluator.js";
import { Batch } from "./batch.js";
import { PhysicalOperator } from "./physical.js";
import { SpillManager, mergeSortedRuns } from "./spill.js";
import type { SortKeySpec } from "./spill.js";

// ---------------------------------------------------------------------------
// WindowSpec dataclass
// ---------------------------------------------------------------------------

export interface WindowSpec {
  readonly outputCol: string;
  readonly funcName: string;
  readonly partitionBy: string[];
  readonly orderBy: SortKey[];
  readonly inputCol?: string;
  readonly lagLeadOffset?: number;
  readonly lagLeadDefault?: unknown;
  readonly ntileBuckets?: number;
  readonly frameStart?: string | null; // "unbounded_preceding" | "current_row" | "offset_preceding" | "offset_following"
  readonly frameEnd?: string | null;
  readonly frameStartOffset?: number;
  readonly frameEndOffset?: number;
  readonly frameType?: string; // "rows" | "range"
  readonly filterNode?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function drainToRows(child: PhysicalOperator): Record<string, unknown>[] {
  const rows: Record<string, unknown>[] = [];
  for (;;) {
    const batch = child.next();
    if (batch === null) break;
    for (const row of batch.toRows()) rows.push(row);
  }
  return rows;
}

function rowKey(row: Record<string, unknown>, columns: string[]): string {
  const parts: string[] = [];
  for (const col of columns) {
    const v = row[col];
    if (v === null || v === undefined) {
      parts.push("\x00NULL");
    } else if (typeof v === "string") {
      parts.push("s:" + v);
    } else {
      parts.push(JSON.stringify(v));
    }
  }
  return parts.join("\x01");
}

function rowsEqualOnColumns(
  a: Record<string, unknown>,
  b: Record<string, unknown>,
  columns: string[],
): boolean {
  for (const c of columns) {
    if (a[c] !== b[c]) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// FilterOp
// ---------------------------------------------------------------------------

export class FilterOp extends PhysicalOperator {
  private readonly _child: PhysicalOperator;
  private readonly _column: string;
  private readonly _predicate: Predicate;

  constructor(child: PhysicalOperator, column: string, predicate: Predicate) {
    super();
    this._child = child;
    this._column = column;
    this._predicate = predicate;
  }

  open(): void {
    this._child.open();
  }

  next(): Batch | null {
    for (;;) {
      const batch = this._child.next();
      if (batch === null) return null;
      const rows = batch.toRows();
      const filtered: Record<string, unknown>[] = [];
      for (const row of rows) {
        if (this._predicate.evaluate(row[this._column])) filtered.push(row);
      }
      if (filtered.length > 0) return Batch.fromRows(filtered);
    }
  }

  close(): void {
    this._child.close();
  }
}

// ---------------------------------------------------------------------------
// ExprFilterOp
// ---------------------------------------------------------------------------

export class ExprFilterOp extends PhysicalOperator {
  private readonly _child: PhysicalOperator;
  private readonly _exprNode: Record<string, unknown>;
  private readonly _evaluator: ExprEvaluator;

  constructor(
    child: PhysicalOperator,
    exprNode: Record<string, unknown>,
    evaluator?: ExprEvaluator,
  ) {
    super();
    this._child = child;
    this._exprNode = exprNode;
    this._evaluator = evaluator ?? new ExprEvaluator();
  }

  open(): void {
    this._child.open();
  }

  next(): Batch | null {
    for (;;) {
      const batch = this._child.next();
      if (batch === null) return null;
      const rows = batch.toRows();
      const filtered: Record<string, unknown>[] = [];
      for (const row of rows) {
        const result = this._evaluator.evaluate(this._exprNode, row);
        if (result === true) filtered.push(row);
      }
      if (filtered.length > 0) return Batch.fromRows(filtered);
    }
  }

  close(): void {
    this._child.close();
  }
}

// ---------------------------------------------------------------------------
// ProjectOp
// ---------------------------------------------------------------------------

export class ProjectOp extends PhysicalOperator {
  private readonly _child: PhysicalOperator;
  private readonly _columns: string[];
  private readonly _aliases: Map<string, string>;

  constructor(
    child: PhysicalOperator,
    columns: string[],
    aliases?: Map<string, string>,
  ) {
    super();
    this._child = child;
    this._columns = columns;
    this._aliases = aliases ?? new Map<string, string>();
  }

  open(): void {
    this._child.open();
  }

  next(): Batch | null {
    const batch = this._child.next();
    if (batch === null) return null;
    return batch.selectColumns(this._columns, this._aliases);
  }

  close(): void {
    this._child.close();
  }
}

// ---------------------------------------------------------------------------
// ExprProjectOp
// ---------------------------------------------------------------------------

export interface ExprProjectSpec {
  readonly outputCol: string;
  readonly exprNode: Record<string, unknown>;
}

export class ExprProjectOp extends PhysicalOperator {
  private readonly _child: PhysicalOperator;
  private readonly _specs: ExprProjectSpec[];
  private readonly _evaluator: ExprEvaluator;

  constructor(
    child: PhysicalOperator,
    specs: ExprProjectSpec[],
    evaluator?: ExprEvaluator,
  ) {
    super();
    this._child = child;
    this._specs = specs;
    this._evaluator = evaluator ?? new ExprEvaluator();
  }

  open(): void {
    this._child.open();
  }

  next(): Batch | null {
    const batch = this._child.next();
    if (batch === null) return null;

    // Attempt vectorized evaluation for simple ColumnRef projections
    const vectorized = this._tryVectorized(batch);
    if (vectorized !== null) return vectorized;

    // Fallback: row-by-row evaluation
    const rows = batch.toRows();
    const result: Record<string, unknown>[] = [];
    for (const row of rows) {
      const outRow: Record<string, unknown> = {};
      for (const spec of this._specs) {
        outRow[spec.outputCol] = this._evaluator.evaluate(spec.exprNode, row);
      }
      result.push(outRow);
    }
    if (result.length === 0) return null;
    return Batch.fromRows(result);
  }

  /**
   * Attempt vectorized column copy for simple ColumnRef-only projections.
   * Returns null if any spec is not a simple column reference.
   */
  private _tryVectorized(batch: Batch): Batch | null {
    const cols = new Map<string, unknown[]>();
    for (const spec of this._specs) {
      const colName = this._extractColumnName(spec.exprNode);
      if (colName === null) return null; // not a simple column ref
      const srcCol = batch.getColumn(colName);
      if (srcCol === null) return null; // column not found
      cols.set(spec.outputCol, srcCol);
    }
    return new Batch(cols, batch.length);
  }

  /**
   * Extract a simple column name from a ColumnRef AST node.
   * Returns null if the node is not a simple ColumnRef.
   */
  private _extractColumnName(node: Record<string, unknown>): string | null {
    const keys = Object.keys(node);
    if (keys.length !== 1 || keys[0] !== "ColumnRef") return null;
    const inner = node["ColumnRef"] as Record<string, unknown> | undefined;
    if (!inner) return null;
    const fields = inner["fields"];
    if (!Array.isArray(fields) || fields.length === 0) return null;
    // Single-field column ref
    if (fields.length === 1) {
      const f = fields[0] as Record<string, unknown>;
      const strNode = f["String"] ?? f["str"];
      if (strNode !== null && strNode !== undefined && typeof strNode === "object") {
        const obj = strNode as Record<string, unknown>;
        return (
          (obj["sval"] as string | undefined) ??
          (obj["str"] as string | undefined) ??
          null
        );
      }
      if (typeof strNode === "string") return strNode;
      const sval = f["sval"];
      if (typeof sval === "string") return sval;
    }
    // Two-field: table.column -- just take the column part
    if (fields.length === 2) {
      const f = fields[1] as Record<string, unknown>;
      const strNode = f["String"] ?? f["str"];
      if (strNode !== null && strNode !== undefined && typeof strNode === "object") {
        const obj = strNode as Record<string, unknown>;
        return (
          (obj["sval"] as string | undefined) ??
          (obj["str"] as string | undefined) ??
          null
        );
      }
      if (typeof strNode === "string") return strNode;
    }
    return null;
  }

  close(): void {
    this._child.close();
  }
}

// ---------------------------------------------------------------------------
// SortOp
// ---------------------------------------------------------------------------

export interface SortKey {
  readonly column: string;
  readonly ascending: boolean;
  readonly nullsFirst: boolean;
}

export class SortOp extends PhysicalOperator {
  private readonly _child: PhysicalOperator;
  private readonly _sortKeys: SortKey[];
  private _sorted: Record<string, unknown>[] | null = null;
  private _index = 0;
  private readonly _batchSize: number;
  private readonly _spillThreshold: number;

  private _spillManager: SpillManager | null = null;

  constructor(
    child: PhysicalOperator,
    sortKeys: SortKey[],
    batchSize = 1024,
    spillThreshold = 0,
  ) {
    super();
    this._child = child;
    this._sortKeys = sortKeys;
    this._batchSize = batchSize;
    this._spillThreshold = spillThreshold;
  }

  open(): void {
    this._child.open();

    if (this._spillThreshold > 0) {
      // External merge sort: split into sorted runs, then merge
      this._sorted = this._externalMergeSort();
    } else {
      // In-memory sort
      const rows = drainToRows(this._child);
      this._child.close();
      SortOp._sortRows(rows, this._sortKeys);
      this._sorted = rows;
    }
    this._index = 0;
  }

  /**
   * External merge sort: read child in chunks of _spillThreshold rows,
   * sort each chunk in memory, spill to SpillManager, then k-way merge
   * all sorted runs.
   */
  private _externalMergeSort(): Record<string, unknown>[] {
    const spill = new SpillManager();
    this._spillManager = spill;
    const runSize = this._spillThreshold;

    let buffer: Record<string, unknown>[] = [];
    const runIndices: number[] = [];

    for (;;) {
      const batch = this._child.next();
      if (batch === null) break;
      for (const row of batch.toRows()) {
        buffer.push(row);
        if (buffer.length >= runSize) {
          // Sort and spill
          SortOp._sortRows(buffer, this._sortKeys);
          const runIdx = spill.newRun();
          spill.writeRows(runIdx, buffer);
          runIndices.push(runIdx);
          buffer = [];
        }
      }
    }
    this._child.close();

    // Final partial run
    if (buffer.length > 0) {
      SortOp._sortRows(buffer, this._sortKeys);
      const runIdx = spill.newRun();
      spill.writeRows(runIdx, buffer);
      runIndices.push(runIdx);
    }

    if (runIndices.length === 0) return [];
    if (runIndices.length === 1) return spill.readRows(runIndices[0]!);

    // K-way merge
    const runs = runIndices.map((idx) => spill.readRows(idx));
    const sortKeySpecs: SortKeySpec[] = this._sortKeys.map((k) => [
      k.column,
      k.ascending,
      k.nullsFirst,
    ]);
    return mergeSortedRuns(runs, sortKeySpecs);
  }

  next(): Batch | null {
    if (this._sorted === null || this._index >= this._sorted.length) return null;
    const end = Math.min(this._index + this._batchSize, this._sorted.length);
    const slice = this._sorted.slice(this._index, end);
    this._index = end;
    return Batch.fromRows(slice);
  }

  close(): void {
    this._child.close();
    this._sorted = null;
    this._index = 0;
    if (this._spillManager !== null) {
      this._spillManager.cleanup();
      this._spillManager = null;
    }
  }

  static _sortRows(rows: Record<string, unknown>[], sortKeys: SortKey[]): void {
    rows.sort((a, b) => {
      for (const key of sortKeys) {
        const av = a[key.column];
        const bv = b[key.column];
        const aNull = av === null || av === undefined;
        const bNull = bv === null || bv === undefined;
        if (aNull && bNull) continue;
        if (aNull) return key.nullsFirst ? -1 : 1;
        if (bNull) return key.nullsFirst ? 1 : -1;
        let cmp: number;
        if (typeof av === "number" && typeof bv === "number") {
          cmp = av - bv;
        } else if (typeof av === "string" && typeof bv === "string") {
          cmp = av < bv ? -1 : av > bv ? 1 : 0;
        } else {
          const sa = String(av as string | number);
          const sb = String(bv as string | number);
          cmp = sa < sb ? -1 : sa > sb ? 1 : 0;
        }
        if (!key.ascending) cmp = -cmp;
        if (cmp !== 0) return cmp;
      }
      return 0;
    });
  }

  private _compare(a: Record<string, unknown>, b: Record<string, unknown>): number {
    for (const key of this._sortKeys) {
      const av = a[key.column];
      const bv = b[key.column];
      const aNull = av === null || av === undefined;
      const bNull = bv === null || bv === undefined;
      if (aNull && bNull) continue;
      if (aNull) return key.nullsFirst ? -1 : 1;
      if (bNull) return key.nullsFirst ? 1 : -1;
      let cmp: number;
      if (typeof av === "number" && typeof bv === "number") {
        cmp = av - bv;
      } else {
        const sa = String(av as string | number);
        const sb = String(bv as string | number);
        cmp = sa < sb ? -1 : sa > sb ? 1 : 0;
      }
      if (!key.ascending) cmp = -cmp;
      if (cmp !== 0) return cmp;
    }
    return 0;
  }
}

// ---------------------------------------------------------------------------
// LimitOp
// ---------------------------------------------------------------------------

export class LimitOp extends PhysicalOperator {
  private readonly _child: PhysicalOperator;
  private readonly _limit: number;
  private readonly _offset: number;
  private _emitted = 0;
  private _skipped = 0;

  constructor(child: PhysicalOperator, limit: number, offset = 0) {
    super();
    this._child = child;
    this._limit = limit;
    this._offset = offset;
  }

  open(): void {
    this._child.open();
    this._emitted = 0;
    this._skipped = 0;
  }

  next(): Batch | null {
    if (this._emitted >= this._limit) return null;
    for (;;) {
      const batch = this._child.next();
      if (batch === null) return null;
      const rows = batch.toRows();
      const result: Record<string, unknown>[] = [];
      for (const row of rows) {
        if (this._skipped < this._offset) {
          this._skipped++;
          continue;
        }
        if (this._emitted >= this._limit) break;
        result.push(row);
        this._emitted++;
      }
      if (result.length > 0) return Batch.fromRows(result);
      if (this._emitted >= this._limit) return null;
    }
  }

  close(): void {
    this._child.close();
    this._emitted = 0;
    this._skipped = 0;
  }
}

// ---------------------------------------------------------------------------
// HashAggOp
// ---------------------------------------------------------------------------

export interface AggregateSpec {
  readonly outputCol: string;
  readonly funcName: string;
  readonly inputCol?: string;
  readonly distinct?: boolean;
  readonly extra?: unknown;
  readonly filterNode?: Record<string, unknown>;
  readonly orderKeys?: [string, boolean][];
}

export class HashAggOp extends PhysicalOperator {
  private readonly _child: PhysicalOperator;
  private readonly _groupByCols: string[];
  private readonly _aggSpecs: AggregateSpec[];
  private readonly _groupAliases: Map<string, string>;
  private _result: Record<string, unknown>[] | null = null;
  private _index = 0;
  private readonly _batchSize: number;
  private readonly _spillThreshold: number;
  private _filterEvaluator: ExprEvaluator | null = null;
  private static readonly NUM_PARTITIONS = 16;

  constructor(
    child: PhysicalOperator,
    groupByCols: string[],
    aggSpecs: AggregateSpec[],
    opts?: {
      batchSize?: number;
      groupAliases?: Map<string, string>;
      spillThreshold?: number;
    },
  ) {
    super();
    this._child = child;
    this._groupByCols = groupByCols;
    this._aggSpecs = aggSpecs;
    this._batchSize = opts?.batchSize ?? 1024;
    this._groupAliases = opts?.groupAliases ?? new Map<string, string>();
    this._spillThreshold = opts?.spillThreshold ?? 0;
  }

  open(): void {
    this._child.open();
    const hasFilter = this._aggSpecs.some((s) => s.filterNode != null);
    if (hasFilter) this._filterEvaluator = new ExprEvaluator();

    const rows = drainToRows(this._child);
    this._child.close();

    // If spill threshold is set and data exceeds it, use partition-based strategy
    if (this._spillThreshold > 0 && rows.length > this._spillThreshold) {
      this._result = this._aggregateWithSpill(rows);
    } else if (this._canStream()) {
      this._result = this._aggregateStreaming(rows);
    } else {
      this._result = this._aggregateMaterialized(rows);
    }
    this._index = 0;
  }

  /**
   * 16-partition spill strategy for large aggregations.
   * Hash-partition rows into 16 buckets, aggregate each bucket independently,
   * then combine results.
   */
  private _aggregateWithSpill(
    rows: Record<string, unknown>[],
  ): Record<string, unknown>[] {
    const numParts = HashAggOp.NUM_PARTITIONS;
    const partitions: Record<string, unknown>[][] = [];
    for (let i = 0; i < numParts; i++) {
      partitions.push([]);
    }

    // Hash-partition rows
    for (const row of rows) {
      const key = rowKey(row, this._groupByCols);
      const hash = HashAggOp._hashString(key);
      const partIdx = hash % numParts;
      partitions[partIdx]!.push(row);
    }

    // Aggregate each partition independently
    const allResults: Record<string, unknown>[] = [];
    for (const partRows of partitions) {
      if (partRows.length === 0) continue;
      const partResult = this._aggregateMaterialized(partRows);
      for (const r of partResult) {
        allResults.push(r);
      }
    }

    return allResults;
  }

  /**
   * Simple string hash for partitioning.
   */
  private static _hashString(s: string): number {
    let hash = 0;
    for (let i = 0; i < s.length; i++) {
      hash = ((hash << 5) - hash + s.charCodeAt(i)) | 0;
    }
    return Math.abs(hash);
  }

  private _canStream(): boolean {
    const incremental = new Set([
      "COUNT",
      "SUM",
      "AVG",
      "MIN",
      "MAX",
      "BOOL_AND",
      "BOOL_OR",
    ]);
    for (const spec of this._aggSpecs) {
      if (!incremental.has(spec.funcName)) return false;
      if (
        spec.distinct ||
        spec.filterNode != null ||
        (spec.orderKeys && spec.orderKeys.length > 0)
      )
        return false;
    }
    return true;
  }

  private _aggregateStreaming(
    rows: Record<string, unknown>[],
  ): Record<string, unknown>[] {
    const accumulators = new Map<string, [number, number, unknown, unknown][]>();
    const groupKeysOrder: string[] = [];
    const groupFirstRow = new Map<string, Record<string, unknown>>();

    for (const row of rows) {
      const key = rowKey(row, this._groupByCols);
      if (!accumulators.has(key)) {
        accumulators.set(
          key,
          this._aggSpecs.map(() => [0, 0, undefined, undefined]),
        );
        groupKeysOrder.push(key);
        groupFirstRow.set(key, row);
      }
      const accs = accumulators.get(key)!;
      for (let i = 0; i < this._aggSpecs.length; i++) {
        const spec = this._aggSpecs[i]!;
        const acc = accs[i]!;
        const fn = spec.funcName;
        if (fn === "COUNT" && spec.inputCol == null) {
          acc[0]++;
          continue;
        }
        const val = spec.inputCol != null ? row[spec.inputCol] : undefined;
        if (fn === "COUNT") {
          if (val != null) acc[0]++;
        } else if (fn === "SUM" || fn === "AVG") {
          if (typeof val === "number") {
            acc[0]++;
            acc[1] = acc[1] + val;
          }
        } else if (fn === "MIN") {
          if (val != null && (typeof val === "number" || typeof val === "string")) {
            acc[0]++;
            if (acc[2] === undefined || val < (acc[2] as typeof val)) acc[2] = val;
          }
        } else if (fn === "MAX") {
          if (val != null && (typeof val === "number" || typeof val === "string")) {
            acc[0]++;
            if (acc[3] === undefined || val > (acc[3] as typeof val)) acc[3] = val;
          }
        } else if (fn === "BOOL_AND") {
          if (val != null) {
            acc[0]++;
            acc[2] =
              acc[2] === undefined ? Boolean(val) : (acc[2] as boolean) && Boolean(val);
          }
        } else if (fn === "BOOL_OR") {
          if (val != null) {
            acc[0]++;
            acc[2] =
              acc[2] === undefined ? Boolean(val) : (acc[2] as boolean) || Boolean(val);
          }
        }
      }
    }

    if (accumulators.size === 0 && this._groupByCols.length === 0) {
      accumulators.set(
        "",
        this._aggSpecs.map(() => [0, 0, undefined, undefined]),
      );
      groupKeysOrder.push("");
    }

    const result: Record<string, unknown>[] = [];
    for (const key of groupKeysOrder) {
      const rowOut: Record<string, unknown> = {};
      // Reconstruct group columns from first matching row (simplified)
      const accs = accumulators.get(key)!;
      // Use stored first row for group column values
      const groupRow = groupFirstRow.get(key);
      if (groupRow) {
        for (const col of this._groupByCols) {
          rowOut[col] = groupRow[col];
          const alias = this._groupAliases.get(col);
          if (alias && alias !== col) rowOut[alias] = groupRow[col];
        }
      }
      for (let i = 0; i < this._aggSpecs.length; i++) {
        const spec = this._aggSpecs[i]!;
        const acc = accs[i]!;
        const cnt = acc[0];
        let value: unknown;
        if (spec.funcName === "COUNT") value = cnt;
        else if (spec.funcName === "SUM") value = cnt > 0 ? acc[1] : null;
        else if (spec.funcName === "AVG") value = cnt > 0 ? acc[1] / cnt : null;
        else if (spec.funcName === "MIN") value = acc[2] ?? null;
        else if (spec.funcName === "MAX") value = acc[3] ?? null;
        else if (spec.funcName === "BOOL_AND" || spec.funcName === "BOOL_OR")
          value = acc[2] ?? null;
        else value = null;
        rowOut[spec.outputCol] = value;
        // Also store with natural name (func_col) for compatibility
        const argCol = spec.inputCol;
        const natural =
          argCol == null
            ? spec.funcName.toLowerCase()
            : `${spec.funcName.toLowerCase()}_${argCol}`;
        if (natural !== spec.outputCol) {
          rowOut[natural] = value;
        }
      }
      result.push(rowOut);
    }
    return result;
  }

  private _aggregateMaterialized(
    rows: Record<string, unknown>[],
  ): Record<string, unknown>[] {
    const groups = new Map<string, Record<string, unknown>[]>();
    const groupOrder: string[] = [];
    for (const row of rows) {
      const key = rowKey(row, this._groupByCols);
      if (!groups.has(key)) {
        groups.set(key, []);
        groupOrder.push(key);
      }
      groups.get(key)!.push(row);
    }
    if (groups.size === 0 && this._groupByCols.length === 0) {
      groups.set("", []);
      groupOrder.push("");
    }

    const result: Record<string, unknown>[] = [];
    for (const key of groupOrder) {
      const groupRows = groups.get(key)!;
      const rowOut: Record<string, unknown> = {};
      if (groupRows.length > 0) {
        for (const col of this._groupByCols) {
          rowOut[col] = groupRows[0]![col];
          const alias = this._groupAliases.get(col);
          if (alias && alias !== col) rowOut[alias] = groupRows[0]![col];
        }
      }
      for (const spec of this._aggSpecs) {
        let aggRows = groupRows;
        if (spec.filterNode != null && this._filterEvaluator) {
          aggRows = aggRows.filter((r) =>
            this._filterEvaluator!.evaluate(spec.filterNode!, r),
          );
        }
        if (spec.orderKeys && spec.orderKeys.length > 0) {
          aggRows = [...aggRows];
          for (let k = spec.orderKeys.length - 1; k >= 0; k--) {
            const [col, desc] = spec.orderKeys[k]!;
            aggRows.sort((a, b) => {
              const av = a[col];
              const bv = b[col];
              if (av == null && bv == null) return 0;
              if (av == null) return 1;
              if (bv == null) return -1;
              const cmp =
                (av as number) < (bv as number)
                  ? -1
                  : (av as number) > (bv as number)
                    ? 1
                    : 0;
              return desc ? -cmp : cmp;
            });
          }
        }
        const value = computeAggregate(
          spec.funcName,
          spec.inputCol ?? null,
          aggRows,
          spec.distinct,
          spec.extra,
        );
        rowOut[spec.outputCol] = value;
      }
      result.push(rowOut);
    }
    return result;
  }

  next(): Batch | null {
    if (this._result === null || this._index >= this._result.length) return null;
    const end = Math.min(this._index + this._batchSize, this._result.length);
    const slice = this._result.slice(this._index, end);
    this._index = end;
    return Batch.fromRows(slice);
  }

  close(): void {
    this._child.close();
    this._result = null;
    this._filterEvaluator = null;
    this._index = 0;
  }
}

// ---------------------------------------------------------------------------
// DistinctOp
// ---------------------------------------------------------------------------

export class DistinctOp extends PhysicalOperator {
  private readonly _child: PhysicalOperator;
  private readonly _columns: string[];
  private readonly _spillThreshold: number;
  private _seen: Set<string> = new Set();
  private _spillPartitions: Map<number, Set<string>> | null = null;
  private static readonly NUM_PARTITIONS = 16;

  constructor(child: PhysicalOperator, columns: string[], spillThreshold = 0) {
    super();
    this._child = child;
    this._columns = columns;
    this._spillThreshold = spillThreshold;
  }

  open(): void {
    this._child.open();
    this._seen = new Set();
    this._spillPartitions = null;
  }

  next(): Batch | null {
    for (;;) {
      const batch = this._child.next();
      if (batch === null) return null;
      const rows = batch.toRows();
      const deduped: Record<string, unknown>[] = [];
      for (const row of rows) {
        const key = rowKey(row, this._columns);

        // If we exceed the spill threshold, switch to partitioned dedup
        if (
          this._spillThreshold > 0 &&
          this._seen.size >= this._spillThreshold &&
          this._spillPartitions === null
        ) {
          this._spillPartitions = new Map();
          // Distribute existing keys into partitions
          for (const existingKey of this._seen) {
            const partIdx =
              DistinctOp._hashKey(existingKey) % DistinctOp.NUM_PARTITIONS;
            if (!this._spillPartitions.has(partIdx)) {
              this._spillPartitions.set(partIdx, new Set());
            }
            this._spillPartitions.get(partIdx)!.add(existingKey);
          }
          this._seen.clear(); // Free main set memory
        }

        if (this._spillPartitions !== null) {
          // Partitioned dedup
          const partIdx = DistinctOp._hashKey(key) % DistinctOp.NUM_PARTITIONS;
          if (!this._spillPartitions.has(partIdx)) {
            this._spillPartitions.set(partIdx, new Set());
          }
          const partSet = this._spillPartitions.get(partIdx)!;
          if (!partSet.has(key)) {
            partSet.add(key);
            deduped.push(row);
          }
        } else {
          // Standard in-memory dedup
          if (!this._seen.has(key)) {
            this._seen.add(key);
            deduped.push(row);
          }
        }
      }
      if (deduped.length > 0) return Batch.fromRows(deduped);
    }
  }

  private static _hashKey(s: string): number {
    let hash = 0;
    for (let i = 0; i < s.length; i++) {
      hash = ((hash << 5) - hash + s.charCodeAt(i)) | 0;
    }
    return Math.abs(hash);
  }

  close(): void {
    this._child.close();
    this._seen = new Set();
    this._spillPartitions = null;
  }
}

// ---------------------------------------------------------------------------
// WindowOp
// ---------------------------------------------------------------------------

export class WindowOp extends PhysicalOperator {
  private readonly _child: PhysicalOperator;
  private readonly _windowSpecs: WindowSpec[];
  private _result: Record<string, unknown>[] | null = null;
  private _index = 0;
  private readonly _batchSize: number;

  constructor(child: PhysicalOperator, windowSpecs: WindowSpec[], batchSize = 1024) {
    super();
    this._child = child;
    this._windowSpecs = windowSpecs;
    this._batchSize = batchSize;
  }

  open(): void {
    this._child.open();
    const allRows = drainToRows(this._child);
    this._child.close();

    for (const spec of this._windowSpecs) {
      // Sort by partition keys + order keys
      const sortedRows = [...allRows];
      for (let k = spec.orderBy.length - 1; k >= 0; k--) {
        const key = spec.orderBy[k]!;
        sortedRows.sort((a, b) => {
          const av = a[key.column];
          const bv = b[key.column];
          if (av == null && bv == null) return 0;
          if (av == null) return 1;
          if (bv == null) return -1;
          const cmp =
            (av as number) < (bv as number)
              ? -1
              : (av as number) > (bv as number)
                ? 1
                : 0;
          return key.ascending ? cmp : -cmp;
        });
      }
      if (spec.partitionBy.length > 0) {
        sortedRows.sort((a, b) => {
          for (const col of spec.partitionBy) {
            const av = a[col];
            const bv = b[col];
            if (av == null && bv == null) continue;
            if (av == null) return -1;
            if (bv == null) return 1;
            if (av < bv) return -1;
            if (av > bv) return 1;
          }
          return 0;
        });
      }

      // Build row_id -> sorted_idx mapping
      const rowIdToSortedIdx = new Map<unknown, number>();
      for (let idx = 0; idx < sortedRows.length; idx++) {
        rowIdToSortedIdx.set(sortedRows[idx], idx);
      }

      // Partition sorted rows
      const partitions: number[][] = [];
      let currentKey: string | null = null;
      for (let idx = 0; idx < sortedRows.length; idx++) {
        const key = rowKey(sortedRows[idx]!, spec.partitionBy);
        if (key !== currentKey) {
          partitions.push([]);
          currentKey = key;
        }
        partitions[partitions.length - 1]!.push(idx);
      }

      // Compute window values
      const winValues = new Map<number, unknown>();
      for (const partIndices of partitions) {
        const partRows = partIndices.map((i) => sortedRows[i]!);
        const values = computeWindowFunction(spec, partRows);
        for (let i = 0; i < partIndices.length; i++) {
          winValues.set(partIndices[i]!, values[i]);
        }
      }

      // Apply computed values back to original rows
      for (const row of allRows) {
        const sortedIdx = rowIdToSortedIdx.get(row);
        if (sortedIdx !== undefined) {
          row[spec.outputCol] = winValues.get(sortedIdx);
        }
      }
    }

    this._result = allRows;
    this._index = 0;
  }

  next(): Batch | null {
    if (this._result === null || this._index >= this._result.length) return null;
    const end = Math.min(this._index + this._batchSize, this._result.length);
    const slice = this._result.slice(this._index, end);
    this._index = end;
    return Batch.fromRows(slice);
  }

  close(): void {
    this._result = null;
    this._index = 0;
  }
}

// ---------------------------------------------------------------------------
// Window function computation
// ---------------------------------------------------------------------------

function computeWindowFunction(
  spec: WindowSpec,
  partitionRows: Record<string, unknown>[],
): unknown[] {
  const fn = spec.funcName;
  const n = partitionRows.length;
  const orderCols = spec.orderBy.map((k) => k.column);

  if (fn === "ROW_NUMBER") return Array.from({ length: n }, (_, i) => i + 1);

  if (fn === "RANK") {
    const ranks: number[] = [];
    for (let i = 0; i < n; i++) {
      if (i === 0) ranks.push(1);
      else if (rowsEqualOnColumns(partitionRows[i - 1]!, partitionRows[i]!, orderCols))
        ranks.push(ranks[ranks.length - 1]!);
      else ranks.push(i + 1);
    }
    return ranks;
  }

  if (fn === "DENSE_RANK") {
    const ranks: number[] = [];
    let currentRank = 0;
    for (let i = 0; i < n; i++) {
      if (
        i === 0 ||
        !rowsEqualOnColumns(partitionRows[i - 1]!, partitionRows[i]!, orderCols)
      )
        currentRank++;
      ranks.push(currentRank);
    }
    return ranks;
  }

  if (fn === "NTILE") {
    const buckets = spec.ntileBuckets ?? spec.lagLeadOffset ?? 1;
    return Array.from({ length: n }, (_, i) => Math.floor((i * buckets) / n) + 1);
  }

  if (fn === "LAG") {
    const offset = spec.lagLeadOffset ?? 1;
    const def = spec.lagLeadDefault ?? null;
    return Array.from({ length: n }, (_, i) =>
      i - offset >= 0
        ? spec.inputCol != null
          ? partitionRows[i - offset]![spec.inputCol]
          : null
        : def,
    );
  }

  if (fn === "LEAD") {
    const offset = spec.lagLeadOffset ?? 1;
    const def = spec.lagLeadDefault ?? null;
    return Array.from({ length: n }, (_, i) =>
      i + offset < n
        ? spec.inputCol != null
          ? partitionRows[i + offset]![spec.inputCol]
          : null
        : def,
    );
  }

  if (fn === "PERCENT_RANK") {
    if (n <= 1) return Array(n).fill(0.0);
    const ranks: number[] = [];
    for (let i = 0; i < n; i++) {
      if (i === 0) ranks.push(1);
      else if (rowsEqualOnColumns(partitionRows[i - 1]!, partitionRows[i]!, orderCols))
        ranks.push(ranks[ranks.length - 1]!);
      else ranks.push(i + 1);
    }
    return ranks.map((r) => (r - 1) / (n - 1));
  }

  if (fn === "CUME_DIST") {
    if (n === 0) return [];
    const values: number[] = new Array<number>(n).fill(0);
    let i = 0;
    while (i < n) {
      let j = i + 1;
      while (
        j < n &&
        rowsEqualOnColumns(partitionRows[i]!, partitionRows[j]!, orderCols)
      )
        j++;
      const val = j / n;
      for (let k = i; k < j; k++) values[k] = val;
      i = j;
    }
    return values;
  }

  if (fn === "NTH_VALUE") {
    const nth = spec.ntileBuckets ?? spec.lagLeadOffset ?? 1;
    if (n === 0 || nth < 1 || nth > n) return Array(n).fill(null);
    const val = spec.inputCol != null ? partitionRows[nth - 1]![spec.inputCol] : null;
    return Array(n).fill(val);
  }

  if (fn === "FIRST_VALUE") {
    if (n === 0) return [];
    const val = spec.inputCol != null ? partitionRows[0]![spec.inputCol] : null;
    return Array(n).fill(val);
  }

  if (fn === "LAST_VALUE") {
    if (n === 0) return [];
    const val = spec.inputCol != null ? partitionRows[n - 1]![spec.inputCol] : null;
    return Array(n).fill(val);
  }

  // Aggregate window functions
  const aggFuncs = new Set([
    "SUM",
    "COUNT",
    "AVG",
    "MIN",
    "MAX",
    "STRING_AGG",
    "ARRAY_AGG",
    "BOOL_AND",
    "BOOL_OR",
  ]);

  if (aggFuncs.has(fn)) {
    let filteredRows = partitionRows;
    if (spec.filterNode != null) {
      const evaluator = new ExprEvaluator();
      filteredRows = partitionRows.filter((r) =>
        evaluator.evaluate(spec.filterNode!, r),
      );
    }

    // Frame support
    if (spec.frameStart != null) {
      return computeFramedAggregate(fn, spec.inputCol ?? null, filteredRows, spec);
    }
    // Default frame when ORDER BY present
    if (spec.orderBy.length > 0) {
      const defaultSpec: WindowSpec = {
        ...spec,
        frameStart: "unbounded_preceding",
        frameEnd: "current_row",
        frameType: "rows",
      };
      return computeFramedAggregate(
        fn,
        spec.inputCol ?? null,
        filteredRows,
        defaultSpec,
      );
    }
    const aggVal = computeAggregate(fn, spec.inputCol ?? null, filteredRows);
    return Array(n).fill(aggVal);
  }

  throw new Error(`Unknown window function: ${fn}`);
}

function computeFramedAggregate(
  funcName: string,
  argCol: string | null,
  partitionRows: Record<string, unknown>[],
  spec: WindowSpec,
): unknown[] {
  const n = partitionRows.length;
  const results: unknown[] = [];

  if (spec.frameType === "range" && spec.orderBy.length > 0) {
    const orderCol = spec.orderBy[0]!.column;
    for (let i = 0; i < n; i++) {
      const start = resolveRangeFrameIndex(
        i,
        n,
        partitionRows,
        orderCol,
        spec.frameStart ?? null,
        spec.frameStartOffset ?? 0,
        true,
      );
      const end = resolveRangeFrameIndex(
        i,
        n,
        partitionRows,
        orderCol,
        spec.frameEnd ?? null,
        spec.frameEndOffset ?? 0,
        false,
      );
      const frameRows = partitionRows.slice(start, end + 1);
      results.push(computeAggregate(funcName, argCol, frameRows));
    }
    return results;
  }

  for (let i = 0; i < n; i++) {
    const start = resolveFrameIndex(
      i,
      n,
      spec.frameStart ?? null,
      spec.frameStartOffset ?? 0,
    );
    const end = resolveFrameIndex(
      i,
      n,
      spec.frameEnd ?? null,
      spec.frameEndOffset ?? 0,
    );
    const frameRows = partitionRows.slice(start, end + 1);
    results.push(computeAggregate(funcName, argCol, frameRows));
  }
  return results;
}

function resolveFrameIndex(
  current: number,
  n: number,
  bound: string | null,
  offset: number,
): number {
  if (bound === null || bound === "unbounded_preceding") return 0;
  if (bound === "unbounded_following") return n - 1;
  if (bound === "current_row") return current;
  if (bound === "offset_preceding") return Math.max(0, current - offset);
  if (bound === "offset_following") return Math.min(n - 1, current + offset);
  return current;
}

function resolveRangeFrameIndex(
  current: number,
  n: number,
  rows: Record<string, unknown>[],
  orderCol: string,
  bound: string | null,
  offset: number,
  isStart: boolean,
): number {
  if (bound === null || bound === "unbounded_preceding") return 0;
  if (bound === "unbounded_following") return n - 1;
  if (bound === "current_row") {
    const curVal = rows[current]![orderCol];
    if (isStart) {
      let idx = current;
      while (idx > 0 && rows[idx - 1]![orderCol] === curVal) idx--;
      return idx;
    } else {
      let idx = current;
      while (idx < n - 1 && rows[idx + 1]![orderCol] === curVal) idx++;
      return idx;
    }
  }
  if (bound === "offset_preceding" || bound === "offset_following") {
    const curVal = rows[current]![orderCol];
    if (curVal == null) return isStart ? 0 : n - 1;
    const target =
      bound === "offset_preceding"
        ? (curVal as number) - offset
        : (curVal as number) + offset;
    // Binary search
    if (isStart) {
      let lo = 0;
      let hi = n;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        const v = rows[mid]![orderCol];
        if (v != null && (v as number) < target) lo = mid + 1;
        else hi = mid;
      }
      return lo < n ? lo : n;
    } else {
      let lo = -1;
      let hi = n - 1;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        const v = rows[mid]![orderCol];
        if (v != null && (v as number) <= target) lo = mid;
        else hi = mid - 1;
      }
      return lo >= 0 ? lo : -1;
    }
  }
  return current;
}

// ---------------------------------------------------------------------------
// Aggregate computation
// ---------------------------------------------------------------------------

function computeAggregate(
  funcName: string,
  argCol: string | null,
  rows: Record<string, unknown>[],
  distinct = false,
  extra?: unknown,
): unknown {
  if (typeof extra === "function")
    return (extra as (rows: Record<string, unknown>[]) => unknown)(rows);

  if (funcName === "COUNT") {
    if (argCol === null) return rows.length;
    if (distinct) {
      const seen = new Set<unknown>();
      for (const r of rows) {
        const v = r[argCol];
        if (v != null) seen.add(v);
      }
      return seen.size;
    }
    return rows.filter((r) => r[argCol] != null).length;
  }

  if (funcName === "STRING_AGG") {
    const sep = extra != null ? String(extra as string | number) : ",";
    let vals = rows.filter((r) => r[argCol!] != null).map((r) => String(r[argCol!]));
    if (distinct) {
      const seen = new Set<string>();
      vals = vals.filter((v) => {
        if (seen.has(v)) return false;
        seen.add(v);
        return true;
      });
    }
    return vals.length > 0 ? vals.join(sep) : null;
  }

  if (funcName === "ARRAY_AGG") {
    let vals = rows.filter((r) => r[argCol!] != null).map((r) => r[argCol!]);
    if (distinct) {
      const seen = new Set<string>();
      vals = vals.filter((v) => {
        const k = JSON.stringify(v);
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
      });
    }
    return vals.length > 0 ? vals : null;
  }

  if (funcName === "BOOL_AND") {
    const vals = rows.filter((r) => r[argCol!] != null).map((r) => r[argCol!]);
    if (vals.length === 0) return null;
    return vals.every(Boolean);
  }
  if (funcName === "BOOL_OR") {
    const vals = rows.filter((r) => r[argCol!] != null).map((r) => r[argCol!]);
    if (vals.length === 0) return null;
    return vals.some(Boolean);
  }

  if (funcName === "MODE") {
    const vals = rows.filter((r) => r[argCol!] != null).map((r) => r[argCol!]);
    if (vals.length === 0) return null;
    const counts = new Map<unknown, number>();
    for (const v of vals) counts.set(v, (counts.get(v) ?? 0) + 1);
    let best: unknown = null;
    let bestCount = 0;
    for (const [v, c] of counts) {
      if (c > bestCount) {
        bestCount = c;
        best = v;
      }
    }
    return best;
  }

  // Numeric aggregates
  const values: number[] = [];
  for (const r of rows) {
    const v = r[argCol!];
    if (typeof v === "number") values.push(v);
  }
  if (values.length === 0) return null;

  if (funcName === "SUM") return values.reduce((a, b) => a + b, 0);
  if (funcName === "AVG") return values.reduce((a, b) => a + b, 0) / values.length;
  if (funcName === "MIN") return Math.min(...values);
  if (funcName === "MAX") return Math.max(...values);

  if (funcName === "STDDEV" || funcName === "STDDEV_SAMP") {
    if (values.length < 2) return null;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance =
      values.reduce((a, v) => a + (v - mean) ** 2, 0) / (values.length - 1);
    return Math.sqrt(variance);
  }
  if (funcName === "STDDEV_POP") {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, v) => a + (v - mean) ** 2, 0) / values.length;
    return Math.sqrt(variance);
  }
  if (funcName === "VARIANCE" || funcName === "VAR_SAMP") {
    if (values.length < 2) return null;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return values.reduce((a, v) => a + (v - mean) ** 2, 0) / (values.length - 1);
  }
  if (funcName === "VAR_POP") {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return values.reduce((a, v) => a + (v - mean) ** 2, 0) / values.length;
  }

  if (funcName === "PERCENTILE_CONT") {
    const fraction = Number(extra);
    const sorted = [...values].sort((a, b) => a - b);
    const pos = fraction * (sorted.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, sorted.length - 1);
    const frac = pos - lo;
    return sorted[lo]! + frac * (sorted[hi]! - sorted[lo]!);
  }
  if (funcName === "PERCENTILE_DISC") {
    const fraction = Number(extra);
    const sorted = [...values].sort((a, b) => a - b);
    const idx = Math.floor(fraction * (sorted.length - 1));
    return sorted[idx];
  }

  // Two-argument statistical aggregates
  const twoArgFuncs = new Set([
    "COVAR_POP",
    "COVAR_SAMP",
    "CORR",
    "REGR_COUNT",
    "REGR_AVGX",
    "REGR_AVGY",
    "REGR_SXX",
    "REGR_SYY",
    "REGR_SXY",
    "REGR_SLOPE",
    "REGR_INTERCEPT",
    "REGR_R2",
  ]);
  if (twoArgFuncs.has(funcName)) {
    const xCol = extra as string;
    const pairs: [number, number][] = [];
    for (const r of rows) {
      const yVal = r[argCol!];
      const xVal = r[xCol];
      if (typeof yVal === "number" && typeof xVal === "number") {
        pairs.push([yVal, xVal]);
      }
    }
    const pn = pairs.length;
    if (pn === 0) return null;
    if (funcName === "REGR_COUNT") return pn;

    const ys = pairs.map((p) => p[0]);
    const xs = pairs.map((p) => p[1]);
    const meanY = ys.reduce((a, b) => a + b, 0) / pn;
    const meanX = xs.reduce((a, b) => a + b, 0) / pn;

    if (funcName === "REGR_AVGX") return meanX;
    if (funcName === "REGR_AVGY") return meanY;

    const sxy = pairs.reduce((a, [yi, xi]) => a + (yi - meanY) * (xi - meanX), 0);
    const sxx = xs.reduce((a, xi) => a + (xi - meanX) ** 2, 0);
    const syy = ys.reduce((a, yi) => a + (yi - meanY) ** 2, 0);

    if (funcName === "COVAR_POP") return sxy / pn;
    if (funcName === "COVAR_SAMP") return pn < 2 ? null : sxy / (pn - 1);
    if (funcName === "REGR_SXY") return sxy;
    if (funcName === "REGR_SXX") return sxx;
    if (funcName === "REGR_SYY") return syy;
    if (funcName === "REGR_SLOPE") return sxx === 0 ? null : sxy / sxx;
    if (funcName === "REGR_INTERCEPT")
      return sxx === 0 ? null : meanY - (sxy / sxx) * meanX;
    if (funcName === "CORR") {
      const sdY = Math.sqrt(syy);
      const sdX = Math.sqrt(sxx);
      return sdY === 0 || sdX === 0 ? null : sxy / (sdY * sdX);
    }
    if (funcName === "REGR_R2") {
      return sxx === 0 || syy === 0 ? null : sxy ** 2 / (sxx * syy);
    }
  }

  throw new Error(`Unknown aggregate: ${funcName}`);
}
