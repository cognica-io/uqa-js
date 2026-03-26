//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- spill-to-memory manager (browser-safe)
// 1:1 port of uqa/execution/spill.py
//
// In-browser environments cannot spill to disk, so this implementation
// uses in-memory arrays. The API mirrors a file-backed spill manager
// so the same callers work when a real file backend is added.

// ---------------------------------------------------------------------------
// SpillManager
// ---------------------------------------------------------------------------

export class SpillManager {
  private _runs: Record<string, unknown>[][] = [];

  /** Allocate a new run buffer. Returns the run index. */
  newRun(): number {
    const idx = this._runs.length;
    this._runs.push([]);
    return idx;
  }

  /** Return a new SpillWriter backed by a new run. */
  newWriter(): SpillWriter {
    const idx = this.newRun();
    return new SpillWriter(this, idx);
  }

  /** Append rows to an existing run. */
  writeRows(runIdx: number, rows: Record<string, unknown>[]): void {
    const run = this._runs[runIdx];
    if (run === undefined) {
      throw new Error(`SpillManager: run ${String(runIdx)} does not exist`);
    }
    for (const row of rows) {
      run.push(row);
    }
  }

  /** Read all rows from a run. */
  readRows(runIdx: number): Record<string, unknown>[] {
    const run = this._runs[runIdx];
    if (run === undefined) {
      throw new Error(`SpillManager: run ${String(runIdx)} does not exist`);
    }
    return run;
  }

  /** Return the number of runs. */
  get runCount(): number {
    return this._runs.length;
  }

  /** Release all spill memory. */
  cleanup(): void {
    this._runs = [];
  }
}

// ---------------------------------------------------------------------------
// SpillWriter -- mirrors the Python SpillWriter class
// ---------------------------------------------------------------------------

export class SpillWriter {
  private _manager: SpillManager;
  private _runIdx: number;
  private _rowCount: number;

  constructor(manager: SpillManager, runIdx: number) {
    this._manager = manager;
    this._runIdx = runIdx;
    this._rowCount = 0;
  }

  get runIdx(): number {
    return this._runIdx;
  }

  get rowCount(): number {
    return this._rowCount;
  }

  writeRows(rows: Record<string, unknown>[]): void {
    if (rows.length === 0) return;
    this._manager.writeRows(this._runIdx, rows);
    this._rowCount += rows.length;
  }

  close(): void {
    // No-op: in-memory runs do not need flushing.
  }
}

// ---------------------------------------------------------------------------
// mergeSortedRuns -- k-way merge of pre-sorted run arrays
// ---------------------------------------------------------------------------

/**
 * Merge multiple pre-sorted runs into a single sorted array.
 *
 * Uses a simple k-way merge with a linear scan to find the minimum at each
 * step. For the expected number of runs (typically 2-8), this is efficient
 * enough; a heap-based approach can be added for very large k values.
 *
 * @param runs     Array of sorted run arrays.
 * @param sortKeys Array of [column, ascending] pairs.
 * @returns        Merged sorted array.
 */
export type SortKeySpec = [string, boolean] | [string, boolean, boolean];

export function mergeSortedRuns(
  runs: Record<string, unknown>[][],
  sortKeys: SortKeySpec[],
): Record<string, unknown>[] {
  if (runs.length === 0) return [];
  if (runs.length === 1) return runs[0]!;

  // Cursors for each run
  const cursors: number[] = new Array(runs.length).fill(0) as number[];
  const totalRows = runs.reduce((acc, r) => acc + r.length, 0);
  const result: Record<string, unknown>[] = [];
  result.length = totalRows;

  for (let out = 0; out < totalRows; out++) {
    // Find the run with the smallest current element
    let bestRun = -1;
    let bestRow: Record<string, unknown> | null = null;

    for (let r = 0; r < runs.length; r++) {
      const run = runs[r]!;
      const cursor = cursors[r]!;
      if (cursor >= run.length) continue;

      const row = run[cursor]!;
      if (bestRow === null || compareRows(row, bestRow, sortKeys) < 0) {
        bestRun = r;
        bestRow = row;
      }
    }

    // bestRun is guaranteed to be found since out < totalRows
    result[out] = bestRow!;
    cursors[bestRun]!++;
  }

  return result;
}

/**
 * Compare two rows by the given sort keys.
 */
function compareRows(
  a: Record<string, unknown>,
  b: Record<string, unknown>,
  sortKeys: SortKeySpec[],
): number {
  for (const sk of sortKeys) {
    const column = sk[0];
    const ascending = sk[1];
    // nullsFirst defaults to PostgreSQL convention: NULLS FIRST for DESC, NULLS LAST for ASC
    const nullsFirst = sk.length > 2 ? sk[2] : !ascending;
    const av = a[column];
    const bv = b[column];
    const aNull = av === null || av === undefined;
    const bNull = bv === null || bv === undefined;

    if (aNull && bNull) continue;
    if (aNull) return nullsFirst ? -1 : 1;
    if (bNull) return nullsFirst ? 1 : -1;

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

    if (!ascending) cmp = -cmp;
    if (cmp !== 0) return cmp;
  }
  return 0;
}
