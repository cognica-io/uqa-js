//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- parallel executor (browser-safe sequential fallback)
// 1:1 port of uqa/planner/parallel.py
//
// In-browser environments do not have access to Web Workers from library code
// without bundler support. This implementation falls back to sequential
// execution while exposing the same API that a worker-based version would use.

import type { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "../operators/base.js";
import type { Operator } from "../operators/base.js";

// ---------------------------------------------------------------------------
// ParallelExecutor
// ---------------------------------------------------------------------------

/**
 * Executes multiple independent operator branches.
 *
 * The current implementation runs branches sequentially on the main thread.
 * When Web Worker support is added, branches will be dispatched to a pool
 * of workers for true parallelism.
 */
export class ParallelExecutor {
  private readonly _maxWorkers: number;
  private _shutdown: boolean = false;

  /**
   * @param maxWorkers  Maximum number of concurrent workers (reserved for
   *                    future Web Worker support; currently unused).
   */
  constructor(maxWorkers = 4) {
    this._maxWorkers = maxWorkers;
  }

  /**
   * Whether parallel execution is actually enabled.
   * Returns false in the current sequential implementation.
   */
  get enabled(): boolean {
    // Sequential fallback -- no true parallelism available.
    // Will return true when Web Worker support is enabled and _maxWorkers > 1.
    return this._maxWorkers > 0 && false;
  }

  /**
   * Execute each operator independently and return their results.
   *
   * In the sequential implementation, operators are executed one after
   * another on the main thread.
   *
   * @param operators Array of independent operators to execute.
   * @param context   The shared execution context.
   * @returns         Array of PostingList results, one per operator.
   */
  executeBranches(operators: Operator[], context: ExecutionContext): PostingList[] {
    if (this._shutdown) {
      throw new Error("ParallelExecutor has been shut down");
    }

    const results: PostingList[] = [];
    for (const op of operators) {
      results.push(op.execute(context));
    }
    return results;
  }

  /**
   * Release all resources.
   *
   * In the sequential implementation this is a no-op but it marks the
   * executor as shut down to prevent further use.
   */
  shutdown(): void {
    this._shutdown = true;
  }
}
