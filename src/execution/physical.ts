//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Volcano-model physical operator base
// 1:1 port of uqa/execution/physical.py

import type { Batch } from "./batch.js";

/**
 * Abstract base class for Volcano-model physical operators.
 *
 * Each operator implements `open()`, `next()`, and `close()`.
 * - `open()` initializes state (e.g. opens child iterators).
 * - `next()` returns the next Batch, or null when exhausted.
 * - `close()` releases resources.
 */
export abstract class PhysicalOperator {
  /** Initialize the operator and its children. */
  abstract open(): void;

  /** Return the next batch of rows, or null if exhausted. */
  abstract next(): Batch | null;

  /** Release all resources. */
  abstract close(): void;
}
