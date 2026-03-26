//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Temporal filter
// 1:1 port of uqa/graph/temporal_filter.py

import type { Edge } from "../core/types.js";

// -- TemporalFilter -----------------------------------------------------------

export class TemporalFilter {
  readonly timestamp: number | null;
  readonly timeRange: [number, number] | null;

  constructor(opts?: {
    timestamp?: number | null;
    timeRange?: [number, number] | null;
  }) {
    const ts = opts?.timestamp ?? null;
    const tr = opts?.timeRange ?? null;
    if (ts !== null && tr !== null) {
      throw new Error("TemporalFilter: timestamp and timeRange are mutually exclusive");
    }
    this.timestamp = ts;
    this.timeRange = tr;
  }

  isValid(edge: Edge): boolean {
    // Support valid_from / valid_to range model on edges
    const validFrom = edge.properties["valid_from"];
    const validTo = edge.properties["valid_to"];
    const hasRange = typeof validFrom === "number" || typeof validTo === "number";

    if (hasRange) {
      return this._isValidRange(
        typeof validFrom === "number" ? validFrom : null,
        typeof validTo === "number" ? validTo : null,
      );
    }

    // Fall back to simple timestamp property
    const edgeTimestamp = edge.properties["timestamp"];
    if (typeof edgeTimestamp !== "number") {
      // No temporal properties on edge: pass through
      return true;
    }

    if (this.timestamp !== null) {
      return edgeTimestamp <= this.timestamp;
    }

    if (this.timeRange !== null) {
      const [start, end] = this.timeRange;
      return edgeTimestamp >= start && edgeTimestamp <= end;
    }

    // No filter configured
    return true;
  }

  private _isValidRange(validFrom: number | null, validTo: number | null): boolean {
    if (this.timestamp !== null) {
      // Point-in-time query: timestamp must fall within [valid_from, valid_to]
      if (validFrom !== null && this.timestamp < validFrom) return false;
      if (validTo !== null && this.timestamp > validTo) return false;
      return true;
    }

    if (this.timeRange !== null) {
      // Range overlap: edge range [valid_from, valid_to] must overlap [start, end]
      const [start, end] = this.timeRange;
      if (validFrom !== null && validFrom > end) return false;
      if (validTo !== null && validTo < start) return false;
      return true;
    }

    // No filter: accept all
    return true;
  }
}
