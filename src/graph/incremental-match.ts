//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Incremental pattern matcher
// 1:1 port of uqa/graph/incremental_match.py

import type { PostingEntry } from "../core/types.js";
import { createPostingEntry } from "../core/types.js";
import type { ExecutionContext } from "../operators/base.js";
import type { GraphStore } from "../storage/abc/graph-store.js";
import { GraphPostingList, createGraphPayload } from "./posting-list.js";
import type { GraphPattern } from "./pattern.js";
import { PatternMatchOperator } from "./operators.js";
import type { GraphDelta } from "./delta.js";

// -- IncrementalPatternMatcher ------------------------------------------------

export class IncrementalPatternMatcher {
  readonly pattern: GraphPattern;
  readonly graph: string;
  private _cachedResults: Array<Map<string, number>> | null = null;

  constructor(pattern: GraphPattern, graph: string) {
    this.pattern = pattern;
    this.graph = graph;
  }

  get cachedResults(): ReadonlyArray<ReadonlyMap<string, number>> | null {
    return this._cachedResults;
  }

  // Full pattern match and cache
  fullMatch(context: ExecutionContext): GraphPostingList {
    const op = new PatternMatchOperator(this.pattern, this.graph);
    const result = op.execute(context);

    // Cache assignments from result
    this._cachedResults = [];
    for (const entry of result) {
      const fields = entry.payload.fields as Record<string, unknown>;
      const assignment = new Map<string, number>();
      for (const vp of this.pattern.vertexPatterns) {
        const vid = fields[vp.variable];
        if (typeof vid === "number") {
          assignment.set(vp.variable, vid);
        }
      }
      this._cachedResults.push(assignment);
    }

    return result instanceof GraphPostingList
      ? result
      : new GraphPostingList([...result]);
  }

  // Invalidate affected matches and re-match in the delta region
  incrementalUpdate(delta: GraphDelta, context: ExecutionContext): GraphPostingList {
    if (this._cachedResults === null) {
      return this.fullMatch(context);
    }

    const affectedVertices = delta.affectedVertices;
    const affectedLabels = delta.affectedLabels;

    // Determine which cached matches are affected
    const validMatches: Array<Map<string, number>> = [];
    const invalidatedCount = { value: 0 };

    for (const assignment of this._cachedResults) {
      let affected = false;
      for (const vid of assignment.values()) {
        if (affectedVertices.has(vid)) {
          affected = true;
          break;
        }
      }
      if (!affected) {
        // Check edge labels
        for (const ep of this.pattern.edgePatterns) {
          if (ep.label !== null && affectedLabels.has(ep.label)) {
            affected = true;
            break;
          }
        }
      }

      if (affected) {
        invalidatedCount.value++;
      } else {
        validMatches.push(assignment);
      }
    }

    // Re-run full match if too many invalidated
    if (invalidatedCount.value > validMatches.length) {
      return this.fullMatch(context);
    }

    // Otherwise, try to find new matches only in affected region
    const store = context.graphStore as GraphStore;
    const op = new PatternMatchOperator(this.pattern, this.graph);
    const newResult = op.execute(context);

    // Extract new assignments
    const newAssignments: Array<Map<string, number>> = [];
    for (const entry of newResult) {
      const fields = entry.payload.fields as Record<string, unknown>;
      const assignment = new Map<string, number>();
      for (const vp of this.pattern.vertexPatterns) {
        const vid = fields[vp.variable];
        if (typeof vid === "number") {
          assignment.set(vp.variable, vid);
        }
      }

      // Only keep if it involves affected vertices (new matches)
      let involvesAffected = false;
      for (const vid of assignment.values()) {
        if (affectedVertices.has(vid)) {
          involvesAffected = true;
          break;
        }
      }
      if (involvesAffected) {
        newAssignments.push(assignment);
      }
    }

    // Merge: valid cached + new
    this._cachedResults = [...validMatches, ...newAssignments];

    return this._buildPostingList(store, this._cachedResults);
  }

  // Invalidate all cached results
  invalidate(): void {
    this._cachedResults = null;
  }

  private _buildPostingList(
    _store: GraphStore,
    results: Array<Map<string, number>>,
  ): GraphPostingList {
    const entries: PostingEntry[] = [];

    for (let i = 0; i < results.length; i++) {
      const assignment = results[i]!;

      const fields: Record<string, unknown> = {};
      for (const [variable, vid] of assignment) {
        fields[variable] = vid;
      }

      entries.push(
        createPostingEntry(i, {
          score: 1.0,
          fields,
        }),
      );
    }

    const gpl = new GraphPostingList(entries);
    for (let i = 0; i < results.length; i++) {
      const assignment = results[i]!;
      const vertexIds = new Set(assignment.values());
      gpl.setGraphPayload(
        i,
        createGraphPayload({
          subgraphVertices: vertexIds,
          subgraphEdges: new Set(),
          score: 1.0,
          graphName: this.graph,
        }),
      );
    }
    return gpl;
  }
}
