//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Temporal traverse operator
// 1:1 port of uqa/graph/temporal_traverse.py

import type { PostingEntry, Vertex } from "../core/types.js";
import { createPostingEntry } from "../core/types.js";
import type { PostingList } from "../core/posting-list.js";
import { Operator } from "../operators/base.js";
import type { ExecutionContext } from "../operators/base.js";
import type { GraphStore } from "../storage/abc/graph-store.js";
import { GraphPostingList, createGraphPayload } from "./posting-list.js";
import type { TemporalFilter } from "./temporal-filter.js";

// -- TemporalTraverseOperator -------------------------------------------------

export class TemporalTraverseOperator extends Operator {
  readonly startVertex: number;
  readonly graph: string;
  readonly temporalFilter: TemporalFilter;
  readonly label: string | null;
  readonly maxHops: number;
  readonly vertexPredicate: ((v: Vertex) => boolean) | null;
  readonly score: number;

  constructor(opts: {
    startVertex: number;
    graph: string;
    temporalFilter: TemporalFilter;
    label?: string | null;
    maxHops?: number;
    vertexPredicate?: ((v: Vertex) => boolean) | null;
    score?: number;
  }) {
    super();
    this.startVertex = opts.startVertex;
    this.graph = opts.graph;
    this.temporalFilter = opts.temporalFilter;
    this.label = opts.label ?? null;
    this.maxHops = opts.maxHops ?? Infinity;
    this.vertexPredicate = opts.vertexPredicate ?? null;
    this.score = opts.score ?? 1.0;
  }

  execute(context: ExecutionContext): PostingList {
    const store = context.graphStore as GraphStore;

    const visited = new Set<number>();
    const entries: PostingEntry[] = [];
    const allVertices = new Set<number>();
    const allEdges = new Set<number>();

    // BFS with temporal filtering on edges
    const queue: Array<[number, number]> = [[this.startVertex, 0]];
    visited.add(this.startVertex);

    while (queue.length > 0) {
      const [current, depth] = queue.shift()!;
      const vertex = store.getVertex(current);
      if (!vertex) continue;

      if (this.vertexPredicate && !this.vertexPredicate(vertex)) {
        continue;
      }

      allVertices.add(current);

      const depthScore = this.score / (1 + depth);
      entries.push(
        createPostingEntry(current, {
          score: depthScore,
          fields: { ...vertex.properties, _depth: depth },
        }),
      );

      if (depth < this.maxHops) {
        const outEdges = store.outEdgeIds(current, this.graph);
        for (const eid of outEdges) {
          const edge = store.getEdge(eid);
          if (!edge) continue;
          if (this.label !== null && edge.label !== this.label) continue;

          // Apply temporal filter
          if (!this.temporalFilter.isValid(edge)) continue;

          allEdges.add(eid);

          if (!visited.has(edge.targetId)) {
            visited.add(edge.targetId);
            queue.push([edge.targetId, depth + 1]);
          }
        }
      }
    }

    const gpl = new GraphPostingList(entries);
    for (const entry of entries) {
      gpl.setGraphPayload(
        entry.docId,
        createGraphPayload({
          subgraphVertices: allVertices,
          subgraphEdges: allEdges,
          score: entry.payload.score,
          graphName: this.graph,
        }),
      );
    }
    return gpl;
  }
}
