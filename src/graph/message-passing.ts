//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Message passing operator
// 1:1 port of uqa/graph/message_passing.py

import type { PostingEntry } from "../core/types.js";
import { createPostingEntry } from "../core/types.js";
import type { PostingList } from "../core/posting-list.js";
import { Operator } from "../operators/base.js";
import type { ExecutionContext } from "../operators/base.js";
import type { GraphStore } from "../storage/abc/graph-store.js";
import { GraphPostingList, createGraphPayload } from "./posting-list.js";

// -- MessagePassingOperator ---------------------------------------------------

export class MessagePassingOperator extends Operator {
  readonly kLayers: number;
  readonly aggregation: "mean" | "sum" | "max";
  readonly propertyName: string;
  readonly graph: string;

  constructor(opts: {
    kLayers?: number;
    aggregation?: "mean" | "sum" | "max";
    propertyName?: string;
    graph: string;
  }) {
    super();
    this.kLayers = opts.kLayers ?? 2;
    this.aggregation = opts.aggregation ?? "mean";
    this.propertyName = opts.propertyName ?? "feature";
    this.graph = opts.graph;
  }

  execute(context: ExecutionContext): PostingList {
    const store = context.graphStore as GraphStore;
    const vertexIds = [...store.vertexIdsInGraph(this.graph)];
    const n = vertexIds.length;
    if (n === 0) return new GraphPostingList();

    const vidToIdx = new Map<number, number>();
    for (let i = 0; i < vertexIds.length; i++) {
      vidToIdx.set(vertexIds[i]!, i);
    }

    // Initialize features from vertex properties
    let features = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const vid = vertexIds[i]!;
      const vertex = store.getVertex(vid);
      if (vertex) {
        const prop = vertex.properties[this.propertyName];
        if (typeof prop === "number") {
          features[i] = prop;
        } else {
          features[i] = 0.0;
        }
      }
    }

    // Message passing iterations
    for (let layer = 0; layer < this.kLayers; layer++) {
      const newFeatures = new Float64Array(n);

      for (let i = 0; i < n; i++) {
        const vid = vertexIds[i]!;
        // Gather neighbor features
        const outNeighbors = store.neighbors(vid, this.graph, null, "out");
        const inNeighbors = store.neighbors(vid, this.graph, null, "in");
        const allNeighborIds = new Set([...outNeighbors, ...inNeighbors]);

        const neighborValues: number[] = [];
        for (const nid of allNeighborIds) {
          const nIdx = vidToIdx.get(nid);
          if (nIdx !== undefined) {
            neighborValues.push(features[nIdx]!);
          }
        }

        // Aggregate
        let aggregated: number;
        if (neighborValues.length === 0) {
          aggregated = features[i]!;
        } else if (this.aggregation === "mean") {
          let s = 0;
          for (const v of neighborValues) s += v;
          aggregated = s / neighborValues.length;
        } else if (this.aggregation === "sum") {
          let s = 0;
          for (const v of neighborValues) s += v;
          aggregated = s;
        } else {
          // max
          let mx = neighborValues[0]!;
          for (let j = 1; j < neighborValues.length; j++) {
            if (neighborValues[j]! > mx) mx = neighborValues[j]!;
          }
          aggregated = mx;
        }

        // Combine with self (GNN update rule: new_h = agg(neighbors) + self)
        newFeatures[i] = aggregated + features[i]!;
      }

      features = newFeatures;
    }

    // Sigmoid calibration
    for (let i = 0; i < n; i++) {
      features[i] = 1.0 / (1.0 + Math.exp(-features[i]!));
    }

    // Build posting list
    const entries: PostingEntry[] = [];
    for (let i = 0; i < n; i++) {
      const vid = vertexIds[i]!;
      entries.push(
        createPostingEntry(vid, {
          score: features[i]!,
          fields: { _mp_score: features[i]! },
        }),
      );
    }

    const allVertices = new Set(vertexIds);
    const gpl = new GraphPostingList(entries);
    for (const entry of entries) {
      gpl.setGraphPayload(
        entry.docId,
        createGraphPayload({
          subgraphVertices: allVertices,
          subgraphEdges: new Set(),
          score: entry.payload.score,
          graphName: this.graph,
        }),
      );
    }
    return gpl;
  }
}
