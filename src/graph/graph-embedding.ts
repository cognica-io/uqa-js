//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Graph embedding operator
// 1:1 port of uqa/graph/graph_embedding.py

import type { PostingEntry } from "../core/types.js";
import { createPostingEntry } from "../core/types.js";
import type { PostingList } from "../core/posting-list.js";
import { Operator } from "../operators/base.js";
import type { ExecutionContext } from "../operators/base.js";
import type { GraphStore } from "../storage/abc/graph-store.js";
import { GraphPostingList, createGraphPayload } from "./posting-list.js";

// -- GraphEmbeddingOperator ---------------------------------------------------
// Produces structure-based graph embeddings using:
//   - Degree features
//   - Label distribution
//   - k-hop connectivity

export class GraphEmbeddingOperator extends Operator {
  readonly graph: string;
  readonly kHops: number;

  constructor(opts: { graph: string; kHops?: number }) {
    super();
    this.graph = opts.graph;
    this.kHops = opts.kHops ?? 2;
  }

  execute(context: ExecutionContext): PostingList {
    const store = context.graphStore as GraphStore;
    const vertexIds = [...store.vertexIdsInGraph(this.graph)];
    const n = vertexIds.length;
    if (n === 0) return new GraphPostingList();

    // Collect all labels for label distribution vector
    const labelCounts = store.vertexLabelCounts(this.graph);
    const allLabels = [...labelCounts.keys()].sort();
    const labelToIdx = new Map<string, number>();
    for (let i = 0; i < allLabels.length; i++) {
      labelToIdx.set(allLabels[i]!, i);
    }

    const entries: PostingEntry[] = [];
    const allVertexSet = new Set(vertexIds);

    for (const vid of vertexIds) {
      const vertex = store.getVertex(vid);
      if (!vertex) continue;

      // Feature 1: normalized degree
      const outEdges = store.outEdgeIds(vid, this.graph);
      const inEdges = store.inEdgeIds(vid, this.graph);
      const outDegree = outEdges.size;
      const inDegree = inEdges.size;
      const totalDegree = outDegree + inDegree;
      const normDegree = n > 1 ? totalDegree / (2 * (n - 1)) : 0;

      // Feature 2: label distribution of neighbors
      const labelDist = new Float64Array(allLabels.length);
      const neighbors = new Set<number>();
      for (const nid of store.neighbors(vid, this.graph, null, "out")) {
        neighbors.add(nid);
      }
      for (const nid of store.neighbors(vid, this.graph, null, "in")) {
        neighbors.add(nid);
      }

      if (neighbors.size > 0) {
        for (const nid of neighbors) {
          const nVertex = store.getVertex(nid);
          if (nVertex) {
            const idx = labelToIdx.get(nVertex.label);
            if (idx !== undefined) {
              labelDist[idx] = labelDist[idx]! + 1;
            }
          }
        }
        // Normalize
        for (let i = 0; i < labelDist.length; i++) {
          labelDist[i] = labelDist[i]! / neighbors.size;
        }
      }

      // Feature 3: k-hop connectivity (number of reachable vertices at each hop)
      const kHopCounts = new Float64Array(this.kHops);
      const visited = new Set<number>([vid]);
      let frontier = new Set<number>([vid]);

      for (let hop = 0; hop < this.kHops; hop++) {
        const nextFrontier = new Set<number>();
        for (const fvid of frontier) {
          for (const nid of store.neighbors(fvid, this.graph, null, "out")) {
            if (!visited.has(nid)) {
              visited.add(nid);
              nextFrontier.add(nid);
            }
          }
          for (const nid of store.neighbors(fvid, this.graph, null, "in")) {
            if (!visited.has(nid)) {
              visited.add(nid);
              nextFrontier.add(nid);
            }
          }
        }
        kHopCounts[hop] = n > 1 ? nextFrontier.size / (n - 1) : 0;
        frontier = nextFrontier;
      }

      // Combine features into embedding vector
      const embDim = 1 + allLabels.length + this.kHops;
      const embedding = new Float64Array(embDim);
      embedding[0] = normDegree;
      for (let i = 0; i < labelDist.length; i++) {
        embedding[1 + i] = labelDist[i]!;
      }
      for (let i = 0; i < kHopCounts.length; i++) {
        embedding[1 + allLabels.length + i] = kHopCounts[i]!;
      }

      // Score: use norm of embedding as relevance indicator
      let normSq = 0;
      for (let i = 0; i < embDim; i++) {
        normSq += embedding[i]! * embedding[i]!;
      }
      const score = Math.sqrt(normSq);

      entries.push(
        createPostingEntry(vid, {
          score,
          fields: {
            embedding,
            out_degree: outDegree,
            in_degree: inDegree,
            label: vertex.label,
          },
        }),
      );
    }

    const gpl = new GraphPostingList(entries);
    for (const entry of entries) {
      gpl.setGraphPayload(
        entry.docId,
        createGraphPayload({
          subgraphVertices: allVertexSet,
          subgraphEdges: new Set(),
          score: entry.payload.score,
          graphName: this.graph,
        }),
      );
    }
    return gpl;
  }
}
