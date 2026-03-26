//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Centrality operators
// 1:1 port of uqa/graph/centrality.py

import type { PostingEntry } from "../core/types.js";
import { createPostingEntry } from "../core/types.js";
import type { PostingList } from "../core/posting-list.js";
import { Operator } from "../operators/base.js";
import type { ExecutionContext } from "../operators/base.js";
import type { GraphStore } from "../storage/abc/graph-store.js";
import { GraphPostingList, createGraphPayload } from "./posting-list.js";

// -- PageRankOperator ---------------------------------------------------------

export class PageRankOperator extends Operator {
  readonly damping: number;
  readonly maxIterations: number;
  readonly tolerance: number;
  readonly graph: string;

  constructor(opts: {
    damping?: number;
    maxIterations?: number;
    tolerance?: number;
    graph: string;
  }) {
    super();
    this.damping = opts.damping ?? 0.85;
    this.maxIterations = opts.maxIterations ?? 100;
    this.tolerance = opts.tolerance ?? 1e-6;
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

    // Initialize scores
    let scores = new Float64Array(n);
    scores.fill(1.0 / n);

    // Compute out-degree for each vertex
    const outDegree = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const vid = vertexIds[i]!;
      const outEdges = store.outEdgeIds(vid, this.graph);
      outDegree[i] = outEdges.size;
    }

    // Iterative computation
    for (let iter = 0; iter < this.maxIterations; iter++) {
      const newScores = new Float64Array(n);
      newScores.fill((1.0 - this.damping) / n);

      for (let i = 0; i < n; i++) {
        const vid = vertexIds[i]!;
        if (outDegree[i]! === 0) {
          // Dangling node: distribute evenly
          const share = scores[i]! / n;
          for (let j = 0; j < n; j++) {
            newScores[j] = newScores[j]! + this.damping * share;
          }
        } else {
          const share = scores[i]! / outDegree[i]!;
          const neighbors = store.neighbors(vid, this.graph, null, "out");
          for (const nid of neighbors) {
            const nIdx = vidToIdx.get(nid);
            if (nIdx !== undefined) {
              newScores[nIdx] = newScores[nIdx]! + this.damping * share;
            }
          }
        }
      }

      // Check convergence
      let diff = 0;
      for (let i = 0; i < n; i++) {
        diff += Math.abs(newScores[i]! - scores[i]!);
      }

      scores = newScores;
      if (diff < this.tolerance) break;
    }

    // Build posting list
    const entries: PostingEntry[] = [];
    for (let i = 0; i < n; i++) {
      const vid = vertexIds[i]!;
      entries.push(
        createPostingEntry(vid, {
          score: scores[i]!,
          fields: { pagerank: scores[i]! },
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

// -- HITSOperator -------------------------------------------------------------

export class HITSOperator extends Operator {
  readonly maxIterations: number;
  readonly tolerance: number;
  readonly graph: string;

  constructor(opts: { maxIterations?: number; tolerance?: number; graph: string }) {
    super();
    this.maxIterations = opts.maxIterations ?? 100;
    this.tolerance = opts.tolerance ?? 1e-6;
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

    let hub = new Float64Array(n);
    let auth = new Float64Array(n);
    hub.fill(1.0);
    auth.fill(1.0);

    for (let iter = 0; iter < this.maxIterations; iter++) {
      const newAuth = new Float64Array(n);
      const newHub = new Float64Array(n);

      // Authority update: auth(v) = sum of hub(u) for all u -> v
      for (let i = 0; i < n; i++) {
        const vid = vertexIds[i]!;
        const inNeighbors = store.neighbors(vid, this.graph, null, "in");
        let s = 0;
        for (const nid of inNeighbors) {
          const nIdx = vidToIdx.get(nid);
          if (nIdx !== undefined) {
            s += hub[nIdx]!;
          }
        }
        newAuth[i] = s;
      }

      // Hub update: hub(v) = sum of auth(u) for all v -> u
      for (let i = 0; i < n; i++) {
        const vid = vertexIds[i]!;
        const outNeighbors = store.neighbors(vid, this.graph, null, "out");
        let s = 0;
        for (const nid of outNeighbors) {
          const nIdx = vidToIdx.get(nid);
          if (nIdx !== undefined) {
            s += newAuth[nIdx]!;
          }
        }
        newHub[i] = s;
      }

      // Normalize
      let authNorm = 0;
      let hubNorm = 0;
      for (let i = 0; i < n; i++) {
        authNorm += newAuth[i]! * newAuth[i]!;
        hubNorm += newHub[i]! * newHub[i]!;
      }
      authNorm = Math.sqrt(authNorm);
      hubNorm = Math.sqrt(hubNorm);

      if (authNorm > 0) {
        for (let i = 0; i < n; i++) {
          newAuth[i] = newAuth[i]! / authNorm;
        }
      }
      if (hubNorm > 0) {
        for (let i = 0; i < n; i++) {
          newHub[i] = newHub[i]! / hubNorm;
        }
      }

      // Check convergence
      let diff = 0;
      for (let i = 0; i < n; i++) {
        diff += Math.abs(newAuth[i]! - auth[i]!);
        diff += Math.abs(newHub[i]! - hub[i]!);
      }

      auth = newAuth;
      hub = newHub;
      if (diff < this.tolerance) break;
    }

    const entries: PostingEntry[] = [];
    for (let i = 0; i < n; i++) {
      const vid = vertexIds[i]!;
      // Combined score: authority + hub
      const score = auth[i]! + hub[i]!;
      entries.push(
        createPostingEntry(vid, {
          score,
          fields: { authority: auth[i]!, hub: hub[i]! },
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

// -- BetweennessCentralityOperator --------------------------------------------

export class BetweennessCentralityOperator extends Operator {
  readonly graph: string;

  constructor(opts: { graph: string }) {
    super();
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

    // Brandes algorithm
    const centrality = new Float64Array(n);

    for (let s = 0; s < n; s++) {
      const stack: number[] = [];
      const predecessors: Set<number>[] = new Array<Set<number>>(n);
      for (let i = 0; i < n; i++) {
        predecessors[i] = new Set();
      }

      const sigma = new Float64Array(n);
      sigma[s] = 1.0;

      const dist = new Float64Array(n);
      dist.fill(-1);
      dist[s] = 0;

      // BFS
      const queue: number[] = [s];
      let qIdx = 0;

      while (qIdx < queue.length) {
        const v = queue[qIdx]!;
        qIdx++;
        stack.push(v);

        const vid = vertexIds[v]!;
        const neighbors = store.neighbors(vid, this.graph, null, "out");
        const inNeighbors = store.neighbors(vid, this.graph, null, "in");
        const allNeighborIds = new Set([...neighbors, ...inNeighbors]);

        for (const nid of allNeighborIds) {
          const w = vidToIdx.get(nid);
          if (w === undefined) continue;

          // w found for first time?
          if (dist[w]! < 0) {
            queue.push(w);
            dist[w] = dist[v]! + 1;
          }
          // Shortest path to w via v?
          if (dist[w]! === dist[v]! + 1) {
            sigma[w] = sigma[w]! + sigma[v]!;
            predecessors[w]!.add(v);
          }
        }
      }

      // Accumulation
      const delta = new Float64Array(n);
      while (stack.length > 0) {
        const w = stack.pop()!;
        for (const v of predecessors[w]!) {
          delta[v] = delta[v]! + (sigma[v]! / sigma[w]!) * (1.0 + delta[w]!);
        }
        if (w !== s) {
          centrality[w] = centrality[w]! + delta[w]!;
        }
      }
    }

    // Normalize for undirected (divide by 2)
    for (let i = 0; i < n; i++) {
      centrality[i] = centrality[i]! / 2.0;
    }

    const entries: PostingEntry[] = [];
    for (let i = 0; i < n; i++) {
      const vid = vertexIds[i]!;
      entries.push(
        createPostingEntry(vid, {
          score: centrality[i]!,
          fields: { betweenness: centrality[i]! },
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
