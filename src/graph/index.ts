//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Graph indexes
// 1:1 port of uqa/graph/index.py

import type { GraphStore } from "../storage/abc/graph-store.js";
import type { GraphPattern } from "./pattern.js";
import { PatternMatchOperator } from "./operators.js";
import type { ExecutionContext } from "../operators/base.js";

// -- LabelIndex ---------------------------------------------------------------

export class LabelIndex {
  private _labelToEdges: Map<string, number[]> = new Map();
  private _labelToVertices: Map<string, Set<number>> = new Map();

  build(store: GraphStore, graphName: string): void {
    this._labelToEdges.clear();
    this._labelToVertices.clear();

    for (const vid of store.vertexIdsInGraph(graphName)) {
      for (const eid of store.outEdgeIds(vid, graphName)) {
        const edge = store.getEdge(eid);
        if (!edge) continue;
        let edgeList = this._labelToEdges.get(edge.label);
        if (!edgeList) {
          edgeList = [];
          this._labelToEdges.set(edge.label, edgeList);
        }
        edgeList.push(eid);

        let vertexSet = this._labelToVertices.get(edge.label);
        if (!vertexSet) {
          vertexSet = new Set();
          this._labelToVertices.set(edge.label, vertexSet);
        }
        vertexSet.add(edge.sourceId);
        vertexSet.add(edge.targetId);
      }
    }

    // Sort edge lists
    for (const edges of this._labelToEdges.values()) {
      edges.sort((a, b) => a - b);
    }
  }

  edgesByLabel(label: string): number[] {
    return this._labelToEdges.get(label) ?? [];
  }

  verticesByLabel(label: string): Set<number> {
    return this._labelToVertices.get(label) ?? new Set();
  }

  labels(): string[] {
    return [...this._labelToEdges.keys()].sort();
  }

  labelCount(label: string): number {
    return (this._labelToEdges.get(label) ?? []).length;
  }
}

// -- NeighborhoodIndex --------------------------------------------------------

export class NeighborhoodIndex {
  readonly maxHops: number;
  private _cache: Map<number, Map<number, Set<number>>> = new Map();

  constructor(maxHops = 2) {
    this.maxHops = maxHops;
  }

  build(store: GraphStore, graphName: string, label?: string | null): void {
    this._cache.clear();
    const maxH = this.maxHops;

    for (const vid of [...store.vertexIdsInGraph(graphName)].sort((a, b) => a - b)) {
      const vidCache = new Map<number, Set<number>>();
      this._cache.set(vid, vidCache);

      const visited = new Set<number>([vid]);
      let frontier = new Set<number>([vid]);

      for (let hop = 1; hop <= maxH; hop++) {
        const nextFrontier = new Set<number>();
        for (const v of frontier) {
          for (const eid of store.outEdgeIds(v, graphName)) {
            const edge = store.getEdge(eid);
            if (!edge) continue;
            if (label != null && edge.label !== label) continue;
            if (!visited.has(edge.targetId)) {
              nextFrontier.add(edge.targetId);
            }
          }
        }
        for (const nv of nextFrontier) visited.add(nv);
        frontier = nextFrontier;
        vidCache.set(hop, new Set(visited));
      }
    }
  }

  neighbors(vertexId: number, hops: number): Set<number> {
    const vidCache = this._cache.get(vertexId);
    if (!vidCache) return new Set();
    const clamped = Math.min(hops, this.maxHops);
    return vidCache.get(clamped) ?? new Set();
  }

  hasVertex(vertexId: number): boolean {
    return this._cache.has(vertexId);
  }
}

// -- PathIndex ----------------------------------------------------------------

export class PathIndex {
  // Maps (graphName, labelSeqKey) -> Set of (startVid, endVid) pairs
  private _index: Map<string, Set<string>> = new Map();

  private _makeKey(labels: string[], graphName: string): string {
    return `${graphName}\0${labels.join("\0")}`;
  }

  private _pairKey(start: number, end: number): string {
    return `${String(start)}\0${String(end)}`;
  }

  private _parsePairKey(key: string): [number, number] {
    const parts = key.split("\0");
    return [parseInt(parts[0]!, 10), parseInt(parts[1]!, 10)];
  }

  build(store: GraphStore, graphName: string, labelSequences: string[][]): void {
    for (const labels of labelSequences) {
      const key = this._makeKey(labels, graphName);
      const pairs = new Set<string>();

      const allVertexIds = store.vertexIdsInGraph(graphName);

      for (const startVid of allVertexIds) {
        const reachable = this._findReachable(store, graphName, startVid, labels);
        for (const endVid of reachable) {
          pairs.add(this._pairKey(startVid, endVid));
        }
      }

      this._index.set(key, pairs);
    }
  }

  private _findReachable(
    store: GraphStore,
    graphName: string,
    startVid: number,
    labels: string[],
  ): number[] {
    // BFS through label sequence
    let current = new Set<number>([startVid]);

    for (const label of labels) {
      const next = new Set<number>();
      for (const vid of current) {
        const neighbors = store.neighbors(vid, graphName, label, "out");
        for (const nid of neighbors) {
          next.add(nid);
        }
      }
      current = next;
      if (current.size === 0) break;
    }

    return [...current];
  }

  lookup(labels: string[], graphName: string): Array<[number, number]> {
    const key = this._makeKey(labels, graphName);
    const pairs = this._index.get(key);
    if (!pairs) return [];
    return [...pairs].map((p) => this._parsePairKey(p));
  }

  hasPath(labels: string[], graphName: string): boolean {
    const key = this._makeKey(labels, graphName);
    return this._index.has(key);
  }

  indexedPaths(): string[][] {
    const result: string[][] = [];
    for (const key of this._index.keys()) {
      const parts = key.split("\0");
      // First part is graphName, rest are labels
      result.push(parts.slice(1));
    }
    return result;
  }
}

// -- SubgraphIndex ------------------------------------------------------------

export class SubgraphIndex {
  // Cache pattern match results by serialized pattern + graph name
  private _cache: Map<string, Array<Map<string, number>>> = new Map();

  private _patternKey(pattern: GraphPattern, graphName: string): string {
    // Simple serialization: variable names + edge patterns
    const vps = pattern.vertexPatterns
      .map((vp) => vp.variable)
      .sort()
      .join(",");
    const eps = pattern.edgePatterns
      .map(
        (ep) =>
          `${ep.sourceVar}->${ep.targetVar}:${ep.label ?? "*"}${ep.negated ? "!" : ""}`,
      )
      .sort()
      .join(",");
    return `${graphName}\0${vps}\0${eps}`;
  }

  build(
    _store: GraphStore,
    graphName: string,
    patterns: GraphPattern[],
    context: ExecutionContext,
  ): void {
    for (const pattern of patterns) {
      const key = this._patternKey(pattern, graphName);
      const op = new PatternMatchOperator(pattern, graphName);
      const result = op.execute(context);
      // Extract assignments from posting list fields
      const assignments: Array<Map<string, number>> = [];
      for (const entry of result) {
        const fields = entry.payload.fields as Record<string, unknown>;
        const assignment = new Map<string, number>();
        for (const vp of pattern.vertexPatterns) {
          const vid = fields[vp.variable];
          if (typeof vid === "number") {
            assignment.set(vp.variable, vid);
          }
        }
        assignments.push(assignment);
      }
      this._cache.set(key, assignments);
    }
  }

  lookup(pattern: GraphPattern, graphName: string): Array<Map<string, number>> | null {
    const key = this._patternKey(pattern, graphName);
    return this._cache.get(key) ?? null;
  }

  hasPattern(pattern: GraphPattern, graphName: string): boolean {
    const key = this._patternKey(pattern, graphName);
    return this._cache.has(key);
  }

  invalidate(pattern: GraphPattern, graphName: string): void {
    const key = this._patternKey(pattern, graphName);
    this._cache.delete(key);
  }
}

// -- VertexPropertyIndex ------------------------------------------------------

export class VertexPropertyIndex {
  // Hash index: propertyName -> value -> Set<vertexId>
  private _hashIndex: Map<string, Map<unknown, Set<number>>> = new Map();
  // Sorted index: propertyName -> sorted array of [value, vertexId]
  private _sortedIndex: Map<string, Array<[number, number]>> = new Map();
  private _indexedProperties: Set<string> = new Set();

  build(store: GraphStore, graphName: string, propertyNames: string[]): void {
    for (const propName of propertyNames) {
      this._indexedProperties.add(propName);
      const hashMap = new Map<unknown, Set<number>>();
      const sorted: Array<[number, number]> = [];

      const vertices = store.verticesInGraph(graphName);
      for (const vertex of vertices) {
        const value = vertex.properties[propName];
        if (value === undefined) continue;

        // Hash index
        let bucket = hashMap.get(value);
        if (!bucket) {
          bucket = new Set();
          hashMap.set(value, bucket);
        }
        bucket.add(vertex.vertexId);

        // Sorted index (only for numeric values)
        if (typeof value === "number") {
          sorted.push([value, vertex.vertexId]);
        }
      }

      this._hashIndex.set(propName, hashMap);
      sorted.sort((a, b) => a[0] - b[0]);
      this._sortedIndex.set(propName, sorted);
    }
  }

  lookupEq(propertyName: string, value: unknown): Set<number> {
    const hashMap = this._hashIndex.get(propertyName);
    if (!hashMap) return new Set();
    return hashMap.get(value) ?? new Set();
  }

  lookupRange(propertyName: string, low: number, high: number): number[] {
    const sorted = this._sortedIndex.get(propertyName);
    if (!sorted) return [];

    const result: number[] = [];
    // Binary search for lower bound
    let lo = 0;
    let hi = sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (sorted[mid]![0] < low) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }

    for (let i = lo; i < sorted.length; i++) {
      const [val, vid] = sorted[i]!;
      if (val > high) break;
      result.push(vid);
    }

    return result;
  }

  hasProperty(propertyName: string): boolean {
    return this._indexedProperties.has(propertyName);
  }
}

// -- EdgePropertyIndex --------------------------------------------------------

export class EdgePropertyIndex {
  private _hashIndex: Map<string, Map<unknown, Set<number>>> = new Map();
  private _sortedIndex: Map<string, Array<[number, number]>> = new Map();
  private _indexedProperties: Set<string> = new Set();

  build(store: GraphStore, graphName: string, propertyNames: string[]): void {
    for (const propName of propertyNames) {
      this._indexedProperties.add(propName);
      const hashMap = new Map<unknown, Set<number>>();
      const sorted: Array<[number, number]> = [];

      const edges = store.edgesInGraph(graphName);
      for (const edge of edges) {
        const value = edge.properties[propName];
        if (value === undefined) continue;

        let bucket = hashMap.get(value);
        if (!bucket) {
          bucket = new Set();
          hashMap.set(value, bucket);
        }
        bucket.add(edge.edgeId);

        if (typeof value === "number") {
          sorted.push([value, edge.edgeId]);
        }
      }

      this._hashIndex.set(propName, hashMap);
      sorted.sort((a, b) => a[0] - b[0]);
      this._sortedIndex.set(propName, sorted);
    }
  }

  lookupEq(propertyName: string, value: unknown): Set<number> {
    const hashMap = this._hashIndex.get(propertyName);
    if (!hashMap) return new Set();
    return hashMap.get(value) ?? new Set();
  }

  lookupRange(propertyName: string, low: number, high: number): number[] {
    const sorted = this._sortedIndex.get(propertyName);
    if (!sorted) return [];

    const result: number[] = [];
    let lo = 0;
    let hi = sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (sorted[mid]![0] < low) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }

    for (let i = lo; i < sorted.length; i++) {
      const [val, eid] = sorted[i]!;
      if (val > high) break;
      result.push(eid);
    }

    return result;
  }

  hasProperty(propertyName: string): boolean {
    return this._indexedProperties.has(propertyName);
  }
}
