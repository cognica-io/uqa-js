//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- MemoryGraphStore
// 1:1 port of uqa/graph/store.py

import type { Edge, Vertex } from "../core/types.js";
import { GraphStore } from "../storage/abc/graph-store.js";

// -- Graph partition (internal) -----------------------------------------------

class _GraphPartition {
  readonly vertexIds: Set<number> = new Set();
  readonly edgeIds: Set<number> = new Set();
  readonly adjOut: Map<number, Set<number>> = new Map();
  readonly adjIn: Map<number, Set<number>> = new Map();
  readonly labelIndex: Map<string, Set<number>> = new Map();
  readonly vertexLabelIndex: Map<string, Set<number>> = new Map();

  addVertex(vertexId: number, label: string): void {
    this.vertexIds.add(vertexId);
    if (!this.adjOut.has(vertexId)) {
      this.adjOut.set(vertexId, new Set());
    }
    if (!this.adjIn.has(vertexId)) {
      this.adjIn.set(vertexId, new Set());
    }
    let labelSet = this.vertexLabelIndex.get(label);
    if (!labelSet) {
      labelSet = new Set();
      this.vertexLabelIndex.set(label, labelSet);
    }
    labelSet.add(vertexId);
  }

  removeVertex(vertexId: number, label: string): void {
    this.vertexIds.delete(vertexId);
    this.adjOut.delete(vertexId);
    this.adjIn.delete(vertexId);
    // Remove from all adj lists
    for (const [, outSet] of this.adjOut) {
      outSet.delete(vertexId);
    }
    for (const [, inSet] of this.adjIn) {
      inSet.delete(vertexId);
    }
    const labelSet = this.vertexLabelIndex.get(label);
    if (labelSet) {
      labelSet.delete(vertexId);
      if (labelSet.size === 0) {
        this.vertexLabelIndex.delete(label);
      }
    }
  }

  addEdge(edgeId: number, sourceId: number, targetId: number, label: string): void {
    this.edgeIds.add(edgeId);
    let outSet = this.adjOut.get(sourceId);
    if (!outSet) {
      outSet = new Set();
      this.adjOut.set(sourceId, outSet);
    }
    outSet.add(edgeId);

    let inSet = this.adjIn.get(targetId);
    if (!inSet) {
      inSet = new Set();
      this.adjIn.set(targetId, inSet);
    }
    inSet.add(edgeId);

    let edgeSet = this.labelIndex.get(label);
    if (!edgeSet) {
      edgeSet = new Set();
      this.labelIndex.set(label, edgeSet);
    }
    edgeSet.add(edgeId);
  }

  removeEdge(edgeId: number, sourceId: number, targetId: number, label: string): void {
    this.edgeIds.delete(edgeId);
    const outSet = this.adjOut.get(sourceId);
    if (outSet) {
      outSet.delete(edgeId);
    }
    const inSet = this.adjIn.get(targetId);
    if (inSet) {
      inSet.delete(edgeId);
    }
    const edgeSet = this.labelIndex.get(label);
    if (edgeSet) {
      edgeSet.delete(edgeId);
      if (edgeSet.size === 0) {
        this.labelIndex.delete(label);
      }
    }
  }

  neighbors(
    vertexId: number,
    edges: Map<number, Edge>,
    label: string | null,
    direction: "out" | "in",
  ): number[] {
    const result: number[] = [];
    if (direction === "out") {
      const outSet = this.adjOut.get(vertexId);
      if (outSet) {
        for (const edgeId of outSet) {
          const edge = edges.get(edgeId);
          if (edge && (label === null || edge.label === label)) {
            result.push(edge.targetId);
          }
        }
      }
    } else {
      const inSet = this.adjIn.get(vertexId);
      if (inSet) {
        for (const edgeId of inSet) {
          const edge = edges.get(edgeId);
          if (edge && (label === null || edge.label === label)) {
            result.push(edge.sourceId);
          }
        }
      }
    }
    return result;
  }

  verticesByLabel(label: string): Set<number> {
    return this.vertexLabelIndex.get(label) ?? new Set();
  }
}

// -- MemoryGraphStore ---------------------------------------------------------

export class MemoryGraphStore extends GraphStore {
  private _vertices: Map<number, Vertex> = new Map();
  private _edges: Map<number, Edge> = new Map();
  private _graphs: Map<string, _GraphPartition> = new Map();
  private _vertexMembership: Map<number, Set<string>> = new Map();
  private _edgeMembership: Map<number, Set<string>> = new Map();
  private _nextVertexId: number = 1;
  private _nextEdgeId: number = 1;

  // -- Graph lifecycle --------------------------------------------------------

  createGraph(name: string): void {
    if (!this._graphs.has(name)) {
      this._graphs.set(name, new _GraphPartition());
    }
  }

  dropGraph(name: string): void {
    const partition = this._graphs.get(name);
    if (!partition) return;

    // Clean up membership
    for (const vid of partition.vertexIds) {
      const membership = this._vertexMembership.get(vid);
      if (membership) {
        membership.delete(name);
        if (membership.size === 0) {
          this._vertexMembership.delete(vid);
        }
      }
    }
    for (const eid of partition.edgeIds) {
      const membership = this._edgeMembership.get(eid);
      if (membership) {
        membership.delete(name);
        if (membership.size === 0) {
          this._edgeMembership.delete(eid);
        }
      }
    }

    this._graphs.delete(name);
  }

  graphNames(): string[] {
    return [...this._graphs.keys()];
  }

  hasGraph(name: string): boolean {
    return this._graphs.has(name);
  }

  // -- Graph algebra ----------------------------------------------------------

  unionGraphs(g1: string, g2: string, target: string): void {
    this.createGraph(target);
    const src1 = this._graphs.get(g1);
    const src2 = this._graphs.get(g2);

    if (src1) {
      for (const vid of src1.vertexIds) {
        const vertex = this._vertices.get(vid);
        if (vertex) this.addVertex(vertex, target);
      }
      for (const eid of src1.edgeIds) {
        const edge = this._edges.get(eid);
        if (edge) this.addEdge(edge, target);
      }
    }
    if (src2) {
      for (const vid of src2.vertexIds) {
        const vertex = this._vertices.get(vid);
        if (vertex) this.addVertex(vertex, target);
      }
      for (const eid of src2.edgeIds) {
        const edge = this._edges.get(eid);
        if (edge) this.addEdge(edge, target);
      }
    }
  }

  intersectGraphs(g1: string, g2: string, target: string): void {
    this.createGraph(target);
    const src1 = this._graphs.get(g1);
    const src2 = this._graphs.get(g2);
    if (!src1 || !src2) return;

    for (const vid of src1.vertexIds) {
      if (src2.vertexIds.has(vid)) {
        const vertex = this._vertices.get(vid);
        if (vertex) this.addVertex(vertex, target);
      }
    }
    for (const eid of src1.edgeIds) {
      if (src2.edgeIds.has(eid)) {
        const edge = this._edges.get(eid);
        if (edge) this.addEdge(edge, target);
      }
    }
  }

  differenceGraphs(g1: string, g2: string, target: string): void {
    this.createGraph(target);
    const src1 = this._graphs.get(g1);
    const src2 = this._graphs.get(g2);
    if (!src1) return;

    const excludeV = src2 ? src2.vertexIds : new Set<number>();
    const excludeE = src2 ? src2.edgeIds : new Set<number>();

    for (const vid of src1.vertexIds) {
      if (!excludeV.has(vid)) {
        const vertex = this._vertices.get(vid);
        if (vertex) this.addVertex(vertex, target);
      }
    }
    for (const eid of src1.edgeIds) {
      if (!excludeE.has(eid)) {
        const edge = this._edges.get(eid);
        if (edge) this.addEdge(edge, target);
      }
    }
  }

  copyGraph(source: string, target: string): void {
    this.createGraph(target);
    const src = this._graphs.get(source);
    if (!src) return;

    for (const vid of src.vertexIds) {
      const vertex = this._vertices.get(vid);
      if (vertex) this.addVertex(vertex, target);
    }
    for (const eid of src.edgeIds) {
      const edge = this._edges.get(eid);
      if (edge) this.addEdge(edge, target);
    }
  }

  // -- Mutation ---------------------------------------------------------------

  addVertex(vertex: Vertex, graph: string): void {
    this._vertices.set(vertex.vertexId, vertex);
    if (vertex.vertexId >= this._nextVertexId) {
      this._nextVertexId = vertex.vertexId + 1;
    }

    const partition = this._graphs.get(graph);
    if (partition) {
      partition.addVertex(vertex.vertexId, vertex.label);
    }

    let membership = this._vertexMembership.get(vertex.vertexId);
    if (!membership) {
      membership = new Set();
      this._vertexMembership.set(vertex.vertexId, membership);
    }
    membership.add(graph);
  }

  addEdge(edge: Edge, graph: string): void {
    this._edges.set(edge.edgeId, edge);
    if (edge.edgeId >= this._nextEdgeId) {
      this._nextEdgeId = edge.edgeId + 1;
    }

    const partition = this._graphs.get(graph);
    if (partition) {
      partition.addEdge(edge.edgeId, edge.sourceId, edge.targetId, edge.label);
    }

    let membership = this._edgeMembership.get(edge.edgeId);
    if (!membership) {
      membership = new Set();
      this._edgeMembership.set(edge.edgeId, membership);
    }
    membership.add(graph);
  }

  removeVertex(vertexId: number, graph: string): void {
    const vertex = this._vertices.get(vertexId);
    if (!vertex) return;

    const partition = this._graphs.get(graph);
    if (partition) {
      // Remove edges incident to this vertex in this graph
      const edgesToRemove: number[] = [];
      const outSet = partition.adjOut.get(vertexId);
      if (outSet) {
        for (const eid of outSet) edgesToRemove.push(eid);
      }
      const inSet = partition.adjIn.get(vertexId);
      if (inSet) {
        for (const eid of inSet) edgesToRemove.push(eid);
      }
      for (const eid of edgesToRemove) {
        this.removeEdge(eid, graph);
      }
      partition.removeVertex(vertexId, vertex.label);
    }

    const membership = this._vertexMembership.get(vertexId);
    if (membership) {
      membership.delete(graph);
      if (membership.size === 0) {
        this._vertexMembership.delete(vertexId);
        this._vertices.delete(vertexId);
      }
    }
  }

  removeEdge(edgeId: number, graph: string): void {
    const edge = this._edges.get(edgeId);
    if (!edge) return;

    const partition = this._graphs.get(graph);
    if (partition) {
      partition.removeEdge(edgeId, edge.sourceId, edge.targetId, edge.label);
    }

    const membership = this._edgeMembership.get(edgeId);
    if (membership) {
      membership.delete(graph);
      if (membership.size === 0) {
        this._edgeMembership.delete(edgeId);
        this._edges.delete(edgeId);
      }
    }
  }

  // -- Query ------------------------------------------------------------------

  neighbors(
    vertexId: number,
    graph: string,
    label?: string | null,
    direction?: "out" | "in",
  ): number[] {
    const partition = this._graphs.get(graph);
    if (!partition) return [];
    return partition.neighbors(
      vertexId,
      this._edges,
      label ?? null,
      direction ?? "out",
    );
  }

  verticesByLabel(label: string, graph: string): Vertex[] {
    const partition = this._graphs.get(graph);
    if (!partition) return [];
    const ids = partition.verticesByLabel(label);
    const result: Vertex[] = [];
    for (const vid of ids) {
      const v = this._vertices.get(vid);
      if (v) result.push(v);
    }
    return result;
  }

  verticesInGraph(graph: string): Vertex[] {
    const partition = this._graphs.get(graph);
    if (!partition) return [];
    const result: Vertex[] = [];
    for (const vid of partition.vertexIds) {
      const v = this._vertices.get(vid);
      if (v) result.push(v);
    }
    return result;
  }

  edgesInGraph(graph: string): Edge[] {
    const partition = this._graphs.get(graph);
    if (!partition) return [];
    const result: Edge[] = [];
    for (const eid of partition.edgeIds) {
      const e = this._edges.get(eid);
      if (e) result.push(e);
    }
    return result;
  }

  vertexGraphs(vertexId: number): Set<string> {
    return this._vertexMembership.get(vertexId) ?? new Set();
  }

  // -- Graph-scoped adjacency -------------------------------------------------

  outEdgeIds(vertexId: number, graph: string): Set<number> {
    const partition = this._graphs.get(graph);
    if (!partition) return new Set();
    return partition.adjOut.get(vertexId) ?? new Set();
  }

  inEdgeIds(vertexId: number, graph: string): Set<number> {
    const partition = this._graphs.get(graph);
    if (!partition) return new Set();
    return partition.adjIn.get(vertexId) ?? new Set();
  }

  edgeIdsByLabel(label: string, graph: string): Set<number> {
    const partition = this._graphs.get(graph);
    if (!partition) return new Set();
    return partition.labelIndex.get(label) ?? new Set();
  }

  vertexIdsInGraph(graph: string): Set<number> {
    const partition = this._graphs.get(graph);
    if (!partition) return new Set();
    return new Set(partition.vertexIds);
  }

  // -- Statistics -------------------------------------------------------------

  degreeDistribution(graph: string): Map<number, number> {
    const partition = this._graphs.get(graph);
    if (!partition) return new Map();
    const dist = new Map<number, number>();
    for (const vid of partition.vertexIds) {
      const outSet = partition.adjOut.get(vid);
      const inSet = partition.adjIn.get(vid);
      const degree = (outSet ? outSet.size : 0) + (inSet ? inSet.size : 0);
      dist.set(degree, (dist.get(degree) ?? 0) + 1);
    }
    return dist;
  }

  labelDegree(label: string, graph: string): number {
    const partition = this._graphs.get(graph);
    if (!partition) return 0;
    const edgeSet = partition.labelIndex.get(label);
    return edgeSet ? edgeSet.size : 0;
  }

  vertexLabelCounts(graph: string): Map<string, number> {
    const partition = this._graphs.get(graph);
    if (!partition) return new Map();
    const counts = new Map<string, number>();
    for (const [label, ids] of partition.vertexLabelIndex) {
      counts.set(label, ids.size);
    }
    return counts;
  }

  // -- Global accessors -------------------------------------------------------

  getVertex(vertexId: number): Vertex | null {
    return this._vertices.get(vertexId) ?? null;
  }

  getEdge(edgeId: number): Edge | null {
    return this._edges.get(edgeId) ?? null;
  }

  nextVertexId(): number {
    return this._nextVertexId++;
  }

  nextEdgeId(): number {
    return this._nextEdgeId++;
  }

  clear(): void {
    this._vertices.clear();
    this._edges.clear();
    this._graphs.clear();
    this._vertexMembership.clear();
    this._edgeMembership.clear();
    this._nextVertexId = 1;
    this._nextEdgeId = 1;
  }

  get vertices(): Map<number, Vertex> {
    return this._vertices;
  }

  get edges(): Map<number, Edge> {
    return this._edges;
  }

  get vertexCount(): number {
    return this._vertices.size;
  }

  get edgeCount(): number {
    return this._edges.size;
  }

  // -- Bulk mutation ----------------------------------------------------------

  addVertices(vertices: Vertex[], graph: string): void {
    for (const v of vertices) {
      this.addVertex(v, graph);
    }
  }

  addEdges(edges: Edge[], graph: string): void {
    for (const e of edges) {
      this.addEdge(e, graph);
    }
  }

  // -- Property access -------------------------------------------------------

  vertexProperty(vertexId: number, key: string): unknown {
    const v = this._vertices.get(vertexId);
    if (v === undefined) return undefined;
    return v.properties[key];
  }

  setVertexProperty(vertexId: number, key: string, value: unknown): void {
    const v = this._vertices.get(vertexId);
    if (v === undefined) throw new Error(`Vertex ${String(vertexId)} not found`);
    (v.properties as Record<string, unknown>)[key] = value;
  }

  edgeProperty(edgeId: number, key: string): unknown {
    const e = this._edges.get(edgeId);
    if (e === undefined) return undefined;
    return e.properties[key];
  }

  setEdgeProperty(edgeId: number, key: string, value: unknown): void {
    const e = this._edges.get(edgeId);
    if (e === undefined) throw new Error(`Edge ${String(edgeId)} not found`);
    (e.properties as Record<string, unknown>)[key] = value;
  }

  // -- Label queries ---------------------------------------------------------

  edgesByLabel(label: string, graph: string): Edge[] {
    const result: Edge[] = [];
    const graphSet = this._graphs.get(graph);
    if (!graphSet) return result;
    for (const [eid, e] of this._edges) {
      if (e.label === label) {
        const membership = this._edgeMembership.get(eid);
        if (membership && membership.has(graph)) {
          result.push(e);
        }
      }
    }
    return result;
  }

  vertexLabels(graph: string): Set<string> {
    const labels = new Set<string>();
    for (const v of this.verticesInGraph(graph)) {
      if (v.label) labels.add(v.label);
    }
    return labels;
  }

  edgeLabels(graph: string): Set<string> {
    const labels = new Set<string>();
    for (const e of this.edgesInGraph(graph)) {
      labels.add(e.label);
    }
    return labels;
  }

  // -- Degree queries --------------------------------------------------------

  outDegree(vertexId: number, graph: string): number {
    return this.outEdgeIds(vertexId, graph).size;
  }

  inDegree(vertexId: number, graph: string): number {
    return this.inEdgeIds(vertexId, graph).size;
  }

  // -- Edge-centric queries --------------------------------------------------

  edgesBetween(
    sourceId: number,
    targetId: number,
    graph: string,
    label?: string | null,
  ): Edge[] {
    const result: Edge[] = [];
    const outEdges = this.outEdgeIds(sourceId, graph);
    for (const eid of outEdges) {
      const edge = this._edges.get(eid);
      if (edge && edge.targetId === targetId) {
        if (label === null || label === undefined || edge.label === label) {
          result.push(edge);
        }
      }
    }
    return result;
  }

  // -- Subgraph extraction ---------------------------------------------------

  subgraph(vertexIds: Set<number>, graph: string, target: string): void {
    if (!this._graphs.has(target)) {
      this.createGraph(target);
    }
    for (const vid of vertexIds) {
      const v = this._vertices.get(vid);
      if (v) {
        this.addVertex(v, target);
      }
    }
    for (const [eid, e] of this._edges) {
      const membership = this._edgeMembership.get(eid);
      if (membership && membership.has(graph)) {
        if (vertexIds.has(e.sourceId) && vertexIds.has(e.targetId)) {
          this.addEdge(e, target);
        }
      }
    }
  }

  // -- Temporal support -------------------------------------------------------

  minTimestamp(graph: string): number | null {
    let min: number | null = null;
    for (const e of this.edgesInGraph(graph)) {
      const ts = e.properties["timestamp"];
      if (typeof ts === "number") {
        if (min === null || ts < min) min = ts;
      }
    }
    return min;
  }

  maxTimestamp(graph: string): number | null {
    let max: number | null = null;
    for (const e of this.edgesInGraph(graph)) {
      const ts = e.properties["timestamp"];
      if (typeof ts === "number") {
        if (max === null || ts > max) max = ts;
      }
    }
    return max;
  }

  edgesInTimeRange(graph: string, startTime: number, endTime: number): Edge[] {
    const result: Edge[] = [];
    for (const e of this.edgesInGraph(graph)) {
      const ts = e.properties["timestamp"];
      if (typeof ts === "number" && ts >= startTime && ts <= endTime) {
        result.push(e);
      }
    }
    return result;
  }
}
