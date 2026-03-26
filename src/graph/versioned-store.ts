//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Versioned graph store
// 1:1 port of uqa/graph/versioned_store.py

import type { Vertex, Edge } from "../core/types.js";
import { GraphStore } from "../storage/abc/graph-store.js";
import type { GraphDelta } from "./delta.js";

// -- VersionedGraphStore ------------------------------------------------------

export class VersionedGraphStore extends GraphStore {
  private _inner: GraphStore;
  private _versions: GraphDelta[] = [];
  private _currentVersion: number = 0;

  constructor(inner: GraphStore) {
    super();
    this._inner = inner;
  }

  get currentVersion(): number {
    return this._currentVersion;
  }

  get versionCount(): number {
    return this._versions.length;
  }

  get inner(): GraphStore {
    return this._inner;
  }

  // -- Apply a delta and record it as a version --------------------------------

  apply(delta: GraphDelta, graph: string): void {
    // Truncate any versions after current (if we rolled back and are branching)
    this._versions.length = this._currentVersion;

    for (const op of delta.ops) {
      switch (op.kind) {
        case "add_vertex":
          if (op.vertex) {
            this._inner.addVertex(op.vertex, graph);
          }
          break;
        case "remove_vertex":
          if (op.vertexId !== undefined) {
            this._inner.removeVertex(op.vertexId, graph);
          }
          break;
        case "add_edge":
          if (op.edge) {
            this._inner.addEdge(op.edge, graph);
          }
          break;
        case "remove_edge":
          if (op.edgeId !== undefined) {
            this._inner.removeEdge(op.edgeId, graph);
          }
          break;
      }
    }

    this._versions.push(delta);
    this._currentVersion = this._versions.length;
  }

  // -- Rollback to a previous version ------------------------------------------

  rollback(targetVersion: number, graph: string): void {
    if (targetVersion < 0 || targetVersion > this._currentVersion) {
      throw new Error(
        `Invalid target version: ${String(targetVersion)} (current: ${String(this._currentVersion)})`,
      );
    }

    // Undo operations from current version back to target
    for (let v = this._currentVersion - 1; v >= targetVersion; v--) {
      const delta = this._versions[v]!;
      // Undo in reverse order
      const ops = delta.ops;
      for (let i = ops.length - 1; i >= 0; i--) {
        const op = ops[i]!;
        switch (op.kind) {
          case "add_vertex":
            if (op.vertex) {
              this._inner.removeVertex(op.vertex.vertexId, graph);
            }
            break;
          case "remove_vertex":
            if (op.vertex) {
              this._inner.addVertex(op.vertex, graph);
            }
            break;
          case "add_edge":
            if (op.edge) {
              this._inner.removeEdge(op.edge.edgeId, graph);
            }
            break;
          case "remove_edge":
            if (op.edge) {
              this._inner.addEdge(op.edge, graph);
            }
            break;
        }
      }
    }

    this._currentVersion = targetVersion;
  }

  // -- Delegate all GraphStore abstract methods to inner -----------------------

  createGraph(name: string): void {
    this._inner.createGraph(name);
  }

  dropGraph(name: string): void {
    this._inner.dropGraph(name);
  }

  graphNames(): string[] {
    return this._inner.graphNames();
  }

  hasGraph(name: string): boolean {
    return this._inner.hasGraph(name);
  }

  unionGraphs(g1: string, g2: string, target: string): void {
    this._inner.unionGraphs(g1, g2, target);
  }

  intersectGraphs(g1: string, g2: string, target: string): void {
    this._inner.intersectGraphs(g1, g2, target);
  }

  differenceGraphs(g1: string, g2: string, target: string): void {
    this._inner.differenceGraphs(g1, g2, target);
  }

  copyGraph(source: string, target: string): void {
    this._inner.copyGraph(source, target);
  }

  addVertex(vertex: Vertex, graph: string): void {
    this._inner.addVertex(vertex, graph);
  }

  addEdge(edge: Edge, graph: string): void {
    this._inner.addEdge(edge, graph);
  }

  removeVertex(vertexId: number, graph: string): void {
    this._inner.removeVertex(vertexId, graph);
  }

  removeEdge(edgeId: number, graph: string): void {
    this._inner.removeEdge(edgeId, graph);
  }

  neighbors(
    vertexId: number,
    graph: string,
    label?: string | null,
    direction?: "out" | "in",
  ): number[] {
    return this._inner.neighbors(vertexId, graph, label, direction);
  }

  verticesByLabel(label: string, graph: string): Vertex[] {
    return this._inner.verticesByLabel(label, graph);
  }

  verticesInGraph(graph: string): Vertex[] {
    return this._inner.verticesInGraph(graph);
  }

  edgesInGraph(graph: string): Edge[] {
    return this._inner.edgesInGraph(graph);
  }

  vertexGraphs(vertexId: number): Set<string> {
    return this._inner.vertexGraphs(vertexId);
  }

  outEdgeIds(vertexId: number, graph: string): Set<number> {
    return this._inner.outEdgeIds(vertexId, graph);
  }

  inEdgeIds(vertexId: number, graph: string): Set<number> {
    return this._inner.inEdgeIds(vertexId, graph);
  }

  edgeIdsByLabel(label: string, graph: string): Set<number> {
    return this._inner.edgeIdsByLabel(label, graph);
  }

  vertexIdsInGraph(graph: string): Set<number> {
    return this._inner.vertexIdsInGraph(graph);
  }

  degreeDistribution(graph: string): Map<number, number> {
    return this._inner.degreeDistribution(graph);
  }

  labelDegree(label: string, graph: string): number {
    return this._inner.labelDegree(label, graph);
  }

  vertexLabelCounts(graph: string): Map<string, number> {
    return this._inner.vertexLabelCounts(graph);
  }

  getVertex(vertexId: number): Vertex | null {
    return this._inner.getVertex(vertexId);
  }

  getEdge(edgeId: number): Edge | null {
    return this._inner.getEdge(edgeId);
  }

  nextVertexId(): number {
    return this._inner.nextVertexId();
  }

  nextEdgeId(): number {
    return this._inner.nextEdgeId();
  }

  clear(): void {
    this._inner.clear();
    this._versions = [];
    this._currentVersion = 0;
  }

  get vertices(): Map<number, Vertex> {
    return this._inner.vertices;
  }

  get edges(): Map<number, Edge> {
    return this._inner.edges;
  }

  get vertexCount(): number {
    return this._inner.vertexCount;
  }

  get edgeCount(): number {
    return this._inner.edgeCount;
  }

  addVertices(vertices: Vertex[], graph: string): void {
    this._inner.addVertices(vertices, graph);
  }

  addEdges(edges: Edge[], graph: string): void {
    this._inner.addEdges(edges, graph);
  }

  vertexProperty(vertexId: number, key: string): unknown {
    return this._inner.vertexProperty(vertexId, key);
  }

  setVertexProperty(vertexId: number, key: string, value: unknown): void {
    this._inner.setVertexProperty(vertexId, key, value);
  }

  edgeProperty(edgeId: number, key: string): unknown {
    return this._inner.edgeProperty(edgeId, key);
  }

  setEdgeProperty(edgeId: number, key: string, value: unknown): void {
    this._inner.setEdgeProperty(edgeId, key, value);
  }

  edgesByLabel(label: string, graph: string): Edge[] {
    return this._inner.edgesByLabel(label, graph);
  }

  vertexLabels(graph: string): Set<string> {
    return this._inner.vertexLabels(graph);
  }

  edgeLabels(graph: string): Set<string> {
    return this._inner.edgeLabels(graph);
  }

  outDegree(vertexId: number, graph: string): number {
    return this._inner.outDegree(vertexId, graph);
  }

  inDegree(vertexId: number, graph: string): number {
    return this._inner.inDegree(vertexId, graph);
  }

  edgesBetween(
    sourceId: number,
    targetId: number,
    graph: string,
    label?: string | null,
  ): Edge[] {
    return this._inner.edgesBetween(sourceId, targetId, graph, label);
  }

  subgraph(vertexIds: Set<number>, graph: string, target: string): void {
    this._inner.subgraph(vertexIds, graph, target);
  }

  minTimestamp(graph: string): number | null {
    return this._inner.minTimestamp(graph);
  }

  maxTimestamp(graph: string): number | null {
    return this._inner.maxTimestamp(graph);
  }

  edgesInTimeRange(graph: string, startTime: number, endTime: number): Edge[] {
    return this._inner.edgesInTimeRange(graph, startTime, endTime);
  }
}
