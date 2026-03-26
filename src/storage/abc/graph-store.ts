//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- GraphStore abstract interface
// 1:1 port of uqa/storage/abc/graph_store.py
//
// A graph store manages named graphs with vertices and edges, supporting
// graph lifecycle, mutations, queries, adjacency access, and statistics.
// Concrete implementations include in-memory and SQLite-backed stores.

import type { Edge, Vertex } from "../../core/types.js";

/**
 * Abstract interface for graph storage backends.
 *
 * A graph store manages named graphs with vertices and edges, supporting
 * graph lifecycle, mutations, queries, adjacency access, and statistics.
 * Concrete implementations include in-memory and SQLite-backed stores.
 */
export abstract class GraphStore {
  // -- Graph lifecycle --------------------------------------------------------

  /** Create a new named graph. */
  abstract createGraph(name: string): void;

  /** Drop a named graph and its data. */
  abstract dropGraph(name: string): void;

  /** Return sorted list of graph names. */
  abstract graphNames(): string[];

  /** Return true if graph exists. */
  abstract hasGraph(name: string): boolean;

  // -- Graph algebra ----------------------------------------------------------

  /** Create target graph as union of g1 and g2. */
  abstract unionGraphs(g1: string, g2: string, target: string): void;

  /** Create target graph as intersection of g1 and g2. */
  abstract intersectGraphs(g1: string, g2: string, target: string): void;

  /** Create target graph as g1 - g2. */
  abstract differenceGraphs(g1: string, g2: string, target: string): void;

  /** Copy source graph to target graph. */
  abstract copyGraph(source: string, target: string): void;

  // -- Mutation ---------------------------------------------------------------

  /** Add a vertex to a named graph. */
  abstract addVertex(vertex: Vertex, graph: string): void;

  /** Add an edge to a named graph. */
  abstract addEdge(edge: Edge, graph: string): void;

  /** Remove a vertex from a named graph. */
  abstract removeVertex(vertexId: number, graph: string): void;

  /** Remove an edge from a named graph. */
  abstract removeEdge(edgeId: number, graph: string): void;

  // -- Query ------------------------------------------------------------------

  /** Return neighbor vertex IDs. */
  abstract neighbors(
    vertexId: number,
    graph: string,
    label?: string | null,
    direction?: "out" | "in",
  ): number[];

  /** Return vertices with a given label in a graph. */
  abstract verticesByLabel(label: string, graph: string): Vertex[];

  /** Return all vertices in a graph. */
  abstract verticesInGraph(graph: string): Vertex[];

  /** Return all edges in a graph. */
  abstract edgesInGraph(graph: string): Edge[];

  /** Return set of graph names a vertex belongs to. */
  abstract vertexGraphs(vertexId: number): Set<string>;

  // -- Graph-scoped adjacency -------------------------------------------------

  /** Return outgoing edge IDs for vertex in a specific graph. */
  abstract outEdgeIds(vertexId: number, graph: string): Set<number>;

  /** Return incoming edge IDs for vertex in a specific graph. */
  abstract inEdgeIds(vertexId: number, graph: string): Set<number>;

  /** Return edge IDs with a given label in a specific graph. */
  abstract edgeIdsByLabel(label: string, graph: string): Set<number>;

  /** Return all vertex IDs in a specific graph. */
  abstract vertexIdsInGraph(graph: string): Set<number>;

  // -- Statistics -------------------------------------------------------------

  /** Out-degree distribution for vertices in graph. */
  abstract degreeDistribution(graph: string): Map<number, number>;

  /** Average out-degree for edges with given label in graph. */
  abstract labelDegree(label: string, graph: string): number;

  /** Count of vertices per vertex label in graph. */
  abstract vertexLabelCounts(graph: string): Map<string, number>;

  // -- Bulk mutation ----------------------------------------------------------

  /** Add multiple vertices to a named graph in a single batch. */
  abstract addVertices(vertices: Vertex[], graph: string): void;

  /** Add multiple edges to a named graph in a single batch. */
  abstract addEdges(edges: Edge[], graph: string): void;

  // -- Property access -------------------------------------------------------

  /** Return a property value from a vertex. */
  abstract vertexProperty(vertexId: number, key: string): unknown;

  /** Set a property value on a vertex. */
  abstract setVertexProperty(vertexId: number, key: string, value: unknown): void;

  /** Return a property value from an edge. */
  abstract edgeProperty(edgeId: number, key: string): unknown;

  /** Set a property value on an edge. */
  abstract setEdgeProperty(edgeId: number, key: string, value: unknown): void;

  // -- Label queries ---------------------------------------------------------

  /** Return edges with a given label in a specific graph. */
  abstract edgesByLabel(label: string, graph: string): Edge[];

  /** Return the set of all vertex labels in a graph. */
  abstract vertexLabels(graph: string): Set<string>;

  /** Return the set of all edge labels in a graph. */
  abstract edgeLabels(graph: string): Set<string>;

  // -- Degree queries --------------------------------------------------------

  /** Return the out-degree of a vertex in a graph. */
  abstract outDegree(vertexId: number, graph: string): number;

  /** Return the in-degree of a vertex in a graph. */
  abstract inDegree(vertexId: number, graph: string): number;

  // -- Edge-centric queries --------------------------------------------------

  /** Return edges between two specific vertices, optionally filtered by label. */
  abstract edgesBetween(
    sourceId: number,
    targetId: number,
    graph: string,
    label?: string | null,
  ): Edge[];

  // -- Subgraph extraction ---------------------------------------------------

  /** Extract a subgraph containing only the specified vertex IDs. */
  abstract subgraph(vertexIds: Set<number>, graph: string, target: string): void;

  // -- Temporal support (optional, returns null/empty for non-temporal stores) -

  /** Return the minimum timestamp across all edges in the graph. */
  abstract minTimestamp(graph: string): number | null;

  /** Return the maximum timestamp across all edges in the graph. */
  abstract maxTimestamp(graph: string): number | null;

  /** Return edges within a time range in the graph. */
  abstract edgesInTimeRange(graph: string, startTime: number, endTime: number): Edge[];

  // -- Global accessors -------------------------------------------------------

  /** Return vertex by ID, or null if not found. */
  abstract getVertex(vertexId: number): Vertex | null;

  /** Return edge by ID, or null if not found. */
  abstract getEdge(edgeId: number): Edge | null;

  /** Return and advance the next available vertex ID. */
  abstract nextVertexId(): number;

  /** Return and advance the next available edge ID. */
  abstract nextEdgeId(): number;

  /** Remove all vertices, edges, and graphs. */
  abstract clear(): void;

  /** Return a copy of all vertices. */
  abstract get vertices(): Map<number, Vertex>;

  /** Return a copy of all edges. */
  abstract get edges(): Map<number, Edge>;

  /** Return the total number of vertices. */
  abstract get vertexCount(): number;

  /** Return the total number of edges. */
  abstract get edgeCount(): number;
}
