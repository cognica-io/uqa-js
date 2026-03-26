//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- SQLite-backed GraphStore
// 1:1 port of uqa/storage/sqlite_graph_store.py

import type { Edge, Vertex } from "../core/types.js";
import { GraphStore } from "./abc/graph-store.js";
import type { ManagedConnection } from "./managed-connection.js";

export class SQLiteGraphStore extends GraphStore {
  private _conn: ManagedConnection;
  private _tableName: string;

  constructor(conn: ManagedConnection, tableName?: string | null) {
    super();
    this._conn = conn;
    this._tableName = tableName ?? "_uqa_graph";
    this._ensureTables();
  }

  private _ensureTables(): void {
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS "${this._tableName}_vertices" (
        vertex_id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        properties_json TEXT NOT NULL DEFAULT '{}'
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS "${this._tableName}_edges" (
        edge_id INTEGER PRIMARY KEY,
        source_id INTEGER NOT NULL,
        target_id INTEGER NOT NULL,
        label TEXT NOT NULL,
        properties_json TEXT NOT NULL DEFAULT '{}'
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS "${this._tableName}_graphs" (
        graph_name TEXT PRIMARY KEY
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS "${this._tableName}_vertex_membership" (
        vertex_id INTEGER NOT NULL,
        graph_name TEXT NOT NULL,
        PRIMARY KEY (vertex_id, graph_name)
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS "${this._tableName}_edge_membership" (
        edge_id INTEGER NOT NULL,
        graph_name TEXT NOT NULL,
        PRIMARY KEY (edge_id, graph_name)
      )
    `);
  }

  // -- Graph lifecycle --------------------------------------------------------

  createGraph(name: string): void {
    this._conn.execute(
      `INSERT OR IGNORE INTO "${this._tableName}_graphs" (graph_name) VALUES (?)`,
      [name],
    );
  }

  dropGraph(name: string): void {
    // Remove all membership entries for this graph
    this._conn.execute(
      `DELETE FROM "${this._tableName}_vertex_membership" WHERE graph_name = ?`,
      [name],
    );
    this._conn.execute(
      `DELETE FROM "${this._tableName}_edge_membership" WHERE graph_name = ?`,
      [name],
    );
    this._conn.execute(`DELETE FROM "${this._tableName}_graphs" WHERE graph_name = ?`, [
      name,
    ]);
    // Clean up orphaned vertices/edges (no membership anywhere)
    this._conn.execute(`
      DELETE FROM "${this._tableName}_vertices"
      WHERE vertex_id NOT IN (
        SELECT DISTINCT vertex_id FROM "${this._tableName}_vertex_membership"
      )
    `);
    this._conn.execute(`
      DELETE FROM "${this._tableName}_edges"
      WHERE edge_id NOT IN (
        SELECT DISTINCT edge_id FROM "${this._tableName}_edge_membership"
      )
    `);
  }

  graphNames(): string[] {
    const rows = this._conn.query(
      `SELECT graph_name FROM "${this._tableName}_graphs" ORDER BY graph_name`,
    );
    return rows.map((r) => r["graph_name"] as string);
  }

  hasGraph(name: string): boolean {
    const row = this._conn.queryOne(
      `SELECT 1 AS found FROM "${this._tableName}_graphs" WHERE graph_name = ?`,
      [name],
    );
    return row !== null;
  }

  // -- Graph algebra ----------------------------------------------------------

  unionGraphs(g1: string, g2: string, target: string): void {
    this.createGraph(target);
    // Copy vertices from both graphs
    this._conn.execute(
      `
      INSERT OR IGNORE INTO "${this._tableName}_vertex_membership" (vertex_id, graph_name)
      SELECT vertex_id, ? FROM "${this._tableName}_vertex_membership"
      WHERE graph_name = ? OR graph_name = ?
    `,
      [target, g1, g2],
    );
    // Copy edges from both graphs
    this._conn.execute(
      `
      INSERT OR IGNORE INTO "${this._tableName}_edge_membership" (edge_id, graph_name)
      SELECT edge_id, ? FROM "${this._tableName}_edge_membership"
      WHERE graph_name = ? OR graph_name = ?
    `,
      [target, g1, g2],
    );
  }

  intersectGraphs(g1: string, g2: string, target: string): void {
    this.createGraph(target);
    // Vertices in both graphs
    this._conn.execute(
      `
      INSERT OR IGNORE INTO "${this._tableName}_vertex_membership" (vertex_id, graph_name)
      SELECT a.vertex_id, ? FROM "${this._tableName}_vertex_membership" a
      INNER JOIN "${this._tableName}_vertex_membership" b
        ON a.vertex_id = b.vertex_id
      WHERE a.graph_name = ? AND b.graph_name = ?
    `,
      [target, g1, g2],
    );
    // Edges in both graphs
    this._conn.execute(
      `
      INSERT OR IGNORE INTO "${this._tableName}_edge_membership" (edge_id, graph_name)
      SELECT a.edge_id, ? FROM "${this._tableName}_edge_membership" a
      INNER JOIN "${this._tableName}_edge_membership" b
        ON a.edge_id = b.edge_id
      WHERE a.graph_name = ? AND b.graph_name = ?
    `,
      [target, g1, g2],
    );
  }

  differenceGraphs(g1: string, g2: string, target: string): void {
    this.createGraph(target);
    // Vertices in g1 but not g2
    this._conn.execute(
      `
      INSERT OR IGNORE INTO "${this._tableName}_vertex_membership" (vertex_id, graph_name)
      SELECT vertex_id, ? FROM "${this._tableName}_vertex_membership"
      WHERE graph_name = ?
        AND vertex_id NOT IN (
          SELECT vertex_id FROM "${this._tableName}_vertex_membership" WHERE graph_name = ?
        )
    `,
      [target, g1, g2],
    );
    // Edges in g1 but not g2
    this._conn.execute(
      `
      INSERT OR IGNORE INTO "${this._tableName}_edge_membership" (edge_id, graph_name)
      SELECT edge_id, ? FROM "${this._tableName}_edge_membership"
      WHERE graph_name = ?
        AND edge_id NOT IN (
          SELECT edge_id FROM "${this._tableName}_edge_membership" WHERE graph_name = ?
        )
    `,
      [target, g1, g2],
    );
  }

  copyGraph(source: string, target: string): void {
    this.createGraph(target);
    this._conn.execute(
      `
      INSERT OR IGNORE INTO "${this._tableName}_vertex_membership" (vertex_id, graph_name)
      SELECT vertex_id, ? FROM "${this._tableName}_vertex_membership" WHERE graph_name = ?
    `,
      [target, source],
    );
    this._conn.execute(
      `
      INSERT OR IGNORE INTO "${this._tableName}_edge_membership" (edge_id, graph_name)
      SELECT edge_id, ? FROM "${this._tableName}_edge_membership" WHERE graph_name = ?
    `,
      [target, source],
    );
  }

  // -- Mutation ---------------------------------------------------------------

  addVertex(vertex: Vertex, graph: string): void {
    const propsJSON = JSON.stringify(vertex.properties);
    this._conn.execute(
      `INSERT OR REPLACE INTO "${this._tableName}_vertices"
       (vertex_id, label, properties_json) VALUES (?, ?, ?)`,
      [vertex.vertexId, vertex.label, propsJSON],
    );
    this._conn.execute(
      `INSERT OR IGNORE INTO "${this._tableName}_vertex_membership"
       (vertex_id, graph_name) VALUES (?, ?)`,
      [vertex.vertexId, graph],
    );
  }

  addEdge(edge: Edge, graph: string): void {
    const propsJSON = JSON.stringify(edge.properties);
    this._conn.execute(
      `INSERT OR REPLACE INTO "${this._tableName}_edges"
       (edge_id, source_id, target_id, label, properties_json)
       VALUES (?, ?, ?, ?, ?)`,
      [edge.edgeId, edge.sourceId, edge.targetId, edge.label, propsJSON],
    );
    this._conn.execute(
      `INSERT OR IGNORE INTO "${this._tableName}_edge_membership"
       (edge_id, graph_name) VALUES (?, ?)`,
      [edge.edgeId, graph],
    );
  }

  removeVertex(vertexId: number, graph: string): void {
    // Remove edges incident to this vertex in this graph
    const edgeRows = this._conn.query(
      `
      SELECT e.edge_id FROM "${this._tableName}_edges" e
      INNER JOIN "${this._tableName}_edge_membership" em
        ON e.edge_id = em.edge_id
      WHERE em.graph_name = ? AND (e.source_id = ? OR e.target_id = ?)
    `,
      [graph, vertexId, vertexId],
    );
    for (const row of edgeRows) {
      this.removeEdge(row["edge_id"] as number, graph);
    }

    // Remove vertex membership
    this._conn.execute(
      `DELETE FROM "${this._tableName}_vertex_membership" WHERE vertex_id = ? AND graph_name = ?`,
      [vertexId, graph],
    );

    // If vertex has no remaining memberships, delete it
    const remaining = this._conn.queryOne(
      `SELECT 1 AS found FROM "${this._tableName}_vertex_membership" WHERE vertex_id = ?`,
      [vertexId],
    );
    if (remaining === null) {
      this._conn.execute(
        `DELETE FROM "${this._tableName}_vertices" WHERE vertex_id = ?`,
        [vertexId],
      );
    }
  }

  removeEdge(edgeId: number, graph: string): void {
    this._conn.execute(
      `DELETE FROM "${this._tableName}_edge_membership" WHERE edge_id = ? AND graph_name = ?`,
      [edgeId, graph],
    );

    // If edge has no remaining memberships, delete it
    const remaining = this._conn.queryOne(
      `SELECT 1 AS found FROM "${this._tableName}_edge_membership" WHERE edge_id = ?`,
      [edgeId],
    );
    if (remaining === null) {
      this._conn.execute(`DELETE FROM "${this._tableName}_edges" WHERE edge_id = ?`, [
        edgeId,
      ]);
    }
  }

  // -- Query ------------------------------------------------------------------

  neighbors(
    vertexId: number,
    graph: string,
    label?: string | null,
    direction?: "out" | "in",
  ): number[] {
    const dir = direction ?? "out";
    const resolvedLabel = label ?? null;

    let sql: string;
    const params: unknown[] = [graph];

    if (dir === "out") {
      sql = `
        SELECT e.target_id AS neighbor
        FROM "${this._tableName}_edges" e
        INNER JOIN "${this._tableName}_edge_membership" em ON e.edge_id = em.edge_id
        WHERE em.graph_name = ? AND e.source_id = ?
      `;
      params.push(vertexId);
    } else {
      sql = `
        SELECT e.source_id AS neighbor
        FROM "${this._tableName}_edges" e
        INNER JOIN "${this._tableName}_edge_membership" em ON e.edge_id = em.edge_id
        WHERE em.graph_name = ? AND e.target_id = ?
      `;
      params.push(vertexId);
    }

    if (resolvedLabel !== null) {
      sql += " AND e.label = ?";
      params.push(resolvedLabel);
    }

    const rows = this._conn.query(sql, params);
    return rows.map((r) => r["neighbor"] as number);
  }

  verticesByLabel(label: string, graph: string): Vertex[] {
    const rows = this._conn.query(
      `
      SELECT v.vertex_id, v.label, v.properties_json
      FROM "${this._tableName}_vertices" v
      INNER JOIN "${this._tableName}_vertex_membership" vm ON v.vertex_id = vm.vertex_id
      WHERE vm.graph_name = ? AND v.label = ?
      ORDER BY v.vertex_id
    `,
      [graph, label],
    );
    return rows.map((r) => ({
      vertexId: r["vertex_id"] as number,
      label: r["label"] as string,
      properties: JSON.parse(r["properties_json"] as string) as Record<string, unknown>,
    }));
  }

  verticesInGraph(graph: string): Vertex[] {
    const rows = this._conn.query(
      `
      SELECT v.vertex_id, v.label, v.properties_json
      FROM "${this._tableName}_vertices" v
      INNER JOIN "${this._tableName}_vertex_membership" vm ON v.vertex_id = vm.vertex_id
      WHERE vm.graph_name = ?
      ORDER BY v.vertex_id
    `,
      [graph],
    );
    return rows.map((r) => ({
      vertexId: r["vertex_id"] as number,
      label: r["label"] as string,
      properties: JSON.parse(r["properties_json"] as string) as Record<string, unknown>,
    }));
  }

  edgesInGraph(graph: string): Edge[] {
    const rows = this._conn.query(
      `
      SELECT e.edge_id, e.source_id, e.target_id, e.label, e.properties_json
      FROM "${this._tableName}_edges" e
      INNER JOIN "${this._tableName}_edge_membership" em ON e.edge_id = em.edge_id
      WHERE em.graph_name = ?
      ORDER BY e.edge_id
    `,
      [graph],
    );
    return rows.map((r) => ({
      edgeId: r["edge_id"] as number,
      sourceId: r["source_id"] as number,
      targetId: r["target_id"] as number,
      label: r["label"] as string,
      properties: JSON.parse(r["properties_json"] as string) as Record<string, unknown>,
    }));
  }

  vertexGraphs(vertexId: number): Set<string> {
    const rows = this._conn.query(
      `SELECT graph_name FROM "${this._tableName}_vertex_membership" WHERE vertex_id = ?`,
      [vertexId],
    );
    return new Set(rows.map((r) => r["graph_name"] as string));
  }

  // -- Graph-scoped adjacency -------------------------------------------------

  outEdgeIds(vertexId: number, graph: string): Set<number> {
    const rows = this._conn.query(
      `
      SELECT e.edge_id
      FROM "${this._tableName}_edges" e
      INNER JOIN "${this._tableName}_edge_membership" em ON e.edge_id = em.edge_id
      WHERE em.graph_name = ? AND e.source_id = ?
    `,
      [graph, vertexId],
    );
    return new Set(rows.map((r) => r["edge_id"] as number));
  }

  inEdgeIds(vertexId: number, graph: string): Set<number> {
    const rows = this._conn.query(
      `
      SELECT e.edge_id
      FROM "${this._tableName}_edges" e
      INNER JOIN "${this._tableName}_edge_membership" em ON e.edge_id = em.edge_id
      WHERE em.graph_name = ? AND e.target_id = ?
    `,
      [graph, vertexId],
    );
    return new Set(rows.map((r) => r["edge_id"] as number));
  }

  edgeIdsByLabel(label: string, graph: string): Set<number> {
    const rows = this._conn.query(
      `
      SELECT e.edge_id
      FROM "${this._tableName}_edges" e
      INNER JOIN "${this._tableName}_edge_membership" em ON e.edge_id = em.edge_id
      WHERE em.graph_name = ? AND e.label = ?
    `,
      [graph, label],
    );
    return new Set(rows.map((r) => r["edge_id"] as number));
  }

  vertexIdsInGraph(graph: string): Set<number> {
    const rows = this._conn.query(
      `SELECT vertex_id FROM "${this._tableName}_vertex_membership" WHERE graph_name = ?`,
      [graph],
    );
    return new Set(rows.map((r) => r["vertex_id"] as number));
  }

  // -- Statistics -------------------------------------------------------------

  degreeDistribution(graph: string): Map<number, number> {
    // Compute in-app: get all vertex ids and count edges for each
    const vertexIds = this.vertexIdsInGraph(graph);
    const dist = new Map<number, number>();
    for (const vid of vertexIds) {
      const outCount = this.outEdgeIds(vid, graph).size;
      const inCount = this.inEdgeIds(vid, graph).size;
      const degree = outCount + inCount;
      dist.set(degree, (dist.get(degree) ?? 0) + 1);
    }
    return dist;
  }

  labelDegree(label: string, graph: string): number {
    return this.edgeIdsByLabel(label, graph).size;
  }

  vertexLabelCounts(graph: string): Map<string, number> {
    const rows = this._conn.query(
      `
      SELECT v.label, COUNT(*) AS cnt
      FROM "${this._tableName}_vertices" v
      INNER JOIN "${this._tableName}_vertex_membership" vm ON v.vertex_id = vm.vertex_id
      WHERE vm.graph_name = ?
      GROUP BY v.label
    `,
      [graph],
    );
    const counts = new Map<string, number>();
    for (const row of rows) {
      counts.set(row["label"] as string, row["cnt"] as number);
    }
    return counts;
  }

  // -- Global accessors -------------------------------------------------------

  getVertex(vertexId: number): Vertex | null {
    const row = this._conn.queryOne(
      `SELECT vertex_id, label, properties_json FROM "${this._tableName}_vertices" WHERE vertex_id = ?`,
      [vertexId],
    );
    if (row === null) return null;
    return {
      vertexId: row["vertex_id"] as number,
      label: row["label"] as string,
      properties: JSON.parse(row["properties_json"] as string) as Record<
        string,
        unknown
      >,
    };
  }

  getEdge(edgeId: number): Edge | null {
    const row = this._conn.queryOne(
      `SELECT edge_id, source_id, target_id, label, properties_json
       FROM "${this._tableName}_edges" WHERE edge_id = ?`,
      [edgeId],
    );
    if (row === null) return null;
    return {
      edgeId: row["edge_id"] as number,
      sourceId: row["source_id"] as number,
      targetId: row["target_id"] as number,
      label: row["label"] as string,
      properties: JSON.parse(row["properties_json"] as string) as Record<
        string,
        unknown
      >,
    };
  }

  nextVertexId(): number {
    const row = this._conn.queryOne(
      `SELECT COALESCE(MAX(vertex_id), -1) + 1 AS next_id FROM "${this._tableName}_vertices"`,
    );
    return row !== null ? (row["next_id"] as number) : 0;
  }

  nextEdgeId(): number {
    const row = this._conn.queryOne(
      `SELECT COALESCE(MAX(edge_id), -1) + 1 AS next_id FROM "${this._tableName}_edges"`,
    );
    return row !== null ? (row["next_id"] as number) : 0;
  }

  clear(): void {
    this._conn.execute(`DELETE FROM "${this._tableName}_edge_membership"`);
    this._conn.execute(`DELETE FROM "${this._tableName}_vertex_membership"`);
    this._conn.execute(`DELETE FROM "${this._tableName}_edges"`);
    this._conn.execute(`DELETE FROM "${this._tableName}_vertices"`);
    this._conn.execute(`DELETE FROM "${this._tableName}_graphs"`);
  }

  get vertices(): Map<number, Vertex> {
    const rows = this._conn.query(
      `SELECT vertex_id, label, properties_json FROM "${this._tableName}_vertices" ORDER BY vertex_id`,
    );
    const result = new Map<number, Vertex>();
    for (const row of rows) {
      const v: Vertex = {
        vertexId: row["vertex_id"] as number,
        label: row["label"] as string,
        properties: JSON.parse(row["properties_json"] as string) as Record<
          string,
          unknown
        >,
      };
      result.set(v.vertexId, v);
    }
    return result;
  }

  get edges(): Map<number, Edge> {
    const rows = this._conn.query(
      `SELECT edge_id, source_id, target_id, label, properties_json
       FROM "${this._tableName}_edges" ORDER BY edge_id`,
    );
    const result = new Map<number, Edge>();
    for (const row of rows) {
      const e: Edge = {
        edgeId: row["edge_id"] as number,
        sourceId: row["source_id"] as number,
        targetId: row["target_id"] as number,
        label: row["label"] as string,
        properties: JSON.parse(row["properties_json"] as string) as Record<
          string,
          unknown
        >,
      };
      result.set(e.edgeId, e);
    }
    return result;
  }

  get vertexCount(): number {
    const row = this._conn.queryOne(
      `SELECT COUNT(*) AS cnt FROM "${this._tableName}_vertices"`,
    );
    return row !== null ? (row["cnt"] as number) : 0;
  }

  get edgeCount(): number {
    const row = this._conn.queryOne(
      `SELECT COUNT(*) AS cnt FROM "${this._tableName}_edges"`,
    );
    return row !== null ? (row["cnt"] as number) : 0;
  }

  // -- Bulk mutation ----------------------------------------------------------

  addVertices(vertices: Vertex[], graph: string): void {
    const wasInTx = this._conn.inTransaction;
    if (!wasInTx) this._conn.beginTransaction();
    try {
      for (const v of vertices) this.addVertex(v, graph);
      if (!wasInTx) this._conn.commit();
    } catch (e) {
      if (!wasInTx) this._conn.rollback();
      throw e;
    }
  }

  addEdges(edges: Edge[], graph: string): void {
    const wasInTx = this._conn.inTransaction;
    if (!wasInTx) this._conn.beginTransaction();
    try {
      for (const e of edges) this.addEdge(e, graph);
      if (!wasInTx) this._conn.commit();
    } catch (e) {
      if (!wasInTx) this._conn.rollback();
      throw e;
    }
  }

  // -- Property access -------------------------------------------------------

  vertexProperty(vertexId: number, key: string): unknown {
    const v = this.getVertex(vertexId);
    if (v === null) return undefined;
    return v.properties[key];
  }

  setVertexProperty(vertexId: number, key: string, value: unknown): void {
    const v = this.getVertex(vertexId);
    if (v === null) throw new Error(`Vertex ${String(vertexId)} not found`);
    const props = { ...v.properties, [key]: value };
    this._conn.execute(
      `UPDATE "${this._tableName}_vertices" SET properties_json = ? WHERE vertex_id = ?`,
      [JSON.stringify(props), vertexId],
    );
  }

  edgeProperty(edgeId: number, key: string): unknown {
    const e = this.getEdge(edgeId);
    if (e === null) return undefined;
    return e.properties[key];
  }

  setEdgeProperty(edgeId: number, key: string, value: unknown): void {
    const e = this.getEdge(edgeId);
    if (e === null) throw new Error(`Edge ${String(edgeId)} not found`);
    const props = { ...e.properties, [key]: value };
    this._conn.execute(
      `UPDATE "${this._tableName}_edges" SET properties_json = ? WHERE edge_id = ?`,
      [JSON.stringify(props), edgeId],
    );
  }

  // -- Label queries ---------------------------------------------------------

  edgesByLabel(label: string, graph: string): Edge[] {
    const rows = this._conn.query(
      `SELECT e.edge_id, e.source_id, e.target_id, e.label, e.properties_json
       FROM "${this._tableName}_edges" e
       JOIN "${this._tableName}_edge_membership" m ON e.edge_id = m.edge_id
       WHERE e.label = ? AND m.graph_name = ?`,
      [label, graph],
    );
    return rows.map((row) => ({
      edgeId: row["edge_id"] as number,
      sourceId: row["source_id"] as number,
      targetId: row["target_id"] as number,
      label: row["label"] as string,
      properties: JSON.parse(row["properties_json"] as string) as Record<
        string,
        unknown
      >,
    }));
  }

  vertexLabels(graph: string): Set<string> {
    const rows = this._conn.query(
      `SELECT DISTINCT v.label FROM "${this._tableName}_vertices" v
       JOIN "${this._tableName}_vertex_membership" m ON v.vertex_id = m.vertex_id
       WHERE m.graph_name = ? AND v.label IS NOT NULL AND v.label != ''`,
      [graph],
    );
    return new Set(rows.map((r) => r["label"] as string));
  }

  edgeLabels(graph: string): Set<string> {
    const rows = this._conn.query(
      `SELECT DISTINCT e.label FROM "${this._tableName}_edges" e
       JOIN "${this._tableName}_edge_membership" m ON e.edge_id = m.edge_id
       WHERE m.graph_name = ?`,
      [graph],
    );
    return new Set(rows.map((r) => r["label"] as string));
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
    let sql = `SELECT e.edge_id, e.source_id, e.target_id, e.label, e.properties_json
       FROM "${this._tableName}_edges" e
       JOIN "${this._tableName}_edge_membership" m ON e.edge_id = m.edge_id
       WHERE e.source_id = ? AND e.target_id = ? AND m.graph_name = ?`;
    const params: unknown[] = [sourceId, targetId, graph];
    if (label !== null && label !== undefined) {
      sql += ` AND e.label = ?`;
      params.push(label);
    }
    const rows = this._conn.query(sql, params);
    return rows.map((row) => ({
      edgeId: row["edge_id"] as number,
      sourceId: row["source_id"] as number,
      targetId: row["target_id"] as number,
      label: row["label"] as string,
      properties: JSON.parse(row["properties_json"] as string) as Record<
        string,
        unknown
      >,
    }));
  }

  // -- Subgraph extraction ---------------------------------------------------

  subgraph(vertexIds: Set<number>, graph: string, target: string): void {
    if (!this.hasGraph(target)) this.createGraph(target);
    for (const vid of vertexIds) {
      const v = this.getVertex(vid);
      if (v) this.addVertex(v, target);
    }
    for (const e of this.edgesInGraph(graph)) {
      if (vertexIds.has(e.sourceId) && vertexIds.has(e.targetId)) {
        this.addEdge(e, target);
      }
    }
  }

  // -- Temporal support -------------------------------------------------------

  minTimestamp(graph: string): number | null {
    const edges = this.edgesInGraph(graph);
    let min: number | null = null;
    for (const e of edges) {
      const ts = e.properties["timestamp"];
      if (typeof ts === "number") {
        if (min === null || ts < min) min = ts;
      }
    }
    return min;
  }

  maxTimestamp(graph: string): number | null {
    const edges = this.edgesInGraph(graph);
    let max: number | null = null;
    for (const e of edges) {
      const ts = e.properties["timestamp"];
      if (typeof ts === "number") {
        if (max === null || ts > max) max = ts;
      }
    }
    return max;
  }

  edgesInTimeRange(graph: string, startTime: number, endTime: number): Edge[] {
    // Use JSON extraction for timestamp filtering in SQLite
    const allEdges = this.edgesInGraph(graph);
    return allEdges.filter((e) => {
      const ts = e.properties["timestamp"];
      return typeof ts === "number" && ts >= startTime && ts <= endTime;
    });
  }
}
