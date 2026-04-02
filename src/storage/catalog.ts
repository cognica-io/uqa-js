//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- SQLite-based metadata catalog
// 1:1 port of uqa/storage/catalog.py
//
// SQLite tables:
//   _metadata          -- key-value engine configuration
//   _catalog_tables    -- table schemas (name, columns JSON)
//   _documents         -- documents per table
//   _graph_vertices    -- graph vertices with properties
//   _graph_edges       -- graph edges with label and properties
//   _vectors           -- vector embeddings as binary blobs
//   _postings          -- inverted index posting entries
//   _doc_lengths       -- per-document per-field token lengths (BM25)
//   _column_stats      -- ANALYZE results for query optimizer
//   _scoring_params    -- Bayesian calibration parameters
//   _catalog_indexes   -- index definitions
//   _named_graphs      -- registered named graphs
//   _analyzers         -- named analyzer configurations
//   _foreign_servers   -- FDW server definitions
//   _foreign_tables    -- FDW table definitions
//   _table_field_analyzers -- field-to-analyzer mappings
//   _path_indexes      -- path index configurations
//   _models            -- trained model configurations

import type { ManagedConnection } from "./managed-connection.js";

export interface IndexDef {
  name: string;
  indexType: string;
  tableName: string;
  columns: string[];
  parameters: Record<string, unknown>;
}

export class Catalog {
  private _conn: ManagedConnection;
  private _inTransaction: boolean;

  constructor(conn: ManagedConnection) {
    this._conn = conn;
    this._inTransaction = false;
    this._initSchema();
  }

  get conn(): ManagedConnection {
    return this._conn;
  }

  // -- Schema bootstrap -------------------------------------------------------

  private _initSchema(): void {
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _metadata (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _catalog_tables (
        name         TEXT PRIMARY KEY,
        columns_json TEXT NOT NULL
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _documents (
        table_name TEXT    NOT NULL,
        doc_id     INTEGER NOT NULL,
        data_json  TEXT    NOT NULL,
        PRIMARY KEY (table_name, doc_id)
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _graph_vertices (
        vertex_id       INTEGER PRIMARY KEY,
        label           TEXT    NOT NULL DEFAULT '',
        properties_json TEXT    NOT NULL
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _graph_edges (
        edge_id         INTEGER PRIMARY KEY,
        source_id       INTEGER NOT NULL,
        target_id       INTEGER NOT NULL,
        label           TEXT    NOT NULL,
        properties_json TEXT    NOT NULL
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _vectors (
        doc_id     INTEGER PRIMARY KEY,
        dimensions INTEGER NOT NULL,
        embedding  BLOB    NOT NULL
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _postings (
        table_name TEXT    NOT NULL,
        field      TEXT    NOT NULL,
        term       TEXT    NOT NULL,
        doc_id     INTEGER NOT NULL,
        positions  TEXT    NOT NULL,
        PRIMARY KEY (table_name, field, term, doc_id)
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _doc_lengths (
        table_name TEXT NOT NULL,
        doc_id     INTEGER NOT NULL,
        lengths    TEXT NOT NULL,
        PRIMARY KEY (table_name, doc_id)
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _column_stats (
        table_name      TEXT    NOT NULL,
        column_name     TEXT    NOT NULL,
        distinct_count  INTEGER NOT NULL DEFAULT 0,
        null_count      INTEGER NOT NULL DEFAULT 0,
        min_value       TEXT,
        max_value       TEXT,
        row_count       INTEGER NOT NULL DEFAULT 0,
        histogram       TEXT    NOT NULL DEFAULT '[]',
        mcv_values      TEXT    NOT NULL DEFAULT '[]',
        mcv_frequencies TEXT    NOT NULL DEFAULT '[]',
        PRIMARY KEY (table_name, column_name)
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _scoring_params (
        name        TEXT PRIMARY KEY,
        params_json TEXT NOT NULL
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _catalog_indexes (
        name       TEXT PRIMARY KEY,
        index_type TEXT NOT NULL,
        table_name TEXT NOT NULL,
        columns    TEXT NOT NULL,
        parameters TEXT NOT NULL
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _named_graphs (
        name TEXT PRIMARY KEY
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _analyzers (
        name        TEXT PRIMARY KEY,
        config_json TEXT NOT NULL
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _foreign_servers (
        name     TEXT PRIMARY KEY,
        fdw_type TEXT NOT NULL,
        options  TEXT NOT NULL
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _foreign_tables (
        name         TEXT PRIMARY KEY,
        server_name  TEXT NOT NULL,
        columns_json TEXT NOT NULL,
        options      TEXT NOT NULL
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _table_field_analyzers (
        table_name    TEXT NOT NULL,
        field         TEXT NOT NULL,
        phase         TEXT NOT NULL,
        analyzer_name TEXT NOT NULL,
        PRIMARY KEY (table_name, field, phase)
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _path_indexes (
        graph_name       TEXT PRIMARY KEY,
        label_sequences  TEXT NOT NULL
      )
    `);
    this._conn.execute(`
      CREATE TABLE IF NOT EXISTS _models (
        model_name  TEXT PRIMARY KEY,
        config_json TEXT NOT NULL
      )
    `);
  }

  // -- Transaction management -------------------------------------------------

  begin(): void {
    this._inTransaction = true;
  }

  commit(): void {
    // The ManagedConnection handles actual SQLite transaction semantics
    this._inTransaction = false;
  }

  rollback(): void {
    this._inTransaction = false;
  }

  private _autoCommit(): void {
    if (!this._inTransaction) {
      // no-op for now; ManagedConnection auto-commits
    }
  }

  // -- Metadata ---------------------------------------------------------------

  setMetadata(key: string, value: string): void {
    this._conn.execute(`INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)`, [
      key,
      value,
    ]);
    this._autoCommit();
  }

  getMetadata(key: string): string | null {
    const row = this._conn.queryOne(`SELECT value FROM _metadata WHERE key = ?`, [key]);
    return row !== null ? (row["value"] as string) : null;
  }

  // -- Table schemas ----------------------------------------------------------

  saveTableSchema(name: string, columns: Record<string, unknown>[]): void {
    this._conn.execute(
      `INSERT OR REPLACE INTO _catalog_tables (name, columns_json) VALUES (?, ?)`,
      [name, JSON.stringify(columns)],
    );
    this._autoCommit();
  }

  dropTableSchema(name: string): void {
    this._conn.execute(`DELETE FROM _catalog_tables WHERE name = ?`, [name]);

    // Drop per-table SQLite tables (new format)
    this._conn.execute(`DROP TABLE IF EXISTS "_data_${name}"`);
    this._conn.execute(`DROP TABLE IF EXISTS "_field_stats_${name}"`);
    this._conn.execute(`DROP TABLE IF EXISTS "_doc_lengths_${name}"`);
    this._conn.execute(`DROP TABLE IF EXISTS "_graph_vertices_${name}"`);
    this._conn.execute(`DROP TABLE IF EXISTS "_graph_edges_${name}"`);

    // Drop all per-field inverted, skip, block-max, and IVF tables
    const prefixes = [
      `_inverted_${name}_`,
      `_skip_${name}_`,
      `_blockmax_${name}_`,
      `_ivf_centroids_${name}_`,
      `_ivf_lists_${name}_`,
    ];
    for (const prefix of prefixes) {
      const rows = this._conn.query(
        `SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?`,
        [prefix + "%"],
      );
      for (const row of rows) {
        const tblName = row["name"] as string;
        this._conn.execute(`DROP TABLE IF EXISTS "${tblName}"`);
      }
    }

    // Drop index catalog entries for this table
    this._conn.execute(`DELETE FROM _catalog_indexes WHERE table_name = ?`, [name]);

    // Clean shared catalog tables
    this._conn.execute(`DELETE FROM _documents WHERE table_name = ?`, [name]);
    this._conn.execute(`DELETE FROM _postings WHERE table_name = ?`, [name]);
    this._conn.execute(`DELETE FROM _doc_lengths WHERE table_name = ?`, [name]);
    this._conn.execute(`DELETE FROM _column_stats WHERE table_name = ?`, [name]);
    this._conn.execute(`DELETE FROM _table_field_analyzers WHERE table_name = ?`, [
      name,
    ]);
    this._autoCommit();
  }

  loadTableSchemas(): [string, Record<string, unknown>[]][] {
    const rows = this._conn.query(`SELECT name, columns_json FROM _catalog_tables`);
    return rows.map((row) => [
      row["name"] as string,
      JSON.parse(row["columns_json"] as string) as Record<string, unknown>[],
    ]);
  }

  // -- Documents --------------------------------------------------------------

  saveDocument(tableName: string, docId: number, data: Record<string, unknown>): void {
    this._conn.execute(
      `INSERT OR REPLACE INTO _documents (table_name, doc_id, data_json) VALUES (?, ?, ?)`,
      [tableName, docId, JSON.stringify(data)],
    );
    this._autoCommit();
  }

  deleteDocument(tableName: string, docId: number): void {
    this._conn.execute(`DELETE FROM _documents WHERE table_name = ? AND doc_id = ?`, [
      tableName,
      docId,
    ]);
    this._conn.execute(`DELETE FROM _postings WHERE table_name = ? AND doc_id = ?`, [
      tableName,
      docId,
    ]);
    this._conn.execute(`DELETE FROM _doc_lengths WHERE table_name = ? AND doc_id = ?`, [
      tableName,
      docId,
    ]);
    this._autoCommit();
  }

  loadDocuments(tableName: string): [number, Record<string, unknown>][] {
    const rows = this._conn.query(
      `SELECT doc_id, data_json FROM _documents WHERE table_name = ?`,
      [tableName],
    );
    return rows.map((row) => [
      row["doc_id"] as number,
      JSON.parse(row["data_json"] as string) as Record<string, unknown>,
    ]);
  }

  // -- Postings (inverted index entries) --------------------------------------

  savePostings(
    tableName: string,
    docId: number,
    fieldLengths: Record<string, number>,
    postings: Map<string, number[]>,
  ): void {
    for (const [key, positions] of postings) {
      const [field, term] = key.split("\0");
      this._conn.execute(
        `INSERT OR REPLACE INTO _postings (table_name, field, term, doc_id, positions) VALUES (?, ?, ?, ?, ?)`,
        [tableName, field, term, docId, JSON.stringify(positions)],
      );
    }
    this._conn.execute(
      `INSERT OR REPLACE INTO _doc_lengths (table_name, doc_id, lengths) VALUES (?, ?, ?)`,
      [tableName, docId, JSON.stringify(fieldLengths)],
    );
    this._autoCommit();
  }

  deletePostings(tableName: string, docId: number): void {
    this._conn.execute(`DELETE FROM _postings WHERE table_name = ? AND doc_id = ?`, [
      tableName,
      docId,
    ]);
    this._conn.execute(`DELETE FROM _doc_lengths WHERE table_name = ? AND doc_id = ?`, [
      tableName,
      docId,
    ]);
    this._autoCommit();
  }

  loadPostings(tableName: string): [string, string, number, number[]][] {
    const rows = this._conn.query(
      `SELECT field, term, doc_id, positions FROM _postings WHERE table_name = ?`,
      [tableName],
    );
    return rows.map((row) => [
      row["field"] as string,
      row["term"] as string,
      row["doc_id"] as number,
      JSON.parse(row["positions"] as string) as number[],
    ]);
  }

  loadDocLengths(tableName: string): [number, Record<string, number>][] {
    const rows = this._conn.query(
      `SELECT doc_id, lengths FROM _doc_lengths WHERE table_name = ?`,
      [tableName],
    );
    return rows.map((row) => [
      row["doc_id"] as number,
      JSON.parse(row["lengths"] as string) as Record<string, number>,
    ]);
  }

  // -- Graph vertices ---------------------------------------------------------

  saveVertex(vertexId: number, properties: Record<string, unknown>): void {
    this._conn.execute(
      `INSERT OR REPLACE INTO _graph_vertices (vertex_id, properties_json) VALUES (?, ?)`,
      [vertexId, JSON.stringify(properties)],
    );
    this._autoCommit();
  }

  saveEdge(
    edgeId: number,
    sourceId: number,
    targetId: number,
    label: string,
    properties: Record<string, unknown>,
  ): void {
    this._conn.execute(
      `INSERT OR REPLACE INTO _graph_edges (edge_id, source_id, target_id, label, properties_json) VALUES (?, ?, ?, ?, ?)`,
      [edgeId, sourceId, targetId, label, JSON.stringify(properties)],
    );
    this._autoCommit();
  }

  loadVertices(): [number, Record<string, unknown>][] {
    const rows = this._conn.query(
      `SELECT vertex_id, properties_json FROM _graph_vertices`,
    );
    return rows.map((row) => [
      row["vertex_id"] as number,
      JSON.parse(row["properties_json"] as string) as Record<string, unknown>,
    ]);
  }

  loadEdges(): [number, number, number, string, Record<string, unknown>][] {
    const rows = this._conn.query(
      `SELECT edge_id, source_id, target_id, label, properties_json FROM _graph_edges`,
    );
    return rows.map((row) => [
      row["edge_id"] as number,
      row["source_id"] as number,
      row["target_id"] as number,
      row["label"] as string,
      JSON.parse(row["properties_json"] as string) as Record<string, unknown>,
    ]);
  }

  // -- Vectors ----------------------------------------------------------------

  saveVector(docId: number, embedding: Float64Array): void {
    // Convert to binary blob (Float32 for space efficiency)
    const f32 = new Float32Array(embedding);
    const blob = new Uint8Array(f32.buffer);
    this._conn.execute(
      `INSERT OR REPLACE INTO _vectors (doc_id, dimensions, embedding) VALUES (?, ?, ?)`,
      [docId, embedding.length, blob],
    );
    this._autoCommit();
  }

  deleteVector(docId: number): void {
    this._conn.execute(`DELETE FROM _vectors WHERE doc_id = ?`, [docId]);
    this._autoCommit();
  }

  loadVectors(): [number, Float64Array][] {
    const rows = this._conn.query(`SELECT doc_id, dimensions, embedding FROM _vectors`);
    return rows.map((row) => {
      const blob = row["embedding"] as Uint8Array;
      const f32 = new Float32Array(blob.buffer, blob.byteOffset, blob.byteLength / 4);
      return [row["doc_id"] as number, new Float64Array(f32)];
    });
  }

  // -- Column statistics (ANALYZE results) ------------------------------------

  saveColumnStats(
    tableName: string,
    columnName: string,
    distinctCount: number,
    nullCount: number,
    minValue: unknown,
    maxValue: unknown,
    rowCount: number,
    histogram?: unknown[] | null,
    mcvValues?: unknown[] | null,
    mcvFrequencies?: number[] | null,
  ): void {
    this._conn.execute(
      `INSERT OR REPLACE INTO _column_stats
       (table_name, column_name, distinct_count, null_count,
        min_value, max_value, row_count,
        histogram, mcv_values, mcv_frequencies)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        tableName,
        columnName,
        distinctCount,
        nullCount,
        JSON.stringify(minValue),
        JSON.stringify(maxValue),
        rowCount,
        JSON.stringify(histogram ?? []),
        JSON.stringify(mcvValues ?? []),
        JSON.stringify(mcvFrequencies ?? []),
      ],
    );
    this._autoCommit();
  }

  loadColumnStats(
    tableName: string,
  ): [
    string,
    number,
    number,
    unknown,
    unknown,
    number,
    unknown[],
    unknown[],
    number[],
  ][] {
    const rows = this._conn.query(
      `SELECT column_name, distinct_count, null_count,
              min_value, max_value, row_count,
              histogram, mcv_values, mcv_frequencies
       FROM _column_stats WHERE table_name = ?`,
      [tableName],
    );
    return rows.map((row) => [
      row["column_name"] as string,
      row["distinct_count"] as number,
      row["null_count"] as number,
      JSON.parse(row["min_value"] as string),
      JSON.parse(row["max_value"] as string),
      row["row_count"] as number,
      JSON.parse(row["histogram"] as string) as unknown[],
      JSON.parse(row["mcv_values"] as string) as unknown[],
      JSON.parse(row["mcv_frequencies"] as string) as number[],
    ]);
  }

  deleteColumnStats(tableName: string): void {
    this._conn.execute(`DELETE FROM _column_stats WHERE table_name = ?`, [tableName]);
    this._autoCommit();
  }

  // -- Scoring / calibration parameters ---------------------------------------

  saveScoringParams(name: string, params: Record<string, unknown>): void {
    this._conn.execute(
      `INSERT OR REPLACE INTO _scoring_params (name, params_json) VALUES (?, ?)`,
      [name, JSON.stringify(params)],
    );
    this._autoCommit();
  }

  loadScoringParams(name: string): Record<string, unknown> | null {
    const row = this._conn.queryOne(
      `SELECT params_json FROM _scoring_params WHERE name = ?`,
      [name],
    );
    return row !== null
      ? (JSON.parse(row["params_json"] as string) as Record<string, unknown>)
      : null;
  }

  loadAllScoringParams(): [string, Record<string, unknown>][] {
    const rows = this._conn.query(`SELECT name, params_json FROM _scoring_params`);
    return rows.map((row) => [
      row["name"] as string,
      JSON.parse(row["params_json"] as string) as Record<string, unknown>,
    ]);
  }

  deleteScoringParams(name: string): void {
    this._conn.execute(`DELETE FROM _scoring_params WHERE name = ?`, [name]);
    this._autoCommit();
  }

  // -- Indexes ----------------------------------------------------------------

  saveIndex(indexDef: IndexDef): void {
    this._conn.execute(
      `INSERT OR REPLACE INTO _catalog_indexes
       (name, index_type, table_name, columns, parameters)
       VALUES (?, ?, ?, ?, ?)`,
      [
        indexDef.name,
        indexDef.indexType,
        indexDef.tableName,
        JSON.stringify(indexDef.columns),
        JSON.stringify(indexDef.parameters),
      ],
    );
    this._autoCommit();
  }

  dropIndex(name: string): void {
    this._conn.execute(`DELETE FROM _catalog_indexes WHERE name = ?`, [name]);
    this._autoCommit();
  }

  loadIndexes(): [string, string, string, string[], Record<string, unknown>][] {
    const rows = this._conn.query(
      `SELECT name, index_type, table_name, columns, parameters FROM _catalog_indexes`,
    );
    return rows.map((row) => [
      row["name"] as string,
      row["index_type"] as string,
      row["table_name"] as string,
      JSON.parse(row["columns"] as string) as string[],
      JSON.parse(row["parameters"] as string) as Record<string, unknown>,
    ]);
  }

  loadIndexesForTable(
    tableName: string,
  ): [string, string, string, string[], Record<string, unknown>][] {
    const rows = this._conn.query(
      `SELECT name, index_type, table_name, columns, parameters
       FROM _catalog_indexes WHERE table_name = ?`,
      [tableName],
    );
    return rows.map((row) => [
      row["name"] as string,
      row["index_type"] as string,
      row["table_name"] as string,
      JSON.parse(row["columns"] as string) as string[],
      JSON.parse(row["parameters"] as string) as Record<string, unknown>,
    ]);
  }

  // -- Named graphs -----------------------------------------------------------

  saveNamedGraph(name: string): void {
    this._conn.execute(`INSERT OR IGNORE INTO _named_graphs (name) VALUES (?)`, [name]);
    this._autoCommit();
  }

  dropNamedGraph(name: string): void {
    this._conn.execute(`DELETE FROM _named_graphs WHERE name = ?`, [name]);
    this._autoCommit();
  }

  loadNamedGraphs(): string[] {
    const rows = this._conn.query(`SELECT name FROM _named_graphs`);
    return rows.map((row) => row["name"] as string);
  }

  // -- Path indexes -----------------------------------------------------------

  savePathIndex(graphName: string, labelSequences: string[][]): void {
    this._conn.execute(
      `INSERT OR REPLACE INTO _path_indexes (graph_name, label_sequences) VALUES (?, ?)`,
      [graphName, JSON.stringify(labelSequences)],
    );
    this._autoCommit();
  }

  loadPathIndexes(): [string, string[][]][] {
    try {
      const rows = this._conn.query(
        `SELECT graph_name, label_sequences FROM _path_indexes`,
      );
      return rows.map((row) => [
        row["graph_name"] as string,
        JSON.parse(row["label_sequences"] as string) as string[][],
      ]);
    } catch {
      return [];
    }
  }

  dropPathIndex(graphName: string): void {
    this._conn.execute(`DELETE FROM _path_indexes WHERE graph_name = ?`, [graphName]);
    this._autoCommit();
  }

  // -- Models (deep_learn) ----------------------------------------------------

  saveModel(modelName: string, config: Record<string, unknown>): void {
    this._conn.execute(
      `INSERT OR REPLACE INTO _models (model_name, config_json) VALUES (?, ?)`,
      [modelName, JSON.stringify(config)],
    );
    this._autoCommit();
  }

  loadModel(modelName: string): Record<string, unknown> | null {
    const row = this._conn.queryOne(
      `SELECT config_json FROM _models WHERE model_name = ?`,
      [modelName],
    );
    return row !== null
      ? (JSON.parse(row["config_json"] as string) as Record<string, unknown>)
      : null;
  }

  deleteModel(modelName: string): void {
    this._conn.execute(`DELETE FROM _models WHERE model_name = ?`, [modelName]);
    this._autoCommit();
  }

  // -- Analyzers --------------------------------------------------------------

  saveAnalyzer(name: string, config: Record<string, unknown>): void {
    this._conn.execute(
      `INSERT OR REPLACE INTO _analyzers (name, config_json) VALUES (?, ?)`,
      [name, JSON.stringify(config)],
    );
    this._autoCommit();
  }

  dropAnalyzer(name: string): void {
    this._conn.execute(`DELETE FROM _analyzers WHERE name = ?`, [name]);
    this._autoCommit();
  }

  loadAnalyzers(): [string, Record<string, unknown>][] {
    const rows = this._conn.query(`SELECT name, config_json FROM _analyzers`);
    return rows.map((row) => [
      row["name"] as string,
      JSON.parse(row["config_json"] as string) as Record<string, unknown>,
    ]);
  }

  // -- Table field analyzers --------------------------------------------------

  saveTableFieldAnalyzer(
    tableName: string,
    field: string,
    phase: string,
    analyzerName: string,
  ): void {
    this._conn.execute(
      `INSERT OR REPLACE INTO _table_field_analyzers
       (table_name, field, phase, analyzer_name) VALUES (?, ?, ?, ?)`,
      [tableName, field, phase, analyzerName],
    );
    this._autoCommit();
  }

  loadTableFieldAnalyzers(): [string, string, string, string][] {
    const rows = this._conn.query(
      `SELECT table_name, field, phase, analyzer_name FROM _table_field_analyzers`,
    );
    return rows.map((row) => [
      row["table_name"] as string,
      row["field"] as string,
      row["phase"] as string,
      row["analyzer_name"] as string,
    ]);
  }

  dropTableFieldAnalyzers(tableName: string): void {
    this._conn.execute(`DELETE FROM _table_field_analyzers WHERE table_name = ?`, [
      tableName,
    ]);
    this._autoCommit();
  }

  // -- Foreign servers --------------------------------------------------------

  saveForeignServer(
    name: string,
    fdwType: string,
    options: Record<string, string>,
  ): void {
    this._conn.execute(
      `INSERT INTO _foreign_servers (name, fdw_type, options) VALUES (?, ?, ?)`,
      [name, fdwType, JSON.stringify(options)],
    );
    this._autoCommit();
  }

  dropForeignServer(name: string): void {
    this._conn.execute(`DELETE FROM _foreign_servers WHERE name = ?`, [name]);
    this._autoCommit();
  }

  loadForeignServers(): [string, string, Record<string, string>][] {
    const rows = this._conn.query(
      `SELECT name, fdw_type, options FROM _foreign_servers`,
    );
    return rows.map((row) => [
      row["name"] as string,
      row["fdw_type"] as string,
      JSON.parse(row["options"] as string) as Record<string, string>,
    ]);
  }

  // -- Foreign tables ---------------------------------------------------------

  saveForeignTable(
    name: string,
    serverName: string,
    columnsJSON: Record<string, unknown>[],
    options: Record<string, string>,
  ): void {
    this._conn.execute(
      `INSERT INTO _foreign_tables (name, server_name, columns_json, options) VALUES (?, ?, ?, ?)`,
      [name, serverName, JSON.stringify(columnsJSON), JSON.stringify(options)],
    );
    this._autoCommit();
  }

  dropForeignTable(name: string): void {
    this._conn.execute(`DELETE FROM _foreign_tables WHERE name = ?`, [name]);
    this._autoCommit();
  }

  loadForeignTables(): [
    string,
    string,
    Record<string, unknown>[],
    Record<string, string>,
  ][] {
    const rows = this._conn.query(
      `SELECT name, server_name, columns_json, options FROM _foreign_tables`,
    );
    return rows.map((row) => [
      row["name"] as string,
      row["server_name"] as string,
      JSON.parse(row["columns_json"] as string) as Record<string, unknown>[],
      JSON.parse(row["options"] as string) as Record<string, string>,
    ]);
  }

  // -- Lifecycle --------------------------------------------------------------

  close(): void {
    // Checkpoint WAL to ensure all committed data is flushed to
    // the main database file before closing.
    try {
      this._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)");
    } catch {
      // Ignore errors (in-memory databases do not support WAL).
    }
  }
}
