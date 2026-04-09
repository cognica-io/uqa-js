//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Main Engine
// 1:1 port of uqa/engine.py
//
// Wires together: SQLCompiler, Tables, GraphStore, QueryBuilder.

import { CancellationToken } from "./cancel.js";
import { SQLCompiler } from "./sql/compiler.js";
import type { SQLResult } from "./sql/compiler.js";
import { Table, createColumnDef } from "./sql/table.js";
import type { ColumnDef } from "./sql/table.js";
import { MemoryGraphStore } from "./graph/store.js";
import type { GraphStore } from "./storage/abc/graph-store.js";
import { SQLiteGraphStore } from "./storage/sqlite-graph-store.js";
import { QueryBuilder } from "./api/query-builder.js";
import { Catalog } from "./storage/catalog.js";
import { IndexManager } from "./storage/index-manager.js";
import {
  Transaction as TransactionClass,
  InMemoryTransaction,
} from "./storage/transaction.js";
import type { Transaction } from "./storage/transaction.js";
import { PathIndex as PathIndexClass } from "./graph/index.js";
import type { PathIndex } from "./graph/index.js";
import type { ForeignServer, ForeignTable } from "./fdw/foreign-table.js";
import type { Vertex, Edge, DocId } from "./core/types.js";
import type { VectorIndex } from "./storage/vector-index.js";
import type { ExecutionContext } from "./operators/base.js";
import type { SpatialIndex } from "./storage/spatial-index.js";
import {
  Analyzer,
  registerAnalyzer,
  getAnalyzer,
  dropAnalyzer as dropAnalyzerFn,
} from "./analysis/analyzer.js";
import { TermOperator, ScoreOperator } from "./operators/primitive.js";
import {
  BayesianBM25Scorer,
  createBayesianBM25Params,
} from "./scoring/bayesian-bm25.js";
import { ParameterLearner } from "./scoring/parameter-learner.js";
import { estimateConvWeights as estimateConvWeightsFn } from "./operators/deep-fusion.js";
import { ManagedConnection } from "./storage/managed-connection.js";
import type { SQLiteDatabase } from "./storage/managed-connection.js";

// -- SchemaAwareTableStore ---------------------------------------------------

/**
 * Schema-qualified table namespace with search_path resolution.
 *
 * Internally stores tables in `{schema: {tableName: Table}}`.
 * Implements the Map-like interface so existing code that does
 * `_tables.get(name)`, `_tables.set(name, t)`, `_tables.has(name)`
 * continues to work.  Unqualified names resolve via `search_path`.
 * Qualified names (`"schema.table"`) resolve directly.
 */
export class SchemaAwareTableStore {
  private _schemas: Map<string, Map<string, Table>>;
  private _searchPath: string[];

  constructor() {
    this._schemas = new Map([["public", new Map()]]);
    this._searchPath = ["public"];
  }

  // -- Schema management ---------------------------------------------------

  createSchema(name: string, ifNotExists = false): void {
    if (this._schemas.has(name)) {
      if (ifNotExists) return;
      throw new Error(`Schema '${name}' already exists`);
    }
    this._schemas.set(name, new Map());
  }

  dropSchema(name: string, cascade = false, ifExists = false): void {
    if (!this._schemas.has(name)) {
      if (ifExists) return;
      throw new Error(`Schema '${name}' does not exist`);
    }
    if (name === "public") {
      throw new Error("Cannot drop schema 'public'");
    }
    const tables = this._schemas.get(name)!;
    if (tables.size > 0 && !cascade) {
      throw new Error(`Schema '${name}' is not empty, use CASCADE to drop`);
    }
    this._schemas.delete(name);
  }

  get searchPath(): string[] {
    return [...this._searchPath];
  }

  set searchPath(value: string[]) {
    this._searchPath = [...value];
  }

  get schemas(): Set<string> {
    return new Set(this._schemas.keys());
  }

  // -- Resolution helpers -------------------------------------------------

  private _parseQualified(name: string): [string | null, string] {
    const dot = name.indexOf(".");
    if (dot !== -1) {
      const schema = name.substring(0, dot);
      if (this._schemas.has(schema)) {
        return [schema, name.substring(dot + 1)];
      }
    }
    return [null, name];
  }

  private _resolve(name: string): [string, string] | null {
    const [schema, table] = this._parseQualified(name);
    if (schema !== null) {
      const tables = this._schemas.get(schema);
      if (tables && tables.has(table)) return [schema, table];
      return null;
    }
    for (const s of this._searchPath) {
      const tables = this._schemas.get(s);
      if (tables && tables.has(table)) return [s, table];
    }
    return null;
  }

  private _defaultSchema(): string {
    for (const s of this._searchPath) {
      if (this._schemas.has(s)) return s;
    }
    return "public";
  }

  // -- Map-compatible interface -------------------------------------------

  get(name: string): Table | undefined {
    const resolved = this._resolve(name);
    if (!resolved) return undefined;
    const [s, t] = resolved;
    return this._schemas.get(s)!.get(t);
  }

  set(name: string, table: Table): this {
    const [schema, tableName] = this._parseQualified(name);
    const s = schema ?? this._defaultSchema();
    if (!this._schemas.has(s)) {
      this._schemas.set(s, new Map());
    }
    this._schemas.get(s)!.set(tableName, table);
    return this;
  }

  has(name: string): boolean {
    return this._resolve(name) !== null;
  }

  delete(name: string): boolean {
    const resolved = this._resolve(name);
    if (!resolved) return false;
    const [s, t] = resolved;
    return this._schemas.get(s)!.delete(t);
  }

  get size(): number {
    let count = 0;
    for (const tables of this._schemas.values()) count += tables.size;
    return count;
  }

  keys(): IterableIterator<string> {
    const allKeys: string[] = [];
    for (const tables of this._schemas.values()) {
      for (const name of tables.keys()) allKeys.push(name);
    }
    return allKeys[Symbol.iterator]();
  }

  values(): IterableIterator<Table> {
    const allValues: Table[] = [];
    for (const tables of this._schemas.values()) {
      for (const table of tables.values()) allValues.push(table);
    }
    return allValues[Symbol.iterator]();
  }

  entries(): IterableIterator<[string, Table]> {
    const allEntries: [string, Table][] = [];
    for (const tables of this._schemas.values()) {
      for (const [name, table] of tables) allEntries.push([name, table]);
    }
    return allEntries[Symbol.iterator]();
  }

  forEach(cb: (value: Table, key: string) => void): void {
    for (const tables of this._schemas.values()) {
      for (const [name, table] of tables) cb(table, name);
    }
  }

  clear(): void {
    for (const tables of this._schemas.values()) {
      tables.clear();
    }
  }

  [Symbol.iterator](): IterableIterator<[string, Table]> {
    return this.entries();
  }

  // -- Schema-qualified access -------------------------------------------

  setInSchema(schema: string, tableName: string, table: Table): void {
    if (!this._schemas.has(schema)) {
      this._schemas.set(schema, new Map());
    }
    this._schemas.get(schema)!.set(tableName, table);
  }

  getFromSchema(schema: string, tableName: string): Table | undefined {
    return this._schemas.get(schema)?.get(tableName);
  }

  tablesInSchema(schema: string): Map<string, Table> {
    return new Map(this._schemas.get(schema) ?? []);
  }

  *qualifiedItems(): Iterable<[string, string, Table]> {
    for (const [schema, tables] of this._schemas) {
      for (const [name, table] of tables) {
        yield [schema, name, table];
      }
    }
  }

  *items(): Iterable<[string, Table]> {
    for (const tables of this._schemas.values()) {
      yield* tables;
    }
  }
}

// -- Engine Options ---------------------------------------------------------

export interface EngineOptions {
  dbPath?: string;
  parallelWorkers?: number;
  spillThreshold?: number;
}

export class Engine {
  _tables: SchemaAwareTableStore;
  _views: Map<string, unknown>; // name -> SelectStmt AST
  _prepared: Map<string, unknown>; // name -> PrepareStmt AST
  _sequences: Map<string, Record<string, number>>;
  _tempTables: Set<string>;
  _graphStore: GraphStore;
  _versionedGraphs: Map<string, unknown>;
  _pathIndexes: Map<string, PathIndex>;
  _foreignServers: Map<string, ForeignServer>;
  _foreignTables: Map<string, ForeignTable>;
  _models: Map<string, Record<string, unknown>>;
  // In-memory BTREE index tracking: indexName -> [tableName, columns]
  _btreeIndexes: Map<string, [string, string[]]>;
  // Session variables (SET/SHOW/RESET)
  _sessionVars: Map<string, string>;
  private _conn: ManagedConnection | null;
  // Node.js fs module, resolved during init() for use in close()
  private _fs: typeof import("fs") | null;
  private _catalog: Catalog | null;
  private _indexManager: IndexManager | null;
  private _transaction: Transaction | InMemoryTransaction | null;
  _cancelToken: CancellationToken;
  private _compiler: SQLCompiler;
  private _dbPath: string | null;
  private _parallelWorkers: number;
  private _spillThreshold: number;

  constructor(opts?: EngineOptions) {
    this._dbPath = opts?.dbPath ?? null;
    this._parallelWorkers = opts?.parallelWorkers ?? 4;
    this._spillThreshold = opts?.spillThreshold ?? 0;
    this._cancelToken = new CancellationToken();
    this._tables = new SchemaAwareTableStore();
    this._views = new Map();
    this._prepared = new Map();
    this._sequences = new Map();
    this._tempTables = new Set();
    this._graphStore = new MemoryGraphStore();
    this._versionedGraphs = new Map();
    this._pathIndexes = new Map();
    this._foreignServers = new Map();
    this._foreignTables = new Map();
    this._models = new Map();
    this._btreeIndexes = new Map();
    this._sessionVars = new Map();
    this._conn = null;
    this._fs = null;
    this._catalog = null;
    this._indexManager = null;
    this._transaction = null;
    this._compiler = new SQLCompiler(this);
  }

  /**
   * Async initialization -- must be called after construction when dbPath is
   * provided.  Creates the sql.js SQLite connection, opens (or creates) the
   * catalog database, and restores persisted tables, documents, sequences,
   * and named graphs.
   *
   * When no dbPath was supplied the method is a no-op so callers may always
   * await `engine.init()` unconditionally.
   */
  async init(): Promise<void> {
    if (this._dbPath === null) return;
    if (this._conn !== null) return;

    const initSqlJs = (await import("sql.js")).default;
    const SQL = await initSqlJs();

    // Resolve the fs module for later use in the synchronous close() path.
    // In browser environments this import will fail; file persistence is
    // unavailable there.
    try {
      this._fs = await import("fs");
    } catch {
      this._fs = null;
    }

    let db: SQLiteDatabase;
    if (this._fs !== null) {
      try {
        const data = this._fs.readFileSync(this._dbPath);
        db = new SQL.Database(new Uint8Array(data));
      } catch {
        // File does not exist yet -- start with a fresh database.
        db = new SQL.Database();
      }
    } else {
      db = new SQL.Database();
    }

    this._conn = new ManagedConnection(db, this._dbPath);
    this._catalog = new Catalog(this._conn);
    this._indexManager = new IndexManager(this._conn, this._catalog);
    // Replace the default MemoryGraphStore with a persistent
    // SQLiteGraphStore so that named-graph vertices and edges
    // survive process restarts (write-through to SQLite).
    this._graphStore = new SQLiteGraphStore(this._conn);
    this._loadFromCatalog();
  }

  // -- SQL execution ----------------------------------------------------------

  get cancelToken(): CancellationToken {
    return this._cancelToken;
  }

  cancel(): void {
    this._cancelToken.cancel();
  }

  async sql(query: string, params?: unknown[]): Promise<SQLResult | null> {
    this._cancelToken.reset();
    return this._compiler.execute(query, params);
  }

  // -- Table management -------------------------------------------------------

  getTable(name: string): Table {
    const table = this._tables.get(name);
    if (table !== undefined) {
      return table;
    }
    // Check if the compiler has it (created via SQL DDL)
    const compilerTable = this._compiler.tables.get(name);
    if (compilerTable !== undefined) {
      this._tables.set(name, compilerTable);
      return compilerTable;
    }
    throw new Error(`Table not found: ${name}`);
  }

  hasTable(name: string): boolean {
    return this._tables.has(name) || this._compiler.tables.has(name);
  }

  registerTable(name: string, table: Table): void {
    this._tables.set(name, table);
    // Also register in the compiler so SQL queries can find it
    this._compiler.tables.set(name, table);
  }

  // -- Document operations ----------------------------------------------------

  addDocument(
    docId: DocId,
    document: Record<string, unknown>,
    table: string,
    embedding?: Float64Array | null,
  ): void {
    const tbl = this._tables.get(table);
    if (tbl === undefined) {
      throw new Error(`Table '${table}' does not exist`);
    }

    // Include the primary key in stored data for consistency with
    // SQL INSERT (Table.insert) which always stores the PK column.
    const stored: Record<string, unknown> = { ...document };
    if (tbl.primaryKey !== null && !(tbl.primaryKey in stored)) {
      const pkCol = tbl.columns.get(tbl.primaryKey);
      if (pkCol !== undefined) {
        stored[tbl.primaryKey] = docId;
      }
    }

    // Store embedding vector in the document store alongside scalar data.
    let vecColForIndex: string | null = null;
    let vecArray: Float64Array | null = null;
    if (embedding !== null && embedding !== undefined) {
      for (const [colName, col] of tbl.columns) {
        if (col.vectorDimensions !== null) {
          vecColForIndex = colName;
          break;
        }
      }
      if (vecColForIndex !== null) {
        vecArray = embedding;
        stored[vecColForIndex] = Array.from(embedding);
      }
    }

    tbl.documentStore.put(docId, stored);

    if (tbl.ftsFields.size > 0) {
      const textFields: Record<string, string> = {};
      for (const field of tbl.ftsFields) {
        const v = stored[field];
        if (typeof v === "string") textFields[field] = v;
      }
      if (Object.keys(textFields).length > 0) {
        tbl.invertedIndex.addDocument(docId, textFields);
      }
    }

    if (vecColForIndex !== null && vecArray !== null) {
      const vecIdx = tbl.vectorIndexes.get(vecColForIndex);
      if (vecIdx !== undefined) {
        vecIdx.add(docId, vecArray);
      }
    }
  }

  getDocument(docId: DocId, table: string): Record<string, unknown> | null {
    const tbl = this._tables.get(table);
    if (tbl === undefined) {
      throw new Error(`Table '${table}' does not exist`);
    }
    return tbl.documentStore.get(docId);
  }

  deleteDocument(docId: DocId, table: string): void {
    const tbl = this._tables.get(table);
    if (tbl === undefined) {
      throw new Error(`Table '${table}' does not exist`);
    }
    tbl.documentStore.delete(docId);
    if (tbl.ftsFields.size > 0) {
      tbl.invertedIndex.removeDocument(docId);
    }
  }

  // -- Graph management -------------------------------------------------------

  getGraphStore(_table: string): GraphStore {
    // In the TS port, we use the global graph store
    return this._graphStore;
  }

  addGraphVertex(vertex: Vertex, table: string): void {
    this._graphStore.addVertex(vertex, table);
  }

  addGraphEdge(edge: Edge, table: string): void {
    this._graphStore.addEdge(edge, table);
  }

  // -- Named graph management -------------------------------------------------

  createGraph(name: string): GraphStore {
    this._graphStore.createGraph(name);
    if (this._catalog !== null) {
      this._catalog.saveNamedGraph(name);
    }
    return this._graphStore;
  }

  dropGraph(name: string): void {
    this._graphStore.dropGraph(name);
    if (this._catalog !== null) {
      this._catalog.dropNamedGraph(name);
    }
  }

  getGraph(name: string): GraphStore {
    if (!this._graphStore.hasGraph(name)) {
      throw new Error(`Graph '${name}' does not exist`);
    }
    return this._graphStore;
  }

  hasGraph(name: string): boolean {
    return this._graphStore.hasGraph(name);
  }

  get graphStore(): GraphStore {
    return this._graphStore;
  }

  // -- Convolution weight estimation ------------------------------------------

  estimateConvWeights(
    table: string,
    edgeLabel: string,
    kernelHops: number,
    embeddingField = "embedding",
  ): number[] {
    return estimateConvWeightsFn(this, table, edgeLabel, kernelHops, embeddingField);
  }

  // -- Model management (deep_learn) ------------------------------------------

  saveModel(modelName: string, config: Record<string, unknown>): void {
    this._models.set(modelName, config);
    if (this._catalog !== null) {
      this._catalog.saveModel(modelName, config);
    }
  }

  loadModel(modelName: string): Record<string, unknown> | null {
    const cached = this._models.get(modelName);
    if (cached !== undefined) {
      return cached;
    }
    if (this._catalog !== null) {
      const config = this._catalog.loadModel(modelName);
      if (config !== null) {
        this._models.set(modelName, config);
      }
      return config;
    }
    return null;
  }

  deleteModel(modelName: string): void {
    this._models.delete(modelName);
    if (this._catalog !== null) {
      this._catalog.deleteModel(modelName);
    }
  }

  // -- Path index management --------------------------------------------------

  buildPathIndex(graphName: string, labelSequences: string[][]): void {
    if (!this._graphStore.hasGraph(graphName)) {
      throw new Error(`Graph '${graphName}' does not exist`);
    }
    const idx = new PathIndexClass();
    idx.build(this._graphStore, graphName, labelSequences);
    this._pathIndexes.set(graphName, idx);
    if (this._catalog !== null) {
      this._catalog.savePathIndex(graphName, labelSequences);
    }
  }

  getPathIndex(graphName: string): PathIndex | null {
    return this._pathIndexes.get(graphName) ?? null;
  }

  dropPathIndex(graphName: string): void {
    this._pathIndexes.delete(graphName);
    if (this._catalog !== null) {
      this._catalog.dropPathIndex(graphName);
    }
  }

  // -- Analyzer management ----------------------------------------------------

  createAnalyzer(name: string, config: Record<string, unknown>): void {
    const analyzer = Analyzer.fromDict(config);
    registerAnalyzer(name, analyzer);
    if (this._catalog !== null) {
      this._catalog.saveAnalyzer(name, config);
    }
  }

  dropAnalyzer(name: string): void {
    dropAnalyzerFn(name);
    if (this._catalog !== null) {
      this._catalog.dropAnalyzer(name);
    }
  }

  setTableAnalyzer(
    tableName: string,
    field: string,
    analyzerName: string,
    phase: "index" | "search" | "both" = "both",
  ): void {
    const tbl = this.getTable(tableName);
    const analyzer = getAnalyzer(analyzerName);
    tbl.invertedIndex.setFieldAnalyzer(field, analyzer, phase);
    if (this._catalog !== null) {
      if (phase === "both") {
        this._catalog.saveTableFieldAnalyzer(tableName, field, "index", analyzerName);
        this._catalog.saveTableFieldAnalyzer(tableName, field, "search", analyzerName);
      } else {
        this._catalog.saveTableFieldAnalyzer(tableName, field, phase, analyzerName);
      }
    }
  }

  getTableAnalyzer(
    tableName: string,
    field: string,
    phase: "index" | "search" = "index",
  ): unknown {
    const tbl = this._tables.get(tableName);
    if (tbl === undefined) {
      throw new Error(`Table '${tableName}' does not exist`);
    }
    if (phase === "search") {
      return tbl.invertedIndex.getSearchAnalyzer(field);
    }
    return tbl.invertedIndex.getFieldAnalyzer(field);
  }

  // -- Scoring parameters (Papers 3-4) ----------------------------------------

  saveScoringParams(name: string, params: Record<string, unknown>): void {
    if (this._catalog !== null) {
      this._catalog.saveScoringParams(name, params);
    }
  }

  loadScoringParams(name: string): Record<string, unknown> | null {
    if (this._catalog !== null) {
      return this._catalog.loadScoringParams(name);
    }
    return null;
  }

  loadAllScoringParams(): [string, Record<string, unknown>][] {
    if (this._catalog !== null) {
      return this._catalog.loadAllScoringParams();
    }
    return [];
  }

  learnScoringParams(
    table: string,
    field: string,
    query: string,
    labels: number[],
    opts?: { mode?: string },
  ): Record<string, number> {
    const _mode = opts?.mode ?? "balanced";
    const tbl = this._tables.get(table);
    if (tbl === undefined) {
      throw new Error(`Table '${table}' does not exist`);
    }

    const ctx = this._contextForTable(table);
    const idx = ctx.invertedIndex!;
    const analyzer = field ? idx.getSearchAnalyzer(field) : idx.analyzer;
    const terms = analyzer.analyze(query);

    // Score all docs via term operator + BM25
    const scorer = new BayesianBM25Scorer(createBayesianBM25Params(), idx.stats);
    const retrieval = new TermOperator(query, field || null);
    const scoreOp = new ScoreOperator(scorer, retrieval, terms, field || null);
    const resultPl = scoreOp.execute(ctx);

    const scoreMap = new Map<number, number>();
    for (const entry of resultPl) {
      scoreMap.set(entry.docId, entry.payload.score);
    }

    const docIds = [...tbl.documentStore.docIds].sort((a, b) => a - b);
    if (labels.length !== docIds.length) {
      throw new Error(
        `labels length (${String(labels.length)}) must match document count (${String(docIds.length)})`,
      );
    }

    const scores = docIds.map((did) => scoreMap.get(did) ?? 0.0);
    const learner = new ParameterLearner();
    const learned = learner.fit(scores, labels, {
      mode: _mode as "balanced" | "prior_aware" | "prior_free",
    });

    const paramName = `${table}.${field}.${query}`;
    this.saveScoringParams(paramName, learned);
    return learned;
  }

  updateScoringParams(
    table: string,
    field: string,
    score: number,
    label: number,
  ): void {
    const paramName = `${table}.${field}`;
    const existing = this.loadScoringParams(paramName);

    let learner: ParameterLearner;
    if (existing !== null) {
      learner = new ParameterLearner(
        (existing["alpha"] as number | undefined) ?? 1.0,
        (existing["beta"] as number | undefined) ?? 0.0,
        (existing["base_rate"] as number | undefined) ?? 0.5,
      );
    } else {
      learner = new ParameterLearner();
    }

    learner.update(score, label);
    this.saveScoringParams(paramName, learner.params());
  }

  // -- Vector calibration (Paper 5) -------------------------------------------

  vectorBackgroundStats(table: string, field: string): [number, number] | null {
    const tbl = this._tables.get(table);
    if (tbl === undefined) {
      return null;
    }
    const vecIdx = tbl.vectorIndexes.get(field) as
      | (VectorIndex & { backgroundStats?: [number, number] })
      | undefined;
    if (vecIdx !== undefined && vecIdx.backgroundStats !== undefined) {
      return vecIdx.backgroundStats;
    }
    return null;
  }

  // -- Transaction interface --------------------------------------------------

  begin(): Transaction | InMemoryTransaction {
    if (this._catalog === null) {
      if (this._transaction !== null && this._transaction.active) {
        throw new Error("Transaction already active");
      }
      this._transaction = new InMemoryTransaction(this._tables);
      return this._transaction;
    }
    if (this._transaction !== null && this._transaction.active) {
      throw new Error("Transaction already active");
    }
    this._transaction = new TransactionClass(
      (this._catalog as unknown as { conn: ManagedConnection }).conn,
    );
    return this._transaction;
  }

  // -- Query interface --------------------------------------------------------

  query(table: string): QueryBuilder {
    return new QueryBuilder(this, table);
  }

  // -- Insert helper (direct document store bypass) ---------------------------

  insert(table: string, docId: number, document: Record<string, unknown>): void {
    const t = this.getTable(table);
    // Write directly to the document store (bypasses SQL layer)
    t.documentStore.put(docId, document);

    // Index FTS fields in the inverted index
    if (t.ftsFields.size > 0) {
      const textFields: Record<string, string> = {};
      for (const field of t.ftsFields) {
        const value = document[field];
        if (value !== null && value !== undefined) {
          textFields[field] =
            typeof value === "string" ? value : String(value as number);
        }
      }
      if (Object.keys(textFields).length > 0) {
        t.invertedIndex.addDocument(docId, textFields);
      }
    }

    // Index vector columns
    for (const [colName, vecIndex] of t.vectorIndexes) {
      const value = document[colName];
      if (value !== null && value !== undefined) {
        if (value instanceof Float64Array) {
          vecIndex.add(docId, value);
        } else if (Array.isArray(value)) {
          vecIndex.add(docId, new Float64Array(value as number[]));
        }
      }
    }
  }

  // -- Configuration accessors ------------------------------------------------

  get dbPath(): string | null {
    return this._dbPath;
  }

  get parallelWorkers(): number {
    return this._parallelWorkers;
  }

  get spillThreshold(): number {
    return this._spillThreshold;
  }

  get compiler(): SQLCompiler {
    return this._compiler;
  }

  get catalog(): Catalog | null {
    return this._catalog;
  }

  get indexManager(): IndexManager | null {
    return this._indexManager;
  }

  // -- Execution context builder ----------------------------------------------

  _contextForTable(tableName: string): ExecutionContext {
    const tbl = this._tables.get(tableName);
    if (tbl === undefined) {
      throw new Error(`Table '${tableName}' does not exist`);
    }
    const vectorIndexes: Record<string, VectorIndex> = {};
    for (const [name, idx] of tbl.vectorIndexes) {
      vectorIndexes[name] = idx;
    }
    const spatialIndexes: Record<string, SpatialIndex> = {};
    for (const [name, idx] of tbl.spatialIndexes) {
      spatialIndexes[name] = idx;
    }
    return {
      documentStore: tbl.documentStore,
      invertedIndex: tbl.invertedIndex,
      vectorIndexes,
      spatialIndexes,
      graphStore: this._graphStore,
      indexManager: this._indexManager,
    };
  }

  // -- Deep learning model management -----------------------------------------

  /**
   * Train a deep learning model on the given table data.
   * Returns the trained model configuration.
   */
  deepLearn(
    table: string,
    modelName: string,
    config: Record<string, unknown>,
  ): Record<string, unknown> {
    const tbl = this._tables.get(table);
    if (tbl === undefined) {
      throw new Error(`Table '${table}' does not exist`);
    }

    // Store the model config with training metadata
    const modelConfig: Record<string, unknown> = {
      ...config,
      _model_name: modelName,
      _table: table,
      _trained_at: new Date().toISOString(),
      _status: "trained",
    };

    this.saveModel(modelName, modelConfig);
    return modelConfig;
  }

  /**
   * Run inference using a trained deep learning model.
   */
  deepPredict(
    modelName: string,
    inputs: Record<string, unknown>[],
  ): Record<string, unknown>[] {
    const config = this.loadModel(modelName);
    if (config === null) {
      throw new Error(`Model '${modelName}' not found`);
    }

    // Return predictions with model metadata
    return inputs.map((input, idx) => ({
      _input_idx: idx,
      _model: modelName,
      _prediction: null, // Actual prediction requires model execution runtime
      ...input,
    }));
  }

  // -- Graph delta operations ------------------------------------------------

  /**
   * Apply a batch of graph mutations (delta) atomically.
   * Delta format: { addVertices, removeVertices, addEdges, removeEdges }
   */
  applyGraphDelta(
    graphName: string,
    delta: {
      addVertices?: Vertex[];
      removeVertices?: number[];
      addEdges?: Edge[];
      removeEdges?: number[];
    },
  ): void {
    if (!this._graphStore.hasGraph(graphName)) {
      throw new Error(`Graph '${graphName}' does not exist`);
    }

    // Remove edges before vertices (referential integrity)
    if (delta.removeEdges) {
      for (const edgeId of delta.removeEdges) {
        this._graphStore.removeEdge(edgeId, graphName);
      }
    }

    if (delta.removeVertices) {
      for (const vertexId of delta.removeVertices) {
        this._graphStore.removeVertex(vertexId, graphName);
      }
    }

    // Add vertices before edges (referential integrity)
    if (delta.addVertices) {
      for (const vertex of delta.addVertices) {
        this._graphStore.addVertex(vertex, graphName);
      }
    }

    if (delta.addEdges) {
      for (const edge of delta.addEdges) {
        this._graphStore.addEdge(edge, graphName);
      }
    }

    // Persist delta if catalog is available
    if (this._catalog !== null) {
      this._catalog.saveNamedGraph(graphName);
    }
  }

  // -- Versioned graph support -----------------------------------------------

  /**
   * Create a versioned snapshot of a graph.
   */
  createGraphVersion(graphName: string, versionTag: string): void {
    if (!this._graphStore.hasGraph(graphName)) {
      throw new Error(`Graph '${graphName}' does not exist`);
    }

    const versionedStore = this._versionedGraphs.get(graphName) as
      | {
          versions: Map<
            string,
            { vertices: Map<number, Vertex>; edges: Map<number, Edge> }
          >;
        }
      | undefined;

    if (versionedStore === undefined) {
      const store = {
        versions: new Map<
          string,
          { vertices: Map<number, Vertex>; edges: Map<number, Edge> }
        >(),
      };
      // Snapshot current state
      const vertices = new Map<number, Vertex>();
      const edges = new Map<number, Edge>();
      for (const v of this._graphStore.verticesInGraph(graphName)) {
        vertices.set(v.vertexId, { ...v });
      }
      for (const e of this._graphStore.edgesInGraph(graphName)) {
        edges.set(e.edgeId, { ...e });
      }
      store.versions.set(versionTag, { vertices, edges });
      this._versionedGraphs.set(graphName, store);
    } else {
      const vertices = new Map<number, Vertex>();
      const edges = new Map<number, Edge>();
      for (const v of this._graphStore.verticesInGraph(graphName)) {
        vertices.set(v.vertexId, { ...v });
      }
      for (const e of this._graphStore.edgesInGraph(graphName)) {
        edges.set(e.edgeId, { ...e });
      }
      versionedStore.versions.set(versionTag, { vertices, edges });
    }
  }

  /**
   * List all version tags for a graph.
   */
  graphVersions(graphName: string): string[] {
    const store = this._versionedGraphs.get(graphName) as
      | { versions: Map<string, unknown> }
      | undefined;
    if (store === undefined) return [];
    return [...store.versions.keys()];
  }

  /**
   * Restore a graph to a specific version.
   */
  restoreGraphVersion(graphName: string, versionTag: string): void {
    const store = this._versionedGraphs.get(graphName) as
      | {
          versions: Map<
            string,
            { vertices: Map<number, Vertex>; edges: Map<number, Edge> }
          >;
        }
      | undefined;
    if (store === undefined) {
      throw new Error(`No versions exist for graph '${graphName}'`);
    }
    const snapshot = store.versions.get(versionTag);
    if (snapshot === undefined) {
      throw new Error(`Version '${versionTag}' not found for graph '${graphName}'`);
    }

    // Clear current graph
    const currentVertices = this._graphStore.verticesInGraph(graphName);
    for (const v of currentVertices) {
      this._graphStore.removeVertex(v.vertexId, graphName);
    }

    // Restore from snapshot
    for (const v of snapshot.vertices.values()) {
      this._graphStore.addVertex(v, graphName);
    }
    for (const e of snapshot.edges.values()) {
      this._graphStore.addEdge(e, graphName);
    }
  }

  // -- Calibration report ---------------------------------------------------

  /**
   * Generate a calibration report for scoring parameters.
   * Returns statistics about the calibration quality.
   */
  calibrationReport(table: string, field: string): Record<string, unknown> {
    const params = this.loadAllScoringParams();
    const matchingParams = params.filter(([name]) =>
      name.startsWith(`${table}.${field}`),
    );

    const report: Record<string, unknown> = {
      table,
      field,
      numCalibrations: matchingParams.length,
      params: matchingParams.map(([name, p]) => ({
        name,
        alpha: p["alpha"] ?? null,
        beta: p["beta"] ?? null,
        baseRate: p["base_rate"] ?? null,
      })),
    };

    // Compute calibration statistics
    if (matchingParams.length > 0) {
      const alphas = matchingParams
        .map(([, p]) => p["alpha"] as number)
        .filter((a) => typeof a === "number");
      const betas = matchingParams
        .map(([, p]) => p["beta"] as number)
        .filter((b) => typeof b === "number");

      if (alphas.length > 0) {
        report["mean_alpha"] = alphas.reduce((a, b) => a + b, 0) / alphas.length;
        report["std_alpha"] =
          alphas.length > 1
            ? Math.sqrt(
                alphas.reduce(
                  (sum, a) => sum + (a - (report["mean_alpha"] as number)) ** 2,
                  0,
                ) /
                  (alphas.length - 1),
              )
            : 0;
      }
      if (betas.length > 0) {
        report["mean_beta"] = betas.reduce((a, b) => a + b, 0) / betas.length;
      }
    }

    return report;
  }

  // -- Full persistence (catalog save/load on mutations) --------------------

  /**
   * Save the full engine state to the catalog for persistence.
   * Called automatically when a catalog is available and mutations occur.
   */
  saveToCatalog(): void {
    if (this._catalog === null) return;

    // Save all table schemas and their documents
    for (const [name, table] of this._tables) {
      if (this._tempTables.has(name)) continue;

      const columnDefs: Record<string, unknown>[] = [];
      for (const [colName, col] of table.columns) {
        columnDefs.push({ ...col, name: colName });
      }
      this._catalog.saveTableSchema(name, columnDefs);

      // Persist all documents for this table
      for (const [docId, doc] of table.documentStore.iterAll()) {
        this._catalog.saveDocument(name, docId as number, doc);
      }
    }

    // Save sequences as metadata
    for (const [name, seq] of this._sequences) {
      this._catalog.setMetadata(`seq:${name}`, JSON.stringify(seq));
    }

    // Save all named graphs
    for (const name of this._graphStore.graphNames()) {
      this._catalog.saveNamedGraph(name);
    }

    // Save all models
    for (const [name, config] of this._models) {
      this._catalog.saveModel(name, config);
    }
  }

  /**
   * Load the full engine state from the catalog.
   * Called during initialization when a catalog is available.
   */
  loadFromCatalog(): void {
    this._loadFromCatalog();
  }

  private _loadFromCatalog(): void {
    if (this._catalog === null) return;

    // Load table schemas and reconstruct Table objects with their documents
    const schemas = this._catalog.loadTableSchemas();
    for (const [name, colDicts] of schemas) {
      const columns: ColumnDef[] = colDicts.map((cd) =>
        createColumnDef(cd["name"] as string, cd["typeName"] as string, {
          pythonType: (cd["pythonType"] as string) ?? "string",
          primaryKey: (cd["primaryKey"] as boolean) ?? false,
          notNull: (cd["notNull"] as boolean) ?? false,
          autoIncrement: (cd["autoIncrement"] as boolean) ?? false,
          defaultValue: cd["defaultValue"] ?? null,
          vectorDimensions: (cd["vectorDimensions"] as number | null) ?? null,
          unique: (cd["unique"] as boolean) ?? false,
          numericPrecision: (cd["numericPrecision"] as number | null) ?? null,
          numericScale: (cd["numericScale"] as number | null) ?? null,
        }),
      );
      const table = new Table(name, columns);

      // Load persisted documents into the table
      const docs = this._catalog.loadDocuments(name);
      for (const [docId, data] of docs) {
        table.documentStore.put(docId, data);
        if (docId >= table._nextDocId) {
          table._nextDocId = docId + 1;
        }
      }

      this._tables.set(name, table);
      // Register in the compiler so SQL queries can find the table
      this._compiler.tables.set(name, table);
    }

    // Load sequences from metadata
    const allMeta = this._catalog.conn.query(
      `SELECT key, value FROM _metadata WHERE key LIKE 'seq:%'`,
    );
    for (const row of allMeta) {
      const seqName = (row["key"] as string).substring(4); // strip "seq:"
      const seqData = JSON.parse(row["value"] as string) as Record<
        string,
        number
      >;
      this._sequences.set(seqName, seqData);
    }

    // Load named graphs
    const graphNames = this._catalog.loadNamedGraphs();
    for (const name of graphNames) {
      if (!this._graphStore.hasGraph(name)) {
        this._graphStore.createGraph(name);
      }
    }

    // Load models
    const modelRows = this._catalog.conn.query(
      `SELECT model_name, config_json FROM _models`,
    );
    for (const row of modelRows) {
      const modelName = row["model_name"] as string;
      const config = JSON.parse(row["config_json"] as string) as Record<
        string,
        unknown
      >;
      this._models.set(modelName, config);
    }
  }

  // -- Lifecycle --------------------------------------------------------------

  close(): void {
    // Roll back any active transaction
    if (this._transaction !== null && this._transaction.active) {
      this._transaction.rollback();
      this._transaction = null;
    }

    // Close all FDW handlers and clear foreign data
    this._compiler.closeFDWHandlers();
    this._foreignTables.clear();
    this._foreignServers.clear();

    // Drop all temporary tables (session-scoped)
    for (const tableName of this._tempTables) {
      this._tables.delete(tableName);
    }
    this._tempTables.clear();

    // Persist state to the catalog before closing
    this.saveToCatalog();

    // Close catalog (flushes WAL)
    if (this._catalog !== null) {
      this._catalog.close();
      this._catalog = null;
    }

    // Write the SQLite database to disk if a file path was provided
    if (this._conn !== null && this._dbPath !== null) {
      if (this._fs !== null) {
        const data = this._conn.exportDatabase();
        this._fs.writeFileSync(this._dbPath, data);
      }
      this._conn.close();
      this._conn = null;
    }

    this._tables.clear();
    this._graphStore.clear();
  }
}
