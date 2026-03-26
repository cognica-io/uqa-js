//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- SQLite-backed InvertedIndex
// 1:1 port of uqa/storage/sqlite_inverted_index.py
//
// Performance structures (Phase 2, Section 3.2.2):
//   Skip pointers -- every Nth doc_id per term for fast forward-seeking.
//   Block-max scores -- per-block maximum BM25 scores for BMW pruning.

import type { DocId, FieldName, PostingEntry } from "../core/types.js";
import { IndexStats, createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { AnalyzerLike, IndexedTerms } from "./abc/inverted-index.js";
import { InvertedIndex } from "./abc/inverted-index.js";
import type { ManagedConnection } from "./managed-connection.js";

// Default analyzer: simple whitespace + lowercase split
const DEFAULT_ANALYZER: AnalyzerLike = {
  analyze(text: string): string[] {
    return text
      .toLowerCase()
      .split(/\s+/)
      .filter((t) => t.length > 0);
  },
};

export class SQLiteInvertedIndex extends InvertedIndex {
  static readonly BLOCK_SIZE = 128;

  private _conn: ManagedConnection;
  private _tableName: string;
  private _analyzer: AnalyzerLike;
  private _indexFieldAnalyzers: Map<string, AnalyzerLike>;
  private _searchFieldAnalyzers: Map<string, AnalyzerLike>;
  private _knownFields: Set<string>;
  private _cachedStats: IndexStats | null;
  private _dirtyTerms: Set<string>; // serialized "field\0term" keys

  constructor(
    conn: ManagedConnection,
    tableName: string,
    analyzer?: AnalyzerLike | null,
  ) {
    super();
    this._conn = conn;
    this._tableName = tableName;
    this._analyzer = analyzer ?? DEFAULT_ANALYZER;
    this._indexFieldAnalyzers = new Map();
    this._searchFieldAnalyzers = new Map();
    this._knownFields = new Set();
    this._cachedStats = null;
    this._dirtyTerms = new Set();
    this._ensureSharedTables();
    this._discoverFields();
  }

  // -- Shared tables (field stats, doc lengths) ------

  private _ensureSharedTables(): void {
    this._conn.execute(
      `CREATE TABLE IF NOT EXISTS "_field_stats_${this._tableName}" (
        field        TEXT PRIMARY KEY,
        doc_count    INTEGER NOT NULL DEFAULT 0,
        total_length INTEGER NOT NULL DEFAULT 0
      )`,
    );
    this._conn.execute(
      `CREATE TABLE IF NOT EXISTS "_doc_lengths_${this._tableName}" (
        doc_id  INTEGER NOT NULL,
        field   TEXT    NOT NULL,
        length  INTEGER NOT NULL,
        PRIMARY KEY (doc_id, field)
      )`,
    );
  }

  private _discoverFields(): void {
    const rows = this._conn.query(
      `SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?`,
      [`_inverted_${this._tableName}_%`],
    );
    const prefix = `_inverted_${this._tableName}_`;
    for (const row of rows) {
      const name = row["name"] as string;
      if (name.startsWith(prefix)) {
        this._knownFields.add(name.slice(prefix.length));
      }
    }
  }

  // -- Lazy per-field table creation ------

  private _ensureFieldTable(field: string): void {
    if (this._knownFields.has(field)) return;
    const tbl = this._invertedTableName(field);
    const skipTbl = this._skipTableName(field);
    const bmTbl = this._blockmaxTableName(field);
    this._conn.execute(
      `CREATE TABLE IF NOT EXISTS "${tbl}" (
        term    TEXT    NOT NULL,
        doc_id  INTEGER NOT NULL,
        tf      INTEGER NOT NULL,
        positions TEXT  NOT NULL,
        PRIMARY KEY (term, doc_id)
      )`,
    );
    this._conn.execute(
      `CREATE TABLE IF NOT EXISTS "${skipTbl}" (
        term        TEXT    NOT NULL,
        skip_doc_id INTEGER NOT NULL,
        skip_offset INTEGER NOT NULL,
        PRIMARY KEY (term, skip_doc_id)
      )`,
    );
    this._conn.execute(
      `CREATE TABLE IF NOT EXISTS "${bmTbl}" (
        term      TEXT    NOT NULL,
        block_idx INTEGER NOT NULL,
        max_score REAL    NOT NULL,
        PRIMARY KEY (term, block_idx)
      )`,
    );
    this._knownFields.add(field);
  }

  private _invertedTableName(field: string): string {
    return `_inverted_${this._tableName}_${field}`;
  }

  private _skipTableName(field: string): string {
    return `_skip_${this._tableName}_${field}`;
  }

  private _blockmaxTableName(field: string): string {
    return `_blockmax_${this._tableName}_${field}`;
  }

  // -- Analyzer accessors ------

  get analyzer(): AnalyzerLike {
    return this._analyzer;
  }

  get fieldAnalyzers(): Record<string, AnalyzerLike> {
    const result: Record<string, AnalyzerLike> = {};
    for (const [k, v] of this._indexFieldAnalyzers) {
      result[k] = v;
    }
    return result;
  }

  setFieldAnalyzer(
    field: string,
    analyzer: AnalyzerLike,
    phase: "index" | "search" | "both" = "both",
  ): void {
    if (phase === "index" || phase === "both") {
      this._indexFieldAnalyzers.set(field, analyzer);
    }
    if (phase === "search" || phase === "both") {
      this._searchFieldAnalyzers.set(field, analyzer);
    }
  }

  getFieldAnalyzer(field: string): AnalyzerLike {
    return this._indexFieldAnalyzers.get(field) ?? this._analyzer;
  }

  getSearchAnalyzer(field: string): AnalyzerLike {
    return (
      this._searchFieldAnalyzers.get(field) ??
      this._indexFieldAnalyzers.get(field) ??
      this._analyzer
    );
  }

  // -- Tokenization ------

  private _tokenize(text: string, field?: string | null): string[] {
    const analyzer = field
      ? (this._indexFieldAnalyzers.get(field) ?? this._analyzer)
      : this._analyzer;
    return analyzer.analyze(text);
  }

  // -- Deferred skip pointer rebuild ------

  flushSkipPointers(): void {
    if (this._dirtyTerms.size === 0) return;
    const dirty = this._dirtyTerms;
    this._dirtyTerms = new Set();
    for (const key of dirty) {
      const [field, term] = key.split("\0");
      this._rebuildSkipPointers(field!, term!);
    }
  }

  // -- Indexing ------

  addDocument(docId: DocId, fields: Record<FieldName, string>): IndexedTerms {
    this._cachedStats = null;
    const resultFieldLengths: Record<string, number> = {};
    const resultPostings = new Map<string, readonly number[]>();

    for (const [fieldName, text] of Object.entries(fields)) {
      this._ensureFieldTable(fieldName);
      const tbl = this._invertedTableName(fieldName);
      const tokens = this._tokenize(text, fieldName);
      const length = tokens.length;
      resultFieldLengths[fieldName] = length;

      // Build position index for each token.
      const termPositions = new Map<string, number[]>();
      for (let pos = 0; pos < tokens.length; pos++) {
        const token = tokens[pos]!;
        let positions = termPositions.get(token);
        if (!positions) {
          positions = [];
          termPositions.set(token, positions);
        }
        positions.push(pos);
      }

      // Insert postings
      for (const [term, positions] of termPositions) {
        const key = `${fieldName}\0${term}`;
        const posJSON = JSON.stringify(positions);
        this._conn.execute(
          `INSERT OR REPLACE INTO "${tbl}" (term, doc_id, tf, positions) VALUES (?, ?, ?, ?)`,
          [term, docId, positions.length, posJSON],
        );
        resultPostings.set(key, positions);
      }

      // Doc length
      this._conn.execute(
        `INSERT OR REPLACE INTO "_doc_lengths_${this._tableName}" (doc_id, field, length) VALUES (?, ?, ?)`,
        [docId, fieldName, length],
      );

      // Field stats -- upsert doc_count and total_length
      this._conn.execute(
        `INSERT INTO "_field_stats_${this._tableName}" (field, doc_count, total_length)
         VALUES (?, 1, ?)
         ON CONFLICT(field) DO UPDATE SET
         doc_count = doc_count + 1,
         total_length = total_length + ?`,
        [fieldName, length, length],
      );
    }

    // Defer skip pointer rebuilds until the next query.
    for (const key of resultPostings.keys()) {
      this._dirtyTerms.add(key);
    }

    return { fieldLengths: resultFieldLengths, postings: resultPostings };
  }

  addPosting(field: string, term: string, entry: PostingEntry): void {
    this._cachedStats = null;
    this._ensureFieldTable(field);
    const tbl = this._invertedTableName(field);
    const positions = entry.payload.positions;
    const tf = positions.length;
    this._conn.execute(
      `INSERT OR REPLACE INTO "${tbl}" (term, doc_id, tf, positions) VALUES (?, ?, ?, ?)`,
      [term, entry.docId, tf, JSON.stringify(positions)],
    );
  }

  setDocLength(docId: DocId, lengths: Record<FieldName, number>): void {
    for (const [field, length] of Object.entries(lengths)) {
      this._conn.execute(
        `INSERT OR REPLACE INTO "_doc_lengths_${this._tableName}" (doc_id, field, length) VALUES (?, ?, ?)`,
        [docId, field, length],
      );
    }
  }

  setDocCount(count: number): void {
    this._cachedStats = null;
    this._conn.execute(`UPDATE "_field_stats_${this._tableName}" SET doc_count = ?`, [
      count,
    ]);
  }

  addTotalLength(field: FieldName, length: number): void {
    this._cachedStats = null;
    this._conn.execute(
      `INSERT INTO "_field_stats_${this._tableName}" (field, doc_count, total_length)
       VALUES (?, 0, ?)
       ON CONFLICT(field) DO UPDATE SET total_length = total_length + ?`,
      [field, length, length],
    );
  }

  removeDocument(docId: DocId): void {
    this._cachedStats = null;

    // Collect per-field lengths
    const lengthRows = this._conn.query(
      `SELECT field, length FROM "_doc_lengths_${this._tableName}" WHERE doc_id = ?`,
      [docId],
    );

    // Collect affected (field, term) pairs and delete postings
    for (const field of this._knownFields) {
      const tbl = this._invertedTableName(field);
      const termRows = this._conn.query(`SELECT term FROM "${tbl}" WHERE doc_id = ?`, [
        docId,
      ]);
      for (const row of termRows) {
        this._dirtyTerms.add(`${field}\0${row["term"] as string}`);
      }
      this._conn.execute(`DELETE FROM "${tbl}" WHERE doc_id = ?`, [docId]);
    }

    if (lengthRows.length > 0) {
      for (const row of lengthRows) {
        const field = row["field"] as string;
        const length = row["length"] as number;
        this._conn.execute(
          `UPDATE "_field_stats_${this._tableName}"
           SET doc_count = MAX(doc_count - 1, 0),
               total_length = MAX(total_length - ?, 0)
           WHERE field = ?`,
          [length, field],
        );
      }
      this._conn.execute(
        `DELETE FROM "_doc_lengths_${this._tableName}" WHERE doc_id = ?`,
        [docId],
      );
    }
  }

  clear(): void {
    this._cachedStats = null;
    for (const field of this._knownFields) {
      const tbl = this._invertedTableName(field);
      this._conn.execute(`DELETE FROM "${tbl}"`);
      const skipTbl = this._skipTableName(field);
      this._conn.execute(`DELETE FROM "${skipTbl}"`);
      const bmTbl = this._blockmaxTableName(field);
      this._conn.execute(`DELETE FROM "${bmTbl}"`);
    }
    this._conn.execute(`DELETE FROM "_field_stats_${this._tableName}"`);
    this._conn.execute(`DELETE FROM "_doc_lengths_${this._tableName}"`);
    this._dirtyTerms.clear();
  }

  // -- Querying ------

  private _flushTerm(field: string, term: string): void {
    const key = `${field}\0${term}`;
    if (this._dirtyTerms.has(key)) {
      this._dirtyTerms.delete(key);
      this._rebuildSkipPointers(field, term);
    }
  }

  getPostingList(field: string, term: string): PostingList {
    this._flushTerm(field, term);
    if (!this._knownFields.has(field)) return new PostingList();
    const tbl = this._invertedTableName(field);
    const rows = this._conn.query(
      `SELECT doc_id, tf, positions FROM "${tbl}" WHERE term = ? ORDER BY doc_id`,
      [term],
    );
    const entries: PostingEntry[] = rows.map((row) => ({
      docId: row["doc_id"] as number,
      payload: createPayload({
        positions: JSON.parse(row["positions"] as string) as number[],
        score: 0.0,
      }),
    }));
    return PostingList.fromSorted(entries);
  }

  getPostingListAnyField(term: string): PostingList {
    this.flushSkipPointers();
    const seen = new Set<DocId>();
    const allEntries: PostingEntry[] = [];

    for (const field of [...this._knownFields].sort()) {
      const tbl = this._invertedTableName(field);
      const rows = this._conn.query(
        `SELECT doc_id, tf, positions FROM "${tbl}" WHERE term = ? ORDER BY doc_id`,
        [term],
      );
      for (const row of rows) {
        const docId = row["doc_id"] as number;
        if (!seen.has(docId)) {
          seen.add(docId);
          allEntries.push({
            docId,
            payload: createPayload({
              positions: JSON.parse(row["positions"] as string) as number[],
              score: 0.0,
            }),
          });
        }
      }
    }
    return new PostingList(allEntries);
  }

  docFreq(field: string, term: string): number {
    if (!this._knownFields.has(field)) return 0;
    const tbl = this._invertedTableName(field);
    const row = this._conn.queryOne(
      `SELECT COUNT(*) AS cnt FROM "${tbl}" WHERE term = ?`,
      [term],
    );
    return row !== null ? (row["cnt"] as number) : 0;
  }

  docFreqAnyField(term: string): number {
    if (this._knownFields.size === 0) return 0;
    const parts: string[] = [];
    const params: unknown[] = [];
    for (const field of this._knownFields) {
      const tbl = this._invertedTableName(field);
      parts.push(`SELECT DISTINCT doc_id FROM "${tbl}" WHERE term = ?`);
      params.push(term);
    }
    const sql = `SELECT COUNT(DISTINCT doc_id) AS cnt FROM (${parts.join(" UNION ALL ")})`;
    const row = this._conn.queryOne(sql, params);
    return row !== null ? (row["cnt"] as number) : 0;
  }

  getDocLength(docId: DocId, field: FieldName): number {
    const row = this._conn.queryOne(
      `SELECT length FROM "_doc_lengths_${this._tableName}" WHERE doc_id = ? AND field = ?`,
      [docId, field],
    );
    return row !== null ? (row["length"] as number) : 0;
  }

  getDocLengthsBulk(docIds: DocId[], field: FieldName): Map<DocId, number> {
    const result = new Map<DocId, number>();
    // Process in chunks of 500 to avoid SQLite variable limits
    for (let start = 0; start < docIds.length; start += 500) {
      const chunk = docIds.slice(start, start + 500);
      const placeholders = chunk.map(() => "?").join(",");
      const rows = this._conn.query(
        `SELECT doc_id, length FROM "_doc_lengths_${this._tableName}"
         WHERE field = ? AND doc_id IN (${placeholders})`,
        [field, ...chunk],
      );
      for (const row of rows) {
        result.set(row["doc_id"] as number, row["length"] as number);
      }
    }
    // Fill missing with 0
    for (const docId of docIds) {
      if (!result.has(docId)) result.set(docId, 0);
    }
    return result;
  }

  getTotalDocLength(docId: DocId): number {
    const row = this._conn.queryOne(
      `SELECT SUM(length) AS total FROM "_doc_lengths_${this._tableName}" WHERE doc_id = ?`,
      [docId],
    );
    return row !== null && row["total"] !== null ? (row["total"] as number) : 0;
  }

  getTermFreq(docId: DocId, field: string, term: string): number {
    if (!this._knownFields.has(field)) return 0;
    const tbl = this._invertedTableName(field);
    const row = this._conn.queryOne(
      `SELECT tf FROM "${tbl}" WHERE term = ? AND doc_id = ?`,
      [term, docId],
    );
    return row !== null ? (row["tf"] as number) : 0;
  }

  getTermFreqsBulk(docIds: DocId[], field: string, term: string): Map<DocId, number> {
    const result = new Map<DocId, number>();
    if (!this._knownFields.has(field)) {
      for (const docId of docIds) result.set(docId, 0);
      return result;
    }
    const tbl = this._invertedTableName(field);
    for (let start = 0; start < docIds.length; start += 500) {
      const chunk = docIds.slice(start, start + 500);
      const placeholders = chunk.map(() => "?").join(",");
      const rows = this._conn.query(
        `SELECT doc_id, tf FROM "${tbl}" WHERE term = ? AND doc_id IN (${placeholders})`,
        [term, ...chunk],
      );
      for (const row of rows) {
        result.set(row["doc_id"] as number, row["tf"] as number);
      }
    }
    for (const docId of docIds) {
      if (!result.has(docId)) result.set(docId, 0);
    }
    return result;
  }

  getTotalTermFreq(docId: DocId, term: string): number {
    if (this._knownFields.size === 0) return 0;
    let total = 0;
    for (const field of this._knownFields) {
      total += this.getTermFreq(docId, field, term);
    }
    return total;
  }

  // -- Statistics ------

  get stats(): IndexStats {
    this.flushSkipPointers();
    if (this._cachedStats !== null) return this._cachedStats;

    const rows = this._conn.query(
      `SELECT field, doc_count, total_length FROM "_field_stats_${this._tableName}"`,
    );

    if (rows.length === 0) {
      return new IndexStats(0, 0.0);
    }

    const totalDocs = Math.max(...rows.map((r) => r["doc_count"] as number));
    let totalLength = 0;
    for (const r of rows) {
      totalLength += r["total_length"] as number;
    }
    const avgDocLength = totalDocs > 0 ? totalLength / totalDocs : 0.0;

    const s = new IndexStats(totalDocs, avgDocLength);

    // Build doc frequencies
    for (const field of this._knownFields) {
      const tbl = this._invertedTableName(field);
      const termRows = this._conn.query(
        `SELECT term, COUNT(*) AS df FROM "${tbl}" GROUP BY term`,
      );
      for (const row of termRows) {
        s.setDocFreq(row["term"] as string, row["term"] as string, row["df"] as number);
        // Also set with field key
        s.setDocFreq(field, row["term"] as string, row["df"] as number);
      }
    }

    this._cachedStats = s;
    return s;
  }

  // -- Skip pointers ------

  private _rebuildSkipPointers(field: string, term: string): void {
    if (!this._knownFields.has(field)) return;
    const skipTbl = this._skipTableName(field);
    const invTbl = this._invertedTableName(field);

    // Clear old skips for this term
    this._conn.execute(`DELETE FROM "${skipTbl}" WHERE term = ?`, [term]);

    // Fetch sorted doc_ids for this term
    const rows = this._conn.query(
      `SELECT doc_id FROM "${invTbl}" WHERE term = ? ORDER BY doc_id`,
      [term],
    );

    // Insert skip entries every BLOCK_SIZE docs
    for (let offset = 0; offset < rows.length; offset++) {
      if (offset % SQLiteInvertedIndex.BLOCK_SIZE === 0) {
        this._conn.execute(
          `INSERT INTO "${skipTbl}" (term, skip_doc_id, skip_offset) VALUES (?, ?, ?)`,
          [term, rows[offset]!["doc_id"], offset],
        );
      }
    }
  }

  skipTo(field: string, term: string, targetDocId: number): [number, number] {
    this.flushSkipPointers();
    if (!this._knownFields.has(field)) return [0, 0];
    const skipTbl = this._skipTableName(field);
    const row = this._conn.queryOne(
      `SELECT skip_doc_id, skip_offset FROM "${skipTbl}"
       WHERE term = ? AND skip_doc_id <= ?
       ORDER BY skip_doc_id DESC LIMIT 1`,
      [term, targetDocId],
    );
    if (row === null) return [0, 0];
    return [row["skip_doc_id"] as number, row["skip_offset"] as number];
  }

  // -- Block-max scores ------

  buildBlockMaxScores(
    field: string,
    term: string,
    scorer: { score(tf: number, dl: number, df: number): number },
  ): void {
    if (!this._knownFields.has(field)) return;
    const invTbl = this._invertedTableName(field);
    const bmTbl = this._blockmaxTableName(field);

    const rows = this._conn.query(
      `SELECT doc_id, tf FROM "${invTbl}" WHERE term = ? ORDER BY doc_id`,
      [term],
    );

    const docFreq = rows.length;

    // Clear old block-max entries
    this._conn.execute(`DELETE FROM "${bmTbl}" WHERE term = ?`, [term]);

    // Compute per-block maximums
    const blockSize = SQLiteInvertedIndex.BLOCK_SIZE;
    for (let blockStart = 0; blockStart < rows.length; blockStart += blockSize) {
      const blockEnd = Math.min(blockStart + blockSize, rows.length);
      let maxScore = 0.0;
      for (let i = blockStart; i < blockEnd; i++) {
        const tf = rows[i]!["tf"] as number;
        const score = scorer.score(tf, tf, docFreq);
        maxScore = Math.max(maxScore, score);
      }
      const blockIdx = Math.floor(blockStart / blockSize);
      this._conn.execute(
        `INSERT INTO "${bmTbl}" (term, block_idx, max_score) VALUES (?, ?, ?)`,
        [term, blockIdx, maxScore],
      );
    }
  }

  buildAllBlockMaxScores(
    field: string,
    scorer: { score(tf: number, dl: number, df: number): number },
  ): void {
    if (!this._knownFields.has(field)) return;
    const invTbl = this._invertedTableName(field);
    const terms = this._conn.query(`SELECT DISTINCT term FROM "${invTbl}"`);
    for (const row of terms) {
      this.buildBlockMaxScores(field, row["term"] as string, scorer);
    }
  }

  getBlockMaxScore(field: string, term: string, blockIdx: number): number {
    if (!this._knownFields.has(field)) return 0.0;
    const bmTbl = this._blockmaxTableName(field);
    const row = this._conn.queryOne(
      `SELECT max_score FROM "${bmTbl}" WHERE term = ? AND block_idx = ?`,
      [term, blockIdx],
    );
    return row !== null ? (row["max_score"] as number) : 0.0;
  }

  getAllBlockMaxScores(field: string, term: string): number[] {
    if (!this._knownFields.has(field)) return [];
    const bmTbl = this._blockmaxTableName(field);
    const rows = this._conn.query(
      `SELECT max_score FROM "${bmTbl}" WHERE term = ? ORDER BY block_idx`,
      [term],
    );
    return rows.map((r) => r["max_score"] as number);
  }

  // -- Bulk operations -------------------------------------------------------

  addDocuments(docs: Array<[DocId, Record<FieldName, string>]>): void {
    const wasInTransaction = this._conn.inTransaction;
    if (!wasInTransaction) this._conn.beginTransaction();
    try {
      for (const [docId, fields] of docs) {
        this.addDocument(docId, fields);
      }
      if (!wasInTransaction) this._conn.commit();
    } catch (e) {
      if (!wasInTransaction) this._conn.rollback();
      throw e;
    }
  }

  removeDocuments(docIds: DocId[]): void {
    const wasInTransaction = this._conn.inTransaction;
    if (!wasInTransaction) this._conn.beginTransaction();
    try {
      for (const docId of docIds) {
        this.removeDocument(docId);
      }
      if (!wasInTransaction) this._conn.commit();
    } catch (e) {
      if (!wasInTransaction) this._conn.rollback();
      throw e;
    }
  }

  // -- Term enumeration ------------------------------------------------------

  *terms(field: string): Generator<string> {
    if (!this._knownFields.has(field)) return;
    const postTbl = this._invertedTableName(field);
    const rows = this._conn.query(
      `SELECT DISTINCT term FROM "${postTbl}" ORDER BY term`,
    );
    for (const row of rows) {
      yield row["term"] as string;
    }
  }

  *allTerms(): Generator<[string, string]> {
    for (const field of this._knownFields) {
      for (const term of this.terms(field)) {
        yield [field, term];
      }
    }
  }

  *fieldNames(): Generator<string> {
    for (const field of this._knownFields) {
      yield field;
    }
  }

  // -- Existence checks ------------------------------------------------------

  hasTerm(field: string, term: string): boolean {
    if (!this._knownFields.has(field)) return false;
    const postTbl = this._invertedTableName(field);
    const row = this._conn.queryOne(
      `SELECT 1 AS found FROM "${postTbl}" WHERE term = ? LIMIT 1`,
      [term],
    );
    return row !== null;
  }

  hasDoc(docId: DocId): boolean {
    const row = this._conn.queryOne(
      `SELECT 1 AS found FROM "${this._tableName}__doc_lengths" WHERE doc_id = ? LIMIT 1`,
      [docId],
    );
    return row !== null;
  }

  // -- Document length statistics -------------------------------------------

  avgDocLength(_field: FieldName): number {
    const s = this.stats;
    return s.avgDocLength;
  }

  totalDocCount(): number {
    return this.stats.totalDocs;
  }

  totalFieldLength(field: FieldName): number {
    const row = this._conn.queryOne(
      `SELECT SUM(length) AS total FROM "${this._tableName}__doc_lengths" WHERE field = ?`,
      [field],
    );
    if (row === null || row["total"] === null) return 0;
    return row["total"] as number;
  }

  // -- Position access -------------------------------------------------------

  getPositions(docId: DocId, field: string, term: string): readonly number[] {
    if (!this._knownFields.has(field)) return [];
    const postTbl = this._invertedTableName(field);
    const row = this._conn.queryOne(
      `SELECT positions FROM "${postTbl}" WHERE term = ? AND doc_id = ?`,
      [term, docId],
    );
    if (row === null || row["positions"] === null) return [];
    return JSON.parse(row["positions"] as string) as number[];
  }

  loadBlockMaxInto(blockMaxIndex: { _blockMaxes: Map<string, number[]> }): void {
    for (const field of this._knownFields) {
      const bmTbl = this._blockmaxTableName(field);
      const rows = this._conn.query(
        `SELECT term, block_idx, max_score FROM "${bmTbl}" ORDER BY term, block_idx`,
      );

      let currentTerm: string | null = null;
      let scores: number[] = [];
      for (const row of rows) {
        const term = row["term"] as string;
        if (term !== currentTerm) {
          if (currentTerm !== null) {
            blockMaxIndex._blockMaxes.set(
              `${this._tableName}\0${field}\0${currentTerm}`,
              scores,
            );
          }
          currentTerm = term;
          scores = [];
        }
        scores.push(row["max_score"] as number);
      }
      if (currentTerm !== null) {
        blockMaxIndex._blockMaxes.set(
          `${this._tableName}\0${field}\0${currentTerm}`,
          scores,
        );
      }
    }
  }
}
