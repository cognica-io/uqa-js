//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- SQLite-backed DocumentStore
// 1:1 port of uqa/storage/sqlite_document_store.py

import type { DocId, FieldName, PathExpr } from "../core/types.js";
import { DocumentStore } from "./abc/document-store.js";
import type { ManagedConnection } from "./managed-connection.js";

// Path evaluation with implicit array wildcard (mirrors MemoryDocumentStore)
function evalPathOnDoc(doc: Record<string, unknown>, path: PathExpr): unknown {
  let current: unknown = doc;
  for (const component of path) {
    if (current === null || current === undefined) return undefined;
    if (typeof component === "number") {
      if (!Array.isArray(current)) return undefined;
      current = (current as unknown[])[component];
    } else {
      // Implicit array wildcard: if current is array and component is string,
      // map over array elements extracting named field from each object
      if (Array.isArray(current)) {
        current = (current as unknown[]).map(
          (item) => (item as Record<string, unknown>)[component],
        );
      } else if (typeof current === "object") {
        current = (current as Record<string, unknown>)[component];
      } else {
        return undefined;
      }
    }
  }
  return current;
}

export class SQLiteDocumentStore extends DocumentStore {
  private _conn: ManagedConnection;
  private _tableName: string;
  private _columns: [string, string][];
  private _cachedDocIds: Set<DocId> | null;
  private _cachedLength: number | null;

  constructor(conn: ManagedConnection, tableName: string, columns: [string, string][]) {
    super();
    this._conn = conn;
    this._tableName = tableName;
    this._columns = columns;
    this._cachedDocIds = null;
    this._cachedLength = null;
    this._ensureTable();
  }

  private _ensureTable(): void {
    const colDefs = this._columns.map(([name, type]) => `"${name}" ${type}`).join(", ");
    this._conn.execute(
      `CREATE TABLE IF NOT EXISTS "${this._tableName}" (
        _doc_id INTEGER PRIMARY KEY,
        _doc_json TEXT NOT NULL${colDefs.length > 0 ? ", " + colDefs : ""}
      )`,
    );
  }

  private _invalidateCache(): void {
    this._cachedDocIds = null;
    this._cachedLength = null;
  }

  put(docId: DocId, document: Record<string, unknown>): void {
    this._invalidateCache();
    const json = JSON.stringify(document);
    this._conn.execute(
      `INSERT OR REPLACE INTO "${this._tableName}" (_doc_id, _doc_json) VALUES (?, ?)`,
      [docId, json],
    );
  }

  get(docId: DocId): Record<string, unknown> | null {
    const row = this._conn.queryOne(
      `SELECT _doc_json FROM "${this._tableName}" WHERE _doc_id = ?`,
      [docId],
    );
    if (row === null) return null;
    return JSON.parse(row["_doc_json"] as string) as Record<string, unknown>;
  }

  delete(docId: DocId): void {
    this._invalidateCache();
    this._conn.execute(`DELETE FROM "${this._tableName}" WHERE _doc_id = ?`, [docId]);
  }

  clear(): void {
    this._invalidateCache();
    this._conn.execute(`DELETE FROM "${this._tableName}"`);
  }

  getField(docId: DocId, field: FieldName): unknown {
    const doc = this.get(docId);
    if (doc === null) return undefined;
    return doc[field];
  }

  getFieldsBulk(docIds: DocId[], field: FieldName): Map<DocId, unknown> {
    const result = new Map<DocId, unknown>();
    for (const docId of docIds) {
      const doc = this.get(docId);
      result.set(docId, doc !== null ? doc[field] : undefined);
    }
    return result;
  }

  hasValue(field: FieldName, value: unknown): boolean {
    // Full scan through SQLite rows
    const rows = this._conn.query(`SELECT _doc_json FROM "${this._tableName}"`);
    for (const row of rows) {
      const doc = JSON.parse(row["_doc_json"] as string) as Record<string, unknown>;
      if (doc[field] === value) return true;
    }
    return false;
  }

  evalPath(docId: DocId, path: PathExpr): unknown {
    const doc = this.get(docId);
    if (doc === null) return undefined;
    return evalPathOnDoc(doc, path);
  }

  get docIds(): Set<DocId> {
    if (this._cachedDocIds !== null) return this._cachedDocIds;
    const rows = this._conn.query(
      `SELECT _doc_id FROM "${this._tableName}" ORDER BY _doc_id`,
    );
    const ids = new Set<DocId>();
    for (const row of rows) {
      ids.add(row["_doc_id"] as number);
    }
    this._cachedDocIds = ids;
    return ids;
  }

  get length(): number {
    if (this._cachedLength !== null) return this._cachedLength;
    const row = this._conn.queryOne(`SELECT COUNT(*) AS cnt FROM "${this._tableName}"`);
    const count = row !== null ? (row["cnt"] as number) : 0;
    this._cachedLength = count;
    return count;
  }

  /**
   * Return the maximum document ID in the store, or -1 if empty.
   */
  maxDocId(): number {
    const row = this._conn.queryOne(
      `SELECT MAX(_doc_id) AS max_id FROM "${this._tableName}"`,
    );
    if (row === null || row["max_id"] === null || row["max_id"] === undefined)
      return -1;
    return row["max_id"] as number;
  }

  /**
   * Iterate over all documents in the store.
   * Yields [docId, document] pairs in ascending doc_id order.
   */
  *iterAll(): Generator<[DocId, Record<string, unknown>]> {
    const rows = this._conn.query(
      `SELECT _doc_id, _doc_json FROM "${this._tableName}" ORDER BY _doc_id`,
    );
    for (const row of rows) {
      const docId = row["_doc_id"] as number;
      const doc = JSON.parse(row["_doc_json"] as string) as Record<string, unknown>;
      yield [docId, doc];
    }
  }

  /**
   * Bulk insert multiple documents. More efficient than individual puts
   * because it batches within a transaction.
   */
  putBulk(entries: Array<[DocId, Record<string, unknown>]>): void {
    if (entries.length === 0) return;
    this._invalidateCache();
    const wasInTransaction = this._conn.inTransaction;
    if (!wasInTransaction) {
      this._conn.beginTransaction();
    }
    try {
      for (const [docId, document] of entries) {
        const json = JSON.stringify(document);
        this._conn.execute(
          `INSERT OR REPLACE INTO "${this._tableName}" (_doc_id, _doc_json) VALUES (?, ?)`,
          [docId, json],
        );
      }
      if (!wasInTransaction) {
        this._conn.commit();
      }
    } catch (e) {
      if (!wasInTransaction) {
        this._conn.rollback();
      }
      throw e;
    }
  }

  /**
   * Bulk delete multiple documents by their IDs.
   */
  deleteBulk(docIds: DocId[]): void {
    if (docIds.length === 0) return;
    this._invalidateCache();
    const wasInTransaction = this._conn.inTransaction;
    if (!wasInTransaction) {
      this._conn.beginTransaction();
    }
    try {
      for (const docId of docIds) {
        this._conn.execute(`DELETE FROM "${this._tableName}" WHERE _doc_id = ?`, [
          docId,
        ]);
      }
      if (!wasInTransaction) {
        this._conn.commit();
      }
    } catch (e) {
      if (!wasInTransaction) {
        this._conn.rollback();
      }
      throw e;
    }
  }

  /**
   * Bulk get multiple documents by their IDs.
   */
  getBulk(docIds: DocId[]): Map<DocId, Record<string, unknown>> {
    const result = new Map<DocId, Record<string, unknown>>();
    for (const docId of docIds) {
      const doc = this.get(docId);
      if (doc !== null) {
        result.set(docId, doc);
      }
    }
    return result;
  }

  /**
   * Check whether a document with the given ID exists.
   */
  has(docId: DocId): boolean {
    const row = this._conn.queryOne(
      `SELECT 1 AS found FROM "${this._tableName}" WHERE _doc_id = ?`,
      [docId],
    );
    return row !== null;
  }

  /**
   * Return all documents as a Map.
   */
  toMap(): Map<DocId, Record<string, unknown>> {
    const result = new Map<DocId, Record<string, unknown>>();
    for (const [docId, doc] of this.iterAll()) {
      result.set(docId, doc);
    }
    return result;
  }
}
