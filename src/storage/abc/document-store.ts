//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- DocumentStore abstract interface
// 1:1 port of uqa/storage/abc/document_store.py
//
// A document store maps DocId keys to dict values and supports
// field-level access, bulk retrieval, and hierarchical path evaluation.
// Concrete implementations include in-memory and SQLite-backed stores.

import type { DocId, FieldName, PathExpr } from "../../core/types.js";

/**
 * Abstract interface for document storage backends.
 *
 * A document store maps DocId keys to dict values and supports
 * field-level access, bulk retrieval, and hierarchical path evaluation.
 * Concrete implementations include in-memory and SQLite-backed stores.
 */
export abstract class DocumentStore {
  /** Insert or replace a document keyed by docId. */
  abstract put(docId: DocId, document: Record<string, unknown>): void;

  /** Return the document as a dict, or null if absent. */
  abstract get(docId: DocId): Record<string, unknown> | null;

  /** Delete a document. No error if docId does not exist. */
  abstract delete(docId: DocId): void;

  /** Remove all documents. */
  abstract clear(): void;

  /** Return a single field value, or null/undefined if absent. */
  abstract getField(docId: DocId, field: FieldName): unknown;

  /** Return field values for multiple docIds in a single call. */
  abstract getFieldsBulk(docIds: DocId[], field: FieldName): Map<DocId, unknown>;

  /** Return true if any document has the given field equal to value. */
  abstract hasValue(field: FieldName, value: unknown): boolean;

  /** Evaluate a hierarchical path expression against a document. */
  abstract evalPath(docId: DocId, path: PathExpr): unknown;

  /** Return the set of all stored document IDs. */
  abstract get docIds(): Set<DocId>;

  /** Return the number of stored documents. */
  abstract get length(): number;

  /** Return the maximum document ID in the store, or -1 if empty. */
  abstract maxDocId(): number;

  /** Bulk insert or replace multiple documents. */
  abstract putBulk(docs: Array<[DocId, Record<string, unknown>]>): void;

  /** Bulk delete multiple documents. */
  abstract deleteBulk(docIds: DocId[]): void;

  /**
   * Yield all (docId, document) pairs in ID order.
   *
   * The default implementation fetches each document individually.
   * SQLite-backed stores override this with a single query.
   */
  *iterAll(): IterableIterator<[DocId, Record<string, unknown>]> {
    const sortedIds = [...this.docIds].sort((a, b) => a - b);
    for (const docId of sortedIds) {
      const doc = this.get(docId);
      if (doc !== null) {
        yield [docId, doc];
      }
    }
  }
}
