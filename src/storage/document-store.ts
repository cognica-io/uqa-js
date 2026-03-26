//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- MemoryDocumentStore
// 1:1 port of uqa/storage/document_store.py

import type { DocId, FieldName, PathExpr } from "../core/types.js";
import { DocumentStore } from "./abc/document-store.js";

// Simple path evaluation (full HierarchicalDocument in Phase 13)
function evalPathOnDoc(doc: Record<string, unknown>, path: PathExpr): unknown {
  let current: unknown = doc;
  for (const component of path) {
    if (current === null || current === undefined) return undefined;
    if (typeof component === "number") {
      if (!Array.isArray(current)) return undefined;
      current = (current as unknown[])[component];
    } else {
      // Implicit array wildcard: if current is array and component is string,
      // map over array elements extracting named field from each dict
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

export class MemoryDocumentStore extends DocumentStore {
  private _documents: Map<DocId, Record<string, unknown>>;

  constructor() {
    super();
    this._documents = new Map();
  }

  put(docId: DocId, document: Record<string, unknown>): void {
    this._documents.set(docId, document);
  }

  get(docId: DocId): Record<string, unknown> | null {
    return this._documents.get(docId) ?? null;
  }

  delete(docId: DocId): void {
    this._documents.delete(docId);
  }

  clear(): void {
    this._documents.clear();
  }

  getField(docId: DocId, field: FieldName): unknown {
    const doc = this._documents.get(docId);
    if (doc === undefined) return undefined;
    return doc[field];
  }

  getFieldsBulk(docIds: DocId[], field: FieldName): Map<DocId, unknown> {
    const result = new Map<DocId, unknown>();
    for (const docId of docIds) {
      const doc = this._documents.get(docId);
      result.set(docId, doc !== undefined ? doc[field] : undefined);
    }
    return result;
  }

  hasValue(field: FieldName, value: unknown): boolean {
    for (const doc of this._documents.values()) {
      if (doc[field] === value) return true;
    }
    return false;
  }

  evalPath(docId: DocId, path: PathExpr): unknown {
    const doc = this._documents.get(docId);
    if (doc === undefined) return undefined;
    return evalPathOnDoc(doc, path);
  }

  get docIds(): Set<DocId> {
    return new Set(this._documents.keys());
  }

  get length(): number {
    return this._documents.size;
  }

  maxDocId(): number {
    let max = -1;
    for (const id of this._documents.keys()) {
      if (id > max) max = id;
    }
    return max;
  }

  putBulk(docs: Array<[DocId, Record<string, unknown>]>): void {
    for (const [docId, document] of docs) {
      this._documents.set(docId, document);
    }
  }

  deleteBulk(docIds: DocId[]): void {
    for (const docId of docIds) {
      this._documents.delete(docId);
    }
  }
}
