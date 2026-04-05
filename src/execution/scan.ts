//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- scan operators (sequential and posting-list based)
// 1:1 port of uqa/execution/scan.py

import type { DocId } from "../core/types.js";
import type { PostingList } from "../core/posting-list.js";
import type { DocumentStore } from "../storage/abc/document-store.js";
import { Batch } from "./batch.js";
import { PhysicalOperator } from "./physical.js";

// ---------------------------------------------------------------------------
// SeqScanOp -- sequential full-table scan from a DocumentStore
// ---------------------------------------------------------------------------

export class SeqScanOp extends PhysicalOperator {
  private readonly _store: DocumentStore;
  private readonly _columns: string[] | null;
  private _iterator: IterableIterator<[DocId, Record<string, unknown>]> | null = null;
  private _batchSize: number;

  /**
   * @param store    The document store to scan.
   * @param columns  If provided, only these columns are emitted.
   *                 null means all columns present in the document.
   * @param batchSize Number of rows per batch (default 1024).
   */
  constructor(store: DocumentStore, columns?: string[] | null, batchSize = 1024) {
    super();
    this._store = store;
    this._columns = columns ?? null;
    this._batchSize = batchSize;
  }

  open(): void {
    this._iterator = this._store.iterAll();
  }

  next(): Batch | null {
    this.checkCancelled();
    if (this._iterator === null) {
      return null;
    }

    const rows: Record<string, unknown>[] = [];
    for (let i = 0; i < this._batchSize; i++) {
      const result = this._iterator.next();
      if (result.done) {
        break;
      }
      const [docId, doc] = result.value;
      if (this._columns !== null) {
        const projected: Record<string, unknown> = { _docId: docId };
        for (const col of this._columns) {
          projected[col] = doc[col] ?? null;
        }
        rows.push(projected);
      } else {
        rows.push({ _docId: docId, ...doc });
      }
    }

    if (rows.length === 0) {
      return null;
    }
    return Batch.fromRows(rows);
  }

  close(): void {
    this._iterator = null;
  }
}

// ---------------------------------------------------------------------------
// PostingListScanOp -- scan from a PostingList, optionally enriching
// from a DocumentStore
// ---------------------------------------------------------------------------

export class PostingListScanOp extends PhysicalOperator {
  private readonly _postingList: PostingList;
  private readonly _store: DocumentStore | null;
  private readonly _columns: string[] | null;
  private _index: number = 0;
  private _batchSize: number;

  /**
   * @param postingList The posting list to scan.
   * @param store       If provided, row fields are enriched from the
   *                    document store. Otherwise only docId and score
   *                    are emitted.
   * @param columns     If provided with a store, only these columns are
   *                    fetched.
   * @param batchSize   Number of rows per batch (default 1024).
   */
  constructor(
    postingList: PostingList,
    store?: DocumentStore | null,
    columns?: string[] | null,
    batchSize = 1024,
  ) {
    super();
    this._postingList = postingList;
    this._store = store ?? null;
    this._columns = columns ?? null;
    this._batchSize = batchSize;
  }

  open(): void {
    this._index = 0;
  }

  next(): Batch | null {
    this.checkCancelled();
    const entries = this._postingList.entries;
    if (this._index >= entries.length) {
      return null;
    }

    const rows: Record<string, unknown>[] = [];
    const end = Math.min(this._index + this._batchSize, entries.length);

    for (let i = this._index; i < end; i++) {
      const entry = entries[i]!;
      const row: Record<string, unknown> = {
        _docId: entry.docId,
        _score: entry.payload.score,
      };

      // Copy payload fields
      for (const [key, value] of Object.entries(entry.payload.fields)) {
        row[key] = value;
      }

      // Enrich from document store
      if (this._store !== null) {
        const doc = this._store.get(entry.docId);
        if (doc !== null) {
          if (this._columns !== null) {
            for (const col of this._columns) {
              if (!(col in row)) {
                row[col] = doc[col] ?? null;
              }
            }
          } else {
            for (const [key, value] of Object.entries(doc)) {
              if (!(key in row)) {
                row[key] = value;
              }
            }
          }
        }
      }

      rows.push(row);
    }

    this._index = end;

    if (rows.length === 0) {
      return null;
    }
    return Batch.fromRows(rows);
  }

  close(): void {
    this._index = 0;
  }
}
