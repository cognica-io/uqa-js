//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Transaction wrapper
// 1:1 port of uqa/storage/transaction.py

import type { ManagedConnection } from "./managed-connection.js";

// -- Table snapshot helpers for InMemoryTransaction -------------------------

interface TableSnapshot {
  documents: Map<number, Record<string, unknown>>;
  nextId: number;
  uniqueIndexes: Map<string, Map<unknown, number>>;
}

function _deepCopyMap<K, V>(src: Map<K, V>): Map<K, V> {
  const dst = new Map<K, V>();
  for (const [k, v] of src) {
    if (v && typeof v === "object" && !Array.isArray(v)) {
      dst.set(k, structuredClone(v));
    } else {
      dst.set(k, v);
    }
  }
  return dst;
}

function _snapshotTables(tables: Iterable<[string, unknown]>): Map<string, TableSnapshot> {
  const snapshots = new Map<string, TableSnapshot>();
  for (const [key, table] of tables) {
    const tbl = table as Record<string, unknown>;
    const store = tbl.documentStore as Record<string, unknown>;
    const docsMap = (store as { _documents?: Map<number, Record<string, unknown>> })
      ._documents;
    const docs = docsMap ? _deepCopyMap(docsMap) : new Map<number, Record<string, unknown>>();
    const uniqueRaw = (tbl as { _uniqueIndexes?: Map<string, Map<unknown, number>> })
      ._uniqueIndexes;
    const unique = uniqueRaw
      ? new Map<string, Map<unknown, number>>(
          [...uniqueRaw.entries()].map(([k, v]) => [k, new Map(v)]),
        )
      : new Map<string, Map<unknown, number>>();
    snapshots.set(key, {
      documents: docs,
      nextId: (tbl as { _nextId?: number })._nextId ?? 1,
      uniqueIndexes: unique,
    });
  }
  return snapshots;
}

function _restoreTables(
  tables: { get(name: string): unknown },
  snapshots: Map<string, TableSnapshot>,
): void {
  for (const [key, snap] of snapshots) {
    const table = tables.get(key) as Record<string, unknown> | undefined;
    if (!table) continue;
    const store = table.documentStore as Record<string, unknown>;
    if (store && "_documents" in store) {
      (store as { _documents: Map<number, Record<string, unknown>> })._documents =
        snap.documents;
    }
    (table as { _nextId: number })._nextId = snap.nextId;
    (table as { _uniqueIndexes: Map<string, Map<unknown, number>> })._uniqueIndexes =
      snap.uniqueIndexes;
    (table as { _uniqueIndexesBuilt: boolean })._uniqueIndexesBuilt =
      snap.uniqueIndexes.size > 0;
  }
}

// -- Persistent Transaction -------------------------------------------------

export class Transaction {
  private _conn: ManagedConnection;
  private _finished: boolean;

  constructor(conn: ManagedConnection) {
    this._conn = conn;
    this._finished = false;
    this._conn.beginTransaction();
  }

  get active(): boolean {
    return !this._finished;
  }

  commit(): void {
    if (this._finished) {
      throw new Error("Transaction already finished");
    }
    this._conn.commitTransaction();
    this._finished = true;
  }

  rollback(): void {
    if (this._finished) {
      throw new Error("Transaction already finished");
    }
    this._conn.rollbackTransaction();
    this._finished = true;
  }

  savepoint(name: string): void {
    if (this._finished) {
      throw new Error("Transaction already finished");
    }
    this._conn.savepoint(name);
  }

  releaseSavepoint(name: string): void {
    if (this._finished) {
      throw new Error("Transaction already finished");
    }
    this._conn.releaseSavepoint(name);
  }

  rollbackTo(name: string): void {
    if (this._finished) {
      throw new Error("Transaction already finished");
    }
    this._conn.rollbackToSavepoint(name);
  }

  /** Auto-rollback disposable pattern (analogous to Python __enter__/__exit__). */
  autoRollback(): void {
    if (!this._finished) {
      this.rollback();
    }
  }
}

// -- InMemoryTransaction ----------------------------------------------------

/**
 * Transaction for in-memory engines with real rollback support.
 *
 * On construction, snapshots all table document stores.  `rollback()`
 * restores the snapshot, discarding all writes.  `commit()` discards
 * the snapshot, making writes permanent.  Savepoints create nested
 * snapshots on a stack.
 */
export class InMemoryTransaction {
  private _tables: {
    get(name: string): unknown;
    qualifiedItems?(): Iterable<[string, string, unknown]>;
    items?(): Iterable<[string, unknown]>;
  };
  private _finished: boolean;
  private _snapshot: Map<string, TableSnapshot>;
  private _savepoints: Map<string, Map<string, TableSnapshot>>;

  constructor(tables: {
    get(name: string): unknown;
    qualifiedItems?(): Iterable<[string, string, unknown]>;
    items?(): Iterable<[string, unknown]>;
  }) {
    this._tables = tables;
    this._finished = false;
    this._snapshot = _snapshotTables(this._iterableItems());
    this._savepoints = new Map();
  }

  private _iterableItems(): Iterable<[string, unknown]> {
    if (this._tables.qualifiedItems) {
      const entries: [string, unknown][] = [];
      for (const [s, n, t] of this._tables.qualifiedItems()) {
        entries.push([`${s}.${n}`, t]);
      }
      return entries;
    }
    if (this._tables.items) {
      return this._tables.items();
    }
    return [];
  }

  get active(): boolean {
    return !this._finished;
  }

  commit(): void {
    if (this._finished) {
      throw new Error("Transaction already finished");
    }
    this._snapshot = new Map();
    this._savepoints.clear();
    this._finished = true;
  }

  rollback(): void {
    if (this._finished) {
      throw new Error("Transaction already finished");
    }
    _restoreTables(this._tables, this._snapshot);
    this._snapshot = new Map();
    this._savepoints.clear();
    this._finished = true;
  }

  savepoint(name: string): void {
    if (this._finished) {
      throw new Error("Transaction already finished");
    }
    this._savepoints.set(name, _snapshotTables(this._iterableItems()));
  }

  releaseSavepoint(name: string): void {
    if (this._finished) {
      throw new Error("Transaction already finished");
    }
    this._savepoints.delete(name);
  }

  rollbackTo(name: string): void {
    if (this._finished) {
      throw new Error("Transaction already finished");
    }
    const snap = this._savepoints.get(name);
    if (!snap) {
      throw new Error(`Savepoint '${name}' does not exist`);
    }
    _restoreTables(this._tables, snap);
  }

  /** Auto-rollback disposable pattern. */
  autoRollback(): void {
    if (!this._finished) {
      this.rollback();
    }
  }
}
