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
