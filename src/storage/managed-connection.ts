//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- ManagedConnection wrapper around sql.js Database
// 1:1 port of uqa/storage/managed_connection.py

export interface SQLiteDatabase {
  run(sql: string, params?: unknown[]): void;
  exec(sql: string, params?: unknown[]): { columns: string[]; values: unknown[][] }[];
  getAsObject(sql: string, params?: unknown[]): Record<string, unknown>;
  export(): Uint8Array;
  close(): void;
}

export class ManagedConnection {
  private _db: SQLiteDatabase;
  private _inTransaction: boolean;
  private readonly _dbPath: string | null;

  constructor(db: SQLiteDatabase, dbPath?: string | null) {
    this._db = db;
    this._inTransaction = false;
    this._dbPath = dbPath ?? null;
  }

  get dbPath(): string | null {
    return this._dbPath;
  }

  get inTransaction(): boolean {
    return this._inTransaction;
  }

  execute(sql: string, params?: unknown[]): void {
    this._db.run(sql, params);
  }

  query(sql: string, params?: unknown[]): Record<string, unknown>[] {
    const results = this._db.exec(sql, params);
    if (results.length === 0) return [];

    const first = results[0]!;
    const columns = first.columns;
    const rows: Record<string, unknown>[] = [];
    for (const row of first.values) {
      const obj: Record<string, unknown> = {};
      for (let i = 0; i < columns.length; i++) {
        obj[columns[i]!] = row[i];
      }
      rows.push(obj);
    }
    return rows;
  }

  queryOne(sql: string, params?: unknown[]): Record<string, unknown> | null {
    // sql.js getAsObject returns {} for no results, so use query instead
    // for consistent behavior
    const rows = this.query(sql, params);
    if (rows.length === 0) return null;
    return rows[0]!;
  }

  commit(): void {
    if (this._inTransaction) {
      this._db.run("COMMIT");
      this._inTransaction = false;
    }
  }

  rollback(): void {
    if (this._inTransaction) {
      this._db.run("ROLLBACK");
      this._inTransaction = false;
    }
  }

  beginTransaction(): void {
    if (!this._inTransaction) {
      this._db.run("BEGIN TRANSACTION");
      this._inTransaction = true;
    }
  }

  commitTransaction(): void {
    this.commit();
  }

  rollbackTransaction(): void {
    this.rollback();
  }

  savepoint(name: string): void {
    this._db.run(`SAVEPOINT "${name}"`);
  }

  releaseSavepoint(name: string): void {
    this._db.run(`RELEASE SAVEPOINT "${name}"`);
  }

  rollbackToSavepoint(name: string): void {
    this._db.run(`ROLLBACK TO SAVEPOINT "${name}"`);
  }

  /**
   * Execute a read query on a thread-local connection (Python: read_fetchall).
   * In the browser, there is no threading so this is identical to query().
   */
  readFetchall(sql: string, params?: unknown[]): Record<string, unknown>[] {
    return this.query(sql, params);
  }

  /**
   * Execute a read query on a thread-local connection (Python: read_fetchone).
   */
  readFetchone(sql: string, params?: unknown[]): Record<string, unknown> | null {
    return this.queryOne(sql, params);
  }

  /**
   * Execute a write query and return all rows atomically (Python: execute_fetchall).
   */
  executeFetchall(sql: string, params?: unknown[]): Record<string, unknown>[] {
    return this.query(sql, params);
  }

  /**
   * Execute a write query and return one row atomically (Python: execute_fetchone).
   */
  executeFetchone(sql: string, params?: unknown[]): Record<string, unknown> | null {
    return this.queryOne(sql, params);
  }

  /**
   * Execute a statement multiple times with different parameter sets.
   */
  executemany(sql: string, paramSets: unknown[][]): void {
    for (const params of paramSets) {
      this._db.run(sql, params);
    }
  }

  /**
   * Execute multiple SQL statements in a script (separated by semicolons).
   */
  executescript(script: string): void {
    // sql.js exec() can handle multiple statements separated by semicolons
    const statements = script
      .split(";")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    for (const stmt of statements) {
      this._db.run(stmt);
    }
  }

  /**
   * Return the underlying database handle for direct access.
   */
  get db(): SQLiteDatabase {
    return this._db;
  }

  /**
   * Export the database to a Uint8Array for persistence.
   */
  exportDatabase(): Uint8Array {
    return this._db.export();
  }

  close(): void {
    if (this._inTransaction) {
      this.rollback();
    }
    this._db.close();
  }
}
