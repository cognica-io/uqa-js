import { describe, expect, it } from "vitest";
import { ManagedConnection } from "../../src/storage/managed-connection.js";
import type { SQLiteDatabase } from "../../src/storage/managed-connection.js";
import { Transaction } from "../../src/storage/transaction.js";

// ======================================================================
// Mock SQLiteDatabase for testing ManagedConnection without sql.js
// ======================================================================

class MockSQLiteDatabase implements SQLiteDatabase {
  private _tables: Map<string, Record<string, unknown>[]>;
  private _inTransaction: boolean;
  private _savepoints: Map<string, Map<string, Record<string, unknown>[]>>;
  private _txnSnapshot: Map<string, Record<string, unknown>[]> | null;

  constructor() {
    this._tables = new Map();
    this._inTransaction = false;
    this._savepoints = new Map();
    this._txnSnapshot = null;
  }

  run(sql: string, _params?: unknown[]): void {
    const trimmed = sql.trim().toUpperCase();
    if (trimmed === "BEGIN TRANSACTION" || trimmed === "BEGIN") {
      this._inTransaction = true;
      this._txnSnapshot = this._cloneTables();
      return;
    }
    if (trimmed === "COMMIT") {
      this._inTransaction = false;
      this._txnSnapshot = null;
      return;
    }
    if (trimmed === "ROLLBACK") {
      if (this._txnSnapshot) {
        this._tables = this._txnSnapshot;
      }
      this._inTransaction = false;
      this._txnSnapshot = null;
      return;
    }
    if (trimmed.startsWith("SAVEPOINT")) {
      const name = sql.trim().split(/\s+/)[1]!.replace(/"/g, "");
      this._savepoints.set(name, this._cloneTables());
      return;
    }
    if (trimmed.startsWith("RELEASE SAVEPOINT")) {
      const name = sql.trim().split(/\s+/)[2]!.replace(/"/g, "");
      this._savepoints.delete(name);
      return;
    }
    if (trimmed.startsWith("ROLLBACK TO SAVEPOINT")) {
      const name = sql.trim().split(/\s+/)[3]!.replace(/"/g, "");
      const snapshot = this._savepoints.get(name);
      if (snapshot) {
        this._tables = snapshot;
        this._savepoints.delete(name);
      }
      return;
    }
    if (trimmed.startsWith("CREATE TABLE")) {
      const match = sql.match(/CREATE TABLE (\w+)/i);
      if (match) {
        this._tables.set(match[1]!, []);
      }
      return;
    }
    if (trimmed.startsWith("INSERT INTO")) {
      const match = sql.match(/INSERT INTO (\w+)/i);
      if (match) {
        const tableName = match[1]!;
        const valMatch = sql.match(/VALUES\s*\((.+)\)/i);
        if (valMatch) {
          const val = Number(valMatch[1]!.trim());
          const table = this._tables.get(tableName);
          if (table) {
            table.push({ x: val });
          }
        }
      }
      return;
    }
  }

  exec(sql: string, _params?: unknown[]): { columns: string[]; values: unknown[][] }[] {
    const trimmed = sql.trim().toUpperCase();
    if (trimmed.startsWith("SELECT COUNT")) {
      const match = sql.match(/FROM (\w+)/i);
      if (match) {
        const table = this._tables.get(match[1]!);
        return [{ columns: ["count"], values: [[table ? table.length : 0]] }];
      }
    }
    if (trimmed.startsWith("SELECT X") || trimmed.startsWith("SELECT *")) {
      const match = sql.match(/FROM (\w+)/i);
      if (match) {
        const table = this._tables.get(match[1]!);
        if (table && table.length > 0) {
          const values = table.map((row) => [row["x"]]);
          return [{ columns: ["x"], values }];
        }
        return [];
      }
    }
    return [];
  }

  getAsObject(_sql: string, _params?: unknown[]): Record<string, unknown> {
    return {};
  }

  export(): Uint8Array {
    return new Uint8Array();
  }

  close(): void {
    // noop
  }

  private _cloneTables(): Map<string, Record<string, unknown>[]> {
    const clone = new Map<string, Record<string, unknown>[]>();
    for (const [name, rows] of this._tables) {
      clone.set(
        name,
        rows.map((r) => ({ ...r })),
      );
    }
    return clone;
  }

  // Expose for direct verification
  getTableRows(name: string): Record<string, unknown>[] {
    return this._tables.get(name) ?? [];
  }
}

// ======================================================================
// ManagedConnection
// ======================================================================

describe("ManagedConnection", () => {
  it("commit forwarded when no transaction", () => {
    const db = new MockSQLiteDatabase();
    db.run("CREATE TABLE t (x INTEGER)");
    const conn = new ManagedConnection(db);

    conn.execute("INSERT INTO t VALUES (1)");
    conn.commit();

    const rows = db.getTableRows("t");
    expect(rows.length).toBe(1);
    expect(rows[0]!["x"]).toBe(1);
  });

  it("begins and commits transaction", () => {
    const db = new MockSQLiteDatabase();
    db.run("CREATE TABLE t (x INTEGER)");
    const conn = new ManagedConnection(db);

    conn.beginTransaction();
    expect(conn.inTransaction).toBe(true);
    conn.execute("INSERT INTO t VALUES (42)");
    conn.commitTransaction();
    expect(conn.inTransaction).toBe(false);

    const rows = db.getTableRows("t");
    expect(rows.length).toBe(1);
    expect(rows[0]!["x"]).toBe(42);
  });

  it("rollback transaction discards changes", () => {
    const db = new MockSQLiteDatabase();
    db.run("CREATE TABLE t (x INTEGER)");
    const conn = new ManagedConnection(db);

    conn.beginTransaction();
    conn.execute("INSERT INTO t VALUES (42)");
    conn.rollbackTransaction();

    const rows = db.getTableRows("t");
    expect(rows.length).toBe(0);
  });

  it("savepoint and rollback to savepoint", () => {
    const db = new MockSQLiteDatabase();
    db.run("CREATE TABLE t (x INTEGER)");
    const conn = new ManagedConnection(db);

    conn.beginTransaction();
    conn.execute("INSERT INTO t VALUES (1)");
    conn.savepoint("sp1");
    conn.execute("INSERT INTO t VALUES (2)");
    conn.rollbackToSavepoint("sp1");
    conn.commitTransaction();

    const rows = db.getTableRows("t");
    expect(rows.length).toBe(1);
    expect(rows[0]!["x"]).toBe(1);
  });

  it("release savepoint keeps changes", () => {
    const db = new MockSQLiteDatabase();
    db.run("CREATE TABLE t (x INTEGER)");
    const conn = new ManagedConnection(db);

    conn.beginTransaction();
    conn.execute("INSERT INTO t VALUES (1)");
    conn.savepoint("sp1");
    conn.execute("INSERT INTO t VALUES (2)");
    conn.releaseSavepoint("sp1");
    conn.commitTransaction();

    const rows = db.getTableRows("t");
    expect(rows.length).toBe(2);
  });

  it("in-transaction flag is accurate", () => {
    const db = new MockSQLiteDatabase();
    const conn = new ManagedConnection(db);

    expect(conn.inTransaction).toBe(false);
    conn.beginTransaction();
    expect(conn.inTransaction).toBe(true);
    conn.commitTransaction();
    expect(conn.inTransaction).toBe(false);
  });

  it("close rolls back active transaction", () => {
    const db = new MockSQLiteDatabase();
    db.run("CREATE TABLE t (x INTEGER)");
    const conn = new ManagedConnection(db);

    conn.beginTransaction();
    conn.execute("INSERT INTO t VALUES (1)");
    conn.close();

    // After close with active transaction, changes should be rolled back
    expect(db.getTableRows("t").length).toBe(0);
  });
});

// ======================================================================
// Transaction class
// ======================================================================

describe("Transaction", () => {
  it("commit persists changes", () => {
    const db = new MockSQLiteDatabase();
    db.run("CREATE TABLE t (x INTEGER)");
    const conn = new ManagedConnection(db);

    const txn = new Transaction(conn);
    expect(txn.active).toBe(true);
    conn.execute("INSERT INTO t VALUES (1)");
    txn.commit();
    expect(txn.active).toBe(false);

    const rows = db.getTableRows("t");
    expect(rows.length).toBe(1);
    expect(rows[0]!["x"]).toBe(1);
  });

  it("rollback discards changes", () => {
    const db = new MockSQLiteDatabase();
    db.run("CREATE TABLE t (x INTEGER)");
    const conn = new ManagedConnection(db);

    const txn = new Transaction(conn);
    conn.execute("INSERT INTO t VALUES (1)");
    txn.rollback();
    expect(txn.active).toBe(false);

    const rows = db.getTableRows("t");
    expect(rows.length).toBe(0);
  });

  it("savepoint within transaction", () => {
    const db = new MockSQLiteDatabase();
    db.run("CREATE TABLE t (x INTEGER)");
    const conn = new ManagedConnection(db);

    const txn = new Transaction(conn);
    conn.execute("INSERT INTO t VALUES (1)");
    txn.savepoint("sp1");
    conn.execute("INSERT INTO t VALUES (2)");
    txn.rollbackTo("sp1");
    txn.commit();

    const rows = db.getTableRows("t");
    expect(rows.length).toBe(1);
    expect(rows[0]!["x"]).toBe(1);
  });

  it("savepoint throws when transaction is not active", () => {
    const db = new MockSQLiteDatabase();
    const conn = new ManagedConnection(db);

    const txn = new Transaction(conn);
    txn.commit();
    expect(() => txn.savepoint("sp1")).toThrow("already finished");
  });

  it("rollbackTo throws when transaction is not active", () => {
    const db = new MockSQLiteDatabase();
    const conn = new ManagedConnection(db);

    const txn = new Transaction(conn);
    txn.commit();
    expect(() => txn.rollbackTo("sp1")).toThrow("already finished");
  });

  it("releaseSavepoint throws when transaction is not active", () => {
    const db = new MockSQLiteDatabase();
    const conn = new ManagedConnection(db);

    const txn = new Transaction(conn);
    txn.commit();
    expect(() => txn.releaseSavepoint("sp1")).toThrow("already finished");
  });
});

// ======================================================================
// SQL Transaction Statements (via Engine)
// ======================================================================

describe("Transaction integration", () => {
  it("Transaction wraps ManagedConnection", () => {
    const db = new MockSQLiteDatabase();
    db.run("CREATE TABLE t (x INTEGER)");
    const conn = new ManagedConnection(db);

    // Begin, insert, commit via Transaction
    const txn = new Transaction(conn);
    expect(txn.active).toBe(true);
    conn.execute("INSERT INTO t VALUES (100)");
    txn.commit();
    expect(txn.active).toBe(false);

    const rows = db.getTableRows("t");
    expect(rows.length).toBe(1);
    expect(rows[0]!["x"]).toBe(100);
  });

  it("nested savepoints within transaction", () => {
    const db = new MockSQLiteDatabase();
    db.run("CREATE TABLE t (x INTEGER)");
    const conn = new ManagedConnection(db);

    const txn = new Transaction(conn);
    conn.execute("INSERT INTO t VALUES (1)");
    txn.savepoint("sp1");
    conn.execute("INSERT INTO t VALUES (2)");
    txn.savepoint("sp2");
    conn.execute("INSERT INTO t VALUES (3)");
    txn.rollbackTo("sp2");
    // After rolling back sp2, only rows 1 and 2 remain
    txn.commit();

    const rows = db.getTableRows("t");
    expect(rows.length).toBe(2);
  });
});
