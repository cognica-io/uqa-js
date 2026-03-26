//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- BTreeIndex
// 1:1 port of uqa/storage/btree_index.py

import {
  Between,
  Equals,
  GreaterThan,
  GreaterThanOrEqual,
  InSet,
  LessThan,
  LessThanOrEqual,
  NotEquals,
  createPayload,
} from "../core/types.js";
import type { Predicate, PostingEntry } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { IndexDef } from "./index-types.js";
import type { SQLiteConnection } from "./index-abc.js";
import { Index } from "./index-abc.js";
import type { ManagedConnection } from "./managed-connection.js";

export class BTreeIndex extends Index {
  /* @internal */ readonly _dataTable: string;

  constructor(indexDef: IndexDef, conn: SQLiteConnection) {
    super(indexDef, conn);
    this._dataTable = `_data_${indexDef.tableName}`;
  }

  private get _mc(): ManagedConnection {
    return this._conn as ManagedConnection;
  }

  build(): void {
    const cols = this._indexDef.columns.map((c) => `"${c}"`).join(", ");
    this._mc.execute(
      `CREATE INDEX IF NOT EXISTS "${this._indexDef.name}" ON "${this._dataTable}" (${cols})`,
    );
    this._mc.commit();
  }

  drop(): void {
    this._mc.execute(`DROP INDEX IF EXISTS "${this._indexDef.name}"`);
    this._mc.commit();
  }

  scan(predicate: Predicate): PostingList {
    const [whereClause, params] = this._predicateToSQL(predicate);
    const sql =
      `SELECT _rowid FROM "${this._dataTable}" ` +
      `WHERE ${whereClause} ORDER BY _rowid`;
    const rows = this._mc.query(sql, params);
    const entries: PostingEntry[] = rows.map((row) => ({
      docId: row["_rowid"] as number,
      payload: createPayload({ score: 0.0 }),
    }));
    return PostingList.fromSorted(entries);
  }

  estimateCardinality(predicate: Predicate): number {
    const [whereClause, params] = this._predicateToSQL(predicate);
    const sql = `SELECT COUNT(*) AS cnt FROM "${this._dataTable}" WHERE ${whereClause}`;
    const row = this._mc.queryOne(sql, params);
    return row !== null ? (row["cnt"] as number) : 0;
  }

  scanCost(predicate: Predicate): number {
    const total = this._totalRows();
    if (total === 0) return 0.0;
    const estimated = this.estimateCardinality(predicate);
    return Math.log2(total) + estimated;
  }

  private _totalRows(): number {
    const row = this._mc.queryOne(`SELECT COUNT(*) AS cnt FROM "${this._dataTable}"`);
    return row !== null ? (row["cnt"] as number) : 0;
  }

  private _predicateToSQL(predicate: Predicate): [string, unknown[]] {
    const colName = this._indexDef.columns[0] ?? "";
    const col = `"${colName}"`;

    if (predicate instanceof Equals) {
      return [`${col} = ?`, [predicate.target]];
    }
    if (predicate instanceof NotEquals) {
      return [`${col} != ?`, [predicate.target]];
    }
    if (predicate instanceof GreaterThan) {
      return [`${col} > ?`, [predicate.target]];
    }
    if (predicate instanceof GreaterThanOrEqual) {
      return [`${col} >= ?`, [predicate.target]];
    }
    if (predicate instanceof LessThan) {
      return [`${col} < ?`, [predicate.target]];
    }
    if (predicate instanceof LessThanOrEqual) {
      return [`${col} <= ?`, [predicate.target]];
    }
    if (predicate instanceof Between) {
      return [`${col} BETWEEN ? AND ?`, [predicate.low, predicate.high]];
    }
    if (predicate instanceof InSet) {
      const vals = [...predicate.values];
      const placeholders = vals.map(() => "?").join(", ");
      return [`${col} IN (${placeholders})`, vals];
    }

    throw new Error(
      `BTreeIndex cannot handle predicate type: ${predicate.constructor.name}`,
    );
  }
}
