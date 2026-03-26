//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Index abstract base class
// 1:1 port of uqa/storage/index_abc.py
// Note: SQLite connection type is a placeholder; concrete in Phase 11.

import type { Predicate } from "../core/types.js";
import type { PostingList } from "../core/posting-list.js";
import type { IndexDef } from "./index-types.js";

// Placeholder for SQLite connection (Phase 11: sql.js)
export type SQLiteConnection = unknown;

export abstract class Index {
  protected readonly _indexDef: IndexDef;
  protected readonly _conn: SQLiteConnection;

  constructor(indexDef: IndexDef, conn: SQLiteConnection) {
    this._indexDef = indexDef;
    this._conn = conn;
  }

  get indexDef(): IndexDef {
    return this._indexDef;
  }

  abstract scan(predicate: Predicate): PostingList;
  abstract estimateCardinality(predicate: Predicate): number;
  abstract scanCost(predicate: Predicate): number;
  abstract build(): void;
  abstract drop(): void;
}
