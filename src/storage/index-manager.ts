//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- IndexManager
// 1:1 port of uqa/storage/index_manager.py
// Note: Catalog dependency deferred to Phase 11. Skeleton provided.

import type { Predicate } from "../core/types.js";
import type { IndexDef } from "./index-types.js";
import type { SQLiteConnection, Index } from "./index-abc.js";
import { BTreeIndex } from "./btree-index.js";

// Placeholder for Catalog (Phase 11)
export type CatalogLike = unknown;

export class IndexManager {
  private readonly _conn: SQLiteConnection;
  /* @internal */ readonly _catalog: CatalogLike;
  private readonly _indexes: Map<string, Index>;

  constructor(conn: SQLiteConnection, catalog: CatalogLike) {
    this._conn = conn;
    this._catalog = catalog;
    this._indexes = new Map();
  }

  createIndex(indexDef: IndexDef): Index {
    if (this._indexes.has(indexDef.name)) {
      throw new Error(`Index "${indexDef.name}" already exists`);
    }
    const index = this._makeIndex(indexDef);
    index.build();
    // catalog.saveIndex(indexDef) -- Phase 11
    this._indexes.set(indexDef.name, index);
    return index;
  }

  dropIndex(name: string): void {
    const index = this._indexes.get(name);
    if (!index) {
      throw new Error(`Index "${name}" not found`);
    }
    index.drop();
    // catalog.dropIndex(name) -- Phase 11
    this._indexes.delete(name);
  }

  dropIndexIfExists(name: string): void {
    if (this._indexes.has(name)) {
      this.dropIndex(name);
    }
  }

  dropIndexesForTable(tableName: string): void {
    const toRemove: string[] = [];
    for (const [name, index] of this._indexes) {
      if (index.indexDef.tableName === tableName) {
        index.drop();
        toRemove.push(name);
      }
    }
    for (const name of toRemove) {
      this._indexes.delete(name);
    }
  }

  findCoveringIndex(
    tableName: string,
    column: string,
    predicate: Predicate,
  ): Index | null {
    let best: Index | null = null;
    let bestCost = Infinity;

    for (const index of this._indexes.values()) {
      const idef = index.indexDef;
      if (idef.tableName === tableName && idef.columns[0] === column) {
        const cost = index.scanCost(predicate);
        if (cost < bestCost) {
          bestCost = cost;
          best = index;
        }
      }
    }

    return best;
  }

  getIndexesForTable(tableName: string): Index[] {
    const result: Index[] = [];
    for (const index of this._indexes.values()) {
      if (index.indexDef.tableName === tableName) {
        result.push(index);
      }
    }
    return result;
  }

  hasIndex(name: string): boolean {
    return this._indexes.has(name);
  }

  private _makeIndex(indexDef: IndexDef): Index {
    if (indexDef.indexType === "btree") {
      return new BTreeIndex(indexDef, this._conn);
    }
    throw new Error(`Unsupported index type: ${indexDef.indexType}`);
  }
}
