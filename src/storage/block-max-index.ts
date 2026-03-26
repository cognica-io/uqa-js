//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- BlockMaxIndex
// 1:1 port of uqa/storage/block_max_index.py (in-memory portion)

import type { PostingList } from "../core/posting-list.js";

// BM25Scorer interface (full implementation in scoring module, Phase 4)
export interface BM25ScorerLike {
  score(termFreq: number, docLength: number, docFreq: number): number;
}

export class BlockMaxIndex {
  readonly blockSize: number;
  // key: "${tableName}\x1F${field}\x1F${term}" -> block max scores
  private _blockMaxes: Map<string, number[]>;

  constructor(blockSize = 128) {
    this.blockSize = blockSize;
    this._blockMaxes = new Map();
  }

  private _key(tableName: string, field: string, term: string): string {
    return `${tableName}\x1F${field}\x1F${term}`;
  }

  build(
    postingList: PostingList,
    scorer: BM25ScorerLike,
    field: string,
    term: string,
    tableName = "",
  ): void {
    const entries = postingList.entries;
    const key = this._key(tableName, field, term);

    if (entries.length === 0) {
      this._blockMaxes.set(key, []);
      return;
    }

    const docFreq = entries.length;
    const blockMaxes: number[] = [];

    for (
      let blockStart = 0;
      blockStart < entries.length;
      blockStart += this.blockSize
    ) {
      const blockEnd = Math.min(blockStart + this.blockSize, entries.length);
      let maxScore = 0;

      for (let i = blockStart; i < blockEnd; i++) {
        const entry = entries[i]!;
        const tf =
          entry.payload.positions.length > 0 ? entry.payload.positions.length : 1;
        const s = scorer.score(tf, tf, docFreq);
        if (s > maxScore) maxScore = s;
      }

      blockMaxes.push(maxScore);
    }

    this._blockMaxes.set(key, blockMaxes);
  }

  getBlockMax(field: string, term: string, blockIdx: number, tableName = ""): number {
    const maxes = this._blockMaxes.get(this._key(tableName, field, term));
    if (!maxes || blockIdx >= maxes.length) return 0.0;
    return maxes[blockIdx]!;
  }

  numBlocks(field: string, term: string, tableName = ""): number {
    const maxes = this._blockMaxes.get(this._key(tableName, field, term));
    return maxes ? maxes.length : 0;
  }

  /**
   * Persist the block-max index data to a SQLite database via ManagedConnection.
   */
  saveToSQLite(conn: {
    execute(sql: string, params?: unknown[]): void;
    query(sql: string, params?: unknown[]): Record<string, unknown>[];
  }): void {
    conn.execute(
      `CREATE TABLE IF NOT EXISTS _block_max_index (
        key TEXT PRIMARY KEY,
        block_size INTEGER NOT NULL,
        scores_json TEXT NOT NULL
      )`,
    );

    for (const [key, scores] of this._blockMaxes) {
      conn.execute(
        `INSERT OR REPLACE INTO _block_max_index (key, block_size, scores_json) VALUES (?, ?, ?)`,
        [key, this.blockSize, JSON.stringify(scores)],
      );
    }
  }

  /**
   * Load block-max index data from a SQLite database via ManagedConnection.
   */
  static loadFromSQLite(conn: {
    execute(sql: string, params?: unknown[]): void;
    query(sql: string, params?: unknown[]): Record<string, unknown>[];
  }): BlockMaxIndex {
    // Check if the table exists
    const tableCheck = conn.query(
      `SELECT name FROM sqlite_master WHERE type='table' AND name='_block_max_index'`,
    );
    if (tableCheck.length === 0) {
      return new BlockMaxIndex();
    }

    // Try new schema first
    let rows: Record<string, unknown>[];
    try {
      rows = conn.query(`SELECT key, block_size, scores_json FROM _block_max_index`);
    } catch {
      // Fall back to legacy schema migration
      return BlockMaxIndex._migrateLegacyBlockmax(conn);
    }

    let blockSize = 128;
    const blockMaxes = new Map<string, number[]>();

    for (const row of rows) {
      const key = row["key"] as string;
      blockSize = row["block_size"] as number;
      const scores = JSON.parse(row["scores_json"] as string) as number[];
      blockMaxes.set(key, scores);
    }

    const idx = new BlockMaxIndex(blockSize);
    (idx as unknown as { _blockMaxes: Map<string, number[]> })._blockMaxes = blockMaxes;
    return idx;
  }

  /**
   * Migrate from legacy block-max table schema (table, field, term, block_idx, max_score)
   * to the current compact format.
   */
  private static _migrateLegacyBlockmax(conn: {
    execute(sql: string, params?: unknown[]): void;
    query(sql: string, params?: unknown[]): Record<string, unknown>[];
  }): BlockMaxIndex {
    const idx = new BlockMaxIndex();

    // Check for legacy table with old schema
    let rows: Record<string, unknown>[];
    try {
      rows = conn.query(
        `SELECT table_name, field, term, block_idx, max_score
         FROM _block_max_index
         ORDER BY table_name, field, term, block_idx`,
      );
    } catch {
      return idx;
    }

    const grouped = new Map<string, number[]>();
    for (const row of rows) {
      const tableName = row["table_name"] as string;
      const field = row["field"] as string;
      const term = row["term"] as string;
      const blockIdx = row["block_idx"] as number;
      const maxScore = row["max_score"] as number;
      const key = `${tableName}\x1F${field}\x1F${term}`;
      if (!grouped.has(key)) {
        grouped.set(key, []);
      }
      const arr = grouped.get(key)!;
      while (arr.length <= blockIdx) arr.push(0);
      arr[blockIdx] = maxScore;
    }

    (idx as unknown as { _blockMaxes: Map<string, number[]> })._blockMaxes = grouped;

    // Drop old table and recreate with new schema
    try {
      conn.execute(`DROP TABLE IF EXISTS _block_max_index`);
      idx.saveToSQLite(conn);
    } catch {
      // Migration save failed, but the in-memory index is still valid
    }

    return idx;
  }

  /**
   * Clear all block-max data.
   */
  clear(): void {
    this._blockMaxes.clear();
  }

  /**
   * Return all stored keys.
   */
  keys(): string[] {
    return [...this._blockMaxes.keys()];
  }

  /**
   * Return the global maximum score across all blocks for a given term.
   */
  globalMax(field: string, term: string, tableName = ""): number {
    const maxes = this._blockMaxes.get(this._key(tableName, field, term));
    if (!maxes || maxes.length === 0) return 0.0;
    let max = 0.0;
    for (const s of maxes) {
      if (s > max) max = s;
    }
    return max;
  }
}
