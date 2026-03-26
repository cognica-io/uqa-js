//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- index type definitions
// 1:1 port of uqa/storage/index_types.py

export type IndexType = "btree" | "inverted" | "hnsw" | "ivf" | "graph" | "rtree";

export interface IndexDef {
  readonly name: string;
  readonly indexType: IndexType;
  readonly tableName: string;
  readonly columns: readonly string[];
  readonly parameters: Readonly<Record<string, unknown>>;
}

export function createIndexDef(
  name: string,
  indexType: IndexType,
  tableName: string,
  columns: readonly string[],
  parameters?: Record<string, unknown>,
): IndexDef {
  return { name, indexType, tableName, columns, parameters: parameters ?? {} };
}
