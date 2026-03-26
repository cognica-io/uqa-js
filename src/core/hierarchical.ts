//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- HierarchicalDocument
// 1:1 port of uqa/core/hierarchical.py

import type { PathExpr } from "./types.js";

export class HierarchicalDocument {
  readonly docId: number;
  readonly data: unknown;

  constructor(docId: number, data: unknown) {
    this.docId = docId;
    this.data = data;
  }

  /**
   * Evaluate a path expression against this document's data.
   * Uses the same implicit array wildcard logic as MemoryDocumentStore:
   * when the current value is an array and the path component is a string,
   * map over array elements to extract the named field from each.
   */
  evalPath(path: PathExpr): unknown {
    let current: unknown = this.data;
    for (const component of path) {
      if (current === null || current === undefined) return undefined;
      if (typeof component === "number") {
        if (!Array.isArray(current)) return undefined;
        current = (current as unknown[])[component];
      } else {
        // Implicit array wildcard
        if (Array.isArray(current)) {
          current = (current as unknown[]).map(
            (item) => (item as Record<string, unknown>)[component],
          );
        } else if (typeof current === "object") {
          current = (current as Record<string, unknown>)[component];
        } else {
          return undefined;
        }
      }
    }
    return current;
  }
}

/**
 * Project a set of path expressions against a HierarchicalDocument,
 * returning a flat record keyed by the string representation of each path.
 */
export function projectPaths(
  doc: HierarchicalDocument,
  paths: PathExpr[],
): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const path of paths) {
    const key = path.map((c) => String(c)).join(".");
    result[key] = doc.evalPath(path);
  }
  return result;
}

/**
 * Unnest an array field at the given path, producing one HierarchicalDocument
 * per element. Each child document shares the same docId as the parent.
 */
export function unnestArray(
  doc: HierarchicalDocument,
  path: PathExpr,
): HierarchicalDocument[] {
  const arr = doc.evalPath(path);
  if (!Array.isArray(arr)) return [];
  const results: HierarchicalDocument[] = [];
  for (const element of arr) {
    results.push(new HierarchicalDocument(doc.docId, element));
  }
  return results;
}
