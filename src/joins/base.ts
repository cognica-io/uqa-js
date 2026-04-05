//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- join base types
// 1:1 port of uqa/joins/base.py

import type { CancellationToken } from "../cancel.js";
import type { DocId, GeneralizedPostingEntry, PostingEntry } from "../core/types.js";
import type { GeneralizedPostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "../operators/base.js";

export interface JoinCondition {
  readonly leftField: string;
  readonly rightField: string;
}

export type JoinEntry = PostingEntry | GeneralizedPostingEntry;

export function entryDocId(entry: JoinEntry): DocId {
  if ("docIds" in entry) return entry.docIds[0]!;
  return entry.docId;
}

export abstract class JoinOperator {
  readonly left: unknown;
  readonly right: unknown;
  readonly condition: JoinCondition;
  cancelToken: CancellationToken | null = null;

  constructor(left: unknown, right: unknown, condition: JoinCondition) {
    this.left = left;
    this.right = right;
    this.condition = condition;
  }

  checkCancelled(): void {
    if (this.cancelToken !== null) {
      this.cancelToken.check();
    }
  }

  abstract execute(context: ExecutionContext): GeneralizedPostingList;

  protected _getEntries(source: unknown, context: ExecutionContext): JoinEntry[] {
    if (
      source !== null &&
      typeof source === "object" &&
      "execute" in source &&
      typeof (source as { execute: unknown }).execute === "function"
    ) {
      const result = (
        source as { execute(ctx: ExecutionContext): { entries: JoinEntry[] } }
      ).execute(context);
      return [...result.entries];
    }
    return source as JoinEntry[];
  }
}
