//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- cross join (Cartesian product)
// 1:1 port of uqa/joins/cross.py

import type { GeneralizedPostingEntry } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { GeneralizedPostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "../operators/base.js";
import type { JoinEntry } from "./base.js";
import { entryDocId } from "./base.js";

export class CrossJoinOperator {
  readonly left: unknown;
  readonly right: unknown;

  constructor(left: unknown, right: unknown) {
    this.left = left;
    this.right = right;
  }

  execute(context: ExecutionContext): GeneralizedPostingList {
    const leftEntries = getEntries(this.left, context);
    const rightEntries = getEntries(this.right, context);

    const result: GeneralizedPostingEntry[] = [];
    for (const le of leftEntries) {
      for (const re of rightEntries) {
        result.push({
          docIds: [entryDocId(le), entryDocId(re)],
          payload: createPayload({
            score: le.payload.score + re.payload.score,
            fields: {
              ...(le.payload.fields as Record<string, unknown>),
              ...(re.payload.fields as Record<string, unknown>),
            },
          }),
        });
      }
    }

    return new GeneralizedPostingList(result);
  }
}

function getEntries(source: unknown, context: ExecutionContext): JoinEntry[] {
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
