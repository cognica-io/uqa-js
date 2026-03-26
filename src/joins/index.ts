//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- index join (binary search)
// 1:1 port of uqa/joins/index.py

import type { GeneralizedPostingEntry } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { GeneralizedPostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "../operators/base.js";
import type { JoinEntry } from "./base.js";
import { JoinOperator, entryDocId } from "./base.js";

export class IndexJoinOperator extends JoinOperator {
  execute(context: ExecutionContext): GeneralizedPostingList {
    const leftEntries = this._getEntries(this.left, context);
    const rightEntries = this._getEntries(this.right, context);

    // Build sorted index on right
    const keyed: { key: unknown; entry: JoinEntry }[] = [];
    for (const entry of rightEntries) {
      const key = (entry.payload.fields as Record<string, unknown>)[
        this.condition.rightField
      ];
      if (key !== undefined && key !== null) {
        keyed.push({ key, entry });
      }
    }
    keyed.sort((a, b) =>
      (a.key as number) < (b.key as number)
        ? -1
        : (a.key as number) > (b.key as number)
          ? 1
          : 0,
    );
    const rightKeys = keyed.map((k) => k.key);

    const result: GeneralizedPostingEntry[] = [];
    for (const leftEntry of leftEntries) {
      const leftKey = (leftEntry.payload.fields as Record<string, unknown>)[
        this.condition.leftField
      ];
      if (leftKey === undefined || leftKey === null) continue;

      // Binary search for leftKey
      let lo = bisectLeft(rightKeys, leftKey);
      while (lo < rightKeys.length && rightKeys[lo] === leftKey) {
        const rightEntry = keyed[lo]!.entry;
        result.push({
          docIds: [entryDocId(leftEntry), entryDocId(rightEntry)],
          payload: createPayload({
            score: leftEntry.payload.score + rightEntry.payload.score,
            fields: {
              ...(leftEntry.payload.fields as Record<string, unknown>),
              ...(rightEntry.payload.fields as Record<string, unknown>),
            },
          }),
        });
        lo++;
      }
    }

    return new GeneralizedPostingList(result);
  }
}

function bisectLeft(arr: unknown[], target: unknown): number {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if ((arr[mid] as number) < (target as number)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}
