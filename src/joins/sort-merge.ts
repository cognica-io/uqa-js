//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- sort-merge join
// 1:1 port of uqa/joins/sort_merge.py

import type { GeneralizedPostingEntry } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { GeneralizedPostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "../operators/base.js";
import { JoinOperator, entryDocId } from "./base.js";

export class SortMergeJoinOperator extends JoinOperator {
  execute(context: ExecutionContext): GeneralizedPostingList {
    const leftEntries = this._getEntries(this.left, context);
    const rightEntries = this._getEntries(this.right, context);

    const lf = this.condition.leftField;
    const rf = this.condition.rightField;

    const getKey = (
      e: { payload: { fields: Readonly<Record<string, unknown>> } },
      field: string,
    ): number | string =>
      ((e.payload.fields as Record<string, unknown>)[field] ?? 0) as number | string;

    const leftSorted = leftEntries.slice().sort((a, b) => {
      const ak = getKey(a, lf);
      const bk = getKey(b, lf);
      return ak < bk ? -1 : ak > bk ? 1 : 0;
    });
    const rightSorted = rightEntries.slice().sort((a, b) => {
      const ak = getKey(a, rf);
      const bk = getKey(b, rf);
      return ak < bk ? -1 : ak > bk ? 1 : 0;
    });

    const result: GeneralizedPostingEntry[] = [];
    let i = 0;
    let j = 0;

    while (i < leftSorted.length && j < rightSorted.length) {
      const leftKey = getKey(leftSorted[i]!, lf);
      const rightKey = getKey(rightSorted[j]!, rf);

      if (leftKey === rightKey) {
        // Collect all matching right entries
        let rightEnd = j;
        while (
          rightEnd < rightSorted.length &&
          getKey(rightSorted[rightEnd]!, rf) === rightKey
        ) {
          rightEnd++;
        }
        // Collect all matching left entries
        const leftStart = i;
        while (i < leftSorted.length && getKey(leftSorted[i]!, lf) === leftKey) {
          i++;
        }
        // Cartesian product
        for (let li = leftStart; li < i; li++) {
          for (let ri = j; ri < rightEnd; ri++) {
            const le = leftSorted[li]!;
            const re = rightSorted[ri]!;
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
        j = rightEnd;
      } else if (leftKey < rightKey) {
        i++;
      } else {
        j++;
      }
    }

    return new GeneralizedPostingList(result);
  }
}
