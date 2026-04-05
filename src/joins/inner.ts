//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- inner join
// 1:1 port of uqa/joins/inner.py

import type { GeneralizedPostingEntry } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { GeneralizedPostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "../operators/base.js";
import type { JoinEntry } from "./base.js";
import { JoinOperator, entryDocId } from "./base.js";

export class InnerJoinOperator extends JoinOperator {
  execute(context: ExecutionContext): GeneralizedPostingList {
    const leftEntries = this._getEntries(this.left, context);
    const rightEntries = this._getEntries(this.right, context);

    // Build hash index on smaller side
    const leftSmaller = leftEntries.length <= rightEntries.length;
    const buildEntries = leftSmaller ? leftEntries : rightEntries;
    const probeEntries = leftSmaller ? rightEntries : leftEntries;
    const buildField = leftSmaller
      ? this.condition.leftField
      : this.condition.rightField;
    const probeField = leftSmaller
      ? this.condition.rightField
      : this.condition.leftField;

    const index = new Map<unknown, JoinEntry[]>();
    for (const entry of buildEntries) {
      const key = (entry.payload.fields as Record<string, unknown>)[buildField];
      if (key === undefined || key === null) continue;
      let bucket = index.get(key);
      if (!bucket) {
        bucket = [];
        index.set(key, bucket);
      }
      bucket.push(entry);
    }

    this.checkCancelled();
    const result: GeneralizedPostingEntry[] = [];
    for (const probeEntry of probeEntries) {
      const probeKey = (probeEntry.payload.fields as Record<string, unknown>)[
        probeField
      ];
      if (probeKey === undefined || probeKey === null) continue;
      const matches = index.get(probeKey);
      if (!matches) continue;

      for (const buildEntry of matches) {
        const leftE = leftSmaller ? buildEntry : probeEntry;
        const rightE = leftSmaller ? probeEntry : buildEntry;
        const leftId = entryDocId(leftE);
        const rightId = entryDocId(rightE);

        result.push({
          docIds: [leftId, rightId],
          payload: createPayload({
            score: leftE.payload.score + rightE.payload.score,
            fields: {
              ...(leftE.payload.fields as Record<string, unknown>),
              ...(rightE.payload.fields as Record<string, unknown>),
            },
          }),
        });
      }
    }

    return new GeneralizedPostingList(result);
  }
}
