//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- sparse threshold operator
// 1:1 port of uqa/operators/sparse.py

import type { IndexStats } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "./base.js";
import { Operator } from "./base.js";

export class SparseThresholdOperator extends Operator {
  readonly source: Operator;
  readonly threshold: number;

  constructor(source: Operator, threshold: number) {
    super();
    this.source = source;
    this.threshold = threshold;
  }

  execute(context: ExecutionContext): PostingList {
    const sourcePl = this.source.execute(context);
    const entries: { docId: number; payload: ReturnType<typeof createPayload> }[] = [];

    for (const e of sourcePl) {
      const adjusted = e.payload.score - this.threshold;
      if (adjusted > 0) {
        entries.push({
          docId: e.docId,
          payload: createPayload({
            positions: e.payload.positions as number[],
            score: adjusted,
            fields: e.payload.fields as Record<string, unknown>,
          }),
        });
      }
    }

    return PostingList.fromSorted(entries);
  }

  costEstimate(stats: IndexStats): number {
    return this.source.costEstimate(stats);
  }
}
