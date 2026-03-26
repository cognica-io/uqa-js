//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- multi-stage retrieval operator
// 1:1 port of uqa/operators/multi_stage.py

import type { IndexStats } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "./base.js";
import { Operator } from "./base.js";

export class MultiStageOperator extends Operator {
  readonly stages: [Operator, number][];

  constructor(stages: [Operator, number][]) {
    super();
    if (stages.length === 0) {
      throw new Error("MultiStageOperator requires at least one stage");
    }
    this.stages = stages;
  }

  execute(context: ExecutionContext): PostingList {
    if (this.stages.length === 0) return new PostingList();

    let current = this.stages[0]![0].execute(context);
    current = applyCutoff(current, this.stages[0]![1]);

    for (let i = 1; i < this.stages.length; i++) {
      const [op, cutoff] = this.stages[i]!;
      const stageResult = op.execute(context);

      // Re-score surviving candidates
      const stageMap = new Map<number, number>();
      for (const e of stageResult) {
        stageMap.set(e.docId, e.payload.score);
      }

      const entries = current.entries.map((e) => ({
        docId: e.docId,
        payload: createPayload({
          positions: e.payload.positions as number[],
          score: stageMap.get(e.docId) ?? e.payload.score,
          fields: e.payload.fields as Record<string, unknown>,
        }),
      }));
      current = applyCutoff(new PostingList(entries), cutoff);
    }

    return current;
  }

  costEstimate(stats: IndexStats): number {
    let total = 0;
    let cardinality = stats.totalDocs;
    for (const [op, cutoff] of this.stages) {
      total += op.costEstimate(stats) * (cardinality / Math.max(stats.totalDocs, 1));
      cardinality =
        typeof cutoff === "number" && Number.isInteger(cutoff)
          ? Math.min(cutoff, cardinality)
          : cardinality * 0.5;
    }
    return total;
  }

  static applyCutoff(pl: PostingList, cutoff: number): PostingList {
    return applyCutoff(pl, cutoff);
  }
}

function applyCutoff(pl: PostingList, cutoff: number): PostingList {
  if (Number.isInteger(cutoff) && cutoff > 0) {
    return pl.topK(cutoff);
  }
  // Float threshold
  const entries = pl.entries.filter((e) => e.payload.score >= cutoff);
  return new PostingList(entries);
}
