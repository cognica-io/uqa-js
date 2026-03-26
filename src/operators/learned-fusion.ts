//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- learned fusion operator
// 1:1 port of uqa/operators/learned_fusion.py

import type { DocId, IndexStats } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "./base.js";
import { Operator } from "./base.js";
import { coverageBasedDefault } from "./hybrid.js";

// Learned fusion interface (from bayesian-bm25 LearnableLogOddsWeights)
export interface LearnedFusionLike {
  fuse(probs: number[]): number;
}

export class LearnedFusionOperator extends Operator {
  readonly signals: Operator[];
  readonly learned: LearnedFusionLike;

  constructor(signals: Operator[], learned: LearnedFusionLike) {
    super();
    this.signals = signals;
    this.learned = learned;
  }

  execute(context: ExecutionContext): PostingList {
    const signalResults = this.signals.map((s) => s.execute(context));
    const scoreMaps: Map<DocId, number>[] = [];
    const allDocIds = new Set<DocId>();
    let numDocs = 0;

    for (const pl of signalResults) {
      const m = new Map<DocId, number>();
      for (const e of pl) {
        m.set(e.docId, e.payload.score);
        allDocIds.add(e.docId);
      }
      scoreMaps.push(m);
      numDocs = Math.max(numDocs, m.size);
    }

    const defaults = scoreMaps.map((m) => coverageBasedDefault(m.size, numDocs));

    const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
    for (const docId of allDocIds) {
      const probs: number[] = [];
      for (let s = 0; s < scoreMaps.length; s++) {
        probs.push(scoreMaps[s]!.get(docId) ?? defaults[s]!);
      }
      const fused = this.learned.fuse(probs);
      entries.push({ docId, payload: createPayload({ score: fused }) });
    }

    return new PostingList(entries);
  }

  costEstimate(stats: IndexStats): number {
    let sum = 0;
    for (const s of this.signals) sum += s.costEstimate(stats);
    return sum;
  }
}
