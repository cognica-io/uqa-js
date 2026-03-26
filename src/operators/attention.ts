//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- attention fusion operator
// 1:1 port of uqa/operators/attention.py

import type { DocId, IndexStats } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "./base.js";
import { Operator } from "./base.js";
import { coverageBasedDefault } from "./hybrid.js";

// Attention interface (from bayesian-bm25 AttentionLogOddsWeights / MultiHeadAttentionLogOddsWeights)
export interface AttentionFusionLike {
  fuse(probs: number[], queryFeatures: Float64Array | number[]): number;
}

export class AttentionFusionOperator extends Operator {
  readonly signals: Operator[];
  readonly attention: AttentionFusionLike;
  readonly queryFeatures: Float64Array;

  constructor(
    signals: Operator[],
    attention: AttentionFusionLike,
    queryFeatures: Float64Array,
  ) {
    super();
    this.signals = signals;
    this.attention = attention;
    this.queryFeatures = queryFeatures;
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
      const fused = this.attention.fuse(probs, this.queryFeatures);
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
