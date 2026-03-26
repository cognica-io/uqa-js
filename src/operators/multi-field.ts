//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- multi-field search operator
// 1:1 port of uqa/operators/multi_field.py

import { logOddsConjunction } from "bayesian-bm25";
import type { DocId, IndexStats } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "./base.js";
import { Operator } from "./base.js";
import { TermOperator } from "./primitive.js";
import { ScoreOperator } from "./primitive.js";
import {
  BayesianBM25Scorer,
  createBayesianBM25Params,
} from "../scoring/bayesian-bm25.js";

export class MultiFieldSearchOperator extends Operator {
  readonly fields: string[];
  readonly query: string;
  readonly weights: number[];

  constructor(fields: string[], query: string, weights?: number[] | null) {
    super();
    this.fields = fields;
    this.query = query;
    this.weights = weights ?? fields.map(() => 1.0);
  }

  execute(context: ExecutionContext): PostingList {
    const idx = context.invertedIndex;
    if (!idx) return new PostingList();

    const stats = idx.stats;
    const signalMaps: Map<DocId, number>[] = [];
    const allDocIds = new Set<DocId>();

    for (const field of this.fields) {
      const termOp = new TermOperator(this.query, field);
      const scorer = new BayesianBM25Scorer(createBayesianBM25Params(), stats);
      const analyzer = idx.getSearchAnalyzer(field);
      const tokens = analyzer.analyze(this.query);
      const scoreOp = new ScoreOperator(scorer, termOp, tokens, field);
      const pl = scoreOp.execute(context);

      const m = new Map<DocId, number>();
      for (const entry of pl) {
        m.set(entry.docId, entry.payload.score);
        allDocIds.add(entry.docId);
      }
      signalMaps.push(m);
    }

    // Normalize weights
    let wSum = 0;
    for (const w of this.weights) wSum += w;
    const normWeights = this.weights.map((w) => w / wSum);

    const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
    for (const docId of allDocIds) {
      const probs: number[] = [];
      for (let i = 0; i < this.fields.length; i++) {
        probs.push(signalMaps[i]!.get(docId) ?? 0.5);
      }
      const fused =
        probs.length === 1 ? probs[0]! : logOddsConjunction(probs, 0.0, normWeights);
      entries.push({ docId, payload: createPayload({ score: fused }) });
    }

    return new PostingList(entries);
  }

  costEstimate(stats: IndexStats): number {
    return stats.totalDocs * this.fields.length;
  }
}
