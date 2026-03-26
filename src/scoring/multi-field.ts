//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- multi-field Bayesian scorer
// 1:1 port of uqa/scoring/multi_field.py

import { logOddsConjunction } from "bayesian-bm25";
import type { IndexStats } from "../core/types.js";
import type { BayesianBM25Params } from "./bayesian-bm25.js";
import { BayesianBM25Scorer } from "./bayesian-bm25.js";

export class MultiFieldBayesianScorer {
  private readonly _fieldNames: string[];
  private readonly _scorers: BayesianBM25Scorer[];
  private readonly _weights: number[];

  constructor(
    fieldConfigs: [string, BayesianBM25Params, number][],
    indexStats: IndexStats,
  ) {
    this._fieldNames = [];
    this._scorers = [];
    this._weights = [];
    for (const [name, params, weight] of fieldConfigs) {
      this._fieldNames.push(name);
      this._scorers.push(new BayesianBM25Scorer(params, indexStats));
      this._weights.push(weight);
    }
  }

  scoreDocument(
    _docId: number,
    termFreqPerField: Record<string, number>,
    docLengthPerField: Record<string, number>,
    docFreqPerField: Record<string, number>,
  ): number {
    const probs: number[] = [];
    const weights: number[] = [];

    for (let i = 0; i < this._fieldNames.length; i++) {
      const field = this._fieldNames[i]!;
      const tf = termFreqPerField[field] ?? 0;
      const dl = docLengthPerField[field] ?? 1;
      const df = docFreqPerField[field] ?? 1;

      if (tf === 0) {
        probs.push(0.5);
      } else {
        probs.push(this._scorers[i]!.score(tf, dl, df));
      }
      weights.push(this._weights[i]!);
    }

    if (probs.length === 1) return probs[0]!;

    // Normalize weights
    let wSum = 0;
    for (const w of weights) wSum += w;
    const normWeights = weights.map((w) => w / wSum);

    return logOddsConjunction(probs, 0.0, normWeights);
  }
}
