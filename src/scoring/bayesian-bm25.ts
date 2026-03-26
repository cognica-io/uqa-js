//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Bayesian BM25 scorer
// 1:1 port of uqa/scoring/bayesian_bm25.py
// Delegates to bayesian-bm25 npm package

import { BayesianProbabilityTransform, logOddsConjunction } from "bayesian-bm25";
import type { IndexStats } from "../core/types.js";
import type { BM25Params } from "./bm25.js";
import { BM25Scorer, createBM25Params } from "./bm25.js";

export interface BayesianBM25Params {
  readonly bm25: BM25Params;
  readonly alpha: number;
  readonly beta: number;
  readonly baseRate: number;
}

export function createBayesianBM25Params(
  opts?: Partial<BayesianBM25Params>,
): BayesianBM25Params {
  return {
    bm25: opts?.bm25 ?? createBM25Params(),
    alpha: opts?.alpha ?? 1.0,
    beta: opts?.beta ?? 0.0,
    baseRate: opts?.baseRate ?? 0.5,
  };
}

export class BayesianBM25Scorer {
  private readonly _bm25: BM25Scorer;
  private readonly _transform: BayesianProbabilityTransform;

  constructor(params: BayesianBM25Params, indexStats: IndexStats) {
    this._bm25 = new BM25Scorer(params.bm25, indexStats);
    this._transform = new BayesianProbabilityTransform(
      params.alpha,
      params.beta,
      params.baseRate === 0.5 ? null : params.baseRate,
    );
  }

  get bm25(): BM25Scorer {
    return this._bm25;
  }

  idf(docFreq: number): number {
    return this._bm25.idf(docFreq);
  }

  score(termFreq: number, docLength: number, docFreq: number): number {
    return this.scoreWithIdf(termFreq, docLength, this._bm25.idf(docFreq));
  }

  scoreWithIdf(termFreq: number, docLength: number, idfVal: number): number {
    const raw = this._bm25.scoreWithIdf(termFreq, docLength, idfVal);
    const avgdl = this._bm25.params.k1 > 0 ? docLength : 1;
    const docLenRatio = avgdl > 0 ? docLength / avgdl : 1.0;
    return this._transform.scoreToProbability(raw, termFreq, docLenRatio);
  }

  combineScores(scores: number[]): number {
    if (scores.length === 0) return 0.5;
    if (scores.length === 1) return scores[0]!;
    return logOddsConjunction(scores, 0.0);
  }

  upperBound(docFreq: number): number {
    const bm25Ub = this._bm25.upperBound(docFreq);
    return this._transform.wandUpperBound(bm25Ub);
  }
}
