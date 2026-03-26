//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- external prior scorer
// 1:1 port of uqa/scoring/external_prior.py
//
// Bayesian BM25 scorer with external prior features (Section 12.2 #6, Paper 3).
//
// Combines the BM25 likelihood with a document-level prior via log-odds
// addition:
//
//     logit(posterior) = logit(likelihood) + logit(prior)
//
// The prior is computed by a user-supplied function that maps document
// fields to a probability in (0, 1).

import { BayesianProbabilityTransform } from "bayesian-bm25";
import type { IndexStats } from "../core/types.js";
import type { BayesianBM25Params } from "./bayesian-bm25.js";
import { BM25Scorer } from "./bm25.js";

/**
 * Numerically stable sigmoid function.
 * Avoids overflow for large negative x by using the exp(x)/(1+exp(x)) form.
 */
function sigmoidStable(x: number): number {
  if (x >= 0) return 1.0 / (1.0 + Math.exp(-x));
  const ex = Math.exp(x);
  return ex / (1.0 + ex);
}

/**
 * Clamped logit function.
 * Returns -10 for p <= 0 and +10 for p >= 1 to avoid infinities.
 */
function logitClamped(p: number): number {
  if (p <= 0) return -10.0;
  if (p >= 1) return 10.0;
  return Math.log(p / (1.0 - p));
}

/**
 * A prior function that maps document fields to a probability in (0, 1).
 * Returns 0.5 (neutral prior) when no prior information is available.
 */
export type PriorFn = (docFields: Record<string, unknown>) => number;

/**
 * Bayesian BM25 scorer with external prior.
 *
 * Computes the BM25 likelihood probability, then combines it with
 * an external prior via log-odds addition to produce a posterior.
 */
export class ExternalPriorScorer {
  private readonly _bm25: BM25Scorer;
  private readonly _transform: BayesianProbabilityTransform;
  private readonly _priorFn: PriorFn;

  constructor(params: BayesianBM25Params, indexStats: IndexStats, priorFn: PriorFn) {
    this._bm25 = new BM25Scorer(params.bm25, indexStats);
    this._transform = new BayesianProbabilityTransform(
      params.alpha,
      params.beta,
      params.baseRate === 0.5 ? null : params.baseRate,
    );
    this._priorFn = priorFn;
  }

  /**
   * Compute posterior with external prior via log-odds fusion.
   *
   * 1. Compute BM25 raw score.
   * 2. Convert to likelihood probability via BayesianProbabilityTransform.
   * 3. Compute prior from document fields.
   * 4. Combine via log-odds addition: logit(posterior) = logit(likelihood) + logit(prior).
   * 5. Convert back to probability via sigmoid.
   */
  scoreWithPrior(
    termFreq: number,
    docLength: number,
    docFreq: number,
    docFields: Record<string, unknown>,
  ): number {
    // Compute BM25 likelihood probability
    const idfVal = this._bm25.idf(docFreq);
    const rawBm25 = this._bm25.scoreWithIdf(termFreq, docLength, idfVal);
    const avgdl = this._bm25.params.k1 > 0 ? docLength : 1;
    const docLenRatio = avgdl > 0 ? docLength / avgdl : 1.0;
    const likelihood = this._transform.scoreToProbability(
      rawBm25,
      termFreq,
      docLenRatio,
    );

    // Compute prior from document fields
    let prior = this._priorFn(docFields);
    prior = Math.max(1e-10, Math.min(1.0 - 1e-10, prior));

    // Combine via log-odds addition
    const logitPosterior = logitClamped(likelihood) + logitClamped(prior);
    return sigmoidStable(logitPosterior);
  }
}

// -- Prior factory functions --------------------------------------------------

/**
 * Create a recency-based prior function.
 *
 * Documents with a more recent timestamp in the given field receive higher
 * prior probability. The prior decays exponentially with age:
 *
 *     prior = 0.5 + 0.4 * exp(-age_days / decay_days)
 *
 * Returns 0.5 (neutral prior) when the field is missing or unparseable.
 *
 * @param field - The document field containing a date/timestamp.
 * @param decayDays - Half-life in days for the exponential decay.
 */
export function recencyPrior(field: string, decayDays = 30.0): PriorFn {
  return (docFields: Record<string, unknown>): number => {
    const val = docFields[field];
    if (val === null || val === undefined) return 0.5;

    let date: Date;
    if (val instanceof Date) {
      date = val;
    } else if (typeof val === "string") {
      date = new Date(val);
      if (isNaN(date.getTime())) return 0.5;
    } else {
      return 0.5;
    }

    const ageDays = (Date.now() - date.getTime()) / 86400000;
    return 0.5 + 0.4 * Math.exp(-ageDays / decayDays);
  };
}

/**
 * Create an authority-based prior function.
 *
 * Maps categorical authority levels to prior probabilities.
 * Default levels: {"high": 0.8, "medium": 0.6, "low": 0.4}.
 * Returns 0.5 (neutral) when the field is missing or unrecognized.
 *
 * @param field - The document field containing the authority level.
 * @param levels - Mapping from level name to prior probability.
 */
export function authorityPrior(
  field: string,
  levels?: Record<string, number> | null,
): PriorFn {
  const _levels = levels ?? { high: 0.8, medium: 0.6, low: 0.4 };
  return (docFields: Record<string, unknown>): number => {
    const val = docFields[field];
    if (val === null || val === undefined) return 0.5;
    const key = typeof val === "string" ? val : String(val as number);
    return _levels[key] ?? 0.5;
  };
}
