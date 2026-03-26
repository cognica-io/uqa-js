//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- parameter learner
// 1:1 port of uqa/scoring/parameter_learner.py
//
// Online parameter learning for Bayesian BM25 (Section 8, Paper 3).
//
// Wraps BayesianProbabilityTransform.fit() and .update() to learn
// calibration parameters (alpha, beta, base_rate) from relevance
// judgments.

import { BayesianProbabilityTransform } from "bayesian-bm25";
import type { FitOptions, UpdateOptions } from "bayesian-bm25";

/**
 * Online parameter learning for Bayesian BM25.
 *
 * Wraps BayesianProbabilityTransform.fit() and .update() to learn
 * calibration parameters (alpha, beta, baseRate) from relevance
 * judgments. Supports both batch learning (fit) and incremental
 * online updates (update).
 */
export class ParameterLearner {
  private readonly _transform: BayesianProbabilityTransform;

  /**
   * @param alpha - Initial scaling exponent (default 1.0).
   * @param beta - Initial offset (default 0.0).
   * @param baseRate - Prior base rate of relevance (default 0.5).
   */
  constructor(alpha = 1.0, beta = 0.0, baseRate = 0.5) {
    this._transform = new BayesianProbabilityTransform(
      alpha,
      beta,
      baseRate === 0.5 ? null : baseRate,
    );
  }

  /**
   * Batch-learn calibration parameters from scored documents.
   *
   * @param scores - Array of raw BM25 scores.
   * @param labels - Array of binary relevance labels (0 or 1).
   * @param options - Optional fit options (mode, tfs, doc_len_ratios).
   * @returns The learned parameters as a dict with keys: alpha, beta, baseRate.
   */
  fit(
    scores: number[],
    labels: number[],
    options?: FitOptions,
  ): Record<string, number> {
    this._transform.fit(scores, labels, options);
    return this.params();
  }

  /**
   * Online update with a single observation.
   *
   * @param score - A single raw BM25 score.
   * @param label - Binary relevance label (0 or 1).
   * @param options - Optional update options (learning_rate, tf, doc_len_ratio).
   */
  update(score: number, label: number, options?: UpdateOptions): void {
    this._transform.update(score, label, options);
  }

  /**
   * Return current learned parameters.
   *
   * @returns Dict with keys: alpha, beta, baseRate.
   */
  params(): Record<string, number> {
    return {
      alpha: this._transform.alpha,
      beta: this._transform.beta,
      baseRate: this._transform.baseRate ?? 0.5,
    };
  }
}
