//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- log-odds fusion
// 1:1 port of uqa/fusion/log_odds.py
//
// Log-odds conjunction framework (Section 4, Paper 4).
//
// Resolves the conjunction shrinkage problem while preserving:
// - Scale neutrality (Theorem 4.1.2): P_i = p for all i => P_final = p
// - Sign preservation (Theorem 4.2.2): sgn(adjusted) = sgn(mean)
// - Irrelevance preservation (Theorem 4.5.1 iii): all P_i < 0.5 => P_final < 0.5
// - Relevance preservation (Theorem 4.5.1 iv): all P_i > 0.5 => P_final > 0.5

import { logOddsConjunction } from "bayesian-bm25";

/**
 * Log-odds conjunction framework (Section 4, Paper 4).
 *
 * Delegates to bayesian-bm25 package's log_odds_conjunction.
 */
export class LogOddsFusion {
  /**
   * Confidence scaling exponent.
   * alpha=0.5 yields the sqrt(n) scaling law (Theorem 4.4.1).
   */
  readonly alpha: number;

  /**
   * Gating mechanism for log-odds signals.
   * "none" (default), "relu", or "swish".
   */
  readonly gating: string | null;

  /**
   * @param confidenceAlpha - Confidence scaling exponent (default 0.5).
   * @param gating - Gating mechanism: "none", "relu", or "swish".
   */
  constructor(confidenceAlpha = 0.5, gating?: string | null) {
    this.alpha = confidenceAlpha;
    this.gating = gating ?? null;
  }

  /**
   * Combine calibrated probability signals via log-odds conjunction.
   *
   * Returns 0.5 for an empty list and the single value for a single element.
   */
  fuse(probabilities: number[]): number {
    if (probabilities.length === 0) return 0.5;
    if (probabilities.length === 1) return probabilities[0]!;
    return logOddsConjunction(
      probabilities,
      this.alpha,
      undefined,
      this.gating ?? "none",
    );
  }

  /**
   * Log-odds mean aggregation (Definition 4.1.1, Paper 4).
   *
   * Computes the arithmetic mean in log-odds space and maps back to
   * probability via sigmoid. Unlike fuse(), no confidence scaling
   * (n^alpha) is applied -- the result is scale-neutral: if all
   * signals report the same probability p, the output is exactly p
   * regardless of n.
   *
   * This is the normalized Logarithmic Opinion Pool (Theorem 4.1.2a).
   */
  fuseMean(probabilities: number[]): number {
    if (probabilities.length === 0) return 0.5;
    if (probabilities.length === 1) return probabilities[0]!;

    const eps = 1e-15;
    let sumLogit = 0;
    for (const p of probabilities) {
      const c = Math.max(eps, Math.min(1.0 - eps, p));
      sumLogit += Math.log(c / (1.0 - c));
    }
    const meanLogit = sumLogit / probabilities.length;
    return 1.0 / (1.0 + Math.exp(-meanLogit));
  }

  /**
   * Weighted log-odds conjunction (attention-like, Section 8, Paper 4).
   *
   * Each signal gets a per-signal weight applied in log-odds space.
   */
  fuseWeighted(probabilities: number[], weights: number[]): number {
    if (probabilities.length === 0) return 0.5;
    return logOddsConjunction(
      probabilities,
      this.alpha,
      weights,
      this.gating ?? "none",
    );
  }
}

// -- SignalQuality -----------------------------------------------------------

/**
 * Quality metrics for a single signal (Paper 4, Section 6).
 *
 * Used to compute per-signal confidence scaling in AdaptiveLogOddsFusion.
 */
export interface SignalQuality {
  /** Fraction of candidate docs returned by this signal. */
  readonly coverageRatio: number;
  /** Variance of signal scores. */
  readonly scoreVariance: number;
  /** Mean absolute calibration error. */
  readonly calibrationError: number;
}

// -- AdaptiveLogOddsFusion ---------------------------------------------------

/**
 * Log-odds fusion with per-signal adaptive confidence scaling.
 *
 * Instead of a uniform alpha, each signal gets an alpha computed
 * from its quality metrics: signals with higher coverage, lower
 * variance, and lower calibration error get higher confidence.
 *
 * alpha_i = base_alpha * (coverage * (1 - cal_error)) / (1 + variance)
 */
export class AdaptiveLogOddsFusion extends LogOddsFusion {
  readonly baseAlpha: number;

  constructor(baseAlpha = 0.5, gating?: string | null) {
    super(baseAlpha, gating);
    this.baseAlpha = baseAlpha;
  }

  /**
   * Compute per-signal confidence scaling from quality metrics.
   *
   * Higher coverage, lower calibration error, and lower variance
   * all increase the signal's confidence alpha.
   */
  computeSignalAlpha(quality: SignalQuality): number {
    const coverage = Math.max(0, Math.min(1, quality.coverageRatio));
    const calError = Math.max(0, Math.min(1, quality.calibrationError));
    const variance = Math.max(0, quality.scoreVariance);
    const alpha = (this.baseAlpha * (coverage * (1 - calError))) / (1 + variance);
    return Math.max(0.01, Math.min(1.0, alpha));
  }

  /**
   * Fuse with per-signal adaptive weights.
   *
   * Computes raw alpha for each signal from its quality metrics,
   * then normalizes to sum to 1.0 (required by log_odds_conjunction).
   */
  fuseAdaptive(probabilities: number[], qualities: SignalQuality[]): number {
    if (probabilities.length === 0) return 0.5;
    if (probabilities.length === 1) return probabilities[0]!;

    const rawWeights = qualities.map((q) => this.computeSignalAlpha(q));
    let wSum = 0;
    for (const w of rawWeights) wSum += w;
    const normalized = rawWeights.map((w) => w / wSum);

    return this.fuseWeighted(probabilities, normalized);
  }
}
