//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- calibration metrics
// 1:1 port of uqa/scoring/calibration.py
//
// Calibration diagnostics (Section 11.3 Paper 3; Section 8.3 Paper 5).
//
// Wraps bayesian_bm25 calibration functions into a unified API for
// evaluating how well predicted relevance probabilities match actual
// relevance rates.

import {
  brierScore,
  calibrationReport,
  expectedCalibrationError,
  reliabilityDiagram,
} from "bayesian-bm25";

/**
 * Calibration diagnostics for evaluating how well predicted relevance
 * probabilities match actual relevance rates.
 *
 * Provides log loss, expected calibration error (ECE), Brier score,
 * full calibration reports, and reliability diagram data.
 */
// eslint-disable-next-line @typescript-eslint/no-extraneous-class
export class CalibrationMetrics {
  /**
   * Negative log-likelihood (log loss).
   *
   * Strictly proper scoring rule that penalizes the probabilistic
   * model directly. Lower is better.
   *
   * L = -(1/N) * sum[ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
   *
   * @param probabilities - Predicted probabilities in (0, 1).
   * @param labels - Binary relevance labels (0 or 1).
   * @returns Log loss value.
   */
  static logLoss(probabilities: number[], labels: number[]): number {
    const n = probabilities.length;
    if (n === 0) return 0;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const p = Math.max(1e-15, Math.min(1.0 - 1e-15, probabilities[i]!));
      const y = labels[i]!;
      sum += y * Math.log(p) + (1 - y) * Math.log(1 - p);
    }
    return -sum / n;
  }

  /**
   * Expected Calibration Error.
   *
   * Measures how well predicted probabilities match actual relevance
   * rates. Lower is better. Perfect calibration = 0.
   *
   * @param probabilities - Predicted probabilities.
   * @param labels - Binary relevance labels.
   * @param nBins - Number of bins for bucketing (default 10).
   * @returns ECE value.
   */
  static ece(probabilities: number[], labels: number[], nBins = 10): number {
    return expectedCalibrationError(probabilities, labels, nBins);
  }

  /**
   * Brier score: mean squared error between probabilities and labels.
   *
   * Decomposes into calibration + discrimination. Lower is better.
   *
   * @param probabilities - Predicted probabilities.
   * @param labels - Binary relevance labels.
   * @returns Brier score value.
   */
  static brier(probabilities: number[], labels: number[]): number {
    return brierScore(probabilities, labels);
  }

  /**
   * Full calibration diagnostic report.
   *
   * Returns a dict with keys: ece, brier, nSamples, nBins, reliability,
   * and any additional diagnostics from the bayesian_bm25 package.
   *
   * @param probabilities - Predicted probabilities.
   * @param labels - Binary relevance labels.
   * @param nBins - Number of bins (default 10).
   * @returns Calibration report dict.
   */
  static report(
    probabilities: number[],
    labels: number[],
    nBins = 10,
  ): Record<string, unknown> {
    const r = calibrationReport(probabilities, labels, nBins);
    return {
      ece: r.ece,
      brier: r.brier,
      nSamples: r.nSamples,
      nBins: r.nBins,
      reliability: r.reliability,
      summary: r.summary(),
    };
  }

  /**
   * Compute reliability diagram data: (avgPredicted, avgActual, count) per bin.
   *
   * Perfect calibration means avgPredicted == avgActual for every bin.
   *
   * @param probabilities - Predicted probabilities.
   * @param labels - Binary relevance labels.
   * @param nBins - Number of bins (default 10).
   * @returns Array of [avgPredicted, avgActual, count] tuples.
   */
  static reliabilityDiagram(
    probabilities: number[],
    labels: number[],
    nBins = 10,
  ): [number, number, number][] {
    return reliabilityDiagram(probabilities, labels, nBins);
  }
}
