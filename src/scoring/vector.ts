//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- vector scoring
// 1:1 port of uqa/scoring/vector.py
//
// Vector similarity to probability conversion.
//
// Two modes:
//     - Uncalibrated (Definition 7.1.2, Paper 3):
//           P_vector = (1 + score) / 2
//     - Calibrated (Theorem 3.1.1, Paper 5):
//           P_vector via log(f_R(d) / f_G(d)) + logit(base_rate)

import { cosineToProbability } from "bayesian-bm25";
import type { VectorProbabilityTransform } from "bayesian-bm25";
import { cosine } from "../math/linalg.js";

/**
 * Vector similarity to probability conversion.
 *
 * Provides both uncalibrated and calibrated modes for converting
 * cosine similarity scores into relevance probabilities.
 */
// eslint-disable-next-line @typescript-eslint/no-extraneous-class
export class VectorScorer {
  /**
   * Compute cosine similarity between two vectors.
   *
   * Returns 0.0 if either vector has zero norm.
   *
   * @param a - First vector.
   * @param b - Second vector.
   * @returns Cosine similarity in [-1, 1].
   */
  static cosineSimilarity(a: Float64Array, b: Float64Array): number {
    return cosine(a, b);
  }

  /**
   * Definition 7.1.2: P_vector = (1 + score) / 2 (uncalibrated).
   *
   * Converts a cosine similarity in [-1, 1] to a probability in [0, 1]
   * using the simple linear mapping.
   *
   * @param cosineSim - Cosine similarity value.
   * @returns Probability in [0, 1].
   */
  static similarityToProbability(cosineSim: number): number {
    return cosineToProbability(cosineSim);
  }

  /**
   * Likelihood ratio calibration (Theorem 3.1.1, Paper 5).
   *
   * Converts cosine similarities to calibrated probabilities using a
   * pre-configured VectorProbabilityTransform with background distribution.
   *
   * @param similarities - Cosine similarities in [-1, 1] for the top-K results.
   * @param calibrator - Pre-configured calibrator with background distribution.
   * @param weights - External relevance weights. If null, uniform weights.
   * @returns Calibrated probabilities in (0, 1).
   */
  static calibratedProbabilities(
    similarities: number[],
    calibrator: VectorProbabilityTransform,
    weights?: number[] | null,
  ): number[] {
    const distances = similarities.map((s) => 1.0 - s);
    const calibrated = calibrator.calibrate(
      distances,
      weights !== null && weights !== undefined ? { weights } : undefined,
    );
    return calibrated;
  }
}
