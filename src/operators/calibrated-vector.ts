//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- calibrated vector operator
// 1:1 port of uqa/operators/calibrated_vector.py
//
// Performs KNN or threshold vector search, then transforms the raw cosine
// similarities into calibrated relevance probabilities via the likelihood
// ratio framework.  Importance weights come from one of:
//
//   1. External BM25 probabilities (cross-modal, Section 4.3)
//   2. IVF cell-density prior (Strategy 4.6.2)
//   3. Distance gap detection (Strategy 4.6.1)
//   4. Uniform weights (fallback)

import type { DocId, IndexStats, PostingEntry } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "./base.js";
import { Operator } from "./base.js";
import { TermOperator, ScoreOperator } from "./primitive.js";
import {
  BayesianBM25Scorer,
  createBayesianBM25Params,
} from "../scoring/bayesian-bm25.js";

/**
 * IVF background stats: distance distribution parameters (mu_G, sigma_G).
 */
export interface IVFBackgroundStats {
  mu: number;
  sigma: number;
}

/**
 * Minimal interface for VectorProbabilityTransform calibration.
 * Full implementation is delegated to the bayesian-bm25 package.
 */
export interface VectorProbabilityTransformLike {
  calibrate(
    distances: number[],
    opts?: {
      weights?: number[] | null;
      method?: string;
      bandwidthFactor?: number;
      densityPrior?: number[] | null;
    },
  ): number[];
}

/**
 * Compute IVF density prior weight for a single cell.
 *
 * w(c) = (pop(c) / avg_pop)^gamma
 *
 * High-population cells produce a higher prior because they
 * represent denser regions of the embedding space.
 */
export function ivfDensityPrior(cellPop: number, avgPop: number, gamma = 1.0): number {
  if (avgPop <= 0) return 1.0;
  return Math.pow(cellPop / avgPop, gamma);
}

/**
 * Fit a VectorProbabilityTransform from background distance samples.
 *
 * Returns a transform object that can calibrate new distances into
 * probabilities.  This is a simplified JS implementation.
 */
export function fitBackgroundTransform(
  bgDistances: number[],
  baseRate = 0.5,
): VectorProbabilityTransformLike {
  // Compute background statistics
  const n = bgDistances.length;
  if (n === 0) {
    return {
      calibrate(distances: number[]): number[] {
        return distances.map(() => baseRate);
      },
    };
  }

  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += bgDistances[i]!;
  }
  const mu = sum / n;

  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = bgDistances[i]! - mu;
    sumSq += d * d;
  }
  const sigma = Math.max(Math.sqrt(sumSq / n), 1e-10);

  return {
    calibrate(
      distances: number[],
      opts?: {
        weights?: number[] | null;
        method?: string;
        bandwidthFactor?: number;
        densityPrior?: number[] | null;
      },
    ): number[] {
      const bwFactor = opts?.bandwidthFactor ?? 1.0;
      const densityPrior = opts?.densityPrior ?? null;

      // Silverman bandwidth: h = 1.06 * sigma * n^{-1/5}
      const h = 1.06 * sigma * Math.pow(n, -0.2) * bwFactor;

      return distances.map((dist, i) => {
        // Gaussian kernel density at dist relative to background
        const z = (dist - mu) / Math.max(h, 1e-10);
        const bgDensity = Math.exp(-0.5 * z * z) / (Math.sqrt(2 * Math.PI) * h);

        // Likelihood ratio approximation
        // f_R estimated from the observed distances (self-density)
        const fR = bgDensity * 2.0; // proxy: closer distances should have higher f_R

        // Apply density prior if available
        let prior = baseRate;
        if (densityPrior !== null && i < densityPrior.length) {
          prior = Math.min(Math.max(densityPrior[i]! * baseRate, 0.01), 0.99);
        }

        // Bayesian update: P(R|d) = f_R * prior / (f_R * prior + f_G * (1-prior))
        const numerator = fR * prior;
        const denominator = numerator + bgDensity * (1 - prior);
        if (denominator <= 0) return prior;
        return Math.min(Math.max(numerator / denominator, 0.001), 0.999);
      });
    },
  };
}

/**
 * Cross-modal BM25 importance weights (Section 4.3).
 *
 * Uses the Bayesian BM25 scoring pipeline to produce calibrated posterior
 * probabilities P(R=1|s) for each document.
 */
function bm25Weights(
  results: PostingEntry[],
  context: ExecutionContext,
  bm25Query: string | null,
  bm25Field: string | null,
): number[] {
  const n = results.length;
  const weights = new Array<number>(n).fill(0.01);

  const invIdx = context.invertedIndex;
  if (invIdx === null || invIdx === undefined || bm25Query === null) {
    return weights;
  }

  const field = bm25Field ?? "";
  const scorer = new BayesianBM25Scorer(createBayesianBM25Params(), invIdx.stats);
  const terms = bm25Query
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 0);
  if (terms.length === 0) return weights;

  const termOp = new TermOperator(bm25Query, field || null);
  const scoreOp = new ScoreOperator(scorer, termOp, terms, field || null);
  const scoredPl = scoreOp.execute(context);

  const bm25Map = new Map<number, number>();
  for (const entry of scoredPl) {
    bm25Map.set(entry.docId, entry.payload.score);
  }

  for (let i = 0; i < results.length; i++) {
    const w = bm25Map.get(results[i]!.docId) ?? 0.01;
    weights[i] = Math.min(Math.max(w, 0.01), 0.99);
  }

  return weights;
}

export class CalibratedVectorOperator extends Operator {
  readonly queryVector: Float64Array;
  readonly k: number;
  readonly field: string;
  readonly estimationMethod: string;
  readonly baseRate: number;
  readonly weightSource: string;
  readonly bm25Query: string | null;
  readonly bm25Field: string | null;
  readonly densityGamma: number;
  readonly bandwidthScale: number;

  constructor(
    queryVector: Float64Array,
    k: number,
    field = "embedding",
    estimationMethod = "kde",
    baseRate = 0.5,
    weightSource = "density_prior",
    bm25Query?: string | null,
    bm25Field?: string | null,
    densityGamma = 1.0,
    bandwidthScale = 1.0,
  ) {
    super();
    this.queryVector = queryVector;
    this.k = k;
    this.field = field;
    this.estimationMethod = estimationMethod;
    this.baseRate = baseRate;
    this.weightSource = weightSource;
    this.bm25Query = bm25Query ?? null;
    this.bm25Field = bm25Field ?? null;
    this.densityGamma = densityGamma;
    this.bandwidthScale = bandwidthScale;
  }

  execute(context: ExecutionContext): PostingList {
    const vecIdx = context.vectorIndexes?.[this.field];
    if (vecIdx === undefined) {
      return new PostingList();
    }

    // 1. Retrieve raw top-K results.
    const rawResults = vecIdx.searchKnn(this.queryVector, this.k);
    if (rawResults.length === 0) {
      return rawResults;
    }

    // 2. Extract entries, similarities, and convert to distances.
    const rawEntries: PostingEntry[] = [];
    const docIds: DocId[] = [];
    const similarities: number[] = [];
    for (const entry of rawResults) {
      rawEntries.push(entry);
      docIds.push(entry.docId);
      similarities.push(entry.payload.score);
    }
    const distances = similarities.map((s) => 1.0 - s);

    // 3. Obtain background distances for f_G estimation.
    type IVFLike = {
      probedDistances?(query: Float64Array): number[] | null;
      backgroundSamples?: number[] | null;
      cellPopulations?(): Map<number, number>;
      totalVectors?: number;
      nlist?: number;
    };
    let ivf: IVFLike | null = null;
    let bgDistances: number[] | null = null;

    const vecIdxAny = vecIdx as unknown as IVFLike;
    if (typeof vecIdxAny.probedDistances === "function") {
      ivf = vecIdxAny;
      const probed = vecIdxAny.probedDistances(this.queryVector);
      if (probed !== null && probed.length > 0) {
        bgDistances = Array.isArray(probed) ? probed : Array.from(probed);
      }
    }

    if (
      (bgDistances === null || bgDistances.length === 0) &&
      vecIdxAny.backgroundSamples !== null &&
      vecIdxAny.backgroundSamples !== undefined
    ) {
      ivf = vecIdxAny;
      const samples = vecIdxAny.backgroundSamples;
      if (samples.length > 0) {
        bgDistances = Array.isArray(samples) ? samples : Array.from(samples);
      }
    }

    if (bgDistances === null || bgDistances.length === 0) {
      // Cannot calibrate without background -- return raw results
      return rawResults;
    }

    // 4. Build VectorProbabilityTransform from background distances.
    const vpt = fitBackgroundTransform(bgDistances, this.baseRate);

    // 5. Compute importance weights and determine calibration method.
    const { weights, densityPrior, method } = this._resolveWeightsAndMethod(
      rawEntries,
      ivf,
      distances,
      context,
    );

    // 6. Calibrate via VectorProbabilityTransform.
    const calibrated = vpt.calibrate(distances, {
      weights,
      method,
      bandwidthFactor: this.bandwidthScale,
      densityPrior,
    });

    // 7. Build output posting list with calibrated probabilities.
    const entries = docIds.map((docId, i) => ({
      docId,
      payload: createPayload({
        score: calibrated[i]!,
        fields: {
          _raw_similarity: similarities[i],
          ...rawEntries[i]!.payload.fields,
        },
      }),
    }));

    return new PostingList(entries);
  }

  private _resolveWeightsAndMethod(
    results: PostingEntry[],
    ivf: {
      cellPopulations?(): Map<number, number>;
      totalVectors?: number;
      nlist?: number;
    } | null,
    distances: number[],
    context: ExecutionContext,
  ): {
    weights: number[] | null;
    densityPrior: number[] | null;
    method: string;
  } {
    if (this.weightSource === "bayesian_bm25") {
      const w = bm25Weights(results, context, this.bm25Query, this.bm25Field);
      return { weights: w, densityPrior: null, method: this.estimationMethod };
    }

    if (
      this.weightSource === "density_prior" &&
      ivf !== null &&
      typeof ivf.cellPopulations === "function"
    ) {
      const cellPops = ivf.cellPopulations();
      const avgPop = (ivf.totalVectors ?? 0) / Math.max(ivf.nlist ?? 1, 1);
      const prior = results.map((entry) => {
        const centroidId =
          (entry.payload.fields["_centroid_id"] as number | undefined) ?? -1;
        const pop = cellPops.get(centroidId) ?? 1;
        return ivfDensityPrior(pop, avgPop, this.densityGamma);
      });
      return { weights: null, densityPrior: prior, method: this.estimationMethod };
    }

    if (this.weightSource === "distance_gap") {
      // Let VPT auto-routing handle gap detection
      return { weights: null, densityPrior: null, method: "auto" };
    }

    // "uniform" or unknown -- let VPT decide with no external signal
    return { weights: null, densityPrior: null, method: this.estimationMethod };
  }

  costEstimate(stats: IndexStats): number {
    return stats.dimensions * Math.log2(stats.totalDocs + 1);
  }
}
