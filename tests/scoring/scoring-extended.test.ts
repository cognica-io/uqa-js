import { describe, expect, it } from "vitest";
import { IndexStats } from "../../src/core/types.js";
import { BM25Scorer, createBM25Params } from "../../src/scoring/bm25.js";
import type { BM25Params } from "../../src/scoring/bm25.js";
import {
  BayesianBM25Scorer,
  createBayesianBM25Params,
} from "../../src/scoring/bayesian-bm25.js";
import { BayesianProbabilityTransform } from "bayesian-bm25";
import { VectorScorer } from "../../src/scoring/vector.js";

// -- Fixtures -----------------------------------------------------------------

function makeIndexStats(): IndexStats {
  return new IndexStats(10000, 200.0);
}

function makeBM25Scorer(stats?: IndexStats, params?: BM25Params): BM25Scorer {
  return new BM25Scorer(params ?? createBM25Params(), stats ?? makeIndexStats());
}

function makeBayesianScorer(stats?: IndexStats): BayesianBM25Scorer {
  return new BayesianBM25Scorer(createBayesianBM25Params(), stats ?? makeIndexStats());
}

// =============================================================================
// BM25 Tests
// =============================================================================

describe("BM25Scorer", () => {
  it("IDF is positive for rare terms", () => {
    const scorer = makeBM25Scorer();
    const idf = scorer.idf(10);
    expect(idf).toBeGreaterThan(0);
  });

  it("IDF decreases with frequency", () => {
    const scorer = makeBM25Scorer();
    const idfRare = scorer.idf(10);
    const idfCommon = scorer.idf(5000);
    expect(idfRare).toBeGreaterThan(idfCommon);
  });

  it("monotonicity: higher term frequency produces higher score", () => {
    const scorer = makeBM25Scorer();
    const scoreLow = scorer.score(1, 200, 100);
    const scoreMid = scorer.score(5, 200, 100);
    const scoreHigh = scorer.score(20, 200, 100);
    expect(scoreLow).toBeLessThan(scoreMid);
    expect(scoreMid).toBeLessThan(scoreHigh);
  });

  it("monotonicity: shorter documents produce higher scores", () => {
    const scorer = makeBM25Scorer();
    const scoreShort = scorer.score(5, 50, 100);
    const scoreAvg = scorer.score(5, 200, 100);
    const scoreLong = scorer.score(5, 500, 100);
    expect(scoreShort).toBeGreaterThan(scoreAvg);
    expect(scoreAvg).toBeGreaterThan(scoreLong);
  });

  it("upper bound is always greater than any score", () => {
    const scorer = makeBM25Scorer();
    const ub = scorer.upperBound(100);
    const termFreqs = [1, 5, 10, 50, 100, 1000];
    const docLengths = [1, 50, 100, 200, 500, 1000];
    for (const tf of termFreqs) {
      for (const dl of docLengths) {
        const score = scorer.score(tf, dl, 100);
        expect(score).toBeLessThan(ub);
      }
    }
  });

  it("score is non-negative", () => {
    const scorer = makeBM25Scorer();
    const score = scorer.score(1, 200, 100);
    expect(score).toBeGreaterThanOrEqual(0.0);
  });

  it("boosted upper bound is respected", () => {
    const stats = makeIndexStats();
    const params = createBM25Params({ boost: 2.5 });
    const scorer = new BM25Scorer(params, stats);
    const ub = scorer.upperBound(50);
    const score = scorer.score(100, 10, 50);
    expect(score).toBeLessThan(ub);
  });
});

// =============================================================================
// Bayesian BM25 Tests
// =============================================================================

describe("BayesianBM25Scorer", () => {
  it("output is in [0, 1]", () => {
    const scorer = makeBayesianScorer();
    const termFreqs = [0, 1, 5, 10, 50];
    const docLengths = [10, 100, 200, 500, 1000];
    const docFreqs = [1, 10, 100, 1000, 5000];
    for (const tf of termFreqs) {
      for (const dl of docLengths) {
        for (const df of docFreqs) {
          const p = scorer.score(tf, dl, df);
          expect(p).toBeGreaterThanOrEqual(0.0);
          expect(p).toBeLessThanOrEqual(1.0);
        }
      }
    }
  });

  it("monotonicity: higher BM25 score produces higher posterior", () => {
    const scorer = makeBayesianScorer();
    const pLow = scorer.score(1, 200, 100);
    const pMid = scorer.score(5, 200, 100);
    const pHigh = scorer.score(20, 200, 100);
    expect(pLow).toBeLessThan(pMid);
    expect(pMid).toBeLessThan(pHigh);
  });

  it("composite prior bounds are in [0.1, 0.9]", () => {
    const termFreqs = [0, 1, 5, 10, 50, 100];
    const dlRatios = [0.005, 0.05, 0.5, 1.0, 2.5, 10.0];
    for (const tf of termFreqs) {
      for (const dlRatio of dlRatios) {
        const prior = BayesianProbabilityTransform.compositePrior(tf, dlRatio);
        expect(prior).toBeGreaterThanOrEqual(0.1);
        expect(prior).toBeLessThanOrEqual(0.9);
      }
    }
  });

  it("base_rate=0.5 is identity", () => {
    const stats = makeIndexStats();
    const scorer05 = new BayesianBM25Scorer(
      createBayesianBM25Params({ baseRate: 0.5 }),
      stats,
    );
    const scorerDefault = new BayesianBM25Scorer(createBayesianBM25Params(), stats);
    for (const tf of [1, 5, 10]) {
      for (const dl of [100, 200, 500]) {
        const p05 = scorer05.score(tf, dl, 100);
        const pDef = scorerDefault.score(tf, dl, 100);
        expect(Math.abs(p05 - pDef)).toBeLessThan(1e-12);
      }
    }
  });

  it("non-0.5 base rate shifts posterior", () => {
    const stats = makeIndexStats();
    const pLowBr = new BayesianBM25Scorer(
      createBayesianBM25Params({ baseRate: 0.2 }),
      stats,
    ).score(5, 200, 100);
    const pDefault = new BayesianBM25Scorer(
      createBayesianBM25Params({ baseRate: 0.5 }),
      stats,
    ).score(5, 200, 100);
    const pHighBr = new BayesianBM25Scorer(
      createBayesianBM25Params({ baseRate: 0.8 }),
      stats,
    ).score(5, 200, 100);
    expect(pLowBr).toBeLessThan(pDefault);
    expect(pDefault).toBeLessThan(pHighBr);
  });

  it("upper bound is at least any score", () => {
    const scorer = makeBayesianScorer();
    const ub = scorer.upperBound(100);
    for (const tf of [1, 5, 10, 50]) {
      for (const dl of [50, 200, 500]) {
        const score = scorer.score(tf, dl, 100);
        expect(score).toBeLessThanOrEqual(ub + 1e-10);
      }
    }
  });

  it("likelihood is numerically stable for extreme inputs", () => {
    const transform = new BayesianProbabilityTransform(1.0, 0.0);
    expect(transform.likelihood(500.0)).toBeCloseTo(1.0, 6);
    expect(transform.likelihood(-500.0)).toBeCloseTo(0.0, 6);
    expect(transform.likelihood(0.0)).toBeCloseTo(0.5, 6);
  });
});

// =============================================================================
// Vector Scorer Tests
// =============================================================================

describe("VectorScorer", () => {
  it("cosine similarity of identical vectors is 1", () => {
    const v = new Float64Array([1.0, 2.0, 3.0]);
    expect(VectorScorer.cosineSimilarity(v, v)).toBeCloseTo(1.0, 6);
  });

  it("cosine similarity of opposite vectors is -1", () => {
    const v = new Float64Array([1.0, 0.0, 0.0]);
    const neg = new Float64Array([-1.0, 0.0, 0.0]);
    expect(VectorScorer.cosineSimilarity(v, neg)).toBeCloseTo(-1.0, 6);
  });

  it("cosine similarity of orthogonal vectors is 0", () => {
    const a = new Float64Array([1.0, 0.0]);
    const b = new Float64Array([0.0, 1.0]);
    expect(VectorScorer.cosineSimilarity(a, b)).toBeCloseTo(0.0, 6);
  });

  it("cosine similarity with zero vector is 0", () => {
    const a = new Float64Array([1.0, 2.0, 3.0]);
    const z = new Float64Array(3);
    expect(VectorScorer.cosineSimilarity(a, z)).toBe(0.0);
    expect(VectorScorer.cosineSimilarity(z, z)).toBe(0.0);
  });

  it("similarity to probability range", () => {
    expect(VectorScorer.similarityToProbability(1.0)).toBeCloseTo(1.0, 6);
    expect(VectorScorer.similarityToProbability(-1.0)).toBeCloseTo(0.0, 6);
    expect(VectorScorer.similarityToProbability(0.0)).toBeCloseTo(0.5, 2);
  });

  it("similarity to probability is monotonic", () => {
    const sims = [-0.8, -0.3, 0.0, 0.4, 0.9];
    const probs = sims.map((s) => VectorScorer.similarityToProbability(s));
    for (let i = 0; i < probs.length - 1; i++) {
      expect(probs[i]!).toBeLessThan(probs[i + 1]!);
    }
  });
});
