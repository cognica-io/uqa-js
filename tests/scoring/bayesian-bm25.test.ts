import { describe, expect, it } from "vitest";
import { IndexStats } from "../../src/core/types.js";
import {
  BayesianBM25Scorer,
  createBayesianBM25Params,
} from "../../src/scoring/bayesian-bm25.js";

describe("BayesianBM25Scorer", () => {
  function makeScorer() {
    const stats = new IndexStats(1000, 50);
    return new BayesianBM25Scorer(createBayesianBM25Params(), stats);
  }

  it("returns probability in [0, 1]", () => {
    const scorer = makeScorer();
    const p = scorer.score(3, 50, 10);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it("preserves BM25 ranking order (monotonicity)", () => {
    const scorer = makeScorer();
    const p1 = scorer.score(1, 50, 10);
    const p5 = scorer.score(5, 50, 10);
    expect(p5).toBeGreaterThan(p1);
  });

  it("combineScores returns probability", () => {
    const scorer = makeScorer();
    const combined = scorer.combineScores([0.7, 0.8]);
    expect(combined).toBeGreaterThanOrEqual(0);
    expect(combined).toBeLessThanOrEqual(1);
  });

  it("combineScores with empty returns 0.5", () => {
    const scorer = makeScorer();
    expect(scorer.combineScores([])).toBe(0.5);
  });

  it("combineScores with single returns same", () => {
    const scorer = makeScorer();
    expect(scorer.combineScores([0.7])).toBe(0.7);
  });

  it("upperBound returns probability", () => {
    const scorer = makeScorer();
    const ub = scorer.upperBound(10);
    expect(ub).toBeGreaterThanOrEqual(0);
    expect(ub).toBeLessThanOrEqual(1);
  });

  it("idf delegates to BM25", () => {
    const scorer = makeScorer();
    expect(scorer.idf(10)).toBeGreaterThan(0);
  });
});
