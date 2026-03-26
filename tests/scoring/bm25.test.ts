import { describe, expect, it } from "vitest";
import { IndexStats } from "../../src/core/types.js";
import { BM25Scorer, createBM25Params } from "../../src/scoring/bm25.js";

describe("BM25Scorer", () => {
  function makeScorer(totalDocs = 100, avgDocLength = 50) {
    const stats = new IndexStats(totalDocs, avgDocLength);
    return new BM25Scorer(createBM25Params(), stats);
  }

  describe("idf", () => {
    it("higher for rarer terms", () => {
      const scorer = makeScorer(1000);
      expect(scorer.idf(1)).toBeGreaterThan(scorer.idf(100));
      expect(scorer.idf(100)).toBeGreaterThan(scorer.idf(500));
    });

    it("non-negative for any doc freq", () => {
      const scorer = makeScorer(100);
      expect(scorer.idf(0)).toBeGreaterThan(0);
      expect(scorer.idf(50)).toBeGreaterThan(0);
      expect(scorer.idf(100)).toBeGreaterThanOrEqual(0);
    });
  });

  describe("score", () => {
    it("positive for matching terms", () => {
      const scorer = makeScorer();
      expect(scorer.score(3, 50, 10)).toBeGreaterThan(0);
    });

    it("increases with term frequency (saturation)", () => {
      const scorer = makeScorer();
      const s1 = scorer.score(1, 50, 10);
      const s5 = scorer.score(5, 50, 10);
      const s20 = scorer.score(20, 50, 10);
      expect(s5).toBeGreaterThan(s1);
      expect(s20).toBeGreaterThan(s5);
      // Saturation: marginal gain decreases
      expect(s5 - s1).toBeGreaterThan(s20 - s5);
    });

    it("penalizes longer documents", () => {
      const scorer = makeScorer();
      const short = scorer.score(3, 30, 10);
      const long = scorer.score(3, 200, 10);
      expect(short).toBeGreaterThan(long);
    });
  });

  describe("scoreWithIdf", () => {
    it("matches score() result", () => {
      const scorer = makeScorer();
      const idf = scorer.idf(10);
      expect(scorer.scoreWithIdf(3, 50, idf)).toBeCloseTo(scorer.score(3, 50, 10), 10);
    });
  });

  describe("combineScores", () => {
    it("sums scores", () => {
      const scorer = makeScorer();
      expect(scorer.combineScores([1.0, 2.0, 3.0])).toBe(6.0);
    });
  });

  describe("upperBound", () => {
    it("returns boost * idf", () => {
      const scorer = makeScorer();
      expect(scorer.upperBound(10)).toBeCloseTo(scorer.idf(10), 10);
    });

    it("scales with boost", () => {
      const stats = new IndexStats(100, 50);
      const boosted = new BM25Scorer(createBM25Params({ boost: 2.0 }), stats);
      const normal = new BM25Scorer(createBM25Params({ boost: 1.0 }), stats);
      expect(boosted.upperBound(10)).toBeCloseTo(2.0 * normal.upperBound(10), 10);
    });
  });

  describe("custom params", () => {
    it("k1=0 gives binary score", () => {
      const stats = new IndexStats(100, 50);
      const scorer = new BM25Scorer(createBM25Params({ k1: 0 }), stats);
      // With k1=0, invNorm -> inf, score saturates immediately
      const s1 = scorer.score(1, 50, 10);
      const s10 = scorer.score(10, 50, 10);
      // Both should be close to upper bound
      expect(Math.abs(s1 - s10)).toBeLessThan(0.01);
    });
  });
});
