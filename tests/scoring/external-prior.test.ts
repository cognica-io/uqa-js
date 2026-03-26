import { describe, expect, it } from "vitest";
import { IndexStats } from "../../src/core/types.js";
import { createBayesianBM25Params } from "../../src/scoring/bayesian-bm25.js";
import {
  ExternalPriorScorer,
  authorityPrior,
  recencyPrior,
} from "../../src/scoring/external-prior.js";

describe("ExternalPriorScorer", () => {
  it("returns probability in [0, 1]", () => {
    const stats = new IndexStats(100, 50);
    const scorer = new ExternalPriorScorer(
      createBayesianBM25Params(),
      stats,
      () => 0.7,
    );
    const p = scorer.scoreWithPrior(3, 50, 10, {});
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it("high prior increases score", () => {
    const stats = new IndexStats(100, 50);
    const lowPrior = new ExternalPriorScorer(
      createBayesianBM25Params(),
      stats,
      () => 0.3,
    );
    const highPrior = new ExternalPriorScorer(
      createBayesianBM25Params(),
      stats,
      () => 0.9,
    );
    const low = lowPrior.scoreWithPrior(3, 50, 10, {});
    const high = highPrior.scoreWithPrior(3, 50, 10, {});
    expect(high).toBeGreaterThan(low);
  });
});

describe("recencyPrior", () => {
  it("returns ~0.9 for recent documents", () => {
    const fn = recencyPrior("date");
    const now = new Date().toISOString();
    const p = fn({ date: now });
    expect(p).toBeGreaterThan(0.8);
    expect(p).toBeLessThanOrEqual(0.9);
  });

  it("returns ~0.5 for old documents", () => {
    const fn = recencyPrior("date", 30);
    const old = new Date(Date.now() - 365 * 86400000).toISOString();
    const p = fn({ date: old });
    expect(p).toBeCloseTo(0.5, 1);
  });

  it("returns 0.5 for missing field", () => {
    const fn = recencyPrior("date");
    expect(fn({})).toBe(0.5);
  });
});

describe("authorityPrior", () => {
  it("returns mapped levels", () => {
    const fn = authorityPrior("level");
    expect(fn({ level: "high" })).toBe(0.8);
    expect(fn({ level: "medium" })).toBe(0.6);
    expect(fn({ level: "low" })).toBe(0.4);
  });

  it("returns 0.5 for unknown level", () => {
    const fn = authorityPrior("level");
    expect(fn({ level: "unknown" })).toBe(0.5);
  });

  it("returns 0.5 for missing field", () => {
    const fn = authorityPrior("level");
    expect(fn({})).toBe(0.5);
  });

  it("accepts custom levels", () => {
    const fn = authorityPrior("rank", { expert: 0.95, novice: 0.3 });
    expect(fn({ rank: "expert" })).toBe(0.95);
    expect(fn({ rank: "novice" })).toBe(0.3);
  });
});
