import { describe, expect, it } from "vitest";
import { IndexStats } from "../../src/core/types.js";
import { createBayesianBM25Params } from "../../src/scoring/bayesian-bm25.js";
import {
  ExternalPriorScorer,
  authorityPrior,
  recencyPrior,
} from "../../src/scoring/external-prior.js";
import { Engine } from "../../src/engine.js";

// -- TestExternalPriorScorer --

describe("TestExternalPriorScorer", () => {
  it("score with neutral prior", () => {
    const stats = new IndexStats(100, 10.0);
    stats.setDocFreq("_default", "test", 10);

    const neutralFn = () => 0.5;

    const scorer = new ExternalPriorScorer(
      createBayesianBM25Params(),
      stats,
      neutralFn,
    );
    const score = scorer.scoreWithPrior(3, 10, 10, {});
    expect(score).toBeGreaterThan(0.0);
    expect(score).toBeLessThan(1.0);
  });

  it("high prior boosts score", () => {
    const stats = new IndexStats(100, 10.0);
    stats.setDocFreq("_default", "test", 10);

    const neutralFn = () => 0.5;
    const highFn = () => 0.9;

    const scorerNeutral = new ExternalPriorScorer(
      createBayesianBM25Params(),
      stats,
      neutralFn,
    );
    const scorerHigh = new ExternalPriorScorer(
      createBayesianBM25Params(),
      stats,
      highFn,
    );

    const baseScore = scorerNeutral.scoreWithPrior(3, 10, 10, {});
    const boostedScore = scorerHigh.scoreWithPrior(3, 10, 10, {});
    expect(boostedScore).toBeGreaterThan(baseScore);
  });

  it("low prior reduces score", () => {
    const stats = new IndexStats(100, 10.0);
    stats.setDocFreq("_default", "test", 10);

    const neutralFn = () => 0.5;
    const lowFn = () => 0.1;

    const scorerNeutral = new ExternalPriorScorer(
      createBayesianBM25Params(),
      stats,
      neutralFn,
    );
    const scorerLow = new ExternalPriorScorer(createBayesianBM25Params(), stats, lowFn);

    const baseScore = scorerNeutral.scoreWithPrior(3, 10, 10, {});
    const reducedScore = scorerLow.scoreWithPrior(3, 10, 10, {});
    expect(reducedScore).toBeLessThan(baseScore);
  });

  it("score in probability range", () => {
    const stats = new IndexStats(100, 10.0);
    stats.setDocFreq("_default", "test", 10);

    const fn = () => 0.7;
    const scorer = new ExternalPriorScorer(createBayesianBM25Params(), stats, fn);
    const score = scorer.scoreWithPrior(2, 8, 10, { authority: "high" });
    expect(score).toBeGreaterThan(0.0);
    expect(score).toBeLessThan(1.0);
  });
});

// -- TestRecencyPrior --

describe("TestRecencyPrior", () => {
  it("missing field returns neutral", () => {
    const fn = recencyPrior("timestamp");
    expect(fn({})).toBeCloseTo(0.5);
  });

  it("recent date gives high prior", () => {
    const fn = recencyPrior("timestamp", 30.0);
    const now = new Date().toISOString();
    const prior = fn({ timestamp: now });
    expect(prior).toBeGreaterThan(0.7);
  });

  it("old date gives lower prior", () => {
    const fn = recencyPrior("timestamp", 30.0);
    const old = new Date(Date.now() - 365 * 86400000).toISOString();
    const prior = fn({ timestamp: old });
    expect(prior).toBeLessThan(0.6);
  });

  it("invalid date returns neutral", () => {
    const fn = recencyPrior("timestamp");
    expect(fn({ timestamp: "not-a-date" })).toBeCloseTo(0.5);
  });
});

// -- TestAuthorityPrior --

describe("TestAuthorityPrior", () => {
  it("high authority", () => {
    const fn = authorityPrior("level");
    expect(fn({ level: "high" })).toBeCloseTo(0.8);
  });

  it("medium authority", () => {
    const fn = authorityPrior("level");
    expect(fn({ level: "medium" })).toBeCloseTo(0.6);
  });

  it("low authority", () => {
    const fn = authorityPrior("level");
    expect(fn({ level: "low" })).toBeCloseTo(0.4);
  });

  it("missing field returns neutral", () => {
    const fn = authorityPrior("level");
    expect(fn({})).toBeCloseTo(0.5);
  });

  it("unknown level returns neutral", () => {
    const fn = authorityPrior("level");
    expect(fn({ level: "unknown" })).toBeCloseTo(0.5);
  });

  it("custom levels", () => {
    const fn = authorityPrior("rank", { expert: 0.95, novice: 0.3 });
    expect(fn({ rank: "expert" })).toBeCloseTo(0.95);
    expect(fn({ rank: "novice" })).toBeCloseTo(0.3);
  });
});

// -- TestExternalPriorSQL --

describe("TestExternalPriorSQL", () => {
  it("bayesian with prior produces valid scores", () => {
    // Test the ExternalPriorScorer directly with an authority-like prior
    const stats = new IndexStats(100, 10.0);
    stats.setDocFreq("_default", "test", 10);

    const fn = authorityPrior("authority");
    const scorer = new ExternalPriorScorer(createBayesianBM25Params(), stats, fn);
    const score = scorer.scoreWithPrior(3, 10, 10, { authority: "high" });
    expect(score).toBeGreaterThan(0.0);
    expect(score).toBeLessThan(1.0);
  });

  it("bayesian with prior unknown authority returns neutral range", () => {
    const stats = new IndexStats(100, 10.0);
    stats.setDocFreq("_default", "test", 10);

    const fn = authorityPrior("authority");
    const scorer = new ExternalPriorScorer(createBayesianBM25Params(), stats, fn);
    // "invalid" is not a known authority level, so prior returns 0.5 (neutral)
    const score = scorer.scoreWithPrior(3, 10, 10, { authority: "invalid" });
    expect(score).toBeGreaterThan(0.0);
    expect(score).toBeLessThan(1.0);
  });
});

// -- TestExternalPriorQueryBuilder --

describe("TestExternalPriorQueryBuilder", () => {
  it("fluent api with authority prior via scorer", () => {
    // Test the scorer with authority prior
    const stats = new IndexStats(100, 10.0);
    stats.setDocFreq("_default", "test", 10);

    const priorFn = authorityPrior("source");
    const scorer = new ExternalPriorScorer(createBayesianBM25Params(), stats, priorFn);

    const highScore = scorer.scoreWithPrior(3, 10, 10, { source: "high" });
    const lowScore = scorer.scoreWithPrior(3, 10, 10, { source: "low" });
    expect(highScore).toBeGreaterThan(lowScore);
  });

  it("fluent api requires prior fn", () => {
    const e = new Engine();
    expect(() => e.query("t").scoreBayesianWithPrior("test")).toThrow(
      /priorFn is required/,
    );
  });
});
