import { describe, expect, it } from "vitest";
import { createPayload } from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import { AdaptiveWANDScorer, BoundTightnessAnalyzer } from "../../src/scoring/wand.js";
import { TightenedFusionWANDScorer } from "../../src/scoring/fusion-wand.js";
import type { WANDScorerLike } from "../../src/scoring/wand.js";

// -- Helpers --

function makePostingList(entries: [number, number][]): PostingList {
  return new PostingList(
    entries.map(([docId, score]) => ({
      docId,
      payload: createPayload({ score }),
    })),
  );
}

class MockScorer implements WANDScorerLike {
  private readonly _score: number;

  constructor(score: number) {
    this._score = score;
  }

  score(_tf: number, _docLength: number, _df: number): number {
    return this._score;
  }

  upperBound(_df: number): number {
    return this._score * 2.0; // Deliberately loose bound
  }
}

// -- BoundTightnessAnalyzer tests --

describe("BoundTightnessAnalyzer", () => {
  it("empty analyzer returns 1.0", () => {
    const analyzer = new BoundTightnessAnalyzer();
    expect(analyzer.tightnessRatio()).toBe(1.0);
  });

  it("perfect tightness", () => {
    const analyzer = new BoundTightnessAnalyzer();
    analyzer.record(5.0, 5.0);
    analyzer.record(3.0, 3.0);
    expect(analyzer.tightnessRatio()).toBeCloseTo(1.0);
  });

  it("loose bounds", () => {
    const analyzer = new BoundTightnessAnalyzer();
    // upper_bound=10, actual_max=5 -> ratio=0.5
    analyzer.record(10.0, 5.0);
    expect(analyzer.tightnessRatio()).toBeCloseTo(0.5);
  });

  it("slack", () => {
    const analyzer = new BoundTightnessAnalyzer();
    analyzer.record(10.0, 5.0); // ratio=0.5, slack=0.5
    expect(analyzer.slack()).toBeCloseTo(0.5);

    analyzer.clear();
    analyzer.record(4.0, 4.0); // ratio=1.0, slack=0.0
    expect(analyzer.slack()).toBeCloseTo(0.0);
  });

  it("worst bound index", () => {
    const analyzer = new BoundTightnessAnalyzer();
    analyzer.record(10.0, 9.0); // ratio=0.9
    analyzer.record(10.0, 2.0); // ratio=0.2 (worst)
    analyzer.record(10.0, 7.0); // ratio=0.7
    expect(analyzer.worstBoundIndex()).toBe(1);
  });

  it("worst bound index empty", () => {
    const analyzer = new BoundTightnessAnalyzer();
    expect(analyzer.worstBoundIndex()).toBe(0);
  });

  it("zero upper bound", () => {
    const analyzer = new BoundTightnessAnalyzer();
    analyzer.record(0.0, 0.0); // ub=0 -> ratio defaults to 1.0
    expect(analyzer.tightnessRatio()).toBeCloseTo(1.0);
  });

  it("clear", () => {
    const analyzer = new BoundTightnessAnalyzer();
    analyzer.record(10.0, 5.0);
    expect(analyzer.tightnessRatio()).toBeCloseTo(0.5);
    analyzer.clear();
    expect(analyzer.tightnessRatio()).toBeCloseTo(1.0);
  });
});

// -- AdaptiveWANDScorer tests --

describe("AdaptiveWANDScorer", () => {
  it("adaptive wand tightening", () => {
    const scorer1 = new MockScorer(1.0);
    const scorer2 = new MockScorer(2.0);
    const pl1 = makePostingList([
      [1, 0.8],
      [2, 0.6],
    ]);
    const pl2 = makePostingList([
      [1, 0.9],
      [3, 0.5],
    ]);

    const adaptive = new AdaptiveWANDScorer(
      [scorer1, scorer2],
      2,
      [pl1, pl2],
      null,
      null,
      null,
      0.8,
    );

    // The upper bounds should be tightened by the factor
    // scorer1.upperBound(2) = 2.0, * 0.8 = 1.6
    // scorer2.upperBound(2) = 4.0, * 0.8 = 3.2
    // Verify indirectly by checking that scoring produces results
    const result = adaptive.scoreTopK();
    expect(result.length).toBeGreaterThan(0);
  });

  it("adaptive wand produces results", () => {
    const scorer1 = new MockScorer(1.0);
    const scorer2 = new MockScorer(0.5);
    const pl1 = makePostingList([
      [1, 0.9],
      [2, 0.7],
      [3, 0.5],
    ]);
    const pl2 = makePostingList([
      [1, 0.8],
      [2, 0.6],
      [4, 0.3],
    ]);

    const adaptive = new AdaptiveWANDScorer(
      [scorer1, scorer2],
      2,
      [pl1, pl2],
      null,
      null,
      null,
      0.9,
    );
    const result = adaptive.scoreTopK();
    expect(result.length).toBeLessThanOrEqual(2);
    expect(result.length).toBeGreaterThan(0);
  });

  it("adaptive wand analyzer populated", () => {
    const scorer1 = new MockScorer(1.0);
    const pl1 = makePostingList([
      [1, 0.5],
      [2, 0.8],
    ]);

    const adaptive = new AdaptiveWANDScorer([scorer1], 2, [pl1], null, null, null, 0.9);
    adaptive.scoreTopK();

    // Analyzer should be accessible
    expect(adaptive._analyzer).toBeDefined();
  });
});

// -- TightenedFusionWANDScorer tests --

describe("TightenedFusionWANDScorer", () => {
  it("tightened fusion wand produces results", () => {
    const pl1 = makePostingList([
      [1, 0.9],
      [2, 0.7],
      [3, 0.5],
    ]);
    const pl2 = makePostingList([
      [1, 0.8],
      [2, 0.6],
      [4, 0.4],
    ]);

    const scorer = new TightenedFusionWANDScorer(
      [pl1, pl2],
      [0.95, 0.85],
      0.5,
      2,
      null,
      0.9,
    );
    const result = scorer.scoreTopK();
    expect(result.length).toBeLessThanOrEqual(2);
    expect(result.length).toBeGreaterThan(0);
  });

  it("tightened fusion analyzer", () => {
    const pl1 = makePostingList([
      [1, 0.9],
      [2, 0.7],
    ]);
    const pl2 = makePostingList([
      [1, 0.8],
      [3, 0.4],
    ]);

    const scorer = new TightenedFusionWANDScorer(
      [pl1, pl2],
      [1.0, 1.0],
      0.5,
      2,
      null,
      0.85,
    );
    scorer.scoreTopK();

    // Analyzer should be accessible
    expect(scorer.boundAnalyzer).toBeDefined();
  });

  it("tightened fusion preserves original bounds", () => {
    const pl1 = makePostingList([[1, 0.9]]);
    const scorer = new TightenedFusionWANDScorer([pl1], [1.0], 0.5, 1, null, 0.8);
    // original_bounds should be untouched
    expect(scorer["_originalBounds"]).toEqual([1.0]);
    // But signal_upper_bounds should be tightened
    expect(scorer["_signalUpperBounds"][0]).toBeCloseTo(0.8);
  });
});
