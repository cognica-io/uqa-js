import { describe, expect, it } from "vitest";
import { createPayload, IndexStats } from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import { AdaptiveLogOddsFusion, LogOddsFusion } from "../../src/fusion/log-odds.js";
import type { SignalQuality } from "../../src/fusion/log-odds.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import { Operator } from "../../src/operators/base.js";
import { AdaptiveLogOddsFusionOperator } from "../../src/operators/hybrid.js";

// -- Helper --

class ConstantOperator extends Operator {
  private readonly _pl: PostingList;

  constructor(pl: PostingList) {
    super();
    this._pl = pl;
  }

  execute(_context: ExecutionContext): PostingList {
    return this._pl;
  }
}

// -- SignalQuality tests --

describe("SignalQuality", () => {
  it("signal quality creation", () => {
    const sq: SignalQuality = {
      coverageRatio: 0.8,
      scoreVariance: 0.05,
      calibrationError: 0.1,
    };
    expect(sq.coverageRatio).toBe(0.8);
    expect(sq.scoreVariance).toBe(0.05);
    expect(sq.calibrationError).toBe(0.1);
  });
});

// -- compute_signal_alpha tests --

describe("computeSignalAlpha", () => {
  it("high quality", () => {
    const fusion = new AdaptiveLogOddsFusion(0.5);
    const sq: SignalQuality = {
      coverageRatio: 1.0,
      scoreVariance: 0.0,
      calibrationError: 0.0,
    };
    const alpha = fusion.computeSignalAlpha(sq);
    // alpha = 0.5 * (1.0 * 1.0) / 1.0 = 0.5
    expect(alpha).toBeCloseTo(0.5, 8);
  });

  it("low quality", () => {
    const fusion = new AdaptiveLogOddsFusion(0.5);
    const sq: SignalQuality = {
      coverageRatio: 0.1,
      scoreVariance: 5.0,
      calibrationError: 0.4,
    };
    const alpha = fusion.computeSignalAlpha(sq);
    // alpha = 0.5 * (0.1 * 0.6) / 6.0 = 0.005 -> clamped to 0.01
    expect(alpha).toBeCloseTo(0.01, 8);
  });

  it("clamping", () => {
    const fusion = new AdaptiveLogOddsFusion(0.5);

    // Near-zero coverage should clamp to 0.01
    const sqLow: SignalQuality = {
      coverageRatio: 0.0,
      scoreVariance: 0.0,
      calibrationError: 0.0,
    };
    const alphaLow = fusion.computeSignalAlpha(sqLow);
    expect(alphaLow).toBeCloseTo(0.01, 8);

    // Very high base_alpha with perfect quality should clamp to 1.0
    const fusionHigh = new AdaptiveLogOddsFusion(5.0);
    const sqHigh: SignalQuality = {
      coverageRatio: 1.0,
      scoreVariance: 0.0,
      calibrationError: 0.0,
    };
    const alphaHigh = fusionHigh.computeSignalAlpha(sqHigh);
    expect(alphaHigh).toBeCloseTo(1.0, 8);
  });
});

// -- fuse_adaptive tests --

describe("fuseAdaptive", () => {
  it("single signal returns original probability", () => {
    const fusion = new AdaptiveLogOddsFusion(0.5);
    const sq: SignalQuality = {
      coverageRatio: 1.0,
      scoreVariance: 0.0,
      calibrationError: 0.0,
    };
    const result = fusion.fuseAdaptive([0.8], [sq]);
    expect(result).toBeCloseTo(0.8, 8);
  });

  it("uniform quality", () => {
    const adaptive = new AdaptiveLogOddsFusion(0.5);
    const standard = new LogOddsFusion(0.5);

    const probs = [0.7, 0.8, 0.6];
    const sq: SignalQuality = {
      coverageRatio: 1.0,
      scoreVariance: 0.0,
      calibrationError: 0.0,
    };
    const qualities = [sq, sq, sq];

    const adaptiveResult = adaptive.fuseAdaptive(probs, qualities);
    const standardResult = standard.fuse(probs);

    // Both should produce results on the same side of 0.5
    expect(adaptiveResult).toBeGreaterThan(0.5);
    expect(standardResult).toBeGreaterThan(0.5);
  });

  it("mixed quality", () => {
    const fusion = new AdaptiveLogOddsFusion(0.5);

    const highQ: SignalQuality = {
      coverageRatio: 1.0,
      scoreVariance: 0.0,
      calibrationError: 0.0,
    };
    const lowQ: SignalQuality = {
      coverageRatio: 0.1,
      scoreVariance: 5.0,
      calibrationError: 0.4,
    };

    // High-quality signal says relevant (0.9), low-quality says irrelevant (0.1)
    const resultHighFirst = fusion.fuseAdaptive([0.9, 0.1], [highQ, lowQ]);
    // Swap: low-quality says relevant (0.9), high-quality says irrelevant (0.1)
    const resultLowFirst = fusion.fuseAdaptive([0.1, 0.9], [highQ, lowQ]);

    // When high-quality signal says relevant, the result should be higher
    expect(resultHighFirst).toBeGreaterThan(resultLowFirst);
  });
});

// -- AdaptiveLogOddsFusionOperator tests --
// NOTE: AdaptiveLogOddsFusionOperator is not implemented in the JS port.
// These tests are skipped.

describe("AdaptiveLogOddsFusionOperator", () => {
  it("adaptive operator basic", () => {
    const pl1 = PostingList.fromSorted([
      { docId: 1, payload: createPayload({ score: 0.8 }) },
      { docId: 2, payload: createPayload({ score: 0.7 }) },
      { docId: 3, payload: createPayload({ score: 0.6 }) },
    ]);
    const pl2 = PostingList.fromSorted([
      { docId: 1, payload: createPayload({ score: 0.9 }) },
      { docId: 2, payload: createPayload({ score: 0.3 }) },
    ]);

    const op = new AdaptiveLogOddsFusionOperator(
      [new ConstantOperator(pl1), new ConstantOperator(pl2)],
      0.5,
    );
    const ctx: ExecutionContext = {};
    const result = op.execute(ctx);

    expect(result.length).toBe(3);
    const docIds = result.entries.map((e) => e.docId).sort((a, b) => a - b);
    expect(docIds).toEqual([1, 2, 3]);

    // All fused scores should be in (0, 1)
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });

  it("adaptive operator empty", () => {
    const op = new AdaptiveLogOddsFusionOperator([], 0.5);
    const ctx: ExecutionContext = {};
    const result = op.execute(ctx);
    expect(result.length).toBe(0);
  });

  it("adaptive operator cost estimate", () => {
    const pl1 = PostingList.fromSorted([
      { docId: 1, payload: createPayload({ score: 0.5 }) },
    ]);
    const pl2 = PostingList.fromSorted([
      { docId: 2, payload: createPayload({ score: 0.5 }) },
    ]);

    const op = new AdaptiveLogOddsFusionOperator([
      new ConstantOperator(pl1),
      new ConstantOperator(pl2),
    ]);
    const stats = new IndexStats(100);
    const cost = op.costEstimate(stats);
    // Cost should be positive (sum of child costs)
    expect(cost).toBeGreaterThan(0);
  });
});
