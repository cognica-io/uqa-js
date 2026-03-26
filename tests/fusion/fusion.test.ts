import { describe, expect, it } from "vitest";
import { ProbabilisticBoolean } from "../../src/fusion/boolean.js";
import { AdaptiveLogOddsFusion, LogOddsFusion } from "../../src/fusion/log-odds.js";
import {
  AttentionFusion,
  MultiHeadAttentionFusion,
} from "../../src/fusion/attention.js";
import { LearnedFusion } from "../../src/fusion/learned.js";
import { QueryFeatureExtractor } from "../../src/fusion/query-features.js";
import { MemoryInvertedIndex } from "../../src/storage/inverted-index.js";

// -- ProbabilisticBoolean ----------------------------------------------------

describe("ProbabilisticBoolean", () => {
  it("prob_and returns product-like probability", () => {
    const result = ProbabilisticBoolean.probAnd([0.8, 0.9]);
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThanOrEqual(0.8);
  });

  it("prob_or returns union-like probability", () => {
    const result = ProbabilisticBoolean.probOr([0.3, 0.4]);
    expect(result).toBeGreaterThanOrEqual(0.4);
    expect(result).toBeLessThanOrEqual(1);
  });

  it("prob_not returns complement", () => {
    expect(ProbabilisticBoolean.probNot(0.7)).toBeCloseTo(0.3, 8);
    expect(ProbabilisticBoolean.probNot(0.0)).toBeCloseTo(1.0, 8);
    expect(ProbabilisticBoolean.probNot(1.0)).toBeCloseTo(0.0, 8);
  });
});

// -- LogOddsFusion -----------------------------------------------------------

describe("LogOddsFusion", () => {
  it("fuse returns 0.5 for empty", () => {
    const f = new LogOddsFusion();
    expect(f.fuse([])).toBe(0.5);
  });

  it("fuse returns single probability unchanged", () => {
    const f = new LogOddsFusion();
    expect(f.fuse([0.7])).toBe(0.7);
  });

  it("fuse combines multiple probabilities", () => {
    const f = new LogOddsFusion(0.5);
    const result = f.fuse([0.8, 0.9]);
    expect(result).toBeGreaterThan(0.5);
    expect(result).toBeLessThanOrEqual(1);
  });

  it("fuseMean returns log-odds mean", () => {
    const f = new LogOddsFusion();
    const result = f.fuseMean([0.8, 0.8]);
    expect(result).toBeCloseTo(0.8, 5);
  });

  it("fuseMean returns 0.5 for empty", () => {
    expect(new LogOddsFusion().fuseMean([])).toBe(0.5);
  });

  it("fuseWeighted applies weights", () => {
    const f = new LogOddsFusion(0.5);
    const result = f.fuseWeighted([0.9, 0.1], [1.0, 0.0]);
    // With weight 0 on second signal, should be close to first
    expect(result).toBeGreaterThan(0.5);
  });
});

describe("AdaptiveLogOddsFusion", () => {
  it("computes signal alpha from quality", () => {
    const f = new AdaptiveLogOddsFusion(0.5);
    const alpha = f.computeSignalAlpha({
      coverageRatio: 0.8,
      scoreVariance: 0.1,
      calibrationError: 0.05,
    });
    expect(alpha).toBeGreaterThan(0);
    expect(alpha).toBeLessThanOrEqual(1);
  });

  it("low quality gives low alpha", () => {
    const f = new AdaptiveLogOddsFusion(0.5);
    const low = f.computeSignalAlpha({
      coverageRatio: 0.1,
      scoreVariance: 2.0,
      calibrationError: 0.5,
    });
    const high = f.computeSignalAlpha({
      coverageRatio: 0.9,
      scoreVariance: 0.01,
      calibrationError: 0.01,
    });
    expect(high).toBeGreaterThan(low);
  });

  it("fuseAdaptive returns probability", () => {
    const f = new AdaptiveLogOddsFusion(0.5);
    const result = f.fuseAdaptive(
      [0.7, 0.8],
      [
        { coverageRatio: 0.9, scoreVariance: 0.1, calibrationError: 0.05 },
        { coverageRatio: 0.5, scoreVariance: 0.3, calibrationError: 0.1 },
      ],
    );
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThan(1);
  });
});

// -- AttentionFusion ---------------------------------------------------------

describe("AttentionFusion", () => {
  it("fuse returns probability", () => {
    const af = new AttentionFusion(2, 3, 0.5);
    const result = af.fuse([0.7, 0.8], [1, 0, 0]);
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThan(1);
  });

  it("state_dict round-trips", () => {
    const af = new AttentionFusion(2, 3);
    const state = af.stateDict();
    expect(state["n_signals"]).toBe(2);
    expect(state["n_query_features"]).toBe(3);
    const af2 = new AttentionFusion(2, 3);
    af2.loadStateDict(state);
    expect(af2.nSignals).toBe(2);
  });
});

describe("MultiHeadAttentionFusion", () => {
  it("fuse returns probability", () => {
    const mh = new MultiHeadAttentionFusion(2, 2, 3, 0.5);
    const result = mh.fuse([0.7, 0.8], [1, 0, 0]);
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThan(1);
  });

  it("state_dict round-trips", () => {
    const mh = new MultiHeadAttentionFusion(3, 2, 4);
    const state = mh.stateDict();
    expect(state["n_heads"]).toBe(2);
    expect(state["n_signals"]).toBe(3);
    const mh2 = new MultiHeadAttentionFusion(3, 2, 4);
    mh2.loadStateDict(state);
    expect(mh2.nQueryFeatures).toBe(4);
  });
});

// -- LearnedFusion -----------------------------------------------------------

describe("LearnedFusion", () => {
  it("fuse returns probability", () => {
    const lf = new LearnedFusion(2, 0.5);
    const result = lf.fuse([0.7, 0.8]);
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThan(1);
  });

  it("state_dict round-trips", () => {
    const lf = new LearnedFusion(3);
    const state = lf.stateDict();
    expect(state["n_signals"]).toBe(3);
    const lf2 = new LearnedFusion(3);
    lf2.loadStateDict(state);
    expect(lf2.nSignals).toBe(3);
  });
});

// -- QueryFeatureExtractor ---------------------------------------------------

describe("QueryFeatureExtractor", () => {
  it("extracts 6 features", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { text: "hello world" });
    idx.addDocument(2, { text: "world peace" });
    const qfe = new QueryFeatureExtractor(idx);
    expect(qfe.nFeatures).toBe(6);
    const features = qfe.extract(["hello", "world"], "text");
    expect(features.length).toBe(6);
  });

  it("returns zeros for empty index", () => {
    const idx = new MemoryInvertedIndex();
    const qfe = new QueryFeatureExtractor(idx);
    const features = qfe.extract(["test"]);
    expect(features.length).toBe(6);
    expect([...features].every((v) => v === 0)).toBe(true);
  });

  it("mean_idf is positive for matching terms", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { text: "hello world" });
    idx.addDocument(2, { text: "world peace" });
    const qfe = new QueryFeatureExtractor(idx);
    const features = qfe.extract(["hello"], "text");
    expect(features[0]).toBeGreaterThan(0); // mean_idf
  });

  it("query_length reflects input", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { text: "a b c" });
    const qfe = new QueryFeatureExtractor(idx);
    const features = qfe.extract(["a", "b", "c"]);
    expect(features[4]).toBe(3); // query_length
  });
});
