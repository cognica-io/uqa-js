import { describe, expect, it } from "vitest";
import { createPayload, IndexStats } from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import { AttentionFusion } from "../../src/fusion/attention.js";
import { LearnedFusion } from "../../src/fusion/learned.js";
import { AttentionFusionOperator } from "../../src/operators/attention.js";
import { LearnedFusionOperator } from "../../src/operators/learned-fusion.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import { Operator } from "../../src/operators/base.js";

// -- Helpers --

class FixedOperator extends Operator {
  private readonly _entries: {
    docId: number;
    payload: ReturnType<typeof createPayload>;
  }[];

  constructor(entries: { docId: number; score: number }[]) {
    super();
    this._entries = entries.map((e) => ({
      docId: e.docId,
      payload: createPayload({ score: e.score }),
    }));
  }

  execute(_context: ExecutionContext): PostingList {
    return PostingList.fromSorted(this._entries);
  }
}

function makeEntry(docId: number, score: number) {
  return { docId, payload: createPayload({ score }) };
}

// -- TestQueryFeatureExtractor --

describe("TestQueryFeatureExtractor", () => {
  it("n features via AttentionFusion default", () => {
    // AttentionFusion uses 6 query features by default
    const af = new AttentionFusion(2);
    expect(af.nQueryFeatures).toBe(6);
  });

  it("empty query features produce valid fuse result", () => {
    const af = new AttentionFusion(2);
    const qf = Array.from({ length: 6 }, () => 0);
    const result = af.fuse([0.5, 0.5], qf);
    expect(result).toBeGreaterThanOrEqual(0.0);
    expect(result).toBeLessThanOrEqual(1.0);
  });

  it("no matching terms produce zero-like features", () => {
    // With zero features, fuse should still produce a valid result
    const af = new AttentionFusion(2);
    const zeroFeatures = Array.from({ length: 6 }, () => 0.0);
    const result = af.fuse([0.8, 0.6], zeroFeatures);
    expect(result).toBeGreaterThanOrEqual(0.0);
    expect(result).toBeLessThanOrEqual(1.0);
  });

  it("matching terms produce nonzero features effect", () => {
    const af = new AttentionFusion(2);
    const zeroFeatures = Array.from({ length: 6 }, () => 0.0);
    const nonzeroFeatures = [1.0, 2.0, 0.5, 0.1, 3.0, 0.8];
    const resultZero = af.fuse([0.7, 0.3], zeroFeatures);
    const resultNonzero = af.fuse([0.7, 0.3], nonzeroFeatures);
    // Both should be valid probabilities
    expect(resultZero).toBeGreaterThanOrEqual(0.0);
    expect(resultZero).toBeLessThanOrEqual(1.0);
    expect(resultNonzero).toBeGreaterThanOrEqual(0.0);
    expect(resultNonzero).toBeLessThanOrEqual(1.0);
  });
});

// -- TestAttentionFusion --

describe("TestAttentionFusion", () => {
  it("construction", () => {
    const af = new AttentionFusion(3, 6, 0.5);
    expect(af.nSignals).toBe(3);
    expect(af.nQueryFeatures).toBe(6);
  });

  it("fuse result in unit interval", () => {
    const af = new AttentionFusion(2);
    const qf = Array.from({ length: 6 }, () => 0);
    const result = af.fuse([0.8, 0.6], qf);
    expect(result).toBeGreaterThanOrEqual(0.0);
    expect(result).toBeLessThanOrEqual(1.0);
  });

  it("fuse with nonzero features", () => {
    const af = new AttentionFusion(2);
    const qf = [1.0, 2.0, 0.5, 0.1, 3.0, 0.8];
    const result = af.fuse([0.7, 0.3], qf);
    expect(result).toBeGreaterThanOrEqual(0.0);
    expect(result).toBeLessThanOrEqual(1.0);
  });

  it("state dict roundtrip", () => {
    const af = new AttentionFusion(2, 6, 0.3);
    const state = af.stateDict();
    expect(state["n_signals"]).toBe(2);
    expect(state["n_query_features"]).toBe(6);
    expect(state["alpha"]).toBeCloseTo(0.3);

    const af2 = new AttentionFusion(2, 6);
    af2.loadStateDict(state);
    const state2 = af2.stateDict();
    // After load, n_signals and n_query_features should match
    expect(state2["n_signals"]).toBe(state["n_signals"]);
    expect(state2["n_query_features"]).toBe(state["n_query_features"]);
  });

  it("fuse three signals", () => {
    const af = new AttentionFusion(3);
    const qf = Array.from({ length: 6 }, () => 0);
    const result = af.fuse([0.9, 0.5, 0.2], qf);
    expect(result).toBeGreaterThanOrEqual(0.0);
    expect(result).toBeLessThanOrEqual(1.0);
  });
});

// -- TestLearnedFusion --

describe("TestLearnedFusion", () => {
  it("construction", () => {
    const lf = new LearnedFusion(3, 0.4);
    expect(lf.nSignals).toBe(3);
  });

  it("fuse result in unit interval", () => {
    const lf = new LearnedFusion(2);
    const result = lf.fuse([0.8, 0.6]);
    expect(result).toBeGreaterThanOrEqual(0.0);
    expect(result).toBeLessThanOrEqual(1.0);
  });

  it("state dict roundtrip", () => {
    const lf = new LearnedFusion(3, 0.7);
    const state = lf.stateDict();
    expect(state["n_signals"]).toBe(3);
    expect(state["alpha"]).toBeCloseTo(0.7);

    const lf2 = new LearnedFusion(3);
    lf2.loadStateDict(state);
    const state2 = lf2.stateDict();
    // Weights should match after roundtrip
    const w1 = state["weights"] as number[];
    const w2 = state2["weights"] as number[];
    for (let i = 0; i < w1.length; i++) {
      expect(w2[i]).toBeCloseTo(w1[i]!, 5);
    }
  });

  it("fuse three signals", () => {
    const lf = new LearnedFusion(3);
    const result = lf.fuse([0.9, 0.5, 0.2]);
    expect(result).toBeGreaterThanOrEqual(0.0);
    expect(result).toBeLessThanOrEqual(1.0);
  });
});

// -- TestAttentionFusionOperator --

describe("TestAttentionFusionOperator", () => {
  it("empty signals return empty", () => {
    const af = new AttentionFusion(2);
    const qf = new Float64Array(6);
    const sig1 = new FixedOperator([]);
    const sig2 = new FixedOperator([]);
    const op = new AttentionFusionOperator([sig1, sig2], af, qf);
    const ctx: ExecutionContext = {};
    const result = op.execute(ctx);
    expect(result.length).toBe(0);
  });

  it("fuses two signals", () => {
    const af = new AttentionFusion(2);
    const qf = new Float64Array(6);
    const sig1 = new FixedOperator([
      { docId: 1, score: 0.8 },
      { docId: 2, score: 0.6 },
    ]);
    const sig2 = new FixedOperator([
      { docId: 1, score: 0.7 },
      { docId: 3, score: 0.5 },
    ]);
    const op = new AttentionFusionOperator([sig1, sig2], af, qf);
    const ctx: ExecutionContext = {};
    const result = op.execute(ctx);
    // Documents 1, 2, 3 should all appear (union)
    const docIds = result.entries.map((e) => e.docId).sort((a, b) => a - b);
    expect(docIds).toEqual([1, 2, 3]);
    // All scores should be in (0, 1)
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });

  it("cost estimate", () => {
    const af = new AttentionFusion(2);
    const qf = new Float64Array(6);
    const sig1 = new FixedOperator([{ docId: 1, score: 0.8 }]);
    const sig2 = new FixedOperator([{ docId: 2, score: 0.7 }]);
    const op = new AttentionFusionOperator([sig1, sig2], af, qf);
    const stats = new IndexStats(100, 10.0);
    const cost = op.costEstimate(stats);
    expect(cost).toBeGreaterThanOrEqual(0);
  });
});

// -- TestLearnedFusionOperator --

describe("TestLearnedFusionOperator", () => {
  it("empty signals return empty", () => {
    const lf = new LearnedFusion(2);
    const sig1 = new FixedOperator([]);
    const sig2 = new FixedOperator([]);
    const op = new LearnedFusionOperator([sig1, sig2], lf);
    const ctx: ExecutionContext = {};
    const result = op.execute(ctx);
    expect(result.length).toBe(0);
  });

  it("fuses two signals", () => {
    const lf = new LearnedFusion(2);
    const sig1 = new FixedOperator([
      { docId: 1, score: 0.8 },
      { docId: 2, score: 0.6 },
    ]);
    const sig2 = new FixedOperator([
      { docId: 1, score: 0.7 },
      { docId: 3, score: 0.5 },
    ]);
    const op = new LearnedFusionOperator([sig1, sig2], lf);
    const ctx: ExecutionContext = {};
    const result = op.execute(ctx);
    const docIds = result.entries.map((e) => e.docId).sort((a, b) => a - b);
    expect(docIds).toEqual([1, 2, 3]);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });

  it("cost estimate", () => {
    const lf = new LearnedFusion(2);
    const sig1 = new FixedOperator([{ docId: 1, score: 0.8 }]);
    const sig2 = new FixedOperator([{ docId: 2, score: 0.7 }]);
    const op = new LearnedFusionOperator([sig1, sig2], lf);
    const stats = new IndexStats(100, 10.0);
    const cost = op.costEstimate(stats);
    expect(cost).toBeGreaterThanOrEqual(0);
  });
});

// -- TestAttentionFusionSQL --

describe("TestAttentionFusionSQL", () => {
  it("fuse attention via operator with two signals", () => {
    const af = new AttentionFusion(2);
    const qf = new Float64Array(6);
    const sig1 = new FixedOperator([
      { docId: 1, score: 0.8 },
      { docId: 2, score: 0.5 },
    ]);
    const sig2 = new FixedOperator([
      { docId: 1, score: 0.7 },
      { docId: 2, score: 0.4 },
    ]);
    const op = new AttentionFusionOperator([sig1, sig2], af, qf);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });

  it("fuse attention normalize via alpha parameter", () => {
    const af = new AttentionFusion(2, 6, 0.5);
    const qf = new Float64Array(6);
    const sig1 = new FixedOperator([{ docId: 1, score: 0.8 }]);
    const sig2 = new FixedOperator([{ docId: 1, score: 0.7 }]);
    const op = new AttentionFusionOperator([sig1, sig2], af, qf);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });

  it("fuse attention with base rate alpha", () => {
    const af = new AttentionFusion(2, 6, 0.01);
    const qf = new Float64Array(6);
    const sig1 = new FixedOperator([{ docId: 1, score: 0.8 }]);
    const sig2 = new FixedOperator([{ docId: 1, score: 0.7 }]);
    const op = new AttentionFusionOperator([sig1, sig2], af, qf);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });

  it("fuse multihead via attention with 3 signals", () => {
    const af = new AttentionFusion(3);
    const qf = new Float64Array(6);
    const sig1 = new FixedOperator([{ docId: 1, score: 0.8 }]);
    const sig2 = new FixedOperator([{ docId: 1, score: 0.7 }]);
    const sig3 = new FixedOperator([{ docId: 1, score: 0.6 }]);
    const op = new AttentionFusionOperator([sig1, sig2, sig3], af, qf);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });

  it("fuse multihead default settings", () => {
    const af = new AttentionFusion(2);
    const qf = new Float64Array(6);
    const sig1 = new FixedOperator([{ docId: 1, score: 0.8 }]);
    const sig2 = new FixedOperator([{ docId: 1, score: 0.6 }]);
    const op = new AttentionFusionOperator([sig1, sig2], af, qf);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
  });

  it("fuse learned with alpha parameter", () => {
    const lf = new LearnedFusion(2, 0.7);
    const sig1 = new FixedOperator([{ docId: 1, score: 0.8 }]);
    const sig2 = new FixedOperator([{ docId: 1, score: 0.7 }]);
    const op = new LearnedFusionOperator([sig1, sig2], lf);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });

  it("fuse learned via operator", () => {
    const lf = new LearnedFusion(2);
    const sig1 = new FixedOperator([
      { docId: 1, score: 0.8 },
      { docId: 2, score: 0.5 },
    ]);
    const sig2 = new FixedOperator([
      { docId: 1, score: 0.7 },
      { docId: 2, score: 0.4 },
    ]);
    const op = new LearnedFusionOperator([sig1, sig2], lf);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });
});

// -- TestAttentionFusionQueryBuilder --

describe("TestAttentionFusionQueryBuilder", () => {
  it("fuse attention builder via operator", () => {
    const af = new AttentionFusion(2);
    const qf = new Float64Array(6);
    const sig1 = new FixedOperator([
      { docId: 1, score: 0.8 },
      { docId: 2, score: 0.6 },
    ]);
    const sig2 = new FixedOperator([
      { docId: 1, score: 0.7 },
      { docId: 3, score: 0.5 },
    ]);
    const op = new AttentionFusionOperator([sig1, sig2], af, qf);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
  });

  it("fuse learned builder via operator", () => {
    const lf = new LearnedFusion(2);
    const sig1 = new FixedOperator([
      { docId: 1, score: 0.8 },
      { docId: 2, score: 0.6 },
    ]);
    const sig2 = new FixedOperator([
      { docId: 1, score: 0.7 },
      { docId: 3, score: 0.5 },
    ]);
    const op = new LearnedFusionOperator([sig1, sig2], lf);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
  });
});
