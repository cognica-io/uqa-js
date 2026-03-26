import { describe, expect, it } from "vitest";
import { createPayload } from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import { FusionWANDScorer } from "../../src/scoring/fusion-wand.js";
import { LogOddsFusionOperator } from "../../src/operators/hybrid.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import { Operator } from "../../src/operators/base.js";

// -- Helpers --

function makePostingList(entries: [number, number][]): PostingList {
  return new PostingList(
    entries.map(([docId, score]) => ({
      docId,
      payload: createPayload({ score }),
    })),
  );
}

class FixedOperator extends Operator {
  private readonly _entries: [number, number][];

  constructor(entries: [number, number][]) {
    super();
    this._entries = entries;
  }

  execute(_context: ExecutionContext): PostingList {
    return makePostingList(this._entries);
  }
}

// -- TestFusionWANDScorer --

describe("TestFusionWANDScorer", () => {
  it("basic top k", () => {
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
    const scorer = new FusionWANDScorer([pl1, pl2], [0.9, 0.8], 0.5, 2);
    const result = scorer.scoreTopK();
    expect(result.length).toBe(2);
  });

  it("top k returns highest", () => {
    const pl1 = makePostingList([
      [1, 0.9],
      [2, 0.3],
      [3, 0.1],
    ]);
    const pl2 = makePostingList([
      [1, 0.8],
      [2, 0.2],
      [3, 0.1],
    ]);
    const scorer = new FusionWANDScorer([pl1, pl2], [0.9, 0.8], 0.5, 1);
    const result = scorer.scoreTopK();
    expect(result.length).toBe(1);
    expect(result.entries[0]!.docId).toBe(1);
  });

  it("top k larger than docs", () => {
    const pl1 = makePostingList([[1, 0.7]]);
    const pl2 = makePostingList([[1, 0.6]]);
    const scorer = new FusionWANDScorer([pl1, pl2], [0.7, 0.6], 0.5, 10);
    const result = scorer.scoreTopK();
    expect(result.length).toBe(1);
  });

  it("empty signals", () => {
    const scorer = new FusionWANDScorer([], [], 0.5, 5);
    const result = scorer.scoreTopK();
    expect(result.length).toBe(0);
  });

  it("single signal", () => {
    const pl = makePostingList([
      [1, 0.9],
      [2, 0.3],
    ]);
    const scorer = new FusionWANDScorer([pl], [0.9], 0.5, 1);
    const result = scorer.scoreTopK();
    expect(result.length).toBe(1);
    expect(result.entries[0]!.docId).toBe(1);
  });

  it("fused upper bound", () => {
    // Create a scorer with known upper bounds and verify the fused bound is in (0, 1)
    const pl1 = makePostingList([
      [1, 0.9],
      [2, 0.7],
    ]);
    const pl2 = makePostingList([
      [1, 0.8],
      [2, 0.5],
    ]);
    const scorer = new FusionWANDScorer([pl1, pl2], [0.9, 0.8], 0.5, 5);
    const result = scorer.scoreTopK();
    // All fused scores should be bounded by the fused upper bound
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });

  it("scores are probabilities", () => {
    const pl1 = makePostingList([
      [1, 0.7],
      [2, 0.6],
    ]);
    const pl2 = makePostingList([
      [1, 0.8],
      [2, 0.5],
    ]);
    const scorer = new FusionWANDScorer([pl1, pl2], [0.8, 0.8], 0.5, 5);
    const result = scorer.scoreTopK();
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });

  it("alpha parameter", () => {
    const pl1 = makePostingList([[1, 0.7]]);
    const pl2 = makePostingList([[1, 0.6]]);
    const s1 = new FusionWANDScorer([pl1, pl2], [0.7, 0.6], 0.1, 5);
    const s2 = new FusionWANDScorer([pl1, pl2], [0.7, 0.6], 0.9, 5);
    const r1 = s1.scoreTopK();
    const r2 = s2.scoreTopK();
    // Different alpha should produce different scores
    expect(
      Math.abs(r1.entries[0]!.payload.score - r2.entries[0]!.payload.score),
    ).toBeGreaterThan(1e-3);
  });

  it("wand gating relu", () => {
    const pl1 = makePostingList([
      [1, 0.7],
      [2, 0.6],
    ]);
    const pl2 = makePostingList([
      [1, 0.8],
      [2, 0.5],
    ]);
    const scorer = new FusionWANDScorer([pl1, pl2], [0.8, 0.8], 0.5, 5, "relu");
    const result = scorer.scoreTopK();
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThanOrEqual(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });

  it("wand gating swish", () => {
    const pl1 = makePostingList([
      [1, 0.7],
      [2, 0.6],
    ]);
    const pl2 = makePostingList([
      [1, 0.8],
      [2, 0.5],
    ]);
    const scorer = new FusionWANDScorer([pl1, pl2], [0.8, 0.8], 0.5, 5, "swish");
    const result = scorer.scoreTopK();
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThanOrEqual(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });
});

// -- TestLogOddsFusionTopK --

describe("TestLogOddsFusionTopK", () => {
  it("top k parameter", () => {
    const sig1 = new FixedOperator([
      [1, 0.9],
      [2, 0.7],
      [3, 0.5],
      [4, 0.3],
    ]);
    const sig2 = new FixedOperator([
      [1, 0.8],
      [2, 0.6],
      [3, 0.4],
      [4, 0.2],
    ]);
    const op = new LogOddsFusionOperator([sig1, sig2], 0.5, 2);
    const result = op.execute({});
    expect(result.length).toBe(2);
  });

  it("without top k returns all", () => {
    const sig1 = new FixedOperator([
      [1, 0.9],
      [2, 0.7],
    ]);
    const sig2 = new FixedOperator([
      [1, 0.8],
      [2, 0.6],
    ]);
    const op = new LogOddsFusionOperator([sig1, sig2]);
    const result = op.execute({});
    expect(result.length).toBe(2);
  });

  it("top k preserves ranking", () => {
    const sig1 = new FixedOperator([
      [1, 0.9],
      [2, 0.3],
      [3, 0.1],
    ]);
    const sig2 = new FixedOperator([
      [1, 0.8],
      [2, 0.2],
      [3, 0.1],
    ]);
    const op = new LogOddsFusionOperator([sig1, sig2], 0.5, 2);
    const result = op.execute({});
    const docIds = result.entries.map((e) => e.docId).sort((a, b) => a - b);
    expect(docIds).toContain(1);
  });

  it("top k results match full results", () => {
    const sig1a = new FixedOperator([
      [1, 0.9],
      [2, 0.7],
      [3, 0.5],
    ]);
    const sig2a = new FixedOperator([
      [1, 0.8],
      [2, 0.6],
      [3, 0.4],
    ]);
    const fullOp = new LogOddsFusionOperator([sig1a, sig2a]);
    const fullResult = fullOp.execute({});
    const fullScores = fullResult.entries
      .map((e) => ({ docId: e.docId, score: e.payload.score }))
      .sort((a, b) => b.score - a.score);

    const sig1b = new FixedOperator([
      [1, 0.9],
      [2, 0.7],
      [3, 0.5],
    ]);
    const sig2b = new FixedOperator([
      [1, 0.8],
      [2, 0.6],
      [3, 0.4],
    ]);
    const topOp = new LogOddsFusionOperator([sig1b, sig2b], 0.5, 2);
    const topResult = topOp.execute({});
    const topScores = topResult.entries
      .map((e) => ({ docId: e.docId, score: e.payload.score }))
      .sort((a, b) => b.score - a.score);

    for (const { docId, score } of topScores) {
      const matching = fullScores.filter((fs) => fs.docId === docId);
      expect(matching.length).toBe(1);
      expect(score).toBeCloseTo(matching[0]!.score, 5);
    }
  });
});

// -- TestFusionWANDSQL --

describe("TestFusionWANDSQL", () => {
  it("log odds fusion with limit", () => {
    // Test top-k behavior: create operator with k=1 and verify only 1 result
    const sig1 = new FixedOperator([
      [1, 0.9],
      [2, 0.7],
      [3, 0.5],
    ]);
    const sig2 = new FixedOperator([
      [1, 0.8],
      [2, 0.6],
      [3, 0.4],
    ]);
    const op = new LogOddsFusionOperator([sig1, sig2], 0.5, 1);
    const result = op.execute({});
    expect(result.length).toBe(1);
  });

  it("fusion result scores", () => {
    const sig1 = new FixedOperator([
      [1, 0.9],
      [2, 0.7],
    ]);
    const sig2 = new FixedOperator([
      [1, 0.8],
      [2, 0.6],
    ]);
    const op = new LogOddsFusionOperator([sig1, sig2]);
    const result = op.execute({});
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });

  it("log odds with gating relu", () => {
    const pl1 = makePostingList([
      [1, 0.7],
      [2, 0.6],
    ]);
    const pl2 = makePostingList([
      [1, 0.8],
      [2, 0.5],
    ]);
    const scorer = new FusionWANDScorer([pl1, pl2], [0.8, 0.8], 0.5, 5, "relu");
    const result = scorer.scoreTopK();
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThanOrEqual(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });

  it("log odds with gating swish", () => {
    const pl1 = makePostingList([
      [1, 0.7],
      [2, 0.6],
    ]);
    const pl2 = makePostingList([
      [1, 0.8],
      [2, 0.5],
    ]);
    const scorer = new FusionWANDScorer([pl1, pl2], [0.8, 0.8], 0.5, 5, "swish");
    const result = scorer.scoreTopK();
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThanOrEqual(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });
});
