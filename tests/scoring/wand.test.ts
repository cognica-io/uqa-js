import { describe, expect, it } from "vitest";
import { createPayload } from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import { BoundTightnessAnalyzer, WANDScorer } from "../../src/scoring/wand.js";

describe("WANDScorer", () => {
  function makePostingList(
    docs: { id: number; tf: number; score: number }[],
  ): PostingList {
    return new PostingList(
      docs.map((d) => ({
        docId: d.id,
        payload: createPayload({
          positions: Array.from({ length: d.tf }, (_, i) => i),
          score: d.score,
        }),
      })),
    );
  }

  const simpleScorer = {
    score(tf: number, _dl: number, _df: number): number {
      return tf * 0.5;
    },
    upperBound(_df: number): number {
      return 10.0;
    },
  };

  it("returns top-k results", () => {
    const pl1 = makePostingList([
      { id: 1, tf: 3, score: 0 },
      { id: 2, tf: 1, score: 0 },
      { id: 3, tf: 5, score: 0 },
    ]);
    const pl2 = makePostingList([
      { id: 1, tf: 2, score: 0 },
      { id: 3, tf: 4, score: 0 },
      { id: 4, tf: 1, score: 0 },
    ]);

    const wand = new WANDScorer([simpleScorer, simpleScorer], 2, [pl1, pl2]);
    const result = wand.scoreTopK();
    expect(result.length).toBeLessThanOrEqual(2);
    expect(result.length).toBeGreaterThan(0);
  });

  it("returns empty for empty posting lists", () => {
    const wand = new WANDScorer([simpleScorer], 5, [new PostingList()]);
    expect(wand.scoreTopK().length).toBe(0);
  });

  it("returns all docs when k > total docs", () => {
    const pl = makePostingList([
      { id: 1, tf: 1, score: 0 },
      { id: 2, tf: 2, score: 0 },
    ]);
    const wand = new WANDScorer([simpleScorer], 100, [pl]);
    const result = wand.scoreTopK();
    expect(result.length).toBe(2);
  });
});

describe("BoundTightnessAnalyzer", () => {
  it("computes tightness ratio", () => {
    const a = new BoundTightnessAnalyzer();
    a.record(10, 8); // 0.8
    a.record(10, 10); // 1.0
    expect(a.tightnessRatio()).toBeCloseTo(0.9, 5);
  });

  it("computes slack", () => {
    const a = new BoundTightnessAnalyzer();
    a.record(10, 8);
    a.record(10, 10);
    expect(a.slack()).toBeCloseTo(0.1, 5);
  });

  it("returns 1.0 for empty", () => {
    expect(new BoundTightnessAnalyzer().tightnessRatio()).toBe(1.0);
  });

  it("finds worst bound index", () => {
    const a = new BoundTightnessAnalyzer();
    a.record(10, 9); // 0.9
    a.record(10, 5); // 0.5 -- worst
    a.record(10, 8); // 0.8
    expect(a.worstBoundIndex()).toBe(1);
  });

  it("clears data", () => {
    const a = new BoundTightnessAnalyzer();
    a.record(10, 5);
    a.clear();
    expect(a.tightnessRatio()).toBe(1.0);
  });
});
