import { describe, expect, it } from "vitest";
import { BlockMaxIndex } from "../../src/storage/block-max-index.js";
import { PostingList } from "../../src/core/posting-list.js";
import { createPayload } from "../../src/core/types.js";

describe("BlockMaxIndex", () => {
  function makePostingList(n: number): PostingList {
    const entries = Array.from({ length: n }, (_, i) => ({
      docId: i,
      payload: createPayload({ positions: [0], score: 0 }),
    }));
    return new PostingList(entries);
  }

  const scorer = {
    score(termFreq: number, _docLength: number, _docFreq: number): number {
      return termFreq * 1.5;
    },
  };

  it("builds block max scores", () => {
    const bmi = new BlockMaxIndex(3);
    const pl = makePostingList(7);
    bmi.build(pl, scorer, "title", "hello");

    // 7 entries, block size 3 -> blocks: [0,1,2], [3,4,5], [6]
    expect(bmi.numBlocks("title", "hello")).toBe(3);
  });

  it("getBlockMax returns correct value", () => {
    const bmi = new BlockMaxIndex(3);
    const pl = makePostingList(6);
    bmi.build(pl, scorer, "title", "test");

    // Each entry has positions=[0], so tf=1, score=1.5
    expect(bmi.getBlockMax("title", "test", 0)).toBeCloseTo(1.5, 5);
    expect(bmi.getBlockMax("title", "test", 1)).toBeCloseTo(1.5, 5);
  });

  it("getBlockMax returns 0 for out-of-range block", () => {
    const bmi = new BlockMaxIndex(3);
    const pl = makePostingList(3);
    bmi.build(pl, scorer, "title", "test");
    expect(bmi.getBlockMax("title", "test", 99)).toBe(0);
  });

  it("getBlockMax returns 0 for unknown term", () => {
    const bmi = new BlockMaxIndex();
    expect(bmi.getBlockMax("title", "unknown", 0)).toBe(0);
  });

  it("numBlocks returns 0 for unknown term", () => {
    const bmi = new BlockMaxIndex();
    expect(bmi.numBlocks("title", "unknown")).toBe(0);
  });

  it("handles empty posting list", () => {
    const bmi = new BlockMaxIndex();
    const pl = new PostingList();
    bmi.build(pl, scorer, "title", "empty");
    expect(bmi.numBlocks("title", "empty")).toBe(0);
  });

  it("supports table_name parameter", () => {
    const bmi = new BlockMaxIndex(2);
    const pl = makePostingList(4);
    bmi.build(pl, scorer, "title", "hello", "articles");
    expect(bmi.numBlocks("title", "hello", "articles")).toBe(2);
    expect(bmi.numBlocks("title", "hello")).toBe(0); // different table
  });
});
