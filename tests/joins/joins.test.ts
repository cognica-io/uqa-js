import { describe, expect, it } from "vitest";
import type { PostingEntry } from "../../src/core/types.js";
import { createPayload } from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import type { Operator } from "../../src/operators/base.js";
import { InnerJoinOperator } from "../../src/joins/inner.js";
import {
  FullOuterJoinOperator,
  LeftOuterJoinOperator,
  RightOuterJoinOperator,
} from "../../src/joins/outer.js";
import { AntiJoinOperator, SemiJoinOperator } from "../../src/joins/semi.js";
import { SortMergeJoinOperator } from "../../src/joins/sort-merge.js";
import { IndexJoinOperator } from "../../src/joins/index.js";
import { CrossJoinOperator } from "../../src/joins/cross.js";
import { TextSimilarityJoinOperator } from "../../src/joins/cross-paradigm.js";

function makeEntries(
  data: { id: number; fields: Record<string, unknown> }[],
): PostingEntry[] {
  return data.map((d) => ({
    docId: d.id,
    payload: createPayload({ score: 1.0, fields: d.fields }),
  }));
}

const leftData = makeEntries([
  { id: 1, fields: { dept: "eng", name: "Alice" } },
  { id: 2, fields: { dept: "sales", name: "Bob" } },
  { id: 3, fields: { dept: "eng", name: "Charlie" } },
]);

const rightData = makeEntries([
  { id: 10, fields: { dept: "eng", budget: 100 } },
  { id: 20, fields: { dept: "hr", budget: 50 } },
]);

const cond = { leftField: "dept", rightField: "dept" };

describe("InnerJoinOperator", () => {
  it("joins on matching field", () => {
    const op = new InnerJoinOperator(leftData, rightData, cond);
    const result = op.execute({});
    // Alice(eng) + eng, Charlie(eng) + eng = 2 results
    expect(result.length).toBe(2);
  });

  it("produces empty on no match", () => {
    const left = makeEntries([{ id: 1, fields: { dept: "xyz" } }]);
    const op = new InnerJoinOperator(left, rightData, cond);
    expect(op.execute({}).length).toBe(0);
  });
});

describe("LeftOuterJoinOperator", () => {
  it("preserves all left entries", () => {
    const op = new LeftOuterJoinOperator(leftData, rightData, cond);
    const result = op.execute({});
    // Alice+eng, Charlie+eng, Bob (no match) = 3
    expect(result.length).toBe(3);
  });

  it("unmatched left entries have single-id tuple", () => {
    const op = new LeftOuterJoinOperator(leftData, rightData, cond);
    const result = op.execute({});
    const singleId = result.entries.find((e) => e.docIds.length === 1);
    expect(singleId).toBeDefined();
    expect(singleId!.docIds[0]).toBe(2); // Bob
  });
});

describe("RightOuterJoinOperator", () => {
  it("preserves all right entries", () => {
    const op = new RightOuterJoinOperator(leftData, rightData, cond);
    const result = op.execute({});
    // eng matches Alice+Charlie, hr unmatched = 3
    expect(result.length).toBe(3);
  });
});

describe("FullOuterJoinOperator", () => {
  it("preserves all entries from both sides", () => {
    const op = new FullOuterJoinOperator(leftData, rightData, cond);
    const result = op.execute({});
    // Alice+eng, Charlie+eng, Bob(unmatched), hr(unmatched) = 4
    expect(result.length).toBe(4);
  });
});

describe("SemiJoinOperator", () => {
  it("keeps left entries with matching right", () => {
    const leftPl = new PostingList(leftData);
    const rightPl = new PostingList([leftData[0]!, leftData[2]!]); // ids 1, 3
    const leftOp = {
      execute: () => leftPl,
      compose: () => null,
      costEstimate: () => 0,
    };
    const rightOp = {
      execute: () => rightPl,
      compose: () => null,
      costEstimate: () => 0,
    };
    const op = new SemiJoinOperator(
      leftOp as unknown as Operator,
      rightOp as unknown as Operator,
    );
    const result = op.execute({});
    expect(result.length).toBe(2);
    expect([...result.docIds]).toContain(1);
    expect([...result.docIds]).toContain(3);
  });
});

describe("AntiJoinOperator", () => {
  it("keeps left entries without matching right", () => {
    const leftPl = new PostingList(leftData);
    const rightPl = new PostingList([leftData[0]!]); // id 1
    const leftOp = {
      execute: () => leftPl,
      compose: () => null,
      costEstimate: () => 0,
    };
    const rightOp = {
      execute: () => rightPl,
      compose: () => null,
      costEstimate: () => 0,
    };
    const op = new AntiJoinOperator(
      leftOp as unknown as Operator,
      rightOp as unknown as Operator,
    );
    const result = op.execute({});
    expect(result.length).toBe(2);
    expect([...result.docIds]).not.toContain(1);
  });
});

describe("SortMergeJoinOperator", () => {
  it("joins via sort-merge", () => {
    const op = new SortMergeJoinOperator(leftData, rightData, cond);
    const result = op.execute({});
    expect(result.length).toBe(2);
  });
});

describe("IndexJoinOperator", () => {
  it("joins via binary search", () => {
    const op = new IndexJoinOperator(leftData, rightData, cond);
    const result = op.execute({});
    expect(result.length).toBe(2);
  });
});

describe("CrossJoinOperator", () => {
  it("produces Cartesian product", () => {
    const op = new CrossJoinOperator(leftData, rightData);
    const result = op.execute({});
    expect(result.length).toBe(leftData.length * rightData.length); // 3 * 2 = 6
  });
});

describe("TextSimilarityJoinOperator", () => {
  it("joins on text similarity", () => {
    const left = makeEntries([
      { id: 1, fields: { text: "machine learning algorithms" } },
      { id: 2, fields: { text: "web development" } },
    ]);
    const right = makeEntries([
      { id: 10, fields: { text: "learning machine algorithms" } },
      { id: 20, fields: { text: "cooking recipes" } },
    ]);
    const op = new TextSimilarityJoinOperator(left, right, "text", "text", 0.3);
    const result = op.execute({});
    // doc 1 and 10 have high Jaccard overlap
    expect(result.length).toBeGreaterThanOrEqual(1);
  });
});
