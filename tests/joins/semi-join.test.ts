import { describe, expect, it } from "vitest";
import { PostingList } from "../../src/core/posting-list.js";
import { IndexStats, createPostingEntry, createPayload } from "../../src/core/types.js";
import type { PostingEntry } from "../../src/core/types.js";
import { Operator } from "../../src/operators/base.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import { SemiJoinOperator, AntiJoinOperator } from "../../src/joins/semi.js";

class ConstantOperator extends Operator {
  private _pl: PostingList;
  constructor(pl: PostingList) {
    super();
    this._pl = pl;
  }
  execute(_context: ExecutionContext): PostingList {
    return this._pl;
  }
}

function makePl(...docIds: number[]): PostingList {
  const entries = docIds.map((d) => createPostingEntry(d, { score: d }));
  return PostingList.fromSorted(entries);
}

const context: ExecutionContext = {};

describe("SemiJoin", () => {
  it("basic semi join", () => {
    const left = new ConstantOperator(makePl(1, 2, 3));
    const right = new ConstantOperator(makePl(2, 3, 4));
    const result = new SemiJoinOperator(left, right).execute(context);
    expect(result.entries.map((e) => e.docId)).toEqual([2, 3]);
  });

  it("basic anti join", () => {
    const left = new ConstantOperator(makePl(1, 2, 3));
    const right = new ConstantOperator(makePl(2, 3, 4));
    const result = new AntiJoinOperator(left, right).execute(context);
    expect(result.entries.map((e) => e.docId)).toEqual([1]);
  });

  it("semi join with custom condition", () => {
    const leftPl = PostingList.fromSorted([
      createPostingEntry(1, { fields: { dept: "eng" } }),
      createPostingEntry(2, { fields: { dept: "sales" } }),
      createPostingEntry(3, { fields: { dept: "hr" } }),
    ]);
    const rightPl = PostingList.fromSorted([
      createPostingEntry(10, { fields: { department: "eng" } }),
      createPostingEntry(11, { fields: { department: "sales" } }),
    ]);

    const deptMatch = (l: PostingEntry, r: PostingEntry): boolean =>
      l.payload.fields["dept"] === r.payload.fields["department"];

    const result = new SemiJoinOperator(
      new ConstantOperator(leftPl),
      new ConstantOperator(rightPl),
      deptMatch,
    ).execute(context);

    expect(result.entries.map((e) => e.docId)).toEqual([1, 2]);
  });

  it("anti join with custom condition", () => {
    const leftPl = PostingList.fromSorted([
      createPostingEntry(1, { fields: { dept: "eng" } }),
      createPostingEntry(2, { fields: { dept: "sales" } }),
      createPostingEntry(3, { fields: { dept: "hr" } }),
    ]);
    const rightPl = PostingList.fromSorted([
      createPostingEntry(10, { fields: { department: "eng" } }),
      createPostingEntry(11, { fields: { department: "sales" } }),
    ]);

    const deptMatch = (l: PostingEntry, r: PostingEntry): boolean =>
      l.payload.fields["dept"] === r.payload.fields["department"];

    const result = new AntiJoinOperator(
      new ConstantOperator(leftPl),
      new ConstantOperator(rightPl),
      deptMatch,
    ).execute(context);

    expect(result.entries.map((e) => e.docId)).toEqual([3]);
  });

  it("empty left", () => {
    const left = new ConstantOperator(PostingList.fromSorted([]));
    const right = new ConstantOperator(makePl(1, 2));

    const semiResult = new SemiJoinOperator(left, right).execute(context);
    const antiResult = new AntiJoinOperator(left, right).execute(context);
    expect(semiResult.entries.length).toBe(0);
    expect(antiResult.entries.length).toBe(0);
  });

  it("empty right", () => {
    const left = new ConstantOperator(makePl(1, 2, 3));
    const right = new ConstantOperator(PostingList.fromSorted([]));

    const semiResult = new SemiJoinOperator(left, right).execute(context);
    const antiResult = new AntiJoinOperator(left, right).execute(context);
    expect(semiResult.entries.length).toBe(0);
    expect(antiResult.entries.map((e) => e.docId)).toEqual([1, 2, 3]);
  });

  it("no overlap", () => {
    const left = new ConstantOperator(makePl(1, 2, 3));
    const right = new ConstantOperator(makePl(4, 5, 6));

    const semiResult = new SemiJoinOperator(left, right).execute(context);
    const antiResult = new AntiJoinOperator(left, right).execute(context);
    expect(semiResult.entries.length).toBe(0);
    expect(antiResult.entries.map((e) => e.docId)).toEqual([1, 2, 3]);
  });

  it("not commutative", () => {
    // Demonstrate identity difference via payloads
    const leftPl = PostingList.fromSorted([
      createPostingEntry(1, { score: 0.1, fields: { src: "left" } }),
    ]);
    const rightPl = PostingList.fromSorted([
      createPostingEntry(1, { score: 0.9, fields: { src: "right" } }),
    ]);
    const p = new ConstantOperator(leftPl);
    const q = new ConstantOperator(rightPl);

    const resultPQ = new SemiJoinOperator(p, q).execute(context);
    const resultQP = new SemiJoinOperator(q, p).execute(context);

    expect(resultPQ.entries[0]!.payload.fields["src"]).toBe("left");
    expect(resultQP.entries[0]!.payload.fields["src"]).toBe("right");
    expect(resultPQ.entries[0]!.payload.score).not.toBe(
      resultQP.entries[0]!.payload.score,
    );
  });

  it("payload preservation", () => {
    const leftPl = PostingList.fromSorted([
      createPostingEntry(1, {
        positions: [10, 20],
        score: 0.95,
        fields: { name: "Alice", role: "admin" },
      }),
      createPostingEntry(2, {
        positions: [30],
        score: 0.5,
        fields: { name: "Bob", role: "user" },
      }),
      createPostingEntry(3, {
        positions: [],
        score: 0.1,
        fields: { name: "Charlie", role: "guest" },
      }),
    ]);
    const rightPl = PostingList.fromSorted([
      createPostingEntry(2, { score: 999.0, fields: { unrelated: true } }),
      createPostingEntry(3, { score: 888.0, fields: { other: "data" } }),
    ]);

    const result = new SemiJoinOperator(
      new ConstantOperator(leftPl),
      new ConstantOperator(rightPl),
    ).execute(context);

    expect(result.entries.length).toBe(2);
    const entry2 = result.entries[0]!;
    expect(entry2.docId).toBe(2);
    expect(entry2.payload.score).toBe(0.5);
    expect(entry2.payload.positions).toEqual([30]);
    expect(entry2.payload.fields).toEqual({ name: "Bob", role: "user" });

    const entry3 = result.entries[1]!;
    expect(entry3.docId).toBe(3);
    expect(entry3.payload.score).toBe(0.1);
    expect(entry3.payload.positions).toEqual([]);
    expect(entry3.payload.fields).toEqual({ name: "Charlie", role: "guest" });
  });

  it("cost estimate", () => {
    const left = new ConstantOperator(makePl(1, 2));
    const right = new ConstantOperator(makePl(3, 4));
    const stats = new IndexStats(100);
    const op = new SemiJoinOperator(left, right);
    expect(op.costEstimate(stats)).toBe(200.0);
    const antiOp = new AntiJoinOperator(left, right);
    expect(antiOp.costEstimate(stats)).toBe(200.0);
  });

  it("semi and anti are complementary", () => {
    const left = new ConstantOperator(makePl(1, 2, 3, 4, 5));
    const right = new ConstantOperator(makePl(2, 4));

    const semiResult = new SemiJoinOperator(left, right).execute(context);
    const antiResult = new AntiJoinOperator(left, right).execute(context);

    const semiIds = new Set(semiResult.entries.map((e) => e.docId));
    const antiIds = new Set(antiResult.entries.map((e) => e.docId));

    expect(semiIds).toEqual(new Set([2, 4]));
    expect(antiIds).toEqual(new Set([1, 3, 5]));
    expect(new Set([...semiIds, ...antiIds])).toEqual(new Set([1, 2, 3, 4, 5]));
    // Intersection should be empty
    const intersection = new Set([...semiIds].filter((x) => antiIds.has(x)));
    expect(intersection.size).toBe(0);
  });

  it("full overlap", () => {
    const left = new ConstantOperator(makePl(1, 2, 3));
    const right = new ConstantOperator(makePl(1, 2, 3, 4, 5));

    const semiResult = new SemiJoinOperator(left, right).execute(context);
    const antiResult = new AntiJoinOperator(left, right).execute(context);

    expect(semiResult.entries.map((e) => e.docId)).toEqual([1, 2, 3]);
    expect(antiResult.entries.length).toBe(0);
  });
});
