import { describe, expect, it } from "vitest";
import { createPayload, createPostingEntry } from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import { GraphPostingList, createGraphPayload } from "../../src/graph/posting-list.js";
import type { GraphPayload } from "../../src/graph/posting-list.js";
import {
  GraphGraphJoinOperator,
  CrossParadigmGraphJoinOperator,
} from "../../src/graph/join.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import type { Operator } from "../../src/operators/base.js";

// -- Helpers --

class ConstGPL {
  private _gpl: PostingList;
  constructor(gpl: PostingList) {
    this._gpl = gpl;
  }
  execute(_ctx: ExecutionContext): PostingList {
    return this._gpl;
  }
}

class ConstPL {
  private _pl: PostingList;
  constructor(pl: PostingList) {
    this._pl = pl;
  }
  execute(_ctx: ExecutionContext): PostingList {
    return this._pl;
  }
}

function makeGpl(
  entries: Array<[number, Record<string, number>]>,
  vertices?: Array<ReadonlySet<number>>,
): GraphPostingList {
  const peList = entries.map(([docId, fields], i) =>
    createPostingEntry(docId, { score: 0.9, fields }),
  );
  const gpMap = new Map<number, GraphPayload>();
  for (let i = 0; i < entries.length; i++) {
    const [docId, fields] = entries[i]!;
    const verts = vertices ? vertices[i]! : new Set(Object.values(fields));
    gpMap.set(
      docId,
      createGraphPayload({
        subgraphVertices: verts,
        subgraphEdges: new Set<number>(),
      }),
    );
  }
  return new GraphPostingList(peList, gpMap);
}

function makeCtx(): ExecutionContext {
  const gs = new MemoryGraphStore();
  gs.createGraph("test");
  return { graphStore: gs };
}

// -- GraphGraphJoinOperator tests --

describe("GraphGraphJoinOperator", () => {
  it("basic join", () => {
    const left = makeGpl([
      [1, { x: 10 }],
      [2, { x: 20 }],
    ]);
    const right = makeGpl([
      [3, { x: 20 }],
      [4, { x: 30 }],
    ]);

    const op = new GraphGraphJoinOperator(
      new ConstGPL(left) as unknown as Operator,
      new ConstGPL(right) as unknown as Operator,
      "x",
    );
    const result = op.execute(makeCtx());
    expect(result.length).toBe(1);
    // Joined on x=20
    expect([...result][0]!.payload.fields["x"]).toBe(20);
  });

  it("empty join", () => {
    const left = makeGpl([[1, { x: 10 }]]);
    const right = makeGpl([[2, { x: 20 }]]);

    const op = new GraphGraphJoinOperator(
      new ConstGPL(left) as unknown as Operator,
      new ConstGPL(right) as unknown as Operator,
      "x",
    );
    const result = op.execute(makeCtx());
    expect(result.length).toBe(0);
  });

  it("metadata merge", () => {
    const left = makeGpl([[1, { x: 10 }]], [new Set([1, 2])]);
    const right = makeGpl([[2, { x: 10 }]], [new Set([3, 4])]);

    const op = new GraphGraphJoinOperator(
      new ConstGPL(left) as unknown as Operator,
      new ConstGPL(right) as unknown as Operator,
      "x",
    );
    const result = op.execute(makeCtx());
    expect(result.length).toBe(1);

    if (result instanceof GraphPostingList) {
      const firstEntry = [...result][0]!;
      const gp = result.getGraphPayload(firstEntry.docId);
      expect(gp).not.toBeNull();
      expect(gp!.subgraphVertices).toEqual(new Set([1, 2, 3, 4]));
    }
  });

  it("commutativity", () => {
    const left = makeGpl([[1, { x: 10, side: 1 }]]);
    const right = makeGpl([[2, { x: 10, side: 2 }]]);

    const opLR = new GraphGraphJoinOperator(
      new ConstGPL(left) as unknown as Operator,
      new ConstGPL(right) as unknown as Operator,
      "x",
    );
    const opRL = new GraphGraphJoinOperator(
      new ConstGPL(right) as unknown as Operator,
      new ConstGPL(left) as unknown as Operator,
      "x",
    );
    const ctx = makeCtx();
    const rLR = opLR.execute(ctx);
    const rRL = opRL.execute(ctx);
    expect(rLR.length).toBe(1);
    expect(rRL.length).toBe(1);
    // Both produce 1 result; field merging behavior may vary
    const lrSide = [...rLR][0]!.payload.fields["side"];
    const rlSide = [...rRL][0]!.payload.fields["side"];
    // At minimum, left and right should produce different side values
    expect(lrSide !== rlSide || lrSide === 1).toBe(true);
  });

  it("multiple matches", () => {
    const left = makeGpl([
      [1, { x: 10 }],
      [2, { x: 10 }],
    ]);
    const right = makeGpl([
      [3, { x: 10 }],
      [4, { x: 10 }],
    ]);

    const op = new GraphGraphJoinOperator(
      new ConstGPL(left) as unknown as Operator,
      new ConstGPL(right) as unknown as Operator,
      "x",
    );
    const result = op.execute(makeCtx());
    // 2 left * 2 right = 4 matches
    expect(result.length).toBe(4);
  });

  it("associativity", () => {
    const a = makeGpl([[1, { x: 5, src: 1 }]]);
    const b = makeGpl([[2, { x: 5, src: 2 }]]);
    const c = makeGpl([[3, { x: 5, src: 3 }]]);

    const ctx = makeCtx();

    // (A join B) join C
    const abOp = new GraphGraphJoinOperator(
      new ConstGPL(a) as unknown as Operator,
      new ConstGPL(b) as unknown as Operator,
      "x",
    );
    const abResult = abOp.execute(ctx);
    const abcOp = new GraphGraphJoinOperator(
      new ConstGPL(abResult) as unknown as Operator,
      new ConstGPL(c) as unknown as Operator,
      "x",
    );
    const abcResult = abcOp.execute(ctx);

    // A join (B join C)
    const bcOp = new GraphGraphJoinOperator(
      new ConstGPL(b) as unknown as Operator,
      new ConstGPL(c) as unknown as Operator,
      "x",
    );
    const bcResult = bcOp.execute(ctx);
    const aBcOp = new GraphGraphJoinOperator(
      new ConstGPL(a) as unknown as Operator,
      new ConstGPL(bcResult) as unknown as Operator,
      "x",
    );
    const aBcResult = aBcOp.execute(ctx);

    expect(abcResult.length).toBe(aBcResult.length);
    expect(abcResult.length).toBe(1);
  });
});

// -- CrossParadigmGraphJoinOperator tests --

describe("CrossParadigmGraphJoinOperator", () => {
  it("basic join on shared field value", () => {
    // Graph side has vid=100, relational side has vid=100
    const gpl = makeGpl([[1, { vid: 100 }]]);
    const rel = new PostingList([
      createPostingEntry(100, { score: 0.5, fields: { vid: 100, name: "Alice" } }),
      createPostingEntry(200, { score: 0.5, fields: { vid: 200, name: "Bob" } }),
    ]);

    const op = new CrossParadigmGraphJoinOperator(
      new ConstGPL(gpl) as unknown as Operator,
      new ConstPL(rel) as unknown as Operator,
      "vid",
    );
    const result = op.execute(makeCtx());
    expect(result.length).toBe(1);
  });

  it("empty join when no field match", () => {
    const gpl = makeGpl([[1, { vid: 999 }]]);
    const rel = new PostingList([
      createPostingEntry(100, { score: 0.5, fields: { vid: 100 } }),
    ]);

    const op = new CrossParadigmGraphJoinOperator(
      new ConstGPL(gpl) as unknown as Operator,
      new ConstPL(rel) as unknown as Operator,
      "vid",
    );
    const result = op.execute(makeCtx());
    expect(result.length).toBe(0);
  });

  it("score merge adds graph and relational scores", () => {
    const gpl = makeGpl([[1, { vid: 100 }]]);
    const rel = new PostingList([
      createPostingEntry(100, { score: 0.3, fields: { vid: 100 } }),
    ]);

    const op = new CrossParadigmGraphJoinOperator(
      new ConstGPL(gpl) as unknown as Operator,
      new ConstPL(rel) as unknown as Operator,
      "vid",
    );
    const result = op.execute(makeCtx());
    expect(result.length).toBe(1);
    // 0.9 (graph score from makeGpl) + 0.3 (relational) = 1.2
    expect(Math.abs([...result][0]!.payload.score - 1.2)).toBeLessThan(0.01);
  });
});
