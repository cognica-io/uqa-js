import { describe, expect, it } from "vitest";
import { createVertex, createEdge } from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import { VertexPropertyIndex, EdgePropertyIndex } from "../../src/graph/index.js";

function makeStore(): MemoryGraphStore {
  const gs = new MemoryGraphStore();
  gs.createGraph("g");
  gs.addVertex(createVertex(1, "person", { name: "Alice", age: 30 }), "g");
  gs.addVertex(createVertex(2, "person", { name: "Bob", age: 25 }), "g");
  gs.addVertex(createVertex(3, "person", { name: "Carol", age: 35 }), "g");
  gs.addVertex(createVertex(4, "company", { name: "Acme", size: 100 }), "g");
  gs.addEdge(createEdge(10, 1, 2, "knows", { since: 2020, weight: 0.8 }), "g");
  gs.addEdge(createEdge(11, 2, 3, "knows", { since: 2021, weight: 0.6 }), "g");
  gs.addEdge(createEdge(12, 1, 4, "works_at", { since: 2019, weight: 1.0 }), "g");
  return gs;
}

function buildVertexIdx(
  gs: MemoryGraphStore,
  graph: string,
  props: string[],
): VertexPropertyIndex {
  const idx = new VertexPropertyIndex();
  idx.build(gs, graph, props);
  return idx;
}

function buildEdgeIdx(
  gs: MemoryGraphStore,
  graph: string,
  props: string[],
): EdgePropertyIndex {
  const idx = new EdgePropertyIndex();
  idx.build(gs, graph, props);
  return idx;
}

// -- VertexPropertyIndex tests --

describe("VertexPropertyIndex", () => {
  it("build", () => {
    const gs = makeStore();
    const idx = buildVertexIdx(gs, "g", ["name", "age"]);
    expect(idx.hasProperty("name")).toBe(true);
    expect(idx.hasProperty("age")).toBe(true);
    expect(idx.hasProperty("nonexistent")).toBe(false);
  });

  it("eq lookup", () => {
    const gs = makeStore();
    const idx = buildVertexIdx(gs, "g", ["name"]);
    expect(idx.lookupEq("name", "Alice")).toEqual(new Set([1]));
    expect(idx.lookupEq("name", "Bob")).toEqual(new Set([2]));
    expect(idx.lookupEq("name", "Nonexistent")).toEqual(new Set());
  });

  it("range lookup", () => {
    const gs = makeStore();
    const idx = buildVertexIdx(gs, "g", ["age"]);
    const result = idx.lookupRange("age", 25, 30);
    expect(result.sort()).toEqual([1, 2]);
  });

  it("range narrow", () => {
    const gs = makeStore();
    const idx = buildVertexIdx(gs, "g", ["age"]);
    const result = idx.lookupRange("age", 31, 40);
    expect(result).toEqual([3]);
  });

  it("range empty", () => {
    const gs = makeStore();
    const idx = buildVertexIdx(gs, "g", ["age"]);
    expect(idx.lookupRange("age", 100, 200)).toEqual([]);
  });

  it("multiple properties", () => {
    const gs = makeStore();
    const idx = buildVertexIdx(gs, "g", ["name", "age"]);
    expect(idx.lookupEq("name", "Carol")).toEqual(new Set([3]));
    const result = idx.lookupRange("age", 30, 35);
    expect(result.sort()).toEqual([1, 3]);
  });

  it("missing values", () => {
    const gs = makeStore();
    const idx = buildVertexIdx(gs, "g", ["size"]);
    expect(idx.lookupEq("size", 100)).toEqual(new Set([4]));
    expect(idx.lookupEq("size", 200)).toEqual(new Set());
  });
});

// -- EdgePropertyIndex tests --

describe("EdgePropertyIndex", () => {
  it("build", () => {
    const gs = makeStore();
    const idx = buildEdgeIdx(gs, "g", ["since", "weight"]);
    expect(idx.hasProperty("since")).toBe(true);
    expect(idx.hasProperty("weight")).toBe(true);
  });

  it("eq lookup", () => {
    const gs = makeStore();
    const idx = buildEdgeIdx(gs, "g", ["since"]);
    expect(idx.lookupEq("since", 2020)).toEqual(new Set([10]));
    expect(idx.lookupEq("since", 2021)).toEqual(new Set([11]));
    expect(idx.lookupEq("since", 9999)).toEqual(new Set());
  });

  it("range lookup", () => {
    const gs = makeStore();
    const idx = buildEdgeIdx(gs, "g", ["weight"]);
    const result = idx.lookupRange("weight", 0.7, 1.0);
    expect(result.sort()).toEqual([10, 12]);
  });

  it("range all", () => {
    const gs = makeStore();
    const idx = buildEdgeIdx(gs, "g", ["since"]);
    const result = idx.lookupRange("since", 2019, 2021);
    expect(result.sort()).toEqual([10, 11, 12]);
  });

  it("empty graph", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("empty");
    const idx = buildEdgeIdx(gs, "empty", ["weight"]);
    expect(idx.lookupEq("weight", 1.0)).toEqual(new Set());
    expect(idx.lookupRange("weight", 0, 10)).toEqual([]);
  });
});

// -- Graph isolation --

describe("PropertyIndexGraphScope", () => {
  it("respects graph scope", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g1");
    gs.createGraph("g2");
    gs.addVertex(createVertex(1, "a", { val: 10 }), "g1");
    gs.addVertex(createVertex(2, "b", { val: 20 }), "g2");

    const idx1 = buildVertexIdx(gs, "g1", ["val"]);
    const idx2 = buildVertexIdx(gs, "g2", ["val"]);

    expect(idx1.lookupEq("val", 10)).toEqual(new Set([1]));
    expect(idx1.lookupEq("val", 20)).toEqual(new Set());
    expect(idx2.lookupEq("val", 20)).toEqual(new Set([2]));
    expect(idx2.lookupEq("val", 10)).toEqual(new Set());
  });
});
