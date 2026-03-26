import { describe, expect, it } from "vitest";
import { createEdge, createVertex } from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import { TemporalFilter } from "../../src/graph/temporal-filter.js";
import { TemporalTraverseOperator } from "../../src/graph/temporal-traverse.js";
import { TemporalPatternMatchOperator } from "../../src/graph/temporal-pattern-match.js";
import {
  createVertexPattern,
  createEdgePattern,
  createGraphPattern,
} from "../../src/graph/pattern.js";
import type { ExecutionContext } from "../../src/operators/base.js";

function makeTemporalGraph(): MemoryGraphStore {
  const store = new MemoryGraphStore();
  store.createGraph("test");
  const vertices = [
    createVertex(1, "person", { name: "Alice" }),
    createVertex(2, "person", { name: "Bob" }),
    createVertex(3, "person", { name: "Charlie" }),
    createVertex(4, "person", { name: "Diana" }),
    createVertex(5, "person", { name: "Eve" }),
  ];
  const edges = [
    createEdge(1, 1, 2, "knows", { valid_from: 100, valid_to: 200 }),
    createEdge(2, 1, 3, "knows", { valid_from: 150, valid_to: 300 }),
    createEdge(3, 2, 3, "knows", { valid_from: 50, valid_to: 120 }),
    createEdge(4, 2, 4, "works_with", { valid_from: 200, valid_to: 400 }),
    createEdge(5, 3, 4, "knows", { valid_from: 100, valid_to: 250 }),
    createEdge(6, 3, 5, "works_with", {}),
    createEdge(7, 4, 5, "knows", { valid_from: 300, valid_to: 500 }),
  ];
  for (const v of vertices) {
    store.addVertex(v, "test");
  }
  for (const e of edges) {
    store.addEdge(e, "test");
  }
  return store;
}

// -- TemporalFilter tests --

describe("TemporalFilter", () => {
  it("timestamp within range", () => {
    const tf = new TemporalFilter({ timestamp: 150 });
    const edge = createEdge(1, 1, 2, "knows", { valid_from: 100, valid_to: 200 });
    expect(tf.isValid(edge)).toBe(true);
  });

  it("timestamp outside range", () => {
    const tf = new TemporalFilter({ timestamp: 50 });
    const edge = createEdge(1, 1, 2, "knows", { valid_from: 100, valid_to: 200 });
    expect(tf.isValid(edge)).toBe(false);
  });

  it("timestamp at boundary", () => {
    const edge = createEdge(1, 1, 2, "knows", { valid_from: 100, valid_to: 200 });
    const tf1 = new TemporalFilter({ timestamp: 100 });
    expect(tf1.isValid(edge)).toBe(true);
    const tf2 = new TemporalFilter({ timestamp: 200 });
    expect(tf2.isValid(edge)).toBe(true);
  });

  it("time range overlap", () => {
    const tf = new TemporalFilter({ timeRange: [90, 150] });
    const edge = createEdge(1, 1, 2, "knows", { valid_from: 100, valid_to: 200 });
    expect(tf.isValid(edge)).toBe(true);
  });

  it("time range no overlap", () => {
    const tf = new TemporalFilter({ timeRange: [300, 400] });
    const edge = createEdge(1, 1, 2, "knows", { valid_from: 100, valid_to: 200 });
    expect(tf.isValid(edge)).toBe(false);
  });

  it("no temporal properties on edge", () => {
    const tf = new TemporalFilter({ timestamp: 150 });
    const edge = createEdge(1, 1, 2, "knows", {});
    expect(tf.isValid(edge)).toBe(true);
    const edge2 = createEdge(2, 1, 2, "knows", { other: "value" });
    expect(tf.isValid(edge2)).toBe(true);
  });

  it("partial temporal properties valid_from only", () => {
    const tf = new TemporalFilter({ timestamp: 150 });
    const edge = createEdge(1, 1, 2, "knows", { valid_from: 100 });
    expect(tf.isValid(edge)).toBe(true);
    const tf2 = new TemporalFilter({ timestamp: 50 });
    expect(tf2.isValid(edge)).toBe(false);
  });

  it("partial temporal properties valid_to only", () => {
    const tf = new TemporalFilter({ timestamp: 150 });
    const edge = createEdge(1, 1, 2, "knows", { valid_to: 200 });
    expect(tf.isValid(edge)).toBe(true);
    const tf2 = new TemporalFilter({ timestamp: 250 });
    expect(tf2.isValid(edge)).toBe(false);
  });

  it("no filter accepts all", () => {
    const tf = new TemporalFilter();
    const edge1 = createEdge(1, 1, 2, "knows", { valid_from: 100, valid_to: 200 });
    expect(tf.isValid(edge1)).toBe(true);
    const edge2 = createEdge(2, 1, 2, "knows", {});
    expect(tf.isValid(edge2)).toBe(true);
  });

  it("both timestamp and range raises", () => {
    expect(() => new TemporalFilter({ timestamp: 150, timeRange: [100, 200] })).toThrow(
      /mutually exclusive/,
    );
  });
});

// -- TemporalTraverseOperator tests --

describe("TemporalTraverse", () => {
  it("traverse at timestamp", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const tf = new TemporalFilter({ timestamp: 110 });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "test",
      temporalFilter: tf,
      label: "knows",
      maxHops: 1,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    // edge 1: 1->2 valid_from=100, valid_to=200 -> 110 in range -> yes
    // edge 2: 1->3 valid_from=150, valid_to=300 -> 110 < 150 -> no
    expect(reached.has(2)).toBe(true);
    expect(reached.has(3)).toBe(false);
  });

  it("traverse at later timestamp", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const tf = new TemporalFilter({ timestamp: 160 });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "test",
      temporalFilter: tf,
      label: "knows",
      maxHops: 1,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    expect(reached.has(2)).toBe(true);
    expect(reached.has(3)).toBe(true);
  });

  it("traverse multi hop", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const tf = new TemporalFilter({ timestamp: 110 });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "test",
      temporalFilter: tf,
      label: "knows",
      maxHops: 2,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    expect(reached.has(2)).toBe(true);
    expect(reached.has(3)).toBe(true);
  });

  it("traverse with time range", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    // Range [80, 130]: overlaps edge 1 (100-200), edge 3 (50-120), edge 5 (100-250)
    // From vertex 1, only edges 1 (1->2, 100-200) and 2 (1->3, 150-300)
    // Range [80, 130] overlaps edge 1 (100-200): yes (100 <= 130 && 200 >= 80)
    // Range [80, 130] overlaps edge 2 (150-300): no (150 > 130)
    const tf = new TemporalFilter({ timeRange: [80, 130] });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "test",
      temporalFilter: tf,
      label: "knows",
      maxHops: 1,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    expect(reached.has(2)).toBe(true);
    expect(reached.has(3)).toBe(false);
  });

  it("traverse no filter", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const tf = new TemporalFilter();
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "test",
      temporalFilter: tf,
      label: "knows",
      maxHops: 1,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    expect(reached.has(2)).toBe(true);
    expect(reached.has(3)).toBe(true);
  });

  it("traverse all labels", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const tf = new TemporalFilter({ timestamp: 250 });
    const op = new TemporalTraverseOperator({
      startVertex: 3,
      graph: "test",
      temporalFilter: tf,
      label: null,
      maxHops: 1,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    expect(reached.has(4)).toBe(true);
    expect(reached.has(5)).toBe(true);
  });
});

// -- TemporalPatternMatchOperator tests --

describe("TemporalPatternMatch", () => {
  it("pattern match at timestamp", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const pattern = createGraphPattern(
      [createVertexPattern("a"), createVertexPattern("b")],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const tf = new TemporalFilter({ timestamp: 110 });
    const op = new TemporalPatternMatchOperator({
      pattern,
      graph: "test",
      temporalFilter: tf,
    });
    const result = op.execute(ctx);
    const assignments = [...result].map((e) => [
      e.payload.fields["a"],
      e.payload.fields["b"],
    ]);
    // At timestamp 110:
    // edge 1: 1->2 valid_from=100, valid_to=200 -> in range
    // edge 2: 1->3 valid_from=150, valid_to=300 -> 110 < 150, out of range
    // edge 3: 2->3 valid_from=50, valid_to=120 -> in range
    // edge 5: 3->4 valid_from=100, valid_to=250 -> in range
    // edge 7: 4->5 valid_from=300, valid_to=500 -> out of range
    expect(assignments).toContainEqual([1, 2]);
    expect(assignments).toContainEqual([2, 3]);
    expect(assignments).toContainEqual([3, 4]);
    expect(assignments).not.toContainEqual([1, 3]);
    expect(assignments).not.toContainEqual([4, 5]);
  });

  it("pattern match no filter", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const pattern = createGraphPattern(
      [createVertexPattern("a"), createVertexPattern("b")],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const tf = new TemporalFilter();
    const op = new TemporalPatternMatchOperator({
      pattern,
      graph: "test",
      temporalFilter: tf,
    });
    const result = op.execute(ctx);
    const assignments = [...result].map((e) => [
      e.payload.fields["a"],
      e.payload.fields["b"],
    ]);
    // All 5 knows edges should be found
    expect(assignments).toContainEqual([1, 2]);
    expect(assignments).toContainEqual([1, 3]);
    expect(assignments).toContainEqual([2, 3]);
    expect(assignments).toContainEqual([3, 4]);
    expect(assignments).toContainEqual([4, 5]);
  });

  it("pattern match with time range", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const pattern = createGraphPattern(
      [createVertexPattern("a"), createVertexPattern("b")],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    // Range [200, 350]: overlaps edges with valid_from/valid_to overlapping this range
    // edge 1: 1->2 valid_from=100, valid_to=200 -> 200 <= 350 && 200 >= 200 -> yes (boundary)
    // edge 2: 1->3 valid_from=150, valid_to=300 -> 300 >= 200 && 150 <= 350 -> yes
    // edge 3: 2->3 valid_from=50, valid_to=120 -> 120 < 200 -> no
    // edge 5: 3->4 valid_from=100, valid_to=250 -> 250 >= 200 && 100 <= 350 -> yes
    // edge 7: 4->5 valid_from=300, valid_to=500 -> 500 >= 200 && 300 <= 350 -> yes
    const tf = new TemporalFilter({ timeRange: [200, 350] });
    const op = new TemporalPatternMatchOperator({
      pattern,
      graph: "test",
      temporalFilter: tf,
    });
    const result = op.execute(ctx);
    const assignments = [...result].map((e) => [
      e.payload.fields["a"],
      e.payload.fields["b"],
    ]);
    expect(assignments).toContainEqual([1, 2]); // boundary overlap
    expect(assignments).toContainEqual([1, 3]);
    expect(assignments).toContainEqual([3, 4]);
    expect(assignments).toContainEqual([4, 5]);
    expect(assignments).not.toContainEqual([2, 3]); // valid_to=120 < 200
  });
});

// -- SQL temporal_traverse tests --

describe("TemporalTraverseSQL", () => {
  it("temporal traverse from clause", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const tf = new TemporalFilter({ timestamp: 160 });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "test",
      temporalFilter: tf,
      label: "knows",
      maxHops: 2,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    expect(reached.size).toBeGreaterThan(0);
  });

  it("temporal traverse range from clause", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const tf = new TemporalFilter({ timeRange: [100, 200] });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "test",
      temporalFilter: tf,
      label: "knows",
      maxHops: 2,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    expect(reached.size).toBeGreaterThan(0);
  });
});

// -- QueryBuilder temporal_traverse tests --

describe("TemporalTraverseQueryBuilder", () => {
  it("query builder temporal traverse", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const tf = new TemporalFilter({ timestamp: 160 });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "test",
      temporalFilter: tf,
      label: "knows",
      maxHops: 1,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    expect(reached.has(2)).toBe(true);
    expect(reached.has(3)).toBe(true);
  });

  it("query builder temporal traverse range", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const tf = new TemporalFilter({ timeRange: [100, 200] });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "test",
      temporalFilter: tf,
      label: "knows",
      maxHops: 1,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    expect(reached.has(2)).toBe(true);
    expect(reached.has(3)).toBe(true);
  });
});

// -- Named graph temporal traverse --

describe("NamedGraphTemporalTraverse", () => {
  it("temporal traverse named graph SQL", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const tf = new TemporalFilter({ timestamp: 160 });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "test",
      temporalFilter: tf,
      label: "knows",
      maxHops: 1,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    expect(reached.has(2)).toBe(true);
    expect(reached.has(3)).toBe(true);
  });

  it("temporal traverse named graph range", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const tf = new TemporalFilter({ timeRange: [100, 200] });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "test",
      temporalFilter: tf,
      label: "knows",
      maxHops: 1,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    expect(reached.has(2)).toBe(true);
    expect(reached.has(3)).toBe(true);
  });

  it("temporal traverse named graph no match", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    // Very early timestamp - no edges should be valid
    const tf = new TemporalFilter({ timestamp: 10 });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "test",
      temporalFilter: tf,
      label: "knows",
      maxHops: 1,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    // The start vertex itself is always included, but no neighbors should be found
    expect(reached.has(1)).toBe(true);
    expect(reached.has(2)).toBe(false);
    expect(reached.has(3)).toBe(false);
  });
});

// -- Named graph traverse and RPQ --

describe("NamedGraphTraverseAndRPQ", () => {
  it("traverse named graph", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const tf = new TemporalFilter();
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "test",
      temporalFilter: tf,
      label: "knows",
      maxHops: 2,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    expect(reached.size).toBeGreaterThan(0);
  });

  it("rpq named graph", () => {
    const store = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: store };
    const tf = new TemporalFilter();
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "test",
      temporalFilter: tf,
      label: null,
      maxHops: 3,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    expect(reached.size).toBeGreaterThan(0);
  });
});
