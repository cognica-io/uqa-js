import { describe, expect, it } from "vitest";
import { createEdge, createVertex, IndexStats } from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import { TemporalFilter } from "../../src/graph/temporal-filter.js";
import { TemporalTraverseOperator } from "../../src/graph/temporal-traverse.js";
import { TemporalPatternMatchOperator } from "../../src/graph/temporal-pattern-match.js";
import {
  createGraphPattern,
  createVertexPattern,
  createEdgePattern,
} from "../../src/graph/pattern.js";

// -- Temporal graph fixture ---------------------------------------------------
// Uses the "timestamp" property on edges, as that is what the TS TemporalFilter checks.
// The Python version uses valid_from/valid_to ranges, but the TS TemporalFilter
// checks `edge.properties["timestamp"]`.

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

  // Edges with timestamps
  const edges = [
    createEdge(1, 1, 2, "knows", { timestamp: 100 }),
    createEdge(2, 1, 3, "knows", { timestamp: 150 }),
    createEdge(3, 2, 3, "knows", { timestamp: 50 }),
    createEdge(4, 2, 4, "works_with", { timestamp: 200 }),
    createEdge(5, 3, 4, "knows", { timestamp: 100 }),
    createEdge(6, 3, 5, "works_with"), // no timestamp -- always valid
    createEdge(7, 4, 5, "knows", { timestamp: 300 }),
  ];

  for (const v of vertices) store.addVertex(v, "test");
  for (const e of edges) store.addEdge(e, "test");

  return store;
}

interface ExecutionContext {
  graphStore: MemoryGraphStore;
}

// =============================================================================
// TemporalFilter tests
// =============================================================================

describe("TemporalFilter", () => {
  it("timestamp within range", () => {
    const tf = new TemporalFilter({ timestamp: 150 });
    const edge = createEdge(1, 1, 2, "e", { timestamp: 100 });
    // timestamp 100 <= 150 should be valid
    expect(tf.isValid(edge)).toBe(true);
  });

  it("timestamp at boundary", () => {
    const tf = new TemporalFilter({ timestamp: 100 });
    const edge = createEdge(1, 1, 2, "e", { timestamp: 100 });
    expect(tf.isValid(edge)).toBe(true);
  });

  it("timestamp outside range", () => {
    const tf = new TemporalFilter({ timestamp: 50 });
    const edge = createEdge(1, 1, 2, "e", { timestamp: 100 });
    // timestamp 100 > 50, so invalid
    expect(tf.isValid(edge)).toBe(false);
  });

  it("time range overlap", () => {
    const tf = new TemporalFilter({ timeRange: [90, 150] });
    const edge = createEdge(1, 1, 2, "e", { timestamp: 100 });
    // 100 is within [90, 150]
    expect(tf.isValid(edge)).toBe(true);
  });

  it("time range no overlap", () => {
    const tf = new TemporalFilter({ timeRange: [210, 300] });
    const edge = createEdge(1, 1, 2, "e", { timestamp: 100 });
    // 100 is not in [210, 300]
    expect(tf.isValid(edge)).toBe(false);
  });

  it("no temporal properties -- always valid", () => {
    const tf = new TemporalFilter({ timestamp: 150 });
    const edge = createEdge(1, 1, 2, "e");
    expect(tf.isValid(edge)).toBe(true);
    const edge2 = createEdge(2, 1, 2, "e", { other: "value" });
    expect(tf.isValid(edge2)).toBe(true);
  });

  it("no filter accepts all", () => {
    const tf = new TemporalFilter();
    expect(tf.isValid(createEdge(1, 1, 2, "e", { timestamp: 100 }))).toBe(true);
    expect(tf.isValid(createEdge(2, 1, 2, "e"))).toBe(true);
  });
});

// =============================================================================
// TemporalTraverseOperator tests
// =============================================================================

describe("TemporalTraverseOperator", () => {
  it("traverse at timestamp", () => {
    const gs = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: gs };
    // At timestamp=110, from vertex 1: edge 1 (timestamp=100, 100<=110 valid)
    // edge 2 (timestamp=150, 150>110 invalid)
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
    expect(reached.has(2)).toBe(true);
    expect(reached.has(3)).toBe(false);
  });

  it("traverse at later timestamp", () => {
    const gs = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: gs };
    // At timestamp=160, both edges from vertex 1 are valid
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
    const gs = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: gs };
    // At timestamp=110, from vertex 1 with 2 hops (knows):
    // Hop 1: 1->2 valid (ts=100<=110). 1->3 invalid (ts=150>110).
    // Hop 2: 2->3 valid (ts=50<=110).
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
    const gs = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: gs };
    // Range [90, 110]: from vertex 1 with "knows":
    // Edge 1 (ts=100): 90<=100<=110 -> valid
    // Edge 2 (ts=150): 150>110 -> invalid
    const tf = new TemporalFilter({ timeRange: [90, 110] });
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
    const gs = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: gs };
    // Without a temporal filter, all edges are traversable
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
    const gs = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: gs };
    // At timestamp=250, from vertex 3:
    // Edge 5 (3->4, knows, ts=100): valid (100<=250)
    // Edge 6 (3->5, works_with, no ts): always valid
    const tf = new TemporalFilter({ timestamp: 250 });
    const op = new TemporalTraverseOperator({
      startVertex: 3,
      graph: "test",
      temporalFilter: tf,
      maxHops: 1,
    });
    const result = op.execute(ctx);
    const reached = new Set([...result].map((e) => e.docId));
    expect(reached.has(4)).toBe(true);
    expect(reached.has(5)).toBe(true);
  });
});

// =============================================================================
// TemporalPatternMatchOperator tests
// =============================================================================

describe("TemporalPatternMatchOperator", () => {
  it("pattern match at timestamp", () => {
    const gs = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: gs };
    // Find (a)-[:knows]->(b) at timestamp=110.
    // Valid knows edges at ts<=110:
    //   1->2 (ts=100): yes
    //   1->3 (ts=150): no (150>110)
    //   2->3 (ts=50):  yes
    //   3->4 (ts=100): yes
    //   4->5 (ts=300): no (300>110)
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
    const assignments: Array<[number, number]> = [];
    for (const entry of result) {
      assignments.push([
        entry.payload.fields["a"] as number,
        entry.payload.fields["b"] as number,
      ]);
    }
    expect(assignments).toContainEqual([1, 2]);
    expect(assignments).toContainEqual([2, 3]);
    expect(assignments).toContainEqual([3, 4]);
    expect(assignments).not.toContainEqual([1, 3]);
    expect(assignments).not.toContainEqual([4, 5]);
  });

  it("pattern match no filter", () => {
    const gs = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: gs };
    // Without temporal filter, all knows edges participate.
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
    const assignments: Array<[number, number]> = [];
    for (const entry of result) {
      assignments.push([
        entry.payload.fields["a"] as number,
        entry.payload.fields["b"] as number,
      ]);
    }
    // All 5 knows edges should be found
    expect(assignments).toContainEqual([1, 2]);
    expect(assignments).toContainEqual([1, 3]);
    expect(assignments).toContainEqual([2, 3]);
    expect(assignments).toContainEqual([3, 4]);
    expect(assignments).toContainEqual([4, 5]);
  });

  it("pattern match with time range", () => {
    const gs = makeTemporalGraph();
    const ctx: ExecutionContext = { graphStore: gs };
    // Range [50, 150]: valid knows edges where timestamp is in [50, 150]:
    //   1->2 (ts=100): yes
    //   1->3 (ts=150): yes
    //   2->3 (ts=50):  yes
    //   3->4 (ts=100): yes
    //   4->5 (ts=300): no
    const pattern = createGraphPattern(
      [createVertexPattern("a"), createVertexPattern("b")],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const tf = new TemporalFilter({ timeRange: [50, 150] });
    const op = new TemporalPatternMatchOperator({
      pattern,
      graph: "test",
      temporalFilter: tf,
    });
    const result = op.execute(ctx);
    const assignments: Array<[number, number]> = [];
    for (const entry of result) {
      assignments.push([
        entry.payload.fields["a"] as number,
        entry.payload.fields["b"] as number,
      ]);
    }
    expect(assignments).toContainEqual([1, 2]);
    expect(assignments).toContainEqual([1, 3]);
    expect(assignments).toContainEqual([2, 3]);
    expect(assignments).toContainEqual([3, 4]);
    expect(assignments).not.toContainEqual([4, 5]);
  });
});
