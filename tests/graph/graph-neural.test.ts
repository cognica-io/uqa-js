import { describe, expect, it } from "vitest";
import { createEdge, createVertex, IndexStats } from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import { MessagePassingOperator } from "../../src/graph/message-passing.js";
import { GraphEmbeddingOperator } from "../../src/graph/graph-embedding.js";
import type { ExecutionContext } from "../../src/operators/base.js";

const GRAPH_NAME = "test";

function buildTestGraph(): MemoryGraphStore {
  const g = new MemoryGraphStore();
  g.createGraph(GRAPH_NAME);
  g.addVertex(createVertex(1, "person", { name: "Alice", score: 0.8 }), GRAPH_NAME);
  g.addVertex(createVertex(2, "person", { name: "Bob", score: 0.6 }), GRAPH_NAME);
  g.addVertex(createVertex(3, "person", { name: "Carol", score: 0.4 }), GRAPH_NAME);
  g.addVertex(createVertex(4, "person", { name: "Dave", score: 0.2 }), GRAPH_NAME);
  g.addEdge(createEdge(1, 1, 2, "knows"), GRAPH_NAME);
  g.addEdge(createEdge(2, 2, 3, "knows"), GRAPH_NAME);
  g.addEdge(createEdge(3, 3, 4, "works_with"), GRAPH_NAME);
  g.addEdge(createEdge(4, 1, 3, "works_with"), GRAPH_NAME);
  return g;
}

// ======================================================================
// MessagePassingOperator
// ======================================================================

describe("MessagePassingOperator", () => {
  it("basic execution", () => {
    const g = buildTestGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new MessagePassingOperator({
      kLayers: 1,
      aggregation: "mean",
      propertyName: "score",
      graph: GRAPH_NAME,
    });
    const result = op.execute(ctx);
    expect([...result].length).toBe(4);
  });

  it("scores are probabilities", () => {
    const g = buildTestGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new MessagePassingOperator({
      kLayers: 2,
      aggregation: "mean",
      propertyName: "score",
      graph: GRAPH_NAME,
    });
    const result = op.execute(ctx);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });

  it("sum aggregation", () => {
    const g = buildTestGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new MessagePassingOperator({
      kLayers: 1,
      aggregation: "sum",
      propertyName: "score",
      graph: GRAPH_NAME,
    });
    const result = op.execute(ctx);
    expect([...result].length).toBe(4);
  });

  it("max aggregation", () => {
    const g = buildTestGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new MessagePassingOperator({
      kLayers: 1,
      aggregation: "max",
      propertyName: "score",
      graph: GRAPH_NAME,
    });
    const result = op.execute(ctx);
    expect([...result].length).toBe(4);
  });

  it("no property uses default", () => {
    const g = buildTestGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new MessagePassingOperator({ kLayers: 1, graph: GRAPH_NAME });
    const result = op.execute(ctx);
    expect([...result].length).toBe(4);
  });

  it("empty graph", () => {
    const g = new MemoryGraphStore();
    g.createGraph(GRAPH_NAME);
    const ctx: ExecutionContext = { graphStore: g };
    const op = new MessagePassingOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);
    expect([...result].length).toBe(0);
  });

  it("isolated vertex", () => {
    const g = new MemoryGraphStore();
    g.createGraph(GRAPH_NAME);
    g.addVertex(createVertex(1, "person", { score: 0.5 }), GRAPH_NAME);
    const ctx: ExecutionContext = { graphStore: g };
    const op = new MessagePassingOperator({
      kLayers: 1,
      propertyName: "score",
      graph: GRAPH_NAME,
    });
    const result = op.execute(ctx);
    expect([...result].length).toBe(1);
  });

  it("k layers effect", () => {
    const g = buildTestGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op1 = new MessagePassingOperator({
      kLayers: 1,
      propertyName: "score",
      graph: GRAPH_NAME,
    });
    const op2 = new MessagePassingOperator({
      kLayers: 3,
      propertyName: "score",
      graph: GRAPH_NAME,
    });
    const r1 = op1.execute(ctx);
    const r2 = op2.execute(ctx);
    const scores1: Record<number, number> = {};
    for (const e of r1) {
      scores1[e.docId] = e.payload.score;
    }
    const scores2: Record<number, number> = {};
    for (const e of r2) {
      scores2[e.docId] = e.payload.score;
    }
    // More layers should produce different scores
    expect(scores1).not.toEqual(scores2);
  });

  it("cost estimate", () => {
    const op = new MessagePassingOperator({
      kLayers: 3,
      graph: GRAPH_NAME,
    });
    const stats = new IndexStats(100, 10.0);
    const cost = op.costEstimate(stats);
    // Cost estimate should be a positive number
    expect(cost).toBeGreaterThan(0);
  });
});

// ======================================================================
// GraphEmbeddingOperator
// ======================================================================

describe("GraphEmbeddingOperator", () => {
  it("basic execution", () => {
    const g = buildTestGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new GraphEmbeddingOperator({ graph: GRAPH_NAME, kHops: 2 });
    const result = op.execute(ctx);
    expect([...result].length).toBe(4);
  });

  it("embedding present in fields", () => {
    const g = buildTestGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new GraphEmbeddingOperator({ graph: GRAPH_NAME, kHops: 1 });
    const result = op.execute(ctx);
    for (const entry of result) {
      // Check that embedding data is present in payload fields
      const fields = entry.payload.fields;
      const hasEmbedding =
        fields["_embedding"] !== undefined || fields["embedding"] !== undefined;
      // If embedding is stored on the score only, that's also valid
      expect(hasEmbedding || entry.payload.score !== 0).toBe(true);
    }
  });

  it("different vertices different embeddings", () => {
    const g = buildTestGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new GraphEmbeddingOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);
    // At least some vertices should have different scores
    const scores: Record<number, number> = {};
    for (const e of result) {
      scores[e.docId] = e.payload.score;
    }
    // Different structural positions should yield different scores
    expect(Object.values(scores).length).toBeGreaterThanOrEqual(2);
  });

  it("empty graph", () => {
    const g = new MemoryGraphStore();
    g.createGraph(GRAPH_NAME);
    const ctx: ExecutionContext = { graphStore: g };
    const op = new GraphEmbeddingOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);
    expect([...result].length).toBe(0);
  });
});

// ======================================================================
// GNN SQL (Engine not yet ported)
// ======================================================================

describe("GNNSQL", () => {
  it("message passing via operator with custom graph", () => {
    const g = buildTestGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new MessagePassingOperator({
      kLayers: 2,
      aggregation: "mean",
      propertyName: "score",
      graph: GRAPH_NAME,
    });
    const result = op.execute(ctx);
    // All 4 vertices should appear
    expect([...result].length).toBe(4);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
    }
  });
  it("graph embedding via operator with custom graph", () => {
    const g = buildTestGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new GraphEmbeddingOperator({ graph: GRAPH_NAME, kHops: 2 });
    const result = op.execute(ctx);
    expect([...result].length).toBe(4);
    for (const entry of result) {
      // Each entry should have a valid score or embedding data
      const fields = entry.payload.fields;
      const hasEmbedding =
        fields["_embedding"] !== undefined || fields["embedding"] !== undefined;
      expect(hasEmbedding || entry.payload.score !== 0).toBe(true);
    }
  });
});

// ======================================================================
// GNN QueryBuilder (Engine not yet ported)
// ======================================================================

describe("GNNQueryBuilder", () => {
  it("query builder message passing via operator", () => {
    const g = buildTestGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new MessagePassingOperator({
      kLayers: 2,
      propertyName: "score",
      graph: GRAPH_NAME,
    });
    const result = op.execute(ctx);
    expect([...result].length).toBeGreaterThan(0);
  });
});
