import { describe, expect, it } from "vitest";
import { createVertex, createEdge } from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import { PatternMatchOperator } from "../../src/graph/operators.js";
import {
  createGraphPattern,
  createVertexPattern,
  createEdgePattern,
} from "../../src/graph/pattern.js";
import type { ExecutionContext } from "../../src/operators/base.js";

function makeSocialGraph(): MemoryGraphStore {
  const gs = new MemoryGraphStore();
  gs.createGraph("g");
  gs.addVertex(createVertex(1, "person", { name: "Alice" }), "g");
  gs.addVertex(createVertex(2, "person", { name: "Bob" }), "g");
  gs.addVertex(createVertex(3, "person", { name: "Carol" }), "g");
  gs.addVertex(createVertex(4, "person", { name: "Dave" }), "g");
  gs.addEdge(createEdge(10, 1, 2, "knows", {}), "g");
  gs.addEdge(createEdge(11, 1, 3, "knows", {}), "g");
  gs.addEdge(createEdge(12, 2, 3, "knows", {}), "g");
  gs.addEdge(createEdge(13, 1, 4, "blocks", {}), "g");
  return gs;
}

describe("PatternNegation", () => {
  it("positive edge pattern", () => {
    const gs = makeSocialGraph();
    const pattern = createGraphPattern(
      [createVertexPattern("a"), createVertexPattern("b")],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new PatternMatchOperator(pattern, "g");
    const result = op.execute(ctx);
    expect(result.entries.length).toBe(3);
  });

  it("negated edge basic", () => {
    const gs = makeSocialGraph();
    const pattern = createGraphPattern(
      [createVertexPattern("a"), createVertexPattern("b")],
      [
        createEdgePattern("a", "b", { label: "knows" }),
        createEdgePattern("a", "b", { label: "blocks", negated: true }),
      ],
    );
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new PatternMatchOperator(pattern, "g");
    const result = op.execute(ctx);
    expect(result.entries.length).toBe(3);
  });

  it("negated edge filters match", () => {
    const gs = makeSocialGraph();
    const pattern = createGraphPattern(
      [createVertexPattern("a"), createVertexPattern("b")],
      [
        createEdgePattern("a", "b", { label: "blocks" }),
        createEdgePattern("a", "b", { label: "knows", negated: true }),
      ],
    );
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new PatternMatchOperator(pattern, "g");
    const result = op.execute(ctx);
    expect(result.entries.length).toBe(1);
    expect(result.entries[0]!.payload.fields["a"]).toBe(1);
    expect(result.entries[0]!.payload.fields["b"]).toBe(4);
  });

  it("negated edge removes all", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g");
    gs.addVertex(createVertex(1, "a", {}), "g");
    gs.addVertex(createVertex(2, "b", {}), "g");
    gs.addEdge(createEdge(10, 1, 2, "knows", {}), "g");
    gs.addEdge(createEdge(11, 1, 2, "blocks", {}), "g");

    const pattern = createGraphPattern(
      [createVertexPattern("a"), createVertexPattern("b")],
      [
        createEdgePattern("a", "b", { label: "knows" }),
        createEdgePattern("a", "b", { label: "blocks", negated: true }),
      ],
    );
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new PatternMatchOperator(pattern, "g");
    const result = op.execute(ctx);
    expect(result.entries.length).toBe(0);
  });

  it("negated only pattern", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g");
    gs.addVertex(createVertex(1, "a", {}), "g");
    gs.addVertex(createVertex(2, "b", {}), "g");
    gs.addVertex(createVertex(3, "c", {}), "g");
    gs.addEdge(createEdge(10, 1, 2, "e", {}), "g");

    const pattern = createGraphPattern(
      [createVertexPattern("a"), createVertexPattern("b")],
      [createEdgePattern("a", "b", { label: "e", negated: true })],
    );
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new PatternMatchOperator(pattern, "g");
    const result = op.execute(ctx);
    expect(result.entries.length).toBe(5);
  });

  it("negated edge no label", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g");
    gs.addVertex(createVertex(1, "a", {}), "g");
    gs.addVertex(createVertex(2, "b", {}), "g");
    gs.addVertex(createVertex(3, "c", {}), "g");
    gs.addEdge(createEdge(10, 1, 2, "e1", {}), "g");

    const pattern = createGraphPattern(
      [createVertexPattern("a"), createVertexPattern("b")],
      [createEdgePattern("a", "b", { label: null, negated: true })],
    );
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new PatternMatchOperator(pattern, "g");
    const result = op.execute(ctx);
    expect(result.entries.length).toBe(5);
  });

  it("negated edge default false", () => {
    const ep = createEdgePattern("a", "b", { label: "knows" });
    expect(ep.negated).toBe(false);
    const epNeg = createEdgePattern("a", "b", { label: "knows", negated: true });
    expect(epNeg.negated).toBe(true);
  });
});
