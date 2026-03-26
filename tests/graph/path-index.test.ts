import { describe, expect, it } from "vitest";
import { createEdge, createVertex, IndexStats } from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import { PathIndex } from "../../src/graph/index.js";
import { RegularPathQueryOperator } from "../../src/graph/operators.js";
import { Label, Concat, Alternation, KleeneStar } from "../../src/graph/pattern.js";
import { CostModel } from "../../src/planner/cost-model.js";
import { Engine } from "../../src/engine.js";
import type { ExecutionContext } from "../../src/operators/base.js";

const GRAPH_NAME = "test";

function buildTestGraph(): MemoryGraphStore {
  const g = new MemoryGraphStore();
  g.createGraph(GRAPH_NAME);
  g.addVertex(createVertex(1, "person", { name: "Alice" }), GRAPH_NAME);
  g.addVertex(createVertex(2, "person", { name: "Bob" }), GRAPH_NAME);
  g.addVertex(createVertex(3, "person", { name: "Carol" }), GRAPH_NAME);
  g.addVertex(createVertex(4, "person", { name: "Dave" }), GRAPH_NAME);
  g.addEdge(createEdge(1, 1, 2, "knows"), GRAPH_NAME);
  g.addEdge(createEdge(2, 2, 3, "works_with"), GRAPH_NAME);
  g.addEdge(createEdge(3, 3, 4, "manages"), GRAPH_NAME);
  g.addEdge(createEdge(4, 1, 3, "knows"), GRAPH_NAME);
  return g;
}

// ======================================================================
// PathIndex
// ======================================================================

describe("PathIndex", () => {
  it("build single label", () => {
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows"]]);
    const pairs = idx.lookup(["knows"], GRAPH_NAME);
    expect(pairs.length).toBeGreaterThan(0);
    const pairSet = pairs.map(([s, e]) => `${String(s)}->${String(e)}`);
    expect(pairSet).toContain("1->2");
    expect(pairSet).toContain("1->3");
  });

  it("build two label sequence", () => {
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows", "works_with"]]);
    const pairs = idx.lookup(["knows", "works_with"], GRAPH_NAME);
    expect(pairs.length).toBeGreaterThan(0);
    const pairSet = pairs.map(([s, e]) => `${String(s)}->${String(e)}`);
    expect(pairSet).toContain("1->3");
  });

  it("lookup unindexed path", () => {
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows"]]);
    const pairs = idx.lookup(["works_with"], GRAPH_NAME);
    expect(pairs.length).toBe(0);
  });

  it("has path", () => {
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows"]]);
    expect(idx.hasPath(["knows"], GRAPH_NAME)).toBe(true);
    expect(idx.hasPath(["manages"], GRAPH_NAME)).toBe(false);
  });

  it("indexed paths", () => {
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows"], ["works_with"]]);
    const paths = idx.indexedPaths();
    // indexedPaths returns string[][] -- flatten to check labels
    const flatLabels = paths.map((p) => p.join("/"));
    expect(flatLabels).toContain("knows");
    expect(flatLabels).toContain("works_with");
  });
});

// ======================================================================
// RPQ with PathIndex
// ======================================================================

describe("RPQWithPathIndex", () => {
  it("rpq uses index for simple label", () => {
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows"]]);
    const ctx: ExecutionContext = { graphStore: g, pathIndex: idx };

    const op = new RegularPathQueryOperator(new Label("knows"), GRAPH_NAME, 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(docIds.has(2)).toBe(true);
    expect(docIds.has(3)).toBe(true);
  });

  it("rpq uses index for concat", () => {
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows", "works_with"]]);
    const ctx: ExecutionContext = { graphStore: g, pathIndex: idx };

    const expr = new Concat(new Label("knows"), new Label("works_with"));
    const op = new RegularPathQueryOperator(expr, GRAPH_NAME, 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(docIds.has(3)).toBe(true);
  });

  it("rpq falls back for alternation", () => {
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows"]]);
    const ctx: ExecutionContext = { graphStore: g, pathIndex: idx };

    const expr = new Alternation(new Label("knows"), new Label("works_with"));
    const op = new RegularPathQueryOperator(expr, GRAPH_NAME, 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(docIds.has(2) || docIds.has(3)).toBe(true);
  });

  it("rpq falls back for kleene star", () => {
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows"]]);
    const ctx: ExecutionContext = { graphStore: g, pathIndex: idx };

    const expr = new KleeneStar(new Label("knows"));
    const op = new RegularPathQueryOperator(expr, GRAPH_NAME, 1);
    const result = op.execute(ctx);
    expect([...result].length).toBeGreaterThan(0);
  });

  it("rpq without index", () => {
    const g = buildTestGraph();
    const ctx: ExecutionContext = { graphStore: g };

    const op = new RegularPathQueryOperator(new Label("knows"), GRAPH_NAME, 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(docIds.has(2)).toBe(true);
  });

  it("rpq all starts with index", () => {
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows"]]);
    const ctx: ExecutionContext = { graphStore: g, pathIndex: idx };

    const op = new RegularPathQueryOperator(new Label("knows"), GRAPH_NAME);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(docIds.has(2)).toBe(true);
    expect(docIds.has(3)).toBe(true);
  });

  it("extract label sequence single via index lookup", () => {
    // Verify _extractLabelSequence(Label("knows")) == ["knows"] indirectly:
    // If the RPQ operator can use an index for Label("knows"), the label sequence is ["knows"].
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows"]]);
    const ctx: ExecutionContext = { graphStore: g, pathIndex: idx };

    const op = new RegularPathQueryOperator(new Label("knows"), GRAPH_NAME, 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    // If label sequence extraction works, index is used and results are correct
    expect(docIds.has(2)).toBe(true);
    expect(docIds.has(3)).toBe(true);
  });
  it("extract label sequence concat via index lookup", () => {
    // Verify _extractLabelSequence(Concat(Label("a"), Label("b"))) == ["a", "b"] indirectly
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows", "works_with"]]);
    const ctx: ExecutionContext = { graphStore: g, pathIndex: idx };

    const expr = new Concat(new Label("knows"), new Label("works_with"));
    const op = new RegularPathQueryOperator(expr, GRAPH_NAME, 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(docIds.has(3)).toBe(true);
  });
  it("extract label sequence nested concat via index lookup", () => {
    // Verify nested concat: Concat(Concat(Label("a"), Label("b")), Label("c")) == ["a","b","c"]
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows", "works_with", "manages"]]);
    const ctx: ExecutionContext = { graphStore: g, pathIndex: idx };

    const expr = new Concat(
      new Concat(new Label("knows"), new Label("works_with")),
      new Label("manages"),
    );
    const op = new RegularPathQueryOperator(expr, GRAPH_NAME, 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    // 1->knows->2->works_with->3->manages->4, and 1->knows->3->manages->4
    expect(docIds.has(4)).toBe(true);
  });
  it("extract label sequence alternation falls back to non-index path", () => {
    // Alternation cannot produce a label sequence, so the operator falls back
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows"]]);
    const ctx: ExecutionContext = { graphStore: g, pathIndex: idx };

    const expr = new Alternation(new Label("knows"), new Label("works_with"));
    const op = new RegularPathQueryOperator(expr, GRAPH_NAME, 1);
    const result = op.execute(ctx);
    // Falls back to graph traversal; still returns results
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(docIds.has(2) || docIds.has(3)).toBe(true);
  });
  it("extract label sequence kleene star falls back to non-index path", () => {
    // KleeneStar cannot produce a label sequence, so the operator falls back
    const g = buildTestGraph();
    const idx = new PathIndex();
    idx.build(g, GRAPH_NAME, [["knows"]]);
    const ctx: ExecutionContext = { graphStore: g, pathIndex: idx };

    const expr = new KleeneStar(new Label("knows"));
    const op = new RegularPathQueryOperator(expr, GRAPH_NAME, 1);
    const result = op.execute(ctx);
    expect([...result].length).toBeGreaterThan(0);
  });
});

// ======================================================================
// Engine path index management (Engine not yet ported)
// ======================================================================

describe("EnginePathIndex", () => {
  it("build and get path index", () => {
    const e = new Engine();
    const g = e.createGraph("social");
    g.addVertex(createVertex(1, "person", { name: "Alice" }), "social");
    g.addVertex(createVertex(2, "person", { name: "Bob" }), "social");
    g.addEdge(createEdge(1, 1, 2, "knows"), "social");

    e.buildPathIndex("social", [["knows"]]);
    const idx = e.getPathIndex("social");
    expect(idx).not.toBeNull();
    expect(idx!.hasPath(["knows"], "social")).toBe(true);
  });
  it("drop path index", () => {
    const e = new Engine();
    const g = e.createGraph("social");
    g.addVertex(createVertex(1, "person"), "social");
    g.addVertex(createVertex(2, "person"), "social");
    g.addEdge(createEdge(1, 1, 2, "knows"), "social");

    e.buildPathIndex("social", [["knows"]]);
    e.dropPathIndex("social");
    expect(e.getPathIndex("social")).toBeNull();
  });
  it("build index nonexistent graph", () => {
    const e = new Engine();
    expect(() => e.buildPathIndex("nonexistent", [["knows"]])).toThrow(
      /does not exist/,
    );
  });
  it("path index persistence via rebuild", () => {
    const e = new Engine();
    const g = e.createGraph("social");
    g.addVertex(createVertex(1, "person", { name: "Alice" }), "social");
    g.addVertex(createVertex(2, "person", { name: "Bob" }), "social");
    g.addEdge(createEdge(1, 1, 2, "knows"), "social");
    e.buildPathIndex("social", [["knows"]]);

    // Verify the index is usable for RPQ
    const idx = e.getPathIndex("social")!;
    expect(idx.hasPath(["knows"], "social")).toBe(true);
    const pairs = idx.lookup(["knows"], "social");
    expect(pairs.length).toBeGreaterThan(0);
  });
});

// ======================================================================
// Cost model with path index (CostModel not yet ported)
// ======================================================================

describe("CostModelWithPathIndex", () => {
  it("indexable rpq cheaper", () => {
    const stats = new IndexStats(100);
    const model = new CostModel();

    const simpleOp = new RegularPathQueryOperator(new Label("knows"), GRAPH_NAME);
    const complexOp = new RegularPathQueryOperator(
      new Alternation(new Label("knows"), new Label("works_with")),
      GRAPH_NAME,
    );

    const simpleCost = model.estimate(simpleOp, stats);
    const complexCost = model.estimate(complexOp, stats);
    expect(simpleCost).toBeLessThan(complexCost);
  });
});
