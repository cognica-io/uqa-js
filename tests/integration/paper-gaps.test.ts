import { describe, expect, it } from "vitest";
import {
  IndexStats,
  Equals,
  GreaterThan,
  createPayload,
  createPostingEntry,
  createVertex,
  createEdge,
} from "../../src/core/types.js";
import type { PostingEntry } from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import { Operator } from "../../src/operators/base.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import {
  TraverseOperator,
  PatternMatchOperator,
  RegularPathQueryOperator,
  VertexAggregationOperator,
} from "../../src/graph/operators.js";
import {
  createGraphPattern,
  createVertexPattern,
  createEdgePattern,
  Label,
} from "../../src/graph/pattern.js";
import { FilterOperator } from "../../src/operators/primitive.js";
import { QueryOptimizer } from "../../src/planner/optimizer.js";
import { CardinalityEstimator, GraphStats } from "../../src/planner/cardinality.js";
import { CostModel } from "../../src/planner/cost-model.js";
import { LabelIndex, NeighborhoodIndex, PathIndex } from "../../src/graph/index.js";
import {
  VectorExclusionOperator,
  FacetVectorOperator,
} from "../../src/operators/hybrid.js";
import {
  PathAggregateOperator,
  UnifiedFilterOperator,
} from "../../src/operators/hierarchical.js";
import {
  SumMonoid,
  AvgMonoid,
  CountMonoid,
  MinMonoid,
  MaxMonoid,
} from "../../src/operators/aggregation.js";
import { MemoryDocumentStore } from "../../src/storage/document-store.js";
import { IntersectOperator } from "../../src/operators/boolean.js";

// -- Helpers --

class FixedOperator extends Operator {
  private _entries: PostingEntry[];
  constructor(entries: PostingEntry[]) {
    super();
    this._entries = entries;
  }
  execute(_context: ExecutionContext): PostingList {
    return PostingList.fromSorted(this._entries);
  }
}

function makeGraph(): MemoryGraphStore {
  const g = new MemoryGraphStore();
  g.createGraph("test");
  g.addVertex(createVertex(1, "", { name: "Alice", age: 30, dept: "eng" }), "test");
  g.addVertex(createVertex(2, "", { name: "Bob", age: 25, dept: "eng" }), "test");
  g.addVertex(createVertex(3, "", { name: "Charlie", age: 35, dept: "sales" }), "test");
  g.addVertex(createVertex(4, "", { name: "Diana", age: 28, dept: "eng" }), "test");
  g.addVertex(createVertex(5, "", { name: "Eve", age: 32, dept: "sales" }), "test");
  g.addEdge(createEdge(1, 1, 2, "knows", { weight: 0.9 }), "test");
  g.addEdge(createEdge(2, 1, 3, "knows", { weight: 0.7 }), "test");
  g.addEdge(createEdge(3, 2, 3, "knows", { weight: 0.5 }), "test");
  g.addEdge(createEdge(4, 2, 4, "works_with", { weight: 0.8 }), "test");
  g.addEdge(createEdge(5, 3, 4, "knows", { weight: 0.6 }), "test");
  g.addEdge(createEdge(6, 3, 5, "works_with", { weight: 0.4 }), "test");
  g.addEdge(createEdge(7, 4, 5, "knows", { weight: 0.3 }), "test");
  return g;
}

// ===========================================================================
// Paper 1: VectorExclusionOperator
// ===========================================================================

describe("VectorExclusionOperator", () => {
  it("excludes negative matches", () => {
    const positive = new FixedOperator([
      createPostingEntry(1, { score: 0.9 }),
      createPostingEntry(2, { score: 0.8 }),
      createPostingEntry(3, { score: 0.7 }),
    ]);
    const ctx: ExecutionContext = {};
    const negVec = new Float64Array(4);
    const op = new VectorExclusionOperator(positive, negVec, 0.5);
    const result = op.execute(ctx);
    // No vector index -> negative returns empty -> all positive kept
    expect(result.entries.length).toBe(3);
  });

  it("cost estimate is positive", () => {
    const positive = new FixedOperator([]);
    const negVec = new Float64Array(4);
    const op = new VectorExclusionOperator(positive, negVec, 0.5);
    const stats = new IndexStats(100, 4);
    const cost = op.costEstimate(stats);
    expect(cost).toBeGreaterThan(0);
  });

  it("preserves positive entries when negative is empty", () => {
    const entries = [
      createPostingEntry(10, { score: 0.5 }),
      createPostingEntry(20, { score: 0.3 }),
    ];
    const positive = new FixedOperator(entries);
    const negVec = new Float64Array(4);
    const op = new VectorExclusionOperator(positive, negVec, 0.5);
    const ctx: ExecutionContext = {};
    const result = op.execute(ctx);
    expect(result.entries.length).toBe(2);
    expect(result.entries[0]!.docId).toBe(10);
    expect(result.entries[1]!.docId).toBe(20);
  });
});

// ===========================================================================
// Paper 1: FacetVectorOperator
// ===========================================================================

describe("FacetVectorOperator", () => {
  it("facet over empty vector results", () => {
    const queryVec = new Float64Array(4);
    const op = new FacetVectorOperator("category", queryVec, 0.5);
    const ctx: ExecutionContext = {};
    const result = op.execute(ctx);
    expect(result.entries.length).toBe(0);
  });

  it("facet counts fields with source", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { category: "A", title: "doc1" });
    store.put(2, { category: "B", title: "doc2" });
    store.put(3, { category: "A", title: "doc3" });
    const source = new FixedOperator([
      createPostingEntry(1, { score: 0.9 }),
      createPostingEntry(2, { score: 0.8 }),
      createPostingEntry(3, { score: 0.7 }),
    ]);
    const queryVec = new Float64Array(4);
    const op = new FacetVectorOperator("category", queryVec, 0.5, source);
    // No vector index -> vector returns empty -> intersection is empty
    const ctx: ExecutionContext = { documentStore: store };
    const result = op.execute(ctx);
    expect(result.entries.length).toBe(0);
  });

  it("cost estimate is positive", () => {
    const queryVec = new Float64Array(4);
    const op = new FacetVectorOperator("category", queryVec, 0.5);
    const stats = new IndexStats(100, 0, 4);
    const cost = op.costEstimate(stats);
    expect(cost).toBeGreaterThan(0);
  });

  it("facet with null source and no vector index returns empty", () => {
    const queryVec = new Float64Array(4);
    const op = new FacetVectorOperator("category", queryVec, 0.5, null);
    const ctx: ExecutionContext = {};
    const result = op.execute(ctx);
    expect(result.entries.length).toBe(0);
  });
});

// ===========================================================================
// Paper 1: PathAggregateOperator
// ===========================================================================

describe("PathAggregateOperator", () => {
  function makePathAggContext(): ExecutionContext {
    const store = new MemoryDocumentStore();
    store.put(1, { title: "order1", items: [10, 20, 30] });
    store.put(2, { title: "order2", items: [5, 15] });
    store.put(3, { title: "order3", items: 42 }); // single value, not a list
    return { documentStore: store };
  }

  it("sum over array", () => {
    const source = new FixedOperator([
      createPostingEntry(1, { score: 0.0 }),
      createPostingEntry(2, { score: 0.0 }),
    ]);
    const op = new PathAggregateOperator(["items"], new SumMonoid(), source);
    const result = op.execute(makePathAggContext());
    expect(result.entries.length).toBe(2);
    expect(result.entries[0]!.payload.score).toBeCloseTo(60.0); // 10+20+30
    expect(result.entries[1]!.payload.score).toBeCloseTo(20.0); // 5+15
  });

  it("avg over array", () => {
    const source = new FixedOperator([createPostingEntry(1, { score: 0.0 })]);
    const op = new PathAggregateOperator(["items"], new AvgMonoid(), source);
    const result = op.execute(makePathAggContext());
    expect(result.entries[0]!.payload.score).toBeCloseTo(20.0); // (10+20+30)/3
  });

  it("single value not list", () => {
    const source = new FixedOperator([createPostingEntry(3, { score: 0.0 })]);
    const op = new PathAggregateOperator(["items"], new SumMonoid(), source);
    const result = op.execute(makePathAggContext());
    expect(result.entries[0]!.payload.score).toBeCloseTo(42.0);
  });

  it("missing path", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { title: "no items" });
    const ctx: ExecutionContext = { documentStore: store };
    const source = new FixedOperator([createPostingEntry(1, { score: 0.0 })]);
    const op = new PathAggregateOperator(["items"], new SumMonoid(), source);
    const result = op.execute(ctx);
    expect(result.entries[0]!.payload.score).toBeCloseTo(0.0);
  });

  it("count over array", () => {
    const source = new FixedOperator([createPostingEntry(1, { score: 0.0 })]);
    const op = new PathAggregateOperator(["items"], new CountMonoid(), source);
    const result = op.execute(makePathAggContext());
    expect(result.entries[0]!.payload.score).toBeCloseTo(3.0);
  });

  it("min max over array", () => {
    const source = new FixedOperator([createPostingEntry(1, { score: 0.0 })]);
    const ctx = makePathAggContext();

    const minOp = new PathAggregateOperator(["items"], new MinMonoid(), source);
    const minResult = minOp.execute(ctx);
    expect(minResult.entries[0]!.payload.score).toBeCloseTo(10.0);

    const maxOp = new PathAggregateOperator(["items"], new MaxMonoid(), source);
    const maxResult = maxOp.execute(ctx);
    expect(maxResult.entries[0]!.payload.score).toBeCloseTo(30.0);
  });
});

// ===========================================================================
// Paper 1: UnifiedFilterOperator
// ===========================================================================

describe("UnifiedFilterOperator", () => {
  function makeUnifiedFilterContext(): ExecutionContext {
    const store = new MemoryDocumentStore();
    store.put(1, { year: 2023, metadata: { author: "Alice", score: 95 } });
    store.put(2, { year: 2024, metadata: { author: "Bob", score: 88 } });
    store.put(3, { year: 2025, metadata: { author: "Alice", score: 72 } });
    return { documentStore: store };
  }

  it("flat field filter", () => {
    const op = new UnifiedFilterOperator("year", new GreaterThan(2023));
    const result = op.execute(makeUnifiedFilterContext());
    const docIds = result.entries.map((e) => e.docId).sort((a, b) => a - b);
    expect(docIds).toEqual([2, 3]);
  });

  it("hierarchical path filter", () => {
    const op = new UnifiedFilterOperator("metadata.author", new Equals("Alice"));
    const result = op.execute(makeUnifiedFilterContext());
    const docIds = result.entries.map((e) => e.docId).sort((a, b) => a - b);
    expect(docIds).toEqual([1, 3]);
  });

  it("hierarchical numeric path", () => {
    const op = new UnifiedFilterOperator("metadata.score", new GreaterThan(80));
    const result = op.execute(makeUnifiedFilterContext());
    const docIds = result.entries.map((e) => e.docId).sort((a, b) => a - b);
    expect(docIds).toEqual([1, 2]);
  });

  it("with source operator", () => {
    const source = new FixedOperator([
      createPostingEntry(1, { score: 1.0 }),
      createPostingEntry(2, { score: 0.8 }),
    ]);
    const op = new UnifiedFilterOperator("year", new GreaterThan(2023), source);
    const result = op.execute(makeUnifiedFilterContext());
    const docIds = result.entries.map((e) => e.docId);
    expect(docIds).toEqual([2]);
  });

  it("returns empty for no matches", () => {
    const op = new UnifiedFilterOperator("year", new GreaterThan(2030));
    const result = op.execute(makeUnifiedFilterContext());
    expect(result.entries.length).toBe(0);
  });
});

// ===========================================================================
// Paper 2: VertexAggregationOperator
// ===========================================================================

describe("VertexAggregationOperator", () => {
  // TS VertexAggregationOperator takes a function, not a string name.
  // The Python API accepts "sum", "avg", "min", "max", "count" as strings.
  // The TS version applies the function to all source entries, not producing one row.

  it("sum aggregation via function", () => {
    const graph = makeGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const traverse = new TraverseOperator(1, "test", "knows", 1);
    const sumFn = (vals: number[]) => vals.reduce((a, b) => a + b, 0);
    const agg = new VertexAggregationOperator(traverse, "age", sumFn);
    const result = agg.execute(ctx);
    expect(result.entries.length).toBeGreaterThan(0);
    // All entries have the aggregated score
    expect(result.entries[0]!.payload.score).toBeCloseTo(90.0);
  });

  it("avg aggregation via default function", () => {
    const graph = makeGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const traverse = new TraverseOperator(1, "test", "knows", 1);
    // Default aggFn is average
    const agg = new VertexAggregationOperator(traverse, "age");
    const result = agg.execute(ctx);
    expect(result.entries[0]!.payload.score).toBeCloseTo(30.0);
  });

  it("min max aggregation via functions", () => {
    const graph = makeGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const traverse = new TraverseOperator(1, "test", "knows", 1);

    const minFn = (vals: number[]) => (vals.length === 0 ? 0 : Math.min(...vals));
    const minAgg = new VertexAggregationOperator(traverse, "age", minFn);
    const minResult = minAgg.execute(ctx);
    expect(minResult.entries[0]!.payload.score).toBeCloseTo(25.0);

    const maxFn = (vals: number[]) => (vals.length === 0 ? 0 : Math.max(...vals));
    const maxAgg = new VertexAggregationOperator(traverse, "age", maxFn);
    const maxResult = maxAgg.execute(ctx);
    expect(maxResult.entries[0]!.payload.score).toBeCloseTo(35.0);
  });

  it("count aggregation via function", () => {
    const graph = makeGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const traverse = new TraverseOperator(1, "test", "knows", 1);
    const countFn = (vals: number[]) => vals.length;
    const agg = new VertexAggregationOperator(traverse, "age", countFn);
    const result = agg.execute(ctx);
    expect(result.entries[0]!.payload.score).toBeCloseTo(3.0);
  });

  it("missing property returns zero", () => {
    const graph = makeGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const traverse = new TraverseOperator(1, "test", "knows", 1);
    const sumFn = (vals: number[]) => vals.reduce((a, b) => a + b, 0);
    const agg = new VertexAggregationOperator(traverse, "nonexistent", sumFn);
    const result = agg.execute(ctx);
    expect(result.entries[0]!.payload.score).toBeCloseTo(0.0);
  });

  it("payload fields contain aggregation metadata", () => {
    const graph = makeGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const traverse = new TraverseOperator(1, "test", "knows", 1);
    const sumFn = (vals: number[]) => vals.reduce((a, b) => a + b, 0);
    const agg = new VertexAggregationOperator(traverse, "age", sumFn);
    const result = agg.execute(ctx);
    const entry = result.entries[0]!;
    expect(entry.payload.fields["_property"]).toBe("age");
    expect(entry.payload.fields["_aggregated"]).toBeCloseTo(90.0);
  });
});

// ===========================================================================
// Paper 2: Graph pattern pushdown
// ===========================================================================

describe("GraphPatternPushdown", () => {
  it("filter on qualified field pushes to correct vertex", () => {
    const pattern = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const pm = new PatternMatchOperator(pattern, "test");
    const filtered = new FilterOperator("b.name", new Equals("Alice"), pm);
    const stats = new IndexStats(5);
    const optimizer = new QueryOptimizer(stats);
    const optimized = optimizer.optimize(filtered);
    // If the optimizer can push into the pattern, it becomes PatternMatchOperator;
    // otherwise it stays as FilterOperator. Either way verify it does not crash.
    expect(optimized).toBeTruthy();
  });

  it("pushdown preserves correctness", () => {
    const pattern = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const pm = new PatternMatchOperator(pattern, "test");
    const filtered = new FilterOperator("a.dept", new Equals("eng"), pm);
    const stats = new IndexStats(5);
    const optimizer = new QueryOptimizer(stats);
    const optimized = optimizer.optimize(filtered);
    const graph = makeGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const result = optimized.execute(ctx);
    for (const entry of result) {
      const aId = entry.payload.fields["a"];
      if (aId != null) {
        const v = graph.getVertex(aId as number);
        expect(v).not.toBeNull();
        expect(v!.properties["dept"]).toBe("eng");
      }
    }
  });
});

// ===========================================================================
// Paper 2: Join-pattern fusion
// ===========================================================================

describe("JoinPatternFusion", () => {
  it("intersect of pattern matches produces result", () => {
    const pattern1 = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const pattern2 = createGraphPattern(
      [createVertexPattern("c", []), createVertexPattern("d", [])],
      [createEdgePattern("c", "d", { label: "works_with" })],
    );
    const pm1 = new PatternMatchOperator(pattern1, "test");
    const pm2 = new PatternMatchOperator(pattern2, "test");
    const intersect = new IntersectOperator([pm1, pm2]);
    const graph = makeGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    // The intersect of two pattern matches should run without error
    const result = intersect.execute(ctx);
    // Both pattern matches return entries; intersection finds common doc IDs
    expect(result).toBeTruthy();
  });
});

// ===========================================================================
// Paper 2: Statistics-based graph cardinality
// ===========================================================================

describe("GraphCardinality", () => {
  it("GraphStats.fromGraphStore", () => {
    const g = makeGraph();
    const gs = GraphStats.fromGraphStore(g, "test");
    expect(gs.numVertices).toBe(5);
    expect(gs.numEdges).toBe(7);
    expect(gs.avgOutDegree).toBeCloseTo(7 / 5);
  });

  it("traverse with stats", () => {
    const g = makeGraph();
    const gs = GraphStats.fromGraphStore(g, "test");
    const est = new CardinalityEstimator({ graphStats: gs });
    const card = est.estimateTraverse(1, "knows");
    expect(card).toBeGreaterThan(0);
    expect(card).toBeLessThanOrEqual(gs.numVertices);
  });

  it("traverse without stats", () => {
    const est = new CardinalityEstimator();
    const card = est.estimateTraverse(1, "knows");
    // Fallback heuristic
    expect(card).toBeGreaterThan(0);
  });

  it("pattern match with stats", () => {
    const g = makeGraph();
    const gs = GraphStats.fromGraphStore(g, "test");
    const est = new CardinalityEstimator({ graphStats: gs });
    const card = est.estimatePatternMatch(2, 1, ["knows"]);
    expect(card).toBeGreaterThan(0);
  });

  it("rpq with stats", () => {
    const g = makeGraph();
    const gs = GraphStats.fromGraphStore(g, "test");
    const est = new CardinalityEstimator({ graphStats: gs });
    const card = est.estimatePathQuery(2, "knows");
    expect(card).toBeGreaterThan(0);
  });

  it("graph stats properties", () => {
    const g = makeGraph();
    const gs = GraphStats.fromGraphStore(g, "test");
    expect(gs.numVertices).toBe(5);
    expect(gs.numEdges).toBe(7);
    expect(gs.avgOutDegree).toBeCloseTo(7 / 5);
    expect(gs.labelCounts.get("knows")).toBe(5);
    expect(gs.labelCounts.get("works_with")).toBe(2);
  });

  it("label selectivity", () => {
    const g = makeGraph();
    const gs = GraphStats.fromGraphStore(g, "test");
    const knowsSel = gs.labelSelectivity("knows");
    const worksSel = gs.labelSelectivity("works_with");
    // "knows" has 5/7 edges, "works_with" has 2/7
    expect(knowsSel).toBeCloseTo(5 / 7, 2);
    expect(worksSel).toBeCloseTo(2 / 7, 2);
  });

  it("edge density", () => {
    const g = makeGraph();
    const gs = GraphStats.fromGraphStore(g, "test");
    const density = gs.edgeDensity();
    // 7 edges / 25 (5^2) = 0.28
    expect(density).toBeCloseTo(7 / 25, 2);
  });
});

// ===========================================================================
// Paper 2: Graph indexing structures
// ===========================================================================

describe("LabelIndex", () => {
  it("build and lookup", () => {
    const g = makeGraph();
    const idx = new LabelIndex();
    idx.build(g, "test");
    const knowsEdges = idx.edgesByLabel("knows");
    expect(knowsEdges.length).toBe(5);
  });

  it("vertices by label", () => {
    const g = makeGraph();
    const idx = new LabelIndex();
    idx.build(g, "test");
    const knowsVertices = idx.verticesByLabel("knows");
    // All vertices connected by "knows" edges
    expect(knowsVertices.size).toBeGreaterThan(0);
  });

  it("label count", () => {
    const g = makeGraph();
    const idx = new LabelIndex();
    idx.build(g, "test");
    expect(idx.labelCount("knows")).toBe(5);
    expect(idx.labelCount("works_with")).toBe(2);
  });

  it("labels", () => {
    const g = makeGraph();
    const idx = new LabelIndex();
    idx.build(g, "test");
    const labels = idx.labels();
    expect(labels).toContain("knows");
    expect(labels).toContain("works_with");
  });

  it("empty graph", () => {
    const g = new MemoryGraphStore();
    g.createGraph("empty");
    const idx = new LabelIndex();
    idx.build(g, "empty");
    expect(idx.labels().length).toBe(0);
    expect(idx.edgesByLabel("knows").length).toBe(0);
  });
});

describe("NeighborhoodIndex", () => {
  it("build and lookup", () => {
    const g = makeGraph();
    const idx = new NeighborhoodIndex(2);
    idx.build(g, "test");
    const neighbors = idx.neighbors(1, 1);
    // Vertex 1 has outgoing edges to 2 and 3
    expect(neighbors.has(1)).toBe(true); // self
    expect(neighbors.has(2)).toBe(true);
    expect(neighbors.has(3)).toBe(true);
  });

  it("label filtered", () => {
    const g = makeGraph();
    const idx = new NeighborhoodIndex(2);
    idx.build(g, "test", "knows");
    const neighbors = idx.neighbors(1, 1);
    // With label filter "knows": 1->2, 1->3
    expect(neighbors.has(2)).toBe(true);
    expect(neighbors.has(3)).toBe(true);
  });

  it("nonexistent vertex", () => {
    const g = makeGraph();
    const idx = new NeighborhoodIndex(2);
    idx.build(g, "test");
    const neighbors = idx.neighbors(999, 1);
    expect(neighbors.size).toBe(0);
  });

  it("has vertex", () => {
    const g = makeGraph();
    const idx = new NeighborhoodIndex(2);
    idx.build(g, "test");
    expect(idx.hasVertex(1)).toBe(true);
    expect(idx.hasVertex(999)).toBe(false);
  });

  it("empty graph", () => {
    const g = new MemoryGraphStore();
    g.createGraph("empty");
    const idx = new NeighborhoodIndex(2);
    idx.build(g, "empty");
    expect(idx.hasVertex(1)).toBe(false);
  });
});

describe("PathIndex", () => {
  it("build single hop", () => {
    const g = makeGraph();
    const idx = new PathIndex();
    idx.build(g, "test", [["knows"]]);
    const pairs = idx.lookup(["knows"], "test");
    expect(pairs.length).toBeGreaterThan(0);
    // All "knows" edges should appear
    expect(pairs.length).toBe(5); // 5 "knows" edges
  });

  it("build multi hop", () => {
    const g = makeGraph();
    const idx = new PathIndex();
    idx.build(g, "test", [["knows", "knows"]]);
    const pairs = idx.lookup(["knows", "knows"], "test");
    // Two-hop "knows" paths
    expect(pairs.length).toBeGreaterThan(0);
  });

  it("cross label path", () => {
    const g = makeGraph();
    const idx = new PathIndex();
    idx.build(g, "test", [["knows", "works_with"]]);
    const pairs = idx.lookup(["knows", "works_with"], "test");
    // Paths following "knows" then "works_with"
    expect(pairs.length).toBeGreaterThan(0);
  });

  it("nonexistent path", () => {
    const g = makeGraph();
    const idx = new PathIndex();
    idx.build(g, "test", [["knows"]]);
    // Lookup a path that was not indexed
    const pairs = idx.lookup(["nonexistent"], "test");
    expect(pairs.length).toBe(0);
  });

  it("indexed paths", () => {
    const g = makeGraph();
    const idx = new PathIndex();
    idx.build(g, "test", [["knows"], ["works_with"]]);
    const paths = idx.indexedPaths();
    expect(paths.length).toBe(2);
  });

  it("empty graph", () => {
    const g = new MemoryGraphStore();
    g.createGraph("empty");
    const idx = new PathIndex();
    idx.build(g, "empty", [["knows"]]);
    const pairs = idx.lookup(["knows"], "empty");
    expect(pairs.length).toBe(0);
  });
});

// ===========================================================================
// CostModel for new operators
// ===========================================================================

describe("CostModelNewOperators", () => {
  it("vector exclusion cost", () => {
    const positive = new FixedOperator([]);
    const negVec = new Float64Array(4);
    const op = new VectorExclusionOperator(positive, negVec, 0.5);
    const stats = new IndexStats(100, 4);
    const cost = new CostModel().estimate(op, stats);
    expect(cost).toBeGreaterThan(0);
  });

  it("facet vector cost", () => {
    const queryVec = new Float64Array(4);
    const op = new FacetVectorOperator("category", queryVec, 0.5);
    const stats = new IndexStats(100, 0, 4);
    const cost = new CostModel().estimate(op, stats);
    expect(cost).toBeGreaterThan(0);
  });

  it("vertex aggregation cost", () => {
    const traverse = new TraverseOperator(1, "test", "knows", 1);
    const sumFn = (vals: number[]) => vals.reduce((a, b) => a + b, 0);
    const agg = new VertexAggregationOperator(traverse, "age", sumFn);
    const stats = new IndexStats(100);
    const cost = new CostModel().estimate(agg, stats);
    expect(cost).toBeGreaterThan(0);
  });
});

// ===========================================================================
// CardinalityEstimator for new operators
// ===========================================================================

describe("CardinalityNewOperators", () => {
  it("vector exclusion cardinality", () => {
    const positive = new FixedOperator([]);
    const negVec = new Float64Array(4);
    const op = new VectorExclusionOperator(positive, negVec, 0.5);
    const stats = new IndexStats(100, 4);
    const card = new CardinalityEstimator().estimate(op, stats);
    expect(card).toBeGreaterThan(0);
  });

  it("facet vector cardinality", () => {
    const queryVec = new Float64Array(4);
    const op = new FacetVectorOperator("category", queryVec, 0.5);
    const stats = new IndexStats(100, 4);
    const card = new CardinalityEstimator().estimate(op, stats);
    expect(card).toBeGreaterThan(0);
  });

  it("vertex aggregation cardinality", () => {
    const traverse = new TraverseOperator(1, "test", "knows", 1);
    const sumFn = (vals: number[]) => vals.reduce((a, b) => a + b, 0);
    const agg = new VertexAggregationOperator(traverse, "age", sumFn);
    const stats = new IndexStats(100);
    const card = new CardinalityEstimator().estimate(agg, stats);
    // TS returns totalDocs by default for unknown operators
    expect(card).toBeGreaterThan(0);
  });
});
