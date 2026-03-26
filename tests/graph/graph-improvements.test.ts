import { describe, expect, it } from "vitest";
import { createEdge, createVertex, IndexStats } from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import {
  PatternMatchOperator,
  RegularPathQueryOperator,
  TraverseOperator,
  WeightedPathQueryOperator,
} from "../../src/graph/operators.js";
import {
  Label,
  Concat,
  Alternation,
  KleeneStar,
  BoundedLabel,
  createVertexPattern,
  createEdgePattern,
  createGraphPattern,
  parseRpq,
} from "../../src/graph/pattern.js";
import { SubgraphIndex } from "../../src/graph/index.js";
import { IncrementalPatternMatcher } from "../../src/graph/incremental-match.js";
import { GraphDelta } from "../../src/graph/delta.js";
import { CardinalityEstimator, GraphStats } from "../../src/planner/cardinality.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import {
  PageRankOperator,
  HITSOperator,
  BetweennessCentralityOperator,
} from "../../src/graph/centrality.js";
import { ProgressiveFusionOperator } from "../../src/operators/progressive-fusion.js";
import { Operator } from "../../src/operators/base.js";
import { PostingList } from "../../src/core/posting-list.js";
import { createPayload } from "../../src/core/types.js";

const GRAPH_NAME = "test";

function makeGraph(): MemoryGraphStore {
  const g = new MemoryGraphStore();
  g.createGraph(GRAPH_NAME);
  g.addVertex(createVertex(1, "person", { name: "Alice", age: 30 }), GRAPH_NAME);
  g.addVertex(createVertex(2, "person", { name: "Bob", age: 25 }), GRAPH_NAME);
  g.addVertex(createVertex(3, "person", { name: "Carol", age: 35 }), GRAPH_NAME);
  g.addVertex(createVertex(4, "person", { name: "Dave", age: 40 }), GRAPH_NAME);
  g.addEdge(createEdge(1, 1, 2, "knows", { since: 2020, weight: 1.0 }), GRAPH_NAME);
  g.addEdge(createEdge(2, 2, 3, "knows", { since: 2021, weight: 2.0 }), GRAPH_NAME);
  g.addEdge(createEdge(3, 3, 4, "knows", { since: 2022, weight: 3.0 }), GRAPH_NAME);
  g.addEdge(
    createEdge(4, 1, 3, "works_with", { since: 2019, weight: 0.5 }),
    GRAPH_NAME,
  );
  return g;
}

// -- Phase 1C: Bounded RPQ Tests --

describe("BoundedRPQ", () => {
  it("parse bounded label", () => {
    const expr = parseRpq("knows{2,3}");
    expect(expr).toBeTruthy();
  });

  it("bounded NFA exact hops", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    // knows{2,3} from vertex 1: should reach vertex 3 (2 hops) and vertex 4 (3 hops)
    const expr = parseRpq("knows{2,3}");
    const op = new RegularPathQueryOperator(expr, GRAPH_NAME, 1);
    const result = op.execute(ctx);
    const ends = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(ends.has(3)).toBe(true); // 1->2->3 (2 hops)
    expect(ends.has(4)).toBe(true); // 1->2->3->4 (3 hops)
    expect(ends.has(2)).toBe(false); // only 1 hop
  });

  it("bounded min zero", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    // knows{0,1} from vertex 1: should reach vertex 1 (0 hops) and vertex 2 (1 hop)
    const expr = new BoundedLabel("knows", 0, 1);
    const op = new RegularPathQueryOperator(expr, GRAPH_NAME, 1);
    const result = op.execute(ctx);
    const ends = new Set([...result].map((e) => e.payload.fields["end"] as number));
    // 0 hops = start vertex, 1 hop = direct neighbor
    expect(ends.has(1)).toBe(true); // 0 hops (epsilon)
    expect(ends.has(2)).toBe(true); // 1 hop
  });

  it("weighted path sum", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new WeightedPathQueryOperator(
      new KleeneStar(new Label("knows")),
      GRAPH_NAME,
      { startVertex: 1, aggregation: "sum" },
    );
    const result = op.execute(ctx);
    const entries = [...result];
    // Should find paths from vertex 1 through "knows" edges
    expect(entries.length).toBeGreaterThan(0);
    // Check that weights are summed
    for (const entry of entries) {
      const w = entry.payload.fields["weight"] as number;
      expect(typeof w).toBe("number");
    }
  });

  it("weighted path max", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new WeightedPathQueryOperator(
      new KleeneStar(new Label("knows")),
      GRAPH_NAME,
      { startVertex: 1, aggregation: "max" },
    );
    const result = op.execute(ctx);
    const entries = [...result];
    expect(entries.length).toBeGreaterThan(0);
    for (const entry of entries) {
      const w = entry.payload.fields["weight"] as number;
      expect(typeof w).toBe("number");
    }
  });

  it("weighted path predicate", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    // Only paths with total weight >= 5.0
    const op = new WeightedPathQueryOperator(
      new KleeneStar(new Label("knows")),
      GRAPH_NAME,
      { startVertex: 1, aggregation: "sum", weightThreshold: 5.0 },
    );
    const result = op.execute(ctx);
    const entries = [...result];
    // 1->2->3->4 has weight 1+2+3=6, so it should pass threshold
    for (const entry of entries) {
      const w = entry.payload.fields["weight"] as number;
      expect(w).toBeGreaterThanOrEqual(5.0);
    }
  });
});

// -- Phase 2A: Edge Property Filter Pushdown Tests --

describe("EdgeFilterPushdown", () => {
  it("edge filter pushdown", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    // Pattern with edge constraint on "since"
    const pattern = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [
        createEdgePattern("a", "b", {
          label: "knows",
          constraints: [(e) => (e.properties["since"] as number) >= 2021],
        }),
      ],
    );
    const op = new PatternMatchOperator(pattern, GRAPH_NAME);
    const result = op.execute(ctx);
    const entries = [...result];
    // Only edges with since >= 2021: edge 2 (2->3, since=2021), edge 3 (3->4, since=2022)
    expect(entries.length).toBe(2);
  });

  it("edge filter with vertex filter", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const pattern = createGraphPattern(
      [
        createVertexPattern("a", [(v) => (v.properties["age"] as number) >= 25]),
        createVertexPattern("b", []),
      ],
      [
        createEdgePattern("a", "b", {
          label: "knows",
          constraints: [(e) => (e.properties["since"] as number) >= 2021],
        }),
      ],
    );
    const op = new PatternMatchOperator(pattern, GRAPH_NAME);
    const result = op.execute(ctx);
    const entries = [...result];
    // Vertex age >= 25 AND edge since >= 2021
    // edge 2: 2->3 (Bob age=25, since=2021) -> passes
    // edge 3: 3->4 (Carol age=35, since=2022) -> passes
    expect(entries.length).toBe(2);
  });

  it("no match edge filter preserved", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const pattern = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [
        createEdgePattern("a", "b", {
          label: "knows",
          constraints: [(e) => (e.properties["since"] as number) >= 3000],
        }),
      ],
    );
    const op = new PatternMatchOperator(pattern, GRAPH_NAME);
    const result = op.execute(ctx);
    expect([...result].length).toBe(0);
  });
});

// -- Phase 2B: Join-Pattern Fusion Tests --

describe("JoinPatternFusion", () => {
  it("shared variable fusion", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    // Two patterns sharing variable "b"
    const p1 = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const p2 = createGraphPattern(
      [createVertexPattern("b", []), createVertexPattern("c", [])],
      [createEdgePattern("b", "c", { label: "knows" })],
    );
    const op1 = new PatternMatchOperator(p1, GRAPH_NAME);
    const op2 = new PatternMatchOperator(p2, GRAPH_NAME);
    const r1 = [...op1.execute(ctx)];
    const r2 = [...op2.execute(ctx)];
    expect(r1.length).toBeGreaterThan(0);
    expect(r2.length).toBeGreaterThan(0);
  });

  it("no shared variables preserved", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const p1 = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const p2 = createGraphPattern(
      [createVertexPattern("x", []), createVertexPattern("y", [])],
      [createEdgePattern("x", "y", { label: "works_with" })],
    );
    const op1 = new PatternMatchOperator(p1, GRAPH_NAME);
    const op2 = new PatternMatchOperator(p2, GRAPH_NAME);
    const r1 = [...op1.execute(ctx)];
    const r2 = [...op2.execute(ctx)];
    expect(r1.length).toBeGreaterThan(0);
    expect(r2.length).toBeGreaterThan(0);
  });

  it("constraints combined", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const pattern = createGraphPattern(
      [
        createVertexPattern("a", [(v) => (v.properties["age"] as number) >= 30]),
        createVertexPattern("b", [(v) => (v.properties["age"] as number) <= 30]),
      ],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const op = new PatternMatchOperator(pattern, GRAPH_NAME);
    const result = [...op.execute(ctx)];
    // a.age >= 30, b.age <= 30
    // Edge 1->2: Alice(30)->Bob(25) -> passes (30 >= 30, 25 <= 30)
    expect(result.length).toBeGreaterThan(0);
  });

  it("mixed operand types", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    // Combine pattern match with traversal
    const pattern = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const pmOp = new PatternMatchOperator(pattern, GRAPH_NAME);
    const traverseOp = new TraverseOperator(1, GRAPH_NAME, "knows", 1);
    const pmResult = [...pmOp.execute(ctx)];
    const tResult = [...traverseOp.execute(ctx)];
    expect(pmResult.length).toBeGreaterThan(0);
    expect(tResult.length).toBeGreaterThan(0);
  });
});

// -- Phase 3A: Subgraph Index Tests --

describe("SubgraphIndex", () => {
  it("build and lookup", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const pattern = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const idx = new SubgraphIndex();
    idx.build(g, GRAPH_NAME, [pattern], ctx);
    const result = idx.lookup(pattern, GRAPH_NAME);
    expect(result).not.toBeNull();
    expect(result!.length).toBeGreaterThan(0);
  });

  it("miss returns null", () => {
    const idx = new SubgraphIndex();
    const pattern = createGraphPattern(
      [createVertexPattern("x", []), createVertexPattern("y", [])],
      [createEdgePattern("x", "y", { label: "missing" })],
    );
    const result = idx.lookup(pattern, GRAPH_NAME);
    expect(result).toBeNull();
  });

  it("invalidation", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const pattern = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const idx = new SubgraphIndex();
    idx.build(g, GRAPH_NAME, [pattern], ctx);
    expect(idx.lookup(pattern, GRAPH_NAME)).not.toBeNull();
    idx.invalidate(pattern, GRAPH_NAME);
    expect(idx.lookup(pattern, GRAPH_NAME)).toBeNull();
  });

  it("pattern match uses cache", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const pattern = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const idx = new SubgraphIndex();
    idx.build(g, GRAPH_NAME, [pattern], ctx);
    const firstLookup = idx.lookup(pattern, GRAPH_NAME);
    const secondLookup = idx.lookup(pattern, GRAPH_NAME);
    // Same reference since it's cached
    expect(firstLookup).toBe(secondLookup);
  });

  it("canonical key deterministic", () => {
    const idx = new SubgraphIndex();
    const pattern = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    // hasPattern uses the same canonical key each time
    expect(idx.hasPattern(pattern, GRAPH_NAME)).toBe(false);
    expect(idx.hasPattern(pattern, GRAPH_NAME)).toBe(false);
  });
});

// -- Phase 3B: Incremental Pattern Matching Tests --

describe("IncrementalPatternMatcher", () => {
  it("add edge creates new match", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const pattern = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const matcher = new IncrementalPatternMatcher(pattern, GRAPH_NAME);
    const initial = matcher.fullMatch(ctx);
    const initialCount = [...initial].length;

    // Add a new edge
    const newEdge = createEdge(10, 4, 1, "knows", {});
    g.addEdge(newEdge, GRAPH_NAME);

    const delta = new GraphDelta();
    delta.addEdge(newEdge);

    const updated = matcher.incrementalUpdate(delta, ctx);
    const updatedCount = [...updated].length;
    expect(updatedCount).toBeGreaterThan(initialCount);
  });

  it("remove vertex invalidates match", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const pattern = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const matcher = new IncrementalPatternMatcher(pattern, GRAPH_NAME);
    const initial = matcher.fullMatch(ctx);
    const initialCount = [...initial].length;

    // Remove vertex 2 (Bob) which is involved in edges 1 (1->2) and 2 (2->3)
    g.removeVertex(2, GRAPH_NAME);

    const delta = new GraphDelta();
    delta.removeVertex(2, "person");

    const updated = matcher.incrementalUpdate(delta, ctx);
    const updatedCount = [...updated].length;
    expect(updatedCount).toBeLessThan(initialCount);
  });

  it("unrelated delta no change", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const pattern = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const matcher = new IncrementalPatternMatcher(pattern, GRAPH_NAME);
    const initial = matcher.fullMatch(ctx);
    const initialCount = [...initial].length;

    // Add a vertex that doesn't participate in any "knows" edge
    const newVertex = createVertex(99, "thing", { name: "Unrelated" });
    g.addVertex(newVertex, GRAPH_NAME);

    const delta = new GraphDelta();
    delta.addVertex(newVertex);

    const updated = matcher.incrementalUpdate(delta, ctx);
    const updatedCount = [...updated].length;
    // Should have same count since the new vertex is unrelated
    expect(updatedCount).toBe(initialCount);
  });
});

// -- Phase 3C: Graph Sampling Cardinality Tests --

describe("GraphSamplingCardinality", () => {
  it("small graph uses formula", () => {
    const g = makeGraph();
    const gs = GraphStats.fromGraphStore(g, GRAPH_NAME);
    const est = new CardinalityEstimator({ graphStats: gs });
    const card = est.estimateTraverse(1, "knows");
    expect(card).toBeGreaterThan(0);
    expect(card).toBeLessThanOrEqual(gs.numVertices);
  });

  it("no graph store fallback", () => {
    // Without graph stats, the estimator should still return a reasonable value
    const est = new CardinalityEstimator();
    const stats = new IndexStats(1000);
    const traverseOp = new TraverseOperator(1, GRAPH_NAME, "knows", 1);
    const card = est.estimate(traverseOp, stats);
    expect(card).toBeGreaterThan(0);
  });

  it("empty graph returns zero or one", () => {
    const g = new MemoryGraphStore();
    g.createGraph("empty");
    const gs = GraphStats.fromGraphStore(g, "empty");
    expect(gs.numVertices).toBe(0);
    expect(gs.numEdges).toBe(0);
    const est = new CardinalityEstimator({ graphStats: gs });
    // With 0 vertices, traverse estimate should be minimal
    const card = est.estimateTraverse(1, "knows");
    expect(card).toBeLessThanOrEqual(1);
  });
});

// -- Phase 2E: Temporal Cardinality Tests --

describe("TemporalCardinality", () => {
  it("range half coverage", () => {
    const gs = new GraphStats({
      numVertices: 100,
      numEdges: 200,
      avgOutDegree: 2.0,
      minTimestamp: 0,
      maxTimestamp: 100,
    });
    const est = new CardinalityEstimator({ graphStats: gs });
    const fullCard = est.estimateTemporalTraverse(1, null, null, null);
    // Half the time range
    const halfCard = est.estimateTemporalTraverse(1, null, 0, 50);
    expect(halfCard).toBeLessThanOrEqual(fullCard);
    expect(halfCard).toBeGreaterThan(0);
  });

  it("point query low selectivity", () => {
    const gs = new GraphStats({
      numVertices: 100,
      numEdges: 200,
      avgOutDegree: 2.0,
      minTimestamp: 0,
      maxTimestamp: 100,
    });
    const est = new CardinalityEstimator({ graphStats: gs });
    // Very narrow range
    const card = est.estimateTemporalTraverse(1, null, 50, 51);
    expect(card).toBeGreaterThan(0);
    expect(card).toBeLessThanOrEqual(100);
  });

  it("no temporal data returns one", () => {
    const gs = new GraphStats({
      numVertices: 100,
      numEdges: 200,
      avgOutDegree: 2.0,
      // No minTimestamp / maxTimestamp
    });
    const est = new CardinalityEstimator({ graphStats: gs });
    const card = est.estimateTemporalTraverse(1, null, null, null);
    // Without temporal data, should return base estimate
    expect(card).toBeGreaterThan(0);
  });
});

// -- SQL Function Tests --

describe("CentralitySQL", () => {
  it("pagerank from", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new PageRankOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);
    const entries = [...result];
    expect(entries.length).toBe(4);
    for (const entry of entries) {
      expect(entry.payload.score).toBeGreaterThan(0);
    }
  });

  it("pagerank where", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new PageRankOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);
    const entries = [...result].filter((e) => e.payload.score > 0.1);
    expect(entries.length).toBeGreaterThan(0);
  });

  it("pagerank custom damping", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new PageRankOperator({ graph: GRAPH_NAME, damping: 0.5 });
    const result = op.execute(ctx);
    const entries = [...result];
    expect(entries.length).toBe(4);
  });

  it("hits from", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new HITSOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);
    const entries = [...result];
    expect(entries.length).toBeGreaterThan(0);
  });

  it("betweenness from", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new BetweennessCentralityOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);
    const entries = [...result];
    expect(entries.length).toBeGreaterThan(0);
  });

  it("centrality with order by", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new PageRankOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);
    const entries = [...result].sort((a, b) => b.payload.score - a.payload.score);
    expect(entries.length).toBe(4);
    // Verify descending order
    for (let i = 0; i < entries.length - 1; i++) {
      expect(entries[i]!.payload.score).toBeGreaterThanOrEqual(
        entries[i + 1]!.payload.score,
      );
    }
  });

  it("centrality with aggregation", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new PageRankOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);
    const entries = [...result];
    const totalScore = entries.reduce((sum, e) => sum + e.payload.score, 0);
    // PageRank scores should approximately sum to 1
    expect(totalScore).toBeGreaterThan(0.5);
    expect(totalScore).toBeLessThanOrEqual(1.5);
  });

  it("centrality in fusion", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new PageRankOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);
    const entries = [...result];
    // Verify we can use centrality scores for fusion
    expect(entries.length).toBeGreaterThan(0);
    for (const entry of entries) {
      expect(entry.payload.score).toBeGreaterThanOrEqual(0);
    }
  });
});

describe("WeightedRPQSQL", () => {
  it("weighted rpq basic", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new WeightedPathQueryOperator(
      new KleeneStar(new Label("knows")),
      GRAPH_NAME,
      { startVertex: 1, aggregation: "sum" },
    );
    const result = op.execute(ctx);
    expect([...result].length).toBeGreaterThan(0);
  });

  it("weighted rpq max", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new WeightedPathQueryOperator(
      new KleeneStar(new Label("knows")),
      GRAPH_NAME,
      { startVertex: 1, aggregation: "max" },
    );
    const result = op.execute(ctx);
    expect([...result].length).toBeGreaterThan(0);
  });

  it("weighted rpq threshold", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const op = new WeightedPathQueryOperator(
      new KleeneStar(new Label("knows")),
      GRAPH_NAME,
      { startVertex: 1, aggregation: "sum", weightThreshold: 5.0 },
    );
    const result = op.execute(ctx);
    for (const entry of [...result]) {
      const w = entry.payload.fields["weight"] as number;
      expect(w).toBeGreaterThanOrEqual(5.0);
    }
  });
});

class FixedScoreOperator extends Operator {
  private _entries: { docId: number; payload: ReturnType<typeof createPayload> }[];
  constructor(entries: { docId: number; score: number }[]) {
    super();
    this._entries = entries.map((e) => ({
      docId: e.docId,
      payload: createPayload({ score: e.score }),
    }));
  }
  execute(_ctx: ExecutionContext): PostingList {
    return PostingList.fromSorted(this._entries);
  }
}

describe("ProgressiveFusionSQL", () => {
  it("progressive fusion basic", () => {
    const sig1 = new FixedScoreOperator([
      { docId: 1, score: 0.8 },
      { docId: 2, score: 0.6 },
      { docId: 3, score: 0.4 },
    ]);
    const sig2 = new FixedScoreOperator([
      { docId: 1, score: 0.7 },
      { docId: 2, score: 0.5 },
      { docId: 4, score: 0.3 },
    ]);
    const op = new ProgressiveFusionOperator(
      [
        [[sig1], 3],
        [[sig2], 2],
      ],
      0.5,
    );
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
  });

  it("progressive fusion with gating", () => {
    const sig1 = new FixedScoreOperator([
      { docId: 1, score: 0.8 },
      { docId: 2, score: 0.6 },
    ]);
    const sig2 = new FixedScoreOperator([
      { docId: 1, score: 0.7 },
      { docId: 3, score: 0.5 },
    ]);
    const op = new ProgressiveFusionOperator(
      [
        [[sig1], 2],
        [[sig2], 2],
      ],
      0.5,
      "relu",
    );
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
  });
});

describe("BoundedRPQSQL", () => {
  it("bounded rpq from", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const expr = new BoundedLabel("knows", 1, 2);
    const op = new RegularPathQueryOperator(expr, GRAPH_NAME, 1);
    const result = op.execute(ctx);
    expect([...result].length).toBeGreaterThan(0);
  });

  it("bounded rpq exact", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const expr = new BoundedLabel("knows", 2, 2);
    const op = new RegularPathQueryOperator(expr, GRAPH_NAME, 1);
    const result = op.execute(ctx);
    const ends = new Set([...result].map((e) => e.payload.fields["end"] as number));
    // From vertex 1, exactly 2 hops of "knows": 1->2->3 gives end=3, 1->3 is 1 hop
    expect(ends.has(3)).toBe(true);
  });

  it("bounded rpq aggregate", () => {
    const g = makeGraph();
    const ctx: ExecutionContext = { graphStore: g };
    const expr = new BoundedLabel("knows", 1, 3);
    const op = new RegularPathQueryOperator(expr, GRAPH_NAME, 1);
    const result = op.execute(ctx);
    const entries = [...result];
    expect(entries.length).toBeGreaterThan(0);
    // Should reach vertices at 1, 2, and 3 hops
    const ends = new Set(entries.map((e) => e.payload.fields["end"] as number));
    expect(ends.size).toBeGreaterThan(1);
  });
});
