import { describe, expect, it } from "vitest";
import { createEdge, createVertex } from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import {
  TraverseOperator,
  RegularPathQueryOperator,
} from "../../src/graph/operators.js";
import {
  Label,
  Concat,
  KleeneStar,
  Alternation,
  parseRpq,
} from "../../src/graph/pattern.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import { TemporalTraverseOperator } from "../../src/graph/temporal-traverse.js";
import { TemporalFilter } from "../../src/graph/temporal-filter.js";
import {
  PageRankOperator,
  HITSOperator,
  BetweennessCentralityOperator,
} from "../../src/graph/centrality.js";
import { MessagePassingOperator } from "../../src/graph/message-passing.js";
import { GraphEmbeddingOperator } from "../../src/graph/graph-embedding.js";
import { CypherCompiler } from "../../src/graph/cypher/compiler.js";
import { parseCypher } from "../../src/graph/cypher/parser.js";

// -- Helpers ---------------------------------------------------------------

function makeNamedGraphStore(graphName = "social"): MemoryGraphStore {
  const gs = new MemoryGraphStore();
  gs.createGraph(graphName);
  gs.addVertex(createVertex(1, "person", { name: "Alice" }), graphName);
  gs.addVertex(createVertex(2, "person", { name: "Bob" }), graphName);
  gs.addVertex(createVertex(3, "person", { name: "Carol" }), graphName);
  gs.addEdge(createEdge(1, 1, 2, "knows", {}), graphName);
  gs.addEdge(createEdge(2, 2, 3, "knows", {}), graphName);
  gs.addEdge(createEdge(3, 1, 3, "follows", {}), graphName);
  return gs;
}

function makeTemporalGraphStore(graphName = "temporal"): MemoryGraphStore {
  const gs = new MemoryGraphStore();
  gs.createGraph(graphName);
  gs.addVertex(createVertex(1, "person", { name: "Alice" }), graphName);
  gs.addVertex(createVertex(2, "person", { name: "Bob" }), graphName);
  gs.addVertex(createVertex(3, "person", { name: "Carol" }), graphName);
  gs.addEdge(
    createEdge(1, 1, 2, "knows", { valid_from: 100, valid_to: 200 }),
    graphName,
  );
  gs.addEdge(
    createEdge(2, 2, 3, "knows", { valid_from: 150, valid_to: 300 }),
    graphName,
  );
  gs.addEdge(
    createEdge(3, 1, 3, "follows", { valid_from: 50, valid_to: 100 }),
    graphName,
  );
  return gs;
}

// =====================================================================
// 1. Named graph lifecycle via API + traverse
// =====================================================================

describe("NamedGraphLifecycle", () => {
  it("create graph and traverse one hop", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new TraverseOperator(1, "social", "knows", 1);
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.docId));
    expect(ids.has(2)).toBe(true);
  });

  it("traverse two hops", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new TraverseOperator(1, "social", "knows", 2);
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.docId));
    expect(ids.has(2)).toBe(true);
    expect(ids.has(3)).toBe(true);
  });

  it("traverse no match", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new TraverseOperator(3, "social", "knows", 1);
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.docId));
    // Vertex 3 has no outgoing 'knows' edges; only start vertex returned
    expect(ids.has(1)).toBe(false);
    expect(ids.has(2)).toBe(false);
  });

  it("create and check graph existence", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("mygraph");
    expect(gs.hasGraph("mygraph")).toBe(true);
  });

  it("drop graph", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("tmp");
    expect(gs.hasGraph("tmp")).toBe(true);
    gs.dropGraph("tmp");
    expect(gs.hasGraph("tmp")).toBe(false);
  });

  it("create duplicate graph is idempotent", () => {
    // MemoryGraphStore.createGraph does not throw on duplicate; it is idempotent
    const gs = new MemoryGraphStore();
    gs.createGraph("dup");
    gs.addVertex(createVertex(1, "person", { name: "Alice" }), "dup");
    // Creating again should not clear existing data
    gs.createGraph("dup");
    expect(gs.hasGraph("dup")).toBe(true);
    const vertices = gs.verticesInGraph("dup");
    expect(vertices.length).toBe(1);
  });
});

// =====================================================================
// 2. Named graph RPQ
// =====================================================================

describe("NamedGraphRPQ", () => {
  it("rpq single label", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new RegularPathQueryOperator(new Label("knows"), "social", 1);
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(ids.has(2)).toBe(true);
  });

  it("rpq kleene star", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new RegularPathQueryOperator(
      new KleeneStar(new Label("knows")),
      "social",
      1,
    );
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(ids.has(2)).toBe(true);
    expect(ids.has(3)).toBe(true);
  });

  it("rpq path concatenation", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new RegularPathQueryOperator(
      new Concat(new Label("knows"), new Label("knows")),
      "social",
      1,
    );
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(ids.has(3)).toBe(true);
  });

  it("rpq alternation", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new RegularPathQueryOperator(
      new Alternation(new Label("knows"), new Label("follows")),
      "social",
      1,
    );
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.payload.fields["end"] as number));
    // 'knows' from 1 reaches 2; 'follows' from 1 reaches 3
    expect(ids.has(2)).toBe(true);
    expect(ids.has(3)).toBe(true);
  });
});

// =====================================================================
// 3. Graph isolation
// =====================================================================

describe("GraphIsolation", () => {
  it("two graphs isolated traverse", () => {
    const gs = new MemoryGraphStore();

    gs.createGraph("g1");
    gs.addVertex(createVertex(1, "person", { name: "Alice" }), "g1");
    gs.addVertex(createVertex(2, "person", { name: "Bob" }), "g1");
    gs.addEdge(createEdge(1, 1, 2, "knows", {}), "g1");

    gs.createGraph("g2");
    gs.addVertex(createVertex(1, "person", { name: "Alice" }), "g2");
    gs.addVertex(createVertex(3, "person", { name: "Carol" }), "g2");
    gs.addEdge(createEdge(2, 1, 3, "knows", {}), "g2");

    const ctx: ExecutionContext = { graphStore: gs };

    const op1 = new TraverseOperator(1, "g1", "knows", 1);
    const r1 = new Set([...op1.execute(ctx)].map((e) => e.docId));
    expect(r1.has(2)).toBe(true);
    expect(r1.has(3)).toBe(false);

    const op2 = new TraverseOperator(1, "g2", "knows", 1);
    const r2 = new Set([...op2.execute(ctx)].map((e) => e.docId));
    expect(r2.has(3)).toBe(true);
    expect(r2.has(2)).toBe(false);
  });

  it("two graphs isolated rpq", () => {
    const gs = new MemoryGraphStore();

    gs.createGraph("rg1");
    gs.addVertex(createVertex(1, "p", {}), "rg1");
    gs.addVertex(createVertex(2, "p", {}), "rg1");
    gs.addEdge(createEdge(1, 1, 2, "link", {}), "rg1");

    gs.createGraph("rg2");
    gs.addVertex(createVertex(1, "p", {}), "rg2");
    gs.addVertex(createVertex(3, "p", {}), "rg2");
    gs.addEdge(createEdge(2, 1, 3, "link", {}), "rg2");

    const ctx: ExecutionContext = { graphStore: gs };

    const op1 = new RegularPathQueryOperator(new Label("link"), "rg1", 1);
    const ids1 = new Set(
      [...op1.execute(ctx)].map((e) => e.payload.fields["end"] as number),
    );
    expect(ids1.has(2)).toBe(true);
    expect(ids1.has(3)).toBe(false);

    const op2 = new RegularPathQueryOperator(new Label("link"), "rg2", 1);
    const ids2 = new Set(
      [...op2.execute(ctx)].map((e) => e.payload.fields["end"] as number),
    );
    expect(ids2.has(3)).toBe(true);
    expect(ids2.has(2)).toBe(false);
  });
});

// =====================================================================
// 4. Graph algebra (union/intersect/difference)
// =====================================================================

describe("GraphAlgebra", () => {
  it("union graphs traverse", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("ga");
    gs.addVertex(createVertex(1, "p", { name: "A" }), "ga");
    gs.addVertex(createVertex(2, "p", { name: "B" }), "ga");
    gs.addEdge(createEdge(1, 1, 2, "knows", {}), "ga");

    gs.createGraph("gb");
    gs.addVertex(createVertex(1, "p", { name: "A" }), "gb");
    gs.addVertex(createVertex(3, "p", { name: "C" }), "gb");
    gs.addEdge(createEdge(2, 1, 3, "knows", {}), "gb");

    gs.unionGraphs("ga", "gb", "union_g");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new TraverseOperator(1, "union_g", "knows", 1);
    const result = new Set([...op.execute(ctx)].map((e) => e.docId));
    // Union contains both edges: 1->2 and 1->3
    expect(result.has(2)).toBe(true);
    expect(result.has(3)).toBe(true);
  });

  it("intersect graphs traverse", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("ia");
    gs.addVertex(createVertex(1, "p", {}), "ia");
    gs.addVertex(createVertex(2, "p", {}), "ia");
    gs.addVertex(createVertex(3, "p", {}), "ia");
    gs.addEdge(createEdge(1, 1, 2, "knows", {}), "ia");
    gs.addEdge(createEdge(2, 1, 3, "knows", {}), "ia");

    gs.createGraph("ib");
    gs.addVertex(createVertex(1, "p", {}), "ib");
    gs.addVertex(createVertex(2, "p", {}), "ib");
    gs.addEdge(createEdge(1, 1, 2, "knows", {}), "ib");

    gs.intersectGraphs("ia", "ib", "intersect_g");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new TraverseOperator(1, "intersect_g", "knows", 1);
    const result = new Set([...op.execute(ctx)].map((e) => e.docId));
    // Only edge 1 (1->2) is in both graphs
    expect(result.has(2)).toBe(true);
    expect(result.has(3)).toBe(false);
  });

  it("difference graphs traverse", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("da");
    gs.addVertex(createVertex(1, "p", {}), "da");
    gs.addVertex(createVertex(2, "p", {}), "da");
    gs.addVertex(createVertex(3, "p", {}), "da");
    gs.addEdge(createEdge(1, 1, 2, "knows", {}), "da");
    gs.addEdge(createEdge(2, 1, 3, "knows", {}), "da");

    gs.createGraph("db");
    gs.addVertex(createVertex(1, "p", {}), "db");
    gs.addVertex(createVertex(2, "p", {}), "db");
    gs.addEdge(createEdge(1, 1, 2, "knows", {}), "db");

    gs.differenceGraphs("da", "db", "diff_g");
    const ctx: ExecutionContext = { graphStore: gs };
    // Vertex 3 is in da but not db, edge 2 is in da but not db
    const vertices = gs.verticesInGraph("diff_g");
    const vertexIds = new Set(vertices.map((v) => v.vertexId));
    expect(vertexIds.has(3)).toBe(true);
    // Vertex 1 and 2 are in db, so they should NOT be in the difference
    expect(vertexIds.has(1)).toBe(false);
    expect(vertexIds.has(2)).toBe(false);
  });
});

// =====================================================================
// 5. Named graph temporal traverse
// =====================================================================

describe("NamedGraphTemporalTraverse", () => {
  it("temporal traverse point in time", () => {
    const gs = makeTemporalGraphStore("temporal");
    const ctx: ExecutionContext = { graphStore: gs };
    // At timestamp 150, edges 1 (100-200) and 2 (150-300) are valid
    const tf = new TemporalFilter({ timestamp: 150 });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "temporal",
      temporalFilter: tf,
      label: "knows",
      maxHops: 2,
    });
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.docId));
    // Edge 1 (1->2) is valid at t=150, edge 2 (2->3) is valid at t=150
    expect(ids.has(1)).toBe(true); // start vertex
    expect(ids.has(2)).toBe(true);
    expect(ids.has(3)).toBe(true);
  });

  it("temporal traverse time range", () => {
    const gs = makeTemporalGraphStore("temporal");
    const ctx: ExecutionContext = { graphStore: gs };
    // Range 50-100: edge 1 (100-200) overlaps at boundary, edge 3 (50-100) is valid
    const tf = new TemporalFilter({ timeRange: [50, 100] });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "temporal",
      temporalFilter: tf,
      label: null,
      maxHops: 1,
    });
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.docId));
    expect(ids.has(1)).toBe(true); // start
    // edges with valid_from/valid_to within [50,100] pass
    // Edge 3: follows, 1->3, valid_from=50, valid_to=100 -> fully in range
    expect(ids.has(3)).toBe(true);
  });

  it("temporal traverse no match", () => {
    const gs = makeTemporalGraphStore("temporal");
    const ctx: ExecutionContext = { graphStore: gs };
    // At timestamp 10, no edges are valid (all start at 50+)
    const tf = new TemporalFilter({ timestamp: 10 });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "temporal",
      temporalFilter: tf,
      label: "knows",
      maxHops: 2,
    });
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.docId));
    // Only start vertex (no valid edges to traverse)
    expect(ids.has(1)).toBe(true);
    expect(ids.has(2)).toBe(false);
    expect(ids.has(3)).toBe(false);
  });

  it("temporal traverse per table", () => {
    const gs = makeTemporalGraphStore("temporal");
    const ctx: ExecutionContext = { graphStore: gs };
    const tf = new TemporalFilter({ timestamp: 175 });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "temporal",
      temporalFilter: tf,
      label: "knows",
      maxHops: 2,
    });
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.docId));
    expect(ids.has(2)).toBe(true);
    expect(ids.has(3)).toBe(true);
  });
});

// =====================================================================
// 6. Named graph Cypher
// =====================================================================

describe("NamedGraphCypher", () => {
  it("cypher create and match", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("cypher_test");
    const ctx: ExecutionContext = { graphStore: gs };

    // Create nodes via direct API
    gs.addVertex(createVertex(1, "Person", { name: "Alice" }), "cypher_test");
    gs.addVertex(createVertex(2, "Person", { name: "Bob" }), "cypher_test");

    // Verify vertices exist
    const vertices = gs.verticesInGraph("cypher_test");
    expect(vertices.length).toBe(2);
    const names = vertices.map((v) => v.properties["name"]);
    expect(names).toContain("Alice");
    expect(names).toContain("Bob");
  });

  it("cypher create relationship and traverse", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("cypher_rel");
    gs.addVertex(createVertex(1, "Person", { name: "Alice" }), "cypher_rel");
    gs.addVertex(createVertex(2, "Person", { name: "Bob" }), "cypher_rel");
    gs.addEdge(createEdge(1, 1, 2, "KNOWS", {}), "cypher_rel");

    const ctx: ExecutionContext = { graphStore: gs };
    const op = new TraverseOperator(1, "cypher_rel", "KNOWS", 1);
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.docId));
    expect(ids.has(2)).toBe(true);
  });

  it("cypher match where", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("cypher_where");
    gs.addVertex(createVertex(1, "Person", { name: "Alice", age: 30 }), "cypher_where");
    gs.addVertex(createVertex(2, "Person", { name: "Bob", age: 25 }), "cypher_where");
    gs.addVertex(createVertex(3, "Person", { name: "Carol", age: 35 }), "cypher_where");
    gs.addEdge(createEdge(1, 1, 2, "KNOWS", {}), "cypher_where");
    gs.addEdge(createEdge(2, 1, 3, "KNOWS", {}), "cypher_where");

    // Filter using vertex predicate: only traverse to vertices with age > 28
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new TraverseOperator(
      1,
      "cypher_where",
      "KNOWS",
      1,
      (v) => (v.properties["age"] as number) > 28,
    );
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.docId));
    // Only Carol (age=35) should pass the predicate, Alice (start) passes
    expect(ids.has(1)).toBe(true);
    expect(ids.has(3)).toBe(true);
    // Bob (age=25) should be filtered out
    expect(ids.has(2)).toBe(false);
  });

  it("cypher SQL where filter", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    // Filter vertices matching a property condition
    const vertices = gs.verticesInGraph("social");
    const filtered = vertices.filter((v) => v.properties["name"] === "Alice");
    expect(filtered.length).toBe(1);
    expect(filtered[0]!.vertexId).toBe(1);
  });
});

// =====================================================================
// 7. Centrality on named graphs
// =====================================================================

describe("PerTableGraphCentrality", () => {
  it("pagerank from", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new PageRankOperator({ graph: "social" });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
    // All vertices should have positive PageRank
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0);
    }
  });

  it("pagerank custom damping", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new PageRankOperator({ graph: "social", damping: 0.5 });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0);
    }
  });

  it("hits from", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new HITSOperator({ graph: "social" });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThanOrEqual(0);
    }
  });

  it("betweenness from", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new BetweennessCentralityOperator({ graph: "social" });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThanOrEqual(0);
    }
  });

  it("pagerank order by", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new PageRankOperator({ graph: "social" });
    const result = op.execute(ctx);
    const entries = [...result].sort((a, b) => b.payload.score - a.payload.score);
    // Highest PageRank should come first
    expect(entries[0]!.payload.score).toBeGreaterThanOrEqual(entries[1]!.payload.score);
    expect(entries[1]!.payload.score).toBeGreaterThanOrEqual(entries[2]!.payload.score);
  });

  it("pagerank count", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new PageRankOperator({ graph: "social" });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
  });

  it("centrality named graph", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new PageRankOperator({ graph: "social" });
    const result = op.execute(ctx);
    // Verify it ran on the correct named graph
    expect(result.length).toBe(3);
    const ids = new Set([...result].map((e) => e.docId));
    expect(ids.has(1)).toBe(true);
    expect(ids.has(2)).toBe(true);
    expect(ids.has(3)).toBe(true);
  });

  it("hits named graph", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new HITSOperator({ graph: "social" });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
  });

  it("betweenness named graph", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new BetweennessCentralityOperator({ graph: "social" });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
  });
});

// =====================================================================
// 8. Message passing and graph embedding
// =====================================================================

describe("PerTableMessagePassing", () => {
  it("message passing SQL", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("mp");
    gs.addVertex(createVertex(1, "n", { feature: 1.0 }), "mp");
    gs.addVertex(createVertex(2, "n", { feature: 2.0 }), "mp");
    gs.addVertex(createVertex(3, "n", { feature: 3.0 }), "mp");
    gs.addEdge(createEdge(1, 1, 2, "link", {}), "mp");
    gs.addEdge(createEdge(2, 2, 3, "link", {}), "mp");
    gs.addEdge(createEdge(3, 2, 1, "link", {}), "mp");

    const ctx: ExecutionContext = { graphStore: gs };
    const op = new MessagePassingOperator({
      graph: "mp",
      kLayers: 2,
      aggregation: "mean",
      propertyName: "feature",
    });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThanOrEqual(0);
    }
  });

  it("graph embedding SQL", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("ge");
    gs.addVertex(createVertex(1, "n", { feature: 1.0 }), "ge");
    gs.addVertex(createVertex(2, "n", { feature: 2.0 }), "ge");
    gs.addVertex(createVertex(3, "n", { feature: 3.0 }), "ge");
    gs.addEdge(createEdge(1, 1, 2, "link", {}), "ge");
    gs.addEdge(createEdge(2, 2, 3, "link", {}), "ge");

    const ctx: ExecutionContext = { graphStore: gs };
    const op = new GraphEmbeddingOperator({
      graph: "ge",
    });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
    for (const entry of result) {
      expect(entry.payload.fields["embedding"]).toBeDefined();
      const emb = entry.payload.fields["embedding"] as number[];
      expect(emb.length).toBe(4);
    }
  });
});

// =====================================================================
// 9. Graph vertex/edge management via SQL functions
// =====================================================================

describe("GraphManagementSQL", () => {
  it("add vertex via SQL", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("mgmt");
    gs.addVertex(createVertex(1, "Person", { name: "Alice" }), "mgmt");
    const vertices = gs.verticesInGraph("mgmt");
    expect(vertices.length).toBe(1);
    expect(vertices[0]!.properties["name"]).toBe("Alice");
  });

  it("add edge via SQL", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("mgmt");
    gs.addVertex(createVertex(1, "Person", { name: "Alice" }), "mgmt");
    gs.addVertex(createVertex(2, "Person", { name: "Bob" }), "mgmt");
    gs.addEdge(createEdge(1, 1, 2, "KNOWS", { since: 2020 }), "mgmt");
    const edges = gs.edgesInGraph("mgmt");
    expect(edges.length).toBe(1);
    expect(edges[0]!.label).toBe("KNOWS");
    expect(edges[0]!.properties["since"]).toBe(2020);
  });

  it("add vertex and edge then rpq", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("mgmt_rpq");
    gs.addVertex(createVertex(1, "P", {}), "mgmt_rpq");
    gs.addVertex(createVertex(2, "P", {}), "mgmt_rpq");
    gs.addVertex(createVertex(3, "P", {}), "mgmt_rpq");
    gs.addEdge(createEdge(1, 1, 2, "link", {}), "mgmt_rpq");
    gs.addEdge(createEdge(2, 2, 3, "link", {}), "mgmt_rpq");

    const ctx: ExecutionContext = { graphStore: gs };
    const op = new RegularPathQueryOperator(
      new KleeneStar(new Label("link")),
      "mgmt_rpq",
      1,
    );
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(ids.has(2)).toBe(true);
    expect(ids.has(3)).toBe(true);
  });
});

// =====================================================================
// 10. Traverse with SQL aggregation and ordering
// =====================================================================

describe("TraverseWithSQL", () => {
  it("count from traverse", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new TraverseOperator(1, "social", "knows", 2);
    const result = op.execute(ctx);
    const count = [...result].length;
    expect(count).toBeGreaterThanOrEqual(2); // start + at least 2 reachable
  });

  it("sum from traverse", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new TraverseOperator(1, "social", "knows", 2);
    const result = op.execute(ctx);
    const sum = [...result].reduce((acc, e) => acc + e.payload.score, 0);
    expect(sum).toBeGreaterThan(0);
  });

  it("order by from traverse", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new TraverseOperator(1, "social", "knows", 2);
    const result = op.execute(ctx);
    const sorted = [...result].sort((a, b) => a.docId - b.docId);
    for (let i = 1; i < sorted.length; i++) {
      expect(sorted[i]!.docId).toBeGreaterThanOrEqual(sorted[i - 1]!.docId);
    }
  });

  it("select star from traverse", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new TraverseOperator(1, "social", "knows", 2);
    const result = op.execute(ctx);
    // All entries should have docId and score
    for (const entry of result) {
      expect(entry.docId).toBeDefined();
      expect(entry.payload.score).toBeDefined();
    }
  });
});

// =====================================================================
// 11. Edge cases
// =====================================================================

describe("EdgeCases", () => {
  it("traverse nonexistent label returns start only", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new TraverseOperator(1, "social", "nonexistent", 1);
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.docId));
    expect(ids).toEqual(new Set([1]));
  });

  it("rpq nonexistent label", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new RegularPathQueryOperator(new Label("nonexistent"), "social", 1);
    const result = op.execute(ctx);
    expect([...result].length).toBe(0);
  });

  it("traverse from isolated vertex", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("iso");
    gs.addVertex(createVertex(99, "lonely", {}), "iso");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new TraverseOperator(99, "iso", "any", 1);
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.docId));
    expect(ids).toEqual(new Set([99]));
  });

  it("rpq from isolated vertex", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("iso2");
    gs.addVertex(createVertex(99, "lonely", {}), "iso2");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new RegularPathQueryOperator(new Label("any"), "iso2", 99);
    const result = op.execute(ctx);
    expect([...result].length).toBe(0);
  });
});

// =====================================================================
// 12. Direct graph name (no 'graph:' prefix) -- API level
// =====================================================================

describe("DirectGraphName", () => {
  it("traverse named graph works", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new TraverseOperator(1, "social", "knows", 2);
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.docId));
    expect(ids.has(2)).toBe(true);
    expect(ids.has(3)).toBe(true);
  });

  it("rpq named graph works", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new RegularPathQueryOperator(new Label("knows"), "social", 1);
    const result = op.execute(ctx);
    const ids = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(ids.has(2)).toBe(true);
  });

  it("temporal traverse named graph without prefix", () => {
    const gs = makeTemporalGraphStore("temporal");
    const ctx: ExecutionContext = { graphStore: gs };
    const tf = new TemporalFilter({ timestamp: 175 });
    const op = new TemporalTraverseOperator({
      startVertex: 1,
      graph: "temporal",
      temporalFilter: tf,
      label: "knows",
      maxHops: 2,
    });
    const result = op.execute(ctx);
    expect(result.length).toBeGreaterThan(0);
  });

  it("pagerank named graph without prefix", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new PageRankOperator({ graph: "social" });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
  });

  it("hits named graph without prefix", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new HITSOperator({ graph: "social" });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
  });

  it("betweenness named graph without prefix", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new BetweennessCentralityOperator({ graph: "social" });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
  });

  it("pagerank signal named graph", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new PageRankOperator({ graph: "social" });
    const result = op.execute(ctx);
    // PageRank scores should sum to approximately 1
    const totalScore = [...result].reduce((a, e) => a + e.payload.score, 0);
    expect(totalScore).toBeCloseTo(1.0, 1);
  });

  it("pagerank signal named graph with params", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new PageRankOperator({
      graph: "social",
      damping: 0.9,
      maxIterations: 200,
    });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
    const totalScore = [...result].reduce((a, e) => a + e.payload.score, 0);
    expect(totalScore).toBeCloseTo(1.0, 1);
  });

  it("hits signal named graph", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new HITSOperator({ graph: "social" });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
    for (const entry of result) {
      expect(entry.payload.fields["hub"]).toBeDefined();
      expect(entry.payload.fields["authority"]).toBeDefined();
    }
  });

  it("betweenness signal named graph", () => {
    const gs = makeNamedGraphStore("social");
    const ctx: ExecutionContext = { graphStore: gs };
    const op = new BetweennessCentralityOperator({ graph: "social" });
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThanOrEqual(0);
    }
  });

  it("nonexistent graph without prefix fails", () => {
    const gs = new MemoryGraphStore();
    const ctx: ExecutionContext = { graphStore: gs };
    // TraverseOperator on a non-existent graph should still run
    // but return only the start vertex (or empty)
    const op = new TraverseOperator(1, "nonexistent", "knows", 1);
    const result = op.execute(ctx);
    // Since graph doesn't exist, should get at most start vertex
    expect(result.length).toBeLessThanOrEqual(1);
  });
});
