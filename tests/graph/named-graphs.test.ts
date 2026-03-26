import { describe, expect, it } from "vitest";
import { createEdge, createVertex } from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import {
  PatternMatchOperator,
  RegularPathQueryOperator,
  TraverseOperator,
} from "../../src/graph/operators.js";
import {
  Label,
  createGraphPattern,
  createVertexPattern,
  createEdgePattern,
} from "../../src/graph/pattern.js";
import { createGraphPayload } from "../../src/graph/posting-list.js";

// -- Helpers ------------------------------------------------------------------

function makeGraphStore(): MemoryGraphStore {
  const gs = new MemoryGraphStore();
  gs.createGraph("g1");
  gs.createGraph("g2");
  return gs;
}

function addTriangle(gs: MemoryGraphStore, graph: string, startVid: number = 1): void {
  const a = startVid;
  const b = startVid + 1;
  const c = startVid + 2;
  gs.addVertex(createVertex(a, "person", { name: "Alice" }), graph);
  gs.addVertex(createVertex(b, "person", { name: "Bob" }), graph);
  gs.addVertex(createVertex(c, "person", { name: "Carol" }), graph);
  gs.addEdge(createEdge(a * 10, a, b, "knows"), graph);
  gs.addEdge(createEdge(a * 10 + 1, b, c, "knows"), graph);
  gs.addEdge(createEdge(a * 10 + 2, c, a, "knows"), graph);
}

// =============================================================================
// Graph lifecycle tests
// =============================================================================

describe("Graph lifecycle", () => {
  it("create graph", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("test");
    expect(gs.hasGraph("test")).toBe(true);
    expect(gs.graphNames()).toContain("test");
  });

  it("drop graph", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("test");
    gs.addVertex(createVertex(1, "node", {}), "test");
    gs.dropGraph("test");
    expect(gs.hasGraph("test")).toBe(false);
    expect(gs.graphNames()).not.toContain("test");
  });

  it("graph names sorted", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("zebra");
    gs.createGraph("apple");
    gs.createGraph("mango");
    const names = gs.graphNames().sort();
    expect(names).toEqual(["apple", "mango", "zebra"]);
  });
});

// =============================================================================
// Graph isolation tests
// =============================================================================

describe("Graph isolation", () => {
  it("vertices are isolated", () => {
    const gs = makeGraphStore();
    gs.addVertex(createVertex(1, "person", { name: "Alice" }), "g1");
    gs.addVertex(createVertex(2, "person", { name: "Bob" }), "g2");

    expect(gs.vertexIdsInGraph("g1")).toEqual(new Set([1]));
    expect(gs.vertexIdsInGraph("g2")).toEqual(new Set([2]));
  });

  it("edges are isolated", () => {
    const gs = makeGraphStore();
    gs.addVertex(createVertex(1, "person", {}), "g1");
    gs.addVertex(createVertex(2, "person", {}), "g1");
    gs.addVertex(createVertex(3, "person", {}), "g2");
    gs.addVertex(createVertex(4, "person", {}), "g2");
    gs.addEdge(createEdge(10, 1, 2, "knows"), "g1");
    gs.addEdge(createEdge(20, 3, 4, "knows"), "g2");

    expect(gs.outEdgeIds(1, "g1")).toEqual(new Set([10]));
    expect(gs.outEdgeIds(1, "g2")).toEqual(new Set());
    expect(gs.outEdgeIds(3, "g2")).toEqual(new Set([20]));
    expect(gs.outEdgeIds(3, "g1")).toEqual(new Set());
  });

  it("neighbors are isolated", () => {
    const gs = makeGraphStore();
    gs.addVertex(createVertex(1, "person", {}), "g1");
    gs.addVertex(createVertex(2, "person", {}), "g1");
    gs.addVertex(createVertex(1, "person", {}), "g2");
    gs.addVertex(createVertex(3, "person", {}), "g2");
    gs.addEdge(createEdge(10, 1, 2, "knows"), "g1");
    gs.addEdge(createEdge(20, 1, 3, "knows"), "g2");

    expect(gs.neighbors(1, "g1")).toEqual([2]);
    expect(gs.neighbors(1, "g2")).toEqual([3]);
  });

  it("traversal is isolated", () => {
    const gs = makeGraphStore();
    addTriangle(gs, "g1", 1);
    gs.addVertex(createVertex(10, "person", {}), "g2");
    gs.addVertex(createVertex(11, "person", {}), "g2");
    gs.addEdge(createEdge(100, 10, 11, "knows"), "g2");

    const ctx = { graphStore: gs };
    const op = new TraverseOperator(1, "g1", "knows", 2);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.docId));
    // Should find vertices 1, 2, 3 (triangle in g1)
    expect(docIds).toEqual(new Set([1, 2, 3]));
  });
});

// =============================================================================
// Cross-graph vertex sharing
// =============================================================================

describe("Cross-graph vertex sharing", () => {
  it("shared vertex in multiple graphs", () => {
    const gs = makeGraphStore();
    const v = createVertex(1, "shared", { name: "shared" });
    gs.addVertex(v, "g1");
    gs.addVertex(v, "g2");

    expect(gs.vertexIdsInGraph("g1").has(1)).toBe(true);
    expect(gs.vertexIdsInGraph("g2").has(1)).toBe(true);
    expect(gs.vertexGraphs(1)).toEqual(new Set(["g1", "g2"]));
  });

  it("drop graph preserves shared vertex", () => {
    const gs = makeGraphStore();
    const v = createVertex(1, "shared", {});
    gs.addVertex(v, "g1");
    gs.addVertex(v, "g2");
    gs.dropGraph("g1");

    // Vertex should still exist in g2
    expect(gs.getVertex(1)).not.toBeNull();
    expect(gs.vertexIdsInGraph("g2").has(1)).toBe(true);
  });

  it("drop graph removes unshared vertex from graph", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g1");
    gs.addVertex(createVertex(1, "node", {}), "g1");
    gs.dropGraph("g1");

    // After dropping, vertex is no longer in any graph's partition
    expect(gs.hasGraph("g1")).toBe(false);
    // The vertex membership is cleared
    expect(gs.vertexGraphs(1).size).toBe(0);
  });
});

// =============================================================================
// Graph algebra tests
// =============================================================================

describe("Graph algebra", () => {
  it("union graphs", () => {
    const gs = makeGraphStore();
    gs.addVertex(createVertex(1, "a", {}), "g1");
    gs.addVertex(createVertex(2, "b", {}), "g1");
    gs.addVertex(createVertex(2, "b", {}), "g2");
    gs.addVertex(createVertex(3, "c", {}), "g2");

    gs.unionGraphs("g1", "g2", "union");
    expect(gs.vertexIdsInGraph("union")).toEqual(new Set([1, 2, 3]));
  });

  it("intersect graphs", () => {
    const gs = makeGraphStore();
    gs.addVertex(createVertex(1, "a", {}), "g1");
    gs.addVertex(createVertex(2, "b", {}), "g1");
    gs.addVertex(createVertex(2, "b", {}), "g2");
    gs.addVertex(createVertex(3, "c", {}), "g2");

    gs.intersectGraphs("g1", "g2", "inter");
    expect(gs.vertexIdsInGraph("inter")).toEqual(new Set([2]));
  });

  it("difference graphs", () => {
    const gs = makeGraphStore();
    gs.addVertex(createVertex(1, "a", {}), "g1");
    gs.addVertex(createVertex(2, "b", {}), "g1");
    gs.addVertex(createVertex(2, "b", {}), "g2");
    gs.addVertex(createVertex(3, "c", {}), "g2");

    gs.differenceGraphs("g1", "g2", "diff");
    expect(gs.vertexIdsInGraph("diff")).toEqual(new Set([1]));
  });

  it("copy graph", () => {
    const gs = makeGraphStore();
    addTriangle(gs, "g1");
    gs.copyGraph("g1", "copy");

    expect(gs.vertexIdsInGraph("copy")).toEqual(gs.vertexIdsInGraph("g1"));
    expect(gs.edgesInGraph("copy").length).toBe(gs.edgesInGraph("g1").length);
  });

  it("union graphs with edges", () => {
    const gs = makeGraphStore();
    gs.addVertex(createVertex(1, "a", {}), "g1");
    gs.addVertex(createVertex(2, "b", {}), "g1");
    gs.addEdge(createEdge(10, 1, 2, "e1"), "g1");

    gs.addVertex(createVertex(2, "b", {}), "g2");
    gs.addVertex(createVertex(3, "c", {}), "g2");
    gs.addEdge(createEdge(20, 2, 3, "e2"), "g2");

    gs.unionGraphs("g1", "g2", "union");
    const edges = gs.edgesInGraph("union");
    const edgeIds = new Set(edges.map((e) => e.edgeId));
    expect(edgeIds).toEqual(new Set([10, 20]));
  });
});

// =============================================================================
// Scoped traversal / pattern match / RPQ
// =============================================================================

describe("Scoped operations", () => {
  it("scoped traverse", () => {
    const gs = makeGraphStore();
    gs.addVertex(createVertex(1, "a", {}), "g1");
    gs.addVertex(createVertex(2, "b", {}), "g1");
    gs.addEdge(createEdge(10, 1, 2, "link"), "g1");

    const ctx = { graphStore: gs };
    const op = new TraverseOperator(1, "g1", "link", 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.docId));
    expect(docIds.has(2)).toBe(true);
  });

  it("scoped pattern match", () => {
    const gs = makeGraphStore();
    gs.addVertex(createVertex(1, "person", { name: "Alice" }), "g1");
    gs.addVertex(createVertex(2, "person", { name: "Bob" }), "g1");
    gs.addEdge(createEdge(10, 1, 2, "knows"), "g1");

    const pattern = createGraphPattern(
      [createVertexPattern("a"), createVertexPattern("b")],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const ctx = { graphStore: gs };
    const op = new PatternMatchOperator(pattern, "g1");
    const result = op.execute(ctx);
    expect(result.length).toBe(1);
    expect(result.entries[0]!.payload.fields["a"]).toBe(1);
    expect(result.entries[0]!.payload.fields["b"]).toBe(2);
  });

  it("scoped RPQ", () => {
    const gs = makeGraphStore();
    gs.addVertex(createVertex(1, "a", {}), "g1");
    gs.addVertex(createVertex(2, "b", {}), "g1");
    gs.addVertex(createVertex(3, "c", {}), "g1");
    gs.addEdge(createEdge(10, 1, 2, "knows"), "g1");
    gs.addEdge(createEdge(11, 2, 3, "knows"), "g1");

    const ctx = { graphStore: gs };
    const op = new RegularPathQueryOperator(new Label("knows"), "g1", 1);
    const result = op.execute(ctx);
    const endIds = new Set<number>();
    for (const entry of result) {
      endIds.add(entry.payload.fields["end"] as number);
    }
    expect(endIds.has(2)).toBe(true);
  });
});

// =============================================================================
// Statistics (per-graph)
// =============================================================================

describe("Per-graph statistics", () => {
  it("degree distribution", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g");
    gs.addVertex(createVertex(1, "a", {}), "g");
    gs.addVertex(createVertex(2, "b", {}), "g");
    gs.addVertex(createVertex(3, "c", {}), "g");
    gs.addEdge(createEdge(10, 1, 2, "e"), "g");
    gs.addEdge(createEdge(11, 1, 3, "e"), "g");

    const dist = gs.degreeDistribution("g");
    // vertex 1 has out-degree 2, vertices 2,3 have in-degree 1
    // The distribution counts total degree (out+in)
    expect(dist.get(2)).toBe(1); // vertex 1 has degree 2
  });

  it("vertex label counts", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g");
    gs.addVertex(createVertex(1, "person", {}), "g");
    gs.addVertex(createVertex(2, "person", {}), "g");
    gs.addVertex(createVertex(3, "company", {}), "g");

    const counts = gs.vertexLabelCounts("g");
    expect(counts.get("person")).toBe(2);
    expect(counts.get("company")).toBe(1);
  });
});

// =============================================================================
// GraphPayload graph_name field
// =============================================================================

describe("GraphPayload", () => {
  it("has graph name", () => {
    const gp = createGraphPayload({
      subgraphVertices: new Set([1, 2]),
      subgraphEdges: new Set(),
      graphName: "my_graph",
    });
    expect(gp.graphName).toBe("my_graph");
  });

  it("default graph name is empty", () => {
    const gp = createGraphPayload();
    expect(gp.graphName).toBe("");
  });
});

// =============================================================================
// Remove vertex/edge
// =============================================================================

describe("Remove vertex/edge", () => {
  it("remove vertex from graph", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g");
    gs.addVertex(createVertex(1, "a", {}), "g");
    gs.addVertex(createVertex(2, "b", {}), "g");
    gs.addEdge(createEdge(10, 1, 2, "e"), "g");

    gs.removeVertex(1, "g");
    expect(gs.vertexIdsInGraph("g")).toEqual(new Set([2]));
    // Edge should also be removed from partition
    expect(gs.edgesInGraph("g").length).toBe(0);
  });

  it("remove edge from graph", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g");
    gs.addVertex(createVertex(1, "a", {}), "g");
    gs.addVertex(createVertex(2, "b", {}), "g");
    gs.addEdge(createEdge(10, 1, 2, "e"), "g");

    gs.removeEdge(10, "g");
    expect(gs.edgesInGraph("g").length).toBe(0);
    // Vertices should still exist
    expect(gs.vertexIdsInGraph("g")).toEqual(new Set([1, 2]));
  });
});

// =============================================================================
// Clear
// =============================================================================

describe("Clear", () => {
  it("removes all", () => {
    const gs = makeGraphStore();
    addTriangle(gs, "g1");
    addTriangle(gs, "g2", 10);
    gs.clear();

    expect(gs.graphNames()).toEqual([]);
    expect(gs.vertices.size).toBe(0);
    expect(gs.edges.size).toBe(0);
  });
});

// =============================================================================
// Vertex labels
// =============================================================================

describe("VertexLabels", () => {
  it("vertex has label", () => {
    const v = createVertex(1, "Person", { name: "Alice" });
    expect(v.label).toBe("Person");
  });

  it("vertices by label", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("test");
    gs.addVertex(createVertex(1, "Person", { name: "Alice" }), "test");
    gs.addVertex(createVertex(2, "Person", { name: "Bob" }), "test");
    gs.addVertex(createVertex(3, "City", { name: "NYC" }), "test");

    const persons = gs.verticesByLabel("Person", "test");
    expect(persons.length).toBe(2);
    const cities = gs.verticesByLabel("City", "test");
    expect(cities.length).toBe(1);
  });

  it("next vertex id", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("test");
    gs.addVertex(createVertex(5, "X", {}), "test");
    const id1 = gs.nextVertexId();
    const id2 = gs.nextVertexId();
    expect(id2).toBe(id1 + 1);
  });

  it("next edge id", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("test");
    gs.addEdge(createEdge(10, 1, 2, "X"), "test");
    const eid = gs.nextEdgeId();
    expect(eid).toBeGreaterThanOrEqual(11);
  });
});
