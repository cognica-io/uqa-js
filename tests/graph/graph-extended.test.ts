import { describe, expect, it } from "vitest";
import {
  createEdge,
  createVertex,
  createPostingEntry,
  createPayload,
} from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import { createGraphPayload, GraphPostingList } from "../../src/graph/posting-list.js";
import {
  PatternMatchOperator,
  RegularPathQueryOperator,
  TraverseOperator,
} from "../../src/graph/operators.js";
import {
  Alternation,
  Concat,
  KleeneStar,
  Label,
  parseRpq,
  createGraphPattern,
  createVertexPattern,
  createEdgePattern,
} from "../../src/graph/pattern.js";
import type { GraphPattern } from "../../src/graph/pattern.js";
import type { Vertex } from "../../src/core/types.js";

// -- Sample graph from conftest.py -------------------------------------------

function makeSampleGraph(): MemoryGraphStore {
  const store = new MemoryGraphStore();
  store.createGraph("test");

  const vertices = [
    createVertex(1, "", { name: "Alice", age: 30 }),
    createVertex(2, "", { name: "Bob", age: 25 }),
    createVertex(3, "", { name: "Charlie", age: 35 }),
    createVertex(4, "", { name: "Diana", age: 28 }),
    createVertex(5, "", { name: "Eve", age: 32 }),
  ];
  const edges = [
    createEdge(1, 1, 2, "knows", { since: 2020 }),
    createEdge(2, 1, 3, "knows", { since: 2019 }),
    createEdge(3, 2, 3, "knows", { since: 2021 }),
    createEdge(4, 2, 4, "works_with", { project: "alpha" }),
    createEdge(5, 3, 4, "knows", { since: 2022 }),
    createEdge(6, 3, 5, "works_with", { project: "beta" }),
    createEdge(7, 4, 5, "knows", { since: 2023 }),
  ];

  for (const v of vertices) store.addVertex(v, "test");
  for (const e of edges) store.addEdge(e, "test");

  return store;
}

interface ExecutionContext {
  graphStore: MemoryGraphStore;
}

function makeCtx(gs: MemoryGraphStore): ExecutionContext {
  return { graphStore: gs };
}

// =============================================================================
// GraphStore tests
// =============================================================================

describe("TestGraphStore", () => {
  it("add and get vertex", () => {
    const gs = makeSampleGraph();
    const v = gs.getVertex(1);
    expect(v).not.toBeNull();
    expect(v!.properties["name"]).toBe("Alice");
  });

  it("add and get edge", () => {
    const gs = makeSampleGraph();
    const e = gs.getEdge(1);
    expect(e).not.toBeNull();
    expect(e!.sourceId).toBe(1);
    expect(e!.targetId).toBe(2);
    expect(e!.label).toBe("knows");
  });

  it("neighbors out", () => {
    const gs = makeSampleGraph();
    const neighbors = gs.neighbors(1, "test", null, "out");
    expect(new Set(neighbors)).toEqual(new Set([2, 3]));
  });

  it("neighbors out with label", () => {
    const gs = makeSampleGraph();
    const neighbors = gs.neighbors(3, "test", "works_with", "out");
    expect(new Set(neighbors)).toEqual(new Set([5]));
  });

  it("neighbors in", () => {
    const gs = makeSampleGraph();
    const neighbors = gs.neighbors(3, "test", null, "in");
    expect(new Set(neighbors)).toEqual(new Set([1, 2]));
  });

  it("missing vertex returns null", () => {
    const gs = makeSampleGraph();
    expect(gs.getVertex(999)).toBeNull();
  });

  it("missing edge returns null", () => {
    const gs = makeSampleGraph();
    expect(gs.getEdge(999)).toBeNull();
  });
});

// =============================================================================
// TraverseOperator tests
// =============================================================================

describe("TestTraverseOperator", () => {
  it("single hop", () => {
    const gs = makeSampleGraph();
    const ctx = makeCtx(gs);
    const op = new TraverseOperator(1, "test", null, 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.docId));
    // From vertex 1: can reach 2 and 3 in 1 hop, plus start vertex 1
    expect(docIds).toEqual(new Set([1, 2, 3]));
  });

  it("two hops", () => {
    const gs = makeSampleGraph();
    const ctx = makeCtx(gs);
    const op = new TraverseOperator(1, "test", null, 2);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.docId));
    // Hop 1: {2, 3}, Hop 2: from 2->{3,4}, from 3->{4,5}
    expect(docIds).toEqual(new Set([1, 2, 3, 4, 5]));
  });

  it("with label filter", () => {
    const gs = makeSampleGraph();
    const ctx = makeCtx(gs);
    const op = new TraverseOperator(1, "test", "knows", 2);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.docId));
    // Hop 1: {2,3} via "knows", Hop 2: 2->3 knows, 3->4 knows
    expect(docIds).toEqual(new Set([1, 2, 3, 4]));
  });

  it("max hops zero", () => {
    const gs = makeSampleGraph();
    const ctx = makeCtx(gs);
    const op = new TraverseOperator(1, "test", null, 0);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.docId));
    // Zero hops: only the start vertex
    expect(docIds).toEqual(new Set([1]));
  });

  it("BFS correctness at each depth", () => {
    const gs = makeSampleGraph();
    const ctx = makeCtx(gs);

    // depth 1
    const op1 = new TraverseOperator(1, "test", null, 1);
    const r1 = new Set([...op1.execute(ctx)].map((e) => e.docId));

    // depth 2
    const op2 = new TraverseOperator(1, "test", null, 2);
    const r2 = new Set([...op2.execute(ctx)].map((e) => e.docId));

    // depth 2 should be a superset of depth 1
    for (const v of r1) {
      expect(r2.has(v)).toBe(true);
    }

    // depth 3
    const op3 = new TraverseOperator(1, "test", null, 3);
    const r3 = new Set([...op3.execute(ctx)].map((e) => e.docId));
    for (const v of r2) {
      expect(r3.has(v)).toBe(true);
    }
  });
});

// =============================================================================
// PatternMatchOperator tests
// =============================================================================

describe("TestPatternMatchOperator", () => {
  it("find triangles", () => {
    const gs = makeSampleGraph();
    const ctx = makeCtx(gs);
    const pattern = createGraphPattern(
      [createVertexPattern("a"), createVertexPattern("b"), createVertexPattern("c")],
      [
        createEdgePattern("a", "b", { label: "knows" }),
        createEdgePattern("b", "c", { label: "knows" }),
        createEdgePattern("a", "c", { label: "knows" }),
      ],
    );
    const op = new PatternMatchOperator(pattern, "test");
    const result = op.execute(ctx);
    // Triangle: (1,2,3) - edges: 1->2 knows, 2->3 knows, 1->3 knows
    expect(result.length).toBeGreaterThanOrEqual(1);

    // Verify at least one match has vertices {1,2,3}
    let foundTriangle = false;
    for (const entry of result) {
      const gpl = result as GraphPostingList;
      const gp = gpl.getGraphPayload(entry.docId);
      if (gp !== null) {
        const verts = new Set(gp.subgraphVertices);
        if (verts.has(1) && verts.has(2) && verts.has(3) && verts.size === 3) {
          foundTriangle = true;
          break;
        }
      }
    }
    expect(foundTriangle).toBe(true);
  });

  it("find star pattern", () => {
    const gs = makeSampleGraph();
    const ctx = makeCtx(gs);
    const pattern = createGraphPattern(
      [
        createVertexPattern("center"),
        createVertexPattern("leaf1"),
        createVertexPattern("leaf2"),
      ],
      [
        createEdgePattern("center", "leaf1", { label: "knows" }),
        createEdgePattern("center", "leaf2", { label: "knows" }),
      ],
    );
    const op = new PatternMatchOperator(pattern, "test");
    const result = op.execute(ctx);
    // Vertex 1 knows 2 and 3
    expect(result.length).toBeGreaterThanOrEqual(1);
  });

  it("vertex constraints", () => {
    const gs = makeSampleGraph();
    const ctx = makeCtx(gs);
    const pattern = createGraphPattern(
      [
        createVertexPattern("a", [(v: Vertex) => (v.properties["age"] as number) < 30]),
        createVertexPattern("b"),
      ],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const op = new PatternMatchOperator(pattern, "test");
    const result = op.execute(ctx);
    // Only Bob (25) and Diana (28) have age < 30
    for (const entry of result) {
      const gpl = result as GraphPostingList;
      const gp = gpl.getGraphPayload(entry.docId);
      expect(gp).not.toBeNull();
      for (const vid of gp!.subgraphVertices) {
        const v = gs.getVertex(vid);
        expect(v).not.toBeNull();
      }
    }
  });

  it("empty pattern match", () => {
    const gs = makeSampleGraph();
    const ctx = makeCtx(gs);
    const pattern = createGraphPattern(
      [createVertexPattern("a"), createVertexPattern("b")],
      [createEdgePattern("a", "b", { label: "nonexistent_label" })],
    );
    const op = new PatternMatchOperator(pattern, "test");
    const result = op.execute(ctx);
    expect(result.length).toBe(0);
  });
});

// =============================================================================
// RegularPathQueryOperator tests
// =============================================================================

describe("TestRegularPathQueryOperator", () => {
  it("single label", () => {
    const gs = makeSampleGraph();
    const ctx = makeCtx(gs);
    const expr = new Label("knows");
    const op = new RegularPathQueryOperator(expr, "test", 1);
    const result = op.execute(ctx);
    // Collect "end" vertex from fields
    const endIds = new Set<number>();
    for (const entry of result) {
      const end = entry.payload.fields["end"] as number;
      endIds.add(end);
    }
    // 1 -knows-> 2, 1 -knows-> 3
    expect(endIds).toEqual(new Set([2, 3]));
  });

  it("concat", () => {
    const gs = makeSampleGraph();
    const ctx = makeCtx(gs);
    const expr = new Concat(new Label("knows"), new Label("knows"));
    const op = new RegularPathQueryOperator(expr, "test", 1);
    const result = op.execute(ctx);
    const endIds = new Set<number>();
    for (const entry of result) {
      const end = entry.payload.fields["end"] as number;
      endIds.add(end);
    }
    // 1-knows->2-knows->3, 1-knows->3-knows->4
    expect(endIds.has(3)).toBe(true);
    expect(endIds.has(4)).toBe(true);
  });

  it("alternation", () => {
    const gs = makeSampleGraph();
    const ctx = makeCtx(gs);
    const expr = new Alternation(new Label("knows"), new Label("works_with"));
    const op = new RegularPathQueryOperator(expr, "test", 2);
    const result = op.execute(ctx);
    const endIds = new Set<number>();
    for (const entry of result) {
      const end = entry.payload.fields["end"] as number;
      endIds.add(end);
    }
    // 2-knows->3, 2-works_with->4
    expect(endIds).toEqual(new Set([3, 4]));
  });

  it("Kleene star", () => {
    const gs = makeSampleGraph();
    const ctx = makeCtx(gs);
    const expr = new KleeneStar(new Label("knows"));
    const op = new RegularPathQueryOperator(expr, "test", 1);
    const result = op.execute(ctx);
    const endIds = new Set<number>();
    for (const entry of result) {
      const end = entry.payload.fields["end"] as number;
      endIds.add(end);
    }
    // Kleene star includes zero hops (start vertex itself)
    expect(endIds.has(1)).toBe(true);
    // And transitive closure via "knows": 1->2, 1->3, 2->3, 3->4, 4->5
    expect(endIds.has(2)).toBe(true);
    expect(endIds.has(3)).toBe(true);
    expect(endIds.has(4)).toBe(true);
    expect(endIds.has(5)).toBe(true);
  });

  it("all vertices start (no start_vertex)", () => {
    const gs = makeSampleGraph();
    const ctx = makeCtx(gs);
    const expr = new Label("works_with");
    const op = new RegularPathQueryOperator(expr, "test");
    const result = op.execute(ctx);
    const endIds = new Set<number>();
    for (const entry of result) {
      const end = entry.payload.fields["end"] as number;
      endIds.add(end);
    }
    // edges: 2-works_with->4, 3-works_with->5
    expect(endIds.has(4)).toBe(true);
    expect(endIds.has(5)).toBe(true);
  });
});

// =============================================================================
// RPQ parser tests
// =============================================================================

describe("TestParseRPQ", () => {
  it("single label", () => {
    const result = parseRpq("knows");
    expect(result).toBeInstanceOf(Label);
    expect((result as Label).name).toBe("knows");
  });

  it("concat", () => {
    const result = parseRpq("knows/works_with");
    expect(result).toBeInstanceOf(Concat);
    expect((result as Concat).left).toBeInstanceOf(Label);
    expect((result as Concat).right).toBeInstanceOf(Label);
  });

  it("alternation", () => {
    const result = parseRpq("knows|works_with");
    expect(result).toBeInstanceOf(Alternation);
  });

  it("Kleene star", () => {
    const result = parseRpq("knows*");
    expect(result).toBeInstanceOf(KleeneStar);
    expect((result as KleeneStar).inner).toBeInstanceOf(Label);
  });

  it("parentheses", () => {
    const result = parseRpq("(knows|works_with)*");
    expect(result).toBeInstanceOf(KleeneStar);
    expect((result as KleeneStar).inner).toBeInstanceOf(Alternation);
  });

  it("complex expression", () => {
    const result = parseRpq("knows/works_with|knows*");
    // alternation is lowest precedence: (knows/works_with) | (knows*)
    expect(result).toBeInstanceOf(Alternation);
  });

  it("invalid expression", () => {
    expect(() => parseRpq(")")).toThrow();
  });
});

// =============================================================================
// GraphPostingList isomorphism tests
// =============================================================================

describe("TestGraphPostingListIsomorphism", () => {
  it("round trip", () => {
    const entries = [
      createPostingEntry(1, { score: 0.5 }),
      createPostingEntry(2, { score: 0.8 }),
    ];
    const gpl = new GraphPostingList(entries);
    gpl.setGraphPayload(
      1,
      createGraphPayload({
        subgraphVertices: new Set([1, 2]),
        subgraphEdges: new Set([10]),
        score: 0.5,
      }),
    );
    gpl.setGraphPayload(
      2,
      createGraphPayload({
        subgraphVertices: new Set([2, 3]),
        subgraphEdges: new Set([20]),
        score: 0.8,
      }),
    );

    // Convert to standard posting list and back
    const pl = gpl.toPostingList();
    const entries2 = [...pl];
    expect(entries2.length).toBe(2);

    // Verify graph payload data is preserved in fields
    const f1 = entries2.find((e) => e.docId === 1)!.payload.fields as Record<
      string,
      unknown
    >;
    expect(f1["_subgraph_vertices"]).toBeDefined();
    expect(f1["_subgraph_edges"]).toBeDefined();

    const f2 = entries2.find((e) => e.docId === 2)!.payload.fields as Record<
      string,
      unknown
    >;
    expect(f2["_subgraph_vertices"]).toBeDefined();
    expect(f2["_subgraph_edges"]).toBeDefined();
  });

  it("functor property union", () => {
    const entriesA = [
      createPostingEntry(1, { score: 0.5 }),
      createPostingEntry(3, { score: 0.3 }),
    ];
    const entriesB = [
      createPostingEntry(2, { score: 0.8 }),
      createPostingEntry(3, { score: 0.4 }),
    ];

    const gplA = new GraphPostingList(entriesA);
    gplA.setGraphPayload(
      1,
      createGraphPayload({ subgraphVertices: new Set([1]), score: 0.5 }),
    );
    gplA.setGraphPayload(
      3,
      createGraphPayload({ subgraphVertices: new Set([3]), score: 0.3 }),
    );

    const gplB = new GraphPostingList(entriesB);
    gplB.setGraphPayload(
      2,
      createGraphPayload({ subgraphVertices: new Set([2]), score: 0.8 }),
    );
    gplB.setGraphPayload(
      3,
      createGraphPayload({ subgraphVertices: new Set([3]), score: 0.4 }),
    );

    // Phi(A union B) -- compute union at graph level, then convert
    const gplUnion = gplA.union(gplB);
    const plOfUnion = new GraphPostingList(gplUnion.entries).toPostingList();

    // Phi(A) union Phi(B) -- convert each, then union
    const plA = gplA.toPostingList();
    const plB = gplB.toPostingList();
    const unionOfPl = plA.union(plB);

    // Both should have the same doc_ids
    const idsA = new Set([...plOfUnion].map((e) => e.docId));
    const idsB = new Set([...unionOfPl].map((e) => e.docId));
    expect(idsA).toEqual(idsB);
  });
});
