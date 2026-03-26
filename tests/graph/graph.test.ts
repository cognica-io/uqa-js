import { describe, expect, it } from "vitest";
import { createEdge, createVertex } from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import { createGraphPayload, GraphPostingList } from "../../src/graph/posting-list.js";
import {
  Alternation,
  Concat,
  KleeneStar,
  Label,
  parseRpq,
} from "../../src/graph/pattern.js";
import type { GraphPattern } from "../../src/graph/pattern.js";
import { PatternMatchOperator, TraverseOperator } from "../../src/graph/operators.js";
import { TemporalFilter } from "../../src/graph/temporal-filter.js";
import { GraphDelta } from "../../src/graph/delta.js";

function makeGraph(): MemoryGraphStore {
  const gs = new MemoryGraphStore();
  gs.createGraph("test");
  gs.addVertex(createVertex(1, "Person", { name: "Alice" }), "test");
  gs.addVertex(createVertex(2, "Person", { name: "Bob" }), "test");
  gs.addVertex(createVertex(3, "Person", { name: "Charlie" }), "test");
  gs.addVertex(createVertex(4, "City", { name: "NYC" }), "test");
  gs.addEdge(createEdge(1, 1, 2, "knows"), "test");
  gs.addEdge(createEdge(2, 2, 3, "knows"), "test");
  gs.addEdge(createEdge(3, 1, 4, "lives_in"), "test");
  return gs;
}

// -- MemoryGraphStore --------------------------------------------------------

describe("MemoryGraphStore", () => {
  it("creates and queries graph", () => {
    const gs = makeGraph();
    expect(gs.hasGraph("test")).toBe(true);
    expect(gs.graphNames()).toEqual(["test"]);
  });

  it("vertices and edges", () => {
    const gs = makeGraph();
    expect(gs.verticesInGraph("test").length).toBe(4);
    expect(gs.edgesInGraph("test").length).toBe(3);
  });

  it("neighbors out", () => {
    const gs = makeGraph();
    const n = gs.neighbors(1, "test", null, "out");
    expect(n).toContain(2);
    expect(n).toContain(4);
  });

  it("neighbors with label filter", () => {
    const gs = makeGraph();
    const n = gs.neighbors(1, "test", "knows", "out");
    expect(n).toEqual([2]);
  });

  it("neighbors in", () => {
    const gs = makeGraph();
    const n = gs.neighbors(2, "test", null, "in");
    expect(n).toContain(1);
  });

  it("remove vertex cascades edges", () => {
    const gs = makeGraph();
    gs.removeVertex(2, "test");
    expect(gs.verticesInGraph("test").length).toBe(3);
    expect(gs.edgesInGraph("test").length).toBe(1);
  });

  it("graph algebra union", () => {
    const gs = makeGraph();
    gs.createGraph("g2");
    gs.addVertex(createVertex(5, "Person", { name: "Dave" }), "g2");
    gs.addVertex(createVertex(1, "Person", { name: "Alice" }), "g2");
    gs.createGraph("union");
    gs.unionGraphs("test", "g2", "union");
    expect(gs.vertexIdsInGraph("union").has(1)).toBe(true);
    expect(gs.vertexIdsInGraph("union").has(5)).toBe(true);
  });

  it("drop graph", () => {
    const gs = makeGraph();
    gs.dropGraph("test");
    expect(gs.hasGraph("test")).toBe(false);
  });

  it("vertices by label", () => {
    const gs = makeGraph();
    const persons = gs.verticesByLabel("Person", "test");
    expect(persons.length).toBe(3);
  });

  it("next vertex and edge ids", () => {
    const gs = new MemoryGraphStore();
    const v1 = gs.nextVertexId();
    const v2 = gs.nextVertexId();
    expect(v2).toBe(v1 + 1);
  });
});

// -- GraphPostingList --------------------------------------------------------

describe("GraphPostingList", () => {
  it("stores and retrieves graph payloads", () => {
    const gpl = new GraphPostingList();
    const gp = createGraphPayload({
      subgraphVertices: new Set([1, 2]),
      subgraphEdges: new Set([10]),
      score: 0.9,
    });
    gpl.setGraphPayload(1, gp);
    expect(gpl.getGraphPayload(1)).toBe(gp);
    expect(gpl.getGraphPayload(999)).toBeNull();
  });

  it("toPostingList encodes graph data", () => {
    const gpl = new GraphPostingList([
      { docId: 1, payload: { positions: [], score: 0.5, fields: {} } },
    ]);
    gpl.setGraphPayload(
      1,
      createGraphPayload({
        subgraphVertices: new Set([1, 2]),
        subgraphEdges: new Set([10]),
      }),
    );
    const pl = gpl.toPostingList();
    const fields = pl.entries[0]!.payload.fields as Record<string, unknown>;
    expect(fields["_subgraph_vertices"]).toBeDefined();
    expect(fields["_subgraph_edges"]).toBeDefined();
  });
});

// -- RPQ Parser --------------------------------------------------------------

describe("parseRpq", () => {
  it("parses simple label", () => {
    const expr = parseRpq("knows");
    expect(expr).toBeInstanceOf(Label);
    expect((expr as Label).name).toBe("knows");
  });

  it("parses concatenation", () => {
    const expr = parseRpq("knows/likes");
    expect(expr).toBeInstanceOf(Concat);
  });

  it("parses alternation", () => {
    const expr = parseRpq("knows|likes");
    expect(expr).toBeInstanceOf(Alternation);
  });

  it("parses Kleene star", () => {
    const expr = parseRpq("knows*");
    expect(expr).toBeInstanceOf(KleeneStar);
  });

  it("parses parenthesized expression", () => {
    const expr = parseRpq("(knows|likes)*");
    expect(expr).toBeInstanceOf(KleeneStar);
  });
});

// -- TraverseOperator --------------------------------------------------------

describe("TraverseOperator", () => {
  it("traverses 1 hop", () => {
    const gs = makeGraph();
    const op = new TraverseOperator(1, "test", null, 1);
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThanOrEqual(2);
  });

  it("traverses with label filter", () => {
    const gs = makeGraph();
    const op = new TraverseOperator(1, "test", "knows", 2);
    const result = op.execute({ graphStore: gs });
    const ids = [...result.docIds];
    expect(ids).toContain(2);
    expect(ids).toContain(3);
  });
});

// -- PatternMatchOperator ----------------------------------------------------

describe("PatternMatchOperator", () => {
  it("matches simple edge pattern", () => {
    const gs = makeGraph();
    const pattern: GraphPattern = {
      vertexPatterns: [
        { variable: "a", constraints: [] },
        { variable: "b", constraints: [] },
      ],
      edgePatterns: [
        {
          sourceVar: "a",
          targetVar: "b",
          label: "knows",
          constraints: [],
          negated: false,
        },
      ],
    };
    const op = new PatternMatchOperator(pattern, "test");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBe(2);
  });
});

// -- TemporalFilter ----------------------------------------------------------

describe("TemporalFilter", () => {
  it("validates point-in-time", () => {
    const tf = new TemporalFilter({ timestamp: 5.0 });
    const e1 = createEdge(1, 1, 2, "e", { timestamp: 5.0 });
    const e2 = createEdge(2, 1, 2, "e", { timestamp: 15.0 });
    expect(tf.isValid(e1)).toBe(true);
    expect(tf.isValid(e2)).toBe(false);
  });

  it("always valid without temporal properties", () => {
    const tf = new TemporalFilter({ timestamp: 5.0 });
    const e = createEdge(1, 1, 2, "e");
    expect(tf.isValid(e)).toBe(true);
  });
});

// -- GraphDelta --------------------------------------------------------------

describe("GraphDelta", () => {
  it("tracks operations", () => {
    const delta = new GraphDelta();
    delta.addVertex(createVertex(1, "Person"));
    delta.addEdge(createEdge(1, 1, 2, "knows"));
    delta.removeVertex(3);
    expect(delta.ops.length).toBe(3);
    expect(delta.affectedVertices.has(1)).toBe(true);
    expect(delta.affectedVertices.has(3)).toBe(true);
    expect(delta.affectedLabels.has("knows")).toBe(true);
  });

  it("reports empty correctly", () => {
    expect(new GraphDelta().isEmpty).toBe(true);
  });
});
