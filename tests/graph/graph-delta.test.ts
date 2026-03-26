import { describe, expect, it } from "vitest";
import { createEdge, createVertex } from "../../src/core/types.js";
import { GraphDelta } from "../../src/graph/delta.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import { VersionedGraphStore } from "../../src/graph/versioned-store.js";
import { Engine } from "../../src/engine.js";

const GRAPH_NAME = "test";

// ======================================================================
// GraphDelta
// ======================================================================

describe("GraphDelta", () => {
  it("empty delta", () => {
    const delta = new GraphDelta();
    expect(delta.isEmpty).toBe(true);
    expect(delta.ops.length).toBe(0);
  });

  it("add vertex op", () => {
    const delta = new GraphDelta();
    const v = createVertex(1, "person", { name: "Alice" });
    delta.addVertex(v);
    expect(delta.isEmpty).toBe(false);
    expect(delta.ops.length).toBe(1);
  });

  it("remove vertex op", () => {
    const delta = new GraphDelta();
    delta.removeVertex(1);
    expect(delta.ops.length).toBe(1);
  });

  it("add edge op", () => {
    const delta = new GraphDelta();
    const e = createEdge(1, 1, 2, "knows");
    delta.addEdge(e);
    expect(delta.ops.length).toBe(1);
  });

  it("remove edge op", () => {
    const delta = new GraphDelta();
    delta.removeEdge(1);
    expect(delta.ops.length).toBe(1);
  });

  it("affected vertex ids", () => {
    const delta = new GraphDelta();
    delta.addVertex(createVertex(1, "person"));
    delta.addEdge(createEdge(1, 2, 3, "knows"));
    const ids = delta.affectedVertices;
    expect(ids.has(1)).toBe(true);
    expect(ids.has(2)).toBe(true);
    expect(ids.has(3)).toBe(true);
  });

  it("affected edge labels", () => {
    const delta = new GraphDelta();
    delta.addEdge(createEdge(1, 1, 2, "knows"));
    delta.addEdge(createEdge(2, 2, 3, "works_with"));
    const labels = delta.affectedLabels;
    expect(labels.has("knows")).toBe(true);
    expect(labels.has("works_with")).toBe(true);
  });

  it("multiple ops", () => {
    const delta = new GraphDelta();
    delta.addVertex(createVertex(1, "person"));
    delta.addVertex(createVertex(2, "person"));
    delta.addEdge(createEdge(1, 1, 2, "knows"));
    expect(delta.ops.length).toBe(3);
  });
});

// ======================================================================
// VersionedGraphStore
// ======================================================================

describe("VersionedGraphStore", () => {
  it("initial version", () => {
    const g = new MemoryGraphStore();
    g.createGraph(GRAPH_NAME);
    const vg = new VersionedGraphStore(g);
    expect(vg.currentVersion).toBe(0);
  });

  it("apply increments version", () => {
    const g = new MemoryGraphStore();
    g.createGraph(GRAPH_NAME);
    const vg = new VersionedGraphStore(g);
    const delta = new GraphDelta();
    delta.addVertex(createVertex(1, "person", { name: "Alice" }));
    vg.apply(delta, GRAPH_NAME);
    expect(vg.currentVersion).toBe(1);
  });

  it("apply adds vertex", () => {
    const g = new MemoryGraphStore();
    g.createGraph(GRAPH_NAME);
    const vg = new VersionedGraphStore(g);
    const delta = new GraphDelta();
    delta.addVertex(createVertex(1, "person", { name: "Alice" }));
    vg.apply(delta, GRAPH_NAME);
    expect(g.getVertex(1)).not.toBeNull();
    expect(g.getVertex(1)!.properties["name"]).toBe("Alice");
  });

  it("apply adds edge", () => {
    const g = new MemoryGraphStore();
    g.createGraph(GRAPH_NAME);
    g.addVertex(createVertex(1, "person"), GRAPH_NAME);
    g.addVertex(createVertex(2, "person"), GRAPH_NAME);
    const vg = new VersionedGraphStore(g);
    const delta = new GraphDelta();
    delta.addEdge(createEdge(1, 1, 2, "knows"));
    vg.apply(delta, GRAPH_NAME);
    expect(g.getEdge(1)).not.toBeNull();
  });

  it("apply removes vertex", () => {
    const g = new MemoryGraphStore();
    g.createGraph(GRAPH_NAME);
    g.addVertex(createVertex(1, "person"), GRAPH_NAME);
    const vg = new VersionedGraphStore(g);
    const delta = new GraphDelta();
    delta.removeVertex(1);
    vg.apply(delta, GRAPH_NAME);
    expect(g.getVertex(1)).toBeNull();
  });

  it("apply removes edge", () => {
    const g = new MemoryGraphStore();
    g.createGraph(GRAPH_NAME);
    g.addVertex(createVertex(1, "person"), GRAPH_NAME);
    g.addVertex(createVertex(2, "person"), GRAPH_NAME);
    g.addEdge(createEdge(1, 1, 2, "knows"), GRAPH_NAME);
    const vg = new VersionedGraphStore(g);
    const delta = new GraphDelta();
    delta.removeEdge(1);
    vg.apply(delta, GRAPH_NAME);
    expect(g.getEdge(1)).toBeNull();
  });

  it("rollback to initial", () => {
    const g = new MemoryGraphStore();
    g.createGraph(GRAPH_NAME);
    const vg = new VersionedGraphStore(g);
    const delta = new GraphDelta();
    delta.addVertex(createVertex(1, "person", { name: "Alice" }));
    vg.apply(delta, GRAPH_NAME);
    expect(g.getVertex(1)).not.toBeNull();
    vg.rollback(0, GRAPH_NAME);
    expect(vg.currentVersion).toBe(0);
    expect(g.getVertex(1)).toBeNull();
  });

  it("rollback partial", () => {
    const g = new MemoryGraphStore();
    g.createGraph(GRAPH_NAME);
    const vg = new VersionedGraphStore(g);
    const d1 = new GraphDelta();
    d1.addVertex(createVertex(1, "person"));
    vg.apply(d1, GRAPH_NAME);
    const d2 = new GraphDelta();
    d2.addVertex(createVertex(2, "person"));
    vg.apply(d2, GRAPH_NAME);
    expect(vg.currentVersion).toBe(2);
    vg.rollback(1, GRAPH_NAME);
    expect(vg.currentVersion).toBe(1);
    expect(g.getVertex(1)).not.toBeNull();
    expect(g.getVertex(2)).toBeNull();
  });

  it("rollback edge removal", () => {
    const g = new MemoryGraphStore();
    g.createGraph(GRAPH_NAME);
    g.addVertex(createVertex(1, "person"), GRAPH_NAME);
    g.addVertex(createVertex(2, "person"), GRAPH_NAME);
    const edge = createEdge(1, 1, 2, "knows");
    g.addEdge(edge, GRAPH_NAME);
    const vg = new VersionedGraphStore(g);
    const delta = new GraphDelta();
    // Pass the edge object so rollback can restore it
    delta.removeEdge(1, edge);
    vg.apply(delta, GRAPH_NAME);
    expect(g.getEdge(1)).toBeNull();
    vg.rollback(0, GRAPH_NAME);
    expect(g.getEdge(1)).not.toBeNull();
  });

  it("rollback invalid version", () => {
    const g = new MemoryGraphStore();
    g.createGraph(GRAPH_NAME);
    const vg = new VersionedGraphStore(g);
    expect(() => vg.rollback(5, GRAPH_NAME)).toThrow();
  });

  it("multiple deltas", () => {
    const g = new MemoryGraphStore();
    g.createGraph(GRAPH_NAME);
    const vg = new VersionedGraphStore(g);
    for (let i = 1; i <= 5; i++) {
      const d = new GraphDelta();
      d.addVertex(createVertex(i, "person"));
      vg.apply(d, GRAPH_NAME);
    }
    expect(vg.currentVersion).toBe(5);
    expect(g.verticesInGraph(GRAPH_NAME).length).toBe(5);
  });
});

// ======================================================================
// Engine.apply_graph_delta (Engine not yet ported)
// ======================================================================

describe("EngineGraphDelta", () => {
  it("apply delta", () => {
    const e = new Engine();
    e.createGraph("social");
    e.applyGraphDelta("social", {
      addVertices: [
        createVertex(1, "person", { name: "Alice" }),
        createVertex(2, "person", { name: "Bob" }),
      ],
      addEdges: [createEdge(1, 1, 2, "knows")],
    });
    expect(e._graphStore.getVertex(1)).not.toBeNull();
    expect(e._graphStore.getVertex(2)).not.toBeNull();
    expect(e._graphStore.getEdge(1)).not.toBeNull();
  });
  it("apply delta nonexistent graph", () => {
    const e = new Engine();
    expect(() =>
      e.applyGraphDelta("nonexistent", {
        addVertices: [createVertex(1, "person")],
      }),
    ).toThrow(/does not exist/);
  });
  it("apply delta with edge adds after vertex adds", () => {
    const e = new Engine();
    e.createGraph("social");
    e.applyGraphDelta("social", {
      addVertices: [createVertex(1, "person"), createVertex(2, "person")],
      addEdges: [createEdge(1, 1, 2, "knows")],
    });
    const edge = e._graphStore.getEdge(1);
    expect(edge).not.toBeNull();
    expect(edge!.label).toBe("knows");
  });
  it("apply delta removes edges", () => {
    const e = new Engine();
    const g = e.createGraph("social");
    g.addVertex(createVertex(1, "person"), "social");
    g.addVertex(createVertex(2, "person"), "social");
    g.addEdge(createEdge(1, 1, 2, "knows"), "social");

    e.applyGraphDelta("social", { removeEdges: [1] });
    expect(e._graphStore.getEdge(1)).toBeNull();
  });
  it("multiple deltas", () => {
    const e = new Engine();
    e.createGraph("social");
    e.applyGraphDelta("social", {
      addVertices: [createVertex(1, "person")],
    });
    e.applyGraphDelta("social", {
      addVertices: [createVertex(2, "person")],
    });
    expect(e._graphStore.getVertex(1)).not.toBeNull();
    expect(e._graphStore.getVertex(2)).not.toBeNull();
  });
});
