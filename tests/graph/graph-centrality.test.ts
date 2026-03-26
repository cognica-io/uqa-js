import { describe, expect, it } from "vitest";
import { createEdge, createVertex, IndexStats } from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import {
  PageRankOperator,
  HITSOperator,
  BetweennessCentralityOperator,
} from "../../src/graph/centrality.js";
import type { ExecutionContext } from "../../src/operators/base.js";

const GRAPH_NAME = "test";

function makeStarGraph(): MemoryGraphStore {
  const g = new MemoryGraphStore();
  g.createGraph(GRAPH_NAME);
  for (let i = 1; i <= 4; i++) {
    g.addVertex(createVertex(i, "node", {}), GRAPH_NAME);
  }
  g.addEdge(createEdge(1, 1, 2, "link", {}), GRAPH_NAME);
  g.addEdge(createEdge(2, 1, 3, "link", {}), GRAPH_NAME);
  g.addEdge(createEdge(3, 1, 4, "link", {}), GRAPH_NAME);
  g.addEdge(createEdge(4, 2, 1, "link", {}), GRAPH_NAME);
  g.addEdge(createEdge(5, 3, 1, "link", {}), GRAPH_NAME);
  g.addEdge(createEdge(6, 4, 1, "link", {}), GRAPH_NAME);
  return g;
}

function makeChainGraph(): MemoryGraphStore {
  const g = new MemoryGraphStore();
  g.createGraph(GRAPH_NAME);
  for (let i = 1; i <= 4; i++) {
    g.addVertex(createVertex(i, "node", {}), GRAPH_NAME);
  }
  g.addEdge(createEdge(1, 1, 2, "link", {}), GRAPH_NAME);
  g.addEdge(createEdge(2, 2, 3, "link", {}), GRAPH_NAME);
  g.addEdge(createEdge(3, 3, 4, "link", {}), GRAPH_NAME);
  return g;
}

function makeBipartiteGraph(): MemoryGraphStore {
  const g = new MemoryGraphStore();
  g.createGraph(GRAPH_NAME);
  for (let i = 1; i <= 5; i++) {
    g.addVertex(createVertex(i, "node", {}), GRAPH_NAME);
  }
  g.addEdge(createEdge(1, 1, 3, "link", {}), GRAPH_NAME);
  g.addEdge(createEdge(2, 1, 4, "link", {}), GRAPH_NAME);
  g.addEdge(createEdge(3, 1, 5, "link", {}), GRAPH_NAME);
  g.addEdge(createEdge(4, 2, 3, "link", {}), GRAPH_NAME);
  g.addEdge(createEdge(5, 2, 4, "link", {}), GRAPH_NAME);
  g.addEdge(createEdge(6, 2, 5, "link", {}), GRAPH_NAME);
  return g;
}

function makeLineGraph(): MemoryGraphStore {
  const g = new MemoryGraphStore();
  g.createGraph(GRAPH_NAME);
  for (let i = 1; i <= 3; i++) {
    g.addVertex(createVertex(i, "node", {}), GRAPH_NAME);
  }
  g.addEdge(createEdge(1, 1, 2, "link", {}), GRAPH_NAME);
  g.addEdge(createEdge(2, 2, 3, "link", {}), GRAPH_NAME);
  return g;
}

// ======================================================================
// PageRank
// ======================================================================

describe("PageRankOperator", () => {
  it("star topology - center has highest rank", () => {
    const graph = makeStarGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const op = new PageRankOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);

    const scores: Record<number, number> = {};
    for (const e of result) {
      scores[e.docId] = e.payload.score;
    }
    expect(scores[1]).toBe(Math.max(...Object.values(scores)));
  });

  it("chain graph - last vertex has highest rank", () => {
    const graph = makeChainGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const op = new PageRankOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);

    const scores: Record<number, number> = {};
    for (const e of result) {
      scores[e.docId] = e.payload.score;
    }
    expect(scores[4]).toBe(Math.max(...Object.values(scores)));
  });

  it("empty graph", () => {
    const graph = new MemoryGraphStore();
    graph.createGraph(GRAPH_NAME);
    const ctx: ExecutionContext = { graphStore: graph };
    const op = new PageRankOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);
    expect([...result].length).toBe(0);
  });

  it("single vertex", () => {
    const graph = new MemoryGraphStore();
    graph.createGraph(GRAPH_NAME);
    graph.addVertex(createVertex(1, "node", {}), GRAPH_NAME);
    const ctx: ExecutionContext = { graphStore: graph };
    const op = new PageRankOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);

    const entries = [...result];
    expect(entries.length).toBe(1);
    expect(entries[0]!.payload.score).toBeCloseTo(1.0);
  });

  it("convergence idempotent", () => {
    const graph = makeStarGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const op = new PageRankOperator({ graph: GRAPH_NAME });
    const result1 = op.execute(ctx);
    const result2 = op.execute(ctx);

    const scores1: Record<number, number> = {};
    for (const e of result1) {
      scores1[e.docId] = e.payload.score;
    }
    const scores2: Record<number, number> = {};
    for (const e of result2) {
      scores2[e.docId] = e.payload.score;
    }
    for (const vid of Object.keys(scores1)) {
      expect(scores1[Number(vid)]).toBeCloseTo(scores2[Number(vid)]!);
    }
  });

  it("cost estimate", () => {
    const stats = new IndexStats(1000);
    const op = new PageRankOperator({ graph: GRAPH_NAME });
    const cost = op.costEstimate(stats);
    // Cost estimate should be positive
    expect(cost).toBeGreaterThan(0);
  });
});

// ======================================================================
// HITS
// ======================================================================

describe("HITSOperator", () => {
  it("bipartite graph", () => {
    const graph = makeBipartiteGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const op = new HITSOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);

    const hubScores: Record<number, number> = {};
    const authScores: Record<number, number> = {};
    for (const entry of result) {
      hubScores[entry.docId] = entry.payload.fields["hub"] as number;
      authScores[entry.docId] = entry.payload.fields["authority"] as number;
    }

    const hubVertices = [1, 2];
    const authorityVertices = [3, 4, 5];

    const maxHubAmongHubs = Math.max(...hubVertices.map((v) => hubScores[v]!));
    const maxHubAmongAuths = Math.max(...authorityVertices.map((v) => hubScores[v]!));
    expect(maxHubAmongHubs).toBeGreaterThan(maxHubAmongAuths);

    const maxAuthAmongAuths = Math.max(...authorityVertices.map((v) => authScores[v]!));
    const maxAuthAmongHubs = Math.max(...hubVertices.map((v) => authScores[v]!));
    expect(maxAuthAmongAuths).toBeGreaterThan(maxAuthAmongHubs);
  });

  it("hub authority fields present", () => {
    const graph = makeStarGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const op = new HITSOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);

    for (const entry of result) {
      expect("hub" in entry.payload.fields).toBe(true);
      expect("authority" in entry.payload.fields).toBe(true);
    }
  });

  it("empty graph", () => {
    const graph = new MemoryGraphStore();
    graph.createGraph(GRAPH_NAME);
    const ctx: ExecutionContext = { graphStore: graph };
    const op = new HITSOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);
    expect([...result].length).toBe(0);
  });
});

// ======================================================================
// Betweenness Centrality
// ======================================================================

describe("BetweennessCentralityOperator", () => {
  it("line graph - middle vertex highest", () => {
    const graph = makeLineGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const op = new BetweennessCentralityOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);

    const scores: Record<number, number> = {};
    for (const e of result) {
      scores[e.docId] = e.payload.score;
    }
    expect(scores[2]).toBe(Math.max(...Object.values(scores)));
    expect(scores[2]!).toBeGreaterThan(0.0);
  });

  it("star topology - center highest", () => {
    const graph = makeStarGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const op = new BetweennessCentralityOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);

    const scores: Record<number, number> = {};
    for (const e of result) {
      scores[e.docId] = e.payload.score;
    }
    expect(scores[1]).toBe(Math.max(...Object.values(scores)));
  });

  it("scores are non-negative", () => {
    const graph = makeStarGraph();
    const ctx: ExecutionContext = { graphStore: graph };
    const op = new BetweennessCentralityOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);

    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThanOrEqual(0.0);
    }
  });

  it("empty graph", () => {
    const graph = new MemoryGraphStore();
    graph.createGraph(GRAPH_NAME);
    const ctx: ExecutionContext = { graphStore: graph };
    const op = new BetweennessCentralityOperator({ graph: GRAPH_NAME });
    const result = op.execute(ctx);
    expect([...result].length).toBe(0);
  });
});
