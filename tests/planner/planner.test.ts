import { describe, expect, it } from "vitest";
import { IndexStats } from "../../src/core/types.js";
import { CostModel } from "../../src/planner/cost-model.js";
import { CardinalityEstimator } from "../../src/planner/cardinality.js";
import { JoinGraph } from "../../src/planner/join-graph.js";
import { DPccp } from "../../src/planner/join-enumerator.js";
import { PlanExecutor } from "../../src/planner/executor.js";
import { ParallelExecutor } from "../../src/planner/parallel.js";
import { TermOperator } from "../../src/operators/primitive.js";
import { IntersectOperator, UnionOperator } from "../../src/operators/boolean.js";
import { MemoryInvertedIndex } from "../../src/storage/inverted-index.js";

describe("CostModel", () => {
  it("estimates term operator cost", () => {
    const stats = new IndexStats(1000, 50);
    stats.setDocFreq("title", "hello", 100);
    const cm = new CostModel();
    const op = new TermOperator("hello", "title");
    const cost = cm.estimate(op, stats);
    expect(cost).toBeGreaterThan(0);
  });

  it("estimates union cost as sum", () => {
    const stats = new IndexStats(1000, 50);
    stats.setDocFreq("t", "a", 50);
    stats.setDocFreq("t", "b", 30);
    const cm = new CostModel();
    const op = new UnionOperator([
      new TermOperator("a", "t"),
      new TermOperator("b", "t"),
    ]);
    const cost = cm.estimate(op, stats);
    expect(cost).toBeGreaterThan(0);
  });
});

describe("CardinalityEstimator", () => {
  it("estimates cardinality", () => {
    const stats = new IndexStats(1000, 50);
    stats.setDocFreq("title", "hello", 100);
    const ce = new CardinalityEstimator();
    const op = new TermOperator("hello", "title");
    const card = ce.estimate(op, stats);
    expect(card).toBeGreaterThan(0);
    expect(card).toBeLessThanOrEqual(1000);
  });

  it("estimates join cardinality", () => {
    const ce = new CardinalityEstimator();
    // estimateJoin(leftCard, rightCard, domainSize) = leftCard * rightCard / domainSize
    const card = ce.estimateJoin(100, 200, 100);
    expect(card).toBeCloseTo(200, 0); // 100 * 200 / 100
  });
});

describe("JoinGraph", () => {
  it("builds and queries", () => {
    const jg = new JoinGraph();
    const n0 = jg.addNode("a", null, null, 100);
    const n1 = jg.addNode("b", null, null, 200);
    jg.addEdge(n0, n1, "id", "a_id", 0.01);
    expect(jg.length).toBe(2);
    expect(jg.neighbors(n0)).toContain(n1);
  });
});

describe("DPccp", () => {
  it("finds optimal join order for 3 relations", () => {
    const jg = new JoinGraph();
    jg.addNode("a", null, null, 100);
    jg.addNode("b", null, null, 200);
    jg.addNode("c", null, null, 50);
    jg.addEdge(0, 1, "id", "a_id", 0.01);
    jg.addEdge(1, 2, "id", "b_id", 0.02);
    const dp = new DPccp(jg);
    const plan = dp.optimize();
    expect(plan.relations.size).toBe(3);
    expect(plan.cost).toBeGreaterThan(0);
  });
});

describe("PlanExecutor", () => {
  it("executes and collects stats", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { title: "hello world" });
    idx.addDocument(2, { title: "hello earth" });
    const ctx = { invertedIndex: idx };
    const pe = new PlanExecutor(ctx);
    const op = new TermOperator("hello", "title");
    const result = pe.execute(op);
    expect(result.length).toBe(2);
    expect(pe.lastStats).not.toBeNull();
    expect(pe.lastStats!.elapsedMs).toBeGreaterThanOrEqual(0);
  });

  it("explains operator", () => {
    const op = new IntersectOperator([
      new TermOperator("a", "t"),
      new TermOperator("b", "t"),
    ]);
    const pe = new PlanExecutor({});
    const plan = pe.explain(op);
    expect(plan).toContain("Intersect");
  });
});

describe("ParallelExecutor", () => {
  it("executes sequentially in browser mode", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { title: "hello" });
    idx.addDocument(2, { title: "world" });
    const ctx = { invertedIndex: idx };
    const pe = new ParallelExecutor();
    const results = pe.executeBranches(
      [new TermOperator("hello", "title"), new TermOperator("world", "title")],
      ctx,
    );
    expect(results.length).toBe(2);
    expect(results[0]!.length).toBe(1);
    expect(results[1]!.length).toBe(1);
  });
});
