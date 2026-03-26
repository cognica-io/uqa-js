import { describe, expect, it } from "vitest";
import { IndexStats, Equals, GreaterThan } from "../../src/core/types.js";
import { CardinalityEstimator } from "../../src/planner/cardinality.js";
import { CostModel } from "../../src/planner/cost-model.js";
import { QueryOptimizer } from "../../src/planner/optimizer.js";
import { PlanExecutor } from "../../src/planner/executor.js";
import {
  TermOperator,
  FilterOperator,
  KNNOperator,
  VectorSimilarityOperator,
  ScoreOperator,
} from "../../src/operators/primitive.js";
import { IntersectOperator } from "../../src/operators/boolean.js";
import {
  LogOddsFusionOperator,
  ProbBoolFusionOperator,
  ProbNotOperator,
  HybridTextVectorOperator,
  SemanticFilterOperator,
} from "../../src/operators/hybrid.js";
import {
  TraverseOperator,
  PatternMatchOperator,
  RegularPathQueryOperator,
} from "../../src/graph/operators.js";
import {
  createGraphPattern,
  createVertexPattern,
  createEdgePattern,
  Label,
} from "../../src/graph/pattern.js";

// Dummy scorer for testing
class DummyScorer {
  score(tf: number, _dl: number, _df: number): number {
    return tf;
  }
  combineScores(scores: number[]): number {
    return scores.reduce((a, b) => a + b, 0);
  }
}

// ==================================================================
// Cross-paradigm cardinality estimation
// ==================================================================

describe("CrossParadigmCardinality", () => {
  const stats = new IndexStats(1000, 16);

  it("score operator delegates to source", () => {
    const termOp = new TermOperator("test", "title");
    const scorer = new DummyScorer();
    const scoreOp = new ScoreOperator(scorer as never, termOp, ["test"], "title");
    const est = new CardinalityEstimator();
    const termCard = est.estimate(termOp, stats);
    const scoreCard = est.estimate(scoreOp, stats);
    expect(scoreCard).toBe(termCard);
  });

  it("traverse operator cardinality heuristic", () => {
    const traverseOp = new TraverseOperator(1, "test", "knows", 2);
    const est = new CardinalityEstimator();
    const card = est.estimate(traverseOp, stats);
    // Without graph stats, fallback heuristic returns > 0
    expect(card).toBeGreaterThan(0);
  });

  it("pattern match high cardinality", () => {
    const pattern = createGraphPattern([], []);
    const op = new PatternMatchOperator(pattern, "test");
    const est = new CardinalityEstimator();
    const card = est.estimate(op, stats);
    expect(card).toBeGreaterThanOrEqual(stats.totalDocs);
  });

  it("LogOddsFusionOperator cardinality estimation", () => {
    const term1 = new TermOperator("hello", "body");
    const term2 = new TermOperator("world", "body");
    const fusion = new LogOddsFusionOperator([term1, term2], 0.5);
    const est = new CardinalityEstimator();
    const card = est.estimate(fusion, stats);
    // Fusion uses costEstimate fallback
    expect(card).toBeGreaterThan(0);
  });

  it("ProbBoolFusionOperator cardinality estimation", () => {
    const term1 = new TermOperator("hello", "body");
    const term2 = new TermOperator("world", "body");
    const fusion = new ProbBoolFusionOperator([term1, term2], "and");
    const est = new CardinalityEstimator();
    const card = est.estimate(fusion, stats);
    expect(card).toBeGreaterThan(0);
  });

  it("ProbNotOperator cardinality estimation", () => {
    const term1 = new TermOperator("hello", "body");
    const probNot = new ProbNotOperator(term1, 0.01);
    const est = new CardinalityEstimator();
    const card = est.estimate(probNot, stats);
    expect(card).toBeGreaterThan(0);
  });

  it("HybridTextVectorOperator cardinality estimation", () => {
    const vec = new Float64Array(16);
    vec[0] = 1.0;
    const hybrid = new HybridTextVectorOperator("test", vec, 0.5);
    const est = new CardinalityEstimator();
    const card = est.estimate(hybrid, stats);
    expect(card).toBeGreaterThan(0);
  });

  it("SemanticFilterOperator cardinality estimation", () => {
    const vec = new Float64Array(16);
    vec[0] = 1.0;
    const term = new TermOperator("hello", "body");
    const semantic = new SemanticFilterOperator(term, vec, 0.5);
    const est = new CardinalityEstimator();
    const card = est.estimate(semantic, stats);
    expect(card).toBeGreaterThan(0);
  });
});

// ==================================================================
// Fusion signal reordering
// ==================================================================

describe("FusionSignalReordering", () => {
  it("LogOddsFusionOperator signal reordering by cost", () => {
    const stats = new IndexStats(1000, 16);
    const term1 = new TermOperator("rare", "body");
    const term2 = new TermOperator("common", "body");
    const fusion = new LogOddsFusionOperator([term1, term2], 0.5);
    const optimizer = new QueryOptimizer(stats);
    const optimized = optimizer.optimize(fusion);
    // Should still be a LogOddsFusionOperator
    expect(optimized).toBeInstanceOf(LogOddsFusionOperator);
  });

  it("ProbBoolFusionOperator signal reordering", () => {
    const stats = new IndexStats(1000, 16);
    const term1 = new TermOperator("rare", "body");
    const term2 = new TermOperator("common", "body");
    const fusion = new ProbBoolFusionOperator([term1, term2], "and");
    const optimizer = new QueryOptimizer(stats);
    const optimized = optimizer.optimize(fusion);
    expect(optimized).toBeInstanceOf(ProbBoolFusionOperator);
  });

  it("fusion preserves alpha", () => {
    const stats = new IndexStats(1000, 16);
    const term1 = new TermOperator("hello", "body");
    const term2 = new TermOperator("world", "body");
    const fusion = new LogOddsFusionOperator([term1, term2], 0.7);
    const optimizer = new QueryOptimizer(stats);
    const optimized = optimizer.optimize(fusion) as LogOddsFusionOperator;
    expect(optimized.alpha).toBe(0.7);
  });

  it("fusion preserves mode", () => {
    const stats = new IndexStats(1000, 16);
    const term1 = new TermOperator("hello", "body");
    const term2 = new TermOperator("world", "body");
    const fusion = new ProbBoolFusionOperator([term1, term2], "or");
    const optimizer = new QueryOptimizer(stats);
    const optimized = optimizer.optimize(fusion) as ProbBoolFusionOperator;
    expect(optimized.mode).toBe("or");
  });

  it("nested fusion in intersect", () => {
    const stats = new IndexStats(1000, 16);
    const term1 = new TermOperator("hello", "body");
    const term2 = new TermOperator("world", "body");
    const fusion = new LogOddsFusionOperator([term1, term2], 0.5);
    const filter = new FilterOperator("age", new Equals(25));
    const intersect = new IntersectOperator([fusion, filter]);
    const optimizer = new QueryOptimizer(stats);
    const optimized = optimizer.optimize(intersect);
    // Should still work without errors
    expect(optimized).toBeTruthy();
  });
});

// ==================================================================
// Cross-paradigm cost model
// ==================================================================

describe("CrossParadigmCostModel", () => {
  const stats = new IndexStats(1000, 16);

  it("ScoreOperator cost includes overhead", () => {
    const statsWithDims = new IndexStats(1000, 16, 16);
    const vec = new Float64Array(16);
    vec[0] = 1.0;
    const vecOp = new VectorSimilarityOperator(vec, 0.5, "embedding");
    const scorer = new DummyScorer();
    const scoreOp = new ScoreOperator(scorer as never, vecOp, ["test"], "embedding");
    const costModel = new CostModel();
    const vecCost = costModel.estimate(vecOp, statsWithDims);
    const scoreCost = costModel.estimate(scoreOp, statsWithDims);
    // ScoreOperator cost = source cost * 1.1 overhead
    expect(vecCost).toBeGreaterThan(0);
    expect(scoreCost).toBeGreaterThan(vecCost);
  });

  it("fusion cost is sum of signals", () => {
    const term1 = new TermOperator("hello", "body");
    const term2 = new TermOperator("world", "body");
    const fusion = new LogOddsFusionOperator([term1, term2], 0.5);
    const costModel = new CostModel();
    const cost1 = costModel.estimate(term1, stats);
    const cost2 = costModel.estimate(term2, stats);
    const fusionCost = costModel.estimate(fusion, stats);
    expect(fusionCost).toBe(cost1 + cost2);
  });

  it("TraverseOperator cost model", () => {
    const traverse = new TraverseOperator(1, "test", "knows", 2);
    const costModel = new CostModel();
    const cost = costModel.estimate(traverse, stats);
    expect(cost).toBeGreaterThan(0);
  });

  it("PatternMatchOperator cost model", () => {
    const pattern = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const pm = new PatternMatchOperator(pattern, "test");
    const costModel = new CostModel();
    const cost = costModel.estimate(pm, stats);
    expect(cost).toBeGreaterThan(0);
  });
});

// ==================================================================
// EXPLAIN for cross-paradigm operators
// ==================================================================

describe("CrossParadigmExplain", () => {
  it("PlanExecutor.explain for ScoreOperator", () => {
    const term = new TermOperator("test", "body");
    const scorer = new DummyScorer();
    const scoreOp = new ScoreOperator(scorer as never, term, ["test"], "body");
    const executor = new PlanExecutor({});
    const explained = executor.explain(scoreOp);
    expect(explained).toContain("Score");
    expect(explained).toContain("TermScan");
  });

  it("PlanExecutor.explain for LogOddsFusion", () => {
    const term1 = new TermOperator("hello", "body");
    const term2 = new TermOperator("world", "body");
    const fusion = new LogOddsFusionOperator([term1, term2], 0.5);
    const executor = new PlanExecutor({});
    const explained = executor.explain(fusion);
    expect(explained).toContain("LogOddsFusionOperator");
  });

  it("PlanExecutor.explain for ProbBoolFusion", () => {
    const term1 = new TermOperator("hello", "body");
    const term2 = new TermOperator("world", "body");
    const fusion = new ProbBoolFusionOperator([term1, term2], "and");
    const executor = new PlanExecutor({});
    const explained = executor.explain(fusion);
    expect(explained).toContain("ProbBoolFusionOperator");
  });

  it("PlanExecutor.explain for ProbNot", () => {
    const term = new TermOperator("hello", "body");
    const probNot = new ProbNotOperator(term, 0.01);
    const executor = new PlanExecutor({});
    const explained = executor.explain(probNot);
    expect(explained).toContain("ProbNotOperator");
  });

  it("PlanExecutor.explain for TraverseOp", () => {
    const traverse = new TraverseOperator(1, "test", "knows", 2);
    const executor = new PlanExecutor({});
    const explained = executor.explain(traverse);
    expect(explained).toContain("TraverseOperator");
    expect(explained).toContain("graph=test");
  });

  it("PlanExecutor.explain for PatternMatchOp", () => {
    const pattern = createGraphPattern(
      [createVertexPattern("a", []), createVertexPattern("b", [])],
      [createEdgePattern("a", "b", { label: "knows" })],
    );
    const pm = new PatternMatchOperator(pattern, "test");
    const executor = new PlanExecutor({});
    const explained = executor.explain(pm);
    expect(explained).toContain("PatternMatchOperator");
    expect(explained).toContain("graph=test");
  });

  it("PlanExecutor.explain for RPQOp", () => {
    const rpq = new RegularPathQueryOperator(new Label("knows"), "test", 1);
    const executor = new PlanExecutor({});
    const explained = executor.explain(rpq);
    expect(explained).toContain("RegularPathQueryOperator");
    expect(explained).toContain("graph=test");
  });
});

// ==================================================================
// End-to-end optimizer correctness
// ==================================================================

describe("CrossParadigmOptimizerCorrectness", () => {
  it("fusion reorder preserves operator type", () => {
    const stats = new IndexStats(1000, 16);
    const term1 = new TermOperator("hello", "body");
    const term2 = new TermOperator("world", "body");
    // Create fusion with specific order
    const fusionOrig = new LogOddsFusionOperator([term1, term2], 0.5);
    const optimizer = new QueryOptimizer(stats);
    const optimized = optimizer.optimize(fusionOrig);
    // After optimization, should still be LogOddsFusionOperator
    expect(optimized).toBeInstanceOf(LogOddsFusionOperator);
    // Alpha should be preserved
    expect((optimized as LogOddsFusionOperator).alpha).toBe(0.5);
    // Signal count should be preserved
    expect((optimized as LogOddsFusionOperator).signals.length).toBe(2);
  });

  it("intersect reorder with mixed paradigms", () => {
    const stats = new IndexStats(1000, 16);
    const term = new TermOperator("hello", "body");
    const filter = new FilterOperator("category", new Equals("ml"));
    const intersect = new IntersectOperator([filter, term]);
    const optimizer = new QueryOptimizer(stats);
    const optimized = optimizer.optimize(intersect);
    // The optimizer should produce a valid operator tree
    expect(optimized).toBeTruthy();
    // The result should still contain both operand types
    if (optimized instanceof IntersectOperator) {
      expect(optimized.operands.length).toBe(2);
    }
  });
});

// ==================================================================
// SQL-level integration tests
// ==================================================================

describe("SQLCrossParadigmOptimizer", () => {
  it("explain shows plan for text query", async () => {
    const { Engine } = await import("../../src/engine.js");
    const e = new Engine();
    await e.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT, body TEXT)");
    for (let i = 1; i <= 5; i++) {
      await e.sql(
        `INSERT INTO docs (id, title, body) VALUES ` +
          `(${i}, 'neural network ${i}', 'deep learning paper ${i}')`,
      );
    }
    const r = await e.sql(
      "EXPLAIN SELECT title FROM docs WHERE title = 'neural network 1'",
    );
    const planText = (r?.rows ?? [])
      .map((row: Record<string, unknown>) =>
        String(row["QUERY PLAN"] ?? row["plan"] ?? ""),
      )
      .join(" ");
    expect(planText.length).toBeGreaterThan(0);
  });

  it("explain shows plan for traverse query", async () => {
    const { Engine } = await import("../../src/engine.js");
    const e = new Engine();
    await e.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, name TEXT)");
    for (let i = 1; i <= 3; i++) {
      await e.sql(`INSERT INTO docs (id, name) VALUES (${i}, 'v${i}')`);
    }
    const r = await e.sql("EXPLAIN SELECT name FROM docs WHERE id = 1");
    const planText = (r?.rows ?? [])
      .map((row: Record<string, unknown>) =>
        String(row["QUERY PLAN"] ?? row["plan"] ?? ""),
      )
      .join(" ");
    expect(planText.length).toBeGreaterThan(0);
  });
});
