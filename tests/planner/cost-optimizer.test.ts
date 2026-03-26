import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";
import {
  IndexStats,
  Equals,
  Between,
  GreaterThan,
  LessThan,
} from "../../src/core/types.js";
import { CardinalityEstimator } from "../../src/planner/cardinality.js";
import { FilterOperator } from "../../src/operators/primitive.js";
import type { ColumnStats } from "../../src/sql/table.js";
import { Table } from "../../src/sql/table.js";

// ==================================================================
// Histogram construction
// ==================================================================

describe("Histogram", () => {
  it("histogram basic via Table.analyze", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
    for (let i = 1; i <= 10; i++) {
      await engine.sql(`INSERT INTO t (id, val) VALUES (${i}, ${i})`);
    }
    const table = engine.getTable("t");
    table.analyze();
    const stats = table.getColumnStats("val")!;
    expect(stats.histogram.length).toBeGreaterThanOrEqual(2);
    expect(stats.histogram[0]).toBe(1);
    expect(stats.histogram[stats.histogram.length - 1]).toBe(10);
  });

  it("histogram empty table", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
    const table = engine.getTable("t");
    table.analyze();
    const stats = table.getColumnStats("val")!;
    expect(stats.histogram.length).toBe(0);
  });

  it("histogram single value", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
    await engine.sql("INSERT INTO t (id, val) VALUES (1, 42)");
    const table = engine.getTable("t");
    table.analyze();
    const stats = table.getColumnStats("val")!;
    expect(stats.histogram).toEqual([42, 42]);
  });

  it("histogram duplicates", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
    const vals = [1, 1, 1, 1, 2, 2, 3];
    for (let i = 0; i < vals.length; i++) {
      await engine.sql(`INSERT INTO t (id, val) VALUES (${i + 1}, ${vals[i]})`);
    }
    const table = engine.getTable("t");
    table.analyze();
    const stats = table.getColumnStats("val")!;
    expect(stats.histogram[0]).toBe(1);
    expect(stats.histogram[stats.histogram.length - 1]).toBe(3);
  });

  it("histogram strings", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)");
    const vals = ["a", "b", "c", "d", "e"];
    for (let i = 0; i < vals.length; i++) {
      await engine.sql(`INSERT INTO t (id, val) VALUES (${i + 1}, '${vals[i]}')`);
    }
    const table = engine.getTable("t");
    table.analyze();
    const stats = table.getColumnStats("val")!;
    expect(stats.histogram[0]).toBe("a");
    expect(stats.histogram[stats.histogram.length - 1]).toBe("e");
  });
});

// ==================================================================
// MCV construction
// ==================================================================

describe("MCV", () => {
  it("mcv skewed data", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, cat TEXT)");
    // 'x' appears 50 times, 'y' 5 times, others once each
    for (let i = 1; i <= 50; i++) {
      await engine.sql(`INSERT INTO t (id, cat) VALUES (${i}, 'x')`);
    }
    for (let i = 51; i <= 55; i++) {
      await engine.sql(`INSERT INTO t (id, cat) VALUES (${i}, 'y')`);
    }
    for (let i = 56; i <= 100; i++) {
      await engine.sql(`INSERT INTO t (id, cat) VALUES (${i}, 'z${i}')`);
    }
    const table = engine.getTable("t");
    table.analyze();
    const stats = table.getColumnStats("cat")!;
    expect(stats.mcvValues).toContain("x");
    const idxX = (stats.mcvValues as string[]).indexOf("x");
    expect(stats.mcvFrequencies[idxX]).toBeCloseTo(0.5, 1);
  });

  it("mcv uniform data", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
    // All values appear once -- no MCV above average
    for (let i = 1; i <= 100; i++) {
      await engine.sql(`INSERT INTO t (id, val) VALUES (${i}, ${i})`);
    }
    const table = engine.getTable("t");
    table.analyze();
    const stats = table.getColumnStats("val")!;
    expect(stats.mcvValues.length).toBe(0);
    expect(stats.mcvFrequencies.length).toBe(0);
  });

  it("mcv empty table", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
    const table = engine.getTable("t");
    table.analyze();
    const stats = table.getColumnStats("val")!;
    expect(stats.mcvValues.length).toBe(0);
  });
});

// ==================================================================
// ANALYZE integration
// ==================================================================

describe("AnalyzeHistogramMCV", () => {
  it("analyze creates histogram via SQL", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
    for (let i = 1; i <= 100; i++) {
      await engine.sql(`INSERT INTO t (id, val) VALUES (${i}, ${i})`);
    }
    await engine.sql("ANALYZE t");
    const table = engine.getTable("t");
    const stats = table.getColumnStats("val")!;
    expect(stats.histogram.length).toBeGreaterThanOrEqual(2);
    expect(stats.histogram[0]).toBe(1);
    expect(stats.histogram[stats.histogram.length - 1]).toBe(100);
  });

  it("analyze creates mcv via SQL", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, cat TEXT)");
    for (let i = 1; i <= 100; i++) {
      const cat = i <= 50 ? "A" : i <= 80 ? "B" : "C";
      await engine.sql(`INSERT INTO t (id, cat) VALUES (${i}, '${cat}')`);
    }
    await engine.sql("ANALYZE t");
    const table = engine.getTable("t");
    const stats = table.getColumnStats("cat")!;
    expect(stats.mcvValues).toContain("A");
    const idxA = (stats.mcvValues as string[]).indexOf("A");
    expect(Math.abs(stats.mcvFrequencies[idxA]! - 0.5)).toBeLessThan(0.01);
  });

  it("analyze without table name analyzes all tables", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val INTEGER)");
    await engine.sql("CREATE TABLE t2 (id INTEGER PRIMARY KEY, val INTEGER)");
    for (let i = 1; i <= 10; i++) {
      await engine.sql(`INSERT INTO t1 (id, val) VALUES (${i}, ${i})`);
      await engine.sql(`INSERT INTO t2 (id, val) VALUES (${i}, ${i * 10})`);
    }
    await engine.sql("ANALYZE");
    const stats1 = engine.getTable("t1").getColumnStats("val")!;
    const stats2 = engine.getTable("t2").getColumnStats("val")!;
    expect(stats1.histogram.length).toBeGreaterThanOrEqual(2);
    expect(stats2.histogram.length).toBeGreaterThanOrEqual(2);
  });
});

// ==================================================================
// Selectivity estimation with histograms/MCVs
// ==================================================================

describe("SelectivityEstimation", () => {
  function makeEstimatorWithStats(): CardinalityEstimator {
    const cs: ColumnStats = {
      distinctCount: 100,
      nullCount: 0,
      minValue: 1,
      maxValue: 1000,
      rowCount: 1000,
      histogram: [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
      mcvValues: [42, 100],
      mcvFrequencies: [0.15, 0.1],
    };
    return new CardinalityEstimator({ columnStats: new Map([["val", cs]]) });
  }

  it("CardinalityEstimator with columnStats histogram/MCV", () => {
    const est = makeEstimatorWithStats();
    const stats = new IndexStats(1000);
    // A filter on "val" should use the column stats
    const filter = new FilterOperator("val", new Equals(42));
    const card = est.estimate(filter, stats);
    expect(card).toBeGreaterThan(0);
    expect(card).toBeLessThanOrEqual(1000);
  });

  it("equality mcv hit", () => {
    const est = makeEstimatorWithStats();
    const stats = new IndexStats(1000);
    // Value 42 has MCV frequency 0.15
    const filter = new FilterOperator("val", new Equals(42));
    const card = est.estimate(filter, stats);
    // Expected: 1000 * 0.15 = 150 (approximately, with entropy clamping)
    expect(card).toBeGreaterThan(50);
    expect(card).toBeLessThanOrEqual(200);
  });

  it("equality mcv miss", () => {
    const est = makeEstimatorWithStats();
    const stats = new IndexStats(1000);
    // Value 999 is not in MCV, falls back to 1/ndv = 1/100 = 0.01
    const filter = new FilterOperator("val", new Equals(999));
    const card = est.estimate(filter, stats);
    // 1000 * (1/100) = 10, possibly clamped by entropy
    expect(card).toBeGreaterThan(0);
    expect(card).toBeLessThanOrEqual(100);
  });

  it("range histogram", () => {
    const est = makeEstimatorWithStats();
    const stats = new IndexStats(1000);
    // Range [100, 500] covers roughly 4 out of 10 histogram buckets
    const filter = new FilterOperator("val", new Between(100, 500));
    const card = est.estimate(filter, stats);
    expect(card).toBeGreaterThan(100);
    expect(card).toBeLessThanOrEqual(600);
  });

  it("gt histogram", () => {
    const est = makeEstimatorWithStats();
    const stats = new IndexStats(1000);
    // GT 500 covers roughly half the histogram
    const filter = new FilterOperator("val", new GreaterThan(500));
    const card = est.estimate(filter, stats);
    expect(card).toBeGreaterThan(100);
    expect(card).toBeLessThanOrEqual(700);
  });

  it("lt histogram", () => {
    const est = makeEstimatorWithStats();
    const stats = new IndexStats(1000);
    // LT 200 covers roughly 2 out of 10 histogram buckets
    const filter = new FilterOperator("val", new LessThan(200));
    const card = est.estimate(filter, stats);
    expect(card).toBeGreaterThan(50);
    expect(card).toBeLessThanOrEqual(400);
  });

  it("no stats fallback", () => {
    // Without column stats, should fall back to default selectivity
    const est = new CardinalityEstimator();
    const stats = new IndexStats(1000);
    const filter = new FilterOperator("unknown_field", new Equals(42));
    const card = est.estimate(filter, stats);
    // Default selectivity is 0.5
    expect(card).toBeGreaterThan(0);
    expect(card).toBeLessThanOrEqual(1000);
  });
});

// ==================================================================
// End-to-end optimizer with ANALYZE
// ==================================================================

describe("OptimizerEndToEnd", () => {
  it("analyze improves explain", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
    for (let i = 1; i <= 50; i++) {
      await engine.sql(`INSERT INTO t (id, val) VALUES (${i}, ${i})`);
    }
    await engine.sql("ANALYZE t");
    const r = await engine.sql("EXPLAIN SELECT val FROM t WHERE val > 25");
    const planText = (r?.rows ?? [])
      .map((row: Record<string, unknown>) =>
        String(row["QUERY PLAN"] ?? row["plan"] ?? ""),
      )
      .join(" ");
    // After ANALYZE, the plan should contain cost info
    expect(planText.length).toBeGreaterThan(0);
  });

  it("histogram data accessible after analyze", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
    for (let i = 1; i <= 100; i++) {
      await engine.sql(`INSERT INTO t (id, val) VALUES (${i}, ${i})`);
    }
    await engine.sql("ANALYZE t");
    const stats = engine.getTable("t").getColumnStats("val")!;
    expect(stats.histogram.length).toBeGreaterThanOrEqual(2);
    expect(stats.histogram[0]).toBe(1);
    expect(stats.histogram[stats.histogram.length - 1]).toBe(100);
  });

  it("mcv data accessible after analyze", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, cat TEXT)");
    for (let i = 1; i <= 100; i++) {
      const cat = i <= 60 ? "A" : "B";
      await engine.sql(`INSERT INTO t (id, cat) VALUES (${i}, '${cat}')`);
    }
    await engine.sql("ANALYZE t");
    const stats = engine.getTable("t").getColumnStats("cat")!;
    expect(stats.mcvValues).toContain("A");
  });

  it("column stats row count matches", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
    for (let i = 1; i <= 50; i++) {
      await engine.sql(`INSERT INTO t (id, val) VALUES (${i}, ${i})`);
    }
    await engine.sql("ANALYZE t");
    const stats = engine.getTable("t").getColumnStats("val")!;
    expect(stats.rowCount).toBe(50);
    expect(stats.distinctCount).toBe(50);
    expect(stats.nullCount).toBe(0);
  });
});
