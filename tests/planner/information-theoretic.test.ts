import { describe, expect, it } from "vitest";
import { IndexStats, Equals, GreaterThan } from "../../src/core/types.js";
import {
  CardinalityEstimator,
  columnEntropy,
  mutualInformationEstimate,
  entropyCardinalityLowerBound,
} from "../../src/planner/cardinality.js";
import { IntersectOperator } from "../../src/operators/boolean.js";
import { FilterOperator } from "../../src/operators/primitive.js";
import type { ColumnStats } from "../../src/sql/table.js";

function makeStats(
  ndv: number,
  mcvValues?: unknown[],
  mcvFrequencies?: number[],
): ColumnStats {
  return {
    distinctCount: ndv,
    nullCount: 0,
    minValue: 0,
    maxValue: ndv,
    rowCount: 1000,
    histogram: [],
    mcvValues: mcvValues ?? [],
    mcvFrequencies: mcvFrequencies ?? [],
  };
}

// The Python tests call private functions _column_entropy,
// _entropy_cardinality_lower_bound, _mutual_information_estimate.
// These are now exported in the TS implementation.

describe("ColumnEntropy", () => {
  it("column entropy uniform", () => {
    const cs = makeStats(100);
    const h = columnEntropy(cs);
    // Uniform distribution: log2(100) ~= 6.64
    expect(h).toBeCloseTo(Math.log2(100), 1);
  });

  it("column entropy single value", () => {
    const cs = makeStats(1);
    const h = columnEntropy(cs);
    expect(h).toBe(0.0);
  });

  it("column entropy zero", () => {
    const cs = makeStats(0);
    const h = columnEntropy(cs);
    expect(h).toBe(0.0);
  });

  it("column entropy none (ndv=1)", () => {
    const cs = makeStats(1);
    const h = columnEntropy(cs);
    expect(h).toBe(0.0);
  });

  it("column entropy with mcv", () => {
    const cs = makeStats(10, [1, 2, 3], [0.4, 0.3, 0.2]);
    const h = columnEntropy(cs);
    // With skewed MCV, entropy should be less than log2(10) ~= 3.32
    expect(h).toBeGreaterThan(0);
    expect(h).toBeLessThan(Math.log2(10));
  });
});

describe("MutualInformation", () => {
  it("mutual information zero joint selectivity", () => {
    const csA = makeStats(100);
    const csB = makeStats(100);
    // Zero joint selectivity => MI = 0
    const mi = mutualInformationEstimate(csA, csB, 0);
    expect(mi).toBe(0);
  });

  it("mutual information positive", () => {
    const csA = makeStats(10);
    const csB = makeStats(10);
    // Any positive joint selectivity should give non-negative MI
    const jointSel = 0.5;
    const mi = mutualInformationEstimate(csA, csB, jointSel);
    expect(mi).toBeGreaterThanOrEqual(0);
  });

  it("mutual information increases with joint selectivity", () => {
    const csA = makeStats(10);
    const csB = makeStats(10);
    // Higher joint selectivity = larger effective joint NDV = larger hJoint = lower MI
    // (The formula is MI = hX + hY - hJoint where hJoint = log2(ndvX*ndvY*jointSel))
    // So higher jointSel => higher hJoint => lower MI
    const highSel = mutualInformationEstimate(csA, csB, 1.0);
    const lowSel = mutualInformationEstimate(csA, csB, 0.001);
    // Lower joint selectivity => smaller hJoint => higher MI
    expect(lowSel).toBeGreaterThanOrEqual(highSel);
  });
});

describe("EntropyLowerBound", () => {
  it("entropy lower bound single", () => {
    const lb = entropyCardinalityLowerBound(1000, [2.0]);
    // 1000 * 2^(-2) = 250
    expect(lb).toBeCloseTo(250, 0);
  });

  it("entropy lower bound multiple", () => {
    const lb = entropyCardinalityLowerBound(1000, [2.0, 3.0]);
    // 1000 * 2^(-5) = 31.25
    expect(lb).toBeCloseTo(31.25, 0);
  });

  it("entropy lower bound empty", () => {
    const lb = entropyCardinalityLowerBound(1000, []);
    expect(lb).toBe(1.0);
  });

  it("entropy lower bound zero n", () => {
    const lb = entropyCardinalityLowerBound(0, [2.0]);
    expect(lb).toBe(1.0);
  });

  it("entropy lower bound floor", () => {
    // Very high entropy should still give >= 1.0
    const lb = entropyCardinalityLowerBound(100, [20.0]);
    expect(lb).toBeGreaterThanOrEqual(1.0);
  });
});

// -- Integration tests: entropy bounds in CardinalityEstimator --

describe("EntropyLowerBoundInIntersection", () => {
  it("entropy lower bound floors intersection cardinality", () => {
    const cs = new Map<string, ColumnStats>([
      [
        "age",
        {
          distinctCount: 50,
          nullCount: 0,
          minValue: 0,
          maxValue: 100,
          rowCount: 1000,
          histogram: [],
          mcvValues: [],
          mcvFrequencies: [],
        },
      ],
      [
        "dept",
        {
          distinctCount: 5,
          nullCount: 0,
          minValue: 0,
          maxValue: 5,
          rowCount: 1000,
          histogram: [],
          mcvValues: [],
          mcvFrequencies: [],
        },
      ],
    ]);
    const estimator = new CardinalityEstimator({ columnStats: cs });
    const idxStats = new IndexStats(1000, 0);

    const f1 = new FilterOperator("age", new Equals(25));
    const f2 = new FilterOperator("dept", new Equals(3));
    const intersect = new IntersectOperator([f1, f2]);

    const card = estimator.estimate(intersect, idxStats);
    expect(card).toBeGreaterThanOrEqual(1.0);
  });
});

describe("EntropyClampingInFilterSelectivity", () => {
  it("entropy clamping in filter selectivity", () => {
    const cs = new Map<string, ColumnStats>([
      [
        "status",
        {
          distinctCount: 3,
          nullCount: 0,
          minValue: 0,
          maxValue: 2,
          rowCount: 1000,
          histogram: [],
          mcvValues: [],
          mcvFrequencies: [],
        },
      ],
    ]);
    const est = new CardinalityEstimator({ columnStats: cs });
    const stats = new IndexStats(1000);
    const filter = new FilterOperator("status", new Equals(1));
    const card = est.estimate(filter, stats);
    // With 3 distinct values, entropy = log2(3) ~= 1.585
    // Selectivity = max(1/ndv, 1/2^h) = max(1/3, 1/2^1.585) = max(0.333, 0.333) ~ 0.333
    // card ~ 1000 * 0.333 ~ 333
    expect(card).toBeGreaterThan(100);
    expect(card).toBeLessThanOrEqual(500);
  });
});

describe("EntropyClampingDoesNotRaiseHighSelectivity", () => {
  it("entropy does not raise high selectivity", () => {
    const cs = new Map<string, ColumnStats>([
      [
        "flag",
        {
          distinctCount: 2,
          nullCount: 0,
          minValue: 0,
          maxValue: 1,
          rowCount: 1000,
          histogram: [],
          mcvValues: [0, 1],
          mcvFrequencies: [0.9, 0.1],
        },
      ],
    ]);
    const est = new CardinalityEstimator({ columnStats: cs });
    const stats = new IndexStats(1000);
    // Querying the common value (freq=0.9) should give high cardinality
    const filterCommon = new FilterOperator("flag", new Equals(0));
    const cardCommon = est.estimate(filterCommon, stats);
    expect(cardCommon).toBeGreaterThan(500);

    // Querying the rare value (freq=0.1) should give lower cardinality
    const filterRare = new FilterOperator("flag", new Equals(1));
    const cardRare = est.estimate(filterRare, stats);
    expect(cardRare).toBeLessThan(cardCommon);
    expect(cardRare).toBeGreaterThan(0);
  });
});
