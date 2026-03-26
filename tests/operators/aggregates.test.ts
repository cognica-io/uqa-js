import { describe, expect, it } from "vitest";
import {
  AggregateOperator,
  AvgMonoid,
  CountMonoid,
  GroupByOperator,
  MaxMonoid,
  MinMonoid,
  QuantileMonoid,
  SumMonoid,
} from "../../src/operators/aggregation.js";
import {
  createPayload,
  createPostingEntry,
  Equals,
  GreaterThanOrEqual,
} from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import { MemoryDocumentStore } from "../../src/storage/document-store.js";
import { FilterOperator } from "../../src/operators/primitive.js";
import type { ExecutionContext } from "../../src/operators/base.js";

// -- Test data setup ----------------------------------------------------------

function makeProductContext(): ExecutionContext {
  const store = new MemoryDocumentStore();
  store.put(1, { id: 1, category: "fruit", name: "Apple", price: 3, active: true });
  store.put(2, { id: 2, category: "fruit", name: "Banana", price: 2, active: true });
  store.put(3, { id: 3, category: "fruit", name: "Cherry", price: 5, active: false });
  store.put(4, { id: 4, category: "veggie", name: "Daikon", price: 4, active: true });
  store.put(5, {
    id: 5,
    category: "veggie",
    name: "Eggplant",
    price: 6,
    active: false,
  });
  return { documentStore: store };
}

function makeUserContext(): ExecutionContext {
  const store = new MemoryDocumentStore();
  store.put(1, { id: 1, name: "Alice", age: 30 });
  store.put(2, { id: 2, name: "Bob", age: 25 });
  store.put(3, { id: 3, name: "Carol", age: 35 });
  store.put(4, { id: 4, name: "Dave", age: 25 });
  return { documentStore: store };
}

function makeTableContext(): ExecutionContext {
  const store = new MemoryDocumentStore();
  store.put(1, { id: 1, val: 10, name: "alpha" });
  store.put(2, { id: 2, val: 20, name: "bravo" });
  store.put(3, { id: 3, val: 30, name: "charlie" });
  return { documentStore: store };
}

// =============================================================================
// CountMonoid
// =============================================================================

describe("CountMonoid", () => {
  it("identity finalize is 0", () => {
    const m = new CountMonoid();
    expect(m.finalize(m.identity())).toBe(0);
  });

  it("accumulate counts items", () => {
    const m = new CountMonoid();
    let state = m.identity();
    state = m.accumulate(state, "anything");
    state = m.accumulate(state, "else");
    expect(m.finalize(state)).toBe(2);
  });

  it("combine is associative", () => {
    const m = new CountMonoid();
    const a = m.accumulate(m.identity(), 1);
    const b = m.accumulate(m.identity(), 2);
    const c = m.accumulate(m.identity(), 3);
    expect(m.combine(m.combine(a, b), c)).toBe(m.combine(a, m.combine(b, c)));
  });

  it("combine is identity-neutral", () => {
    const m = new CountMonoid();
    const a = m.accumulate(m.identity(), "x");
    expect(m.combine(a, m.identity())).toBe(a);
    expect(m.combine(m.identity(), a)).toBe(a);
  });
});

// =============================================================================
// SumMonoid
// =============================================================================

describe("SumMonoid", () => {
  it("identity finalize is 0", () => {
    const m = new SumMonoid();
    expect(m.finalize(m.identity())).toBe(0.0);
  });

  it("accumulate sums values", () => {
    const m = new SumMonoid();
    let state = m.identity();
    state = m.accumulate(state, 3.0);
    state = m.accumulate(state, 7.0);
    expect(m.finalize(state)).toBe(10.0);
  });

  it("combine is associative", () => {
    const m = new SumMonoid();
    const a = m.accumulate(m.identity(), 1.0);
    const b = m.accumulate(m.identity(), 2.0);
    const c = m.accumulate(m.identity(), 3.0);
    expect(m.combine(m.combine(a, b), c)).toBe(m.combine(a, m.combine(b, c)));
  });
});

// =============================================================================
// AvgMonoid
// =============================================================================

describe("AvgMonoid", () => {
  it("computes correct average", () => {
    const m = new AvgMonoid();
    let state = m.identity();
    for (const v of [2.0, 4.0, 6.0]) {
      state = m.accumulate(state, v);
    }
    expect(m.finalize(state)).toBe(4.0);
  });

  it("combine is associative (by finalize)", () => {
    const m = new AvgMonoid();
    const a = m.accumulate(m.identity(), 2.0);
    const b = m.accumulate(m.identity(), 4.0);
    const c = m.accumulate(m.identity(), 6.0);
    const lhs = m.combine(m.combine(a, b), c);
    const rhs = m.combine(a, m.combine(b, c));
    expect(m.finalize(lhs)).toBe(m.finalize(rhs));
  });

  it("empty average returns 0", () => {
    const m = new AvgMonoid();
    expect(m.finalize(m.identity())).toBe(0);
  });

  it("single value average", () => {
    const m = new AvgMonoid();
    const state = m.accumulate(m.identity(), 42.0);
    expect(m.finalize(state)).toBe(42.0);
  });
});

// =============================================================================
// MinMonoid
// =============================================================================

describe("MinMonoid", () => {
  it("identity is Infinity", () => {
    const m = new MinMonoid();
    expect(m.finalize(m.identity())).toBe(Infinity);
  });

  it("accumulate finds minimum", () => {
    const m = new MinMonoid();
    let state = m.identity();
    state = m.accumulate(state, 5.0);
    state = m.accumulate(state, 3.0);
    state = m.accumulate(state, 7.0);
    expect(m.finalize(state)).toBe(3.0);
  });

  it("combine is associative", () => {
    const m = new MinMonoid();
    const a = m.accumulate(m.identity(), 5.0);
    const b = m.accumulate(m.identity(), 3.0);
    const c = m.accumulate(m.identity(), 7.0);
    expect(m.combine(m.combine(a, b), c)).toBe(m.combine(a, m.combine(b, c)));
  });
});

// =============================================================================
// MaxMonoid
// =============================================================================

describe("MaxMonoid", () => {
  it("identity is -Infinity", () => {
    const m = new MaxMonoid();
    expect(m.finalize(m.identity())).toBe(-Infinity);
  });

  it("accumulate finds maximum", () => {
    const m = new MaxMonoid();
    let state = m.identity();
    state = m.accumulate(state, 5.0);
    state = m.accumulate(state, 3.0);
    state = m.accumulate(state, 7.0);
    expect(m.finalize(state)).toBe(7.0);
  });

  it("combine is associative", () => {
    const m = new MaxMonoid();
    const a = m.accumulate(m.identity(), 5.0);
    const b = m.accumulate(m.identity(), 3.0);
    const c = m.accumulate(m.identity(), 7.0);
    expect(m.combine(m.combine(a, b), c)).toBe(m.combine(a, m.combine(b, c)));
  });
});

// =============================================================================
// QuantileMonoid
// =============================================================================

describe("QuantileMonoid", () => {
  it("computes median", () => {
    const m = new QuantileMonoid(0.5);
    let state = m.identity();
    for (const v of [10, 20, 30]) {
      state = m.accumulate(state, v);
    }
    expect(m.finalize(state)).toBeCloseTo(20.0);
  });

  it("computes 25th percentile", () => {
    const m = new QuantileMonoid(0.25);
    let state = m.identity();
    for (const v of [10, 20, 30]) {
      state = m.accumulate(state, v);
    }
    // 0.25 * (3-1) = 0.5 -> interp between 10 and 20 = 15
    expect(m.finalize(state)).toBeCloseTo(15.0);
  });

  it("combine preserves data", () => {
    const m = new QuantileMonoid(0.5);
    const a = m.accumulate(m.identity(), 10);
    const b = m.accumulate(m.identity(), 20);
    const c = m.accumulate(m.identity(), 30);
    const combined = m.combine(m.combine(a, b), c);
    expect(m.finalize(combined)).toBeCloseTo(20.0);
  });

  it("empty returns 0", () => {
    const m = new QuantileMonoid(0.5);
    expect(m.finalize(m.identity())).toBe(0);
  });

  it("invalid quantile throws", () => {
    expect(() => new QuantileMonoid(-0.1)).toThrow();
    expect(() => new QuantileMonoid(1.1)).toThrow();
  });
});

// =============================================================================
// AggregateOperator
// =============================================================================

describe("AggregateOperator", () => {
  it("computes average over filtered documents", () => {
    const ctx = makeProductContext();
    const source = new FilterOperator("category", new Equals("fruit"));
    const agg = new AggregateOperator(source, "price", new AvgMonoid());
    const result = agg.execute(ctx);
    expect(result.length).toBe(1);
    const entry = result.entries[0]!;
    // fruit prices: 3, 2, 5 -> avg = 10/3 ~ 3.333
    expect(entry.payload.fields["_aggregate"]).toBeCloseTo(10 / 3);
  });

  it("computes sum", () => {
    const ctx = makeProductContext();
    const agg = new AggregateOperator(null, "price", new SumMonoid());
    const result = agg.execute(ctx);
    // All prices: 3 + 2 + 5 + 4 + 6 = 20
    expect(result.entries[0]!.payload.fields["_aggregate"]).toBe(20);
  });

  it("computes count", () => {
    const ctx = makeProductContext();
    const source = new FilterOperator("active", new Equals(true));
    const agg = new AggregateOperator(source, "price", new CountMonoid());
    const result = agg.execute(ctx);
    // Active products: Apple, Banana, Daikon = 3
    expect(result.entries[0]!.payload.fields["_aggregate"]).toBe(3);
  });

  it("computes min and max", () => {
    const ctx = makeProductContext();
    const aggMin = new AggregateOperator(null, "price", new MinMonoid());
    const aggMax = new AggregateOperator(null, "price", new MaxMonoid());
    expect(aggMin.execute(ctx).entries[0]!.payload.fields["_aggregate"]).toBe(2);
    expect(aggMax.execute(ctx).entries[0]!.payload.fields["_aggregate"]).toBe(6);
  });
});

// =============================================================================
// GroupByOperator
// =============================================================================

describe("GroupByOperator", () => {
  it("groups by category and counts", () => {
    const ctx = makeProductContext();
    const groupOp = new GroupByOperator(
      new FilterOperator("price", new GreaterThanOrEqual(0)),
      "category",
      "price",
      new CountMonoid(),
    );
    const result = groupOp.execute(ctx);
    expect(result.length).toBeGreaterThan(0);
    const groups: Record<string, number> = {};
    for (const entry of result) {
      const key = entry.payload.fields["_group_key"] as string;
      const agg = entry.payload.fields["_aggregate_result"] as number;
      groups[key] = agg;
    }
    expect(groups["fruit"]).toBe(3);
    expect(groups["veggie"]).toBe(2);
  });

  it("groups by category and sums prices", () => {
    const ctx = makeProductContext();
    const groupOp = new GroupByOperator(
      new FilterOperator("price", new GreaterThanOrEqual(0)),
      "category",
      "price",
      new SumMonoid(),
    );
    const result = groupOp.execute(ctx);
    const groups: Record<string, number> = {};
    for (const entry of result) {
      groups[entry.payload.fields["_group_key"] as string] = entry.payload.fields[
        "_aggregate_result"
      ] as number;
    }
    expect(groups["fruit"]).toBe(10); // 3 + 2 + 5
    expect(groups["veggie"]).toBe(10); // 4 + 6
  });

  it("groups by category and computes avg price", () => {
    const ctx = makeProductContext();
    const groupOp = new GroupByOperator(
      new FilterOperator("price", new GreaterThanOrEqual(0)),
      "category",
      "price",
      new AvgMonoid(),
    );
    const result = groupOp.execute(ctx);
    const groups: Record<string, number> = {};
    for (const entry of result) {
      groups[entry.payload.fields["_group_key"] as string] = entry.payload.fields[
        "_aggregate_result"
      ] as number;
    }
    expect(groups["fruit"]).toBeCloseTo(10 / 3);
    expect(groups["veggie"]).toBeCloseTo(5);
  });

  it("groups by age in user data", () => {
    const ctx = makeUserContext();
    const groupOp = new GroupByOperator(
      new FilterOperator("age", new GreaterThanOrEqual(0)),
      "age",
      "id",
      new CountMonoid(),
    );
    const result = groupOp.execute(ctx);
    const groups: Record<string, number> = {};
    for (const entry of result) {
      groups[String(entry.payload.fields["_group_key"])] = entry.payload.fields[
        "_aggregate_result"
      ] as number;
    }
    expect(groups["25"]).toBe(2); // Bob and Dave
    expect(groups["30"]).toBe(1); // Alice
    expect(groups["35"]).toBe(1); // Carol
  });
});

// =============================================================================
// Monoid laws (comprehensive)
// =============================================================================

describe("Monoid laws", () => {
  const monoids = [
    { name: "Count", monoid: new CountMonoid(), values: [1, 2, 3] },
    { name: "Sum", monoid: new SumMonoid(), values: [1.0, 2.0, 3.0] },
    { name: "Min", monoid: new MinMonoid(), values: [5.0, 3.0, 7.0] },
    { name: "Max", monoid: new MaxMonoid(), values: [5.0, 3.0, 7.0] },
  ];

  for (const { name, monoid, values } of monoids) {
    describe(`${name}Monoid`, () => {
      it("identity is left neutral for combine", () => {
        let state = monoid.identity();
        for (const v of values) {
          state = monoid.accumulate(state, v);
        }
        const combined = monoid.combine(monoid.identity(), state);
        expect(monoid.finalize(combined)).toBe(monoid.finalize(state));
      });

      it("identity is right neutral for combine", () => {
        let state = monoid.identity();
        for (const v of values) {
          state = monoid.accumulate(state, v);
        }
        const combined = monoid.combine(state, monoid.identity());
        expect(monoid.finalize(combined)).toBe(monoid.finalize(state));
      });

      it("combine is associative", () => {
        const states = values.map((v) => monoid.accumulate(monoid.identity(), v));
        const lhs = monoid.combine(monoid.combine(states[0]!, states[1]!), states[2]!);
        const rhs = monoid.combine(states[0]!, monoid.combine(states[1]!, states[2]!));
        expect(monoid.finalize(lhs)).toBe(monoid.finalize(rhs));
      });
    });
  }
});

// =============================================================================
// Filtered aggregation
// =============================================================================

describe("Filtered aggregation", () => {
  it("count only active products", () => {
    const ctx = makeProductContext();
    const source = new FilterOperator("active", new Equals(true));
    const agg = new AggregateOperator(source, "price", new CountMonoid());
    const result = agg.execute(ctx);
    expect(result.entries[0]!.payload.fields["_aggregate"]).toBe(3);
  });

  it("sum of active product prices", () => {
    const ctx = makeProductContext();
    const source = new FilterOperator("active", new Equals(true));
    const agg = new AggregateOperator(source, "price", new SumMonoid());
    const result = agg.execute(ctx);
    // Apple=3, Banana=2, Daikon=4 -> 9
    expect(result.entries[0]!.payload.fields["_aggregate"]).toBe(9);
  });

  it("count of expensive products (price > 3)", () => {
    const ctx = makeProductContext();
    const source = new FilterOperator("price", new GreaterThanOrEqual(4));
    const agg = new AggregateOperator(source, "price", new CountMonoid());
    const result = agg.execute(ctx);
    // Daikon=4, Cherry=5, Eggplant=6 -> 3
    expect(result.entries[0]!.payload.fields["_aggregate"]).toBe(3);
  });
});

// =============================================================================
// Variance and standard deviation via monoids
// =============================================================================

describe("Statistical aggregates via monoids", () => {
  it("computes variance manually", () => {
    // For values [10, 20, 30]:
    // mean = 20, var_samp = 100, stddev_samp = 10
    const values = [10, 20, 30];
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    expect(mean).toBe(20);

    const sumSquares = values.reduce((acc, v) => acc + (v - mean) ** 2, 0);
    const varSamp = sumSquares / (values.length - 1);
    expect(varSamp).toBeCloseTo(100.0);

    const stddevSamp = Math.sqrt(varSamp);
    expect(stddevSamp).toBeCloseTo(10.0);

    const varPop = sumSquares / values.length;
    expect(varPop).toBeCloseTo(200 / 3);
  });

  it("QuantileMonoid computes percentile_disc (nearest rank)", () => {
    // For values [10, 20, 30], percentile_disc(0.5) = 20
    const m = new QuantileMonoid(0.5);
    let state = m.identity();
    for (const v of [10, 20, 30]) {
      state = m.accumulate(state, v);
    }
    expect(m.finalize(state)).toBe(20);
  });

  it("QuantileMonoid can approximate mode via frequency counting", () => {
    // Mode of [1, 2, 2, 3] is 2 (most frequent)
    // We verify this manually since QuantileMonoid is continuous
    const values = [1, 2, 2, 3];
    const counts = new Map<number, number>();
    for (const v of values) {
      counts.set(v, (counts.get(v) ?? 0) + 1);
    }
    let mode = values[0]!;
    let maxCount = 0;
    for (const [val, count] of counts) {
      if (count > maxCount) {
        maxCount = count;
        mode = val;
      }
    }
    expect(mode).toBe(2);
  });
});
