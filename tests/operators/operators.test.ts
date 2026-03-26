import { describe, expect, it } from "vitest";
import { createPayload } from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import { MemoryDocumentStore } from "../../src/storage/document-store.js";
import { MemoryInvertedIndex } from "../../src/storage/inverted-index.js";
import { FlatVectorIndex } from "../../src/storage/vector-index.js";
import type { ExecutionContext, Operator } from "../../src/operators/base.js";
import { ComposedOperator } from "../../src/operators/base.js";
import {
  FacetOperator,
  FilterOperator,
  KNNOperator,
  TermOperator,
  VectorSimilarityOperator,
} from "../../src/operators/primitive.js";
import {
  ComplementOperator,
  IntersectOperator,
  UnionOperator,
} from "../../src/operators/boolean.js";
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
import { SparseThresholdOperator } from "../../src/operators/sparse.js";
import { Equals, GreaterThanOrEqual } from "../../src/core/types.js";

function makeContext(): ExecutionContext {
  const docStore = new MemoryDocumentStore();
  docStore.put(1, { title: "machine learning basics", year: 2020, category: "ai" });
  docStore.put(2, {
    title: "deep learning neural networks",
    year: 2021,
    category: "ai",
  });
  docStore.put(3, { title: "web development guide", year: 2019, category: "web" });
  docStore.put(4, { title: "machine learning advanced", year: 2022, category: "ai" });

  const idx = new MemoryInvertedIndex();
  idx.addDocument(1, { title: "machine learning basics" });
  idx.addDocument(2, { title: "deep learning neural networks" });
  idx.addDocument(3, { title: "web development guide" });
  idx.addDocument(4, { title: "machine learning advanced" });

  return { documentStore: docStore, invertedIndex: idx };
}

function makeVectorContext(): ExecutionContext {
  const docStore = new MemoryDocumentStore();
  const vecIdx = new FlatVectorIndex(3);
  docStore.put(1, { name: "a" });
  vecIdx.add(1, new Float64Array([1, 0, 0]));
  docStore.put(2, { name: "b" });
  vecIdx.add(2, new Float64Array([0, 1, 0]));
  docStore.put(3, { name: "c" });
  vecIdx.add(3, new Float64Array([0.7, 0.7, 0]));
  return { documentStore: docStore, vectorIndexes: { embedding: vecIdx } };
}

// -- Primitive operators -----------------------------------------------------

describe("TermOperator", () => {
  it("finds documents containing term", () => {
    const ctx = makeContext();
    const op = new TermOperator("learning", "title");
    const result = op.execute(ctx);
    // docs 1, 2, 4 contain "learning"
    expect(result.length).toBe(3);
    const ids = [...result.docIds];
    expect(ids).toContain(1);
    expect(ids).toContain(2);
    expect(ids).toContain(4);
  });

  it("returns empty for missing term", () => {
    const ctx = makeContext();
    expect(new TermOperator("nonexistent", "title").execute(ctx).length).toBe(0);
  });

  it("returns empty with no index", () => {
    expect(new TermOperator("test").execute({}).length).toBe(0);
  });
});

describe("FilterOperator", () => {
  it("filters by equality predicate", () => {
    const ctx = makeContext();
    const op = new FilterOperator("category", new Equals("ai"));
    const result = op.execute(ctx);
    expect(result.length).toBe(3); // docs 1, 2, 4
  });

  it("filters by range predicate", () => {
    const ctx = makeContext();
    const op = new FilterOperator("year", new GreaterThanOrEqual(2021));
    const result = op.execute(ctx);
    expect(result.length).toBe(2); // docs 2, 4
  });

  it("filters source operator results", () => {
    const ctx = makeContext();
    const source = new TermOperator("learning", "title");
    const op = new FilterOperator("year", new GreaterThanOrEqual(2021), source);
    const result = op.execute(ctx);
    // learning: 1, 2, 4; year >= 2021: 2, 4
    expect(result.length).toBe(2);
  });
});

describe("FacetOperator", () => {
  it("counts field values", () => {
    const ctx = makeContext();
    const op = new FacetOperator("category");
    const result = op.execute(ctx);
    expect(result.length).toBe(2); // "ai" and "web"
    const aiEntry = result.entries.find(
      (e) => (e.payload.fields as Record<string, unknown>)["_facet_value"] === "ai",
    );
    expect(aiEntry).toBeDefined();
    expect((aiEntry!.payload.fields as Record<string, unknown>)["_facet_count"]).toBe(
      3,
    );
  });
});

// -- Vector operators --------------------------------------------------------

describe("KNNOperator", () => {
  it("finds nearest neighbors", () => {
    const ctx = makeVectorContext();
    const op = new KNNOperator(new Float64Array([1, 0, 0]), 2, "embedding");
    const result = op.execute(ctx);
    expect(result.length).toBe(2);
    expect([...result.docIds]).toContain(1); // exact match
  });
});

describe("VectorSimilarityOperator", () => {
  it("finds docs above threshold", () => {
    const ctx = makeVectorContext();
    const op = new VectorSimilarityOperator(
      new Float64Array([1, 0, 0]),
      0.5,
      "embedding",
    );
    const result = op.execute(ctx);
    // doc 1: cos=1.0, doc 3: cos~0.707 -> both above 0.5
    expect(result.length).toBe(2);
  });
});

// -- Boolean operators -------------------------------------------------------

describe("UnionOperator", () => {
  it("unions results", () => {
    const ctx = makeContext();
    const op = new UnionOperator([
      new TermOperator("machine", "title"),
      new TermOperator("web", "title"),
    ]);
    const result = op.execute(ctx);
    expect([...result.docIds]).toContain(1);
    expect([...result.docIds]).toContain(3);
    expect([...result.docIds]).toContain(4);
  });
});

describe("IntersectOperator", () => {
  it("intersects results", () => {
    const ctx = makeContext();
    const op = new IntersectOperator([
      new TermOperator("machine", "title"),
      new TermOperator("learning", "title"),
    ]);
    const result = op.execute(ctx);
    // docs 1, 4 have both "machine" and "learning"
    expect(result.length).toBe(2);
    expect([...result.docIds]).toContain(1);
    expect([...result.docIds]).toContain(4);
  });

  it("returns empty on no operands", () => {
    expect(new IntersectOperator([]).execute(makeContext()).length).toBe(0);
  });

  it("short-circuits on empty intermediate", () => {
    const ctx = makeContext();
    const op = new IntersectOperator([
      new TermOperator("nonexistent", "title"),
      new TermOperator("machine", "title"),
    ]);
    expect(op.execute(ctx).length).toBe(0);
  });
});

describe("ComplementOperator", () => {
  it("returns all docs NOT in operand", () => {
    const ctx = makeContext();
    const op = new ComplementOperator(new TermOperator("machine", "title"));
    const result = op.execute(ctx);
    // machine: docs 1, 4 -> complement: docs 2, 3
    expect(result.length).toBe(2);
    expect([...result.docIds]).toContain(2);
    expect([...result.docIds]).toContain(3);
  });
});

// -- Aggregation operators ---------------------------------------------------

describe("AggregateOperator", () => {
  it("counts documents", () => {
    const ctx = makeContext();
    const op = new AggregateOperator(null, "year", new CountMonoid());
    const result = op.execute(ctx);
    expect(result.entries[0]!.payload.score).toBe(4);
  });

  it("sums field values", () => {
    const ctx = makeContext();
    const op = new AggregateOperator(null, "year", new SumMonoid());
    const result = op.execute(ctx);
    expect(result.entries[0]!.payload.score).toBe(2020 + 2021 + 2019 + 2022);
  });

  it("averages field values", () => {
    const ctx = makeContext();
    const op = new AggregateOperator(null, "year", new AvgMonoid());
    const result = op.execute(ctx);
    expect(result.entries[0]!.payload.score).toBeCloseTo(
      (2020 + 2021 + 2019 + 2022) / 4,
      5,
    );
  });

  it("finds min", () => {
    const ctx = makeContext();
    const op = new AggregateOperator(null, "year", new MinMonoid());
    const result = op.execute(ctx);
    expect(result.entries[0]!.payload.score).toBe(2019);
  });

  it("finds max", () => {
    const ctx = makeContext();
    const op = new AggregateOperator(null, "year", new MaxMonoid());
    const result = op.execute(ctx);
    expect(result.entries[0]!.payload.score).toBe(2022);
  });

  it("computes quantile (median)", () => {
    const ctx = makeContext();
    const op = new AggregateOperator(null, "year", new QuantileMonoid(0.5));
    const result = op.execute(ctx);
    // sorted years: 2019, 2020, 2021, 2022 -> median = (2020+2021)/2 = 2020.5
    expect(result.entries[0]!.payload.score).toBeCloseTo(2020.5, 5);
  });
});

describe("GroupByOperator", () => {
  it("groups and aggregates", () => {
    const ctx = makeContext();
    const source = new FilterOperator("year", new GreaterThanOrEqual(0));
    const op = new GroupByOperator(source, "category", "year", new CountMonoid());
    const result = op.execute(ctx);
    expect(result.length).toBe(2); // "ai" and "web"
  });
});

// -- Sparse threshold --------------------------------------------------------

describe("SparseThresholdOperator", () => {
  it("applies ReLU thresholding", () => {
    const entries = [
      { docId: 1, payload: createPayload({ score: 0.8 }) },
      { docId: 2, payload: createPayload({ score: 0.3 }) },
      { docId: 3, payload: createPayload({ score: 0.6 }) },
    ];
    const source = {
      execute: () => new PostingList(entries),
      compose: () =>
        null as unknown as ReturnType<typeof ComposedOperator.prototype.compose>,
      costEstimate: () => 0,
    };
    const op = new SparseThresholdOperator(source as unknown as Operator, 0.5);
    const result = op.execute({});
    expect(result.length).toBe(2); // docs 1 (0.3) and 3 (0.1)
    const scores = result.entries.map((e) => e.payload.score).sort((a, b) => a - b);
    expect(scores[0]).toBeCloseTo(0.1, 10); // 0.6 - 0.5
    expect(scores[1]).toBeCloseTo(0.3, 10); // 0.8 - 0.5
  });
});

// -- Composed operator -------------------------------------------------------

describe("ComposedOperator", () => {
  it("returns last operator result", () => {
    const ctx = makeContext();
    const op = new ComposedOperator([
      new TermOperator("machine", "title"),
      new TermOperator("web", "title"),
    ]);
    const result = op.execute(ctx);
    // Returns last: "web" results
    expect([...result.docIds]).toContain(3);
  });
});
