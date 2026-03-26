import { describe, expect, it } from "vitest";
import {
  createPayload,
  createPostingEntry,
  Equals,
  NotEquals,
  GreaterThan,
  GreaterThanOrEqual,
  LessThan,
  LessThanOrEqual,
  InSet,
  Between,
} from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import { MemoryDocumentStore } from "../../src/storage/document-store.js";
import { MemoryInvertedIndex } from "../../src/storage/inverted-index.js";
import { FlatVectorIndex } from "../../src/storage/vector-index.js";
import type { ExecutionContext } from "../../src/operators/base.js";
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
  HybridTextVectorOperator,
  SemanticFilterOperator,
} from "../../src/operators/hybrid.js";
import {
  PathFilterOperator,
  PathProjectOperator,
  PathUnnestOperator,
} from "../../src/operators/hierarchical.js";
import {
  AggregateOperator,
  AvgMonoid,
  CountMonoid,
  GroupByOperator,
  MaxMonoid,
  MinMonoid,
  SumMonoid,
} from "../../src/operators/aggregation.js";
import { whitespaceAnalyzer } from "../../src/analysis/analyzer.js";

// -- Shared test data ---------------------------------------------------------

const SAMPLE_DOCUMENTS = [
  {
    doc_id: 1,
    title: "introduction to neural networks",
    abstract: "neural networks are computational models",
    year: 2023,
    category: "machine learning",
  },
  {
    doc_id: 2,
    title: "deep learning transformers",
    abstract: "transformers use attention mechanisms",
    year: 2024,
    category: "deep learning",
  },
  {
    doc_id: 3,
    title: "graph neural networks",
    abstract: "graph neural networks extend neural networks to graph data",
    year: 2024,
    category: "machine learning",
  },
  {
    doc_id: 4,
    title: "bayesian optimization methods",
    abstract: "bayesian methods for hyperparameter optimization",
    year: 2025,
    category: "optimization",
  },
  {
    doc_id: 5,
    title: "reinforcement learning agents",
    abstract: "reinforcement learning for decision making",
    year: 2025,
    category: "reinforcement learning",
  },
];

function makeSeededVector(seed: number, dim: number): Float64Array {
  // Simple deterministic PRNG for vectors
  let x = seed;
  const vec = new Float64Array(dim);
  for (let i = 0; i < dim; i++) {
    x = ((x * 1103515245 + 12345) >>> 0) & 0x7fffffff;
    vec[i] = (x / 0x7fffffff) * 2 - 1;
  }
  return vec;
}

function makeSampleVectors(): Record<number, Float64Array> {
  const vectors: Record<number, Float64Array> = {};
  for (let i = 1; i <= 5; i++) {
    vectors[i] = makeSeededVector(42 + i, 64);
  }
  return vectors;
}

function makeContext(): ExecutionContext {
  const docStore = new MemoryDocumentStore();
  const invIndex = new MemoryInvertedIndex(whitespaceAnalyzer());
  const sampleVectors = makeSampleVectors();

  for (const doc of SAMPLE_DOCUMENTS) {
    docStore.put(doc.doc_id, doc);
    const fields: Record<string, string> = {};
    if (doc.title) fields["title"] = doc.title;
    if (doc.abstract) fields["abstract"] = doc.abstract;
    invIndex.addDocument(doc.doc_id, fields);
  }

  const vecIdx = new FlatVectorIndex(64);
  for (const [docId, vec] of Object.entries(sampleVectors)) {
    vecIdx.add(Number(docId), vec);
  }

  return {
    documentStore: docStore,
    invertedIndex: invIndex,
    vectorIndexes: { embedding: vecIdx },
  };
}

// =============================================================================
// TermOperator
// =============================================================================

describe("TermOperator", () => {
  it("returns correct posting list", () => {
    const ctx = makeContext();
    const op = new TermOperator("neural", "title");
    const result = op.execute(ctx);
    expect(result.length).toBeGreaterThan(0);
    const docIds = result.docIds;
    expect(docIds.has(1)).toBe(true);
    expect(docIds.has(3)).toBe(true);
  });

  it("missing term returns empty", () => {
    const ctx = makeContext();
    const op = new TermOperator("nonexistent_term_xyz", "title");
    const result = op.execute(ctx);
    expect(result.length).toBe(0);
  });

  it("term with positions", () => {
    const ctx = makeContext();
    const op = new TermOperator("neural", "title");
    const result = op.execute(ctx);
    for (const entry of result) {
      expect(entry.payload.positions.length).toBeGreaterThan(0);
    }
  });
});

// =============================================================================
// FilterOperator
// =============================================================================

describe("FilterOperator", () => {
  it("Equals", () => {
    const ctx = makeContext();
    const op = new FilterOperator("category", new Equals("machine learning"));
    const result = op.execute(ctx);
    expect(result.docIds).toEqual(new Set([1, 3]));
  });

  it("NotEquals", () => {
    const ctx = makeContext();
    const op = new FilterOperator("category", new NotEquals("machine learning"));
    const result = op.execute(ctx);
    expect(result.docIds.has(1)).toBe(false);
    expect(result.docIds.has(3)).toBe(false);
    expect(result.length).toBe(3);
  });

  it("GreaterThan", () => {
    const ctx = makeContext();
    const op = new FilterOperator("year", new GreaterThan(2024));
    const result = op.execute(ctx);
    expect(result.docIds).toEqual(new Set([4, 5]));
  });

  it("GreaterThanOrEqual", () => {
    const ctx = makeContext();
    const op = new FilterOperator("year", new GreaterThanOrEqual(2024));
    const result = op.execute(ctx);
    expect(result.docIds).toEqual(new Set([2, 3, 4, 5]));
  });

  it("LessThan", () => {
    const ctx = makeContext();
    const op = new FilterOperator("year", new LessThan(2024));
    const result = op.execute(ctx);
    expect(result.docIds).toEqual(new Set([1]));
  });

  it("LessThanOrEqual", () => {
    const ctx = makeContext();
    const op = new FilterOperator("year", new LessThanOrEqual(2024));
    const result = op.execute(ctx);
    expect(result.docIds).toEqual(new Set([1, 2, 3]));
  });

  it("InSet", () => {
    const ctx = makeContext();
    const op = new FilterOperator(
      "category",
      new InSet(["machine learning", "optimization"]),
    );
    const result = op.execute(ctx);
    expect(result.docIds).toEqual(new Set([1, 3, 4]));
  });

  it("Between", () => {
    const ctx = makeContext();
    const op = new FilterOperator("year", new Between(2024, 2025));
    const result = op.execute(ctx);
    expect(result.docIds).toEqual(new Set([2, 3, 4, 5]));
  });
});

// =============================================================================
// Boolean operators
// =============================================================================

describe("Boolean operators", () => {
  it("UnionOperator", () => {
    const ctx = makeContext();
    const opA = new TermOperator("neural", "title");
    const opB = new TermOperator("bayesian", "title");
    const unionOp = new UnionOperator([opA, opB]);
    const result = unionOp.execute(ctx);
    // neural: docs 1, 3; bayesian: doc 4
    for (const id of [1, 3, 4]) {
      expect(result.docIds.has(id)).toBe(true);
    }
  });

  it("IntersectOperator", () => {
    const ctx = makeContext();
    const opA = new TermOperator("neural", "title");
    const opB = new TermOperator("networks", "title");
    const intersectOp = new IntersectOperator([opA, opB]);
    const result = intersectOp.execute(ctx);
    expect(result.docIds).toEqual(new Set([1, 3]));
  });

  it("ComplementOperator", () => {
    const ctx = makeContext();
    const op = new TermOperator("neural", "title");
    const complementOp = new ComplementOperator(op);
    const result = complementOp.execute(ctx);
    const neuralDocs = op.execute(ctx).docIds;
    for (const docId of result.docIds) {
      expect(neuralDocs.has(docId)).toBe(false);
    }
  });
});

// =============================================================================
// HybridTextVectorOperator
// =============================================================================

describe("HybridTextVectorOperator", () => {
  it("hybrid is intersect of term and vector results", () => {
    const ctx = makeContext();
    const sampleVectors = makeSampleVectors();
    const queryVec = sampleVectors[1]!;

    const hybridOp = new HybridTextVectorOperator("neural", queryVec, -1.0);
    const hybridResult = hybridOp.execute(ctx);

    // Manually compute intersect
    const termResult = new TermOperator("neural", "title").execute(ctx);
    const vecResult = new VectorSimilarityOperator(queryVec, -1.0).execute(ctx);
    const manualIntersect = termResult.intersect(vecResult);

    expect(hybridResult.docIds).toEqual(manualIntersect.docIds);
  });
});

describe("SemanticFilterOperator", () => {
  it("result is subset of filter result", () => {
    const ctx = makeContext();
    const sampleVectors = makeSampleVectors();
    const queryVec = sampleVectors[1]!;

    const source = new FilterOperator("year", new GreaterThanOrEqual(2024));
    const semFilter = new SemanticFilterOperator(source, queryVec, -1.0);
    const result = semFilter.execute(ctx);

    const filterResult = source.execute(ctx);
    for (const docId of result.docIds) {
      expect(filterResult.docIds.has(docId)).toBe(true);
    }
  });
});

// =============================================================================
// Aggregation Monoids
// =============================================================================

describe("AggregationMonoids", () => {
  it("count identity", () => {
    const m = new CountMonoid();
    expect(m.finalize(m.identity())).toBe(0);
  });

  it("count accumulate", () => {
    const m = new CountMonoid();
    let state = m.identity();
    state = m.accumulate(state, "anything");
    state = m.accumulate(state, "else");
    expect(m.finalize(state)).toBe(2);
  });

  it("count combine associativity", () => {
    const m = new CountMonoid();
    const a = m.accumulate(m.identity(), 1);
    const b = m.accumulate(m.identity(), 2);
    const c = m.accumulate(m.identity(), 3);
    expect(m.combine(m.combine(a, b), c)).toBe(m.combine(a, m.combine(b, c)));
  });

  it("sum identity", () => {
    const m = new SumMonoid();
    expect(m.finalize(m.identity())).toBe(0.0);
  });

  it("sum accumulate", () => {
    const m = new SumMonoid();
    let state = m.identity();
    state = m.accumulate(state, 3.0);
    state = m.accumulate(state, 7.0);
    expect(m.finalize(state)).toBe(10.0);
  });

  it("sum combine associativity", () => {
    const m = new SumMonoid();
    const a = m.accumulate(m.identity(), 1.0);
    const b = m.accumulate(m.identity(), 2.0);
    const c = m.accumulate(m.identity(), 3.0);
    expect(m.combine(m.combine(a, b), c)).toBe(m.combine(a, m.combine(b, c)));
  });

  it("avg correct", () => {
    const m = new AvgMonoid();
    let state = m.identity();
    for (const v of [2.0, 4.0, 6.0]) {
      state = m.accumulate(state, v);
    }
    expect(m.finalize(state)).toBe(4.0);
  });

  it("avg combine associativity", () => {
    const m = new AvgMonoid();
    const a = m.accumulate(m.identity(), 2.0);
    const b = m.accumulate(m.identity(), 4.0);
    const c = m.accumulate(m.identity(), 6.0);
    const lhs = m.combine(m.combine(a, b), c);
    const rhs = m.combine(a, m.combine(b, c));
    expect(m.finalize(lhs)).toBe(m.finalize(rhs));
  });

  it("min identity", () => {
    const m = new MinMonoid();
    expect(m.finalize(m.identity())).toBe(Infinity);
  });

  it("min accumulate", () => {
    const m = new MinMonoid();
    let state = m.identity();
    state = m.accumulate(state, 5.0);
    state = m.accumulate(state, 3.0);
    state = m.accumulate(state, 7.0);
    expect(m.finalize(state)).toBe(3.0);
  });

  it("max identity", () => {
    const m = new MaxMonoid();
    expect(m.finalize(m.identity())).toBe(-Infinity);
  });

  it("max accumulate", () => {
    const m = new MaxMonoid();
    let state = m.identity();
    state = m.accumulate(state, 5.0);
    state = m.accumulate(state, 3.0);
    state = m.accumulate(state, 7.0);
    expect(m.finalize(state)).toBe(7.0);
  });

  it("aggregate operator computes avg", () => {
    const ctx = makeContext();
    const source = new FilterOperator("year", new GreaterThanOrEqual(2024));
    const agg = new AggregateOperator(source, "year", new AvgMonoid());
    const result = agg.execute(ctx);
    expect(result.length).toBe(1);
    const entry = result.entries[0]!;
    // Years 2024, 2024, 2025, 2025 -> avg = 2024.5
    expect(entry.payload.fields["_aggregate"]).toBeCloseTo(2024.5);
  });

  it("group by operator", () => {
    const ctx = makeContext();
    const source = new FilterOperator("year", new GreaterThanOrEqual(2023));
    const groupOp = new GroupByOperator(source, "category", "year", new CountMonoid());
    const result = groupOp.execute(ctx);
    expect(result.length).toBeGreaterThan(0);
    const groups = new Set(result.entries.map((e) => e.payload.fields["_group_key"]));
    expect(groups.has("machine learning")).toBe(true);
  });
});

// =============================================================================
// Hierarchical operators
// =============================================================================

describe("Hierarchical operators", () => {
  function makeHierContext(): ExecutionContext {
    const store = new MemoryDocumentStore();
    store.put(1, {
      title: "test document",
      metadata: {
        author: "Alice",
        tags: ["python", "search", "algebra"],
      },
      sections: [
        { heading: "Introduction", content: "This is the intro" },
        { heading: "Methods", content: "We use posting lists" },
      ],
    });
    return { documentStore: store };
  }

  it("path filter matches", () => {
    const ctx = makeHierContext();
    const op = new PathFilterOperator(["metadata", "author"], new Equals("Alice"));
    const result = op.execute(ctx);
    expect(result.length).toBe(1);
    expect(result.docIds.has(1)).toBe(true);
  });

  it("path filter no match", () => {
    const ctx = makeHierContext();
    const op = new PathFilterOperator(["metadata", "author"], new Equals("Bob"));
    const result = op.execute(ctx);
    expect(result.length).toBe(0);
  });

  it("path project", () => {
    const ctx = makeHierContext();
    const source = new PathFilterOperator(["metadata", "author"], new Equals("Alice"));
    const projectOp = new PathProjectOperator(
      [["title"], ["metadata", "author"]],
      source,
    );
    const result = projectOp.execute(ctx);
    expect(result.length).toBe(1);
    const entry = result.entries[0]!;
    expect(entry.payload.fields["title"]).toBe("test document");
    expect(entry.payload.fields["metadata.author"]).toBe("Alice");
  });

  it("path unnest", () => {
    const ctx = makeHierContext();
    const source = new PathFilterOperator(["metadata", "author"], new Equals("Alice"));
    const unnestOp = new PathUnnestOperator(["metadata", "tags"], source);
    const result = unnestOp.execute(ctx);
    expect(result.length).toBe(1);
    expect(result.docIds.has(1)).toBe(true);
    const entry = result.entries[0]!;
    expect(entry.payload.fields).toHaveProperty("_unnested_data");
  });
});

// =============================================================================
// Vector operators
// =============================================================================

describe("Vector operators", () => {
  it("KNN returns k results", () => {
    const ctx = makeContext();
    const sampleVectors = makeSampleVectors();
    const query = sampleVectors[1]!;
    const op = new KNNOperator(query, 3);
    const result = op.execute(ctx);
    expect(result.length).toBe(3);
  });

  it("vector similarity threshold", () => {
    const ctx = makeContext();
    const sampleVectors = makeSampleVectors();
    const query = sampleVectors[1]!;
    const op = new VectorSimilarityOperator(query, 0.99);
    const result = op.execute(ctx);
    expect(result.docIds.has(1)).toBe(true);
  });
});

// =============================================================================
// FacetOperator
// =============================================================================

describe("FacetOperator", () => {
  it("facet counts", () => {
    const ctx = makeContext();
    const source = new FilterOperator("year", new GreaterThanOrEqual(2023));
    const facetOp = new FacetOperator("category", source);
    const result = facetOp.execute(ctx);
    expect(result.length).toBeGreaterThan(0);
    const facetValues = new Set(
      result.entries.map((e) => e.payload.fields["_facet_value"]),
    );
    expect(facetValues.has("machine learning")).toBe(true);
  });
});

// =============================================================================
// Storage layer (DocumentStore, InvertedIndex)
// =============================================================================

describe("DocumentStore", () => {
  it("put and get", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { title: "test" });
    expect(store.get(1)).toEqual({ title: "test" });
  });

  it("delete", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { title: "test" });
    store.delete(1);
    expect(store.get(1)).toBeNull();
  });

  it("getField", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { title: "test", year: 2024 });
    expect(store.getField(1, "title")).toBe("test");
    expect(store.getField(1, "year")).toBe(2024);
  });

  it("evalPath", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { metadata: { author: "Alice" } });
    expect(store.evalPath(1, ["metadata", "author"])).toBe("Alice");
  });
});

describe("InvertedIndex", () => {
  it("add and retrieve", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { title: "hello world" });
    const pl = idx.getPostingList("title", "hello");
    expect(pl.length).toBe(1);
    expect(pl.docIds.has(1)).toBe(true);
  });

  it("doc freq", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { title: "hello world" });
    idx.addDocument(2, { title: "hello there" });
    expect(idx.docFreq("title", "hello")).toBe(2);
    expect(idx.docFreq("title", "world")).toBe(1);
  });

  it("positions", () => {
    const idx = new MemoryInvertedIndex(whitespaceAnalyzer());
    idx.addDocument(1, { title: "the quick brown fox the" });
    const pl = idx.getPostingList("title", "the");
    const entry = pl.getEntry(1);
    expect(entry).not.toBeNull();
    expect(entry!.payload.positions).toContain(0);
    expect(entry!.payload.positions).toContain(4);
  });

  it("stats", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { title: "hello world" });
    idx.addDocument(2, { title: "hello there friend" });
    const stats = idx.stats;
    expect(stats.totalDocs).toBe(2);
    expect(stats.avgDocLength).toBeGreaterThan(0);
  });
});

// =============================================================================
// Vector index
// =============================================================================

describe("FlatVectorIndex", () => {
  it("add and knn", () => {
    const idx = new FlatVectorIndex(16);
    const vectors: Record<number, Float64Array> = {};
    for (let i = 1; i <= 5; i++) {
      vectors[i] = makeSeededVector(42 + i, 16);
      idx.add(i, vectors[i]!);
    }
    const result = idx.searchKnn(vectors[1]!, 3);
    expect(result.length).toBe(3);
    expect(result.docIds.has(1)).toBe(true);
  });

  it("search threshold", () => {
    const idx = new FlatVectorIndex(16);
    const vectors: Record<number, Float64Array> = {};
    for (let i = 1; i <= 5; i++) {
      vectors[i] = makeSeededVector(42 + i, 16);
      idx.add(i, vectors[i]!);
    }
    const result = idx.searchThreshold(vectors[1]!, 0.99);
    expect(result.docIds.has(1)).toBe(true);
  });
});
