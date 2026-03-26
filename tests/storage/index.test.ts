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
  IndexStats,
} from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import { MemoryDocumentStore } from "../../src/storage/document-store.js";
import { MemoryInvertedIndex } from "../../src/storage/inverted-index.js";
import { FlatVectorIndex } from "../../src/storage/vector-index.js";
import { FilterOperator } from "../../src/operators/primitive.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import { whitespaceAnalyzer, standardAnalyzer } from "../../src/analysis/analyzer.js";

// -- Helpers ------------------------------------------------------------------

interface Employee {
  id: number;
  name: string;
  age: number;
  salary: number;
}

const EMPLOYEES: Employee[] = [
  { id: 1, name: "Alice", age: 30, salary: 70000.0 },
  { id: 2, name: "Bob", age: 25, salary: 55000.0 },
  { id: 3, name: "Charlie", age: 35, salary: 90000.0 },
  { id: 4, name: "Diana", age: 28, salary: 65000.0 },
  { id: 5, name: "Eve", age: 40, salary: 95000.0 },
];

function makeEmployeeContext(): ExecutionContext {
  const store = new MemoryDocumentStore();
  for (const emp of EMPLOYEES) {
    store.put(emp.id, emp as unknown as Record<string, unknown>);
  }
  return { documentStore: store };
}

// =============================================================================
// DocumentStore
// =============================================================================

describe("MemoryDocumentStore", () => {
  it("put and get", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { title: "test" });
    expect(store.get(1)).toEqual({ title: "test" });
  });

  it("get missing returns null", () => {
    const store = new MemoryDocumentStore();
    expect(store.get(999)).toBeNull();
  });

  it("delete removes document", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { title: "test" });
    store.delete(1);
    expect(store.get(1)).toBeNull();
  });

  it("getField retrieves specific field", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { title: "test", year: 2024 });
    expect(store.getField(1, "title")).toBe("test");
    expect(store.getField(1, "year")).toBe(2024);
  });

  it("getField returns undefined for missing field", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { title: "test" });
    expect(store.getField(1, "nonexistent")).toBeUndefined();
  });

  it("getField returns undefined for missing doc", () => {
    const store = new MemoryDocumentStore();
    expect(store.getField(999, "title")).toBeUndefined();
  });

  it("evalPath traverses nested objects", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { metadata: { author: "Alice" } });
    expect(store.evalPath(1, ["metadata", "author"])).toBe("Alice");
  });

  it("evalPath handles missing path", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { a: 1 });
    expect(store.evalPath(1, ["x", "y"])).toBeUndefined();
  });

  it("evalPath handles arrays", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { items: [10, 20, 30] });
    expect(store.evalPath(1, ["items", 1])).toBe(20);
  });

  it("evalPath implicit array wildcard", () => {
    const store = new MemoryDocumentStore();
    store.put(1, {
      people: [{ name: "Alice" }, { name: "Bob" }],
    });
    expect(store.evalPath(1, ["people", "name"])).toEqual(["Alice", "Bob"]);
  });

  it("clear removes all documents", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { a: 1 });
    store.put(2, { b: 2 });
    store.clear();
    expect(store.get(1)).toBeNull();
    expect(store.get(2)).toBeNull();
  });

  it("hasValue checks for existence", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { title: "test", year: 2024 });
    store.put(2, { title: "other", year: 2025 });
    expect(store.hasValue("title", "test")).toBe(true);
    expect(store.hasValue("title", "missing")).toBe(false);
  });

  it("getFieldsBulk retrieves multiple docs", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { title: "a" });
    store.put(2, { title: "b" });
    store.put(3, { title: "c" });
    const result = store.getFieldsBulk([1, 3], "title");
    expect(result.get(1)).toBe("a");
    expect(result.get(3)).toBe("c");
  });

  it("docIds returns all doc IDs", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { a: 1 });
    store.put(2, { b: 2 });
    store.put(3, { c: 3 });
    const ids = store.docIds;
    expect(ids.has(1)).toBe(true);
    expect(ids.has(2)).toBe(true);
    expect(ids.has(3)).toBe(true);
    expect(ids.size).toBe(3);
  });
});

// =============================================================================
// InvertedIndex
// =============================================================================

describe("MemoryInvertedIndex", () => {
  it("add and retrieve", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { title: "hello world" });
    const pl = idx.getPostingList("title", "hello");
    expect(pl.length).toBe(1);
    expect(pl.docIds.has(1)).toBe(true);
  });

  it("doc freq across documents", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { title: "hello world" });
    idx.addDocument(2, { title: "hello there" });
    expect(idx.docFreq("title", "hello")).toBe(2);
    expect(idx.docFreq("title", "world")).toBe(1);
  });

  it("positions are tracked", () => {
    const idx = new MemoryInvertedIndex(whitespaceAnalyzer());
    idx.addDocument(1, { title: "the quick brown fox the" });
    const pl = idx.getPostingList("title", "the");
    const entry = pl.getEntry(1);
    expect(entry).not.toBeNull();
    expect(entry!.payload.positions).toContain(0);
    expect(entry!.payload.positions).toContain(4);
  });

  it("stats are computed correctly", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { title: "hello world" });
    idx.addDocument(2, { title: "hello there friend" });
    const stats = idx.stats;
    expect(stats.totalDocs).toBe(2);
    expect(stats.avgDocLength).toBeGreaterThan(0);
  });

  it("missing term returns empty posting list", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { title: "hello world" });
    const pl = idx.getPostingList("title", "nonexistent");
    expect(pl.length).toBe(0);
  });

  it("multiple fields indexed separately", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { title: "hello world", body: "goodbye world" });
    expect(idx.getPostingList("title", "hello").length).toBe(1);
    expect(idx.getPostingList("body", "goodbye").length).toBe(1);
    // "world" appears in both fields
    expect(idx.getPostingList("title", "world").length).toBe(1);
    expect(idx.getPostingList("body", "world").length).toBe(1);
  });

  it("removeDocument cleans up", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { title: "hello world" });
    idx.addDocument(2, { title: "hello there" });
    idx.removeDocument(1);
    expect(idx.getPostingList("title", "hello").length).toBe(1);
    expect(idx.getPostingList("title", "world").length).toBe(0);
  });

  it("per-field analyzer", () => {
    const idx = new MemoryInvertedIndex();
    idx.setFieldAnalyzer("title", standardAnalyzer());
    idx.setFieldAnalyzer("body", whitespaceAnalyzer());
    idx.addDocument(1, { title: "The Quick Fox", body: "The body" });
    // "the" removed from title (standard has stop words) but kept in body
    expect(idx.getPostingList("title", "the").length).toBe(0);
    expect(idx.getPostingList("body", "the").length).toBe(1);
  });

  it("dual analyzer (index vs search)", () => {
    const idx = new MemoryInvertedIndex();
    const idxAnalyzer = whitespaceAnalyzer();
    const searchAnalyzer = whitespaceAnalyzer();
    idx.setFieldAnalyzer("body", idxAnalyzer, "index");
    idx.setFieldAnalyzer("body", searchAnalyzer, "search");
    expect(idx.getFieldAnalyzer("body")).toBe(idxAnalyzer);
    expect(idx.getSearchAnalyzer("body")).toBe(searchAnalyzer);
  });
});

// =============================================================================
// FlatVectorIndex
// =============================================================================

describe("FlatVectorIndex", () => {
  function makeSeededVector(seed: number, dim: number): Float64Array {
    let x = seed;
    const vec = new Float64Array(dim);
    for (let i = 0; i < dim; i++) {
      x = ((x * 1103515245 + 12345) >>> 0) & 0x7fffffff;
      vec[i] = (x / 0x7fffffff) * 2 - 1;
    }
    return vec;
  }

  it("add and KNN search", () => {
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

  it("threshold search finds exact matches", () => {
    const idx = new FlatVectorIndex(16);
    const vectors: Record<number, Float64Array> = {};
    for (let i = 1; i <= 5; i++) {
      vectors[i] = makeSeededVector(42 + i, 16);
      idx.add(i, vectors[i]!);
    }
    // Exact vector should have similarity ~1.0
    const result = idx.searchThreshold(vectors[1]!, 0.99);
    expect(result.docIds.has(1)).toBe(true);
  });

  it("dimension mismatch throws", () => {
    const idx = new FlatVectorIndex(16);
    expect(() => idx.add(1, new Float64Array(8))).toThrow(/dimension/i);
  });

  it("delete removes vector", () => {
    const idx = new FlatVectorIndex(4);
    idx.add(1, new Float64Array([1, 0, 0, 0]));
    idx.add(2, new Float64Array([0, 1, 0, 0]));
    expect(idx.count()).toBe(2);
    idx.delete(1);
    expect(idx.count()).toBe(1);
  });

  it("clear removes all vectors", () => {
    const idx = new FlatVectorIndex(4);
    idx.add(1, new Float64Array([1, 0, 0, 0]));
    idx.add(2, new Float64Array([0, 1, 0, 0]));
    idx.clear();
    expect(idx.count()).toBe(0);
  });

  it("KNN on empty index returns empty", () => {
    const idx = new FlatVectorIndex(4);
    const result = idx.searchKnn(new Float64Array([1, 0, 0, 0]), 3);
    expect(result.length).toBe(0);
  });

  it("KNN returns scores in descending order", () => {
    const idx = new FlatVectorIndex(4);
    idx.add(1, new Float64Array([1, 0, 0, 0]));
    idx.add(2, new Float64Array([0.9, 0.1, 0, 0]));
    idx.add(3, new Float64Array([0, 0, 0, 1]));
    const result = idx.searchKnn(new Float64Array([1, 0, 0, 0]), 3);
    const scores = result.entries.map((e) => e.payload.score);
    for (let i = 0; i < scores.length - 1; i++) {
      expect(scores[i]!).toBeGreaterThanOrEqual(scores[i + 1]!);
    }
  });
});

// =============================================================================
// Filter scan correctness (simulated index scan via FilterOperator)
// =============================================================================

describe("FilterOperator scan correctness", () => {
  it("equality scan", () => {
    const ctx = makeEmployeeContext();
    const op = new FilterOperator("age", new Equals(30));
    const result = op.execute(ctx);
    expect(result.docIds).toEqual(new Set([1])); // Alice
  });

  it("range scan greater than", () => {
    const ctx = makeEmployeeContext();
    const op = new FilterOperator("age", new GreaterThan(30));
    const result = op.execute(ctx);
    expect(result.docIds).toEqual(new Set([3, 5])); // Charlie, Eve
  });

  it("range scan less than", () => {
    const ctx = makeEmployeeContext();
    const op = new FilterOperator("age", new LessThan(30));
    const result = op.execute(ctx);
    expect(result.docIds).toEqual(new Set([2, 4])); // Bob, Diana
  });

  it("between scan", () => {
    const ctx = makeEmployeeContext();
    const op = new FilterOperator("age", new Between(28, 35));
    const result = op.execute(ctx);
    expect(result.docIds).toEqual(new Set([1, 3, 4])); // Alice, Charlie, Diana
  });

  it("in scan", () => {
    const ctx = makeEmployeeContext();
    const op = new FilterOperator("age", new InSet([25, 40]));
    const result = op.execute(ctx);
    expect(result.docIds).toEqual(new Set([2, 5])); // Bob, Eve
  });

  it("AND filter (chained)", () => {
    const ctx = makeEmployeeContext();
    const ageOp = new FilterOperator("age", new GreaterThan(28));
    const salaryOp = new FilterOperator("salary", new GreaterThan(70000));
    // Manually intersect results
    const ageResult = ageOp.execute(ctx);
    const salaryResult = salaryOp.execute(ctx);
    const result = ageResult.intersect(salaryResult);
    expect(result.docIds).toEqual(new Set([3, 5])); // Charlie, Eve
  });

  it("scan matches full scan", () => {
    const ctx = makeEmployeeContext();
    const op = new FilterOperator("age", new GreaterThanOrEqual(30));
    const result = op.execute(ctx);
    // Manually verify
    const expected = EMPLOYEES.filter((e) => e.age >= 30).map((e) => e.id);
    expect(result.docIds).toEqual(new Set(expected));
  });
});

// =============================================================================
// IndexStats
// =============================================================================

describe("IndexStats", () => {
  it("initializes with defaults", () => {
    const stats = new IndexStats();
    expect(stats.totalDocs).toBe(0);
    expect(stats.avgDocLength).toBe(0);
    expect(stats.dimensions).toBe(0);
  });

  it("stores doc freq per field+term", () => {
    const stats = new IndexStats(100, 50.0);
    stats.setDocFreq("title", "hello", 42);
    expect(stats.docFreq("title", "hello")).toBe(42);
    expect(stats.docFreq("title", "missing")).toBe(0);
  });

  it("allows updating totalDocs and avgDocLength", () => {
    const stats = new IndexStats(10, 20.0);
    stats.totalDocs = 100;
    stats.avgDocLength = 50.0;
    expect(stats.totalDocs).toBe(100);
    expect(stats.avgDocLength).toBe(50.0);
  });
});

// =============================================================================
// Predicate system
// =============================================================================

describe("Predicate system", () => {
  it("Equals", () => {
    const p = new Equals(42);
    expect(p.evaluate(42)).toBe(true);
    expect(p.evaluate(43)).toBe(false);
    expect(p.evaluate("42")).toBe(false);
  });

  it("NotEquals", () => {
    const p = new NotEquals(42);
    expect(p.evaluate(42)).toBe(false);
    expect(p.evaluate(43)).toBe(true);
  });

  it("GreaterThan", () => {
    const p = new GreaterThan(10);
    expect(p.evaluate(11)).toBe(true);
    expect(p.evaluate(10)).toBe(false);
    expect(p.evaluate(9)).toBe(false);
  });

  it("GreaterThanOrEqual", () => {
    const p = new GreaterThanOrEqual(10);
    expect(p.evaluate(11)).toBe(true);
    expect(p.evaluate(10)).toBe(true);
    expect(p.evaluate(9)).toBe(false);
  });

  it("LessThan", () => {
    const p = new LessThan(10);
    expect(p.evaluate(9)).toBe(true);
    expect(p.evaluate(10)).toBe(false);
    expect(p.evaluate(11)).toBe(false);
  });

  it("LessThanOrEqual", () => {
    const p = new LessThanOrEqual(10);
    expect(p.evaluate(9)).toBe(true);
    expect(p.evaluate(10)).toBe(true);
    expect(p.evaluate(11)).toBe(false);
  });

  it("InSet", () => {
    const p = new InSet([1, 2, 3]);
    expect(p.evaluate(1)).toBe(true);
    expect(p.evaluate(4)).toBe(false);
  });

  it("Between", () => {
    const p = new Between(5, 10);
    expect(p.evaluate(5)).toBe(true);
    expect(p.evaluate(7)).toBe(true);
    expect(p.evaluate(10)).toBe(true);
    expect(p.evaluate(4)).toBe(false);
    expect(p.evaluate(11)).toBe(false);
  });
});
