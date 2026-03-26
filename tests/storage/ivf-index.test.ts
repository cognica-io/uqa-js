import { describe, expect, it } from "vitest";
import { IVFIndex } from "../../src/storage/ivf-index.js";
import { FlatVectorIndex } from "../../src/storage/vector-index.js";
import { cosine, norm } from "../../src/math/linalg.js";

// -- Helpers -----------------------------------------------------------------

function randomVector(dim: number, seed: number): Float64Array {
  // Simple seeded pseudo-random for reproducibility
  let state = seed;
  const values = new Float64Array(dim);
  for (let i = 0; i < dim; i++) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    values[i] = (state / 0x7fffffff) * 2 - 1;
  }
  return values;
}

function normalizedVector(dim: number, seed: number): Float64Array {
  const v = randomVector(dim, seed);
  const n = norm(v);
  if (n === 0) return v;
  const result = new Float64Array(dim);
  for (let i = 0; i < dim; i++) {
    result[i] = v[i]! / n;
  }
  return result;
}

// ======================================================================
// IVFIndex -- add and search
// ======================================================================

describe("IVFIndex add", () => {
  it("adds a single vector", () => {
    const idx = new IVFIndex(8);
    const vec = normalizedVector(8, 1);
    idx.add(1, vec);
    expect(idx.count()).toBe(1);
  });

  it("adds multiple vectors", () => {
    const idx = new IVFIndex(8);
    for (let i = 1; i <= 5; i++) {
      idx.add(i, normalizedVector(8, i));
    }
    expect(idx.count()).toBe(5);
  });

  it("rejects dimension mismatch", () => {
    const idx = new IVFIndex(8);
    expect(() => {
      idx.add(1, new Float64Array([1, 2, 3]));
    }).toThrow("dimension mismatch");
  });
});

// ======================================================================
// Search correctness
// ======================================================================

describe("IVFIndex search", () => {
  it("KNN search finds nearest vector", () => {
    const idx = new IVFIndex(8);
    const v1 = new Float64Array([1, 0, 0, 0, 0, 0, 0, 0]);
    const v2 = new Float64Array([0, 0, 0, 0, 0, 0, 0, 1]);
    idx.add(1, v1);
    idx.add(2, v2);

    const query = new Float64Array([1, 0, 0, 0, 0, 0, 0, 0]);
    const result = idx.searchKnn(query, 1);
    expect(result.length).toBe(1);
    expect(result.entries[0]!.docId).toBe(1);
  });

  it("threshold search filters by similarity", () => {
    const idx = new IVFIndex(8);
    const v1 = new Float64Array([1, 0, 0, 0, 0, 0, 0, 0]);
    const v2 = new Float64Array([0, 0, 0, 0, 0, 0, 0, 1]);
    idx.add(1, v1);
    idx.add(2, v2);

    const query = new Float64Array([1, 0, 0, 0, 0, 0, 0, 0]);
    const result = idx.searchThreshold(query, 0.9);
    const docIds = new Set([...result.docIds]);
    expect(docIds.has(1)).toBe(true);
    expect(docIds.has(2)).toBe(false);
  });

  it("KNN on empty index returns empty", () => {
    const idx = new IVFIndex(8);
    const query = new Float64Array([1, 0, 0, 0, 0, 0, 0, 0]);
    const result = idx.searchKnn(query, 5);
    expect(result.length).toBe(0);
  });

  it("KNN returns correct top-k from many vectors", () => {
    const idx = new IVFIndex(8);
    for (let i = 1; i <= 20; i++) {
      idx.add(i, normalizedVector(8, i));
    }
    const query = normalizedVector(8, 5);
    const result = idx.searchKnn(query, 1);
    expect(result.length).toBe(1);
    // The closest vector should be doc_id 5 (same seed)
    expect(result.entries[0]!.docId).toBe(5);
  });

  it("KNN returns k results when available", () => {
    const idx = new IVFIndex(8);
    for (let i = 1; i <= 10; i++) {
      idx.add(i, normalizedVector(8, i));
    }
    const query = normalizedVector(8, 3);
    const result = idx.searchKnn(query, 5);
    expect(result.length).toBe(5);
  });
});

// ======================================================================
// Delete
// ======================================================================

describe("IVFIndex delete", () => {
  it("delete removes vector from search results", () => {
    const idx = new IVFIndex(8);
    idx.add(1, normalizedVector(8, 1));
    idx.add(2, normalizedVector(8, 2));
    idx.delete(1);
    // HNSW uses lazy deletion so count may still include deleted nodes.
    // What matters is the deleted vector is excluded from search results.
    const query = normalizedVector(8, 1);
    const result = idx.searchKnn(query, 5);
    const docIds = new Set([...result.docIds]);
    expect(docIds.has(1)).toBe(false);
  });

  it("delete nonexistent is safe", () => {
    const idx = new IVFIndex(8);
    idx.delete(999);
    expect(idx.count()).toBe(0);
  });

  it("KNN after delete excludes deleted", () => {
    const idx = new IVFIndex(8);
    const v1 = new Float64Array([1, 0, 0, 0, 0, 0, 0, 0]);
    const v2 = new Float64Array([0, 1, 0, 0, 0, 0, 0, 0]);
    idx.add(1, v1);
    idx.add(2, v2);
    idx.delete(1);

    const query = new Float64Array([1, 0, 0, 0, 0, 0, 0, 0]);
    const result = idx.searchKnn(query, 2);
    const docIds = new Set([...result.docIds]);
    expect(docIds.has(1)).toBe(false);
    expect(docIds.has(2)).toBe(true);
  });
});

// ======================================================================
// Clear
// ======================================================================

describe("IVFIndex clear", () => {
  it("clear removes all vectors", () => {
    const idx = new IVFIndex(8);
    for (let i = 1; i <= 10; i++) {
      idx.add(i, normalizedVector(8, i));
    }
    expect(idx.count()).toBe(10);
    idx.clear();
    expect(idx.count()).toBe(0);
  });

  it("search after clear returns empty", () => {
    const idx = new IVFIndex(8);
    idx.add(1, new Float64Array([1, 0, 0, 0, 0, 0, 0, 0]));
    idx.clear();
    const result = idx.searchKnn(new Float64Array([1, 0, 0, 0, 0, 0, 0, 0]), 5);
    expect(result.length).toBe(0);
  });
});

// ======================================================================
// Train (k-means)
// ======================================================================

describe("IVFIndex train", () => {
  it("train does not crash on few vectors", () => {
    const idx = new IVFIndex(8, 4);
    const vectors: Float64Array[] = [];
    for (let i = 0; i < 5; i++) {
      const v = normalizedVector(8, i);
      vectors.push(v);
      idx.add(i, v);
    }
    idx.train(vectors);
    // After training, search should still work
    const query = normalizedVector(8, 3);
    const result = idx.searchKnn(query, 1);
    expect(result.entries[0]!.docId).toBe(3);
  });

  it("train with many vectors converges", () => {
    const idx = new IVFIndex(8, 4);
    const vectors: Float64Array[] = [];
    for (let i = 0; i < 100; i++) {
      const v = normalizedVector(8, i);
      vectors.push(v);
      idx.add(i, v);
    }
    idx.train(vectors);
    // Post-training search should still find the correct vector
    const query = normalizedVector(8, 50);
    const result = idx.searchKnn(query, 1);
    expect(result.length).toBe(1);
    expect(result.entries[0]!.docId).toBe(50);
  });

  it("train on empty vectors does not crash", () => {
    const idx = new IVFIndex(8, 4);
    idx.train([]);
    expect(idx.count()).toBe(0);
  });
});

// ======================================================================
// FlatVectorIndex comparison (basic tests to ensure IVF behaves similarly)
// ======================================================================

describe("IVFIndex vs FlatVectorIndex", () => {
  it("both find same nearest neighbor", () => {
    const flat = new FlatVectorIndex(4);
    const ivf = new IVFIndex(4);

    const vectors = [
      new Float64Array([1, 0, 0, 0]),
      new Float64Array([0, 1, 0, 0]),
      new Float64Array([0, 0, 1, 0]),
      new Float64Array([0, 0, 0, 1]),
      new Float64Array([1, 1, 0, 0]),
    ];
    for (let i = 0; i < vectors.length; i++) {
      flat.add(i + 1, vectors[i]!);
      ivf.add(i + 1, vectors[i]!);
    }

    const query = new Float64Array([1, 0, 0, 0]);
    const flatResult = flat.searchKnn(query, 1);
    const ivfResult = ivf.searchKnn(query, 1);

    expect(flatResult.entries[0]!.docId).toBe(ivfResult.entries[0]!.docId);
  });

  it("both return same threshold results", () => {
    const flat = new FlatVectorIndex(4);
    const ivf = new IVFIndex(4);

    const vectors = [new Float64Array([1, 0, 0, 0]), new Float64Array([0, 0, 0, 1])];
    for (let i = 0; i < vectors.length; i++) {
      flat.add(i + 1, vectors[i]!);
      ivf.add(i + 1, vectors[i]!);
    }

    const query = new Float64Array([1, 0, 0, 0]);
    const flatResult = flat.searchThreshold(query, 0.9);
    const ivfResult = ivf.searchThreshold(query, 0.9);

    const flatIds = new Set([...flatResult.docIds]);
    const ivfIds = new Set([...ivfResult.docIds]);
    expect(flatIds.has(1)).toBe(true);
    expect(flatIds.has(2)).toBe(false);
    expect(ivfIds.has(1)).toBe(true);
    expect(ivfIds.has(2)).toBe(false);
  });
});

// ======================================================================
// Add, delete, re-search correctness
// ======================================================================

describe("IVFIndex add-delete-search cycles", () => {
  it("re-adding after delete works", () => {
    const idx = new IVFIndex(4);
    const v = new Float64Array([1, 0, 0, 0]);
    idx.add(1, v);
    idx.delete(1);
    // After delete the vector should not appear in search
    const resultAfterDelete = idx.searchKnn(v, 1);
    const idsAfterDelete = new Set([...resultAfterDelete.docIds]);
    expect(idsAfterDelete.has(1)).toBe(false);

    // Re-add and verify it appears in search again
    idx.add(1, v);
    const result = idx.searchKnn(v, 1);
    expect(result.entries[0]!.docId).toBe(1);
  });

  it("bulk add and search", () => {
    const idx = new IVFIndex(16);
    for (let i = 0; i < 50; i++) {
      idx.add(i, normalizedVector(16, i));
    }
    expect(idx.count()).toBe(50);

    // Search for a specific vector
    const query = normalizedVector(16, 25);
    const result = idx.searchKnn(query, 3);
    expect(result.length).toBe(3);
    // doc 25 should be the best match
    expect(result.entries[0]!.docId).toBe(25);
  });
});
