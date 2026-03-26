import { describe, expect, it } from "vitest";
import { FlatVectorIndex } from "../../src/storage/vector-index.js";

describe("FlatVectorIndex", () => {
  function makeIndex() {
    const idx = new FlatVectorIndex(3);
    idx.add(1, new Float64Array([1, 0, 0]));
    idx.add(2, new Float64Array([0, 1, 0]));
    idx.add(3, new Float64Array([0, 0, 1]));
    idx.add(4, new Float64Array([1, 1, 0])); // normalized: [0.707, 0.707, 0]
    return idx;
  }

  it("searchKnn returns top-k by cosine similarity", () => {
    const idx = makeIndex();
    const query = new Float64Array([1, 0, 0]);
    const result = idx.searchKnn(query, 2);
    expect(result.length).toBe(2);
    // doc 1 should be top (exact match), doc 4 second (cos = 0.707)
    const ids = result.entries.map((e) => e.docId);
    expect(ids).toContain(1);
    expect(ids).toContain(4);
  });

  it("searchKnn returns all if k >= count", () => {
    const idx = makeIndex();
    const query = new Float64Array([1, 0, 0]);
    const result = idx.searchKnn(query, 100);
    expect(result.length).toBe(4);
  });

  it("searchThreshold returns docs above threshold", () => {
    const idx = makeIndex();
    const query = new Float64Array([1, 0, 0]);
    const result = idx.searchThreshold(query, 0.5);
    // doc 1: cos=1.0, doc 4: cos~0.707 -> both above 0.5
    expect(result.length).toBe(2);
    const ids = [...result.docIds];
    expect(ids).toContain(1);
    expect(ids).toContain(4);
  });

  it("searchThreshold with high threshold", () => {
    const idx = makeIndex();
    const query = new Float64Array([1, 0, 0]);
    const result = idx.searchThreshold(query, 0.99);
    expect(result.length).toBe(1);
    expect([...result.docIds]).toContain(1);
  });

  it("add throws on dimension mismatch", () => {
    const idx = new FlatVectorIndex(3);
    expect(() => {
      idx.add(1, new Float64Array([1, 2]));
    }).toThrow("dimension mismatch");
  });

  it("delete removes vector", () => {
    const idx = makeIndex();
    idx.delete(1);
    expect(idx.count()).toBe(3);
    const result = idx.searchKnn(new Float64Array([1, 0, 0]), 1);
    expect(result.entries[0]!.docId).toBe(4); // next best match
  });

  it("clear removes all", () => {
    const idx = makeIndex();
    idx.clear();
    expect(idx.count()).toBe(0);
  });

  it("count returns number of vectors", () => {
    const idx = makeIndex();
    expect(idx.count()).toBe(4);
  });

  it("searchKnn on empty index returns empty", () => {
    const idx = new FlatVectorIndex(3);
    const result = idx.searchKnn(new Float64Array([1, 0, 0]), 5);
    expect(result.length).toBe(0);
  });
});
