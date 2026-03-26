import { describe, expect, it } from "vitest";
import { MemoryInvertedIndex } from "../../src/storage/inverted-index.js";

describe("MemoryInvertedIndex", () => {
  function makeIndex() {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { title: "hello world", body: "the quick brown fox" });
    idx.addDocument(2, { title: "world peace", body: "jumps over the lazy dog" });
    idx.addDocument(3, { title: "hello again", body: "the fox runs fast" });
    return idx;
  }

  describe("addDocument", () => {
    it("indexes documents and returns IndexedTerms", () => {
      const idx = new MemoryInvertedIndex();
      const result = idx.addDocument(1, { title: "hello world" });
      expect(result.fieldLengths["title"]).toBe(2);
      expect(result.postings.size).toBeGreaterThan(0);
    });

    it("handles multiple fields", () => {
      const idx = new MemoryInvertedIndex();
      const result = idx.addDocument(1, { title: "a", body: "b c" });
      expect(result.fieldLengths["title"]).toBe(1);
      expect(result.fieldLengths["body"]).toBe(2);
    });
  });

  describe("getPostingList", () => {
    it("retrieves posting list for a term in a field", () => {
      const idx = makeIndex();
      const pl = idx.getPostingList("title", "hello");
      expect(pl.length).toBe(2);
      const docIds = [...pl.docIds];
      expect(docIds).toContain(1);
      expect(docIds).toContain(3);
    });

    it("returns empty for missing term", () => {
      const idx = makeIndex();
      expect(idx.getPostingList("title", "xyz").length).toBe(0);
    });
  });

  describe("getPostingListAnyField", () => {
    it("finds term across all fields", () => {
      const idx = makeIndex();
      const pl = idx.getPostingListAnyField("the");
      // "the" appears in body of doc 1, 2, 3
      expect(pl.length).toBe(3);
    });
  });

  describe("docFreq", () => {
    it("returns document frequency", () => {
      const idx = makeIndex();
      expect(idx.docFreq("title", "world")).toBe(2);
      expect(idx.docFreq("title", "hello")).toBe(2);
      expect(idx.docFreq("title", "peace")).toBe(1);
    });

    it("returns 0 for unknown term", () => {
      const idx = makeIndex();
      expect(idx.docFreq("title", "unknown")).toBe(0);
    });
  });

  describe("docFreqAnyField", () => {
    it("counts unique docs across fields", () => {
      const idx = makeIndex();
      expect(idx.docFreqAnyField("the")).toBe(3);
    });
  });

  describe("getDocLength", () => {
    it("returns field length for document", () => {
      const idx = makeIndex();
      expect(idx.getDocLength(1, "title")).toBe(2);
      expect(idx.getDocLength(1, "body")).toBe(4);
    });

    it("returns 0 for missing doc", () => {
      const idx = makeIndex();
      expect(idx.getDocLength(999, "title")).toBe(0);
    });
  });

  describe("getTotalDocLength", () => {
    it("returns sum of all field lengths for a doc", () => {
      const idx = makeIndex();
      expect(idx.getTotalDocLength(1)).toBe(6); // title:2 + body:4
    });
  });

  describe("getTermFreq", () => {
    it("returns term frequency in specific field", () => {
      const idx = makeIndex();
      // "the" appears twice in doc 2 body: "jumps over the lazy dog" -> 1 time
      expect(idx.getTermFreq(1, "body", "the")).toBe(1);
    });

    it("returns 0 for missing", () => {
      const idx = makeIndex();
      expect(idx.getTermFreq(1, "body", "xyz")).toBe(0);
    });
  });

  describe("removeDocument", () => {
    it("removes document from index", () => {
      const idx = makeIndex();
      idx.removeDocument(1);
      expect(idx.docFreq("title", "hello")).toBe(1);
      expect(idx.getPostingList("title", "hello").length).toBe(1);
    });

    it("decrements doc count", () => {
      const idx = makeIndex();
      const statsBefore = idx.stats;
      expect(statsBefore.totalDocs).toBe(3);
      idx.removeDocument(1);
      expect(idx.stats.totalDocs).toBe(2);
    });
  });

  describe("clear", () => {
    it("removes everything", () => {
      const idx = makeIndex();
      idx.clear();
      expect(idx.stats.totalDocs).toBe(0);
      expect(idx.getPostingList("title", "hello").length).toBe(0);
    });
  });

  describe("addPosting", () => {
    it("adds a single posting entry", () => {
      const idx = new MemoryInvertedIndex();
      idx.addPosting("title", "test", {
        docId: 42,
        payload: { positions: [0, 5], score: 1.0, fields: {} },
      });
      expect(idx.docFreq("title", "test")).toBe(1);
      const pl = idx.getPostingList("title", "test");
      expect(pl.length).toBe(1);
      expect(pl.entries[0]!.docId).toBe(42);
    });
  });

  describe("stats", () => {
    it("returns cached stats", () => {
      const idx = makeIndex();
      const s1 = idx.stats;
      const s2 = idx.stats;
      expect(s1).toBe(s2); // same object (cached)
    });

    it("invalidates cache on mutation", () => {
      const idx = makeIndex();
      const s1 = idx.stats;
      idx.addDocument(4, { title: "new doc" });
      const s2 = idx.stats;
      expect(s1).not.toBe(s2);
      expect(s2.totalDocs).toBe(4);
    });

    it("has correct avg doc length", () => {
      const idx = new MemoryInvertedIndex();
      idx.addDocument(1, { text: "a b c" }); // 3 tokens
      idx.addDocument(2, { text: "a b" }); // 2 tokens
      const s = idx.stats;
      expect(s.totalDocs).toBe(2);
      expect(s.avgDocLength).toBeCloseTo(2.5, 10);
    });
  });

  describe("analyzers", () => {
    it("uses default whitespace analyzer", () => {
      const idx = new MemoryInvertedIndex();
      idx.addDocument(1, { text: "Hello World" });
      expect(idx.getPostingList("text", "hello").length).toBe(1);
      expect(idx.getPostingList("text", "Hello").length).toBe(0);
    });

    it("uses custom field analyzer", () => {
      const uppercase = { analyze: (t: string) => t.toUpperCase().split(/\s+/) };
      const idx = new MemoryInvertedIndex(null, { text: uppercase });
      idx.addDocument(1, { text: "hello world" });
      expect(idx.getPostingList("text", "HELLO").length).toBe(1);
      expect(idx.getPostingList("text", "hello").length).toBe(0);
    });

    it("setFieldAnalyzer with search phase", () => {
      const idx = new MemoryInvertedIndex();
      const searchAnalyzer = { analyze: (t: string) => [t.toUpperCase()] };
      idx.setFieldAnalyzer("title", searchAnalyzer, "search");
      expect(idx.getSearchAnalyzer("title")).toBe(searchAnalyzer);
      // Index analyzer unchanged
      expect(idx.getFieldAnalyzer("title")).toBe(idx.analyzer);
    });
  });

  describe("bulk operations", () => {
    it("getDocLengthsBulk returns map", () => {
      const idx = makeIndex();
      const result = idx.getDocLengthsBulk([1, 2], "title");
      expect(result.get(1)).toBe(2);
      expect(result.get(2)).toBe(2);
    });

    it("getTermFreqsBulk returns map", () => {
      const idx = makeIndex();
      const result = idx.getTermFreqsBulk([1, 2, 3], "title", "world");
      expect(result.get(1)).toBe(1);
      expect(result.get(2)).toBe(1);
      expect(result.get(3)).toBe(0);
    });
  });
});
