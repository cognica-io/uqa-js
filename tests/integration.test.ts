import { describe, expect, it } from "vitest";
import { Engine } from "../src/engine.js";
import { Table, createColumnDef } from "../src/sql/table.js";
import { MemoryDocumentStore } from "../src/storage/document-store.js";
import { MemoryInvertedIndex } from "../src/storage/inverted-index.js";
import { PostingList } from "../src/core/posting-list.js";
import {
  createPayload,
  createPostingEntry,
  GreaterThanOrEqual,
  Equals,
} from "../src/core/types.js";

// ======================================================================
// Table insert / query pipeline
// ======================================================================

describe("Table CRUD", () => {
  it("insert and retrieve document", () => {
    const table = new Table("papers", [
      createColumnDef("id", "integer", {
        pythonType: "number",
        primaryKey: true,
        autoIncrement: true,
      }),
      createColumnDef("title", "text", { pythonType: "string" }),
      createColumnDef("year", "integer", { pythonType: "number" }),
      createColumnDef("category", "text", { pythonType: "string" }),
    ]);
    table.insert({ title: "neural networks", year: 2023, category: "ml" });
    table.insert({ title: "transformers", year: 2024, category: "dl" });

    expect(table.rowCount).toBe(2);
    const doc = table.documentStore.get(1);
    expect(doc).not.toBeNull();
    expect(doc!["title"]).toBe("neural networks");
    expect(doc!["year"]).toBe(2023);
  });

  it("insert indexes text fields in inverted index", () => {
    const table = new Table("papers", [
      createColumnDef("id", "integer", {
        pythonType: "number",
        primaryKey: true,
        autoIncrement: true,
      }),
      createColumnDef("title", "text", { pythonType: "string" }),
      createColumnDef("abstract", "text", { pythonType: "string" }),
    ]);
    // Register TEXT columns for FTS indexing (equivalent to CREATE INDEX USING gin)
    table.ftsFields.add("title");
    table.ftsFields.add("abstract");
    table.insert({
      title: "neural network basics",
      abstract: "introduction to neural networks",
    });
    table.insert({
      title: "transformer architecture",
      abstract: "attention mechanisms",
    });

    // Search for "neural" in the inverted index
    const pl = table.invertedIndex.getPostingList("title", "neural");
    expect(pl.length).toBeGreaterThan(0);
  });

  it("delete removes document", () => {
    const table = new Table("test", [
      createColumnDef("id", "integer", {
        pythonType: "number",
        primaryKey: true,
        autoIncrement: true,
      }),
      createColumnDef("val", "integer", { pythonType: "number" }),
    ]);
    table.insert({ val: 10 });
    table.insert({ val: 20 });
    table.insert({ val: 30 });

    expect(table.rowCount).toBe(3);
    table.documentStore.delete(2);
    expect(table.documentStore.get(2)).toBeNull();
  });
});

// ======================================================================
// Full pipeline: add documents with inverted index queries
// ======================================================================

describe("Full pipeline: term search", () => {
  it("finds documents by term", () => {
    const store = new MemoryDocumentStore();
    const index = new MemoryInvertedIndex();

    const docs = [
      { title: "neural network basics", abstract: "introduction to neural networks" },
      { title: "transformer architecture", abstract: "attention is all you need" },
      { title: "graph neural networks", abstract: "neural networks for graphs" },
    ];

    for (let i = 0; i < docs.length; i++) {
      const docId = i + 1;
      store.put(docId, docs[i]!);
      index.addDocument(docId, {
        title: docs[i]!["title"],
        abstract: docs[i]!["abstract"],
      });
    }

    // Search for "neural"
    const pl = index.getPostingListAnyField("neural");
    expect(pl.length).toBeGreaterThan(0);
    const ids = [...pl.docIds];
    expect(ids).toContain(1); // "neural network basics"
    expect(ids).toContain(3); // "graph neural networks"
  });

  it("boolean AND of two terms", () => {
    const index = new MemoryInvertedIndex();
    index.addDocument(1, {
      title: "neural network basics",
      abstract: "introduction to neural networks",
    });
    index.addDocument(2, {
      title: "transformer architecture",
      abstract: "attention is all you need",
    });
    index.addDocument(3, {
      title: "graph neural networks",
      abstract: "neural networks for graphs",
    });

    const plNeural = index.getPostingListAnyField("neural");
    const plNetworks = index.getPostingListAnyField("networks");
    const intersection = plNeural.intersect(plNetworks);

    // Both "neural" and "networks" appear in doc 1 and 3
    const ids = [...intersection.docIds];
    expect(ids).toContain(1);
    expect(ids).toContain(3);
    expect(ids).not.toContain(2);
  });

  it("boolean OR of two terms", () => {
    const index = new MemoryInvertedIndex();
    index.addDocument(1, { title: "neural networks" });
    index.addDocument(2, { title: "bayesian optimization" });
    index.addDocument(3, { title: "graph algorithms" });

    const plNeural = index.getPostingListAnyField("neural");
    const plBayesian = index.getPostingListAnyField("bayesian");
    const union = plNeural.union(plBayesian);

    const ids = [...union.docIds];
    expect(ids).toContain(1);
    expect(ids).toContain(2);
    expect(ids).not.toContain(3);
  });
});

// ======================================================================
// Vector search pipeline
// ======================================================================

describe("Vector search pipeline", () => {
  it("KNN search with FlatVectorIndex", async () => {
    const { FlatVectorIndex } = await import("../src/storage/vector-index.js");
    const idx = new FlatVectorIndex(4);
    idx.add(1, new Float64Array([1, 0, 0, 0]));
    idx.add(2, new Float64Array([0, 1, 0, 0]));
    idx.add(3, new Float64Array([0, 0, 1, 0]));

    const result = idx.searchKnn(new Float64Array([1, 0, 0, 0]), 1);
    expect(result.length).toBe(1);
    expect(result.entries[0]!.docId).toBe(1);
  });

  it("threshold search filters by similarity", async () => {
    const { FlatVectorIndex } = await import("../src/storage/vector-index.js");
    const idx = new FlatVectorIndex(4);
    idx.add(1, new Float64Array([1, 0, 0, 0]));
    idx.add(2, new Float64Array([0, 0, 0, 1]));

    const result = idx.searchThreshold(new Float64Array([1, 0, 0, 0]), 0.9);
    expect(result.length).toBe(1);
    expect([...result.docIds]).toContain(1);
  });
});

// ======================================================================
// PostingList operations
// ======================================================================

describe("PostingList operations", () => {
  it("union combines entries", () => {
    const a = new PostingList([
      createPostingEntry(1, { score: 0.9 }),
      createPostingEntry(3, { score: 0.7 }),
    ]);
    const b = new PostingList([
      createPostingEntry(2, { score: 0.8 }),
      createPostingEntry(3, { score: 0.6 }),
    ]);
    const result = a.union(b);
    expect(result.length).toBe(3);
    expect([...result.docIds]).toEqual(expect.arrayContaining([1, 2, 3]));
  });

  it("intersect finds common entries", () => {
    const a = new PostingList([
      createPostingEntry(1, { score: 0.9 }),
      createPostingEntry(2, { score: 0.8 }),
      createPostingEntry(3, { score: 0.7 }),
    ]);
    const b = new PostingList([
      createPostingEntry(2, { score: 0.6 }),
      createPostingEntry(3, { score: 0.5 }),
      createPostingEntry(4, { score: 0.4 }),
    ]);
    const result = a.intersect(b);
    expect(result.length).toBe(2);
    expect([...result.docIds]).toEqual(expect.arrayContaining([2, 3]));
  });

  it("difference excludes entries", () => {
    const a = new PostingList([
      createPostingEntry(1, { score: 0.9 }),
      createPostingEntry(2, { score: 0.8 }),
      createPostingEntry(3, { score: 0.7 }),
    ]);
    const b = new PostingList([createPostingEntry(2, { score: 0.6 })]);
    const result = a.difference(b);
    expect(result.length).toBe(2);
    expect([...result.docIds]).toContain(1);
    expect([...result.docIds]).toContain(3);
    expect([...result.docIds]).not.toContain(2);
  });

  it("topK returns highest scoring entries", () => {
    const pl = new PostingList([
      createPostingEntry(1, { score: 0.5 }),
      createPostingEntry(2, { score: 0.9 }),
      createPostingEntry(3, { score: 0.7 }),
      createPostingEntry(4, { score: 0.3 }),
    ]);
    const top = pl.topK(2);
    expect(top.length).toBe(2);
    expect(top.entries[0]!.docId).toBe(2); // highest score
  });
});

// ======================================================================
// Document store operations
// ======================================================================

describe("Document store operations", () => {
  it("put and get", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { name: "Alice", age: 30 });
    store.put(2, { name: "Bob", age: 25 });

    expect(store.get(1)).toEqual({ name: "Alice", age: 30 });
    expect(store.get(2)).toEqual({ name: "Bob", age: 25 });
    expect(store.get(3)).toBeNull();
  });

  it("delete removes document", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { name: "Alice" });
    store.delete(1);
    expect(store.get(1)).toBeNull();
  });

  it("clear removes all", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { name: "Alice" });
    store.put(2, { name: "Bob" });
    store.clear();
    expect(store.length).toBe(0);
  });

  it("getField retrieves single field", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { name: "Alice", age: 30 });
    expect(store.getField(1, "name")).toBe("Alice");
    expect(store.getField(1, "age")).toBe(30);
    expect(store.getField(1, "missing")).toBeUndefined();
  });

  it("getFieldsBulk retrieves field for multiple docs", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { name: "Alice" });
    store.put(2, { name: "Bob" });
    store.put(3, { name: "Carol" });

    const result = store.getFieldsBulk([1, 2, 3], "name");
    expect(result.get(1)).toBe("Alice");
    expect(result.get(2)).toBe("Bob");
    expect(result.get(3)).toBe("Carol");
  });

  it("hasValue checks for value existence", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { name: "Alice" });
    store.put(2, { name: "Bob" });

    expect(store.hasValue("name", "Alice")).toBe(true);
    expect(store.hasValue("name", "Carol")).toBe(false);
  });

  it("iterAll iterates in doc id order", () => {
    const store = new MemoryDocumentStore();
    store.put(3, { val: 30 });
    store.put(1, { val: 10 });
    store.put(2, { val: 20 });

    const pairs: [number, Record<string, unknown>][] = [];
    for (const pair of store.iterAll()) {
      pairs.push(pair);
    }
    expect(pairs.length).toBe(3);
    expect(pairs[0]![0]).toBe(1);
    expect(pairs[1]![0]).toBe(2);
    expect(pairs[2]![0]).toBe(3);
  });
});

// ======================================================================
// Engine DDL (CREATE TABLE only -- known to work)
// ======================================================================

describe("Engine DDL", () => {
  it("creates table via SQL", async () => {
    const engine = new Engine();
    await engine.sql(
      "CREATE TABLE papers (" +
        "id SERIAL PRIMARY KEY, " +
        "title TEXT, " +
        "year INTEGER" +
        ")",
    );
    expect(engine.hasTable("papers")).toBe(true);
    engine.close();
  });

  it("creates multiple tables", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t1 (id SERIAL PRIMARY KEY, val TEXT)");
    await engine.sql("CREATE TABLE t2 (id SERIAL PRIMARY KEY, data INTEGER)");
    expect(engine.hasTable("t1")).toBe(true);
    expect(engine.hasTable("t2")).toBe(true);
    engine.close();
  });
});
