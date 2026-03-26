import { describe, expect, it } from "vitest";
import { MemoryDocumentStore } from "../../src/storage/document-store.js";

describe("MemoryDocumentStore", () => {
  it("put and get", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { title: "hello", body: "world" });
    expect(store.get(1)).toEqual({ title: "hello", body: "world" });
  });

  it("get returns null for missing", () => {
    const store = new MemoryDocumentStore();
    expect(store.get(999)).toBeNull();
  });

  it("delete removes document", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { x: 1 });
    store.delete(1);
    expect(store.get(1)).toBeNull();
  });

  it("delete is no-op for missing", () => {
    const store = new MemoryDocumentStore();
    store.delete(999); // should not throw
  });

  it("clear removes all", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { a: 1 });
    store.put(2, { a: 2 });
    store.clear();
    expect(store.length).toBe(0);
  });

  it("getField returns field value", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { title: "test", year: 2024 });
    expect(store.getField(1, "title")).toBe("test");
    expect(store.getField(1, "year")).toBe(2024);
  });

  it("getField returns undefined for missing doc", () => {
    const store = new MemoryDocumentStore();
    expect(store.getField(1, "x")).toBeUndefined();
  });

  it("getField returns undefined for missing field", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { title: "test" });
    expect(store.getField(1, "missing")).toBeUndefined();
  });

  it("getFieldsBulk returns field values", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { x: 10 });
    store.put(2, { x: 20 });
    store.put(3, { x: 30 });
    const result = store.getFieldsBulk([1, 2, 3], "x");
    expect(result.get(1)).toBe(10);
    expect(result.get(2)).toBe(20);
    expect(result.get(3)).toBe(30);
  });

  it("hasValue finds matching value", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { color: "red" });
    store.put(2, { color: "blue" });
    expect(store.hasValue("color", "red")).toBe(true);
    expect(store.hasValue("color", "green")).toBe(false);
  });

  it("docIds returns set of all ids", () => {
    const store = new MemoryDocumentStore();
    store.put(1, {});
    store.put(5, {});
    store.put(10, {});
    expect(store.docIds).toEqual(new Set([1, 5, 10]));
  });

  it("length returns count", () => {
    const store = new MemoryDocumentStore();
    expect(store.length).toBe(0);
    store.put(1, {});
    store.put(2, {});
    expect(store.length).toBe(2);
  });

  it("evalPath traverses nested objects", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { a: { b: { c: 42 } } });
    expect(store.evalPath(1, ["a", "b", "c"])).toBe(42);
  });

  it("evalPath traverses arrays by index", () => {
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

  it("evalPath returns undefined for missing path", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { a: 1 });
    expect(store.evalPath(1, ["x", "y"])).toBeUndefined();
  });

  it("iterAll yields sorted entries", () => {
    const store = new MemoryDocumentStore();
    store.put(3, { v: 3 });
    store.put(1, { v: 1 });
    store.put(2, { v: 2 });
    const entries = [...store.iterAll()];
    expect(entries.map(([id]) => id)).toEqual([1, 2, 3]);
  });
});
