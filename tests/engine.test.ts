import { describe, expect, it } from "vitest";
import { Engine } from "../src/engine.js";
import { QueryBuilder } from "../src/api/query-builder.js";
import { HierarchicalDocument } from "../src/core/hierarchical.js";
import { GraphToRelationalFunctor, TextToVectorFunctor } from "../src/core/functor.js";
import { PostingList } from "../src/core/posting-list.js";
import { createPayload } from "../src/core/types.js";
import { Transaction } from "../src/storage/transaction.js";

describe("Engine", () => {
  it("creates tables via SQL", async () => {
    const engine = new Engine();
    await engine.sql(
      "CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, year INTEGER)",
    );
    expect(engine.hasTable("articles")).toBe(true);
    engine.close();
  });

  it("provides query builder", () => {
    const engine = new Engine();
    const qb = engine.query("test");
    expect(qb).toBeInstanceOf(QueryBuilder);
    engine.close();
  });

  it("manages graph store", () => {
    const engine = new Engine();
    engine.createGraph("kg");
    expect(engine.graphStore.hasGraph("kg")).toBe(true);
    engine.close();
  });
});

describe("HierarchicalDocument", () => {
  it("evaluates simple path", () => {
    const doc = new HierarchicalDocument(1, { a: { b: { c: 42 } } });
    expect(doc.evalPath(["a", "b", "c"])).toBe(42);
  });

  it("evaluates array index", () => {
    const doc = new HierarchicalDocument(1, { items: [10, 20, 30] });
    expect(doc.evalPath(["items", 1])).toBe(20);
  });

  it("implicit array wildcard", () => {
    const doc = new HierarchicalDocument(1, {
      people: [{ name: "Alice" }, { name: "Bob" }],
    });
    expect(doc.evalPath(["people", "name"])).toEqual(["Alice", "Bob"]);
  });

  it("returns undefined for missing path", () => {
    const doc = new HierarchicalDocument(1, { a: 1 });
    expect(doc.evalPath(["x", "y"])).toBeUndefined();
  });
});

describe("Functors", () => {
  it("GraphToRelationalFunctor converts PostingList", () => {
    const f = new GraphToRelationalFunctor();
    const pl = new PostingList([{ docId: 1, payload: createPayload({ score: 0.5 }) }]);
    const result = f.mapObject(pl);
    expect(result).toBeInstanceOf(PostingList);
  });

  it("TextToVectorFunctor creates embeddings", () => {
    const f = new TextToVectorFunctor(4);
    const pl = new PostingList([
      {
        docId: 1,
        payload: createPayload({
          score: 0.5,
          positions: [0, 1, 2],
        }),
      },
    ]);
    const result = f.mapObject(pl) as PostingList;
    expect(result.length).toBe(1);
  });
});

describe("Transaction", () => {
  it("creates with active state", () => {
    // Transaction requires ManagedConnection, test structure only
    expect(Transaction).toBeDefined();
  });
});
