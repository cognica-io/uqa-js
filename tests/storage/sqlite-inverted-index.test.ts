import { describe, expect, it } from "vitest";
import initSqlJs from "sql.js";
import { SQLiteInvertedIndex } from "../../src/storage/sqlite-inverted-index.js";
import { ManagedConnection } from "../../src/storage/managed-connection.js";

async function makeIndex(tableName = "docs"): Promise<SQLiteInvertedIndex> {
  const SQL = await initSqlJs();
  const db = new SQL.Database();
  const conn = new ManagedConnection(db);
  return new SQLiteInvertedIndex(conn, tableName);
}

// ======================================================================
// Basic add / retrieve
// ======================================================================

describe("AddDocumentAndRetrieve", () => {
  it("single document single field", async () => {
    const idx = await makeIndex();
    const result = idx.addDocument(1, { title: "hello world" });

    expect(result.fieldLengths["title"]).toBe(2);

    const pl = idx.getPostingList("title", "hello");
    expect(pl.length).toBe(1);
    const entries = [...pl];
    expect(entries[0]!.docId).toBe(1);
    expect(entries[0]!.payload.score).toBe(0.0);
  });

  it("posting list sorted by doc id", async () => {
    const idx = await makeIndex();
    idx.addDocument(3, { body: "alpha" });
    idx.addDocument(1, { body: "alpha" });
    idx.addDocument(2, { body: "alpha" });

    const pl = idx.getPostingList("body", "alpha");
    const docIds = [...pl].map((e) => e.docId);
    expect(docIds).toEqual([1, 2, 3]);
  });

  it("multiple terms in one document", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { title: "the quick brown fox" });

    for (const term of ["the", "quick", "brown", "fox"]) {
      const pl = idx.getPostingList("title", term);
      expect(pl.length).toBe(1);
      expect([...pl][0]!.docId).toBe(1);
    }
  });

  it("multiple documents shared term", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "hello world" });
    idx.addDocument(2, { body: "hello there" });
    idx.addDocument(3, { body: "goodbye world" });

    const plHello = idx.getPostingList("body", "hello");
    expect(plHello.length).toBe(2);
    const helloDocIds = new Set([...plHello].map((e) => e.docId));
    expect(helloDocIds).toEqual(new Set([1, 2]));

    const plWorld = idx.getPostingList("body", "world");
    expect(plWorld.length).toBe(2);
    const worldDocIds = new Set([...plWorld].map((e) => e.docId));
    expect(worldDocIds).toEqual(new Set([1, 3]));
  });

  it("positions are preserved", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "the cat sat on the mat" });

    const pl = idx.getPostingList("body", "the");
    expect(pl.length).toBe(1);
    const entry = [...pl][0]!;
    expect(entry.payload.positions).toEqual([0, 4]);

    const plCat = idx.getPostingList("body", "cat");
    const catEntry = [...plCat][0]!;
    expect(catEntry.payload.positions).toEqual([1]);
  });

  it("empty posting list for nonexistent term", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { title: "hello" });

    const pl = idx.getPostingList("title", "nonexistent");
    expect(pl.length).toBe(0);
  });

  it("empty posting list for nonexistent field", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { title: "hello" });

    const pl = idx.getPostingList("unknown_field", "hello");
    expect(pl.length).toBe(0);
  });
});

// ======================================================================
// doc_freq / term frequency
// ======================================================================

describe("DocFreqAndTermFreq", () => {
  it("doc freq single term", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "hello world" });
    idx.addDocument(2, { body: "hello there" });
    idx.addDocument(3, { body: "goodbye" });

    expect(idx.docFreq("body", "hello")).toBe(2);
    expect(idx.docFreq("body", "world")).toBe(1);
    expect(idx.docFreq("body", "goodbye")).toBe(1);
    expect(idx.docFreq("body", "missing")).toBe(0);
  });

  it("doc freq nonexistent field", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "hello" });
    expect(idx.docFreq("other", "hello")).toBe(0);
  });

  it("get term freq", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "the cat sat on the mat" });

    expect(idx.getTermFreq(1, "body", "the")).toBe(2);
    expect(idx.getTermFreq(1, "body", "cat")).toBe(1);
    expect(idx.getTermFreq(1, "body", "missing")).toBe(0);
    expect(idx.getTermFreq(99, "body", "the")).toBe(0);
  });

  it("get total term freq", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { title: "hello world", body: "hello there hello" });

    // "hello" appears 1 time in title, 2 times in body = 3 total
    expect(idx.getTotalTermFreq(1, "hello")).toBe(3);
    expect(idx.getTotalTermFreq(1, "world")).toBe(1);
    expect(idx.getTotalTermFreq(1, "missing")).toBe(0);
  });

  it("doc freq any field", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { title: "hello", body: "hello world" });
    idx.addDocument(2, { title: "goodbye", body: "hello" });

    expect(idx.docFreqAnyField("hello")).toBe(2);
    expect(idx.docFreqAnyField("world")).toBe(1);
    expect(idx.docFreqAnyField("goodbye")).toBe(1);
    expect(idx.docFreqAnyField("missing")).toBe(0);
  });
});

// ======================================================================
// remove_document
// ======================================================================

describe("RemoveDocument", () => {
  it("removes all postings", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "hello world" });
    idx.addDocument(2, { body: "hello there" });

    idx.removeDocument(1);

    const pl = idx.getPostingList("body", "hello");
    expect(pl.length).toBe(1);
    expect([...pl][0]!.docId).toBe(2);

    const plWorld = idx.getPostingList("body", "world");
    expect(plWorld.length).toBe(0);
  });

  it("updates stats on remove", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "hello world" });
    idx.addDocument(2, { body: "goodbye" });

    const statsBefore = idx.stats;
    expect(statsBefore.totalDocs).toBe(2);

    idx.removeDocument(1);

    const statsAfter = idx.stats;
    expect(statsAfter.totalDocs).toBe(1);
  });

  it("removes doc lengths", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "hello world" });
    idx.addDocument(2, { body: "goodbye" });

    idx.removeDocument(1);

    expect(idx.getDocLength(1, "body")).toBe(0);
    expect(idx.getDocLength(2, "body")).toBe(1);
  });

  it("remove nonexistent is safe", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "hello" });
    // Should not throw
    idx.removeDocument(999);
    expect(idx.getPostingList("body", "hello").length).toBe(1);
  });
});

// ======================================================================
// Doc lengths
// ======================================================================

describe("DocLengths", () => {
  it("get doc length", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { title: "hello world", body: "a b c d e" });

    expect(idx.getDocLength(1, "title")).toBe(2);
    expect(idx.getDocLength(1, "body")).toBe(5);
  });

  it("get doc length missing doc", async () => {
    const idx = await makeIndex();
    expect(idx.getDocLength(99, "body")).toBe(0);
  });

  it("get total doc length", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { title: "hello world", body: "a b c d e" });
    expect(idx.getTotalDocLength(1)).toBe(7); // 2 + 5
  });

  it("get total doc length missing doc", async () => {
    const idx = await makeIndex();
    expect(idx.getTotalDocLength(99)).toBe(0);
  });
});

// ======================================================================
// get_posting_list_any_field
// ======================================================================

describe("PostingListAnyField", () => {
  it("merges across fields", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { title: "hello", body: "world" });
    idx.addDocument(2, { title: "world", body: "hello" });

    const pl = idx.getPostingListAnyField("hello");
    expect(pl.length).toBe(2);
    const docIds = new Set([...pl].map((e) => e.docId));
    expect(docIds).toEqual(new Set([1, 2]));
  });

  it("deduplicates across fields", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { title: "hello", body: "hello world" });

    const pl = idx.getPostingListAnyField("hello");
    expect(pl.length).toBe(1);
    expect([...pl][0]!.docId).toBe(1);
  });

  it("empty for missing term", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { title: "hello" });

    const pl = idx.getPostingListAnyField("nonexistent");
    expect(pl.length).toBe(0);
  });
});

// ======================================================================
// stats property
// ======================================================================

describe("Stats", () => {
  it("empty index", async () => {
    const idx = await makeIndex();
    const s = idx.stats;
    expect(s.totalDocs).toBe(0);
    expect(s.avgDocLength).toBe(0.0);
  });

  it("single document", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "hello world test" });

    const s = idx.stats;
    expect(s.totalDocs).toBe(1);
    expect(s.avgDocLength).toBe(3.0);
  });

  it("multiple documents", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "hello world" }); // length 2
    idx.addDocument(2, { body: "a b c d" }); // length 4

    const s = idx.stats;
    expect(s.totalDocs).toBe(2);
    // total_length = 6, avg = 3.0
    expect(s.avgDocLength).toBe(3.0);
  });

  it("stats doc freqs", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "hello world" });
    idx.addDocument(2, { body: "hello there" });

    const s = idx.stats;
    expect(s.docFreq("body", "hello")).toBe(2);
    expect(s.docFreq("body", "world")).toBe(1);
    expect(s.docFreq("body", "there")).toBe(1);
  });

  it("stats with multiple fields", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { title: "hello", body: "world foo" });

    const s = idx.stats;
    expect(s.totalDocs).toBe(1);
    // total_length = 1 (title) + 2 (body) = 3
    expect(s.avgDocLength).toBe(3.0);
  });
});

// ======================================================================
// Tokenization
// ======================================================================

describe("Tokenization", () => {
  it("lowercased", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { title: "Hello WORLD" });

    const pl = idx.getPostingList("title", "hello");
    expect(pl.length).toBe(1);

    const plUpper = idx.getPostingList("title", "Hello");
    expect(plUpper.length).toBe(0);
  });

  it("split on whitespace", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { title: "hello   world" });

    const pl = idx.getPostingList("title", "hello");
    expect(pl.length).toBe(1);
    const pl2 = idx.getPostingList("title", "world");
    expect(pl2.length).toBe(1);
  });
});

// ======================================================================
// Table isolation (two tables sharing one connection)
// ======================================================================

describe("TableIsolation", () => {
  it("separate tables same connection", async () => {
    const SQL = await initSqlJs();
    const db = new SQL.Database();
    const conn = new ManagedConnection(db);

    const idxA = new SQLiteInvertedIndex(conn, "table_a");
    const idxB = new SQLiteInvertedIndex(conn, "table_b");

    idxA.addDocument(1, { body: "hello world" });
    idxB.addDocument(1, { body: "goodbye moon" });

    const plA = idxA.getPostingList("body", "hello");
    const plB = idxB.getPostingList("body", "hello");
    expect(plA.length).toBe(1);
    expect(plB.length).toBe(0);

    const plB2 = idxB.getPostingList("body", "goodbye");
    expect(plB2.length).toBe(1);

    expect(idxA.stats.totalDocs).toBe(1);
    expect(idxB.stats.totalDocs).toBe(1);
  });

  it("remove does not affect other table", async () => {
    const SQL = await initSqlJs();
    const db = new SQL.Database();
    const conn = new ManagedConnection(db);

    const idxA = new SQLiteInvertedIndex(conn, "table_a");
    const idxB = new SQLiteInvertedIndex(conn, "table_b");

    idxA.addDocument(1, { body: "hello" });
    idxB.addDocument(1, { body: "hello" });

    idxA.removeDocument(1);

    expect(idxA.getPostingList("body", "hello").length).toBe(0);
    expect(idxB.getPostingList("body", "hello").length).toBe(1);
  });
});

// ======================================================================
// IndexedTerms return value
// ======================================================================

describe("IndexedTermsReturn", () => {
  it("field lengths", async () => {
    const idx = await makeIndex();
    const result = idx.addDocument(1, { title: "a b c", body: "x y" });

    expect(result.fieldLengths["title"]).toBe(3);
    expect(result.fieldLengths["body"]).toBe(2);
  });

  it("postings keys", async () => {
    const idx = await makeIndex();
    const result = idx.addDocument(1, { title: "hello world" });

    expect(result.postings.has("title\0hello")).toBe(true);
    expect(result.postings.has("title\0world")).toBe(true);
  });
});

// ======================================================================
// Persistence across reconnection
// ======================================================================

describe("Persistence", () => {
  it("data survives export/import", async () => {
    const SQL = await initSqlJs();
    const db1 = new SQL.Database();
    const conn1 = new ManagedConnection(db1);
    const idx1 = new SQLiteInvertedIndex(conn1, "docs");
    idx1.addDocument(1, { title: "hello world" });
    idx1.addDocument(2, { title: "foo bar" });

    const data = db1.export();
    const db2 = new SQL.Database(data);
    const conn2 = new ManagedConnection(db2);
    const idx2 = new SQLiteInvertedIndex(conn2, "docs");

    const pl = idx2.getPostingList("title", "hello");
    expect(pl.length).toBe(1);
    expect([...pl][0]!.docId).toBe(1);

    expect(idx2.getDocLength(1, "title")).toBe(2);
    expect(idx2.stats.totalDocs).toBe(2);
  });
});
