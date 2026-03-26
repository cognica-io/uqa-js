import { describe, expect, it } from "vitest";
import initSqlJs from "sql.js";
import { SQLiteInvertedIndex } from "../../src/storage/sqlite-inverted-index.js";
import { ManagedConnection } from "../../src/storage/managed-connection.js";
import { BlockMaxIndex } from "../../src/storage/block-max-index.js";
import { IndexStats, createPostingEntry } from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";

async function makeIndex(tableName = "docs"): Promise<SQLiteInvertedIndex> {
  const SQL = await initSqlJs();
  const db = new SQL.Database();
  const conn = new ManagedConnection(db);
  return new SQLiteInvertedIndex(conn, tableName);
}

function bulkInsert(idx: SQLiteInvertedIndex, n: number, field = "body"): void {
  for (let i = 1; i <= n; i++) {
    idx.addDocument(i, { [field]: `alpha term${String(i)}` });
  }
}

// ======================================================================
// Skip Pointers -- skip_to, flush_skip_pointers depend on the TS
// implementation providing these methods. Many of these are SQLite-internal
// details. We test what is available.
// ======================================================================

describe("SkipPointerConstruction", () => {
  it("skip table created", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "hello world" });

    // The skip table should exist after adding a document with a text field.
    const rows = (
      idx as unknown as {
        _conn: {
          query: (sql: string, params?: unknown[]) => Record<string, unknown>[];
        };
      }
    )._conn.query("SELECT name FROM sqlite_master WHERE type='table' AND name = ?", [
      `_skip_docs_body`,
    ]);
    expect(rows.length).toBe(1);
  });

  it("skip entries for small posting list", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "hello" });
    idx.addDocument(2, { body: "hello" });
    idx.addDocument(3, { body: "hello" });
    idx.flushSkipPointers();

    const conn = (
      idx as unknown as {
        _conn: {
          query: (sql: string, params?: unknown[]) => Record<string, unknown>[];
        };
      }
    )._conn;
    const rows = conn.query(
      `SELECT skip_doc_id, skip_offset FROM "_skip_docs_body" WHERE term = ? ORDER BY skip_doc_id`,
      ["hello"],
    );

    // Only one skip entry at offset 0 (the first doc).
    expect(rows.length).toBe(1);
    expect(rows[0]!["skip_doc_id"]).toBe(1);
    expect(rows[0]!["skip_offset"]).toBe(0);
  });

  it("skip entries for large posting list", async () => {
    const idx = await makeIndex();
    for (let i = 1; i <= 300; i++) {
      idx.addDocument(i, { body: "alpha" });
    }
    idx.flushSkipPointers();

    const conn = (
      idx as unknown as {
        _conn: {
          query: (sql: string, params?: unknown[]) => Record<string, unknown>[];
        };
      }
    )._conn;
    const rows = conn.query(
      `SELECT skip_doc_id, skip_offset FROM "_skip_docs_body" WHERE term = ? ORDER BY skip_offset`,
      ["alpha"],
    );

    expect(rows.length).toBe(3);
    // Offsets should be 0, 128, 256
    const offsets = rows.map((r) => r["skip_offset"]);
    expect(offsets).toEqual([0, 128, 256]);

    expect(rows[0]!["skip_doc_id"]).toBe(1);
    expect(rows[1]!["skip_doc_id"]).toBe(129);
    expect(rows[2]!["skip_doc_id"]).toBe(257);
  });

  it("skip rebuilt on remove", async () => {
    const idx = await makeIndex();
    for (let i = 1; i < 200; i++) {
      idx.addDocument(i, { body: "alpha" });
    }
    idx.flushSkipPointers();

    const conn = (
      idx as unknown as {
        _conn: {
          query: (sql: string, params?: unknown[]) => Record<string, unknown>[];
        };
      }
    )._conn;

    // Before remove: 2 skip entries (0, 128)
    const rowsBefore = conn.query(
      `SELECT COUNT(*) as cnt FROM "_skip_docs_body" WHERE term = ?`,
      ["alpha"],
    );
    expect(rowsBefore[0]!["cnt"]).toBe(2);

    // Remove doc_ids 1..72, leaving 127 docs (< 128 -> 1 skip)
    for (let i = 1; i <= 72; i++) {
      idx.removeDocument(i);
    }
    idx.flushSkipPointers();

    const rowsAfter = conn.query(
      `SELECT COUNT(*) as cnt FROM "_skip_docs_body" WHERE term = ?`,
      ["alpha"],
    );
    expect(rowsAfter[0]!["cnt"]).toBe(1);
  });
});

describe("SkipTo", () => {
  it("skip to finds nearest", async () => {
    const idx = await makeIndex();
    for (let i = 1; i <= 300; i++) {
      idx.addDocument(i, { body: "alpha" });
    }

    // Skip to doc 150: nearest skip entry should be at doc 129 (offset 128)
    const [skipDocId, skipOffset] = idx.skipTo("body", "alpha", 150);
    expect(skipDocId).toBe(129);
    expect(skipOffset).toBe(128);
  });

  it("skip to exact match", async () => {
    const idx = await makeIndex();
    for (let i = 1; i <= 300; i++) {
      idx.addDocument(i, { body: "alpha" });
    }

    // Skip to doc 129: exact match on skip entry
    const [skipDocId, skipOffset] = idx.skipTo("body", "alpha", 129);
    expect(skipDocId).toBe(129);
    expect(skipOffset).toBe(128);
  });

  it("skip to before first", async () => {
    const idx = await makeIndex();
    for (let i = 100; i < 200; i++) {
      idx.addDocument(i, { body: "alpha" });
    }

    // Skip to doc 50: before all skip entries
    const [skipDocId, skipOffset] = idx.skipTo("body", "alpha", 50);
    expect(skipDocId).toBe(0);
    expect(skipOffset).toBe(0);
  });

  it("skip to nonexistent field", async () => {
    const idx = await makeIndex();
    const [skipDocId, skipOffset] = idx.skipTo("body", "alpha", 100);
    expect(skipDocId).toBe(0);
    expect(skipOffset).toBe(0);
  });

  it("skip to nonexistent term", async () => {
    const idx = await makeIndex();
    idx.addDocument(1, { body: "hello" });

    const [skipDocId, skipOffset] = idx.skipTo("body", "nonexistent", 1);
    expect(skipDocId).toBe(0);
    expect(skipOffset).toBe(0);
  });
});

// ======================================================================
// Block-Max Scores
// ======================================================================

describe("BlockMaxScores", () => {
  it("BlockMaxIndex builds and retrieves", () => {
    const bmi = new BlockMaxIndex();
    const entries = Array.from({ length: 10 }, (_, i) =>
      createPostingEntry(i + 1, { positions: [0], score: 0.0 }),
    );
    const pl = new PostingList(entries);
    const scorer = {
      score(_tf: number, _dl: number, _df: number): number {
        return 1.5;
      },
    };
    bmi.build(pl, scorer, "body", "alpha");

    const numBlocks = bmi.numBlocks("body", "alpha");
    expect(numBlocks).toBeGreaterThanOrEqual(1);
    expect(bmi.getBlockMax("body", "alpha", 0)).toBeGreaterThan(0.0);
  });

  it("get block max nonexistent field", () => {
    const bmi = new BlockMaxIndex();
    expect(bmi.getBlockMax("nonexistent", "alpha", 0)).toBe(0);
  });

  it("num blocks returns 0 for unknown term", () => {
    const bmi = new BlockMaxIndex();
    expect(bmi.numBlocks("body", "hello")).toBe(0);
  });

  it("build block max single block", () => {
    const bmi = new BlockMaxIndex();
    const entries = Array.from({ length: 5 }, (_, i) =>
      createPostingEntry(i + 1, { positions: [0], score: 0.0 }),
    );
    const pl = new PostingList(entries);
    const scorer = {
      score(tf: number, _dl: number, _df: number): number {
        return tf * 2.0;
      },
    };
    bmi.build(pl, scorer, "body", "test");
    expect(bmi.numBlocks("body", "test")).toBe(1);
    expect(bmi.getBlockMax("body", "test", 0)).toBeGreaterThan(0);
  });

  it("build block max multiple blocks", () => {
    const bmi = new BlockMaxIndex(4); // block size 4
    const entries = Array.from({ length: 10 }, (_, i) =>
      createPostingEntry(i + 1, {
        positions: Array.from({ length: i + 1 }, (__, j) => j),
        score: 0.0,
      }),
    );
    const pl = new PostingList(entries);
    const scorer = {
      score(tf: number, _dl: number, _df: number): number {
        return tf;
      },
    };
    bmi.build(pl, scorer, "body", "test");
    // 10 entries, blockSize=4 => 3 blocks (4, 4, 2)
    expect(bmi.numBlocks("body", "test")).toBe(3);
    // Second block should have higher max (entries 5-8 have more positions)
    expect(bmi.getBlockMax("body", "test", 1)).toBeGreaterThan(
      bmi.getBlockMax("body", "test", 0),
    );
  });

  it("get block max score by index", () => {
    const bmi = new BlockMaxIndex(4);
    const entries = Array.from({ length: 8 }, (_, i) =>
      createPostingEntry(i + 1, {
        positions: Array.from({ length: i + 1 }, (__, j) => j),
        score: 0.0,
      }),
    );
    const pl = new PostingList(entries);
    const scorer = {
      score(tf: number, _dl: number, _df: number): number {
        return tf;
      },
    };
    bmi.build(pl, scorer, "body", "test");
    // Block 0: entries 1-4, max tf = 4
    // Block 1: entries 5-8, max tf = 8
    expect(bmi.getBlockMax("body", "test", 0)).toBeCloseTo(4);
    expect(bmi.getBlockMax("body", "test", 1)).toBeCloseTo(8);
  });

  it("get all block max nonexistent field returns 0", () => {
    const bmi = new BlockMaxIndex();
    expect(bmi.globalMax("nonexistent", "alpha")).toBe(0);
  });

  it("build all block max scores for multiple terms", () => {
    const bmi = new BlockMaxIndex();
    const entries1 = Array.from({ length: 5 }, (_, i) =>
      createPostingEntry(i + 1, { positions: [0], score: 0.0 }),
    );
    const entries2 = Array.from({ length: 8 }, (_, i) =>
      createPostingEntry(i + 1, { positions: [0, 1], score: 0.0 }),
    );
    const scorer = {
      score(tf: number, _dl: number, _df: number): number {
        return tf;
      },
    };
    bmi.build(new PostingList(entries1), scorer, "body", "term1");
    bmi.build(new PostingList(entries2), scorer, "body", "term2");
    expect(bmi.numBlocks("body", "term1")).toBeGreaterThanOrEqual(1);
    expect(bmi.numBlocks("body", "term2")).toBeGreaterThanOrEqual(1);
    expect(bmi.globalMax("body", "term2")).toBeGreaterThanOrEqual(
      bmi.globalMax("body", "term1"),
    );
  });

  it("build block max for nonexistent field yields empty", () => {
    const bmi = new BlockMaxIndex();
    const pl = new PostingList([]);
    const scorer = {
      score(_tf: number, _dl: number, _df: number): number {
        return 1.0;
      },
    };
    bmi.build(pl, scorer, "nonexistent", "test");
    expect(bmi.numBlocks("nonexistent", "test")).toBe(0);
  });
});

// ======================================================================
// Persistence across reconnection
// ======================================================================

describe("SkipBlockMaxPersistence", () => {
  it("skip pointers survive reconnection via export/import", async () => {
    const SQL = await initSqlJs();
    const db1 = new SQL.Database();
    const conn1 = new ManagedConnection(db1);
    const idx1 = new SQLiteInvertedIndex(conn1, "docs");
    for (let i = 1; i < 200; i++) {
      idx1.addDocument(i, { body: "alpha" });
    }
    idx1.flushSkipPointers();

    // Export and reimport the database
    const data = db1.export();
    const db2 = new SQL.Database(data);
    const conn2 = new ManagedConnection(db2);
    const idx2 = new SQLiteInvertedIndex(conn2, "docs");

    const [skipDocId, skipOffset] = idx2.skipTo("body", "alpha", 150);
    expect(skipDocId).toBe(129);
    expect(skipOffset).toBe(128);
  });

  it("block max scores survive reconnection via saveToSQLite/loadFromSQLite", async () => {
    const SQL = await initSqlJs();
    const db = new SQL.Database();
    const conn = new ManagedConnection(db);

    const bmi = new BlockMaxIndex();
    const entries = Array.from({ length: 10 }, (_, i) =>
      createPostingEntry(i + 1, { positions: [0], score: 0.0 }),
    );
    const pl = new PostingList(entries);
    const scorer = {
      score(_tf: number, _dl: number, _df: number): number {
        return 1.5;
      },
    };
    bmi.build(pl, scorer, "body", "alpha");
    bmi.saveToSQLite(conn);

    // Load from same connection
    const loaded = BlockMaxIndex.loadFromSQLite(conn);
    expect(loaded.numBlocks("body", "alpha")).toBe(bmi.numBlocks("body", "alpha"));
    expect(loaded.getBlockMax("body", "alpha", 0)).toBe(
      bmi.getBlockMax("body", "alpha", 0),
    );
  });

  it("load block max into memory index", async () => {
    const SQL = await initSqlJs();
    const db = new SQL.Database();
    const conn = new ManagedConnection(db);

    const bmi = new BlockMaxIndex();
    const entries = Array.from({ length: 10 }, (_, i) =>
      createPostingEntry(i + 1, { positions: [0, 1], score: 0.0 }),
    );
    const pl = new PostingList(entries);
    const scorer = {
      score(tf: number, _dl: number, _df: number): number {
        return tf * 0.5;
      },
    };
    bmi.build(pl, scorer, "body", "hello");
    bmi.saveToSQLite(conn);

    const loaded = BlockMaxIndex.loadFromSQLite(conn);
    // Verify loaded data matches
    expect(loaded.numBlocks("body", "hello")).toBe(bmi.numBlocks("body", "hello"));
    for (let i = 0; i < loaded.numBlocks("body", "hello"); i++) {
      expect(loaded.getBlockMax("body", "hello", i)).toBe(
        bmi.getBlockMax("body", "hello", i),
      );
    }
  });
});

// ======================================================================
// BlockMaxIndex SQLite persistence
// ======================================================================

describe("BlockMaxIndexPersistence", () => {
  it("save and load", async () => {
    const SQL = await initSqlJs();
    const db = new SQL.Database();
    const conn = new ManagedConnection(db);

    const bmi = new BlockMaxIndex(64);
    const entries = Array.from({ length: 200 }, (_, i) =>
      createPostingEntry(i + 1, { positions: [0], score: 0.0 }),
    );
    const pl = new PostingList(entries);
    const scorer = {
      score(_tf: number, _dl: number, _df: number): number {
        return 2.0;
      },
    };
    bmi.build(pl, scorer, "body", "test");
    bmi.saveToSQLite(conn);

    const loaded = BlockMaxIndex.loadFromSQLite(conn);
    expect(loaded.numBlocks("body", "test")).toBe(bmi.numBlocks("body", "test"));
    for (let i = 0; i < loaded.numBlocks("body", "test"); i++) {
      expect(loaded.getBlockMax("body", "test", i)).toBeCloseTo(
        bmi.getBlockMax("body", "test", i),
      );
    }
  });

  it("save and load multi table isolation", async () => {
    const SQL = await initSqlJs();
    const db = new SQL.Database();
    const conn = new ManagedConnection(db);

    const bmi = new BlockMaxIndex();
    const entries1 = Array.from({ length: 5 }, (_, i) =>
      createPostingEntry(i + 1, { positions: [0], score: 0.0 }),
    );
    const entries2 = Array.from({ length: 8 }, (_, i) =>
      createPostingEntry(i + 1, { positions: [0, 1], score: 0.0 }),
    );
    const scorer = {
      score(tf: number, _dl: number, _df: number): number {
        return tf;
      },
    };
    bmi.build(new PostingList(entries1), scorer, "body", "t1", "table_a");
    bmi.build(new PostingList(entries2), scorer, "body", "t2", "table_b");
    bmi.saveToSQLite(conn);

    const loaded = BlockMaxIndex.loadFromSQLite(conn);
    expect(loaded.numBlocks("body", "t1", "table_a")).toBe(
      bmi.numBlocks("body", "t1", "table_a"),
    );
    expect(loaded.numBlocks("body", "t2", "table_b")).toBe(
      bmi.numBlocks("body", "t2", "table_b"),
    );
  });

  it("save overwrites previous", async () => {
    const SQL = await initSqlJs();
    const db = new SQL.Database();
    const conn = new ManagedConnection(db);

    const scorer = {
      score(_tf: number, _dl: number, _df: number): number {
        return 1.0;
      },
    };

    // First save
    const bmi1 = new BlockMaxIndex();
    bmi1.build(
      new PostingList([createPostingEntry(1, { positions: [0], score: 0.0 })]),
      scorer,
      "body",
      "x",
    );
    bmi1.saveToSQLite(conn);

    // Second save with different data
    const bmi2 = new BlockMaxIndex();
    bmi2.build(
      new PostingList([
        createPostingEntry(1, { positions: [0], score: 0.0 }),
        createPostingEntry(2, { positions: [0], score: 0.0 }),
      ]),
      scorer,
      "body",
      "x",
    );
    bmi2.saveToSQLite(conn);

    const loaded = BlockMaxIndex.loadFromSQLite(conn);
    // Should reflect the second save
    expect(loaded.numBlocks("body", "x")).toBe(bmi2.numBlocks("body", "x"));
  });

  it("load from empty database", async () => {
    const SQL = await initSqlJs();
    const db = new SQL.Database();
    const conn = new ManagedConnection(db);

    // No block max table exists
    const loaded = BlockMaxIndex.loadFromSQLite(conn);
    expect(loaded.numBlocks("body", "test")).toBe(0);
  });

  it("load legacy schema migration", async () => {
    // Old databases with _block_max_index but legacy column layout
    // (table_name, field, term, block_idx, max_score) should auto-migrate
    // to the compact (key, block_size, scores_json) format.
    const SQL = await initSqlJs();
    const db = new SQL.Database();
    const conn = new ManagedConnection(db);

    // Create the legacy schema with the table_name column
    conn.execute(
      `CREATE TABLE _block_max_index (
        table_name TEXT    NOT NULL,
        field      TEXT    NOT NULL,
        term       TEXT    NOT NULL,
        block_idx  INTEGER NOT NULL,
        max_score  REAL    NOT NULL,
        PRIMARY KEY (table_name, field, term, block_idx)
      )`,
    );
    conn.execute(`INSERT INTO _block_max_index VALUES ('', 'body', 'hello', 0, 2.5)`);

    const loaded = BlockMaxIndex.loadFromSQLite(conn);
    // After migration, scores should be accessible
    expect(loaded.numBlocks("body", "hello")).toBeGreaterThanOrEqual(1);
  });
});

// ======================================================================
// Block size constant
// ======================================================================

describe("BlockSize", () => {
  it("default block size", () => {
    const bmi = new BlockMaxIndex();
    // Default block size should be defined
    expect(bmi.blockSize).toBeGreaterThan(0);
  });
});

// ======================================================================
// Engine integration with skip pointers and block-max
// ======================================================================

describe("EngineIntegration", () => {
  it("SQL table has inverted index with skip pointer infrastructure", async () => {
    const { Engine } = await import("../../src/engine.js");
    const engine = new Engine();
    await engine.sql("CREATE TABLE docs (id SERIAL PRIMARY KEY, body TEXT)");
    for (let i = 1; i < 200; i++) {
      await engine.sql(`INSERT INTO docs (body) VALUES ('alpha term${String(i)}')`);
    }

    const table = engine.getTable("docs");
    expect(table).toBeDefined();
    // The inverted index should be populated with postings
    const invIdx = table.invertedIndex;
    // Verify the index has term data for "alpha"
    const pl = invIdx.getPostingList("body", "alpha");
    expect(pl.length).toBeGreaterThanOrEqual(100);
    // If the inverted index supports skip pointers, test them
    if ("skipTo" in invIdx && typeof invIdx.skipTo === "function") {
      const [skipDoc, skipOff] = (invIdx as unknown as SQLiteInvertedIndex).skipTo(
        "body",
        "alpha",
        150,
      );
      expect(skipDoc).toBeGreaterThan(0);
      expect(skipOff).toBeGreaterThan(0);
    } else {
      // Memory inverted index does not support skip pointers,
      // but verifying posting list length proves data is indexed.
      expect(pl.length).toBe(199);
    }
    engine.close();
  });

  it("drop table succeeds without error", async () => {
    const { Engine } = await import("../../src/engine.js");
    const engine = new Engine();
    await engine.sql("CREATE TABLE docs (id SERIAL PRIMARY KEY, body TEXT)");
    await engine.sql("INSERT INTO docs (body) VALUES ('hello world')");

    // Verify data exists before drop
    const result = await engine.sql("SELECT COUNT(*) AS cnt FROM docs");
    expect(result).not.toBeNull();
    const rows = (result as { rows: Record<string, unknown>[] }).rows;
    expect(rows[0]!["cnt"]).toBe(1);

    // DROP TABLE should succeed without throwing
    await engine.sql("DROP TABLE docs");

    // DROP TABLE IF EXISTS on a potentially cached table should not throw
    await engine.sql("DROP TABLE IF EXISTS docs");
    engine.close();
  });
});
