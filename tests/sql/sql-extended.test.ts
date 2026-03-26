import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";
import { Table, createColumnDef } from "../../src/sql/table.js";

// =============================================================================
// Helpers: create tables via Table API, register with Engine for SQL queries
// =============================================================================

function makeEngine(): Engine {
  const e = new Engine();
  const table = new Table("papers", [
    createColumnDef("id", "INTEGER", { primaryKey: true, pythonType: "number" }),
    createColumnDef("title", "TEXT"),
    createColumnDef("year", "INTEGER", { pythonType: "number" }),
    createColumnDef("venue", "TEXT"),
    createColumnDef("field", "TEXT"),
    createColumnDef("citations", "INTEGER", { pythonType: "number" }),
  ]);
  table.insert({
    id: 1,
    title: "attention is all you need",
    year: 2017,
    venue: "NeurIPS",
    field: "nlp",
    citations: 90000,
  });
  table.insert({
    id: 2,
    title: "bert pre-training of deep bidirectional transformers",
    year: 2019,
    venue: "NAACL",
    field: "nlp",
    citations: 75000,
  });
  table.insert({
    id: 3,
    title: "graph attention networks",
    year: 2018,
    venue: "ICLR",
    field: "graph",
    citations: 15000,
  });
  table.insert({
    id: 4,
    title: "vision transformer for image recognition",
    year: 2021,
    venue: "ICLR",
    field: "cv",
    citations: 25000,
  });
  table.insert({
    id: 5,
    title: "scaling language models methods and insights",
    year: 2020,
    venue: "arXiv",
    field: "nlp",
    citations: 8000,
  });
  e.registerTable("papers", table);
  return e;
}

function makeUserEngine(): Engine {
  const e = new Engine();
  const t = new Table("users", [
    createColumnDef("id", "INTEGER", { primaryKey: true, pythonType: "number" }),
    createColumnDef("name", "TEXT"),
    createColumnDef("age", "INTEGER", { pythonType: "number" }),
  ]);
  t.insert({ id: 1, name: "Alice", age: 30 });
  t.insert({ id: 2, name: "Bob", age: 25 });
  t.insert({ id: 3, name: "Carol", age: 35 });
  t.insert({ id: 4, name: "Dave", age: 25 });
  e.registerTable("users", t);
  return e;
}

// =============================================================================
// DQL: Basic SELECT
// =============================================================================

describe("BasicSelect", () => {
  it("select all", async () => {
    const engine = makeEngine();
    const result = await engine.sql("SELECT * FROM papers");
    expect(result!.rows.length).toBe(5);
  });

  it("select columns", async () => {
    const engine = makeEngine();
    const result = await engine.sql("SELECT title, year FROM papers");
    expect(result!.columns).toEqual(["title", "year"]);
    expect(result!.rows.length).toBe(5);
  });

  it("select order by asc", async () => {
    const engine = makeEngine();
    const result = await engine.sql("SELECT title, year FROM papers ORDER BY year");
    const years = result!.rows.map((r: Record<string, unknown>) => r["year"] as number);
    expect(years).toEqual([...years].sort((a, b) => a - b));
  });

  it("select order by desc", async () => {
    const engine = makeEngine();
    const result = await engine.sql(
      "SELECT title, year FROM papers ORDER BY year DESC",
    );
    const years = result!.rows.map((r: Record<string, unknown>) => r["year"] as number);
    expect(years).toEqual([...years].sort((a, b) => b - a));
  });
});

// =============================================================================
// DQL: WHERE clause
// =============================================================================

describe("WhereClause", () => {
  it("greater than", async () => {
    const engine = makeEngine();
    const result = await engine.sql("SELECT * FROM papers WHERE year > 2019");
    for (const row of result!.rows) {
      expect((row as Record<string, unknown>)["year"] as number).toBeGreaterThan(2019);
    }
  });

  it("and", async () => {
    const engine = makeEngine();
    const result = await engine.sql(
      "SELECT * FROM papers WHERE field = 'nlp' AND year >= 2019",
    );
    for (const row of result!.rows) {
      expect((row as Record<string, unknown>)["field"]).toBe("nlp");
      expect((row as Record<string, unknown>)["year"] as number).toBeGreaterThanOrEqual(
        2019,
      );
    }
  });

  it("complex boolean", async () => {
    const engine = makeEngine();
    const result = await engine.sql(
      "SELECT * FROM papers WHERE (field = 'nlp' OR field = 'cv') AND year >= 2020",
    );
    for (const row of result!.rows) {
      expect(["nlp", "cv"]).toContain((row as Record<string, unknown>)["field"]);
      expect((row as Record<string, unknown>)["year"] as number).toBeGreaterThanOrEqual(
        2020,
      );
    }
  });
});

// =============================================================================
// DQL: Aggregation
// =============================================================================

describe("Aggregation", () => {
  it("group by", async () => {
    const engine = makeEngine();
    const result = await engine.sql(
      "SELECT field, COUNT(*) AS cnt FROM papers GROUP BY field ORDER BY cnt DESC",
    );
    expect(result!.rows.length).toBeGreaterThanOrEqual(3);
    const counts = result!.rows.map((r: Record<string, unknown>) => r["cnt"] as number);
    expect(counts).toEqual([...counts].sort((a, b) => b - a));
  });

  it("having", async () => {
    const engine = makeEngine();
    const result = await engine.sql(
      "SELECT field, COUNT(*) AS cnt FROM papers " +
        "GROUP BY field HAVING COUNT(*) >= 2",
    );
    for (const row of result!.rows) {
      expect((row as Record<string, unknown>)["cnt"] as number).toBeGreaterThanOrEqual(
        2,
      );
    }
  });
});

// =============================================================================
// DISTINCT
// =============================================================================

describe("Distinct", () => {
  it("select distinct", async () => {
    const engine = makeEngine();
    const result = await engine.sql("SELECT DISTINCT field FROM papers");
    const fields = result!.rows.map((r: Record<string, unknown>) => r["field"]);
    expect(fields.length).toBe(new Set(fields).size);
  });

  it("distinct with order by", async () => {
    const engine = makeEngine();
    const result = await engine.sql("SELECT DISTINCT field FROM papers ORDER BY field");
    const fields = result!.rows.map(
      (r: Record<string, unknown>) => r["field"] as string,
    );
    expect(fields).toEqual([...fields].sort());
    expect(fields.length).toBe(new Set(fields).size);
  });

  it("distinct preserves all unique rows", async () => {
    const engine = makeEngine();
    const allResult = await engine.sql("SELECT year FROM papers");
    const distinctResult = await engine.sql("SELECT DISTINCT year FROM papers");
    const allYears = new Set(
      allResult!.rows.map((r: Record<string, unknown>) => r["year"]),
    );
    const distinctYears = new Set(
      distinctResult!.rows.map((r: Record<string, unknown>) => r["year"]),
    );
    expect(allYears).toEqual(distinctYears);
  });
});

// =============================================================================
// Edge cases
// =============================================================================

describe("EdgeCases", () => {
  it("no where clause returns all rows", async () => {
    const engine = makeEngine();
    const result = await engine.sql("SELECT * FROM papers");
    expect(result!.rows.length).toBe(5);
  });

  it("empty result", async () => {
    const engine = makeEngine();
    const result = await engine.sql("SELECT * FROM papers WHERE year > 9999");
    expect(result!.rows.length).toBe(0);
  });
});

// =============================================================================
// Column alias in ORDER BY
// =============================================================================

describe("ColumnAliasOrderBy", () => {
  it("order by alias", async () => {
    const e = makeUserEngine();
    const result = await e.sql(
      "SELECT name, age AS user_age FROM users ORDER BY user_age DESC",
    );
    const ages = result!.rows.map((r: Record<string, unknown>) => r["user_age"]);
    expect(ages).toEqual([35, 30, 25, 25]);
  });
});

// =============================================================================
// UNION / INTERSECT / EXCEPT
// =============================================================================

describe("SetOperations", () => {
  function makeTwoTables(): Engine {
    const e = new Engine();
    const t1 = new Table("t1", [
      createColumnDef("id", "INTEGER"),
      createColumnDef("val", "TEXT"),
    ]);
    t1.insert({ id: 1, val: "a" });
    t1.insert({ id: 2, val: "b" });
    t1.insert({ id: 3, val: "c" });
    e.registerTable("t1", t1);

    const t2 = new Table("t2", [
      createColumnDef("id", "INTEGER"),
      createColumnDef("val", "TEXT"),
    ]);
    t2.insert({ id: 2, val: "b" });
    t2.insert({ id: 3, val: "c" });
    t2.insert({ id: 4, val: "d" });
    e.registerTable("t2", t2);
    return e;
  }

  it("union all", async () => {
    const e = makeTwoTables();
    const result = await e.sql(
      "SELECT id, val FROM t1 UNION ALL SELECT id, val FROM t2",
    );
    expect(result!.rows.length).toBe(6);
  });

  it("union distinct", async () => {
    const e = makeTwoTables();
    const result = await e.sql("SELECT id, val FROM t1 UNION SELECT id, val FROM t2");
    expect(result!.rows.length).toBe(4);
  });

  it("intersect", async () => {
    const e = makeTwoTables();
    const result = await e.sql(
      "SELECT id, val FROM t1 INTERSECT SELECT id, val FROM t2",
    );
    expect(result!.rows.length).toBe(2);
  });

  it("except", async () => {
    const e = makeTwoTables();
    const result = await e.sql("SELECT id, val FROM t1 EXCEPT SELECT id, val FROM t2");
    expect(result!.rows.length).toBe(1);
    expect(result!.rows[0]!["val"]).toBe("a");
  });
});

// =============================================================================
// Table API
// =============================================================================

describe("Table API", () => {
  it("table insert and query", () => {
    const table = new Table("test", [
      createColumnDef("id", "INTEGER", { primaryKey: true, autoIncrement: true }),
      createColumnDef("name", "TEXT"),
    ]);
    table.insert({ name: "alice" });
    table.insert({ name: "bob" });
    expect(table.rowCount).toBe(2);
  });

  it("enforces NOT NULL", () => {
    const table = new Table("test", [
      createColumnDef("id", "INTEGER", { primaryKey: true }),
      createColumnDef("name", "TEXT", { notNull: true }),
    ]);
    expect(() => table.insert({ id: 1 })).toThrow();
  });

  it("auto increment", () => {
    const table = new Table("test", [
      createColumnDef("id", "INTEGER", { primaryKey: true, autoIncrement: true }),
      createColumnDef("name", "TEXT"),
    ]);
    const [id1] = table.insert({ name: "a" });
    const [id2] = table.insert({ name: "b" });
    expect(id1).toBe(1);
    expect(id2).toBe(2);
  });

  it("analyze produces stats", () => {
    const table = new Table("test", [
      createColumnDef("id", "INTEGER", { primaryKey: true, pythonType: "number" }),
      createColumnDef("year", "INTEGER", { pythonType: "number" }),
    ]);
    table.insert({ id: 1, year: 2017 });
    table.insert({ id: 2, year: 2019 });
    table.insert({ id: 3, year: 2021 });
    const stats = table.analyze();
    const yearStats = stats.get("year");
    expect(yearStats).toBeDefined();
    expect(yearStats!.minValue).toBe(2017);
    expect(yearStats!.maxValue).toBe(2021);
    expect(yearStats!.rowCount).toBe(3);
  });
});

// =============================================================================
// NULLS ordering
// =============================================================================

describe("NullsOrdering", () => {
  function makeNullData(): Engine {
    const e = new Engine();
    const t = new Table("t", [
      createColumnDef("id", "INTEGER", { pythonType: "number" }),
      createColumnDef("val", "INTEGER", { pythonType: "number" }),
    ]);
    t.insert({ id: 1, val: 10 });
    t.insert({ id: 2, val: 20 });
    t.insert({ id: 3, val: null });
    t.insert({ id: 4, val: 5 });
    e.registerTable("t", t);
    return e;
  }

  it("nulls first asc", async () => {
    const e = makeNullData();
    const result = await e.sql("SELECT id, val FROM t ORDER BY val ASC NULLS FIRST");
    const vals = result!.rows.map((r: Record<string, unknown>) => r["val"]);
    expect(vals[0]).toBeNull();
    expect(vals.slice(1)).toEqual([5, 10, 20]);
  });

  it("nulls last asc", async () => {
    const e = makeNullData();
    const result = await e.sql("SELECT id, val FROM t ORDER BY val ASC NULLS LAST");
    const vals = result!.rows.map((r: Record<string, unknown>) => r["val"]);
    expect(vals[vals.length - 1]).toBeNull();
    expect(vals.slice(0, -1)).toEqual([5, 10, 20]);
  });

  it("default nulls last", async () => {
    const e = makeNullData();
    const result = await e.sql("SELECT id, val FROM t ORDER BY val ASC");
    const vals = result!.rows.map((r: Record<string, unknown>) => r["val"]);
    expect(vals[vals.length - 1]).toBeNull();
  });
});
