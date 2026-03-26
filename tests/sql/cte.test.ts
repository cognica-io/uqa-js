import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// Helpers
// =============================================================================

async function makeCTEEngine(): Promise<Engine> {
  const e = new Engine();
  await e.sql("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT NOT NULL)");
  await e.sql(
    "INSERT INTO departments (id, name) VALUES " +
      "(1, 'Engineering'), " +
      "(2, 'Marketing'), " +
      "(3, 'Sales')",
  );
  await e.sql(
    "CREATE TABLE employees (" +
      "id INTEGER PRIMARY KEY, " +
      "name TEXT NOT NULL, " +
      "dept_id INTEGER, " +
      "salary REAL" +
      ")",
  );
  await e.sql(
    "INSERT INTO employees (id, name, dept_id, salary) VALUES " +
      "(1, 'Alice', 1, 90000), " +
      "(2, 'Bob', 2, 75000), " +
      "(3, 'Carol', 1, 85000), " +
      "(4, 'Dave', 3, 70000), " +
      "(5, 'Eve', 1, 95000)",
  );
  return e;
}

// =============================================================================
// Basic CTE
// =============================================================================

describe("CTEBasic", () => {
  // Python parity gap: not yet implemented in TS SQL compiler
  it("simple cte", async () => {
    const e = await makeCTEEngine();
    const r = await e.sql(
      "WITH eng AS (" +
        "  SELECT name FROM employees WHERE dept_id = 1" +
        ") " +
        "SELECT name FROM eng ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice", "Carol", "Eve"]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("cte with filter", async () => {
    const e = await makeCTEEngine();
    const r = await e.sql(
      "WITH high_sal AS (" +
        "  SELECT name, salary FROM employees WHERE salary > 80000" +
        ") " +
        "SELECT name FROM high_sal WHERE salary > 90000",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Eve"]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("cte with aggregate", async () => {
    const e = await makeCTEEngine();
    const r = await e.sql(
      "WITH dept_stats AS (" +
        "  SELECT dept_id, COUNT(*) AS cnt, AVG(salary) AS avg_sal " +
        "  FROM employees GROUP BY dept_id" +
        ") " +
        "SELECT dept_id, cnt FROM dept_stats ORDER BY dept_id",
    );
    expect(r!.rows[0]!["dept_id"]).toBe(1);
    expect(r!.rows[0]!["cnt"]).toBe(3);
    expect(r!.rows[1]!["cnt"]).toBe(1);
    expect(r!.rows[2]!["cnt"]).toBe(1);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("cte with order and limit", async () => {
    const e = await makeCTEEngine();
    const r = await e.sql(
      "WITH ranked AS (" +
        "  SELECT name, salary FROM employees ORDER BY salary DESC" +
        ") " +
        "SELECT name FROM ranked LIMIT 3",
    );
    const names = r!.rows.map((row) => row["name"]);
    expect(names.length).toBe(3);
    expect(names[0]).toBe("Eve");
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("cte cleanup", async () => {
    const e = await makeCTEEngine();
    await e.sql("WITH temp_cte AS (SELECT 1 AS val) SELECT val FROM temp_cte");
    // The CTE table should not exist after the query
    expect(e.hasTable("temp_cte")).toBe(false);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("cte does not shadow real table", async () => {
    const e = await makeCTEEngine();
    await e.sql("WITH x AS (SELECT name FROM employees LIMIT 1) SELECT name FROM x");
    const r = await e.sql("SELECT COUNT(*) AS cnt FROM employees");
    expect(r!.rows[0]!["cnt"]).toBe(5);
  });
});

// =============================================================================
// Multiple CTEs
// =============================================================================

describe("MultipleCTEs", () => {
  // Python parity gap: not yet implemented in TS SQL compiler
  it("two ctes", async () => {
    const e = await makeCTEEngine();
    const r = await e.sql(
      "WITH " +
        "  eng AS (SELECT id, name FROM employees WHERE dept_id = 1), " +
        "  mkt AS (SELECT id, name FROM employees WHERE dept_id = 2) " +
        "SELECT name FROM eng ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice", "Carol", "Eve"]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("second cte used", async () => {
    const e = await makeCTEEngine();
    const r = await e.sql(
      "WITH " +
        "  eng AS (SELECT name FROM employees WHERE dept_id = 1), " +
        "  mkt AS (SELECT name FROM employees WHERE dept_id = 2) " +
        "SELECT name FROM mkt",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Bob"]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("cte referencing another", async () => {
    const e = await makeCTEEngine();
    const r = await e.sql(
      "WITH " +
        "  high_sal AS (SELECT name, salary FROM employees WHERE salary > 80000), " +
        "  very_high AS (SELECT name FROM high_sal WHERE salary > 90000) " +
        "SELECT name FROM very_high",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Eve"]);
  });
});

// =============================================================================
// CTE with subqueries
// =============================================================================

describe("CTEWithSubquery", () => {
  // Python parity gap: not yet implemented in TS SQL compiler
  it("cte used in subquery", async () => {
    const e = await makeCTEEngine();
    const r = await e.sql(
      "WITH eng_ids AS (" +
        "  SELECT id FROM employees WHERE dept_id = 1" +
        ") " +
        "SELECT name FROM employees " +
        "WHERE id IN (SELECT id FROM eng_ids) " +
        "ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice", "Carol", "Eve"]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("cte with distinct", async () => {
    const e = await makeCTEEngine();
    const r = await e.sql(
      "WITH dept_ids AS (" +
        "  SELECT DISTINCT dept_id FROM employees" +
        ") " +
        "SELECT dept_id FROM dept_ids ORDER BY dept_id",
    );
    expect(r!.rows.map((row) => row["dept_id"])).toEqual([1, 2, 3]);
  });
});

// =============================================================================
// WITH RECURSIVE
// =============================================================================

describe("WithRecursive", () => {
  // Python parity gap: not yet implemented in TS SQL compiler
  it("recursive count", async () => {
    const e = new Engine();
    const result = await e.sql(
      "WITH RECURSIVE cnt(x) AS (" +
        "  SELECT 1" +
        "  UNION ALL" +
        "  SELECT x + 1 FROM cnt WHERE x < 5" +
        ") SELECT x FROM cnt",
    );
    const values = result!.rows.map((r) => r["x"]);
    expect(values).toEqual([1, 2, 3, 4, 5]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("recursive union dedup", async () => {
    const e = new Engine();
    const result = await e.sql(
      "WITH RECURSIVE seq(n) AS (" +
        "  SELECT 1" +
        "  UNION" +
        "  SELECT n + 1 FROM seq WHERE n < 3" +
        ") SELECT n FROM seq",
    );
    const values = result!.rows.map((r) => r["n"] as number).sort((a, b) => a - b);
    expect(values).toEqual([1, 2, 3]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("recursive hierarchy", async () => {
    const e = new Engine();
    await e.sql(
      "CREATE TABLE employees (" +
        "  eid INTEGER PRIMARY KEY, ename TEXT, manager_id INTEGER" +
        ")",
    );
    await e.sql("INSERT INTO employees (eid, ename, manager_id) VALUES (1, 'CEO', 0)");
    await e.sql("INSERT INTO employees (eid, ename, manager_id) VALUES (2, 'VP', 1)");
    await e.sql(
      "INSERT INTO employees (eid, ename, manager_id) VALUES (3, 'Manager', 2)",
    );
    await e.sql(
      "INSERT INTO employees (eid, ename, manager_id) VALUES (4, 'Developer', 3)",
    );
    const result = await e.sql(
      "WITH RECURSIVE chain(cid, cname, lvl) AS (" +
        "  SELECT eid, ename, 0 FROM employees WHERE eid = 1" +
        "  UNION ALL" +
        "  SELECT e.eid, e.ename, c.lvl + 1 " +
        "  FROM employees e " +
        "  INNER JOIN chain c ON e.manager_id = c.cid" +
        ") SELECT cname, lvl FROM chain",
    );
    expect(result!.rows.length).toBe(4);
    const names = new Set(result!.rows.map((r) => r["cname"]));
    expect(names).toEqual(new Set(["CEO", "VP", "Manager", "Developer"]));
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("recursive empty base", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)");
    const result = await e.sql(
      "WITH RECURSIVE r(n) AS (" +
        "  SELECT id FROM t WHERE id = 999" +
        "  UNION ALL" +
        "  SELECT n + 1 FROM r WHERE n < 5" +
        ") SELECT n FROM r",
    );
    expect(result!.rows.length).toBe(0);
  });
});
