import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// Helpers
// =============================================================================

async function makeSubqueryEngine(): Promise<Engine> {
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
      "(5, 'Eve', NULL, 95000)",
  );
  return e;
}

// =============================================================================
// IN (SELECT ...)
// =============================================================================

describe("InSubquery", () => {
  // Python parity gap: not yet implemented in TS SQL compiler
  it("in subquery basic", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name FROM employees " +
        "WHERE dept_id IN (SELECT id FROM departments WHERE name = 'Engineering') " +
        "ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice", "Carol"]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("in subquery multiple values", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name FROM employees " +
        "WHERE dept_id IN (SELECT id FROM departments WHERE name != 'Sales') " +
        "ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice", "Bob", "Carol"]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("in subquery no match", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name FROM employees " +
        "WHERE dept_id IN (SELECT id FROM departments WHERE name = 'HR')",
    );
    expect(r!.rows).toEqual([]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("in subquery null excluded", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name FROM employees WHERE dept_id IN (SELECT id FROM departments)",
    );
    const names = r!.rows.map((row) => row["name"] as string).sort();
    expect(names).not.toContain("Eve");
    expect(names.length).toBe(4);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("not in subquery", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name FROM employees " +
        "WHERE dept_id NOT IN (SELECT id FROM departments WHERE name = 'Engineering') " +
        "ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Bob", "Dave", "Eve"]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("in subquery with aggregate", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name FROM employees " +
        "WHERE salary IN (" +
        "  SELECT MAX(salary) AS max_sal FROM employees" +
        ") " +
        "ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Eve"]);
  });
});

// =============================================================================
// EXISTS (SELECT ...)
// =============================================================================

describe("ExistsSubquery", () => {
  // Python parity gap: not yet implemented in TS SQL compiler
  it("exists true", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name FROM employees " +
        "WHERE EXISTS (SELECT 1 FROM departments WHERE name = 'Engineering') " +
        "ORDER BY name",
    );
    expect(r!.rows.length).toBe(5);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("exists false", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name FROM employees " +
        "WHERE EXISTS (SELECT 1 FROM departments WHERE name = 'HR')",
    );
    expect(r!.rows).toEqual([]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("not exists", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name FROM employees " +
        "WHERE NOT EXISTS (SELECT 1 FROM departments WHERE name = 'HR') " +
        "ORDER BY name",
    );
    expect(r!.rows.length).toBe(5);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("not exists with results", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name FROM employees WHERE NOT EXISTS (SELECT 1 FROM departments)",
    );
    expect(r!.rows).toEqual([]);
  });
});

// =============================================================================
// Scalar subquery in SELECT
// =============================================================================

describe("ScalarSubquery", () => {
  // Python parity gap: not yet implemented in TS SQL compiler
  it("scalar subquery in select", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name, " +
        "(SELECT COUNT(*) FROM departments) AS dept_count " +
        "FROM employees ORDER BY name LIMIT 1",
    );
    expect(r!.rows[0]!["name"]).toBe("Alice");
    expect(r!.rows[0]!["dept_count"]).toBe(3);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("scalar subquery aggregate", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name, " +
        "(SELECT MAX(salary) FROM employees) AS max_salary " +
        "FROM employees WHERE id = 1",
    );
    expect(r!.rows[0]!["max_salary"]).toBe(95000);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("scalar subquery empty", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name, " +
        "(SELECT salary FROM employees WHERE id = 999) AS other_sal " +
        "FROM employees WHERE id = 1",
    );
    expect(r!.rows[0]!["other_sal"] ?? null).toBeNull();
  });
});

// =============================================================================
// Subqueries with other features
// =============================================================================

describe("SubqueryIntegration", () => {
  // Python parity gap: not yet implemented in TS SQL compiler
  it("in subquery with like", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name FROM employees " +
        "WHERE dept_id IN (" +
        "  SELECT id FROM departments WHERE name LIKE 'Eng%'" +
        ") ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice", "Carol"]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("in subquery with order and limit", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name FROM employees " +
        "WHERE dept_id IN (SELECT id FROM departments) " +
        "ORDER BY name LIMIT 2",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice", "Bob"]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("in subquery with and", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name FROM employees " +
        "WHERE dept_id IN (SELECT id FROM departments WHERE name = 'Engineering') " +
        "AND salary > 87000",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice"]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("multiple subquery conditions", async () => {
    const e = await makeSubqueryEngine();
    const r = await e.sql(
      "SELECT name FROM employees " +
        "WHERE dept_id IN (SELECT id FROM departments) " +
        "AND salary IN (SELECT salary FROM employees WHERE salary >= 85000) " +
        "ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice", "Carol"]);
  });
});
