import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// Helpers
// =============================================================================

async function makeCorrelatedEngine(): Promise<Engine> {
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
// Correlated scalar subquery
// =============================================================================

describe("CorrelatedScalar", () => {
  it("correlated subqueries basic", async () => {
    const e = await makeCorrelatedEngine();
    const r = await e.sql(
      "SELECT name, " +
        "(SELECT COUNT(*) FROM employees e2 WHERE e2.dept_id = e.dept_id) AS dept_size " +
        "FROM employees e WHERE id = 1",
    );
    expect(r!.rows[0]!["name"]).toBe("Alice");
    expect(r!.rows[0]!["dept_size"]).toBe(3);
  });

  it("salary above dept avg", async () => {
    const e = await makeCorrelatedEngine();
    const r = await e.sql(
      "SELECT name FROM employees e " +
        "WHERE salary > (" +
        "  SELECT AVG(salary) FROM employees e2 WHERE e2.dept_id = e.dept_id" +
        ") ORDER BY name",
    );
    // avg(dept_id=1) = (90000+85000+95000)/3 = 90000; Alice=90000 is not > avg
    expect(r!.rows.map((row) => row["name"])).toEqual(["Eve"]);
  });

  it("salary equal dept max", async () => {
    const e = await makeCorrelatedEngine();
    const r = await e.sql(
      "SELECT name FROM employees e " +
        "WHERE salary = (" +
        "  SELECT MAX(salary) FROM employees e2 WHERE e2.dept_id = e.dept_id" +
        ") ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Bob", "Dave", "Eve"]);
  });

  it("correlated count", async () => {
    const e = await makeCorrelatedEngine();
    const r = await e.sql(
      "SELECT d.name, " +
        "(SELECT COUNT(*) FROM employees e WHERE e.dept_id = d.id) AS emp_count " +
        "FROM departments d ORDER BY d.name",
    );
    expect(r!.rows.map((row) => row["emp_count"])).toEqual([3, 1, 1]);
  });
});

// =============================================================================
// Correlated EXISTS
// =============================================================================

describe("CorrelatedExists", () => {
  it("exists basic", async () => {
    const e = await makeCorrelatedEngine();
    const r = await e.sql(
      "SELECT name FROM departments d " +
        "WHERE EXISTS (SELECT 1 FROM employees e WHERE e.dept_id = d.id) " +
        "ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual([
      "Engineering",
      "Marketing",
      "Sales",
    ]);
  });

  it("not exists", async () => {
    const e = await makeCorrelatedEngine();
    await e.sql("INSERT INTO departments (id, name) VALUES (4, 'HR')");
    const r = await e.sql(
      "SELECT name FROM departments d " +
        "WHERE NOT EXISTS (SELECT 1 FROM employees e WHERE e.dept_id = d.id) " +
        "ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["HR"]);
  });

  it("exists with additional condition", async () => {
    const e = await makeCorrelatedEngine();
    const r = await e.sql(
      "SELECT name FROM departments d " +
        "WHERE EXISTS (" +
        "  SELECT 1 FROM employees e WHERE e.dept_id = d.id AND e.salary > 80000" +
        ") ORDER BY name",
    );
    // Engineering has Alice(90k), Carol(85k), Eve(95k); Marketing & Sales have none > 80k
    expect(r!.rows.map((row) => row["name"])).toEqual(["Engineering"]);
  });
});

// =============================================================================
// Correlated IN
// =============================================================================

describe("CorrelatedIn", () => {
  it("correlated IN subquery", async () => {
    const e = await makeCorrelatedEngine();
    const r = await e.sql(
      "SELECT name FROM departments d " +
        "WHERE id IN (SELECT dept_id FROM employees WHERE salary > 80000) " +
        "ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Engineering"]);
  });

  it("correlated in", async () => {
    const e = await makeCorrelatedEngine();
    const r = await e.sql(
      "SELECT name FROM employees " +
        "WHERE dept_id IN (SELECT id FROM departments WHERE name = 'Engineering') " +
        "ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice", "Carol", "Eve"]);
  });
});

// =============================================================================
// Edge cases
// =============================================================================

describe("CorrelatedEdgeCases", () => {
  it("correlated with min", async () => {
    const e = await makeCorrelatedEngine();
    const r = await e.sql(
      "SELECT name FROM employees e " +
        "WHERE salary = (" +
        "  SELECT MIN(salary) FROM employees e2 WHERE e2.dept_id = e.dept_id" +
        ") ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Bob", "Carol", "Dave"]);
  });

  it("correlated subquery in and", async () => {
    const e = await makeCorrelatedEngine();
    const r = await e.sql(
      "SELECT name FROM employees e " +
        "WHERE salary > 80000 AND salary < (" +
        "  SELECT MAX(salary) FROM employees e2 WHERE e2.dept_id = e.dept_id" +
        ") ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice", "Carol"]);
  });

  it("non correlated still works", async () => {
    const e = await makeCorrelatedEngine();
    const r = await e.sql(
      "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees) ORDER BY name",
    );
    // avg = (90000+75000+85000+70000+95000)/5 = 83000
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice", "Carol", "Eve"]);
  });

  it("exists non correlated still works", async () => {
    const e = await makeCorrelatedEngine();
    const r = await e.sql(
      "SELECT name FROM employees WHERE EXISTS (SELECT 1 FROM departments WHERE name = 'Engineering') ORDER BY name",
    );
    expect(r!.rows.length).toBe(5);
  });
});
