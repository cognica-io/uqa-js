import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// Helpers
// =============================================================================

async function makeViewsEngine(): Promise<Engine> {
  const e = new Engine();
  await e.sql(
    "CREATE TABLE employees (" +
      "id INTEGER PRIMARY KEY, " +
      "name TEXT NOT NULL, " +
      "dept TEXT, " +
      "salary REAL" +
      ")",
  );
  await e.sql(
    "INSERT INTO employees (id, name, dept, salary) VALUES " +
      "(1, 'Alice', 'eng', 90000), " +
      "(2, 'Bob', 'mkt', 75000), " +
      "(3, 'Carol', 'eng', 85000), " +
      "(4, 'Dave', 'sales', 70000), " +
      "(5, 'Eve', 'eng', 95000), " +
      "(6, 'Frank', 'mkt', 80000)",
  );
  return e;
}

// =============================================================================
// CREATE VIEW
// =============================================================================

describe("CreateView", () => {
  it("create view basic", async () => {
    const e = await makeViewsEngine();
    await e.sql(
      "CREATE VIEW eng_employees AS " +
        "SELECT name, salary FROM employees WHERE dept = 'eng'",
    );
    expect(e.compiler.views.has("eng_employees")).toBe(true);
  });

  it("create view duplicate raises", async () => {
    const e = await makeViewsEngine();
    await e.sql("CREATE VIEW v AS SELECT name FROM employees");
    await expect(e.sql("CREATE VIEW v AS SELECT name FROM employees")).rejects.toThrow(
      /already exists/,
    );
  });

  it("create view name conflicts with table", async () => {
    const e = await makeViewsEngine();
    await expect(
      e.sql("CREATE VIEW employees AS SELECT name FROM employees"),
    ).rejects.toThrow(/already exists as a table/);
  });
});

// =============================================================================
// SELECT from view
// =============================================================================

describe("SelectFromView", () => {
  it("select all from view", async () => {
    const e = await makeViewsEngine();
    await e.sql(
      "CREATE VIEW eng AS SELECT name, salary FROM employees WHERE dept = 'eng'",
    );
    const r = await e.sql("SELECT name FROM eng ORDER BY name");
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice", "Carol", "Eve"]);
  });

  it("view with filter", async () => {
    const e = await makeViewsEngine();
    await e.sql(
      "CREATE VIEW high_sal AS " +
        "SELECT name, salary FROM employees WHERE salary > 80000",
    );
    const r = await e.sql("SELECT name FROM high_sal WHERE salary > 90000");
    expect(r!.rows.map((row) => row["name"])).toEqual(["Eve"]);
  });

  it("view with aggregate", async () => {
    const e = await makeViewsEngine();
    await e.sql(
      "CREATE VIEW dept_stats AS " +
        "SELECT dept, COUNT(*) AS cnt, AVG(salary) AS avg_sal " +
        "FROM employees GROUP BY dept",
    );
    const r = await e.sql("SELECT dept, cnt FROM dept_stats ORDER BY dept");
    expect(r!.rows[0]!["dept"]).toBe("eng");
    expect(r!.rows[0]!["cnt"]).toBe(3);
  });

  it("view with order and limit", async () => {
    const e = await makeViewsEngine();
    await e.sql(
      "CREATE VIEW ranked AS " +
        "SELECT name, salary FROM employees ORDER BY salary DESC",
    );
    const r = await e.sql("SELECT name FROM ranked LIMIT 3");
    expect(r!.rows.length).toBe(3);
    expect(r!.rows[0]!["name"]).toBe("Eve");
  });

  it("view preserves column types", async () => {
    const e = await makeViewsEngine();
    await e.sql("CREATE VIEW v AS SELECT name, salary FROM employees");
    const r = await e.sql("SELECT salary FROM v WHERE name = 'Alice'");
    expect(r!.rows[0]!["salary"]).toBe(90000.0);
  });

  it("view with distinct", async () => {
    const e = await makeViewsEngine();
    await e.sql("CREATE VIEW depts AS SELECT DISTINCT dept FROM employees");
    const r = await e.sql("SELECT dept FROM depts ORDER BY dept");
    expect(r!.rows.map((row) => row["dept"])).toEqual(["eng", "mkt", "sales"]);
  });
});

// =============================================================================
// View cleanup
// =============================================================================

describe("ViewCleanup", () => {
  it("view does not leak temp table", async () => {
    const e = await makeViewsEngine();
    await e.sql("CREATE VIEW v AS SELECT name FROM employees");
    await e.sql("SELECT name FROM v");
    expect(e.hasTable("v")).toBe(false);
    expect(e.compiler.views.has("v")).toBe(true);
  });

  it("view does not shadow real table", async () => {
    const e = await makeViewsEngine();
    await e.sql("CREATE VIEW v AS SELECT name FROM employees LIMIT 1");
    await e.sql("SELECT name FROM v");
    const r = await e.sql("SELECT COUNT(*) AS cnt FROM employees");
    expect(r!.rows[0]!["cnt"]).toBe(6);
  });

  it("multiple view queries", async () => {
    const e = await makeViewsEngine();
    await e.sql("CREATE VIEW v AS SELECT name, salary FROM employees");
    const r1 = await e.sql("SELECT COUNT(*) AS cnt FROM v");
    const r2 = await e.sql("SELECT name FROM v WHERE salary > 90000");
    expect(r1!.rows[0]!["cnt"]).toBe(6);
    expect(r2!.rows.map((row) => row["name"])).toEqual(["Eve"]);
  });
});

// =============================================================================
// DROP VIEW
// =============================================================================

describe("DropView", () => {
  it("drop view", async () => {
    const e = await makeViewsEngine();
    await e.sql("CREATE VIEW v AS SELECT name FROM employees");
    await e.sql("DROP VIEW v");
    expect(e.compiler.views.has("v")).toBe(false);
  });

  it("drop view if exists", async () => {
    const e = await makeViewsEngine();
    // Should not raise
    await e.sql("DROP VIEW IF EXISTS nonexistent");
  });

  it("drop view nonexistent raises", async () => {
    const e = await makeViewsEngine();
    await expect(e.sql("DROP VIEW nonexistent")).rejects.toThrow(/does not exist/);
  });

  it("drop view then select raises", async () => {
    const e = await makeViewsEngine();
    await e.sql("CREATE VIEW v AS SELECT name FROM employees");
    await e.sql("DROP VIEW v");
    await expect(e.sql("SELECT name FROM v")).rejects.toThrow(/does not exist/);
  });

  it("recreate view after drop", async () => {
    const e = await makeViewsEngine();
    await e.sql("CREATE VIEW v AS SELECT name FROM employees WHERE dept = 'eng'");
    await e.sql("DROP VIEW v");
    await e.sql("CREATE VIEW v AS SELECT name FROM employees WHERE dept = 'mkt'");
    const r = await e.sql("SELECT name FROM v ORDER BY name");
    expect(r!.rows.map((row) => row["name"])).toEqual(["Bob", "Frank"]);
  });
});

// =============================================================================
// Views with other SQL features
// =============================================================================

describe("ViewIntegration", () => {
  it("view reflects data changes", async () => {
    const e = await makeViewsEngine();
    await e.sql("CREATE VIEW v AS SELECT name, salary FROM employees");
    await e.sql(
      "INSERT INTO employees (id, name, dept, salary) VALUES " +
        "(7, 'Grace', 'eng', 100000)",
    );
    const r = await e.sql("SELECT COUNT(*) AS cnt FROM v");
    expect(r!.rows[0]!["cnt"]).toBe(7);
  });

  it("view with window function", async () => {
    const e = await makeViewsEngine();
    await e.sql(
      "CREATE VIEW ranked AS " +
        "SELECT name, salary, " +
        "ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn " +
        "FROM employees",
    );
    const r = await e.sql("SELECT name, rn FROM ranked WHERE rn <= 3 ORDER BY rn");
    expect(r!.rows.length).toBe(3);
    expect(r!.rows[0]!["name"]).toBe("Eve");
  });

  it("view used in subquery", async () => {
    const e = await makeViewsEngine();
    await e.sql("CREATE VIEW eng_ids AS SELECT id FROM employees WHERE dept = 'eng'");
    const r = await e.sql(
      "SELECT name FROM employees " +
        "WHERE id IN (SELECT id FROM eng_ids) " +
        "ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice", "Carol", "Eve"]);
  });

  it("view of view", async () => {
    const e = await makeViewsEngine();
    await e.sql(
      "CREATE VIEW high_sal AS " +
        "SELECT name, salary FROM employees WHERE salary > 80000",
    );
    await e.sql(
      "CREATE VIEW very_high AS SELECT name FROM high_sal WHERE salary > 90000",
    );
    const r = await e.sql("SELECT name FROM very_high");
    expect(r!.rows.map((row) => row["name"])).toEqual(["Eve"]);
  });

  it("cte and view together", async () => {
    const e = await makeViewsEngine();
    await e.sql(
      "CREATE VIEW eng AS SELECT name, salary FROM employees WHERE dept = 'eng'",
    );
    const r = await e.sql(
      "WITH top AS (SELECT name FROM eng WHERE salary > 90000) " +
        "SELECT name FROM top",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Eve"]);
  });
});
