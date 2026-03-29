import { describe, expect, it, beforeEach } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// FDW integration tests -- exercise the full SQL compiler path:
//   CREATE FOREIGN SERVER -> CREATE FOREIGN TABLE -> SELECT
//
// Both DuckDB FDW and Arrow FDW handlers are tested via their inline JSON
// data fallback paths (no actual DuckDB WASM or Arrow IPC runtime required).
// =============================================================================

const EMPLOYEES_JSON = JSON.stringify([
  { id: 1, name: "Alice", age: 30, dept: "eng", salary: 90000 },
  { id: 2, name: "Bob", age: 25, dept: "sales", salary: 70000 },
  { id: 3, name: "Charlie", age: 35, dept: "eng", salary: 110000 },
  { id: 4, name: "Diana", age: 28, dept: "hr", salary: 65000 },
  { id: 5, name: "Eve", age: 32, dept: "eng", salary: 95000 },
]);

const DEPARTMENTS_JSON = JSON.stringify([
  { dept: "eng", budget: 500000 },
  { dept: "sales", budget: 200000 },
  { dept: "hr", budget: 150000 },
]);

// =============================================================================
// DuckDB FDW -- inline JSON data path
// =============================================================================

describe("DuckDBFDWIntegration", () => {
  let engine: Engine;

  beforeEach(async () => {
    engine = new Engine();
    await engine.sql(
      "CREATE SERVER myserver FOREIGN DATA WRAPPER duckdb_fdw",
    );
    await engine.sql(`
      CREATE FOREIGN TABLE employees (
        id INTEGER,
        name TEXT,
        age INTEGER,
        dept TEXT,
        salary INTEGER
      ) SERVER myserver OPTIONS (data '${EMPLOYEES_JSON.replace(/'/g, "''")}')
    `);
  });

  it("select all rows", async () => {
    const result = await engine.sql("SELECT * FROM employees");
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(5);
  });

  it("select with column projection", async () => {
    const result = await engine.sql("SELECT name, age FROM employees");
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(5);
    expect(result!.rows[0]).toHaveProperty("name");
    expect(result!.rows[0]).toHaveProperty("age");
  });

  it("select with WHERE equality", async () => {
    const result = await engine.sql(
      "SELECT * FROM employees WHERE dept = 'eng'",
    );
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(3);
    for (const row of result!.rows) {
      expect(row["dept"]).toBe("eng");
    }
  });

  it("select with WHERE comparison", async () => {
    const result = await engine.sql(
      "SELECT name, age FROM employees WHERE age > 30",
    );
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(2);
    for (const row of result!.rows) {
      expect(row["age"] as number).toBeGreaterThan(30);
    }
  });

  it("select with LIMIT", async () => {
    const result = await engine.sql("SELECT * FROM employees LIMIT 2");
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(2);
  });

  it("select with ORDER BY", async () => {
    const result = await engine.sql(
      "SELECT name, salary FROM employees ORDER BY salary DESC",
    );
    expect(result).not.toBeNull();
    const salaries = result!.rows.map((r) => r["salary"] as number);
    for (let i = 1; i < salaries.length; i++) {
      expect(salaries[i]!).toBeLessThanOrEqual(salaries[i - 1]!);
    }
  });

  it("select with aggregate", async () => {
    const result = await engine.sql(
      "SELECT dept, COUNT(*) AS cnt, AVG(salary) AS avg_sal FROM employees GROUP BY dept ORDER BY dept",
    );
    expect(result).not.toBeNull();
    const engRow = result!.rows.find((r) => r["dept"] === "eng");
    expect(engRow).toBeDefined();
    expect(engRow!["cnt"]).toBe(3);
  });

  it("aliased foreign table", async () => {
    const result = await engine.sql(
      "SELECT e.name FROM employees e WHERE e.age = 30",
    );
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(1);
    expect(result!.rows[0]!["name"]).toBe("Alice");
  });

  it("foreign table in information_schema", async () => {
    const result = await engine.sql(
      "SELECT table_name, table_type FROM information_schema.tables WHERE table_type = 'FOREIGN TABLE'",
    );
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(1);
    expect(result!.rows[0]!["table_name"]).toBe("employees");
    expect(result!.rows[0]!["table_type"]).toBe("FOREIGN TABLE");
  });

  it("foreign table columns in information_schema", async () => {
    const result = await engine.sql(
      "SELECT column_name FROM information_schema.columns WHERE table_name = 'employees' ORDER BY ordinal_position",
    );
    expect(result).not.toBeNull();
    const cols = result!.rows.map((r) => r["column_name"]);
    expect(cols).toEqual(["id", "name", "age", "dept", "salary"]);
  });

  it("drop foreign table then select raises", async () => {
    await engine.sql("DROP FOREIGN TABLE employees");
    await expect(engine.sql("SELECT * FROM employees")).rejects.toThrow();
  });

  it("drop server with dependent table raises", async () => {
    await expect(
      engine.sql("DROP SERVER myserver"),
    ).rejects.toThrow(/depends on it/);
  });

  it("drop foreign table then drop server", async () => {
    await engine.sql("DROP FOREIGN TABLE employees");
    await engine.sql("DROP SERVER myserver");
    await expect(
      engine.sql("SELECT * FROM employees"),
    ).rejects.toThrow();
  });

  it("engine close cleans up", () => {
    engine.close();
    // No error should be thrown
  });
});

// =============================================================================
// Arrow FDW -- inline JSON data path
// =============================================================================

describe("ArrowFDWIntegration", () => {
  let engine: Engine;

  beforeEach(async () => {
    engine = new Engine();
    await engine.sql(
      "CREATE SERVER arrowserver FOREIGN DATA WRAPPER arrow_fdw",
    );
    await engine.sql(`
      CREATE FOREIGN TABLE employees (
        id INTEGER,
        name TEXT,
        age INTEGER,
        dept TEXT,
        salary INTEGER
      ) SERVER arrowserver OPTIONS (data '${EMPLOYEES_JSON.replace(/'/g, "''")}')
    `);
  });

  it("select all rows", async () => {
    const result = await engine.sql("SELECT * FROM employees");
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(5);
  });

  it("select with WHERE equality", async () => {
    const result = await engine.sql(
      "SELECT * FROM employees WHERE dept = 'eng'",
    );
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(3);
  });

  it("select with column projection", async () => {
    const result = await engine.sql("SELECT name, salary FROM employees");
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(5);
    expect(result!.rows[0]).toHaveProperty("name");
    expect(result!.rows[0]).toHaveProperty("salary");
  });

  it("select with LIMIT", async () => {
    const result = await engine.sql("SELECT * FROM employees LIMIT 3");
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(3);
  });

  it("select with aggregate", async () => {
    const result = await engine.sql(
      "SELECT COUNT(*) AS cnt FROM employees",
    );
    expect(result).not.toBeNull();
    expect(result!.rows[0]!["cnt"]).toBe(5);
  });

  it("foreign table in information_schema", async () => {
    const result = await engine.sql(
      "SELECT table_type FROM information_schema.tables WHERE table_name = 'employees'",
    );
    expect(result).not.toBeNull();
    expect(result!.rows[0]!["table_type"]).toBe("FOREIGN TABLE");
  });
});

// =============================================================================
// Mixed queries: foreign table + local table
// =============================================================================

describe("FDWMixedQueries", () => {
  let engine: Engine;

  beforeEach(async () => {
    engine = new Engine();

    // Foreign table
    await engine.sql(
      "CREATE SERVER myserver FOREIGN DATA WRAPPER duckdb_fdw",
    );
    await engine.sql(`
      CREATE FOREIGN TABLE employees (
        id INTEGER,
        name TEXT,
        age INTEGER,
        dept TEXT,
        salary INTEGER
      ) SERVER myserver OPTIONS (data '${EMPLOYEES_JSON.replace(/'/g, "''")}')
    `);

    // Local table
    await engine.sql(
      "CREATE TABLE departments (dept TEXT PRIMARY KEY, budget INTEGER)",
    );
    await engine.sql(
      "INSERT INTO departments (dept, budget) VALUES ('eng', 500000)",
    );
    await engine.sql(
      "INSERT INTO departments (dept, budget) VALUES ('sales', 200000)",
    );
    await engine.sql(
      "INSERT INTO departments (dept, budget) VALUES ('hr', 150000)",
    );
  });

  it("join foreign table with local table", async () => {
    const result = await engine.sql(
      "SELECT e.name, e.salary, d.budget FROM employees e JOIN departments d ON e.dept = d.dept ORDER BY e.name",
    );
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(5);
    const alice = result!.rows.find((r) => r["name"] === "Alice");
    expect(alice).toBeDefined();
    expect(alice!["budget"]).toBe(500000);
  });

  it("subquery on foreign table", async () => {
    const result = await engine.sql(
      "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)",
    );
    expect(result).not.toBeNull();
    // avg salary = (90000+70000+110000+65000+95000)/5 = 86000
    // Charlie(110000) and Eve(95000) and Alice(90000) are above
    expect(result!.rows.length).toBe(3);
  });
});

// =============================================================================
// Multiple foreign servers
// =============================================================================

describe("FDWMultipleServers", () => {
  it("two duckdb servers", async () => {
    const engine = new Engine();
    await engine.sql("CREATE SERVER srv1 FOREIGN DATA WRAPPER duckdb_fdw");
    await engine.sql("CREATE SERVER srv2 FOREIGN DATA WRAPPER duckdb_fdw");

    await engine.sql(`
      CREATE FOREIGN TABLE employees (id INTEGER, name TEXT, dept TEXT)
      SERVER srv1 OPTIONS (data '${JSON.stringify([
        { id: 1, name: "Alice", dept: "eng" },
        { id: 2, name: "Bob", dept: "sales" },
      ]).replace(/'/g, "''")}')
    `);
    await engine.sql(`
      CREATE FOREIGN TABLE departments (dept TEXT, budget INTEGER)
      SERVER srv2 OPTIONS (data '${DEPARTMENTS_JSON.replace(/'/g, "''")}')
    `);

    const result = await engine.sql(
      "SELECT e.name, d.budget FROM employees e JOIN departments d ON e.dept = d.dept ORDER BY e.name",
    );
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(2);
    expect(result!.rows[0]!["name"]).toBe("Alice");
    expect(result!.rows[0]!["budget"]).toBe(500000);
  });

  it("mixed duckdb and arrow servers", async () => {
    const engine = new Engine();
    await engine.sql("CREATE SERVER duck FOREIGN DATA WRAPPER duckdb_fdw");
    await engine.sql("CREATE SERVER arrow FOREIGN DATA WRAPPER arrow_fdw");

    await engine.sql(`
      CREATE FOREIGN TABLE emp (id INTEGER, name TEXT, dept TEXT)
      SERVER duck OPTIONS (data '${JSON.stringify([
        { id: 1, name: "Alice", dept: "eng" },
      ]).replace(/'/g, "''")}')
    `);
    await engine.sql(`
      CREATE FOREIGN TABLE dept (dept TEXT, budget INTEGER)
      SERVER arrow OPTIONS (data '${JSON.stringify([
        { dept: "eng", budget: 500000 },
      ]).replace(/'/g, "''")}')
    `);

    const result = await engine.sql(
      "SELECT emp.name, dept.budget FROM emp JOIN dept ON emp.dept = dept.dept",
    );
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(1);
    expect(result!.rows[0]!["name"]).toBe("Alice");
    expect(result!.rows[0]!["budget"]).toBe(500000);
  });
});

// =============================================================================
// Server options propagation
// =============================================================================

describe("FDWServerOptions", () => {
  it("data option on server propagates to handler", async () => {
    const engine = new Engine();
    // Set data on the server, not the foreign table
    await engine.sql(`
      CREATE SERVER srv FOREIGN DATA WRAPPER duckdb_fdw
      OPTIONS (data '${EMPLOYEES_JSON.replace(/'/g, "''")}')
    `);
    await engine.sql(`
      CREATE FOREIGN TABLE employees (
        id INTEGER, name TEXT, age INTEGER, dept TEXT, salary INTEGER
      ) SERVER srv
    `);

    const result = await engine.sql("SELECT * FROM employees");
    expect(result).not.toBeNull();
    expect(result!.rows.length).toBe(5);
  });
});

// =============================================================================
// Error cases
// =============================================================================

describe("FDWErrors", () => {
  it("unsupported fdw type", async () => {
    const engine = new Engine();
    await expect(
      engine.sql("CREATE SERVER srv FOREIGN DATA WRAPPER postgres_fdw"),
    ).rejects.toThrow(/Unsupported FDW type/);
  });

  it("create foreign table on nonexistent server", async () => {
    const engine = new Engine();
    await expect(
      engine.sql(
        "CREATE FOREIGN TABLE t (id INTEGER) SERVER nosuchserver",
      ),
    ).rejects.toThrow(/does not exist/);
  });

  it("duplicate foreign server", async () => {
    const engine = new Engine();
    await engine.sql("CREATE SERVER srv FOREIGN DATA WRAPPER duckdb_fdw");
    await expect(
      engine.sql("CREATE SERVER srv FOREIGN DATA WRAPPER duckdb_fdw"),
    ).rejects.toThrow(/already exists/);
  });

  it("duplicate foreign table", async () => {
    const engine = new Engine();
    await engine.sql("CREATE SERVER srv FOREIGN DATA WRAPPER duckdb_fdw");
    await engine.sql(
      `CREATE FOREIGN TABLE t (id INTEGER) SERVER srv OPTIONS (data '[]')`,
    );
    await expect(
      engine.sql(
        `CREATE FOREIGN TABLE t (id INTEGER) SERVER srv OPTIONS (data '[]')`,
      ),
    ).rejects.toThrow(/already exists/);
  });

  it("foreign table name conflicts with regular table", async () => {
    const engine = new Engine();
    await engine.sql("CREATE TABLE t (id INTEGER)");
    await engine.sql("CREATE SERVER srv FOREIGN DATA WRAPPER duckdb_fdw");
    await expect(
      engine.sql(
        `CREATE FOREIGN TABLE t (id INTEGER) SERVER srv OPTIONS (data '[]')`,
      ),
    ).rejects.toThrow(/already exists/);
  });

  it("if not exists on duplicate server", async () => {
    const engine = new Engine();
    await engine.sql("CREATE SERVER srv FOREIGN DATA WRAPPER duckdb_fdw");
    // Should not throw
    await engine.sql(
      "CREATE SERVER IF NOT EXISTS srv FOREIGN DATA WRAPPER duckdb_fdw",
    );
  });

  it("drop server if exists on nonexistent", async () => {
    const engine = new Engine();
    // Should not throw
    await engine.sql("DROP SERVER IF EXISTS nosuchserver");
  });

  it("drop foreign table if exists on nonexistent", async () => {
    const engine = new Engine();
    // Should not throw
    await engine.sql("DROP FOREIGN TABLE IF EXISTS nosuch");
  });
});
