import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// Helpers
// =============================================================================

async function makeOrdersEngine(): Promise<Engine> {
  const e = new Engine();
  await e.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)");
  await e.sql("INSERT INTO users (id, name) VALUES (1, 'Alice')");
  await e.sql("INSERT INTO users (id, name) VALUES (2, 'Bob')");
  await e.sql("INSERT INTO users (id, name) VALUES (3, 'Carol')");
  await e.sql(
    "CREATE TABLE orders (oid INTEGER PRIMARY KEY, user_id INTEGER, product TEXT)",
  );
  await e.sql("INSERT INTO orders (oid, user_id, product) VALUES (10, 1, 'Book')");
  await e.sql("INSERT INTO orders (oid, user_id, product) VALUES (11, 1, 'Pen')");
  await e.sql("INSERT INTO orders (oid, user_id, product) VALUES (12, 2, 'Notebook')");
  return e;
}

// =============================================================================
// INNER JOIN
// =============================================================================

describe("InnerJoin", () => {
  it("inner join basic", async () => {
    const e = await makeOrdersEngine();
    const result = await e.sql(
      "SELECT users.name, orders.product " +
        "FROM users INNER JOIN orders ON users.id = orders.user_id",
    );
    expect(result!.rows.length).toBe(3);
    const products = new Set(result!.rows.map((r) => r["product"]));
    expect(products).toEqual(new Set(["Book", "Pen", "Notebook"]));
  });

  it("inner join excludes unmatched", async () => {
    const e = await makeOrdersEngine();
    const result = await e.sql(
      "SELECT users.name " +
        "FROM users INNER JOIN orders ON users.id = orders.user_id",
    );
    const names = new Set(result!.rows.map((r) => r["name"]));
    expect(names).not.toContain("Carol");
  });
});

// =============================================================================
// LEFT JOIN
// =============================================================================

describe("LeftJoin", () => {
  it("left join preserves left", async () => {
    const e = await makeOrdersEngine();
    const result = await e.sql(
      "SELECT users.name, orders.product " +
        "FROM users LEFT JOIN orders ON users.id = orders.user_id",
    );
    expect(result!.rows.length).toBe(4);
    const names = new Set(result!.rows.map((r) => r["name"]));
    expect(names).toContain("Carol");
  });

  it("left join null for unmatched", async () => {
    const e = await makeOrdersEngine();
    const result = await e.sql(
      "SELECT users.name, orders.product " +
        "FROM users LEFT JOIN orders ON users.id = orders.user_id",
    );
    const carolRows = result!.rows.filter((r) => r["name"] === "Carol");
    expect(carolRows.length).toBe(1);
    expect(carolRows[0]!["product"] ?? null).toBeNull();
  });
});

// =============================================================================
// CROSS JOIN
// =============================================================================

describe("CrossJoin", () => {
  it("cross join cartesian", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE a (id INTEGER PRIMARY KEY, val TEXT)");
    await e.sql("INSERT INTO a (id, val) VALUES (1, 'x')");
    await e.sql("INSERT INTO a (id, val) VALUES (2, 'y')");
    await e.sql("CREATE TABLE b (id INTEGER PRIMARY KEY, label TEXT)");
    await e.sql("INSERT INTO b (id, label) VALUES (10, 'p')");
    await e.sql("INSERT INTO b (id, label) VALUES (20, 'q')");
    await e.sql("INSERT INTO b (id, label) VALUES (30, 'r')");
    const result = await e.sql("SELECT a.val, b.label FROM a CROSS JOIN b");
    expect(result!.rows.length).toBe(6);
  });

  it("cross join empty side", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE a (id INTEGER PRIMARY KEY, val TEXT)");
    await e.sql("INSERT INTO a (id, val) VALUES (1, 'x')");
    await e.sql("CREATE TABLE b (id INTEGER PRIMARY KEY, label TEXT)");
    const result = await e.sql("SELECT * FROM a CROSS JOIN b");
    expect(result!.rows.length).toBe(0);
  });
});

// =============================================================================
// RIGHT JOIN
// =============================================================================

describe("RightJoin", () => {
  it("right join preserves right", async () => {
    const e = await makeOrdersEngine();
    await e.sql("INSERT INTO orders (oid, user_id, product) VALUES (13, 99, 'Ghost')");
    const result = await e.sql(
      "SELECT users.name, orders.product " +
        "FROM users RIGHT JOIN orders ON users.id = orders.user_id",
    );
    const products = new Set(result!.rows.map((r) => r["product"]));
    expect(products).toContain("Ghost");
    expect(result!.rows.length).toBe(4);
  });

  it("right join null for unmatched left", async () => {
    const e = await makeOrdersEngine();
    await e.sql("INSERT INTO orders (oid, user_id, product) VALUES (13, 99, 'Ghost')");
    const result = await e.sql(
      "SELECT users.name, orders.product " +
        "FROM users RIGHT JOIN orders ON users.id = orders.user_id",
    );
    const ghostRows = result!.rows.filter((r) => r["product"] === "Ghost");
    expect(ghostRows.length).toBe(1);
    expect(ghostRows[0]!["name"] ?? null).toBeNull();
  });
});

// =============================================================================
// FULL OUTER JOIN
// =============================================================================

describe("FullOuterJoin", () => {
  it("full join preserves both", async () => {
    const e = await makeOrdersEngine();
    await e.sql("INSERT INTO orders (oid, user_id, product) VALUES (13, 99, 'Ghost')");
    const result = await e.sql(
      "SELECT users.name, orders.product " +
        "FROM users FULL OUTER JOIN orders " +
        "ON users.id = orders.user_id",
    );
    expect(result!.rows.length).toBe(5);
    const names = new Set(result!.rows.map((r) => r["name"] ?? null));
    expect(names).toContain("Carol");
    const products = new Set(result!.rows.map((r) => r["product"] ?? null));
    expect(products).toContain("Ghost");
  });

  it("full join no overlap", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE a (id INTEGER PRIMARY KEY, val TEXT)");
    await e.sql("INSERT INTO a (id, val) VALUES (1, 'x')");
    await e.sql("CREATE TABLE b (id INTEGER PRIMARY KEY, val TEXT)");
    await e.sql("INSERT INTO b (id, val) VALUES (2, 'y')");
    const result = await e.sql("SELECT * FROM a FULL OUTER JOIN b ON a.id = b.id");
    expect(result!.rows.length).toBe(2);
  });
});

// =============================================================================
// Multiple FROM tables
// =============================================================================

describe("MultipleFromTables", () => {
  it("implicit cross join", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE a (id INTEGER PRIMARY KEY, val TEXT)");
    await e.sql("INSERT INTO a (id, val) VALUES (1, 'x')");
    await e.sql("INSERT INTO a (id, val) VALUES (2, 'y')");
    await e.sql("CREATE TABLE b (id INTEGER PRIMARY KEY, label TEXT)");
    await e.sql("INSERT INTO b (id, label) VALUES (10, 'p')");
    await e.sql("INSERT INTO b (id, label) VALUES (20, 'q')");
    const result = await e.sql("SELECT a.val, b.label FROM a, b");
    expect(result!.rows.length).toBe(4);
  });

  it("implicit cross join with where", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)");
    await e.sql("INSERT INTO users (id, name) VALUES (1, 'Alice')");
    await e.sql("INSERT INTO users (id, name) VALUES (2, 'Bob')");
    await e.sql(
      "CREATE TABLE orders (" +
        "  oid INTEGER PRIMARY KEY, user_id INTEGER, product TEXT" +
        ")",
    );
    await e.sql("INSERT INTO orders (oid, user_id, product) VALUES (10, 1, 'Book')");
    await e.sql("INSERT INTO orders (oid, user_id, product) VALUES (11, 2, 'Pen')");
    const result = await e.sql(
      "SELECT users.name, orders.product " +
        "FROM users, orders " +
        "WHERE users.id = orders.user_id",
    );
    expect(result!.rows.length).toBe(2);
  });

  it("three table cross join", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE a (id INTEGER PRIMARY KEY, x TEXT)");
    await e.sql("INSERT INTO a (id, x) VALUES (1, 'a')");
    await e.sql("CREATE TABLE b (id INTEGER PRIMARY KEY, y TEXT)");
    await e.sql("INSERT INTO b (id, y) VALUES (1, 'b')");
    await e.sql("CREATE TABLE c (id INTEGER PRIMARY KEY, z TEXT)");
    await e.sql("INSERT INTO c (id, z) VALUES (1, 'c')");
    await e.sql("INSERT INTO c (id, z) VALUES (2, 'd')");
    const result = await e.sql("SELECT a.x, b.y, c.z FROM a, b, c");
    expect(result!.rows.length).toBe(2);
  });
});

// =============================================================================
// LATERAL subquery
// =============================================================================

describe("Lateral", () => {
  async function makeLateralEngine(): Promise<Engine> {
    const e = new Engine();
    await e.sql("CREATE TABLE depts (id INT PRIMARY KEY, dept_name TEXT)");
    await e.sql(
      "CREATE TABLE emps " +
        "(id INT PRIMARY KEY, emp_name TEXT, dept_id INT, salary INT)",
    );
    await e.sql("INSERT INTO depts VALUES (1, 'Engineering')");
    await e.sql("INSERT INTO depts VALUES (2, 'Sales')");
    await e.sql("INSERT INTO emps VALUES (1, 'Alice', 1, 90000)");
    await e.sql("INSERT INTO emps VALUES (2, 'Bob', 1, 80000)");
    await e.sql("INSERT INTO emps VALUES (3, 'Charlie', 2, 70000)");
    await e.sql("INSERT INTO emps VALUES (4, 'Diana', 2, 75000)");
    return e;
  }

  it("lateral subquery with aggregate", async () => {
    const e = await makeLateralEngine();
    const r = await e.sql(
      "SELECT d.dept_name, sub.top_salary " +
        "FROM depts d, " +
        "LATERAL (SELECT MAX(salary) AS top_salary " +
        "FROM emps WHERE emps.dept_id = d.id) sub " +
        "ORDER BY d.dept_name",
    );
    expect(r!.rows.length).toBe(2);
    expect(r!.rows[0]!["dept_name"]).toBe("Engineering");
    expect(r!.rows[0]!["top_salary"]).toBe(90000);
    expect(r!.rows[1]!["dept_name"]).toBe("Sales");
    expect(r!.rows[1]!["top_salary"]).toBe(75000);
  });

  it("lateral with limit", async () => {
    const e = await makeLateralEngine();
    const r = await e.sql(
      "SELECT d.dept_name, sub.top_emp, sub.top_sal " +
        "FROM depts d, " +
        "LATERAL (SELECT emp_name AS top_emp, salary AS top_sal " +
        "FROM emps WHERE emps.dept_id = d.id " +
        "ORDER BY salary DESC LIMIT 1) sub " +
        "ORDER BY d.dept_name",
    );
    expect(r!.rows.length).toBe(2);
    expect(r!.rows[0]!["top_emp"]).toBe("Alice");
    expect(r!.rows[0]!["top_sal"]).toBe(90000);
    expect(r!.rows[1]!["top_emp"]).toBe("Diana");
    expect(r!.rows[1]!["top_sal"]).toBe(75000);
  });

  it("lateral with count", async () => {
    const e = await makeLateralEngine();
    const r = await e.sql(
      "SELECT d.dept_name, sub.emp_count " +
        "FROM depts d, " +
        "LATERAL (SELECT COUNT(*) AS emp_count " +
        "FROM emps WHERE emps.dept_id = d.id) sub " +
        "ORDER BY d.dept_name",
    );
    expect(r!.rows[0]!["dept_name"]).toBe("Engineering");
    expect(r!.rows[0]!["emp_count"]).toBe(2);
    expect(r!.rows[1]!["dept_name"]).toBe("Sales");
    expect(r!.rows[1]!["emp_count"]).toBe(2);
  });
});
