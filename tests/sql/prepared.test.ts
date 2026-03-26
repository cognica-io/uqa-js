import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// Helpers
// =============================================================================

async function makePreparedEngine(): Promise<Engine> {
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
      "(5, 'Eve', 'eng', 95000)",
  );
  return e;
}

// =============================================================================
// PREPARE
// =============================================================================

describe("Prepare", () => {
  // Python parity gap: not yet implemented in TS SQL compiler
  it("prepare select", async () => {
    const e = await makePreparedEngine();
    await e.sql(
      "PREPARE get_by_id (INTEGER) AS SELECT name FROM employees WHERE id = $1",
    );
    // Verify the prepared statement exists by executing it
    const r = await e.sql("EXECUTE get_by_id (1)");
    expect(r!.rows[0]!["name"]).toBe("Alice");
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("prepare duplicate raises", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE q AS SELECT name FROM employees");
    await expect(e.sql("PREPARE q AS SELECT name FROM employees")).rejects.toThrow(
      /already exists/,
    );
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("prepare insert", async () => {
    const e = await makePreparedEngine();
    await e.sql(
      "PREPARE ins AS " +
        "INSERT INTO employees (id, name, dept, salary) " +
        "VALUES ($1, $2, $3, $4)",
    );
    await e.sql("EXECUTE ins (6, 'Frank', 'mkt', 80000)");
    const r = await e.sql("SELECT name FROM employees WHERE id = 6");
    expect(r!.rows[0]!["name"]).toBe("Frank");
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("prepare update", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE upd AS UPDATE employees SET salary = $1 WHERE id = $2");
    await e.sql("EXECUTE upd (100000, 1)");
    const r = await e.sql("SELECT salary FROM employees WHERE id = 1");
    expect(r!.rows[0]!["salary"]).toBe(100000.0);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("prepare delete", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE del AS DELETE FROM employees WHERE id = $1");
    await e.sql("EXECUTE del (4)");
    const r = await e.sql("SELECT COUNT(*) AS cnt FROM employees");
    expect(r!.rows[0]!["cnt"]).toBe(4);
  });
});

// =============================================================================
// EXECUTE
// =============================================================================

describe("Execute", () => {
  // Python parity gap: not yet implemented in TS SQL compiler
  it("execute select single param", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE get_by_id AS SELECT name FROM employees WHERE id = $1");
    const r = await e.sql("EXECUTE get_by_id (1)");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["name"]).toBe("Alice");
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("execute select different params", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE get_by_id AS SELECT name FROM employees WHERE id = $1");
    const r1 = await e.sql("EXECUTE get_by_id (1)");
    const r2 = await e.sql("EXECUTE get_by_id (3)");
    expect(r1!.rows[0]!["name"]).toBe("Alice");
    expect(r2!.rows[0]!["name"]).toBe("Carol");
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("execute select multiple params", async () => {
    const e = await makePreparedEngine();
    await e.sql(
      "PREPARE get_by_dept_sal AS " +
        "SELECT name FROM employees " +
        "WHERE dept = $1 AND salary > $2 " +
        "ORDER BY name",
    );
    const r = await e.sql("EXECUTE get_by_dept_sal ('eng', 87000)");
    expect(r!.rows.map((row) => row["name"])).toEqual(["Alice", "Eve"]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("execute insert", async () => {
    const e = await makePreparedEngine();
    await e.sql(
      "PREPARE ins AS " +
        "INSERT INTO employees (id, name, dept, salary) " +
        "VALUES ($1, $2, $3, $4)",
    );
    await e.sql("EXECUTE ins (6, 'Frank', 'mkt', 80000)");
    const r = await e.sql("SELECT name FROM employees WHERE id = 6");
    expect(r!.rows[0]!["name"]).toBe("Frank");
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("execute update", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE upd AS UPDATE employees SET salary = $1 WHERE id = $2");
    await e.sql("EXECUTE upd (100000, 1)");
    const r = await e.sql("SELECT salary FROM employees WHERE id = 1");
    expect(r!.rows[0]!["salary"]).toBe(100000.0);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("execute delete", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE del AS DELETE FROM employees WHERE id = $1");
    await e.sql("EXECUTE del (4)");
    const r = await e.sql("SELECT COUNT(*) AS cnt FROM employees");
    expect(r!.rows[0]!["cnt"]).toBe(4);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("execute nonexistent raises", async () => {
    const e = await makePreparedEngine();
    await expect(e.sql("EXECUTE nonexistent (1)")).rejects.toThrow(/does not exist/);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("execute missing param raises", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE q AS SELECT name FROM employees WHERE id = $1 AND dept = $2");
    await expect(e.sql("EXECUTE q (1)")).rejects.toThrow(/No value supplied/);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("execute reusable", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE get_name AS SELECT name FROM employees WHERE id = $1");
    const names: unknown[] = [];
    for (let i = 1; i <= 5; i++) {
      const r = await e.sql(`EXECUTE get_name (${i})`);
      names.push(r!.rows[0]!["name"]);
    }
    expect(names).toEqual(["Alice", "Bob", "Carol", "Dave", "Eve"]);
  });
});

// =============================================================================
// DEALLOCATE
// =============================================================================

describe("Deallocate", () => {
  // Python parity gap: not yet implemented in TS SQL compiler
  it("deallocate", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE q AS SELECT name FROM employees");
    await e.sql("DEALLOCATE q");
    await expect(e.sql("EXECUTE q")).rejects.toThrow(/does not exist/);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("deallocate nonexistent raises", async () => {
    const e = await makePreparedEngine();
    await expect(e.sql("DEALLOCATE nonexistent")).rejects.toThrow(/does not exist/);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("deallocate all", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE q1 AS SELECT name FROM employees");
    await e.sql("PREPARE q2 AS SELECT dept FROM employees");
    await e.sql("DEALLOCATE ALL");
    await expect(e.sql("EXECUTE q1")).rejects.toThrow(/does not exist/);
    await expect(e.sql("EXECUTE q2")).rejects.toThrow(/does not exist/);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("execute after deallocate raises", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE q AS SELECT name FROM employees WHERE id = $1");
    await e.sql("DEALLOCATE q");
    await expect(e.sql("EXECUTE q (1)")).rejects.toThrow(/does not exist/);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("reprepare after deallocate", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE q AS SELECT name FROM employees WHERE dept = $1");
    await e.sql("DEALLOCATE q");
    await e.sql("PREPARE q AS SELECT salary FROM employees WHERE id = $1");
    const r = await e.sql("EXECUTE q (1)");
    expect(r!.rows[0]!["salary"]).toBe(90000.0);
  });
});

// =============================================================================
// Edge cases and integration
// =============================================================================

describe("PreparedIntegration", () => {
  // Python parity gap: not yet implemented in TS SQL compiler
  it("prepare with typed params", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE q (INTEGER) AS SELECT name FROM employees WHERE id = $1");
    const r = await e.sql("EXECUTE q (2)");
    expect(r!.rows[0]!["name"]).toBe("Bob");
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("prepare select with order and limit", async () => {
    const e = await makePreparedEngine();
    await e.sql(
      "PREPARE top_earners AS " +
        "SELECT name, salary FROM employees " +
        "WHERE dept = $1 ORDER BY salary DESC LIMIT 2",
    );
    const r = await e.sql("EXECUTE top_earners ('eng')");
    expect(r!.rows.length).toBe(2);
    expect(r!.rows[0]!["name"]).toBe("Eve");
    expect(r!.rows[1]!["name"]).toBe("Alice");
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("prepare select no params", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE all_names AS SELECT name FROM employees ORDER BY name");
    const r = await e.sql("EXECUTE all_names");
    expect(r!.rows.map((row) => row["name"])).toEqual([
      "Alice",
      "Bob",
      "Carol",
      "Dave",
      "Eve",
    ]);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("prepare with null param", async () => {
    const e = await makePreparedEngine();
    await e.sql(
      "INSERT INTO employees (id, name, dept, salary) VALUES " +
        "(6, 'Frank', NULL, 80000)",
    );
    await e.sql(
      "PREPARE get_null_dept AS SELECT name FROM employees WHERE dept IS NULL",
    );
    const r = await e.sql("EXECUTE get_null_dept");
    expect(r!.rows[0]!["name"]).toBe("Frank");
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("multiple prepared coexist", async () => {
    const e = await makePreparedEngine();
    await e.sql("PREPARE by_id AS SELECT name FROM employees WHERE id = $1");
    await e.sql(
      "PREPARE by_dept AS " +
        "SELECT name FROM employees WHERE dept = $1 ORDER BY name",
    );
    const r1 = await e.sql("EXECUTE by_id (1)");
    const r2 = await e.sql("EXECUTE by_dept ('eng')");
    expect(r1!.rows[0]!["name"]).toBe("Alice");
    expect(r2!.rows.map((row) => row["name"])).toEqual(["Alice", "Carol", "Eve"]);
  });
});
