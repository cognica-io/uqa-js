import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// Helpers
// =============================================================================

async function makeUsersEngine(): Promise<Engine> {
  const e = new Engine();
  await e.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)");
  await e.sql("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)");
  await e.sql("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)");
  await e.sql("INSERT INTO users (id, name, age) VALUES (3, 'Carol', 35)");
  return e;
}

// =============================================================================
// ALTER TABLE -- ADD COLUMN
// =============================================================================

describe("AlterTableAddColumn", () => {
  it("add column", async () => {
    const e = await makeUsersEngine();
    await e.sql("ALTER TABLE users ADD COLUMN email TEXT");
    await e.sql("UPDATE users SET email = 'alice@test.com' WHERE id = 1");
    const result = await e.sql("SELECT email FROM users WHERE id = 1");
    expect(result!.rows[0]!["email"]).toBe("alice@test.com");
  });

  it("add column duplicate raises", async () => {
    const e = await makeUsersEngine();
    await expect(e.sql("ALTER TABLE users ADD COLUMN name TEXT")).rejects.toThrow(
      /already exists/,
    );
  });

  it("add column with default", async () => {
    const e = await makeUsersEngine();
    await e.sql("ALTER TABLE users ADD COLUMN active BOOLEAN DEFAULT TRUE");
    await e.sql("INSERT INTO users (id, name, age) VALUES (4, 'Dave', 28)");
    const result = await e.sql("SELECT active FROM users WHERE id = 4");
    expect(result!.rows[0]!["active"]).toBe(true);
  });
});

// =============================================================================
// ALTER TABLE -- DROP COLUMN
// =============================================================================

describe("AlterTableDropColumn", () => {
  it("drop column", async () => {
    const e = await makeUsersEngine();
    await e.sql("ALTER TABLE users DROP COLUMN age");
    const result = await e.sql("SELECT name FROM users WHERE id = 1");
    expect(result!.rows[0]!["name"]).toBe("Alice");
    const result2 = await e.sql("SELECT * FROM users WHERE id = 1");
    expect(result2!.rows[0]!["age"]).toBeUndefined();
  });

  it("drop column nonexistent raises", async () => {
    const e = await makeUsersEngine();
    await expect(e.sql("ALTER TABLE users DROP COLUMN nonexistent")).rejects.toThrow(
      /does not exist/,
    );
  });

  it("drop column if exists", async () => {
    const e = await makeUsersEngine();
    // Should not raise with IF EXISTS
    await e.sql("ALTER TABLE users DROP COLUMN IF EXISTS nonexistent");
  });
});

// =============================================================================
// ALTER TABLE -- RENAME COLUMN
// =============================================================================

describe("AlterTableRenameColumn", () => {
  it("rename column", async () => {
    const e = await makeUsersEngine();
    await e.sql("ALTER TABLE users RENAME COLUMN name TO full_name");
    const result = await e.sql("SELECT full_name FROM users WHERE id = 1");
    expect(result!.rows[0]!["full_name"]).toBe("Alice");
  });

  it("rename column nonexistent raises", async () => {
    const e = await makeUsersEngine();
    await expect(e.sql("ALTER TABLE users RENAME COLUMN xyz TO abc")).rejects.toThrow(
      /does not exist/,
    );
  });

  it("rename column duplicate raises", async () => {
    const e = await makeUsersEngine();
    await expect(e.sql("ALTER TABLE users RENAME COLUMN name TO age")).rejects.toThrow(
      /already exists/,
    );
  });
});

// =============================================================================
// ALTER TABLE -- RENAME TO
// =============================================================================

describe("AlterTableRenameTo", () => {
  it("rename table", async () => {
    const e = await makeUsersEngine();
    await e.sql("ALTER TABLE users RENAME TO people");
    const result = await e.sql("SELECT COUNT(*) AS cnt FROM people");
    expect(result!.rows[0]!["cnt"]).toBe(3);
    await expect(e.sql("SELECT * FROM users")).rejects.toThrow(/does not exist/);
  });

  it("rename table duplicate raises", async () => {
    const e = await makeUsersEngine();
    await e.sql("CREATE TABLE other (id INTEGER)");
    await expect(e.sql("ALTER TABLE users RENAME TO other")).rejects.toThrow(
      /already exists/,
    );
  });
});

// =============================================================================
// ALTER TABLE -- SET/DROP DEFAULT
// =============================================================================

describe("AlterTableDefault", () => {
  it("set default", async () => {
    const e = await makeUsersEngine();
    await e.sql("ALTER TABLE users ALTER COLUMN age SET DEFAULT 18");
    await e.sql("INSERT INTO users (id, name) VALUES (4, 'Dave')");
    const result = await e.sql("SELECT age FROM users WHERE id = 4");
    expect(result!.rows[0]!["age"]).toBe(18);
  });

  it("drop default", async () => {
    const e = await makeUsersEngine();
    await e.sql("ALTER TABLE users ALTER COLUMN age SET DEFAULT 18");
    await e.sql("ALTER TABLE users ALTER COLUMN age DROP DEFAULT");
    await e.sql("INSERT INTO users (id, name) VALUES (5, 'Eve')");
    const result = await e.sql("SELECT age FROM users WHERE id = 5");
    expect(result!.rows[0]!["age"] ?? null).toBeNull();
  });
});

// =============================================================================
// ALTER TABLE -- SET/DROP NOT NULL
// =============================================================================

describe("AlterTableNotNull", () => {
  it("set not null", async () => {
    const e = await makeUsersEngine();
    await e.sql("ALTER TABLE users ALTER COLUMN name SET NOT NULL");
    await expect(e.sql("INSERT INTO users (id, age) VALUES (4, 28)")).rejects.toThrow(
      /NOT NULL/,
    );
  });

  it("set not null with existing nulls raises", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER, val TEXT)");
    await e.sql("INSERT INTO t (id) VALUES (1)");
    await expect(e.sql("ALTER TABLE t ALTER COLUMN val SET NOT NULL")).rejects.toThrow(
      /contains NULL/,
    );
  });

  it("drop not null", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER, val TEXT NOT NULL)");
    await e.sql("ALTER TABLE t ALTER COLUMN val DROP NOT NULL");
    await e.sql("INSERT INTO t (id) VALUES (1)");
    const result = await e.sql("SELECT val FROM t WHERE id = 1");
    expect(result!.rows[0]!["val"] ?? null).toBeNull();
  });
});

// =============================================================================
// TRUNCATE TABLE
// =============================================================================

describe("TruncateTable", () => {
  it("truncate basic", async () => {
    const e = await makeUsersEngine();
    await e.sql("TRUNCATE TABLE users");
    const result = await e.sql("SELECT COUNT(*) AS cnt FROM users");
    expect(result!.rows[0]!["cnt"]).toBe(0);
  });

  it("truncate resets auto increment", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id SERIAL PRIMARY KEY, val TEXT)");
    await e.sql("INSERT INTO t (val) VALUES ('a')");
    await e.sql("INSERT INTO t (val) VALUES ('b')");
    await e.sql("TRUNCATE TABLE t");
    await e.sql("INSERT INTO t (val) VALUES ('c')");
    const result = await e.sql("SELECT id FROM t");
    expect(result!.rows[0]!["id"]).toBe(1);
  });

  it("truncate preserves schema", async () => {
    const e = await makeUsersEngine();
    await e.sql("TRUNCATE TABLE users");
    // Use a fresh id (unique indexes are not cleared by truncate)
    await e.sql("INSERT INTO users (id, name, age) VALUES (10, 'New', 20)");
    const result = await e.sql("SELECT name FROM users WHERE id = 10");
    expect(result!.rows[0]!["name"]).toBe("New");
  });

  it("truncate nonexistent raises", async () => {
    const e = new Engine();
    await expect(e.sql("TRUNCATE TABLE nonexistent")).rejects.toThrow(/does not exist/);
  });
});

// =============================================================================
// UNIQUE constraint
// =============================================================================

describe("UniqueConstraint", () => {
  it("unique basic", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER, email TEXT UNIQUE)");
    await e.sql("INSERT INTO t (id, email) VALUES (1, 'a@test.com')");
    await expect(
      e.sql("INSERT INTO t (id, email) VALUES (2, 'a@test.com')"),
    ).rejects.toThrow(/UNIQUE constraint/);
  });

  it("unique allows different values", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER, email TEXT UNIQUE)");
    await e.sql("INSERT INTO t (id, email) VALUES (1, 'a@test.com')");
    await e.sql("INSERT INTO t (id, email) VALUES (2, 'b@test.com')");
    const result = await e.sql("SELECT COUNT(*) AS cnt FROM t");
    expect(result!.rows[0]!["cnt"]).toBe(2);
  });

  it("unique allows null", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER, email TEXT UNIQUE)");
    await e.sql("INSERT INTO t (id, email) VALUES (1, 'a@test.com')");
    await e.sql("INSERT INTO t (id) VALUES (2)");
    await e.sql("INSERT INTO t (id) VALUES (3)");
    const result = await e.sql("SELECT COUNT(*) AS cnt FROM t");
    expect(result!.rows[0]!["cnt"]).toBe(3);
  });

  it("primary key enforces uniqueness", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)");
    await e.sql("INSERT INTO t (id, val) VALUES (1, 'a')");
    await expect(e.sql("INSERT INTO t (id, val) VALUES (1, 'b')")).rejects.toThrow(
      /UNIQUE constraint/,
    );
  });
});

// =============================================================================
// CHECK constraint
// =============================================================================

describe("CheckConstraint", () => {
  it("check basic", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER, age INTEGER CHECK (age > 0))");
    await e.sql("INSERT INTO t (id, age) VALUES (1, 25)");
    await expect(e.sql("INSERT INTO t (id, age) VALUES (2, -1)")).rejects.toThrow(
      /CHECK constraint/,
    );
  });

  it("check allows valid", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER, age INTEGER CHECK (age > 0))");
    await e.sql("INSERT INTO t (id, age) VALUES (1, 1)");
    await e.sql("INSERT INTO t (id, age) VALUES (2, 100)");
    const result = await e.sql("SELECT COUNT(*) AS cnt FROM t");
    expect(result!.rows[0]!["cnt"]).toBe(2);
  });

  it("check with comparison", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER, price REAL CHECK (price >= 0.0))");
    await e.sql("INSERT INTO t (id, price) VALUES (1, 9.99)");
    await expect(e.sql("INSERT INTO t (id, price) VALUES (2, -0.01)")).rejects.toThrow(
      /CHECK constraint/,
    );
  });
});

// =============================================================================
// ALTER TABLE -- ALTER COLUMN TYPE
// =============================================================================

describe("AlterColumnType", () => {
  it("change type", async () => {
    const e = await makeUsersEngine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER, name TEXT)");
    await e.sql("INSERT INTO t (id, val, name) VALUES (1, 10, 'alpha')");
    await e.sql("INSERT INTO t (id, val, name) VALUES (2, 20, 'bravo')");
    await e.sql("INSERT INTO t (id, val, name) VALUES (3, 30, 'charlie')");
    await e.sql("ALTER TABLE t ALTER COLUMN val TYPE TEXT");
    const result = await e.sql("SELECT val FROM t WHERE id = 1");
    expect(typeof result!.rows[0]!["val"]).toBe("string");
    expect(result!.rows[0]!["val"]).toBe("10");
  });
});

// =============================================================================
// FOREIGN KEY constraint
// =============================================================================

describe("ForeignKey", () => {
  async function makeFKEngine(): Promise<Engine> {
    const e = new Engine();
    await e.sql("CREATE TABLE parents (id INT PRIMARY KEY, name TEXT)");
    await e.sql("INSERT INTO parents VALUES (1, 'Parent1')");
    await e.sql("INSERT INTO parents VALUES (2, 'Parent2')");
    return e;
  }

  it("basic fk insert", async () => {
    const e = await makeFKEngine();
    await e.sql(
      "CREATE TABLE children " +
        "(id INT PRIMARY KEY, parent_id INT REFERENCES parents(id), " +
        "val TEXT)",
    );
    await e.sql("INSERT INTO children VALUES (1, 1, 'child1')");
    const r = await e.sql("SELECT id, parent_id, val FROM children");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["parent_id"]).toBe(1);
  });

  it("fk insert violation", async () => {
    const e = await makeFKEngine();
    await e.sql(
      "CREATE TABLE children " +
        "(id INT PRIMARY KEY, parent_id INT REFERENCES parents(id), " +
        "val TEXT)",
    );
    await expect(e.sql("INSERT INTO children VALUES (1, 999, 'bad')")).rejects.toThrow(
      /[Ff]oreign [Kk]ey/,
    );
  });

  it("fk null allowed", async () => {
    const e = await makeFKEngine();
    await e.sql(
      "CREATE TABLE children " +
        "(id INT PRIMARY KEY, parent_id INT REFERENCES parents(id), " +
        "val TEXT)",
    );
    await e.sql("INSERT INTO children VALUES (1, NULL, 'orphan')");
    const r = await e.sql("SELECT id, parent_id, val FROM children");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["parent_id"] ?? null).toBeNull();
  });

  it("fk delete violation", async () => {
    const e = await makeFKEngine();
    await e.sql(
      "CREATE TABLE children " +
        "(id INT PRIMARY KEY, parent_id INT REFERENCES parents(id), " +
        "val TEXT)",
    );
    await e.sql("INSERT INTO children VALUES (1, 1, 'child1')");
    await expect(e.sql("DELETE FROM parents WHERE id = 1")).rejects.toThrow(
      /FOREIGN KEY constraint violated/,
    );
  });

  it("fk delete unreferenced", async () => {
    const e = await makeFKEngine();
    await e.sql(
      "CREATE TABLE children " +
        "(id INT PRIMARY KEY, parent_id INT REFERENCES parents(id), " +
        "val TEXT)",
    );
    await e.sql("INSERT INTO children VALUES (1, 1, 'child1')");
    await e.sql("DELETE FROM parents WHERE id = 2");
    const r = await e.sql("SELECT id, name FROM parents");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["id"]).toBe(1);
  });

  it("fk update violation", async () => {
    const e = await makeFKEngine();
    await e.sql(
      "CREATE TABLE children " +
        "(id INT PRIMARY KEY, parent_id INT REFERENCES parents(id), " +
        "val TEXT)",
    );
    await e.sql("INSERT INTO children VALUES (1, 1, 'child1')");
    await expect(
      e.sql("UPDATE children SET parent_id = 999 WHERE id = 1"),
    ).rejects.toThrow(/FOREIGN KEY constraint violated/);
  });

  it("fk update valid", async () => {
    const e = await makeFKEngine();
    await e.sql(
      "CREATE TABLE children " +
        "(id INT PRIMARY KEY, parent_id INT REFERENCES parents(id), " +
        "val TEXT)",
    );
    await e.sql("INSERT INTO children VALUES (1, 1, 'child1')");
    await e.sql("UPDATE children SET parent_id = 2 WHERE id = 1");
    const r = await e.sql("SELECT parent_id FROM children WHERE id = 1");
    expect(r!.rows[0]!["parent_id"]).toBe(2);
  });

  it("fk update parent pk violation", async () => {
    const e = await makeFKEngine();
    await e.sql(
      "CREATE TABLE children " +
        "(id INT PRIMARY KEY, parent_id INT REFERENCES parents(id), " +
        "val TEXT)",
    );
    await e.sql("INSERT INTO children VALUES (1, 1, 'child1')");
    await expect(e.sql("UPDATE parents SET id = 99 WHERE id = 1")).rejects.toThrow(
      /FOREIGN KEY constraint violated/,
    );
  });
});
