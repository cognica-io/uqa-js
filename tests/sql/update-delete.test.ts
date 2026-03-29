import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// Helpers
// =============================================================================

async function makeProductsEngine(): Promise<Engine> {
  const e = new Engine();
  await e.sql(
    "CREATE TABLE products (" +
      "id INTEGER PRIMARY KEY, " +
      "name TEXT NOT NULL, " +
      "price REAL, " +
      "quantity INTEGER, " +
      "category TEXT" +
      ")",
  );
  await e.sql(
    "INSERT INTO products (id, name, price, quantity, category) VALUES " +
      "(1, 'Widget', 10.50, 100, 'tools'), " +
      "(2, 'Gadget', 25.00, 50, 'electronics'), " +
      "(3, 'Doohickey', 5.75, 200, NULL)",
  );
  return e;
}

async function makeUsersEngine(): Promise<Engine> {
  const e = new Engine();
  await e.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)");
  await e.sql("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)");
  await e.sql("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)");
  await e.sql("INSERT INTO users (id, name, age) VALUES (3, 'Carol', 35)");
  return e;
}

// =============================================================================
// UPDATE -- basic
// =============================================================================

describe("UpdateBasic", () => {
  it("update single column", async () => {
    const engine = await makeProductsEngine();
    const r = await engine.sql("UPDATE products SET price = 12.00 WHERE id = 1");
    expect(r!.rows[0]!["updated"]).toBe(1);
    const r2 = await engine.sql("SELECT price FROM products WHERE id = 1");
    expect(r2!.rows[0]!["price"]).toBe(12.0);
  });

  it("update multiple columns", async () => {
    const engine = await makeProductsEngine();
    const r = await engine.sql(
      "UPDATE products SET price = 15.00, quantity = 75 WHERE id = 2",
    );
    expect(r!.rows[0]!["updated"]).toBe(1);
    const r2 = await engine.sql("SELECT price, quantity FROM products WHERE id = 2");
    expect(r2!.rows[0]!["price"]).toBe(15.0);
    expect(r2!.rows[0]!["quantity"]).toBe(75);
  });

  it("update multiple rows", async () => {
    const engine = await makeProductsEngine();
    const r = await engine.sql(
      "UPDATE products SET category = 'sale' WHERE price < 20",
    );
    expect(r!.rows[0]!["updated"]).toBe(2);
    const r2 = await engine.sql(
      "SELECT id FROM products WHERE category = 'sale' ORDER BY id",
    );
    expect(r2!.rows.map((row) => row["id"])).toEqual([1, 3]);
  });

  it("update all rows", async () => {
    const engine = await makeProductsEngine();
    const r = await engine.sql("UPDATE products SET quantity = 0");
    expect(r!.rows[0]!["updated"]).toBe(3);
    const r2 = await engine.sql("SELECT quantity FROM products");
    for (const row of r2!.rows) {
      expect(row["quantity"]).toBe(0);
    }
  });

  it("update no match", async () => {
    const engine = await makeProductsEngine();
    const r = await engine.sql("UPDATE products SET price = 0 WHERE id = 999");
    expect(r!.rows[0]!["updated"]).toBe(0);
  });

  it("update returns count", async () => {
    const engine = await makeProductsEngine();
    const r = await engine.sql("UPDATE products SET price = 1.00");
    expect(r!.columns).toEqual(["updated"]);
    expect(r!.rows[0]!["updated"]).toBe(3);
  });
});

// =============================================================================
// UPDATE -- expressions
// =============================================================================

describe("UpdateExpressions", () => {
  it("update with arithmetic", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("UPDATE products SET price = price * 1.1 WHERE id = 1");
    const r = await engine.sql("SELECT price FROM products WHERE id = 1");
    expect(Math.abs((r!.rows[0]!["price"] as number) - 11.55)).toBeLessThan(0.01);
  });

  it("update with addition", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("UPDATE products SET quantity = quantity + 10 WHERE id = 2");
    const r = await engine.sql("SELECT quantity FROM products WHERE id = 2");
    expect(r!.rows[0]!["quantity"]).toBe(60);
  });

  it("update set to null", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("UPDATE products SET category = NULL WHERE id = 1");
    const r = await engine.sql("SELECT category FROM products WHERE id = 1");
    expect(r!.rows[0]!["category"] ?? null).toBeNull();
  });

  it("update with coalesce", async () => {
    const engine = await makeProductsEngine();
    await engine.sql(
      "UPDATE products SET category = COALESCE(category, 'uncategorized')",
    );
    const r = await engine.sql("SELECT id, category FROM products WHERE id = 3");
    expect(r!.rows[0]!["category"]).toBe("uncategorized");
  });

  it("update with case", async () => {
    const engine = await makeProductsEngine();
    await engine.sql(
      "UPDATE products SET category = " +
        "CASE WHEN price > 20 THEN 'premium' ELSE 'standard' END",
    );
    const r = await engine.sql("SELECT id, category FROM products ORDER BY id");
    expect(r!.rows[0]!["category"]).toBe("standard");
    expect(r!.rows[1]!["category"]).toBe("premium");
    expect(r!.rows[2]!["category"]).toBe("standard");
  });

  it("update with string function", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("UPDATE products SET name = UPPER(name) WHERE id = 1");
    const r = await engine.sql("SELECT name FROM products WHERE id = 1");
    expect(r!.rows[0]!["name"]).toBe("WIDGET");
  });

  it("update with concat", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("UPDATE products SET name = name || ' (v2)' WHERE id = 1");
    const r = await engine.sql("SELECT name FROM products WHERE id = 1");
    expect(r!.rows[0]!["name"]).toBe("Widget (v2)");
  });
});

// =============================================================================
// UPDATE -- constraints
// =============================================================================

describe("UpdateConstraints", () => {
  it("update not null violation", async () => {
    const engine = await makeProductsEngine();
    await expect(
      engine.sql("UPDATE products SET name = NULL WHERE id = 1"),
    ).rejects.toThrow(/NOT NULL/);
  });

  it("update unknown column", async () => {
    const engine = await makeProductsEngine();
    await expect(
      engine.sql("UPDATE products SET nonexistent = 1 WHERE id = 1"),
    ).rejects.toThrow(/[Uu]nknown column/);
  });

  it("update nonexistent table", async () => {
    const engine = await makeProductsEngine();
    await expect(engine.sql("UPDATE nonexistent SET x = 1")).rejects.toThrow(
      /does not exist/,
    );
  });
});

// =============================================================================
// UPDATE -- WHERE clause variations
// =============================================================================

describe("UpdateWithWhere", () => {
  it("update where is null", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("UPDATE products SET category = 'misc' WHERE category IS NULL");
    const r = await engine.sql("SELECT category FROM products WHERE id = 3");
    expect(r!.rows[0]!["category"]).toBe("misc");
  });

  it("update where is not null", async () => {
    const engine = await makeProductsEngine();
    await engine.sql(
      "UPDATE products SET price = price * 2 WHERE category IS NOT NULL",
    );
    const r = await engine.sql("SELECT id, price FROM products ORDER BY id");
    expect(r!.rows[0]!["price"]).toBe(21.0);
    expect(r!.rows[1]!["price"]).toBe(50.0);
    expect(r!.rows[2]!["price"]).toBe(5.75);
  });

  it("update where in", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("UPDATE products SET quantity = 0 WHERE id IN (1, 3)");
    const r = await engine.sql("SELECT id, quantity FROM products ORDER BY id");
    expect(r!.rows[0]!["quantity"]).toBe(0);
    expect(r!.rows[1]!["quantity"]).toBe(50);
    expect(r!.rows[2]!["quantity"]).toBe(0);
  });

  it("update where between", async () => {
    const engine = await makeProductsEngine();
    await engine.sql(
      "UPDATE products SET category = 'mid' WHERE price BETWEEN 5 AND 15",
    );
    const r = await engine.sql(
      "SELECT id FROM products WHERE category = 'mid' ORDER BY id",
    );
    expect(r!.rows.map((row) => row["id"])).toEqual([1, 3]);
  });

  it("update where and", async () => {
    const engine = await makeProductsEngine();
    await engine.sql(
      "UPDATE products SET price = 0 WHERE category = 'tools' AND quantity > 50",
    );
    const r = await engine.sql("SELECT price FROM products WHERE id = 1");
    expect(r!.rows[0]!["price"]).toBe(0.0);
  });
});

// =============================================================================
// UPDATE -- text index
// =============================================================================

describe("UpdateTextIndex", () => {
  // Python parity gap: not yet implemented in TS SQL compiler
  it("update reindexes text", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("CREATE INDEX idx_products_name ON products USING gin (name)");
    const r1 = await engine.sql(
      "SELECT id FROM products WHERE text_match(name, 'widget')",
    );
    expect(r1!.rows.length).toBe(1);

    await engine.sql("UPDATE products SET name = 'Sprocket' WHERE id = 1");

    const r2 = await engine.sql(
      "SELECT id FROM products WHERE text_match(name, 'widget')",
    );
    expect(r2!.rows.length).toBe(0);

    const r3 = await engine.sql(
      "SELECT id FROM products WHERE text_match(name, 'sprocket')",
    );
    expect(r3!.rows.length).toBe(1);
    expect(r3!.rows[0]!["id"]).toBe(1);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("update non text preserves index", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("CREATE INDEX idx_products_name ON products USING gin (name)");
    await engine.sql("UPDATE products SET price = 99.99 WHERE id = 1");
    const r = await engine.sql(
      "SELECT id FROM products WHERE text_match(name, 'widget')",
    );
    expect(r!.rows.length).toBe(1);
  });
});

// =============================================================================
// DELETE -- basic
// =============================================================================

describe("DeleteBasic", () => {
  it("delete single row", async () => {
    const engine = await makeProductsEngine();
    const r = await engine.sql("DELETE FROM products WHERE id = 2");
    expect(r!.rows[0]!["deleted"]).toBe(1);
    const r2 = await engine.sql("SELECT * FROM products WHERE id = 2");
    expect(r2!.rows).toEqual([]);
  });

  it("delete multiple rows", async () => {
    const engine = await makeProductsEngine();
    const r = await engine.sql("DELETE FROM products WHERE price < 20");
    expect(r!.rows[0]!["deleted"]).toBe(2);
    const r2 = await engine.sql("SELECT id FROM products");
    expect(r2!.rows.map((row) => row["id"])).toEqual([2]);
  });

  it("delete all rows", async () => {
    const engine = await makeProductsEngine();
    const r = await engine.sql("DELETE FROM products");
    expect(r!.rows[0]!["deleted"]).toBe(3);
    const r2 = await engine.sql("SELECT * FROM products");
    expect(r2!.rows.length).toBe(0);
  });

  it("delete no match", async () => {
    const engine = await makeProductsEngine();
    const r = await engine.sql("DELETE FROM products WHERE id = 999");
    expect(r!.rows[0]!["deleted"]).toBe(0);
  });

  it("delete returns count", async () => {
    const engine = await makeProductsEngine();
    const r = await engine.sql("DELETE FROM products WHERE id = 1");
    expect(r!.columns).toEqual(["deleted"]);
    expect(r!.rows[0]!["deleted"]).toBe(1);
  });

  it("delete nonexistent table", async () => {
    const engine = await makeProductsEngine();
    await expect(engine.sql("DELETE FROM nonexistent WHERE id = 1")).rejects.toThrow(
      /does not exist/,
    );
  });
});

// =============================================================================
// DELETE -- WHERE clause variations
// =============================================================================

describe("DeleteWithWhere", () => {
  it("delete where is null", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("DELETE FROM products WHERE category IS NULL");
    const r = await engine.sql("SELECT id FROM products ORDER BY id");
    expect(r!.rows.map((row) => row["id"])).toEqual([1, 2]);
  });

  it("delete where is not null", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("DELETE FROM products WHERE category IS NOT NULL");
    const r = await engine.sql("SELECT id FROM products");
    expect(r!.rows.map((row) => row["id"])).toEqual([3]);
  });

  it("delete where comparison", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("DELETE FROM products WHERE price > 10");
    const r = await engine.sql("SELECT id FROM products ORDER BY id");
    expect(r!.rows.map((row) => row["id"])).toEqual([3]);
  });

  it("delete where and", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("DELETE FROM products WHERE category = 'tools' AND price > 5");
    const r = await engine.sql("SELECT id FROM products ORDER BY id");
    expect(r!.rows.map((row) => row["id"])).toEqual([2, 3]);
  });
});

// =============================================================================
// DELETE -- text index
// =============================================================================

describe("DeleteTextIndex", () => {
  // Python parity gap: not yet implemented in TS SQL compiler
  it("delete removes from index", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("CREATE INDEX idx_products_name ON products USING gin (name)");
    const r1 = await engine.sql(
      "SELECT id FROM products WHERE text_match(name, 'widget')",
    );
    expect(r1!.rows.length).toBe(1);

    await engine.sql("DELETE FROM products WHERE id = 1");

    const r2 = await engine.sql(
      "SELECT id FROM products WHERE text_match(name, 'widget')",
    );
    expect(r2!.rows.length).toBe(0);
  });

  // Python parity gap: not yet implemented in TS SQL compiler
  it("delete preserves other index entries", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("CREATE INDEX idx_products_name ON products USING gin (name)");
    await engine.sql("DELETE FROM products WHERE id = 1");
    const r = await engine.sql(
      "SELECT id FROM products WHERE text_match(name, 'gadget')",
    );
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["id"]).toBe(2);
  });
});

// =============================================================================
// Combined operations
// =============================================================================

describe("CombinedOperations", () => {
  it("insert update select", async () => {
    const engine = await makeProductsEngine();
    await engine.sql(
      "INSERT INTO products (id, name, price, quantity, category) " +
        "VALUES (4, 'NewItem', 1.00, 1, 'new')",
    );
    await engine.sql("UPDATE products SET price = 99.99 WHERE id = 4");
    const r = await engine.sql("SELECT price FROM products WHERE id = 4");
    expect(r!.rows[0]!["price"]).toBe(99.99);
  });

  it("insert delete count", async () => {
    const engine = await makeProductsEngine();
    await engine.sql(
      "INSERT INTO products (id, name, price, quantity) " +
        "VALUES (4, 'Extra', 1.00, 1)",
    );
    const r1 = await engine.sql("SELECT * FROM products");
    expect(r1!.rows.length).toBe(4);
    await engine.sql("DELETE FROM products WHERE id = 4");
    const r2 = await engine.sql("SELECT * FROM products");
    expect(r2!.rows.length).toBe(3);
  });

  it("update then aggregate", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("UPDATE products SET price = 10 WHERE price < 10");
    const r = await engine.sql("SELECT MIN(price) AS min_p FROM products");
    expect(r!.rows[0]!["min_p"]).toBe(10.0);
  });

  it("delete then aggregate", async () => {
    const engine = await makeProductsEngine();
    await engine.sql("DELETE FROM products WHERE id = 3");
    const r = await engine.sql("SELECT COUNT(*) AS cnt FROM products");
    expect(r!.rows[0]!["cnt"]).toBe(2);
  });
});

// =============================================================================
// INSERT ... RETURNING
// =============================================================================

describe("InsertReturning", () => {
  it("returning star", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)");
    const result = await e.sql(
      "INSERT INTO t (id, name, age) VALUES (1, 'Alice', 30) RETURNING *",
    );
    expect(result!.rows.length).toBe(1);
    expect(result!.rows[0]!["id"]).toBe(1);
    expect(result!.rows[0]!["name"]).toBe("Alice");
    expect(result!.rows[0]!["age"]).toBe(30);
  });

  it("returning specific columns", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)");
    const result = await e.sql(
      "INSERT INTO t (id, name, age) VALUES (1, 'Alice', 30) RETURNING id, name",
    );
    expect(result!.columns).toEqual(["id", "name"]);
    expect(result!.rows[0]!["id"]).toBe(1);
    expect(result!.rows[0]!["name"]).toBe("Alice");
    expect(result!.rows[0]!["age"]).toBeUndefined();
  });

  it("returning with alias", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
    const result = await e.sql(
      "INSERT INTO t (id, name) VALUES (1, 'Alice') " +
        "RETURNING id AS user_id, name AS user_name",
    );
    expect(result!.rows[0]!["user_id"]).toBe(1);
    expect(result!.rows[0]!["user_name"]).toBe("Alice");
  });

  it("returning multi row", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
    const result = await e.sql(
      "INSERT INTO t (id, name) VALUES (1, 'Alice'), (2, 'Bob') " +
        "RETURNING id, name",
    );
    expect(result!.rows.length).toBe(2);
    const ids = result!.rows.map((r) => r["id"]);
    expect(ids).toContain(1);
    expect(ids).toContain(2);
  });

  it("returning serial", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id SERIAL PRIMARY KEY, name TEXT)");
    const result = await e.sql("INSERT INTO t (name) VALUES ('Alice') RETURNING id");
    expect(result!.rows[0]!["id"]).toBe(1);
  });
});

// =============================================================================
// UPDATE ... RETURNING
// =============================================================================

describe("UpdateReturning", () => {
  it("returning updated values", async () => {
    const e = await makeUsersEngine();
    const result = await e.sql(
      "UPDATE users SET age = 31 WHERE id = 1 RETURNING id, age",
    );
    expect(result!.rows.length).toBe(1);
    expect(result!.rows[0]!["id"]).toBe(1);
    expect(result!.rows[0]!["age"]).toBe(31);
  });

  it("returning star", async () => {
    const e = await makeUsersEngine();
    const result = await e.sql("UPDATE users SET age = 99 WHERE id = 2 RETURNING *");
    expect(result!.rows[0]!["id"]).toBe(2);
    expect(result!.rows[0]!["name"]).toBe("Bob");
    expect(result!.rows[0]!["age"]).toBe(99);
  });

  it("returning multiple rows", async () => {
    const e = await makeUsersEngine();
    const result = await e.sql(
      "UPDATE users SET age = age + 1 WHERE age < 35 RETURNING id, age",
    );
    expect(result!.rows.length).toBe(2);
    for (const row of result!.rows) {
      if (row["id"] === 1) {
        expect(row["age"]).toBe(31);
      } else if (row["id"] === 2) {
        expect(row["age"]).toBe(26);
      }
    }
  });

  it("returning no match", async () => {
    const e = await makeUsersEngine();
    const result = await e.sql("UPDATE users SET age = 0 WHERE id = 999 RETURNING id");
    expect(result!.rows.length).toBe(0);
  });
});

// =============================================================================
// DELETE ... RETURNING
// =============================================================================

describe("DeleteReturning", () => {
  it("returning deleted row", async () => {
    const e = await makeUsersEngine();
    const result = await e.sql("DELETE FROM users WHERE id = 1 RETURNING *");
    expect(result!.rows.length).toBe(1);
    expect(result!.rows[0]!["id"]).toBe(1);
    expect(result!.rows[0]!["name"]).toBe("Alice");
    const check = await e.sql("SELECT COUNT(*) AS cnt FROM users WHERE id = 1");
    expect(check!.rows[0]!["cnt"]).toBe(0);
  });

  it("returning specific columns", async () => {
    const e = await makeUsersEngine();
    const result = await e.sql("DELETE FROM users WHERE id = 2 RETURNING name");
    expect(result!.columns).toEqual(["name"]);
    expect(result!.rows[0]!["name"]).toBe("Bob");
  });

  it("returning multiple deletes", async () => {
    const e = await makeUsersEngine();
    const result = await e.sql("DELETE FROM users WHERE age >= 30 RETURNING id, name");
    expect(result!.rows.length).toBe(2);
    const names = new Set(result!.rows.map((r) => r["name"]));
    expect(names).toEqual(new Set(["Alice", "Carol"]));
  });

  it("returning no match", async () => {
    const e = await makeUsersEngine();
    const result = await e.sql("DELETE FROM users WHERE id = 999 RETURNING id");
    expect(result!.columns).toEqual(["id"]);
    expect(result!.rows.length).toBe(0);
  });
});

// =============================================================================
// INSERT ... ON CONFLICT DO NOTHING
// =============================================================================

describe("OnConflictDoNothing", () => {
  it("skip on conflict", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
    await e.sql("INSERT INTO t (id, name) VALUES (1, 'Alice')");
    await e.sql(
      "INSERT INTO t (id, name) VALUES (1, 'Bob') ON CONFLICT (id) DO NOTHING",
    );
    const result = await e.sql("SELECT name FROM t WHERE id = 1");
    expect(result!.rows[0]!["name"]).toBe("Alice");
  });

  it("insert non conflicting", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
    await e.sql("INSERT INTO t (id, name) VALUES (1, 'Alice')");
    await e.sql(
      "INSERT INTO t (id, name) VALUES (2, 'Bob') ON CONFLICT (id) DO NOTHING",
    );
    const result = await e.sql("SELECT COUNT(*) AS cnt FROM t");
    expect(result!.rows[0]!["cnt"]).toBe(2);
  });

  it("multi row partial conflict", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
    await e.sql("INSERT INTO t (id, name) VALUES (1, 'Alice')");
    await e.sql(
      "INSERT INTO t (id, name) VALUES (1, 'Dup'), (2, 'Bob') " +
        "ON CONFLICT (id) DO NOTHING",
    );
    const result = await e.sql("SELECT COUNT(*) AS cnt FROM t");
    expect(result!.rows[0]!["cnt"]).toBe(2);
    const r2 = await e.sql("SELECT name FROM t WHERE id = 1");
    expect(r2!.rows[0]!["name"]).toBe("Alice");
  });
});

// =============================================================================
// INSERT ... ON CONFLICT DO UPDATE (UPSERT)
// =============================================================================

describe("OnConflictDoUpdate", () => {
  it("upsert basic", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)");
    await e.sql("INSERT INTO t (id, name, score) VALUES (1, 'Alice', 100)");
    await e.sql(
      "INSERT INTO t (id, name, score) VALUES (1, 'Alice', 200) " +
        "ON CONFLICT (id) DO UPDATE SET score = excluded.score",
    );
    const result = await e.sql("SELECT score FROM t WHERE id = 1");
    expect(result!.rows[0]!["score"]).toBe(200);
  });

  it("upsert multiple set columns", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)");
    await e.sql("INSERT INTO t (id, name, score) VALUES (1, 'Alice', 100)");
    await e.sql(
      "INSERT INTO t (id, name, score) VALUES (1, 'Alicia', 200) " +
        "ON CONFLICT (id) DO UPDATE " +
        "SET name = excluded.name, score = excluded.score",
    );
    const result = await e.sql("SELECT name, score FROM t WHERE id = 1");
    expect(result!.rows[0]!["name"]).toBe("Alicia");
    expect(result!.rows[0]!["score"]).toBe(200);
  });

  it("upsert no conflict inserts", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
    await e.sql(
      "INSERT INTO t (id, name) VALUES (1, 'Alice') " +
        "ON CONFLICT (id) DO UPDATE SET name = excluded.name",
    );
    const result = await e.sql("SELECT name FROM t WHERE id = 1");
    expect(result!.rows[0]!["name"]).toBe("Alice");
  });

  it("upsert with returning", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)");
    await e.sql("INSERT INTO t (id, name, score) VALUES (1, 'Alice', 100)");
    const result = await e.sql(
      "INSERT INTO t (id, name, score) VALUES (1, 'Alice', 200) " +
        "ON CONFLICT (id) DO UPDATE SET score = excluded.score " +
        "RETURNING id, score",
    );
    expect(result!.rows.length).toBe(1);
    expect(result!.rows[0]!["id"]).toBe(1);
    expect(result!.rows[0]!["score"]).toBe(200);
  });

  it("upsert on unique column", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER, email TEXT UNIQUE, name TEXT)");
    await e.sql("INSERT INTO t (id, email, name) VALUES (1, 'a@b.com', 'Alice')");
    await e.sql(
      "INSERT INTO t (id, email, name) " +
        "VALUES (2, 'a@b.com', 'Bob') " +
        "ON CONFLICT (email) DO UPDATE SET name = excluded.name",
    );
    const result = await e.sql("SELECT name FROM t WHERE email = 'a@b.com'");
    expect(result!.rows[0]!["name"]).toBe("Bob");
    const cnt = await e.sql("SELECT COUNT(*) AS cnt FROM t");
    expect(cnt!.rows[0]!["cnt"]).toBe(1);
  });
});

// =============================================================================
// UPDATE ... FROM
// =============================================================================

describe("UpdateFrom", () => {
  async function makeUFEngine(): Promise<Engine> {
    const e = new Engine();
    await e.sql(
      "CREATE TABLE employees " +
        "(id INT PRIMARY KEY, name TEXT, dept_id INT, salary INT)",
    );
    await e.sql("CREATE TABLE departments (id INT PRIMARY KEY, name TEXT, budget INT)");
    await e.sql("INSERT INTO departments VALUES (1, 'Engineering', 100000)");
    await e.sql("INSERT INTO departments VALUES (2, 'Sales', 50000)");
    await e.sql("INSERT INTO employees VALUES (1, 'Alice', 1, 50000)");
    await e.sql("INSERT INTO employees VALUES (2, 'Bob', 2, 40000)");
    await e.sql("INSERT INTO employees VALUES (3, 'Charlie', 1, 60000)");
    return e;
  }

  it("basic update from", async () => {
    const e = await makeUFEngine();
    await e.sql(
      "UPDATE employees SET salary = departments.budget / 2 " +
        "FROM departments " +
        "WHERE employees.dept_id = departments.id " +
        "AND departments.name = 'Engineering'",
    );
    const r = await e.sql(
      "SELECT id, name, dept_id, salary FROM employees ORDER BY id",
    );
    expect(r!.rows[0]!["salary"]).toBe(50000);
    expect(r!.rows[1]!["salary"]).toBe(40000);
    expect(r!.rows[2]!["salary"]).toBe(50000);
  });

  it("update from returning", async () => {
    const e = await makeUFEngine();
    const r = await e.sql(
      "UPDATE employees SET salary = 99999 " +
        "FROM departments " +
        "WHERE employees.dept_id = departments.id " +
        "AND departments.name = 'Sales' " +
        "RETURNING employees.id, employees.salary",
    );
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["id"]).toBe(2);
    expect(r!.rows[0]!["salary"]).toBe(99999);
  });

  it("update from no match", async () => {
    const e = await makeUFEngine();
    const r = await e.sql(
      "UPDATE employees SET salary = 0 " +
        "FROM departments " +
        "WHERE employees.dept_id = departments.id " +
        "AND departments.name = 'Marketing'",
    );
    expect(r!.rows[0]!["updated"]).toBe(0);
  });

  it("update from multiple matches", async () => {
    const e = await makeUFEngine();
    await e.sql(
      "UPDATE employees SET salary = salary + 1000 " +
        "FROM departments " +
        "WHERE employees.dept_id = departments.id " +
        "AND departments.name = 'Engineering'",
    );
    const r = await e.sql("SELECT id, salary FROM employees ORDER BY id");
    expect(r!.rows[0]!["salary"]).toBe(51000);
    expect(r!.rows[1]!["salary"]).toBe(40000);
    expect(r!.rows[2]!["salary"]).toBe(61000);
  });
});

// =============================================================================
// DELETE ... USING
// =============================================================================

describe("DeleteUsing", () => {
  async function makeDUEngine(): Promise<Engine> {
    const e = new Engine();
    await e.sql("CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT, total INT)");
    await e.sql("CREATE TABLE blacklist (customer_id INT PRIMARY KEY)");
    await e.sql("INSERT INTO orders VALUES (1, 10, 100)");
    await e.sql("INSERT INTO orders VALUES (2, 20, 200)");
    await e.sql("INSERT INTO orders VALUES (3, 10, 300)");
    await e.sql("INSERT INTO blacklist VALUES (10)");
    return e;
  }

  it("basic delete using", async () => {
    const e = await makeDUEngine();
    await e.sql(
      "DELETE FROM orders USING blacklist " +
        "WHERE orders.customer_id = blacklist.customer_id",
    );
    const r = await e.sql("SELECT id, customer_id, total FROM orders ORDER BY id");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["id"]).toBe(2);
  });

  it("delete using returning", async () => {
    const e = await makeDUEngine();
    const r = await e.sql(
      "DELETE FROM orders USING blacklist " +
        "WHERE orders.customer_id = blacklist.customer_id " +
        "RETURNING orders.id",
    );
    expect(r!.rows.length).toBe(2);
    const ids = new Set(r!.rows.map((row) => row["id"]));
    expect(ids).toEqual(new Set([1, 3]));
  });

  it("delete using no match", async () => {
    const e = await makeDUEngine();
    await e.sql("DELETE FROM blacklist WHERE customer_id = 10");
    const r = await e.sql(
      "DELETE FROM orders USING blacklist " +
        "WHERE orders.customer_id = blacklist.customer_id",
    );
    expect(r!.rows[0]!["deleted"]).toBe(0);
  });

  it("delete using preserves unmatched", async () => {
    const e = await makeDUEngine();
    await e.sql(
      "DELETE FROM orders USING blacklist " +
        "WHERE orders.customer_id = blacklist.customer_id",
    );
    const r = await e.sql("SELECT id, customer_id, total FROM orders");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["customer_id"]).toBe(20);
    expect(r!.rows[0]!["total"]).toBe(200);
  });
});
