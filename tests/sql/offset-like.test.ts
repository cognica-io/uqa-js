import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// Helpers
// =============================================================================

async function makeItemsEngine(): Promise<Engine> {
  const e = new Engine();
  await e.sql(
    "CREATE TABLE items (" +
      "id INTEGER PRIMARY KEY, " +
      "name TEXT NOT NULL, " +
      "category TEXT, " +
      "price REAL" +
      ")",
  );
  await e.sql(
    "INSERT INTO items (id, name, category, price) VALUES " +
      "(1, 'Apple', 'fruit', 1.50), " +
      "(2, 'Banana', 'fruit', 0.75), " +
      "(3, 'Carrot', 'vegetable', 2.00), " +
      "(4, 'Date', 'fruit', 5.00), " +
      "(5, 'Eggplant', 'vegetable', 3.50), " +
      "(6, 'Fig', 'fruit', 4.00), " +
      "(7, 'Grape', 'fruit', 2.50), " +
      "(8, 'Habanero', 'pepper', 1.00)",
  );
  return e;
}

// =============================================================================
// OFFSET
// =============================================================================

describe("Offset", () => {
  it("limit offset", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT id FROM items ORDER BY id LIMIT 3 OFFSET 2");
    expect(r!.rows.map((row) => row["id"])).toEqual([3, 4, 5]);
  });

  it("offset zero", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT id FROM items ORDER BY id LIMIT 3 OFFSET 0");
    expect(r!.rows.map((row) => row["id"])).toEqual([1, 2, 3]);
  });

  it("offset past end", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT id FROM items ORDER BY id LIMIT 5 OFFSET 100");
    expect(r!.rows).toEqual([]);
  });

  it("offset last rows", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT id FROM items ORDER BY id LIMIT 10 OFFSET 6");
    expect(r!.rows.map((row) => row["id"])).toEqual([7, 8]);
  });

  it("offset with where", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql(
      "SELECT id FROM items WHERE category = 'fruit' ORDER BY id LIMIT 2 OFFSET 1",
    );
    expect(r!.rows.map((row) => row["id"])).toEqual([2, 4]);
  });

  it("offset with order desc", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT id FROM items ORDER BY id DESC LIMIT 3 OFFSET 2");
    expect(r!.rows.map((row) => row["id"])).toEqual([6, 5, 4]);
  });

  it("offset single row", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT id FROM items ORDER BY id LIMIT 1 OFFSET 4");
    expect(r!.rows.map((row) => row["id"])).toEqual([5]);
  });

  it("offset with aggregation", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql(
      "SELECT category, COUNT(*) AS cnt FROM items " +
        "GROUP BY category ORDER BY category LIMIT 2 OFFSET 1",
    );
    expect(r!.rows.length).toBe(2);
    expect(r!.rows[0]!["category"]).toBe("pepper");
    expect(r!.rows[1]!["category"]).toBe("vegetable");
  });
});

// =============================================================================
// LIKE
// =============================================================================

describe("Like", () => {
  it("like prefix", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT name FROM items WHERE name LIKE 'A%'");
    expect(r!.rows.map((row) => row["name"])).toEqual(["Apple"]);
  });

  it("like suffix", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT name FROM items WHERE name LIKE '%e'");
    const names = r!.rows.map((row) => row["name"] as string).sort();
    expect(names).toEqual(["Apple", "Date", "Grape"]);
  });

  it("like contains", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT name FROM items WHERE name LIKE '%an%'");
    const names = r!.rows.map((row) => row["name"] as string).sort();
    expect(names).toEqual(["Banana", "Eggplant", "Habanero"]);
  });

  it("like single char wildcard", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT name FROM items WHERE name LIKE '_ig'");
    expect(r!.rows.map((row) => row["name"])).toEqual(["Fig"]);
  });

  it("like exact match", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT name FROM items WHERE name LIKE 'Apple'");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["name"]).toBe("Apple");
  });

  it("like no match", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT name FROM items WHERE name LIKE 'Xyz%'");
    expect(r!.rows).toEqual([]);
  });

  it("like case sensitive", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT name FROM items WHERE name LIKE 'apple'");
    expect(r!.rows).toEqual([]);
  });

  it("not like", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql(
      "SELECT name FROM items WHERE name NOT LIKE '%a%' ORDER BY name",
    );
    const names = r!.rows.map((row) => row["name"]);
    expect(names).toEqual(["Apple", "Fig"]);
  });
});

// =============================================================================
// ILIKE
// =============================================================================

describe("ILike", () => {
  it("ilike case insensitive", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT name FROM items WHERE name ILIKE 'apple'");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["name"]).toBe("Apple");
  });

  it("ilike prefix", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT name FROM items WHERE name ILIKE 'a%'");
    expect(r!.rows.map((row) => row["name"])).toEqual(["Apple"]);
  });

  it("ilike suffix", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT name FROM items WHERE name ILIKE '%E'");
    const names = r!.rows.map((row) => row["name"] as string).sort();
    expect(names).toEqual(["Apple", "Date", "Grape"]);
  });

  it("ilike pattern mixed case", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT name FROM items WHERE name ILIKE '%BANANA%'");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["name"]).toBe("Banana");
  });

  it("not ilike", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql(
      "SELECT name FROM items WHERE name NOT ILIKE '%A%' ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Fig"]);
  });
});

// =============================================================================
// LIKE with expressions
// =============================================================================

describe("LikeWithExpressions", () => {
  it("like in case", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql(
      "SELECT name, " +
        "CASE WHEN name LIKE 'A%' THEN 'starts_A' " +
        "     WHEN name LIKE 'B%' THEN 'starts_B' " +
        "     ELSE 'other' END AS grp " +
        "FROM items ORDER BY id LIMIT 3",
    );
    expect(r!.rows[0]!["grp"]).toBe("starts_A");
    expect(r!.rows[1]!["grp"]).toBe("starts_B");
    expect(r!.rows[2]!["grp"]).toBe("other");
  });

  it("like with and", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql(
      "SELECT name FROM items " +
        "WHERE name LIKE '%a%' AND category = 'fruit' " +
        "ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Banana", "Date", "Grape"]);
  });

  it("like with or", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql(
      "SELECT name FROM items " +
        "WHERE name LIKE 'A%' OR name LIKE 'B%' " +
        "ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Apple", "Banana"]);
  });

  it("like with order and limit", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql(
      "SELECT name FROM items WHERE name LIKE '%a%' ORDER BY name LIMIT 2",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual(["Banana", "Carrot"]);
  });

  it("ilike in where expr", async () => {
    const e = await makeItemsEngine();
    const r = await e.sql("SELECT name FROM items WHERE name ILIKE '%egg%'");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["name"]).toBe("Eggplant");
  });
});

// =============================================================================
// LIKE with UPDATE/DELETE
// =============================================================================

describe("LikeUpdate", () => {
  it("update where like", async () => {
    const e = await makeItemsEngine();
    await e.sql("UPDATE items SET category = 'tropical' WHERE name LIKE '%an%'");
    const r = await e.sql(
      "SELECT name FROM items WHERE category = 'tropical' ORDER BY name",
    );
    expect(r!.rows.map((row) => row["name"])).toEqual([
      "Banana",
      "Eggplant",
      "Habanero",
    ]);
  });

  it("delete where like", async () => {
    const e = await makeItemsEngine();
    await e.sql("DELETE FROM items WHERE name LIKE 'E%'");
    const r = await e.sql("SELECT id FROM items ORDER BY id");
    const ids = r!.rows.map((row) => row["id"]);
    expect(ids).not.toContain(5);
    expect(ids.length).toBe(7);
  });
});
