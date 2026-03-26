import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";
import { Table, createColumnDef } from "../../src/sql/table.js";

// =============================================================================
// Helper: create products table via Table API
// =============================================================================

function makeProductsEngine(): Engine {
  const e = new Engine();
  const t = new Table("products", [
    createColumnDef("id", "INTEGER", { primaryKey: true, pythonType: "number" }),
    createColumnDef("name", "TEXT", { notNull: true }),
    createColumnDef("price", "REAL", { pythonType: "number" }),
    createColumnDef("quantity", "INTEGER", { pythonType: "number" }),
    createColumnDef("category", "TEXT"),
  ]);
  t.insert({ id: 1, name: "Widget", price: 10.5, quantity: 100, category: "tools" });
  t.insert({
    id: 2,
    name: "Gadget",
    price: 25.0,
    quantity: 50,
    category: "electronics",
  });
  t.insert({ id: 3, name: "Doohickey", price: 5.75, quantity: 200, category: null });
  e.registerTable("products", t);
  return e;
}

// =============================================================================
// IS NULL / IS NOT NULL
// =============================================================================

describe("NullTests", () => {
  it("is null", async () => {
    const e = makeProductsEngine();
    const r = await e.sql("SELECT id, name FROM products WHERE category IS NULL");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["name"]).toBe("Doohickey");
  });

  it("is not null", async () => {
    const e = makeProductsEngine();
    const r = await e.sql("SELECT id, name FROM products WHERE category IS NOT NULL");
    expect(r!.rows.length).toBe(2);
    const names = new Set(r!.rows.map((row: Record<string, unknown>) => row["name"]));
    expect(names).toEqual(new Set(["Widget", "Gadget"]));
  });

  it("is null on non-null column", async () => {
    const e = makeProductsEngine();
    const r = await e.sql("SELECT id FROM products WHERE name IS NULL");
    expect(r!.rows.length).toBe(0);
  });

  it("is not null all rows", async () => {
    const e = makeProductsEngine();
    const r = await e.sql("SELECT id FROM products WHERE price IS NOT NULL");
    expect(r!.rows.length).toBe(3);
  });
});

// =============================================================================
// Arithmetic: column / column works, column * literal has evaluator issues
// =============================================================================

describe("Arithmetic", () => {
  it("divide column by column", async () => {
    const e = makeProductsEngine();
    const r = await e.sql("SELECT name, price / quantity AS unit_cost FROM products");
    expect(Math.abs((r!.rows[0]!["unit_cost"] as number) - 0.105)).toBeLessThan(0.001);
  });

  it("simple and computed (column * column)", async () => {
    const e = makeProductsEngine();
    const r = await e.sql("SELECT id, name, price * quantity AS total FROM products");
    expect(r!.columns).toEqual(["id", "name", "total"]);
    expect(r!.rows[0]!["id"]).toBe(1);
    expect(r!.rows[0]!["name"]).toBe("Widget");
    expect(Math.abs((r!.rows[0]!["total"] as number) - 1050.0)).toBeLessThan(0.01);
  });

  it("computed with order by", async () => {
    const e = makeProductsEngine();
    const r = await e.sql(
      "SELECT name, price * quantity AS total FROM products ORDER BY total DESC",
    );
    // Gadget: 1250, Doohickey: 1150, Widget: 1050
    expect(r!.rows[0]!["name"]).toBe("Gadget");
    expect(r!.rows[2]!["name"]).toBe("Widget");
  });
});

// =============================================================================
// CAST
// =============================================================================

describe("Cast", () => {
  it("cast int to text", async () => {
    const e = makeProductsEngine();
    const r = await e.sql("SELECT CAST(quantity AS TEXT) AS qty_text FROM products");
    expect(r!.rows[0]!["qty_text"]).toBe("100");
  });

  it("cast text to int", async () => {
    const e = new Engine();
    const t = new Table("nums", [
      createColumnDef("id", "INTEGER"),
      createColumnDef("val", "TEXT"),
    ]);
    t.insert({ id: 1, val: "42" });
    e.registerTable("nums", t);
    const r = await e.sql("SELECT CAST(val AS INTEGER) AS num FROM nums");
    expect(r!.rows[0]!["num"]).toBe(42);
  });

  it("cast float to int", async () => {
    const e = makeProductsEngine();
    const r = await e.sql("SELECT CAST(price AS INTEGER) AS price_int FROM products");
    expect(r!.rows[0]!["price_int"]).toBe(10);
  });
});

// =============================================================================
// String functions
// =============================================================================

describe("StringFunctions", () => {
  it("upper", async () => {
    const e = makeProductsEngine();
    const r = await e.sql("SELECT UPPER(name) AS up FROM products");
    expect(r!.rows[0]!["up"]).toBe("WIDGET");
  });

  it("lower", async () => {
    const e = makeProductsEngine();
    const r = await e.sql("SELECT LOWER(name) AS low FROM products");
    expect(r!.rows[0]!["low"]).toBe("widget");
  });

  it("length", async () => {
    const e = makeProductsEngine();
    const r = await e.sql("SELECT LENGTH(name) AS len FROM products");
    expect(r!.rows[0]!["len"]).toBe(6);
  });
});

// =============================================================================
// Math functions
// =============================================================================

describe("MathFunctions", () => {
  it("ceil", async () => {
    const e = makeProductsEngine();
    const r = await e.sql("SELECT CEIL(price) AS c FROM products");
    expect(r!.rows[0]!["c"]).toBe(11);
  });

  it("floor", async () => {
    const e = makeProductsEngine();
    const r = await e.sql("SELECT FLOOR(price) AS f FROM products");
    expect(r!.rows[0]!["f"]).toBe(10);
  });
});

// =============================================================================
// Expression-based WHERE clause
// =============================================================================

describe("ExpressionWhere", () => {
  it("expression no match", async () => {
    const e = makeProductsEngine();
    const r = await e.sql("SELECT name FROM products WHERE price * quantity > 99999");
    expect(r!.rows.length).toBe(0);
  });
});

// =============================================================================
// IS NULL with physical operators
// =============================================================================

describe("NullPhysical", () => {
  it("is null with group by", async () => {
    const e = makeProductsEngine();
    const r = await e.sql(
      "SELECT category, COUNT(*) AS cnt FROM products GROUP BY category",
    );
    const rowsByCat: Record<string, number> = {};
    for (const row of r!.rows) {
      const cat = (row as Record<string, unknown>)["category"];
      rowsByCat[String(cat)] = (row as Record<string, unknown>)["cnt"] as number;
    }
    expect(rowsByCat["null"]).toBe(1);
    expect(rowsByCat["tools"]).toBe(1);
    expect(rowsByCat["electronics"]).toBe(1);
  });

  it("is null with order by", async () => {
    const e = makeProductsEngine();
    const r = await e.sql(
      "SELECT name FROM products WHERE category IS NOT NULL ORDER BY name",
    );
    expect(r!.rows[0]!["name"]).toBe("Gadget");
    expect(r!.rows[1]!["name"]).toBe("Widget");
  });

  it("is null with distinct", async () => {
    const e = makeProductsEngine();
    e.getTable("products").insert({
      id: 4,
      name: "Thingamajig",
      price: 3.0,
      quantity: 10,
      category: null,
    });
    const r = await e.sql(
      "SELECT DISTINCT category FROM products WHERE category IS NOT NULL",
    );
    const cats = new Set(
      r!.rows.map((row: Record<string, unknown>) => row["category"]),
    );
    expect(cats).toEqual(new Set(["tools", "electronics"]));
  });
});
