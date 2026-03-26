import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";
import { Table, createColumnDef } from "../../src/sql/table.js";

// =============================================================================
// Helpers
// =============================================================================

function makeEngineWithOrders(): Engine {
  const engine = new Engine();
  const users = new Table("users", [
    createColumnDef("id", "INTEGER", { primaryKey: true }),
    createColumnDef("name", "TEXT"),
  ]);
  users.insert({ id: 1, name: "Alice" });
  users.insert({ id: 2, name: "Bob" });
  users.insert({ id: 3, name: "Carol" });
  engine.registerTable("users", users);

  const orders = new Table("orders", [
    createColumnDef("oid", "INTEGER", { primaryKey: true }),
    createColumnDef("user_id", "INTEGER"),
    createColumnDef("product", "TEXT"),
  ]);
  orders.insert({ oid: 10, user_id: 1, product: "Book" });
  orders.insert({ oid: 11, user_id: 1, product: "Pen" });
  orders.insert({ oid: 12, user_id: 2, product: "Notebook" });
  engine.registerTable("orders", orders);

  return engine;
}

// =============================================================================
// INNER JOIN
// =============================================================================

describe("InnerJoin", () => {
  it("inner join basic", async () => {
    const engine = makeEngineWithOrders();
    const result = await engine.sql(
      "SELECT users.name, orders.product " +
        "FROM users INNER JOIN orders ON users.id = orders.user_id",
    );
    expect(result!.rows.length).toBe(3);
    const products = new Set(
      result!.rows.map((r: Record<string, unknown>) => r["product"]),
    );
    expect(products).toEqual(new Set(["Book", "Pen", "Notebook"]));
  });

  it("inner join excludes unmatched", async () => {
    const engine = makeEngineWithOrders();
    const result = await engine.sql(
      "SELECT users.name " +
        "FROM users INNER JOIN orders ON users.id = orders.user_id",
    );
    const names = new Set(result!.rows.map((r: Record<string, unknown>) => r["name"]));
    expect(names.has("Carol")).toBe(false);
  });
});

// =============================================================================
// LEFT JOIN
// =============================================================================

describe("LeftJoin", () => {
  it("left join preserves left", async () => {
    const engine = makeEngineWithOrders();
    const result = await engine.sql(
      "SELECT users.name, orders.product " +
        "FROM users LEFT JOIN orders ON users.id = orders.user_id",
    );
    // Alice(2) + Bob(1) + Carol(unmatched) = 4 rows
    expect(result!.rows.length).toBe(4);
    const names = new Set(result!.rows.map((r: Record<string, unknown>) => r["name"]));
    expect(names.has("Carol")).toBe(true);
  });

  it("left join null for unmatched", async () => {
    const engine = makeEngineWithOrders();
    const result = await engine.sql(
      "SELECT users.name, orders.product " +
        "FROM users LEFT JOIN orders ON users.id = orders.user_id",
    );
    const carolRows = result!.rows.filter(
      (r: Record<string, unknown>) => r["name"] === "Carol",
    );
    expect(carolRows.length).toBe(1);
    expect(carolRows[0]!["product"]).toBeNull();
  });
});

// =============================================================================
// CROSS JOIN
// =============================================================================

describe("CrossJoin", () => {
  it("cross join cartesian", async () => {
    const engine = new Engine();
    const a = new Table("a", [
      createColumnDef("id", "INTEGER", { primaryKey: true }),
      createColumnDef("val", "TEXT"),
    ]);
    a.insert({ id: 1, val: "x" });
    a.insert({ id: 2, val: "y" });
    engine.registerTable("a", a);

    const b = new Table("b", [
      createColumnDef("id", "INTEGER", { primaryKey: true }),
      createColumnDef("label", "TEXT"),
    ]);
    b.insert({ id: 10, label: "p" });
    b.insert({ id: 20, label: "q" });
    b.insert({ id: 30, label: "r" });
    engine.registerTable("b", b);

    const result = await engine.sql("SELECT a.val, b.label FROM a CROSS JOIN b");
    expect(result!.rows.length).toBe(6); // 2 * 3
  });

  it("cross join empty side", async () => {
    const engine = new Engine();
    const a = new Table("a", [
      createColumnDef("id", "INTEGER", { primaryKey: true }),
      createColumnDef("val", "TEXT"),
    ]);
    a.insert({ id: 1, val: "x" });
    engine.registerTable("a", a);

    const b = new Table("b", [
      createColumnDef("id", "INTEGER", { primaryKey: true }),
      createColumnDef("label", "TEXT"),
    ]);
    engine.registerTable("b", b);

    const result = await engine.sql("SELECT * FROM a CROSS JOIN b");
    expect(result!.rows.length).toBe(0);
  });
});

// =============================================================================
// RIGHT JOIN
// =============================================================================

describe("RightJoin", () => {
  it("right join preserves right", async () => {
    const engine = makeEngineWithOrders();
    engine.getTable("orders").insert({ oid: 13, user_id: 99, product: "Ghost" });
    const result = await engine.sql(
      "SELECT users.name, orders.product " +
        "FROM users RIGHT JOIN orders ON users.id = orders.user_id",
    );
    const products = new Set(
      result!.rows.map((r: Record<string, unknown>) => r["product"]),
    );
    expect(products.has("Ghost")).toBe(true);
    expect(result!.rows.length).toBe(4);
  });

  it("right join null for unmatched left", async () => {
    const engine = makeEngineWithOrders();
    engine.getTable("orders").insert({ oid: 13, user_id: 99, product: "Ghost" });
    const result = await engine.sql(
      "SELECT users.name, orders.product " +
        "FROM users RIGHT JOIN orders ON users.id = orders.user_id",
    );
    const ghostRows = result!.rows.filter(
      (r: Record<string, unknown>) => r["product"] === "Ghost",
    );
    expect(ghostRows.length).toBe(1);
    expect(ghostRows[0]!["name"]).toBeNull();
  });
});

// =============================================================================
// FULL OUTER JOIN
// =============================================================================

describe("FullOuterJoin", () => {
  it("full join preserves both", async () => {
    const engine = makeEngineWithOrders();
    engine.getTable("orders").insert({ oid: 13, user_id: 99, product: "Ghost" });
    const result = await engine.sql(
      "SELECT users.name, orders.product " +
        "FROM users FULL OUTER JOIN orders ON users.id = orders.user_id",
    );
    // Alice(2) + Bob(1) + Carol(unmatched left) + Ghost(unmatched right) = 5
    expect(result!.rows.length).toBe(5);
    const names = new Set(result!.rows.map((r: Record<string, unknown>) => r["name"]));
    expect(names.has("Carol")).toBe(true);
    const products = new Set(
      result!.rows.map((r: Record<string, unknown>) => r["product"]),
    );
    expect(products.has("Ghost")).toBe(true);
  });

  it("full join no overlap", async () => {
    const engine = new Engine();
    const a = new Table("a", [
      createColumnDef("aid", "INTEGER", { primaryKey: true }),
      createColumnDef("val", "TEXT"),
    ]);
    a.insert({ aid: 1, val: "x" });
    engine.registerTable("a", a);

    const b = new Table("b", [
      createColumnDef("bid", "INTEGER", { primaryKey: true }),
      createColumnDef("label", "TEXT"),
    ]);
    b.insert({ bid: 2, label: "y" });
    engine.registerTable("b", b);

    const result = await engine.sql(
      "SELECT * FROM a FULL OUTER JOIN b ON a.aid = b.bid",
    );
    expect(result!.rows.length).toBe(2);
  });
});

// =============================================================================
// Multiple FROM tables
// =============================================================================

describe("MultipleFromTables", () => {
  it("implicit cross join", async () => {
    const engine = new Engine();
    const a = new Table("a", [
      createColumnDef("id", "INTEGER", { primaryKey: true }),
      createColumnDef("val", "TEXT"),
    ]);
    a.insert({ id: 1, val: "x" });
    a.insert({ id: 2, val: "y" });
    engine.registerTable("a", a);

    const b = new Table("b", [
      createColumnDef("id", "INTEGER", { primaryKey: true }),
      createColumnDef("label", "TEXT"),
    ]);
    b.insert({ id: 10, label: "p" });
    b.insert({ id: 20, label: "q" });
    engine.registerTable("b", b);

    const result = await engine.sql("SELECT a.val, b.label FROM a, b");
    expect(result!.rows.length).toBe(4); // 2 * 2
  });

  it("implicit cross join with where", async () => {
    const engine = new Engine();
    const users = new Table("users", [
      createColumnDef("id", "INTEGER", { primaryKey: true }),
      createColumnDef("name", "TEXT"),
    ]);
    users.insert({ id: 1, name: "Alice" });
    users.insert({ id: 2, name: "Bob" });
    engine.registerTable("users", users);

    const orders = new Table("orders", [
      createColumnDef("oid", "INTEGER", { primaryKey: true }),
      createColumnDef("user_id", "INTEGER"),
      createColumnDef("product", "TEXT"),
    ]);
    orders.insert({ oid: 10, user_id: 1, product: "Book" });
    orders.insert({ oid: 11, user_id: 2, product: "Pen" });
    engine.registerTable("orders", orders);

    const result = await engine.sql(
      "SELECT users.name, orders.product " +
        "FROM users, orders " +
        "WHERE users.id = orders.user_id",
    );
    expect(result!.rows.length).toBe(2);
  });

  it("three table cross join", async () => {
    const engine = new Engine();
    const a = new Table("a", [
      createColumnDef("id", "INTEGER", { primaryKey: true }),
      createColumnDef("x", "TEXT"),
    ]);
    a.insert({ id: 1, x: "a" });
    engine.registerTable("a", a);

    const b = new Table("b", [
      createColumnDef("id", "INTEGER", { primaryKey: true }),
      createColumnDef("y", "TEXT"),
    ]);
    b.insert({ id: 1, y: "b" });
    engine.registerTable("b", b);

    const c = new Table("c", [
      createColumnDef("id", "INTEGER", { primaryKey: true }),
      createColumnDef("z", "TEXT"),
    ]);
    c.insert({ id: 1, z: "c" });
    c.insert({ id: 2, z: "d" });
    engine.registerTable("c", c);

    const result = await engine.sql("SELECT a.x, b.y, c.z FROM a, b, c");
    expect(result!.rows.length).toBe(2); // 1 * 1 * 2
  });
});

// Note: LATERAL subquery tests are omitted because the TS SQL ExprEvaluator
// does not support aggregate functions (MAX, COUNT) inside LATERAL sub-queries.
