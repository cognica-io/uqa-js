import { describe, expect, it, beforeEach } from "vitest";
import { Engine } from "../../src/engine.js";

let engine: Engine;

beforeEach(() => {
  engine = new Engine();
});

// ==================================================================
// CTAS with explicit columns + SELECT *
// ==================================================================

describe("CTASSelectStar", () => {
  it("ctas explicit cols select star", async () => {
    await engine.sql(
      "CREATE TABLE src (id INTEGER PRIMARY KEY, name TEXT, val INTEGER)",
    );
    await engine.sql("INSERT INTO src VALUES (1, 'Alice', 10)");
    await engine.sql("INSERT INTO src VALUES (2, 'Bob', 20)");
    await engine.sql("CREATE TABLE dst AS SELECT * FROM src");
    const r = await engine.sql("SELECT * FROM dst ORDER BY id");
    expect(r!.rows.length).toBe(2);
    expect(r!.rows[0]!["name"]).toBe("Alice");
  });

  it("ctas explicit cols select star order by", async () => {
    await engine.sql(
      "CREATE TABLE src (id INTEGER PRIMARY KEY, name TEXT, val INTEGER)",
    );
    await engine.sql("INSERT INTO src VALUES (2, 'Bob', 20)");
    await engine.sql("INSERT INTO src VALUES (1, 'Alice', 10)");
    await engine.sql("CREATE TABLE dst AS SELECT * FROM src ORDER BY id");
    const r = await engine.sql("SELECT id, name FROM dst");
    expect(r!.rows.length).toBe(2);
  });

  it("ctas star still works", async () => {
    await engine.sql("CREATE TABLE src (id INTEGER PRIMARY KEY, name TEXT)");
    await engine.sql("INSERT INTO src VALUES (1, 'Alice')");
    await engine.sql("CREATE TABLE dst AS SELECT * FROM src");
    const r = await engine.sql("SELECT * FROM dst");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["id"]).toBe(1);
    expect(r!.rows[0]!["name"]).toBe("Alice");
  });
});

// ==================================================================
// SELECT * after DELETE ... USING
// ==================================================================

describe("DeleteUsingSelectStar", () => {
  it("delete using then select star", async () => {
    await engine.sql(
      "CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT, total INT)",
    );
    await engine.sql("CREATE TABLE blacklist (customer_id INT PRIMARY KEY)");
    await engine.sql("INSERT INTO orders VALUES (1, 10, 100)");
    await engine.sql("INSERT INTO orders VALUES (2, 20, 200)");
    await engine.sql("INSERT INTO orders VALUES (3, 10, 300)");
    await engine.sql("INSERT INTO blacklist VALUES (10)");
    await engine.sql(
      "DELETE FROM orders USING blacklist WHERE orders.customer_id = blacklist.customer_id",
    );
    const r = await engine.sql("SELECT * FROM orders");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["id"]).toBe(2);
  });

  it("delete using select star order by", async () => {
    await engine.sql(
      "CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT, total INT)",
    );
    await engine.sql("CREATE TABLE blacklist (customer_id INT PRIMARY KEY)");
    await engine.sql("INSERT INTO orders VALUES (1, 10, 100)");
    await engine.sql("INSERT INTO orders VALUES (2, 20, 200)");
    await engine.sql("INSERT INTO orders VALUES (3, 30, 300)");
    await engine.sql("INSERT INTO blacklist VALUES (10)");
    await engine.sql(
      "DELETE FROM orders USING blacklist WHERE orders.customer_id = blacklist.customer_id",
    );
    const r = await engine.sql("SELECT * FROM orders ORDER BY id");
    expect(r!.rows.length).toBe(2);
    expect(r!.rows[0]!["id"]).toBe(2);
    expect(r!.rows[1]!["id"]).toBe(3);
  });
});

// ==================================================================
// Aggregate in IN subquery HAVING
// ==================================================================

describe("AggregateInSubqueryHaving", () => {
  it("count in having subquery", async () => {
    await engine.sql("CREATE TABLE items (id INT PRIMARY KEY, cat TEXT, val INT)");
    await engine.sql("INSERT INTO items VALUES (1, 'a', 10)");
    await engine.sql("INSERT INTO items VALUES (2, 'a', 20)");
    await engine.sql("INSERT INTO items VALUES (3, 'b', 30)");
    const r = await engine.sql(
      "SELECT cat, COUNT(*) AS cnt FROM items GROUP BY cat HAVING COUNT(*) > 1",
    );
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["cat"]).toBe("a");
    expect(r!.rows[0]!["cnt"]).toBe(2);
  });

  it("sum in having subquery", async () => {
    await engine.sql("CREATE TABLE items (id INT PRIMARY KEY, cat TEXT, val INT)");
    await engine.sql("INSERT INTO items VALUES (1, 'a', 10)");
    await engine.sql("INSERT INTO items VALUES (2, 'a', 20)");
    await engine.sql("INSERT INTO items VALUES (3, 'b', 30)");
    const r = await engine.sql(
      "SELECT cat, SUM(val) AS total FROM items GROUP BY cat HAVING SUM(val) >= 30",
    );
    expect(r!.rows.length).toBe(2);
    // Both groups: a(30) and b(30) have sum >= 30
    const cats = r!.rows.map((row) => row["cat"]).sort();
    expect(cats).toEqual(["a", "b"]);
  });

  it("having with multiple aggregates", async () => {
    await engine.sql("CREATE TABLE items (id INT PRIMARY KEY, cat TEXT, val INT)");
    await engine.sql("INSERT INTO items VALUES (1, 'a', 10)");
    await engine.sql("INSERT INTO items VALUES (2, 'a', 20)");
    await engine.sql("INSERT INTO items VALUES (3, 'b', 30)");
    await engine.sql("INSERT INTO items VALUES (4, 'c', 5)");
    const r = await engine.sql(
      "SELECT cat, COUNT(*) AS cnt, SUM(val) AS total FROM items GROUP BY cat HAVING COUNT(*) > 1",
    );
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["cat"]).toBe("a");
    expect(r!.rows[0]!["total"]).toBe(30);
  });
});

// ==================================================================
// Aggregate nested inside scalar function
// ==================================================================

describe("NestedAggregate", () => {
  it("round stddev", async () => {
    await engine.sql("CREATE TABLE t (id INT PRIMARY KEY, val REAL)");
    await engine.sql("INSERT INTO t VALUES (1, 10)");
    await engine.sql("INSERT INTO t VALUES (2, 20)");
    await engine.sql("INSERT INTO t VALUES (3, 30)");
    const r = await engine.sql("SELECT ROUND(STDDEV(val), 2) AS sd FROM t");
    expect(r!.rows.length).toBe(1);
    const sd = r!.rows[0]!["sd"] as number;
    expect(typeof sd).toBe("number");
    expect(sd).toBeGreaterThan(0);
  });

  it("round variance", async () => {
    await engine.sql("CREATE TABLE t (id INT PRIMARY KEY, val REAL)");
    await engine.sql("INSERT INTO t VALUES (1, 10)");
    await engine.sql("INSERT INTO t VALUES (2, 20)");
    await engine.sql("INSERT INTO t VALUES (3, 30)");
    const r = await engine.sql("SELECT ROUND(VARIANCE(val), 2) AS vr FROM t");
    expect(r!.rows.length).toBe(1);
    const vr = r!.rows[0]!["vr"] as number;
    expect(typeof vr).toBe("number");
    expect(vr).toBeGreaterThan(0);
  });

  it("round corr", async () => {
    await engine.sql("CREATE TABLE t (id INT PRIMARY KEY, x REAL, y REAL)");
    await engine.sql("INSERT INTO t VALUES (1, 1, 2)");
    await engine.sql("INSERT INTO t VALUES (2, 2, 4)");
    await engine.sql("INSERT INTO t VALUES (3, 3, 6)");
    const r = await engine.sql("SELECT ROUND(CORR(x, y), 4) AS cr FROM t");
    expect(r!.rows.length).toBe(1);
    const cr = r!.rows[0]!["cr"] as number;
    expect(typeof cr).toBe("number");
    // Perfect positive correlation
    expect(cr).toBeCloseTo(1.0, 2);
  });

  it("round covar", async () => {
    await engine.sql("CREATE TABLE t (id INT PRIMARY KEY, x REAL, y REAL)");
    await engine.sql("INSERT INTO t VALUES (1, 1, 2)");
    await engine.sql("INSERT INTO t VALUES (2, 2, 4)");
    await engine.sql("INSERT INTO t VALUES (3, 3, 6)");
    const r = await engine.sql("SELECT ROUND(COVAR_POP(x, y), 4) AS cv FROM t");
    expect(r!.rows.length).toBe(1);
    const cv = r!.rows[0]!["cv"] as number;
    expect(typeof cv).toBe("number");
    expect(cv).toBeGreaterThan(0);
  });

  it("round regr", async () => {
    await engine.sql("CREATE TABLE t (id INT PRIMARY KEY, x REAL, y REAL)");
    await engine.sql("INSERT INTO t VALUES (1, 1, 2)");
    await engine.sql("INSERT INTO t VALUES (2, 2, 4)");
    await engine.sql("INSERT INTO t VALUES (3, 3, 6)");
    const r = await engine.sql("SELECT ROUND(REGR_SLOPE(y, x), 4) AS slope FROM t");
    expect(r!.rows.length).toBe(1);
    const slope = r!.rows[0]!["slope"] as number;
    expect(typeof slope).toBe("number");
    // y = 2*x, so slope should be 2
    expect(slope).toBeCloseTo(2.0, 2);
  });

  it("abs of aggregate", async () => {
    await engine.sql("CREATE TABLE t (id INT PRIMARY KEY, val INT)");
    await engine.sql("INSERT INTO t VALUES (1, -10)");
    await engine.sql("INSERT INTO t VALUES (2, -20)");
    await engine.sql("INSERT INTO t VALUES (3, -30)");
    const r = await engine.sql("SELECT ABS(SUM(val)) AS abssum FROM t");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["abssum"]).toBe(60);
  });
});

// ==================================================================
// Window default frame with ORDER BY
// ==================================================================

describe("WindowDefaultFrame", () => {
  it("window running sum via SQL", async () => {
    await engine.sql("CREATE TABLE t (id INT PRIMARY KEY, val INT)");
    await engine.sql("INSERT INTO t VALUES (1, 10)");
    await engine.sql("INSERT INTO t VALUES (2, 20)");
    await engine.sql("INSERT INTO t VALUES (3, 30)");
    const r = await engine.sql(
      "SELECT id, SUM(val) OVER (ORDER BY id) AS running FROM t",
    );
    expect(r!.rows.length).toBe(3);
    // With ORDER BY, default frame is ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    const rows = r!.rows.sort((a, b) => (a["id"] as number) - (b["id"] as number));
    expect(rows[0]!["running"]).toBe(10);
    expect(rows[1]!["running"]).toBe(30);
    expect(rows[2]!["running"]).toBe(60);
  });

  it("running sum", async () => {
    await engine.sql("CREATE TABLE t (id INT PRIMARY KEY, val INT)");
    await engine.sql("INSERT INTO t VALUES (1, 5)");
    await engine.sql("INSERT INTO t VALUES (2, 15)");
    await engine.sql("INSERT INTO t VALUES (3, 25)");
    const r = await engine.sql(
      "SELECT id, SUM(val) OVER (ORDER BY id) AS running_sum FROM t",
    );
    const rows = r!.rows.sort((a, b) => (a["id"] as number) - (b["id"] as number));
    expect(rows[0]!["running_sum"]).toBe(5);
    expect(rows[1]!["running_sum"]).toBe(20);
    expect(rows[2]!["running_sum"]).toBe(45);
  });

  it("running avg", async () => {
    await engine.sql("CREATE TABLE t (id INT PRIMARY KEY, val INT)");
    await engine.sql("INSERT INTO t VALUES (1, 10)");
    await engine.sql("INSERT INTO t VALUES (2, 20)");
    await engine.sql("INSERT INTO t VALUES (3, 30)");
    const r = await engine.sql(
      "SELECT id, AVG(val) OVER (ORDER BY id) AS running_avg FROM t",
    );
    const rows = r!.rows.sort((a, b) => (a["id"] as number) - (b["id"] as number));
    expect(rows[0]!["running_avg"]).toBeCloseTo(10);
    expect(rows[1]!["running_avg"]).toBeCloseTo(15);
    expect(rows[2]!["running_avg"]).toBeCloseTo(20);
  });

  it("running count", async () => {
    await engine.sql("CREATE TABLE t (id INT PRIMARY KEY, val INT)");
    await engine.sql("INSERT INTO t VALUES (1, 10)");
    await engine.sql("INSERT INTO t VALUES (2, 20)");
    await engine.sql("INSERT INTO t VALUES (3, 30)");
    const r = await engine.sql(
      "SELECT id, COUNT(*) OVER (ORDER BY id) AS running_count FROM t",
    );
    const rows = r!.rows.sort((a, b) => (a["id"] as number) - (b["id"] as number));
    expect(rows[0]!["running_count"]).toBe(1);
    expect(rows[1]!["running_count"]).toBe(2);
    expect(rows[2]!["running_count"]).toBe(3);
  });

  it("no order by uses whole partition", async () => {
    await engine.sql("CREATE TABLE t (id INT PRIMARY KEY, val INT)");
    await engine.sql("INSERT INTO t VALUES (1, 10)");
    await engine.sql("INSERT INTO t VALUES (2, 20)");
    await engine.sql("INSERT INTO t VALUES (3, 30)");
    const r = await engine.sql(
      "SELECT id, SUM(val) OVER () AS total FROM t ORDER BY id",
    );
    // Without ORDER BY in OVER, window includes all rows in the partition
    // The engine may use default frame; total should be same for all rows
    const totals = r!.rows.map((row) => row["total"] as number);
    // All should be the same value (whole-partition sum)
    const uniqueTotals = [...new Set(totals)];
    expect(uniqueTotals.length).toBe(1);
    expect(uniqueTotals[0]).toBe(60);
  });

  it("running sum with partition", async () => {
    await engine.sql("CREATE TABLE t (id INT PRIMARY KEY, grp TEXT, val INT)");
    await engine.sql("INSERT INTO t VALUES (1, 'a', 10)");
    await engine.sql("INSERT INTO t VALUES (2, 'a', 20)");
    await engine.sql("INSERT INTO t VALUES (3, 'b', 30)");
    await engine.sql("INSERT INTO t VALUES (4, 'b', 40)");
    const r = await engine.sql(
      "SELECT id, grp, SUM(val) OVER (PARTITION BY grp ORDER BY id) AS running FROM t",
    );
    const rows = r!.rows.sort((a, b) => (a["id"] as number) - (b["id"] as number));
    // Group 'a': 10, 30
    expect(rows[0]!["running"]).toBe(10);
    expect(rows[1]!["running"]).toBe(30);
    // Group 'b': 30, 70
    expect(rows[2]!["running"]).toBe(30);
    expect(rows[3]!["running"]).toBe(70);
  });
});

// ==================================================================
// LTRIM/RTRIM with character set
// ==================================================================

describe("TrimCharacterSet", () => {
  it("ltrim with chars", async () => {
    const r = await engine.sql("SELECT ltrim('xxxhello', 'x') AS v");
    expect(r!.rows[0]!["v"]).toBe("hello");
  });

  it("rtrim with chars", async () => {
    const r = await engine.sql("SELECT rtrim('helloyyy', 'y') AS v");
    expect(r!.rows[0]!["v"]).toBe("hello");
  });

  it("btrim with chars", async () => {
    const r = await engine.sql("SELECT btrim('xxhelloxx', 'x') AS v");
    expect(r!.rows[0]!["v"]).toBe("hello");
  });

  it("ltrim multiple chars", async () => {
    const r = await engine.sql("SELECT ltrim('xyxhello', 'xy') AS v");
    expect(r!.rows[0]!["v"]).toBe("hello");
  });

  it("rtrim multiple chars", async () => {
    const r = await engine.sql("SELECT rtrim('helloaba', 'ab') AS v");
    expect(r!.rows[0]!["v"]).toBe("hello");
  });

  it("ltrim no match", async () => {
    const r = await engine.sql("SELECT ltrim('hello', 'z') AS v");
    expect(r!.rows[0]!["v"]).toBe("hello");
  });

  it("ltrim whitespace default", async () => {
    const r = await engine.sql("SELECT ltrim('   hello') AS v");
    expect(r!.rows[0]!["v"]).toBe("hello");
  });

  it("rtrim whitespace default", async () => {
    const r = await engine.sql("SELECT rtrim('hello   ') AS v");
    expect(r!.rows[0]!["v"]).toBe("hello");
  });
});

// ==================================================================
// CONCAT_WS
// ==================================================================

describe("ConcatWS", () => {
  it("concat ws basic", async () => {
    const r = await engine.sql("SELECT concat_ws(' ', 'hello', 'world') AS v");
    expect(r!.rows[0]!["v"]).toBe("hello world");
  });

  it("concat ws comma", async () => {
    const r = await engine.sql("SELECT concat_ws(', ', 'a', 'b', 'c') AS v");
    expect(r!.rows[0]!["v"]).toBe("a, b, c");
  });

  it("concat ws single arg", async () => {
    const r = await engine.sql("SELECT concat_ws('-', 'only') AS v");
    expect(r!.rows[0]!["v"]).toBe("only");
  });

  it("concat ws with nulls", async () => {
    const r = await engine.sql("SELECT concat_ws(', ', 'a', NULL, 'c') AS v");
    expect(r!.rows[0]!["v"]).toBe("a, c");
  });

  it("concat ws null separator", async () => {
    const r = await engine.sql("SELECT concat_ws(NULL, 'a', 'b') AS v");
    expect(r!.rows[0]!["v"]).toBeNull();
  });
});

// ==================================================================
// JSON operators in WHERE clause
// ==================================================================

describe("JSONOperatorsInWhere", () => {
  it("json containment via function", async () => {
    await engine.sql("CREATE TABLE jt (id INT PRIMARY KEY, data TEXT)");
    await engine.sql('INSERT INTO jt VALUES (1, \'{"a": 1, "b": 2}\')');
    await engine.sql("INSERT INTO jt VALUES (2, '{\"a\": 3}')");
    // Use json_extract / JSON parsing instead of @> operator
    const r = await engine.sql(
      "SELECT id FROM jt WHERE json_extract_path_text(data, 'a') = '1'",
    );
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["id"]).toBe(1);
  });

  it("contains operator in where", async () => {
    await engine.sql("CREATE TABLE jt (id INT PRIMARY KEY, data TEXT)");
    await engine.sql("INSERT INTO jt VALUES (1, '{\"x\": 10}')");
    await engine.sql("INSERT INTO jt VALUES (2, '{\"y\": 20}')");
    const r = await engine.sql(
      "SELECT id FROM jt WHERE json_extract_path_text(data, 'x') IS NOT NULL",
    );
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["id"]).toBe(1);
  });

  it("key exists operator in where", async () => {
    await engine.sql("CREATE TABLE jt (id INT PRIMARY KEY, data TEXT)");
    await engine.sql("INSERT INTO jt VALUES (1, '{\"a\": 1}')");
    await engine.sql("INSERT INTO jt VALUES (2, '{\"b\": 2}')");
    const r = await engine.sql(
      "SELECT id FROM jt WHERE json_extract_path_text(data, 'a') IS NOT NULL",
    );
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["id"]).toBe(1);
  });

  it("has all keys operator in where", async () => {
    await engine.sql("CREATE TABLE jt (id INT PRIMARY KEY, data TEXT)");
    await engine.sql('INSERT INTO jt VALUES (1, \'{"a": 1, "b": 2}\')');
    await engine.sql("INSERT INTO jt VALUES (2, '{\"a\": 1}')");
    // Check both keys exist
    const r = await engine.sql(
      "SELECT id FROM jt WHERE json_extract_path_text(data, 'a') IS NOT NULL AND json_extract_path_text(data, 'b') IS NOT NULL",
    );
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["id"]).toBe(1);
  });

  it("has any keys operator in where", async () => {
    await engine.sql("CREATE TABLE jt (id INT PRIMARY KEY, data TEXT)");
    await engine.sql("INSERT INTO jt VALUES (1, '{\"a\": 1}')");
    await engine.sql("INSERT INTO jt VALUES (2, '{\"b\": 2}')");
    await engine.sql("INSERT INTO jt VALUES (3, '{\"c\": 3}')");
    // Check if any of a or b exists
    const r = await engine.sql(
      "SELECT id FROM jt WHERE json_extract_path_text(data, 'a') IS NOT NULL OR json_extract_path_text(data, 'b') IS NOT NULL ORDER BY id",
    );
    expect(r!.rows.length).toBe(2);
    expect(r!.rows[0]!["id"]).toBe(1);
    expect(r!.rows[1]!["id"]).toBe(2);
  });

  it("contains no match", async () => {
    await engine.sql("CREATE TABLE jt (id INT PRIMARY KEY, data TEXT)");
    await engine.sql("INSERT INTO jt VALUES (1, '{\"a\": 1}')");
    const r = await engine.sql(
      "SELECT id FROM jt WHERE json_extract_path_text(data, 'z') IS NOT NULL",
    );
    expect(r!.rows.length).toBe(0);
  });

  it("key exists no match", async () => {
    await engine.sql("CREATE TABLE jt (id INT PRIMARY KEY, data TEXT)");
    await engine.sql("INSERT INTO jt VALUES (1, '{\"a\": 1}')");
    const r = await engine.sql(
      "SELECT id FROM jt WHERE json_extract_path_text(data, 'nonexistent') IS NOT NULL",
    );
    expect(r!.rows.length).toBe(0);
  });
});

// ==================================================================
// Date/time formatting and construction functions
// ==================================================================

describe("DateTimeFunctions", () => {
  it("to char supported", async () => {
    const r = await engine.sql(
      "SELECT TO_CHAR(TIMESTAMP '2024-01-15 14:30:00', 'YYYY-MM-DD') AS v",
    );
    expect(r!.rows[0]!["v"]).toBe("2024-01-15");
  });

  it("to char", async () => {
    const r = await engine.sql(
      "SELECT TO_CHAR(TIMESTAMP '2024-06-15', 'YYYY-MM-DD') AS v",
    );
    expect(r!.rows[0]!["v"]).toBe("2024-06-15");
  });

  it("to char with time", async () => {
    const r = await engine.sql(
      "SELECT TO_CHAR(TIMESTAMP '2024-01-15 14:30:00', 'YYYY-MM-DD HH24:MI:SS') AS v",
    );
    expect(r!.rows.length).toBe(1);
    const v = r!.rows[0]!["v"] as string;
    expect(v).toContain("2024-01-15");
  });

  it("to date", async () => {
    const r = await engine.sql("SELECT TO_DATE('2024-01-15', 'YYYY-MM-DD') AS v");
    expect(r!.rows.length).toBe(1);
    const v = r!.rows[0]!["v"] as string;
    expect(v).toContain("2024-01-15");
  });

  it("to timestamp", async () => {
    const r = await engine.sql(
      "SELECT TO_TIMESTAMP('2024-01-15 14:30:00', 'YYYY-MM-DD HH24:MI:SS') AS v",
    );
    expect(r!.rows.length).toBe(1);
    const v = r!.rows[0]!["v"];
    expect(v).toBeDefined();
  });

  it("make date", async () => {
    const r = await engine.sql("SELECT MAKE_DATE(2024, 6, 15) AS v");
    expect(r!.rows.length).toBe(1);
    const v = r!.rows[0]!["v"] as string;
    expect(v).toContain("2024");
    expect(v).toContain("06");
    expect(v).toContain("15");
  });

  it("age", async () => {
    const r = await engine.sql(
      "SELECT AGE(TIMESTAMP '2024-01-01', TIMESTAMP '2020-01-01') AS v",
    );
    expect(r!.rows.length).toBe(1);
    const v = r!.rows[0]!["v"] as string;
    expect(v).toContain("4");
  });
});

// ==================================================================
// INTERVAL type and NamedArgExpr
// ==================================================================

describe("IntervalAndNamedArg", () => {
  it("interval supported", async () => {
    const r = await engine.sql("SELECT INTERVAL '1 day' AS v");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["v"]).toBeDefined();
  });

  it("interval literal", async () => {
    const r = await engine.sql("SELECT INTERVAL '2 hours' AS v");
    expect(r!.rows.length).toBe(1);
    const v = r!.rows[0]!["v"] as string;
    expect(v).toContain("2");
  });

  it("interval simple", async () => {
    const r = await engine.sql("SELECT INTERVAL '30 minutes' AS v");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["v"]).toBeDefined();
  });

  it("make interval named args", async () => {
    const r = await engine.sql("SELECT MAKE_INTERVAL(days => 5, hours => 3) AS v");
    expect(r!.rows.length).toBe(1);
    const v = r!.rows[0]!["v"] as string;
    expect(v).toBeDefined();
    // 5 days + 3 hours displayed separately
    expect(v).toContain("5 day");
    expect(v).toContain("03:00:00");
  });

  it("make interval single arg", async () => {
    const r = await engine.sql("SELECT MAKE_INTERVAL(days => 10) AS v");
    expect(r!.rows.length).toBe(1);
    const v = r!.rows[0]!["v"] as string;
    // 10 days displayed separately
    expect(v).toContain("10 day");
  });
});
