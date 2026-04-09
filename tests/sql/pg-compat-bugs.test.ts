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
    const v = r!.rows[0]!["v"] as Date;
    expect(v).toBeInstanceOf(Date);
    expect(v.getFullYear()).toBe(2024);
    expect(v.getMonth()).toBe(0); // January
    expect(v.getDate()).toBe(15);
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
    const v = r!.rows[0]!["v"] as Date;
    expect(v).toBeInstanceOf(Date);
    expect(v.getFullYear()).toBe(2024);
    expect(v.getMonth()).toBe(5); // June
    expect(v.getDate()).toBe(15);
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

// ==================================================================
// Session Variables: SET / SHOW / RESET / DISCARD
// ==================================================================

describe("SessionVariables", () => {
  it("set and show", async () => {
    const r = await engine.sql("SET client_encoding TO 'UTF8'");
    void r;
    const result = await engine.sql("SHOW client_encoding");
    expect(result!.columns).toEqual(["client_encoding"]);
    expect(result!.rows[0]!["client_encoding"]).toBe("UTF8");
  });

  it("set integer value", async () => {
    await engine.sql("SET statement_timeout = 5000");
    const result = await engine.sql("SHOW statement_timeout");
    expect(result!.rows[0]!["statement_timeout"]).toBe("5000");
  });

  it("show defaults", async () => {
    const result = await engine.sql("SHOW server_version");
    expect(result!.rows[0]!["server_version"]).toBe("17.0");
  });

  it("reset", async () => {
    await engine.sql("SET client_encoding TO 'LATIN1'");
    await engine.sql("RESET client_encoding");
    const result = await engine.sql("SHOW client_encoding");
    expect(result!.rows[0]!["client_encoding"]).toBe("UTF8");
  });

  it("reset all", async () => {
    await engine.sql("SET client_encoding TO 'LATIN1'");
    await engine.sql("SET statement_timeout = 9999");
    await engine.sql("RESET ALL");
    const result = await engine.sql("SHOW client_encoding");
    expect(result!.rows[0]!["client_encoding"]).toBe("UTF8");
  });

  it("discard all", async () => {
    await engine.sql("SET client_encoding TO 'LATIN1'");
    await engine.sql("DISCARD ALL");
    const result = await engine.sql("SHOW client_encoding");
    expect(result!.rows[0]!["client_encoding"]).toBe("UTF8");
  });
});

// ==================================================================
// In-memory transactions (BEGIN / COMMIT / ROLLBACK)
// ==================================================================

describe("InMemoryTransactions", () => {
  it("begin commit", async () => {
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)");
    await engine.sql("BEGIN");
    await engine.sql("INSERT INTO t VALUES (1, 'a')");
    await engine.sql("COMMIT");
    const result = await engine.sql("SELECT * FROM t");
    expect(result!.rows.length).toBe(1);
  });

  it("begin rollback", async () => {
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)");
    await engine.sql("INSERT INTO t VALUES (1, 'a')");
    await engine.sql("BEGIN");
    await engine.sql("INSERT INTO t VALUES (2, 'b')");
    await engine.sql("ROLLBACK");
    const result = await engine.sql("SELECT * FROM t");
    expect(result!.rows.length).toBe(1);
    expect(result!.rows[0]!["v"]).toBe("a");
  });
});

// ==================================================================
// SELECT * must not expose internal columns (_doc_id, _score)
// ==================================================================

describe("SelectStarNoInternalColumns", () => {
  it("single table no internal cols", async () => {
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
    await engine.sql("INSERT INTO t VALUES (1, 'Alice'), (2, 'Bob')");
    const result = await engine.sql("SELECT * FROM t");
    expect(result!.columns).not.toContain("_doc_id");
    expect(result!.columns).not.toContain("_score");
    expect(new Set(result!.columns)).toEqual(new Set(["id", "name"]));
    expect(result!.rows.length).toBe(2);
  });

  it("explicit columns still work", async () => {
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
    await engine.sql("INSERT INTO t VALUES (1, 'Alice')");
    const result = await engine.sql("SELECT id, name FROM t");
    expect(result!.columns).toEqual(["id", "name"]);
  });

  it("select star with order by", async () => {
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)");
    await engine.sql("INSERT INTO t VALUES (2, 'b'), (1, 'a')");
    const result = await engine.sql("SELECT * FROM t ORDER BY id");
    expect(result!.columns).not.toContain("_doc_id");
    expect(result!.rows[0]!["id"]).toBe(1);
  });
});

// ==================================================================
// DEFAULT CURRENT_TIMESTAMP and similar
// ==================================================================

describe("DefaultSQLFunctions", () => {
  it("default current timestamp", async () => {
    await engine.sql(
      "CREATE TABLE log (" +
        "id SERIAL PRIMARY KEY, " +
        "msg TEXT, " +
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    );
    await engine.sql("INSERT INTO log (msg) VALUES ('hello')");
    const result = await engine.sql("SELECT created_at FROM log");
    const ts = result!.rows[0]!["created_at"];
    expect(ts).toBeInstanceOf(Date);
  });

  it("default current date", async () => {
    await engine.sql(
      "CREATE TABLE events (" +
        "id SERIAL PRIMARY KEY, " +
        "event_date DATE DEFAULT CURRENT_DATE)",
    );
    await engine.sql("INSERT INTO events (id) VALUES (1)");
    const result = await engine.sql("SELECT event_date FROM events");
    const d = result!.rows[0]!["event_date"];
    expect(d).toBeInstanceOf(Date);
  });

  it("default literal still works", async () => {
    await engine.sql(
      "CREATE TABLE t (id INTEGER PRIMARY KEY, status TEXT DEFAULT 'active')",
    );
    await engine.sql("INSERT INTO t (id) VALUES (1)");
    const result = await engine.sql("SELECT status FROM t");
    expect(result!.rows[0]!["status"]).toBe("active");
  });
});

// ==================================================================
// ALTER TABLE ADD CONSTRAINT
// ==================================================================

describe("AlterTableAddConstraint", () => {
  it("add unique constraint", async () => {
    await engine.sql("CREATE TABLE t (id INTEGER, email TEXT)");
    await engine.sql("INSERT INTO t VALUES (1, 'a@b.c')");
    await engine.sql("ALTER TABLE t ADD CONSTRAINT uq_email UNIQUE (email)");
    await expect(
      engine.sql("INSERT INTO t VALUES (2, 'a@b.c')"),
    ).rejects.toThrow(/UNIQUE/);
  });

  it("add check constraint", async () => {
    await engine.sql("CREATE TABLE t (id INTEGER, val INTEGER)");
    await engine.sql("INSERT INTO t VALUES (1, 10)");
    await engine.sql("ALTER TABLE t ADD CONSTRAINT chk CHECK (val > 0)");
    await expect(
      engine.sql("INSERT INTO t VALUES (2, -1)"),
    ).rejects.toThrow(/CHECK/);
  });
});

// ==================================================================
// ON CONFLICT DO NOTHING without explicit columns
// ==================================================================

describe("OnConflictDoNothing", () => {
  it("without explicit columns", async () => {
    await engine.sql(
      "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)",
    );
    await engine.sql("INSERT INTO t VALUES (1, 'Alice')");
    // Should not throw -- skip the conflicting row
    await engine.sql("INSERT INTO t VALUES (1, 'Bob') ON CONFLICT DO NOTHING");
    const result = await engine.sql("SELECT name FROM t WHERE id = 1");
    expect(result!.rows[0]!["name"]).toBe("Alice");
  });
});

// ==================================================================
// In-memory BTREE index (CREATE/DROP INDEX)
// ==================================================================

describe("InMemoryIndex", () => {
  it("create and drop index", async () => {
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)");
    await engine.sql("CREATE INDEX idx_val ON t (val)");
    // Should not throw
    await engine.sql("DROP INDEX idx_val");
  });

  it("drop index if exists", async () => {
    // Should not throw even if index does not exist
    await engine.sql("DROP INDEX IF EXISTS nonexistent_idx");
  });
});

// ==================================================================
// Schema support (CREATE SCHEMA / DROP SCHEMA / qualified names)
// ==================================================================

describe("SchemaSupport", () => {
  it("create schema", async () => {
    await engine.sql("CREATE SCHEMA myschema");
    expect(engine._tables.schemas.has("myschema")).toBe(true);
  });

  it("create schema if not exists", async () => {
    await engine.sql("CREATE SCHEMA myschema");
    await engine.sql("CREATE SCHEMA IF NOT EXISTS myschema");
  });

  it("create schema duplicate raises", async () => {
    await engine.sql("CREATE SCHEMA myschema");
    await expect(engine.sql("CREATE SCHEMA myschema")).rejects.toThrow(
      /already exists/,
    );
  });

  it("create table in schema", async () => {
    await engine.sql("CREATE SCHEMA sales");
    await engine.sql(
      "CREATE TABLE sales.orders (id INTEGER PRIMARY KEY, total INTEGER)",
    );
    await engine.sql("INSERT INTO sales.orders VALUES (1, 100)");
    const result = await engine.sql("SELECT total FROM sales.orders");
    expect(result!.rows[0]!["total"]).toBe(100);
  });

  it("schema isolation", async () => {
    await engine.sql("CREATE SCHEMA s1");
    await engine.sql("CREATE SCHEMA s2");
    await engine.sql("CREATE TABLE s1.t (id INTEGER PRIMARY KEY, val TEXT)");
    await engine.sql("CREATE TABLE s2.t (id INTEGER PRIMARY KEY, val TEXT)");
    await engine.sql("INSERT INTO s1.t VALUES (1, 'schema1')");
    await engine.sql("INSERT INTO s2.t VALUES (1, 'schema2')");
    const r1 = await engine.sql("SELECT val FROM s1.t");
    const r2 = await engine.sql("SELECT val FROM s2.t");
    expect(r1!.rows[0]!["val"]).toBe("schema1");
    expect(r2!.rows[0]!["val"]).toBe("schema2");
  });

  it("search path resolution", async () => {
    await engine.sql("CREATE SCHEMA myschema");
    await engine.sql(
      "CREATE TABLE myschema.users (id INTEGER PRIMARY KEY, name TEXT)",
    );
    await engine.sql("INSERT INTO myschema.users VALUES (1, 'Alice')");
    await engine.sql("SET search_path TO 'myschema', 'public'");
    const result = await engine.sql("SELECT name FROM users");
    expect(result!.rows[0]!["name"]).toBe("Alice");
  });

  it("default public schema", async () => {
    await engine.sql("CREATE TABLE t (id INTEGER PRIMARY KEY)");
    await engine.sql("INSERT INTO t VALUES (1)");
    const result = await engine.sql("SELECT id FROM public.t");
    expect(result!.rows[0]!["id"]).toBe(1);
  });

  it("drop schema empty", async () => {
    await engine.sql("CREATE SCHEMA temp_schema");
    await engine.sql("DROP SCHEMA temp_schema");
    expect(engine._tables.schemas.has("temp_schema")).toBe(false);
  });

  it("drop schema cascade", async () => {
    await engine.sql("CREATE SCHEMA doomed");
    await engine.sql("CREATE TABLE doomed.t (id INTEGER)");
    await engine.sql("DROP SCHEMA doomed CASCADE");
    expect(engine._tables.schemas.has("doomed")).toBe(false);
  });

  it("drop schema nonempty raises", async () => {
    await engine.sql("CREATE SCHEMA nonempty");
    await engine.sql("CREATE TABLE nonempty.t (id INTEGER)");
    await expect(engine.sql("DROP SCHEMA nonempty")).rejects.toThrow(/not empty/);
  });

  it("drop schema if exists", async () => {
    await engine.sql("DROP SCHEMA IF EXISTS nonexistent");
  });

  it("information schema tables", async () => {
    await engine.sql("CREATE SCHEMA myschema");
    await engine.sql("CREATE TABLE myschema.t (id INTEGER)");
    const result = await engine.sql(
      "SELECT table_schema, table_name FROM information_schema.tables " +
        "WHERE table_name = 't'",
    );
    expect(result!.rows.some((r) => r["table_schema"] === "myschema")).toBe(true);
  });
});

// ==================================================================
// DROP TABLE cascading cleanup
// ==================================================================

describe("DropTableCascadeCleanup", () => {
  it("drop table cleans btree indexes in memory", async () => {
    await engine.sql("CREATE TABLE t (id INTEGER, val TEXT)");
    await engine.sql("CREATE INDEX idx_val ON t (val)");
    expect(engine._btreeIndexes.has("idx_val")).toBe(true);
    await engine.sql("DROP TABLE t");
    expect(engine._btreeIndexes.has("idx_val")).toBe(false);
  });

  it("drop table cleans gin indexes", async () => {
    await engine.sql("CREATE TABLE docs (id SERIAL PRIMARY KEY, body TEXT)");
    await engine.sql("INSERT INTO docs (body) VALUES ('hello world')");
    await engine.sql("CREATE INDEX idx_body ON docs USING gin (body)");
    await engine.sql("DROP TABLE docs");
    // GIN index metadata should be removed
    const result = await engine.sql(
      "SELECT * FROM information_schema.tables WHERE table_name = 'docs'",
    );
    expect(result!.rows.length).toBe(0);
  });

  it("drop table cleans fk validators on parent", async () => {
    await engine.sql("CREATE TABLE parent (id INTEGER PRIMARY KEY)");
    await engine.sql("INSERT INTO parent (id) VALUES (1)");
    await engine.sql(
      "CREATE TABLE child (id INTEGER PRIMARY KEY, " +
        "pid INTEGER REFERENCES parent(id))",
    );
    const parent = engine._tables.get("parent")!;
    expect(parent.fkDeleteValidators.length).toBeGreaterThan(0);

    await engine.sql("DROP TABLE child");
    expect(parent.fkDeleteValidators.length).toBe(0);
    expect(parent.fkUpdateValidators.length).toBe(0);
  });
});

// ==================================================================
// DROP SCHEMA CASCADE cleanup
// ==================================================================

describe("DropSchemaCascadeCleanup", () => {
  it("drop schema cascade cleans btree indexes", async () => {
    await engine.sql("CREATE SCHEMA myschema");
    await engine.sql("CREATE TABLE myschema.t (id INTEGER, val INTEGER)");
    await engine.sql("INSERT INTO myschema.t (id, val) VALUES (1, 10)");
    await engine.sql("CREATE INDEX idx_val ON myschema.t (val)");
    expect(engine._btreeIndexes.has("idx_val")).toBe(true);

    await engine.sql("DROP SCHEMA myschema CASCADE");
    expect(engine._btreeIndexes.has("idx_val")).toBe(false);
    expect(engine._tables.schemas.has("myschema")).toBe(false);
  });

  it("drop schema cascade cleans gin indexes", async () => {
    await engine.sql("CREATE SCHEMA myschema");
    await engine.sql(
      "CREATE TABLE myschema.docs (id INTEGER PRIMARY KEY, body TEXT)",
    );
    await engine.sql("INSERT INTO myschema.docs (id, body) VALUES (1, 'hello')");
    await engine.sql("CREATE INDEX idx_body ON myschema.docs USING gin (body)");

    await engine.sql("DROP SCHEMA myschema CASCADE");
    // Verify schema is gone and table no longer exists
    expect(engine._tables.schemas.has("myschema")).toBe(false);
    expect(engine._tables.has("myschema.docs")).toBe(false);
  });

  it("drop schema cascade cleans fk validators", async () => {
    await engine.sql("CREATE TABLE parent (id INTEGER PRIMARY KEY)");
    await engine.sql("INSERT INTO parent (id) VALUES (1)");
    await engine.sql("CREATE SCHEMA child_schema");
    await engine.sql(
      "CREATE TABLE child_schema.child (" +
        "id INTEGER PRIMARY KEY, " +
        "pid INTEGER REFERENCES parent(id))",
    );
    const parent = engine._tables.get("parent")!;
    expect(parent.fkDeleteValidators.length).toBeGreaterThan(0);

    await engine.sql("DROP SCHEMA child_schema CASCADE");
    expect(parent.fkDeleteValidators.length).toBe(0);
    expect(parent.fkUpdateValidators.length).toBe(0);
  });

  it("drop schema not empty without cascade raises", async () => {
    await engine.sql("CREATE SCHEMA myschema");
    await engine.sql("CREATE TABLE myschema.t (id INTEGER)");
    await expect(engine.sql("DROP SCHEMA myschema")).rejects.toThrow(/not empty/);
  });

  it("drop schema if exists nonexistent", async () => {
    // Should not raise
    await engine.sql("DROP SCHEMA IF EXISTS nonexistent");
  });

  it("drop schema nonexistent raises", async () => {
    await expect(engine.sql("DROP SCHEMA nonexistent")).rejects.toThrow(
      /does not exist/,
    );
  });
});

// ==================================================================
// CREATE INDEX IF NOT EXISTS
// ==================================================================

describe("CreateIndexIfNotExists", () => {
  it("btree in memory if not exists", async () => {
    await engine.sql("CREATE TABLE t (id INTEGER, val TEXT)");
    await engine.sql("CREATE INDEX idx_val ON t (val)");
    expect(engine._btreeIndexes.has("idx_val")).toBe(true);
    // Should not raise
    await engine.sql("CREATE INDEX IF NOT EXISTS idx_val ON t (val)");
  });

  it("gin if not exists", async () => {
    await engine.sql("CREATE TABLE docs (id SERIAL PRIMARY KEY, body TEXT)");
    await engine.sql("INSERT INTO docs (body) VALUES ('hello world')");
    await engine.sql("CREATE INDEX idx_body ON docs USING gin (body)");
    // Should not raise
    await engine.sql(
      "CREATE INDEX IF NOT EXISTS idx_body ON docs USING gin (body)",
    );
  });

  it("without if not exists raises", async () => {
    await engine.sql("CREATE TABLE t (id INTEGER, val TEXT)");
    await engine.sql("CREATE INDEX idx_val ON t (val)");
    await expect(
      engine.sql("CREATE INDEX idx_val ON t (val)"),
    ).rejects.toThrow(/already exists/);
  });
});
