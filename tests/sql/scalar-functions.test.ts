import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// Helpers
// =============================================================================

async function makeDataEngine(): Promise<Engine> {
  const e = new Engine();
  await e.sql("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)");
  await e.sql("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)");
  await e.sql("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)");
  await e.sql("INSERT INTO users (id, name, age) VALUES (3, 'Carol', 35)");
  await e.sql("INSERT INTO users (id, name, age) VALUES (4, 'Dave', 25)");
  return e;
}

async function makeTableEngine(): Promise<Engine> {
  const e = new Engine();
  await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER, name TEXT)");
  await e.sql("INSERT INTO t (id, val, name) VALUES (1, 10, 'alpha')");
  await e.sql("INSERT INTO t (id, val, name) VALUES (2, 20, 'bravo')");
  await e.sql("INSERT INTO t (id, val, name) VALUES (3, 30, 'charlie')");
  return e;
}

// =============================================================================
// GREATEST / LEAST / NULLIF
// =============================================================================

describe("GreatestLeastNullif", () => {
  it("greatest basic", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT GREATEST(1, 5, 3)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe(5);
  });

  it("greatest with nulls", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT GREATEST(1, NULL, 3)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe(3);
  });

  it("greatest all null", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT GREATEST(NULL, NULL)");
    expect(result!.rows[0]![result!.columns[0]!] ?? null).toBeNull();
  });

  it("least basic", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT LEAST(10, 5, 8)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe(5);
  });

  it("least with nulls", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT LEAST(10, NULL, 3)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe(3);
  });

  it("nullif equal", async () => {
    const e = await makeDataEngine();
    const result = await e.sql(
      "SELECT NULLIF(age, 25) AS result FROM users WHERE name = 'Bob'",
    );
    expect(result!.rows[0]!["result"] ?? null).toBeNull();
  });

  it("nullif not equal", async () => {
    const e = await makeDataEngine();
    const result = await e.sql(
      "SELECT NULLIF(age, 99) AS result FROM users WHERE name = 'Alice'",
    );
    expect(result!.rows[0]!["result"]).toBe(30);
  });

  it("nullif null", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT NULLIF(NULL, NULL)");
    expect(result!.rows[0]![result!.columns[0]!] ?? null).toBeNull();
  });
});

// =============================================================================
// String functions
// =============================================================================

describe("StringFunctions", () => {
  it("position", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT POSITION('lo' IN 'hello world')");
    expect(result!.rows[0]![result!.columns[0]!]).toBe(4);
  });

  it("position not found", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT POSITION('xyz' IN 'hello')");
    expect(result!.rows[0]![result!.columns[0]!]).toBe(0);
  });

  it("char length", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT CHAR_LENGTH('hello')");
    expect(result!.rows[0]![result!.columns[0]!]).toBe(5);
  });

  it("lpad", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT LPAD('hi', 5, 'x')");
    expect(result!.rows[0]![result!.columns[0]!]).toBe("xxxhi");
  });

  it("lpad default fill", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT LPAD('hi', 5)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe("   hi");
  });

  it("lpad truncate", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT LPAD('hello', 3)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe("hel");
  });

  it("rpad", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT RPAD('hi', 5, 'x')");
    expect(result!.rows[0]![result!.columns[0]!]).toBe("hixxx");
  });

  it("repeat", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT REPEAT('ab', 3)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe("ababab");
  });

  it("reverse", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT REVERSE('hello')");
    expect(result!.rows[0]![result!.columns[0]!]).toBe("olleh");
  });

  it("split part", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT SPLIT_PART('a,b,c', ',', 2)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe("b");
  });

  it("split part out of range", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT SPLIT_PART('a,b', ',', 5)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe("");
  });
});

// =============================================================================
// String function aliases
// =============================================================================

describe("StringFunctionAliases", () => {
  it("character length", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT CHARACTER_LENGTH('hello') AS len");
    expect(result!.rows[0]!["len"]).toBe(5);
  });

  it("strpos", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT STRPOS('hello world', 'lo') AS pos");
    expect(result!.rows[0]!["pos"]).toBe(4);
  });
});

// =============================================================================
// Math functions
// =============================================================================

describe("MathFunctions", () => {
  it("power", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT POWER(2, 10)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe(1024);
  });

  it("pow", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT POW(3, 2)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe(9);
  });

  it("sqrt", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT SQRT(16)");
    expect(result!.rows[0]![result!.columns[0]!]).toBeCloseTo(4.0);
  });

  it("log", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT LOG(100)");
    expect(result!.rows[0]![result!.columns[0]!]).toBeCloseTo(2.0);
  });

  it("ln", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT LN(1)");
    expect(result!.rows[0]![result!.columns[0]!]).toBeCloseTo(0.0);
  });

  it("exp", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT EXP(0)");
    expect(result!.rows[0]![result!.columns[0]!]).toBeCloseTo(1.0);
  });

  it("mod", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT MOD(10, 3)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe(1);
  });

  it("trunc", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT TRUNC(3.7)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe(3);
  });

  it("trunc with precision", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT TRUNC(3.456, 2)");
    expect(result!.rows[0]![result!.columns[0]!]).toBeCloseTo(3.45);
  });

  it("sign positive", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT SIGN(42)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe(1);
  });

  it("sign negative", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT SIGN(-5)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe(-1);
  });

  it("sign zero", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT SIGN(0)");
    expect(result!.rows[0]![result!.columns[0]!]).toBe(0);
  });

  it("pi", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT PI()");
    expect(result!.rows[0]![result!.columns[0]!]).toBeCloseTo(Math.PI);
  });

  it("random", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT RANDOM()");
    const val = result!.rows[0]![result!.columns[0]!] as number;
    expect(val).toBeGreaterThanOrEqual(0.0);
    expect(val).toBeLessThan(1.0);
  });
});

// =============================================================================
// LOG with two arguments
// =============================================================================

describe("LogTwoArgs", () => {
  it("log base 2", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT LOG(2, 8) AS val");
    expect(result!.rows[0]!["val"]).toBeCloseTo(3.0);
  });

  it("log base 10 explicit", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT LOG(10, 1000) AS val");
    expect(result!.rows[0]!["val"]).toBeCloseTo(3.0);
  });

  it("log single arg unchanged", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT LOG(100) AS val");
    expect(result!.rows[0]!["val"]).toBeCloseTo(2.0);
  });
});

// =============================================================================
// Scalar functions (Step 7): initcap, translate, ascii, chr, starts_with
// =============================================================================

describe("ScalarFunctionsStep7", () => {
  it("initcap", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY)");
    await e.sql("INSERT INTO t (id) VALUES (1)");
    const result = await e.sql("SELECT initcap('hello world') AS v FROM t");
    expect(result!.rows[0]!["v"]).toBe("Hello World");
  });

  it("translate", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY)");
    await e.sql("INSERT INTO t (id) VALUES (1)");
    const result = await e.sql("SELECT translate('12345', '143', 'ax') AS v FROM t");
    expect(result!.rows[0]!["v"]).toBe("a2x5");
  });

  it("ascii", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY)");
    await e.sql("INSERT INTO t (id) VALUES (1)");
    const result = await e.sql("SELECT ascii('A') AS v FROM t");
    expect(result!.rows[0]!["v"]).toBe(65);
  });

  it("chr", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY)");
    await e.sql("INSERT INTO t (id) VALUES (1)");
    const result = await e.sql("SELECT chr(65) AS v FROM t");
    expect(result!.rows[0]!["v"]).toBe("A");
  });

  it("starts with", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
    await e.sql("INSERT INTO t (id, name) VALUES (1, 'PostgreSQL')");
    const result = await e.sql(
      "SELECT starts_with(name, 'Post') AS v FROM t WHERE id = 1",
    );
    expect(result!.rows[0]!["v"]).toBe(true);
  });
});

// =============================================================================
// OCTET_LENGTH
// =============================================================================

describe("OctetLength", () => {
  it("ascii", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT octet_length('hello') AS v FROM t WHERE id = 1");
    expect(result!.rows[0]!["v"]).toBe(5);
  });

  it("multibyte", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT octet_length(name) AS v FROM t WHERE id = 1");
    expect(result!.rows[0]!["v"]).toBe(5); // 'alpha' = 5 bytes
  });
});

// =============================================================================
// MD5
// =============================================================================

describe("MD5", () => {
  it("basic", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT md5('hello') AS v FROM t WHERE id = 1");
    expect(result!.rows[0]!["v"]).toBe("5d41402abc4b2a76b9719d911017c592");
  });
});

// =============================================================================
// FORMAT
// =============================================================================

describe("Format", () => {
  it("basic", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      "SELECT format('Hello %s, you are %s', 'World', 'great') AS v " +
        "FROM t WHERE id = 1",
    );
    expect(result!.rows[0]!["v"]).toBe("Hello World, you are great");
  });
});

// =============================================================================
// REGEXP_MATCH
// =============================================================================

describe("RegexpMatch", () => {
  it("basic", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      "SELECT regexp_match('foobarbaz', 'b(.)r') AS v FROM t WHERE id = 1",
    );
    expect(result!.rows[0]!["v"]).toEqual(["a"]);
  });

  it("no match", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      "SELECT regexp_match('hello', 'xyz') AS v FROM t WHERE id = 1",
    );
    expect(result!.rows[0]!["v"] ?? null).toBeNull();
  });
});

// =============================================================================
// REGEXP_REPLACE
// =============================================================================

describe("RegexpReplace", () => {
  it("basic", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      "SELECT regexp_replace('hello world', 'world', 'there') AS v " +
        "FROM t WHERE id = 1",
    );
    expect(result!.rows[0]!["v"]).toBe("hello there");
  });

  it("global flag", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      "SELECT regexp_replace('aaa', 'a', 'b', 'g') AS v FROM t WHERE id = 1",
    );
    expect(result!.rows[0]!["v"]).toBe("bbb");
  });
});

// =============================================================================
// OVERLAY
// =============================================================================

describe("Overlay", () => {
  it("basic", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      "SELECT overlay('Txxxxas' placing 'hom' from 2 for 4) AS v " +
        "FROM t WHERE id = 1",
    );
    expect(result!.rows[0]!["v"]).toBe("Thomas");
  });
});

// =============================================================================
// CBRT
// =============================================================================

describe("Cbrt", () => {
  it("basic", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT cbrt(27) AS v FROM t WHERE id = 1");
    expect(Math.abs((result!.rows[0]!["v"] as number) - 3.0)).toBeLessThan(0.001);
  });

  it("negative", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT cbrt(-8) AS v FROM t WHERE id = 1");
    expect(Math.abs((result!.rows[0]!["v"] as number) - -2.0)).toBeLessThan(0.001);
  });
});

// =============================================================================
// Trigonometric functions
// =============================================================================

describe("TrigFunctions", () => {
  it("sin", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT sin(0) AS v FROM t WHERE id = 1");
    expect(Math.abs(result!.rows[0]!["v"] as number)).toBeLessThan(0.001);
  });

  it("cos", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT cos(0) AS v FROM t WHERE id = 1");
    expect(Math.abs((result!.rows[0]!["v"] as number) - 1.0)).toBeLessThan(0.001);
  });

  it("tan", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT tan(0) AS v FROM t WHERE id = 1");
    expect(Math.abs(result!.rows[0]!["v"] as number)).toBeLessThan(0.001);
  });

  it("asin", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT asin(1) AS v FROM t WHERE id = 1");
    expect(Math.abs((result!.rows[0]!["v"] as number) - Math.PI / 2)).toBeLessThan(
      0.001,
    );
  });

  it("acos", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT acos(1) AS v FROM t WHERE id = 1");
    expect(Math.abs(result!.rows[0]!["v"] as number)).toBeLessThan(0.001);
  });

  it("atan", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT atan(1) AS v FROM t WHERE id = 1");
    expect(Math.abs((result!.rows[0]!["v"] as number) - Math.PI / 4)).toBeLessThan(
      0.001,
    );
  });

  it("atan2", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT atan2(1, 1) AS v FROM t WHERE id = 1");
    expect(Math.abs((result!.rows[0]!["v"] as number) - Math.PI / 4)).toBeLessThan(
      0.001,
    );
  });
});

// =============================================================================
// DEGREES / RADIANS
// =============================================================================

describe("DegreesRadians", () => {
  it("degrees", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT degrees(pi()) AS v FROM t WHERE id = 1");
    expect(Math.abs((result!.rows[0]!["v"] as number) - 180.0)).toBeLessThan(0.001);
  });

  it("radians", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT radians(180) AS v FROM t WHERE id = 1");
    expect(Math.abs((result!.rows[0]!["v"] as number) - Math.PI)).toBeLessThan(0.001);
  });
});

// =============================================================================
// DIV
// =============================================================================

describe("Div", () => {
  it("basic", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT div(7, 2) AS v FROM t WHERE id = 1");
    expect(result!.rows[0]!["v"]).toBe(3);
  });

  it("negative", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT div(-7, 2) AS v FROM t WHERE id = 1");
    expect(result!.rows[0]!["v"]).toBe(-3);
  });
});

// =============================================================================
// GCD / LCM
// =============================================================================

describe("GcdLcm", () => {
  it("gcd", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT gcd(12, 8) AS v FROM t WHERE id = 1");
    expect(result!.rows[0]!["v"]).toBe(4);
  });

  it("lcm", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT lcm(12, 8) AS v FROM t WHERE id = 1");
    expect(result!.rows[0]!["v"]).toBe(24);
  });
});

// =============================================================================
// WIDTH_BUCKET
// =============================================================================

describe("WidthBucket", () => {
  it("in range", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      "SELECT width_bucket(5.0, 0, 10, 5) AS v FROM t WHERE id = 1",
    );
    expect(result!.rows[0]!["v"]).toBe(3);
  });

  it("below range", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      "SELECT width_bucket(-1, 0, 10, 5) AS v FROM t WHERE id = 1",
    );
    expect(result!.rows[0]!["v"]).toBe(0);
  });

  it("above range", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      "SELECT width_bucket(15, 0, 10, 5) AS v FROM t WHERE id = 1",
    );
    expect(result!.rows[0]!["v"]).toBe(6);
  });
});
