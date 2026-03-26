import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// Helpers
// =============================================================================

async function makeJSONEngine(): Promise<Engine> {
  const e = new Engine();
  await e.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, data JSON, label TEXT)");
  await e.sql(
    "INSERT INTO docs (id, data, label) VALUES " +
      '(1, \'{"name": "Alice", "age": 30, "tags": ["a", "b"]}\', \'first\')',
  );
  await e.sql(
    "INSERT INTO docs (id, data, label) VALUES " +
      '(2, \'{"name": "Bob", "age": 25, "tags": ["c"]}\', \'second\')',
  );
  await e.sql(
    "INSERT INTO docs (id, data, label) VALUES " +
      '(3, \'{"name": "Carol", "nested": {"x": 10}}\', \'third\')',
  );
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
// JSON/JSONB type
// =============================================================================

describe("JSONType", () => {
  it("create table with json", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, data JSON)");
    await e.sql("INSERT INTO t (id, data) VALUES (1, '{\"x\":1}')");
    const result = await e.sql("SELECT * FROM t");
    expect(result!.columns).toContain("data");
  });

  it("create table with jsonb", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, data JSONB)");
    await e.sql("INSERT INTO t (id, data) VALUES (1, '{\"x\":1}')");
    const result = await e.sql("SELECT * FROM t");
    expect(result!.columns).toContain("data");
  });

  it("insert json string", async () => {
    const e = await makeJSONEngine();
    const result = await e.sql("SELECT data FROM docs WHERE id = 1");
    const data = result!.rows[0]!["data"] as Record<string, unknown>;
    expect(typeof data).toBe("object");
    expect(data["name"]).toBe("Alice");
    expect(data["age"]).toBe(30);
  });

  it("insert json array", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, items JSON)");
    await e.sql("INSERT INTO t (id, items) VALUES (1, '[1, 2, 3]')");
    const result = await e.sql("SELECT items FROM t WHERE id = 1");
    expect(result!.rows[0]!["items"]).toEqual([1, 2, 3]);
  });
});

// =============================================================================
// JSON operators
// =============================================================================

describe("JSONOperators", () => {
  it("arrow text key", async () => {
    const e = await makeJSONEngine();
    const result = await e.sql("SELECT data->'name' AS name FROM docs WHERE id = 1");
    expect(result!.rows[0]!["name"]).toBe("Alice");
  });

  it("double arrow text key", async () => {
    const e = await makeJSONEngine();
    const result = await e.sql("SELECT data->>'name' AS name FROM docs WHERE id = 1");
    expect(result!.rows[0]!["name"]).toBe("Alice");
    expect(typeof result!.rows[0]!["name"]).toBe("string");
  });

  it("arrow integer key", async () => {
    const e = await makeJSONEngine();
    const result = await e.sql(
      "SELECT data->'tags'->0 AS first_tag FROM docs WHERE id = 1",
    );
    expect(result!.rows[0]!["first_tag"]).toBe("a");
  });

  it("arrow nested object", async () => {
    const e = await makeJSONEngine();
    const result = await e.sql(
      "SELECT data->'nested'->'x' AS x FROM docs WHERE id = 3",
    );
    expect(result!.rows[0]!["x"]).toBe(10);
  });

  it("double arrow returns text for nested", async () => {
    const e = await makeJSONEngine();
    const result = await e.sql("SELECT data->>'tags' AS tags FROM docs WHERE id = 1");
    expect(typeof result!.rows[0]!["tags"]).toBe("string");
    expect(result!.rows[0]!["tags"] as string).toContain('"a"');
  });

  it("arrow missing key returns null", async () => {
    const e = await makeJSONEngine();
    const result = await e.sql(
      "SELECT data->'nonexistent' AS v FROM docs WHERE id = 1",
    );
    expect(result!.rows[0]!["v"] ?? null).toBeNull();
  });

  it("json in where", async () => {
    const e = await makeJSONEngine();
    const result = await e.sql("SELECT id FROM docs WHERE data->>'name' = 'Bob'");
    expect(result!.rows.length).toBe(1);
    expect(result!.rows[0]!["id"]).toBe(2);
  });
});

// =============================================================================
// JSON functions
// =============================================================================

describe("JSONFunctions", () => {
  it("json build object", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY)");
    await e.sql("INSERT INTO t (id) VALUES (1)");
    const result = await e.sql(
      "SELECT json_build_object('a', 1, 'b', 2) AS obj FROM t",
    );
    expect(result!.rows[0]!["obj"]).toEqual({ a: 1, b: 2 });
  });

  it("json build array", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY)");
    await e.sql("INSERT INTO t (id) VALUES (1)");
    const result = await e.sql("SELECT json_build_array(1, 2, 3) AS arr FROM t");
    expect(result!.rows[0]!["arr"]).toEqual([1, 2, 3]);
  });

  it("json build array mixed types", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT json_build_array(1, 2, 3, 'four') AS arr");
    const arr = result!.rows[0]!["arr"] as unknown[];
    expect(arr.length).toBe(4);
    expect(arr[3]).toBe("four");
  });

  it("json build array mixed int float str bool", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT json_build_array(1, 2.5, 'hello', true) AS arr");
    const arr = result!.rows[0]!["arr"] as unknown[];
    expect(arr.length).toBe(4);
  });

  it("json build array empty", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT json_build_array() AS arr");
    expect(result!.rows[0]!["arr"]).toEqual([]);
  });

  it("json typeof object", async () => {
    const e = await makeJSONEngine();
    const result = await e.sql("SELECT json_typeof(data) AS t FROM docs WHERE id = 1");
    expect(result!.rows[0]!["t"]).toBe("object");
  });

  it("json typeof array", async () => {
    const e = await makeJSONEngine();
    const result = await e.sql(
      "SELECT json_typeof(data->'tags') AS t FROM docs WHERE id = 1",
    );
    expect(result!.rows[0]!["t"]).toBe("array");
  });

  it("json array length", async () => {
    const e = await makeJSONEngine();
    const result = await e.sql(
      "SELECT json_array_length(data->'tags') AS n FROM docs WHERE id = 1",
    );
    expect(result!.rows[0]!["n"]).toBe(2);
  });

  it("json array length single", async () => {
    const e = await makeJSONEngine();
    const result = await e.sql(
      "SELECT json_array_length(data->'tags') AS n FROM docs WHERE id = 2",
    );
    expect(result!.rows[0]!["n"]).toBe(1);
  });

  it("json extract path", async () => {
    const e = await makeJSONEngine();
    const result = await e.sql(
      "SELECT json_extract_path(data, 'nested', 'x') AS v FROM docs WHERE id = 3",
    );
    expect(result!.rows[0]!["v"]).toBe(10);
  });

  it("json extract path text", async () => {
    const e = await makeJSONEngine();
    const result = await e.sql(
      "SELECT json_extract_path_text(data, 'name') AS v FROM docs WHERE id = 1",
    );
    expect(result!.rows[0]!["v"]).toBe("Alice");
    expect(typeof result!.rows[0]!["v"]).toBe("string");
  });

  it("cast to json", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, raw TEXT)");
    await e.sql("INSERT INTO t (id, raw) VALUES (1, '{\"x\": 42}')");
    const result = await e.sql(
      "SELECT CAST(raw AS json)->'x' AS v FROM t WHERE id = 1",
    );
    expect(result!.rows[0]!["v"]).toBe(42);
  });
});

// =============================================================================
// JSON object aggregation
// =============================================================================

describe("JSONObjectAgg", () => {
  it("basic", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT json_object_agg(name, val) AS v FROM t");
    const v = result!.rows[0]!["v"] as Record<string, unknown>;
    expect(typeof v).toBe("object");
    expect(v["alpha"]).toBe(10);
    expect(v["bravo"]).toBe(20);
    expect(v["charlie"]).toBe(30);
  });

  it("jsonb variant", async () => {
    const e = await makeTableEngine();
    const result = await e.sql("SELECT jsonb_object_agg(name, val) AS v FROM t");
    const v = result!.rows[0]!["v"] as Record<string, unknown>;
    expect(v["alpha"]).toBe(10);
  });
});

// =============================================================================
// JSON path operators
// =============================================================================

describe("JSONPathOperator", () => {
  it("hash gt", async () => {
    const e = await makeTableEngine();
    await e.sql("CREATE TABLE jdoc (id SERIAL PRIMARY KEY, data JSONB)");
    await e.sql('INSERT INTO jdoc (data) VALUES (\'{"a": {"b": 42}}\'::jsonb)');
    const result = await e.sql("SELECT data #> '{a,b}' AS v FROM jdoc WHERE id = 1");
    expect(result!.rows[0]!["v"]).toBe(42);
  });

  it("hash gt gt", async () => {
    const e = await makeTableEngine();
    await e.sql("CREATE TABLE jd2 (id SERIAL PRIMARY KEY, data JSONB)");
    await e.sql('INSERT INTO jd2 (data) VALUES (\'{"a": {"b": 42}}\'::jsonb)');
    const result = await e.sql("SELECT data #>> '{a,b}' AS v FROM jd2 WHERE id = 1");
    expect(result!.rows[0]!["v"]).toBe("42");
  });
});

// =============================================================================
// JSON containment
// =============================================================================

describe("JSONContainment", () => {
  it("contains", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      'SELECT \'{"a": 1, "b": 2}\'::jsonb @> \'{"a": 1}\'::jsonb AS v ' +
        "FROM t WHERE id = 1",
    );
    expect(result!.rows[0]!["v"]).toBe(true);
  });

  it("not contains", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      "SELECT '{\"a\": 1}'::jsonb @> '{\"a\": 2}'::jsonb AS v FROM t WHERE id = 1",
    );
    expect(result!.rows[0]!["v"]).toBe(false);
  });

  it("contained by", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      'SELECT \'{"a": 1}\'::jsonb <@ \'{"a": 1, "b": 2}\'::jsonb AS v ' +
        "FROM t WHERE id = 1",
    );
    expect(result!.rows[0]!["v"]).toBe(true);
  });
});

// =============================================================================
// JSONB set
// =============================================================================

describe("JSONBSet", () => {
  it("basic", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      "SELECT jsonb_set('{\"a\": 1}'::jsonb, '{b}', '2'::jsonb) AS v " +
        "FROM t WHERE id = 1",
    );
    const v = result!.rows[0]!["v"] as Record<string, unknown>;
    expect(v["a"]).toBe(1);
    expect(v["b"]).toBe(2);
  });

  it("replace", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      "SELECT jsonb_set('{\"a\": 1}'::jsonb, '{a}', '99'::jsonb) AS v " +
        "FROM t WHERE id = 1",
    );
    expect((result!.rows[0]!["v"] as Record<string, unknown>)["a"]).toBe(99);
  });
});

// =============================================================================
// JSON object keys
// =============================================================================

describe("JSONObjectKeys", () => {
  it("basic", async () => {
    const e = await makeTableEngine();
    const result = await e.sql(
      'SELECT json_object_keys(\'{"a": 1, "b": 2}\'::json) AS v ' +
        "FROM t WHERE id = 1",
    );
    const v = result!.rows[0]!["v"] as string[];
    expect(new Set(v)).toEqual(new Set(["a", "b"]));
  });
});

// =============================================================================
// JSON key existence
// =============================================================================

describe("JSONKeyExistence", () => {
  async function makeKeyEngine(): Promise<Engine> {
    const e = new Engine();
    await e.sql("CREATE TABLE t (id INT PRIMARY KEY)");
    await e.sql("INSERT INTO t VALUES (1)");
    return e;
  }

  it("has key present", async () => {
    const e = await makeKeyEngine();
    const r = await e.sql(
      'SELECT \'{"a": 1, "b": 2, "c": 3}\'::jsonb ? \'a\' AS v ' +
        "FROM t WHERE id = 1",
    );
    expect(r!.rows[0]!["v"]).toBe(true);
  });

  it("has key missing", async () => {
    const e = await makeKeyEngine();
    const r = await e.sql(
      "SELECT '{\"a\": 1, \"b\": 2}'::jsonb ? 'z' AS v FROM t WHERE id = 1",
    );
    expect(r!.rows[0]!["v"]).toBe(false);
  });

  it("has any key match", async () => {
    const e = await makeKeyEngine();
    const r = await e.sql(
      'SELECT \'{"a": 1, "b": 2, "c": 3}\'::jsonb ' +
        "?| ARRAY['a', 'z'] AS v " +
        "FROM t WHERE id = 1",
    );
    expect(r!.rows[0]!["v"]).toBe(true);
  });

  it("has any key no match", async () => {
    const e = await makeKeyEngine();
    const r = await e.sql(
      "SELECT '{\"a\": 1}'::jsonb ?| ARRAY['x', 'y'] AS v FROM t WHERE id = 1",
    );
    expect(r!.rows[0]!["v"]).toBe(false);
  });

  it("has all keys present", async () => {
    const e = await makeKeyEngine();
    const r = await e.sql(
      'SELECT \'{"a": 1, "b": 2, "c": 3}\'::jsonb ' +
        "?& ARRAY['a', 'b'] AS v " +
        "FROM t WHERE id = 1",
    );
    expect(r!.rows[0]!["v"]).toBe(true);
  });

  it("has all keys missing one", async () => {
    const e = await makeKeyEngine();
    const r = await e.sql(
      'SELECT \'{"a": 1, "b": 2}\'::jsonb ' +
        "?& ARRAY['a', 'z'] AS v " +
        "FROM t WHERE id = 1",
    );
    expect(r!.rows[0]!["v"]).toBe(false);
  });

  it("has key on empty object", async () => {
    const e = await makeKeyEngine();
    const r = await e.sql("SELECT '{}'::jsonb ? 'a' AS v FROM t WHERE id = 1");
    expect(r!.rows[0]!["v"]).toBe(false);
  });

  it("has all keys on single key", async () => {
    const e = await makeKeyEngine();
    const r = await e.sql(
      "SELECT '{\"x\": 10}'::jsonb ?& ARRAY['x'] AS v FROM t WHERE id = 1",
    );
    expect(r!.rows[0]!["v"]).toBe(true);
  });
});

// =============================================================================
// JSON each
// =============================================================================

describe("JSONEach", () => {
  it("json each", async () => {
    const e = new Engine();
    const r = await e.sql('SELECT * FROM json_each(\'{"a": 1, "b": 2}\')');
    expect(r!.rows.length).toBe(2);
    const keys = new Set(r!.rows.map((row) => row["key"]));
    expect(keys).toEqual(new Set(["a", "b"]));
  });

  it("json each key value pairs", async () => {
    const e = new Engine();
    const r = await e.sql('SELECT key, value FROM json_each(\'{"x": 10, "y": 20}\')');
    expect(r!.rows.length).toBe(2);
    const kv: Record<string, unknown> = {};
    for (const row of r!.rows) {
      kv[row["key"] as string] = row["value"];
    }
    expect(Number(kv["x"])).toBe(10);
    expect(Number(kv["y"])).toBe(20);
  });

  it("json each text", async () => {
    const e = new Engine();
    const r = await e.sql(
      'SELECT * FROM json_each_text(\'{"name": "Alice", "age": "30"}\')',
    );
    expect(r!.rows.length).toBe(2);
    for (const row of r!.rows) {
      expect(typeof row["value"]).toBe("string");
    }
  });

  it("json each text values", async () => {
    const e = new Engine();
    const r = await e.sql(
      'SELECT key, value FROM json_each_text(\'{"k1": "v1", "k2": "v2"}\')',
    );
    const kv: Record<string, unknown> = {};
    for (const row of r!.rows) {
      kv[row["key"] as string] = row["value"];
    }
    expect(kv["k1"]).toBe("v1");
    expect(kv["k2"]).toBe("v2");
  });

  it("jsonb each", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT * FROM jsonb_each('{\"p\": 100}')");
    expect(r!.rows.length).toBe(1);
    expect(r!.rows[0]!["key"]).toBe("p");
  });
});

// =============================================================================
// JSON array elements
// =============================================================================

describe("JSONArrayElements", () => {
  it("basic", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT * FROM json_array_elements('[1, 2, 3]')");
    expect(r!.rows.length).toBe(3);
  });

  it("values", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT value FROM json_array_elements('[10, 20, 30]')");
    const values = r!.rows.map((row) => Number(row["value"]));
    expect(values).toContain(10);
    expect(values).toContain(20);
    expect(values).toContain(30);
  });

  it("text variant", async () => {
    const e = new Engine();
    const r = await e.sql(
      'SELECT * FROM json_array_elements_text(\'["a", "b", "c"]\')',
    );
    expect(r!.rows.length).toBe(3);
    const values = r!.rows.map((row) => row["value"]);
    expect(values).toContain("a");
    expect(values).toContain("b");
    expect(values).toContain("c");
  });

  it("jsonb array elements", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT * FROM jsonb_array_elements('[4, 5]')");
    expect(r!.rows.length).toBe(2);
  });

  it("single element array", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT * FROM json_array_elements('[42]')");
    expect(r!.rows.length).toBe(1);
    expect(Number(r!.rows[0]!["value"])).toBe(42);
  });
});

// =============================================================================
// JSON array elements table function
// =============================================================================

describe("JSONArrayElementsTableFunction", () => {
  it("array elements from literal", async () => {
    const e = new Engine();
    const r = await e.sql(
      'SELECT value FROM json_array_elements(\'["python", "sql", "rust"]\')',
    );
    expect(r!.rows.length).toBe(3);
    const values = r!.rows.map((row) => row["value"]);
    expect(values).toContain("python");
    expect(values).toContain("sql");
    expect(values).toContain("rust");
  });

  it("array elements integers", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT value FROM json_array_elements('[1, 2, 3]')");
    expect(r!.rows.length).toBe(3);
  });
});
