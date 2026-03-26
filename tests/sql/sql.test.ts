import { describe, expect, it } from "vitest";
import { Table, createColumnDef, resolveType } from "../../src/sql/table.js";
import { parseFtsQuery } from "../../src/sql/fts-query.js";
import { ExprEvaluator } from "../../src/sql/expr-evaluator.js";

// -- resolveType -------------------------------------------------------------

describe("resolveType", () => {
  it("resolves INTEGER", () => {
    expect(resolveType(["INTEGER"])).toEqual(["integer", "number"]);
  });

  it("resolves TEXT", () => {
    expect(resolveType(["TEXT"])).toEqual(["text", "string"]);
  });

  it("resolves BOOLEAN", () => {
    expect(resolveType(["BOOLEAN"])).toEqual(["boolean", "boolean"]);
  });

  it("resolves VARCHAR", () => {
    expect(resolveType(["VARCHAR"])).toEqual(["text", "string"]);
  });

  it("resolves FLOAT", () => {
    expect(resolveType(["FLOAT"])).toEqual(["float", "number"]);
  });
});

// -- Table -------------------------------------------------------------------

describe("Table", () => {
  it("creates and inserts", () => {
    const table = new Table("test", [
      createColumnDef("id", "INTEGER", { primaryKey: true, autoIncrement: true }),
      createColumnDef("name", "TEXT"),
      createColumnDef("year", "INTEGER"),
    ]);
    const [id1] = table.insert({ name: "Alice", year: 2020 });
    const [id2] = table.insert({ name: "Bob", year: 2021 });
    expect(id1).toBe(1);
    expect(id2).toBe(2);
    expect(table.rowCount).toBe(2);
  });

  it("enforces NOT NULL", () => {
    const table = new Table("test", [
      createColumnDef("id", "INTEGER", { primaryKey: true }),
      createColumnDef("name", "TEXT", { notNull: true }),
    ]);
    expect(() => {
      table.insert({ id: 1 });
    }).toThrow();
  });

  it("applies defaults", () => {
    const table = new Table("test", [
      createColumnDef("id", "INTEGER", { primaryKey: true }),
      createColumnDef("status", "TEXT", { defaultValue: "active" }),
    ]);
    table.insert({ id: 1 });
    const doc = table.documentStore.get(1);
    expect(doc).not.toBeNull();
    expect(doc!["status"]).toBe("active");
  });

  it("returns column names", () => {
    const table = new Table("t", [
      createColumnDef("a", "INTEGER"),
      createColumnDef("b", "TEXT"),
    ]);
    expect(table.columnNames).toEqual(["a", "b"]);
  });

  it("analyze produces stats", () => {
    const table = new Table("test", [
      createColumnDef("id", "INTEGER", { primaryKey: true, pythonType: "number" }),
      createColumnDef("val", "INTEGER", { pythonType: "number" }),
    ]);
    table.insert({ id: 1, val: 10 });
    table.insert({ id: 2, val: 20 });
    table.insert({ id: 3, val: 10 });
    const stats = table.analyze();
    const valStats = stats.get("val");
    expect(valStats).toBeDefined();
    expect(valStats!.rowCount).toBe(3);
    expect(valStats!.distinctCount).toBe(2);
    expect(valStats!.minValue).toBe(10);
    expect(valStats!.maxValue).toBe(20);
  });
});

// -- FTS Query Parser --------------------------------------------------------

describe("FTS Query Parser", () => {
  it("parses single term", () => {
    const node = parseFtsQuery("hello");
    expect(node.type).toBe("term");
    if (node.type === "term") expect(node.term).toBe("hello");
  });

  it("parses phrase", () => {
    const node = parseFtsQuery('"hello world"');
    expect(node.type).toBe("phrase");
    if (node.type === "phrase") expect(node.phrase).toBe("hello world");
  });

  it("parses AND", () => {
    const node = parseFtsQuery("hello AND world");
    expect(node.type).toBe("and");
  });

  it("parses OR", () => {
    const node = parseFtsQuery("hello OR world");
    expect(node.type).toBe("or");
  });

  it("parses NOT", () => {
    const node = parseFtsQuery("NOT hello");
    expect(node.type).toBe("not");
  });

  it("parses implicit AND", () => {
    const node = parseFtsQuery("hello world");
    expect(node.type).toBe("and");
  });

  it("parses field:value", () => {
    const node = parseFtsQuery("title:hello");
    expect(node.type).toBe("term");
    if (node.type === "term") {
      expect(node.field).toBe("title");
      expect(node.term).toBe("hello");
    }
  });

  it("parses vector literal", () => {
    const node = parseFtsQuery("[1.0, 2.0, 3.0]");
    expect(node.type).toBe("vector");
    if (node.type === "vector") {
      expect(node.values).toEqual([1.0, 2.0, 3.0]);
    }
  });

  it("parses parenthesized expressions", () => {
    const node = parseFtsQuery("(hello OR world) AND foo");
    expect(node.type).toBe("and");
  });
});

// -- ExprEvaluator -----------------------------------------------------------

describe("ExprEvaluator", () => {
  const evaluator = new ExprEvaluator();

  it("evaluates ColumnRef", () => {
    const node = { ColumnRef: { fields: [{ String: { sval: "name" } }] } };
    expect(evaluator.evaluate(node, { name: "Alice" })).toBe("Alice");
  });

  it("evaluates A_Const integer", () => {
    const node = { A_Const: { ival: 42 } };
    expect(evaluator.evaluate(node, {})).toBe(42);
  });

  it("evaluates A_Const string", () => {
    const node = { A_Const: { sval: "hello" } };
    expect(evaluator.evaluate(node, {})).toBe("hello");
  });

  it("evaluates addition", () => {
    const node = {
      A_Expr: {
        kind: 0,
        name: [{ String: { sval: "+" } }],
        lexpr: { A_Const: { ival: 3 } },
        rexpr: { A_Const: { ival: 4 } },
      },
    };
    expect(evaluator.evaluate(node, {})).toBe(7);
  });

  it("evaluates comparison", () => {
    const node = {
      A_Expr: {
        kind: 0,
        name: [{ String: { sval: ">" } }],
        lexpr: { ColumnRef: { fields: [{ String: { sval: "x" } }] } },
        rexpr: { A_Const: { ival: 5 } },
      },
    };
    expect(evaluator.evaluate(node, { x: 10 })).toBe(true);
    expect(evaluator.evaluate(node, { x: 3 })).toBe(false);
  });

  it("evaluates BoolExpr AND", () => {
    const node = {
      BoolExpr: {
        boolop: 0, // AND
        args: [{ A_Const: { boolval: true } }, { A_Const: { boolval: false } }],
      },
    };
    expect(evaluator.evaluate(node, {})).toBe(false);
  });

  it("evaluates NullTest IS NULL", () => {
    const node = {
      NullTest: {
        arg: { ColumnRef: { fields: [{ String: { sval: "x" } }] } },
        nulltesttype: 0, // IS NULL
      },
    };
    expect(evaluator.evaluate(node, { x: null })).toBe(true);
    expect(evaluator.evaluate(node, { x: 42 })).toBe(false);
  });

  it("evaluates FuncCall UPPER", () => {
    const node = {
      FuncCall: {
        funcname: [{ String: { sval: "upper" } }],
        args: [{ ColumnRef: { fields: [{ String: { sval: "name" } }] } }],
      },
    };
    expect(evaluator.evaluate(node, { name: "hello" })).toBe("HELLO");
  });

  it("evaluates with params", () => {
    const e = new ExprEvaluator({ params: [42, "hello"] });
    const node = { ParamRef: { number: 1 } };
    expect(e.evaluate(node, {})).toBe(42);
  });
});
