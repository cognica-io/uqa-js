import { describe, expect, it } from "vitest";
import {
  Between,
  createEdge,
  createPayload,
  createPostingEntry,
  createVertex,
  Equals,
  GreaterThan,
  GreaterThanOrEqual,
  ILike,
  IndexStats,
  InSet,
  IsNotNull,
  IsNull,
  isNullPredicate,
  LessThan,
  LessThanOrEqual,
  Like,
  NotEquals,
  NotILike,
  NotLike,
} from "../../src/core/types.js";

// -- Factory functions -------------------------------------------------------

describe("createPayload", () => {
  it("creates with defaults", () => {
    const p = createPayload();
    expect(p.positions).toEqual([]);
    expect(p.score).toBe(0.0);
    expect(p.fields).toEqual({});
  });

  it("creates with all fields", () => {
    const p = createPayload({
      positions: [1, 3, 5],
      score: 2.5,
      fields: { title: "hello" },
    });
    expect(p.positions).toEqual([1, 3, 5]);
    expect(p.score).toBe(2.5);
    expect(p.fields).toEqual({ title: "hello" });
  });
});

describe("createPostingEntry", () => {
  it("creates with docId and default payload", () => {
    const e = createPostingEntry(42);
    expect(e.docId).toBe(42);
    expect(e.payload.score).toBe(0.0);
  });

  it("creates with custom payload", () => {
    const e = createPostingEntry(1, { score: 3.14 });
    expect(e.payload.score).toBe(3.14);
  });
});

describe("createVertex", () => {
  it("creates vertex with defaults", () => {
    const v = createVertex(1, "Person");
    expect(v.vertexId).toBe(1);
    expect(v.label).toBe("Person");
    expect(v.properties).toEqual({});
  });

  it("creates vertex with properties", () => {
    const v = createVertex(1, "Person", { name: "Alice" });
    expect(v.properties).toEqual({ name: "Alice" });
  });
});

describe("createEdge", () => {
  it("creates edge with defaults", () => {
    const e = createEdge(1, 10, 20, "knows");
    expect(e.edgeId).toBe(1);
    expect(e.sourceId).toBe(10);
    expect(e.targetId).toBe(20);
    expect(e.label).toBe("knows");
    expect(e.properties).toEqual({});
  });
});

// -- IndexStats --------------------------------------------------------------

describe("IndexStats", () => {
  it("has correct defaults", () => {
    const stats = new IndexStats();
    expect(stats.totalDocs).toBe(0);
    expect(stats.avgDocLength).toBe(0.0);
    expect(stats.dimensions).toBe(0);
  });

  it("gets and sets doc freq", () => {
    const stats = new IndexStats();
    expect(stats.docFreq("title", "hello")).toBe(0);
    stats.setDocFreq("title", "hello", 42);
    expect(stats.docFreq("title", "hello")).toBe(42);
  });

  it("distinguishes field+term combinations", () => {
    const stats = new IndexStats();
    stats.setDocFreq("title", "hello", 10);
    stats.setDocFreq("body", "hello", 20);
    stats.setDocFreq("title", "world", 30);
    expect(stats.docFreq("title", "hello")).toBe(10);
    expect(stats.docFreq("body", "hello")).toBe(20);
    expect(stats.docFreq("title", "world")).toBe(30);
  });

  it("is mutable", () => {
    const stats = new IndexStats(100, 50.5, 128);
    expect(stats.totalDocs).toBe(100);
    stats.totalDocs = 200;
    expect(stats.totalDocs).toBe(200);
  });
});

// -- Predicates --------------------------------------------------------------

describe("Equals", () => {
  it("matches equal values", () => {
    expect(new Equals(42).evaluate(42)).toBe(true);
    expect(new Equals("hello").evaluate("hello")).toBe(true);
  });

  it("rejects non-equal values", () => {
    expect(new Equals(42).evaluate(43)).toBe(false);
  });
});

describe("NotEquals", () => {
  it("matches non-equal values", () => {
    expect(new NotEquals(42).evaluate(43)).toBe(true);
  });

  it("rejects equal values", () => {
    expect(new NotEquals(42).evaluate(42)).toBe(false);
  });
});

describe("GreaterThan", () => {
  it("matches greater values", () => {
    expect(new GreaterThan(5).evaluate(6)).toBe(true);
  });

  it("rejects equal or lesser values", () => {
    expect(new GreaterThan(5).evaluate(5)).toBe(false);
    expect(new GreaterThan(5).evaluate(4)).toBe(false);
  });
});

describe("GreaterThanOrEqual", () => {
  it("matches greater or equal values", () => {
    expect(new GreaterThanOrEqual(5).evaluate(5)).toBe(true);
    expect(new GreaterThanOrEqual(5).evaluate(6)).toBe(true);
  });

  it("rejects lesser values", () => {
    expect(new GreaterThanOrEqual(5).evaluate(4)).toBe(false);
  });
});

describe("LessThan", () => {
  it("matches lesser values", () => {
    expect(new LessThan(5).evaluate(4)).toBe(true);
  });

  it("rejects equal or greater values", () => {
    expect(new LessThan(5).evaluate(5)).toBe(false);
    expect(new LessThan(5).evaluate(6)).toBe(false);
  });
});

describe("LessThanOrEqual", () => {
  it("matches lesser or equal values", () => {
    expect(new LessThanOrEqual(5).evaluate(5)).toBe(true);
    expect(new LessThanOrEqual(5).evaluate(4)).toBe(true);
  });

  it("rejects greater values", () => {
    expect(new LessThanOrEqual(5).evaluate(6)).toBe(false);
  });
});

describe("InSet", () => {
  it("matches values in the set", () => {
    const pred = new InSet([1, 2, 3]);
    expect(pred.evaluate(1)).toBe(true);
    expect(pred.evaluate(3)).toBe(true);
  });

  it("rejects values not in the set", () => {
    const pred = new InSet([1, 2, 3]);
    expect(pred.evaluate(4)).toBe(false);
  });
});

describe("Between", () => {
  it("matches values in range (inclusive)", () => {
    const pred = new Between(1, 10);
    expect(pred.evaluate(1)).toBe(true);
    expect(pred.evaluate(5)).toBe(true);
    expect(pred.evaluate(10)).toBe(true);
  });

  it("rejects values outside range", () => {
    const pred = new Between(1, 10);
    expect(pred.evaluate(0)).toBe(false);
    expect(pred.evaluate(11)).toBe(false);
  });
});

describe("IsNull", () => {
  it("matches null", () => {
    expect(new IsNull().evaluate(null)).toBe(true);
  });

  it("matches undefined", () => {
    expect(new IsNull().evaluate(undefined)).toBe(true);
  });

  it("rejects non-null values", () => {
    expect(new IsNull().evaluate(0)).toBe(false);
    expect(new IsNull().evaluate("")).toBe(false);
  });
});

describe("IsNotNull", () => {
  it("matches non-null values", () => {
    expect(new IsNotNull().evaluate(0)).toBe(true);
    expect(new IsNotNull().evaluate("")).toBe(true);
  });

  it("rejects null and undefined", () => {
    expect(new IsNotNull().evaluate(null)).toBe(false);
    expect(new IsNotNull().evaluate(undefined)).toBe(false);
  });
});

// -- LIKE predicates ---------------------------------------------------------

describe("Like", () => {
  it("matches prefix pattern", () => {
    expect(new Like("hel%").evaluate("hello")).toBe(true);
    expect(new Like("hel%").evaluate("world")).toBe(false);
  });

  it("matches suffix pattern", () => {
    expect(new Like("%llo").evaluate("hello")).toBe(true);
  });

  it("matches contains pattern", () => {
    expect(new Like("%ell%").evaluate("hello")).toBe(true);
  });

  it("matches single character wildcard", () => {
    expect(new Like("h_llo").evaluate("hello")).toBe(true);
    expect(new Like("h_llo").evaluate("hallo")).toBe(true);
    expect(new Like("h_llo").evaluate("hillo")).toBe(true);
    expect(new Like("h_llo").evaluate("hoollo")).toBe(false);
  });

  it("matches exact string", () => {
    expect(new Like("hello").evaluate("hello")).toBe(true);
    expect(new Like("hello").evaluate("hello!")).toBe(false);
  });

  it("is case-sensitive", () => {
    expect(new Like("Hello").evaluate("hello")).toBe(false);
    expect(new Like("Hello").evaluate("Hello")).toBe(true);
  });

  it("handles escaped wildcards", () => {
    expect(new Like("100\\%").evaluate("100%")).toBe(true);
    expect(new Like("100\\%").evaluate("100abc")).toBe(false);
  });
});

describe("NotLike", () => {
  it("inverts Like", () => {
    expect(new NotLike("hel%").evaluate("hello")).toBe(false);
    expect(new NotLike("hel%").evaluate("world")).toBe(true);
  });
});

describe("ILike", () => {
  it("matches case-insensitively", () => {
    expect(new ILike("hello").evaluate("HELLO")).toBe(true);
    expect(new ILike("HELLO").evaluate("hello")).toBe(true);
    expect(new ILike("%ELLO").evaluate("hello")).toBe(true);
  });
});

describe("NotILike", () => {
  it("inverts ILike", () => {
    expect(new NotILike("hello").evaluate("HELLO")).toBe(false);
    expect(new NotILike("hello").evaluate("world")).toBe(true);
  });
});

// -- isNullPredicate ---------------------------------------------------------

describe("isNullPredicate", () => {
  it("returns true for IsNull and IsNotNull", () => {
    expect(isNullPredicate(new IsNull())).toBe(true);
    expect(isNullPredicate(new IsNotNull())).toBe(true);
  });

  it("returns false for other predicates", () => {
    expect(isNullPredicate(new Equals(1))).toBe(false);
    expect(isNullPredicate(new Like("x"))).toBe(false);
    expect(isNullPredicate(new Between(0, 10))).toBe(false);
  });
});
