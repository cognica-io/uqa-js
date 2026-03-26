import { describe, expect, it } from "vitest";
import { IndexStats, Equals } from "../../src/core/types.js";
import { IntersectOperator, UnionOperator } from "../../src/operators/boolean.js";
import { TermOperator, FilterOperator } from "../../src/operators/primitive.js";
import { QueryOptimizer } from "../../src/planner/optimizer.js";

function makeOptimizer(): QueryOptimizer {
  return new QueryOptimizer(new IndexStats(100));
}

// ==================================================================
// Idempotent intersection
// ==================================================================

describe("IdempotentIntersection", () => {
  it("duplicate operand removed", () => {
    const a = new TermOperator("hello", "text");
    const op = new IntersectOperator([a, a]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBe(a);
  });

  it("triple duplicate reduced", () => {
    const a = new TermOperator("hello", "text");
    const op = new IntersectOperator([a, a, a]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBe(a);
  });

  it("distinct operands preserved", () => {
    const a = new TermOperator("hello", "text");
    const b = new TermOperator("world", "text");
    const op = new IntersectOperator([a, b]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBeInstanceOf(IntersectOperator);
    expect((result as IntersectOperator).operands.length).toBe(2);
  });

  it("partial duplicate", () => {
    const a = new TermOperator("hello", "text");
    const b = new TermOperator("world", "text");
    const op = new IntersectOperator([a, b, a]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBeInstanceOf(IntersectOperator);
    expect((result as IntersectOperator).operands.length).toBe(2);
  });
});

// ==================================================================
// Idempotent union
// ==================================================================

describe("IdempotentUnion", () => {
  it("duplicate operand removed", () => {
    const a = new TermOperator("hello", "text");
    const op = new UnionOperator([a, a]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBe(a);
  });

  it("triple duplicate reduced", () => {
    const a = new TermOperator("hello", "text");
    const op = new UnionOperator([a, a, a]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBe(a);
  });

  it("distinct operands preserved", () => {
    const a = new TermOperator("hello", "text");
    const b = new TermOperator("world", "text");
    const op = new UnionOperator([a, b]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBeInstanceOf(UnionOperator);
    expect((result as UnionOperator).operands.length).toBe(2);
  });

  it("partial duplicate", () => {
    const a = new TermOperator("hello", "text");
    const b = new TermOperator("world", "text");
    const op = new UnionOperator([a, b, a]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBeInstanceOf(UnionOperator);
    expect((result as UnionOperator).operands.length).toBe(2);
  });
});

// ==================================================================
// Absorption law
// ==================================================================

describe("Absorption", () => {
  it("union absorbs intersect: Union(A, Intersect(A, B)) => A", () => {
    const a = new TermOperator("hello", "text");
    const b = new TermOperator("world", "text");
    const inner = new IntersectOperator([a, b]);
    const op = new UnionOperator([a, inner]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBe(a);
  });

  it("union absorbs intersect reversed: Union(Intersect(A, B), A) => A", () => {
    const a = new TermOperator("hello", "text");
    const b = new TermOperator("world", "text");
    const inner = new IntersectOperator([a, b]);
    const op = new UnionOperator([inner, a]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBe(a);
  });

  it("intersect absorbs union: Intersect(A, Union(A, B)) => A", () => {
    const a = new TermOperator("hello", "text");
    const b = new TermOperator("world", "text");
    const inner = new UnionOperator([a, b]);
    const op = new IntersectOperator([a, inner]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBe(a);
  });

  it("intersect absorbs union reversed: Intersect(Union(A, B), A) => A", () => {
    const a = new TermOperator("hello", "text");
    const b = new TermOperator("world", "text");
    const inner = new UnionOperator([a, b]);
    const op = new IntersectOperator([inner, a]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBe(a);
  });

  it("no absorption without identity: Union(A, Intersect(C, B)) stays when C is not A", () => {
    const a = new TermOperator("hello", "text");
    const b = new TermOperator("world", "text");
    const c = new TermOperator("hello", "text"); // Same term but different object
    const inner = new IntersectOperator([c, b]);
    const op = new UnionOperator([a, inner]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBeInstanceOf(UnionOperator);
    expect((result as UnionOperator).operands.length).toBe(2);
  });

  it("no absorption when no shared identity", () => {
    const a = new TermOperator("hello", "text");
    const b = new TermOperator("world", "text");
    const c = new TermOperator("foo", "text");
    const inner = new IntersectOperator([b, c]);
    const op = new UnionOperator([a, inner]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBeInstanceOf(UnionOperator);
    expect((result as UnionOperator).operands.length).toBe(2);
  });
});

// ==================================================================
// Empty elimination
// ==================================================================

describe("EmptyElimination", () => {
  it("intersect with empty yields empty", () => {
    const a = new TermOperator("hello", "text");
    const empty = new IntersectOperator([]);
    const op = new IntersectOperator([a, empty]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBeInstanceOf(IntersectOperator);
    expect((result as IntersectOperator).operands.length).toBe(0);
  });

  it("union with empty drops empty", () => {
    const a = new TermOperator("hello", "text");
    const empty = new UnionOperator([]);
    const op = new UnionOperator([a, empty]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBe(a);
  });

  it("union all empty", () => {
    const e1 = new UnionOperator([]);
    const e2 = new IntersectOperator([]);
    const op = new UnionOperator([e1, e2]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBeInstanceOf(UnionOperator);
    expect((result as UnionOperator).operands.length).toBe(0);
  });

  it("intersect with empty nested", () => {
    const a = new TermOperator("hello", "text");
    const b = new TermOperator("world", "text");
    const unionAB = new UnionOperator([a, b]);
    const empty = new IntersectOperator([]);
    const op = new IntersectOperator([unionAB, empty]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBeInstanceOf(IntersectOperator);
    expect((result as IntersectOperator).operands.length).toBe(0);
  });

  it("intersect with empty preserves semantics", () => {
    const a = new TermOperator("hello", "text");
    const b = new TermOperator("world", "text");
    const empty = new IntersectOperator([]);
    const op = new IntersectOperator([a, b, empty]);
    const result = makeOptimizer().optimize(op);
    expect(result).toBeInstanceOf(IntersectOperator);
    expect((result as IntersectOperator).operands.length).toBe(0);
  });
});

// ==================================================================
// Nested simplification
// ==================================================================

describe("NestedSimplification", () => {
  it("nested idempotent", () => {
    const a = new TermOperator("hello", "text");
    const b = new TermOperator("world", "text");
    const inner = new IntersectOperator([a, a]); // should simplify to a
    const outer = new UnionOperator([inner, b]);
    const result = makeOptimizer().optimize(outer);
    expect(result).toBeInstanceOf(UnionOperator);
    expect((result as UnionOperator).operands.length).toBe(2);
  });

  it("simplification through filter", () => {
    const a = new TermOperator("hello", "text");
    const inner = new UnionOperator([a, a]);
    const filtered = new FilterOperator("text", new Equals("hello"), inner);
    const result = makeOptimizer().optimize(filtered);
    expect(result).toBeInstanceOf(FilterOperator);
  });
});
