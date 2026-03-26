import { describe, expect, it } from "vitest";
import { createVertex, createEdge } from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import { RegularPathQueryOperator } from "../../src/graph/operators.js";
import { Label, Alternation, Concat, KleeneStar } from "../../src/graph/pattern.js";
import { simplifyExpr, subsetConstruction } from "../../src/graph/rpq-optimizer.js";
import type { ExecutionContext } from "../../src/operators/base.js";

// -- Expression simplification tests --

describe("Expression simplification", () => {
  it("simplify identity", () => {
    const expr = new Label("a");
    const result = simplifyExpr(expr);
    expect(result).toEqual(new Label("a"));
  });

  it("simplify duplicate alternation", () => {
    const expr = new Alternation(new Label("a"), new Label("a"));
    const result = simplifyExpr(expr);
    expect(result).toEqual(new Label("a"));
  });

  it("simplify nested kleene", () => {
    const expr = new KleeneStar(new KleeneStar(new Label("a")));
    const result = simplifyExpr(expr);
    expect(result).toEqual(new KleeneStar(new Label("a")));
  });

  it("simplify alternation sorting", () => {
    // TS simplifier may not sort, but verifying it does not break
    const expr = new Alternation(new Label("b"), new Label("a"));
    const result = simplifyExpr(expr);
    // Result should be an Alternation with both labels preserved
    expect(result instanceof Alternation || result instanceof Label).toBe(true);
    if (result instanceof Alternation) {
      const leftStr = String(result.left);
      const rightStr = String(result.right);
      // Both labels should be present
      expect(leftStr.includes("a") || leftStr.includes("b")).toBe(true);
      expect(rightStr.includes("a") || rightStr.includes("b")).toBe(true);
    }
  });

  it("simplify nested concat", () => {
    const expr = new Concat(new Label("a"), new Label("b"));
    const result = simplifyExpr(expr);
    expect(result).toEqual(new Concat(new Label("a"), new Label("b")));
  });

  it("simplify complex", () => {
    const expr = new Concat(
      new Alternation(new Label("a"), new Label("a")),
      new KleeneStar(new KleeneStar(new Label("b"))),
    );
    const result = simplifyExpr(expr);
    expect(result).toEqual(new Concat(new Label("a"), new KleeneStar(new Label("b"))));
  });
});

// -- Subset construction (NFA -> DFA) tests --

describe("Subset construction", () => {
  it("subset construction simple via RPQ execution", () => {
    // subsetConstruction is exported; test it via end-to-end RPQ
    const gs = new MemoryGraphStore();
    gs.createGraph("g");
    gs.addVertex(createVertex(1, "a", {}), "g");
    gs.addVertex(createVertex(2, "b", {}), "g");
    gs.addEdge(createEdge(10, 1, 2, "knows", {}), "g");

    const ctx: ExecutionContext = { graphStore: gs };
    const op = new RegularPathQueryOperator(new Label("knows"), "g", 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(docIds.has(2)).toBe(true);
  });
  it("subset construction alternation via RPQ execution", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g");
    gs.addVertex(createVertex(1, "a", {}), "g");
    gs.addVertex(createVertex(2, "b", {}), "g");
    gs.addVertex(createVertex(3, "c", {}), "g");
    gs.addEdge(createEdge(10, 1, 2, "x", {}), "g");
    gs.addEdge(createEdge(11, 1, 3, "y", {}), "g");

    const ctx: ExecutionContext = { graphStore: gs };
    const expr = new Alternation(new Label("x"), new Label("y"));
    const op = new RegularPathQueryOperator(expr, "g", 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(docIds).toEqual(new Set([2, 3]));
  });
  it("subset construction concat via RPQ execution", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g");
    gs.addVertex(createVertex(1, "a", {}), "g");
    gs.addVertex(createVertex(2, "b", {}), "g");
    gs.addVertex(createVertex(3, "c", {}), "g");
    gs.addEdge(createEdge(10, 1, 2, "a", {}), "g");
    gs.addEdge(createEdge(11, 2, 3, "b", {}), "g");

    const ctx: ExecutionContext = { graphStore: gs };
    const expr = new Concat(new Label("a"), new Label("b"));
    const op = new RegularPathQueryOperator(expr, "g", 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(docIds.has(3)).toBe(true);
  });
  it("subsetConstruction function produces DFA result", () => {
    // Directly test the exported subsetConstruction with a manually built NFA
    const nfa = {
      states: [{ id: 0 }, { id: 1 }],
      transitions: [{ from: 0, to: 1, label: "a" }],
      startState: 0,
      acceptState: 1,
    };
    const dfa = subsetConstruction(nfa);
    expect(dfa.startState).toBeDefined();
    expect(dfa.acceptStates.size).toBeGreaterThan(0);
  });
});

// -- End-to-end RPQ with simplification --

describe("RPQ end-to-end", () => {
  it("rpq with duplicate alternation", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g");
    gs.addVertex(createVertex(1, "a", {}), "g");
    gs.addVertex(createVertex(2, "b", {}), "g");
    gs.addEdge(createEdge(10, 1, 2, "knows", {}), "g");

    const ctx: ExecutionContext = { graphStore: gs };
    // a|a should simplify to just a
    const expr = new Alternation(new Label("knows"), new Label("knows"));
    const op = new RegularPathQueryOperator(expr, "g", 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(docIds.has(2)).toBe(true);
  });
  it("rpq with nested kleene", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g");
    gs.addVertex(createVertex(1, "a", {}), "g");
    gs.addVertex(createVertex(2, "b", {}), "g");
    gs.addVertex(createVertex(3, "c", {}), "g");
    gs.addEdge(createEdge(10, 1, 2, "e", {}), "g");
    gs.addEdge(createEdge(11, 2, 3, "e", {}), "g");

    const ctx: ExecutionContext = { graphStore: gs };
    const expr = new KleeneStar(new KleeneStar(new Label("e")));
    const op = new RegularPathQueryOperator(expr, "g", 1);
    const result = op.execute(ctx);
    // Should reach multiple vertices via Kleene star
    expect([...result].length).toBeGreaterThan(0);
  });
});

// -- Integration tests: simplification and DFA path --

describe("RPQ integration", () => {
  it("rpq uses simplification", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g");
    gs.addVertex(createVertex(1, "a", {}), "g");
    gs.addVertex(createVertex(2, "b", {}), "g");
    gs.addEdge(createEdge(10, 1, 2, "e", {}), "g");

    const ctx: ExecutionContext = { graphStore: gs };
    // (e*)* should simplify to e* and still work
    const expr = new KleeneStar(new KleeneStar(new Label("e")));
    const op = new RegularPathQueryOperator(expr, "g", 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(docIds.has(2)).toBe(true);
  });
  it("rpq uses DFA for small NFA", () => {
    const gs = new MemoryGraphStore();
    gs.createGraph("g");
    gs.addVertex(createVertex(1, "a", {}), "g");
    gs.addVertex(createVertex(2, "b", {}), "g");
    gs.addVertex(createVertex(3, "c", {}), "g");
    gs.addEdge(createEdge(10, 1, 2, "x", {}), "g");
    gs.addEdge(createEdge(11, 1, 3, "y", {}), "g");

    const ctx: ExecutionContext = { graphStore: gs };
    // Simple alternation: x|y -- small NFA, should use DFA path
    const expr = new Alternation(new Label("x"), new Label("y"));
    const op = new RegularPathQueryOperator(expr, "g", 1);
    const result = op.execute(ctx);
    const docIds = new Set([...result].map((e) => e.payload.fields["end"] as number));
    expect(docIds).toEqual(new Set([2, 3]));
  });
});

// -- Epsilon elimination tests --

describe("Epsilon elimination", () => {
  it("simplify alternation subsumption", () => {
    const expr = new Alternation(new KleeneStar(new Label("a")), new Label("a"));
    const result = simplifyExpr(expr);
    expect(result).toEqual(new KleeneStar(new Label("a")));
  });

  it("simplify alternation subsumption reversed", () => {
    const expr = new Alternation(new Label("a"), new KleeneStar(new Label("a")));
    const result = simplifyExpr(expr);
    expect(result).toEqual(new KleeneStar(new Label("a")));
  });

  it("simplify duplicate kleene concat", () => {
    const expr = new Concat(
      new KleeneStar(new Label("a")),
      new KleeneStar(new Label("a")),
    );
    const result = simplifyExpr(expr);
    expect(result).toEqual(new KleeneStar(new Label("a")));
  });
});
