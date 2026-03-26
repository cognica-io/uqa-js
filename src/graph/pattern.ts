//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Graph pattern types and RPQ parser
// 1:1 port of uqa/graph/pattern.py

import type { Vertex, Edge } from "../core/types.js";

// -- Pattern types ------------------------------------------------------------

export interface VertexPattern {
  readonly variable: string;
  readonly constraints: ((v: Vertex) => boolean)[];
}

export interface EdgePattern {
  readonly sourceVar: string;
  readonly targetVar: string;
  readonly label: string | null;
  readonly constraints: ((e: Edge) => boolean)[];
  readonly negated: boolean;
}

export interface GraphPattern {
  readonly vertexPatterns: VertexPattern[];
  readonly edgePatterns: EdgePattern[];
}

export function createVertexPattern(
  variable: string,
  constraints?: ((v: Vertex) => boolean)[],
): VertexPattern {
  return { variable, constraints: constraints ?? [] };
}

export function createEdgePattern(
  sourceVar: string,
  targetVar: string,
  opts?: Partial<{
    label: string | null;
    constraints: ((e: Edge) => boolean)[];
    negated: boolean;
  }>,
): EdgePattern {
  return {
    sourceVar,
    targetVar,
    label: opts?.label ?? null,
    constraints: opts?.constraints ?? [],
    negated: opts?.negated ?? false,
  };
}

export function createGraphPattern(
  vertexPatterns?: VertexPattern[],
  edgePatterns?: EdgePattern[],
): GraphPattern {
  return {
    vertexPatterns: vertexPatterns ?? [],
    edgePatterns: edgePatterns ?? [],
  };
}

// -- Regular Path Expression AST ----------------------------------------------

export abstract class RegularPathExpr {
  abstract equals(other: RegularPathExpr): boolean;
  abstract toString(): string;
}

export class Label extends RegularPathExpr {
  readonly name: string;

  constructor(name: string) {
    super();
    this.name = name;
  }

  equals(other: RegularPathExpr): boolean {
    return other instanceof Label && other.name === this.name;
  }

  toString(): string {
    return this.name;
  }
}

export class Concat extends RegularPathExpr {
  readonly left: RegularPathExpr;
  readonly right: RegularPathExpr;

  constructor(left: RegularPathExpr, right: RegularPathExpr) {
    super();
    this.left = left;
    this.right = right;
  }

  equals(other: RegularPathExpr): boolean {
    return (
      other instanceof Concat &&
      this.left.equals(other.left) &&
      this.right.equals(other.right)
    );
  }

  toString(): string {
    return `${this.left.toString()} / ${this.right.toString()}`;
  }
}

export class Alternation extends RegularPathExpr {
  readonly left: RegularPathExpr;
  readonly right: RegularPathExpr;

  constructor(left: RegularPathExpr, right: RegularPathExpr) {
    super();
    this.left = left;
    this.right = right;
  }

  equals(other: RegularPathExpr): boolean {
    return (
      other instanceof Alternation &&
      this.left.equals(other.left) &&
      this.right.equals(other.right)
    );
  }

  toString(): string {
    return `(${this.left.toString()} | ${this.right.toString()})`;
  }
}

export class KleeneStar extends RegularPathExpr {
  readonly inner: RegularPathExpr;

  constructor(inner: RegularPathExpr) {
    super();
    this.inner = inner;
  }

  equals(other: RegularPathExpr): boolean {
    return other instanceof KleeneStar && this.inner.equals(other.inner);
  }

  toString(): string {
    return `(${this.inner.toString()})*`;
  }
}

export class BoundedLabel extends RegularPathExpr {
  readonly name: string;
  readonly minCount: number;
  readonly maxCount: number;

  constructor(name: string, minCount: number, maxCount: number) {
    super();
    this.name = name;
    this.minCount = minCount;
    this.maxCount = maxCount;
  }

  equals(other: RegularPathExpr): boolean {
    return (
      other instanceof BoundedLabel &&
      other.name === this.name &&
      other.minCount === this.minCount &&
      other.maxCount === this.maxCount
    );
  }

  toString(): string {
    return `${this.name}{${String(this.minCount)},${String(this.maxCount)}}`;
  }
}

// -- RPQ tokenizer and parser -------------------------------------------------

interface Token {
  readonly type:
    | "LABEL"
    | "PIPE"
    | "SLASH"
    | "STAR"
    | "LPAREN"
    | "RPAREN"
    | "LBRACE"
    | "RBRACE"
    | "COMMA"
    | "NUMBER";
  readonly value: string;
}

function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < input.length) {
    const ch = input[i]!;
    if (ch === " " || ch === "\t" || ch === "\n" || ch === "\r") {
      i++;
      continue;
    }
    if (ch === "|") {
      tokens.push({ type: "PIPE", value: "|" });
      i++;
    } else if (ch === "/") {
      tokens.push({ type: "SLASH", value: "/" });
      i++;
    } else if (ch === "*") {
      tokens.push({ type: "STAR", value: "*" });
      i++;
    } else if (ch === "(") {
      tokens.push({ type: "LPAREN", value: "(" });
      i++;
    } else if (ch === ")") {
      tokens.push({ type: "RPAREN", value: ")" });
      i++;
    } else if (ch === "{") {
      tokens.push({ type: "LBRACE", value: "{" });
      i++;
    } else if (ch === "}") {
      tokens.push({ type: "RBRACE", value: "}" });
      i++;
    } else if (ch === ",") {
      tokens.push({ type: "COMMA", value: "," });
      i++;
    } else if (ch >= "0" && ch <= "9") {
      let num = "";
      while (i < input.length && input[i]! >= "0" && input[i]! <= "9") {
        num += input[i]!;
        i++;
      }
      tokens.push({ type: "NUMBER", value: num });
    } else {
      // Label: alphanumeric + underscore + hyphen
      let label = "";
      while (
        i < input.length &&
        input[i] !== " " &&
        input[i] !== "\t" &&
        input[i] !== "\n" &&
        input[i] !== "\r" &&
        input[i] !== "|" &&
        input[i] !== "/" &&
        input[i] !== "*" &&
        input[i] !== "(" &&
        input[i] !== ")" &&
        input[i] !== "{" &&
        input[i] !== "}" &&
        input[i] !== ","
      ) {
        label += input[i]!;
        i++;
      }
      if (label.length > 0) {
        tokens.push({ type: "LABEL", value: label });
      }
    }
  }
  return tokens;
}

// Recursive descent parser for RPQ expressions
// Grammar:
//   expr       -> concat (PIPE concat)*
//   concat     -> unary (SLASH unary)*
//   unary      -> primary (STAR | bounded)?
//   primary    -> LABEL | LPAREN expr RPAREN
//   bounded    -> LBRACE NUMBER COMMA NUMBER RBRACE

class RPQParser {
  private tokens: Token[];
  private pos: number;

  constructor(tokens: Token[]) {
    this.tokens = tokens;
    this.pos = 0;
  }

  private peek(): Token | null {
    if (this.pos < this.tokens.length) {
      return this.tokens[this.pos]!;
    }
    return null;
  }

  private consume(expectedType?: string): Token {
    const tok = this.peek();
    if (!tok) {
      throw new Error("RPQ parse error: unexpected end of input");
    }
    if (expectedType !== undefined && tok.type !== expectedType) {
      throw new Error(
        `RPQ parse error: expected ${expectedType}, got ${tok.type} ('${tok.value}')`,
      );
    }
    this.pos++;
    return tok;
  }

  parseExpr(): RegularPathExpr {
    let left = this.parseConcat();
    while (this.peek()?.type === "PIPE") {
      this.consume("PIPE");
      const right = this.parseConcat();
      left = new Alternation(left, right);
    }
    return left;
  }

  private parseConcat(): RegularPathExpr {
    let left = this.parseUnary();
    while (this.peek()?.type === "SLASH") {
      this.consume("SLASH");
      const right = this.parseUnary();
      left = new Concat(left, right);
    }
    return left;
  }

  private parseUnary(): RegularPathExpr {
    let node = this.parsePrimary();
    const next = this.peek();
    if (next?.type === "STAR") {
      this.consume("STAR");
      node = new KleeneStar(node);
    } else if (next?.type === "LBRACE") {
      // Bounded repetition: only valid on Label
      this.consume("LBRACE");
      const minTok = this.consume("NUMBER");
      this.consume("COMMA");
      const maxTok = this.consume("NUMBER");
      this.consume("RBRACE");
      if (node instanceof Label) {
        node = new BoundedLabel(
          node.name,
          parseInt(minTok.value, 10),
          parseInt(maxTok.value, 10),
        );
      } else {
        throw new Error("RPQ parse error: bounded repetition only supported on labels");
      }
    }
    return node;
  }

  private parsePrimary(): RegularPathExpr {
    const tok = this.peek();
    if (!tok) {
      throw new Error("RPQ parse error: unexpected end of input");
    }
    if (tok.type === "LPAREN") {
      this.consume("LPAREN");
      const expr = this.parseExpr();
      this.consume("RPAREN");
      return expr;
    }
    if (tok.type === "LABEL") {
      this.consume("LABEL");
      return new Label(tok.value);
    }
    throw new Error(`RPQ parse error: unexpected token ${tok.type} ('${tok.value}')`);
  }
}

export function parseRpq(exprStr: string): RegularPathExpr {
  const tokens = tokenize(exprStr);
  if (tokens.length === 0) {
    throw new Error("RPQ parse error: empty expression");
  }
  const parser = new RPQParser(tokens);
  const result = parser.parseExpr();
  return result;
}
