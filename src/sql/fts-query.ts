//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Full-Text Search query parser and compiler
// 1:1 port of uqa/sql/fts_query.py
//
// Grammar:
//   query      = or_expr
//   or_expr    = and_expr ( 'OR' and_expr )*
//   and_expr   = unary ( ('AND' | <implicit>) unary )*
//   unary      = 'NOT' unary | primary
//   primary    = '(' or_expr ')'
//              | TERM ':' PHRASE          -- field:"phrase"
//              | TERM ':' VECTOR          -- field:[0.1, 0.2]
//              | TERM ':' TERM            -- field:term
//              | PHRASE                   -- "phrase"
//              | TERM                     -- bare term

import type { Operator, ExecutionContext } from "../operators/base.js";
import {
  UnionOperator,
  IntersectOperator,
  ComplementOperator,
} from "../operators/boolean.js";
import { TermOperator, ScoreOperator } from "../operators/primitive.js";
import { PostingList } from "../core/posting-list.js";
import {
  BayesianBM25Scorer,
  createBayesianBM25Params,
} from "../scoring/bayesian-bm25.js";
import { CalibratedVectorOperator } from "../operators/calibrated-vector.js";
import { LogOddsFusionOperator } from "../operators/hybrid.js";

// -- Token types ----------------------------------------------------------------

export type FTSTokenType =
  | "TERM"
  | "PHRASE"
  | "VECTOR"
  | "AND"
  | "OR"
  | "NOT"
  | "LPAREN"
  | "RPAREN"
  | "COLON"
  | "EOF";

export interface FTSToken {
  readonly type: FTSTokenType;
  readonly value: string;
  readonly pos: number;
}

// -- Tokenizer ------------------------------------------------------------------

function isWhitespace(ch: string): boolean {
  return ch === " " || ch === "\t" || ch === "\n" || ch === "\r";
}

function isWordChar(ch: string): boolean {
  return (
    !isWhitespace(ch) &&
    ch !== "(" &&
    ch !== ")" &&
    ch !== ":" &&
    ch !== '"' &&
    ch !== "[" &&
    ch !== "]"
  );
}

const KEYWORD_MAP: Readonly<Record<string, FTSTokenType>> = {
  and: "AND",
  or: "OR",
  not: "NOT",
};

export function tokenizeFts(source: string): FTSToken[] {
  const tokens: FTSToken[] = [];
  let i = 0;
  const n = source.length;

  while (i < n) {
    const ch = source[i]!;

    // Skip whitespace
    if (isWhitespace(ch)) {
      i++;
      continue;
    }

    // Single-character tokens
    if (ch === "(") {
      tokens.push({ type: "LPAREN", value: "(", pos: i });
      i++;
      continue;
    }
    if (ch === ")") {
      tokens.push({ type: "RPAREN", value: ")", pos: i });
      i++;
      continue;
    }
    if (ch === ":") {
      tokens.push({ type: "COLON", value: ":", pos: i });
      i++;
      continue;
    }

    // Quoted phrase
    if (ch === '"') {
      const start = i;
      i++;
      let phrase = "";
      while (i < n && source[i] !== '"') {
        phrase += source[i]!;
        i++;
      }
      if (i >= n) {
        throw new Error(
          `Unterminated quoted phrase starting at position ${String(start)}`,
        );
      }
      tokens.push({ type: "PHRASE", value: phrase, pos: start });
      i++; // skip closing quote
      continue;
    }

    // Vector literal [...]
    if (ch === "[") {
      const start = i;
      i++;
      let content = "";
      while (i < n && source[i] !== "]") {
        content += source[i]!;
        i++;
      }
      if (i >= n) {
        throw new Error(
          `Unterminated vector literal starting at position ${String(start)}`,
        );
      }
      tokens.push({ type: "VECTOR", value: content.trim(), pos: start });
      i++; // skip closing bracket
      continue;
    }

    // Bare word (term or keyword)
    if (isWordChar(ch)) {
      const start = i;
      let word = "";
      while (i < n && isWordChar(source[i]!)) {
        word += source[i]!;
        i++;
      }
      const kw = KEYWORD_MAP[word.toLowerCase()];
      if (kw !== undefined) {
        tokens.push({ type: kw, value: word, pos: start });
      } else {
        tokens.push({ type: "TERM", value: word, pos: start });
      }
      continue;
    }

    throw new Error(`Unexpected character '${ch}' at position ${String(i)}`);
  }

  tokens.push({ type: "EOF", value: "", pos: n });
  return tokens;
}

// -- AST nodes ------------------------------------------------------------------

export interface TermNode {
  readonly type: "term";
  readonly field: string | null;
  readonly term: string;
}

export interface PhraseNode {
  readonly type: "phrase";
  readonly field: string | null;
  readonly phrase: string;
}

export interface VectorNode {
  readonly type: "vector";
  readonly field: string | null;
  readonly values: readonly number[];
}

export interface AndNode {
  readonly type: "and";
  readonly left: FTSNode;
  readonly right: FTSNode;
}

export interface OrNode {
  readonly type: "or";
  readonly left: FTSNode;
  readonly right: FTSNode;
}

export interface NotNode {
  readonly type: "not";
  readonly operand: FTSNode;
}

export type FTSNode = TermNode | PhraseNode | VectorNode | AndNode | OrNode | NotNode;

// -- Parser ---------------------------------------------------------------------

export class FTSParser {
  private _tokens: FTSToken[];
  private _pos: number;

  constructor(tokens: FTSToken[]) {
    this._tokens = tokens;
    this._pos = 0;
  }

  parse(): FTSNode {
    if (this._peek().type === "EOF") {
      throw new Error("Empty query");
    }
    const node = this._orExpr();
    if (this._peek().type !== "EOF") {
      const tok = this._peek();
      throw new Error(`Unexpected token '${tok.value}' at position ${String(tok.pos)}`);
    }
    return node;
  }

  private _peek(): FTSToken {
    return this._tokens[this._pos] ?? { type: "EOF" as const, value: "", pos: -1 };
  }

  private _advance(): FTSToken {
    const tok = this._peek();
    if (tok.type !== "EOF") {
      this._pos++;
    }
    return tok;
  }

  private _expect(type: FTSTokenType): FTSToken {
    const tok = this._advance();
    if (tok.type !== type) {
      throw new Error(
        `Expected ${type}, got ${tok.type} ('${tok.value}') at position ${String(tok.pos)}`,
      );
    }
    return tok;
  }

  private _orExpr(): FTSNode {
    let left = this._andExpr();
    while (this._peek().type === "OR") {
      this._advance(); // consume OR
      const right = this._andExpr();
      left = { type: "or", left, right };
    }
    return left;
  }

  private _andExpr(): FTSNode {
    let left = this._unary();
    for (;;) {
      const tok = this._peek();
      if (tok.type === "AND") {
        this._advance(); // consume AND
        const right = this._unary();
        left = { type: "and", left, right };
      } else if (
        tok.type === "TERM" ||
        tok.type === "PHRASE" ||
        tok.type === "VECTOR" ||
        tok.type === "LPAREN" ||
        tok.type === "NOT"
      ) {
        // Implicit AND
        const right = this._unary();
        left = { type: "and", left, right };
      } else {
        break;
      }
    }
    return left;
  }

  private _unary(): FTSNode {
    if (this._peek().type === "NOT") {
      this._advance(); // consume NOT
      const operand = this._unary();
      return { type: "not", operand };
    }
    return this._primary();
  }

  private _primary(): FTSNode {
    const tok = this._peek();

    if (tok.type === "LPAREN") {
      this._advance(); // consume (
      const node = this._orExpr();
      this._expect("RPAREN");
      return node;
    }

    if (tok.type === "PHRASE") {
      this._advance();
      return { type: "phrase", field: null, phrase: tok.value };
    }

    if (tok.type === "VECTOR") {
      this._advance();
      return { type: "vector", field: null, values: parseVectorValues(tok.value) };
    }

    if (tok.type === "TERM") {
      this._advance();
      // Lookahead for field:value
      if (this._peek().type === "COLON") {
        this._advance(); // consume :
        const nextTok = this._peek();
        if (nextTok.type === "PHRASE") {
          this._advance();
          return { type: "phrase", field: tok.value, phrase: nextTok.value };
        }
        if (nextTok.type === "VECTOR") {
          this._advance();
          return {
            type: "vector",
            field: tok.value,
            values: parseVectorValues(nextTok.value),
          };
        }
        if (nextTok.type === "TERM") {
          this._advance();
          return { type: "term", field: tok.value, term: nextTok.value };
        }
        throw new Error(
          `Expected term, phrase, or vector after ':', got ${nextTok.type} at position ${String(nextTok.pos)}`,
        );
      }
      return { type: "term", field: null, term: tok.value };
    }

    throw new Error(
      `Unexpected token ${tok.type} ('${tok.value}') at position ${String(tok.pos)}`,
    );
  }
}

function parseVectorValues(raw: string): readonly number[] {
  const content = raw.trim();
  if (content.length === 0) {
    throw new Error("Empty vector literal");
  }
  return content.split(",").map((s) => {
    const n = Number(s.trim());
    if (Number.isNaN(n)) {
      throw new Error(`Malformed vector literal: invalid number '${s.trim()}'`);
    }
    return n;
  });
}

// -- Convenience parse function -----------------------------------------------

export function parseFtsQuery(source: string): FTSNode {
  const tokens = tokenizeFts(source);
  const parser = new FTSParser(tokens);
  return parser.parse();
}

// -- AST-to-Operator Compiler -------------------------------------------------

interface CompilerLike {
  _makeTextSearchOp(
    field: string | null,
    query: string,
    ctx: ExecutionContext,
    opts?: { bayesian?: boolean },
  ): Operator;
}

export function compileFtsMatch(
  queryString: string,
  defaultField: string | null,
  ctx: ExecutionContext,
  compiler: CompilerLike,
): Operator {
  const tokens = tokenizeFts(queryString);
  const ast = new FTSParser(tokens).parse();
  return compileNode(ast, defaultField, ctx, compiler);
}

function compileNode(
  node: FTSNode,
  defaultField: string | null,
  ctx: ExecutionContext,
  compiler: CompilerLike,
): Operator {
  if (node.type === "term") {
    const field = resolveField(node.field, defaultField);
    return compiler._makeTextSearchOp(field, node.term, ctx, { bayesian: true });
  }

  if (node.type === "phrase") {
    return compilePhrase(node, defaultField, ctx);
  }

  if (node.type === "vector") {
    return compileVector(node, defaultField);
  }

  if (node.type === "and") {
    return compileAnd(node, defaultField, ctx, compiler);
  }

  if (node.type === "or") {
    const left = compileNode(node.left, defaultField, ctx, compiler);
    const right = compileNode(node.right, defaultField, ctx, compiler);
    return new UnionOperator([left, right]);
  }

  // node.type === "not" (exhaustive)
  const operand = compileNode(node.operand, defaultField, ctx, compiler);
  return new ComplementOperator(operand);
}

function compilePhrase(
  node: PhraseNode,
  defaultField: string | null,
  ctx: ExecutionContext,
): Operator {
  const field = resolveField(node.field, defaultField);
  const idx = ctx.invertedIndex;
  if (idx === null || idx === undefined) {
    return {
      execute: () => new PostingList(),
      costEstimate: () => 0.0,
      compose: (other: Operator) => other,
    } as unknown as Operator;
  }

  const analyzer = field ? idx.getSearchAnalyzer(field) : idx.analyzer;
  const terms = analyzer.analyze(node.phrase);
  if (terms.length === 0) {
    return {
      execute: () => new PostingList(),
      costEstimate: () => 0.0,
      compose: (other: Operator) => other,
    } as unknown as Operator;
  }

  const termOps: Operator[] = terms.map((t: string) => new TermOperator(t, field));
  const retrieval = termOps.length === 1 ? termOps[0]! : new IntersectOperator(termOps);

  const scorer = new BayesianBM25Scorer(createBayesianBM25Params(), idx.stats);
  return new ScoreOperator(scorer, retrieval, terms, field);
}

function compileVector(node: VectorNode, defaultField: string | null): Operator {
  const field = node.field ?? defaultField ?? "embedding";
  const queryVec = new Float64Array(node.values);

  return new CalibratedVectorOperator(queryVec, 10000, field);
}

function compileAnd(
  node: AndNode,
  defaultField: string | null,
  ctx: ExecutionContext,
  compiler: CompilerLike,
): Operator {
  const leftOp = compileNode(node.left, defaultField, ctx, compiler);
  const rightOp = compileNode(node.right, defaultField, ctx, compiler);

  if (hasVectorSignal(node.left) !== hasVectorSignal(node.right)) {
    // Mixed text + vector: use log-odds fusion for calibrated combination
    return new LogOddsFusionOperator([leftOp, rightOp]);
  }

  // Same-kind AND: use intersection
  return new IntersectOperator([leftOp, rightOp]);
}

function hasVectorSignal(node: FTSNode): boolean {
  if (node.type === "vector") return true;
  if (node.type === "term" || node.type === "phrase") return false;
  if (node.type === "and") {
    return hasVectorSignal(node.left) || hasVectorSignal(node.right);
  }
  if (node.type === "or") {
    return hasVectorSignal(node.left) || hasVectorSignal(node.right);
  }
  // node.type === "not" (exhaustive)
  return hasVectorSignal(node.operand);
}

function resolveField(
  nodeField: string | null,
  defaultField: string | null,
): string | null {
  const field = nodeField !== null ? nodeField : defaultField;
  if (field === "_all") return null;
  return field;
}
