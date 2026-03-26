//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Cypher parser
// 1:1 port of uqa/graph/cypher/parser.py

import type { Token } from "./lexer.js";
import { TokenType, tokenize } from "./lexer.js";
import type {
  CypherClause,
  CypherExpr,
  CypherQuery,
  MatchClause,
  NodePattern,
  OrderByItem,
  PathPattern,
  RelationshipPattern,
  ReturnClause,
  ReturnItem,
  SetClause,
  SetItem,
  WithClause,
} from "./ast.js";
import {
  createBinaryOp,
  createCypherQuery,
  createFunctionCall,
  createIdentifier,
  createLiteral,
  createNodePattern,
  createParameter,
  createPathPattern,
  createPropertyAccess,
  createRelationshipPattern,
  createUnaryOp,
} from "./ast.js";

// -- CypherParser -------------------------------------------------------------

export class CypherParser {
  private tokens: Token[];
  private pos: number;

  constructor(input: string) {
    this.tokens = tokenize(input);
    this.pos = 0;
  }

  private peek(): Token {
    return this.tokens[this.pos] ?? { type: TokenType.EOF, value: "", position: -1 };
  }

  private advance(): Token {
    const tok = this.peek();
    if (tok.type !== TokenType.EOF) {
      this.pos++;
    }
    return tok;
  }

  private expect(type: TokenType): Token {
    const tok = this.peek();
    if (tok.type !== type) {
      throw new Error(
        `Cypher parse error at position ${String(tok.position)}: expected ${type}, got ${tok.type} ('${tok.value}')`,
      );
    }
    return this.advance();
  }

  private match(type: TokenType): boolean {
    if (this.peek().type === type) {
      this.advance();
      return true;
    }
    return false;
  }

  private atClauseStart(): boolean {
    const tok = this.peek();
    const clauseTokens = new Set([
      TokenType.MATCH,
      TokenType.OPTIONAL,
      TokenType.CREATE,
      TokenType.MERGE,
      TokenType.SET,
      TokenType.DELETE,
      TokenType.DETACH,
      TokenType.RETURN,
      TokenType.WITH,
      TokenType.UNWIND,
    ]);
    return clauseTokens.has(tok.type);
  }

  // -- Top-level --------------------------------------------------------------

  parse(): CypherQuery {
    const clauses: CypherClause[] = [];
    while (this.peek().type !== TokenType.EOF) {
      clauses.push(this.parseClause());
    }
    return createCypherQuery(clauses);
  }

  private parseClause(): CypherClause {
    const tok = this.peek();
    switch (tok.type) {
      case TokenType.MATCH:
        return this.parseMatchClause(false);
      case TokenType.OPTIONAL:
        this.advance();
        this.expect(TokenType.MATCH);
        return this.parseMatchClauseInner(true);
      case TokenType.RETURN:
        return this.parseReturnClause();
      case TokenType.WITH:
        return this.parseWithClause();
      case TokenType.CREATE:
        return this.parseCreateClause();
      case TokenType.DELETE:
        return this.parseDeleteClause(false);
      case TokenType.DETACH:
        this.advance();
        this.expect(TokenType.DELETE);
        return this.parseDeleteClauseInner(true);
      case TokenType.SET:
        return this.parseSetClause();
      case TokenType.REMOVE:
        return this.parseRemoveClause();
      case TokenType.MERGE:
        return this.parseMergeClause();
      case TokenType.UNWIND:
        return this.parseUnwindClause();
      case TokenType.CALL:
        return this.parseCallClause();
      default:
        throw new Error(
          `Cypher parse error at position ${String(tok.position)}: unexpected token ${tok.type} ('${tok.value}')`,
        );
    }
  }

  // -- MATCH ------------------------------------------------------------------

  private parseMatchClause(optional: boolean): MatchClause {
    this.expect(TokenType.MATCH);
    return this.parseMatchClauseInner(optional);
  }

  private parseMatchClauseInner(optional: boolean): MatchClause {
    const patterns = this.parsePatternList();
    let where: CypherExpr | null = null;
    if (this.peek().type === TokenType.WHERE) {
      this.advance();
      where = this.parseExpr();
    }
    return { kind: "match", patterns, optional, where };
  }

  // -- RETURN -----------------------------------------------------------------

  private parseReturnClause(): ReturnClause {
    this.expect(TokenType.RETURN);
    const distinct = this.match(TokenType.DISTINCT);
    const items = this.parseReturnItems();
    const orderBy = this.parseOptionalOrderBy();
    const skip = this.parseOptionalSkip();
    const limit = this.parseOptionalLimit();
    return { kind: "return", items, distinct, orderBy, skip, limit };
  }

  // -- WITH -------------------------------------------------------------------

  private parseWithClause(): WithClause {
    this.expect(TokenType.WITH);
    const distinct = this.match(TokenType.DISTINCT);
    const items = this.parseReturnItems();
    let where: CypherExpr | null = null;
    if (this.peek().type === TokenType.WHERE) {
      this.advance();
      where = this.parseExpr();
    }
    const orderBy = this.parseOptionalOrderBy();
    const skip = this.parseOptionalSkip();
    const limit = this.parseOptionalLimit();
    return { kind: "with", items, distinct, where, orderBy, skip, limit };
  }

  // -- CREATE -----------------------------------------------------------------

  private parseCreateClause(): CypherClause {
    this.expect(TokenType.CREATE);
    const patterns = this.parsePatternList();
    return { kind: "create", patterns };
  }

  // -- DELETE -----------------------------------------------------------------

  private parseDeleteClause(detach: boolean): CypherClause {
    this.expect(TokenType.DELETE);
    return this.parseDeleteClauseInner(detach);
  }

  private parseDeleteClauseInner(detach: boolean): CypherClause {
    const exprs: CypherExpr[] = [this.parseExpr()];
    while (this.match(TokenType.COMMA)) {
      exprs.push(this.parseExpr());
    }
    return { kind: "delete", exprs, detach };
  }

  // -- SET --------------------------------------------------------------------

  private parseSetClause(): CypherClause {
    this.expect(TokenType.SET);
    const items: SetItem[] = [];
    do {
      const obj = this.parseExpr();
      let operator = "=";
      if (this.peek().type === TokenType.PLUS_EQ) {
        this.advance();
        operator = "+=";
      } else {
        this.expect(TokenType.EQ);
      }
      const value = this.parseExpr();
      if (obj.kind !== "property_access") {
        throw new Error("SET requires property access on left side");
      }
      items.push({ property: obj, value, operator });
    } while (this.match(TokenType.COMMA));
    return { kind: "set", items };
  }

  // -- REMOVE -----------------------------------------------------------------

  private parseRemoveClause(): CypherClause {
    this.expect(TokenType.REMOVE);
    const items: CypherExpr[] = [this.parseExpr()];
    while (this.match(TokenType.COMMA)) {
      items.push(this.parseExpr());
    }
    return { kind: "remove", items };
  }

  // -- MERGE ------------------------------------------------------------------

  private parseMergeClause(): CypherClause {
    this.expect(TokenType.MERGE);
    const patterns = this.parsePatternList();
    const pattern = patterns[0]!;

    let onCreate: SetClause | null = null;
    let onMatch: SetClause | null = null;

    while (this.peek().type === TokenType.ON) {
      this.advance();
      const nextTok = this.peek();
      if (nextTok.type === TokenType.CREATE) {
        this.advance();
        this.expect(TokenType.SET);
        const items: SetItem[] = [];
        do {
          const obj = this.parseExpr();
          let op = "=";
          if (this.peek().type === TokenType.PLUS_EQ) {
            this.advance();
            op = "+=";
          } else {
            this.expect(TokenType.EQ);
          }
          const value = this.parseExpr();
          if (obj.kind !== "property_access") {
            throw new Error("ON CREATE SET requires property access");
          }
          items.push({ property: obj, value, operator: op });
        } while (this.match(TokenType.COMMA));
        onCreate = { kind: "set", items };
      } else if (nextTok.type === TokenType.MATCH) {
        this.advance();
        this.expect(TokenType.SET);
        const items: SetItem[] = [];
        do {
          const obj = this.parseExpr();
          let op = "=";
          if (this.peek().type === TokenType.PLUS_EQ) {
            this.advance();
            op = "+=";
          } else {
            this.expect(TokenType.EQ);
          }
          const value = this.parseExpr();
          if (obj.kind !== "property_access") {
            throw new Error("ON MATCH SET requires property access");
          }
          items.push({ property: obj, value, operator: op });
        } while (this.match(TokenType.COMMA));
        onMatch = { kind: "set", items };
      }
    }

    return {
      kind: "merge",
      pattern,
      onCreate,
      onMatch,
    };
  }

  // -- UNWIND -----------------------------------------------------------------

  private parseUnwindClause(): CypherClause {
    this.expect(TokenType.UNWIND);
    const expr = this.parseExpr();
    this.expect(TokenType.AS);
    const alias = this.expect(TokenType.IDENTIFIER).value;
    return { kind: "unwind", expr, alias };
  }

  // -- CALL -------------------------------------------------------------------

  private parseCallClause(): CypherClause {
    this.expect(TokenType.CALL);
    let procedure = this.expect(TokenType.IDENTIFIER).value;
    while (this.match(TokenType.DOT)) {
      procedure += "." + this.expect(TokenType.IDENTIFIER).value;
    }
    this.expect(TokenType.LPAREN);
    const args: CypherExpr[] = [];
    if (this.peek().type !== TokenType.RPAREN) {
      args.push(this.parseExpr());
      while (this.match(TokenType.COMMA)) {
        args.push(this.parseExpr());
      }
    }
    this.expect(TokenType.RPAREN);

    let yields: string[] | null = null;
    if (this.peek().type === TokenType.YIELD) {
      this.advance();
      yields = [this.expect(TokenType.IDENTIFIER).value];
      while (this.match(TokenType.COMMA)) {
        yields.push(this.expect(TokenType.IDENTIFIER).value);
      }
    }

    return { kind: "call", procedure, args, yields };
  }

  // -- Patterns ---------------------------------------------------------------

  private parsePatternList(): PathPattern[] {
    const patterns: PathPattern[] = [this.parsePathPattern()];
    while (this.match(TokenType.COMMA)) {
      patterns.push(this.parsePathPattern());
    }
    return patterns;
  }

  private parsePathPattern(): PathPattern {
    const elements: Array<NodePattern | RelationshipPattern> = [];
    elements.push(this.parseNodePattern());

    while (this.isRelationshipStart()) {
      elements.push(this.parseRelationshipPattern());
      elements.push(this.parseNodePattern());
    }

    return createPathPattern(elements);
  }

  private isRelationshipStart(): boolean {
    const t = this.peek().type;
    return (
      t === TokenType.DASH || t === TokenType.ARROW_LEFT || t === TokenType.ARROW_RIGHT
    );
  }

  private parseNodePattern(): NodePattern {
    this.expect(TokenType.LPAREN);
    let variable: string | null = null;
    const labels: string[] = [];
    const properties = new Map<string, CypherExpr>();

    if (this.peek().type === TokenType.IDENTIFIER) {
      variable = this.advance().value;
    }

    while (this.peek().type === TokenType.COLON) {
      this.advance();
      labels.push(this.expect(TokenType.IDENTIFIER).value);
    }

    if (this.peek().type === TokenType.LBRACE) {
      this.parsePropertyMap(properties);
    }

    this.expect(TokenType.RPAREN);
    return createNodePattern(variable, labels, properties);
  }

  private parseRelationshipPattern(): RelationshipPattern {
    let direction: "out" | "in" | "both" = "both";
    let leftArrow = false;

    if (this.peek().type === TokenType.ARROW_LEFT) {
      this.advance();
      leftArrow = true;
    } else {
      this.expect(TokenType.DASH);
    }

    let variable: string | null = null;
    const types: string[] = [];
    const properties = new Map<string, CypherExpr>();
    let minHops: number | null = null;
    let maxHops: number | null = null;

    if (this.peek().type === TokenType.LBRACKET) {
      this.advance();

      if (this.peek().type === TokenType.IDENTIFIER) {
        variable = this.advance().value;
      }

      while (this.peek().type === TokenType.COLON) {
        this.advance();
        types.push(this.expect(TokenType.IDENTIFIER).value);
        while (this.match(TokenType.PIPE)) {
          // Skip optional colon after pipe
          if (this.peek().type === TokenType.COLON) this.advance();
          types.push(this.expect(TokenType.IDENTIFIER).value);
        }
      }

      // Variable-length: *min..max
      if (this.peek().type === TokenType.STAR) {
        this.advance();
        if (this.peek().type === TokenType.INTEGER) {
          minHops = parseInt(this.advance().value, 10);
        }
        if (this.match(TokenType.DOTDOT)) {
          if (this.peek().type === TokenType.INTEGER) {
            maxHops = parseInt(this.advance().value, 10);
          }
        } else if (minHops !== null) {
          // *n means exactly n hops
          maxHops = minHops;
        }
      }

      if (this.peek().type === TokenType.LBRACE) {
        this.parsePropertyMap(properties);
      }

      this.expect(TokenType.RBRACKET);
    }

    if (this.peek().type === TokenType.ARROW_RIGHT) {
      this.advance();
      direction = leftArrow ? "both" : "out";
    } else {
      this.expect(TokenType.DASH);
      direction = leftArrow ? "in" : "both";
    }

    return createRelationshipPattern({
      variable,
      types,
      properties,
      direction,
      minHops,
      maxHops,
    });
  }

  private parsePropertyMap(properties: Map<string, CypherExpr>): void {
    this.expect(TokenType.LBRACE);
    if (this.peek().type !== TokenType.RBRACE) {
      const key = this.expect(TokenType.IDENTIFIER).value;
      this.expect(TokenType.COLON);
      const value = this.parseExpr();
      properties.set(key, value);
      while (this.match(TokenType.COMMA)) {
        const k = this.expect(TokenType.IDENTIFIER).value;
        this.expect(TokenType.COLON);
        const v = this.parseExpr();
        properties.set(k, v);
      }
    }
    this.expect(TokenType.RBRACE);
  }

  // -- Return items -----------------------------------------------------------

  private parseReturnItems(): ReturnItem[] {
    const items: ReturnItem[] = [this.parseReturnItem()];
    while (this.match(TokenType.COMMA)) {
      items.push(this.parseReturnItem());
    }
    return items;
  }

  private parseReturnItem(): ReturnItem {
    // Handle *
    if (this.peek().type === TokenType.STAR) {
      this.advance();
      return { expr: createIdentifier("*"), alias: null };
    }
    const expr = this.parseExpr();
    let alias: string | null = null;
    if (this.match(TokenType.AS)) {
      alias = this.expect(TokenType.IDENTIFIER).value;
    }
    return { expr, alias };
  }

  // -- ORDER BY / SKIP / LIMIT -----------------------------------------------

  private parseOptionalOrderBy(): OrderByItem[] | null {
    if (this.peek().type !== TokenType.ORDER) return null;
    this.advance();
    this.expect(TokenType.BY);
    const items: OrderByItem[] = [this.parseOrderByItem()];
    while (this.match(TokenType.COMMA)) {
      items.push(this.parseOrderByItem());
    }
    return items;
  }

  private parseOrderByItem(): OrderByItem {
    const expr = this.parseExpr();
    let ascending = true;
    if (this.peek().type === TokenType.DESC) {
      this.advance();
      ascending = false;
    } else if (this.peek().type === TokenType.ASC) {
      this.advance();
    }
    return { expr, ascending };
  }

  private parseOptionalSkip(): CypherExpr | null {
    if (this.peek().type !== TokenType.SKIP) return null;
    this.advance();
    return this.parseExpr();
  }

  private parseOptionalLimit(): CypherExpr | null {
    if (this.peek().type !== TokenType.LIMIT) return null;
    this.advance();
    return this.parseExpr();
  }

  // -- Expression parser (Pratt / precedence climbing) ------------------------

  private parseExpr(): CypherExpr {
    return this.parseOr();
  }

  private parseOr(): CypherExpr {
    let left = this.parseXor();
    while (this.peek().type === TokenType.OR) {
      this.advance();
      const right = this.parseXor();
      left = createBinaryOp("OR", left, right);
    }
    return left;
  }

  private parseXor(): CypherExpr {
    let left = this.parseAnd();
    while (this.peek().type === TokenType.XOR) {
      this.advance();
      const right = this.parseAnd();
      left = createBinaryOp("XOR", left, right);
    }
    return left;
  }

  private parseAnd(): CypherExpr {
    let left = this.parseNot();
    while (this.peek().type === TokenType.AND) {
      this.advance();
      const right = this.parseNot();
      left = createBinaryOp("AND", left, right);
    }
    return left;
  }

  private parseNot(): CypherExpr {
    if (this.peek().type === TokenType.NOT) {
      this.advance();
      const operand = this.parseNot();
      return createUnaryOp("NOT", operand);
    }
    return this.parseComparison();
  }

  private parseComparison(): CypherExpr {
    let left = this.parseAddSub();

    const compOps = new Set([
      TokenType.EQ,
      TokenType.NEQ,
      TokenType.LT,
      TokenType.GT,
      TokenType.LTE,
      TokenType.GTE,
      TokenType.STARTS_WITH,
      TokenType.ENDS_WITH,
      TokenType.CONTAINS,
    ]);

    while (compOps.has(this.peek().type)) {
      const opTok = this.advance();
      const right = this.parseAddSub();
      left = createBinaryOp(opTok.value.toUpperCase(), left, right);
    }

    // IS [NOT] NULL
    if (this.peek().type === TokenType.IS) {
      this.advance();
      const negated = this.match(TokenType.NOT);
      this.expect(TokenType.NULL);
      return { kind: "is_null", value: left, negated };
    }

    // IN
    if (this.peek().type === TokenType.IN) {
      this.advance();
      const list = this.parseAddSub();
      return { kind: "in", value: left, list };
    }

    return left;
  }

  private parseAddSub(): CypherExpr {
    let left = this.parseMulDiv();
    while (
      this.peek().type === TokenType.PLUS ||
      this.peek().type === TokenType.MINUS
    ) {
      const opTok = this.advance();
      const right = this.parseMulDiv();
      left = createBinaryOp(opTok.value, left, right);
    }
    return left;
  }

  private parseMulDiv(): CypherExpr {
    let left = this.parseUnaryExpr();
    while (
      this.peek().type === TokenType.STAR ||
      this.peek().type === TokenType.SLASH ||
      this.peek().type === TokenType.PERCENT
    ) {
      const opTok = this.advance();
      const right = this.parseUnaryExpr();
      left = createBinaryOp(opTok.value, left, right);
    }
    return left;
  }

  private parseUnaryExpr(): CypherExpr {
    if (this.peek().type === TokenType.MINUS) {
      this.advance();
      const operand = this.parsePrimary();
      return createUnaryOp("-", operand);
    }
    return this.parsePrimary();
  }

  private parsePrimary(): CypherExpr {
    const tok = this.peek();

    switch (tok.type) {
      case TokenType.INTEGER:
        this.advance();
        return createLiteral(parseInt(tok.value, 10));

      case TokenType.FLOAT:
        this.advance();
        return createLiteral(parseFloat(tok.value));

      case TokenType.STRING:
        this.advance();
        return createLiteral(tok.value);

      case TokenType.TRUE:
        this.advance();
        return createLiteral(true);

      case TokenType.FALSE:
        this.advance();
        return createLiteral(false);

      case TokenType.NULL:
        this.advance();
        return createLiteral(null);

      case TokenType.PARAMETER:
        this.advance();
        return createParameter(tok.value);

      case TokenType.LBRACKET: {
        // List literal
        this.advance();
        const elements: CypherExpr[] = [];
        if (this.peek().type !== TokenType.RBRACKET) {
          elements.push(this.parseExpr());
          while (this.match(TokenType.COMMA)) {
            elements.push(this.parseExpr());
          }
        }
        this.expect(TokenType.RBRACKET);
        return { kind: "list", elements };
      }

      case TokenType.CASE:
        return this.parseCaseExpr();

      case TokenType.EXISTS: {
        this.advance();
        this.expect(TokenType.LPAREN);
        const pattern = this.parseNodePattern();
        this.expect(TokenType.RPAREN);
        return { kind: "exists", pattern };
      }

      case TokenType.LPAREN: {
        this.advance();
        const inner = this.parseExpr();
        this.expect(TokenType.RPAREN);
        return inner;
      }

      case TokenType.IDENTIFIER: {
        this.advance();
        // Check for function call
        if (this.peek().type === TokenType.LPAREN) {
          this.advance();
          const args: CypherExpr[] = [];
          // Handle DISTINCT in aggregate functions
          const isDistinct = this.match(TokenType.DISTINCT);
          if (this.peek().type !== TokenType.RPAREN) {
            args.push(this.parseExpr());
            while (this.match(TokenType.COMMA)) {
              args.push(this.parseExpr());
            }
          }
          this.expect(TokenType.RPAREN);
          const fname = isDistinct ? tok.value + "_distinct" : tok.value;
          let result: CypherExpr = createFunctionCall(fname, args);
          // Property access and index access chain after function call
          while (
            this.peek().type === TokenType.DOT ||
            this.peek().type === TokenType.LBRACKET
          ) {
            if (this.peek().type === TokenType.DOT) {
              this.advance();
              const prop = this.expect(TokenType.IDENTIFIER).value;
              result = createPropertyAccess(result, prop);
            } else {
              this.advance();
              const idx = this.parseExpr();
              this.expect(TokenType.RBRACKET);
              result = { kind: "index", value: result, index: idx };
            }
          }
          return result;
        }

        // Property access and index access
        let result: CypherExpr = createIdentifier(tok.value);
        while (
          this.peek().type === TokenType.DOT ||
          this.peek().type === TokenType.LBRACKET
        ) {
          if (this.peek().type === TokenType.DOT) {
            this.advance();
            const prop = this.expect(TokenType.IDENTIFIER).value;
            result = createPropertyAccess(result, prop);
          } else {
            this.advance();
            const idx = this.parseExpr();
            this.expect(TokenType.RBRACKET);
            result = { kind: "index", value: result, index: idx };
          }
        }
        return result;
      }

      default:
        throw new Error(
          `Cypher parse error at position ${String(tok.position)}: unexpected token ${tok.type} ('${tok.value}')`,
        );
    }
  }

  private parseCaseExpr(): CypherExpr {
    this.expect(TokenType.CASE);
    let operand: CypherExpr | null = null;

    // Simple CASE vs searched CASE
    if (this.peek().type !== TokenType.WHEN) {
      operand = this.parseExpr();
    }

    const whens: Array<{ when: CypherExpr; then: CypherExpr }> = [];
    while (this.peek().type === TokenType.WHEN) {
      this.advance();
      const when = this.parseExpr();
      this.expect(TokenType.THEN);
      const then = this.parseExpr();
      whens.push({ when, then });
    }

    let elseExpr: CypherExpr | null = null;
    if (this.match(TokenType.ELSE)) {
      elseExpr = this.parseExpr();
    }

    this.expect(TokenType.END);
    return { kind: "case", operand, whens, elseExpr };
  }
}

// -- Convenience function -----------------------------------------------------

export function parseCypher(input: string): CypherQuery {
  const parser = new CypherParser(input);
  return parser.parse();
}
