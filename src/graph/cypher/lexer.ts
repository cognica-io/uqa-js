//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Cypher lexer
// 1:1 port of uqa/graph/cypher/lexer.py

// -- TokenType ----------------------------------------------------------------

export enum TokenType {
  // Keywords
  MATCH = "MATCH",
  OPTIONAL = "OPTIONAL",
  WHERE = "WHERE",
  RETURN = "RETURN",
  WITH = "WITH",
  CREATE = "CREATE",
  DELETE = "DELETE",
  DETACH = "DETACH",
  SET = "SET",
  REMOVE = "REMOVE",
  MERGE = "MERGE",
  ON = "ON",
  ORDER = "ORDER",
  BY = "BY",
  ASC = "ASC",
  DESC = "DESC",
  SKIP = "SKIP",
  LIMIT = "LIMIT",
  DISTINCT = "DISTINCT",
  AS = "AS",
  AND = "AND",
  OR = "OR",
  XOR = "XOR",
  NOT = "NOT",
  IN = "IN",
  IS = "IS",
  NULL = "NULL",
  TRUE = "TRUE",
  FALSE = "FALSE",
  CASE = "CASE",
  WHEN = "WHEN",
  THEN = "THEN",
  ELSE = "ELSE",
  END = "END",
  EXISTS = "EXISTS",
  UNWIND = "UNWIND",
  CALL = "CALL",
  YIELD = "YIELD",

  // Literals and identifiers
  INTEGER = "INTEGER",
  FLOAT = "FLOAT",
  STRING = "STRING",
  IDENTIFIER = "IDENTIFIER",
  PARAMETER = "PARAMETER",

  // Operators
  EQ = "EQ", // =
  NEQ = "NEQ", // <>
  LT = "LT", // <
  GT = "GT", // >
  LTE = "LTE", // <=
  GTE = "GTE", // >=
  PLUS = "PLUS", // +
  PLUS_EQ = "PLUS_EQ", // +=
  MINUS = "MINUS", // -
  STAR = "STAR", // *
  SLASH = "SLASH", // /
  PERCENT = "PERCENT", // %
  STARTS_WITH = "STARTS_WITH",
  ENDS_WITH = "ENDS_WITH",
  CONTAINS = "CONTAINS",

  // Delimiters
  LPAREN = "LPAREN", // (
  RPAREN = "RPAREN", // )
  LBRACKET = "LBRACKET", // [
  RBRACKET = "RBRACKET", // ]
  LBRACE = "LBRACE", // {
  RBRACE = "RBRACE", // }
  COMMA = "COMMA", // ,
  DOT = "DOT", // .
  COLON = "COLON", // :
  PIPE = "PIPE", // |
  DOTDOT = "DOTDOT", // ..

  // Arrow parts
  DASH = "DASH", // -
  ARROW_RIGHT = "ARROW_RIGHT", // ->
  ARROW_LEFT = "ARROW_LEFT", // <-

  // Special
  EOF = "EOF",
}

// -- Token --------------------------------------------------------------------

export interface Token {
  readonly type: TokenType;
  readonly value: string;
  readonly position: number;
}

// -- Keyword map --------------------------------------------------------------

const KEYWORDS: Map<string, TokenType> = new Map([
  ["match", TokenType.MATCH],
  ["optional", TokenType.OPTIONAL],
  ["where", TokenType.WHERE],
  ["return", TokenType.RETURN],
  ["with", TokenType.WITH],
  ["create", TokenType.CREATE],
  ["delete", TokenType.DELETE],
  ["detach", TokenType.DETACH],
  ["set", TokenType.SET],
  ["remove", TokenType.REMOVE],
  ["merge", TokenType.MERGE],
  ["on", TokenType.ON],
  ["order", TokenType.ORDER],
  ["by", TokenType.BY],
  ["asc", TokenType.ASC],
  ["ascending", TokenType.ASC],
  ["desc", TokenType.DESC],
  ["descending", TokenType.DESC],
  ["skip", TokenType.SKIP],
  ["limit", TokenType.LIMIT],
  ["distinct", TokenType.DISTINCT],
  ["as", TokenType.AS],
  ["and", TokenType.AND],
  ["or", TokenType.OR],
  ["xor", TokenType.XOR],
  ["not", TokenType.NOT],
  ["in", TokenType.IN],
  ["is", TokenType.IS],
  ["null", TokenType.NULL],
  ["true", TokenType.TRUE],
  ["false", TokenType.FALSE],
  ["case", TokenType.CASE],
  ["when", TokenType.WHEN],
  ["then", TokenType.THEN],
  ["else", TokenType.ELSE],
  ["end", TokenType.END],
  ["exists", TokenType.EXISTS],
  ["unwind", TokenType.UNWIND],
  ["call", TokenType.CALL],
  ["yield", TokenType.YIELD],
  ["starts", TokenType.STARTS_WITH],
  ["ends", TokenType.ENDS_WITH],
  ["contains", TokenType.CONTAINS],
]);

// -- Tokenizer ----------------------------------------------------------------

export function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  while (i < input.length) {
    const ch = input[i]!;

    // Skip whitespace
    if (ch === " " || ch === "\t" || ch === "\n" || ch === "\r") {
      i++;
      continue;
    }

    // Skip single-line comments
    if (ch === "/" && i + 1 < input.length && input[i + 1] === "/") {
      while (i < input.length && input[i] !== "\n") i++;
      continue;
    }

    const pos = i;

    // String literals
    if (ch === "'" || ch === '"') {
      const quote = ch;
      i++;
      let str = "";
      while (i < input.length && input[i] !== quote) {
        if (input[i] === "\\" && i + 1 < input.length) {
          i++;
          const escaped = input[i]!;
          if (escaped === "n") str += "\n";
          else if (escaped === "t") str += "\t";
          else if (escaped === "\\") str += "\\";
          else if (escaped === quote) str += quote;
          else str += escaped;
        } else {
          str += input[i]!;
        }
        i++;
      }
      if (i < input.length) i++; // skip closing quote
      tokens.push({ type: TokenType.STRING, value: str, position: pos });
      continue;
    }

    // Numbers
    if (
      (ch >= "0" && ch <= "9") ||
      (ch === "." &&
        i + 1 < input.length &&
        input[i + 1]! >= "0" &&
        input[i + 1]! <= "9")
    ) {
      let num = "";
      let isFloat = false;
      while (
        i < input.length &&
        ((input[i]! >= "0" && input[i]! <= "9") || input[i] === ".")
      ) {
        if (input[i] === ".") {
          if (isFloat) break; // second dot
          // Check for .. operator
          if (i + 1 < input.length && input[i + 1] === ".") break;
          isFloat = true;
        }
        num += input[i]!;
        i++;
      }
      tokens.push({
        type: isFloat ? TokenType.FLOAT : TokenType.INTEGER,
        value: num,
        position: pos,
      });
      continue;
    }

    // Parameter ($name)
    if (ch === "$") {
      i++;
      let name = "";
      while (
        i < input.length &&
        ((input[i]! >= "a" && input[i]! <= "z") ||
          (input[i]! >= "A" && input[i]! <= "Z") ||
          (input[i]! >= "0" && input[i]! <= "9") ||
          input[i] === "_")
      ) {
        name += input[i]!;
        i++;
      }
      tokens.push({ type: TokenType.PARAMETER, value: name, position: pos });
      continue;
    }

    // Identifiers and keywords
    if ((ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z") || ch === "_") {
      let ident = "";
      while (
        i < input.length &&
        ((input[i]! >= "a" && input[i]! <= "z") ||
          (input[i]! >= "A" && input[i]! <= "Z") ||
          (input[i]! >= "0" && input[i]! <= "9") ||
          input[i] === "_")
      ) {
        ident += input[i]!;
        i++;
      }

      const lower = ident.toLowerCase();
      const kwType = KEYWORDS.get(lower);

      // Handle multi-word keywords: STARTS WITH, ENDS WITH
      if (kwType === TokenType.STARTS_WITH) {
        // Look ahead for WITH
        let j = i;
        while (j < input.length && (input[j] === " " || input[j] === "\t")) j++;
        let next = "";
        while (
          j < input.length &&
          input[j] !== " " &&
          input[j] !== "\t" &&
          input[j] !== "\n"
        ) {
          next += input[j]!;
          j++;
        }
        if (next.toLowerCase() === "with") {
          tokens.push({
            type: TokenType.STARTS_WITH,
            value: "STARTS WITH",
            position: pos,
          });
          i = j;
          continue;
        }
      }
      if (kwType === TokenType.ENDS_WITH) {
        let j = i;
        while (j < input.length && (input[j] === " " || input[j] === "\t")) j++;
        let next = "";
        while (
          j < input.length &&
          input[j] !== " " &&
          input[j] !== "\t" &&
          input[j] !== "\n"
        ) {
          next += input[j]!;
          j++;
        }
        if (next.toLowerCase() === "with") {
          tokens.push({ type: TokenType.ENDS_WITH, value: "ENDS WITH", position: pos });
          i = j;
          continue;
        }
      }

      if (
        kwType !== undefined &&
        kwType !== TokenType.STARTS_WITH &&
        kwType !== TokenType.ENDS_WITH
      ) {
        tokens.push({ type: kwType, value: ident, position: pos });
      } else {
        tokens.push({ type: TokenType.IDENTIFIER, value: ident, position: pos });
      }
      continue;
    }

    // Backtick-quoted identifiers
    if (ch === "`") {
      i++;
      let ident = "";
      while (i < input.length && input[i] !== "`") {
        ident += input[i]!;
        i++;
      }
      if (i < input.length) i++; // skip closing backtick
      tokens.push({ type: TokenType.IDENTIFIER, value: ident, position: pos });
      continue;
    }

    // Operators and delimiters
    if (ch === "(") {
      tokens.push({ type: TokenType.LPAREN, value: "(", position: pos });
      i++;
      continue;
    }
    if (ch === ")") {
      tokens.push({ type: TokenType.RPAREN, value: ")", position: pos });
      i++;
      continue;
    }
    if (ch === "[") {
      tokens.push({ type: TokenType.LBRACKET, value: "[", position: pos });
      i++;
      continue;
    }
    if (ch === "]") {
      tokens.push({ type: TokenType.RBRACKET, value: "]", position: pos });
      i++;
      continue;
    }
    if (ch === "{") {
      tokens.push({ type: TokenType.LBRACE, value: "{", position: pos });
      i++;
      continue;
    }
    if (ch === "}") {
      tokens.push({ type: TokenType.RBRACE, value: "}", position: pos });
      i++;
      continue;
    }
    if (ch === ",") {
      tokens.push({ type: TokenType.COMMA, value: ",", position: pos });
      i++;
      continue;
    }
    if (ch === ":") {
      tokens.push({ type: TokenType.COLON, value: ":", position: pos });
      i++;
      continue;
    }
    if (ch === "|") {
      tokens.push({ type: TokenType.PIPE, value: "|", position: pos });
      i++;
      continue;
    }
    if (ch === "+") {
      if (i + 1 < input.length && input[i + 1] === "=") {
        tokens.push({ type: TokenType.PLUS_EQ, value: "+=", position: pos });
        i += 2;
      } else {
        tokens.push({ type: TokenType.PLUS, value: "+", position: pos });
        i++;
      }
      continue;
    }
    if (ch === "*") {
      tokens.push({ type: TokenType.STAR, value: "*", position: pos });
      i++;
      continue;
    }
    if (ch === "%") {
      tokens.push({ type: TokenType.PERCENT, value: "%", position: pos });
      i++;
      continue;
    }

    // Dot and dotdot
    if (ch === ".") {
      if (i + 1 < input.length && input[i + 1] === ".") {
        tokens.push({ type: TokenType.DOTDOT, value: "..", position: pos });
        i += 2;
        continue;
      }
      tokens.push({ type: TokenType.DOT, value: ".", position: pos });
      i++;
      continue;
    }

    // Comparison operators and arrows
    if (ch === "=") {
      tokens.push({ type: TokenType.EQ, value: "=", position: pos });
      i++;
      continue;
    }

    if (ch === "<") {
      if (i + 1 < input.length) {
        if (input[i + 1] === ">") {
          tokens.push({ type: TokenType.NEQ, value: "<>", position: pos });
          i += 2;
          continue;
        }
        if (input[i + 1] === "=") {
          tokens.push({ type: TokenType.LTE, value: "<=", position: pos });
          i += 2;
          continue;
        }
        if (input[i + 1] === "-") {
          tokens.push({ type: TokenType.ARROW_LEFT, value: "<-", position: pos });
          i += 2;
          continue;
        }
      }
      tokens.push({ type: TokenType.LT, value: "<", position: pos });
      i++;
      continue;
    }

    if (ch === ">") {
      if (i + 1 < input.length && input[i + 1] === "=") {
        tokens.push({ type: TokenType.GTE, value: ">=", position: pos });
        i += 2;
        continue;
      }
      tokens.push({ type: TokenType.GT, value: ">", position: pos });
      i++;
      continue;
    }

    if (ch === "-") {
      if (i + 1 < input.length && input[i + 1] === ">") {
        tokens.push({ type: TokenType.ARROW_RIGHT, value: "->", position: pos });
        i += 2;
        continue;
      }
      tokens.push({ type: TokenType.DASH, value: "-", position: pos });
      i++;
      continue;
    }

    if (ch === "/") {
      tokens.push({ type: TokenType.SLASH, value: "/", position: pos });
      i++;
      continue;
    }

    // Unknown character - skip
    i++;
  }

  tokens.push({ type: TokenType.EOF, value: "", position: i });
  return tokens;
}
