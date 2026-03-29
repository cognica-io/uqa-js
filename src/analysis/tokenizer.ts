//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- tokenizers
// 1:1 port of uqa/analysis/tokenizer.py

export abstract class Tokenizer {
  abstract tokenize(text: string): string[];
  abstract toDict(): Record<string, unknown>;

  static fromDict(d: Record<string, unknown>): Tokenizer {
    const type = d["type"] as string;
    switch (type) {
      case "whitespace":
        return WhitespaceTokenizer._fromDict(d);
      case "standard":
        return StandardTokenizer._fromDict(d);
      case "letter":
        return LetterTokenizer._fromDict(d);
      case "ngram":
        return NGramTokenizer._fromDict(d);
      case "pattern":
        return PatternTokenizer._fromDict(d);
      case "keyword":
        return KeywordTokenizer._fromDict(d);
      default:
        throw new Error(`Unknown Tokenizer type: ${type}`);
    }
  }
}

// -- WhitespaceTokenizer ------------------------------------------------------

export class WhitespaceTokenizer extends Tokenizer {
  tokenize(text: string): string[] {
    return text.split(/\s+/).filter((t) => t.length > 0);
  }

  toDict(): Record<string, unknown> {
    return { type: "whitespace" };
  }

  static _fromDict(_d: Record<string, unknown>): WhitespaceTokenizer {
    return new WhitespaceTokenizer();
  }
}

// -- StandardTokenizer --------------------------------------------------------

const WORD_RE = /[\p{L}\p{N}_]+/gu;

export class StandardTokenizer extends Tokenizer {
  tokenize(text: string): string[] {
    return [...text.matchAll(WORD_RE)].map((m) => m[0]);
  }

  toDict(): Record<string, unknown> {
    return { type: "standard" };
  }

  static _fromDict(_d: Record<string, unknown>): StandardTokenizer {
    return new StandardTokenizer();
  }
}

// -- LetterTokenizer ----------------------------------------------------------

const LETTER_RE = /[a-zA-Z]+/g;

export class LetterTokenizer extends Tokenizer {
  tokenize(text: string): string[] {
    return [...text.matchAll(LETTER_RE)].map((m) => m[0]);
  }

  toDict(): Record<string, unknown> {
    return { type: "letter" };
  }

  static _fromDict(_d: Record<string, unknown>): LetterTokenizer {
    return new LetterTokenizer();
  }
}

// -- NGramTokenizer -----------------------------------------------------------

export class NGramTokenizer extends Tokenizer {
  private readonly _minGram: number;
  private readonly _maxGram: number;

  constructor(minGram = 1, maxGram = 2) {
    super();
    if (minGram < 1) throw new Error("minGram must be >= 1");
    if (maxGram < minGram) throw new Error("maxGram must be >= minGram");
    this._minGram = minGram;
    this._maxGram = maxGram;
  }

  tokenize(text: string): string[] {
    const words = text.split(/\s+/).filter((w) => w.length > 0);
    const result: string[] = [];
    for (const word of words) {
      for (let n = this._minGram; n <= this._maxGram; n++) {
        for (let i = 0; i <= word.length - n; i++) {
          result.push(word.substring(i, i + n));
        }
      }
    }
    return result;
  }

  toDict(): Record<string, unknown> {
    return { type: "ngram", min_gram: this._minGram, max_gram: this._maxGram };
  }

  static _fromDict(d: Record<string, unknown>): NGramTokenizer {
    return new NGramTokenizer(d["min_gram"] as number, d["max_gram"] as number);
  }
}

// -- PatternTokenizer ---------------------------------------------------------

export class PatternTokenizer extends Tokenizer {
  private readonly _pattern: string;
  private readonly _re: RegExp;

  constructor(pattern = "\\W+") {
    super();
    this._pattern = pattern;
    this._re = new RegExp(pattern);
  }

  tokenize(text: string): string[] {
    return text.split(this._re).filter((t) => t.length > 0);
  }

  toDict(): Record<string, unknown> {
    return { type: "pattern", pattern: this._pattern };
  }

  static _fromDict(d: Record<string, unknown>): PatternTokenizer {
    return new PatternTokenizer(d["pattern"] as string);
  }
}

// -- KeywordTokenizer ---------------------------------------------------------

export class KeywordTokenizer extends Tokenizer {
  tokenize(text: string): string[] {
    return text.length > 0 ? [text] : [];
  }

  toDict(): Record<string, unknown> {
    return { type: "keyword" };
  }

  static _fromDict(_d: Record<string, unknown>): KeywordTokenizer {
    return new KeywordTokenizer();
  }
}
