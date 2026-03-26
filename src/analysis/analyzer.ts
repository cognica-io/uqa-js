//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Analyzer pipeline
// 1:1 port of uqa/analysis/analyzer.py

import type { CharFilter } from "./char-filter.js";
import { CharFilter as CharFilterBase } from "./char-filter.js";
import type { TokenFilter } from "./token-filter.js";
import {
  ASCIIFoldingFilter,
  LowerCaseFilter,
  NGramFilter,
  PorterStemFilter,
  StopWordFilter,
  TokenFilter as TokenFilterBase,
} from "./token-filter.js";
import type { Tokenizer } from "./tokenizer.js";
import {
  KeywordTokenizer,
  StandardTokenizer,
  Tokenizer as TokenizerBase,
  WhitespaceTokenizer,
} from "./tokenizer.js";

export class Analyzer {
  private readonly _tokenizer: Tokenizer;
  private readonly _tokenFilters: TokenFilter[];
  private readonly _charFilters: CharFilter[];

  constructor(
    tokenizer?: Tokenizer | null,
    tokenFilters?: TokenFilter[] | null,
    charFilters?: CharFilter[] | null,
  ) {
    this._tokenizer = tokenizer ?? new WhitespaceTokenizer();
    this._tokenFilters = tokenFilters ?? [];
    this._charFilters = charFilters ?? [];
  }

  get tokenizer(): Tokenizer {
    return this._tokenizer;
  }

  get tokenFilters(): TokenFilter[] {
    return this._tokenFilters;
  }

  get charFilters(): CharFilter[] {
    return this._charFilters;
  }

  analyze(text: string): string[] {
    // Pipeline: char_filters -> tokenizer -> token_filters
    let processed = text;
    for (const cf of this._charFilters) {
      processed = cf.filter(processed);
    }
    let tokens = this._tokenizer.tokenize(processed);
    for (const tf of this._tokenFilters) {
      tokens = tf.filter(tokens);
    }
    return tokens;
  }

  toDict(): Record<string, unknown> {
    return {
      tokenizer: this._tokenizer.toDict(),
      token_filters: this._tokenFilters.map((f) => f.toDict()),
      char_filters: this._charFilters.map((f) => f.toDict()),
    };
  }

  toJSON(): string {
    return JSON.stringify(this.toDict());
  }

  static fromDict(d: Record<string, unknown>): Analyzer {
    const tokenizer = TokenizerBase.fromDict(d["tokenizer"] as Record<string, unknown>);
    const tokenFilters = (d["token_filters"] as Record<string, unknown>[]).map((f) =>
      TokenFilterBase.fromDict(f),
    );
    const charFilters = d["char_filters"]
      ? (d["char_filters"] as Record<string, unknown>[]).map((f) =>
          CharFilterBase.fromDict(f),
        )
      : [];
    return new Analyzer(tokenizer, tokenFilters, charFilters);
  }

  static fromJSON(s: string): Analyzer {
    return Analyzer.fromDict(JSON.parse(s) as Record<string, unknown>);
  }
}

// -- Factory functions --------------------------------------------------------

export function whitespaceAnalyzer(): Analyzer {
  return new Analyzer(new WhitespaceTokenizer(), [new LowerCaseFilter()]);
}

export function standardAnalyzer(language = "english"): Analyzer {
  return new Analyzer(new StandardTokenizer(), [
    new LowerCaseFilter(),
    new ASCIIFoldingFilter(),
    new StopWordFilter(language),
    new PorterStemFilter(),
  ]);
}

export function standardCJKAnalyzer(language = "english"): Analyzer {
  return new Analyzer(new StandardTokenizer(), [
    new LowerCaseFilter(),
    new ASCIIFoldingFilter(),
    new StopWordFilter(language),
    new PorterStemFilter(),
    new NGramFilter(2, 3, true),
  ]);
}

export function keywordAnalyzer(): Analyzer {
  return new Analyzer(new KeywordTokenizer(), []);
}

export const DEFAULT_ANALYZER = standardAnalyzer();

// -- Analyzer registry --------------------------------------------------------

const BUILTIN_ANALYZERS: Record<string, Analyzer> = {
  whitespace: whitespaceAnalyzer(),
  standard: standardAnalyzer(),
  standard_cjk: standardCJKAnalyzer(),
  keyword: keywordAnalyzer(),
};

const customAnalyzers = new Map<string, Analyzer>();

export function registerAnalyzer(name: string, analyzer: Analyzer): void {
  if (name in BUILTIN_ANALYZERS) {
    throw new Error(`Cannot override built-in analyzer: ${name}`);
  }
  customAnalyzers.set(name, analyzer);
}

export function getAnalyzer(name: string): Analyzer {
  const custom = customAnalyzers.get(name);
  if (custom) return custom;
  const builtin = BUILTIN_ANALYZERS[name];
  if (builtin) return builtin;
  throw new Error(`Unknown analyzer: ${name}`);
}

export function dropAnalyzer(name: string): void {
  if (name in BUILTIN_ANALYZERS) {
    throw new Error(`Cannot drop built-in analyzer: ${name}`);
  }
  if (!customAnalyzers.has(name)) {
    throw new Error(`Analyzer not found: ${name}`);
  }
  customAnalyzers.delete(name);
}

export function listAnalyzers(): string[] {
  return [...Object.keys(BUILTIN_ANALYZERS), ...customAnalyzers.keys()].sort();
}
