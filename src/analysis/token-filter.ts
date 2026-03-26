//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- token filters
// 1:1 port of uqa/analysis/token_filter.py

// -- Abstract base ------------------------------------------------------------

export abstract class TokenFilter {
  abstract filter(tokens: string[]): string[];
  abstract toDict(): Record<string, unknown>;

  static fromDict(d: Record<string, unknown>): TokenFilter {
    const type = d["type"] as string;
    switch (type) {
      case "lowercase":
        return LowerCaseFilter._fromDict(d);
      case "stop":
        return StopWordFilter._fromDict(d);
      case "porter_stem":
        return PorterStemFilter._fromDict(d);
      case "ascii_folding":
        return ASCIIFoldingFilter._fromDict(d);
      case "synonym":
        return SynonymFilter._fromDict(d);
      case "ngram":
        return NGramFilter._fromDict(d);
      case "edge_ngram":
        return EdgeNGramFilter._fromDict(d);
      case "length":
        return LengthFilter._fromDict(d);
      default:
        throw new Error(`Unknown TokenFilter type: ${type}`);
    }
  }
}

// -- LowerCaseFilter ----------------------------------------------------------

export class LowerCaseFilter extends TokenFilter {
  filter(tokens: string[]): string[] {
    return tokens.map((t) => t.toLowerCase());
  }

  toDict(): Record<string, unknown> {
    return { type: "lowercase" };
  }

  static _fromDict(_d: Record<string, unknown>): LowerCaseFilter {
    return new LowerCaseFilter();
  }
}

// -- StopWordFilter -----------------------------------------------------------

const STOP_WORDS: Record<string, Set<string>> = {
  english: new Set([
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "me",
    "my",
    "no",
    "nor",
    "not",
    "of",
    "on",
    "or",
    "our",
    "own",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "too",
    "us",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "which",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
  ]),
};

export class StopWordFilter extends TokenFilter {
  private readonly _language: string;
  private readonly _customWords: Set<string>;
  private readonly _words: Set<string>;

  constructor(language = "english", customWords?: Set<string> | null) {
    super();
    this._language = language;
    this._customWords = customWords ?? new Set();
    const base = STOP_WORDS[language] ?? new Set<string>();
    this._words = new Set([...base, ...this._customWords]);
  }

  filter(tokens: string[]): string[] {
    return tokens.filter((t) => !this._words.has(t));
  }

  toDict(): Record<string, unknown> {
    const d: Record<string, unknown> = { type: "stop", language: this._language };
    if (this._customWords.size > 0) {
      d["custom_words"] = [...this._customWords].sort();
    }
    return d;
  }

  static _fromDict(d: Record<string, unknown>): StopWordFilter {
    const custom = d["custom_words"] ? new Set(d["custom_words"] as string[]) : null;
    return new StopWordFilter(
      (d["language"] as string | undefined) ?? "english",
      custom,
    );
  }
}

// -- PorterStemFilter ---------------------------------------------------------

function isConsonant(w: string, i: number): boolean {
  const ch = w[i]!;
  if (ch === "a" || ch === "e" || ch === "i" || ch === "o" || ch === "u") {
    return false;
  }
  if (ch === "y") {
    return i === 0 || !isConsonant(w, i - 1);
  }
  return true;
}

function measure(w: string, j: number): number {
  let n = 0;
  let i = 0;
  // Skip initial consonants
  while (i <= j && isConsonant(w, i)) i++;
  while (i <= j) {
    // Count vowels
    while (i <= j && !isConsonant(w, i)) i++;
    n++;
    // Count consonants
    while (i <= j && isConsonant(w, i)) i++;
  }
  return n;
}

function vowelInStem(w: string, j: number): boolean {
  for (let i = 0; i <= j; i++) {
    if (!isConsonant(w, i)) return true;
  }
  return false;
}

function doubleConsonant(w: string, j: number): boolean {
  if (j < 1) return false;
  return w[j] === w[j - 1] && isConsonant(w, j);
}

function cvc(w: string, i: number): boolean {
  if (i < 2) return false;
  if (!isConsonant(w, i - 2) || isConsonant(w, i - 1) || !isConsonant(w, i)) {
    return false;
  }
  const ch = w[i]!;
  return ch !== "w" && ch !== "x" && ch !== "y";
}

function porterStem(word: string): string {
  if (word.length <= 2) return word;

  let w = word;

  // Step 1a
  if (w.endsWith("sses")) {
    w = w.slice(0, -2);
  } else if (w.endsWith("ies")) {
    w = w.slice(0, -2);
  } else if (!w.endsWith("ss") && w.endsWith("s")) {
    w = w.slice(0, -1);
  }

  // Step 1b
  if (w.endsWith("eed")) {
    const stem = w.slice(0, -3);
    if (measure(w, stem.length - 1) > 0) {
      w = w.slice(0, -1); // remove "d" -> "ee"
    }
  } else {
    let found = false;
    let stemLen = 0;
    if (w.endsWith("ed")) {
      stemLen = w.length - 2;
      found = vowelInStem(w, stemLen - 1);
    } else if (w.endsWith("ing")) {
      stemLen = w.length - 3;
      found = vowelInStem(w, stemLen - 1);
    }
    if (found) {
      w = w.slice(0, stemLen);
      if (w.endsWith("at") || w.endsWith("bl") || w.endsWith("iz")) {
        w = w + "e";
      } else if (
        doubleConsonant(w, w.length - 1) &&
        w[w.length - 1] !== "l" &&
        w[w.length - 1] !== "s" &&
        w[w.length - 1] !== "z"
      ) {
        w = w.slice(0, -1);
      } else if (measure(w, w.length - 1) === 1 && cvc(w, w.length - 1)) {
        w = w + "e";
      }
    }
  }

  // Step 1c
  if (w.endsWith("y") && vowelInStem(w, w.length - 2)) {
    w = w.slice(0, -1) + "i";
  }

  // Step 2
  const step2: [string, string][] = [
    ["ational", "ate"],
    ["tional", "tion"],
    ["enci", "ence"],
    ["anci", "ance"],
    ["izer", "ize"],
    ["abli", "able"],
    ["alli", "al"],
    ["entli", "ent"],
    ["eli", "e"],
    ["ousli", "ous"],
    ["ization", "ize"],
    ["ation", "ate"],
    ["ator", "ate"],
    ["alism", "al"],
    ["iveness", "ive"],
    ["fulness", "ful"],
    ["ousness", "ous"],
    ["aliti", "al"],
    ["iviti", "ive"],
    ["biliti", "ble"],
  ];
  for (const [suffix, replacement] of step2) {
    if (w.endsWith(suffix)) {
      const stem = w.slice(0, -suffix.length);
      if (measure(w, stem.length - 1) > 0) {
        w = stem + replacement;
      }
      break;
    }
  }

  // Step 3
  const step3: [string, string][] = [
    ["icate", "ic"],
    ["ative", ""],
    ["alize", "al"],
    ["iciti", "ic"],
    ["ical", "ic"],
    ["ful", ""],
    ["ness", ""],
  ];
  for (const [suffix, replacement] of step3) {
    if (w.endsWith(suffix)) {
      const stem = w.slice(0, -suffix.length);
      if (measure(w, stem.length - 1) > 0) {
        w = stem + replacement;
      }
      break;
    }
  }

  // Step 4
  const step4suffixes = [
    "al",
    "ance",
    "ence",
    "er",
    "ic",
    "able",
    "ible",
    "ant",
    "ement",
    "ment",
    "ent",
    "ion",
    "ou",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
  ];
  for (const suffix of step4suffixes) {
    if (w.endsWith(suffix)) {
      const stem = w.slice(0, -suffix.length);
      if (measure(w, stem.length - 1) > 1) {
        if (suffix === "ion") {
          const lastChar = stem[stem.length - 1];
          if (lastChar === "s" || lastChar === "t") {
            w = stem;
          }
        } else {
          w = stem;
        }
      }
      break;
    }
  }

  // Step 5a
  if (w.endsWith("e")) {
    const stem = w.slice(0, -1);
    const m = measure(w, stem.length - 1);
    if (m > 1 || (m === 1 && !cvc(w, stem.length - 1))) {
      w = stem;
    }
  }

  // Step 5b
  if (
    measure(w, w.length - 1) > 1 &&
    doubleConsonant(w, w.length - 1) &&
    w[w.length - 1] === "l"
  ) {
    w = w.slice(0, -1);
  }

  return w;
}

export class PorterStemFilter extends TokenFilter {
  filter(tokens: string[]): string[] {
    return tokens.map((t) => porterStem(t));
  }

  toDict(): Record<string, unknown> {
    return { type: "porter_stem" };
  }

  static _fromDict(_d: Record<string, unknown>): PorterStemFilter {
    return new PorterStemFilter();
  }
}

// -- ASCIIFoldingFilter -------------------------------------------------------

function isASCII(s: string): boolean {
  for (let i = 0; i < s.length; i++) {
    if (s.charCodeAt(i) > 127) return false;
  }
  return true;
}

function foldChar(ch: string): string {
  if (ch.charCodeAt(0) <= 127) return ch;
  // NFKD normalize then strip non-ASCII
  const normalized = ch.normalize("NFKD");
  let result = "";
  for (let i = 0; i < normalized.length; i++) {
    if (normalized.charCodeAt(i) <= 127) {
      result += normalized.charAt(i);
    }
  }
  return result.length > 0 ? result : ch;
}

export class ASCIIFoldingFilter extends TokenFilter {
  filter(tokens: string[]): string[] {
    return tokens.map((t) => ASCIIFoldingFilter._fold(t));
  }

  private static _fold(token: string): string {
    if (isASCII(token)) return token;
    let result = "";
    for (let i = 0; i < token.length; i++) {
      result += foldChar(token[i]!);
    }
    return result;
  }

  toDict(): Record<string, unknown> {
    return { type: "ascii_folding" };
  }

  static _fromDict(_d: Record<string, unknown>): ASCIIFoldingFilter {
    return new ASCIIFoldingFilter();
  }
}

// -- parseSynonymText ---------------------------------------------------------

/**
 * Parse a Solr/Elasticsearch-format synonym definition string.
 *
 * Supported formats (one rule per line):
 *   Explicit mapping:    car => automobile, vehicle
 *   Equivalent synonyms: car, automobile, vehicle
 *
 * Lines starting with '#' are comments. Blank lines are skipped.
 */
export function parseSynonymText(text: string): Record<string, string[]> {
  const synonyms: Record<string, string[]> = {};
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    if (line.includes("=>")) {
      const [lhs, rhs] = line.split("=>", 2);
      const key = lhs!.trim();
      const values = rhs!
        .split(",")
        .map((v) => v.trim())
        .filter((v) => v.length > 0);
      if (key && values.length > 0) {
        const existing = synonyms[key] ?? [];
        existing.push(...values);
        synonyms[key] = existing;
      }
    } else {
      const terms = line
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);
      if (terms.length < 2) continue;
      for (const term of terms) {
        const others = terms.filter((t) => t !== term);
        const existing = synonyms[term] ?? [];
        existing.push(...others);
        synonyms[term] = existing;
      }
    }
  }

  // Deduplicate values while preserving order
  for (const key of Object.keys(synonyms)) {
    const seen = new Set<string>();
    const deduped: string[] = [];
    for (const v of synonyms[key]!) {
      if (!seen.has(v)) {
        seen.add(v);
        deduped.push(v);
      }
    }
    synonyms[key] = deduped;
  }

  return synonyms;
}

// -- SynonymFilter ------------------------------------------------------------

export class SynonymFilter extends TokenFilter {
  private readonly _synonyms: Record<string, string[]>;

  constructor(synonyms: Record<string, string[]>) {
    super();
    this._synonyms = synonyms;
  }

  filter(tokens: string[]): string[] {
    const result: string[] = [];
    for (const token of tokens) {
      result.push(token);
      const syns = this._synonyms[token];
      if (syns) {
        result.push(...syns);
      }
    }
    return result;
  }

  toDict(): Record<string, unknown> {
    return { type: "synonym", synonyms: this._synonyms };
  }

  static _fromDict(d: Record<string, unknown>): SynonymFilter {
    return new SynonymFilter(d["synonyms"] as Record<string, string[]>);
  }
}

// -- NGramFilter --------------------------------------------------------------

export class NGramFilter extends TokenFilter {
  private readonly _minGram: number;
  private readonly _maxGram: number;
  private readonly _keepShort: boolean;

  constructor(minGram = 2, maxGram = 3, keepShort = false) {
    super();
    if (minGram < 1) throw new Error("minGram must be >= 1");
    if (maxGram < minGram) throw new Error("maxGram must be >= minGram");
    this._minGram = minGram;
    this._maxGram = maxGram;
    this._keepShort = keepShort;
  }

  filter(tokens: string[]): string[] {
    const result: string[] = [];
    for (const token of tokens) {
      if (token.length < this._minGram) {
        if (this._keepShort) result.push(token);
        continue;
      }
      for (let n = this._minGram; n <= this._maxGram; n++) {
        for (let i = 0; i <= token.length - n; i++) {
          result.push(token.substring(i, i + n));
        }
      }
    }
    return result;
  }

  toDict(): Record<string, unknown> {
    const d: Record<string, unknown> = {
      type: "ngram",
      min_gram: this._minGram,
      max_gram: this._maxGram,
    };
    if (this._keepShort) d["keep_short"] = true;
    return d;
  }

  static _fromDict(d: Record<string, unknown>): NGramFilter {
    return new NGramFilter(
      d["min_gram"] as number,
      d["max_gram"] as number,
      (d["keep_short"] as boolean | undefined) ?? false,
    );
  }
}

// -- EdgeNGramFilter ----------------------------------------------------------

export class EdgeNGramFilter extends TokenFilter {
  private readonly _minGram: number;
  private readonly _maxGram: number;

  constructor(minGram = 1, maxGram = 20) {
    super();
    this._minGram = minGram;
    this._maxGram = maxGram;
  }

  filter(tokens: string[]): string[] {
    const result: string[] = [];
    for (const token of tokens) {
      const upper = Math.min(this._maxGram, token.length);
      for (let n = this._minGram; n <= upper; n++) {
        result.push(token.substring(0, n));
      }
    }
    return result;
  }

  toDict(): Record<string, unknown> {
    return {
      type: "edge_ngram",
      min_gram: this._minGram,
      max_gram: this._maxGram,
    };
  }

  static _fromDict(d: Record<string, unknown>): EdgeNGramFilter {
    return new EdgeNGramFilter(d["min_gram"] as number, d["max_gram"] as number);
  }
}

// -- LengthFilter -------------------------------------------------------------

export class LengthFilter extends TokenFilter {
  private readonly _minLength: number;
  private readonly _maxLength: number;

  constructor(minLength = 0, maxLength = 0) {
    super();
    this._minLength = minLength;
    this._maxLength = maxLength;
  }

  filter(tokens: string[]): string[] {
    return tokens.filter((t) => {
      if (t.length < this._minLength) return false;
      if (this._maxLength > 0 && t.length > this._maxLength) return false;
      return true;
    });
  }

  toDict(): Record<string, unknown> {
    return {
      type: "length",
      min_length: this._minLength,
      max_length: this._maxLength,
    };
  }

  static _fromDict(d: Record<string, unknown>): LengthFilter {
    return new LengthFilter(d["min_length"] as number, d["max_length"] as number);
  }
}
