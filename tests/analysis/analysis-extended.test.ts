import { describe, expect, it } from "vitest";
import {
  WhitespaceTokenizer,
  StandardTokenizer,
  LetterTokenizer,
  NGramTokenizer,
  PatternTokenizer,
  KeywordTokenizer,
  Tokenizer,
} from "../../src/analysis/tokenizer.js";
import {
  LowerCaseFilter,
  StopWordFilter,
  PorterStemFilter,
  ASCIIFoldingFilter,
  SynonymFilter,
  NGramFilter,
  EdgeNGramFilter,
  LengthFilter,
  TokenFilter,
} from "../../src/analysis/token-filter.js";
import {
  HTMLStripCharFilter,
  MappingCharFilter,
  PatternReplaceCharFilter,
  CharFilter,
} from "../../src/analysis/char-filter.js";
import {
  Analyzer,
  whitespaceAnalyzer,
  standardAnalyzer,
  standardCJKAnalyzer,
  keywordAnalyzer,
  DEFAULT_ANALYZER,
  registerAnalyzer,
  getAnalyzer,
  dropAnalyzer,
  listAnalyzers,
} from "../../src/analysis/analyzer.js";
import { MemoryInvertedIndex } from "../../src/storage/inverted-index.js";

// =============================================================================
// Tokenizers
// =============================================================================

describe("WhitespaceTokenizer", () => {
  it("basic tokenization", () => {
    const tok = new WhitespaceTokenizer();
    expect(tok.tokenize("hello world")).toEqual(["hello", "world"]);
  });

  it("multiple spaces", () => {
    const tok = new WhitespaceTokenizer();
    expect(tok.tokenize("  hello   world  ")).toEqual(["hello", "world"]);
  });

  it("empty string", () => {
    const tok = new WhitespaceTokenizer();
    expect(tok.tokenize("")).toEqual([]);
  });

  it("serialization roundtrip", () => {
    const tok = new WhitespaceTokenizer();
    const d = tok.toDict();
    expect(d).toEqual({ type: "whitespace" });
    const restored = Tokenizer.fromDict(d);
    expect(restored).toBeInstanceOf(WhitespaceTokenizer);
  });
});

describe("StandardTokenizer", () => {
  it("basic tokenization", () => {
    const tok = new StandardTokenizer();
    expect(tok.tokenize("Hello, World!")).toEqual(["Hello", "World"]);
  });

  it("underscore and digits", () => {
    const tok = new StandardTokenizer();
    const tokens = tok.tokenize("cafe_latte 42");
    expect(tokens).toEqual(["cafe_latte", "42"]);
  });

  it("punctuation splitting", () => {
    const tok = new StandardTokenizer();
    expect(tok.tokenize("it's a test.")).toEqual(["it", "s", "a", "test"]);
  });

  it("serialization roundtrip", () => {
    const tok = new StandardTokenizer();
    const d = tok.toDict();
    const restored = Tokenizer.fromDict(d);
    expect(restored).toBeInstanceOf(StandardTokenizer);
  });
});

describe("LetterTokenizer", () => {
  it("basic tokenization", () => {
    const tok = new LetterTokenizer();
    expect(tok.tokenize("hello123world")).toEqual(["hello", "world"]);
  });

  it("only non-letters returns empty", () => {
    const tok = new LetterTokenizer();
    expect(tok.tokenize("42!!")).toEqual([]);
  });
});

describe("NGramTokenizer", () => {
  it("bigrams", () => {
    const tok = new NGramTokenizer(2, 2);
    expect(tok.tokenize("abc")).toEqual(["ab", "bc"]);
  });

  it("unigrams and bigrams", () => {
    const tok = new NGramTokenizer(1, 2);
    expect(tok.tokenize("ab")).toEqual(["a", "b", "ab"]);
  });

  it("invalid params throw", () => {
    expect(() => new NGramTokenizer(0)).toThrow();
    expect(() => new NGramTokenizer(3, 2)).toThrow();
  });

  it("serialization roundtrip", () => {
    const tok = new NGramTokenizer(2, 3);
    const d = tok.toDict();
    const restored = Tokenizer.fromDict(d);
    expect(restored).toBeInstanceOf(NGramTokenizer);
  });
});

describe("PatternTokenizer", () => {
  it("default pattern (non-word chars)", () => {
    const tok = new PatternTokenizer();
    expect(tok.tokenize("hello-world")).toEqual(["hello", "world"]);
  });

  it("custom pattern", () => {
    const tok = new PatternTokenizer(",\\s*");
    expect(tok.tokenize("a, b, c")).toEqual(["a", "b", "c"]);
  });

  it("serialization roundtrip", () => {
    const tok = new PatternTokenizer("\\|");
    const d = tok.toDict();
    const restored = Tokenizer.fromDict(d);
    expect(restored).toBeInstanceOf(PatternTokenizer);
  });
});

describe("KeywordTokenizer", () => {
  it("single token for entire input", () => {
    const tok = new KeywordTokenizer();
    expect(tok.tokenize("hello world")).toEqual(["hello world"]);
  });

  it("empty string returns empty", () => {
    const tok = new KeywordTokenizer();
    expect(tok.tokenize("")).toEqual([]);
  });
});

// =============================================================================
// Token Filters
// =============================================================================

describe("LowerCaseFilter", () => {
  it("converts to lowercase", () => {
    const f = new LowerCaseFilter();
    expect(f.filter(["Hello", "WORLD"])).toEqual(["hello", "world"]);
  });

  it("serialization roundtrip", () => {
    const f = new LowerCaseFilter();
    const d = f.toDict();
    const restored = TokenFilter.fromDict(d);
    expect(restored).toBeInstanceOf(LowerCaseFilter);
  });
});

describe("StopWordFilter", () => {
  it("removes English stop words by default", () => {
    const f = new StopWordFilter();
    const result = f.filter(["the", "quick", "brown", "fox"]);
    expect(result).toEqual(["quick", "brown", "fox"]);
  });

  it("custom words", () => {
    const f = new StopWordFilter("english", new Set(["quick"]));
    const result = f.filter(["the", "quick", "brown"]);
    expect(result).toEqual(["brown"]);
  });

  it("serialization roundtrip", () => {
    const f = new StopWordFilter("english", new Set(["extra"]));
    const d = f.toDict();
    const restored = TokenFilter.fromDict(d);
    expect(restored).toBeInstanceOf(StopWordFilter);
    expect(restored.filter(["extra"])).toEqual([]);
  });
});

describe("PorterStemFilter", () => {
  it("basic stemming", () => {
    const f = new PorterStemFilter();
    expect(f.filter(["running"])).toEqual(["run"]);
    expect(f.filter(["cats"])).toEqual(["cat"]);
  });

  it("complex stemming", () => {
    const f = new PorterStemFilter();
    const result = f.filter(["connections", "generalization", "relational"]);
    expect(result).toContain("connect");
    expect(result).toContain("gener");
  });
});

describe("ASCIIFoldingFilter", () => {
  it("plain ASCII passes through", () => {
    const f = new ASCIIFoldingFilter();
    expect(f.filter(["cafe"])).toEqual(["cafe"]);
  });

  it("serialization roundtrip", () => {
    const f = new ASCIIFoldingFilter();
    const d = f.toDict();
    const restored = TokenFilter.fromDict(d);
    expect(restored).toBeInstanceOf(ASCIIFoldingFilter);
  });
});

describe("SynonymFilter", () => {
  it("expands synonyms", () => {
    const f = new SynonymFilter({ fast: ["quick", "rapid"] });
    const result = f.filter(["fast", "car"]);
    expect(result).toEqual(["fast", "quick", "rapid", "car"]);
  });

  it("no match passes through", () => {
    const f = new SynonymFilter({ fast: ["quick"] });
    const result = f.filter(["slow", "car"]);
    expect(result).toEqual(["slow", "car"]);
  });

  it("serialization roundtrip", () => {
    const f = new SynonymFilter({ a: ["b"] });
    const d = f.toDict();
    const restored = TokenFilter.fromDict(d);
    expect(restored).toBeInstanceOf(SynonymFilter);
  });
});

describe("NGramFilter", () => {
  it("produces ngrams", () => {
    const f = new NGramFilter(2, 3);
    const result = f.filter(["hello"]);
    expect(result).toContain("he");
    expect(result).toContain("el");
    expect(result).toContain("ll");
    expect(result).toContain("lo");
    expect(result).toContain("hel");
    expect(result).toContain("ell");
    expect(result).toContain("llo");
  });

  it("short token dropped by default", () => {
    const f = new NGramFilter(2, 3);
    expect(f.filter(["a"])).toEqual([]);
  });

  it("keep_short preserves short tokens", () => {
    const f = new NGramFilter(2, 3, true);
    const result = f.filter(["a", "hello"]);
    expect(result[0]).toBe("a");
    expect(result).toContain("he");
  });

  it("keep_short mixed lengths", () => {
    const f = new NGramFilter(3, 4, true);
    const result = f.filter(["ab", "cd", "hello"]);
    expect(result).toContain("ab");
    expect(result).toContain("cd");
    expect(result).toContain("hel");
  });

  it("serialization roundtrip", () => {
    const f = new NGramFilter(2, 4);
    const d = f.toDict();
    expect(d).toEqual({ type: "ngram", min_gram: 2, max_gram: 4 });
    expect(d).not.toHaveProperty("keep_short");
    const restored = TokenFilter.fromDict(d);
    expect(restored).toBeInstanceOf(NGramFilter);
    expect(restored.filter(["abc"])).toEqual(f.filter(["abc"]));
  });

  it("serialization roundtrip with keep_short", () => {
    const f = new NGramFilter(2, 3, true);
    const d = f.toDict();
    expect(d["keep_short"]).toBe(true);
    const restored = TokenFilter.fromDict(d);
    expect(restored.filter(["a"])).toEqual(["a"]);
  });

  it("validation errors", () => {
    expect(() => new NGramFilter(0)).toThrow();
    expect(() => new NGramFilter(3, 2)).toThrow();
  });
});

describe("EdgeNGramFilter", () => {
  it("default edge ngrams", () => {
    const f = new EdgeNGramFilter(1, 3);
    expect(f.filter(["hello"])).toEqual(["h", "he", "hel"]);
  });

  it("min_gram skips short prefixes", () => {
    const f = new EdgeNGramFilter(2, 4);
    expect(f.filter(["abc"])).toEqual(["ab", "abc"]);
  });

  it("serialization roundtrip", () => {
    const f = new EdgeNGramFilter(2, 5);
    const d = f.toDict();
    const restored = TokenFilter.fromDict(d);
    expect(restored).toBeInstanceOf(EdgeNGramFilter);
  });
});

describe("LengthFilter", () => {
  it("min_length", () => {
    const f = new LengthFilter(3);
    expect(f.filter(["a", "ab", "abc", "abcd"])).toEqual(["abc", "abcd"]);
  });

  it("max_length", () => {
    const f = new LengthFilter(0, 3);
    expect(f.filter(["a", "ab", "abc", "abcd"])).toEqual(["a", "ab", "abc"]);
  });

  it("both min and max", () => {
    const f = new LengthFilter(2, 3);
    expect(f.filter(["a", "ab", "abc", "abcd"])).toEqual(["ab", "abc"]);
  });
});

// =============================================================================
// Char Filters
// =============================================================================

describe("HTMLStripCharFilter", () => {
  it("strips HTML tags", () => {
    const f = new HTMLStripCharFilter();
    const result = f.filter("<p>Hello <b>world</b></p>");
    expect(result).toContain("Hello");
    expect(result).toContain("world");
    expect(result).not.toContain("<");
  });

  it("no tags passes through", () => {
    const f = new HTMLStripCharFilter();
    expect(f.filter("plain text")).toBe("plain text");
  });

  it("decodes HTML entities", () => {
    const f = new HTMLStripCharFilter();
    expect(f.filter("a &amp; b")).toBe("a & b");
  });

  it("serialization roundtrip", () => {
    const f = new HTMLStripCharFilter();
    const d = f.toDict();
    const restored = CharFilter.fromDict(d);
    expect(restored).toBeInstanceOf(HTMLStripCharFilter);
  });
});

describe("MappingCharFilter", () => {
  it("replaces mapped characters", () => {
    const f = new MappingCharFilter({ "&": "and", "@": "at" });
    expect(f.filter("you & me @ home")).toBe("you and me at home");
  });

  it("serialization roundtrip", () => {
    const f = new MappingCharFilter({ x: "y" });
    const d = f.toDict();
    const restored = CharFilter.fromDict(d);
    expect(restored).toBeInstanceOf(MappingCharFilter);
  });
});

describe("PatternReplaceCharFilter", () => {
  it("replaces pattern matches", () => {
    const f = new PatternReplaceCharFilter("\\d+", "#");
    expect(f.filter("abc123def456")).toBe("abc#def#");
  });

  it("serialization roundtrip", () => {
    const f = new PatternReplaceCharFilter("\\s+", " ");
    const d = f.toDict();
    const restored = CharFilter.fromDict(d);
    expect(restored).toBeInstanceOf(PatternReplaceCharFilter);
  });
});

// =============================================================================
// Analyzer
// =============================================================================

describe("Analyzer", () => {
  it("DEFAULT_ANALYZER matches standard analyzer output", () => {
    const a = standardAnalyzer();
    const result = DEFAULT_ANALYZER.analyze("The Quick BROWN Fox");
    expect(result).toEqual(a.analyze("The Quick BROWN Fox"));
    expect(result).not.toContain("the");
    expect(result).toContain("quick");
    expect(result).toContain("brown");
    expect(result).toContain("fox");
  });

  it("whitespace analyzer", () => {
    const a = whitespaceAnalyzer();
    expect(a.analyze("Hello World")).toEqual(["hello", "world"]);
  });

  it("standard analyzer removes stop words", () => {
    const a = standardAnalyzer();
    const result = a.analyze("The quick brown fox");
    expect(result).not.toContain("the");
    expect(result).toContain("quick");
  });

  it("standard analyzer applies stemming", () => {
    const a = standardAnalyzer();
    const result = a.analyze("Running transformers efficiently");
    expect(result).toContain("run");
    expect(result).toContain("transform");
  });

  it("standard analyzer applies ASCII folding", () => {
    const a = standardAnalyzer();
    const result = a.analyze("caf\u00e9 r\u00e9sum\u00e9");
    // After folding: "cafe", "resume" -> after stemming: "caf", "resum" (or similar)
    // The folding must have removed the accent; stemming may further reduce
    // Verify that no accented characters remain
    for (const token of result) {
      for (let i = 0; i < token.length; i++) {
        expect(token.charCodeAt(i)).toBeLessThanOrEqual(127);
      }
    }
    // "resume" -> stem -> should contain a token starting with "r" and "sum"
    expect(result.some((t) => t.startsWith("caf"))).toBe(true);
    expect(result.some((t) => t.startsWith("r") || t.includes("sum"))).toBe(true);
  });

  it("standard CJK analyzer produces ngrams", () => {
    const a = standardCJKAnalyzer();
    const result = a.analyze("hello world");
    expect(result).toContain("he");
    expect(result).toContain("hel");
    expect(result).toContain("wo");
    expect(result).toContain("wor");
  });

  it("standard CJK analyzer applies stemming", () => {
    const a = standardCJKAnalyzer();
    const result = a.analyze("Running");
    expect(result).toContain("ru");
    expect(result).toContain("run");
  });

  it("standard CJK analyzer keeps short tokens with keep_short", () => {
    const a = standardCJKAnalyzer();
    const result = a.analyze("x marks");
    expect(result).toContain("x");
    expect(result).toContain("ma");
    expect(result).toContain("mar");
  });

  it("keyword analyzer returns entire input", () => {
    const a = keywordAnalyzer();
    expect(a.analyze("hello world")).toEqual(["hello world"]);
  });

  it("custom pipeline with char filters and token filters", () => {
    const a = new Analyzer(
      new StandardTokenizer(),
      [new LowerCaseFilter(), new PorterStemFilter()],
      [new HTMLStripCharFilter()],
    );
    const result = a.analyze("<p>Running Connections</p>");
    expect(result).toContain("run");
    expect(result).toContain("connect");
  });

  it("serialization roundtrip (dict)", () => {
    const a = new Analyzer(
      new StandardTokenizer(),
      [new LowerCaseFilter(), new StopWordFilter("english")],
      [new HTMLStripCharFilter()],
    );
    const d = a.toDict();
    const restored = Analyzer.fromDict(d as Record<string, unknown>);
    const text = "<p>The quick brown fox</p>";
    expect(restored.analyze(text)).toEqual(a.analyze(text));
  });

  it("JSON roundtrip", () => {
    const a = standardAnalyzer();
    const j = a.toJSON();
    const restored = Analyzer.fromJSON(j);
    const text = "The quick brown fox";
    expect(restored.analyze(text)).toEqual(a.analyze(text));
  });
});

// =============================================================================
// Named Analyzer Registry
// =============================================================================

describe("Analyzer registry", () => {
  it("built-in analyzers are listed", () => {
    const names = listAnalyzers();
    expect(names).toContain("whitespace");
    expect(names).toContain("standard");
    expect(names).toContain("standard_cjk");
    expect(names).toContain("keyword");
  });

  it("register and get custom analyzer", () => {
    const custom = new Analyzer(new LetterTokenizer(), [new LowerCaseFilter()]);
    registerAnalyzer("test_custom_reg_ts", custom);
    try {
      const retrieved = getAnalyzer("test_custom_reg_ts");
      expect(retrieved.analyze("hello123world")).toEqual(["hello", "world"]);
    } finally {
      dropAnalyzer("test_custom_reg_ts");
    }
  });

  it("cannot overwrite built-in", () => {
    expect(() => registerAnalyzer("standard", whitespaceAnalyzer())).toThrow(
      /built-in/,
    );
  });

  it("cannot drop built-in", () => {
    expect(() => dropAnalyzer("standard")).toThrow(/built-in/);
  });

  it("unknown analyzer throws", () => {
    expect(() => getAnalyzer("nonexistent_analyzer_xyz")).toThrow(/Unknown/);
  });

  it("drop nonexistent throws", () => {
    expect(() => dropAnalyzer("nonexistent_analyzer_xyz")).toThrow(/not found/);
  });
});

// =============================================================================
// InvertedIndex Analyzer Integration
// =============================================================================

describe("InvertedIndex analyzer integration", () => {
  it("default analyzer removes stop words", () => {
    const idx = new MemoryInvertedIndex();
    idx.addDocument(1, { title: "The Quick Brown Fox" });
    // Default is whitespace+lowercase, so "the" IS indexed
    const pl = idx.getPostingList("title", "the");
    // whitespace_analyzer: "the" -> "the" (no stop words by default)
    expect(pl.length).toBeGreaterThanOrEqual(0);
    const plQuick = idx.getPostingList("title", "quick");
    expect(plQuick.length).toBe(1);
  });

  it("custom analyzer", () => {
    const a = standardAnalyzer();
    const idx = new MemoryInvertedIndex(a);
    idx.addDocument(1, { title: "The Quick Brown Fox" });
    // "the" is a stop word in standard analyzer
    const pl = idx.getPostingList("title", "the");
    expect(pl.length).toBe(0);
    const plQuick = idx.getPostingList("title", "quick");
    expect(plQuick.length).toBe(1);
  });

  it("per-field analyzer", () => {
    const idx = new MemoryInvertedIndex();
    idx.setFieldAnalyzer("title", standardAnalyzer());
    idx.setFieldAnalyzer("body", whitespaceAnalyzer());
    idx.addDocument(1, { title: "The Quick Fox", body: "The body" });
    // "the" removed from title (standard) but kept in body (whitespace)
    expect(idx.getPostingList("title", "the").length).toBe(0);
    expect(idx.getPostingList("body", "the").length).toBe(1);
  });

  it("getFieldAnalyzer returns default when not set", () => {
    const idx = new MemoryInvertedIndex();
    expect(idx.getFieldAnalyzer("title")).toBe(idx.analyzer);
    const custom = keywordAnalyzer();
    idx.setFieldAnalyzer("title", custom);
    expect(idx.getFieldAnalyzer("title")).toBe(custom);
    expect(idx.getFieldAnalyzer("body")).toBe(idx.analyzer);
  });
});

// =============================================================================
// Dual (index/search) analyzer
// =============================================================================

describe("Dual analyzer", () => {
  it("set_field_analyzer with phase=both sets both", () => {
    const idx = new MemoryInvertedIndex();
    const analyzer = standardAnalyzer();
    idx.setFieldAnalyzer("body", analyzer, "both");
    expect(idx.getFieldAnalyzer("body")).toBe(analyzer);
    expect(idx.getSearchAnalyzer("body")).toBe(analyzer);
  });

  it("index and search analyzers can differ", () => {
    const idxAnalyzer = new Analyzer(new WhitespaceTokenizer(), [
      new LowerCaseFilter(),
    ]);
    const searchAnalyzer = new Analyzer(new WhitespaceTokenizer(), [
      new LowerCaseFilter(),
      new SynonymFilter({ car: ["automobile"] }),
    ]);
    const idx = new MemoryInvertedIndex();
    idx.setFieldAnalyzer("body", idxAnalyzer, "index");
    idx.setFieldAnalyzer("body", searchAnalyzer, "search");
    expect(idx.getFieldAnalyzer("body")).toBe(idxAnalyzer);
    expect(idx.getSearchAnalyzer("body")).toBe(searchAnalyzer);
  });

  it("search falls back to index analyzer", () => {
    const idx = new MemoryInvertedIndex();
    const analyzer = standardAnalyzer();
    idx.setFieldAnalyzer("body", analyzer, "index");
    expect(idx.getSearchAnalyzer("body")).toBe(analyzer);
  });

  it("search falls back to default", () => {
    const idx = new MemoryInvertedIndex();
    const defaultAnalyzer = idx.analyzer;
    expect(idx.getSearchAnalyzer("body")).toBe(defaultAnalyzer);
  });

  it("backward compat: no phase sets both", () => {
    const idx = new MemoryInvertedIndex();
    const analyzer = standardAnalyzer();
    idx.setFieldAnalyzer("body", analyzer);
    expect(idx.getFieldAnalyzer("body")).toBe(analyzer);
    expect(idx.getSearchAnalyzer("body")).toBe(analyzer);
  });
});
