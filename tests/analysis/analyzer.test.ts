import { describe, expect, it } from "vitest";
import {
  Analyzer,
  DEFAULT_ANALYZER,
  dropAnalyzer,
  getAnalyzer,
  keywordAnalyzer,
  listAnalyzers,
  registerAnalyzer,
  standardAnalyzer,
  standardCJKAnalyzer,
  whitespaceAnalyzer,
} from "../../src/analysis/analyzer.js";
import { HTMLStripCharFilter } from "../../src/analysis/char-filter.js";

describe("Analyzer", () => {
  it("default pipeline: whitespace tokenize", () => {
    const a = new Analyzer();
    expect(a.analyze("Hello World")).toEqual(["Hello", "World"]);
  });

  it("applies char filters before tokenizer", () => {
    const a = new Analyzer(null, null, [new HTMLStripCharFilter()]);
    expect(a.analyze("<b>hello</b> world")).toEqual(["hello", "world"]);
  });

  it("applies token filters after tokenizer", () => {
    const a = whitespaceAnalyzer();
    expect(a.analyze("Hello World")).toEqual(["hello", "world"]);
  });
});

describe("whitespaceAnalyzer", () => {
  it("lowercases and splits", () => {
    expect(whitespaceAnalyzer().analyze("The Quick FOX")).toEqual([
      "the",
      "quick",
      "fox",
    ]);
  });
});

describe("standardAnalyzer", () => {
  it("tokenizes, lowercases, folds, removes stops, stems", () => {
    const a = standardAnalyzer();
    const tokens = a.analyze("The running foxes jumped");
    // "the" -> removed (stop word)
    // "running" -> "run" (stem)
    // "foxes" -> "fox" (stem)
    // "jumped" -> "jump" (stem)
    expect(tokens).toContain("run");
    expect(tokens).toContain("fox");
    expect(tokens).toContain("jump");
    expect(tokens).not.toContain("the");
  });
});

describe("standardCJKAnalyzer", () => {
  it("includes ngrams", () => {
    const a = standardCJKAnalyzer();
    const tokens = a.analyze("test");
    // "test" -> stem -> ngrams
    expect(tokens.length).toBeGreaterThan(1);
  });
});

describe("keywordAnalyzer", () => {
  it("returns entire input as one token", () => {
    expect(keywordAnalyzer().analyze("hello world")).toEqual(["hello world"]);
  });
});

describe("DEFAULT_ANALYZER", () => {
  it("is a standard analyzer", () => {
    const tokens = DEFAULT_ANALYZER.analyze("Running tests");
    expect(tokens).toContain("run");
    expect(tokens).toContain("test");
  });
});

describe("serialization", () => {
  it("round-trips via toDict/fromDict", () => {
    const a = standardAnalyzer();
    const d = a.toDict();
    const restored = Analyzer.fromDict(d);
    expect(restored.analyze("Running foxes")).toEqual(a.analyze("Running foxes"));
  });

  it("round-trips via toJSON/fromJSON", () => {
    const a = whitespaceAnalyzer();
    const json = a.toJSON();
    const restored = Analyzer.fromJSON(json);
    expect(restored.analyze("Hello World")).toEqual(["hello", "world"]);
  });
});

describe("analyzer registry", () => {
  it("lists builtin analyzers", () => {
    const names = listAnalyzers();
    expect(names).toContain("whitespace");
    expect(names).toContain("standard");
    expect(names).toContain("standard_cjk");
    expect(names).toContain("keyword");
  });

  it("gets builtin analyzer", () => {
    const a = getAnalyzer("whitespace");
    expect(a.analyze("A B")).toEqual(["a", "b"]);
  });

  it("registers and retrieves custom analyzer", () => {
    const custom = keywordAnalyzer();
    registerAnalyzer("my_keyword", custom);
    expect(getAnalyzer("my_keyword")).toBe(custom);
    dropAnalyzer("my_keyword");
  });

  it("throws when overriding builtin", () => {
    expect(() => {
      registerAnalyzer("standard", keywordAnalyzer());
    }).toThrow("Cannot override built-in");
  });

  it("throws when getting unknown", () => {
    expect(() => getAnalyzer("nonexistent")).toThrow("Unknown analyzer");
  });

  it("throws when dropping builtin", () => {
    expect(() => {
      dropAnalyzer("standard");
    }).toThrow("Cannot drop built-in");
  });

  it("throws when dropping nonexistent", () => {
    expect(() => {
      dropAnalyzer("nonexistent");
    }).toThrow("not found");
  });
});
