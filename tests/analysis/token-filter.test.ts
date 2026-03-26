import { describe, expect, it } from "vitest";
import {
  ASCIIFoldingFilter,
  EdgeNGramFilter,
  LengthFilter,
  LowerCaseFilter,
  NGramFilter,
  PorterStemFilter,
  StopWordFilter,
  SynonymFilter,
  TokenFilter,
} from "../../src/analysis/token-filter.js";

describe("LowerCaseFilter", () => {
  it("lowercases tokens", () => {
    expect(new LowerCaseFilter().filter(["Hello", "WORLD"])).toEqual([
      "hello",
      "world",
    ]);
  });

  it("serializes/deserializes", () => {
    const d = new LowerCaseFilter().toDict();
    expect(TokenFilter.fromDict(d)).toBeInstanceOf(LowerCaseFilter);
  });
});

describe("StopWordFilter", () => {
  it("removes English stop words", () => {
    const f = new StopWordFilter();
    expect(f.filter(["the", "quick", "brown", "fox"])).toEqual([
      "quick",
      "brown",
      "fox",
    ]);
  });

  it("accepts custom words", () => {
    const f = new StopWordFilter("english", new Set(["custom"]));
    expect(f.filter(["custom", "hello"])).toEqual(["hello"]);
  });

  it("serializes/deserializes with custom words", () => {
    const f = new StopWordFilter("english", new Set(["extra"]));
    const d = f.toDict();
    expect(d["custom_words"]).toEqual(["extra"]);
    const restored = TokenFilter.fromDict(d) as StopWordFilter;
    expect(restored.filter(["extra", "hello"])).toEqual(["hello"]);
  });
});

describe("PorterStemFilter", () => {
  it("stems common words", () => {
    const f = new PorterStemFilter();
    expect(f.filter(["running"])).toEqual(["run"]);
    expect(f.filter(["caresses"])).toEqual(["caress"]);
    expect(f.filter(["ponies"])).toEqual(["poni"]);
    expect(f.filter(["cats"])).toEqual(["cat"]);
  });

  it("handles short words unchanged", () => {
    const f = new PorterStemFilter();
    expect(f.filter(["a"])).toEqual(["a"]);
    expect(f.filter(["is"])).toEqual(["is"]);
  });

  it("stems -eed words", () => {
    const f = new PorterStemFilter();
    expect(f.filter(["agreed"])).toEqual(["agre"]);
    expect(f.filter(["feed"])).toEqual(["feed"]);
  });

  it("stems -ed words", () => {
    const f = new PorterStemFilter();
    expect(f.filter(["plastered"])).toEqual(["plaster"]);
    expect(f.filter(["bled"])).toEqual(["bled"]);
  });

  it("stems -ing words", () => {
    const f = new PorterStemFilter();
    expect(f.filter(["motoring"])).toEqual(["motor"]);
    expect(f.filter(["sing"])).toEqual(["sing"]);
  });

  it("handles step 1c (y -> i)", () => {
    const f = new PorterStemFilter();
    expect(f.filter(["happy"])).toEqual(["happi"]);
  });

  it("handles step 2 suffixes", () => {
    const f = new PorterStemFilter();
    expect(f.filter(["relational"])).toEqual(["relat"]);
    expect(f.filter(["conditional"])).toEqual(["condit"]);
    expect(f.filter(["formalization"])).toEqual(["formal"]);
  });

  it("handles step 3 suffixes", () => {
    const f = new PorterStemFilter();
    expect(f.filter(["triplicate"])).toEqual(["triplic"]);
    expect(f.filter(["hopeful"])).toEqual(["hope"]);
    expect(f.filter(["goodness"])).toEqual(["good"]);
  });

  it("handles step 4 suffixes", () => {
    const f = new PorterStemFilter();
    expect(f.filter(["revival"])).toEqual(["reviv"]);
    expect(f.filter(["allowance"])).toEqual(["allow"]);
    expect(f.filter(["adoption"])).toEqual(["adopt"]);
  });

  it("serializes/deserializes", () => {
    const d = new PorterStemFilter().toDict();
    expect(TokenFilter.fromDict(d)).toBeInstanceOf(PorterStemFilter);
  });
});

describe("ASCIIFoldingFilter", () => {
  it("folds accented characters", () => {
    const f = new ASCIIFoldingFilter();
    expect(f.filter(["cafe"])).toEqual(["cafe"]);
    expect(f.filter(["caf\u00e9"])).toEqual(["cafe"]);
    expect(f.filter(["na\u00efve"])).toEqual(["naive"]);
  });

  it("leaves ASCII unchanged", () => {
    const f = new ASCIIFoldingFilter();
    expect(f.filter(["hello"])).toEqual(["hello"]);
  });

  it("serializes/deserializes", () => {
    const d = new ASCIIFoldingFilter().toDict();
    expect(TokenFilter.fromDict(d)).toBeInstanceOf(ASCIIFoldingFilter);
  });
});

describe("SynonymFilter", () => {
  it("expands synonyms", () => {
    const f = new SynonymFilter({ car: ["automobile", "vehicle"] });
    expect(f.filter(["car", "red"])).toEqual(["car", "automobile", "vehicle", "red"]);
  });

  it("passes through unknown tokens", () => {
    const f = new SynonymFilter({ car: ["automobile"] });
    expect(f.filter(["bike"])).toEqual(["bike"]);
  });

  it("serializes/deserializes", () => {
    const f = new SynonymFilter({ a: ["b"] });
    const d = f.toDict();
    expect(d).toEqual({ type: "synonym", synonyms: { a: ["b"] } });
    const restored = TokenFilter.fromDict(d) as SynonymFilter;
    expect(restored.filter(["a"])).toEqual(["a", "b"]);
  });
});

describe("NGramFilter", () => {
  it("generates ngrams from tokens", () => {
    const f = new NGramFilter(2, 3);
    expect(f.filter(["abc"])).toEqual(["ab", "bc", "abc"]);
  });

  it("skips short tokens by default", () => {
    const f = new NGramFilter(3, 3);
    expect(f.filter(["ab"])).toEqual([]);
  });

  it("keeps short tokens when enabled", () => {
    const f = new NGramFilter(3, 3, true);
    expect(f.filter(["ab"])).toEqual(["ab"]);
  });

  it("serializes/deserializes", () => {
    const f = new NGramFilter(2, 4, true);
    const d = f.toDict();
    expect(d["keep_short"]).toBe(true);
    const restored = TokenFilter.fromDict(d) as NGramFilter;
    expect(restored.filter(["a"])).toEqual(["a"]);
  });
});

describe("EdgeNGramFilter", () => {
  it("generates edge ngrams", () => {
    const f = new EdgeNGramFilter(1, 3);
    expect(f.filter(["hello"])).toEqual(["h", "he", "hel"]);
  });

  it("respects token length", () => {
    const f = new EdgeNGramFilter(1, 10);
    expect(f.filter(["ab"])).toEqual(["a", "ab"]);
  });

  it("serializes/deserializes", () => {
    const d = new EdgeNGramFilter(2, 5).toDict();
    const restored = TokenFilter.fromDict(d) as EdgeNGramFilter;
    expect(restored.filter(["abcde"])).toEqual(["ab", "abc", "abcd", "abcde"]);
  });
});

describe("LengthFilter", () => {
  it("filters by minimum length", () => {
    const f = new LengthFilter(3, 0);
    expect(f.filter(["a", "ab", "abc", "abcd"])).toEqual(["abc", "abcd"]);
  });

  it("filters by maximum length", () => {
    const f = new LengthFilter(0, 3);
    expect(f.filter(["a", "ab", "abc", "abcd"])).toEqual(["a", "ab", "abc"]);
  });

  it("filters by both", () => {
    const f = new LengthFilter(2, 3);
    expect(f.filter(["a", "ab", "abc", "abcd"])).toEqual(["ab", "abc"]);
  });

  it("serializes/deserializes", () => {
    const d = new LengthFilter(2, 5).toDict();
    expect(TokenFilter.fromDict(d)).toBeInstanceOf(LengthFilter);
  });
});

describe("TokenFilter.fromDict", () => {
  it("throws for unknown type", () => {
    expect(() => TokenFilter.fromDict({ type: "unknown" })).toThrow(
      "Unknown TokenFilter type",
    );
  });
});
