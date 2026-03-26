import { describe, expect, it } from "vitest";
import {
  KeywordTokenizer,
  LetterTokenizer,
  NGramTokenizer,
  PatternTokenizer,
  StandardTokenizer,
  Tokenizer,
  WhitespaceTokenizer,
} from "../../src/analysis/tokenizer.js";

describe("WhitespaceTokenizer", () => {
  it("splits on whitespace", () => {
    expect(new WhitespaceTokenizer().tokenize("hello world")).toEqual([
      "hello",
      "world",
    ]);
  });

  it("handles multiple spaces", () => {
    expect(new WhitespaceTokenizer().tokenize("  a  b  ")).toEqual(["a", "b"]);
  });

  it("returns empty for empty string", () => {
    expect(new WhitespaceTokenizer().tokenize("")).toEqual([]);
  });

  it("serializes/deserializes", () => {
    const t = new WhitespaceTokenizer();
    const d = t.toDict();
    expect(d).toEqual({ type: "whitespace" });
    const restored = Tokenizer.fromDict(d);
    expect(restored).toBeInstanceOf(WhitespaceTokenizer);
  });
});

describe("StandardTokenizer", () => {
  it("extracts word characters", () => {
    expect(new StandardTokenizer().tokenize("hello, world!")).toEqual([
      "hello",
      "world",
    ]);
  });

  it("handles numbers", () => {
    expect(new StandardTokenizer().tokenize("test123 foo")).toEqual(["test123", "foo"]);
  });

  it("serializes/deserializes", () => {
    const d = new StandardTokenizer().toDict();
    expect(Tokenizer.fromDict(d)).toBeInstanceOf(StandardTokenizer);
  });
});

describe("LetterTokenizer", () => {
  it("extracts ASCII letters only", () => {
    expect(new LetterTokenizer().tokenize("hello123world")).toEqual(["hello", "world"]);
  });

  it("serializes/deserializes", () => {
    const d = new LetterTokenizer().toDict();
    expect(Tokenizer.fromDict(d)).toBeInstanceOf(LetterTokenizer);
  });
});

describe("NGramTokenizer", () => {
  it("generates ngrams", () => {
    const t = new NGramTokenizer(2, 3);
    expect(t.tokenize("abc")).toEqual(["ab", "bc", "abc"]);
  });

  it("handles multiple words", () => {
    const t = new NGramTokenizer(2, 2);
    expect(t.tokenize("ab cd")).toEqual(["ab", "cd"]);
  });

  it("validates params", () => {
    expect(() => new NGramTokenizer(0, 2)).toThrow("minGram must be >= 1");
    expect(() => new NGramTokenizer(3, 2)).toThrow("maxGram must be >= minGram");
  });

  it("serializes/deserializes", () => {
    const t = new NGramTokenizer(2, 4);
    const restored = Tokenizer.fromDict(t.toDict()) as NGramTokenizer;
    expect(restored.tokenize("abcd")).toEqual(t.tokenize("abcd"));
  });
});

describe("PatternTokenizer", () => {
  it("splits on pattern", () => {
    const t = new PatternTokenizer(",\\s*");
    expect(t.tokenize("a, b, c")).toEqual(["a", "b", "c"]);
  });

  it("filters empty tokens", () => {
    const t = new PatternTokenizer("\\W+");
    expect(t.tokenize("hello world")).toEqual(["hello", "world"]);
  });

  it("serializes/deserializes", () => {
    const t = new PatternTokenizer(",");
    const d = t.toDict();
    expect(d).toEqual({ type: "pattern", pattern: "," });
    const restored = Tokenizer.fromDict(d) as PatternTokenizer;
    expect(restored.tokenize("a,b")).toEqual(["a", "b"]);
  });
});

describe("KeywordTokenizer", () => {
  it("returns entire input as single token", () => {
    expect(new KeywordTokenizer().tokenize("hello world")).toEqual(["hello world"]);
  });

  it("returns empty for empty string", () => {
    expect(new KeywordTokenizer().tokenize("")).toEqual([]);
  });

  it("serializes/deserializes", () => {
    const d = new KeywordTokenizer().toDict();
    expect(Tokenizer.fromDict(d)).toBeInstanceOf(KeywordTokenizer);
  });
});

describe("Tokenizer.fromDict", () => {
  it("throws for unknown type", () => {
    expect(() => Tokenizer.fromDict({ type: "unknown" })).toThrow(
      "Unknown Tokenizer type",
    );
  });
});
