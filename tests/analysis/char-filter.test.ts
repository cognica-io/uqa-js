import { describe, expect, it } from "vitest";
import {
  CharFilter,
  HTMLStripCharFilter,
  MappingCharFilter,
  PatternReplaceCharFilter,
} from "../../src/analysis/char-filter.js";

describe("HTMLStripCharFilter", () => {
  it("removes HTML tags", () => {
    expect(new HTMLStripCharFilter().filter("<p>hello</p>")).toBe(" hello ");
  });

  it("replaces entities", () => {
    expect(new HTMLStripCharFilter().filter("a &amp; b")).toBe("a & b");
    expect(new HTMLStripCharFilter().filter("&lt;tag&gt;")).toBe("<tag>");
  });

  it("handles mixed content", () => {
    const result = new HTMLStripCharFilter().filter("<b>bold</b> &amp; <i>italic</i>");
    expect(result).toBe(" bold  &  italic ");
  });

  it("serializes/deserializes", () => {
    const d = new HTMLStripCharFilter().toDict();
    expect(d).toEqual({ type: "html_strip" });
    expect(CharFilter.fromDict(d)).toBeInstanceOf(HTMLStripCharFilter);
  });
});

describe("MappingCharFilter", () => {
  it("replaces mapped strings", () => {
    const f = new MappingCharFilter({ ":)": "happy", ":(": "sad" });
    expect(f.filter("I am :) today")).toBe("I am happy today");
  });

  it("longest match first", () => {
    const f = new MappingCharFilter({ ab: "X", abc: "Y" });
    // "abc" sorted first (longer), so "abc" -> "Y"
    expect(f.filter("abcd")).toBe("Yd");
  });

  it("serializes/deserializes", () => {
    const f = new MappingCharFilter({ a: "b" });
    const d = f.toDict();
    const restored = CharFilter.fromDict(d) as MappingCharFilter;
    expect(restored.filter("a")).toBe("b");
  });
});

describe("PatternReplaceCharFilter", () => {
  it("replaces pattern", () => {
    const f = new PatternReplaceCharFilter("\\d+", "#");
    expect(f.filter("abc123def456")).toBe("abc#def#");
  });

  it("defaults to empty replacement", () => {
    const f = new PatternReplaceCharFilter("\\d+");
    expect(f.filter("abc123")).toBe("abc");
  });

  it("serializes/deserializes", () => {
    const f = new PatternReplaceCharFilter("x", "y");
    const d = f.toDict();
    expect(d).toEqual({ type: "pattern_replace", pattern: "x", replacement: "y" });
    const restored = CharFilter.fromDict(d) as PatternReplaceCharFilter;
    expect(restored.filter("xax")).toBe("yay");
  });
});

describe("CharFilter.fromDict", () => {
  it("throws for unknown type", () => {
    expect(() => CharFilter.fromDict({ type: "unknown" })).toThrow(
      "Unknown CharFilter type",
    );
  });
});
