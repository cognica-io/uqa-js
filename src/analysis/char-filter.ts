//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- character filters
// 1:1 port of uqa/analysis/char_filter.py

export abstract class CharFilter {
  abstract filter(text: string): string;
  abstract toDict(): Record<string, unknown>;

  static fromDict(d: Record<string, unknown>): CharFilter {
    const type = d["type"] as string;
    switch (type) {
      case "html_strip":
        return HTMLStripCharFilter._fromDict(d);
      case "mapping":
        return MappingCharFilter._fromDict(d);
      case "pattern_replace":
        return PatternReplaceCharFilter._fromDict(d);
      default:
        throw new Error(`Unknown CharFilter type: ${type}`);
    }
  }
}

// -- HTMLStripCharFilter ------------------------------------------------------

const HTML_TAG_RE = /<[^>]+>/g;
const HTML_ENTITIES: Record<string, string> = {
  "&amp;": "&",
  "&lt;": "<",
  "&gt;": ">",
  "&quot;": '"',
  "&#39;": "'",
  "&apos;": "'",
  "&nbsp;": " ",
};

export class HTMLStripCharFilter extends CharFilter {
  filter(text: string): string {
    let result = text.replace(HTML_TAG_RE, " ");
    for (const [entity, char] of Object.entries(HTML_ENTITIES)) {
      result = result.replaceAll(entity, char);
    }
    return result;
  }

  toDict(): Record<string, unknown> {
    return { type: "html_strip" };
  }

  static _fromDict(_d: Record<string, unknown>): HTMLStripCharFilter {
    return new HTMLStripCharFilter();
  }
}

// -- MappingCharFilter --------------------------------------------------------

export class MappingCharFilter extends CharFilter {
  private readonly _mapping: [string, string][];

  constructor(mapping: Record<string, string>) {
    super();
    // Sort by key length descending (longest-match-first)
    this._mapping = Object.entries(mapping).sort((a, b) => b[0].length - a[0].length);
  }

  filter(text: string): string {
    let result = text;
    for (const [key, value] of this._mapping) {
      result = result.replaceAll(key, value);
    }
    return result;
  }

  toDict(): Record<string, unknown> {
    const mapping: Record<string, string> = {};
    for (const [k, v] of this._mapping) {
      mapping[k] = v;
    }
    return { type: "mapping", mapping };
  }

  static _fromDict(d: Record<string, unknown>): MappingCharFilter {
    return new MappingCharFilter(d["mapping"] as Record<string, string>);
  }
}

// -- PatternReplaceCharFilter -------------------------------------------------

export class PatternReplaceCharFilter extends CharFilter {
  private readonly _pattern: string;
  private readonly _replacement: string;
  private readonly _re: RegExp;

  constructor(pattern: string, replacement = "") {
    super();
    this._pattern = pattern;
    this._replacement = replacement;
    this._re = new RegExp(pattern, "g");
  }

  filter(text: string): string {
    return text.replace(this._re, this._replacement);
  }

  toDict(): Record<string, unknown> {
    return {
      type: "pattern_replace",
      pattern: this._pattern,
      replacement: this._replacement,
    };
  }

  static _fromDict(d: Record<string, unknown>): PatternReplaceCharFilter {
    return new PatternReplaceCharFilter(
      d["pattern"] as string,
      (d["replacement"] as string | undefined) ?? "",
    );
  }
}
