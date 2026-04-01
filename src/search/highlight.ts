//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Search result highlighting for full-text queries.
//
// Provides term markup and fragment extraction for displaying search
// results with matched terms visually emphasized.
//
// The highlighter re-tokenizes the original text to locate word
// boundaries, runs each token through the same analysis pipeline used
// for indexing, and checks whether the analyzed form matches any query
// term.  Matched spans are wrapped with configurable tags.
//
// When max_fragments > 0, only the best snippets around matches are
// returned instead of the full text.

import { tokenizeFts, FTSParser } from "../sql/fts-query.js";
import type { FTSNode } from "../sql/fts-query.js";

const WORD_RE = /\w+/gu;

interface Analyzer {
  analyze(text: string): string[];
}

/**
 * Extract searchable terms from an FTS query string.
 *
 * Parses the query using the FTS grammar and collects all term and
 * phrase tokens, ignoring boolean operators and vector literals.
 */
export function extractQueryTerms(queryString: string): string[] {
  try {
    const tokens = tokenizeFts(queryString);
    const ast = new FTSParser(tokens).parse();
    const terms: string[] = [];
    collectTerms(ast, terms);
    return terms;
  } catch {
    // Fallback: treat the whole string as space-separated terms
    return queryString
      .split(/\s+/)
      .filter((t) => !["and", "or", "not"].includes(t.toLowerCase()));
  }
}

function collectTerms(node: FTSNode, out: string[]): void {
  switch (node.type) {
    case "term":
      out.push(node.term);
      break;
    case "phrase":
      for (const word of node.phrase.split(/\s+/)) {
        if (word.length > 0) out.push(word);
      }
      break;
    case "and":
      collectTerms(node.left, out);
      collectTerms(node.right, out);
      break;
    case "or":
      collectTerms(node.left, out);
      collectTerms(node.right, out);
      break;
    case "not":
      collectTerms(node.operand, out);
      break;
    // VectorNode: no text terms to collect
  }
}

/**
 * Highlight query terms in the original text.
 *
 * Tokenizes text to find word boundaries, checks each token against
 * queryTerms after applying the analyzer (or simple lowercasing
 * when no analyzer is provided), and wraps matching spans with
 * startTag / endTag.
 *
 * When maxFragments > 0, extracts the best fragments around matches
 * instead of returning the full text.  Each fragment is at most
 * fragmentSize characters.
 *
 * Returns the original text unmodified when queryTerms is empty.
 */
export function highlight(
  text: string | null | undefined,
  queryTerms: string[],
  options?: {
    startTag?: string;
    endTag?: string;
    maxFragments?: number;
    fragmentSize?: number;
    analyzer?: Analyzer | null;
  },
): string {
  if (!text || queryTerms.length === 0) {
    return text || "";
  }

  const startTag = options?.startTag ?? "<b>";
  const endTag = options?.endTag ?? "</b>";
  const maxFragments = options?.maxFragments ?? 0;
  const fragmentSize = options?.fragmentSize ?? 150;
  const analyzer = options?.analyzer ?? null;

  // Build the set of analyzed query terms for matching
  const analyzedTerms = new Set<string>();
  if (analyzer !== null) {
    for (const qt of queryTerms) {
      for (const t of analyzer.analyze(qt)) {
        analyzedTerms.add(t);
      }
    }
  } else {
    for (const qt of queryTerms) {
      analyzedTerms.add(qt.toLowerCase());
    }
  }

  if (analyzedTerms.size === 0) {
    return text;
  }

  // Tokenize the original text to get (start, end) character offsets
  const matchSpans: [number, number][] = [];
  // Reset the regex (global flag means lastIndex persists)
  WORD_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = WORD_RE.exec(text)) !== null) {
    const token = m[0]!;
    if (analyzer !== null) {
      const analyzed = analyzer.analyze(token);
      if (analyzed.length > 0 && analyzed.some((t) => analyzedTerms.has(t))) {
        matchSpans.push([m.index, m.index + token.length]);
      }
    } else {
      if (analyzedTerms.has(token.toLowerCase())) {
        matchSpans.push([m.index, m.index + token.length]);
      }
    }
  }

  if (matchSpans.length === 0) {
    if (maxFragments > 0) {
      const end = Math.min(text.length, fragmentSize);
      const suffix = end < text.length ? "..." : "";
      return text.slice(0, end) + suffix;
    }
    return text;
  }

  if (maxFragments > 0) {
    return buildFragments(text, matchSpans, startTag, endTag, maxFragments, fragmentSize);
  }

  return buildFullHighlight(text, matchSpans, startTag, endTag);
}

/**
 * Build the full text with matched spans wrapped in tags.
 */
function buildFullHighlight(
  text: string,
  matchSpans: [number, number][],
  startTag: string,
  endTag: string,
): string {
  const parts: string[] = [];
  let prevEnd = 0;
  for (const [start, end] of matchSpans) {
    parts.push(text.slice(prevEnd, start));
    parts.push(startTag);
    parts.push(text.slice(start, end));
    parts.push(endTag);
    prevEnd = end;
  }
  parts.push(text.slice(prevEnd));
  return parts.join("");
}

/**
 * Extract and highlight the best fragments around matches.
 */
function buildFragments(
  text: string,
  matchSpans: [number, number][],
  startTag: string,
  endTag: string,
  maxFragments: number,
  fragmentSize: number,
): string {
  const half = Math.floor(fragmentSize / 2);

  // Group nearby matches into clusters
  const clusters: [number, number][][] = [];
  let current: [number, number][] = [];
  for (const span of matchSpans) {
    if (current.length > 0 && span[0] - current[current.length - 1]![1] > half) {
      clusters.push(current);
      current = [span];
    } else {
      current.push(span);
    }
  }
  if (current.length > 0) {
    clusters.push(current);
  }

  // Select top clusters by match density, then sort by position
  clusters.sort((a, b) => b.length - a.length);
  const selected = clusters.slice(0, maxFragments);
  selected.sort((a, b) => a[0]![0] - b[0]![0]);

  const fragments: string[] = [];
  for (const cluster of selected) {
    const center = Math.floor((cluster[0]![0] + cluster[cluster.length - 1]![1]) / 2);
    let fragStart = Math.max(0, center - half);
    let fragEnd = Math.min(text.length, center + half);

    // Snap to word boundaries
    if (fragStart > 0) {
      const space = text.indexOf(" ", fragStart);
      if (space !== -1 && space < fragStart + 30) {
        fragStart = space + 1;
      }
    }
    if (fragEnd < text.length) {
      const space = text.lastIndexOf(" ", fragEnd);
      if (space !== -1 && space >= fragEnd - 30) {
        fragEnd = space;
      }
    }

    // Collect match spans within this fragment
    const fragMatches: [number, number][] = [];
    for (const [s, e] of cluster) {
      if (s >= fragStart && e <= fragEnd) {
        fragMatches.push([s - fragStart, e - fragStart]);
      }
    }
    const fragText = text.slice(fragStart, fragEnd);
    const highlighted = buildFullHighlight(fragText, fragMatches, startTag, endTag);

    const prefix = fragStart > 0 ? "..." : "";
    const suffix = fragEnd < text.length ? "..." : "";
    fragments.push(`${prefix}${highlighted}${suffix}`);
  }

  return fragments.join(" ");
}
