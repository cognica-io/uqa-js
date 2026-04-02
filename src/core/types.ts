//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- core type definitions
// 1:1 port of uqa/core/types.py

// -- Type aliases ------------------------------------------------------------

export type DocId = number;
export type FieldName = string;
export type TermValue = string;
export type PathExpr = ReadonlyArray<string | number>;

// -- Frozen data interfaces --------------------------------------------------

export interface Payload {
  readonly positions: readonly number[];
  readonly score: number;
  readonly fields: Readonly<Record<FieldName, unknown>>;
}

export interface PostingEntry {
  readonly docId: DocId;
  readonly payload: Payload;
}

export interface GeneralizedPostingEntry {
  readonly docIds: readonly DocId[];
  readonly payload: Payload;
}

export interface Vertex {
  readonly vertexId: number;
  readonly label: string;
  readonly properties: Readonly<Record<string, unknown>>;
}

export interface Edge {
  readonly edgeId: number;
  readonly sourceId: number;
  readonly targetId: number;
  readonly label: string;
  readonly properties: Readonly<Record<string, unknown>>;
}

// -- Factory functions (replace Python dataclass defaults) --------------------

export function createPayload(
  opts?: Partial<Pick<Payload, "positions" | "score" | "fields">>,
): Payload {
  return {
    positions: opts?.positions ?? [],
    score: opts?.score ?? 0.0,
    fields: opts?.fields ?? {},
  };
}

export function createPostingEntry(
  docId: DocId,
  payload?: Partial<Pick<Payload, "positions" | "score" | "fields">>,
): PostingEntry {
  return { docId, payload: createPayload(payload) };
}

export function createGeneralizedPostingEntry(
  docIds: readonly DocId[],
  payload?: Partial<Pick<Payload, "positions" | "score" | "fields">>,
): GeneralizedPostingEntry {
  return { docIds, payload: createPayload(payload) };
}

export function createVertex(
  vertexId: number,
  label: string,
  properties?: Record<string, unknown>,
): Vertex {
  return { vertexId, label, properties: properties ?? {} };
}

export function createEdge(
  edgeId: number,
  sourceId: number,
  targetId: number,
  label: string,
  properties?: Record<string, unknown>,
): Edge {
  return { edgeId, sourceId, targetId, label, properties: properties ?? {} };
}

// -- IndexStats (mutable) ----------------------------------------------------

export class IndexStats {
  totalDocs: number;
  avgDocLength: number;
  dimensions: number;
  private readonly _docFreqs: Map<string, number>;

  constructor(totalDocs = 0, avgDocLength = 0.0, dimensions = 0) {
    this.totalDocs = totalDocs;
    this.avgDocLength = avgDocLength;
    this.dimensions = dimensions;
    this._docFreqs = new Map();
  }

  docFreq(field: string, term: string): number {
    return this._docFreqs.get(`${field}\0${term}`) ?? 0;
  }

  setDocFreq(field: string, term: string, freq: number): void {
    this._docFreqs.set(`${field}\0${term}`, freq);
  }
}

// -- Predicate system --------------------------------------------------------

/**
 * Coerce mismatched date/string types for comparison.
 *
 * When one side is a Date object and the other is a string,
 * parse the string to match so comparisons work correctly.
 */
function _coerceForComparison(value: unknown, target: unknown): [unknown, unknown] {
  if (value instanceof Date && typeof target === "string") {
    try {
      const parsed = new Date(target);
      if (!isNaN(parsed.getTime())) return [value, parsed];
    } catch {
      // fall through
    }
  } else if (target instanceof Date && typeof value === "string") {
    try {
      const parsed = new Date(value);
      if (!isNaN(parsed.getTime())) return [parsed, target];
    } catch {
      // fall through
    }
  }
  return [value, target];
}

export abstract class Predicate {
  abstract evaluate(value: unknown): boolean;
}

export class Equals extends Predicate {
  constructor(readonly target: unknown) {
    super();
  }
  evaluate(value: unknown): boolean {
    const [v, t] = _coerceForComparison(value, this.target);
    if (v instanceof Date && t instanceof Date) return v.getTime() === t.getTime();
    return v === t;
  }
}

export class NotEquals extends Predicate {
  constructor(readonly target: unknown) {
    super();
  }
  evaluate(value: unknown): boolean {
    const [v, t] = _coerceForComparison(value, this.target);
    if (v instanceof Date && t instanceof Date) return v.getTime() !== t.getTime();
    return v !== t;
  }
}

export class GreaterThan extends Predicate {
  constructor(readonly target: number) {
    super();
  }
  evaluate(value: unknown): boolean {
    const [v, t] = _coerceForComparison(value, this.target);
    if (v instanceof Date && t instanceof Date) return v.getTime() > t.getTime();
    return (v as number) > (t as number);
  }
}

export class GreaterThanOrEqual extends Predicate {
  constructor(readonly target: number) {
    super();
  }
  evaluate(value: unknown): boolean {
    const [v, t] = _coerceForComparison(value, this.target);
    if (v instanceof Date && t instanceof Date) return v.getTime() >= t.getTime();
    return (v as number) >= (t as number);
  }
}

export class LessThan extends Predicate {
  constructor(readonly target: number) {
    super();
  }
  evaluate(value: unknown): boolean {
    const [v, t] = _coerceForComparison(value, this.target);
    if (v instanceof Date && t instanceof Date) return v.getTime() < t.getTime();
    return (v as number) < (t as number);
  }
}

export class LessThanOrEqual extends Predicate {
  constructor(readonly target: number) {
    super();
  }
  evaluate(value: unknown): boolean {
    const [v, t] = _coerceForComparison(value, this.target);
    if (v instanceof Date && t instanceof Date) return v.getTime() <= t.getTime();
    return (v as number) <= (t as number);
  }
}

export class InSet extends Predicate {
  private readonly _values: Set<unknown>;
  constructor(values: Iterable<unknown>) {
    super();
    this._values = new Set(values);
  }
  get values(): ReadonlySet<unknown> {
    return this._values;
  }
  evaluate(value: unknown): boolean {
    return this._values.has(value);
  }
}

export class Between extends Predicate {
  constructor(
    readonly low: number,
    readonly high: number,
  ) {
    super();
  }
  evaluate(value: unknown): boolean {
    const v = value as number;
    return v >= this.low && v <= this.high;
  }
}

export class IsNull extends Predicate {
  evaluate(value: unknown): boolean {
    return value === null || value === undefined;
  }
}

export class IsNotNull extends Predicate {
  evaluate(value: unknown): boolean {
    return value !== null && value !== undefined;
  }
}

// -- LIKE predicates with regex cache ----------------------------------------

const likeRegexCache = new Map<string, RegExp>();

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function compileLikeRegex(pattern: string, caseSensitive: boolean): RegExp {
  const key = `${caseSensitive ? "s" : "i"}:${pattern}`;
  const cached = likeRegexCache.get(key);
  if (cached !== undefined) return cached;

  // Evict oldest if cache exceeds 256 entries
  if (likeRegexCache.size >= 256) {
    const firstKey = likeRegexCache.keys().next().value as string;
    likeRegexCache.delete(firstKey);
  }

  let regex = "^";
  for (let i = 0; i < pattern.length; i++) {
    const ch = pattern[i]!;
    if (ch === "%") {
      regex += ".*";
    } else if (ch === "_") {
      regex += ".";
    } else if (ch === "\\" && i + 1 < pattern.length) {
      i++;
      regex += escapeRegex(pattern[i]!);
    } else {
      regex += escapeRegex(ch);
    }
  }
  regex += "$";

  const flags = caseSensitive ? "s" : "is";
  const compiled = new RegExp(regex, flags);
  likeRegexCache.set(key, compiled);
  return compiled;
}

function likeMatch(value: string, pattern: string, caseSensitive: boolean): boolean {
  return compileLikeRegex(pattern, caseSensitive).test(value);
}

export class Like extends Predicate {
  constructor(readonly pattern: string) {
    super();
  }
  evaluate(value: unknown): boolean {
    return likeMatch(String(value), this.pattern, true);
  }
}

export class NotLike extends Predicate {
  constructor(readonly pattern: string) {
    super();
  }
  evaluate(value: unknown): boolean {
    return !likeMatch(String(value), this.pattern, true);
  }
}

export class ILike extends Predicate {
  constructor(readonly pattern: string) {
    super();
  }
  evaluate(value: unknown): boolean {
    return likeMatch(String(value), this.pattern, false);
  }
}

export class NotILike extends Predicate {
  constructor(readonly pattern: string) {
    super();
  }
  evaluate(value: unknown): boolean {
    return !likeMatch(String(value), this.pattern, false);
  }
}

// -- Helper ------------------------------------------------------------------

export function isNullPredicate(pred: Predicate): boolean {
  return pred instanceof IsNull || pred instanceof IsNotNull;
}
