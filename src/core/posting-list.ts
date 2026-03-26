//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- PostingList and GeneralizedPostingList
// 1:1 port of uqa/core/posting_list.py

import type { DocId, GeneralizedPostingEntry, Payload, PostingEntry } from "./types.js";

// -- Module-private helpers --------------------------------------------------

function compareDocIdArrays(a: readonly DocId[], b: readonly DocId[]): number {
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    if (a[i]! < b[i]!) return -1;
    if (a[i]! > b[i]!) return 1;
  }
  return a.length - b.length;
}

function docIdArraysEqual(a: readonly DocId[], b: readonly DocId[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function docIdArrayKey(ids: readonly DocId[]): string {
  return ids.join("\0");
}

// -- PostingList -------------------------------------------------------------

export class PostingList {
  private _entries: PostingEntry[];
  private _docIdsCache: Set<DocId> | null = null;

  constructor(entries?: PostingEntry[]) {
    if (entries && entries.length > 0) {
      const sorted = entries.slice().sort((a, b) => a.docId - b.docId);
      // Deduplicate: keep first occurrence of each docId
      const deduped: PostingEntry[] = [sorted[0]!];
      const seen = new Set<DocId>([sorted[0]!.docId]);
      for (let i = 1; i < sorted.length; i++) {
        const entry = sorted[i]!;
        if (!seen.has(entry.docId)) {
          seen.add(entry.docId);
          deduped.push(entry);
        }
      }
      this._entries = deduped;
    } else {
      this._entries = [];
    }
  }

  static fromSorted(entries: PostingEntry[]): PostingList {
    const pl = Object.create(PostingList.prototype) as PostingList;
    pl._entries = entries;
    pl._docIdsCache = null;
    return pl;
  }

  // -- Boolean algebra operations --------------------------------------------

  union(other: PostingList): PostingList {
    const a = this._entries;
    const b = other._entries;
    const result: PostingEntry[] = [];
    let i = 0;
    let j = 0;
    while (i < a.length && j < b.length) {
      const ea = a[i]!;
      const eb = b[j]!;
      if (ea.docId === eb.docId) {
        result.push({
          docId: ea.docId,
          payload: PostingList.mergePayloads(ea.payload, eb.payload),
        });
        i++;
        j++;
      } else if (ea.docId < eb.docId) {
        result.push(ea);
        i++;
      } else {
        result.push(eb);
        j++;
      }
    }
    while (i < a.length) {
      result.push(a[i]!);
      i++;
    }
    while (j < b.length) {
      result.push(b[j]!);
      j++;
    }
    return PostingList.fromSorted(result);
  }

  intersect(other: PostingList): PostingList {
    const a = this._entries;
    const b = other._entries;
    const result: PostingEntry[] = [];
    let i = 0;
    let j = 0;
    while (i < a.length && j < b.length) {
      const ea = a[i]!;
      const eb = b[j]!;
      if (ea.docId === eb.docId) {
        result.push({
          docId: ea.docId,
          payload: PostingList.mergePayloads(ea.payload, eb.payload),
        });
        i++;
        j++;
      } else if (ea.docId < eb.docId) {
        i++;
      } else {
        j++;
      }
    }
    return PostingList.fromSorted(result);
  }

  difference(other: PostingList): PostingList {
    const otherIds = other.docIds;
    const result = this._entries.filter((e) => !otherIds.has(e.docId));
    return PostingList.fromSorted(result);
  }

  complement(universal: PostingList): PostingList {
    return universal.difference(this);
  }

  // -- Payload merge ---------------------------------------------------------

  static mergePayloads(a: Payload, b: Payload): Payload {
    const posSet = new Set([...a.positions, ...b.positions]);
    const positions = [...posSet].sort((x, y) => x - y);
    const score = a.score + b.score;
    const fields = { ...a.fields, ...b.fields };
    return { positions, score, fields };
  }

  // -- Accessors -------------------------------------------------------------

  get docIds(): Set<DocId> {
    if (this._docIdsCache === null) {
      this._docIdsCache = new Set(this._entries.map((e) => e.docId));
    }
    return this._docIdsCache;
  }

  get entries(): PostingEntry[] {
    return this._entries;
  }

  getEntry(docId: DocId): PostingEntry | null {
    let lo = 0;
    let hi = this._entries.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >>> 1;
      const midId = this._entries[mid]!.docId;
      if (midId === docId) return this._entries[mid]!;
      if (midId < docId) {
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return null;
  }

  topK(k: number): PostingList {
    if (k >= this._entries.length) {
      return PostingList.fromSorted(this._entries.slice());
    }
    const sorted = this._entries
      .slice()
      .sort((a, b) => b.payload.score - a.payload.score);
    const top = sorted.slice(0, k);
    return new PostingList(top);
  }

  withScores(scoreFn: (entry: PostingEntry) => number): PostingList {
    const result: PostingEntry[] = [];
    for (const e of this._entries) {
      const newScore = scoreFn(e);
      result.push({
        docId: e.docId,
        payload: {
          positions: e.payload.positions,
          score: newScore,
          fields: e.payload.fields,
        },
      });
    }
    return PostingList.fromSorted(result);
  }

  get length(): number {
    return this._entries.length;
  }

  [Symbol.iterator](): Iterator<PostingEntry> {
    return this._entries[Symbol.iterator]();
  }

  equals(other: PostingList): boolean {
    if (this._entries.length !== other._entries.length) return false;
    for (let i = 0; i < this._entries.length; i++) {
      if (this._entries[i]!.docId !== other._entries[i]!.docId) return false;
    }
    return true;
  }

  toString(): string {
    const ids = this._entries.map((e) => String(e.docId)).join(", ");
    return `PostingList([${ids}])`;
  }

  // Convenience aliases (TS has no operator overloading)
  and(other: PostingList): PostingList {
    return this.intersect(other);
  }
  or(other: PostingList): PostingList {
    return this.union(other);
  }
  sub(other: PostingList): PostingList {
    return this.difference(other);
  }
}

// -- GeneralizedPostingList --------------------------------------------------

export class GeneralizedPostingList {
  private _entries: GeneralizedPostingEntry[];

  constructor(entries?: GeneralizedPostingEntry[]) {
    this._entries = (entries ?? [])
      .slice()
      .sort((a, b) => compareDocIdArrays(a.docIds, b.docIds));
  }

  static fromSorted(entries: GeneralizedPostingEntry[]): GeneralizedPostingList {
    const gpl = Object.create(
      GeneralizedPostingList.prototype,
    ) as GeneralizedPostingList;
    gpl._entries = entries;
    return gpl;
  }

  get entries(): GeneralizedPostingEntry[] {
    return this._entries.slice();
  }

  get length(): number {
    return this._entries.length;
  }

  [Symbol.iterator](): Iterator<GeneralizedPostingEntry> {
    return this._entries[Symbol.iterator]();
  }

  // -- Boolean algebra -------------------------------------------------------

  union(other: GeneralizedPostingList): GeneralizedPostingList {
    const a = this._entries;
    const b = other._entries;
    const result: GeneralizedPostingEntry[] = [];
    let i = 0;
    let j = 0;
    while (i < a.length && j < b.length) {
      const ea = a[i]!;
      const eb = b[j]!;
      const cmp = compareDocIdArrays(ea.docIds, eb.docIds);
      if (cmp === 0) {
        result.push(ea);
        i++;
        j++;
      } else if (cmp < 0) {
        result.push(ea);
        i++;
      } else {
        result.push(eb);
        j++;
      }
    }
    while (i < a.length) {
      result.push(a[i]!);
      i++;
    }
    while (j < b.length) {
      result.push(b[j]!);
      j++;
    }
    return GeneralizedPostingList.fromSorted(result);
  }

  intersect(other: GeneralizedPostingList): GeneralizedPostingList {
    const a = this._entries;
    const b = other._entries;
    const result: GeneralizedPostingEntry[] = [];
    let i = 0;
    let j = 0;
    while (i < a.length && j < b.length) {
      const ea = a[i]!;
      const eb = b[j]!;
      const cmp = compareDocIdArrays(ea.docIds, eb.docIds);
      if (cmp === 0) {
        result.push(ea);
        i++;
        j++;
      } else if (cmp < 0) {
        i++;
      } else {
        j++;
      }
    }
    return GeneralizedPostingList.fromSorted(result);
  }

  difference(other: GeneralizedPostingList): GeneralizedPostingList {
    const otherIds = other.docIdsSet;
    const result = this._entries.filter((e) => !otherIds.has(docIdArrayKey(e.docIds)));
    return GeneralizedPostingList.fromSorted(result);
  }

  complement(universal: GeneralizedPostingList): GeneralizedPostingList {
    return universal.difference(this);
  }

  get docIdsSet(): Set<string> {
    return new Set(this._entries.map((e) => docIdArrayKey(e.docIds)));
  }

  equals(other: GeneralizedPostingList): boolean {
    if (this._entries.length !== other._entries.length) return false;
    for (let i = 0; i < this._entries.length; i++) {
      if (!docIdArraysEqual(this._entries[i]!.docIds, other._entries[i]!.docIds)) {
        return false;
      }
    }
    return true;
  }

  toString(): string {
    const tuples = this._entries.map((e) => `(${e.docIds.join(", ")})`).join(", ");
    return `GeneralizedPostingList([${tuples}])`;
  }

  and(other: GeneralizedPostingList): GeneralizedPostingList {
    return this.intersect(other);
  }
  or(other: GeneralizedPostingList): GeneralizedPostingList {
    return this.union(other);
  }
  sub(other: GeneralizedPostingList): GeneralizedPostingList {
    return this.difference(other);
  }
}
