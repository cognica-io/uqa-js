//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- WAND scorer
// 1:1 port of uqa/scoring/wand.py

import type { DocId, PostingEntry } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { InvertedIndex } from "../storage/abc/inverted-index.js";

// Scorer interface: any object with score() and upperBound() methods
export interface WANDScorerLike {
  score(termFreq: number, docLength: number, docFreq: number): number;
  upperBound(docFreq: number): number;
}

// Binary search: find first position with docId >= target
function advanceCursor(entries: PostingEntry[], pos: number, target: DocId): number {
  let lo = pos;
  let hi = entries.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (entries[mid]!.docId < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

interface TermCursor {
  termIdx: number;
  docId: DocId;
}

export class WANDScorer {
  protected readonly _scorers: WANDScorerLike[];
  protected readonly _k: number;
  protected readonly _postingLists: PostingList[];
  protected readonly _invertedIndex: InvertedIndex | null;
  protected readonly _fields: string[];
  protected readonly _terms: string[];
  protected readonly _upperBounds: number[];
  private readonly _entries: PostingEntry[][];
  private readonly _cursors: number[];

  constructor(
    scorers: WANDScorerLike[],
    k: number,
    postingLists: PostingList[],
    invertedIndex?: InvertedIndex | null,
    fields?: string[] | null,
    terms?: string[] | null,
  ) {
    this._scorers = scorers;
    this._k = k;
    this._postingLists = postingLists;
    this._invertedIndex = invertedIndex ?? null;
    this._fields = fields ?? scorers.map(() => "");
    this._terms = terms ?? scorers.map(() => "");
    this._entries = postingLists.map((pl) => pl.entries);
    this._cursors = new Array(scorers.length).fill(0) as number[];
    this._upperBounds = this._computeUpperBounds();
  }

  protected _computeUpperBounds(): number[] {
    return this._scorers.map((scorer, i) =>
      scorer.upperBound(this._postingLists[i]!.length),
    );
  }

  scoreTopK(): PostingList {
    const n = this._scorers.length;
    // Min-heap for top-k: [score, docId]
    const heap: [number, DocId][] = [];
    let threshold = 0;

    // Build sorted term list by current docId
    const sortedTerms: TermCursor[] = [];
    for (let i = 0; i < n; i++) {
      if (this._cursors[i]! < this._entries[i]!.length) {
        sortedTerms.push({
          termIdx: i,
          docId: this._entries[i]![this._cursors[i]!]!.docId,
        });
      }
    }
    sortedTerms.sort((a, b) => a.docId - b.docId);

    while (sortedTerms.length > 0) {
      // Find pivot: first position where cumulative UB >= threshold
      let cumUB = 0;
      let pivotPos = -1;
      for (let p = 0; p < sortedTerms.length; p++) {
        cumUB += this._upperBounds[sortedTerms[p]!.termIdx]!;
        if (cumUB >= threshold) {
          pivotPos = p;
          break;
        }
      }
      if (pivotPos === -1) break;

      const pivotDocId = sortedTerms[pivotPos]!.docId;

      // Check if all terms before pivot point to same doc
      if (sortedTerms[0]!.docId === pivotDocId) {
        // Score document
        let totalScore = 0;
        for (let p = 0; p <= pivotPos; p++) {
          const ti = sortedTerms[p]!.termIdx;
          const entry = this._entries[ti]![this._cursors[ti]!]!;
          const tf =
            entry.payload.positions.length > 0 ? entry.payload.positions.length : 1;
          let dl = tf;
          if (this._invertedIndex && this._fields[ti]) {
            dl = this._invertedIndex.getDocLength(pivotDocId, this._fields[ti]);
            if (dl === 0) dl = tf;
          }
          const df = this._entries[ti]!.length;
          totalScore += this._scorers[ti]!.score(tf, dl, df);
        }

        // Update heap
        if (heap.length < this._k) {
          heapPush(heap, [totalScore, pivotDocId]);
          if (heap.length === this._k) {
            threshold = heap[0]![0];
          }
        } else if (totalScore > threshold) {
          heapReplace(heap, [totalScore, pivotDocId]);
          threshold = heap[0]![0];
        }

        // Advance all cursors on pivot doc
        for (let p = pivotPos; p >= 0; p--) {
          const ti = sortedTerms[p]!.termIdx;
          this._cursors[ti] = this._cursors[ti]! + 1;
          if (this._cursors[ti] < this._entries[ti]!.length) {
            sortedTerms[p]!.docId = this._entries[ti]![this._cursors[ti]]!.docId;
          } else {
            sortedTerms.splice(p, 1);
          }
        }
        sortedTerms.sort((a, b) => a.docId - b.docId);
      } else {
        // Advance first term cursor to pivot docId
        const ti = sortedTerms[0]!.termIdx;
        this._cursors[ti] = advanceCursor(
          this._entries[ti]!,
          this._cursors[ti]!,
          pivotDocId,
        );
        if (this._cursors[ti] < this._entries[ti]!.length) {
          sortedTerms[0]!.docId = this._entries[ti]![this._cursors[ti]]!.docId;
        } else {
          sortedTerms.shift();
        }
        sortedTerms.sort((a, b) => a.docId - b.docId);
      }
    }

    // Build result
    const result = heap.map(([score, docId]) => ({
      docId,
      payload: createPayload({ score }),
    }));
    return new PostingList(result);
  }
}

// -- BlockMaxWANDScorer -------------------------------------------------------

export class BlockMaxWANDScorer {
  private readonly _scorers: WANDScorerLike[];
  private readonly _k: number;
  private readonly _blockMaxIndex: {
    getBlockMax(
      field: string,
      term: string,
      blockIdx: number,
      tableName?: string,
    ): number;
  };
  private readonly _postingLists: PostingList[];
  private readonly _invertedIndex: InvertedIndex | null;
  private readonly _fields: string[];
  private readonly _terms: string[];
  private readonly _blockSize: number;
  private readonly _tableName: string;

  constructor(
    scorers: WANDScorerLike[],
    k: number,
    blockMaxIndex: {
      getBlockMax(
        field: string,
        term: string,
        blockIdx: number,
        tableName?: string,
      ): number;
    },
    postingLists: PostingList[],
    invertedIndex?: InvertedIndex | null,
    fields?: string[] | null,
    terms?: string[] | null,
    blockSize = 128,
    tableName = "",
  ) {
    this._scorers = scorers;
    this._k = k;
    this._blockMaxIndex = blockMaxIndex;
    this._postingLists = postingLists;
    this._invertedIndex = invertedIndex ?? null;
    this._fields = fields ?? scorers.map(() => "");
    this._terms = terms ?? scorers.map(() => "");
    this._blockSize = blockSize;
    this._tableName = tableName;
  }

  private _getBlockIdx(position: number): number {
    return Math.floor(position / this._blockSize);
  }

  scoreTopK(): PostingList {
    const n = this._scorers.length;
    const entries = this._postingLists.map((pl) => pl.entries);
    const cursors = new Array(n).fill(0) as number[];
    const heap: [number, DocId][] = [];
    let threshold = 0;

    const sortedTerms: TermCursor[] = [];
    for (let i = 0; i < n; i++) {
      if (entries[i]!.length > 0) {
        sortedTerms.push({ termIdx: i, docId: entries[i]![0]!.docId });
      }
    }
    sortedTerms.sort((a, b) => a.docId - b.docId);

    while (sortedTerms.length > 0) {
      // Use block-max upper bounds
      let cumUB = 0;
      let pivotPos = -1;
      for (let p = 0; p < sortedTerms.length; p++) {
        const ti = sortedTerms[p]!.termIdx;
        const blockIdx = this._getBlockIdx(cursors[ti]!);
        const bmax = this._blockMaxIndex.getBlockMax(
          this._fields[ti]!,
          this._terms[ti]!,
          blockIdx,
          this._tableName,
        );
        cumUB += bmax;
        if (cumUB >= threshold) {
          pivotPos = p;
          break;
        }
      }
      if (pivotPos === -1) break;

      const pivotDocId = sortedTerms[pivotPos]!.docId;

      if (sortedTerms[0]!.docId === pivotDocId) {
        let totalScore = 0;
        for (let p = 0; p <= pivotPos; p++) {
          const ti = sortedTerms[p]!.termIdx;
          const entry = entries[ti]![cursors[ti]!]!;
          const tf =
            entry.payload.positions.length > 0 ? entry.payload.positions.length : 1;
          let dl = tf;
          if (this._invertedIndex && this._fields[ti]) {
            dl = this._invertedIndex.getDocLength(pivotDocId, this._fields[ti]);
            if (dl === 0) dl = tf;
          }
          const df = entries[ti]!.length;
          totalScore += this._scorers[ti]!.score(tf, dl, df);
        }

        if (heap.length < this._k) {
          heapPush(heap, [totalScore, pivotDocId]);
          if (heap.length === this._k) threshold = heap[0]![0];
        } else if (totalScore > threshold) {
          heapReplace(heap, [totalScore, pivotDocId]);
          threshold = heap[0]![0];
        }

        for (let p = pivotPos; p >= 0; p--) {
          const ti = sortedTerms[p]!.termIdx;
          cursors[ti] = cursors[ti]! + 1;
          if (cursors[ti] < entries[ti]!.length) {
            sortedTerms[p]!.docId = entries[ti]![cursors[ti]]!.docId;
          } else {
            sortedTerms.splice(p, 1);
          }
        }
        sortedTerms.sort((a, b) => a.docId - b.docId);
      } else {
        const ti = sortedTerms[0]!.termIdx;
        cursors[ti] = advanceCursor(entries[ti]!, cursors[ti]!, pivotDocId);
        if (cursors[ti] < entries[ti]!.length) {
          sortedTerms[0]!.docId = entries[ti]![cursors[ti]]!.docId;
        } else {
          sortedTerms.shift();
        }
        sortedTerms.sort((a, b) => a.docId - b.docId);
      }
    }

    const result = heap.map(([score, docId]) => ({
      docId,
      payload: createPayload({ score }),
    }));
    return new PostingList(result);
  }
}

// -- BoundTightnessAnalyzer ---------------------------------------------------

export class BoundTightnessAnalyzer {
  private _data: [number, number][] = [];

  record(upperBound: number, actualMax: number): void {
    this._data.push([upperBound, actualMax]);
  }

  tightnessRatio(): number {
    if (this._data.length === 0) return 1.0;
    let sum = 0;
    for (const [ub, am] of this._data) {
      sum += ub > 0 ? Math.min(1.0, am / ub) : 1.0;
    }
    return sum / this._data.length;
  }

  slack(): number {
    return 1.0 - this.tightnessRatio();
  }

  worstBoundIndex(): number {
    if (this._data.length === 0) return 0;
    let worstIdx = 0;
    let worstRatio = Infinity;
    for (let i = 0; i < this._data.length; i++) {
      const [ub, am] = this._data[i]!;
      const ratio = ub > 0 ? am / ub : 1.0;
      if (ratio < worstRatio) {
        worstRatio = ratio;
        worstIdx = i;
      }
    }
    return worstIdx;
  }

  clear(): void {
    this._data = [];
  }
}

// -- AdaptiveWANDScorer -------------------------------------------------------

export class AdaptiveWANDScorer extends WANDScorer {
  private readonly _tighteningFactor: number;
  /* @internal */ readonly _analyzer: BoundTightnessAnalyzer;

  constructor(
    scorers: WANDScorerLike[],
    k: number,
    postingLists: PostingList[],
    invertedIndex?: InvertedIndex | null,
    fields?: string[] | null,
    terms?: string[] | null,
    tighteningFactor = 0.9,
  ) {
    super(scorers, k, postingLists, invertedIndex, fields, terms);
    this._tighteningFactor = tighteningFactor;
    this._analyzer = new BoundTightnessAnalyzer();
    // Re-compute upper bounds now that _tighteningFactor is available.
    // During the parent constructor call above, _computeUpperBounds() was
    // invoked before _tighteningFactor was assigned, producing un-tightened
    // bounds. Overwrite them with the correct tightened values.
    const tightened = this._computeUpperBounds();
    for (let i = 0; i < tightened.length; i++) {
      this._upperBounds[i] = tightened[i]!;
    }
  }

  protected override _computeUpperBounds(): number[] {
    const base = super._computeUpperBounds();
    // During parent constructor, _tighteningFactor is not yet set
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition -- called during super() before field init
    const factor = this._tighteningFactor ?? 1.0;
    return base.map((ub) => ub * factor);
  }
}

// -- Min-heap utilities -------------------------------------------------------

function heapPush(heap: [number, DocId][], item: [number, DocId]): void {
  heap.push(item);
  let i = heap.length - 1;
  while (i > 0) {
    const parent = (i - 1) >>> 1;
    if (heap[parent]![0] <= heap[i]![0]) break;
    [heap[parent], heap[i]] = [heap[i]!, heap[parent]!];
    i = parent;
  }
}

function heapReplace(heap: [number, DocId][], item: [number, DocId]): void {
  heap[0] = item;
  const n = heap.length;
  let i = 0;
  for (;;) {
    let smallest = i;
    const left = 2 * i + 1;
    const right = 2 * i + 2;
    if (left < n && heap[left]![0] < heap[smallest]![0]) smallest = left;
    if (right < n && heap[right]![0] < heap[smallest]![0]) smallest = right;
    if (smallest === i) break;
    [heap[i], heap[smallest]] = [heap[smallest]!, heap[i]!];
    i = smallest;
  }
}
