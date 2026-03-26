//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Fusion WAND scorer
// 1:1 port of uqa/scoring/fusion_wand.py

import { logOddsConjunction } from "bayesian-bm25";
import type { DocId } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import { BoundTightnessAnalyzer } from "./wand.js";

function sigmoidStable(x: number): number {
  if (x >= 0) return 1.0 / (1.0 + Math.exp(-x));
  const ex = Math.exp(x);
  return ex / (1.0 + ex);
}

function logit(p: number): number {
  const clamped = Math.max(1e-10, Math.min(1.0 - 1e-10, p));
  return Math.log(clamped / (1.0 - clamped));
}

// Coverage-based default (from operators/hybrid -- inlined to avoid circular dep)
function coverageBasedDefault(countInMap: number, numDocs: number): number {
  if (numDocs <= 0) return 0.5;
  const coverage = countInMap / numDocs;
  return sigmoidStable(logit(0.5) - 0.5 * coverage);
}

export class FusionWANDScorer {
  protected readonly _signalPostingLists: PostingList[];
  protected readonly _signalUpperBounds: number[];
  protected readonly _alpha: number;
  protected readonly _k: number;
  protected readonly _gating: string | null;

  constructor(
    signalPostingLists: PostingList[],
    signalUpperBounds: number[],
    alpha = 0.5,
    k = 10,
    gating?: string | null,
  ) {
    this._signalPostingLists = signalPostingLists;
    this._signalUpperBounds = signalUpperBounds;
    this._alpha = alpha;
    this._k = k;
    this._gating = gating ?? null;
  }

  private _computeFusedUpperBound(activeUbs: number[]): number {
    return logOddsConjunction(
      activeUbs,
      this._alpha,
      undefined,
      this._gating ?? "none",
    );
  }

  scoreTopK(): PostingList {
    const nSignals = this._signalPostingLists.length;

    // Build score maps per signal
    const scoreMaps: Map<DocId, number>[] = [];
    const allDocIds = new Set<DocId>();
    let numDocs = 0;

    for (let s = 0; s < nSignals; s++) {
      const m = new Map<DocId, number>();
      for (const entry of this._signalPostingLists[s]!) {
        m.set(entry.docId, entry.payload.score);
        allDocIds.add(entry.docId);
      }
      scoreMaps.push(m);
      numDocs = Math.max(numDocs, m.size);
    }

    // Compute coverage-based defaults per signal
    const defaults = scoreMaps.map((m) => coverageBasedDefault(m.size, numDocs));

    // Min-heap for top-k
    const heap: [number, DocId][] = [];
    let threshold = 0;

    for (const docId of allDocIds) {
      // Quick upper bound check
      const activeUbs: number[] = [];
      for (let s = 0; s < nSignals; s++) {
        if (scoreMaps[s]!.has(docId)) {
          activeUbs.push(this._signalUpperBounds[s]!);
        } else {
          activeUbs.push(defaults[s]!);
        }
      }
      const fusedUb = this._computeFusedUpperBound(activeUbs);
      if (heap.length >= this._k && fusedUb <= threshold) continue;

      // Score document
      const probs: number[] = [];
      for (let s = 0; s < nSignals; s++) {
        probs.push(scoreMaps[s]!.get(docId) ?? defaults[s]!);
      }
      const fused = logOddsConjunction(
        probs,
        this._alpha,
        undefined,
        this._gating ?? "none",
      );

      if (heap.length < this._k) {
        heap.push([fused, docId]);
        heapifyUp(heap, heap.length - 1);
        if (heap.length === this._k) threshold = heap[0]![0];
      } else if (fused > threshold) {
        heap[0] = [fused, docId];
        heapifyDown(heap);
        threshold = heap[0][0];
      }
    }

    const result = heap.map(([score, docId]) => ({
      docId,
      payload: createPayload({ score }),
    }));
    return new PostingList(result);
  }
}

// -- TightenedFusionWANDScorer ------------------------------------------------

export class TightenedFusionWANDScorer extends FusionWANDScorer {
  /* @internal */ readonly _tighteningFactor: number;
  /* @internal */ readonly _originalBounds: number[];
  readonly boundAnalyzer: BoundTightnessAnalyzer;

  constructor(
    signalPostingLists: PostingList[],
    signalUpperBounds: number[],
    alpha = 0.5,
    k = 10,
    gating?: string | null,
    tighteningFactor = 0.9,
  ) {
    const tightened = signalUpperBounds.map((ub) => ub * tighteningFactor);
    super(signalPostingLists, tightened, alpha, k, gating);
    this._tighteningFactor = tighteningFactor;
    this._originalBounds = signalUpperBounds;
    this.boundAnalyzer = new BoundTightnessAnalyzer();
  }
}

// -- Min-heap utilities -------------------------------------------------------

function heapifyUp(heap: [number, DocId][], i: number): void {
  while (i > 0) {
    const parent = (i - 1) >>> 1;
    if (heap[parent]![0] <= heap[i]![0]) break;
    [heap[parent], heap[i]] = [heap[i]!, heap[parent]!];
    i = parent;
  }
}

function heapifyDown(heap: [number, DocId][]): void {
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
