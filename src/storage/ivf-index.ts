//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- IVF (Inverted File) vector index
// 1:1 port of uqa/storage/ivf_index.py
//
// Three states:
//   UNTRAINED -- fewer than train_threshold vectors; brute-force scan
//   TRAINED   -- centroids are valid; IVF search
//   STALE     -- >20% deletes since last train; retrain on next search

import type { DocId } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import { VectorIndex } from "./vector-index.js";
import { norm as vecNorm } from "../math/linalg.js";
import { SeededRandom } from "../math/random.js";

const UNTRAINED_CENTROID_ID = -1;
export const BACKGROUND_STATS_ID = -2;

export enum IVFState {
  UNTRAINED = "untrained",
  TRAINED = "trained",
  STALE = "stale",
}

// -- IVF Cell data structure --------------------------------------------------

interface IVFCell {
  centroid: Float64Array;
  docIds: DocId[];
  vectors: Float64Array[];
}

// -- Normalize helper ---------------------------------------------------------

function l2Normalize(vec: Float64Array): Float64Array {
  const n = vecNorm(vec);
  if (n <= 0) return vec;
  const result = new Float64Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    result[i] = vec[i]! / n;
  }
  return result;
}

// -- IVFIndex -----------------------------------------------------------------

export class IVFIndex extends VectorIndex {
  readonly dimensions: number;
  private _nlist: number;
  readonly _nProbe: number;
  private _cells: IVFCell[];
  private _docIdToCell: Map<DocId, number>;
  private _allDocIds: Set<DocId>;
  private _allVectors: Map<DocId, Float64Array>;
  private _state: IVFState;
  private _totalVectors: number;
  private _deletesSinceTrain: number;
  private _backgroundMu: number | null;
  private _backgroundSigma: number | null;
  private _backgroundSamples: number[] | null;

  constructor(dimensions: number, nlist = 100, nProbe = 10) {
    super();
    this.dimensions = dimensions;
    this._nlist = nlist;
    this._nProbe = nProbe;
    this._cells = [];
    this._docIdToCell = new Map();
    this._allDocIds = new Set();
    this._allVectors = new Map();
    this._state = IVFState.UNTRAINED;
    this._totalVectors = 0;
    this._deletesSinceTrain = 0;
    this._backgroundMu = null;
    this._backgroundSigma = null;
    this._backgroundSamples = null;
  }

  get trainThreshold(): number {
    return Math.max(2 * this._nlist, 256);
  }

  get nlist(): number {
    return this._nlist;
  }

  get totalVectors(): number {
    return this._totalVectors;
  }

  get backgroundStats(): [number, number] | null {
    if (this._backgroundMu !== null && this._backgroundSigma !== null) {
      return [this._backgroundMu, this._backgroundSigma];
    }
    return null;
  }

  get backgroundSamples(): number[] | null {
    return this._backgroundSamples;
  }

  get state(): IVFState {
    return this._state;
  }

  get deletesSinceTrain(): number {
    return this._deletesSinceTrain;
  }

  // -- VectorIndex interface --------------------------------------------------

  add(docId: DocId, vector: Float64Array): void {
    if (vector.length !== this.dimensions) {
      throw new Error(
        `Vector dimension mismatch: expected ${String(this.dimensions)}, got ${String(vector.length)}`,
      );
    }

    const normalized = l2Normalize(vector);
    this._allDocIds.add(docId);
    this._allVectors.set(docId, normalized);
    this._totalVectors++;

    if (this._state === IVFState.TRAINED && this._cells.length > 0) {
      const cellIdx = this._assignCentroid(normalized);
      this._cells[cellIdx]!.docIds.push(docId);
      this._cells[cellIdx]!.vectors.push(normalized);
      this._docIdToCell.set(docId, cellIdx);
    }

    // Auto-train when enough vectors accumulate
    if (
      this._state === IVFState.UNTRAINED &&
      this._totalVectors >= this.trainThreshold
    ) {
      this._train();
    }
  }

  delete(docId: DocId): void {
    if (!this._allDocIds.has(docId)) return;

    this._allDocIds.delete(docId);
    this._allVectors.delete(docId);
    this._totalVectors--;
    this._deletesSinceTrain++;

    const cellIdx = this._docIdToCell.get(docId);
    if (cellIdx !== undefined && this._cells[cellIdx]) {
      const cell = this._cells[cellIdx];
      const idx = cell.docIds.indexOf(docId);
      if (idx !== -1) {
        cell.docIds.splice(idx, 1);
        cell.vectors.splice(idx, 1);
      }
      this._docIdToCell.delete(docId);
    }

    // Mark stale if >20% deletes since last train
    if (
      this._state === IVFState.TRAINED &&
      this._totalVectors > 0 &&
      this._deletesSinceTrain > this._totalVectors * 0.2
    ) {
      this._state = IVFState.STALE;
    }
  }

  clear(): void {
    this._cells = [];
    this._docIdToCell = new Map();
    this._allDocIds = new Set();
    this._allVectors = new Map();
    this._state = IVFState.UNTRAINED;
    this._totalVectors = 0;
    this._deletesSinceTrain = 0;
    this._backgroundMu = null;
    this._backgroundSigma = null;
    this._backgroundSamples = null;
  }

  searchKnn(query: Float64Array, k: number): PostingList {
    if (this._totalVectors === 0) return new PostingList();

    if (this._state === IVFState.STALE) this._train();

    const q = l2Normalize(query);

    if (this._state === IVFState.UNTRAINED) {
      return this._bruteForceKnn(q, k);
    }

    return this._ivfKnn(q, k);
  }

  searchThreshold(query: Float64Array, threshold: number): PostingList {
    if (this._totalVectors === 0) return new PostingList();

    if (this._state === IVFState.STALE) this._train();

    const q = l2Normalize(query);

    if (this._state === IVFState.UNTRAINED) {
      return this._bruteForceThreshold(q, threshold);
    }

    return this._ivfThreshold(q, threshold);
  }

  count(): number {
    return this._totalVectors;
  }

  // -- Centroid assignment ----------------------------------------------------

  private _assignCentroid(normalizedVec: Float64Array): number {
    let bestIdx = 0;
    let bestSim = -Infinity;
    for (let i = 0; i < this._cells.length; i++) {
      const sim = this._dot(normalizedVec, this._cells[i]!.centroid);
      if (sim > bestSim) {
        bestSim = sim;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  private _nearestCentroids(normalizedQuery: Float64Array, nprobe: number): number[] {
    const n = this._cells.length;
    const actualNprobe = Math.min(nprobe, n);
    if (actualNprobe >= n) {
      return Array.from({ length: n }, (_, i) => i);
    }

    // Compute similarities and pick top nprobe
    const sims: { idx: number; sim: number }[] = [];
    for (let i = 0; i < n; i++) {
      sims.push({ idx: i, sim: this._dot(normalizedQuery, this._cells[i]!.centroid) });
    }
    sims.sort((a, b) => b.sim - a.sim);
    return sims.slice(0, actualNprobe).map((s) => s.idx);
  }

  // -- Brute-force search (UNTRAINED state) -----------------------------------

  private _bruteForceKnn(q: Float64Array, k: number): PostingList {
    const scored: { docId: DocId; score: number }[] = [];
    for (const [docId, vec] of this._allVectors) {
      const sim = this._dot(q, vec);
      scored.push({ docId, score: sim });
    }

    scored.sort((a, b) => b.score - a.score);
    const topK = scored.slice(0, k);

    return new PostingList(
      topK.map((r) => ({
        docId: r.docId,
        payload: createPayload({ score: r.score }),
      })),
    );
  }

  private _bruteForceThreshold(q: Float64Array, threshold: number): PostingList {
    const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
    for (const [docId, vec] of this._allVectors) {
      const sim = this._dot(q, vec);
      if (sim >= threshold) {
        entries.push({ docId, payload: createPayload({ score: sim }) });
      }
    }
    return new PostingList(entries);
  }

  // -- IVF search (TRAINED state) ---------------------------------------------

  probedDistances(query: Float64Array): number[] {
    const q = l2Normalize(query);

    if (this._state !== IVFState.TRAINED || this._cells.length === 0) {
      // Brute-force: scan all vectors
      const dists: number[] = [];
      for (const vec of this._allVectors.values()) {
        dists.push(1.0 - this._dot(q, vec));
      }
      return dists;
    }

    const centroidIds = this._nearestCentroids(q, this._nProbe);
    const dists: number[] = [];
    for (const cid of centroidIds) {
      const cell = this._cells[cid]!;
      for (const vec of cell.vectors) {
        dists.push(1.0 - this._dot(q, vec));
      }
    }
    // Also scan untrained vectors
    for (const docId of this._allDocIds) {
      if (!this._docIdToCell.has(docId)) {
        const vec = this._allVectors.get(docId);
        if (vec) dists.push(1.0 - this._dot(q, vec));
      }
    }
    return dists;
  }

  private _ivfKnn(q: Float64Array, k: number): PostingList {
    const centroidIds = this._nearestCentroids(q, this._nProbe);

    // Collect candidates from probed cells
    const candidates: { docId: DocId; score: number; centroidId: number }[] = [];
    for (const cid of centroidIds) {
      const cell = this._cells[cid]!;
      for (let i = 0; i < cell.docIds.length; i++) {
        const sim = this._dot(q, cell.vectors[i]!);
        candidates.push({ docId: cell.docIds[i]!, score: sim, centroidId: cid });
      }
    }

    // Also scan untrained vectors (added since last train)
    for (const docId of this._allDocIds) {
      if (!this._docIdToCell.has(docId)) {
        const vec = this._allVectors.get(docId);
        if (vec) {
          const sim = this._dot(q, vec);
          candidates.push({ docId, score: sim, centroidId: UNTRAINED_CENTROID_ID });
        }
      }
    }

    candidates.sort((a, b) => b.score - a.score);
    const topK = candidates.slice(0, k);

    return new PostingList(
      topK.map((r) => ({
        docId: r.docId,
        payload: createPayload({
          score: r.score,
          fields: { _centroid_id: r.centroidId },
        }),
      })),
    );
  }

  private _ivfThreshold(q: Float64Array, threshold: number): PostingList {
    const centroidIds = this._nearestCentroids(q, this._nProbe);

    const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
    for (const cid of centroidIds) {
      const cell = this._cells[cid]!;
      for (let i = 0; i < cell.docIds.length; i++) {
        const sim = this._dot(q, cell.vectors[i]!);
        if (sim >= threshold) {
          entries.push({
            docId: cell.docIds[i]!,
            payload: createPayload({
              score: sim,
              fields: { _centroid_id: cid },
            }),
          });
        }
      }
    }

    // Scan untrained vectors
    for (const docId of this._allDocIds) {
      if (!this._docIdToCell.has(docId)) {
        const vec = this._allVectors.get(docId);
        if (vec) {
          const sim = this._dot(q, vec);
          if (sim >= threshold) {
            entries.push({
              docId,
              payload: createPayload({
                score: sim,
                fields: { _centroid_id: UNTRAINED_CENTROID_ID },
              }),
            });
          }
        }
      }
    }

    return new PostingList(entries);
  }

  // -- Cell populations -------------------------------------------------------

  /** Return the number of IVF cells (centroids). */
  cellCount(): number {
    return this._cells.length;
  }

  /** Return true if the index has been trained. */
  isTrained(): boolean {
    return this._state === IVFState.TRAINED;
  }

  /** Return the centroid vectors (empty if untrained). */
  centroids(): Float64Array[] {
    return this._cells.map((c) => c.centroid);
  }

  cellPopulations(): Map<number, number> {
    const pops = new Map<number, number>();
    for (let i = 0; i < this._cells.length; i++) {
      pops.set(i, this._cells[i]!.docIds.length);
    }
    return pops;
  }

  // -- Public training entry point --------------------------------------------

  train(_vectors: Float64Array[]): void {
    this._train();
  }

  // -- k-means training -------------------------------------------------------

  private _train(): void {
    if (this._allVectors.size === 0) return;

    const docIds: DocId[] = [];
    const data: Float64Array[] = [];
    for (const [docId, vec] of this._allVectors) {
      docIds.push(docId);
      data.push(vec);
    }

    const n = data.length;
    const actualNlist = Math.min(this._nlist, n);
    if (actualNlist < 1) return;

    const centroids = this._kmeans(data, actualNlist);

    // Build cells
    this._cells = centroids.map((c) => ({
      centroid: c,
      docIds: [],
      vectors: [],
    }));

    // Assign all vectors to their nearest centroid
    this._docIdToCell.clear();
    for (let j = 0; j < n; j++) {
      const cellIdx = this._assignCentroid(data[j]!);
      this._cells[cellIdx]!.docIds.push(docIds[j]!);
      this._cells[cellIdx]!.vectors.push(data[j]!);
      this._docIdToCell.set(docIds[j]!, cellIdx);
    }

    // Estimate background distance distribution f_G
    const rng = new SeededRandom(42);
    const nRandomQueries = 100;
    const kPerQuery = Math.min(50, n);
    const allBgDists: number[] = [];

    for (let qi = 0; qi < nRandomQueries; qi++) {
      // Generate random normalized query
      const rq = new Float64Array(this.dimensions);
      for (let d = 0; d < this.dimensions; d++) {
        rq[d] = rng.randn();
      }
      const normalized = l2Normalize(rq);

      // Compute similarities and take top-k distances
      const sims: number[] = [];
      for (const vec of data) {
        sims.push(this._dot(normalized, vec));
      }
      sims.sort((a, b) => b - a);
      const topSims = sims.slice(0, kPerQuery);
      for (const sim of topSims) {
        allBgDists.push(1.0 - sim);
      }
    }

    if (allBgDists.length > 0) {
      let sum = 0;
      for (const d of allBgDists) sum += d;
      this._backgroundMu = sum / allBgDists.length;

      let sumSq = 0;
      for (const d of allBgDists) {
        const diff = d - this._backgroundMu;
        sumSq += diff * diff;
      }
      this._backgroundSigma = Math.max(Math.sqrt(sumSq / allBgDists.length), 1e-10);
      this._backgroundSamples = allBgDists;
    }

    this._state = IVFState.TRAINED;
    this._deletesSinceTrain = 0;
  }

  private _kmeans(
    data: Float64Array[],
    k: number,
    maxIter = 25,
    tol = 1e-4,
  ): Float64Array[] {
    const n = data.length;
    const dims = this.dimensions;
    const rng = new SeededRandom(42);

    // -- k-means++ initialization --
    const centroids: Float64Array[] = [];
    const firstIdx = Math.floor(rng.random() * n);
    centroids.push(new Float64Array(data[firstIdx]!));

    for (let i = 1; i < k; i++) {
      // Compute distances to nearest centroid
      const dists = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        let maxSim = -Infinity;
        for (const c of centroids) {
          const sim = this._dot(data[j]!, c);
          if (sim > maxSim) maxSim = sim;
        }
        dists[j] = Math.max(1.0 - maxSim, 0.0);
      }

      let total = 0;
      for (let j = 0; j < n; j++) total += dists[j]!;

      if (total === 0) {
        centroids.push(new Float64Array(data[Math.floor(rng.random() * n)]!));
      } else {
        // Weighted random selection
        const threshold = rng.random() * total;
        let cumSum = 0;
        let chosenIdx = 0;
        for (let j = 0; j < n; j++) {
          cumSum += dists[j]!;
          if (cumSum >= threshold) {
            chosenIdx = j;
            break;
          }
        }
        centroids.push(new Float64Array(data[chosenIdx]!));
      }
    }

    // -- Lloyd's iterations --
    for (let iter = 0; iter < maxIter; iter++) {
      // Assign each vector to nearest centroid
      const labels = new Int32Array(n);
      for (let j = 0; j < n; j++) {
        let bestCid = 0;
        let bestSim = -Infinity;
        for (let c = 0; c < k; c++) {
          const sim = this._dot(data[j]!, centroids[c]!);
          if (sim > bestSim) {
            bestSim = sim;
            bestCid = c;
          }
        }
        labels[j] = bestCid;
      }

      // Update centroids
      const newCentroids: Float64Array[] = [];
      const counts = new Int32Array(k);
      for (let c = 0; c < k; c++) {
        newCentroids.push(new Float64Array(dims));
      }

      for (let j = 0; j < n; j++) {
        const cid = labels[j]!;
        counts[cid] = (counts[cid] ?? 0) + 1;
        const centroidArr = newCentroids[cid]!;
        const dataArr = data[j]!;
        for (let d = 0; d < dims; d++) {
          centroidArr[d] = centroidArr[d]! + dataArr[d]!;
        }
      }

      for (let c = 0; c < k; c++) {
        if ((counts[c] ?? 0) > 0) {
          const cnt = counts[c]!;
          const centroidArr = newCentroids[c]!;
          for (let d = 0; d < dims; d++) {
            centroidArr[d] = centroidArr[d]! / cnt;
          }
          // Normalize
          const nc = l2Normalize(newCentroids[c]!);
          for (let d = 0; d < dims; d++) {
            newCentroids[c]![d] = nc[d]!;
          }
        } else {
          // Empty cluster: reinitialize from random data point
          const ridx = Math.floor(rng.random() * n);
          for (let d = 0; d < dims; d++) {
            newCentroids[c]![d] = data[ridx]![d]!;
          }
        }
      }

      // Check convergence
      let shift = 0;
      for (let c = 0; c < k; c++) {
        let sumSq = 0;
        for (let d = 0; d < dims; d++) {
          const diff = newCentroids[c]![d]! - centroids[c]![d]!;
          sumSq += diff * diff;
        }
        shift += Math.sqrt(sumSq);
      }

      for (let c = 0; c < k; c++) {
        centroids[c] = newCentroids[c]!;
      }

      if (shift < tol) break;
    }

    return centroids;
  }

  // -- Dot product helper -----------------------------------------------------

  private _dot(a: Float64Array, b: Float64Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += a[i]! * b[i]!;
    }
    return sum;
  }
}
