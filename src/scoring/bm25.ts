//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- BM25 scorer
// 1:1 port of uqa/scoring/bm25.py

import type { IndexStats } from "../core/types.js";

export interface BM25Params {
  readonly k1: number;
  readonly b: number;
  readonly boost: number;
}

export function createBM25Params(opts?: Partial<BM25Params>): BM25Params {
  return {
    k1: opts?.k1 ?? 1.2,
    b: opts?.b ?? 0.75,
    boost: opts?.boost ?? 1.0,
  };
}

export class BM25Scorer {
  private readonly _params: BM25Params;
  private readonly _totalDocs: number;
  private readonly _avgDocLength: number;

  constructor(params: BM25Params, indexStats: IndexStats) {
    this._params = params;
    this._totalDocs = indexStats.totalDocs;
    this._avgDocLength = indexStats.avgDocLength;
  }

  get params(): BM25Params {
    return this._params;
  }

  idf(docFreq: number): number {
    const n = this._totalDocs;
    return Math.log((n - docFreq + 0.5) / (docFreq + 0.5) + 1);
  }

  score(termFreq: number, docLength: number, docFreq: number): number {
    return this.scoreWithIdf(termFreq, docLength, this.idf(docFreq));
  }

  scoreWithIdf(termFreq: number, docLength: number, idfVal: number): number {
    const { k1, b, boost } = this._params;
    const w = boost * idfVal;
    const avgdl = this._avgDocLength > 0 ? this._avgDocLength : 1;
    const invNorm = 1 / (k1 * (1 - b + (b * docLength) / avgdl));
    return w - w / (1 + termFreq * invNorm);
  }

  combineScores(scores: number[]): number {
    let sum = 0;
    for (const s of scores) {
      sum += s;
    }
    return sum;
  }

  upperBound(docFreq: number): number {
    return this._params.boost * this.idf(docFreq);
  }
}
