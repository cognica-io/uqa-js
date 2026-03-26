//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- query feature extractor
// 1:1 port of uqa/fusion/query_features.py

import type { InvertedIndex } from "../storage/abc/inverted-index.js";

export class QueryFeatureExtractor {
  private readonly _index: InvertedIndex;

  constructor(invertedIndex: InvertedIndex) {
    this._index = invertedIndex;
  }

  get nFeatures(): number {
    return 6;
  }

  extract(queryTerms: string[], field?: string | null): Float64Array {
    const stats = this._index.stats;
    const n = stats.totalDocs;
    if (n === 0) return new Float64Array(6);

    const fieldName = field ?? "_default";
    const idfs: number[] = [];
    let vocabHits = 0;

    for (const term of queryTerms) {
      const df = stats.docFreq(fieldName, term);
      if (df > 0) {
        vocabHits++;
        const idf = Math.log((n - df + 0.5) / (df + 0.5) + 1.0);
        idfs.push(idf);
      }
    }

    if (idfs.length === 0) {
      return new Float64Array([0, 0, 0, 0, queryTerms.length, 0]);
    }

    let sum = 0;
    let max = -Infinity;
    let min = Infinity;
    for (const v of idfs) {
      sum += v;
      if (v > max) max = v;
      if (v < min) min = v;
    }

    const meanIdf = sum / idfs.length;
    const coverageRatio = idfs.length / Math.max(1, n);
    const queryLength = queryTerms.length;
    const vocabOverlap = vocabHits / Math.max(1, queryTerms.length);

    return new Float64Array([
      meanIdf,
      max,
      min,
      coverageRatio,
      queryLength,
      vocabOverlap,
    ]);
  }
}
