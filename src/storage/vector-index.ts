//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- VectorIndex abstract + flat brute-force implementation
// 1:1 port of uqa/storage/vector_index.py

import type { DocId } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import { cosine } from "../math/linalg.js";

export abstract class VectorIndex {
  abstract readonly dimensions: number;
  abstract add(docId: DocId, vector: Float64Array): void;
  abstract delete(docId: DocId): void;
  abstract clear(): void;
  abstract searchKnn(query: Float64Array, k: number): PostingList;
  abstract searchThreshold(query: Float64Array, threshold: number): PostingList;
  abstract count(): number;
}

// Flat brute-force implementation (no HNSW/IVF -- those come in ivf-index.ts)
export class FlatVectorIndex extends VectorIndex {
  readonly dimensions: number;
  private _vectors: Map<DocId, Float64Array>;

  constructor(dimensions: number) {
    super();
    this.dimensions = dimensions;
    this._vectors = new Map();
  }

  add(docId: DocId, vector: Float64Array): void {
    if (vector.length !== this.dimensions) {
      throw new Error(
        `Vector dimension mismatch: expected ${String(this.dimensions)}, got ${String(vector.length)}`,
      );
    }
    this._vectors.set(docId, vector);
  }

  delete(docId: DocId): void {
    this._vectors.delete(docId);
  }

  clear(): void {
    this._vectors.clear();
  }

  searchKnn(query: Float64Array, k: number): PostingList {
    if (this._vectors.size === 0) return new PostingList();

    // Compute similarities for all vectors
    const scored: { docId: DocId; score: number }[] = [];
    for (const [docId, vec] of this._vectors) {
      const sim = cosine(query, vec);
      scored.push({ docId, score: sim });
    }

    // Sort descending by score, take top k
    scored.sort((a, b) => b.score - a.score);
    const topK = scored.slice(0, k);

    const entries = topK.map((s) => ({
      docId: s.docId,
      payload: createPayload({ score: s.score }),
    }));

    return new PostingList(entries);
  }

  searchThreshold(query: Float64Array, threshold: number): PostingList {
    const entries: {
      docId: DocId;
      payload: {
        positions: readonly number[];
        score: number;
        fields: Readonly<Record<string, unknown>>;
      };
    }[] = [];

    for (const [docId, vec] of this._vectors) {
      const sim = cosine(query, vec);
      if (sim >= threshold) {
        entries.push({
          docId,
          payload: createPayload({ score: sim }),
        });
      }
    }

    return new PostingList(entries);
  }

  count(): number {
    return this._vectors.size;
  }
}
