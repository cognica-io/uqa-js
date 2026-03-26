//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- primitive operators
// 1:1 port of uqa/operators/primitive.py

import type { DocId, IndexStats, Predicate } from "../core/types.js";
import { createPayload, isNullPredicate } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "./base.js";
import { Operator } from "./base.js";
import type { Index } from "../storage/index-abc.js";
import { cosine, norm } from "../math/linalg.js";
import { haversineDistance } from "../storage/spatial-index.js";

// -- Brute-force helpers -----------------------------------------------------

function bruteForceknn(
  context: ExecutionContext,
  field: string,
  query: Float64Array,
  k: number,
): PostingList {
  const docStore = context.documentStore;
  if (!docStore) return new PostingList();

  const qnorm = norm(query);
  if (qnorm === 0) return new PostingList();

  const scored: { docId: DocId; score: number }[] = [];
  for (const docId of docStore.docIds) {
    const vec = docStore.getField(docId, field);
    if (!vec || !(vec instanceof Float64Array)) continue;
    const vnorm = norm(vec);
    if (vnorm === 0) continue;
    const sim = cosine(query, vec);
    scored.push({ docId, score: sim });
  }

  scored.sort((a, b) => b.score - a.score);
  const top = scored.slice(0, k);
  return new PostingList(
    top.map((s) => ({ docId: s.docId, payload: createPayload({ score: s.score }) })),
  );
}

function bruteForceThreshold(
  context: ExecutionContext,
  field: string,
  query: Float64Array,
  threshold: number,
): PostingList {
  const docStore = context.documentStore;
  if (!docStore) return new PostingList();

  const qnorm = norm(query);
  if (qnorm === 0) return new PostingList();

  const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
  for (const docId of docStore.docIds) {
    const vec = docStore.getField(docId, field);
    if (!vec || !(vec instanceof Float64Array)) continue;
    const vnorm = norm(vec);
    if (vnorm === 0) continue;
    const sim = cosine(query, vec);
    if (sim >= threshold) {
      entries.push({ docId, payload: createPayload({ score: sim }) });
    }
  }
  return new PostingList(entries);
}

// -- TermOperator ------------------------------------------------------------

export class TermOperator extends Operator {
  readonly term: string;
  readonly field: string | null;

  constructor(term: string, field?: string | null) {
    super();
    this.term = term;
    this.field = field ?? null;
  }

  execute(context: ExecutionContext): PostingList {
    const idx = context.invertedIndex;
    if (!idx) return new PostingList();

    const analyzer = this.field ? idx.getSearchAnalyzer(this.field) : idx.analyzer;
    const tokens = analyzer.analyze(this.term);
    if (tokens.length === 0) return new PostingList();

    const lists = tokens.map((t) =>
      this.field ? idx.getPostingList(this.field, t) : idx.getPostingListAnyField(t),
    );

    let result = lists[0]!;
    for (let i = 1; i < lists.length; i++) {
      result = result.union(lists[i]!);
    }
    return result;
  }

  costEstimate(stats: IndexStats): number {
    if (this.field) return stats.docFreq(this.field, this.term);
    return stats.totalDocs;
  }
}

// -- VectorSimilarityOperator ------------------------------------------------

export class VectorSimilarityOperator extends Operator {
  readonly queryVector: Float64Array;
  readonly threshold: number;
  readonly field: string;

  constructor(queryVector: Float64Array, threshold: number, field = "embedding") {
    super();
    this.queryVector = queryVector;
    this.threshold = threshold;
    this.field = field;
  }

  execute(context: ExecutionContext): PostingList {
    const vecIdx = context.vectorIndexes?.[this.field];
    if (vecIdx) return vecIdx.searchThreshold(this.queryVector, this.threshold);
    return bruteForceThreshold(context, this.field, this.queryVector, this.threshold);
  }

  costEstimate(stats: IndexStats): number {
    return stats.dimensions * Math.log2(stats.totalDocs + 1);
  }
}

// -- KNNOperator -------------------------------------------------------------

export class KNNOperator extends Operator {
  readonly queryVector: Float64Array;
  readonly k: number;
  readonly field: string;

  constructor(queryVector: Float64Array, k: number, field = "embedding") {
    super();
    this.queryVector = queryVector;
    this.k = k;
    this.field = field;
  }

  execute(context: ExecutionContext): PostingList {
    const vecIdx = context.vectorIndexes?.[this.field];
    if (vecIdx) return vecIdx.searchKnn(this.queryVector, this.k);
    return bruteForceknn(context, this.field, this.queryVector, this.k);
  }

  costEstimate(stats: IndexStats): number {
    return stats.dimensions * Math.log2(stats.totalDocs + 1);
  }
}

// -- SpatialWithinOperator ---------------------------------------------------

export class SpatialWithinOperator extends Operator {
  readonly field: string;
  readonly centerX: number;
  readonly centerY: number;
  readonly distance: number;

  constructor(field: string, centerX: number, centerY: number, distance: number) {
    super();
    this.field = field;
    this.centerX = centerX;
    this.centerY = centerY;
    this.distance = distance;
  }

  execute(context: ExecutionContext): PostingList {
    const spIdx = context.spatialIndexes?.[this.field];
    if (spIdx) return spIdx.searchWithin(this.centerX, this.centerY, this.distance);
    return this._bruteForceScan(context);
  }

  private _bruteForceScan(context: ExecutionContext): PostingList {
    const docStore = context.documentStore;
    if (!docStore) return new PostingList();

    const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
    const sortedIds = [...docStore.docIds].sort((a, b) => a - b);

    for (const docId of sortedIds) {
      const point = docStore.getField(docId, this.field) as
        | [number, number]
        | null
        | undefined;
      if (!point) continue;
      const dist = haversineDistance(this.centerY, this.centerX, point[1], point[0]);
      if (dist <= this.distance) {
        const score = this.distance > 0 ? 1.0 - dist / this.distance : 1.0;
        entries.push({ docId, payload: createPayload({ score }) });
      }
    }
    return PostingList.fromSorted(entries);
  }

  costEstimate(stats: IndexStats): number {
    return Math.log2(stats.totalDocs + 1);
  }
}

// -- FilterOperator ----------------------------------------------------------

export class FilterOperator extends Operator {
  readonly field: string;
  readonly predicate: Predicate;
  readonly source: Operator | null;

  constructor(field: string, predicate: Predicate, source?: Operator | null) {
    super();
    this.field = field;
    this.predicate = predicate;
    this.source = source ?? null;
  }

  execute(context: ExecutionContext): PostingList {
    const docStore = context.documentStore;
    if (!docStore) return new PostingList();

    const isNullAware = isNullPredicate(this.predicate);
    const hasBulk =
      typeof (docStore as unknown as { getFieldsBulk?: unknown }).getFieldsBulk ===
      "function";

    if (this.source) {
      const sourcePl = this.source.execute(context);
      const sourceEntries = [...sourcePl];
      if (hasBulk && sourceEntries.length > 1) {
        const docIds = sourceEntries.map((e) => e.docId);
        const valueMap = (
          docStore as unknown as {
            getFieldsBulk(ids: DocId[], field: string): Map<DocId, unknown>;
          }
        ).getFieldsBulk(docIds, this.field);
        const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] =
          [];
        for (const entry of sourceEntries) {
          const value = valueMap.get(entry.docId);
          const matched = isNullAware
            ? this.predicate.evaluate(value)
            : value !== null && value !== undefined && this.predicate.evaluate(value);
          if (matched) entries.push(entry);
        }
        return PostingList.fromSorted(entries);
      }
      const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
      for (const entry of sourceEntries) {
        const value = docStore.getField(entry.docId, this.field);
        const matched = isNullAware
          ? this.predicate.evaluate(value)
          : value !== null && value !== undefined && this.predicate.evaluate(value);
        if (matched) entries.push(entry);
      }
      return PostingList.fromSorted(entries);
    }

    const sortedIds = [...docStore.docIds].sort((a, b) => a - b);
    if (hasBulk && sortedIds.length > 1) {
      const valueMap = (
        docStore as unknown as {
          getFieldsBulk(ids: DocId[], field: string): Map<DocId, unknown>;
        }
      ).getFieldsBulk(sortedIds, this.field);
      const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
      for (const docId of sortedIds) {
        const value = valueMap.get(docId);
        const matched = isNullAware
          ? this.predicate.evaluate(value)
          : value !== null && value !== undefined && this.predicate.evaluate(value);
        if (matched) {
          entries.push({ docId, payload: createPayload({ score: 0.0 }) });
        }
      }
      return PostingList.fromSorted(entries);
    }
    const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
    for (const docId of sortedIds) {
      const value = docStore.getField(docId, this.field);
      if (!isNullAware && (value === null || value === undefined)) continue;
      if (this.predicate.evaluate(value)) {
        entries.push({ docId, payload: createPayload({ score: 0.0 }) });
      }
    }
    return PostingList.fromSorted(entries);
  }

  costEstimate(stats: IndexStats): number {
    return stats.totalDocs;
  }
}

// -- FacetOperator -----------------------------------------------------------

export class FacetOperator extends Operator {
  readonly field: string;
  readonly source: Operator | null;

  constructor(field: string, source?: Operator | null) {
    super();
    this.field = field;
    this.source = source ?? null;
  }

  execute(context: ExecutionContext): PostingList {
    const docStore = context.documentStore;
    if (!docStore) return new PostingList();

    let docIds: DocId[];
    if (this.source) {
      const sourcePl = this.source.execute(context);
      docIds = sourcePl.entries.map((e) => e.docId);
    } else {
      docIds = [...docStore.docIds].sort((a, b) => a - b);
    }

    const counts = new Map<string, number>();
    for (const docId of docIds) {
      const value = docStore.getField(docId, this.field);
      if (value !== null && value !== undefined) {
        const key = String(value as string | number);
        counts.set(key, (counts.get(key) ?? 0) + 1);
      }
    }

    const sorted = [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
    const entries = sorted.map(([value, count], idx) => ({
      docId: idx,
      payload: createPayload({
        score: count,
        fields: {
          _facet_field: this.field,
          _facet_value: value,
          _facet_count: count,
        },
      }),
    }));
    return PostingList.fromSorted(entries);
  }

  costEstimate(stats: IndexStats): number {
    return stats.totalDocs;
  }
}

// -- ScoreOperator -----------------------------------------------------------

export class ScoreOperator extends Operator {
  readonly scorer: {
    score(termFreq: number, docLength: number, docFreq: number): number;
    idf?(docFreq: number): number;
    scoreWithIdf?(termFreq: number, docLength: number, idfVal: number): number;
    combineScores?(scores: number[]): number;
  };
  readonly source: Operator;
  readonly queryTerms: string[];
  readonly field: string | null;

  constructor(
    scorer: ScoreOperator["scorer"],
    source: Operator,
    queryTerms: string[],
    field?: string | null,
  ) {
    super();
    this.scorer = scorer;
    this.source = source;
    this.queryTerms = queryTerms;
    this.field = field ?? null;
  }

  execute(context: ExecutionContext): PostingList {
    const sourcePl = this.source.execute(context);
    const idx = context.invertedIndex;
    if (!idx) return sourcePl;

    const hasIdf =
      typeof this.scorer.idf === "function" &&
      typeof this.scorer.scoreWithIdf === "function";
    const hasCombine = typeof this.scorer.combineScores === "function";

    // Pre-compute IDF values
    const idfs: number[] = [];
    if (hasIdf) {
      for (const term of this.queryTerms) {
        const df = this.field
          ? idx.docFreq(this.field, term)
          : idx.docFreqAnyField(term);
        idfs.push(this.scorer.idf!(df));
      }
    }

    const entries = sourcePl.entries;
    const docIds = entries.map((e) => e.docId);
    const hasBulk = typeof idx.getDocLengthsBulk === "function";

    // Bulk prefetch doc lengths
    let dlMap: Map<DocId, number> | null = null;
    if (hasBulk && this.field !== null) {
      dlMap = idx.getDocLengthsBulk(docIds, this.field);
    }

    // Bulk prefetch term frequencies per term
    const tfMaps: Map<DocId, number>[] = [];
    if (hasBulk && this.field !== null) {
      for (const term of this.queryTerms) {
        tfMaps.push(idx.getTermFreqsBulk(docIds, this.field, term));
      }
    }

    const result: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];

    for (const entry of entries) {
      const perTermScores: number[] = [];
      let dl: number;
      if (dlMap !== null) {
        dl = dlMap.get(entry.docId) ?? 0;
      } else if (this.field !== null) {
        dl = idx.getDocLength(entry.docId, this.field);
      } else {
        dl = idx.getTotalDocLength(entry.docId);
      }

      for (let t = 0; t < this.queryTerms.length; t++) {
        const term = this.queryTerms[t]!;
        let tf: number;
        if (tfMaps.length > 0) {
          tf = tfMaps[t]!.get(entry.docId) ?? 0;
        } else if (this.field !== null) {
          tf = idx.getTermFreq(entry.docId, this.field, term);
        } else {
          tf = idx.getTotalTermFreq(entry.docId, term);
        }

        if (hasIdf) {
          perTermScores.push(this.scorer.scoreWithIdf!(tf, dl, idfs[t]!));
        } else {
          const df = this.field
            ? idx.docFreq(this.field, term)
            : idx.docFreqAnyField(term);
          perTermScores.push(this.scorer.score(tf, dl, df));
        }
      }

      const totalScore = hasCombine
        ? this.scorer.combineScores!(perTermScores)
        : perTermScores.reduce((a, b) => a + b, 0);

      result.push({
        docId: entry.docId,
        payload: createPayload({
          positions: entry.payload.positions as number[],
          score: totalScore,
          fields: entry.payload.fields as Record<string, unknown>,
        }),
      });
    }

    return PostingList.fromSorted(result);
  }
}

// -- IndexScanOperator -------------------------------------------------------

export class IndexScanOperator extends Operator {
  readonly index: Index;
  readonly field: string;
  readonly predicate: Predicate;

  constructor(index: Index, field: string, predicate: Predicate) {
    super();
    this.index = index;
    this.field = field;
    this.predicate = predicate;
  }

  execute(_context: ExecutionContext): PostingList {
    return this.index.scan(this.predicate);
  }

  costEstimate(_stats: IndexStats): number {
    return this.index.scanCost(this.predicate);
  }
}
