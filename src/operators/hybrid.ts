//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- hybrid operators
// 1:1 port of uqa/operators/hybrid.py

import { logOddsConjunction, probAnd, probOr } from "bayesian-bm25";
import type { DocId, IndexStats } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "./base.js";
import { Operator } from "./base.js";
import { TermOperator, VectorSimilarityOperator } from "./primitive.js";
import { FusionWANDScorer } from "../scoring/fusion-wand.js";

// -- Coverage-based default (exported for fusion-wand) -----------------------

export function coverageBasedDefault(
  nHits: number,
  nTotal: number,
  floor = 0.01,
): number {
  if (nTotal <= 0) return 0.5;
  const r = nHits / nTotal;
  return 0.5 * (1 - r) + floor * r;
}

// -- HybridTextVectorOperator ------------------------------------------------

export class HybridTextVectorOperator extends Operator {
  private readonly _termOp: TermOperator;
  private readonly _vectorOp: VectorSimilarityOperator;

  constructor(term: string, queryVector: Float64Array, threshold: number) {
    super();
    this._termOp = new TermOperator(term);
    this._vectorOp = new VectorSimilarityOperator(queryVector, threshold);
  }

  execute(context: ExecutionContext): PostingList {
    const textResult = this._termOp.execute(context);
    const vecResult = this._vectorOp.execute(context);
    return textResult.intersect(vecResult);
  }

  costEstimate(stats: IndexStats): number {
    return Math.min(
      this._termOp.costEstimate(stats),
      this._vectorOp.costEstimate(stats),
    );
  }
}

// -- SemanticFilterOperator --------------------------------------------------

export class SemanticFilterOperator extends Operator {
  readonly source: Operator;
  private readonly _vectorOp: VectorSimilarityOperator;

  constructor(source: Operator, queryVector: Float64Array, threshold: number) {
    super();
    this.source = source;
    this._vectorOp = new VectorSimilarityOperator(queryVector, threshold);
  }

  execute(context: ExecutionContext): PostingList {
    const sourceResult = this.source.execute(context);
    const vecResult = this._vectorOp.execute(context);
    return sourceResult.intersect(vecResult);
  }

  costEstimate(stats: IndexStats): number {
    return Math.min(
      this.source.costEstimate(stats),
      this._vectorOp.costEstimate(stats),
    );
  }
}

// -- LogOddsFusionOperator ---------------------------------------------------

export class LogOddsFusionOperator extends Operator {
  readonly signals: Operator[];
  readonly alpha: number;
  readonly topK: number | null;
  readonly gating: string | null;
  readonly gatingBeta: number | null;

  constructor(
    signals: Operator[],
    alpha = 0.5,
    topK?: number | null,
    gating?: string | null,
    gatingBeta?: number | null,
  ) {
    super();
    this.signals = signals;
    this.alpha = alpha;
    this.topK = topK ?? null;
    this.gating = gating ?? null;
    this.gatingBeta = gatingBeta ?? null;
  }

  execute(context: ExecutionContext): PostingList {
    const signalResults = this.signals.map((s) => s.execute(context));

    if (this.topK !== null) {
      const upperBounds = signalResults.map((pl) => {
        let max = 0;
        for (const e of pl) {
          if (e.payload.score > max) max = e.payload.score;
        }
        return Math.max(max, 0.5);
      });
      const fwand = new FusionWANDScorer(
        signalResults,
        upperBounds,
        this.alpha,
        this.topK,
        this.gating,
      );
      return fwand.scoreTopK();
    }

    // Build score maps
    const scoreMaps: Map<DocId, number>[] = [];
    const allDocIds = new Set<DocId>();
    for (const pl of signalResults) {
      const m = new Map<DocId, number>();
      for (const e of pl) {
        m.set(e.docId, e.payload.score);
        allDocIds.add(e.docId);
      }
      scoreMaps.push(m);
    }

    const sortedIds = [...allDocIds].sort((a, b) => a - b);
    const numDocs = sortedIds.length;
    const numSignals = scoreMaps.length;

    if (numDocs === 0) {
      return new PostingList();
    }

    // Compute per-signal default probability based on coverage ratio.
    const defaults = scoreMaps.map((m) => coverageBasedDefault(m.size, numDocs));

    const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
    const alpha = this.alpha;
    for (const docId of sortedIds) {
      const probs: number[] = [];
      for (let j = 0; j < numSignals; j++) {
        probs.push(scoreMaps[j]!.get(docId) ?? defaults[j]!);
      }
      let fused: number;
      if (numSignals === 1) {
        fused = probs[0]!;
      } else {
        fused = logOddsConjunction(
          probs,
          alpha,
          undefined,
          this.gating ?? "none",
          this.gatingBeta ?? undefined,
        );
      }
      entries.push({ docId, payload: createPayload({ score: fused }) });
    }

    return PostingList.fromSorted(entries);
  }

  costEstimate(stats: IndexStats): number {
    let sum = 0;
    for (const s of this.signals) sum += s.costEstimate(stats);
    return sum;
  }
}

// -- ProbBoolFusionOperator --------------------------------------------------

export class ProbBoolFusionOperator extends Operator {
  readonly signals: Operator[];
  readonly mode: "and" | "or";

  constructor(signals: Operator[], mode: "and" | "or" = "and") {
    super();
    this.signals = signals;
    this.mode = mode;
  }

  execute(context: ExecutionContext): PostingList {
    const signalResults = this.signals.map((s) => s.execute(context));
    const scoreMaps: Map<DocId, number>[] = [];
    const allDocIds = new Set<DocId>();
    for (const pl of signalResults) {
      const m = new Map<DocId, number>();
      for (const e of pl) {
        m.set(e.docId, e.payload.score);
        allDocIds.add(e.docId);
      }
      scoreMaps.push(m);
    }

    const sortedIds = [...allDocIds].sort((a, b) => a - b);
    const numDocs = sortedIds.length;
    const defaults = scoreMaps.map((m) => coverageBasedDefault(m.size, numDocs));

    const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
    for (const docId of sortedIds) {
      const probs: number[] = [];
      for (let s = 0; s < scoreMaps.length; s++) {
        probs.push(scoreMaps[s]!.get(docId) ?? defaults[s]!);
      }
      const fused = this.mode === "and" ? probAnd(probs) : probOr(probs);
      entries.push({ docId, payload: createPayload({ score: fused }) });
    }

    return PostingList.fromSorted(entries);
  }

  costEstimate(stats: IndexStats): number {
    let sum = 0;
    for (const s of this.signals) sum += s.costEstimate(stats);
    return sum;
  }
}

// -- VectorExclusionOperator -------------------------------------------------

export class VectorExclusionOperator extends Operator {
  readonly positive: Operator;
  private readonly _negativeOp: VectorSimilarityOperator;

  constructor(
    positive: Operator,
    negativeVector: Float64Array,
    negativeThreshold: number,
    field = "embedding",
  ) {
    super();
    this.positive = positive;
    this._negativeOp = new VectorSimilarityOperator(
      negativeVector,
      negativeThreshold,
      field,
    );
  }

  execute(context: ExecutionContext): PostingList {
    const posResult = this.positive.execute(context);
    const negResult = this._negativeOp.execute(context);
    const negativeIds = new Set<DocId>();
    for (const entry of negResult) {
      negativeIds.add(entry.docId);
    }
    const entries = [...posResult].filter((e) => !negativeIds.has(e.docId));
    return PostingList.fromSorted(entries);
  }

  costEstimate(stats: IndexStats): number {
    return this.positive.costEstimate(stats) + this._negativeOp.costEstimate(stats);
  }
}

// -- FacetVectorOperator -----------------------------------------------------

export class FacetVectorOperator extends Operator {
  readonly facetField: string;
  private readonly _vectorOp: VectorSimilarityOperator;
  readonly source: Operator | null;

  constructor(
    facetField: string,
    queryVector: Float64Array,
    threshold: number,
    source?: Operator | null,
  ) {
    super();
    this.facetField = facetField;
    this._vectorOp = new VectorSimilarityOperator(queryVector, threshold);
    this.source = source ?? null;
  }

  execute(context: ExecutionContext): PostingList {
    const vectorPl = this._vectorOp.execute(context);
    const vectorIds = new Set<DocId>();
    for (const entry of vectorPl) {
      vectorIds.add(entry.docId);
    }

    let candidateIds: DocId[];
    if (this.source !== null) {
      const sourcePl = this.source.execute(context);
      candidateIds = [...sourcePl]
        .filter((e) => vectorIds.has(e.docId))
        .map((e) => e.docId);
    } else {
      candidateIds = [...vectorIds].sort((a, b) => a - b);
    }

    const docStore = context.documentStore;
    if (!docStore) return new PostingList();

    const counts = new Map<string, number>();
    for (const docId of candidateIds) {
      const value = docStore.getField(docId, this.facetField);
      if (value !== null && value !== undefined) {
        const key = String(value as string | number);
        counts.set(key, (counts.get(key) ?? 0) + 1);
      }
    }

    const sorted = [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
    return PostingList.fromSorted(
      sorted.map(([value, count], idx) => ({
        docId: idx,
        payload: createPayload({
          score: count,
          fields: {
            _facet_field: this.facetField,
            _facet_value: value,
            _facet_count: count,
          },
        }),
      })),
    );
  }

  costEstimate(stats: IndexStats): number {
    let cost = this._vectorOp.costEstimate(stats);
    if (this.source !== null) {
      cost += this.source.costEstimate(stats);
    }
    return cost;
  }
}

// -- ProbNotOperator ---------------------------------------------------------

export class ProbNotOperator extends Operator {
  readonly signal: Operator;
  readonly defaultProb: number;

  constructor(signal: Operator, defaultProb = 0.01) {
    super();
    this.signal = signal;
    this.defaultProb = defaultProb;
  }

  execute(context: ExecutionContext): PostingList {
    const signalPl = this.signal.execute(context);
    const docStore = context.documentStore;

    const signalMap = new Map<DocId, number>();
    for (const e of signalPl) {
      signalMap.set(e.docId, e.payload.score);
    }

    const allDocIds = new Set<DocId>(signalMap.keys());
    if (docStore) {
      for (const docId of docStore.docIds) {
        allDocIds.add(docId);
      }
    }

    const sorted = [...allDocIds].sort((a, b) => a - b);
    const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
    for (const docId of sorted) {
      const p = signalMap.get(docId) ?? this.defaultProb;
      const notP = 1.0 - p;
      entries.push({ docId, payload: createPayload({ score: notP }) });
    }

    return PostingList.fromSorted(entries);
  }

  costEstimate(stats: IndexStats): number {
    return this.signal.costEstimate(stats);
  }
}

// -- AdaptiveLogOddsFusionOperator -------------------------------------------

/**
 * Signal quality metrics for adaptive fusion.
 */
export interface SignalQuality {
  coverageRatio: number;
  scoreVariance: number;
  calibrationError: number;
}

export class AdaptiveLogOddsFusionOperator extends Operator {
  readonly signals: Operator[];
  readonly baseAlpha: number;
  readonly gating: string | null;

  constructor(signals: Operator[], baseAlpha = 0.5, gating?: string | null) {
    super();
    this.signals = signals;
    this.baseAlpha = baseAlpha;
    this.gating = gating ?? null;
  }

  execute(context: ExecutionContext): PostingList {
    const postingLists = this.signals.map((s) => s.execute(context));

    const allDocIds = new Set<DocId>();
    const scoreMaps: Map<DocId, number>[] = [];
    for (const pl of postingLists) {
      const smap = new Map<DocId, number>();
      for (const entry of pl) {
        smap.set(entry.docId, entry.payload.score);
        allDocIds.add(entry.docId);
      }
      scoreMaps.push(smap);
    }

    const sortedIds = [...allDocIds].sort((a, b) => a - b);
    const numDocs = sortedIds.length;

    if (numDocs === 0) {
      return new PostingList();
    }

    // Compute signal quality metrics
    const qualities: SignalQuality[] = [];
    for (const smap of scoreMaps) {
      const coverage = numDocs > 0 ? smap.size / numDocs : 0.0;
      const scores = [...smap.values()];
      let variance = 0.0;
      if (scores.length > 1) {
        let meanS = 0;
        for (const s of scores) meanS += s;
        meanS /= scores.length;
        let sumSq = 0;
        for (const s of scores) sumSq += (s - meanS) * (s - meanS);
        variance = sumSq / scores.length;
      }
      // Calibration error: |mean_score - 0.5| as proxy
      let meanScore = 0.5;
      if (scores.length > 0) {
        let sum = 0;
        for (const s of scores) sum += s;
        meanScore = sum / scores.length;
      }
      const calError = Math.abs(meanScore - 0.5);
      qualities.push({
        coverageRatio: coverage,
        scoreVariance: variance,
        calibrationError: calError,
      });
    }

    const defaults = scoreMaps.map((m) => coverageBasedDefault(m.size, numDocs));

    // Adaptive fusion: weight each signal's alpha by its quality
    // Higher coverage + lower calibration error = higher confidence
    const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
    for (const docId of sortedIds) {
      const probs: number[] = [];
      for (let j = 0; j < scoreMaps.length; j++) {
        probs.push(scoreMaps[j]!.get(docId) ?? defaults[j]!);
      }

      // Compute adaptive per-signal alphas
      const adaptiveAlphas: number[] = [];
      for (const q of qualities) {
        // Scale alpha by coverage and inverse calibration error
        const qualityWeight = q.coverageRatio * (1.0 - q.calibrationError);
        adaptiveAlphas.push(this.baseAlpha * (0.5 + qualityWeight));
      }

      // Use the average adaptive alpha for the fusion
      let avgAlpha = this.baseAlpha;
      if (adaptiveAlphas.length > 0) {
        let sum = 0;
        for (const a of adaptiveAlphas) sum += a;
        avgAlpha = sum / adaptiveAlphas.length;
      }

      const fused = logOddsConjunction(
        probs,
        avgAlpha,
        undefined,
        this.gating ?? "none",
      );
      entries.push({ docId, payload: createPayload({ score: fused }) });
    }

    return PostingList.fromSorted(entries);
  }

  costEstimate(stats: IndexStats): number {
    let sum = 0;
    for (const s of this.signals) sum += s.costEstimate(stats);
    return sum;
  }
}
