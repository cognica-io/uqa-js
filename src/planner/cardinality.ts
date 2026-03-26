//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- cardinality estimator
// 1:1 port of uqa/planner/cardinality.py

import type { IndexStats } from "../core/types.js";
import type { ColumnStats } from "../sql/table.js";
import { Operator } from "../operators/base.js";
import {
  UnionOperator,
  IntersectOperator,
  ComplementOperator,
} from "../operators/boolean.js";
import {
  TermOperator,
  VectorSimilarityOperator,
  KNNOperator,
  FilterOperator,
  ScoreOperator,
} from "../operators/primitive.js";
import { SparseThresholdOperator } from "../operators/sparse.js";
import { MultiStageOperator } from "../operators/multi-stage.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GraphStats
// ---------------------------------------------------------------------------

export class GraphStats {
  numVertices: number;
  numEdges: number;
  labelCounts: Map<string, number>;
  avgOutDegree: number;
  degreeDistribution: Map<number, number>;
  minTimestamp: number | null;
  maxTimestamp: number | null;
  graphName: string;
  vertexLabelCounts: Map<string, number>;
  labelDegreeMap: Map<string, number>;

  constructor(opts?: {
    numVertices?: number;
    numEdges?: number;
    labelCounts?: Map<string, number>;
    avgOutDegree?: number;
    degreeDistribution?: Map<number, number>;
    minTimestamp?: number | null;
    maxTimestamp?: number | null;
    graphName?: string;
    vertexLabelCounts?: Map<string, number>;
    labelDegreeMap?: Map<string, number>;
  }) {
    this.numVertices = opts?.numVertices ?? 0;
    this.numEdges = opts?.numEdges ?? 0;
    this.labelCounts = opts?.labelCounts ?? new Map<string, number>();
    this.avgOutDegree = opts?.avgOutDegree ?? 0.0;
    this.degreeDistribution = opts?.degreeDistribution ?? new Map<number, number>();
    this.minTimestamp = opts?.minTimestamp ?? null;
    this.maxTimestamp = opts?.maxTimestamp ?? null;
    this.graphName = opts?.graphName ?? "";
    this.vertexLabelCounts = opts?.vertexLabelCounts ?? new Map<string, number>();
    this.labelDegreeMap = opts?.labelDegreeMap ?? new Map<string, number>();
  }

  /**
   * Compute statistics from a graph store instance.
   */
  static fromGraphStore(graphStore: unknown, graph = ""): GraphStats {
    const store = graphStore as {
      _vertices?: Map<number, unknown>;
      _edges?: Map<number, { label: string }>;
      verticesInGraph?: (g: string) => unknown[];
      edgesInGraph?: (g: string) => { label: string }[];
      vertexLabelCounts?: (g: string) => Map<string, number>;
      labelDegree?: (label: string, g: string) => number;
      degreeDistribution?: (g: string) => Map<number, number>;
    };

    if (graph && store.verticesInGraph && store.edgesInGraph) {
      return GraphStats._fromNamedGraph(store, graph);
    }

    // Fallback: global stats
    const vertices = store._vertices ?? new Map<number, unknown>();
    const edges = store._edges ?? new Map<number, { label: string }>();
    const numV = vertices.size;
    const numE = edges.size;

    const labelCounts = new Map<string, number>();
    for (const edge of edges.values()) {
      const label = edge.label;
      labelCounts.set(label, (labelCounts.get(label) ?? 0) + 1);
    }

    const avgOut = numV > 0 ? numE / numV : 0.0;

    return new GraphStats({
      numVertices: numV,
      numEdges: numE,
      labelCounts,
      avgOutDegree: avgOut,
    });
  }

  private static _fromNamedGraph(store: unknown, graph: string): GraphStats {
    const gs = store as {
      verticesInGraph: (g: string) => unknown[];
      edgesInGraph: (g: string) => { label: string }[];
      vertexLabelCounts: (g: string) => Map<string, number>;
      labelDegree: (label: string, g: string) => number;
      degreeDistribution: (g: string) => Map<number, number>;
    };

    const vertices = gs.verticesInGraph(graph);
    const edges = gs.edgesInGraph(graph);
    const numV = vertices.length;
    const numE = edges.length;

    const labelCounts = new Map<string, number>();
    for (const edge of edges) {
      labelCounts.set(edge.label, (labelCounts.get(edge.label) ?? 0) + 1);
    }

    const avgOut = numV > 0 ? numE / numV : 0.0;
    const vlc = gs.vertexLabelCounts(graph);
    const ldm = new Map<string, number>();
    for (const label of labelCounts.keys()) {
      ldm.set(label, gs.labelDegree(label, graph));
    }
    const dd = gs.degreeDistribution(graph);

    return new GraphStats({
      numVertices: numV,
      numEdges: numE,
      labelCounts,
      avgOutDegree: avgOut,
      degreeDistribution: dd,
      graphName: graph,
      vertexLabelCounts: vlc,
      labelDegreeMap: ldm,
    });
  }

  labelSelectivity(label: string | null): number {
    if (label === null || this.numEdges === 0) return 1.0;
    return (this.labelCounts.get(label) ?? 0) / this.numEdges;
  }

  edgeDensity(): number {
    if (this.numVertices <= 1) return 0.0;
    return this.numEdges / this.numVertices ** 2;
  }
}

// ---------------------------------------------------------------------------
// CardinalityEstimator
// ---------------------------------------------------------------------------

export class CardinalityEstimator {
  private readonly _columnStats: Map<string, ColumnStats>;
  private readonly _graphStats: GraphStats | null;
  private readonly _graphStore: unknown;

  constructor(
    columnStatsOrOpts?:
      | Map<string, ColumnStats>
      | {
          columnStats?: Map<string, ColumnStats>;
          graphStats?: GraphStats;
          graphStore?: unknown;
        }
      | null,
    opts?: { graphStats?: GraphStats; graphStore?: unknown },
  ) {
    if (columnStatsOrOpts instanceof Map) {
      this._columnStats = columnStatsOrOpts;
      this._graphStats = opts?.graphStats ?? null;
      this._graphStore = opts?.graphStore ?? null;
    } else if (
      columnStatsOrOpts !== null &&
      columnStatsOrOpts !== undefined &&
      !(columnStatsOrOpts instanceof Map)
    ) {
      this._columnStats =
        columnStatsOrOpts.columnStats ?? new Map<string, ColumnStats>();
      this._graphStats = columnStatsOrOpts.graphStats ?? opts?.graphStats ?? null;
      this._graphStore = columnStatsOrOpts.graphStore ?? opts?.graphStore ?? null;
    } else {
      this._columnStats = new Map<string, ColumnStats>();
      this._graphStats = opts?.graphStats ?? null;
      this._graphStore = opts?.graphStore ?? null;
    }
  }

  estimate(op: Operator, stats: IndexStats): number {
    const n = stats.totalDocs > 0 ? stats.totalDocs : 1.0;

    if (op instanceof TermOperator) {
      const fieldName = op.field ?? "_default";
      return stats.docFreq(fieldName, op.term);
    }
    if (op instanceof VectorSimilarityOperator) {
      return n * CardinalityEstimator._vectorSelectivity(op.threshold);
    }
    if (op instanceof KNNOperator) {
      return op.k;
    }
    if (op instanceof FilterOperator) {
      return n * this._filterSelectivity(op.field, op.predicate, n);
    }
    if (op instanceof ScoreOperator) {
      return this.estimate(op.source, stats);
    }
    if (op instanceof IntersectOperator) {
      const childCards = op.operands
        .map((o) => this.estimate(o, stats))
        .sort((a, b) => a - b);
      if (childCards.length === 0) return 0.0;

      const damping = this._intersectionDamping(op.operands);
      let result = childCards[0]!;
      for (let i = 1; i < childCards.length; i++) {
        const sel = n > 0 ? childCards[i]! / n : 1.0;
        result *= sel ** damping;
      }

      // Apply entropy-based lower bound when column stats are available
      if (this._columnStats.size > 0) {
        const entropies: number[] = [];
        for (const opItem of op.operands) {
          if (opItem instanceof FilterOperator && opItem.field) {
            const cs = this._columnStats.get(opItem.field);
            if (cs !== undefined) {
              entropies.push(columnEntropy(cs));
            }
          }
        }
        if (entropies.length > 0) {
          const lb = entropyCardinalityLowerBound(n, entropies);
          result = Math.max(result, lb);
        }
      }

      return Math.max(1.0, result);
    }
    if (op instanceof UnionOperator) {
      const childCards = op.operands.map((o) => this.estimate(o, stats));
      return Math.min(
        n,
        childCards.reduce((a, b) => a + b, 0),
      );
    }
    if (op instanceof ComplementOperator) {
      const innerCard = this.estimate(op.operand, stats);
      return Math.max(0.0, n - innerCard);
    }

    // Threshold / score / multi-stage
    if (op instanceof SparseThresholdOperator) {
      return this.estimate(op.source, stats) * 0.5;
    }
    if (op instanceof MultiStageOperator) {
      if (op.stages.length === 0) return 0;
      const [, cutoff] = op.stages[op.stages.length - 1]!;
      if (typeof cutoff === "number") return cutoff;
      return n * 0.5;
    }

    // Fallback: use operator's own cost estimate
    return Math.max(1, Math.floor(op.costEstimate(stats)));
  }

  private _intersectionDamping(ops: Operator[]): number {
    const fields: string[] = [];
    for (const op of ops) {
      if (op instanceof FilterOperator) {
        fields.push(op.field);
      }
    }

    if (fields.length < 2) return 0.5;
    if (new Set(fields).size === 1) return 0.1;

    // Use mutual information estimate when stats are available
    if (this._columnStats.size > 0 && fields.length >= 2) {
      const csA = this._columnStats.get(fields[0]!);
      const csB = this._columnStats.get(fields[1]!);
      if (csA !== undefined && csB !== undefined) {
        const mi = mutualInformationEstimate(csA, csB, 0.1);
        if (mi > 1.0) return 0.2;
        if (mi > 0.5) return 0.3;
      }
    }

    return 0.5;
  }

  private static _vectorSelectivity(threshold: number): number {
    if (threshold >= 0.9) return 0.01;
    if (threshold >= 0.7) return 0.05;
    if (threshold >= 0.5) return 0.1;
    return 0.2;
  }

  /**
   * Estimate cardinality of a join operand, handling untyped objects.
   */
  estimateJoinSide(side: unknown, stats: IndexStats, n: number): number {
    if (side instanceof Operator) {
      return this.estimate(side, stats);
    }
    if (
      side !== null &&
      side !== undefined &&
      typeof (side as { execute?: unknown }).execute === "function"
    ) {
      return n;
    }
    return n;
  }

  /**
   * Approximate cardinality via random walk sampling (Section 6.3, Paper 2).
   * Returns estimated number of pattern matches, or -1 if sampling
   * is unavailable (no graph store or empty graph).
   */
  sampleGraphCardinality(
    pattern: {
      vertexPatterns?: { variable: string; constraints: ((v: unknown) => boolean)[] }[];
      edgePatterns?: {
        sourceVar: string;
        targetVar: string;
        label: string | null;
      }[];
    },
    sampleSize = 100,
  ): number {
    if (this._graphStats === null) return -1;
    const nv = this._graphStats.numVertices;
    if (nv <= 0) return 0;

    const vertexPatterns = pattern.vertexPatterns ?? [];
    const edgePatterns = pattern.edgePatterns ?? [];
    const k = vertexPatterns.length;
    if (k === 0) return 0;

    // Without access to the actual graph store, fall back to formula-based estimate
    const ne = this._graphStats.numEdges;
    const density = ne > 0 && nv > 0 ? ne / (nv * nv) : 0.01;

    // Label selectivity for edges
    let labelSel = 1.0;
    for (const ep of edgePatterns) {
      if (ep.label !== null) {
        labelSel *= this._graphStats.labelSelectivity(ep.label);
      }
    }

    // N^k * density^|E| * label_selectivity
    void sampleSize; // reserved for future random walk implementation
    const rawEstimate =
      Math.pow(nv, k) * Math.pow(density, edgePatterns.length) * labelSel;
    return Math.max(1, Math.floor(rawEstimate));
  }

  estimateJoin(leftCard: number, rightCard: number, domainSize: number): number {
    if (domainSize <= 0) return 0.0;
    return (leftCard * rightCard) / domainSize;
  }

  private _filterSelectivity(field: string, predicate: unknown, _n: number): number {
    const cs = this._columnStats.get(field);
    if (cs === undefined || cs.distinctCount <= 0) return 0.5;

    const ndv = cs.distinctCount;
    let selectivity: number;

    // Check predicate type by duck typing
    const pred = predicate as {
      type?: string;
      target?: unknown;
      low?: unknown;
      high?: unknown;
      values?: unknown[];
    };
    const predType =
      (pred as { constructor?: { name?: string } }).constructor?.name ?? "";

    if (predType === "Equals" || pred.type === "equals") {
      selectivity = CardinalityEstimator._equalitySelectivity(cs, pred.target, ndv);
    } else if (predType === "NotEquals" || pred.type === "not_equals") {
      selectivity =
        1.0 - CardinalityEstimator._equalitySelectivity(cs, pred.target, ndv);
    } else if (predType === "InSet" || pred.type === "in_set") {
      selectivity = Math.min(
        1.0,
        (pred.values ?? []).reduce(
          (sum: number, v: unknown) =>
            sum + CardinalityEstimator._equalitySelectivity(cs, v, ndv),
          0,
        ),
      );
    } else if (predType === "Between" || pred.type === "between") {
      selectivity = this._rangeSelectivity(cs, pred.low, pred.high);
    } else if (
      predType === "GreaterThan" ||
      predType === "GreaterThanOrEqual" ||
      pred.type === "gt" ||
      pred.type === "gte"
    ) {
      selectivity = this._gtSelectivity(cs, pred.target);
    } else if (
      predType === "LessThan" ||
      predType === "LessThanOrEqual" ||
      pred.type === "lt" ||
      pred.type === "lte"
    ) {
      selectivity = this._ltSelectivity(cs, pred.target);
    } else {
      selectivity = 0.5;
    }

    // Entropy-based lower bound
    if (cs.distinctCount > 1) {
      const h = columnEntropy(cs);
      if (h > 0) {
        const minSel = 1.0 / 2.0 ** h;
        selectivity = Math.max(minSel, selectivity);
      }
    }

    return selectivity;
  }

  private static _equalitySelectivity(
    cs: ColumnStats,
    target: unknown,
    ndv: number,
  ): number {
    if (cs.mcvValues.length > 0) {
      for (let i = 0; i < cs.mcvValues.length; i++) {
        if (cs.mcvValues[i] === target) {
          return cs.mcvFrequencies[i]!;
        }
      }
    }
    return ndv > 0 ? 1.0 / ndv : 1.0;
  }

  static histogramFraction(boundaries: unknown[], low: unknown, high: unknown): number {
    if (boundaries.length < 2) return 0.5;

    const nBuckets = boundaries.length - 1;
    let overlapping = 0.0;
    for (let i = 0; i < nBuckets; i++) {
      const bLow = boundaries[i] as number;
      const bHigh = boundaries[i + 1] as number;
      try {
        if ((high as number) < bLow || (low as number) > bHigh) continue;
        if ((low as number) <= bLow && (high as number) >= bHigh) {
          overlapping += 1.0;
        } else {
          const bSpan = bHigh - bLow;
          if (bSpan <= 0) {
            overlapping += 1.0;
            continue;
          }
          const clampLow = Math.max(low as number, bLow);
          const clampHigh = Math.min(high as number, bHigh);
          overlapping += (clampHigh - clampLow) / bSpan;
        }
      } catch {
        overlapping += 1.0;
      }
    }

    return Math.max(0.0, Math.min(1.0, overlapping / nBuckets));
  }

  private _rangeSelectivity(cs: ColumnStats, low: unknown, high: unknown): number {
    if (cs.histogram.length > 0) {
      return CardinalityEstimator.histogramFraction(cs.histogram, low, high);
    }
    if (cs.minValue != null && cs.maxValue != null) {
      try {
        const span = (cs.maxValue as number) - (cs.minValue as number);
        if (span > 0) {
          return Math.max(
            0.0,
            Math.min(1.0, ((high as number) - (low as number)) / span),
          );
        }
      } catch {
        // Fall through
      }
    }
    return 0.25;
  }

  private _gtSelectivity(cs: ColumnStats, target: unknown): number {
    if (cs.histogram.length > 0) {
      return CardinalityEstimator.histogramFraction(
        cs.histogram,
        target,
        cs.histogram[cs.histogram.length - 1],
      );
    }
    if (cs.minValue != null && cs.maxValue != null) {
      try {
        const span = (cs.maxValue as number) - (cs.minValue as number);
        if (span > 0) {
          return Math.max(0.0, ((cs.maxValue as number) - (target as number)) / span);
        }
      } catch {
        // Fall through
      }
    }
    return 1.0 / 3.0;
  }

  private _ltSelectivity(cs: ColumnStats, target: unknown): number {
    if (cs.histogram.length > 0) {
      return CardinalityEstimator.histogramFraction(
        cs.histogram,
        cs.histogram[0],
        target,
      );
    }
    if (cs.minValue != null && cs.maxValue != null) {
      try {
        const span = (cs.maxValue as number) - (cs.minValue as number);
        if (span > 0) {
          return Math.max(0.0, ((target as number) - (cs.minValue as number)) / span);
        }
      } catch {
        // Fall through
      }
    }
    return 1.0 / 3.0;
  }

  // -- Graph cardinality estimators ------------------------------------------

  estimateGraphPattern(patternSize: number, edgeLabels?: string[]): number {
    if (this._graphStats === null) return 1;

    const gs = this._graphStats;
    const nv = gs.numVertices > 0 ? gs.numVertices : 1;

    const vertexCount = Math.max(1, Math.ceil(patternSize / 2));
    const edgeCount = Math.max(0, patternSize - vertexCount);

    const density = nv > 1 ? gs.numEdges / (nv * nv) : 0;

    let labelSel = 1.0;
    if (edgeLabels !== undefined && gs.numEdges > 0) {
      for (const label of edgeLabels) {
        const count = gs.labelCounts.get(label) ?? 0;
        labelSel *= count / gs.numEdges;
      }
    }

    let vertexSel = 1.0;
    if (gs.vertexLabelCounts.size > 0 && nv > 0) {
      let totalVlc = 0;
      for (const v of gs.vertexLabelCounts.values()) totalVlc += v;
      const avgVertexSel = totalVlc / (gs.vertexLabelCounts.size * nv);
      vertexSel = Math.pow(Math.min(1.0, avgVertexSel), vertexCount);
    }

    const estimate =
      Math.pow(nv, vertexCount) * Math.pow(density, edgeCount) * labelSel * vertexSel;

    return Math.max(1, Math.min(nv, Math.floor(estimate)));
  }

  estimatePathQuery(pathLength: number, edgeLabel?: string): number {
    if (this._graphStats === null) return 1;

    const gs = this._graphStats;
    const nv = gs.numVertices > 0 ? gs.numVertices : 1;

    let branching = gs.avgOutDegree;
    if (edgeLabel !== undefined && gs.numEdges > 0) {
      const labelCount = gs.labelCounts.get(edgeLabel) ?? 0;
      const labelSel = labelCount / gs.numEdges;
      branching *= labelSel;
    }

    const raw = nv * Math.pow(branching, pathLength);
    return Math.max(1, Math.min(Math.floor(raw), gs.numEdges));
  }

  estimateTraverse(hops: number, label: string | null = null): number {
    if (this._graphStats !== null) {
      const gs = this._graphStats;
      let branching: number;
      if (label && gs.labelDegreeMap.size > 0) {
        branching =
          gs.labelDegreeMap.get(label) ?? gs.avgOutDegree * gs.labelSelectivity(label);
      } else {
        const sel = gs.labelSelectivity(label);
        branching = gs.avgOutDegree * sel;
      }
      const nv = gs.numVertices > 0 ? gs.numVertices : 1;
      return Math.min(nv, branching ** hops);
    }
    const n = 1000; // fallback
    return Math.min(n, Math.min(n * 0.1, 10.0) ** hops);
  }

  estimatePatternMatch(
    vertexCount: number,
    edgeCount: number,
    edgeLabels: string[] = [],
  ): number {
    if (this._graphStats !== null) {
      const gs = this._graphStats;
      const nv = gs.numVertices > 0 ? gs.numVertices : 1;
      const density = gs.edgeDensity();
      let labelSel = 1.0;
      for (const label of edgeLabels) {
        labelSel *= gs.labelSelectivity(label);
      }
      const estimate = nv ** vertexCount * density ** edgeCount * labelSel;
      return Math.max(1.0, Math.min(nv, estimate));
    }
    return 1;
  }

  // -- Cross-paradigm estimation (fusion/hybrid operators) --------------------

  /**
   * Estimate cardinality for cross-paradigm operators that combine
   * text retrieval, graph traversal, and vector similarity.
   */
  estimateCrossParadigm(op: Operator, stats: IndexStats, paradigm: string): number {
    const n = stats.totalDocs > 0 ? stats.totalDocs : 1.0;

    switch (paradigm) {
      case "text_graph": {
        // Text retrieval intersected with graph traversal
        const textCard = this.estimate(op, stats);
        const graphSel =
          this._graphStats !== null
            ? Math.min(
                1.0,
                this._graphStats.avgOutDegree /
                  Math.max(1, this._graphStats.numVertices),
              )
            : 0.1;
        return Math.max(1, Math.floor(textCard * graphSel));
      }
      case "text_vector": {
        // Text retrieval fused with vector similarity
        const textCard = this.estimate(op, stats);
        return Math.max(1, Math.floor(textCard * 0.3));
      }
      case "graph_vector": {
        // Graph traversal fused with vector similarity
        const graphCard =
          this._graphStats !== null
            ? Math.min(n, this._graphStats.avgOutDegree ** 2)
            : n * 0.1;
        return Math.max(1, Math.floor(graphCard * 0.2));
      }
      case "text_graph_vector": {
        // Triple fusion
        const textCard = this.estimate(op, stats);
        const graphSel =
          this._graphStats !== null
            ? Math.min(
                1.0,
                this._graphStats.avgOutDegree /
                  Math.max(1, this._graphStats.numVertices),
              )
            : 0.1;
        return Math.max(1, Math.floor(textCard * graphSel * 0.3));
      }
      case "log_odds_fusion": {
        // Log-odds fusion: result is bounded by the smallest signal
        return Math.max(1, Math.floor(n * 0.15));
      }
      case "prob_bool_fusion": {
        // Probabilistic boolean fusion
        return Math.max(1, Math.floor(n * 0.25));
      }
      case "attention_fusion": {
        // Attention-weighted fusion
        return Math.max(1, Math.floor(n * 0.2));
      }
      case "learned_fusion": {
        // Learned fusion weights
        return Math.max(1, Math.floor(n * 0.2));
      }
      default:
        return Math.max(1, Math.floor(n * 0.5));
    }
  }

  // -- Temporal pattern estimates ---------------------------------------------

  /**
   * Estimate cardinality for temporal graph pattern queries.
   * Takes into account the time range selectivity.
   */
  estimateTemporalPattern(
    vertexCount: number,
    edgeCount: number,
    edgeLabels: string[],
    timeRangeStart: number | null,
    timeRangeEnd: number | null,
  ): number {
    const baseEstimate = this.estimatePatternMatch(vertexCount, edgeCount, edgeLabels);

    if (this._graphStats === null) return baseEstimate;
    const gs = this._graphStats;

    // Apply time range selectivity
    let timeSel = 1.0;
    if (
      timeRangeStart !== null &&
      timeRangeEnd !== null &&
      gs.minTimestamp !== null &&
      gs.maxTimestamp !== null
    ) {
      const totalSpan = gs.maxTimestamp - gs.minTimestamp;
      if (totalSpan > 0) {
        const querySpan =
          Math.min(timeRangeEnd, gs.maxTimestamp) -
          Math.max(timeRangeStart, gs.minTimestamp);
        timeSel = Math.max(0.0, Math.min(1.0, querySpan / totalSpan));
      }
    } else if (timeRangeStart !== null || timeRangeEnd !== null) {
      // One-sided temporal constraint
      timeSel = 0.5;
    }

    return Math.max(1, Math.floor(baseEstimate * timeSel));
  }

  /**
   * Estimate cardinality for temporal traversal with time-filtered edges.
   */
  estimateTemporalTraverse(
    hops: number,
    label: string | null,
    timeRangeStart: number | null,
    timeRangeEnd: number | null,
  ): number {
    const baseEstimate = this.estimateTraverse(hops, label);

    if (this._graphStats === null) return baseEstimate;
    const gs = this._graphStats;

    let timeSel = 1.0;
    if (
      timeRangeStart !== null &&
      timeRangeEnd !== null &&
      gs.minTimestamp !== null &&
      gs.maxTimestamp !== null
    ) {
      const totalSpan = gs.maxTimestamp - gs.minTimestamp;
      if (totalSpan > 0) {
        const querySpan =
          Math.min(timeRangeEnd, gs.maxTimestamp) -
          Math.max(timeRangeStart, gs.minTimestamp);
        timeSel = Math.max(0.0, Math.min(1.0, querySpan / totalSpan));
      }
    }

    // Temporal selectivity compounds per hop
    return Math.max(1, Math.floor(baseEstimate * Math.pow(timeSel, hops)));
  }

  // -- Weighted RPQ estimates ------------------------------------------------

  /**
   * Estimate cardinality for weighted regular path queries.
   * Weighted RPQ considers edge weights/probabilities alongside the
   * structural path pattern.
   */
  estimateWeightedRPQ(
    pathLength: number,
    edgeLabels: string[],
    weightThreshold: number | null,
  ): number {
    const baseEstimate = this.estimatePathQuery(
      pathLength,
      edgeLabels.length > 0 ? edgeLabels[0] : undefined,
    );

    // Weight threshold filters out low-weight paths
    let weightSel = 1.0;
    if (weightThreshold !== null) {
      // Assume weights are uniformly distributed [0, 1]
      // Paths with all edges above threshold: threshold^pathLength
      weightSel = Math.pow(1.0 - weightThreshold, pathLength);
    }

    // Multiple labels reduce cardinality further
    if (edgeLabels.length > 1 && this._graphStats !== null) {
      const gs = this._graphStats;
      let labelSel = 1.0;
      for (const label of edgeLabels) {
        labelSel *= gs.labelSelectivity(label);
      }
      return Math.max(1, Math.floor(baseEstimate * weightSel * labelSel));
    }

    return Math.max(1, Math.floor(baseEstimate * weightSel));
  }

  /**
   * Estimate cardinality for bounded regular path queries.
   * A bounded RPQ has min/max repetition constraints.
   */
  estimateBoundedRPQ(
    minLength: number,
    maxLength: number,
    edgeLabel: string | null,
  ): number {
    if (this._graphStats === null) return 1;
    const gs = this._graphStats;
    const nv = gs.numVertices > 0 ? gs.numVertices : 1;

    let branching = gs.avgOutDegree;
    if (edgeLabel !== null && gs.numEdges > 0) {
      branching *= gs.labelSelectivity(edgeLabel);
    }

    // Sum of reachable paths for lengths [minLength, maxLength]
    let total = 0;
    for (let len = minLength; len <= maxLength; len++) {
      total += nv * Math.pow(branching, len);
    }

    // Deduplicate (same start/end pair may be reached via different lengths)
    const dedup = total / (maxLength - minLength + 1);
    return Math.max(1, Math.min(Math.floor(dedup), gs.numEdges));
  }
}

// -- Information-Theoretic Bounds (Paper 1, Section 7) --

export function columnEntropy(cs: ColumnStats): number {
  const ndv = cs.distinctCount;
  if (ndv <= 1) return 0.0;

  // If MCV frequencies are available, use them
  const mcvFreqs = cs.mcvFrequencies;
  if (mcvFreqs.length > 0) {
    let entropy = 0.0;
    let remaining = 1.0;
    for (const freq of mcvFreqs) {
      remaining -= freq;
      if (freq > 0) {
        entropy -= freq * Math.log2(freq);
      }
    }
    const remainingNdv = Math.max(1, ndv - mcvFreqs.length);
    if (remaining > 0 && remainingNdv > 0) {
      const p = remaining / remainingNdv;
      entropy -= remaining * Math.log2(p);
    }
    return Math.max(0.0, entropy);
  }

  // If histogram buckets are available
  const histogram = cs.histogram;
  if (histogram.length > 1) {
    const numBuckets = histogram.length - 1;
    const p = 1.0 / numBuckets;
    return -numBuckets * p * Math.log2(p);
  }

  // Uniform assumption
  return Math.log2(ndv);
}

export function mutualInformationEstimate(
  csX: ColumnStats,
  csY: ColumnStats,
  jointSelectivity: number,
): number {
  const hX = columnEntropy(csX);
  const hY = columnEntropy(csY);

  if (jointSelectivity <= 0) return 0.0;

  const ndvX = Math.max(1, csX.distinctCount);
  const ndvY = Math.max(1, csY.distinctCount);
  const independentNdv = ndvX * ndvY;
  const effectiveNdv = Math.max(1, independentNdv * jointSelectivity);
  const hJoint = Math.log2(Math.max(1, effectiveNdv));
  return Math.max(0.0, hX + hY - hJoint);
}

export function entropyCardinalityLowerBound(n: number, entropies: number[]): number {
  if (entropies.length === 0 || n <= 0) return 1.0;
  const totalEntropy = entropies.reduce((a, b) => a + b, 0);
  const lb = n * 2.0 ** -totalEntropy;
  return Math.max(1.0, lb);
}
