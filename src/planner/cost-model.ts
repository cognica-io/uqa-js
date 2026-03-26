//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- cost model for operator trees
// 1:1 port of uqa/planner/cost_model.py

import type { IndexStats } from "../core/types.js";
import type { Operator } from "../operators/base.js";
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
  IndexScanOperator,
} from "../operators/primitive.js";
import { SparseThresholdOperator } from "../operators/sparse.js";
import { MultiStageOperator } from "../operators/multi-stage.js";
import { MultiFieldSearchOperator } from "../operators/multi-field.js";
import {
  HybridTextVectorOperator,
  SemanticFilterOperator,
  LogOddsFusionOperator,
  ProbBoolFusionOperator,
  VectorExclusionOperator,
  FacetVectorOperator,
  ProbNotOperator,
  AdaptiveLogOddsFusionOperator,
} from "../operators/hybrid.js";
import { AggregateOperator, GroupByOperator } from "../operators/aggregation.js";
import { AttentionFusionOperator } from "../operators/attention.js";
import { LearnedFusionOperator } from "../operators/learned-fusion.js";
import { MessagePassingOperator } from "../graph/message-passing.js";
import { GraphEmbeddingOperator } from "../graph/graph-embedding.js";
import {
  TraverseOperator,
  PatternMatchOperator,
  RegularPathQueryOperator,
  VertexAggregationOperator,
} from "../graph/operators.js";
import { TemporalTraverseOperator } from "../graph/temporal-traverse.js";
import { TemporalPatternMatchOperator } from "../graph/temporal-pattern-match.js";
import {
  PageRankOperator,
  HITSOperator,
  BetweennessCentralityOperator,
} from "../graph/centrality.js";
import {
  TextSimilarityJoinOperator,
  VectorSimilarityJoinOperator,
  GraphJoinOperator,
  HybridJoinOperator,
  CrossParadigmJoinOperator,
} from "../joins/cross-paradigm.js";
import { ProgressiveFusionOperator } from "../operators/progressive-fusion.js";

// Named constants for cost estimation weights
const SCORE_OVERHEAD_FACTOR = 1.1;
const FILTER_SCAN_FRACTION = 0.1;
const GROUP_BY_OVERHEAD_FACTOR = 1.5;
const VERTEX_AGG_FRACTION = 0.2;
const TRAVERSE_FRACTION = 0.1;

export interface GraphStats {
  numVertices: number;
  avgOutDegree: number;
  labelSelectivity(label: string | null): number;
}

// ---------------------------------------------------------------------------
// CostModel
// ---------------------------------------------------------------------------

/**
 * Operator cost estimation for query optimization (Definition 6.2.1, Paper 1).
 */
export class CostModel {
  private _graphStats: GraphStats | null;

  constructor(graphStats?: GraphStats | null) {
    this._graphStats = graphStats ?? null;
  }

  estimate(op: Operator, stats: IndexStats): number {
    // -- Terminal operators --

    if (op instanceof TermOperator) {
      const fieldName = op.field ?? "_default";
      return stats.totalDocs > 0 ? stats.docFreq(fieldName, op.term) : 1.0;
    }

    if (op instanceof VectorSimilarityOperator) {
      return stats.dimensions * Math.log2(stats.totalDocs + 1);
    }

    if (op instanceof KNNOperator) {
      return stats.dimensions * Math.log2(stats.totalDocs + 1);
    }

    if (op instanceof IndexScanOperator) {
      return op.costEstimate(stats);
    }

    if (op instanceof ScoreOperator) {
      return this.estimate(op.source, stats) * SCORE_OVERHEAD_FACTOR;
    }

    if (op instanceof FilterOperator) {
      const base = stats.totalDocs;
      if (op.source !== null) {
        return this.estimate(op.source, stats) + base * FILTER_SCAN_FRACTION;
      }
      return base;
    }

    // -- Boolean operators --

    if (op instanceof IntersectOperator) {
      let total = 0;
      for (const child of op.operands) {
        total += this.estimate(child, stats);
      }
      return total;
    }

    if (op instanceof UnionOperator) {
      let total = 0;
      for (const child of op.operands) {
        total += this.estimate(child, stats);
      }
      return total;
    }

    if (op instanceof ComplementOperator) {
      return this.estimate(op.operand, stats) + stats.totalDocs;
    }

    // -- Aggregation operators --

    if (op instanceof AggregateOperator) {
      return stats.totalDocs;
    }

    if (op instanceof GroupByOperator) {
      return stats.totalDocs * GROUP_BY_OVERHEAD_FACTOR;
    }

    // -- Fusion operators --

    if (op instanceof LogOddsFusionOperator) {
      let total = 0;
      for (const s of op.signals) {
        total += this.estimate(s, stats);
      }
      return total;
    }

    if (op instanceof ProbBoolFusionOperator) {
      let total = 0;
      for (const s of op.signals) {
        total += this.estimate(s, stats);
      }
      return total;
    }

    if (op instanceof AdaptiveLogOddsFusionOperator) {
      let total = 0;
      for (const s of (op as unknown as { signals: Operator[] }).signals) {
        total += this.estimate(s, stats);
      }
      return total;
    }

    if (op instanceof AttentionFusionOperator) {
      let total = 0;
      for (const s of (op as unknown as { signals: Operator[] }).signals) {
        total += this.estimate(s, stats);
      }
      return total;
    }

    if (op instanceof LearnedFusionOperator) {
      let total = 0;
      for (const s of (op as unknown as { signals: Operator[] }).signals) {
        total += this.estimate(s, stats);
      }
      return total;
    }

    if (op instanceof ProbNotOperator) {
      return this.estimate(op.signal, stats) + stats.totalDocs;
    }

    // -- Hybrid operators --

    if (op instanceof HybridTextVectorOperator) {
      const htv = op as unknown as { _termOp: Operator; _vectorOp: Operator };
      return this.estimate(htv._termOp, stats) + this.estimate(htv._vectorOp, stats);
    }

    if (op instanceof SemanticFilterOperator) {
      const sf = op as unknown as { source: Operator; _vectorOp: Operator };
      return this.estimate(sf.source, stats) + this.estimate(sf._vectorOp, stats);
    }

    if (op instanceof VectorExclusionOperator) {
      const ve = op as unknown as { positive: Operator; _negativeOp: Operator };
      return this.estimate(ve.positive, stats) + this.estimate(ve._negativeOp, stats);
    }

    if (op instanceof FacetVectorOperator) {
      const fv = op as unknown as { _vectorOp: Operator; source: Operator | null };
      let cost = this.estimate(fv._vectorOp, stats);
      if (fv.source !== null) {
        cost += this.estimate(fv.source, stats);
      }
      return cost;
    }

    // -- Graph operators --

    if (op instanceof VertexAggregationOperator) {
      return stats.totalDocs * VERTEX_AGG_FRACTION;
    }

    if (op instanceof TraverseOperator) {
      if (this._graphStats !== null) {
        const gs = this._graphStats;
        const label = (op as unknown as { label: string | null }).label;
        const hops = (op as unknown as { maxHops: number }).maxHops;
        const sel = gs.labelSelectivity(label);
        const d = gs.avgOutDegree * sel;
        // O(sum d^i for i=1..hops)
        let cost = 0;
        for (let i = 1; i <= hops; i++) {
          cost += Math.pow(d, i);
        }
        return Math.max(1.0, cost);
      }
      return stats.totalDocs * TRAVERSE_FRACTION;
    }

    if (op instanceof PatternMatchOperator) {
      const pm = op as unknown as {
        pattern: { vertexPatterns: unknown[]; edgePatterns: { negated: boolean }[] };
      };
      let baseCost: number;
      if (this._graphStats !== null) {
        const gs = this._graphStats;
        const nv = gs.numVertices > 0 ? gs.numVertices : stats.totalDocs;
        const k = pm.pattern.vertexPatterns.length;
        // O(V^k) with pruning factor
        baseCost = Math.max(1.0, Math.pow(nv, k) * 0.01);
      } else {
        baseCost = Math.pow(stats.totalDocs, 2);
      }
      // Negation scan overhead
      let negatedCount = 0;
      for (const ep of pm.pattern.edgePatterns) {
        if (ep.negated) negatedCount++;
      }
      if (negatedCount > 0) {
        baseCost *= 1.0 + 0.2 * negatedCount;
      }
      return baseCost;
    }

    if (op instanceof TemporalTraverseOperator) {
      return stats.totalDocs * TRAVERSE_FRACTION;
    }

    if (op instanceof TemporalPatternMatchOperator) {
      return Math.pow(stats.totalDocs, 2);
    }

    if (op instanceof RegularPathQueryOperator) {
      // Path-indexable expressions are cheaper
      const rpq = op as unknown as { pathExpr: unknown };
      const labels = _extractLabelSequence(rpq.pathExpr);
      if (labels !== null) {
        return stats.totalDocs * 0.1;
      }
      if (this._graphStats !== null) {
        const gs = this._graphStats;
        const nv = gs.numVertices;
        const rSize = _exprLabelCount(rpq.pathExpr);
        return Math.max(1.0, nv * nv * rSize * 0.001);
      }
      return Math.pow(stats.totalDocs, 2);
    }

    // -- Sparse / Multi-field / Message passing --

    if (op instanceof SparseThresholdOperator) {
      return this.estimate(op.source, stats) * 0.5;
    }

    if (op instanceof MultiFieldSearchOperator) {
      const mf = op as unknown as { fields: string[] };
      return stats.totalDocs * mf.fields.length;
    }

    if (op instanceof MessagePassingOperator) {
      const mp = op as unknown as { kLayers: number };
      return stats.totalDocs * mp.kLayers;
    }

    if (op instanceof GraphEmbeddingOperator) {
      const ge = op as unknown as { kLayers: number };
      return stats.totalDocs * ge.kLayers * 2;
    }

    if (op instanceof MultiStageOperator) {
      return op.costEstimate(stats);
    }

    // -- Join operators --

    const n = stats.totalDocs;

    if (op instanceof PageRankOperator) {
      const pr = op as unknown as { maxIterations: number };
      return n * pr.maxIterations * 0.1;
    }

    if (op instanceof HITSOperator) {
      const hits = op as unknown as { maxIterations: number };
      return n * hits.maxIterations * 0.2;
    }

    if (op instanceof BetweennessCentralityOperator) {
      return n * n * 0.5;
    }

    if (op instanceof TextSimilarityJoinOperator) {
      return 2.0 * n * Math.max(stats.dimensions, 10);
    }

    if (op instanceof VectorSimilarityJoinOperator) {
      return n * n * Math.max(stats.dimensions, 1);
    }

    if (op instanceof GraphJoinOperator) {
      return n * 10.0;
    }

    if (op instanceof HybridJoinOperator) {
      return n + n * Math.max(stats.dimensions, 1);
    }

    if (op instanceof CrossParadigmJoinOperator) {
      return n * 10.0;
    }

    if (op instanceof ProgressiveFusionOperator) {
      return op.costEstimate(stats);
    }

    // Fallback
    return n;
  }
}

// -- Helpers ------------------------------------------------------------------

import {
  Label,
  Concat,
  Alternation,
  KleeneStar,
  BoundedLabel,
} from "../graph/pattern.js";

function _extractLabelSequence(expr: unknown): string[] | null {
  if (expr instanceof Label) {
    return [(expr as unknown as { label: string }).label];
  }
  if (expr instanceof Concat) {
    const c = expr as unknown as { left: unknown; right: unknown };
    const leftLabels = _extractLabelSequence(c.left);
    const rightLabels = _extractLabelSequence(c.right);
    if (leftLabels !== null && rightLabels !== null) {
      return [...leftLabels, ...rightLabels];
    }
    return null;
  }
  return null;
}

function _exprLabelCount(expr: unknown): number {
  if (expr instanceof Label) {
    return 1;
  }
  if (expr instanceof Concat || expr instanceof Alternation) {
    const c = expr as unknown as { left: unknown; right: unknown };
    return _exprLabelCount(c.left) + _exprLabelCount(c.right);
  }
  if (expr instanceof KleeneStar) {
    const k = expr as unknown as { inner: unknown };
    return _exprLabelCount(k.inner) * 2;
  }
  if (expr instanceof BoundedLabel) {
    const b = expr as unknown as { inner: unknown; maxHops: number };
    return _exprLabelCount(b.inner) * b.maxHops;
  }
  return 1;
}
