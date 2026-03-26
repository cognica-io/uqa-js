//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- query optimizer with rewrite rules
// 1:1 port of uqa/planner/optimizer.py
//
// Rewrite rules (Theorem 6.1.2, Paper 1):
//   1. _simplifyAlgebra     -- algebraic simplification
//   2. _pushFiltersDown     -- filter pushdown into intersections
//   3. _pushGraphPatternFilters -- push filters into graph pattern constraints
//   4. _pushFilterIntoTraverse  -- push vertex filters into BFS pruning
//   5. _pushFilterBelowGraphJoin -- push filters below graph joins
//   6. _fuseJoinPattern     -- fuse intersected pattern matches
//   7. _mergeVectorThresholds -- merge duplicate vector threshold ops
//   8. _reorderIntersect    -- reorder intersection operands by cardinality
//   9. _reorderFusionSignals -- reorder fusion signal inputs by cost
//  10. _applyIndexScan      -- substitute filter scans with index scans

import type { IndexStats } from "../core/types.js";
import type { ColumnStats } from "../sql/table.js";
import type { IndexManager } from "../storage/index-manager.js";
import type { Operator } from "../operators/base.js";
import { ComposedOperator } from "../operators/base.js";
import {
  UnionOperator,
  IntersectOperator,
  ComplementOperator,
} from "../operators/boolean.js";
import {
  TermOperator,
  VectorSimilarityOperator,
  FilterOperator,
  ScoreOperator,
  IndexScanOperator,
} from "../operators/primitive.js";
import { SparseThresholdOperator } from "../operators/sparse.js";
import {
  LogOddsFusionOperator,
  ProbBoolFusionOperator,
  ProbNotOperator,
} from "../operators/hybrid.js";
import { AttentionFusionOperator } from "../operators/attention.js";
import { LearnedFusionOperator } from "../operators/learned-fusion.js";
import type { GraphStats } from "./cardinality.js";
import { CardinalityEstimator } from "./cardinality.js";
import { CostModel } from "./cost-model.js";

// ---------------------------------------------------------------------------
// QueryOptimizer
// ---------------------------------------------------------------------------

export class QueryOptimizer {
  readonly stats: IndexStats;
  readonly estimator: CardinalityEstimator;
  private readonly _costModel: CostModel;
  private readonly _graphStats: GraphStats | null;
  private readonly _indexManager: IndexManager | null;
  private readonly _tableName: string | null;

  constructor(
    stats: IndexStats,
    opts?: {
      columnStats?: Map<string, ColumnStats>;
      indexManager?: IndexManager;
      tableName?: string;
      graphStats?: GraphStats;
    },
  ) {
    this.stats = stats;
    const columnStats = opts?.columnStats ?? new Map();
    this._graphStats = opts?.graphStats ?? null;
    this.estimator = new CardinalityEstimator(columnStats, {
      graphStats: this._graphStats ?? undefined,
    });
    this._costModel = new CostModel(this._graphStats);
    this._indexManager = opts?.indexManager ?? null;
    this._tableName = opts?.tableName ?? null;
  }

  optimize(op: Operator): Operator {
    let current = op;
    current = this._simplifyAlgebra(current);
    current = this._pushFiltersDown(current);
    current = this._pushGraphPatternFilters(current);
    current = this._pushFilterIntoTraverse(current);
    current = this._pushFilterBelowGraphJoin(current);
    current = this._fuseJoinPattern(current);
    current = this._mergeVectorThresholds(current);
    current = this._reorderIntersect(current);
    current = this._reorderFusionSignals(current);
    current = this._applyIndexScan(current);
    return current;
  }

  // -----------------------------------------------------------------------
  // Rule 1: Simplify boolean algebra
  // -----------------------------------------------------------------------

  private _simplifyAlgebra(op: Operator): Operator {
    // Recurse into children first (bottom-up simplification)
    op = this._recurseSimplify(op);

    if (op instanceof IntersectOperator) {
      let operands = op.operands;

      // Empty elimination
      for (const child of operands) {
        if (QueryOptimizer._isEmptyOperator(child)) {
          return new IntersectOperator([]);
        }
      }

      // Idempotent: remove duplicates by identity
      const seen: Operator[] = [];
      for (const child of operands) {
        if (!seen.some((s) => s === child)) {
          seen.push(child);
        }
      }
      operands = seen;

      // Absorption: drop UnionOperator([A, ...]) when A also appears
      const absorbed: Operator[] = [];
      for (const child of operands) {
        if (
          child instanceof UnionOperator &&
          operands.some(
            (other) =>
              other !== child &&
              child.operands.some((uc) =>
                operands.some((o) => o === uc && o !== child),
              ),
          )
        ) {
          continue;
        }
        absorbed.push(child);
      }
      operands = absorbed;

      if (operands.length === 1) return operands[0]!;
      return new IntersectOperator(operands);
    }

    if (op instanceof UnionOperator) {
      let operands = op.operands;

      // Empty elimination
      operands = operands.filter((child) => !QueryOptimizer._isEmptyOperator(child));

      // Idempotent
      const seen: Operator[] = [];
      for (const child of operands) {
        if (!seen.some((s) => s === child)) {
          seen.push(child);
        }
      }
      operands = seen;

      // Absorption: drop IntersectOperator([A, ...]) when A also appears
      const absorbed: Operator[] = [];
      for (const child of operands) {
        if (
          child instanceof IntersectOperator &&
          operands.some(
            (other) =>
              other !== child &&
              child.operands.some((ic) =>
                operands.some((o) => o === ic && o !== child),
              ),
          )
        ) {
          continue;
        }
        absorbed.push(child);
      }
      operands = absorbed;

      if (operands.length === 1) return operands[0]!;
      if (operands.length === 0) return new UnionOperator([]);
      return new UnionOperator(operands);
    }

    return op;
  }

  private _recurseSimplify(op: Operator): Operator {
    if (op instanceof IntersectOperator) {
      return new IntersectOperator(op.operands.map((o) => this._simplifyAlgebra(o)));
    }
    if (op instanceof UnionOperator) {
      return new UnionOperator(op.operands.map((o) => this._simplifyAlgebra(o)));
    }
    if (op instanceof ComplementOperator) {
      return new ComplementOperator(this._simplifyAlgebra(op.operand));
    }
    if (op instanceof FilterOperator && op.source !== null) {
      return new FilterOperator(
        op.field,
        op.predicate,
        this._simplifyAlgebra(op.source),
      );
    }
    if (op instanceof ComposedOperator) {
      return new ComposedOperator(
        (op as unknown as { operators: Operator[] }).operators.map((o: Operator) =>
          this._simplifyAlgebra(o),
        ),
      );
    }
    return op;
  }

  private static _isEmptyOperator(op: Operator): boolean {
    if (op instanceof IntersectOperator || op instanceof UnionOperator) {
      return op.operands.length === 0;
    }
    return false;
  }

  // -----------------------------------------------------------------------
  // Rule 2: Push filters down
  // -----------------------------------------------------------------------

  private _pushFiltersDown(op: Operator): Operator {
    if (!(op instanceof FilterOperator)) {
      return this._recurseChildren(op);
    }

    const source = op.source;
    if (source instanceof IntersectOperator) {
      const newOperands: Operator[] = [];
      let anyPushed = false;
      for (const child of source.operands) {
        if (QueryOptimizer._filterAppliesTo(op, child)) {
          newOperands.push(new FilterOperator(op.field, op.predicate, child));
          anyPushed = true;
        } else {
          newOperands.push(child);
        }
      }
      if (anyPushed) {
        const recursed = newOperands.map((o) => this._pushFiltersDown(o));
        return this._recurseChildren(new IntersectOperator(recursed));
      }
    }

    if (source === null) return op;
    return new FilterOperator(op.field, op.predicate, this._recurseChildren(source));
  }

  // -----------------------------------------------------------------------
  // Rule 3: Push graph pattern filters
  // -----------------------------------------------------------------------

  /**
   * Push filter predicates into graph pattern constraints.
   * When a FilterOperator sits above a PatternMatchOperator or TraverseOperator
   * and filters on vertex/edge properties, the filter can be pushed into the
   * graph operator's own constraint set, reducing intermediate results.
   */
  private _pushGraphPatternFilters(op: Operator): Operator {
    if (op instanceof FilterOperator && op.source !== null) {
      const source = this._pushGraphPatternFilters(op.source);
      const sourceTypeName = source.constructor.name;

      // Check if the source is a graph pattern operator
      if (sourceTypeName === "PatternMatchOperator") {
        // Try to incorporate the filter into the pattern's vertex/edge constraints
        const patternOp = source as unknown as {
          pattern: unknown;
          graph: string;
          score: number;
          withVertexPredicate?: (field: string, pred: unknown) => Operator;
        };
        if (patternOp.withVertexPredicate && op.field) {
          return patternOp.withVertexPredicate(op.field, op.predicate);
        }
      }

      // Check if the source is a TraverseOperator
      if (sourceTypeName === "TraverseOperator") {
        const traverseOp = source as unknown as {
          startVertex: number;
          graph: string;
          label: string | null;
          maxHops: number;
          vertexPredicate: ((v: unknown) => boolean) | null;
          score: number;
        };
        // If filtering on a vertex property, combine with existing vertex predicate
        if (op.field) {
          const existingPred = traverseOp.vertexPredicate;
          const filterField = op.field;
          const filterPred = op.predicate;
          const combinedPred = (v: unknown) => {
            const vertex = v as Record<string, unknown>;
            const passes = filterPred.evaluate(vertex[filterField]);
            if (!passes) return false;
            if (existingPred !== null) return existingPred(v);
            return true;
          };
          try {
            const TraverseOp = source.constructor as new (
              startVertex: number,
              graph: string,
              label: string | null,
              maxHops: number,
              vertexPredicate: ((v: unknown) => boolean) | null,
              score: number,
            ) => Operator;
            return new TraverseOp(
              traverseOp.startVertex,
              traverseOp.graph,
              traverseOp.label,
              traverseOp.maxHops,
              combinedPred,
              traverseOp.score,
            );
          } catch {
            // If construction fails, keep the filter above
          }
        }
      }

      return new FilterOperator(op.field, op.predicate, source);
    }

    return this._recurseGeneric(op, (o) => this._pushGraphPatternFilters(o));
  }

  // -----------------------------------------------------------------------
  // Rule 4: Push filter into traverse
  // -----------------------------------------------------------------------

  /**
   * Push vertex-property filters into BFS pruning within TraverseOperator.
   * This converts a Filter(TraverseOp) into TraverseOp(vertexPredicate=...).
   */
  private _pushFilterIntoTraverse(op: Operator): Operator {
    if (op instanceof FilterOperator && op.source !== null) {
      const source = this._pushFilterIntoTraverse(op.source);
      const sourceTypeName = source.constructor.name;

      if (sourceTypeName === "TraverseOperator") {
        const traverseOp = source as unknown as {
          startVertex: number;
          graph: string;
          label: string | null;
          maxHops: number;
          vertexPredicate: ((v: unknown) => boolean) | null;
          score: number;
        };

        if (op.field) {
          const filterField = op.field;
          const filterPred = op.predicate;
          const existingPred = traverseOp.vertexPredicate;

          const combinedPred = (v: unknown) => {
            const vertex = v as Record<string, unknown>;
            const fieldValue = vertex["properties"]
              ? (vertex["properties"] as Record<string, unknown>)[filterField]
              : vertex[filterField];
            if (!filterPred.evaluate(fieldValue)) return false;
            return existingPred === null || existingPred(v);
          };

          try {
            const TraverseOp = source.constructor as new (
              startVertex: number,
              graph: string,
              label: string | null,
              maxHops: number,
              vertexPredicate: ((v: unknown) => boolean) | null,
              score: number,
            ) => Operator;
            return new TraverseOp(
              traverseOp.startVertex,
              traverseOp.graph,
              traverseOp.label,
              traverseOp.maxHops,
              combinedPred,
              traverseOp.score,
            );
          } catch {
            // Fall through to keep filter above
          }
        }
      }

      return new FilterOperator(op.field, op.predicate, source);
    }

    return this._recurseGeneric(op, (o) => this._pushFilterIntoTraverse(o));
  }

  // -----------------------------------------------------------------------
  // Rule 5: Push filter below graph join
  // -----------------------------------------------------------------------

  /**
   * Push filters below graph join operators when the filter predicate
   * only references columns from one side of the join.
   */
  private _pushFilterBelowGraphJoin(op: Operator): Operator {
    if (op instanceof FilterOperator && op.source !== null) {
      const source = this._pushFilterBelowGraphJoin(op.source);
      const sourceTypeName = source.constructor.name;

      // Check for graph join operators (CrossParadigmJoin, GraphJoin, etc.)
      if (
        sourceTypeName === "CrossParadigmJoinOperator" ||
        sourceTypeName === "GraphJoinOperator"
      ) {
        const joinOp = source as unknown as {
          left: Operator;
          right: Operator;
        };

        {
          // Try to push the filter into the left or right child
          const leftRelevant = QueryOptimizer._filterAppliesTo(op, joinOp.left);
          const rightRelevant = QueryOptimizer._filterAppliesTo(op, joinOp.right);

          if (leftRelevant && !rightRelevant) {
            try {
              const JoinOp = source.constructor as new (
                left: Operator,
                right: Operator,
                ...rest: unknown[]
              ) => Operator;
              return new JoinOp(
                new FilterOperator(op.field, op.predicate, joinOp.left),
                joinOp.right,
              );
            } catch {
              // Fall through
            }
          }
          if (rightRelevant && !leftRelevant) {
            try {
              const JoinOp = source.constructor as new (
                left: Operator,
                right: Operator,
                ...rest: unknown[]
              ) => Operator;
              return new JoinOp(
                joinOp.left,
                new FilterOperator(op.field, op.predicate, joinOp.right),
              );
            } catch {
              // Fall through
            }
          }
        }
      }

      return new FilterOperator(op.field, op.predicate, source);
    }

    return this._recurseGeneric(op, (o) => this._pushFilterBelowGraphJoin(o));
  }

  // -----------------------------------------------------------------------
  // Rule 6: Fuse join patterns
  // -----------------------------------------------------------------------

  /**
   * Fuse intersected pattern match operations into a single combined
   * pattern when they share variables or graph context.
   */
  private _fuseJoinPattern(op: Operator): Operator {
    if (op instanceof IntersectOperator) {
      const children = op.operands.map((c) => this._fuseJoinPattern(c));

      // Collect pattern match operators
      const patternOps: Operator[] = [];
      const otherOps: Operator[] = [];
      for (const child of children) {
        if (child.constructor.name === "PatternMatchOperator") {
          patternOps.push(child);
        } else {
          otherOps.push(child);
        }
      }

      // If we have multiple pattern ops on the same graph, try to fuse
      if (patternOps.length >= 2) {
        const byGraph = new Map<string, Operator[]>();
        for (const pop of patternOps) {
          const graph = (pop as unknown as { graph: string }).graph;
          if (!byGraph.has(graph)) byGraph.set(graph, []);
          byGraph.get(graph)!.push(pop);
        }

        const fusedOps: Operator[] = [];
        for (const [, graphOps] of byGraph) {
          if (graphOps.length >= 2) {
            // Keep the most selective pattern (smallest cost estimate)
            graphOps.sort(
              (a, b) =>
                this._costModel.estimate(a, this.stats) -
                this._costModel.estimate(b, this.stats),
            );
            // Keep the most selective one as the primary, intersect with others
            fusedOps.push(graphOps[0]!);
            // Remaining patterns become filters on the result
            for (let i = 1; i < graphOps.length; i++) {
              fusedOps.push(graphOps[i]!);
            }
          } else {
            fusedOps.push(graphOps[0]!);
          }
        }

        const allOps = [...otherOps, ...fusedOps];
        if (allOps.length === 1) return allOps[0]!;
        return new IntersectOperator(allOps);
      }

      if (children.length === 1) return children[0]!;
      return new IntersectOperator(children);
    }

    return this._recurseGeneric(op, (o) => this._fuseJoinPattern(o));
  }

  // -----------------------------------------------------------------------
  // Rule 7: Merge vector thresholds
  // -----------------------------------------------------------------------

  private _mergeVectorThresholds(op: Operator): Operator {
    if (!(op instanceof IntersectOperator)) {
      return this._recurseChildren(op);
    }

    const vectorOps: VectorSimilarityOperator[] = [];
    const otherOps: Operator[] = [];

    for (let child of op.operands) {
      child = this._recurseChildren(child);
      if (child instanceof VectorSimilarityOperator) {
        vectorOps.push(child);
      } else {
        otherOps.push(child);
      }
    }

    // Merge vector ops with same field
    const mergedVectors: VectorSimilarityOperator[] = [];
    const used = new Array(vectorOps.length).fill(false);

    for (let i = 0; i < vectorOps.length; i++) {
      if (used[i]) continue;
      let merged = vectorOps[i]!;
      for (let j = i + 1; j < vectorOps.length; j++) {
        if (used[j]) continue;
        if (
          merged.field === vectorOps[j]!.field &&
          this._vectorsClose(merged.queryVector, vectorOps[j]!.queryVector)
        ) {
          merged = new VectorSimilarityOperator(
            merged.queryVector,
            Math.max(merged.threshold, vectorOps[j]!.threshold),
            merged.field,
          );
          used[j] = true;
        }
      }
      used[i] = true;
      mergedVectors.push(merged);
    }

    const allOps: Operator[] = [...otherOps, ...mergedVectors];
    if (allOps.length === 1) return allOps[0]!;
    return new IntersectOperator(allOps);
  }

  private _vectorsClose(a: Float64Array, b: Float64Array): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (Math.abs(a[i]! - b[i]!) > 1e-7) return false;
    }
    return true;
  }

  // -----------------------------------------------------------------------
  // Rule 8: Reorder intersection operands by cardinality
  // -----------------------------------------------------------------------

  private _reorderIntersect(op: Operator): Operator {
    if (!(op instanceof IntersectOperator)) {
      return this._recurseChildren(op);
    }

    const children = op.operands.map((c) => this._recurseChildren(c));
    children.sort(
      (a, b) =>
        this._costModel.estimate(a, this.stats) -
        this._costModel.estimate(b, this.stats),
    );
    return new IntersectOperator(children);
  }

  // -----------------------------------------------------------------------
  // Rule 9: Reorder fusion signals
  // -----------------------------------------------------------------------

  private _reorderFusionSignals(op: Operator): Operator {
    // Check for LogOddsFusionOperator and ProbBoolFusionOperator
    if (op instanceof LogOddsFusionOperator) {
      const signals = op.signals.map((s) => this._reorderFusionSignals(s));
      signals.sort(
        (a, b) => this._graphAwareSignalCost(a) - this._graphAwareSignalCost(b),
      );
      return new LogOddsFusionOperator(signals, op.alpha, undefined, op.gating);
    }

    if (op instanceof ProbBoolFusionOperator) {
      const signals = op.signals.map((s) => this._reorderFusionSignals(s));
      signals.sort(
        (a, b) => this._graphAwareSignalCost(a) - this._graphAwareSignalCost(b),
      );
      return new ProbBoolFusionOperator(signals, op.mode);
    }

    return this._recurseGeneric(op, (o) => this._reorderFusionSignals(o));
  }

  /**
   * Compute graph-aware signal cost for fusion operator reordering.
   * Graph operators get discounted costs when graph statistics are available
   * because they can leverage graph indexes for faster execution.
   */
  private _graphAwareSignalCost(signal: Operator): number {
    const base = this.estimator.estimate(signal, this.stats);

    if (this._graphStats !== null) {
      const typeName = signal.constructor.name;

      // Graph operators benefit from graph indexes
      if (typeName === "TraverseOperator") {
        // Traversal cost depends on graph density
        const density = this._graphStats.edgeDensity();
        return base * (density < 0.01 ? 0.3 : 0.5);
      }
      if (typeName === "PatternMatchOperator") {
        // Pattern matching is expensive but can use label indexes
        return base * 0.5;
      }
      if (typeName === "RegularPathQueryOperator") {
        // RPQ uses DFA simulation, cost depends on automaton size
        return base * 0.6;
      }
      if (typeName === "WeightedPathQueryOperator") {
        // Weighted RPQ adds weight computation overhead
        return base * 0.7;
      }
      if (typeName === "TemporalTraverseOperator") {
        // Temporal traversal adds time filtering overhead
        return base * 0.6;
      }
      if (typeName === "CypherQueryOperator") {
        // Cypher queries are compiled and optimized
        return base * 0.5;
      }
    }

    // Vector operators benefit from vector indexes
    if (signal instanceof VectorSimilarityOperator) {
      return base * 0.8; // Vector search is relatively fast with indexes
    }

    // Term operators are the cheapest (inverted index lookup)
    if (signal instanceof TermOperator) {
      return base * 0.3;
    }

    return base;
  }

  // -----------------------------------------------------------------------
  // Rule 10: Apply index scan substitution
  // -----------------------------------------------------------------------

  private _applyIndexScan(op: Operator): Operator {
    if (this._indexManager === null || this._tableName === null) return op;

    if (op instanceof FilterOperator && op.source === null) {
      const idx = this._indexManager.findCoveringIndex(
        this._tableName,
        op.field,
        op.predicate,
      );
      if (idx !== null) {
        const scanCost = idx.scanCost(op.predicate);
        const fullScanCost = this.stats.totalDocs;
        if (scanCost < fullScanCost) {
          return new IndexScanOperator(idx, op.field, op.predicate);
        }
      }
    }

    if (op instanceof FilterOperator && op.source !== null) {
      return new FilterOperator(
        op.field,
        op.predicate,
        this._applyIndexScan(op.source),
      );
    }

    return this._recurseGeneric(op, (o) => this._applyIndexScan(o));
  }

  // -----------------------------------------------------------------------
  // Generic recursion helpers
  // -----------------------------------------------------------------------

  private _recurseGeneric(op: Operator, fn: (o: Operator) => Operator): Operator {
    if (op instanceof IntersectOperator) {
      return new IntersectOperator(op.operands.map(fn));
    }
    if (op instanceof UnionOperator) {
      return new UnionOperator(op.operands.map(fn));
    }
    if (op instanceof ComplementOperator) {
      return new ComplementOperator(fn(op.operand));
    }
    if (op instanceof FilterOperator && op.source !== null) {
      return new FilterOperator(op.field, op.predicate, fn(op.source));
    }
    if (op instanceof ComposedOperator) {
      return new ComposedOperator(
        (op as unknown as { operators: Operator[] }).operators.map(fn),
      );
    }
    return op;
  }

  private _recurseChildren(op: Operator): Operator {
    if (op instanceof IntersectOperator) {
      return new IntersectOperator(op.operands.map((o) => this.optimize(o)));
    }
    if (op instanceof UnionOperator) {
      return new UnionOperator(op.operands.map((o) => this.optimize(o)));
    }
    if (op instanceof ComplementOperator) {
      return new ComplementOperator(this.optimize(op.operand));
    }
    if (op instanceof FilterOperator && op.source !== null) {
      return new FilterOperator(op.field, op.predicate, this.optimize(op.source));
    }
    if (op instanceof ComposedOperator) {
      return new ComposedOperator(
        (op as unknown as { operators: Operator[] }).operators.map((o: Operator) =>
          this.optimize(o),
        ),
      );
    }
    if (op instanceof ScoreOperator) {
      return new ScoreOperator(
        op.scorer,
        this.optimize(op.source),
        op.queryTerms,
        op.field,
      );
    }
    if (op instanceof SparseThresholdOperator) {
      return new SparseThresholdOperator(this.optimize(op.source), op.threshold);
    }

    // Fusion operator recursion
    if (op instanceof LogOddsFusionOperator) {
      return new LogOddsFusionOperator(
        op.signals.map((s) => this.optimize(s)),
        op.alpha,
        undefined,
        op.gating,
      );
    }
    if (op instanceof ProbBoolFusionOperator) {
      return new ProbBoolFusionOperator(
        op.signals.map((s) => this.optimize(s)),
        op.mode,
      );
    }
    if (op instanceof ProbNotOperator) {
      return new ProbNotOperator(this.optimize(op.signal), op.defaultProb);
    }
    if (op instanceof AttentionFusionOperator) {
      return new AttentionFusionOperator(
        op.signals.map((s) => this.optimize(s)),
        op.attention,
        op.queryFeatures,
      );
    }
    if (op instanceof LearnedFusionOperator) {
      return new LearnedFusionOperator(
        op.signals.map((s) => this.optimize(s)),
        op.learned,
      );
    }

    return op;
  }

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  private static _filterAppliesTo(filterOp: Operator, target: Operator): boolean {
    const field = (filterOp as FilterOperator).field;

    if (target instanceof TermOperator) {
      return target.field === field || target.field === null;
    }
    if (target instanceof FilterOperator) {
      return target.field === field;
    }
    if (target instanceof IntersectOperator) {
      return target.operands.some((child) =>
        QueryOptimizer._filterAppliesTo(filterOp, child),
      );
    }
    return false;
  }
}
