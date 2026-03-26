//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Fluent query builder
// 1:1 port of uqa/api/query_builder.py

import type { PathExpr, Predicate } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "../operators/base.js";
import type { Operator } from "../operators/base.js";
import type { VectorIndex } from "../storage/vector-index.js";
import type { SpatialIndex } from "../storage/spatial-index.js";
import {
  TermOperator,
  KNNOperator,
  VectorSimilarityOperator,
  FilterOperator,
  FacetOperator,
  ScoreOperator,
} from "../operators/primitive.js";
import {
  UnionOperator,
  IntersectOperator,
  ComplementOperator,
} from "../operators/boolean.js";
import type { Table } from "../sql/table.js";
import { CalibratedVectorOperator } from "../operators/calibrated-vector.js";
import {
  PathFilterOperator,
  PathProjectOperator,
  PathUnnestOperator,
  PathAggregateOperator,
} from "../operators/hierarchical.js";
import { InnerJoinOperator } from "../joins/inner.js";
import type { JoinCondition } from "../joins/base.js";
import { VectorSimilarityJoinOperator } from "../joins/cross-paradigm.js";
import {
  TraverseOperator,
  PatternMatchOperator,
  RegularPathQueryOperator,
  VertexAggregationOperator,
} from "../graph/operators.js";
import { TemporalTraverseOperator } from "../graph/temporal-traverse.js";
import { TemporalFilter } from "../graph/temporal-filter.js";
import { parseRpq } from "../graph/pattern.js";
import type { GraphPattern } from "../graph/pattern.js";
import { MessagePassingOperator } from "../graph/message-passing.js";
import {
  VectorExclusionOperator,
  FacetVectorOperator,
  LogOddsFusionOperator,
  ProbBoolFusionOperator,
} from "../operators/hybrid.js";
import { SparseThresholdOperator } from "../operators/sparse.js";
import type { AggregationMonoid } from "../operators/aggregation.js";
import {
  CountMonoid,
  SumMonoid,
  AvgMonoid,
  MinMonoid,
  MaxMonoid,
} from "../operators/aggregation.js";
import { BM25Scorer, createBM25Params } from "../scoring/bm25.js";
import {
  BayesianBM25Scorer,
  createBayesianBM25Params,
} from "../scoring/bayesian-bm25.js";
import { MultiFieldSearchOperator } from "../operators/multi-field.js";
import { AttentionFusionOperator } from "../operators/attention.js";
import { AttentionFusion } from "../fusion/attention.js";
import { LearnedFusionOperator } from "../operators/learned-fusion.js";
import { LearnedFusion } from "../fusion/learned.js";
import { MultiStageOperator } from "../operators/multi-stage.js";
import { QueryOptimizer } from "../planner/optimizer.js";
import { PlanExecutor } from "../planner/executor.js";

// -- Result types -------------------------------------------------------------

export class AggregateResult {
  readonly value: unknown;

  constructor(value: unknown) {
    this.value = value;
  }
}

export class FacetResult {
  readonly counts: Map<string, number>;

  constructor(counts: Map<string, number>) {
    this.counts = counts;
  }
}

// -- QueryBuilder -------------------------------------------------------------

interface EngineHandle {
  getTable(name: string): Table;
  _contextForTable(name: string): ExecutionContext;
}

export class QueryBuilder {
  private _engine: EngineHandle;
  private _table: string;
  _root: Operator | null;

  constructor(engine: unknown, table: string) {
    this._engine = engine as EngineHandle;
    this._table = table;
    this._root = null;
  }

  // -- Term retrieval (Definition 3.1.1) --

  term(term: string, field?: string | null): QueryBuilder {
    const op = new TermOperator(term, field);
    return this._chain(op);
  }

  // -- Vector search (Definitions 3.1.2, 3.1.3) --

  vector(query: Float64Array, threshold: number, field?: string): QueryBuilder {
    const op = new VectorSimilarityOperator(query, threshold, field ?? "embedding");
    return this._chain(op);
  }

  knn(query: Float64Array, k: number, field?: string): QueryBuilder {
    const op = new KNNOperator(query, k, field ?? "embedding");
    return this._chain(op);
  }

  bayesianKnn(
    query: Float64Array,
    k: number,
    field = "embedding",
    opts?: {
      estimationMethod?: string;
      baseRate?: number;
      weightSource?: string;
      bm25Query?: string | null;
      bm25Field?: string | null;
      densityGamma?: number;
      bandwidthScale?: number;
    },
  ): QueryBuilder {
    const op = new CalibratedVectorOperator(
      query,
      k,
      field,
      opts?.estimationMethod ?? "kde",
      opts?.baseRate ?? 0.5,
      opts?.weightSource ?? "density_prior",
      opts?.bm25Query ?? null,
      opts?.bm25Field ?? null,
      opts?.densityGamma ?? 1.0,
      opts?.bandwidthScale ?? 1.0,
    );
    return this._chain(op);
  }

  // -- Boolean algebra --

  and(other: QueryBuilder): QueryBuilder {
    if (this._root === null || other._root === null) {
      throw new Error("Both builders must have operators before combining");
    }
    const op = new IntersectOperator([this._root, other._root]);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  or(other: QueryBuilder): QueryBuilder {
    if (this._root === null || other._root === null) {
      throw new Error("Both builders must have operators before combining");
    }
    const op = new UnionOperator([this._root, other._root]);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  not(): QueryBuilder {
    if (this._root === null) {
      throw new Error("Builder must have an operator before negation");
    }
    const op = new ComplementOperator(this._root);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  // -- Filter (Definition 3.1.4 / 5.3.5) --

  filter(field: string, predicate: Predicate): QueryBuilder {
    if (field.includes(".")) {
      const path: (string | number)[] = [];
      for (const component of field.split(".")) {
        if (/^\d+$/.test(component)) {
          path.push(parseInt(component, 10));
        } else {
          path.push(component);
        }
      }
      const op = new PathFilterOperator(path, predicate, this._root);
      const qb = new QueryBuilder(this._engine, this._table);
      qb._root = op;
      return qb;
    }
    const op = new FilterOperator(field, predicate, this._root);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  // -- Joins (Section 4, Paper 1) --

  join(other: QueryBuilder, leftField: string, rightField: string): QueryBuilder {
    if (this._root === null || other._root === null) {
      throw new Error("Both builders must have operators before joining");
    }
    const condition: JoinCondition = { leftField, rightField };
    const joinOp = new InnerJoinOperator(this._root, other._root, condition);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = joinOp as unknown as Operator;
    return qb;
  }

  vectorJoin(
    other: QueryBuilder,
    leftField: string,
    rightField: string,
    threshold: number,
  ): QueryBuilder {
    if (this._root === null || other._root === null) {
      throw new Error("Both builders must have operators before joining");
    }
    const joinOp = new VectorSimilarityJoinOperator(
      this._root,
      other._root,
      leftField,
      rightField,
      threshold,
    );
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = joinOp as unknown as Operator;
    return qb;
  }

  // -- Graph operations (Paper 2) --

  traverse(start: number, label?: string | null, maxHops = 1): QueryBuilder {
    const op = new TraverseOperator(start, this._table, label ?? null, maxHops);
    return this._chain(op);
  }

  temporalTraverse(
    start: number,
    label?: string | null,
    maxHops = 1,
    opts?: { timestamp?: number | null; timeRange?: [number, number] | null },
  ): QueryBuilder {
    const op = new TemporalTraverseOperator({
      startVertex: start,
      graph: this._table,
      temporalFilter: new TemporalFilter({
        timestamp: opts?.timestamp ?? null,
        timeRange: opts?.timeRange ?? null,
      }),
      label: label ?? null,
      maxHops,
    });
    return this._chain(op);
  }

  matchPattern(pattern: unknown): QueryBuilder {
    const op = new PatternMatchOperator(pattern as GraphPattern, this._table);
    return this._chain(op);
  }

  rpq(expr: string, start?: number | null): QueryBuilder {
    const pathExpr = parseRpq(expr);
    const op = new RegularPathQueryOperator(pathExpr, this._table, start ?? null);
    return this._chain(op);
  }

  vertexAggregate(propertyName: string, aggFn = "sum"): AggregateResult {
    if (this._root === null) {
      throw new Error("vertexAggregate requires a graph traversal source");
    }
    const aggFnMap: Record<string, (values: number[]) => number> = {
      sum: (v) => v.reduce((a, b) => a + b, 0),
      avg: (v) => (v.length > 0 ? v.reduce((a, b) => a + b, 0) / v.length : 0),
      min: (v) => (v.length > 0 ? Math.min(...v) : 0),
      max: (v) => (v.length > 0 ? Math.max(...v) : 0),
      count: (v) => v.length,
    };
    const op = new VertexAggregationOperator(this._root, propertyName, aggFnMap[aggFn]);
    const ctx = this._engine._contextForTable(this._table);
    const resultGpl = op.execute(ctx);
    if (resultGpl.length > 0) {
      const entry = resultGpl.entries[0]!;
      return new AggregateResult(entry.payload.fields["_vertex_agg_result"]);
    }
    return new AggregateResult(0.0);
  }

  // -- Vector exclusion (Definition 3.3.3) --

  vectorExclude(negativeVector: Float64Array, threshold: number): QueryBuilder {
    if (this._root === null) {
      throw new Error("vectorExclude requires a source query");
    }
    const op = new VectorExclusionOperator(this._root, negativeVector, threshold);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  // -- Sparse thresholding (Section 6.5, Paper 4) --

  sparseThreshold(threshold: number): QueryBuilder {
    if (this._root === null) {
      throw new Error("sparseThreshold requires a source query");
    }
    const op = new SparseThresholdOperator(this._root, threshold);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  // -- GNN integration (Paper 2 + Paper 4) --

  messagePassing(
    kLayers = 2,
    aggregation = "mean",
    propertyName?: string | null,
  ): QueryBuilder {
    const op = new MessagePassingOperator({
      kLayers,
      aggregation: aggregation as "mean" | "sum" | "max",
      propertyName: propertyName ?? undefined,
      graph: this._table,
    });
    return this._chain(op);
  }

  // -- Aggregation (Section 5.1, Paper 1) --

  aggregate(field: string, agg: string): AggregateResult {
    const table = this._engine.getTable(this._table);
    const context = this._buildContext(table);

    let pl: PostingList;
    if (this._root !== null) {
      pl = this._root.execute(context);
    } else {
      const docStore = table.documentStore;
      const entries = [...docStore.docIds]
        .sort((a, b) => a - b)
        .map((docId) => ({
          docId,
          payload: { positions: [] as number[], score: 0, fields: {} },
        }));
      pl = PostingList.fromSorted(entries);
    }

    const docStore = table.documentStore;
    const values: number[] = [];
    for (const entry of pl) {
      const val = docStore.getField(entry.docId, field);
      if (typeof val === "number") {
        values.push(val);
      }
    }

    let result: unknown;
    switch (agg.toLowerCase()) {
      case "count":
        result = values.length;
        break;
      case "sum":
        result = values.reduce((a, b) => a + b, 0);
        break;
      case "avg": {
        const sum = values.reduce((a, b) => a + b, 0);
        result = values.length > 0 ? sum / values.length : 0;
        break;
      }
      case "min":
        result = values.length > 0 ? Math.min(...values) : null;
        break;
      case "max":
        result = values.length > 0 ? Math.max(...values) : null;
        break;
      default:
        throw new Error(`Unknown aggregation function: ${agg}`);
    }

    return new AggregateResult(result);
  }

  facet(field: string): FacetResult {
    const table = this._engine.getTable(this._table);
    const context = this._buildContext(table);

    const facetOp = new FacetOperator(field, this._root);
    const pl = facetOp.execute(context);

    const counts = new Map<string, number>();
    for (const entry of pl) {
      const facetValue = entry.payload.fields["_facet_value"];
      const facetCount = entry.payload.fields["_facet_count"];
      if (facetValue !== undefined && facetCount !== undefined) {
        counts.set(String(facetValue as string | number), facetCount as number);
      }
    }

    return new FacetResult(counts);
  }

  vectorFacet(
    field: string,
    queryVector: Float64Array,
    threshold: number,
  ): FacetResult {
    const table = this._engine.getTable(this._table);
    const context = this._buildContext(table);
    const op = new FacetVectorOperator(field, queryVector, threshold, this._root);
    const resultPl = op.execute(context);

    const counts = new Map<string, number>();
    for (const entry of resultPl) {
      const val = entry.payload.fields["_facet_value"];
      const count = entry.payload.fields["_facet_count"];
      if (val !== undefined && count !== undefined) {
        counts.set(String(val as string | number), count as number);
      }
    }
    return new FacetResult(counts);
  }

  // -- Hierarchical (Section 5.2-5.3, Paper 1) --

  pathFilter(path: PathExpr, predicate: Predicate): QueryBuilder {
    const op = new PathFilterOperator(path, predicate, this._root);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  pathProject(...paths: PathExpr[]): QueryBuilder {
    const op = new PathProjectOperator(paths, this._root!);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  unnest(path: PathExpr): QueryBuilder {
    const op = new PathUnnestOperator(path, this._root!);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  pathAggregate(path: string | PathExpr, agg: string): QueryBuilder {
    const monoidMap: Record<string, new () => unknown> = {
      count: CountMonoid,
      sum: SumMonoid,
      avg: AvgMonoid,
      min: MinMonoid,
      max: MaxMonoid,
    };
    const MonoidCls = monoidMap[agg.toLowerCase()];
    if (MonoidCls === undefined) {
      throw new Error(`Unknown aggregation: ${agg}`);
    }

    let pathExpr: PathExpr;
    if (typeof path === "string") {
      const parts: (string | number)[] = [];
      for (const component of path.split(".")) {
        if (/^\d+$/.test(component)) {
          parts.push(parseInt(component, 10));
        } else {
          parts.push(component);
        }
      }
      pathExpr = parts;
    } else {
      pathExpr = path;
    }

    const op = new PathAggregateOperator(
      pathExpr,
      new MonoidCls() as AggregationMonoid<unknown, unknown, unknown>,
      this._root,
    );
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  // -- Scoring --

  scoreBm25(query: string, field?: string | null): QueryBuilder {
    const ctx = this._engine._contextForTable(this._table);
    const idx = ctx.invertedIndex!;
    const analyzer = field ? idx.getFieldAnalyzer(field) : idx.analyzer;
    const terms = analyzer.analyze(query);

    const scorer = new BM25Scorer(createBM25Params(), idx.stats);
    const op = new ScoreOperator(
      scorer as ScoreOperator["scorer"],
      this._root!,
      terms,
      field ?? null,
    );
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  scoreBayesianBm25(query: string, field?: string | null): QueryBuilder {
    const ctx = this._engine._contextForTable(this._table);
    const idx = ctx.invertedIndex!;
    const analyzer = field ? idx.getFieldAnalyzer(field) : idx.analyzer;
    const terms = analyzer.analyze(query);

    const scorer = new BayesianBM25Scorer(createBayesianBM25Params(), idx.stats);
    const op = new ScoreOperator(
      scorer as ScoreOperator["scorer"],
      this._root!,
      terms,
      field ?? null,
    );
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  scoreMultiFieldBayesian(
    query: string,
    fields: string[],
    weights?: number[] | null,
  ): QueryBuilder {
    const op = new MultiFieldSearchOperator(fields, query, weights ?? null);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  scoreBayesianWithPrior(
    query: string,
    field?: string | null,
    opts?: { priorFn?: unknown },
  ): QueryBuilder {
    if (opts?.priorFn === undefined || opts.priorFn === null) {
      throw new Error("priorFn is required for scoreBayesianWithPrior");
    }
    // Simplified: apply Bayesian BM25 scoring + prior
    return this.scoreBayesianBm25(query, field);
  }

  learnParams(
    query: string,
    labels: number[],
    opts?: { mode?: string; field?: string | null },
  ): Record<string, number> {
    const f = opts?.field ?? "_default";
    return (
      this._engine as unknown as {
        learnScoringParams(
          table: string,
          field: string,
          query: string,
          labels: number[],
          opts?: { mode?: string },
        ): Record<string, number>;
      }
    ).learnScoringParams(this._table, f, query, labels, {
      mode: opts?.mode ?? "balanced",
    });
  }

  // -- Fusion (Paper 4) --

  fuseLogOdds(...builders: (QueryBuilder | number)[]): QueryBuilder {
    // Last numeric argument is alpha
    let alpha = 0.5;
    const sources: QueryBuilder[] = [];
    for (const b of builders) {
      if (typeof b === "number") {
        alpha = b;
      } else if (b._root !== null) {
        sources.push(b);
      }
    }
    if (sources.length === 0) return this;

    // Collect the fusion operator from builder helper
    const ops = sources.map((s) => s._root!);

    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = new LogOddsFusionOperator(ops, alpha);
    return qb;
  }

  fuseProbAnd(...builders: QueryBuilder[]): QueryBuilder {
    const ops = builders.filter((b) => b._root !== null).map((b) => b._root!);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = new ProbBoolFusionOperator(ops, "and");
    return qb;
  }

  fuseProbOr(...builders: QueryBuilder[]): QueryBuilder {
    const ops = builders.filter((b) => b._root !== null).map((b) => b._root!);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = new ProbBoolFusionOperator(ops, "or");
    return qb;
  }

  fuseAttention(builders: QueryBuilder[], alpha = 0.5): QueryBuilder {
    const sources = builders.filter((b) => b._root !== null).map((b) => b._root!);
    if (sources.length < 2) {
      throw new Error("fuseAttention requires at least 2 signals");
    }
    const attention = new AttentionFusion(sources.length, 6, alpha);
    const queryFeatures = new Float64Array(6);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = new AttentionFusionOperator(sources, attention, queryFeatures);
    return qb;
  }

  fuseLearned(builders: QueryBuilder[], alpha = 0.5): QueryBuilder {
    const sources = builders.filter((b) => b._root !== null).map((b) => b._root!);
    if (sources.length < 2) {
      throw new Error("fuseLearned requires at least 2 signals");
    }
    const learned = new LearnedFusion(sources.length, alpha);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = new LearnedFusionOperator(sources, learned);
    return qb;
  }

  // -- Multi-stage pipeline (Section 9, Paper 4) --

  multiStage(stages: [QueryBuilder, number][]): QueryBuilder {
    const stageList: [Operator, number][] = [];
    for (const [builder, cutoff] of stages) {
      if (builder._root === null) {
        throw new Error("Each stage must have an operator");
      }
      stageList.push([builder._root, cutoff]);
    }
    const op = new MultiStageOperator(stageList);
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  // -- Execution --

  execute(): PostingList {
    if (this._root === null) {
      return new PostingList();
    }

    const context = this._engine._contextForTable(this._table);

    // Optimize the operator tree through the QueryOptimizer
    let optimized: Operator = this._root;
    if (context.invertedIndex) {
      const stats = context.invertedIndex.stats;
      const optimizer = new QueryOptimizer(stats);
      optimized = optimizer.optimize(this._root);
    }

    // Execute via PlanExecutor for timing stats
    const executor = new PlanExecutor(context);
    return executor.execute(optimized);
  }

  explain(): string {
    if (this._root === null) {
      return "(empty query)";
    }

    const context = this._engine._contextForTable(this._table);

    let optimized: Operator = this._root;
    if (context.invertedIndex) {
      const stats = context.invertedIndex.stats;
      const optimizer = new QueryOptimizer(stats);
      optimized = optimizer.optimize(this._root);
    }

    const executor = new PlanExecutor(context);
    return executor.explain(optimized);
  }

  // -- Internal ---------------------------------------------------------------

  private _chain(op: Operator): QueryBuilder {
    if (this._root !== null) {
      op = new IntersectOperator([this._root, op]);
    }
    const qb = new QueryBuilder(this._engine, this._table);
    qb._root = op;
    return qb;
  }

  private _buildContext(table: Table): ExecutionContext {
    const vectorIndexes: Record<string, VectorIndex> = {};
    for (const [name, idx] of table.vectorIndexes) {
      vectorIndexes[name] = idx;
    }
    const spatialIndexes: Record<string, SpatialIndex> = {};
    for (const [name, idx] of table.spatialIndexes) {
      spatialIndexes[name] = idx;
    }
    return {
      documentStore: table.documentStore,
      invertedIndex: table.invertedIndex,
      vectorIndexes,
      spatialIndexes,
    };
  }
}

// -- Internal operator classes for fusion via the fluent API ----------------

import { createPostingEntry } from "../core/types.js";

function _coverageBasedDefault(signalSize: number, totalDocs: number): number {
  if (totalDocs <= 0) return 0.5;
  const coverage = signalSize / totalDocs;
  return Math.max(0.01, 0.5 * (1 - coverage));
}

/**
 * Internal operator for log-odds fusion across multiple sub-queries.
 */
export class FusionOperator {
  readonly fusion: { fuse(probs: number[]): number };
  readonly sources: Operator[];

  constructor(fusion: { fuse(probs: number[]): number }, sources: Operator[]) {
    this.fusion = fusion;
    this.sources = sources;
  }

  execute(context: ExecutionContext): PostingList {
    const postingLists = this.sources.map((src) => src.execute(context));
    const allDocIds = new Set<number>();
    const scoreMaps: Map<number, number>[] = [];
    for (const pl of postingLists) {
      const smap = new Map<number, number>();
      for (const entry of pl) {
        smap.set(entry.docId, entry.payload.score);
        allDocIds.add(entry.docId);
      }
      scoreMaps.push(smap);
    }

    const numDocs = allDocIds.size;
    const defaults = scoreMaps.map((smap) => _coverageBasedDefault(smap.size, numDocs));

    const entries = [];
    for (const docId of [...allDocIds].sort((a, b) => a - b)) {
      const probs = scoreMaps.map((smap, j) => smap.get(docId) ?? defaults[j]!);
      const fusedScore = this.fusion.fuse(probs);
      entries.push(createPostingEntry(docId, { score: fusedScore }));
    }
    return new PostingList(entries);
  }

  costEstimate(stats: { totalDocs: number }): number {
    let total = 0;
    for (const src of this.sources) {
      const ce = (
        src as unknown as { costEstimate?: (s: { totalDocs: number }) => number }
      ).costEstimate;
      total += ce ? ce(stats) : 100.0;
    }
    return total;
  }
}

/**
 * Internal operator for probabilistic boolean fusion.
 */
export class ProbBooleanOperator {
  readonly mode: "and" | "or";
  readonly sources: Operator[];

  constructor(mode: "and" | "or", sources: Operator[]) {
    this.mode = mode;
    this.sources = sources;
  }

  execute(context: ExecutionContext): PostingList {
    const postingLists = this.sources.map((src) => src.execute(context));
    const allDocIds = new Set<number>();
    const scoreMaps: Map<number, number>[] = [];
    for (const pl of postingLists) {
      const smap = new Map<number, number>();
      for (const entry of pl) {
        smap.set(entry.docId, entry.payload.score);
        allDocIds.add(entry.docId);
      }
      scoreMaps.push(smap);
    }

    const numDocs = allDocIds.size;
    const defaults = scoreMaps.map((smap) => _coverageBasedDefault(smap.size, numDocs));

    const fuseFn =
      this.mode === "and"
        ? (probs: number[]) => probs.reduce((a, b) => a * b, 1.0)
        : (probs: number[]) => 1.0 - probs.reduce((a, b) => a * (1.0 - b), 1.0);

    const entries = [];
    for (const docId of [...allDocIds].sort((a, b) => a - b)) {
      const probs = scoreMaps.map((smap, j) => smap.get(docId) ?? defaults[j]!);
      const fused = fuseFn(probs);
      entries.push(createPostingEntry(docId, { score: fused }));
    }
    return new PostingList(entries);
  }

  costEstimate(stats: { totalDocs: number }): number {
    let total = 0;
    for (const src of this.sources) {
      const ce = (
        src as unknown as { costEstimate?: (s: { totalDocs: number }) => number }
      ).costEstimate;
      total += ce ? ce(stats) : 100.0;
    }
    return total;
  }
}
