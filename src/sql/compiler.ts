//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- SQL compiler
// 1:1 port of uqa/sql/compiler.py
//
// Compiles SQL statements (parsed via libpg-query WASM) and executes them
// against the in-memory Table storage.
//
// Supported statements:
//   DDL:
//     CREATE [TEMPORARY | TEMP] TABLE name (col type [constraints], ...)
//     CREATE [TEMP] TABLE name AS SELECT ...
//     DROP TABLE [IF EXISTS] name
//     DROP VIEW [IF EXISTS] name
//     DROP INDEX [IF EXISTS] name
//     CREATE VIEW name AS SELECT ...
//     CREATE SEQUENCE name [START n] [INCREMENT n]
//     ALTER SEQUENCE name [RESTART [WITH n]] [INCREMENT [BY] n] [START [WITH] n]
//     CREATE INDEX name ON table [USING method] (col, ...)
//     ALTER TABLE name ADD/DROP/ALTER COLUMN ...
//     RENAME TABLE / COLUMN
//     TRUNCATE name
//   Constraints:
//     PRIMARY KEY, NOT NULL, DEFAULT val, UNIQUE, CHECK (expr)
//     REFERENCES parent(col)           -- column-level FOREIGN KEY
//     FOREIGN KEY (col) REFERENCES parent(col) -- table-level FOREIGN KEY
//   DML:
//     INSERT INTO name (col, ...) VALUES (val, ...), ...
//     INSERT INTO name SELECT ...
//     INSERT INTO name ... ON CONFLICT (col) DO NOTHING / DO UPDATE SET ...
//     UPDATE name SET col = expr, ... [WHERE ...]
//     UPDATE name SET col = expr FROM other_table WHERE ...
//     DELETE FROM name [WHERE ...]
//     DELETE FROM name USING other_table WHERE ...
//   DQL:
//     WITH name AS (SELECT ...) [, ...]     -- common table expressions (CTE)
//     WITH RECURSIVE name AS (base UNION [ALL] recursive)
//     SELECT [DISTINCT] [* | col, ... | expr, ... | aggregates |
//            window_func() OVER ([PARTITION BY ...] [ORDER BY ...])]
//       FROM table
//       [JOIN table ON ...]
//       [LATERAL (subquery)]
//       [WHERE comparisons / boolean / IS [NOT] NULL /
//              LIKE / NOT LIKE / ILIKE / NOT ILIKE /
//              IN (list) / IN (SELECT ...) / EXISTS (SELECT ...) /
//              BETWEEN / NOT BETWEEN /
//              text_match() / knn_match() / bayesian_match() / ...]
//       [GROUP BY col [HAVING ...]]
//       [ORDER BY col [ASC|DESC] [NULLS FIRST|LAST]]
//       [LIMIT n [OFFSET m]]
//     UNION [ALL] / INTERSECT [ALL] / EXCEPT [ALL]
//     VALUES (val, ...), ...
//   Transaction:
//     BEGIN / COMMIT / ROLLBACK
//     SAVEPOINT name / RELEASE SAVEPOINT name / ROLLBACK TO SAVEPOINT name
//   Prepared Statements:
//     PREPARE name [(type, ...)] AS query
//     EXECUTE name [(val, ...)]
//     DEALLOCATE name | ALL
//   Utility:
//     EXPLAIN SELECT ...
//     ANALYZE [table]
//     VACUUM
//   RETURNING clause for INSERT/UPDATE/DELETE

import { parse as pgParse } from "libpg-query";
import { ExprEvaluator } from "./expr-evaluator.js";
import { Table, createColumnDef, resolveType } from "./table.js";
import type { ColumnDef, ForeignKeyDef } from "./table.js";

// -- UQA operator and type imports for WHERE-clause compilation ----------------

import { PostingList } from "../core/posting-list.js";
import {
  createPostingEntry,
  Equals,
  NotEquals,
  GreaterThan,
  GreaterThanOrEqual,
  LessThan,
  LessThanOrEqual,
  InSet,
  Between,
  Like,
  NotLike,
  ILike,
  NotILike,
  IsNull,
  IsNotNull,
  createVertex,
  createEdge,
} from "../core/types.js";
import type { PostingEntry as PostingEntryType, Predicate } from "../core/types.js";
import { listAnalyzers as listAnalyzersFn, getAnalyzer as getAnalyzerFn } from "../analysis/analyzer.js";
import { CypherCompiler } from "../graph/cypher/compiler.js";
import type { GraphStore } from "../storage/abc/graph-store.js";
import type { IndexStats } from "../core/types.js";
import { Operator } from "../operators/base.js";
import type { ExecutionContext } from "../operators/base.js";
import {
  FilterOperator,
  TermOperator,
  KNNOperator,
  ScoreOperator,
  SpatialWithinOperator,
} from "../operators/primitive.js";
import {
  IntersectOperator,
  UnionOperator,
  ComplementOperator,
} from "../operators/boolean.js";
import {
  LogOddsFusionOperator,
  ProbBoolFusionOperator,
  VectorExclusionOperator,
  ProbNotOperator,
} from "../operators/hybrid.js";
import { SparseThresholdOperator } from "../operators/sparse.js";
import { MultiStageOperator } from "../operators/multi-stage.js";
import { MultiFieldSearchOperator } from "../operators/multi-field.js";
import { ProgressiveFusionOperator } from "../operators/progressive-fusion.js";
import { DeepFusionOperator } from "../operators/deep-fusion.js";
import type { FusionLayer } from "../operators/deep-fusion.js";
import { PathFilterOperator } from "../operators/hierarchical.js";
import { AttentionFusionOperator } from "../operators/attention.js";
import { LearnedFusionOperator } from "../operators/learned-fusion.js";
import { CalibratedVectorOperator } from "../operators/calibrated-vector.js";
import { TraverseOperator } from "../graph/operators.js";
import { TemporalTraverseOperator } from "../graph/temporal-traverse.js";
import { TemporalFilter } from "../graph/temporal-filter.js";
import {
  PageRankOperator,
  HITSOperator,
  BetweennessCentralityOperator,
} from "../graph/centrality.js";
import { MessagePassingOperator } from "../graph/message-passing.js";
import { GraphEmbeddingOperator } from "../graph/graph-embedding.js";
import { parseRpq } from "../graph/pattern.js";
import {
  BayesianBM25Scorer,
  createBayesianBM25Params,
} from "../scoring/bayesian-bm25.js";
import { BM25Scorer, createBM25Params } from "../scoring/bm25.js";
import {
  recencyPrior,
  authorityPrior,
  ExternalPriorScorer,
} from "../scoring/external-prior.js";
import { WeightedPathQueryOperator } from "../graph/operators.js";
import { generateKernels as _genKernels } from "../operators/deep-learn.js";
import { AttentionFusion, MultiHeadAttentionFusion } from "../fusion/attention.js";
import { LearnedFusion } from "../fusion/learned.js";
import { QueryFeatureExtractor } from "../fusion/query-features.js";
import { QueryOptimizer } from "../planner/optimizer.js";
import type { ColumnStats } from "./table.js";
import { PlanExecutor } from "../planner/executor.js";

// -- FDW handler imports --------------------------------------------------------

import type { FDWHandler } from "../fdw/handler.js";
import type { ForeignServer, ForeignTable } from "../fdw/foreign-table.js";
import { DuckDBFDWHandler } from "../fdw/duckdb-handler.js";
import { ArrowFDWHandler } from "../fdw/arrow-handler.js";

// -- SQLResult ------------------------------------------------------------------

export interface SQLResult {
  readonly columns: string[];
  readonly rows: Record<string, unknown>[];
}

// -- Aggregate function names ---------------------------------------------------

const AGG_FUNC_NAMES = new Set<string>([
  "count",
  "sum",
  "avg",
  "min",
  "max",
  "string_agg",
  "array_agg",
  "bool_and",
  "bool_or",
  "stddev",
  "stddev_pop",
  "stddev_samp",
  "variance",
  "var_pop",
  "var_samp",
  "percentile_cont",
  "percentile_disc",
  "mode",
  "json_object_agg",
  "jsonb_object_agg",
  "corr",
  "covar_pop",
  "covar_samp",
  "regr_count",
  "regr_avgx",
  "regr_avgy",
  "regr_sxx",
  "regr_syy",
  "regr_sxy",
  "regr_slope",
  "regr_intercept",
  "regr_r2",
]);

// -- UQA-specific WHERE functions that produce posting-list-like results --------

const UQA_WHERE_FUNCTIONS = new Set<string>([
  "text_match",
  "bayesian_match",
  "bayesian_match_with_prior",
  "knn_match",
  "bayesian_knn_match",
  "traverse_match",
  "temporal_traverse",
  "path_filter",
  "vector_exclude",
  "spatial_within",
  "fuse_log_odds",
  "fuse_prob_and",
  "fuse_prob_or",
  "fuse_prob_not",
  "fuse_attention",
  "fuse_multihead",
  "fuse_learned",
  "sparse_threshold",
  "multi_field_match",
  "message_passing",
  "graph_embedding",
  "staged_retrieval",
  "pagerank",
  "hits",
  "betweenness",
  "weighted_rpq",
  "progressive_fusion",
  "deep_fusion",
]);

// -- Side-effecting functions that must not be constant-folded ------------------

const NO_FOLD_FUNCS = new Set<string>([
  "random",
  "nextval",
  "currval",
  "now",
  "current_timestamp",
  "clock_timestamp",
  "statement_timestamp",
  "timeofday",
]);

// -- Helpers for AST node access ------------------------------------------------

function nodeGet(node: Record<string, unknown>, key: string): unknown {
  return node[key] ?? null;
}

function toStr(v: unknown): string {
  if (typeof v === "string") return v;
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return v.toString(10);
  if (typeof v === "boolean") return v ? "true" : "false";
  return JSON.stringify(v);
}

function nodeStr(node: Record<string, unknown>, key: string): string {
  const v = node[key];
  return v === undefined || v === null ? "" : toStr(v);
}

function asObj(value: unknown): Record<string, unknown> {
  if (value === null || value === undefined) return {};
  return value as Record<string, unknown>;
}

function asList(value: unknown): Record<string, unknown>[] {
  if (Array.isArray(value)) return value as Record<string, unknown>[];
  return [];
}

/**
 * Extract a string value from a libpg-query String/str node.
 */
function extractString(node: Record<string, unknown>): string {
  const strNode = nodeGet(node, "String") ?? nodeGet(node, "str");
  if (strNode !== null && typeof strNode === "object") {
    return nodeStr(asObj(strNode), "sval") || nodeStr(asObj(strNode), "str");
  }
  if (typeof strNode === "string") return strNode;
  // Direct sval/str
  const sv = nodeStr(node, "sval") || nodeStr(node, "str");
  if (sv) return sv;
  return "";
}

/**
 * Extract a relation name from a RangeVar node.
 */
function extractRelName(rangeVar: Record<string, unknown>): string {
  return nodeStr(rangeVar, "relname");
}

/**
 * Build a lookup key from a RangeVar, including schema if present.
 * e.g. "myschema.mytable" or just "mytable".
 */
function qualifiedName(rangeVar: Record<string, unknown>): string {
  const schema = nodeGet(rangeVar, "schemaname");
  if (schema && typeof schema === "string") {
    return `${schema}.${nodeStr(rangeVar, "relname")}`;
  }
  return nodeStr(rangeVar, "relname");
}

/**
 * Extract alias from a RangeVar or subquery node.
 */
function extractAlias(node: Record<string, unknown>): string | null {
  const alias = nodeGet(node, "alias");
  if (alias !== null && typeof alias === "object") {
    return nodeStr(asObj(alias), "aliasname") || null;
  }
  return null;
}

/**
 * Extract the schema name from a RangeVar node.
 */
function extractSchemaName(rangeVar: Record<string, unknown>): string | null {
  const s = nodeStr(rangeVar, "schemaname");
  return s || null;
}

/**
 * Extract a column name from a ColumnRef AST node (unwrapped).
 */
function extractColumnName(node: Record<string, unknown>): string {
  // Handle both wrapped {ColumnRef: {fields: ...}} and unwrapped {fields: ...}
  const colRef = asObj(nodeGet(node, "ColumnRef") ?? node);
  const fields = asList(nodeGet(colRef, "fields"));
  if (fields.length > 0) {
    const last = fields[fields.length - 1]!;
    const s = extractString(last);
    if (s) return s;
  }
  // Fallback for direct column ref
  const sv = nodeStr(colRef, "sval") || nodeStr(colRef, "str");
  if (sv) return sv;
  throw new Error("Expected ColumnRef, got " + JSON.stringify(node).slice(0, 200));
}

/**
 * Extract a qualified column name (table.column) from a ColumnRef node.
 */
function extractQualifiedColumnName(node: Record<string, unknown>): string {
  const colRef = asObj(nodeGet(node, "ColumnRef") ?? node);
  const fields = asList(nodeGet(colRef, "fields"));
  if (fields.length >= 2) {
    const parts: string[] = [];
    for (const f of fields) {
      const s = extractString(f);
      if (s) parts.push(s);
    }
    return parts.join(".");
  }
  return extractColumnName(node);
}

/**
 * Check whether a node is a ColumnRef.
 */
function isColumnRef(node: Record<string, unknown>): boolean {
  return nodeGet(node, "ColumnRef") !== null;
}

/**
 * Check whether a node is A_Star or contains A_Star in ColumnRef fields.
 */
function isAStar(node: Record<string, unknown>): boolean {
  if (nodeGet(node, "A_Star") !== null && nodeGet(node, "A_Star") !== undefined) {
    return true;
  }
  const colRef = nodeGet(node, "ColumnRef");
  if (colRef !== null && colRef !== undefined) {
    const fields = asList(nodeGet(asObj(colRef), "fields"));
    for (const f of fields) {
      if (nodeGet(f, "A_Star") !== null && nodeGet(f, "A_Star") !== undefined) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Check whether a node is a FuncCall.
 */
function isFuncCall(node: Record<string, unknown>): boolean {
  return nodeGet(node, "FuncCall") !== null;
}

/**
 * Get function name from a FuncCall node (unwrapped).
 */
function getFuncName(node: Record<string, unknown>): string {
  const fc = asObj(nodeGet(node, "FuncCall") ?? node);
  const nameList = asList(nodeGet(fc, "funcname"));
  if (nameList.length === 0) return "";
  return extractString(nameList[nameList.length - 1]!).toLowerCase();
}

/**
 * Get function args from a FuncCall node (unwrapped).
 */
function getFuncArgs(node: Record<string, unknown>): Record<string, unknown>[] {
  const fc = asObj(nodeGet(node, "FuncCall") ?? node);
  return asList(nodeGet(fc, "args"));
}

/**
 * Check if a FuncCall has agg_star set.
 */
function isAggStar(node: Record<string, unknown>): boolean {
  const fc = asObj(nodeGet(node, "FuncCall") ?? node);
  return nodeGet(fc, "agg_star") === true;
}

/**
 * Check if a FuncCall has agg_distinct set.
 */
function isAggDistinct(node: Record<string, unknown>): boolean {
  const fc = asObj(nodeGet(node, "FuncCall") ?? node);
  return nodeGet(fc, "agg_distinct") === true;
}

/**
 * Check if a FuncCall has OVER clause (window function).
 */
function hasOverClause(node: Record<string, unknown>): boolean {
  const fc = asObj(nodeGet(node, "FuncCall") ?? node);
  const over = nodeGet(fc, "over");
  return over !== null && over !== undefined;
}

/**
 * Check if node is a constant (A_Const).
 */
function isAConst(node: Record<string, unknown>): boolean {
  return nodeGet(node, "A_Const") !== null;
}

/**
 * Check if node is a ParamRef.
 */
function isParamRef(node: Record<string, unknown>): boolean {
  return nodeGet(node, "ParamRef") !== null;
}

/**
 * Check if node is an A_Expr.
 */
function isAExpr(node: Record<string, unknown>): boolean {
  return nodeGet(node, "A_Expr") !== null;
}

/**
 * Check if node is a BoolExpr.
 */
function isBoolExpr(node: Record<string, unknown>): boolean {
  return nodeGet(node, "BoolExpr") !== null;
}

/**
 * Check if node is a NullTest.
 */
function isNullTest(node: Record<string, unknown>): boolean {
  return nodeGet(node, "NullTest") !== null;
}

/**
 * Check if node is a SubLink.
 */
function isSubLink(node: Record<string, unknown>): boolean {
  return nodeGet(node, "SubLink") !== null;
}

/**
 * Check if node is a TypeCast.
 */
function isTypeCast(node: Record<string, unknown>): boolean {
  return nodeGet(node, "TypeCast") !== null;
}

/**
 * Check if node is an A_ArrayExpr.
 */
function isArrayExpr(node: Record<string, unknown>): boolean {
  return nodeGet(node, "A_ArrayExpr") !== null;
}

/**
 * Extract a constant value from an A_Const or ParamRef node.
 */
function extractConstValue(node: Record<string, unknown>, params: unknown[]): unknown {
  // Handle ParamRef ($1, $2, ...)
  const paramRef = nodeGet(node, "ParamRef");
  if (paramRef !== null && paramRef !== undefined) {
    const prObj = asObj(paramRef);
    const num = nodeGet(prObj, "number") as number;
    const idx = num - 1;
    if (idx < 0 || idx >= params.length) {
      throw new Error(`No value supplied for parameter $${String(num)}`);
    }
    return params[idx];
  }

  // Handle A_Const
  const aConst = asObj(nodeGet(node, "A_Const") ?? node);

  // Check isnull
  if (nodeGet(aConst, "isnull") === true) return null;

  const ival = nodeGet(aConst, "ival");
  if (ival !== null && ival !== undefined) {
    if (typeof ival === "number") return ival;
    if (typeof ival === "object") {
      const inner = nodeGet(asObj(ival), "ival");
      if (inner !== null && inner !== undefined) return Number(inner);
    }
    return Number(ival);
  }

  const fval = nodeGet(aConst, "fval");
  if (fval !== null && fval !== undefined) {
    return parseFloat(String(fval as string | number));
  }

  const sval = nodeGet(aConst, "sval");
  if (sval !== null && sval !== undefined) {
    if (typeof sval === "string") return sval;
    if (typeof sval === "object") {
      const inner = nodeGet(asObj(sval), "sval");
      if (inner !== null && inner !== undefined) return String(inner as string);
    }
    return String(sval as string | number);
  }

  const boolval = nodeGet(aConst, "boolval");
  if (boolval !== null && boolval !== undefined) {
    return boolval;
  }

  // Nested val object (older libpg-query format)
  const val = nodeGet(aConst, "val");
  if (val !== null && val !== undefined && typeof val === "object") {
    const vObj = asObj(val);
    const intNode = nodeGet(vObj, "Integer") ?? nodeGet(vObj, "ival");
    if (intNode !== null && intNode !== undefined) {
      if (typeof intNode === "number") return intNode;
      const innerIval = nodeGet(asObj(intNode), "ival");
      if (innerIval !== null) return Number(innerIval);
    }
    const floatNode = nodeGet(vObj, "Float") ?? nodeGet(vObj, "fval");
    if (floatNode !== null && floatNode !== undefined) {
      if (typeof floatNode === "number") return floatNode;
      const innerFval = nodeGet(asObj(floatNode), "fval");
      if (innerFval !== null) return parseFloat(String(innerFval as string | number));
      return parseFloat(String(floatNode as string | number));
    }
    const strNode = nodeGet(vObj, "String") ?? nodeGet(vObj, "sval");
    if (strNode !== null && strNode !== undefined) {
      if (typeof strNode === "string") return strNode;
      const innerSval = nodeGet(asObj(strNode), "sval");
      if (innerSval !== null) return String(innerSval as string | number);
    }
    const boolNode = nodeGet(vObj, "Boolean") ?? nodeGet(vObj, "boolval");
    if (boolNode !== null && boolNode !== undefined) {
      if (typeof boolNode === "boolean") return boolNode;
      const innerBool = nodeGet(asObj(boolNode), "boolval");
      if (innerBool !== null) return innerBool;
    }
  }

  // Direct number/string/boolean
  if (typeof node === "number") return node;
  if (typeof node === "string") return node;

  return null;
}

/**
 * Extract integer value from node.
 */
function extractIntValue(node: Record<string, unknown>, params: unknown[]): number {
  const val = extractConstValue(node, params);
  return Number(val);
}

/**
 * Extract string value from node.
 */
function extractStringValue(node: Record<string, unknown>, params: unknown[]): string {
  const val = extractConstValue(node, params);
  return String(val);
}

/**
 * Extract a vector (array of numbers) from an ARRAY literal or $N param.
 */
function extractVectorArg(
  node: Record<string, unknown>,
  params: unknown[],
): Float64Array {
  const arrayExpr = nodeGet(node, "A_ArrayExpr");
  if (arrayExpr !== null && arrayExpr !== undefined) {
    const elements = asList(nodeGet(asObj(arrayExpr), "elements"));
    return new Float64Array(elements.map((e) => Number(extractConstValue(e, params))));
  }
  const paramRef = nodeGet(node, "ParamRef");
  if (paramRef !== null && paramRef !== undefined) {
    const prObj = asObj(paramRef);
    const num = nodeGet(prObj, "number") as number;
    const idx = num - 1;
    if (idx < 0 || idx >= params.length) {
      throw new Error(`No value supplied for parameter $${String(num)}`);
    }
    const val = params[idx];
    if (val instanceof Float64Array) return val;
    if (val instanceof Float32Array) return new Float64Array(val);
    if (Array.isArray(val)) return new Float64Array(val.map(Number));
    throw new Error("Parameter for vector must be an array");
  }
  throw new Error("Expected ARRAY literal or $N parameter for vector");
}

/**
 * Extract an insert value from a VALUES clause element.
 * Handles A_Const, A_ArrayExpr, FuncCall, TypeCast, ParamRef.
 */
function extractInsertValue(
  node: Record<string, unknown>,
  params: unknown[],
  evaluator: ExprEvaluator,
): unknown {
  // ParamRef
  if (isParamRef(node)) {
    const prObj = asObj(nodeGet(node, "ParamRef"));
    const num = nodeGet(prObj, "number") as number;
    const idx = num - 1;
    if (idx < 0 || idx >= params.length) {
      throw new Error(`Parameter $${String(num)} not provided`);
    }
    const val = params[idx];
    if (
      val !== null &&
      typeof val === "object" &&
      "tolist" in (val as Record<string, unknown>)
    ) {
      return (val as Record<string, unknown>)["tolist"];
    }
    return val;
  }
  // A_ArrayExpr
  if (isArrayExpr(node)) {
    const elements = asList(nodeGet(asObj(nodeGet(node, "A_ArrayExpr")), "elements"));
    if (elements.length === 0) return [];
    return elements.map((e) => extractConstValue(e, params));
  }
  // TypeCast
  if (isTypeCast(node)) {
    return evaluator.evaluate(node, {});
  }
  // FuncCall
  if (isFuncCall(node)) {
    return evaluator.evaluate(node, {});
  }
  // Default: use evaluator for complex expressions or extractConstValue for simple
  try {
    return evaluator.evaluate(node, {});
  } catch {
    return extractConstValue(node, params);
  }
}

// -- Max recursive CTE depth ---------------------------------------------------

const MAX_RECURSIVE_DEPTH = 1000;

// -- SQLCompiler ----------------------------------------------------------------

export class SQLCompiler {
  private _tables: Map<string, Table>;
  private _views: Map<string, Record<string, unknown>>;
  private _sequences: Map<
    string,
    { current: number; increment: number; start?: number }
  >;
  private _engine: unknown;
  private _transactionActive: boolean;
  private _prepared: Map<string, Record<string, unknown>>;
  private _params: unknown[];
  private _expandedViews: string[];
  private _shadowedTables: Map<string, Table>;
  private _inlinedCTEs: Map<string, Record<string, unknown>>;
  private _correlatedOuterRow: Record<string, unknown> | null;
  private _foreignServers: Map<string, ForeignServer>;
  private _foreignTables: Map<string, ForeignTable>;
  private _fdwHandlers: Map<string, FDWHandler>;
  private _indexes: Map<
    string,
    { name: string; tableName: string; method: string; columns: string[] }
  >;
  private _uqaFromFilter: {
    tableName: string;
    scores: Map<unknown, number>;
    pkCol: string | null;
  } | null;

  constructor(engine?: unknown) {
    // Use engine's table store if available (supports schema namespaces)
    const eng = engine as { _tables?: Map<string, Table> } | undefined;
    this._tables = eng?._tables ?? new Map();
    this._views = new Map();
    this._sequences = new Map();
    this._engine = engine ?? null;
    this._transactionActive = false;
    this._prepared = new Map();
    this._params = [];
    this._expandedViews = [];
    this._shadowedTables = new Map();
    this._inlinedCTEs = new Map();
    this._correlatedOuterRow = null;
    this._foreignServers = new Map();
    this._foreignTables = new Map();
    this._fdwHandlers = new Map();
    this._indexes = new Map();
    this._uqaFromFilter = null;
  }

  /**
   * Create a SubqueryExecutor callback that delegates to _compileSelect.
   * Supports correlated subqueries by accepting an optional outer row.
   */
  private _makeSubqueryExecutor(
    params: unknown[],
  ): (
    stmt: Record<string, unknown>,
    outerRow?: Record<string, unknown>,
  ) => { columns: string[]; rows: Record<string, unknown>[] } {
    return (stmt: Record<string, unknown>, outerRow?: Record<string, unknown>) =>
      this._compileSelect(stmt, params, outerRow);
  }

  get tables(): Map<string, Table> {
    return this._tables;
  }

  get views(): Map<string, Record<string, unknown>> {
    return this._views;
  }

  get engine(): unknown {
    return this._engine;
  }

  /**
   * Close all cached FDW handlers and release their resources.
   */
  closeFDWHandlers(): void {
    for (const handler of this._fdwHandlers.values()) {
      handler.close();
    }
    this._fdwHandlers.clear();
  }

  /**
   * Parse and execute a SQL statement. Returns SQLResult for queries,
   * null for DDL/DML statements.
   */
  async execute(sql: string, params?: unknown[]): Promise<SQLResult | null> {
    this._params = params ? [...params] : [];
    const ast = (await pgParse(sql)) as Record<string, unknown>;
    const stmts = asList(nodeGet(asObj(ast), "stmts"));

    if (stmts.length === 0) {
      return null;
    }

    let lastResult: SQLResult | null = null;
    for (const stmtWrapper of stmts) {
      const stmtNode = asObj(nodeGet(stmtWrapper, "stmt"));
      lastResult = this._dispatchStatement(stmtNode, this._params);
    }
    return lastResult;
  }

  private _dispatchStatement(
    stmtNode: Record<string, unknown>,
    params: unknown[],
  ): SQLResult | null {
    const keys = Object.keys(stmtNode);
    if (keys.length === 0) return null;

    const stmtType = keys[0]!;
    const stmt = asObj(stmtNode[stmtType]);

    switch (stmtType) {
      case "SelectStmt":
        return this._compileSelect(stmt, params);
      case "InsertStmt":
        return this._compileInsert(stmt, params);
      case "UpdateStmt":
        return this._compileUpdate(stmt, params);
      case "DeleteStmt":
        return this._compileDelete(stmt, params);
      case "CreateStmt":
        return this._compileCreateTable(stmt);
      case "CreateTableAsStmt":
        return this._compileCreateTableAs(stmt, params);
      case "DropStmt":
        return this._compileDrop(stmt);
      case "ViewStmt":
        return this._compileCreateView(stmt);
      case "TransactionStmt":
        return this._compileTransaction(stmt);
      case "CreateSeqStmt":
        return this._compileCreateSequence(stmt);
      case "AlterSeqStmt":
        return this._compileAlterSequence(stmt);
      case "IndexStmt":
        return this._compileCreateIndex(stmt);
      case "AlterTableStmt":
        return this._compileAlterTable(stmt);
      case "TruncateStmt":
        return this._compileTruncate(stmt);
      case "ExplainStmt":
        return this._compileExplain(stmt, params);
      case "RenameStmt":
        return this._compileRename(stmt);
      case "PrepareStmt":
        return this._compilePrepare(stmt);
      case "ExecuteStmt":
        return this._compileExecute(stmt, params);
      case "DeallocateStmt":
        return this._compileDeallocate(stmt);
      case "VacuumStmt":
        return this._compileAnalyze(stmt);
      case "CreateForeignServerStmt":
        return this._compileCreateForeignServer(stmt);
      case "CreateForeignTableStmt":
        return this._compileCreateForeignTable(stmt);
      case "CreateSchemaStmt":
        return this._compileCreateSchema(stmt);
      case "VariableSetStmt":
        return this._compileSet(stmt);
      case "VariableShowStmt":
        return this._compileShow(stmt);
      case "DiscardStmt":
        return this._compileDiscard();
      // Statements that are no-ops in the in-memory engine
      case "DoStmt":
      case "LockStmt":
        return null;
      default:
        throw new Error(`Unsupported statement type: ${stmtType}`);
    }
  }

  // ==================================================================
  // DDL: CREATE TABLE
  // ==================================================================

  private _compileCreateTable(stmt: Record<string, unknown>): SQLResult | null {
    const relation = asObj(nodeGet(stmt, "relation"));
    const tableName = qualifiedName(relation);
    const ifNotExists = nodeGet(stmt, "if_not_exists") === true;

    if (this._tables.has(tableName)) {
      if (ifNotExists) return null;
      throw new Error(`Table "${tableName}" already exists`);
    }

    const tableElts = asList(nodeGet(stmt, "tableElts"));
    const columns: ColumnDef[] = [];
    const foreignKeys: ForeignKeyDef[] = [];
    const checkConstraints: [string, (row: Record<string, unknown>) => boolean][] = [];

    let tablePrimaryKey: string | null = null;

    for (const elt of tableElts) {
      // Column definition
      const colDef = nodeGet(elt, "ColumnDef");
      if (colDef !== null && colDef !== undefined) {
        const colDefObj = asObj(colDef);
        const col = this._parseColumnDef(colDefObj);
        columns.push(col);
        if (col.primaryKey) {
          tablePrimaryKey = col.name;
        }

        // Extract column-level constraints (CHECK, FOREIGN KEY)
        const colConstraints = asList(nodeGet(colDefObj, "constraints"));
        for (const cc of colConstraints) {
          const ccNode = asObj(nodeGet(cc, "Constraint") ?? cc);
          const ccType = nodeGet(ccNode, "contype");

          // Column-level CHECK constraint
          if (ccType === 1 || ccType === "CONSTR_CHECK") {
            const conname =
              nodeStr(ccNode, "conname") || `check_${String(checkConstraints.length)}`;
            const rawExpr = nodeGet(ccNode, "raw_expr");
            if (rawExpr !== null && rawExpr !== undefined) {
              const evaluator = new ExprEvaluator();
              const exprNode = asObj(rawExpr);
              checkConstraints.push([
                conname,
                (row: Record<string, unknown>) => {
                  const result = evaluator.evaluate(exprNode, row);
                  return result === true;
                },
              ]);
            }
          }

          // Column-level FOREIGN KEY (REFERENCES parent(col))
          if (ccType === 8 || ccType === "CONSTR_FOREIGN") {
            const pkTable = asObj(nodeGet(ccNode, "pktable"));
            const pkCols = asList(nodeGet(ccNode, "pk_attrs"));
            if (pkCols.length > 0) {
              foreignKeys.push({
                column: col.name,
                refTable: extractRelName(pkTable),
                refColumn: extractString(pkCols[0]!),
              });
            }
          }
        }

        continue;
      }

      // Table constraint
      const constraint = nodeGet(elt, "Constraint");
      if (constraint !== null && constraint !== undefined) {
        const constraintNode = asObj(constraint);
        const contype = nodeGet(constraintNode, "contype");

        // PRIMARY KEY constraint on table level
        // CONSTR_PRIMARY = 5
        if (contype === 5 || contype === "CONSTR_PRIMARY") {
          const keysList = asList(nodeGet(constraintNode, "keys"));
          if (keysList.length > 0) {
            tablePrimaryKey = extractString(keysList[0]!);
          }
        }

        // FOREIGN KEY constraint
        // CONSTR_FOREIGN = 8
        if (contype === 8 || contype === "CONSTR_FOREIGN") {
          const fkCols = asList(nodeGet(constraintNode, "fk_attrs"));
          const pkTable = asObj(nodeGet(constraintNode, "pktable"));
          const pkCols = asList(nodeGet(constraintNode, "pk_attrs"));
          if (fkCols.length > 0 && pkCols.length > 0) {
            foreignKeys.push({
              column: extractString(fkCols[0]!),
              refTable: extractRelName(pkTable),
              refColumn: extractString(pkCols[0]!),
            });
          }
        }

        // UNIQUE constraint
        // CONSTR_UNIQUE = 4
        if (contype === 4 || contype === "CONSTR_UNIQUE") {
          const uniqueKeys = asList(nodeGet(constraintNode, "keys"));
          for (const uk of uniqueKeys) {
            const colName = extractString(uk);
            const existingIdx = columns.findIndex((c) => c.name === colName);
            if (existingIdx !== -1) {
              const existing = columns[existingIdx]!;
              columns[existingIdx] = createColumnDef(existing.name, existing.typeName, {
                ...existing,
                unique: true,
              });
            }
          }
        }

        // CHECK constraint
        // CONSTR_CHECK = 1
        if (contype === 1 || contype === "CONSTR_CHECK") {
          const conname =
            nodeStr(constraintNode, "conname") ||
            `check_${String(checkConstraints.length)}`;
          const rawExpr = nodeGet(constraintNode, "raw_expr");
          if (rawExpr !== null && rawExpr !== undefined) {
            const evaluator = new ExprEvaluator();
            const exprNode = asObj(rawExpr);
            checkConstraints.push([
              conname,
              (row: Record<string, unknown>) => {
                const result = evaluator.evaluate(exprNode, row);
                return result === true;
              },
            ]);
          }
        }
      }
    }

    // Apply table-level PRIMARY KEY to column definition
    if (tablePrimaryKey !== null) {
      const idx = columns.findIndex((c) => c.name === tablePrimaryKey);
      if (idx !== -1) {
        const existing = columns[idx]!;
        columns[idx] = createColumnDef(existing.name, existing.typeName, {
          ...existing,
          primaryKey: true,
          notNull: true,
        });
      }
    }

    const table = new Table(tableName, columns);
    table.foreignKeys = foreignKeys;
    table.checkConstraints = checkConstraints;
    this._tables.set(tableName, table);

    // Register foreign key validators
    if (foreignKeys.length > 0) {
      this._registerFkValidators(tableName, table, foreignKeys);
    }

    // Create implicit sequence for SERIAL/BIGSERIAL columns
    for (const col of columns) {
      if (col.autoIncrement) {
        const seqName = `${tableName}_${col.name}_seq`;
        this._sequences.set(seqName, { current: 0, increment: 1, start: 1 });
      }
    }

    return null;
  }

  // ==================================================================
  // DDL: CREATE TABLE AS SELECT
  // ==================================================================

  private _compileCreateTableAs(
    stmt: Record<string, unknown>,
    params: unknown[],
  ): SQLResult | null {
    const into = asObj(nodeGet(stmt, "into"));
    const rel = asObj(nodeGet(into, "rel"));
    const tableName = qualifiedName(rel);

    if (this._tables.has(tableName)) {
      throw new Error(`Table "${tableName}" already exists`);
    }

    const query = asObj(nodeGet(stmt, "query"));
    const selectStmt = asObj(nodeGet(query, "SelectStmt") ?? query);
    const result = this._compileSelect(selectStmt, params);

    // Infer column definitions from query result
    const columns: ColumnDef[] = [];
    for (const colName of result.columns) {
      let typeName = "text";
      let jsType = "string";
      if (result.rows.length > 0) {
        const sample = result.rows[0]![colName];
        if (typeof sample === "number") {
          if (Number.isInteger(sample)) {
            typeName = "integer";
            jsType = "number";
          } else {
            typeName = "real";
            jsType = "number";
          }
        } else if (typeof sample === "boolean") {
          typeName = "boolean";
          jsType = "boolean";
        } else if (Array.isArray(sample)) {
          typeName = "text[]";
          jsType = "array";
        } else if (typeof sample === "object" && sample !== null) {
          typeName = "jsonb";
          jsType = "object";
        }
      }
      columns.push(createColumnDef(colName, typeName, { pythonType: jsType }));
    }

    const table = new Table(tableName, columns);
    this._tables.set(tableName, table);

    // Insert result rows
    let inserted = 0;
    for (const row of result.rows) {
      const clean: Record<string, unknown> = {};
      for (const c of result.columns) {
        clean[c] = row[c] ?? null;
      }
      table.insert(clean);
      inserted++;
    }

    return { columns: ["inserted"], rows: [{ inserted }] };
  }

  // ==================================================================
  // DDL: CREATE SEQUENCE / ALTER SEQUENCE
  // ==================================================================

  private _compileCreateSequence(stmt: Record<string, unknown>): SQLResult | null {
    const sequence = asObj(nodeGet(stmt, "sequence"));
    const seqName = extractRelName(sequence);
    const options = asList(nodeGet(stmt, "options"));

    if (this._sequences.has(seqName)) {
      if (nodeGet(stmt, "if_not_exists") === true) return null;
      throw new Error(`Sequence "${seqName}" already exists`);
    }

    let startVal = 1;
    let incrementVal = 1;

    for (const opt of options) {
      const defElem = asObj(nodeGet(opt, "DefElem") ?? opt);
      const defname = nodeStr(defElem, "defname");
      const arg = nodeGet(defElem, "arg");
      if (defname === "start" && arg !== null) {
        const argObj = asObj(arg);
        startVal = Number(
          nodeGet(argObj, "ival") ??
            nodeGet(asObj(nodeGet(argObj, "Integer") ?? {}), "ival") ??
            1,
        );
      }
      if (defname === "increment" && arg !== null) {
        const argObj = asObj(arg);
        incrementVal = Number(
          nodeGet(argObj, "ival") ??
            nodeGet(asObj(nodeGet(argObj, "Integer") ?? {}), "ival") ??
            1,
        );
      }
    }

    this._sequences.set(seqName, {
      current: startVal - incrementVal,
      increment: incrementVal,
      start: startVal,
    });
    return null;
  }

  private _compileAlterSequence(stmt: Record<string, unknown>): SQLResult | null {
    const sequence = asObj(nodeGet(stmt, "sequence"));
    const seqName = extractRelName(sequence);
    const seq = this._sequences.get(seqName);
    if (seq === undefined) {
      throw new Error(`Sequence "${seqName}" does not exist`);
    }

    const options = asList(nodeGet(stmt, "options"));
    for (const opt of options) {
      const defElem = asObj(nodeGet(opt, "DefElem") ?? opt);
      const defname = nodeStr(defElem, "defname");
      const arg = nodeGet(defElem, "arg");

      if (defname === "restart") {
        if (arg !== null && arg !== undefined) {
          const argObj = asObj(arg);
          const restartVal = Number(
            nodeGet(argObj, "ival") ??
              nodeGet(asObj(nodeGet(argObj, "Integer") ?? {}), "ival") ??
              seq.start ??
              1,
          );
          seq.current = restartVal - seq.increment;
        } else {
          seq.current = (seq.start ?? 1) - seq.increment;
        }
      } else if (defname === "increment") {
        if (arg !== null && arg !== undefined) {
          const argObj = asObj(arg);
          seq.increment = Number(
            nodeGet(argObj, "ival") ??
              nodeGet(asObj(nodeGet(argObj, "Integer") ?? {}), "ival") ??
              1,
          );
        }
      } else if (defname === "start") {
        if (arg !== null && arg !== undefined) {
          const argObj = asObj(arg);
          seq.start = Number(
            nodeGet(argObj, "ival") ??
              nodeGet(asObj(nodeGet(argObj, "Integer") ?? {}), "ival") ??
              1,
          );
        }
      }
    }

    return null;
  }

  private _parseColumnDef(node: Record<string, unknown>): ColumnDef {
    const colName = nodeStr(node, "colname");
    const typeNameNode = asObj(nodeGet(node, "typeName"));

    // Extract type name
    const names = asList(nodeGet(typeNameNode, "names"));
    const typeNames: string[] = [];
    for (const n of names) {
      const s = extractString(n);
      if (s && s !== "pg_catalog") {
        typeNames.push(s);
      }
    }

    // Check for array type
    const arrayBounds = asList(nodeGet(typeNameNode, "arrayBounds"));
    const isArray = arrayBounds.length > 0;

    let [resolvedType, jsType] =
      typeNames.length > 0 ? resolveType(typeNames) : ["text", "string"];

    // Check for SERIAL types
    let isAutoIncrement = false;
    const rawTypeUpper = typeNames.join(" ").toUpperCase();
    if (
      rawTypeUpper === "SERIAL" ||
      rawTypeUpper === "BIGSERIAL" ||
      rawTypeUpper === "SMALLSERIAL"
    ) {
      isAutoIncrement = true;
    }

    if (isArray) {
      resolvedType = "array";
      jsType = "array";
    }

    // Extract type modifiers (precision, scale, vector dimensions)
    const typmods = asList(nodeGet(typeNameNode, "typmods"));
    let numericPrecision: number | null = null;
    let numericScale: number | null = null;
    let vectorDimensions: number | null = null;

    if (typmods.length > 0) {
      const firstMod =
        nodeGet(typmods[0]!, "Integer") ?? nodeGet(typmods[0]!, "A_Const");
      if (firstMod !== null && firstMod !== undefined) {
        const val = nodeGet(asObj(firstMod), "ival");
        if (val !== null && typeof val === "number") {
          if (resolvedType === "vector") {
            vectorDimensions = val;
          } else {
            numericPrecision = val;
          }
        }
      }
      if (typmods.length > 1) {
        const secondMod =
          nodeGet(typmods[1]!, "Integer") ?? nodeGet(typmods[1]!, "A_Const");
        if (secondMod !== null && secondMod !== undefined) {
          const val = nodeGet(asObj(secondMod), "ival");
          if (val !== null && typeof val === "number") {
            numericScale = val;
          }
        }
      }
    }

    // Parse column constraints
    let isPrimaryKey = false;
    let isNotNull = false;
    let isUnique = false;
    let defaultValue: unknown = null;

    const constraints = asList(nodeGet(node, "constraints"));
    for (const constraint of constraints) {
      const conNode = asObj(nodeGet(constraint, "Constraint") ?? constraint);
      const contype = nodeGet(conNode, "contype");

      // CONSTR_NOTNULL = 0 or 1 depending on version
      if (contype === "CONSTR_NOTNULL" || contype === 1) {
        isNotNull = true;
      }
      // CONSTR_PRIMARY = 5
      if (contype === "CONSTR_PRIMARY" || contype === 5) {
        isPrimaryKey = true;
        isNotNull = true;
      }
      // CONSTR_UNIQUE = 4
      if (contype === "CONSTR_UNIQUE" || contype === 4) {
        isUnique = true;
      }
      // CONSTR_DEFAULT = 2
      if (contype === "CONSTR_DEFAULT" || contype === 2) {
        const rawExpr = nodeGet(conNode, "raw_expr");
        if (rawExpr !== null && rawExpr !== undefined) {
          const rawObj = asObj(rawExpr);
          // Check if this is a SQLValueFunction or FuncCall that should
          // be deferred (evaluated at insert time, not parse time).
          const isSQLValueFunc = nodeGet(rawObj, "SQLValueFunction") !== null
            && nodeGet(rawObj, "SQLValueFunction") !== undefined;
          const isFuncCall = nodeGet(rawObj, "FuncCall") !== null
            && nodeGet(rawObj, "FuncCall") !== undefined;
          if (isSQLValueFunc || isFuncCall) {
            // Store as deferred default: evaluate at insert time
            const capturedExpr = rawObj;
            defaultValue = {
              _astDefault: true,
              _evalFn: () => {
                const ev = new ExprEvaluator();
                return ev.evaluate(capturedExpr, {});
              },
            };
          } else {
            const evaluator = new ExprEvaluator();
            defaultValue = evaluator.evaluate(rawObj, {});
          }
        }
      }
      // CONSTR_NULL = 0 (explicit NULL constraint -- allowed)
      // CONSTR_IDENTITY = 9 (GENERATED ALWAYS AS IDENTITY)
      if (contype === "CONSTR_IDENTITY" || contype === 9) {
        isAutoIncrement = true;
        isNotNull = true;
      }
      // CONSTR_FOREIGN = 8
      // (column-level FK -- handled below but doesn't affect column flags)
    }

    // SERIAL implies NOT NULL + autoincrement
    if (isAutoIncrement) {
      isNotNull = true;
    }

    return createColumnDef(colName, resolvedType, {
      pythonType: jsType,
      primaryKey: isPrimaryKey,
      notNull: isNotNull,
      autoIncrement: isAutoIncrement,
      defaultValue,
      vectorDimensions,
      unique: isUnique,
      numericPrecision,
      numericScale,
    });
  }

  // ==================================================================
  // DDL: DROP TABLE / DROP VIEW / DROP INDEX
  // ==================================================================

  private _compileDrop(stmt: Record<string, unknown>): SQLResult | null {
    // removeType: 41=TABLE, 20=INDEX, 51=VIEW, 17=FOREIGN_SERVER, 18=FOREIGN_TABLE, 36=SCHEMA
    const removeType = nodeGet(stmt, "removeType") as number | string;
    if (removeType === 36 || removeType === "OBJECT_SCHEMA") {
      return this._compileDropSchema(stmt);
    }
    if (removeType === 20 || removeType === "OBJECT_INDEX") {
      return this._compileDropIndex(stmt);
    }
    if (removeType === 51 || removeType === "OBJECT_VIEW") {
      return this._compileDropView(stmt);
    }
    if (removeType === 17 || removeType === "OBJECT_FOREIGN_SERVER") {
      return this._compileDropForeignServer(stmt);
    }
    if (removeType === 18 || removeType === "OBJECT_FOREIGN_TABLE") {
      return this._compileDropForeignTable(stmt);
    }
    return this._compileDropTable(stmt);
  }

  private _compileDropTable(stmt: Record<string, unknown>): SQLResult | null {
    const objects = asList(nodeGet(stmt, "objects"));
    const ifExists = nodeGet(stmt, "missing_ok") === true;

    for (const obj of objects) {
      const items = asList(obj);
      let tableName: string;
      if (items.length > 0) {
        tableName = extractString(items[items.length - 1]!);
      } else {
        tableName = extractString(obj);
      }

      if (!tableName) continue;

      if (!this._tables.has(tableName)) {
        if (ifExists) continue;
        throw new Error(`Table "${tableName}" does not exist`);
      }
      this._tables.delete(tableName);

      // Clean up associated sequences
      for (const [seqName] of this._sequences) {
        if (seqName.startsWith(`${tableName}_`)) {
          this._sequences.delete(seqName);
        }
      }
    }
    return null;
  }

  private _compileDropView(stmt: Record<string, unknown>): SQLResult | null {
    const objects = asList(nodeGet(stmt, "objects"));
    const ifExists = nodeGet(stmt, "missing_ok") === true;

    for (const obj of objects) {
      // Objects may be List-wrapped: { List: { items: [...] } }
      let items = asList(obj);
      if (items.length === 0) {
        const listNode = nodeGet(asObj(obj), "List");
        if (listNode !== null && listNode !== undefined) {
          items = asList(nodeGet(asObj(listNode), "items"));
        }
      }
      let viewName: string;
      if (items.length > 0) {
        viewName = extractString(items[items.length - 1]!);
      } else {
        viewName = extractString(obj);
      }

      if (!viewName) continue;

      if (!this._views.has(viewName)) {
        if (ifExists) continue;
        throw new Error(`View "${viewName}" does not exist`);
      }
      this._views.delete(viewName);
    }
    return null;
  }

  private _compileDropIndex(stmt: Record<string, unknown>): SQLResult | null {
    const objects = asList(nodeGet(stmt, "objects"));
    const ifExists = nodeGet(stmt, "missing_ok") === true;

    for (const obj of objects) {
      const items = asList(obj);
      let idxName: string;
      if (items.length > 0) {
        idxName = extractString(items[items.length - 1]!);
      } else {
        idxName = extractString(obj);
      }
      if (!idxName) continue;

      // Check in-memory BTREE index
      const eng = this._engine as { _btreeIndexes?: Map<string, [string, string[]]> } | null;
      if (eng && eng._btreeIndexes && eng._btreeIndexes.has(idxName)) {
        eng._btreeIndexes.delete(idxName);
        this._indexes.delete(idxName);
        continue;
      }

      const idx = this._indexes.get(idxName);
      if (idx === undefined) {
        if (ifExists) continue;
        throw new Error(`Index "${idxName}" does not exist`);
      }

      // If this was a GIN index, remove the FTS fields and clear the index data
      if (idx.method === "gin") {
        const table = this._tables.get(idx.tableName);
        if (table !== undefined) {
          for (const col of idx.columns) {
            table.ftsFields.delete(col);
          }
          // If no FTS fields remain, clear the inverted index entirely
          if (table.ftsFields.size === 0) {
            table.invertedIndex.clear();
          }
        }
      }

      this._indexes.delete(idxName);
    }
    return null;
  }

  // ==================================================================
  // DDL: FOREIGN DATA WRAPPERS
  // ==================================================================

  private _compileCreateForeignServer(stmt: Record<string, unknown>): SQLResult | null {
    const name = nodeStr(stmt, "servername");
    if (this._foreignServers.has(name)) {
      if (nodeGet(stmt, "if_not_exists") === true) return null;
      throw new Error(`Foreign server '${name}' already exists`);
    }

    const fdwType = nodeStr(stmt, "fdwname");
    if (fdwType !== "duckdb_fdw" && fdwType !== "arrow_fdw") {
      throw new Error(`Unsupported FDW type: '${fdwType}'`);
    }

    const options: Record<string, string> = {};
    const optList = asList(nodeGet(stmt, "options"));
    for (const opt of optList) {
      const optObj = asObj(nodeGet(opt, "DefElem") ?? opt);
      const defname = nodeStr(optObj, "defname");
      const argNode = asObj(nodeGet(optObj, "arg"));
      const argVal =
        nodeStr(argNode, "sval") ||
        nodeStr(asObj(nodeGet(argNode, "String") ?? {}), "sval");
      if (defname) options[defname] = argVal;
    }

    this._foreignServers.set(name, { name, fdwType, options });
    return null;
  }

  private _compileCreateForeignTable(stmt: Record<string, unknown>): SQLResult | null {
    const base = asObj(nodeGet(stmt, "base"));
    const relation = asObj(nodeGet(base, "relation"));
    const tableName = extractRelName(relation);
    if (this._foreignTables.has(tableName)) {
      if (nodeGet(base, "if_not_exists") === true) return null;
      throw new Error(`Foreign table '${tableName}' already exists`);
    }
    if (this._tables.has(tableName)) {
      throw new Error(`Table '${tableName}' already exists`);
    }

    const serverName = nodeStr(stmt, "servername");
    if (!this._foreignServers.has(serverName)) {
      throw new Error(`Foreign server '${serverName}' does not exist`);
    }

    const columns = new Map<string, ColumnDef>();
    const tableElts = asList(nodeGet(base, "tableElts"));
    for (const elt of tableElts) {
      const colDefNode = asObj(nodeGet(elt, "ColumnDef") ?? elt);
      const col = this._parseColumnDef(colDefNode);
      columns.set(col.name, col);
    }

    const options: Record<string, string> = {};
    const optList = asList(nodeGet(stmt, "options"));
    for (const opt of optList) {
      const optObj = asObj(nodeGet(opt, "DefElem") ?? opt);
      const defname = nodeStr(optObj, "defname");
      const argNode = asObj(nodeGet(optObj, "arg"));
      const argVal =
        nodeStr(argNode, "sval") ||
        nodeStr(asObj(nodeGet(argNode, "String") ?? {}), "sval");
      if (defname) options[defname] = argVal;
    }

    this._foreignTables.set(tableName, {
      name: tableName,
      serverName,
      columns,
      options,
    });
    return null;
  }

  private _compileDropForeignServer(stmt: Record<string, unknown>): SQLResult | null {
    const objects = asList(nodeGet(stmt, "objects"));
    const ifExists = nodeGet(stmt, "missing_ok") === true;

    for (const obj of objects) {
      // DROP SERVER objects may be a flat string node or list of string nodes
      let name: string;
      if (Array.isArray(obj)) {
        const items = obj as Record<string, unknown>[];
        name = extractString(items[items.length - 1]!);
      } else {
        name = extractString(obj);
      }

      if (this._foreignServers.has(name)) {
        // Validate no foreign tables reference this server
        for (const ft of this._foreignTables.values()) {
          if (ft.serverName === name) {
            throw new Error(
              `Cannot drop server '${name}': foreign table '${ft.name}' depends on it`,
            );
          }
        }
        // Close cached handler if present
        const handler = this._fdwHandlers.get(name);
        if (handler !== undefined) {
          handler.close();
          this._fdwHandlers.delete(name);
        }
        this._foreignServers.delete(name);
      } else if (!ifExists) {
        throw new Error(`Foreign server '${name}' does not exist`);
      }
    }
    return null;
  }

  private _compileDropForeignTable(stmt: Record<string, unknown>): SQLResult | null {
    const objects = asList(nodeGet(stmt, "objects"));
    const ifExists = nodeGet(stmt, "missing_ok") === true;

    for (const obj of objects) {
      // DROP FOREIGN TABLE objects are wrapped in a List node:
      //   { "List": { "items": [ { "String": { "sval": "name" } } ] } }
      let tableName: string;
      const listNode = nodeGet(obj, "List");
      if (listNode !== null && listNode !== undefined) {
        const items = asList(nodeGet(asObj(listNode), "items"));
        tableName = items.length > 0
          ? extractString(items[items.length - 1]!)
          : "";
      } else if (Array.isArray(obj)) {
        const items = obj as Record<string, unknown>[];
        tableName = extractString(items[items.length - 1]!);
      } else {
        tableName = extractString(obj);
      }
      if (!tableName) continue;

      if (this._foreignTables.has(tableName)) {
        this._foreignTables.delete(tableName);
      } else if (!ifExists) {
        throw new Error(`Foreign table '${tableName}' does not exist`);
      }
    }
    return null;
  }

  // ==================================================================
  // DDL: CREATE VIEW
  // ==================================================================

  private _compileCreateView(stmt: Record<string, unknown>): SQLResult | null {
    const view = asObj(nodeGet(stmt, "view"));
    const viewName = extractRelName(view);
    if (this._tables.has(viewName)) {
      throw new Error(`"${viewName}" already exists as a table`);
    }
    if (this._views.has(viewName)) {
      throw new Error(`View "${viewName}" already exists`);
    }
    const query = asObj(nodeGet(stmt, "query"));
    const selectStmt = asObj(nodeGet(query, "SelectStmt") ?? query);
    this._views.set(viewName, selectStmt);
    return null;
  }

  // ==================================================================
  // DDL: CREATE INDEX
  // ==================================================================

  private _compileCreateIndex(stmt: Record<string, unknown>): SQLResult | null {
    const idxName = nodeStr(stmt, "idxname");
    const ifNotExists = nodeGet(stmt, "if_not_exists") === true;
    const relation = asObj(nodeGet(stmt, "relation"));
    const tableName = qualifiedName(relation);
    const method = (nodeStr(stmt, "accessMethod") || "btree").toLowerCase();

    if (this._indexes.has(idxName)) {
      if (ifNotExists) return null;
      throw new Error(`Index "${idxName}" already exists`);
    }

    const table = this._tables.get(tableName);
    if (!table) {
      throw new Error(`Table "${tableName}" does not exist`);
    }

    // Extract column names from indexParams
    const indexParams = asList(nodeGet(stmt, "indexParams"));
    const columns: string[] = [];
    for (const param of indexParams) {
      const elem = asObj(nodeGet(param, "IndexElem") ?? param);
      const colName = nodeStr(elem, "name");
      if (colName) columns.push(colName);
    }

    if (columns.length === 0) {
      throw new Error("CREATE INDEX requires at least one column");
    }

    // Extract WITH options (e.g., analyzer = 'name')
    const options: Record<string, string> = {};
    const optList = asList(nodeGet(stmt, "options"));
    for (const opt of optList) {
      const optObj = asObj(nodeGet(opt, "DefElem") ?? opt);
      const defname = nodeStr(optObj, "defname");
      const argNode = asObj(nodeGet(optObj, "arg"));
      const argVal =
        nodeStr(argNode, "sval") ||
        nodeStr(asObj(nodeGet(argNode, "String") ?? {}), "sval");
      if (defname) options[defname] = argVal;
    }

    if (method === "gin") {
      // GIN index = FTS inverted index on specified columns
      for (const col of columns) {
        if (!table.columns.has(col)) {
          throw new Error(
            `Column "${col}" does not exist in table "${tableName}"`,
          );
        }
        table.ftsFields.add(col);
      }

      // Apply analyzer if specified
      const analyzerName = options["analyzer"];
      if (analyzerName) {
        const analyzer = getAnalyzerFn(analyzerName);
        for (const col of columns) {
          table.invertedIndex.setFieldAnalyzer(col, analyzer, "both");
        }
      }

      // Backfill: index existing rows for the newly added FTS fields
      for (const [docId, doc] of table.documentStore.iterAll()) {
        const textFields: Record<string, string> = {};
        for (const col of columns) {
          const v = (doc as Record<string, unknown>)[col];
          if (typeof v === "string") textFields[col] = v;
        }
        if (Object.keys(textFields).length > 0) {
          table.invertedIndex.addDocument(docId, textFields);
        }
      }
    }
    // btree, hnsw, ivf, rtree: vector/spatial indexes are auto-created
    // from column types. For in-memory engines, track BTREE metadata for
    // optimizer use; actual scans fall back to document store iteration.
    if (method === "btree") {
      const eng = this._engine as { _btreeIndexes?: Map<string, [string, string[]]> } | null;
      if (eng && eng._btreeIndexes) {
        eng._btreeIndexes.set(idxName, [tableName, columns]);
      }
    }

    this._indexes.set(idxName, { name: idxName, tableName, method, columns });
    return null;
  }

  // ==================================================================
  // DDL: ALTER TABLE
  // ==================================================================

  private _compileAlterTable(stmt: Record<string, unknown>): SQLResult | null {
    const relation = asObj(nodeGet(stmt, "relation"));
    const tableName = qualifiedName(relation);
    const table = this._tables.get(tableName);
    if (!table) {
      throw new Error(`Table "${tableName}" does not exist`);
    }

    const cmds = asList(nodeGet(stmt, "cmds"));
    for (const cmdWrapper of cmds) {
      const cmd = asObj(nodeGet(cmdWrapper, "AlterTableCmd") ?? cmdWrapper);
      const subtype = nodeGet(cmd, "subtype") as number | string;
      const subtypeStr = typeof subtype === "string" ? subtype : "";

      if (subtypeStr === "AT_AddColumn" || subtype === 0 || subtype === 14) {
        // ADD COLUMN
        const defNode = asObj(nodeGet(cmd, "def"));
        const colDefNode = asObj(nodeGet(defNode, "ColumnDef") ?? defNode);
        const col = this._parseColumnDef(colDefNode);
        if (table.columns.has(col.name)) {
          throw new Error(
            `Column "${col.name}" already exists in table "${tableName}"`,
          );
        }
        table.columns.set(col.name, col);
      } else if (subtypeStr === "AT_DropColumn" || subtype === 10 || subtype === 12) {
        // DROP COLUMN
        const colName = nodeStr(cmd, "name");
        const missingOk = nodeGet(cmd, "missing_ok") === true;
        if (!table.columns.has(colName)) {
          if (missingOk) continue;
          throw new Error(`Column "${colName}" does not exist in table "${tableName}"`);
        }
        table.columns.delete(colName);
        // Remove field from all documents
        for (const docId of table.documentStore.docIds) {
          const doc = table.documentStore.get(docId);
          if (doc && colName in doc) {
            Reflect.deleteProperty(doc, colName);
            table.documentStore.put(docId, doc);
          }
        }
      } else if (subtypeStr === "AT_ColumnDefault" || subtype === 5 || subtype === 6) {
        // SET DEFAULT / DROP DEFAULT
        const colName = nodeStr(cmd, "name");
        if (!table.columns.has(colName)) {
          throw new Error(`Column "${colName}" does not exist in table "${tableName}"`);
        }
        const defExpr = nodeGet(cmd, "def");
        const existing = table.columns.get(colName)!;
        if (defExpr !== null && defExpr !== undefined) {
          const evaluator = new ExprEvaluator();
          const newDefault = evaluator.evaluate(asObj(defExpr), {});
          table.columns.set(
            colName,
            createColumnDef(existing.name, existing.typeName, {
              ...existing,
              defaultValue: newDefault,
            }),
          );
        } else {
          table.columns.set(
            colName,
            createColumnDef(existing.name, existing.typeName, {
              ...existing,
              defaultValue: null,
            }),
          );
        }
      } else if (subtypeStr === "AT_SetNotNull" || subtype === 7 || subtype === 9) {
        // SET NOT NULL
        const colName = nodeStr(cmd, "name");
        if (!table.columns.has(colName)) {
          throw new Error(`Column "${colName}" does not exist in table "${tableName}"`);
        }
        // Validate existing data
        for (const docId of table.documentStore.docIds) {
          const doc = table.documentStore.get(docId);
          if (doc) {
            const val = doc[colName];
            if (val === null || val === undefined) {
              throw new Error(
                `Column "${colName}" contains NULL values; cannot set NOT NULL`,
              );
            }
          }
        }
        const existing = table.columns.get(colName)!;
        table.columns.set(
          colName,
          createColumnDef(existing.name, existing.typeName, {
            ...existing,
            notNull: true,
          }),
        );
      } else if (subtypeStr === "AT_DropNotNull" || subtype === 6 || subtype === 8) {
        // DROP NOT NULL
        const colName = nodeStr(cmd, "name");
        if (!table.columns.has(colName)) {
          throw new Error(`Column "${colName}" does not exist in table "${tableName}"`);
        }
        const existing = table.columns.get(colName)!;
        table.columns.set(
          colName,
          createColumnDef(existing.name, existing.typeName, {
            ...existing,
            notNull: false,
          }),
        );
      } else if (
        subtypeStr === "AT_AlterColumnType" ||
        subtype === 25 ||
        subtype === 28
      ) {
        // ALTER COLUMN TYPE
        const colName = nodeStr(cmd, "name");
        if (!table.columns.has(colName)) {
          throw new Error(`Column "${colName}" does not exist in table "${tableName}"`);
        }
        const defNode = asObj(nodeGet(cmd, "def"));
        const colDefNode = asObj(nodeGet(defNode, "ColumnDef") ?? defNode);
        const typeNameNode = asObj(nodeGet(colDefNode, "typeName"));
        const typeNames: string[] = [];
        for (const n of asList(nodeGet(typeNameNode, "names"))) {
          const s = extractString(n);
          if (s && s !== "pg_catalog") typeNames.push(s);
        }
        const [newResolvedType, newJsType] =
          typeNames.length > 0 ? resolveType(typeNames) : ["text", "string"];
        const existing = table.columns.get(colName)!;
        table.columns.set(
          colName,
          createColumnDef(colName, newResolvedType, {
            pythonType: newJsType,
            primaryKey: existing.primaryKey,
            notNull: existing.notNull,
            autoIncrement: existing.autoIncrement,
            defaultValue: existing.defaultValue,
            vectorDimensions: existing.vectorDimensions,
            unique: existing.unique,
            numericPrecision: existing.numericPrecision,
            numericScale: existing.numericScale,
          }),
        );
        // Coerce existing data to the new type
        for (const docId of Array.from(table.documentStore.docIds)) {
          const doc = table.documentStore.get(docId);
          if (doc && doc[colName] !== null && doc[colName] !== undefined) {
            let coerced: unknown = doc[colName];
            if (newJsType === "number") {
              coerced = Number(coerced);
              if (isNaN(coerced as number)) coerced = doc[colName];
            } else if (newJsType === "string") {
              coerced = String(coerced);
            } else if (newJsType === "boolean") {
              coerced = Boolean(coerced);
            }
            doc[colName] = coerced;
            table.documentStore.put(docId, doc);
          }
        }
      } else if (
        subtypeStr === "AT_AddConstraint" ||
        subtype === 14 ||
        subtype === 17
      ) {
        // ADD CONSTRAINT (CHECK, UNIQUE, PRIMARY KEY, FOREIGN KEY)
        const defNode = asObj(nodeGet(cmd, "def"));
        const constraintNode = asObj(nodeGet(defNode, "Constraint") ?? defNode);
        const ctRaw = nodeGet(constraintNode, "contype") as number | string;
        const ctStr = typeof ctRaw === "string" ? ctRaw : "";

        if (ctStr === "CONSTR_CHECK" || ctRaw === 3) {
          const rawExpr = nodeGet(constraintNode, "raw_expr");
          const conname = nodeStr(constraintNode, "conname") || "unnamed_check";
          // Validate existing data
          const ev = new ExprEvaluator();
          for (const docId of table.documentStore.docIds) {
            const doc = table.documentStore.get(docId);
            if (doc !== null && !ev.evaluate(asObj(rawExpr!), doc)) {
              throw new Error(
                `CHECK constraint '${conname}' is violated by existing data in table '${tableName}'`,
              );
            }
          }
          table.checkConstraints.push([
            conname,
            (row: Record<string, unknown>) => {
              const evaluator = new ExprEvaluator();
              return Boolean(evaluator.evaluate(asObj(rawExpr!), row));
            },
          ]);
        } else if (ctStr === "CONSTR_UNIQUE" || ctRaw === 4) {
          const keys = asList(nodeGet(constraintNode, "keys"));
          for (const k of keys) {
            const col = extractString(k);
            const existing = table.columns.get(col);
            if (existing) {
              table.columns.set(
                col,
                createColumnDef(existing.name, existing.typeName, {
                  ...existing,
                  unique: true,
                }),
              );
            }
          }
          (table as unknown as { _uniqueIndexesBuilt: boolean })._uniqueIndexesBuilt = false;
        } else if (ctStr === "CONSTR_PRIMARY" || ctRaw === 5) {
          const keys = asList(nodeGet(constraintNode, "keys"));
          if (keys.length > 0) {
            const pkCol = extractString(keys[0]!);
            const existing = table.columns.get(pkCol);
            if (existing) {
              table.columns.set(
                pkCol,
                createColumnDef(existing.name, existing.typeName, {
                  ...existing,
                  primaryKey: true,
                  notNull: true,
                }),
              );
            }
            (table as { primaryKey: string | null }).primaryKey = pkCol;
          }
        } else if (ctStr === "CONSTR_FOREIGN" || ctRaw === 6) {
          const fkAttrs = asList(nodeGet(constraintNode, "fk_attrs"));
          const pkTable = asObj(nodeGet(constraintNode, "pktable")!);
          const pkAttrs = asList(nodeGet(constraintNode, "pk_attrs"));
          if (fkAttrs.length > 0 && pkAttrs.length > 0) {
            const fkDef: ForeignKeyDef = {
              column: extractString(fkAttrs[0]!),
              refTable: extractRelName(pkTable),
              refColumn: extractString(pkAttrs[0]!),
            };
            table.foreignKeys.push(fkDef);
          }
        }
      } else {
        throw new Error(`Unsupported ALTER TABLE subcommand: ${String(subtype)}`);
      }
    }

    return null;
  }

  // ==================================================================
  // DDL: RENAME TABLE / COLUMN
  // ==================================================================

  private _compileRename(stmt: Record<string, unknown>): SQLResult | null {
    const renameType = nodeGet(stmt, "renameType") as number | string;
    const renameTypeStr = typeof renameType === "string" ? renameType : "";

    // OBJECT_TABLE = 36 or 42 depending on pg version
    if (renameTypeStr === "OBJECT_TABLE" || renameType === 36 || renameType === 42) {
      const relation = asObj(nodeGet(stmt, "relation"));
      const oldName = qualifiedName(relation);
      const newName = nodeStr(stmt, "newname");
      const table = this._tables.get(oldName);
      if (!table) {
        throw new Error(`Table "${oldName}" does not exist`);
      }
      if (this._tables.has(newName)) {
        throw new Error(`Table "${newName}" already exists`);
      }
      this._tables.delete(oldName);
      this._tables.set(newName, table);
      return null;
    }

    // OBJECT_COLUMN = 8 or 9 depending on pg version
    if (renameTypeStr === "OBJECT_COLUMN" || renameType === 8 || renameType === 9) {
      const relation = asObj(nodeGet(stmt, "relation"));
      const tableName = qualifiedName(relation);
      const table = this._tables.get(tableName);
      if (!table) {
        throw new Error(`Table "${tableName}" does not exist`);
      }
      const oldCol = nodeStr(stmt, "subname");
      const newCol = nodeStr(stmt, "newname");
      if (!table.columns.has(oldCol)) {
        throw new Error(`Column "${oldCol}" does not exist in table "${tableName}"`);
      }
      if (table.columns.has(newCol)) {
        throw new Error(`Column "${newCol}" already exists in table "${tableName}"`);
      }
      // Rebuild column map preserving insertion order
      const entries = [...table.columns.entries()];
      table.columns.clear();
      for (const [name, col] of entries) {
        if (name === oldCol) {
          const renamed = createColumnDef(newCol, col.typeName, { ...col });
          table.columns.set(newCol, renamed);
        } else {
          table.columns.set(name, col);
        }
      }
      // Rename field in all documents
      for (const docId of table.documentStore.docIds) {
        const doc = table.documentStore.get(docId);
        if (doc && oldCol in doc) {
          doc[newCol] = doc[oldCol];
           
          Reflect.deleteProperty(doc, oldCol);
          table.documentStore.put(docId, doc);
        }
      }
      return null;
    }

    throw new Error(`Unsupported RENAME type: ${String(renameType)}`);
  }

  // ==================================================================
  // DDL: TRUNCATE
  // ==================================================================

  private _compileTruncate(stmt: Record<string, unknown>): SQLResult | null {
    const relations = asList(nodeGet(stmt, "relations"));
    for (const rel of relations) {
      const rangeVar = asObj(nodeGet(rel, "RangeVar") ?? rel);
      const tableName = qualifiedName(rangeVar);
      const table = this._tables.get(tableName);
      if (!table) {
        throw new Error(`Table "${tableName}" does not exist`);
      }
      table.documentStore.clear();
      table.invertedIndex.clear();
      for (const [, vi] of table.vectorIndexes) {
        vi.clear();
      }
      for (const [, si] of table.spatialIndexes) {
        si.clear();
      }
      // Reset auto-increment sequences for SERIAL columns
      table._nextDocId = 1;
      for (const col of table.columns.values()) {
        if (col.autoIncrement) {
          const seqName = `${tableName}_${col.name}_seq`;
          const seq = this._sequences.get(seqName);
          if (seq) {
            seq.current = (seq.start ?? 1) - seq.increment;
          }
        }
      }
    }
    return null;
  }

  // ==================================================================
  // DML: INSERT INTO
  // ==================================================================

  private _compileInsert(
    stmt: Record<string, unknown>,
    params: unknown[],
  ): SQLResult | null {
    const relation = asObj(nodeGet(stmt, "relation"));
    const tableName = qualifiedName(relation);
    const table = this._tables.get(tableName);
    if (!table) {
      throw new Error(`Table "${tableName}" does not exist`);
    }

    // Column names
    const colsNodes = asList(nodeGet(stmt, "cols"));
    let colNames: string[];
    if (colsNodes.length > 0) {
      colNames = colsNodes.map((col) => {
        const target = asObj(nodeGet(col, "ResTarget") ?? col);
        return nodeStr(target, "name");
      });
    } else {
      colNames = table.columnNames;
    }

    // VALUES or SELECT source
    const selectStmtRaw = asObj(nodeGet(stmt, "selectStmt"));
    const selectStmt = asObj(nodeGet(selectStmtRaw, "SelectStmt") ?? selectStmtRaw);
    const valuesLists = asList(nodeGet(selectStmt, "valuesLists"));

    const evaluator = new ExprEvaluator({
      params,
      sequences: this._sequences,
      subqueryExecutor: this._makeSubqueryExecutor(params),
    });

    // Collect source rows
    const sourceRows: Record<string, unknown>[] = [];

    if (valuesLists.length === 0) {
      // INSERT INTO ... SELECT ...
      const result = this._compileSelect(selectStmt, params);
      for (const row of result.rows) {
        const mappedRow: Record<string, unknown> = {};
        for (let i = 0; i < colNames.length; i++) {
          if (i < result.columns.length) {
            mappedRow[colNames[i]!] = row[result.columns[i]!] ?? null;
          }
        }
        sourceRows.push(mappedRow);
      }
    } else {
      // INSERT INTO ... VALUES ...
      for (const valueList of valuesLists) {
        const listNode = asObj(nodeGet(valueList, "List"));
        const items = asList(nodeGet(listNode, "items") ?? valueList);

        if (items.length !== colNames.length) {
          throw new Error(
            `VALUES has ${String(items.length)} columns but ${String(colNames.length)} were specified`,
          );
        }

        const row: Record<string, unknown> = {};
        for (let i = 0; i < colNames.length; i++) {
          row[colNames[i]!] = extractInsertValue(items[i]!, params, evaluator);
        }
        sourceRows.push(row);
      }
    }

    // ON CONFLICT handling
    const onConflictClause = nodeGet(stmt, "onConflictClause");
    let conflictCols: string[] = [];
    let conflictAction: string | number | null = null;
    let conflictTargetList: Record<string, unknown>[] = [];

    if (onConflictClause !== null && onConflictClause !== undefined) {
      const ocObj = asObj(onConflictClause);
      const infer = asObj(nodeGet(ocObj, "infer"));
      const indexElems = asList(nodeGet(infer, "indexElems"));
      conflictCols = indexElems.map((elem) => {
        const ie = asObj(nodeGet(elem, "IndexElem") ?? elem);
        return nodeStr(ie, "name");
      });
      conflictAction = nodeGet(ocObj, "action") as string | number;
      conflictTargetList = asList(nodeGet(ocObj, "targetList"));
    }

    // Build hash index for O(1) conflict lookups
    const conflictIndex = new Map<string, number>();
    if (conflictCols.length > 0) {
      for (const docId of table.documentStore.docIds) {
        const doc = table.documentStore.get(docId);
        if (doc !== null) {
          const key = conflictCols.map((c) => JSON.stringify(doc[c])).join("\0");
          conflictIndex.set(key, docId);
        }
      }
    }

    // RETURNING handling
    const returningList = asList(nodeGet(stmt, "returningList"));
    const hasReturning = returningList.length > 0;
    const returningRows: Record<string, unknown>[] = [];

    // Insert rows with ON CONFLICT and RETURNING support
    let inserted = 0;
    for (const srcRow of sourceRows) {
      if (conflictCols.length > 0 && onConflictClause !== null) {
        const key = conflictCols.map((c) => JSON.stringify(srcRow[c])).join("\0");
        const existingId = conflictIndex.get(key);
        if (existingId !== undefined) {
          // ONCONFLICT_NOTHING = 1, ONCONFLICT_UPDATE = 2
          if (conflictAction === 1 || conflictAction === "ONCONFLICT_NOTHING") {
            continue;
          }
          // DO UPDATE SET ...
          this._doConflictUpdate(
            table,
            existingId,
            srcRow,
            conflictTargetList,
            evaluator,
          );
          // Update conflict index if keys changed
          const updatedDoc = table.documentStore.get(existingId);
          if (updatedDoc !== null) {
            const newKey = conflictCols
              .map((c) => JSON.stringify(updatedDoc[c]))
              .join("\0");
            if (newKey !== key) {
              conflictIndex.delete(key);
              conflictIndex.set(newKey, existingId);
            }
          }
          if (hasReturning && updatedDoc !== null) {
            returningRows.push(
              this._evaluateReturning(returningList, updatedDoc, evaluator, table),
            );
          }
          inserted++;
          continue;
        }
      }

      // When ON CONFLICT DO NOTHING is used without explicit columns,
      // catch UNIQUE/PK violations at insert time and silently skip.
      let docId: number;
      if (
        onConflictClause !== null &&
        onConflictClause !== undefined &&
        conflictCols.length === 0 &&
        (conflictAction === 1 || conflictAction === "ONCONFLICT_NOTHING")
      ) {
        try {
          [docId] = table.insert(srcRow);
        } catch {
          continue;
        }
      } else {
        [docId] = table.insert(srcRow);
      }
      inserted++;

      // Update conflict index for subsequent rows in the batch
      if (conflictCols.length > 0) {
        const newKey = conflictCols.map((c) => JSON.stringify(srcRow[c])).join("\0");
        conflictIndex.set(newKey, docId);
      }

      if (hasReturning) {
        const doc = table.documentStore.get(docId);
        if (doc !== null) {
          returningRows.push(
            this._evaluateReturning(returningList, doc, evaluator, table),
          );
        }
      }
    }

    if (hasReturning) {
      const columns = this._extractReturningColumns(returningList, table);
      return { columns, rows: returningRows };
    }

    return {
      columns: ["inserted"],
      rows: [{ inserted }],
    };
  }

  private _doConflictUpdate(
    table: Table,
    docId: number,
    excludedRow: Record<string, unknown>,
    targetList: Record<string, unknown>[],
    evaluator: ExprEvaluator,
  ): void {
    const oldDoc = table.documentStore.get(docId);
    if (oldDoc === null) return;
    const newDoc = { ...oldDoc };

    // Merge excluded.* into the row for expression evaluation
    const evalRow: Record<string, unknown> = { ...oldDoc };
    for (const [k, v] of Object.entries(excludedRow)) {
      evalRow[`excluded.${k}`] = v;
    }

    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const colName = nodeStr(resTarget, "name");
      const valNode = nodeGet(resTarget, "val");
      if (valNode !== null && valNode !== undefined) {
        const newValue = evaluator.evaluate(asObj(valNode), evalRow);
        if (newValue !== null && newValue !== undefined) {
          newDoc[colName] = newValue;
        } else {
           
          Reflect.deleteProperty(newDoc, colName);
        }
      }
    }

    // Update document
    if (table.ftsFields.size > 0) {
      table.invertedIndex.removeDocument(docId);
    }
    table.documentStore.put(docId, newDoc);

    // Re-index FTS fields
    if (table.ftsFields.size > 0) {
      const textFields: Record<string, string> = {};
      for (const field of table.ftsFields) {
        const v = newDoc[field];
        if (typeof v === "string") textFields[field] = v;
      }
      if (Object.keys(textFields).length > 0) {
        table.invertedIndex.addDocument(docId, textFields);
      }
    }
  }

  // ==================================================================
  // DML: UPDATE
  // ==================================================================

  private _compileUpdate(
    stmt: Record<string, unknown>,
    params: unknown[],
  ): SQLResult | null {
    const relation = asObj(nodeGet(stmt, "relation"));
    const tableName = qualifiedName(relation);
    const table = this._tables.get(tableName);
    if (!table) {
      throw new Error(`Table "${tableName}" does not exist`);
    }

    const fromClause = asList(nodeGet(stmt, "fromClause"));

    // Multi-table UPDATE: UPDATE t1 SET ... FROM t2 WHERE ...
    if (fromClause.length > 0) {
      return this._compileUpdateFrom(stmt, table, params);
    }

    const whereClause = nodeGet(stmt, "whereClause");
    const targetList = asList(nodeGet(stmt, "targetList"));
    const returningList = asList(nodeGet(stmt, "returningList"));
    const hasReturning = returningList.length > 0;
    const returningRows: Record<string, unknown>[] = [];

    const evaluator = new ExprEvaluator({
      params,
      sequences: this._sequences,
      outerRow: this._correlatedOuterRow ?? undefined,
      subqueryExecutor: this._makeSubqueryExecutor(params),
    });

    // Parse SET clause
    const setTargets: [string, Record<string, unknown>][] = [];
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const colName = nodeStr(resTarget, "name");
      if (!table.columns.has(colName)) {
        throw new Error(`Unknown column "${colName}" for table "${tableName}"`);
      }
      setTargets.push([colName, asObj(nodeGet(resTarget, "val"))]);
    }

    // Find matching rows
    let updateCount = 0;
    for (const [docId, doc] of table.documentStore.iterAll()) {
      // Evaluate WHERE clause
      if (whereClause !== null && whereClause !== undefined) {
        const condition = evaluator.evaluate(asObj(whereClause), doc);
        if (condition !== true) continue;
      }

      // Apply SET assignments
      const updatedDoc = { ...doc };
      for (const [colName, valNode] of setTargets) {
        const newValue = evaluator.evaluate(valNode, doc);
        const colDef = table.columns.get(colName);
        if (newValue !== null && newValue !== undefined) {
          updatedDoc[colName] = newValue;
        } else if (colDef && colDef.notNull) {
          throw new Error(
            `NOT NULL constraint violated: column "${colName}" in table "${tableName}"`,
          );
        } else {
           
          Reflect.deleteProperty(updatedDoc, colName);
        }
      }

      // FK update validation
      for (const fkValidator of table.fkUpdateValidators) {
        fkValidator(doc, updatedDoc);
      }

      // Update indexes and document
      if (table.ftsFields.size > 0) {
        table.invertedIndex.removeDocument(docId);
      }
      table.documentStore.put(docId, updatedDoc);

      // Re-index FTS fields
      if (table.ftsFields.size > 0) {
        const textFields: Record<string, string> = {};
        for (const field of table.ftsFields) {
          const v = updatedDoc[field];
          if (typeof v === "string") textFields[field] = v;
        }
        if (Object.keys(textFields).length > 0) {
          table.invertedIndex.addDocument(docId, textFields);
        }
      }

      if (hasReturning) {
        returningRows.push(
          this._evaluateReturning(returningList, updatedDoc, evaluator, table),
        );
      }
      updateCount++;
    }

    if (hasReturning) {
      const columns = this._extractReturningColumns(returningList, table);
      return { columns, rows: returningRows };
    }

    return {
      columns: ["updated"],
      rows: [{ updated: updateCount }],
    };
  }

  private _compileUpdateFrom(
    stmt: Record<string, unknown>,
    table: Table,
    params: unknown[],
  ): SQLResult | null {
    const tableName = table.name;
    const relation = asObj(nodeGet(stmt, "relation"));
    const tableAlias = extractAlias(relation) ?? tableName;

    const fromClause = asList(nodeGet(stmt, "fromClause"));
    const whereClause = nodeGet(stmt, "whereClause");
    const targetList = asList(nodeGet(stmt, "targetList"));
    const returningList = asList(nodeGet(stmt, "returningList"));
    const hasReturning = returningList.length > 0;
    const returningRows: Record<string, unknown>[] = [];

    const evaluator = new ExprEvaluator({
      params,
      sequences: this._sequences,
      outerRow: this._correlatedOuterRow ?? undefined,
      subqueryExecutor: this._makeSubqueryExecutor(params),
    });

    // Parse SET clause
    const setTargets: [string, Record<string, unknown>][] = [];
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const colName = nodeStr(resTarget, "name");
      if (!table.columns.has(colName)) {
        throw new Error(`Unknown column "${colName}" for table "${tableName}"`);
      }
      setTargets.push([colName, asObj(nodeGet(resTarget, "val"))]);
    }

    // Resolve FROM tables
    const fromRows = this._resolveFrom(fromClause, params);

    let updated = 0;
    for (const [docId, doc] of table.documentStore.iterAll()) {
      const targetRow: Record<string, unknown> = { ...doc };
      for (const [k, v] of Object.entries(doc)) {
        targetRow[`${tableAlias}.${k}`] = v;
      }

      for (const fromRow of fromRows) {
        const merged = { ...targetRow, ...fromRow };

        if (
          whereClause !== null &&
          whereClause !== undefined &&
          !evaluator.evaluate(asObj(whereClause), merged)
        ) {
          continue;
        }

        // Apply SET expressions
        const newDoc = { ...doc };
        for (const [colName, valNode] of setTargets) {
          const newValue = evaluator.evaluate(valNode, merged);
          const colDef = table.columns.get(colName);
          if (newValue !== null && newValue !== undefined) {
            newDoc[colName] = newValue;
          } else if (colDef && colDef.notNull) {
            throw new Error(
              `NOT NULL constraint violated: column "${colName}" in table "${tableName}"`,
            );
          } else {
            Reflect.deleteProperty(newDoc, colName);
          }
        }

        if (table.ftsFields.size > 0) {
          table.invertedIndex.removeDocument(docId);
        }
        table.documentStore.put(docId, newDoc);

        if (table.ftsFields.size > 0) {
          const textFields: Record<string, string> = {};
          for (const field of table.ftsFields) {
            const v = newDoc[field];
            if (typeof v === "string") textFields[field] = v;
          }
          if (Object.keys(textFields).length > 0) {
            table.invertedIndex.addDocument(docId, textFields);
          }
        }

        if (hasReturning) {
          returningRows.push(
            this._evaluateReturning(returningList, newDoc, evaluator, table),
          );
        }
        updated++;
        break; // Only update once per target row
      }
    }

    if (hasReturning) {
      const columns = this._extractReturningColumns(returningList, table);
      return { columns, rows: returningRows };
    }

    return { columns: ["updated"], rows: [{ updated }] };
  }

  // ==================================================================
  // DML: DELETE
  // ==================================================================

  private _compileDelete(
    stmt: Record<string, unknown>,
    params: unknown[],
  ): SQLResult | null {
    const relation = asObj(nodeGet(stmt, "relation"));
    const tableName = qualifiedName(relation);
    const table = this._tables.get(tableName);
    if (!table) {
      throw new Error(`Table "${tableName}" does not exist`);
    }

    const usingClause = asList(nodeGet(stmt, "usingClause"));

    // Multi-table DELETE: DELETE FROM t1 USING t2 WHERE ...
    if (usingClause.length > 0) {
      return this._compileDeleteUsing(stmt, table, params);
    }

    const whereClause = nodeGet(stmt, "whereClause");
    const returningList = asList(nodeGet(stmt, "returningList"));
    const hasReturning = returningList.length > 0;
    const returningRows: Record<string, unknown>[] = [];

    const evaluator = new ExprEvaluator({
      params,
      outerRow: this._correlatedOuterRow ?? undefined,
      subqueryExecutor: this._makeSubqueryExecutor(params),
    });

    // Collect IDs to delete
    const toDelete: number[] = [];
    for (const [docId, doc] of table.documentStore.iterAll()) {
      if (whereClause !== null && whereClause !== undefined) {
        const condition = evaluator.evaluate(asObj(whereClause), doc);
        if (condition !== true) continue;
      }

      if (hasReturning) {
        returningRows.push(
          this._evaluateReturning(returningList, doc, evaluator, table),
        );
      }

      toDelete.push(docId);
    }

    // FK delete validation: check if any row to be deleted is referenced
    for (const docId of toDelete) {
      for (const fkValidator of table.fkDeleteValidators) {
        fkValidator(docId);
      }
    }

    for (const docId of toDelete) {
      if (table.ftsFields.size > 0) {
        table.invertedIndex.removeDocument(docId);
      }
      table.documentStore.delete(docId);
    }

    if (hasReturning) {
      const columns = this._extractReturningColumns(returningList, table);
      return { columns, rows: returningRows };
    }

    return {
      columns: ["deleted"],
      rows: [{ deleted: toDelete.length }],
    };
  }

  private _compileDeleteUsing(
    stmt: Record<string, unknown>,
    table: Table,
    params: unknown[],
  ): SQLResult | null {
    const tableName = table.name;
    const relation = asObj(nodeGet(stmt, "relation"));
    const tableAlias = extractAlias(relation) ?? tableName;
    const usingClause = asList(nodeGet(stmt, "usingClause"));
    const whereClause = nodeGet(stmt, "whereClause");
    const returningList = asList(nodeGet(stmt, "returningList"));
    const hasReturning = returningList.length > 0;
    const returningRows: Record<string, unknown>[] = [];

    const evaluator = new ExprEvaluator({
      params,
      outerRow: this._correlatedOuterRow ?? undefined,
      subqueryExecutor: this._makeSubqueryExecutor(params),
    });

    // Resolve USING tables
    const usingRows = this._resolveFrom(usingClause, params);

    const toDelete: number[] = [];
    for (const [docId, doc] of table.documentStore.iterAll()) {
      const targetRow: Record<string, unknown> = { ...doc };
      for (const [k, v] of Object.entries(doc)) {
        targetRow[`${tableAlias}.${k}`] = v;
      }

      for (const usingRow of usingRows) {
        const merged = { ...targetRow, ...usingRow };

        if (
          whereClause !== null &&
          whereClause !== undefined &&
          !evaluator.evaluate(asObj(whereClause), merged)
        ) {
          continue;
        }

        if (hasReturning) {
          returningRows.push(
            this._evaluateReturning(returningList, doc, evaluator, table),
          );
        }
        toDelete.push(docId);
        break; // Only delete once per row
      }
    }

    for (const docId of toDelete) {
      if (table.ftsFields.size > 0) {
        table.invertedIndex.removeDocument(docId);
      }
      table.documentStore.delete(docId);
    }

    if (hasReturning) {
      const columns = this._extractReturningColumns(returningList, table);
      return { columns, rows: returningRows };
    }

    return { columns: ["deleted"], rows: [{ deleted: toDelete.length }] };
  }

  // ==================================================================
  // Transaction: BEGIN / COMMIT / ROLLBACK / SAVEPOINT
  // ==================================================================

  private _compileTransaction(stmt: Record<string, unknown>): SQLResult | null {
    const kind = nodeGet(stmt, "kind") as number | string;
    const eng = this._engine as Record<string, unknown> | null;
    // 0 = BEGIN, 2 = COMMIT, 3 = ROLLBACK
    // 4 = SAVEPOINT, 5 = RELEASE SAVEPOINT, 6 = ROLLBACK TO SAVEPOINT
    if (kind === 0 || kind === "TRANS_STMT_BEGIN" || kind === "TRANS_STMT_START") {
      if (eng && typeof (eng as { begin?: unknown }).begin === "function") {
        (eng as { begin: () => unknown }).begin();
      }
      this._transactionActive = true;
    } else if (kind === 2 || kind === "TRANS_STMT_COMMIT") {
      if (eng) {
        const txn = (eng as { _transaction?: { active: boolean; commit: () => void } })
          ._transaction;
        if (txn && txn.active) txn.commit();
      }
      this._transactionActive = false;
    } else if (kind === 3 || kind === "TRANS_STMT_ROLLBACK") {
      if (eng) {
        const txn = (eng as { _transaction?: { active: boolean; rollback: () => void } })
          ._transaction;
        if (txn && txn.active) txn.rollback();
      }
      this._transactionActive = false;
    } else if (kind === 4 || kind === "TRANS_STMT_SAVEPOINT") {
      const name = nodeStr(asObj(nodeGet(stmt, "options") ?? {}), "str") || "sp";
      if (eng) {
        const txn = (eng as { _transaction?: { active: boolean; savepoint: (n: string) => void } })
          ._transaction;
        if (txn && txn.active) txn.savepoint(name);
      }
    } else if (kind === 5 || kind === "TRANS_STMT_RELEASE") {
      const name = nodeStr(asObj(nodeGet(stmt, "options") ?? {}), "str") || "sp";
      if (eng) {
        const txn = (eng as { _transaction?: { active: boolean; releaseSavepoint: (n: string) => void } })
          ._transaction;
        if (txn && txn.active) txn.releaseSavepoint(name);
      }
    } else if (kind === 6 || kind === "TRANS_STMT_ROLLBACK_TO") {
      const name = nodeStr(asObj(nodeGet(stmt, "options") ?? {}), "str") || "sp";
      if (eng) {
        const txn = (eng as { _transaction?: { active: boolean; rollbackTo: (n: string) => void } })
          ._transaction;
        if (txn && txn.active) txn.rollbackTo(name);
      }
    }
    return null;
  }

  // ==================================================================
  // DDL: CREATE SCHEMA / DROP SCHEMA
  // ==================================================================

  private _compileCreateSchema(stmt: Record<string, unknown>): SQLResult | null {
    const schemaName = nodeStr(stmt, "schemaname");
    const ifNotExists = nodeGet(stmt, "if_not_exists") === true;
    const eng = this._engine as { _tables: { createSchema: (n: string, i: boolean) => void } } | null;
    if (eng) {
      eng._tables.createSchema(schemaName, ifNotExists);
    }
    return null;
  }

  private _compileDropSchema(stmt: Record<string, unknown>): SQLResult | null {
    const objects = asList(nodeGet(stmt, "objects"));
    const cascade = nodeGet(stmt, "behavior") === "DROP_CASCADE" ||
      nodeGet(stmt, "behavior") === 1;
    const ifExists = nodeGet(stmt, "missing_ok") === true;
    const eng = this._engine as {
      _tables: { dropSchema: (n: string, c: boolean, i: boolean) => void };
    } | null;
    if (eng) {
      for (const obj of objects) {
        const items = asList(obj);
        const schemaName = items.length > 0
          ? extractString(items[items.length - 1]!)
          : extractString(obj);
        eng._tables.dropSchema(schemaName, cascade, ifExists);
      }
    }
    return null;
  }

  // ==================================================================
  // Session Variables: SET / SHOW / RESET / DISCARD
  // ==================================================================

  private static readonly _SESSION_DEFAULTS: Record<string, string> = {
    server_version: "17.0",
    server_encoding: "UTF8",
    client_encoding: "UTF8",
    client_min_messages: "notice",
    default_transaction_isolation: "read committed",
    default_transaction_read_only: "off",
    is_superuser: "on",
    session_authorization: "default",
    standard_conforming_strings: "on",
    timezone: "UTC",
    datestyle: "ISO, MDY",
    intervalstyle: "postgres",
    integer_datetimes: "on",
    lc_collate: "en_US.UTF-8",
    lc_ctype: "en_US.UTF-8",
    search_path: '"$user", public',
    statement_timeout: "0",
    lock_timeout: "0",
    idle_in_transaction_session_timeout: "0",
    max_connections: "100",
    shared_buffers: "128MB",
    work_mem: "4MB",
    maintenance_work_mem: "64MB",
    transaction_isolation: "read committed",
    transaction_read_only: "off",
  };

  private _compileSet(stmt: Record<string, unknown>): SQLResult | null {
    const kind = nodeGet(stmt, "kind") as string | number;
    const eng = this._engine as {
      _sessionVars: Map<string, string>;
      _tables: { searchPath: string[] };
    } | null;
    if (!eng) return null;

    const name = nodeStr(stmt, "name");
    if (
      kind === "VAR_SET_VALUE" || kind === 0
    ) {
      const argNodes = asList(nodeGet(stmt, "args"));
      const values: string[] = [];
      for (const arg of argNodes) {
        values.push(String(extractConstValue(arg, this._params)));
      }
      const valueStr = values.join(", ");
      eng._sessionVars.set(name, valueStr);
      if (name === "search_path") {
        eng._tables.searchPath = values
          .map((v) => v.trim().replace(/^['"]|['"]$/g, ""))
          .filter((v) => v.length > 0);
      }
    } else if (
      kind === "VAR_SET_DEFAULT" || kind === 1
    ) {
      eng._sessionVars.delete(name);
      if (name === "search_path") {
        eng._tables.searchPath = ["public"];
      }
    } else if (
      kind === "VAR_RESET" || kind === 2
    ) {
      eng._sessionVars.delete(name);
      if (name === "search_path") {
        eng._tables.searchPath = ["public"];
      }
    } else if (
      kind === "VAR_RESET_ALL" || kind === 3
    ) {
      eng._sessionVars.clear();
      eng._tables.searchPath = ["public"];
    }
    return null;
  }

  private _compileShow(stmt: Record<string, unknown>): SQLResult {
    const name = nodeStr(stmt, "name");
    const eng = this._engine as {
      _sessionVars: Map<string, string>;
    } | null;
    const value = eng?._sessionVars.get(name)
      ?? SQLCompiler._SESSION_DEFAULTS[name]
      ?? "";
    return { columns: [name], rows: [{ [name]: value }] };
  }

  private _compileDiscard(): SQLResult | null {
    const eng = this._engine as {
      _sessionVars: Map<string, string>;
      _prepared: Map<string, unknown>;
      _tempTables: Set<string>;
    } | null;
    if (eng) {
      eng._sessionVars.clear();
      eng._prepared.clear();
      eng._tempTables.clear();
    }
    return null;
  }

  // ==================================================================
  // Prepared Statements: PREPARE / EXECUTE / DEALLOCATE
  // ==================================================================

  private _compilePrepare(stmt: Record<string, unknown>): SQLResult | null {
    const name = nodeStr(stmt, "name");
    if (this._prepared.has(name)) {
      throw new Error(`Prepared statement "${name}" already exists`);
    }
    const query = asObj(nodeGet(stmt, "query"));
    this._prepared.set(name, query);
    return null;
  }

  private _compileExecute(
    stmt: Record<string, unknown>,
    params: unknown[],
  ): SQLResult | null {
    const name = nodeStr(stmt, "name");
    const prep = this._prepared.get(name);
    if (prep === undefined) {
      throw new Error(`Prepared statement "${name}" does not exist`);
    }

    // Collect parameter values from EXECUTE
    const execParams = asList(nodeGet(stmt, "params"));
    const evaluator = new ExprEvaluator({ params });
    const resolvedParams: unknown[] = [];
    for (const p of execParams) {
      resolvedParams.push(evaluator.evaluate(p, {}));
    }

    // Temporarily set instance-level params so WHERE clause compilation
    // (which uses this._params) can resolve $N placeholders.
    const savedParams = this._params;
    this._params = resolvedParams;
    try {
      return this._dispatchStatement(prep, resolvedParams);
    } finally {
      this._params = savedParams;
    }
  }

  private _compileDeallocate(stmt: Record<string, unknown>): SQLResult | null {
    const name = nodeStr(stmt, "name");
    if (!name) {
      // DEALLOCATE ALL
      this._prepared.clear();
    } else {
      if (!this._prepared.has(name)) {
        throw new Error(`Prepared statement "${name}" does not exist`);
      }
      this._prepared.delete(name);
    }
    return null;
  }

  // ==================================================================
  // EXPLAIN / ANALYZE
  // ==================================================================

  private _compileExplain(stmt: Record<string, unknown>, params: unknown[]): SQLResult {
    const query = asObj(nodeGet(stmt, "query"));
    const selectStmt = asObj(nodeGet(query, "SelectStmt") ?? query);
    const fromClause = asList(nodeGet(selectStmt, "fromClause"));
    const whereClause = nodeGet(selectStmt, "whereClause");

    void params;

    // Detect UQA function in WHERE and resolve the target table
    let uqaTarget: string | null = null;
    let uqaFuncName: string | null = null;
    if (whereClause !== null && whereClause !== undefined) {
      const whereObj = asObj(whereClause);
      if (SQLCompiler._containsUQAFunction(whereObj)) {
        uqaFuncName = SQLCompiler._extractTopUQAFuncName(whereObj);
        const singleTable = this._resolveFromTableName(fromClause);
        if (singleTable) {
          uqaTarget = singleTable;
        } else if (fromClause.length > 0) {
          const aliasMap = this._collectFromAliases(fromClause);
          const colAlias = SQLCompiler._extractUQAColumnAlias(whereObj);
          if (colAlias && aliasMap.has(colAlias)) {
            uqaTarget = aliasMap.get(colAlias)!;
          }
        }
      }
    }

    // Collect all referenced tables from the FROM clause
    const allTables: { name: string; alias: string | null }[] = [];
    if (fromClause.length > 0) {
      const aliasMap = this._collectFromAliases(fromClause);
      const seen = new Set<string>();
      for (const [alias, name] of aliasMap) {
        if (seen.has(name)) continue;
        seen.add(name);
        allTables.push({ name, alias: alias !== name ? alias : null });
      }
    }

    const isJoin = fromClause.length > 0 && this._resolveFromTableName(fromClause) === null;
    const planLines: string[] = [];

    if (isJoin) {
      // Determine join strategy from join type and ON condition
      let joinLabel = "Nested Loop";
      if (fromClause.length === 1) {
        const je = nodeGet(fromClause[0]!, "JoinExpr");
        if (je !== null && je !== undefined) {
          const jeObj = asObj(je);
          const jt = nodeGet(jeObj, "jointype");
          const onQuals = nodeGet(jeObj, "quals");
          const isEqui = SQLCompiler._extractEquiJoinKeys(onQuals) !== null;
          if (jt === 5 || jt === "JOIN_CROSS") {
            joinLabel = "Nested Loop";
          } else if (isEqui) {
            if (jt === 1 || jt === "JOIN_LEFT") joinLabel = "Hash Left Join";
            else if (jt === 3 || jt === "JOIN_RIGHT") joinLabel = "Hash Right Join";
            else if (jt === 2 || jt === "JOIN_FULL") joinLabel = "Hash Full Join";
            else joinLabel = "Hash Join";
          }
        }
      }
      planLines.push(joinLabel);
      for (const { name, alias } of allTables) {
        const table = this._tables.get(name);
        const rc = table ? String(table.rowCount) : "?";
        const label = alias ? `"${name}" ${alias}` : `"${name}"`;
        if (name === uqaTarget && uqaFuncName) {
          planLines.push(`  -> GIN Index Scan using ${uqaFuncName} on ${label} (${rc} rows)`);
        } else {
          planLines.push(`  -> Seq Scan on ${label} (${rc} rows)`);
        }
      }
    } else if (allTables.length > 0) {
      const { name } = allTables[0]!;
      const table = this._tables.get(name);
      const rc = table ? String(table.rowCount) : "?";
      if (uqaTarget && uqaFuncName) {
        planLines.push(`GIN Index Scan using ${uqaFuncName} on "${name}" (${rc} rows)`);
      } else {
        planLines.push(`Seq Scan on table "${name}" (${rc} rows)`);
      }
    } else {
      planLines.push("Result");
    }

    return {
      columns: ["QUERY PLAN"],
      rows: planLines.map((line) => ({ "QUERY PLAN": line })),
    };
  }

  /**
   * Extract the outermost UQA function name from a WHERE clause AST.
   */
  private static _extractTopUQAFuncName(node: Record<string, unknown>): string | null {
    if (isFuncCall(node)) {
      const name = getFuncName(node);
      if (UQA_WHERE_FUNCTIONS.has(name)) return name;
    }
    const fc = nodeGet(node, "FuncCall");
    if (fc !== null && fc !== undefined) {
      const name = getFuncName(asObj(fc));
      if (UQA_WHERE_FUNCTIONS.has(name)) return name;
    }
    const boolExpr = asObj(nodeGet(node, "BoolExpr") ?? {});
    if (Object.keys(boolExpr).length > 0) {
      const boolArgs = asList(nodeGet(boolExpr, "args"));
      for (const ba of boolArgs) {
        const result = SQLCompiler._extractTopUQAFuncName(ba);
        if (result) return result;
      }
    }
    return null;
  }

  private _compileAnalyze(stmt: Record<string, unknown>): SQLResult | null {
    // ANALYZE collects column statistics (histogram, MCV) for the query optimizer.
    const rels = stmt["rels"] as unknown[] | undefined;
    if (rels && rels.length > 0) {
      for (const rel of rels) {
        const relObj = rel as Record<string, unknown>;
        // libpg-query wraps each rel in VacuumRelation
        const vacuumRel = (relObj["VacuumRelation"] ?? relObj) as Record<
          string,
          unknown
        >;
        const relation = vacuumRel["relation"] as Record<string, unknown> | undefined;
        const relname = relation ? (relation["relname"] as string | null) : null;
        if (relname && this._tables.has(relname)) {
          this._tables.get(relname)!.analyze();
        }
      }
    } else {
      // ANALYZE without table name -> analyze all tables
      for (const table of this._tables.values()) {
        table.analyze();
      }
    }
    return null;
  }

  // ==================================================================
  // DQL: SELECT
  // ==================================================================

  private _compileSelect(
    stmt: Record<string, unknown>,
    params: unknown[],
    outerRow?: Record<string, unknown>,
  ): SQLResult {
    // 0. Materialize CTEs as temporary in-memory tables
    const withClause = nodeGet(stmt, "withClause");
    let cteNames: string[] = [];
    if (withClause !== null && withClause !== undefined) {
      const wc = asObj(withClause);
      const ctes = asList(nodeGet(wc, "ctes"));
      const recursive = nodeGet(wc, "recursive") === true;
      cteNames = this._materializeCTEs(ctes, params, recursive, stmt);
    }

    // Save and set correlated outer row context
    const prevOuterRow = this._correlatedOuterRow;
    this._correlatedOuterRow = outerRow ?? null;

    const prevInlined = new Map(this._inlinedCTEs);

    try {
      return this._compileSelectBody(stmt, params);
    } finally {
      this._correlatedOuterRow = prevOuterRow;

      // Clean up inlined CTEs
      for (const name of cteNames) {
        this._inlinedCTEs.delete(name);
      }
      this._inlinedCTEs = prevInlined;

      // Clean up CTE temporary tables
      for (const name of cteNames) {
        this._tables.delete(name);
      }

      // Clean up materialized view / derived tables
      for (const name of this._expandedViews) {
        if (this._shadowedTables.has(name)) {
          this._tables.set(name, this._shadowedTables.get(name)!);
          this._shadowedTables.delete(name);
        } else {
          this._tables.delete(name);
        }
      }
      this._expandedViews = [];
    }
  }

  private _compileSelectBody(
    stmt: Record<string, unknown>,
    params: unknown[],
  ): SQLResult {
    // Handle standalone VALUES
    const valuesLists = asList(nodeGet(stmt, "valuesLists"));
    if (valuesLists.length > 0) {
      return this._compileValues(valuesLists, params);
    }

    // Handle set operations (UNION / INTERSECT / EXCEPT)
    const op = nodeGet(stmt, "op");
    if (op !== null && op !== undefined && op !== 0 && op !== "SETOP_NONE") {
      return this._compileSetOp(stmt, params);
    }

    // Predicate pushdown: push safe WHERE predicates into views/derived tables
    stmt = this._tryPredicatePushdown(stmt, params);

    // Standard SELECT
    const fromClause = asList(nodeGet(stmt, "fromClause"));
    const whereClause = nodeGet(stmt, "whereClause");
    const targetList = asList(nodeGet(stmt, "targetList"));
    const groupClause = asList(nodeGet(stmt, "groupClause"));
    const havingClause = nodeGet(stmt, "havingClause");
    const sortClause = asList(nodeGet(stmt, "sortClause"));
    const limitCount = nodeGet(stmt, "limitCount");
    const limitOffset = nodeGet(stmt, "limitOffset");
    const distinctClause = nodeGet(stmt, "distinctClause");
    const windowClause = asList(nodeGet(stmt, "windowClause"));

    // Resolve analyzer for uqa_highlight() support
    let hlAnalyzer: { analyze(text: string): string[] } | null = null;
    {
      const tableName = this._resolveFromTableName(fromClause);
      if (tableName) {
        const table = this._tables.get(tableName);
        if (table && table.invertedIndex) {
          hlAnalyzer = (table.invertedIndex as unknown as { analyzer: { analyze(t: string): string[] } }).analyzer ?? null;
        }
      }
    }

    const evaluator = new ExprEvaluator({
      params,
      sequences: this._sequences,
      outerRow: this._correlatedOuterRow ?? undefined,
      subqueryExecutor: this._makeSubqueryExecutor(params),
      analyzer: hlAnalyzer,
    });

    // 1. FROM clause -- get source rows
    //    When UQA functions appear in WHERE with a JOIN, push the posting-list
    //    scan below the join so only matching rows enter the join.
    let rows: Record<string, unknown>[];
    let uqaApplied = false;
    let uqaScalarNode: Record<string, unknown> | null = null;

    if (fromClause.length === 0) {
      // SELECT without FROM (e.g. SELECT 1+1)
      rows = [{}];
    } else if (
      whereClause !== null &&
      whereClause !== undefined &&
      !this._resolveFromTableName(fromClause)
    ) {
      // Multi-table FROM (JOIN): try to push UQA scan below the join
      const whereObj = asObj(whereClause);
      if (SQLCompiler._containsUQAFunction(whereObj)) {
        const aliasMap = this._collectFromAliases(fromClause);
        const colAlias = SQLCompiler._extractUQAColumnAlias(whereObj);
        const targetName = colAlias ? aliasMap.get(colAlias) : undefined;
        const table = targetName ? this._tables.get(targetName) : undefined;
        if (table && targetName) {
          const ctx = this._contextForTable(table);
          const [uqaNode, scalarNode] = this._splitUQAConjuncts(whereObj);
          let op: Operator | null = null;
          if (uqaNode) {
            op = this._compileWhere(uqaNode, ctx);
          }
          if (op) {
            const pl = op.execute(ctx);
            const scores = new Map<unknown, number>();
            for (const entry of pl) {
              scores.set(entry.docId, entry.payload.score);
            }
            // Activate filter so _resolveFromItem only emits matching docs
            const savedFilter = this._uqaFromFilter;
            this._uqaFromFilter = {
              tableName: targetName,
              scores,
              pkCol: table.primaryKey,
            };
            rows = this._resolveFrom(fromClause, params);
            this._uqaFromFilter = savedFilter;
            uqaApplied = true;
            uqaScalarNode = scalarNode;
          } else {
            rows = this._resolveFrom(fromClause, params);
          }
        } else {
          rows = this._resolveFrom(fromClause, params);
        }
      } else {
        rows = this._resolveFrom(fromClause, params);
      }
    } else {
      rows = this._resolveFrom(fromClause, params);
    }

    // 2. WHERE clause -- filter rows
    if (uqaApplied) {
      // UQA already applied during FROM (JOIN pushdown); only scalar remains
      if (uqaScalarNode) {
        rows = rows.filter((row) => {
          const result = evaluator.evaluate(asObj(uqaScalarNode!), row);
          return result === true;
        });
      }
    } else if (whereClause !== null && whereClause !== undefined) {
      const whereObj = asObj(whereClause);
      if (SQLCompiler._containsUQAFunction(whereObj)) {
        // UQA posting-list path (single-table)
        const tableName = this._resolveFromTableName(fromClause);
        if (tableName) {
          const table = this._tables.get(tableName);
          if (table) {
            const ctx = this._contextForTable(table);
            const [uqaNode, scalarNode] = this._splitUQAConjuncts(whereObj);
            let op: Operator | null = null;
            if (uqaNode) {
              op = this._compileWhere(uqaNode, ctx);
            }
            if (op) {
              const pl = op.execute(ctx);
              const newRows: Record<string, unknown>[] = [];
              const ds = table.documentStore;
              const pkCol = table.primaryKey;
              for (const entry of pl) {
                const doc = ds.get(entry.docId);
                if (!doc) continue;
                const row: Record<string, unknown> = { ...doc, _score: entry.payload.score };
                if (pkCol) row[pkCol] = entry.docId;
                row["_doc_id"] = entry.docId;
                newRows.push(row);
              }
              rows = newRows;
              if (scalarNode) {
                rows = rows.filter((row) => {
                  const result = evaluator.evaluate(asObj(scalarNode), row);
                  return result === true;
                });
              }
            }
          } else {
            rows = rows.filter((row) => {
              const result = evaluator.evaluate(whereObj, row);
              return result === true;
            });
          }
        } else {
          rows = rows.filter((row) => {
            const result = evaluator.evaluate(whereObj, row);
            return result === true;
          });
        }
      } else {
        rows = rows.filter((row) => {
          const result = evaluator.evaluate(whereObj, row);
          return result === true;
        });
      }
    }

    // 2b. Intercept uqa_facets(): return facet counts over filtered rows
    const facetResult = this._tryFacets(targetList, rows);
    if (facetResult !== null) {
      return facetResult;
    }

    // 3. GROUP BY clause
    const hasAggregates = this._hasAggregates(targetList);
    if (groupClause.length > 0) {
      rows = this._applyGroupBy(rows, groupClause, targetList, havingClause, evaluator);
    } else if (hasAggregates) {
      // Aggregate without GROUP BY -- treat entire result as one group
      rows = this._applyGroupBy(rows, [], targetList, havingClause, evaluator);
    }

    // 4. Window functions
    const hasWindow = this._hasWindowFunctions(targetList);
    if (hasWindow) {
      rows = this._applyWindowFunctions(rows, targetList, windowClause, evaluator);
    }

    // 5. SELECT list -- project columns
    let columns: string[];
    let projectedRows: Record<string, unknown>[];
    if (groupClause.length > 0 || hasAggregates || hasWindow) {
      // Rows are already projected by group/window logic
      columns = this._resolveSelectColumnNames(targetList, rows);
      projectedRows = rows;
    } else {
      [columns, projectedRows] = this._projectColumns(targetList, rows, evaluator);
    }

    // 6. DISTINCT
    let resultRows = projectedRows;
    if (distinctClause !== null && distinctClause !== undefined) {
      resultRows = this._applyDistinct(resultRows, columns);
    }

    // 7. ORDER BY
    if (sortClause.length > 0) {
      resultRows = this._applyOrderBy(resultRows, sortClause, evaluator, targetList);
    }

    // 8. LIMIT and OFFSET
    if (limitOffset !== null && limitOffset !== undefined) {
      const offset = Number(evaluator.evaluate(asObj(limitOffset), {}));
      resultRows = resultRows.slice(offset);
    }
    if (limitCount !== null && limitCount !== undefined) {
      const limit = Number(evaluator.evaluate(asObj(limitCount), {}));
      resultRows = resultRows.slice(0, limit);
    }

    return { columns, rows: resultRows };
  }

  // -- CTE materialization ------------------------------------------------

  private _materializeCTEs(
    ctes: Record<string, unknown>[],
    params: unknown[],
    recursive: boolean,
    mainQuery: Record<string, unknown>,
  ): string[] {
    const cteNames: string[] = [];

    for (const cte of ctes) {
      const cteObj = asObj(nodeGet(cte, "CommonTableExpr") ?? cte);
      const name = nodeStr(cteObj, "ctename");
      const cteQuery = asObj(nodeGet(cteObj, "ctequery"));
      const selectStmt = asObj(nodeGet(cteQuery, "SelectStmt") ?? cteQuery);

      const cteOp = nodeGet(selectStmt, "op");
      const isRecursive =
        recursive &&
        cteOp !== null &&
        cteOp !== undefined &&
        cteOp !== 0 &&
        cteOp !== "SETOP_NONE";

      if (isRecursive) {
        this._materializeRecursiveCTE(cteObj, selectStmt, params);
      } else {
        // Count references in main query to determine inline vs materialize
        const refCount = this._countCTERefs(name, mainQuery);
        if (refCount === 1) {
          // Inline single-reference CTEs
          this._inlinedCTEs.set(name, selectStmt);
        } else {
          const result = this._compileSelect(selectStmt, params);
          const table = this._resultToTable(name, result);
          this._tables.set(name, table);
        }
      }
      cteNames.push(name);
    }

    return cteNames;
  }

  private _materializeRecursiveCTE(
    cteObj: Record<string, unknown>,
    selectStmt: Record<string, unknown>,
    params: unknown[],
  ): void {
    const name = nodeStr(cteObj, "ctename");
    const isUnionAll = nodeGet(selectStmt, "all") === true;

    // Column name mapping from alias
    const aliasColNamesRaw = asList(nodeGet(cteObj, "aliascolnames"));
    const aliasCols =
      aliasColNamesRaw.length > 0
        ? aliasColNamesRaw.map((n) => extractString(n))
        : null;

    // 1. Execute base case (larg)
    const larg = asObj(nodeGet(selectStmt, "larg"));
    const lSelectStmt = asObj(nodeGet(larg, "SelectStmt") ?? larg);
    const baseResult = this._compileSelect(lSelectStmt, params);
    const baseColumns = baseResult.columns;

    // Remap columns if alias names are provided
    let columns: string[];
    let allRows: Record<string, unknown>[];
    if (aliasCols !== null) {
      allRows = [];
      for (const row of baseResult.rows) {
        const remapped: Record<string, unknown> = {};
        for (let i = 0; i < aliasCols.length; i++) {
          if (i < baseColumns.length) {
            remapped[aliasCols[i]!] = row[baseColumns[i]!] ?? null;
          }
        }
        allRows.push(remapped);
      }
      columns = aliasCols;
    } else {
      allRows = [...baseResult.rows];
      columns = baseColumns;
    }

    // Track seen tuples for UNION deduplication
    let seen: Set<string> | null = null;
    if (!isUnionAll) {
      seen = new Set<string>();
      const deduped: Record<string, unknown>[] = [];
      for (const row of allRows) {
        const key = columns.map((c) => JSON.stringify(row[c])).join("\0");
        if (!seen.has(key)) {
          seen.add(key);
          deduped.push(row);
        }
      }
      allRows = deduped;
    }

    // Register working table
    let workingRows = [...allRows];

    for (let depth = 0; depth < MAX_RECURSIVE_DEPTH; depth++) {
      // Build temporary table from working rows
      const result: SQLResult = { columns, rows: workingRows };
      const table = this._resultToTable(name, result);
      this._tables.set(name, table);

      // Execute recursive case (rarg)
      const rarg = asObj(nodeGet(selectStmt, "rarg"));
      const rSelectStmt = asObj(nodeGet(rarg, "SelectStmt") ?? rarg);
      const recResult = this._compileSelect(rSelectStmt, params);

      // Remap recursive result columns
      const targetCols = aliasCols ?? columns;
      const newRows: Record<string, unknown>[] = [];
      for (const row of recResult.rows) {
        const remapped: Record<string, unknown> = {};
        for (let i = 0; i < targetCols.length; i++) {
          if (i < recResult.columns.length) {
            remapped[targetCols[i]!] = row[recResult.columns[i]!] ?? null;
          }
        }
        newRows.push(remapped);
      }

      if (newRows.length === 0) break;

      // Deduplicate for UNION (not ALL)
      let filteredRows = newRows;
      if (seen !== null) {
        filteredRows = [];
        for (const row of newRows) {
          const key = columns.map((c) => JSON.stringify(row[c])).join("\0");
          if (!seen.has(key)) {
            seen.add(key);
            filteredRows.push(row);
          }
        }
        if (filteredRows.length === 0) break;
      }

      allRows.push(...filteredRows);
      workingRows = filteredRows;
    }

    // Final table with all accumulated rows
    const finalResult: SQLResult = { columns, rows: allRows };
    const finalTable = this._resultToTable(name, finalResult);
    this._tables.set(name, finalTable);
  }

  private _countCTERefs(name: string, node: unknown): number {
    if (node === null || node === undefined) return 0;
    if (typeof node !== "object") return 0;

    if (Array.isArray(node)) {
      let count = 0;
      for (const item of node) {
        count += this._countCTERefs(name, item);
      }
      return count;
    }

    const obj = node as Record<string, unknown>;
    // Check RangeVar
    const rv = nodeGet(obj, "RangeVar");
    if (rv !== null && rv !== undefined) {
      const relname = nodeStr(asObj(rv), "relname");
      if (relname === name) return 1;
    }
    // Check direct relname
    if (nodeStr(obj, "relname") === name) return 1;

    let count = 0;
    for (const [key, value] of Object.entries(obj)) {
      if (key === "withClause") continue; // Don't count refs in CTE definitions
      if (value !== null && typeof value === "object") {
        count += this._countCTERefs(name, value);
      }
    }
    return count;
  }

  private _resultToTable(name: string, result: SQLResult): Table {
    const columns: ColumnDef[] = [];
    for (const colName of result.columns) {
      let typeName = "text";
      let jsType = "string";
      for (const row of result.rows) {
        const sample = row[colName];
        if (sample !== null && sample !== undefined) {
          if (typeof sample === "boolean") {
            typeName = "text";
            jsType = "string";
          } else if (typeof sample === "number") {
            if (Number.isInteger(sample)) {
              typeName = "integer";
              jsType = "number";
            } else {
              typeName = "real";
              jsType = "number";
            }
          }
          break;
        }
      }
      columns.push(createColumnDef(colName, typeName, { pythonType: jsType }));
    }

    const table = new Table(name, columns);
    for (let i = 0; i < result.rows.length; i++) {
      const docId = i + 1;
      const doc: Record<string, unknown> = { _id: docId };
      Object.assign(doc, result.rows[i]);
      table.documentStore.put(docId, doc);
    }
    return table;
  }

  // -- Set operations (UNION / INTERSECT / EXCEPT) --------------------

  private _compileSetOp(stmt: Record<string, unknown>, params: unknown[]): SQLResult {
    const op = nodeGet(stmt, "op");
    const larg = asObj(nodeGet(stmt, "larg"));
    const rarg = asObj(nodeGet(stmt, "rarg"));
    const all = nodeGet(stmt, "all") === true;

    const leftStmt = asObj(nodeGet(larg, "SelectStmt") ?? larg);
    const rightStmt = asObj(nodeGet(rarg, "SelectStmt") ?? rarg);

    const leftResult = this._compileSelect(leftStmt, params);
    const rightResult = this._compileSelect(rightStmt, params);

    if (leftResult.columns.length !== rightResult.columns.length) {
      throw new Error(
        `Set operation column count mismatch: ` +
          `${String(leftResult.columns.length)} vs ${String(rightResult.columns.length)}`,
      );
    }

    const columns = leftResult.columns;

    // Normalize right result columns to match left column names
    const rightRows = rightResult.rows.map((row) => {
      const normalized: Record<string, unknown> = {};
      for (let i = 0; i < columns.length; i++) {
        normalized[columns[i]!] = row[rightResult.columns[i]!] ?? null;
      }
      return normalized;
    });

    let rows: Record<string, unknown>[];

    // SETOP_UNION = 1, SETOP_INTERSECT = 2, SETOP_EXCEPT = 3
    if (op === 1 || op === "SETOP_UNION") {
      rows = this._setUnion(leftResult.rows, rightRows, columns, all);
    } else if (op === 2 || op === "SETOP_INTERSECT") {
      rows = this._setIntersect(leftResult.rows, rightRows, columns, all);
    } else if (op === 3 || op === "SETOP_EXCEPT") {
      rows = this._setExcept(leftResult.rows, rightRows, columns, all);
    } else {
      throw new Error(`Unsupported set operation: ${String(op)}`);
    }

    // Apply ORDER BY / LIMIT on the combined result
    const sortClause = asList(nodeGet(stmt, "sortClause"));
    if (sortClause.length > 0) {
      const evaluator = new ExprEvaluator({ params });
      const leftTargets = asList(nodeGet(leftStmt, "targetList"));
      rows = this._applyOrderBy(rows, sortClause, evaluator, leftTargets);
    }

    const limitCount = nodeGet(stmt, "limitCount");
    const limitOffset = nodeGet(stmt, "limitOffset");
    if (limitOffset !== null && limitOffset !== undefined) {
      const evaluator = new ExprEvaluator({ params });
      const offset = Number(evaluator.evaluate(asObj(limitOffset), {}));
      rows = rows.slice(offset);
    }
    if (limitCount !== null && limitCount !== undefined) {
      const evaluator = new ExprEvaluator({ params });
      const limit = Number(evaluator.evaluate(asObj(limitCount), {}));
      rows = rows.slice(0, limit);
    }

    return { columns, rows };
  }

  private _setUnion(
    leftRows: Record<string, unknown>[],
    rightRows: Record<string, unknown>[],
    columns: string[],
    isAll: boolean,
  ): Record<string, unknown>[] {
    if (isAll) {
      return [...leftRows, ...rightRows];
    }
    const seen = new Set<string>();
    const rows: Record<string, unknown>[] = [];
    for (const row of [...leftRows, ...rightRows]) {
      const key = columns.map((c) => JSON.stringify(row[c])).join("\0");
      if (!seen.has(key)) {
        seen.add(key);
        rows.push(row);
      }
    }
    return rows;
  }

  private _setIntersect(
    leftRows: Record<string, unknown>[],
    rightRows: Record<string, unknown>[],
    columns: string[],
    isAll: boolean,
  ): Record<string, unknown>[] {
    if (isAll) {
      const rightCounter = new Map<string, number>();
      for (const r of rightRows) {
        const key = columns.map((c) => JSON.stringify(r[c])).join("\0");
        rightCounter.set(key, (rightCounter.get(key) ?? 0) + 1);
      }
      const rows: Record<string, unknown>[] = [];
      for (const row of leftRows) {
        const key = columns.map((c) => JSON.stringify(row[c])).join("\0");
        const count = rightCounter.get(key) ?? 0;
        if (count > 0) {
          rows.push(row);
          rightCounter.set(key, count - 1);
        }
      }
      return rows;
    }

    const rightKeys = new Set(
      rightRows.map((r) => columns.map((c) => JSON.stringify(r[c])).join("\0")),
    );
    const seen = new Set<string>();
    const rows: Record<string, unknown>[] = [];
    for (const row of leftRows) {
      const key = columns.map((c) => JSON.stringify(row[c])).join("\0");
      if (rightKeys.has(key) && !seen.has(key)) {
        seen.add(key);
        rows.push(row);
      }
    }
    return rows;
  }

  private _setExcept(
    leftRows: Record<string, unknown>[],
    rightRows: Record<string, unknown>[],
    columns: string[],
    isAll: boolean,
  ): Record<string, unknown>[] {
    if (isAll) {
      const rightCounter = new Map<string, number>();
      for (const r of rightRows) {
        const key = columns.map((c) => JSON.stringify(r[c])).join("\0");
        rightCounter.set(key, (rightCounter.get(key) ?? 0) + 1);
      }
      const rows: Record<string, unknown>[] = [];
      for (const row of leftRows) {
        const key = columns.map((c) => JSON.stringify(row[c])).join("\0");
        const count = rightCounter.get(key) ?? 0;
        if (count > 0) {
          rightCounter.set(key, count - 1);
        } else {
          rows.push(row);
        }
      }
      return rows;
    }

    const rightKeys = new Set(
      rightRows.map((r) => columns.map((c) => JSON.stringify(r[c])).join("\0")),
    );
    const seen = new Set<string>();
    const rows: Record<string, unknown>[] = [];
    for (const row of leftRows) {
      const key = columns.map((c) => JSON.stringify(row[c])).join("\0");
      if (!rightKeys.has(key) && !seen.has(key)) {
        seen.add(key);
        rows.push(row);
      }
    }
    return rows;
  }

  // -- VALUES clause --------------------------------------------------

  private _compileValues(
    valuesLists: Record<string, unknown>[],
    params: unknown[],
  ): SQLResult {
    const evaluator = new ExprEvaluator({ params });
    const rows: Record<string, unknown>[] = [];
    let numCols = 0;

    for (const valueList of valuesLists) {
      const listNode = asObj(nodeGet(valueList, "List"));
      const items = asList(nodeGet(listNode, "items") ?? valueList);
      numCols = Math.max(numCols, items.length);
      const row: Record<string, unknown> = {};
      for (let i = 0; i < items.length; i++) {
        const colName = `column${String(i + 1)}`;
        row[colName] = evaluator.evaluate(items[i]!, {});
      }
      rows.push(row);
    }

    const columns: string[] = [];
    for (let i = 0; i < numCols; i++) {
      columns.push(`column${String(i + 1)}`);
    }

    return { columns, rows };
  }

  // -- FROM clause resolution -----------------------------------------

  private _resolveFromTableName(fromClause: Record<string, unknown>[]): string | null {
    if (fromClause.length !== 1) return null;
    const item = fromClause[0]!;
    const rv = item["RangeVar"] as Record<string, unknown> | undefined;
    if (rv) return (rv["relname"] as string) ?? null;
    return null;
  }

  /**
   * Build a map from table alias (or table name) to actual table name
   * by traversing the FROM clause, including JoinExpr trees.
   */
  private _collectFromAliases(fromClause: Record<string, unknown>[]): Map<string, string> {
    const aliasMap = new Map<string, string>();
    for (const item of fromClause) {
      this._collectFromAliasesFromItem(item, aliasMap);
    }
    return aliasMap;
  }

  private _collectFromAliasesFromItem(
    node: Record<string, unknown>,
    aliasMap: Map<string, string>,
  ): void {
    const rv = nodeGet(node, "RangeVar");
    if (rv !== null && rv !== undefined) {
      const rvObj = asObj(rv);
      const tableName = qualifiedName(rvObj);
      const alias = extractAlias(rvObj);
      if (alias) {
        aliasMap.set(alias, tableName);
      }
      aliasMap.set(tableName, tableName);
      return;
    }
    const joinExpr = nodeGet(node, "JoinExpr");
    if (joinExpr !== null && joinExpr !== undefined) {
      const je = asObj(joinExpr);
      this._collectFromAliasesFromItem(asObj(nodeGet(je, "larg")), aliasMap);
      this._collectFromAliasesFromItem(asObj(nodeGet(je, "rarg")), aliasMap);
    }
  }

  /**
   * Extract the table alias from the first qualified ColumnRef
   * found within a UQA function subtree.  Returns null when all
   * column references are unqualified.
   */
  private static _extractUQAColumnAlias(node: Record<string, unknown>): string | null {
    const colRef = nodeGet(node, "ColumnRef");
    if (colRef !== null && colRef !== undefined) {
      const crObj = asObj(colRef);
      const fields = asList(nodeGet(crObj, "fields"));
      if (fields.length >= 2) {
        const alias = extractString(fields[0]!);
        if (alias) return alias;
      }
    }
    for (const key of Object.keys(node)) {
      const val = node[key];
      if (val === null || val === undefined) continue;
      if (Array.isArray(val)) {
        for (const item of val) {
          if (typeof item === "object" && item !== null) {
            const result = SQLCompiler._extractUQAColumnAlias(
              item as Record<string, unknown>,
            );
            if (result) return result;
          }
        }
      } else if (typeof val === "object") {
        const result = SQLCompiler._extractUQAColumnAlias(
          val as Record<string, unknown>,
        );
        if (result) return result;
      }
    }
    return null;
  }

  private _resolveFrom(
    fromClause: Record<string, unknown>[],
    params: unknown[],
  ): Record<string, unknown>[] {
    let result: Record<string, unknown>[] | null = null;

    for (const fromItem of fromClause) {
      // Check for LATERAL subquery
      const subselect = nodeGet(fromItem, "RangeSubselect");
      if (subselect !== null && subselect !== undefined) {
        const subNode = asObj(subselect);
        const isLateral = nodeGet(subNode, "lateral") === true;
        if (isLateral && result !== null) {
          result = this._resolveLateralJoin(result, subNode, params);
          continue;
        }
      }

      const itemRows = this._resolveFromItem(fromItem, params);

      if (result === null) {
        result = itemRows;
      } else {
        // Cross join
        result = this._crossJoin(result, itemRows);
      }
    }

    return result ?? [{}];
  }

  private _resolveFromItem(
    item: Record<string, unknown>,
    params: unknown[],
  ): Record<string, unknown>[] {
    // RangeVar -- simple table reference
    const rangeVar = nodeGet(item, "RangeVar");
    if (rangeVar !== null && rangeVar !== undefined) {
      const rvObj = asObj(rangeVar);
      const tableName = qualifiedName(rvObj);
      const rawName = extractRelName(rvObj);
      const alias = extractAlias(rvObj);
      const schemaName = extractSchemaName(rvObj);

      // Virtual schema tables
      if (schemaName === "information_schema") {
        return this._buildInformationSchemaTable(rawName, alias ?? rawName);
      }
      if (schemaName === "pg_catalog") {
        return this._buildPgCatalogTable(rawName, alias ?? rawName);
      }

      // Check for view
      const viewDef = this._views.get(rawName);
      if (viewDef !== undefined) {
        const viewResult = this._compileSelect(viewDef, params);
        return this._applyAlias(viewResult.rows, alias ?? rawName);
      }

      // Check for inlined CTE
      const inlinedQuery = this._inlinedCTEs.get(rawName);
      if (inlinedQuery !== undefined) {
        this._inlinedCTEs.delete(rawName);
        const result = this._compileSelect(inlinedQuery, params);
        const table = this._resultToTable(tableName, result);
        this._tables.set(tableName, table);
        this._expandedViews.push(tableName);
        const rows: Record<string, unknown>[] = [];
        for (const [, doc] of table.documentStore.iterAll()) {
          rows.push({ ...doc });
        }
        return this._applyAlias(rows, alias ?? rawName);
      }

      // Check for foreign table
      const foreignTable = this._foreignTables.get(rawName);
      if (foreignTable !== undefined) {
        const rows = this._scanForeignTable(foreignTable);
        return this._applyAlias(rows, alias ?? tableName);
      }

      const table = this._tables.get(tableName);
      if (!table) {
        throw new Error(`Table "${tableName}" does not exist`);
      }

      // UQA pre-filter: emit only posting-list matches when active
      if (this._uqaFromFilter && tableName === this._uqaFromFilter.tableName) {
        const { scores, pkCol } = this._uqaFromFilter;
        const filtered: Record<string, unknown>[] = [];
        for (const [docId, score] of scores) {
          const doc = table.documentStore.get(docId as number);
          if (!doc) continue;
          const row: Record<string, unknown> = { ...doc, _score: score };
          if (pkCol) row[pkCol] = docId;
          filtered.push(row);
        }
        return this._applyAlias(filtered, alias ?? tableName);
      }

      const rows: Record<string, unknown>[] = [];
      for (const [, doc] of table.documentStore.iterAll()) {
        rows.push({ ...doc });
      }

      return this._applyAlias(rows, alias ?? tableName);
    }

    // JoinExpr -- explicit join
    const joinExpr = nodeGet(item, "JoinExpr");
    if (joinExpr !== null && joinExpr !== undefined) {
      return this._resolveJoin(asObj(joinExpr), params);
    }

    // RangeSubselect -- subquery in FROM
    const subselect = nodeGet(item, "RangeSubselect");
    if (subselect !== null && subselect !== undefined) {
      const subNode = asObj(subselect);
      const subquery = asObj(nodeGet(subNode, "subquery"));
      const selectStmt = asObj(nodeGet(subquery, "SelectStmt") ?? subquery);
      const alias = extractAlias(subNode);
      const subResult = this._compileSelect(selectStmt, params);
      return this._applyAlias(subResult.rows, alias);
    }

    // RangeFunction -- function in FROM
    const rangeFunction = nodeGet(item, "RangeFunction");
    if (rangeFunction !== null && rangeFunction !== undefined) {
      return this._compileFromFunction(asObj(rangeFunction), params);
    }

    throw new Error("Unsupported FROM clause item");
  }

  // ==================================================================
  // FDW: handler dispatch
  // ==================================================================

  /**
   * Get or create an FDW handler for the given server name.
   * Handlers are cached by server name and reused across queries.
   */
  private _getFDWHandler(serverName: string): FDWHandler {
    const cached = this._fdwHandlers.get(serverName);
    if (cached !== undefined) return cached;

    const server = this._foreignServers.get(serverName);
    if (server === undefined) {
      throw new Error(`Foreign server '${serverName}' does not exist`);
    }

    let handler: FDWHandler;
    if (server.fdwType === "duckdb_fdw") {
      handler = new DuckDBFDWHandler(server);
    } else if (server.fdwType === "arrow_fdw") {
      handler = new ArrowFDWHandler(server);
    } else {
      throw new Error(`Unsupported FDW type: '${server.fdwType}'`);
    }

    this._fdwHandlers.set(serverName, handler);
    return handler;
  }

  /**
   * Scan a foreign table via the appropriate FDW handler.
   */
  private _scanForeignTable(foreignTable: ForeignTable): Record<string, unknown>[] {
    const handler = this._getFDWHandler(foreignTable.serverName);
    return handler.scan(foreignTable);
  }

  private _applyAlias(
    rows: Record<string, unknown>[],
    alias: string | null,
  ): Record<string, unknown>[] {
    if (alias === null) return rows;
    return rows.map((row) => {
      const aliased: Record<string, unknown> = { ...row };
      for (const [key, value] of Object.entries(row)) {
        if (!key.includes(".")) {
          aliased[`${alias}.${key}`] = value;
        }
      }
      return aliased;
    });
  }

  private _resolveLateralJoin(
    leftRows: Record<string, unknown>[],
    subNode: Record<string, unknown>,
    params: unknown[],
  ): Record<string, unknown>[] {
    const subquery = asObj(nodeGet(subNode, "subquery"));
    const selectStmt = asObj(nodeGet(subquery, "SelectStmt") ?? subquery);
    const alias = extractAlias(subNode) ?? "_lateral";

    const result: Record<string, unknown>[] = [];

    for (const leftRow of leftRows) {
      // Execute subquery with left row as outer row context
      const subResult = this._compileSelect(selectStmt, params, leftRow);

      for (const subRow of subResult.rows) {
        const rightFields: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(subRow)) {
          rightFields[k] = v;
          rightFields[`${alias}.${k}`] = v;
        }
        result.push({ ...leftRow, ...rightFields });
      }
    }

    return result;
  }

  // -- FROM-clause table functions ------------------------------------

  private _compileFromFunction(
    rangeFunc: Record<string, unknown>,
    params: unknown[],
  ): Record<string, unknown>[] {
    const functions = asList(nodeGet(rangeFunc, "functions"));
    if (functions.length === 0) {
      throw new Error("Empty RangeFunction");
    }
    // functions[0] may be a List wrapper: { List: { items: [...] } }
    const firstFunc = functions[0]!;
    const listNode = nodeGet(firstFunc, "List");
    const firstFuncList =
      listNode !== null && listNode !== undefined
        ? asList(nodeGet(asObj(listNode), "items"))
        : asList(firstFunc);
    if (firstFuncList.length === 0) {
      throw new Error("Empty function list in RangeFunction");
    }
    const funcCallNode = asObj(
      nodeGet(firstFuncList[0]!, "FuncCall") ?? firstFuncList[0]!,
    );
    const funcnameList = asList(nodeGet(funcCallNode, "funcname"));
    const funcName = extractString(
      funcnameList[funcnameList.length - 1]!,
    ).toLowerCase();
    const funcArgs = asList(nodeGet(funcCallNode, "args"));

    const alias = extractAlias(rangeFunc);

    // Determine column name from alias column names
    const aliasNode = asObj(nodeGet(rangeFunc, "alias"));
    const colnames = asList(nodeGet(aliasNode, "colnames"));
    const aliasColName = colnames.length > 0 ? extractString(colnames[0]!) : null;

    const evaluator = new ExprEvaluator({ params });

    if (funcName === "generate_series") {
      return this._buildGenerateSeries(
        funcArgs,
        evaluator,
        alias ?? "generate_series",
        aliasColName ?? "generate_series",
      );
    }

    if (funcName === "unnest") {
      return this._buildUnnest(
        funcArgs,
        evaluator,
        alias ?? "unnest",
        aliasColName ?? "unnest",
      );
    }

    if (
      funcName === "json_each" ||
      funcName === "jsonb_each" ||
      funcName === "json_each_text" ||
      funcName === "jsonb_each_text"
    ) {
      return this._buildJSONEach(
        funcArgs,
        evaluator,
        alias ?? funcName,
        funcName.endsWith("_text"),
      );
    }

    if (
      funcName === "json_array_elements" ||
      funcName === "jsonb_array_elements" ||
      funcName === "json_array_elements_text" ||
      funcName === "jsonb_array_elements_text"
    ) {
      return this._buildJSONArrayElements(
        funcArgs,
        evaluator,
        alias ?? funcName,
        funcName.endsWith("_text"),
      );
    }

    if (funcName === "regexp_split_to_table") {
      return this._buildRegexpSplitToTable(
        funcArgs,
        evaluator,
        alias ?? "regexp_split_to_table",
        aliasColName ?? "regexp_split_to_table",
      );
    }

    if (funcName === "create_graph") return this._buildCreateGraph(funcArgs, evaluator);
    if (funcName === "drop_graph") return this._buildDropGraph(funcArgs, evaluator);
    if (funcName === "create_analyzer") return this._buildCreateAnalyzer(funcArgs, evaluator);
    if (funcName === "drop_analyzer") return this._buildDropAnalyzer(funcArgs, evaluator);
    if (funcName === "list_analyzers") return this._buildListAnalyzers();
    if (funcName === "set_table_analyzer") return this._buildSetTableAnalyzer(funcArgs, evaluator);
    if (funcName === "graph_add_vertex") return this._buildGraphAddVertex(funcArgs, evaluator);
    if (funcName === "graph_add_edge") return this._buildGraphAddEdge(funcArgs, evaluator);
    if (funcName === "build_grid_graph") return this._buildGridGraph(funcArgs, evaluator);
    if (funcName === "cypher") return this._buildCypherFrom(funcArgs, evaluator);

    throw new Error(`Unsupported FROM-clause function: ${funcName}`);
  }

  // -- Graph/Analyzer/Cypher FROM functions -------------------------------

  private _buildCreateGraph(a: Record<string, unknown>[], ev: ExprEvaluator): Record<string, unknown>[] {
    if (!a.length) throw new Error("create_graph() requires a graph name");
    const n = String(ev.evaluate(a[0]!, {}));
    (this._engine as { createGraph(n: string): void }).createGraph(n);
    return [{ create_graph: `graph '${n}' created` }];
  }
  private _buildDropGraph(a: Record<string, unknown>[], ev: ExprEvaluator): Record<string, unknown>[] {
    if (!a.length) throw new Error("drop_graph() requires a graph name");
    const n = String(ev.evaluate(a[0]!, {}) as string | number);
    (this._engine as { dropGraph(n: string): void }).dropGraph(n);
    return [{ drop_graph: `graph '${n}' dropped` }];
  }
  private _buildCreateAnalyzer(a: Record<string, unknown>[], ev: ExprEvaluator): Record<string, unknown>[] {
    if (a.length < 2) throw new Error("create_analyzer() requires (name, config_json)");
    const n = String(ev.evaluate(a[0]!, {}) as string | number);
    const c = JSON.parse(String(ev.evaluate(a[1]!, {}) as string | number)) as Record<string, unknown>;
    (this._engine as { createAnalyzer(n: string, c: Record<string, unknown>): void }).createAnalyzer(n, c);
    return [{ create_analyzer: `analyzer '${n}' created` }];
  }
  private _buildDropAnalyzer(a: Record<string, unknown>[], ev: ExprEvaluator): Record<string, unknown>[] {
    if (!a.length) throw new Error("drop_analyzer() requires a name");
    const n = String(ev.evaluate(a[0]!, {}) as string | number);
    (this._engine as { dropAnalyzer(n: string): void }).dropAnalyzer(n);
    return [{ drop_analyzer: `analyzer '${n}' dropped` }];
  }
  private _buildListAnalyzers(): Record<string, unknown>[] {
    return listAnalyzersFn().map((n) => ({ analyzer_name: n }));
  }
  private _buildSetTableAnalyzer(a: Record<string, unknown>[], ev: ExprEvaluator): Record<string, unknown>[] {
    if (a.length < 3) throw new Error("set_table_analyzer(table, field, analyzer[, phase])");
    const t = String(ev.evaluate(a[0]!, {}) as string | number);
    const f = String(ev.evaluate(a[1]!, {}) as string | number);
    const an = String(ev.evaluate(a[2]!, {}) as string | number);
    const ph = a.length > 3 ? String(ev.evaluate(a[3]!, {}) as string | number) : "both";
    (this._engine as unknown as { setTableAnalyzer(t: string, f: string, a: string, phase: string): void }).setTableAnalyzer(t, f, an, ph);
    return [{ set_table_analyzer: `analyzer '${an}' assigned to ${t}.${f}` }];
  }
  private _buildGraphAddVertex(a: Record<string, unknown>[], ev: ExprEvaluator): Record<string, unknown>[] {
    if (a.length < 3) throw new Error("graph_add_vertex(id, label, table[, props])");
    const vid = Number(ev.evaluate(a[0]!, {}));
    const lbl = String(ev.evaluate(a[1]!, {}) as string | number);
    const tbl = String(ev.evaluate(a[2]!, {}) as string | number);
    const props: Record<string, unknown> = {};
    if (a.length > 3) {
      for (const pair of String(ev.evaluate(a[3]!, {}) as string | number).split(",")) {
        const [k, v] = pair.trim().split("=", 2);
        if (k && v) props[k.trim()] = isNaN(Number(v.trim())) ? v.trim() : Number(v.trim());
      }
    }
    (this._engine as { addGraphVertex(v: unknown, o: { table: string }): void }).addGraphVertex(createVertex(vid, lbl, props), { table: tbl });
    return [{ result: `vertex ${String(vid)} added to ${tbl}` }];
  }
  private _buildGraphAddEdge(a: Record<string, unknown>[], ev: ExprEvaluator): Record<string, unknown>[] {
    if (a.length < 5) throw new Error("graph_add_edge(eid, src, tgt, label, table[, props])");
    const eid = Number(ev.evaluate(a[0]!, {}));
    const src = Number(ev.evaluate(a[1]!, {}));
    const tgt = Number(ev.evaluate(a[2]!, {}));
    const lbl = String(ev.evaluate(a[3]!, {}) as string | number);
    const tbl = String(ev.evaluate(a[4]!, {}) as string | number);
    const props: Record<string, unknown> = {};
    if (a.length > 5) {
      for (const pair of String(ev.evaluate(a[5]!, {}) as string | number).split(",")) {
        const [k, v] = pair.trim().split("=", 2);
        if (k && v) props[k.trim()] = isNaN(Number(v.trim())) ? v.trim() : Number(v.trim());
      }
    }
    (this._engine as { addGraphEdge(e: unknown, o: { table: string }): void }).addGraphEdge(createEdge(eid, src, tgt, lbl, props), { table: tbl });
    return [{ result: `edge ${String(eid)} added to ${tbl}` }];
  }
  private _buildGridGraph(a: Record<string, unknown>[], ev: ExprEvaluator): Record<string, unknown>[] {
    if (a.length < 4) throw new Error("build_grid_graph(table, rows, cols, label)");
    const tbl = String(ev.evaluate(a[0]!, {}) as string | number);
    const rows = Number(ev.evaluate(a[1]!, {}));
    const cols = Number(ev.evaluate(a[2]!, {}));
    const lbl = String(ev.evaluate(a[3]!, {}) as string | number);
    let eid = 1;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const pid = r * cols + c + 1;
        if (c < cols - 1) { (this._engine as { addGraphEdge(e: unknown, o: { table: string }): void }).addGraphEdge(createEdge(eid++, pid, pid + 1, lbl), { table: tbl }); }
        if (r < rows - 1) { (this._engine as { addGraphEdge(e: unknown, o: { table: string }): void }).addGraphEdge(createEdge(eid++, pid, pid + cols, lbl), { table: tbl }); }
      }
    }
    return [{ table_name: tbl, rows, cols, edges: eid - 1 }];
  }
  private _buildCypherFrom(a: Record<string, unknown>[], ev: ExprEvaluator): Record<string, unknown>[] {
    if (a.length < 2) throw new Error("cypher() requires (graph_name, query)");
    const gn = String(ev.evaluate(a[0]!, {}) as string | number);
    const qs = String(ev.evaluate(a[1]!, {}) as string | number);
    const graph = (this._engine as { getGraph(n: string): GraphStore }).getGraph(gn);
    const params: Record<string, unknown> = {};
    for (let i = 0; i < this._params.length; i++) params[String(i + 1)] = this._params[i];
    const compiler = new CypherCompiler(graph, gn, params);
    return compiler.executeRows(qs, params);
  }

  private _buildGenerateSeries(
    funcArgs: Record<string, unknown>[],
    evaluator: ExprEvaluator,
    aliasName: string,
    colName: string,
  ): Record<string, unknown>[] {
    if (funcArgs.length < 2) {
      throw new Error("generate_series requires at least 2 arguments");
    }
    const start = Number(evaluator.evaluate(funcArgs[0]!, {}));
    const stop = Number(evaluator.evaluate(funcArgs[1]!, {}));
    const step = funcArgs.length > 2 ? Number(evaluator.evaluate(funcArgs[2]!, {})) : 1;

    if (step === 0) {
      throw new Error("generate_series step cannot be zero");
    }

    const rows: Record<string, unknown>[] = [];
    if (step > 0) {
      for (let current = start; current <= stop; current += step) {
        const row: Record<string, unknown> = { [colName]: current };
        row[`${aliasName}.${colName}`] = current;
        rows.push(row);
      }
    } else {
      for (let current = start; current >= stop; current += step) {
        const row: Record<string, unknown> = { [colName]: current };
        row[`${aliasName}.${colName}`] = current;
        rows.push(row);
      }
    }
    return rows;
  }

  private _buildUnnest(
    funcArgs: Record<string, unknown>[],
    evaluator: ExprEvaluator,
    aliasName: string,
    colName: string,
  ): Record<string, unknown>[] {
    if (funcArgs.length < 1) {
      throw new Error("unnest requires at least 1 argument");
    }
    let arr = evaluator.evaluate(funcArgs[0]!, {});
    if (!Array.isArray(arr)) {
      arr = [arr];
    }
    const rows: Record<string, unknown>[] = [];
    for (const val of arr as unknown[]) {
      const row: Record<string, unknown> = { [colName]: val };
      row[`${aliasName}.${colName}`] = val;
      rows.push(row);
    }
    return rows;
  }

  private _buildJSONEach(
    funcArgs: Record<string, unknown>[],
    evaluator: ExprEvaluator,
    aliasName: string,
    asText: boolean,
  ): Record<string, unknown>[] {
    if (funcArgs.length < 1) {
      throw new Error("json_each requires at least 1 argument");
    }
    let jsonVal = evaluator.evaluate(funcArgs[0]!, {});
    if (typeof jsonVal === "string") {
      jsonVal = JSON.parse(jsonVal);
    }
    if (jsonVal === null || typeof jsonVal !== "object" || Array.isArray(jsonVal)) {
      throw new Error("json_each expects a JSON object");
    }
    const rows: Record<string, unknown>[] = [];
    for (const [key, value] of Object.entries(jsonVal as Record<string, unknown>)) {
      const displayValue = asText ? String(value) : value;
      const row: Record<string, unknown> = {
        key,
        value: displayValue,
      };
      row[`${aliasName}.key`] = key;
      row[`${aliasName}.value`] = displayValue;
      rows.push(row);
    }
    return rows;
  }

  private _buildJSONArrayElements(
    funcArgs: Record<string, unknown>[],
    evaluator: ExprEvaluator,
    aliasName: string,
    asText: boolean,
  ): Record<string, unknown>[] {
    if (funcArgs.length < 1) {
      throw new Error("json_array_elements requires at least 1 argument");
    }
    let jsonVal = evaluator.evaluate(funcArgs[0]!, {});
    if (typeof jsonVal === "string") {
      jsonVal = JSON.parse(jsonVal);
    }
    if (!Array.isArray(jsonVal)) {
      throw new Error("json_array_elements expects a JSON array");
    }
    const rows: Record<string, unknown>[] = [];
    for (const elem of jsonVal as unknown[]) {
      const displayValue = asText ? String(elem) : elem;
      const row: Record<string, unknown> = { value: displayValue };
      row[`${aliasName}.value`] = displayValue;
      rows.push(row);
    }
    return rows;
  }

  private _buildRegexpSplitToTable(
    funcArgs: Record<string, unknown>[],
    evaluator: ExprEvaluator,
    aliasName: string,
    colName: string,
  ): Record<string, unknown>[] {
    if (funcArgs.length < 2) {
      throw new Error("regexp_split_to_table requires at least 2 arguments");
    }
    const str = String(evaluator.evaluate(funcArgs[0]!, {}));
    const pattern = String(evaluator.evaluate(funcArgs[1]!, {}));
    const flags =
      funcArgs.length > 2 ? String(evaluator.evaluate(funcArgs[2]!, {})) : "";
    const regex = new RegExp(pattern, flags.includes("g") ? flags : flags + "g");
    const parts = str.split(regex);
    const rows: Record<string, unknown>[] = [];
    for (const part of parts) {
      const row: Record<string, unknown> = { [colName]: part };
      row[`${aliasName}.${colName}`] = part;
      rows.push(row);
    }
    return rows;
  }

  // -- JOIN -----------------------------------------------------------

  private _crossJoin(
    left: Record<string, unknown>[],
    right: Record<string, unknown>[],
  ): Record<string, unknown>[] {
    const result: Record<string, unknown>[] = [];
    for (const lRow of left) {
      for (const rRow of right) {
        result.push({ ...lRow, ...rRow });
      }
    }
    return result;
  }

  private _resolveJoin(
    joinExpr: Record<string, unknown>,
    params: unknown[],
  ): Record<string, unknown>[] {
    const joinType = nodeGet(joinExpr, "jointype");
    const larg = asObj(nodeGet(joinExpr, "larg"));
    const rarg = asObj(nodeGet(joinExpr, "rarg"));
    const quals = nodeGet(joinExpr, "quals");

    // Check for LATERAL subquery on the right side
    const rSubselect = nodeGet(rarg, "RangeSubselect");
    if (rSubselect !== null && rSubselect !== undefined) {
      const subNode = asObj(rSubselect);
      if (nodeGet(subNode, "lateral") === true) {
        const leftRows = this._resolveFromItem(larg, params);
        return this._resolveLateralJoin(leftRows, subNode, params);
      }
    }

    const leftRows = this._resolveFromItem(larg, params);
    const rightRows = this._resolveFromItem(rarg, params);

    const evaluator = new ExprEvaluator({ params });

    // JOIN_INNER = 0, JOIN_LEFT = 1, JOIN_FULL = 2, JOIN_RIGHT = 3, JOIN_CROSS = 5
    if (joinType === 5 || joinType === "JOIN_CROSS") {
      return this._crossJoin(leftRows, rightRows);
    }

    // CROSS JOIN: INNER with no quals
    if (
      (joinType === 0 || joinType === "JOIN_INNER") &&
      (quals === null || quals === undefined)
    ) {
      return this._crossJoin(leftRows, rightRows);
    }

    const isLeftJoin = joinType === 1 || joinType === "JOIN_LEFT";
    const isRightJoin = joinType === 3 || joinType === "JOIN_RIGHT";
    const isFullJoin = joinType === 2 || joinType === "JOIN_FULL";

    // Try O(n+m) hash join for equi-join conditions
    const equiKeys = SQLCompiler._extractEquiJoinKeys(quals);
    if (equiKeys) {
      return SQLCompiler._hashJoin(
        leftRows,
        rightRows,
        equiKeys,
        isLeftJoin,
        isRightJoin,
        isFullJoin,
      );
    }

    // Fallback: nested loop for non-equi or complex ON conditions
    const result: Record<string, unknown>[] = [];
    const rightMatched = new Set<number>();

    for (const lRow of leftRows) {
      let matched = false;
      for (let ri = 0; ri < rightRows.length; ri++) {
        const rRow = rightRows[ri]!;
        const combined = { ...lRow, ...rRow };

        if (quals !== null && quals !== undefined) {
          const condition = evaluator.evaluate(asObj(quals), combined);
          if (condition !== true) continue;
        }

        result.push(combined);
        matched = true;
        rightMatched.add(ri);
      }

      if (!matched && (isLeftJoin || isFullJoin)) {
        const merged: Record<string, unknown> = { ...lRow };
        if (rightRows.length > 0) {
          for (const key of Object.keys(rightRows[0]!)) {
            if (!(key in merged)) {
              merged[key] = null;
            }
          }
        }
        result.push(merged);
      }
    }

    if (isRightJoin || isFullJoin) {
      for (let ri = 0; ri < rightRows.length; ri++) {
        if (!rightMatched.has(ri)) {
          const merged: Record<string, unknown> = {};
          if (leftRows.length > 0) {
            for (const key of Object.keys(leftRows[0]!)) {
              merged[key] = null;
            }
          }
          for (const [key, val] of Object.entries(rightRows[ri]!)) {
            merged[key] = val;
          }
          result.push(merged);
        }
      }
    }

    return result;
  }

  /**
   * Extract equi-join key columns from an ON clause.
   * Returns null if the condition is not a pure equi-join.
   */
  private static _extractEquiJoinKeys(
    quals: unknown,
  ): { leftCols: string[]; rightCols: string[] } | null {
    if (quals === null || quals === undefined) return null;
    const q = asObj(quals);
    // Single equality: A_Expr =
    const aExpr = asObj(nodeGet(q, "A_Expr") ?? {});
    if (Object.keys(aExpr).length > 0) {
      return SQLCompiler._extractOneEquality(aExpr);
    }
    // Direct A_Expr (unwrapped)
    if (nodeGet(q, "name") !== null) {
      return SQLCompiler._extractOneEquality(q);
    }
    // BoolExpr AND of equalities
    const boolExpr = asObj(nodeGet(q, "BoolExpr") ?? {});
    if (Object.keys(boolExpr).length > 0) {
      const boolop = nodeGet(boolExpr, "boolop");
      if (boolop !== 0 && boolop !== "AND_EXPR") return null;
      const args = asList(nodeGet(boolExpr, "args"));
      const leftCols: string[] = [];
      const rightCols: string[] = [];
      for (const arg of args) {
        const inner = asObj(nodeGet(arg, "A_Expr") ?? arg);
        const pair = SQLCompiler._extractOneEquality(inner);
        if (!pair) return null;
        leftCols.push(pair.leftCols[0]!);
        rightCols.push(pair.rightCols[0]!);
      }
      if (leftCols.length > 0) return { leftCols, rightCols };
    }
    return null;
  }

  private static _extractOneEquality(
    aExpr: Record<string, unknown>,
  ): { leftCols: string[]; rightCols: string[] } | null {
    const nameList = asList(nodeGet(aExpr, "name"));
    if (nameList.length === 0 || extractString(nameList[0]!) !== "=") return null;
    const lexpr = nodeGet(aExpr, "lexpr");
    const rexpr = nodeGet(aExpr, "rexpr");
    if (!lexpr || !rexpr) return null;
    if (!isColumnRef(asObj(lexpr)) || !isColumnRef(asObj(rexpr))) return null;
    const left = extractQualifiedColumnName(asObj(lexpr));
    const right = extractQualifiedColumnName(asObj(rexpr));
    return { leftCols: [left], rightCols: [right] };
  }

  /**
   * O(n+m) hash join for equi-join conditions.
   * Supports INNER, LEFT, RIGHT, and FULL joins.
   */
  private static _hashJoin(
    leftRows: Record<string, unknown>[],
    rightRows: Record<string, unknown>[],
    keys: { leftCols: string[]; rightCols: string[] },
    isLeftJoin: boolean,
    isRightJoin: boolean,
    isFullJoin: boolean,
  ): Record<string, unknown>[] {
    // keys.leftCols / rightCols are the left/right of the "=" operator,
    // which may not correspond to the left/right of the JOIN.
    // Determine which key set belongs to which side by checking the rows.
    let probeCols = keys.leftCols;
    let buildCols = keys.rightCols;
    const lSample = leftRows.length > 0 ? leftRows[0]! : {};
    if (!(probeCols[0]! in lSample) && buildCols[0]! in lSample) {
      probeCols = keys.rightCols;
      buildCols = keys.leftCols;
    }

    // Build hash table on the right side using buildCols
    const hashMap = new Map<string, number[]>();
    for (let ri = 0; ri < rightRows.length; ri++) {
      const rRow = rightRows[ri]!;
      const key = buildCols.map((c) => String(rRow[c] ?? "\0NULL")).join("\0");
      let bucket = hashMap.get(key);
      if (!bucket) {
        bucket = [];
        hashMap.set(key, bucket);
      }
      bucket.push(ri);
    }

    const result: Record<string, unknown>[] = [];
    const rightMatched = new Set<number>();

    for (const lRow of leftRows) {
      const key = probeCols
        .map((c) => String(lRow[c] ?? "\0NULL"))
        .join("\0");
      const bucket = hashMap.get(key);
      if (bucket) {
        for (const ri of bucket) {
          result.push({ ...lRow, ...rightRows[ri]! });
          rightMatched.add(ri);
        }
      } else if (isLeftJoin || isFullJoin) {
        const merged: Record<string, unknown> = { ...lRow };
        if (rightRows.length > 0) {
          for (const k of Object.keys(rightRows[0]!)) {
            if (!(k in merged)) {
              merged[k] = null;
            }
          }
        }
        result.push(merged);
      }
    }

    if (isRightJoin || isFullJoin) {
      for (let ri = 0; ri < rightRows.length; ri++) {
        if (!rightMatched.has(ri)) {
          const merged: Record<string, unknown> = {};
          if (leftRows.length > 0) {
            for (const k of Object.keys(leftRows[0]!)) {
              merged[k] = null;
            }
          }
          for (const [k, val] of Object.entries(rightRows[ri]!)) {
            merged[k] = val;
          }
          result.push(merged);
        }
      }
    }

    return result;
  }

  // -- information_schema virtual tables ------------------------------

  private _buildInformationSchemaTable(
    viewName: string,
    alias: string,
  ): Record<string, unknown>[] {
    if (viewName === "tables") {
      const rows: Record<string, unknown>[] = [];
      const eng = this._engine as { _tables: { qualifiedItems(): Iterable<[string, string, Table]> } } | null;
      if (eng && typeof eng._tables.qualifiedItems === "function") {
        const items = [...eng._tables.qualifiedItems()].sort((a, b) =>
          a[0] === b[0] ? a[1].localeCompare(b[1]) : a[0].localeCompare(b[0]),
        );
        for (const [schema, tname] of items) {
          rows.push({
            table_catalog: "",
            table_schema: schema,
            table_name: tname,
            table_type: "BASE TABLE",
          });
        }
      } else {
        for (const tname of [...this._tables.keys()].sort()) {
          rows.push({
            table_catalog: "",
            table_schema: "public",
            table_name: tname,
            table_type: "BASE TABLE",
          });
        }
      }
      for (const ftname of [...this._foreignTables.keys()].sort()) {
        rows.push({
          table_catalog: "",
          table_schema: "public",
          table_name: ftname,
          table_type: "FOREIGN TABLE",
        });
      }
      for (const vname of [...this._views.keys()].sort()) {
        rows.push({
          table_catalog: "",
          table_schema: "public",
          table_name: vname,
          table_type: "VIEW",
        });
      }
      return this._applyAlias(rows, alias);
    }
    if (viewName === "columns") {
      const INFO_TYPE_DISPLAY: Record<string, string> = {
        int2: "smallint",
        int4: "integer",
        int8: "bigint",
        float4: "real",
        float8: "double precision",
        bool: "boolean",
      };
      const rows: Record<string, unknown>[] = [];
      const eng2 = this._engine as { _tables: { qualifiedItems(): Iterable<[string, string, Table]> } } | null;
      if (eng2 && typeof eng2._tables.qualifiedItems === "function") {
        const items = [...eng2._tables.qualifiedItems()].sort((a, b) =>
          a[0] === b[0] ? a[1].localeCompare(b[1]) : a[0].localeCompare(b[0]),
        );
        for (const [schema, tname, tbl] of items) {
          let pos = 1;
          for (const [cname, cdef] of tbl.columns) {
            const displayType = INFO_TYPE_DISPLAY[cdef.typeName] ?? cdef.typeName;
            rows.push({
              table_catalog: "",
              table_schema: schema,
              table_name: tname,
              column_name: cname,
              ordinal_position: pos,
              data_type: displayType,
              is_nullable: cdef.notNull ? "NO" : "YES",
            });
            pos++;
          }
        }
      } else {
        for (const tname of [...this._tables.keys()].sort()) {
          const tbl = this._tables.get(tname)!;
          let pos = 1;
          for (const [cname, cdef] of tbl.columns) {
            const displayType = INFO_TYPE_DISPLAY[cdef.typeName] ?? cdef.typeName;
            rows.push({
              table_catalog: "",
              table_schema: "public",
              table_name: tname,
              column_name: cname,
              ordinal_position: pos,
              data_type: displayType,
              is_nullable: cdef.notNull ? "NO" : "YES",
            });
            pos++;
          }
        }
      }
      for (const ftname of [...this._foreignTables.keys()].sort()) {
        const ft = this._foreignTables.get(ftname)!;
        let pos = 1;
        for (const [cname, cdef] of ft.columns) {
          const displayType = INFO_TYPE_DISPLAY[cdef.typeName] ?? cdef.typeName;
          rows.push({
            table_catalog: "",
            table_schema: "public",
            table_name: ftname,
            column_name: cname,
            ordinal_position: pos,
            data_type: displayType,
            is_nullable: cdef.notNull ? "NO" : "YES",
          });
          pos++;
        }
      }
      return this._applyAlias(rows, alias);
    }
    throw new Error(`Unknown information_schema view: "${viewName}"`);
  }

  private _buildPgCatalogTable(
    viewName: string,
    alias: string,
  ): Record<string, unknown>[] {
    if (viewName === "pg_tables") {
      const rows: Record<string, unknown>[] = [];
      const eng = this._engine as { _tables: { qualifiedItems(): Iterable<[string, string, Table]> } } | null;
      if (eng && typeof eng._tables.qualifiedItems === "function") {
        const items = [...eng._tables.qualifiedItems()].sort((a, b) =>
          a[0] === b[0] ? a[1].localeCompare(b[1]) : a[0].localeCompare(b[0]),
        );
        for (const [schema, tname] of items) {
          rows.push({
            schemaname: schema,
            tablename: tname,
            tableowner: "",
            tablespace: "",
          });
        }
      } else {
        for (const tname of [...this._tables.keys()].sort()) {
          rows.push({
            schemaname: "public",
            tablename: tname,
            tableowner: "",
            tablespace: "",
          });
        }
      }
      return this._applyAlias(rows, alias);
    }
    if (viewName === "pg_views") {
      const rows: Record<string, unknown>[] = [];
      for (const vname of [...this._views.keys()].sort()) {
        rows.push({
          schemaname: "public",
          viewname: vname,
          viewowner: "",
          definition: "",
        });
      }
      return this._applyAlias(rows, alias);
    }
    if (viewName === "pg_indexes") {
      const rows: Record<string, unknown>[] = [];
      // If engine has index manager, enumerate indexes
      const eng = this._engine as Record<string, unknown> | null;
      const indexMgr =
        eng !== null
          ? (eng["_indexManager"] as Record<string, unknown> | undefined)
          : undefined;
      if (indexMgr !== undefined) {
        const indexes = indexMgr["_indexes"] as
          | Map<string, Record<string, unknown>>
          | undefined;
        if (indexes !== undefined) {
          for (const idx of indexes.values()) {
            const idxDef = idx["indexDef"] as Record<string, unknown> | undefined;
            if (idxDef !== undefined) {
              const tblName = (idxDef["tableName"] as string | undefined) ?? "";
              const idxName = (idxDef["name"] as string | undefined) ?? "";
              const cols = (idxDef["columns"] as string[] | undefined) ?? [];
              rows.push({
                schemaname: "public",
                tablename: tblName,
                indexname: idxName,
                tablespace: "",
                indexdef: `CREATE INDEX ${idxName} ON ${tblName} (${cols.join(", ")})`,
              });
            }
          }
        }
      }
      // Include indexes tracked by the compiler
      for (const idx of this._indexes.values()) {
        const methodStr = idx.method !== "btree" ? ` USING ${idx.method}` : "";
        rows.push({
          schemaname: "public",
          tablename: idx.tableName,
          indexname: idx.name,
          tablespace: "",
          indexdef: `CREATE INDEX ${idx.name} ON ${idx.tableName}${methodStr} (${idx.columns.join(", ")})`,
        });
      }
      return this._applyAlias(rows, alias);
    }
    if (viewName === "pg_type") {
      const typeEntries: [number, string, number, number, string, string][] = [
        [16, "boolean", 11, 1, "b", "B"],
        [17, "bytea", 11, -1, "b", "U"],
        [20, "bigint", 11, 8, "b", "N"],
        [21, "smallint", 11, 2, "b", "N"],
        [23, "integer", 11, 4, "b", "N"],
        [25, "text", 11, -1, "b", "S"],
        [114, "json", 11, -1, "b", "U"],
        [142, "xml", 11, -1, "b", "U"],
        [700, "real", 11, 4, "b", "N"],
        [701, "float8", 11, 8, "b", "N"],
        [1043, "varchar", 11, -1, "b", "S"],
        [1082, "date", 11, 4, "b", "D"],
        [1083, "time", 11, 8, "b", "D"],
        [1114, "timestamp", 11, 8, "b", "D"],
        [1184, "timestamptz", 11, 8, "b", "D"],
        [1186, "interval", 11, 16, "b", "T"],
        [1700, "numeric", 11, -1, "b", "N"],
        [2950, "uuid", 11, 16, "b", "U"],
        [3802, "jsonb", 11, -1, "b", "U"],
        [16385, "vector", 11, -1, "b", "U"],
      ];
      const rows: Record<string, unknown>[] = typeEntries.map(
        ([oid, typname, typnamespace, typlen, typtype, typcategory]) => ({
          oid,
          typname,
          typnamespace,
          typlen,
          typtype,
          typcategory,
        }),
      );
      return this._applyAlias(rows, alias);
    }
    throw new Error(`Unknown pg_catalog view: "${viewName}"`);
  }

  // -- Column projection -----------------------------------------------

  private _projectColumns(
    targetList: Record<string, unknown>[],
    rows: Record<string, unknown>[],
    evaluator: ExprEvaluator,
  ): [string[], Record<string, unknown>[]] {
    // Check for SELECT *
    if (targetList.length === 0) {
      if (rows.length === 0) return [[], []];
      const columns = Object.keys(rows[0]!).filter(
        (k) => !k.includes(".") && k !== "_doc_id" && k !== "_score",
      );
      return [
        columns,
        rows.map((r) => {
          const out: Record<string, unknown> = {};
          for (const c of columns) out[c] = r[c];
          return out;
        }),
      ];
    }

    const columns: string[] = [];
    const colExprs: { name: string; node: Record<string, unknown> | null }[] = [];

    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const val = nodeGet(resTarget, "val");
      const name = nodeStr(resTarget, "name"); // alias

      if (val !== null && val !== undefined) {
        const valObj = asObj(val);

        // Check for star expansion
        if (isAStar(valObj)) {
          // Check if qualified (table.*)
          const colRef = nodeGet(valObj, "ColumnRef");
          if (colRef !== null && colRef !== undefined) {
            const fields = asList(nodeGet(asObj(colRef), "fields"));
            if (fields.length >= 2) {
              // table.* expansion
              const tableAlias = extractString(fields[0]!);
              if (rows.length > 0) {
                for (const key of Object.keys(rows[0]!)) {
                  const prefix = `${tableAlias}.`;
                  if (key.startsWith(prefix)) {
                    const colName = key.slice(prefix.length);
                    if (!colName.includes(".")) {
                      columns.push(colName);
                      colExprs.push({ name: colName, node: null });
                    }
                  }
                }
              }
              continue;
            }
          }
          // SELECT * expansion (filter internal columns)
          if (rows.length > 0) {
            for (const key of Object.keys(rows[0]!)) {
              if (!key.includes(".") && key !== "_doc_id" && key !== "_score") {
                columns.push(key);
                colExprs.push({ name: key, node: null });
              }
            }
          }
          continue;
        }

        // Normal expression
        let colName = name;
        if (!colName) {
          colName = this._deriveColumnName(valObj);
        }
        columns.push(colName);
        colExprs.push({ name: colName, node: valObj });
      }
    }

    const projectedRows: Record<string, unknown>[] = [];
    for (const row of rows) {
      const projected: Record<string, unknown> = {};
      for (const expr of colExprs) {
        if (expr.node === null) {
          projected[expr.name] = row[expr.name];
        } else {
          projected[expr.name] = evaluator.evaluate(expr.node, row);
        }
      }
      projectedRows.push(projected);
    }

    return [columns, projectedRows];
  }

  private _deriveColumnName(node: Record<string, unknown>): string {
    // ColumnRef
    if (isColumnRef(node)) {
      try {
        return extractColumnName(node);
      } catch {
        // fall through
      }
    }

    // FuncCall
    if (isFuncCall(node)) {
      const fn = getFuncName(node);
      if (AGG_FUNC_NAMES.has(fn)) {
        const args = getFuncArgs(node);
        if (isAggStar(node) || args.length === 0) return fn;
        try {
          const argCol = extractColumnName(args[0]!);
          return `${fn}_${argCol}`;
        } catch {
          return fn;
        }
      }
      return fn;
    }

    // TypeCast
    if (isTypeCast(node)) {
      const tc = asObj(nodeGet(node, "TypeCast"));
      const arg = asObj(nodeGet(tc, "arg"));
      if (isColumnRef(arg)) {
        try {
          return extractColumnName(arg);
        } catch {
          // fall through
        }
      }
    }

    // SubLink (scalar subquery)
    if (isSubLink(node)) {
      return "?column?";
    }

    return "?column?";
  }

  private _resolveSelectColumnNames(
    targetList: Record<string, unknown>[],
    rows: Record<string, unknown>[],
  ): string[] {
    if (targetList.length === 0) {
      if (rows.length === 0) return [];
      return Object.keys(rows[0]!).filter((k) => !k.includes("."));
    }

    const columns: string[] = [];
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const val = nodeGet(resTarget, "val");
      const alias = nodeStr(resTarget, "name");

      if (val !== null && val !== undefined) {
        const valObj = asObj(val);
        if (isAStar(valObj)) {
          if (rows.length > 0) {
            for (const key of Object.keys(rows[0]!)) {
              if (!key.includes(".")) {
                columns.push(key);
              }
            }
          }
          continue;
        }
        columns.push(alias || this._deriveColumnName(valObj));
      }
    }
    return columns;
  }

  // -- GROUP BY -------------------------------------------------------

  private _hasAggregates(targetList: Record<string, unknown>[]): boolean {
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const val = nodeGet(resTarget, "val");
      if (val !== null && val !== undefined) {
        if (this._containsAggregate(asObj(val))) return true;
      }
    }
    return false;
  }

  private _containsAggregate(node: Record<string, unknown>): boolean {
    if (isFuncCall(node)) {
      const fn = getFuncName(node);
      if (AGG_FUNC_NAMES.has(fn) && !hasOverClause(node)) return true;
    }
    // Do not descend into SubLink (subquery) nodes -- aggregates inside
    // subqueries belong to the subquery, not the outer query.
    if (isSubLink(node) || nodeGet(node, "SubLink") !== null) return false;
    // Check nested function calls
    for (const [, value] of Object.entries(node)) {
      if (value !== null && typeof value === "object") {
        if (Array.isArray(value)) {
          for (const item of value) {
            if (item !== null && typeof item === "object") {
              if (this._containsAggregate(item as Record<string, unknown>)) return true;
            }
          }
        } else {
          if (this._containsAggregate(value as Record<string, unknown>)) return true;
        }
      }
    }
    return false;
  }

  private _applyGroupBy(
    rows: Record<string, unknown>[],
    groupClause: Record<string, unknown>[],
    targetList: Record<string, unknown>[],
    havingClause: unknown,
    evaluator: ExprEvaluator,
  ): Record<string, unknown>[] {
    // Compute group keys for each row
    const groups = new Map<string, Record<string, unknown>[]>();

    if (groupClause.length === 0) {
      // Aggregate without GROUP BY -- one group for all rows
      groups.set("__all__", rows);
    } else {
      for (const row of rows) {
        const keyParts: unknown[] = [];
        for (const groupExpr of groupClause) {
          const val = evaluator.evaluate(groupExpr, row);
          keyParts.push(val);
        }
        const key = JSON.stringify(keyParts);
        let group = groups.get(key);
        if (!group) {
          group = [];
          groups.set(key, group);
        }
        group.push(row);
      }
    }

    // For each group, compute aggregates
    const result: Record<string, unknown>[] = [];
    const groupRowsList: Record<string, unknown>[][] = [];
    for (const [, groupRows] of groups) {
      const aggregatedRow = this._computeAggregates(groupRows, targetList, evaluator);
      result.push(aggregatedRow);
      groupRowsList.push(groupRows);
    }

    // Apply HAVING
    if (havingClause !== null && havingClause !== undefined) {
      const filtered: Record<string, unknown>[] = [];
      for (let i = 0; i < result.length; i++) {
        const row = result[i]!;
        const groupRows = groupRowsList[i]!;
        // Resolve aggregate functions in HAVING clause using group rows
        const enrichedRow = this._resolveHavingAggregates(
          asObj(havingClause),
          groupRows,
          row,
          evaluator,
        );
        const condition = evaluator.evaluate(asObj(havingClause), enrichedRow);
        if (condition === true) {
          filtered.push(row);
        }
      }
      return filtered;
    }

    return result;
  }

  private _computeAggregates(
    groupRows: Record<string, unknown>[],
    targetList: Record<string, unknown>[],
    evaluator: ExprEvaluator,
  ): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    const firstRow = groupRows[0]!;

    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const val = nodeGet(resTarget, "val");
      const alias = nodeStr(resTarget, "name");

      if (val === null || val === undefined) continue;

      const valObj = asObj(val);

      if (isFuncCall(valObj)) {
        const funcName = getFuncName(valObj);
        const isStar = isAggStar(valObj);
        const argNodes = getFuncArgs(valObj);
        const distinct = isAggDistinct(valObj);
        const hasOver = hasOverClause(valObj);

        // Window functions are handled separately
        if (hasOver) {
          const colName = alias || funcName;
          result[colName] = evaluator.evaluate(valObj, firstRow);
          continue;
        }

        if (!AGG_FUNC_NAMES.has(funcName)) {
          // Not an aggregate -- but may contain aggregate sub-expressions
          // (e.g., ROUND(STDDEV(val), 2))
          const enrichedRow = { ...firstRow };
          this._collectHavingAggregates(valObj, groupRows, enrichedRow, evaluator);
          const colName = alias || this._deriveColumnName(valObj);
          result[colName] = evaluator.evaluate(valObj, enrichedRow);
          continue;
        }

        let colName = alias;
        if (!colName) {
          if (isStar || argNodes.length === 0) {
            colName = funcName;
          } else {
            try {
              const argCol = extractColumnName(argNodes[0]!);
              colName = `${funcName}_${argCol}`;
            } catch {
              colName = funcName;
            }
          }
        }

        let aggValue: unknown;

        // Get values, optionally with DISTINCT
        const getValues = (): unknown[] => {
          let values: unknown[] = [];
          for (const r of groupRows) {
            const v = evaluator.evaluate(argNodes[0]!, r);
            if (v !== null && v !== undefined) {
              values.push(v);
            }
          }
          if (distinct) {
            const seen = new Set<string>();
            const unique: unknown[] = [];
            for (const v of values) {
              const key = JSON.stringify(v);
              if (!seen.has(key)) {
                seen.add(key);
                unique.push(v);
              }
            }
            values = unique;
          }
          return values;
        };

        switch (funcName) {
          case "count":
            if (isStar) {
              aggValue = groupRows.length;
            } else {
              aggValue = getValues().length;
            }
            break;

          case "sum": {
            const values = getValues();
            aggValue =
              values.length > 0
                ? values.reduce((a: number, b) => a + Number(b), 0)
                : null;
            break;
          }

          case "avg": {
            const values = getValues();
            if (values.length > 0) {
              const sum = values.reduce((a: number, b) => a + Number(b), 0);
              aggValue = sum / values.length;
            } else {
              aggValue = null;
            }
            break;
          }

          case "min": {
            const values = getValues();
            if (values.length === 0) {
              aggValue = null;
            } else {
              aggValue = values.reduce((a, b) =>
                (a as number) < (b as number) ? a : b,
              );
            }
            break;
          }

          case "max": {
            const values = getValues();
            if (values.length === 0) {
              aggValue = null;
            } else {
              aggValue = values.reduce((a, b) =>
                (a as number) > (b as number) ? a : b,
              );
            }
            break;
          }

          case "string_agg": {
            const separator =
              argNodes.length >= 2
                ? String(evaluator.evaluate(argNodes[1]!, firstRow))
                : ",";
            const values: string[] = [];
            for (const r of groupRows) {
              const v = evaluator.evaluate(argNodes[0]!, r);
              if (v !== null && v !== undefined) {
                values.push(toStr(v));
              }
            }
            aggValue = values.length > 0 ? values.join(separator) : null;
            break;
          }

          case "array_agg": {
            const values: unknown[] = [];
            for (const r of groupRows) {
              values.push(evaluator.evaluate(argNodes[0]!, r));
            }
            aggValue = values;
            break;
          }

          case "bool_and": {
            const values = getValues();
            aggValue = values.length > 0 ? values.every((v) => Boolean(v)) : null;
            break;
          }

          case "bool_or": {
            const values = getValues();
            aggValue = values.length > 0 ? values.some((v) => Boolean(v)) : null;
            break;
          }

          case "stddev":
          case "stddev_samp": {
            const values = getValues().map(Number);
            if (values.length < 2) {
              aggValue = null;
            } else {
              const mean = values.reduce((a, b) => a + b, 0) / values.length;
              const variance =
                values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length - 1);
              aggValue = Math.sqrt(variance);
            }
            break;
          }

          case "stddev_pop": {
            const values = getValues().map(Number);
            if (values.length === 0) {
              aggValue = null;
            } else {
              const mean = values.reduce((a, b) => a + b, 0) / values.length;
              const variance =
                values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
              aggValue = Math.sqrt(variance);
            }
            break;
          }

          case "variance":
          case "var_samp": {
            const values = getValues().map(Number);
            if (values.length < 2) {
              aggValue = null;
            } else {
              const mean = values.reduce((a, b) => a + b, 0) / values.length;
              aggValue =
                values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length - 1);
            }
            break;
          }

          case "var_pop": {
            const values = getValues().map(Number);
            if (values.length === 0) {
              aggValue = null;
            } else {
              const mean = values.reduce((a, b) => a + b, 0) / values.length;
              aggValue =
                values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
            }
            break;
          }

          case "json_object_agg":
          case "jsonb_object_agg": {
            if (argNodes.length < 2) {
              aggValue = null;
            } else {
              const obj: Record<string, unknown> = {};
              for (const r of groupRows) {
                const k = evaluator.evaluate(argNodes[0]!, r);
                const v = evaluator.evaluate(argNodes[1]!, r);
                if (k !== null && k !== undefined) {
                  obj[String(k as string | number)] = v;
                }
              }
              aggValue = obj;
            }
            break;
          }

          case "percentile_cont": {
            // percentile_cont(fraction) WITHIN GROUP (ORDER BY col)
            if (argNodes.length > 0) {
              const fraction = Number(evaluator.evaluate(argNodes[0]!, firstRow));
              const fc = asObj(nodeGet(valObj, "FuncCall") ?? valObj);
              const aggOrder = asList(nodeGet(fc, "agg_order"));
              if (aggOrder.length > 0) {
                const orderNode = asObj(
                  nodeGet(aggOrder[0]!, "SortBy") ?? aggOrder[0]!,
                );
                const orderCol = asObj(nodeGet(orderNode, "node"));
                const values: number[] = [];
                for (const r of groupRows) {
                  const v = evaluator.evaluate(orderCol, r);
                  if (v !== null && v !== undefined) values.push(Number(v));
                }
                values.sort((a, b) => a - b);
                if (values.length === 0) {
                  aggValue = null;
                } else {
                  const idx = fraction * (values.length - 1);
                  const lower = Math.floor(idx);
                  const upper = Math.ceil(idx);
                  if (lower === upper) {
                    aggValue = values[lower]!;
                  } else {
                    aggValue =
                      values[lower]! +
                      (idx - lower) * (values[upper]! - values[lower]!);
                  }
                }
              } else {
                aggValue = null;
              }
            } else {
              aggValue = null;
            }
            break;
          }

          case "mode": {
            const fc = asObj(nodeGet(valObj, "FuncCall") ?? valObj);
            const aggOrder = asList(nodeGet(fc, "agg_order"));
            if (aggOrder.length > 0) {
              const orderNode = asObj(nodeGet(aggOrder[0]!, "SortBy") ?? aggOrder[0]!);
              const orderCol = asObj(nodeGet(orderNode, "node"));
              const freq = new Map<string, { count: number; value: unknown }>();
              for (const r of groupRows) {
                const v = evaluator.evaluate(orderCol, r);
                const key = JSON.stringify(v);
                const entry = freq.get(key);
                if (entry) {
                  entry.count++;
                } else {
                  freq.set(key, { count: 1, value: v });
                }
              }
              let maxCount = 0;
              let modeValue: unknown = null;
              for (const entry of freq.values()) {
                if (entry.count > maxCount) {
                  maxCount = entry.count;
                  modeValue = entry.value;
                }
              }
              aggValue = modeValue;
            } else {
              aggValue = null;
            }
            break;
          }

          case "corr":
          case "covar_pop":
          case "covar_samp":
          case "regr_count":
          case "regr_avgx":
          case "regr_avgy":
          case "regr_sxx":
          case "regr_syy":
          case "regr_sxy":
          case "regr_slope":
          case "regr_intercept":
          case "regr_r2": {
            // Two-argument statistical aggregates
            if (argNodes.length < 2) {
              aggValue = null;
              break;
            }
            const yVals: number[] = [];
            const xVals: number[] = [];
            for (const r of groupRows) {
              const y = evaluator.evaluate(argNodes[0]!, r);
              const x = evaluator.evaluate(argNodes[1]!, r);
              if (y !== null && y !== undefined && x !== null && x !== undefined) {
                yVals.push(Number(y));
                xVals.push(Number(x));
              }
            }
            aggValue = this._computeStatAgg(funcName, yVals, xVals);
            break;
          }

          default:
            aggValue = evaluator.evaluate(valObj, firstRow);
            break;
        }

        result[colName] = aggValue;
      } else {
        // Non-aggregate expression -- use value from first row
        const colName = alias || this._deriveColumnName(valObj);
        result[colName] = evaluator.evaluate(valObj, firstRow);
      }
    }

    return result;
  }

  private _computeStatAgg(funcName: string, yVals: number[], xVals: number[]): unknown {
    const n = yVals.length;
    if (n === 0) return null;

    const sumX = xVals.reduce((a, b) => a + b, 0);
    const sumY = yVals.reduce((a, b) => a + b, 0);
    const avgX = sumX / n;
    const avgY = sumY / n;

    let sxx = 0;
    let syy = 0;
    let sxy = 0;
    for (let i = 0; i < n; i++) {
      const dx = xVals[i]! - avgX;
      const dy = yVals[i]! - avgY;
      sxx += dx * dx;
      syy += dy * dy;
      sxy += dx * dy;
    }

    switch (funcName) {
      case "regr_count":
        return n;
      case "regr_avgx":
        return avgX;
      case "regr_avgy":
        return avgY;
      case "regr_sxx":
        return sxx;
      case "regr_syy":
        return syy;
      case "regr_sxy":
        return sxy;
      case "covar_pop":
        return sxy / n;
      case "covar_samp":
        return n < 2 ? null : sxy / (n - 1);
      case "corr": {
        if (sxx === 0 || syy === 0) return null;
        return sxy / Math.sqrt(sxx * syy);
      }
      case "regr_slope": {
        if (sxx === 0) return null;
        return sxy / sxx;
      }
      case "regr_intercept": {
        if (sxx === 0) return null;
        return avgY - (sxy / sxx) * avgX;
      }
      case "regr_r2": {
        if (sxx === 0 || syy === 0) return null;
        const r = sxy / Math.sqrt(sxx * syy);
        return r * r;
      }
      default:
        return null;
    }
  }

  // -- HAVING aggregate resolution ----------------------------------------

  /**
   * Walk the HAVING expression tree and resolve any aggregate function calls
   * (COUNT, SUM, AVG, etc.) by computing them from the group rows. The
   * computed values are injected into the enriched row under a synthetic key
   * so the normal expression evaluator can look them up.
   */
  private _resolveHavingAggregates(
    expr: Record<string, unknown>,
    groupRows: Record<string, unknown>[],
    baseRow: Record<string, unknown>,
    evaluator: ExprEvaluator,
  ): Record<string, unknown> {
    const enriched = { ...baseRow };
    this._collectHavingAggregates(expr, groupRows, enriched, evaluator);
    return enriched;
  }

  private _collectHavingAggregates(
    node: Record<string, unknown>,
    groupRows: Record<string, unknown>[],
    enriched: Record<string, unknown>,
    evaluator: ExprEvaluator,
  ): void {
    // Check if this node is a FuncCall
    const funcCallNode = node["FuncCall"] as Record<string, unknown> | undefined;
    if (funcCallNode) {
      const funcNameParts = funcCallNode["funcname"] as unknown[];
      if (funcNameParts.length > 0) {
        const lastPart = funcNameParts[funcNameParts.length - 1] as Record<
          string,
          unknown
        >;
        const strNode = lastPart["String"] ?? lastPart["str"];
        const name = (
          typeof strNode === "object" && strNode !== null
            ? ((strNode as Record<string, unknown>)["sval"] ??
              (strNode as Record<string, unknown>)["str"])
            : strNode
        ) as string | undefined;
        if (name && AGG_FUNC_NAMES.has(name.toLowerCase())) {
          // Compute the aggregate on group rows
          const aggResult = this._computeInlineAggregate(
            name.toLowerCase(),
            funcCallNode,
            groupRows,
            evaluator,
          );
          // Store under the string representation of the aggregate call
          // Build a synthetic key that the evaluator will return when it encounters this FuncCall
          // We override by adding to the row with the funcname as key
          const isStar = isAggStar(node);
          const args = getFuncArgs(node);
          let key: string;
          if (isStar || args.length === 0) {
            key = `${name.toLowerCase()}(*)`;
          } else {
            try {
              const argCol = extractColumnName(args[0]!);
              key = `${name.toLowerCase()}(${argCol})`;
            } catch {
              key = name.toLowerCase();
            }
          }
          enriched[key] = aggResult;
          // Also store under common aliases the evaluator might generate
          enriched[`__having_agg_${JSON.stringify(node)}`] = aggResult;
          return;
        }
      }
    }

    // Check BoolExpr children (AND/OR)
    const boolExpr = node["BoolExpr"] as Record<string, unknown> | undefined;
    if (boolExpr) {
      const args = asList(boolExpr["args"]);
      for (const arg of args) {
        this._collectHavingAggregates(asObj(arg), groupRows, enriched, evaluator);
      }
      return;
    }

    // Check A_Expr children (comparison like > < =)
    const aExpr = node["A_Expr"] as Record<string, unknown> | undefined;
    if (aExpr) {
      const lexpr = aExpr["lexpr"];
      const rexpr = aExpr["rexpr"];
      if (lexpr) {
        this._collectHavingAggregates(asObj(lexpr), groupRows, enriched, evaluator);
      }
      if (rexpr) {
        this._collectHavingAggregates(asObj(rexpr), groupRows, enriched, evaluator);
      }
      return;
    }

    // Recurse into all object values and arrays
    for (const v of Object.values(node)) {
      if (Array.isArray(v)) {
        for (const item of v) {
          if (typeof item === "object" && item !== null) {
            this._collectHavingAggregates(
              item as Record<string, unknown>,
              groupRows,
              enriched,
              evaluator,
            );
          }
        }
      } else if (typeof v === "object" && v !== null) {
        this._collectHavingAggregates(
          v as Record<string, unknown>,
          groupRows,
          enriched,
          evaluator,
        );
      }
    }
  }

  private _computeInlineAggregate(
    funcName: string,
    funcCallNode: Record<string, unknown>,
    groupRows: Record<string, unknown>[],
    evaluator: ExprEvaluator,
  ): unknown {
    const parentNode = { FuncCall: funcCallNode } as Record<string, unknown>;
    const isStar = isAggStar(parentNode);
    const argNodes = getFuncArgs(parentNode);
    const distinct = isAggDistinct(parentNode);

    const getValues = (): unknown[] => {
      let values: unknown[] = [];
      for (const r of groupRows) {
        const v = argNodes.length > 0 ? evaluator.evaluate(argNodes[0]!, r) : null;
        if (v !== null && v !== undefined) {
          values.push(v);
        }
      }
      if (distinct) {
        const seen = new Set<string>();
        const unique: unknown[] = [];
        for (const v of values) {
          const key = JSON.stringify(v);
          if (!seen.has(key)) {
            seen.add(key);
            unique.push(v);
          }
        }
        values = unique;
      }
      return values;
    };

    // Two-argument aggregate helpers
    const getPairedValues = (): [number, number][] => {
      const pairs: [number, number][] = [];
      if (argNodes.length < 2) return pairs;
      for (const r of groupRows) {
        const v1 = evaluator.evaluate(argNodes[0]!, r);
        const v2 = evaluator.evaluate(argNodes[1]!, r);
        if (v1 !== null && v1 !== undefined && v2 !== null && v2 !== undefined) {
          pairs.push([Number(v1), Number(v2)]);
        }
      }
      return pairs;
    };

    switch (funcName) {
      case "count":
        return isStar ? groupRows.length : getValues().length;
      case "sum": {
        const values = getValues();
        return values.length > 0
          ? values.reduce((a: number, b) => a + Number(b), 0)
          : null;
      }
      case "avg": {
        const values = getValues();
        return values.length > 0
          ? values.reduce((a: number, b) => a + Number(b), 0) / values.length
          : null;
      }
      case "min": {
        const values = getValues();
        return values.length > 0
          ? values.reduce((a, b) => (Number(a) < Number(b) ? a : b))
          : null;
      }
      case "max": {
        const values = getValues();
        return values.length > 0
          ? values.reduce((a, b) => (Number(a) > Number(b) ? a : b))
          : null;
      }
      case "stddev":
      case "stddev_samp": {
        const values = getValues().map(Number);
        if (values.length < 2) return null;
        const avg = values.reduce((a, b) => a + b, 0) / values.length;
        const variance =
          values.reduce((a, b) => a + (b - avg) ** 2, 0) / (values.length - 1);
        return Math.sqrt(variance);
      }
      case "stddev_pop": {
        const values = getValues().map(Number);
        if (values.length === 0) return null;
        const avg = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + (b - avg) ** 2, 0) / values.length;
        return Math.sqrt(variance);
      }
      case "variance":
      case "var_samp": {
        const values = getValues().map(Number);
        if (values.length < 2) return null;
        const avg = values.reduce((a, b) => a + b, 0) / values.length;
        return values.reduce((a, b) => a + (b - avg) ** 2, 0) / (values.length - 1);
      }
      case "var_pop": {
        const values = getValues().map(Number);
        if (values.length === 0) return null;
        const avg = values.reduce((a, b) => a + b, 0) / values.length;
        return values.reduce((a, b) => a + (b - avg) ** 2, 0) / values.length;
      }
      case "corr":
      case "covar_pop":
      case "covar_samp":
      case "regr_slope":
      case "regr_intercept":
      case "regr_r2":
      case "regr_count":
      case "regr_avgx":
      case "regr_avgy":
      case "regr_sxx":
      case "regr_syy":
      case "regr_sxy": {
        const pairs = getPairedValues();
        if (pairs.length === 0) return null;
        const n = pairs.length;
        const avgY = pairs.reduce((a, p) => a + p[0], 0) / n;
        const avgX = pairs.reduce((a, p) => a + p[1], 0) / n;
        let sxx = 0,
          syy = 0,
          sxy = 0;
        for (const [y, x] of pairs) {
          sxx += (x - avgX) ** 2;
          syy += (y - avgY) ** 2;
          sxy += (x - avgX) * (y - avgY);
        }
        switch (funcName) {
          case "corr":
            return sxx === 0 || syy === 0 ? null : sxy / Math.sqrt(sxx * syy);
          case "covar_pop":
            return sxy / n;
          case "covar_samp":
            return n < 2 ? null : sxy / (n - 1);
          case "regr_slope":
            return sxx === 0 ? null : sxy / sxx;
          case "regr_intercept":
            return sxx === 0 ? null : avgY - (sxy / sxx) * avgX;
          case "regr_r2":
            return sxx === 0 || syy === 0 ? null : (sxy * sxy) / (sxx * syy);
          case "regr_count":
            return n;
          case "regr_avgx":
            return avgX;
          case "regr_avgy":
            return avgY;
          case "regr_sxx":
            return sxx;
          case "regr_syy":
            return syy;
          case "regr_sxy":
            return sxy;
          default:
            return null;
        }
      }
      default:
        return null;
    }
  }

  // -- Window functions -----------------------------------------------

  private _hasWindowFunctions(targetList: Record<string, unknown>[]): boolean {
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const val = nodeGet(resTarget, "val");
      if (val !== null && val !== undefined) {
        if (isFuncCall(asObj(val)) && hasOverClause(asObj(val))) {
          return true;
        }
      }
    }
    return false;
  }

  private _applyWindowFunctions(
    rows: Record<string, unknown>[],
    targetList: Record<string, unknown>[],
    windowClause: Record<string, unknown>[],
    evaluator: ExprEvaluator,
  ): Record<string, unknown>[] {
    // Build named window lookup
    const namedWindows = new Map<string, Record<string, unknown>>();
    for (const wdef of windowClause) {
      const wObj = asObj(nodeGet(wdef, "WindowDef") ?? wdef);
      const wName = nodeStr(wObj, "name");
      if (wName) namedWindows.set(wName, wObj);
    }

    // Process each window function in the target list
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const val = nodeGet(resTarget, "val");
      const alias = nodeStr(resTarget, "name");

      if (val === null || val === undefined) continue;
      const valObj = asObj(val);
      if (!isFuncCall(valObj) || !hasOverClause(valObj)) continue;

      const funcName = getFuncName(valObj);
      const outputName = alias || funcName;
      const fc = asObj(nodeGet(valObj, "FuncCall") ?? valObj);
      const argNodes = getFuncArgs(valObj);
      const win = asObj(nodeGet(fc, "over"));

      // Resolve named window reference
      const refName = nodeStr(win, "refname") || nodeStr(win, "name");
      let resolvedWin = win;
      if (refName && namedWindows.has(refName)) {
        resolvedWin = namedWindows.get(refName)!;
      }

      // Partition columns
      const partCols: string[] = [];
      for (const p of asList(nodeGet(resolvedWin, "partitionClause"))) {
        try {
          partCols.push(extractColumnName(p));
        } catch {
          // expression in PARTITION BY -- evaluate and use result
        }
      }

      // Order keys
      const orderKeys: { col: string; desc: boolean }[] = [];
      for (const s of asList(nodeGet(resolvedWin, "orderClause"))) {
        const sortBy = asObj(nodeGet(s, "SortBy") ?? s);
        const sortNode = asObj(nodeGet(sortBy, "node"));
        try {
          const col = extractColumnName(sortNode);
          const sortByDir = nodeGet(sortBy, "sortby_dir");
          const desc = sortByDir === 2 || sortByDir === "SORTBY_DESC";
          orderKeys.push({ col, desc });
        } catch {
          // skip complex expressions
        }
      }

      // Partition the rows
      const partitions = new Map<
        string,
        { rows: Record<string, unknown>[]; indices: number[] }
      >();
      for (let i = 0; i < rows.length; i++) {
        const row = rows[i]!;
        const key = partCols.map((c) => JSON.stringify(row[c])).join("\0");
        let part = partitions.get(key);
        if (!part) {
          part = { rows: [], indices: [] };
          partitions.set(key, part);
        }
        part.rows.push(row);
        part.indices.push(i);
      }

      // Sort within each partition
      for (const part of partitions.values()) {
        if (orderKeys.length > 0) {
          const indexed = part.rows.map((row, idx) => ({
            row,
            origIdx: part.indices[idx]!,
          }));
          indexed.sort((a, b) => {
            for (const ok of orderKeys) {
              const va = a.row[ok.col];
              const vb = b.row[ok.col];
              const aNull = va === null || va === undefined;
              const bNull = vb === null || vb === undefined;
              if (aNull && bNull) continue;
              if (aNull) return ok.desc ? -1 : 1;
              if (bNull) return ok.desc ? 1 : -1;
              let cmp: number;
              if (typeof va === "string" && typeof vb === "string") {
                cmp = va < vb ? -1 : va > vb ? 1 : 0;
              } else {
                cmp = (va as number) - (vb as number);
              }
              if (cmp !== 0) return ok.desc ? -cmp : cmp;
            }
            return 0;
          });
          part.rows = indexed.map((x) => x.row);
          part.indices = indexed.map((x) => x.origIdx);
        }
      }

      // Compute window function values
      for (const part of partitions.values()) {
        const partRows = part.rows;
        const partSize = partRows.length;

        for (let i = 0; i < partSize; i++) {
          let value: unknown = null;

          switch (funcName) {
            case "row_number":
              value = i + 1;
              break;

            case "rank": {
              // Same rank for same values, with gaps
              let rank = 1;
              for (let j = 0; j < i; j++) {
                let same = true;
                for (const ok of orderKeys) {
                  if (partRows[j]![ok.col] !== partRows[i]![ok.col]) {
                    same = false;
                    break;
                  }
                }
                if (!same) rank = j + 1 + 1;
              }
              // Recalculate: rank = number of rows with strictly smaller order values + 1
              rank = 1;
              for (let j = 0; j < i; j++) {
                let same = true;
                for (const ok of orderKeys) {
                  if (partRows[j]![ok.col] !== partRows[i]![ok.col]) {
                    same = false;
                    break;
                  }
                }
                if (!same) rank++;
              }
              // Standard rank: position of first occurrence
              rank = 1;
              for (let j = 0; j < partSize; j++) {
                let same = true;
                for (const ok of orderKeys) {
                  if (partRows[j]![ok.col] !== partRows[i]![ok.col]) {
                    same = false;
                    break;
                  }
                }
                if (same) {
                  rank = j + 1;
                  break;
                }
              }
              value = rank;
              break;
            }

            case "dense_rank": {
              const uniqueKeys = new Set<string>();
              for (let j = 0; j <= i; j++) {
                const key = orderKeys
                  .map((ok) => JSON.stringify(partRows[j]![ok.col]))
                  .join("\0");
                uniqueKeys.add(key);
              }
              value = uniqueKeys.size;
              break;
            }

            case "percent_rank": {
              if (partSize <= 1) {
                value = 0;
              } else {
                // rank - 1 / (partSize - 1)
                let rank = 1;
                for (let j = 0; j < partSize; j++) {
                  let same = true;
                  for (const ok of orderKeys) {
                    if (partRows[j]![ok.col] !== partRows[i]![ok.col]) {
                      same = false;
                      break;
                    }
                  }
                  if (same) {
                    rank = j + 1;
                    break;
                  }
                }
                value = (rank - 1) / (partSize - 1);
              }
              break;
            }

            case "cume_dist": {
              // Count rows with order value <= current
              let count = 0;
              for (let j = 0; j < partSize; j++) {
                let leq = true;
                for (const ok of orderKeys) {
                  const va = partRows[j]![ok.col];
                  const vb = partRows[i]![ok.col];
                  if (!ok.desc) {
                    if ((va as number) > (vb as number)) leq = false;
                  } else {
                    if ((va as number) < (vb as number)) leq = false;
                  }
                }
                if (leq) count++;
              }
              value = count / partSize;
              break;
            }

            case "ntile": {
              const buckets =
                argNodes.length > 0 ? Number(evaluator.evaluate(argNodes[0]!, {})) : 1;
              value = Math.floor((i * buckets) / partSize) + 1;
              break;
            }

            case "lag": {
              const offset =
                argNodes.length > 1 ? Number(evaluator.evaluate(argNodes[1]!, {})) : 1;
              const defaultVal =
                argNodes.length > 2 ? evaluator.evaluate(argNodes[2]!, {}) : null;
              const lagIdx = i - offset;
              if (lagIdx >= 0 && lagIdx < partSize && argNodes.length > 0) {
                value = evaluator.evaluate(argNodes[0]!, partRows[lagIdx]!);
              } else {
                value = defaultVal;
              }
              break;
            }

            case "lead": {
              const offset =
                argNodes.length > 1 ? Number(evaluator.evaluate(argNodes[1]!, {})) : 1;
              const defaultVal =
                argNodes.length > 2 ? evaluator.evaluate(argNodes[2]!, {}) : null;
              const leadIdx = i + offset;
              if (leadIdx >= 0 && leadIdx < partSize && argNodes.length > 0) {
                value = evaluator.evaluate(argNodes[0]!, partRows[leadIdx]!);
              } else {
                value = defaultVal;
              }
              break;
            }

            case "first_value": {
              if (argNodes.length > 0 && partSize > 0) {
                value = evaluator.evaluate(argNodes[0]!, partRows[0]!);
              }
              break;
            }

            case "last_value": {
              if (argNodes.length > 0 && partSize > 0) {
                value = evaluator.evaluate(argNodes[0]!, partRows[partSize - 1]!);
              }
              break;
            }

            case "nth_value": {
              const nth =
                argNodes.length > 1 ? Number(evaluator.evaluate(argNodes[1]!, {})) : 1;
              if (argNodes.length > 0 && nth >= 1 && nth <= partSize) {
                value = evaluator.evaluate(argNodes[0]!, partRows[nth - 1]!);
              }
              break;
            }

            // Aggregate window functions
            // When ORDER BY is present, default frame is ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            // When ORDER BY is absent, default frame is the whole partition
            case "sum": {
              let sum = 0;
              let hasVal = false;
              const frameEnd = orderKeys.length > 0 ? i : partSize - 1;
              for (let j = 0; j <= frameEnd; j++) {
                const v =
                  argNodes.length > 0
                    ? evaluator.evaluate(argNodes[0]!, partRows[j]!)
                    : null;
                if (v !== null && v !== undefined) {
                  sum += Number(v);
                  hasVal = true;
                }
              }
              value = hasVal ? sum : null;
              break;
            }

            case "count": {
              const frameEnd = orderKeys.length > 0 ? i : partSize - 1;
              if (isAggStar(valObj)) {
                value = frameEnd + 1;
              } else {
                let count = 0;
                for (let j = 0; j <= frameEnd; j++) {
                  const v =
                    argNodes.length > 0
                      ? evaluator.evaluate(argNodes[0]!, partRows[j]!)
                      : null;
                  if (v !== null && v !== undefined) count++;
                }
                value = count;
              }
              break;
            }

            case "avg": {
              let sum = 0;
              let count = 0;
              const frameEndAvg = orderKeys.length > 0 ? i : partSize - 1;
              for (let j = 0; j <= frameEndAvg; j++) {
                const v =
                  argNodes.length > 0
                    ? evaluator.evaluate(argNodes[0]!, partRows[j]!)
                    : null;
                if (v !== null && v !== undefined) {
                  sum += Number(v);
                  count++;
                }
              }
              value = count > 0 ? sum / count : null;
              break;
            }

            case "min": {
              let minVal: unknown = null;
              const frameEndMin = orderKeys.length > 0 ? i : partSize - 1;
              for (let j = 0; j <= frameEndMin; j++) {
                const v =
                  argNodes.length > 0
                    ? evaluator.evaluate(argNodes[0]!, partRows[j]!)
                    : null;
                if (v !== null && v !== undefined) {
                  if (minVal === null || (v as number) < (minVal as number)) {
                    minVal = v;
                  }
                }
              }
              value = minVal;
              break;
            }

            case "max": {
              let maxVal: unknown = null;
              const frameEndMax = orderKeys.length > 0 ? i : partSize - 1;
              for (let j = 0; j <= frameEndMax; j++) {
                const v =
                  argNodes.length > 0
                    ? evaluator.evaluate(argNodes[0]!, partRows[j]!)
                    : null;
                if (v !== null && v !== undefined) {
                  if (maxVal === null || (v as number) > (maxVal as number)) {
                    maxVal = v;
                  }
                }
              }
              value = maxVal;
              break;
            }

            default:
              value = null;
              break;
          }

          // Write value back to the row in the original rows array
          const origIdx = part.indices[i]!;
          rows[origIdx]![outputName] = value;
        }
      }
    }

    // Also evaluate non-window targets
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const val = nodeGet(resTarget, "val");
      const alias = nodeStr(resTarget, "name");

      if (val === null || val === undefined) continue;
      const valObj = asObj(val);
      if (isFuncCall(valObj) && hasOverClause(valObj)) continue; // already handled

      const colName = alias || this._deriveColumnName(valObj);
      if (isColumnRef(valObj)) {
        // Simple column reference -- already in rows
        const srcCol = extractColumnName(valObj);
        if (srcCol !== colName) {
          for (const row of rows) {
            row[colName] = row[srcCol];
          }
        }
      } else {
        // Computed expression
        for (const row of rows) {
          row[colName] = evaluator.evaluate(valObj, row);
        }
      }
    }

    return rows;
  }

  // -- Faceted search -------------------------------------------------

  /**
   * Detect uqa_facets() in SELECT and return facet rows.
   *
   * Returns null when the SELECT list does not contain
   * uqa_facets(), allowing normal execution to proceed.
   */
  private _tryFacets(
    targetList: Record<string, unknown>[],
    rows: Record<string, unknown>[],
  ): SQLResult | null {
    if (targetList.length === 0) {
      return null;
    }

    const facetFields: string[] = [];
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const val = nodeGet(resTarget, "val");
      if (val === null || val === undefined) continue;
      const valObj = asObj(val);
      const fc = nodeGet(valObj, "FuncCall");
      if (fc === null || fc === undefined) continue;
      const fn = getFuncName(valObj);
      if (fn === "uqa_facets") {
        const args = getFuncArgs(valObj);
        for (const arg of args) {
          facetFields.push(extractColumnName(arg));
        }
      }
    }

    if (facetFields.length === 0) {
      return null;
    }

    const allRows: Record<string, unknown>[] = [];
    const multi = facetFields.length > 1;
    for (const fieldName of facetFields) {
      const valueCounts = new Map<string, number>();
      for (const row of rows) {
        const value = row[fieldName];
        if (value !== null && value !== undefined) {
          const key = String(value);
          valueCounts.set(key, (valueCounts.get(key) ?? 0) + 1);
        }
      }
      const sorted = [...valueCounts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
      for (const [value, count] of sorted) {
        const facetRow: Record<string, unknown> = {};
        if (multi) {
          facetRow["facet_field"] = fieldName;
        }
        facetRow["facet_value"] = value;
        facetRow["facet_count"] = count;
        allRows.push(facetRow);
      }
    }

    const columns = multi
      ? ["facet_field", "facet_value", "facet_count"]
      : ["facet_value", "facet_count"];
    return { columns, rows: allRows };
  }

  // -- DISTINCT -------------------------------------------------------

  private _applyDistinct(
    rows: Record<string, unknown>[],
    columns: string[],
  ): Record<string, unknown>[] {
    const seen = new Set<string>();
    const result: Record<string, unknown>[] = [];
    for (const row of rows) {
      const key = columns.map((c) => JSON.stringify(row[c])).join("\0");
      if (!seen.has(key)) {
        seen.add(key);
        result.push(row);
      }
    }
    return result;
  }

  // -- ORDER BY -------------------------------------------------------

  private _applyOrderBy(
    rows: Record<string, unknown>[],
    sortClause: Record<string, unknown>[],
    evaluator: ExprEvaluator,
    targetList?: Record<string, unknown>[],
  ): Record<string, unknown>[] {
    const sorted = [...rows];

    // Build ordinal and alias maps for resolution
    const ordinalMap = new Map<number, string>();
    const aliasNames = new Set<string>();
    const originalToAlias = new Map<string, string>();
    if (targetList) {
      for (let idx = 0; idx < targetList.length; idx++) {
        const resTarget = asObj(
          nodeGet(targetList[idx]!, "ResTarget") ?? targetList[idx]!,
        );
        const alias = nodeStr(resTarget, "name");
        const val = nodeGet(resTarget, "val");
        let colName = alias;
        if (!colName && val !== null && val !== undefined) {
          colName = this._deriveColumnName(asObj(val));
        }
        if (colName) {
          ordinalMap.set(idx + 1, colName);
          if (alias) {
            aliasNames.add(alias);
            if (val !== null && isColumnRef(asObj(val))) {
              try {
                const realCol = extractColumnName(asObj(val));
                originalToAlias.set(realCol, alias);
              } catch {
                // skip
              }
            }
          }
        }
      }
    }

    const sortSpecs = sortClause.map((item) => {
      const sortBy = asObj(nodeGet(item, "SortBy") ?? item);
      const sortNode = asObj(nodeGet(sortBy, "node"));
      const sortByDir = nodeGet(sortBy, "sortby_dir");
      const desc = sortByDir === 2 || sortByDir === "SORTBY_DESC";
      const sortByNulls = nodeGet(sortBy, "sortby_nulls");
      let nullsFirst: boolean;
      if (sortByNulls === 1 || sortByNulls === "SORTBY_NULLS_FIRST") {
        nullsFirst = true;
      } else if (sortByNulls === 2 || sortByNulls === "SORTBY_NULLS_LAST") {
        nullsFirst = false;
      } else {
        // PostgreSQL default: NULLS FIRST for DESC, NULLS LAST for ASC
        nullsFirst = desc;
      }

      // Check for ordinal reference (ORDER BY 1, 2, ...)
      let resolvedNode = sortNode;
      if (isAConst(sortNode)) {
        const val = extractConstValue(sortNode, []);
        if (typeof val === "number" && Number.isInteger(val)) {
          const ordinal = val;
          const col = ordinalMap.get(ordinal);
          if (col !== undefined) {
            // Create a ColumnRef-like node for this column
            resolvedNode = { ColumnRef: { fields: [{ String: { sval: col } }] } };
          }
        }
      }

      // If the sort column references an original column that was aliased
      // (e.g., ORDER BY salary when SELECT salary AS top_sal), resolve to the alias
      if (isColumnRef(resolvedNode)) {
        try {
          const colName = extractColumnName(resolvedNode);
          if (originalToAlias.has(colName)) {
            const aliasName = originalToAlias.get(colName)!;
            resolvedNode = { ColumnRef: { fields: [{ String: { sval: aliasName } }] } };
          }
        } catch {
          // ignore extraction errors
        }
      }

      return { node: resolvedNode, desc, nullsFirst };
    });

    sorted.sort((a, b) => {
      for (const spec of sortSpecs) {
        const va = evaluator.evaluate(spec.node, a);
        const vb = evaluator.evaluate(spec.node, b);

        // Handle nulls
        const aNull = va === null || va === undefined;
        const bNull = vb === null || vb === undefined;
        if (aNull && bNull) continue;
        if (aNull) return spec.nullsFirst ? -1 : 1;
        if (bNull) return spec.nullsFirst ? 1 : -1;

        let cmp: number;
        if (typeof va === "string" && typeof vb === "string") {
          cmp = va < vb ? -1 : va > vb ? 1 : 0;
        } else {
          cmp = (va as number) - (vb as number);
        }

        if (cmp !== 0) {
          return spec.desc ? -cmp : cmp;
        }
      }
      return 0;
    });

    return sorted;
  }

  // -- RETURNING clause -----------------------------------------------

  private _evaluateReturning(
    returningCols: Record<string, unknown>[],
    row: Record<string, unknown>,
    evaluator: ExprEvaluator,
    table?: Table,
  ): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    for (const col of returningCols) {
      const resTarget = asObj(nodeGet(col, "ResTarget") ?? col);
      const val = nodeGet(resTarget, "val");
      const alias = nodeStr(resTarget, "name");

      if (val !== null && val !== undefined) {
        const valObj = asObj(val);
        // Check for *
        if (isAStar(valObj)) {
          if (table) {
            for (const colName of table.columnNames) {
              result[colName] = row[colName] ?? null;
            }
          } else {
            Object.assign(result, row);
          }
          continue;
        }

        const colName = alias || this._deriveColumnName(valObj);
        result[colName] = evaluator.evaluate(valObj, row);
      }
    }
    return result;
  }

  private _extractReturningColumns(
    returningCols: Record<string, unknown>[],
    table?: Table,
  ): string[] {
    const columns: string[] = [];
    for (const col of returningCols) {
      const resTarget = asObj(nodeGet(col, "ResTarget") ?? col);
      const val = nodeGet(resTarget, "val");
      const alias = nodeStr(resTarget, "name");

      if (val !== null && val !== undefined) {
        const valObj = asObj(val);
        if (isAStar(valObj)) {
          if (table) {
            columns.push(...table.columnNames);
          }
          continue;
        }
        columns.push(alias || this._deriveColumnName(valObj));
      }
    }
    return columns;
  }

  // ==================================================================
  // Operator-based WHERE compilation (UQA extension functions)
  // ==================================================================
  //
  // When UQA extension functions (text_match, knn_match, fuse_*, etc.)
  // appear in a WHERE clause, they are compiled into Operator trees
  // rather than evaluated row-at-a-time by ExprEvaluator. This yields
  // posting-list-based retrieval with proper scoring.
  //
  // For queries that do not use UQA functions, the existing row-at-a-time
  // WHERE evaluation in _compileSelectBody is used.
  // ==================================================================

  /**
   * Name of the current graph being queried (set from FROM-clause table).
   */
  private _currentGraphName = "";

  /**
   * Build an ExecutionContext for a given table (or null for no-table queries).
   */
  private _contextForTable(table: Table | null): ExecutionContext {
    if (table === null) {
      return {};
    }
    // Convert Map to Record for ExecutionContext compatibility
    const vecIndexes: Record<string, unknown> = {};
    for (const [k, v] of table.vectorIndexes) {
      vecIndexes[k] = v;
    }
    const spIndexes: Record<string, unknown> = {};
    for (const [k, v] of table.spatialIndexes) {
      spIndexes[k] = v;
    }
    return {
      documentStore: table.documentStore,
      invertedIndex: table.invertedIndex,
      vectorIndexes: vecIndexes as Record<string, never>,
      spatialIndexes: spIndexes as Record<string, never>,
    };
  }

  /**
   * Check if an AST subtree contains any UQA posting-list functions.
   */
  private static _containsUQAFunction(node: Record<string, unknown>): boolean {
    // FuncCall check
    if (isFuncCall(node)) {
      const name = getFuncName(node);
      if (UQA_WHERE_FUNCTIONS.has(name)) return true;
    }
    const inner = asObj(nodeGet(node, "FuncCall") ?? {});
    if (Object.keys(inner).length > 0) {
      const name = getFuncName(inner);
      if (UQA_WHERE_FUNCTIONS.has(name)) return true;
    }
    // A_Expr with @@ operator
    const aExpr = asObj(nodeGet(node, "A_Expr") ?? {});
    if (Object.keys(aExpr).length > 0) {
      const nameList = asList(nodeGet(aExpr, "name"));
      if (nameList.length > 0 && extractString(nameList[0]!) === "@@") return true;
    }
    // Recurse into child nodes
    for (const attr of ["lexpr", "rexpr", "args", "arg"]) {
      const child = nodeGet(node, attr) ?? nodeGet(aExpr, attr);
      if (child === null || child === undefined) continue;
      if (Array.isArray(child)) {
        for (const c of child as Record<string, unknown>[]) {
          if (SQLCompiler._containsUQAFunction(asObj(c))) {
            return true;
          }
        }
      } else if (typeof child === "object") {
        if (SQLCompiler._containsUQAFunction(asObj(child))) return true;
      }
    }
    // BoolExpr args
    const boolExpr = asObj(nodeGet(node, "BoolExpr") ?? {});
    if (Object.keys(boolExpr).length > 0) {
      const boolArgs = asList(nodeGet(boolExpr, "args"));
      for (const ba of boolArgs) {
        if (SQLCompiler._containsUQAFunction(ba)) return true;
      }
    }
    return false;
  }

  /**
   * Split a WHERE clause into UQA function conjuncts and scalar conjuncts.
   * Returns [uqaNode, scalarNode] where either can be null.
   */
  private _splitUQAConjuncts(
    whereNode: Record<string, unknown>,
  ): [Record<string, unknown> | null, Record<string, unknown> | null] {
    const conjuncts = this._extractAndConjuncts(whereNode);
    const uqa: Record<string, unknown>[] = [];
    const scalar: Record<string, unknown>[] = [];
    for (const conj of conjuncts) {
      if (SQLCompiler._containsUQAFunction(conj)) {
        uqa.push(conj);
      } else {
        scalar.push(conj);
      }
    }
    const uqaNode =
      uqa.length === 0
        ? null
        : uqa.length === 1
          ? uqa[0]!
          : { BoolExpr: { boolop: 0, args: uqa } };
    const scalarNode =
      scalar.length === 0
        ? null
        : scalar.length === 1
          ? scalar[0]!
          : { BoolExpr: { boolop: 0, args: scalar } };
    return [uqaNode, scalarNode];
  }

  /**
   * Extract AND conjuncts from a WHERE clause node.
   */
  private _extractAndConjuncts(
    node: Record<string, unknown>,
  ): Record<string, unknown>[] {
    const boolExpr = asObj(nodeGet(node, "BoolExpr") ?? {});
    if (Object.keys(boolExpr).length > 0) {
      const boolop = nodeGet(boolExpr, "boolop");
      // AND_EXPR = 0
      if (boolop === 0 || boolop === "AND_EXPR") {
        const args = asList(nodeGet(boolExpr, "args"));
        const result: Record<string, unknown>[] = [];
        for (const arg of args) {
          result.push(...this._extractAndConjuncts(arg));
        }
        return result;
      }
    }
    return [node];
  }

  /**
   * Compile a WHERE AST node into an Operator tree.
   * This handles UQA extension functions, boolean logic, comparisons,
   * null tests, and sublink (IN/EXISTS) predicates.
   */
  private _compileWhere(
    node: Record<string, unknown>,
    ctx: ExecutionContext,
  ): Operator {
    // BoolExpr
    const boolExpr = asObj(nodeGet(node, "BoolExpr") ?? {});
    if (Object.keys(boolExpr).length > 0) {
      return this._compileBoolExpr(boolExpr, ctx);
    }
    // A_Expr
    if (isAExpr(node)) {
      return this._compileComparison(asObj(nodeGet(node, "A_Expr") ?? node), ctx);
    }
    const innerAExpr = nodeGet(node, "A_Expr");
    if (innerAExpr !== null && innerAExpr !== undefined) {
      return this._compileComparison(asObj(innerAExpr), ctx);
    }
    // FuncCall
    if (isFuncCall(node)) {
      return this._compileFuncInWhere(asObj(nodeGet(node, "FuncCall") ?? node), ctx);
    }
    const innerFunc = nodeGet(node, "FuncCall");
    if (innerFunc !== null && innerFunc !== undefined) {
      return this._compileFuncInWhere(asObj(innerFunc), ctx);
    }
    // NullTest
    if (isNullTest(node)) {
      return this._compileNullTest(asObj(nodeGet(node, "NullTest") ?? node));
    }
    const innerNull = nodeGet(node, "NullTest");
    if (innerNull !== null && innerNull !== undefined) {
      return this._compileNullTest(asObj(innerNull));
    }
    // SubLink
    if (isSubLink(node)) {
      return this._compileSublinkInWhere(asObj(nodeGet(node, "SubLink") ?? node), ctx);
    }
    const innerSub = nodeGet(node, "SubLink");
    if (innerSub !== null && innerSub !== undefined) {
      return this._compileSublinkInWhere(asObj(innerSub), ctx);
    }
    // Fallback: expression-based filter
    return new ExprFilterOperator(node, (stmt: Record<string, unknown>) =>
      this._compileSelect(stmt, this._params),
    );
  }

  /**
   * Compile a BoolExpr (AND/OR/NOT) into operators.
   */
  private _compileBoolExpr(
    node: Record<string, unknown>,
    ctx: ExecutionContext,
  ): Operator {
    const boolop = nodeGet(node, "boolop");
    const args = asList(nodeGet(node, "args"));

    // AND_EXPR = 0
    if (boolop === 0 || boolop === "AND_EXPR") {
      return this._compileAnd(args, ctx);
    }
    // OR_EXPR = 1
    if (boolop === 1 || boolop === "OR_EXPR") {
      return new UnionOperator(args.map((a) => this._compileWhere(a, ctx)));
    }
    // NOT_EXPR = 2
    if (boolop === 2 || boolop === "NOT_EXPR") {
      return new ComplementOperator(this._compileWhere(args[0]!, ctx));
    }
    throw new Error(`Unsupported BoolExpr type: ${String(boolop)}`);
  }

  /**
   * Compile AND: chain filters on top of scored retrievals.
   */
  private _compileAnd(
    args: Record<string, unknown>[],
    ctx: ExecutionContext,
  ): Operator {
    const scored: Operator[] = [];
    const filters: FilterOperator[] = [];

    for (const arg of args) {
      const compiled = this._compileWhere(arg, ctx);
      if (compiled instanceof FilterOperator && compiled.source === null) {
        filters.push(compiled);
      } else {
        scored.push(compiled);
      }
    }

    let base: Operator;
    if (scored.length > 0) {
      base = scored.length === 1 ? scored[0]! : new IntersectOperator(scored);
    } else if (filters.length > 0) {
      base = filters.shift()!;
    } else {
      return new ScanOperator();
    }

    for (const f of filters) {
      base = new FilterOperator(f.field, f.predicate, base);
    }
    return base;
  }

  /**
   * Compile a comparison expression (A_Expr) into an operator.
   */
  private _compileComparison(
    node: Record<string, unknown>,
    _ctx: ExecutionContext,
  ): Operator {
    const kind = nodeGet(node, "kind") as number | string;
    const nameList = asList(nodeGet(node, "name"));
    const opName =
      nameList.length > 0 ? extractString(nameList[nameList.length - 1]!) : "";

    // AEXPR_OP = 0
    if (kind === 0 || kind === "AEXPR_OP") {
      // @@ operator for full-text search
      if (opName === "@@") {
        const lexpr = asObj(nodeGet(node, "lexpr"));
        const rexpr = asObj(nodeGet(node, "rexpr"));
        const fieldName = extractColumnName(lexpr);
        const queryString = extractStringValue(rexpr, this._params);
        const effectiveField = fieldName === "_all" ? null : fieldName;
        return this._makeTextSearchOp(effectiveField, queryString, _ctx, false);
      }
      // Simple column op constant
      const lexpr = asObj(nodeGet(node, "lexpr"));
      const rexpr = asObj(nodeGet(node, "rexpr"));
      if (
        isColumnRef(lexpr) &&
        (isAConst(rexpr) || isParamRef(rexpr)) &&
        ["=", "!=", "<>", ">", ">=", "<", "<="].includes(opName)
      ) {
        const fieldName = extractColumnName(lexpr);
        const value = extractConstValue(rexpr, this._params);
        return new FilterOperator(fieldName, _opToPredicate(opName, value));
      }
      // Expression-based comparison
      return new ExprFilterOperator(
        node.kind !== undefined ? node : { A_Expr: node },
        (stmt: Record<string, unknown>) => this._compileSelect(stmt, this._params),
      );
    }

    // AEXPR_IN = 7
    if (kind === 7 || kind === "AEXPR_IN") {
      const lexpr = asObj(nodeGet(node, "lexpr"));
      const fieldName = extractColumnName(lexpr);
      const rexprList = asList(nodeGet(node, "rexpr"));
      const values = new Set<unknown>();
      for (const v of rexprList) {
        values.add(extractConstValue(v, this._params));
      }
      return new FilterOperator(fieldName, new InSet(values));
    }

    // AEXPR_BETWEEN = 11
    if (kind === 11 || kind === "AEXPR_BETWEEN") {
      const lexpr = asObj(nodeGet(node, "lexpr"));
      const fieldName = extractColumnName(lexpr);
      const rexprList = asList(nodeGet(node, "rexpr"));
      const low = Number(extractConstValue(rexprList[0]!, this._params));
      const high = Number(extractConstValue(rexprList[1]!, this._params));
      return new FilterOperator(fieldName, new Between(low, high));
    }

    // AEXPR_NOT_BETWEEN = 12
    if (kind === 12 || kind === "AEXPR_NOT_BETWEEN") {
      const lexpr = asObj(nodeGet(node, "lexpr"));
      const fieldName = extractColumnName(lexpr);
      const rexprList = asList(nodeGet(node, "rexpr"));
      const low = Number(extractConstValue(rexprList[0]!, this._params));
      const high = Number(extractConstValue(rexprList[1]!, this._params));
      return new ComplementOperator(
        new FilterOperator(fieldName, new Between(low, high)),
      );
    }

    // AEXPR_LIKE = 8
    if (kind === 8 || kind === "AEXPR_LIKE") {
      const lexpr = asObj(nodeGet(node, "lexpr"));
      const rexpr = asObj(nodeGet(node, "rexpr"));
      const fieldName = extractColumnName(lexpr);
      const pattern = extractStringValue(rexpr, this._params);
      if (opName === "!~~") {
        return new FilterOperator(fieldName, new NotLike(pattern));
      }
      return new FilterOperator(fieldName, new Like(pattern));
    }

    // AEXPR_ILIKE = 9
    if (kind === 9 || kind === "AEXPR_ILIKE") {
      const lexpr = asObj(nodeGet(node, "lexpr"));
      const rexpr = asObj(nodeGet(node, "rexpr"));
      const fieldName = extractColumnName(lexpr);
      const pattern = extractStringValue(rexpr, this._params);
      if (opName === "!~~*") {
        return new FilterOperator(fieldName, new NotILike(pattern));
      }
      return new FilterOperator(fieldName, new ILike(pattern));
    }

    throw new Error(`Unsupported expression kind: ${String(kind)}`);
  }

  /**
   * Compile a NullTest into a FilterOperator.
   */
  private _compileNullTest(node: Record<string, unknown>): Operator {
    const arg = asObj(nodeGet(node, "arg"));
    const fieldName = extractColumnName(arg);
    const nullTestType = nodeGet(node, "nulltesttype");
    // IS_NULL = 0, IS_NOT_NULL = 1
    if (nullTestType === 0 || nullTestType === "IS_NULL") {
      return new FilterOperator(fieldName, new IsNull());
    }
    return new FilterOperator(fieldName, new IsNotNull());
  }

  /**
   * Compile a SubLink (IN/EXISTS subquery) in WHERE position.
   */
  private _compileSublinkInWhere(
    node: Record<string, unknown>,
    _ctx: ExecutionContext,
  ): Operator {
    const linkType = nodeGet(node, "subLinkType") as number | string;
    const subselect = asObj(nodeGet(node, "subselect"));
    const selectStmt = asObj(nodeGet(subselect, "SelectStmt") ?? subselect);

    // ANY_SUBLINK = 2 (IN subquery)
    if (linkType === 2 || linkType === "ANY_SUBLINK") {
      const innerResult = this._compileSelect(selectStmt, this._params);
      if (innerResult.columns.length === 0) {
        throw new Error("Subquery must return at least one column");
      }
      const subCol = innerResult.columns[0]!;
      const values = new Set<unknown>();
      for (const row of innerResult.rows) {
        const v = row[subCol];
        if (v !== null && v !== undefined) values.add(v);
      }
      const testExpr = asObj(nodeGet(node, "testexpr"));
      const fieldName = extractColumnName(testExpr);
      return new FilterOperator(fieldName, new InSet(values));
    }

    // EXISTS_SUBLINK = 0
    if (linkType === 0 || linkType === "EXISTS_SUBLINK") {
      const innerResult = this._compileSelect(selectStmt, this._params);
      if (innerResult.rows.length > 0) {
        return new ScanOperator();
      }
      return new ComplementOperator(new ScanOperator());
    }

    throw new Error(`Unsupported subquery type: ${String(linkType)}`);
  }

  /**
   * Compile a function call in WHERE position into the appropriate operator.
   */
  private _compileFuncInWhere(
    node: Record<string, unknown>,
    ctx: ExecutionContext,
  ): Operator {
    const name = getFuncName(node).toLowerCase();
    const args = getFuncArgs(node);

    if (name === "text_match") {
      const fieldName = extractColumnName(args[0]!);
      const query = extractStringValue(args[1]!, this._params);
      return this._makeTextSearchOp(fieldName, query, ctx, false);
    }
    if (name === "bayesian_match") {
      const fieldName = extractColumnName(args[0]!);
      const query = extractStringValue(args[1]!, this._params);
      return this._makeTextSearchOp(fieldName, query, ctx, true);
    }
    if (name === "bayesian_match_with_prior") {
      return this._makeBayesianWithPriorOp(args, ctx);
    }
    if (name === "knn_match") {
      return this._makeKnnOp(args);
    }
    if (name === "bayesian_knn_match") {
      return this._makeCalibratedKnnOp(args, ctx, true);
    }
    if (name === "traverse_match") {
      return this._makeTraverseMatchOp(args);
    }
    if (name === "temporal_traverse") {
      return this._makeTemporalTraverseOp(args);
    }
    if (name === "path_filter") {
      return this._makePathFilterOp(args);
    }
    if (name === "vector_exclude") {
      return this._makeVectorExcludeOp(args);
    }
    if (name === "spatial_within") {
      return this._makeSpatialWithinOp(args);
    }
    if (name === "fuse_log_odds") {
      return this._makeFusionOp(args, ctx, "log_odds");
    }
    if (name === "fuse_prob_and") {
      return this._makeFusionOp(args, ctx, "prob_and");
    }
    if (name === "fuse_prob_or") {
      return this._makeFusionOp(args, ctx, "prob_or");
    }
    if (name === "fuse_prob_not") {
      return this._makeProbNotOp(args, ctx);
    }
    if (name === "fuse_attention") {
      return this._makeAttentionFusionOp(args, ctx);
    }
    if (name === "fuse_multihead") {
      return this._makeMultiheadFusionOp(args, ctx);
    }
    if (name === "fuse_learned") {
      return this._makeLearnedFusionOp(args, ctx);
    }
    if (name === "sparse_threshold") {
      return this._makeSparseThresholdOp(args, ctx);
    }
    if (name === "multi_field_match") {
      return this._makeMultiFieldMatchOp(args);
    }
    if (name === "message_passing") {
      return this._makeMessagePassingOp(args);
    }
    if (name === "graph_embedding") {
      return this._makeGraphEmbeddingOp(args);
    }
    if (name === "staged_retrieval") {
      return this._makeStagedRetrievalOp(args, ctx);
    }
    if (name === "pagerank") {
      return this._makePageRankOp(args);
    }
    if (name === "hits") {
      return this._makeHITSOp(args);
    }
    if (name === "betweenness") {
      return this._makeBetweennessOp(args);
    }
    if (name === "weighted_rpq") {
      return this._makeWeightedRpqOp(args);
    }
    if (name === "progressive_fusion") {
      return this._makeProgressiveFusionOp(args, ctx);
    }
    if (name === "deep_fusion") {
      return this._makeDeepFusionOp(args, ctx);
    }
    // Fallback: expression-based filter for scalar functions
    return new ExprFilterOperator(node, (stmt: Record<string, unknown>) =>
      this._compileSelect(stmt, this._params),
    );
  }

  /**
   * Compile a signal function into an operator that produces calibrated
   * probabilities in (0, 1). Used by fusion operators.
   */
  private _compileCalibratedSignal(
    node: Record<string, unknown>,
    ctx: ExecutionContext,
  ): Operator {
    const fc = asObj(nodeGet(node, "FuncCall") ?? node);
    const name = getFuncName(fc).toLowerCase();
    const args = getFuncArgs(fc);

    if (name === "text_match" || name === "bayesian_match") {
      const fieldName = extractColumnName(args[0]!);
      const query = extractStringValue(args[1]!, this._params);
      return this._makeTextSearchOp(fieldName, query, ctx, true);
    }
    if (name === "knn_match") {
      return this._makeCalibratedKnnOp(args, ctx, false);
    }
    if (name === "bayesian_knn_match") {
      return this._makeCalibratedKnnOp(args, ctx, true);
    }
    if (name === "traverse_match") {
      return this._makeTraverseMatchOp(args);
    }
    if (name === "spatial_within") {
      return this._makeSpatialWithinOp(args);
    }
    if (name === "pagerank") {
      return this._makePageRankOp(args);
    }
    if (name === "hits") {
      return this._makeHITSOp(args);
    }
    if (name === "betweenness") {
      return this._makeBetweennessOp(args);
    }
    if (name === "weighted_rpq") {
      return this._makeWeightedRpqOp(args);
    }
    if (name === "message_passing") {
      return this._makeMessagePassingOp(args);
    }
    throw new Error(
      `Unknown signal function for fusion: ${name}. ` +
        `Use text_match, bayesian_match, knn_match, bayesian_knn_match, ` +
        `traverse_match, spatial_within, pagerank, hits, betweenness, or weighted_rpq.`,
    );
  }

  // ==================================================================
  // UQA operator builders
  // ==================================================================

  /**
   * Build a text search operator (TermOperator + scorer).
   * When bayesian=true, uses BayesianBM25Scorer; otherwise BM25Scorer.
   */
  private _makeTextSearchOp(
    fieldName: string | null,
    query: string,
    ctx: ExecutionContext,
    bayesian: boolean,
  ): Operator {
    const idx = ctx.invertedIndex;
    let terms: string[] = [];
    if (idx) {
      const idxAny = idx as unknown as Record<string, unknown>;
      const analyzer =
        fieldName && typeof idxAny["getSearchAnalyzer"] === "function"
          ? (
              idx as unknown as {
                getSearchAnalyzer(f: string): { analyze(q: string): string[] };
              }
            ).getSearchAnalyzer(fieldName)
          : (idx as unknown as { analyzer: { analyze(q: string): string[] } }).analyzer;
      terms = analyzer.analyze(query);
    }
    const retrieval = new TermOperator(query, fieldName ?? undefined);

    if (bayesian) {
      // BayesianBM25 scoring
      try {
        const stats = idx ? (idx as unknown as { stats: unknown }).stats : null;
        if (stats) {
          const scorer = new BayesianBM25Scorer(
            createBayesianBM25Params(),
            stats as IndexStats,
          );
          return new ScoreOperator(
            scorer as never,
            retrieval,
            terms,
            fieldName ?? undefined,
          );
        }
      } catch {
        // BayesianBM25 not available, fall back to BM25
      }
    }

    // BM25 scoring
    try {
      const stats = idx ? (idx as unknown as { stats: unknown }).stats : null;
      if (stats) {
        const scorer = new BM25Scorer(createBM25Params(), stats as IndexStats);
        return new ScoreOperator(
          scorer as never,
          retrieval,
          terms,
          fieldName ?? undefined,
        );
      }
    } catch {
      // BM25 not available
    }

    return retrieval;
  }

  /**
   * multi_field_match(field1, field2, ..., query [, weight1, weight2, ...])
   */
  private _makeMultiFieldMatchOp(args: Record<string, unknown>[]): Operator {
    if (args.length < 3) {
      throw new Error(
        "multi_field_match() requires at least 3 arguments: " +
          "multi_field_match(field1, field2, ..., query)",
      );
    }

    const fields: string[] = [];
    let query: string | null = null;
    const weights: number[] = [];

    for (let i = 0; i < args.length; i++) {
      const arg = args[i]!;
      if (isColumnRef(arg) || nodeGet(arg, "ColumnRef") !== null) {
        fields.push(extractColumnName(arg));
      } else {
        const val = extractConstValue(arg, this._params);
        if (typeof val === "string" && query === null) {
          query = val;
        } else if (typeof val === "number") {
          weights.push(val);
        } else if (typeof val === "string") {
          throw new Error(`Unexpected string argument at position ${String(i)}`);
        }
      }
    }

    if (query === null || fields.length < 2) {
      throw new Error(
        "multi_field_match() requires at least 2 field names and a query string",
      );
    }

    if (weights.length > 0 && weights.length !== fields.length) {
      throw new Error(
        `Number of weights (${String(weights.length)}) must match ` +
          `number of fields (${String(fields.length)})`,
      );
    }

    return new MultiFieldSearchOperator(
      fields,
      query,
      weights.length > 0 ? weights : undefined,
    );
  }

  /**
   * message_passing(k_layers, aggregation, property_name)
   */
  private _makeMessagePassingOp(args: Record<string, unknown>[]): Operator {
    const k = args.length > 0 ? extractIntValue(args[0]!, this._params) : 2;
    const agg = args.length > 1 ? extractStringValue(args[1]!, this._params) : "mean";
    const prop =
      args.length > 2 ? extractStringValue(args[2]!, this._params) : undefined;
    return new MessagePassingOperator({
      kLayers: k,
      aggregation: agg as "mean" | "sum" | "max",
      propertyName: prop,
      graph: this._currentGraphName || "",
    });
  }

  /**
   * graph_embedding(dimensions, k_layers)
   */
  private _makeGraphEmbeddingOp(args: Record<string, unknown>[]): Operator {
    const k = args.length > 1 ? extractIntValue(args[1]!, this._params) : 2;
    return new GraphEmbeddingOperator({
      graph: this._currentGraphName || "",
      kHops: k,
    });
  }

  /**
   * bayesian_match_with_prior(field, query, prior_field, prior_mode)
   */
  private _makeBayesianWithPriorOp(
    args: Record<string, unknown>[],
    ctx: ExecutionContext,
  ): Operator {
    if (args.length < 4) {
      throw new Error(
        "bayesian_match_with_prior() requires 4 arguments: " +
          "bayesian_match_with_prior(field, query, prior_field, prior_mode)",
      );
    }
    const fieldName = extractColumnName(args[0]!);
    const query = extractStringValue(args[1]!, this._params);
    const priorField = extractStringValue(args[2]!, this._params);
    const priorMode = extractStringValue(args[3]!, this._params);

    if (priorMode !== "recency" && priorMode !== "authority") {
      throw new Error(
        `Unknown prior mode: ${priorMode}. Use 'recency' or 'authority'.`,
      );
    }

    // Build prior function
    let priorFn: (doc: Record<string, unknown>) => number;
    if (priorMode === "recency") {
      try {
        priorFn = recencyPrior(priorField);
      } catch {
        priorFn = (_doc: Record<string, unknown>) => 0.5;
      }
    } else {
      try {
        priorFn = authorityPrior(priorField);
      } catch {
        priorFn = (_doc: Record<string, unknown>) => 0.5;
      }
    }

    const idx = ctx.invertedIndex;
    let terms: string[] = [];
    if (idx) {
      const analyzer = (
        idx as unknown as { analyzer: { analyze(q: string): string[] } }
      ).analyzer;
      terms = analyzer.analyze(query);
    }
    const retrieval = new TermOperator(query, fieldName);

    // Build ExternalPriorSearchOperator
    try {
      const stats = idx ? (idx as unknown as { stats: unknown }).stats : null;
      if (stats) {
        const scorer = new ExternalPriorScorer(
          createBayesianBM25Params(),
          stats as IndexStats,
          priorFn,
        );
        return new ExternalPriorSearchOperator(
          retrieval,
          scorer as never,
          terms,
          fieldName,
          ctx.documentStore ?? null,
        );
      }
    } catch {
      // ExternalPriorScorer not available
    }

    return retrieval;
  }

  /**
   * knn_match(field, vector, k)
   */
  private _makeKnnOp(args: Record<string, unknown>[]): Operator {
    if (args.length !== 3) {
      throw new Error("knn_match() requires 3 arguments: knn_match(field, vector, k)");
    }
    const fieldName = extractColumnName(args[0]!);
    const queryVector = extractVectorArg(args[1]!, this._params);
    const k = extractIntValue(args[2]!, this._params);
    return new KNNOperator(queryVector, k, fieldName);
  }

  /**
   * KNN search with calibrated probability scores.
   * When force_calibrated is true or IVF background stats are available,
   * uses CalibratedVectorOperator; otherwise falls back to
   * CalibratedKNNOperator (linear rescaling P = (1 + cos) / 2).
   */
  private _makeCalibratedKnnOp(
    args: Record<string, unknown>[],
    ctx: ExecutionContext,
    forceCal: boolean,
  ): Operator {
    // Separate positional args from named options
    const positional: Record<string, unknown>[] = [];
    const options: Record<string, unknown> = {};
    for (const arg of args) {
      const namedArg = nodeGet(arg, "NamedArgExpr");
      if (namedArg !== null && namedArg !== undefined) {
        const nObj = asObj(namedArg);
        const argName = nodeStr(nObj, "name");
        const argVal = asObj(nodeGet(nObj, "arg"));
        options[argName] = extractConstValue(argVal, this._params);
      } else {
        positional.push(arg);
      }
    }

    if (positional.length !== 3) {
      throw new Error(
        "knn_match() requires 3 positional arguments: knn_match(field, vector, k)",
      );
    }
    const fieldName = extractColumnName(positional[0]!);
    const queryVector = extractVectorArg(positional[1]!, this._params);
    const k = extractIntValue(positional[2]!, this._params);

    // Check for IVF background stats
    const vecIdx = ctx.vectorIndexes?.[fieldName];
    const vecIdxAny = vecIdx as unknown as Record<string, unknown> | null;
    const hasBg =
      vecIdxAny !== null &&
      typeof vecIdxAny === "object" &&
      "backgroundStats" in vecIdxAny &&
      vecIdxAny["backgroundStats"] !== null;

    if (hasBg || forceCal || Object.keys(options).length > 0) {
      const validOptions = new Set([
        "method",
        "weight_source",
        "bm25_query",
        "bm25_field",
        "base_rate",
        "bandwidth_scale",
        "density_gamma",
      ]);
      const unknown = Object.keys(options).filter((k) => !validOptions.has(k));
      if (unknown.length > 0) {
        throw new Error(
          `Unknown option(s) for bayesian_knn_match: ${unknown.join(", ")}. ` +
            `Valid options: ${[...validOptions].sort().join(", ")}`,
        );
      }
      return new CalibratedVectorOperator(
        queryVector,
        k,
        fieldName,
        (options["method"] as string | undefined) ?? "kde",
        Number(options["base_rate"] ?? 0.5),
        (options["weight_source"] as string | undefined) ?? "density_prior",
        options["bm25_query"] != null
          ? String(options["bm25_query"] as string | number)
          : undefined,
        options["bm25_field"] != null
          ? String(options["bm25_field"] as string | number)
          : undefined,
        Number(options["density_gamma"] ?? 1.0),
        Number(options["bandwidth_scale"] ?? 1.0),
      );
    }
    // Fallback: linear calibration P = (1 + cos) / 2
    return new CalibratedKNNOperator(queryVector, k, fieldName);
  }

  /**
   * spatial_within(field, POINT(x, y), distance)
   */
  private _makeSpatialWithinOp(args: Record<string, unknown>[]): Operator {
    if (args.length !== 3) {
      throw new Error(
        "spatial_within() requires 3 arguments: " +
          "spatial_within(field, POINT(x, y), distance)",
      );
    }
    const fieldName = extractColumnName(args[0]!);
    const [cx, cy] = this._extractPointArg(args[1]!);
    const distance = Number(extractConstValue(args[2]!, this._params));
    return new SpatialWithinOperator(fieldName, cx, cy, distance);
  }

  /**
   * Extract (x, y) from a POINT(x, y) FuncCall or $N parameter.
   */
  private _extractPointArg(node: Record<string, unknown>): [number, number] {
    const fc = asObj(nodeGet(node, "FuncCall") ?? {});
    if (Object.keys(fc).length > 0) {
      const ptName = getFuncName(fc);
      if (ptName !== "point") {
        throw new Error(`Expected POINT(x, y), got ${ptName}()`);
      }
      const ptArgs = getFuncArgs(fc);
      if (ptArgs.length !== 2) {
        throw new Error("POINT() requires exactly 2 arguments");
      }
      const x = Number(extractConstValue(ptArgs[0]!, this._params));
      const y = Number(extractConstValue(ptArgs[1]!, this._params));
      return [x, y];
    }
    if (isParamRef(node)) {
      const val = extractConstValue(node, this._params);
      if (Array.isArray(val) && val.length === 2) {
        return [Number(val[0]), Number(val[1])];
      }
      throw new Error("Parameter for POINT must be a [x, y] array");
    }
    throw new Error("Expected POINT(x, y) or $N parameter");
  }

  /**
   * traverse_match(start_id, 'label', max_hops) as WHERE signal.
   */
  private _makeTraverseMatchOp(args: Record<string, unknown>[]): Operator {
    const start = extractIntValue(args[0]!, this._params);
    const label =
      args.length > 1 ? extractStringValue(args[1]!, this._params) : undefined;
    const maxHops = args.length > 2 ? extractIntValue(args[2]!, this._params) : 1;
    return new TraverseOperator(
      start,
      this._currentGraphName || "",
      label ?? null,
      maxHops,
    );
  }

  /**
   * temporal_traverse(start, label, hops, timestamp) or
   * temporal_traverse(start, label, hops, from_ts, to_ts)
   */
  private _makeTemporalTraverseOp(args: Record<string, unknown>[]): Operator {
    if (args.length < 4) {
      throw new Error(
        "temporal_traverse() requires at least 4 arguments: " +
          "temporal_traverse(start, label, hops, timestamp)",
      );
    }
    const start = extractIntValue(args[0]!, this._params);
    const label =
      args.length > 1 ? extractStringValue(args[1]!, this._params) : undefined;
    const maxHops = args.length > 2 ? extractIntValue(args[2]!, this._params) : 1;

    let tf: TemporalFilter;
    if (args.length === 4) {
      const ts = Number(extractConstValue(args[3]!, this._params));
      tf = new TemporalFilter({ timestamp: ts });
    } else if (args.length >= 5) {
      const fromTs = Number(extractConstValue(args[3]!, this._params));
      const toTs = Number(extractConstValue(args[4]!, this._params));
      tf = new TemporalFilter({ timeRange: [fromTs, toTs] });
    } else {
      tf = new TemporalFilter();
    }

    return new TemporalTraverseOperator({
      startVertex: start,
      graph: this._currentGraphName || "",
      temporalFilter: tf,
      label: label ?? null,
      maxHops,
    });
  }

  /**
   * path_filter('path', value) or path_filter('path', 'op', value)
   */
  private _makePathFilterOp(args: Record<string, unknown>[]): Operator {
    if (args.length < 2) {
      throw new Error(
        "path_filter() requires at least 2 arguments: " +
          "path_filter('path', value) or path_filter('path', 'op', value)",
      );
    }

    const pathStr = extractStringValue(args[0]!, this._params);
    const pathExpr: (string | number)[] = [];
    for (const component of pathStr.split(".")) {
      if (/^\d+$/.test(component)) {
        pathExpr.push(parseInt(component, 10));
      } else {
        pathExpr.push(component);
      }
    }

    if (args.length === 2) {
      const value = extractConstValue(args[1]!, this._params);
      return new PathFilterOperator(pathExpr, new Equals(value));
    }

    const opStr = extractStringValue(args[1]!, this._params);
    const value = extractConstValue(args[2]!, this._params);
    return new PathFilterOperator(pathExpr, _opToPredicate(opStr, value));
  }

  /**
   * vector_exclude(field, positive_vector, negative_vector, k, threshold)
   */
  private _makeVectorExcludeOp(args: Record<string, unknown>[]): Operator {
    if (args.length !== 5) {
      throw new Error(
        "vector_exclude() requires 5 arguments: " +
          "vector_exclude(field, positive_vector, negative_vector, k, threshold)",
      );
    }
    const fieldName = extractColumnName(args[0]!);
    const positiveVector = extractVectorArg(args[1]!, this._params);
    const negativeVector = extractVectorArg(args[2]!, this._params);
    const k = extractIntValue(args[3]!, this._params);
    const threshold = Number(extractConstValue(args[4]!, this._params));

    const positive = new KNNOperator(positiveVector, k, fieldName);
    return new VectorExclusionOperator(positive, negativeVector, threshold, fieldName);
  }

  /**
   * fuse_prob_not(signal) -- probabilistic complement.
   */
  private _makeProbNotOp(
    args: Record<string, unknown>[],
    ctx: ExecutionContext,
  ): Operator {
    if (args.length !== 1) {
      throw new Error("fuse_prob_not() requires exactly 1 signal function argument");
    }
    const signal = this._compileCalibratedSignal(args[0]!, ctx);
    return new ProbNotOperator(signal);
  }

  /**
   * sparse_threshold(signal, threshold)
   */
  private _makeSparseThresholdOp(
    args: Record<string, unknown>[],
    ctx: ExecutionContext,
  ): Operator {
    if (args.length !== 2) {
      throw new Error(
        "sparse_threshold() requires 2 arguments: " +
          "sparse_threshold(signal, threshold)",
      );
    }
    const signal = this._compileCalibratedSignal(args[0]!, ctx);
    const threshold = Number(extractConstValue(args[1]!, this._params));
    return new SparseThresholdOperator(signal, threshold);
  }

  /**
   * Build a fusion operator from nested function calls.
   * fuse_log_odds(signal1, signal2, ...[, alpha[, 'gating']])
   * fuse_prob_and(signal1, signal2, ...)
   * fuse_prob_or(signal1, signal2, ...)
   */
  private _makeFusionOp(
    args: Record<string, unknown>[],
    ctx: ExecutionContext,
    mode: string,
  ): Operator {
    const signals: Operator[] = [];
    let alpha = 0.5;
    let gating: string | null = null;
    let gatingBeta: number | null = null;

    for (const arg of args) {
      if (isFuncCall(arg) || nodeGet(arg, "FuncCall") !== null) {
        signals.push(this._compileCalibratedSignal(arg, ctx));
      } else {
        const namedArg = nodeGet(arg, "NamedArgExpr");
        if (namedArg !== null && namedArg !== undefined && mode === "log_odds") {
          const nObj = asObj(namedArg);
          const argName = nodeStr(nObj, "name");
          const val = extractConstValue(asObj(nodeGet(nObj, "arg")), this._params);
          if (argName === "alpha") {
            alpha = Number(val);
          } else if (argName === "gating") {
            gating = String(val);
          } else if (argName === "gating_beta") {
            gatingBeta = Number(val);
          } else {
            throw new Error(
              `Unknown option for fuse_log_odds: ${argName}. ` +
                `Valid options: alpha, gating, gating_beta`,
            );
          }
        } else if (isAConst(arg) && mode === "log_odds") {
          const val = extractConstValue(arg, this._params);
          if (typeof val === "string") {
            gating = val;
          } else {
            alpha = Number(val);
          }
        } else {
          throw new Error(
            "Fusion function arguments must be signal functions " +
              "(text_match, knn_match, etc.)",
          );
        }
      }
    }

    if (signals.length < 2) {
      throw new Error("Fusion requires at least 2 signal functions");
    }

    if (mode === "log_odds") {
      return new LogOddsFusionOperator(signals, alpha, null, gating, gatingBeta);
    }
    if (mode === "prob_and") {
      return new ProbBoolFusionOperator(signals, "and");
    }
    return new ProbBoolFusionOperator(signals, "or");
  }

  /**
   * staged_retrieval(signal1, k1, signal2, k2, ...)
   */
  private _makeStagedRetrievalOp(
    args: Record<string, unknown>[],
    ctx: ExecutionContext,
  ): Operator {
    const stages: [Operator, number][] = [];
    let i = 0;
    while (i < args.length - 1) {
      if (!isFuncCall(args[i]!) && nodeGet(args[i]!, "FuncCall") === null) {
        throw new Error(
          `staged_retrieval: argument ${String(i)} must be a signal function`,
        );
      }
      const signal = this._compileCalibratedSignal(args[i]!, ctx);
      let cutoffVal = extractConstValue(args[i + 1]!, this._params) as number;
      if (typeof cutoffVal === "number" && cutoffVal === Math.floor(cutoffVal)) {
        cutoffVal = Math.floor(cutoffVal);
      }
      stages.push([signal, cutoffVal]);
      i += 2;
    }

    if (stages.length === 0) {
      throw new Error("staged_retrieval requires at least one (signal, cutoff) pair");
    }

    return new MultiStageOperator(stages);
  }

  /**
   * Split centrality args into numeric args and graph name.
   */
  private _splitCentralityArgs(
    args: Record<string, unknown>[],
  ): [Record<string, unknown>[], string] {
    const numeric: Record<string, unknown>[] = [];
    let graphName = this._currentGraphName;
    for (const a of args) {
      try {
        const val = extractConstValue(a, this._params);
        if (typeof val === "string") {
          graphName = val;
          continue;
        }
      } catch {
        // Not a constant
      }
      numeric.push(a);
    }
    return [numeric, graphName];
  }

  /**
   * pagerank([damping[, max_iter[, tolerance]]][, 'graph'])
   */
  private _makePageRankOp(args: Record<string, unknown>[]): Operator {
    const [numericArgs, graphName] = this._splitCentralityArgs(args);
    const damping =
      numericArgs.length > 0
        ? Number(extractConstValue(numericArgs[0]!, this._params))
        : 0.85;
    const maxIter =
      numericArgs.length > 1 ? extractIntValue(numericArgs[1]!, this._params) : 100;
    const tol =
      numericArgs.length > 2
        ? Number(extractConstValue(numericArgs[2]!, this._params))
        : 1e-6;
    return new PageRankOperator({
      damping,
      maxIterations: maxIter,
      tolerance: tol,
      graph: graphName || "",
    });
  }

  /**
   * hits([max_iter[, tolerance]][, 'graph'])
   */
  private _makeHITSOp(args: Record<string, unknown>[]): Operator {
    const [numericArgs, graphName] = this._splitCentralityArgs(args);
    const maxIter =
      numericArgs.length > 0 ? extractIntValue(numericArgs[0]!, this._params) : 100;
    const tol =
      numericArgs.length > 1
        ? Number(extractConstValue(numericArgs[1]!, this._params))
        : 1e-6;
    return new HITSOperator({
      maxIterations: maxIter,
      tolerance: tol,
      graph: graphName || "",
    });
  }

  /**
   * betweenness(['graph'])
   */
  private _makeBetweennessOp(args: Record<string, unknown>[]): Operator {
    const [, graphName] = this._splitCentralityArgs(args);
    return new BetweennessCentralityOperator({
      graph: graphName || "",
    });
  }

  /**
   * weighted_rpq('path_expr', start, 'weight_prop'[, 'agg_fn'[, threshold]])
   */
  private _makeWeightedRpqOp(args: Record<string, unknown>[]): Operator {
    if (args.length < 3) {
      throw new Error(
        "weighted_rpq() requires at least 3 arguments: " +
          "weighted_rpq('path_expr', start, 'weight_property'" +
          "[, 'agg_fn'[, threshold]])",
      );
    }
    const exprStr = extractStringValue(args[0]!, this._params);
    const start = extractIntValue(args[1]!, this._params);
    const weightProp = extractStringValue(args[2]!, this._params);
    const aggFn = args.length > 3 ? extractStringValue(args[3]!, this._params) : "sum";
    let weightThreshold: number | null = null;
    if (args.length > 4) {
      weightThreshold = Number(extractConstValue(args[4]!, this._params));
    }

    // Try to import WeightedPathQueryOperator; if not available, fall back
    // to a basic approach with the existing graph operators.
    try {
      return new WeightedPathQueryOperator(
        parseRpq(exprStr),
        this._currentGraphName || "",
        {
          startVertex: start,
          weightProperty: weightProp,
          aggregation: aggFn as "sum" | "min" | "max" | "product",
          weightThreshold,
        },
      );
    } catch {
      // WeightedPathQueryOperator not available -- fall through to error
      throw new Error("weighted_rpq() requires the WeightedPathQueryOperator");
    }
  }

  /**
   * progressive_fusion(sig1, sig2, k1, sig3, k2[, alpha][, 'gating'])
   */
  private _makeProgressiveFusionOp(
    args: Record<string, unknown>[],
    ctx: ExecutionContext,
  ): Operator {
    const signals: Operator[] = [];
    const stages: [Operator[], number][] = [];
    let alpha = 0.5;
    let gating: string | null = null;

    for (const arg of args) {
      if (isFuncCall(arg) || nodeGet(arg, "FuncCall") !== null) {
        signals.push(this._compileCalibratedSignal(arg, ctx));
      } else if (isAConst(arg) || isParamRef(arg)) {
        const val = extractConstValue(arg, this._params);
        if (typeof val === "string") {
          gating = val;
        } else if (typeof val === "number" && val !== Math.floor(val)) {
          alpha = val;
        } else {
          // Integer: this is a k cutoff
          const k = Math.floor(Number(val));
          if (signals.length === 0) {
            throw new Error("progressive_fusion: k must follow signal functions");
          }
          stages.push([[...signals], k]);
          signals.length = 0;
        }
      } else {
        throw new Error("progressive_fusion: unexpected argument type");
      }
    }

    if (signals.length > 0) {
      throw new Error("progressive_fusion: trailing signals without k cutoff");
    }
    if (stages.length === 0) {
      throw new Error("progressive_fusion requires at least one (signals, k) stage");
    }

    return new ProgressiveFusionOperator(stages, alpha, gating);
  }

  /**
   * deep_fusion(layer(...), propagate(...), ...[, named args])
   *
   * Builds a multi-layer deep fusion operator.
   */
  private _makeDeepFusionOp(
    args: Record<string, unknown>[],
    ctx: ExecutionContext,
  ): Operator {
    const layers: FusionLayer[] = [];
    let alpha = 0.5;
    let gating = "none";

    for (const arg of args) {
      // FuncCall (layer, propagate, convolve, etc.)
      const fc = asObj(nodeGet(arg, "FuncCall") ?? (isFuncCall(arg) ? arg : {}));
      if (Object.keys(fc).length > 0) {
        const innerName = getFuncName(fc).toLowerCase();
        const innerArgs = getFuncArgs(fc);

        if (innerName === "layer") {
          const sigs: Operator[] = [];
          for (const ia of innerArgs) {
            if (isFuncCall(ia) || nodeGet(ia, "FuncCall") !== null) {
              sigs.push(this._compileCalibratedSignal(ia, ctx));
            } else {
              throw new Error("layer() arguments must be signal functions");
            }
          }
          if (sigs.length === 0) {
            throw new Error("layer() requires at least one signal");
          }
          layers.push({ type: "signal", signals: sigs } as unknown as FusionLayer);
        } else if (innerName === "propagate") {
          if (innerArgs.length < 2) {
            throw new Error(
              "propagate() requires at least 2 arguments: " +
                "propagate('edge_label', 'aggregation'[, 'direction'])",
            );
          }
          const edgeLabel = extractStringValue(innerArgs[0]!, this._params);
          const aggregation = extractStringValue(innerArgs[1]!, this._params);
          if (!["mean", "sum", "max"].includes(aggregation)) {
            throw new Error(
              `propagate() aggregation must be 'mean', 'sum', or 'max', got '${aggregation}'`,
            );
          }
          let direction = "both";
          if (innerArgs.length >= 3) {
            direction = extractStringValue(innerArgs[2]!, this._params);
            if (!["both", "out", "in"].includes(direction)) {
              throw new Error(
                `propagate() direction must be 'both', 'out', or 'in', got '${direction}'`,
              );
            }
          }
          layers.push({
            type: "propagate",
            edgeLabel,
            aggregation,
            direction,
          } as unknown as FusionLayer);
        } else if (innerName === "convolve") {
          const convPos: Record<string, unknown>[] = [];
          const convNamed: Record<string, unknown> = {};
          for (const ia of innerArgs) {
            const na = nodeGet(ia, "NamedArgExpr");
            if (na !== null && na !== undefined) {
              const nObj = asObj(na);
              convNamed[nodeStr(nObj, "name")] = extractConstValue(
                asObj(nodeGet(nObj, "arg")),
                this._params,
              );
            } else {
              convPos.push(ia);
            }
          }
          if (convPos.length === 0) {
            throw new Error("convolve() requires edge_label as first argument");
          }
          const edgeLabel = extractStringValue(convPos[0]!, this._params);
          const nChannels = Number(convNamed["n_channels"] ?? 0);
          const direction = "both";

          if (nChannels > 1) {
            // Multi-channel convolution
            const seed = Number(convNamed["seed"] ?? 42);
            const initMode = (convNamed["init"] as string | undefined) ?? "kaiming";
            let prevOutCh = 1;
            for (let li = layers.length - 1; li >= 0; li--) {
              const prev = layers[li]!;
              if ((prev as unknown as Record<string, unknown>)["type"] === "conv") {
                const ks = (prev as unknown as Record<string, unknown>)[
                  "kernelShape"
                ] as number[] | undefined;
                if (ks) prevOutCh = ks[0]!;
                break;
              }
            }
            try {
              const kernels = _genKernels(nChannels, prevOutCh, seed, initMode);
              layers.push({
                type: "conv",
                edgeLabel,
                hopWeights: [1.0],
                direction,
                kernel: Array.from(kernels),
                kernelShape: [nChannels, prevOutCh],
              } as unknown as FusionLayer);
            } catch {
              layers.push({
                type: "conv",
                edgeLabel,
                hopWeights: [1.0],
                direction,
              } as unknown as FusionLayer);
            }
          } else {
            // Single-channel: hop_weights from ARRAY
            if (convPos.length < 2) {
              throw new Error(
                "convolve() requires ARRAY[w0, w1, ...] or n_channels => N",
              );
            }
            const hopWeights = extractVectorArg(convPos[1]!, this._params);
            layers.push({
              type: "conv",
              edgeLabel,
              hopWeights: Array.from(hopWeights),
              direction,
            } as unknown as FusionLayer);
          }
        } else if (innerName === "pool") {
          if (innerArgs.length < 3) {
            throw new Error(
              "pool() requires at least 3 arguments: " +
                "pool('edge_label', 'method', pool_size[, 'direction'])",
            );
          }
          const edgeLabel = extractStringValue(innerArgs[0]!, this._params);
          const method = extractStringValue(innerArgs[1]!, this._params);
          if (!["max", "avg"].includes(method)) {
            throw new Error(`pool() method must be 'max' or 'avg', got '${method}'`);
          }
          const poolSize = extractIntValue(innerArgs[2]!, this._params);
          let direction = "both";
          if (innerArgs.length >= 4) {
            direction = extractStringValue(innerArgs[3]!, this._params);
            if (!["both", "out", "in"].includes(direction)) {
              throw new Error(
                `pool() direction must be 'both', 'out', or 'in', got '${direction}'`,
              );
            }
          }
          layers.push({
            type: "pool",
            edgeLabel,
            poolSize,
            method,
            direction,
          } as unknown as FusionLayer);
        } else if (innerName === "dense") {
          const densePos: Record<string, unknown>[] = [];
          const denseNamed: Record<string, number> = {};
          for (const ia of innerArgs) {
            const na = nodeGet(ia, "NamedArgExpr");
            if (na !== null && na !== undefined) {
              const nObj = asObj(na);
              denseNamed[nodeStr(nObj, "name")] = Number(
                extractConstValue(asObj(nodeGet(nObj, "arg")), this._params),
              );
            } else {
              densePos.push(ia);
            }
          }
          if (densePos.length < 2) {
            throw new Error(
              "dense() requires at least 2 positional arguments: ARRAY[weights], ARRAY[bias]",
            );
          }
          const weights = extractVectorArg(densePos[0]!, this._params);
          const biasVec = extractVectorArg(densePos[1]!, this._params);
          const outCh = denseNamed["output_channels"];
          const inCh = denseNamed["input_channels"];
          if (outCh === undefined || inCh === undefined) {
            throw new Error(
              "dense() requires output_channels and input_channels named arguments",
            );
          }
          if (weights.length !== outCh * inCh) {
            throw new Error(
              `dense() weights array length (${String(weights.length)}) must equal ` +
                `output_channels * input_channels (${String(outCh * inCh)})`,
            );
          }
          if (biasVec.length !== outCh) {
            throw new Error(
              `dense() bias array length (${String(biasVec.length)}) must equal ` +
                `output_channels (${String(outCh)})`,
            );
          }
          layers.push({
            type: "dense",
            weights: Array.from(weights),
            bias: Array.from(biasVec),
            outputChannels: outCh,
            inputChannels: inCh,
          } as unknown as FusionLayer);
        } else if (innerName === "flatten") {
          layers.push({ type: "flatten" } as unknown as FusionLayer);
        } else if (innerName === "softmax") {
          layers.push({ type: "softmax" } as unknown as FusionLayer);
        } else if (innerName === "batch_norm") {
          let epsilon = 1e-5;
          for (const ia of innerArgs) {
            const na = nodeGet(ia, "NamedArgExpr");
            if (na !== null && na !== undefined) {
              const nObj = asObj(na);
              if (nodeStr(nObj, "name") === "epsilon") {
                epsilon = Number(
                  extractConstValue(asObj(nodeGet(nObj, "arg")), this._params),
                );
              }
            }
          }
          layers.push({ type: "batchNorm", epsilon } as unknown as FusionLayer);
        } else if (innerName === "dropout") {
          if (innerArgs.length < 1) {
            throw new Error("dropout() requires 1 argument: dropout(p)");
          }
          const p = Number(extractConstValue(innerArgs[0]!, this._params));
          layers.push({ type: "dropout", p } as unknown as FusionLayer);
        } else if (innerName === "attention") {
          let attnNHeads = 1;
          let attnMode = "content";
          for (const ia of innerArgs) {
            const na = nodeGet(ia, "NamedArgExpr");
            if (na !== null && na !== undefined) {
              const nObj = asObj(na);
              const attrName = nodeStr(nObj, "name");
              if (attrName === "n_heads") {
                attnNHeads = Number(
                  extractConstValue(asObj(nodeGet(nObj, "arg")), this._params),
                );
              } else if (attrName === "mode") {
                attnMode = String(
                  extractConstValue(asObj(nodeGet(nObj, "arg")), this._params),
                );
              }
            }
          }
          layers.push({
            type: "attention",
            nHeads: attnNHeads,
            mode: attnMode,
          } as unknown as FusionLayer);
        } else if (innerName === "embed") {
          const embedPos: Record<string, unknown>[] = [];
          const embedNamed: Record<string, number> = {};
          for (const ia of innerArgs) {
            const na = nodeGet(ia, "NamedArgExpr");
            if (na !== null && na !== undefined) {
              const nObj = asObj(na);
              embedNamed[nodeStr(nObj, "name")] = Number(
                extractConstValue(asObj(nodeGet(nObj, "arg")), this._params),
              );
            } else {
              embedPos.push(ia);
            }
          }
          if (embedPos.length === 0) {
            throw new Error("embed() requires vector argument");
          }
          const vec = extractVectorArg(embedPos[0]!, this._params);
          const eInCh = embedNamed["in_channels"] ?? 1;
          let eGh: number;
          let eGw: number;
          if (
            embedNamed["grid_h"] !== undefined &&
            embedNamed["grid_w"] !== undefined
          ) {
            eGh = embedNamed["grid_h"]!;
            eGw = embedNamed["grid_w"]!;
          } else {
            const side = Math.floor(Math.sqrt(vec.length / eInCh));
            if (side * side * eInCh === vec.length) {
              eGh = side;
              eGw = side;
            } else {
              eGh = 0;
              eGw = 0;
            }
          }
          layers.push({
            type: "embed",
            embedding: Array.from(vec),
            gridH: eGh,
            gridW: eGw,
            inChannels: eInCh,
          } as unknown as FusionLayer);
        } else if (innerName === "global_pool") {
          let gpMethod = "avg";
          for (const ia of innerArgs) {
            const na = nodeGet(ia, "NamedArgExpr");
            if (na !== null && na !== undefined) {
              const nObj = asObj(na);
              if (nodeStr(nObj, "name") === "method") {
                gpMethod = String(
                  extractConstValue(asObj(nodeGet(nObj, "arg")), this._params),
                );
              }
            } else {
              gpMethod = extractStringValue(ia, this._params);
            }
          }
          if (!["avg", "max", "avg_max"].includes(gpMethod)) {
            throw new Error(
              `global_pool() method must be 'avg', 'max', or 'avg_max', got '${gpMethod}'`,
            );
          }
          layers.push({
            type: "globalPool",
            method: gpMethod,
          } as unknown as FusionLayer);
        } else {
          throw new Error(`deep_fusion() unknown layer function: ${innerName}()`);
        }
        continue;
      }
      // NamedArgExpr
      const namedArg = nodeGet(arg, "NamedArgExpr");
      if (namedArg !== null && namedArg !== undefined) {
        const nObj = asObj(namedArg);
        const argName = nodeStr(nObj, "name");
        const val = extractConstValue(asObj(nodeGet(nObj, "arg")), this._params);
        if (argName === "alpha") {
          alpha = Number(val);
        } else if (argName === "gating") {
          gating = String(val);
        } else {
          throw new Error(
            `Unknown option for deep_fusion: ${argName}. Valid options: alpha, gating`,
          );
        }
        continue;
      }
      throw new Error("deep_fusion() arguments must be layer() or propagate() calls");
    }

    if (layers.length === 0) {
      throw new Error("deep_fusion requires at least one layer");
    }

    return new DeepFusionOperator(
      layers,
      alpha,
      gating,
      this._currentGraphName || undefined,
    );
  }

  /**
   * fuse_attention(signal1, signal2, ...[, named options])
   */
  private _makeAttentionFusionOp(
    args: Record<string, unknown>[],
    ctx: ExecutionContext,
  ): Operator {
    const signals: Operator[] = [];
    const options: Record<string, unknown> = {};
    for (const arg of args) {
      if (isFuncCall(arg) || nodeGet(arg, "FuncCall") !== null) {
        signals.push(this._compileCalibratedSignal(arg, ctx));
      } else {
        const namedArg = nodeGet(arg, "NamedArgExpr");
        if (namedArg !== null && namedArg !== undefined) {
          const nObj = asObj(namedArg);
          options[nodeStr(nObj, "name")] = extractConstValue(
            asObj(nodeGet(nObj, "arg")),
            this._params,
          );
        }
      }
    }

    if (signals.length < 2) {
      throw new Error("fuse_attention requires at least 2 signals");
    }

    const validOpts = new Set(["normalized", "alpha", "base_rate"]);
    const unknownOpts = Object.keys(options).filter((k) => !validOpts.has(k));
    if (unknownOpts.length > 0) {
      throw new Error(
        `Unknown option(s) for fuse_attention: ${unknownOpts.join(", ")}. ` +
          `Valid options: ${[...validOpts].sort().join(", ")}`,
      );
    }

    try {
      const nSignals = signals.length;
      const attention = new AttentionFusion(
        nSignals,
        6,
        Number(options["alpha"] ?? 0.5),
        Boolean(options["normalized"] ?? false),
        (options["base_rate"] as number | undefined) ?? undefined,
      );

      const queryFeatures = this._extractQueryFeatures(args, ctx);
      return new AttentionFusionOperator(signals, attention as never, queryFeatures);
    } catch {
      // AttentionFusion not available, fall back to log-odds
      return new LogOddsFusionOperator(signals, Number(options["alpha"] ?? 0.5));
    }
  }

  /**
   * fuse_multihead(signal1, signal2, ...[, named options])
   */
  private _makeMultiheadFusionOp(
    args: Record<string, unknown>[],
    ctx: ExecutionContext,
  ): Operator {
    const signals: Operator[] = [];
    const options: Record<string, unknown> = {};
    for (const arg of args) {
      if (isFuncCall(arg) || nodeGet(arg, "FuncCall") !== null) {
        signals.push(this._compileCalibratedSignal(arg, ctx));
      } else {
        const namedArg = nodeGet(arg, "NamedArgExpr");
        if (namedArg !== null && namedArg !== undefined) {
          const nObj = asObj(namedArg);
          options[nodeStr(nObj, "name")] = extractConstValue(
            asObj(nodeGet(nObj, "arg")),
            this._params,
          );
        }
      }
    }

    if (signals.length < 2) {
      throw new Error("fuse_multihead requires at least 2 signals");
    }

    const validOpts = new Set(["n_heads", "normalized", "alpha"]);
    const unknownOpts = Object.keys(options).filter((k) => !validOpts.has(k));
    if (unknownOpts.length > 0) {
      throw new Error(
        `Unknown option(s) for fuse_multihead: ${unknownOpts.join(", ")}. ` +
          `Valid options: ${[...validOpts].sort().join(", ")}`,
      );
    }

    try {
      const nSignals = signals.length;
      const fusion = new MultiHeadAttentionFusion(
        nSignals,
        Number(options["n_heads"] ?? 4),
        6,
        Number(options["alpha"] ?? 0.5),
        Boolean(options["normalized"] ?? false),
      );

      const queryFeatures = this._extractQueryFeatures(args, ctx);
      return new AttentionFusionOperator(signals, fusion as never, queryFeatures);
    } catch {
      // MultiHeadAttentionFusion not available, fall back to log-odds
      return new LogOddsFusionOperator(signals, Number(options["alpha"] ?? 0.5));
    }
  }

  /**
   * fuse_learned(signal1, signal2, ...[, named options])
   */
  private _makeLearnedFusionOp(
    args: Record<string, unknown>[],
    ctx: ExecutionContext,
  ): Operator {
    const signals: Operator[] = [];
    const options: Record<string, unknown> = {};
    for (const arg of args) {
      if (isFuncCall(arg) || nodeGet(arg, "FuncCall") !== null) {
        signals.push(this._compileCalibratedSignal(arg, ctx));
      } else {
        const namedArg = nodeGet(arg, "NamedArgExpr");
        if (namedArg !== null && namedArg !== undefined) {
          const nObj = asObj(namedArg);
          options[nodeStr(nObj, "name")] = extractConstValue(
            asObj(nodeGet(nObj, "arg")),
            this._params,
          );
        }
      }
    }

    if (signals.length < 2) {
      throw new Error("fuse_learned requires at least 2 signals");
    }

    const validOpts = new Set(["alpha"]);
    const unknownOpts = Object.keys(options).filter((k) => !validOpts.has(k));
    if (unknownOpts.length > 0) {
      throw new Error(
        `Unknown option(s) for fuse_learned: ${unknownOpts.join(", ")}. ` +
          `Valid options: ${[...validOpts].sort().join(", ")}`,
      );
    }

    try {
      const learned = new LearnedFusion(
        signals.length,
        Number(options["alpha"] ?? 0.5),
      );
      return new LearnedFusionOperator(signals, learned as never);
    } catch {
      // LearnedFusion not available, fall back to log-odds
      return new LogOddsFusionOperator(signals, Number(options["alpha"] ?? 0.5));
    }
  }

  /**
   * Extract query features from the first text signal in args.
   */
  private _extractQueryFeatures(
    args: Record<string, unknown>[],
    ctx: ExecutionContext,
  ): Float64Array {
    const queryFeatures = new Float64Array(6);
    if (ctx.invertedIndex) {
      try {
        const extractor = new QueryFeatureExtractor(ctx.invertedIndex);
        for (const arg of args) {
          const fc = asObj(nodeGet(arg, "FuncCall") ?? (isFuncCall(arg) ? arg : {}));
          if (Object.keys(fc).length > 0) {
            const fnName = getFuncName(fc);
            if (fnName === "text_match" || fnName === "bayesian_match") {
              const fArgs = getFuncArgs(fc);
              if (fArgs.length >= 2) {
                const queryStr = extractStringValue(fArgs[1]!, this._params);
                const analyzer = (
                  ctx.invertedIndex as unknown as {
                    analyzer: { analyze(q: string): string[] };
                  }
                ).analyzer;
                const terms = analyzer.analyze(queryStr);
                return extractor.extract(terms);
              }
            }
          }
        }
      } catch {
        // QueryFeatureExtractor not available
      }
    }
    return queryFeatures;
  }

  // ==================================================================
  // Constant folding
  // ==================================================================

  /**
   * Return a new stmt with the WHERE clause constant-folded.
   */
  private _foldStmtWhere(stmt: Record<string, unknown>): Record<string, unknown> {
    const whereClause = nodeGet(stmt, "whereClause");
    if (whereClause === null || whereClause === undefined) return stmt;
    const folded = this._foldConstants(asObj(whereClause));
    if (folded === whereClause) return stmt;
    return { ...stmt, whereClause: folded };
  }

  /**
   * Bottom-up constant folding for AST expressions.
   */
  private _foldConstants(node: Record<string, unknown>): Record<string, unknown> {
    if (Object.keys(node).length === 0) return node;

    // A_Const -- already a constant
    if (isAConst(node) || nodeGet(node, "A_Const") !== null) return node;

    // ColumnRef -- not foldable
    if (isColumnRef(node) || nodeGet(node, "ColumnRef") !== null) return node;

    // A_Expr: fold if both operands are constant
    if (isAExpr(node) || nodeGet(node, "A_Expr") !== null) {
      return this._foldAExpr(node);
    }

    // BoolExpr: fold if all args are constant, or partial fold
    if (isBoolExpr(node) || nodeGet(node, "BoolExpr") !== null) {
      return this._foldBoolExpr(node);
    }

    // FuncCall: fold only if all args are constant and function is pure
    if (isFuncCall(node) || nodeGet(node, "FuncCall") !== null) {
      return this._foldFuncCall(node);
    }

    return node;
  }

  /**
   * Try to fold an A_Expr with constant operands.
   */
  private _foldAExpr(node: Record<string, unknown>): Record<string, unknown> {
    const aExpr = asObj(nodeGet(node, "A_Expr") ?? node);
    const newLexpr = this._foldConstants(asObj(nodeGet(aExpr, "lexpr") ?? {}));
    const newRexpr = this._foldConstants(asObj(nodeGet(aExpr, "rexpr") ?? {}));

    // If both sides are now constant, try to evaluate
    if (
      (isAConst(newLexpr) || nodeGet(newLexpr, "A_Const") !== null) &&
      (isAConst(newRexpr) || nodeGet(newRexpr, "A_Const") !== null)
    ) {
      try {
        const evaluator = new ExprEvaluator({ params: this._params });
        const folded = {
          A_Expr: {
            ...aExpr,
            lexpr: newLexpr,
            rexpr: newRexpr,
          },
        };
        const result = evaluator.evaluate(folded, {});
        return _valueToAConst(result);
      } catch {
        // Cannot fold -- return with updated operands
      }
    }

    if (newLexpr !== nodeGet(aExpr, "lexpr") || newRexpr !== nodeGet(aExpr, "rexpr")) {
      const updated = { ...aExpr, lexpr: newLexpr, rexpr: newRexpr };
      return nodeGet(node, "A_Expr") !== null ? { A_Expr: updated } : updated;
    }
    return node;
  }

  /**
   * Try to fold a BoolExpr, including partial evaluation.
   */
  private _foldBoolExpr(node: Record<string, unknown>): Record<string, unknown> {
    const boolExpr = asObj(nodeGet(node, "BoolExpr") ?? node);
    const boolop = nodeGet(boolExpr, "boolop");
    const args = asList(nodeGet(boolExpr, "args"));
    const newArgs = args.map((a) => this._foldConstants(a));

    // Full fold: all args are constants
    const allConst = newArgs.every(
      (a) => isAConst(a) || nodeGet(a, "A_Const") !== null,
    );
    if (allConst) {
      try {
        const evaluator = new ExprEvaluator({ params: this._params });
        const folded = { BoolExpr: { boolop, args: newArgs } };
        const result = evaluator.evaluate(folded, {});
        return _valueToAConst(result);
      } catch {
        // Cannot fold
      }
    }

    // Partial fold for AND (boolop=0): remove True constants, short-circuit on False
    if (boolop === 0 || boolop === "AND_EXPR") {
      const surviving: Record<string, unknown>[] = [];
      for (const a of newArgs) {
        if (isAConst(a) || nodeGet(a, "A_Const") !== null) {
          const val = _constToBool(a);
          if (val === false) {
            return { A_Const: { boolval: false } };
          }
          // True -> skip (identity for AND)
          continue;
        }
        surviving.push(a);
      }
      if (surviving.length === 0) return { A_Const: { boolval: true } };
      if (surviving.length === 1) return surviving[0]!;
      const updated = { boolop: 0, args: surviving };
      return nodeGet(node, "BoolExpr") !== null ? { BoolExpr: updated } : updated;
    }

    // Partial fold for OR (boolop=1): remove False constants, short-circuit on True
    if (boolop === 1 || boolop === "OR_EXPR") {
      const surviving: Record<string, unknown>[] = [];
      for (const a of newArgs) {
        if (isAConst(a) || nodeGet(a, "A_Const") !== null) {
          const val = _constToBool(a);
          if (val === true) {
            return { A_Const: { boolval: true } };
          }
          // False -> skip (identity for OR)
          continue;
        }
        surviving.push(a);
      }
      if (surviving.length === 0) return { A_Const: { boolval: false } };
      if (surviving.length === 1) return surviving[0]!;
      const updated = { boolop: 1, args: surviving };
      return nodeGet(node, "BoolExpr") !== null ? { BoolExpr: updated } : updated;
    }

    // Return with updated args if anything changed
    const argsChanged = newArgs.some((a, i) => a !== args[i]);
    if (argsChanged) {
      const updated = { boolop, args: newArgs };
      return nodeGet(node, "BoolExpr") !== null ? { BoolExpr: updated } : updated;
    }
    return node;
  }

  /**
   * Fold a FuncCall if all args are constant and function is pure.
   */
  private _foldFuncCall(node: Record<string, unknown>): Record<string, unknown> {
    const fc = asObj(nodeGet(node, "FuncCall") ?? node);
    const funcName = getFuncName(fc);
    if (NO_FOLD_FUNCS.has(funcName)) return node;

    const funcArgs = asList(nodeGet(fc, "args"));
    if (funcArgs.length === 0) return node;

    const newArgs = funcArgs.map((a) => this._foldConstants(a));
    const changed = newArgs.some((a, i) => a !== funcArgs[i]);
    if (changed) {
      const updated = { ...fc, args: newArgs };
      return nodeGet(node, "FuncCall") !== null ? { FuncCall: updated } : updated;
    }
    return node;
  }

  // ==================================================================
  // Query optimization and plan execution
  // ==================================================================

  /**
   * Run the operator tree through the QueryOptimizer.
   */
  private _optimize(
    op: Operator,
    ctx: ExecutionContext,
    _table: Table | null,
  ): Operator {
    try {
      const idx = ctx.invertedIndex;
      if (!idx) return op;
      const stats = (idx as unknown as { stats: unknown }).stats;
      const columnStats = _table
        ? (_table as unknown as { _stats: unknown })._stats
        : null;
      const tableName = _table ? _table.name : undefined;
      const optimizer = new QueryOptimizer(stats as IndexStats, {
        columnStats: columnStats as Map<string, ColumnStats> | undefined,
        indexManager: ctx.indexManager ?? undefined,
        tableName,
      });
      return optimizer.optimize(op);
    } catch {
      return op;
    }
  }

  /**
   * Execute an operator tree via PlanExecutor.
   */
  private _executePlan(op: Operator, ctx: ExecutionContext): PostingList {
    try {
      const executor = new PlanExecutor(ctx);
      return executor.execute(op);
    } catch {
      // PlanExecutor not available, execute directly
      return op.execute(ctx);
    }
  }

  /**
   * Format the optimized query plan as an EXPLAIN result.
   */
  private _explainPlan(op: Operator, ctx: ExecutionContext): SQLResult {
    try {
      const executor = new PlanExecutor(ctx);
      const planText = executor.explain(op);
      const lines = planText.split("\n");
      const rows = lines.map((line: string) => ({ plan: line }));
      return { columns: ["plan"], rows };
    } catch {
      return { columns: ["plan"], rows: [{ plan: op.constructor.name }] };
    }
  }

  /**
   * Scan all documents from a table into a PostingList.
   */
  private _scanAll(ctx: ExecutionContext, limit?: number): PostingList {
    if (!ctx.documentStore) {
      return new PostingList([createPostingEntry(0, { score: 0.0 })]);
    }
    const ds = ctx.documentStore;
    let allIds = [...ds.docIds].sort((a, b) => a - b);
    if (limit !== undefined && limit < allIds.length) {
      allIds = allIds.slice(0, limit);
    }
    return new PostingList(allIds.map((d) => createPostingEntry(d, { score: 0.0 })));
  }

  /**
   * Chain a WHERE operator on top of a source operator.
   */
  private _chainOnSource(whereOp: Operator, sourceOp: Operator): Operator {
    if (whereOp instanceof FilterOperator && whereOp.source === null) {
      return new FilterOperator(whereOp.field, whereOp.predicate, sourceOp);
    }
    return new IntersectOperator([sourceOp, whereOp]);
  }

  /**
   * Check if a source operator is a graph operator.
   */
  private static _isGraphOperator(op: unknown): boolean {
    if (op === null || op === undefined) return false;
    const name = (op as { constructor: { name: string } }).constructor.name;
    return [
      "TraverseOperator",
      "RegularPathQueryOperator",
      "PatternMatchOperator",
      "TemporalTraverseOperator",
      "TemporalPatternMatchOperator",
      "CypherQueryOperator",
    ].includes(name);
  }

  /**
   * Check if a source operator is a join operator.
   */
  private static _isJoinOperator(op: unknown): boolean {
    if (op === null || op === undefined) return false;
    const name = (op as { constructor: { name: string } }).constructor.name;
    return [
      "InnerJoinOperator",
      "OuterJoinOperator",
      "CrossJoinOperator",
      "IndexJoinOperator",
      "SortMergeJoinOperator",
      "ExprJoinOperator",
      "LateralJoinOperator",
      "TableScanOperator",
    ].includes(name);
  }

  /**
   * Collect (alias, columns[]) pairs for all tables in a FROM clause.
   * Used by SELECT * on joins to expand columns in correct order.
   */
  private _collectJoinTables(
    fromClause: Record<string, unknown>[],
  ): [string, string[]][] {
    const result: [string, string[]][] = [];
    for (const fromNode of fromClause) {
      this._walkFromForTables(fromNode, result);
    }
    return result;
  }

  private _walkFromForTables(
    node: Record<string, unknown>,
    result: [string, string[]][],
  ): void {
    // RangeVar
    const rv = asObj(nodeGet(node, "RangeVar") ?? {});
    if (Object.keys(rv).length > 0) {
      const tName = extractRelName(rv);
      const alias = extractAlias(rv) ?? tName;
      const table = this._tables.get(tName);
      if (table) {
        result.push([alias, table.columnNames]);
      }
      return;
    }

    // JoinExpr
    const je = asObj(nodeGet(node, "JoinExpr") ?? {});
    if (Object.keys(je).length > 0) {
      const larg = nodeGet(je, "larg");
      const rarg = nodeGet(je, "rarg");
      if (larg !== null && larg !== undefined)
        this._walkFromForTables(asObj(larg), result);
      if (rarg !== null && rarg !== undefined)
        this._walkFromForTables(asObj(rarg), result);
      return;
    }

    // RangeSubselect
    const rss = asObj(nodeGet(node, "RangeSubselect") ?? {});
    if (Object.keys(rss).length > 0) {
      const alias = extractAlias(rss) ?? "_derived";
      const table = this._tables.get(alias);
      if (table) {
        result.push([alias, table.columnNames]);
      }
    }
  }

  /**
   * Apply deferred WHERE predicates as ExprEvaluator-based filter
   * on already-joined/graph-sourced rows. For simple column-to-constant
   * comparisons, uses direct predicate matching. For cross-table or
   * complex expressions, delegates to ExprEvaluator.
   */
  private _applyDeferredWhere(
    rows: Record<string, unknown>[],
    whereNode: Record<string, unknown>,
    evaluator: ExprEvaluator,
  ): Record<string, unknown>[] {
    return rows.filter((row) => {
      const result = evaluator.evaluate(whereNode, row);
      return result === true;
    });
  }

  // -- AST table reference collection -----------------------------------

  /**
   * Recursively collect all table names from an AST node.
   * Walks FROM, JOINs, subqueries, and CTEs.
   * Virtual schemas (information_schema, pg_catalog) are excluded.
   */
  private static _collectAstTableRefs(node: unknown): Set<string> {
    const refs = new Set<string>();
    SQLCompiler._walkAstForTables(node, refs);
    return refs;
  }

  private static _walkAstForTables(node: unknown, refs: Set<string>): void {
    if (node === null || node === undefined) return;
    if (typeof node !== "object") return;

    if (Array.isArray(node)) {
      for (const item of node) {
        SQLCompiler._walkAstForTables(item, refs);
      }
      return;
    }

    const obj = node as Record<string, unknown>;

    // Check for RangeVar
    const rv = obj["RangeVar"] as Record<string, unknown> | undefined;
    if (rv !== undefined) {
      const schema = nodeStr(rv, "schemaname");
      if (schema !== "information_schema" && schema !== "pg_catalog") {
        const relname = nodeStr(rv, "relname");
        if (relname) refs.add(relname);
      }
      return;
    }

    // Direct relname check
    const relname = obj["relname"];
    if (typeof relname === "string" && relname) {
      const schema = obj["schemaname"];
      if (schema !== "information_schema" && schema !== "pg_catalog") {
        refs.add(relname);
      }
    }

    // Recurse into child values
    for (const value of Object.values(obj)) {
      if (value !== null && typeof value === "object") {
        SQLCompiler._walkAstForTables(value, refs);
      }
    }
  }

  // -- Join predicate pushdown helpers ---------------------------------

  /**
   * Partition WHERE conjuncts into per-alias pushable and remaining.
   * Single-alias conjuncts are pushable; cross-table predicates remain.
   * Returns [pushable_per_alias, remaining_node].
   */
  private _partitionWhereForJoins(
    whereClause: Record<string, unknown>,
    fromAliases: Set<string>,
  ): [Map<string, Record<string, unknown>[]>, Record<string, unknown> | null] {
    const conjuncts = this._extractAndConjuncts(whereClause);
    const pushable = new Map<string, Record<string, unknown>[]>();
    const remaining: Record<string, unknown>[] = [];

    for (const conj of conjuncts) {
      const aliases = SQLCompiler._collectConjunctAliases(conj);
      if (aliases.size === 1) {
        const alias = aliases.values().next().value as string;
        if (fromAliases.has(alias)) {
          if (!pushable.has(alias)) pushable.set(alias, []);
          pushable.get(alias)!.push(conj);
          continue;
        }
      }
      remaining.push(conj);
    }

    const remainingNode = SQLCompiler._reconstructAnd(remaining);
    return [pushable, remainingNode];
  }

  /**
   * Collect table alias prefixes from ColumnRef nodes in an AST subtree.
   */
  private static _collectConjunctAliases(node: Record<string, unknown>): Set<string> {
    const aliases = new Set<string>();
    SQLCompiler._walkForColumnAliases(node, aliases);
    return aliases;
  }

  private static _walkForColumnAliases(
    node: Record<string, unknown>,
    aliases: Set<string>,
  ): void {
    // Check for ColumnRef with qualified name (table.column)
    const cr = asObj(nodeGet(node, "ColumnRef") ?? {});
    if (Object.keys(cr).length > 0) {
      const fields = asList(nodeGet(cr, "fields"));
      if (fields.length >= 2) {
        const first = fields[0]!;
        const alias = extractString(first);
        if (alias) aliases.add(alias);
      }
      return;
    }
    if (isColumnRef(node)) {
      const fields = asList(
        nodeGet(asObj(nodeGet(node, "ColumnRef") ?? node), "fields"),
      );
      if (fields.length >= 2) {
        const first = fields[0]!;
        const alias = extractString(first);
        if (alias) aliases.add(alias);
      }
      return;
    }

    // Recurse
    for (const attr of ["lexpr", "rexpr", "args", "arg", "xpr", "val"]) {
      const child = nodeGet(node, attr);
      if (child === null || child === undefined) continue;
      if (Array.isArray(child)) {
        for (const c of child as Record<string, unknown>[]) {
          if (typeof c === "object") {
            SQLCompiler._walkForColumnAliases(c, aliases);
          }
        }
      } else if (typeof child === "object") {
        SQLCompiler._walkForColumnAliases(child as Record<string, unknown>, aliases);
      }
    }

    // BoolExpr args
    const boolExpr = asObj(nodeGet(node, "BoolExpr") ?? {});
    if (Object.keys(boolExpr).length > 0) {
      const boolArgs = asList(nodeGet(boolExpr, "args"));
      for (const ba of boolArgs) {
        SQLCompiler._walkForColumnAliases(ba, aliases);
      }
    }

    // A_Expr children
    const aExpr = asObj(nodeGet(node, "A_Expr") ?? {});
    if (Object.keys(aExpr).length > 0) {
      const lexpr = nodeGet(aExpr, "lexpr");
      const rexpr = nodeGet(aExpr, "rexpr");
      if (lexpr !== null && typeof lexpr === "object") {
        SQLCompiler._walkForColumnAliases(lexpr as Record<string, unknown>, aliases);
      }
      if (rexpr !== null && typeof rexpr === "object") {
        SQLCompiler._walkForColumnAliases(rexpr as Record<string, unknown>, aliases);
      }
    }
  }

  /**
   * Rebuild a BoolExpr AND node from a list of conjuncts.
   */
  private static _reconstructAnd(
    conjuncts: Record<string, unknown>[],
  ): Record<string, unknown> | null {
    if (conjuncts.length === 0) return null;
    if (conjuncts.length === 1) return conjuncts[0]!;
    return { BoolExpr: { boolop: 0, args: conjuncts } };
  }

  /**
   * Extract equijoin predicates from WHERE for implicit cross joins.
   * Returns an array of {leftAlias, leftCol, rightAlias, rightCol} objects.
   */
  private _extractImplicitEquijoinPredicates(
    whereNode: Record<string, unknown>,
    aliasSet: Set<string>,
  ): { leftAlias: string; leftCol: string; rightAlias: string; rightCol: string }[] {
    const predicates: {
      leftAlias: string;
      leftCol: string;
      rightAlias: string;
      rightCol: string;
    }[] = [];

    // Get conjuncts
    const boolExpr = asObj(nodeGet(whereNode, "BoolExpr") ?? {});
    let conjuncts: Record<string, unknown>[];
    if (Object.keys(boolExpr).length > 0) {
      const boolop = nodeGet(boolExpr, "boolop");
      if (boolop === 0 || boolop === "AND_EXPR") {
        conjuncts = asList(nodeGet(boolExpr, "args"));
      } else {
        conjuncts = [whereNode];
      }
    } else {
      conjuncts = [whereNode];
    }

    for (const node of conjuncts) {
      const aExpr = asObj(nodeGet(node, "A_Expr") ?? {});
      if (Object.keys(aExpr).length === 0) continue;

      // Must be equality operator
      const nameList = asList(nodeGet(aExpr, "name"));
      if (nameList.length === 0) continue;
      const opName = extractString(nameList[nameList.length - 1]!);
      if (opName !== "=") continue;

      // Both sides must be column refs
      const lexpr = nodeGet(aExpr, "lexpr");
      const rexpr = nodeGet(aExpr, "rexpr");
      if (lexpr === null || rexpr === null) continue;
      if (!isColumnRef(asObj(lexpr)) || !isColumnRef(asObj(rexpr))) continue;

      // Both must be qualified (table.column)
      const leftCr = asObj(nodeGet(asObj(lexpr), "ColumnRef") ?? asObj(lexpr));
      const rightCr = asObj(nodeGet(asObj(rexpr), "ColumnRef") ?? asObj(rexpr));
      const leftFields = asList(nodeGet(leftCr, "fields"));
      const rightFields = asList(nodeGet(rightCr, "fields"));

      if (leftFields.length < 2 || rightFields.length < 2) continue;

      const leftAlias = extractString(leftFields[0]!);
      const leftCol = extractString(leftFields[leftFields.length - 1]!);
      const rightAlias = extractString(rightFields[0]!);
      const rightCol = extractString(rightFields[rightFields.length - 1]!);

      if (!leftAlias || !leftCol || !rightAlias || !rightCol) continue;
      if (!aliasSet.has(leftAlias) || !aliasSet.has(rightAlias)) continue;

      predicates.push({ leftAlias, leftCol, rightAlias, rightCol });
    }

    return predicates;
  }

  // -- FK validator registration ------------------------------------------

  /**
   * Register foreign key validators (insert/update/delete hooks).
   * Enforces referential integrity between child and parent tables.
   */
  private _registerFkValidators(
    tableName: string,
    table: Table,
    foreignKeys: ForeignKeyDef[],
  ): void {
    for (const fk of foreignKeys) {
      const parentTableName = fk.refTable;
      const childCol = fk.column;
      const parentCol = fk.refColumn;

      // Insert validator: child FK value must exist in parent
      table.fkInsertValidators.push((doc: Record<string, unknown>) => {
        const val = doc[childCol];
        if (val === null || val === undefined) return; // NULL FK allowed
        const parent = this._tables.get(parentTableName);
        if (!parent) {
          throw new Error(
            `Foreign key violation: parent table "${parentTableName}" does not exist`,
          );
        }
        const hasValue = parent.documentStore.hasValue(parentCol, val);
        if (!hasValue) {
          throw new Error(
            `Foreign key violation: ${tableName}.${childCol} = ${JSON.stringify(val)} ` +
              `not found in ${parentTableName}.${parentCol}`,
          );
        }
      });

      // Update validator: when child row changes FK column, new value must exist in parent
      table.fkUpdateValidators.push(
        (_oldDoc: Record<string, unknown>, newDoc: Record<string, unknown>) => {
          const val = newDoc[childCol];
          if (val === null || val === undefined) return; // NULL FK allowed
          const parent = this._tables.get(parentTableName);
          if (!parent) {
            throw new Error(
              `FOREIGN KEY constraint violated: parent table "${parentTableName}" does not exist`,
            );
          }
          const hasValue = parent.documentStore.hasValue(parentCol, val);
          if (!hasValue) {
            throw new Error(
              `FOREIGN KEY constraint violated: ${tableName}.${childCol} = ${JSON.stringify(val)} ` +
                `not found in ${parentTableName}.${parentCol}`,
            );
          }
        },
      );

      // Delete validator on parent: row cannot be deleted if referenced by child
      const parent = this._tables.get(parentTableName);
      if (parent) {
        const childTableName = tableName;
        const childTable = table;
        parent.fkDeleteValidators.push((docId: number) => {
          const parentDoc = parent.documentStore.get(docId);
          if (!parentDoc) return;
          const parentVal = parentDoc[parentCol];
          if (parentVal === null || parentVal === undefined) return;
          // Check if any child row references this value
          if (childTable.documentStore.hasValue(childCol, parentVal)) {
            throw new Error(
              `FOREIGN KEY constraint violated: row in "${parentTableName}" ` +
                `is still referenced from "${childTableName}"`,
            );
          }
        });

        // Update validator on parent: PK changes must not orphan child rows
        parent.fkUpdateValidators.push(
          (oldDoc: Record<string, unknown>, newDoc: Record<string, unknown>) => {
            const oldVal = oldDoc[parentCol];
            const newVal = newDoc[parentCol];
            // If the referenced column value did not change, no violation
            if (oldVal === newVal) return;
            if (oldVal === null || oldVal === undefined) return;
            // Check if any child row references the old value
            if (childTable.documentStore.hasValue(childCol, oldVal)) {
              throw new Error(
                `FOREIGN KEY constraint violated: row in "${parentTableName}" ` +
                  `is still referenced from "${childTableName}"`,
              );
            }
          },
        );
      }
    }
  }

  // -- Predicate pushdown helpers -----------------------------------------

  /**
   * Check if a WHERE predicate is safe to push down into a view/subquery.
   * Only simple column comparisons referencing subquery output columns qualify.
   */
  private _isPushablePredicate(
    node: Record<string, unknown>,
    subqueryColumns: Set<string>,
  ): boolean {
    // Simple A_Expr comparisons
    const aExpr = nodeGet(node, "A_Expr");
    if (aExpr !== null && aExpr !== undefined) {
      const expr = asObj(aExpr);
      const lexpr = nodeGet(expr, "lexpr");
      const rexpr = nodeGet(expr, "rexpr");
      // Left must be a column ref to a subquery column
      if (lexpr !== null && isColumnRef(asObj(lexpr))) {
        try {
          const col = extractColumnName(asObj(lexpr));
          if (!subqueryColumns.has(col)) return false;
        } catch {
          return false;
        }
      } else {
        return false;
      }
      // Right must be a constant or parameter
      if (rexpr !== null) {
        const rObj = asObj(rexpr);
        if (!isAConst(rObj) && !isParamRef(rObj)) return false;
      }
      return true;
    }

    // NullTest on a subquery column
    const nullTest = nodeGet(node, "NullTest");
    if (nullTest !== null && nullTest !== undefined) {
      const nt = asObj(nullTest);
      const arg = nodeGet(nt, "arg");
      if (arg !== null && isColumnRef(asObj(arg))) {
        try {
          const col = extractColumnName(asObj(arg));
          return subqueryColumns.has(col);
        } catch {
          return false;
        }
      }
    }

    return false;
  }

  /**
   * Push safe WHERE predicates into a view or derived table subquery.
   * Only applies when FROM is a single view/derived table and
   * WHERE references only the subquery's output columns.
   */
  private _tryPredicatePushdown(
    stmt: Record<string, unknown>,
    _params: unknown[],
  ): Record<string, unknown> {
    const whereClause = nodeGet(stmt, "whereClause");
    if (whereClause === null || whereClause === undefined) return stmt;
    const fromClause = asList(nodeGet(stmt, "fromClause"));
    if (fromClause.length !== 1) return stmt;

    const fromNode = fromClause[0]!;

    // Identify the subquery
    let subquery: Record<string, unknown> | null = null;
    const rv = asObj(nodeGet(fromNode, "RangeVar") ?? {});
    const rss = asObj(nodeGet(fromNode, "RangeSubselect") ?? {});

    if (Object.keys(rv).length > 0) {
      const tName = extractRelName(rv);
      // Check if it's a view
      if (this._views.has(tName)) {
        subquery = this._views.get(tName)!;
      } else if (this._inlinedCTEs.has(tName)) {
        subquery = this._inlinedCTEs.get(tName)!;
      }
    } else if (Object.keys(rss).length > 0) {
      const sq = nodeGet(rss, "subquery");
      if (sq !== null && sq !== undefined) {
        subquery = asObj(nodeGet(asObj(sq), "SelectStmt") ?? sq);
      }
    }

    if (subquery === null) return stmt;

    // Do not push into subqueries with aggregates, GROUP BY, DISTINCT, LIMIT, window
    if (
      nodeGet(subquery, "groupClause") !== null &&
      asList(nodeGet(subquery, "groupClause")).length > 0
    )
      return stmt;
    if (
      nodeGet(subquery, "distinctClause") !== null &&
      nodeGet(subquery, "distinctClause") !== undefined
    )
      return stmt;
    if (
      nodeGet(subquery, "limitCount") !== null &&
      nodeGet(subquery, "limitCount") !== undefined
    )
      return stmt;
    const sqTargetList = asList(nodeGet(subquery, "targetList"));
    for (const t of sqTargetList) {
      const rt = asObj(nodeGet(t, "ResTarget") ?? t);
      const val = nodeGet(rt, "val");
      if (
        val !== null &&
        val !== undefined &&
        isFuncCall(asObj(val)) &&
        hasOverClause(asObj(val))
      )
        return stmt;
    }
    if (this._hasAggregates(sqTargetList)) return stmt;

    // Collect output column names
    const subqueryColumns = new Set<string>();
    for (const t of sqTargetList) {
      const rt = asObj(nodeGet(t, "ResTarget") ?? t);
      const alias = nodeStr(rt, "name");
      if (alias) {
        subqueryColumns.add(alias);
      } else {
        const val = nodeGet(rt, "val");
        if (val !== null && isColumnRef(asObj(val))) {
          try {
            subqueryColumns.add(extractColumnName(asObj(val)));
          } catch {
            // skip
          }
        }
      }
    }
    if (subqueryColumns.size === 0) return stmt;

    // Split WHERE into pushable and remaining
    const [pushable, remaining] = this._splitPushable(
      asObj(whereClause),
      subqueryColumns,
    );
    if (pushable.length === 0) return stmt;

    // Build the pushed predicate
    const pushedPred =
      pushable.length === 1
        ? pushable[0]!
        : { BoolExpr: { boolop: 0, args: pushable } };

    // Inject into the subquery/view
    if (Object.keys(rv).length > 0) {
      const tName = extractRelName(rv);
      if (this._views.has(tName)) {
        this._views.set(
          tName,
          SQLCompiler._injectWhere(this._views.get(tName)!, pushedPred),
        );
      } else if (this._inlinedCTEs.has(tName)) {
        this._inlinedCTEs.set(
          tName,
          SQLCompiler._injectWhere(this._inlinedCTEs.get(tName)!, pushedPred),
        );
      }
    }

    // Return stmt with reduced WHERE
    return { ...stmt, whereClause: remaining };
  }

  /**
   * Split a WHERE clause into pushable and remaining predicates.
   */
  private _splitPushable(
    whereNode: Record<string, unknown>,
    subqueryColumns: Set<string>,
  ): [Record<string, unknown>[], Record<string, unknown> | null] {
    const boolExpr = asObj(nodeGet(whereNode, "BoolExpr") ?? {});
    if (Object.keys(boolExpr).length > 0) {
      const boolop = nodeGet(boolExpr, "boolop");
      if (boolop === 0 || boolop === "AND_EXPR") {
        const args = asList(nodeGet(boolExpr, "args"));
        const pushable: Record<string, unknown>[] = [];
        const remaining: Record<string, unknown>[] = [];
        for (const arg of args) {
          if (this._isPushablePredicate(arg, subqueryColumns)) {
            pushable.push(arg);
          } else {
            remaining.push(arg);
          }
        }
        let remainingNode: Record<string, unknown> | null = null;
        if (remaining.length === 1) {
          remainingNode = remaining[0]!;
        } else if (remaining.length > 1) {
          remainingNode = { BoolExpr: { boolop: 0, args: remaining } };
        }
        return [pushable, remainingNode];
      }
    }

    // Single predicate
    if (this._isPushablePredicate(whereNode, subqueryColumns)) {
      return [[whereNode], null];
    }
    return [[], whereNode];
  }

  /**
   * Return a copy of *query* with *predicate* AND-merged into WHERE.
   */
  private static _injectWhere(
    query: Record<string, unknown>,
    predicate: Record<string, unknown>,
  ): Record<string, unknown> {
    const existing = nodeGet(query, "whereClause");
    let newWhere: Record<string, unknown>;
    if (existing === null || existing === undefined) {
      newWhere = predicate;
    } else {
      newWhere = {
        BoolExpr: { boolop: 0, args: [existing, predicate] },
      };
    }
    return { ...query, whereClause: newWhere };
  }

  /**
   * Extract a numeric value from an AST constant node or parameter reference.
   */
  private _extractNumericValue(node: Record<string, unknown>): number {
    const val = extractConstValue(node, this._params);
    if (typeof val === "number") return val;
    if (typeof val === "string") {
      const n = Number(val);
      if (!isNaN(n)) return n;
    }
    throw new Error(
      "Expected numeric constant or $N parameter, got " +
        JSON.stringify(node).slice(0, 200),
    );
  }

  /**
   * Check if a target list contains SELECT *.
   */
  private static _isSelectStar(targetList: Record<string, unknown>[]): boolean {
    if (targetList.length === 0) return true;
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const val = nodeGet(resTarget, "val");
      if (val !== null && val !== undefined && isAStar(asObj(val))) {
        return true;
      }
    }
    return false;
  }

  /**
   * Check if any SELECT target is a computed expression (not a simple column).
   * This includes function calls, CASE, CAST, arithmetic, etc.
   */
  private static _hasComputedExpressions(
    targetList: Record<string, unknown>[],
  ): boolean {
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const val = nodeGet(resTarget, "val");
      if (val === null || val === undefined) continue;
      const valObj = asObj(val);
      if (isColumnRef(valObj)) continue;
      if (isFuncCall(valObj)) {
        const fn = getFuncName(valObj);
        if (AGG_FUNC_NAMES.has(fn)) continue;
        // UQA scoring functions are not computed expressions in the usual sense
        if (
          fn === "text_match" ||
          fn === "bayesian_match" ||
          fn === "bayesian_match_with_prior" ||
          fn === "knn_match" ||
          fn === "bayesian_knn_match" ||
          fn === "traverse_match" ||
          fn === "spatial_within"
        )
          continue;
        return true;
      }
      // A_Const, A_Expr, CaseExpr, TypeCast, CoalesceExpr, NullTest, SubLink
      if (
        isAConst(valObj) ||
        isAExpr(valObj) ||
        nodeGet(valObj, "A_Expr") !== null ||
        nodeGet(valObj, "CaseExpr") !== null ||
        isTypeCast(valObj) ||
        nodeGet(valObj, "CoalesceExpr") !== null ||
        isNullTest(valObj) ||
        isSubLink(valObj) ||
        nodeGet(valObj, "SubLink") !== null
      ) {
        return true;
      }
    }
    return false;
  }

  /**
   * Infer the output column name for a single SELECT target.
   * Follows PostgreSQL naming conventions.
   */
  private _inferTargetName(target: Record<string, unknown>): string {
    const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
    const alias = nodeStr(resTarget, "name");
    if (alias) return alias;

    const val = nodeGet(resTarget, "val");
    if (val === null || val === undefined) return "?column?";
    const valObj = asObj(val);

    if (isColumnRef(valObj)) {
      try {
        return extractColumnName(valObj);
      } catch {
        return "?column?";
      }
    }

    if (isFuncCall(valObj)) {
      const fn = getFuncName(valObj);
      if (AGG_FUNC_NAMES.has(fn)) {
        if (isAggStar(valObj)) return fn;
        const args = getFuncArgs(valObj);
        if (args.length > 0 && isColumnRef(args[0]!)) {
          try {
            const argCol = extractColumnName(args[0]!);
            return `${fn}_${argCol}`;
          } catch {
            return fn;
          }
        }
        return fn;
      }
      if (
        fn === "text_match" ||
        fn === "bayesian_match" ||
        fn === "bayesian_match_with_prior"
      ) {
        return "_score";
      }
      return fn;
    }

    if (isTypeCast(valObj)) {
      const tc = asObj(nodeGet(valObj, "TypeCast") ?? valObj);
      const arg = nodeGet(tc, "arg");
      if (arg !== null && isColumnRef(asObj(arg))) {
        try {
          return extractColumnName(asObj(arg));
        } catch {
          // fall through
        }
      }
    }

    return "?column?";
  }

  /**
   * Build (output_name, ast_node) pairs for SELECT targets.
   * Text search functions are mapped to read _score from the row.
   */
  private _buildExprTargets(
    targetList: Record<string, unknown>[],
  ): [string, Record<string, unknown>][] {
    const targets: [string, Record<string, unknown>][] = [];
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const outputName = this._inferTargetName(target);
      let val = nodeGet(resTarget, "val");
      if (val === null || val === undefined) continue;
      const valObj = asObj(val);
      if (isFuncCall(valObj)) {
        const fn = getFuncName(valObj);
        if (
          fn === "text_match" ||
          fn === "bayesian_match" ||
          fn === "bayesian_match_with_prior"
        ) {
          val = { ColumnRef: { fields: [{ String: { sval: "_score" } }] } };
        }
      }
      targets.push([outputName, asObj(val)]);
    }
    return targets;
  }

  /**
   * Check if ORDER BY references columns not in the SELECT list.
   */
  private _sortNeedsExtraCols(
    sortClause: Record<string, unknown>[],
    targetList: Record<string, unknown>[],
  ): boolean {
    const projected = new Set<string>();
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const val = nodeGet(resTarget, "val");
      if (val !== null && val !== undefined && isAStar(asObj(val))) return false;
      if (val !== null && isColumnRef(asObj(val))) {
        try {
          projected.add(extractColumnName(asObj(val)));
        } catch {
          // skip
        }
        const alias = nodeStr(resTarget, "name");
        if (alias) projected.add(alias);
      } else {
        const alias = nodeStr(resTarget, "name");
        if (alias) projected.add(alias);
      }
    }

    for (const item of sortClause) {
      const sortBy = asObj(nodeGet(item, "SortBy") ?? item);
      const sortNode = asObj(nodeGet(sortBy, "node"));
      if (isColumnRef(sortNode)) {
        try {
          const col = extractColumnName(sortNode);
          if (!projected.has(col)) return true;
        } catch {
          // skip
        }
      }
    }
    return false;
  }

  /**
   * Build group column -> alias mapping for GROUP BY queries.
   */
  private _buildGroupAliases(
    groupCols: string[],
    targetList: Record<string, unknown>[],
  ): Map<string, string> {
    const aliases = new Map<string, string>();
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const val = nodeGet(resTarget, "val");
      const alias = nodeStr(resTarget, "name");
      if (val !== null && isColumnRef(asObj(val)) && alias) {
        try {
          const col = extractColumnName(asObj(val));
          if (groupCols.includes(col)) {
            aliases.set(col, alias);
          }
        } catch {
          // skip
        }
      }
    }
    return aliases;
  }

  /**
   * Build ExprProjectOp targets for non-aggregate computed expressions
   * in a GROUP BY SELECT list. Returns null when no post-group computation needed.
   */
  private _buildPostGroupTargets(
    targetList: Record<string, unknown>[],
    groupCols: string[],
    aggAliases: Set<string>,
  ): [string, Record<string, unknown>][] | null {
    let hasComputed = false;
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const val = nodeGet(resTarget, "val");
      if (val === null || val === undefined) continue;
      const valObj = asObj(val);
      if (isFuncCall(valObj)) {
        const fn = getFuncName(valObj);
        if (!AGG_FUNC_NAMES.has(fn) || hasOverClause(valObj)) {
          hasComputed = true;
          break;
        }
      } else if (!isColumnRef(valObj)) {
        const name = nodeStr(resTarget, "name") || "?column?";
        if (!aggAliases.has(name)) {
          hasComputed = true;
          break;
        }
      }
    }

    if (!hasComputed) return null;

    const groupSet = new Set(groupCols);
    const targets: [string, Record<string, unknown>][] = [];
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const name = nodeStr(resTarget, "name") || this._inferTargetName(target);
      if (aggAliases.has(name) || groupSet.has(name)) {
        targets.push([name, { ColumnRef: { fields: [{ String: { sval: name } }] } }]);
      } else {
        const val = nodeGet(resTarget, "val");
        if (val !== null && isColumnRef(asObj(val))) {
          try {
            const col = extractColumnName(asObj(val));
            targets.push([
              name,
              { ColumnRef: { fields: [{ String: { sval: col } }] } },
            ]);
          } catch {
            targets.push([name, asObj(val)]);
          }
        } else {
          targets.push([name, asObj(val ?? {})]);
        }
      }
    }
    return targets;
  }

  /**
   * Build pre-aggregation targets for GROUP BY computed expressions
   * or aggregate expression args (e.g., SUM(CASE ...)).
   */
  private _buildPreAggTargets(
    groupClause: Record<string, unknown>[],
    groupCols: string[],
    aggSpecs: { inputExpr?: Record<string, unknown> }[],
    table: Table | null,
  ): [string, Record<string, unknown>][] | null {
    const exprTargets: [string, Record<string, unknown>][] = [];

    // GROUP BY computed expressions
    for (let idx = 0; idx < groupClause.length; idx++) {
      const g = groupClause[idx]!;
      if (isFuncCall(g) || nodeGet(g, "FuncCall") !== null) {
        if (idx < groupCols.length) {
          exprTargets.push([groupCols[idx]!, g]);
        }
      }
    }

    // Aggregate expression args
    for (const spec of aggSpecs) {
      if (spec.inputExpr !== undefined) {
        // The expression needs to be pre-computed
        exprTargets.push(["_agg_expr", spec.inputExpr]);
      }
    }

    if (exprTargets.length === 0) return null;

    // Pass through all table columns, then append computed expression columns
    const targets: [string, Record<string, unknown>][] = [];
    if (table) {
      for (const colName of table.columnNames) {
        targets.push([
          colName,
          { ColumnRef: { fields: [{ String: { sval: colName } }] } },
        ]);
      }
    }
    for (const [alias, node] of exprTargets) {
      targets.push([alias, node]);
    }
    return targets;
  }

  /**
   * Resolve GROUP BY items: column names, aliases, or ordinals.
   * Returns the list of group column names.
   */
  private _resolveGroupByCols(
    groupClause: Record<string, unknown>[],
    targetList: Record<string, unknown>[],
  ): string[] {
    // Build alias map from SELECT list
    const aliasMap = new Map<string, string>();
    const selectCols: string[] = [];
    for (const target of targetList) {
      const resTarget = asObj(nodeGet(target, "ResTarget") ?? target);
      const val = nodeGet(resTarget, "val");
      const alias = nodeStr(resTarget, "name");
      if (val !== null && isColumnRef(asObj(val))) {
        try {
          const col = extractColumnName(asObj(val));
          selectCols.push(alias || col);
          if (alias) aliasMap.set(alias, col);
        } catch {
          selectCols.push(alias || "?column?");
        }
      } else if (val !== null && isFuncCall(asObj(val))) {
        const fn = getFuncName(asObj(val));
        let name: string;
        if (isAggStar(asObj(val))) {
          name = fn;
        } else {
          const args = getFuncArgs(asObj(val));
          let colArg: string | null = null;
          for (const a of args) {
            if (isColumnRef(a)) {
              try {
                colArg = extractColumnName(a);
              } catch {
                // skip
              }
            }
          }
          name = colArg !== null ? `${fn}_${colArg}` : fn;
        }
        selectCols.push(alias || name);
      } else {
        selectCols.push(alias || "?column?");
      }
    }

    const result: string[] = [];
    for (const g of groupClause) {
      // Ordinal reference: GROUP BY 1, 2
      if (isAConst(g)) {
        const val = extractConstValue(g, []);
        if (typeof val === "number" && Number.isInteger(val)) {
          const idx = val - 1;
          if (idx >= 0 && idx < selectCols.length) {
            result.push(selectCols[idx]!);
            continue;
          }
          throw new Error(`GROUP BY position ${String(val)} is not in select list`);
        }
      }

      // Column reference
      if (isColumnRef(g)) {
        try {
          const col = extractColumnName(g);
          result.push(aliasMap.get(col) ?? col);
          continue;
        } catch {
          // fall through
        }
      }

      // FuncCall in GROUP BY
      if (isFuncCall(g) || nodeGet(g, "FuncCall") !== null) {
        const fn = getFuncName(g);
        // Match against SELECT targets by function name
        let matched: string | null = null;
        for (const target of targetList) {
          const rt = asObj(nodeGet(target, "ResTarget") ?? target);
          const tVal = nodeGet(rt, "val");
          const tAlias = nodeStr(rt, "name");
          if (
            tVal !== null &&
            isFuncCall(asObj(tVal)) &&
            getFuncName(asObj(tVal)) === fn &&
            tAlias
          ) {
            matched = tAlias;
            break;
          }
        }
        if (matched !== null) {
          result.push(matched);
        } else {
          const args = getFuncArgs(g);
          let colArg: string | null = null;
          for (const a of args) {
            if (isColumnRef(a)) {
              try {
                colArg = extractColumnName(a);
              } catch {
                // skip
              }
            }
          }
          result.push(colArg !== null ? `${fn}_${colArg}` : fn);
        }
        continue;
      }

      // Fallback
      try {
        result.push(extractColumnName(g));
      } catch {
        result.push("?column?");
      }
    }
    return result;
  }

  /**
   * Get a table by name, throwing if it doesn't exist.
   */
  private _getTable(tableName: string): Table {
    const table = this._tables.get(tableName);
    if (!table) {
      throw new Error(`Table "${tableName}" does not exist`);
    }
    return table;
  }

  /**
   * Substitute $N parameter references in an AST with A_Const values.
   */
  private _substituteParams(
    node: Record<string, unknown>,
    paramValues: Map<number, Record<string, unknown>>,
  ): Record<string, unknown> {
    // If this is a ParamRef, replace it
    const paramRef = nodeGet(node, "ParamRef");
    if (paramRef !== null && paramRef !== undefined) {
      const prObj = asObj(paramRef);
      const num = prObj["number"] as number;
      const replacement = paramValues.get(num);
      if (replacement !== undefined) {
        return replacement;
      }
      return node;
    }

    // Recursively process child nodes
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(node)) {
      if (Array.isArray(value)) {
        result[key] = (value as unknown[]).map((item: unknown): unknown => {
          if (item !== null && typeof item === "object") {
            return this._substituteParams(item as Record<string, unknown>, paramValues);
          }
          return item;
        });
      } else if (value !== null && typeof value === "object") {
        result[key] = this._substituteParams(
          value as Record<string, unknown>,
          paramValues,
        );
      } else {
        result[key] = value;
      }
    }
    return result;
  }

  /**
   * Check whether an AST node has outer-row references (correlated subquery).
   */
  private _hasOuterRefs(
    node: Record<string, unknown>,
    outerColumns: Set<string>,
  ): boolean {
    // Check ColumnRef against outer columns
    if (isColumnRef(node)) {
      try {
        const col = extractColumnName(node);
        if (outerColumns.has(col)) return true;
        const qual = extractQualifiedColumnName(node);
        if (outerColumns.has(qual)) return true;
      } catch {
        // Not a valid column ref
      }
    }

    // Recurse into child nodes
    for (const value of Object.values(node)) {
      if (Array.isArray(value)) {
        for (const item of value) {
          if (
            item !== null &&
            typeof item === "object" &&
            this._hasOuterRefs(item as Record<string, unknown>, outerColumns)
          ) {
            return true;
          }
        }
      } else if (value !== null && typeof value === "object") {
        if (this._hasOuterRefs(value as Record<string, unknown>, outerColumns)) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Build sort key specifications from an ORDER BY clause.
   * Handles column names, aliases, ordinal references, and NULLS FIRST/LAST.
   */
  private _buildSortKeys(
    sortClause: Record<string, unknown>[],
    targetList: Record<string, unknown>[],
  ): Array<{ column: string; ascending: boolean; nullsFirst: boolean }> {
    // Build ordinal and alias maps from the SELECT list
    const ordinalMap = new Map<number, string>();
    const aliasNames = new Set<string>();
    const originalToAlias = new Map<string, string>();

    for (let idx = 0; idx < targetList.length; idx++) {
      const resTarget = asObj(
        nodeGet(targetList[idx]!, "ResTarget") ?? targetList[idx],
      );
      const val = nodeGet(resTarget, "val");
      const name = nodeStr(resTarget, "name");
      if (val !== null && val !== undefined && !isAStar(asObj(val))) {
        let colName: string;
        try {
          colName = name || this._deriveColumnName(asObj(val));
        } catch {
          colName = name || "?column?";
        }
        ordinalMap.set(idx + 1, colName);
        if (name) {
          aliasNames.add(name);
          if (isColumnRef(asObj(val))) {
            try {
              const realCol = extractColumnName(asObj(val));
              originalToAlias.set(realCol, name);
            } catch {
              // ignore
            }
          }
        }
      }
    }

    const sortKeys: Array<{ column: string; ascending: boolean; nullsFirst: boolean }> =
      [];
    for (const sortItem of sortClause) {
      const item = asObj(nodeGet(sortItem, "SortBy") ?? sortItem);
      const sortNode = asObj(nodeGet(item, "node"));

      // Determine direction
      const dir = nodeGet(item, "sortby_dir");
      const isDesc = dir === 2 || dir === "SORTBY_DESC" || String(dir).includes("DESC");
      const ascending = !isDesc;

      // Determine nulls placement
      const nullsDir = nodeGet(item, "sortby_nulls");
      let nullsFirst: boolean;
      if (
        nullsDir === 1 ||
        nullsDir === "SORTBY_NULLS_FIRST" ||
        String(nullsDir).includes("FIRST")
      ) {
        nullsFirst = true;
      } else if (
        nullsDir === 2 ||
        nullsDir === "SORTBY_NULLS_LAST" ||
        String(nullsDir).includes("LAST")
      ) {
        nullsFirst = false;
      } else {
        // PostgreSQL default: NULLS FIRST for DESC, NULLS LAST for ASC
        nullsFirst = isDesc;
      }

      // Resolve column name: ordinal, alias, or column reference
      let col: string;
      if (isAConst(sortNode)) {
        const ordinal = Number(extractConstValue(sortNode, this._params));
        const mapped = ordinalMap.get(ordinal);
        if (!mapped) {
          throw new Error(`ORDER BY position ${String(ordinal)} is not in select list`);
        }
        col = mapped;
      } else {
        try {
          col = extractColumnName(sortNode);
        } catch {
          col = this._deriveColumnName(sortNode);
        }
        // If col is an original column that was aliased, use alias
        if (!aliasNames.has(col) && originalToAlias.has(col)) {
          col = originalToAlias.get(col)!;
        }
      }

      sortKeys.push({ column: col, ascending, nullsFirst });
    }

    return sortKeys;
  }
}

// ======================================================================
// Internal operator classes
// ======================================================================

/**
 * Scans all documents in the store.
 */
class ScanOperator extends Operator {
  execute(context: ExecutionContext): PostingList {
    const ds = context.documentStore;
    if (!ds) return new PostingList();
    const allIds = [...ds.docIds].sort((a, b) => a - b);
    return new PostingList(allIds.map((d) => createPostingEntry(d, { score: 0.0 })));
  }

  costEstimate(stats: IndexStats): number {
    return stats.totalDocs;
  }
}

/**
 * Scans a specific table, eagerly loading document fields.
 * Used by JOIN operators that need full document data for ON evaluation.
 */
class TableScanOperator extends Operator {
  readonly _table: Table;
  readonly _alias: string | null;

  constructor(table: Table, alias: string | null = null) {
    super();
    this._table = table;
    this._alias = alias;
  }

  execute(_context: ExecutionContext): PostingList {
    const entries: PostingEntryType[] = [];
    const alias = this._alias;
    const colNames = this._table.columnNames;
    const ds = this._table.documentStore;
    for (const docId of [...ds.docIds].sort((a, b) => a - b)) {
      const doc = ds.get(docId);
      const fields: Record<string, unknown> = doc ? { ...doc } : {};
      for (const colName of colNames) {
        if (!(colName in fields)) {
          fields[colName] = null;
        }
      }
      if (alias) {
        const qualified: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(fields)) {
          qualified[`${alias}.${k}`] = v;
        }
        Object.assign(fields, qualified);
      }
      entries.push(createPostingEntry(docId, { score: 0.0, fields }));
    }
    return new PostingList(entries);
  }

  costEstimate(_stats: { totalDocs: number }): number {
    return this._table.documentStore.docIds.size;
  }
}

/**
 * Execute compiled UQA operators against a table for JOIN predicate pushdown.
 */
export class CompiledWhereScanOperator extends Operator {
  private readonly _table: Table;
  private readonly _uqaOp: Operator;
  private readonly _ctx: ExecutionContext;
  private readonly _alias: string | null;
  private readonly _scalarFilter: Record<string, unknown> | null;
  private readonly _subqueryExecutor:
    | ((stmt: Record<string, unknown>, params: unknown[]) => SQLResult)
    | null;
  private readonly _params: unknown[];

  constructor(
    table: Table,
    uqaOp: Operator,
    ctx: ExecutionContext,
    alias: string | null = null,
    scalarFilter: Record<string, unknown> | null = null,
    subqueryExecutor:
      | ((stmt: Record<string, unknown>, params: unknown[]) => SQLResult)
      | null = null,
    params: unknown[] = [],
  ) {
    super();
    this._table = table;
    this._uqaOp = uqaOp;
    this._ctx = ctx;
    this._alias = alias;
    this._scalarFilter = scalarFilter;
    this._subqueryExecutor = subqueryExecutor;
    this._params = params;
  }

  execute(_context: ExecutionContext): PostingList {
    const pl = this._uqaOp.execute(this._ctx);

    let evaluator: ExprEvaluator | null = null;
    if (this._scalarFilter !== null) {
      evaluator = new ExprEvaluator({ params: this._params });
    }

    const alias = this._alias;
    const colNames = this._table.columnNames;
    const ds = this._table.documentStore;
    const entries: PostingEntryType[] = [];

    for (const entry of pl) {
      const doc = ds.get(entry.docId);
      if (!doc) continue;
      const fields: Record<string, unknown> = { ...doc };
      for (const colName of colNames) {
        if (!(colName in fields)) {
          fields[colName] = null;
        }
      }
      if (alias) {
        const qualified: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(fields)) {
          qualified[`${alias}.${k}`] = v;
        }
        Object.assign(fields, qualified);
      }
      if (evaluator && !evaluator.evaluate(this._scalarFilter!, fields)) {
        continue;
      }
      entries.push(
        createPostingEntry(entry.docId, {
          score: entry.payload.score,
          fields,
        }),
      );
    }
    return new PostingList(entries);
  }

  costEstimate(stats: IndexStats): number {
    return stats.totalDocs * 0.3;
  }
}

/**
 * Wraps a scan operator and filters its output using a WHERE predicate.
 */
export class FilteredScanOperator extends Operator {
  private readonly _scan: TableScanOperator;
  private readonly _whereNode: Record<string, unknown>;
  private readonly _subqueryExecutor:
    | ((stmt: Record<string, unknown>, params: unknown[]) => SQLResult)
    | null;
  private readonly _params: unknown[];

  constructor(
    scan: TableScanOperator,
    whereNode: Record<string, unknown>,
    subqueryExecutor:
      | ((stmt: Record<string, unknown>, params: unknown[]) => SQLResult)
      | null = null,
    params: unknown[] = [],
  ) {
    super();
    this._scan = scan;
    this._whereNode = whereNode;
    this._subqueryExecutor = subqueryExecutor;
    this._params = params;
  }

  execute(context: ExecutionContext): PostingList {
    const pl = this._scan.execute(context);
    const evaluator = new ExprEvaluator({ params: this._params });
    const filtered: PostingEntryType[] = [];
    for (const entry of pl) {
      if (
        evaluator.evaluate(
          this._whereNode,
          entry.payload.fields as Record<string, unknown>,
        )
      ) {
        filtered.push(entry);
      }
    }
    return new PostingList(filtered);
  }

  costEstimate(stats: IndexStats): number {
    return this._scan.costEstimate(stats) * 0.5;
  }
}

/**
 * Filter rows using an arbitrary expression via ExprEvaluator.
 * Used for WHERE clauses that cannot be reduced to simple FilterOperator.
 */
class ExprFilterOperator extends Operator {
  readonly exprNode: Record<string, unknown>;
  private readonly _subqueryExecutor:
    | ((stmt: Record<string, unknown>) => SQLResult)
    | null;

  constructor(
    exprNode: Record<string, unknown>,
    subqueryExecutor: ((stmt: Record<string, unknown>) => SQLResult) | null = null,
  ) {
    super();
    this.exprNode = exprNode;
    this._subqueryExecutor = subqueryExecutor;
  }

  execute(context: ExecutionContext): PostingList {
    const evaluator = new ExprEvaluator();
    const ds = context.documentStore;
    if (!ds) return new PostingList();

    const entries: PostingEntryType[] = [];
    for (const docId of [...ds.docIds].sort((a, b) => a - b)) {
      const doc = ds.get(docId);
      if (!doc) continue;
      const result = evaluator.evaluate(this.exprNode, doc);
      if (result) {
        entries.push(createPostingEntry(docId, { score: 0.0 }));
      }
    }
    return new PostingList(entries);
  }

  costEstimate(stats: IndexStats): number {
    return stats.totalDocs;
  }
}

/**
 * Internal operator for Bayesian BM25 with external prior.
 */
class ExternalPriorSearchOperator extends Operator {
  readonly source: Operator;
  readonly scorer: unknown;
  readonly terms: string[];
  readonly field: string | null;
  readonly documentStore: unknown;

  constructor(
    source: Operator,
    scorer: unknown,
    terms: string[],
    field: string | null,
    documentStore: unknown,
  ) {
    super();
    this.source = source;
    this.scorer = scorer;
    this.terms = terms;
    this.field = field;
    this.documentStore = documentStore;
  }

  execute(context: ExecutionContext): PostingList {
    const sourcePl = this.source.execute(context);
    const docStore = (this.documentStore ?? context.documentStore) as {
      get(id: number): Record<string, unknown> | null;
    } | null;
    const idx = context.invertedIndex;
    const entries: PostingEntryType[] = [];

    const scorerObj = this.scorer as {
      scoreWithPrior(
        tf: number,
        docLength: number,
        docFreq: number,
        docFields: Record<string, unknown>,
      ): number;
    };

    for (const entry of sourcePl) {
      const docId = entry.docId;
      const docFields = docStore?.get(docId) ?? {};
      const tf =
        entry.payload.positions.length > 0 ? entry.payload.positions.length : 1;
      const fieldKey = this.field ?? "_default";
      let docLength = tf;
      const idxObj = idx as unknown as Record<string, unknown>;
      if (idx && typeof idxObj["getDocLength"] === "function") {
        docLength = (
          idx as unknown as { getDocLength(id: number, field: string): number }
        ).getDocLength(docId, fieldKey);
      }
      const docFreq = sourcePl.length;

      const score = scorerObj.scoreWithPrior(tf, docLength, docFreq, docFields);
      entries.push(createPostingEntry(docId, { score }));
    }

    return new PostingList(entries);
  }

  costEstimate(stats: IndexStats): number {
    return this.source.costEstimate(stats) * 1.1;
  }
}

/**
 * KNN search with scores calibrated to probabilities.
 * P_vector = (1 + cosine_similarity) / 2  (Definition 7.1.2, Paper 3)
 */
class CalibratedKNNOperator extends Operator {
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
    // Try vector index first, then fallback to brute-force KNN
    const vecIdx = context.vectorIndexes?.[this.field];
    let rawPl: PostingList;
    if (vecIdx) {
      rawPl = (
        vecIdx as unknown as {
          searchKnn(vec: Float64Array, k: number): PostingList;
        }
      ).searchKnn(this.queryVector, this.k);
    } else {
      // Brute-force KNN
      const knnOp = new KNNOperator(this.queryVector, this.k, this.field);
      rawPl = knnOp.execute(context);
    }
    // Calibrate: P = (1 + cos) / 2
    const entries: PostingEntryType[] = [];
    for (const e of rawPl) {
      entries.push(
        createPostingEntry(e.docId, {
          score: (1.0 + e.payload.score) / 2.0,
        }),
      );
    }
    return new PostingList(entries);
  }

  costEstimate(stats: IndexStats): number {
    return (stats.dimensions || 128) * Math.log2(stats.totalDocs + 1);
  }
}

/**
 * Lazy semi-join filter for decorrelated EXISTS subqueries.
 */
export class SemiJoinFilterOperator extends Operator {
  private readonly _outerCol: string;
  private readonly _innerCol: string;
  private readonly _innerSubselect: Record<string, unknown>;
  private readonly _subqueryExecutor: (
    stmt: Record<string, unknown>,
    params?: unknown[],
  ) => SQLResult;

  constructor(
    outerCol: string,
    innerCol: string,
    innerSubselect: Record<string, unknown>,
    subqueryExecutor: (stmt: Record<string, unknown>, params?: unknown[]) => SQLResult,
  ) {
    super();
    this._outerCol = outerCol;
    this._innerCol = innerCol;
    this._innerSubselect = innerSubselect;
    this._subqueryExecutor = subqueryExecutor;
  }

  execute(context: ExecutionContext): PostingList {
    const innerResult = this._subqueryExecutor(this._innerSubselect);
    if (innerResult.rows.length === 0) {
      return new PostingList();
    }

    const values = new Set<unknown>();
    for (const row of innerResult.rows) {
      const v = row[this._innerCol];
      if (v !== null && v !== undefined) values.add(v);
    }
    return new FilterOperator(this._outerCol, new InSet(values)).execute(context);
  }

  costEstimate(stats: IndexStats): number {
    return stats.totalDocs * 1.5;
  }
}

/**
 * Nested-loop join with arbitrary ON expression evaluation.
 */
export class ExprJoinOperator extends Operator {
  private readonly _left: Operator;
  private readonly _right: Operator;
  private readonly _quals: Record<string, unknown>;
  private readonly _joinType: number | string;
  private readonly _params: unknown[];

  constructor(
    left: Operator,
    right: Operator,
    quals: Record<string, unknown>,
    joinType: number | string,
    params: unknown[] = [],
  ) {
    super();
    this._left = left;
    this._right = right;
    this._quals = quals;
    this._joinType = joinType;
    this._params = params;
  }

  execute(context: ExecutionContext): PostingList {
    const evaluator = new ExprEvaluator({ params: this._params });
    const leftEntries = [...this._left.execute(context)];
    const rightEntries = [...this._right.execute(context)];

    const jt = this._joinType;
    const quals = this._quals;

    // JOIN_INNER = 0
    if (jt === 0 || jt === "JOIN_INNER") {
      return ExprJoinOperator._inner(evaluator, quals, leftEntries, rightEntries);
    }
    // JOIN_LEFT = 1
    if (jt === 1 || jt === "JOIN_LEFT") {
      return ExprJoinOperator._leftOuter(evaluator, quals, leftEntries, rightEntries);
    }
    // JOIN_RIGHT = 2
    if (jt === 2 || jt === "JOIN_RIGHT") {
      return ExprJoinOperator._rightOuter(evaluator, quals, leftEntries, rightEntries);
    }
    // JOIN_FULL = 3
    if (jt === 3 || jt === "JOIN_FULL") {
      return ExprJoinOperator._fullOuter(evaluator, quals, leftEntries, rightEntries);
    }
    throw new Error(`Unsupported join type for expression join: ${String(jt)}`);
  }

  private static _inner(
    evaluator: ExprEvaluator,
    quals: Record<string, unknown>,
    leftEntries: PostingEntryType[],
    rightEntries: PostingEntryType[],
  ): PostingList {
    const result: PostingEntryType[] = [];
    for (const left of leftEntries) {
      for (const right of rightEntries) {
        const merged = {
          ...(left.payload.fields as Record<string, unknown>),
          ...(right.payload.fields as Record<string, unknown>),
        };
        if (evaluator.evaluate(quals, merged)) {
          result.push(
            createPostingEntry(left.docId, {
              score: left.payload.score + right.payload.score,
              fields: merged,
            }),
          );
        }
      }
    }
    return new PostingList(result);
  }

  private static _leftOuter(
    evaluator: ExprEvaluator,
    quals: Record<string, unknown>,
    leftEntries: PostingEntryType[],
    rightEntries: PostingEntryType[],
  ): PostingList {
    const result: PostingEntryType[] = [];
    for (const left of leftEntries) {
      let matched = false;
      for (const right of rightEntries) {
        const merged = {
          ...(left.payload.fields as Record<string, unknown>),
          ...(right.payload.fields as Record<string, unknown>),
        };
        if (evaluator.evaluate(quals, merged)) {
          matched = true;
          result.push(
            createPostingEntry(left.docId, {
              score: left.payload.score + right.payload.score,
              fields: merged,
            }),
          );
        }
      }
      if (!matched) {
        result.push(
          createPostingEntry(left.docId, {
            score: left.payload.score,
            fields: { ...(left.payload.fields as Record<string, unknown>) },
          }),
        );
      }
    }
    return new PostingList(result);
  }

  private static _rightOuter(
    evaluator: ExprEvaluator,
    quals: Record<string, unknown>,
    leftEntries: PostingEntryType[],
    rightEntries: PostingEntryType[],
  ): PostingList {
    const result: PostingEntryType[] = [];
    const matchedRight = new Set<number>();
    for (const left of leftEntries) {
      for (const right of rightEntries) {
        const merged = {
          ...(left.payload.fields as Record<string, unknown>),
          ...(right.payload.fields as Record<string, unknown>),
        };
        if (evaluator.evaluate(quals, merged)) {
          matchedRight.add(right.docId);
          result.push(
            createPostingEntry(left.docId, {
              score: left.payload.score + right.payload.score,
              fields: merged,
            }),
          );
        }
      }
    }
    for (const right of rightEntries) {
      if (!matchedRight.has(right.docId)) {
        result.push(
          createPostingEntry(right.docId, {
            score: right.payload.score,
            fields: { ...(right.payload.fields as Record<string, unknown>) },
          }),
        );
      }
    }
    return new PostingList(result);
  }

  private static _fullOuter(
    evaluator: ExprEvaluator,
    quals: Record<string, unknown>,
    leftEntries: PostingEntryType[],
    rightEntries: PostingEntryType[],
  ): PostingList {
    const result: PostingEntryType[] = [];
    const matchedRight = new Set<number>();
    for (const left of leftEntries) {
      let matched = false;
      for (const right of rightEntries) {
        const merged = {
          ...(left.payload.fields as Record<string, unknown>),
          ...(right.payload.fields as Record<string, unknown>),
        };
        if (evaluator.evaluate(quals, merged)) {
          matched = true;
          matchedRight.add(right.docId);
          result.push(
            createPostingEntry(left.docId, {
              score: left.payload.score + right.payload.score,
              fields: merged,
            }),
          );
        }
      }
      if (!matched) {
        result.push(
          createPostingEntry(left.docId, {
            score: left.payload.score,
            fields: { ...(left.payload.fields as Record<string, unknown>) },
          }),
        );
      }
    }
    for (const right of rightEntries) {
      if (!matchedRight.has(right.docId)) {
        result.push(
          createPostingEntry(right.docId, {
            score: right.payload.score,
            fields: { ...(right.payload.fields as Record<string, unknown>) },
          }),
        );
      }
    }
    return new PostingList(result);
  }
}

/**
 * LATERAL subquery join operator.
 */
export class LateralJoinOperator extends Operator {
  private readonly _left: Operator;
  private readonly _subquery: Record<string, unknown>;
  private readonly _alias: string;
  private readonly _subqueryExecutor: (
    stmt: Record<string, unknown>,
    params: unknown[],
    outerRow?: Record<string, unknown>,
  ) => SQLResult;

  constructor(
    left: Operator,
    subquery: Record<string, unknown>,
    alias: string,
    subqueryExecutor: (
      stmt: Record<string, unknown>,
      params: unknown[],
      outerRow?: Record<string, unknown>,
    ) => SQLResult,
  ) {
    super();
    this._left = left;
    this._subquery = subquery;
    this._alias = alias;
    this._subqueryExecutor = subqueryExecutor;
  }

  execute(context: ExecutionContext): PostingList {
    const leftEntries = [...this._left.execute(context)];
    const result: PostingEntryType[] = [];
    const alias = this._alias;

    for (const leftEntry of leftEntries) {
      const leftFields = leftEntry.payload.fields as Record<string, unknown>;
      const outerRow: Record<string, unknown> = { ...leftFields };
      const subResult = this._subqueryExecutor(this._subquery, [], outerRow);

      for (const row of subResult.rows) {
        const rightFields: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(row)) {
          rightFields[k] = v;
          rightFields[`${alias}.${k}`] = v;
        }
        const merged = { ...leftFields, ...rightFields };
        result.push(
          createPostingEntry(leftEntry.docId, {
            score: leftEntry.payload.score,
            fields: merged,
          }),
        );
      }
    }

    return new PostingList(result);
  }
}

// ======================================================================
// Module-level helpers
// ======================================================================

/**
 * Convert an operator comparison string to a Predicate object.
 */
function _opToPredicate(opName: string, value: unknown): Predicate {
  switch (opName) {
    case "=":
      return new Equals(value);
    case "!=":
    case "<>":
      return new NotEquals(value);
    case ">":
      return new GreaterThan(Number(value));
    case ">=":
      return new GreaterThanOrEqual(Number(value));
    case "<":
      return new LessThan(Number(value));
    case "<=":
      return new LessThanOrEqual(Number(value));
    default:
      throw new Error(`Unsupported operator: ${opName}`);
  }
}

/**
 * Convert a Python value to an A_Const-like AST node.
 */
function _valueToAConst(value: unknown): Record<string, unknown> {
  if (value === null || value === undefined) {
    return { A_Const: { isnull: true } };
  }
  if (typeof value === "boolean") {
    return { A_Const: { boolval: value } };
  }
  if (typeof value === "number") {
    if (Number.isInteger(value)) {
      return { A_Const: { ival: value } };
    }
    return { A_Const: { fval: String(value) } };
  }
  if (typeof value === "string") {
    return { A_Const: { sval: value } };
  }
  throw new Error(`Cannot convert ${typeof value} to A_Const`);
}

/**
 * Extract boolean truth value from an A_Const node.
 */
function _constToBool(node: Record<string, unknown>): boolean | null {
  const aConst = asObj(nodeGet(node, "A_Const") ?? node);
  if (nodeGet(aConst, "isnull") === true) return null;
  const boolval = nodeGet(aConst, "boolval");
  if (boolval !== null && boolval !== undefined) return Boolean(boolval);
  const ival = nodeGet(aConst, "ival");
  if (ival !== null && ival !== undefined) return Number(ival) !== 0;
  const fval = nodeGet(aConst, "fval");
  if (fval !== null && fval !== undefined)
    return parseFloat(String(fval as string | number)) !== 0.0;
  const sval = nodeGet(aConst, "sval");
  if (sval !== null && sval !== undefined)
    return String(sval as string | number).length > 0;
  return null;
}
