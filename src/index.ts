//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Public API re-exports

// Core
export { PostingList, GeneralizedPostingList } from "./core/posting-list.js";
export type { DocId, Payload, PostingEntry, Vertex, Edge } from "./core/types.js";
export {
  IndexStats,
  Equals,
  NotEquals,
  GreaterThan,
  GreaterThanOrEqual,
  LessThan,
  LessThanOrEqual,
  InSet,
  Between,
  IsNull,
  IsNotNull,
  Like,
  ILike,
} from "./core/types.js";
export { HierarchicalDocument } from "./core/hierarchical.js";

// Analysis
export {
  Analyzer,
  standardAnalyzer,
  whitespaceAnalyzer,
  keywordAnalyzer,
} from "./analysis/analyzer.js";

// Storage
export { MemoryDocumentStore } from "./storage/document-store.js";
export { MemoryInvertedIndex } from "./storage/inverted-index.js";
export { FlatVectorIndex } from "./storage/vector-index.js";
export { MemoryGraphStore } from "./graph/store.js";

// Scoring
export { BM25Scorer, createBM25Params } from "./scoring/bm25.js";
export {
  BayesianBM25Scorer,
  createBayesianBM25Params,
} from "./scoring/bayesian-bm25.js";

// Operators
export { Operator } from "./operators/base.js";
export type { ExecutionContext } from "./operators/base.js";
export { TermOperator, KNNOperator, FilterOperator } from "./operators/primitive.js";
export {
  UnionOperator,
  IntersectOperator,
  ComplementOperator,
} from "./operators/boolean.js";

// Engine
export { Engine } from "./engine.js";
export { QueryBuilder, AggregateResult, FacetResult } from "./api/query-builder.js";

// SQL
export type { SQLResult } from "./sql/compiler.js";
export { Table } from "./sql/table.js";
