//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Operator base classes
// 1:1 port of uqa/operators/base.py

import type { IndexStats } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { DocumentStore } from "../storage/abc/document-store.js";
import type { InvertedIndex } from "../storage/abc/inverted-index.js";
import type { VectorIndex } from "../storage/vector-index.js";
import type { SpatialIndex } from "../storage/spatial-index.js";
import type { BlockMaxIndex } from "../storage/block-max-index.js";
import type { IndexManager } from "../storage/index-manager.js";

export interface ExecutionContext {
  documentStore?: DocumentStore | null;
  invertedIndex?: InvertedIndex | null;
  vectorIndexes?: Record<string, VectorIndex>;
  spatialIndexes?: Record<string, SpatialIndex>;
  graphStore?: unknown;
  pathIndex?: unknown;
  blockMaxIndex?: BlockMaxIndex | null;
  indexManager?: IndexManager | null;
  parallelExecutor?: unknown;
  subgraphIndex?: unknown;
}

export abstract class Operator {
  abstract execute(context: ExecutionContext): PostingList;

  compose(other: Operator): ComposedOperator {
    return new ComposedOperator([this, other]);
  }

  costEstimate(stats: IndexStats): number {
    return stats.totalDocs;
  }
}

export class ComposedOperator extends Operator {
  readonly operators: Operator[];

  constructor(operators: Operator[]) {
    super();
    this.operators = operators;
  }

  execute(context: ExecutionContext): PostingList {
    let result: PostingList | null = null;
    for (const op of this.operators) {
      result = op.execute(context);
    }
    // Return last result, or empty if no operators
    if (result === null) {
      return new PostingList();
    }
    return result;
  }

  costEstimate(stats: IndexStats): number {
    let sum = 0;
    for (const op of this.operators) {
      sum += op.costEstimate(stats);
    }
    return sum;
  }
}
