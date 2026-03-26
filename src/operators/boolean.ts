//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- boolean operators
// 1:1 port of uqa/operators/boolean.py

import type { IndexStats } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "./base.js";
import { Operator } from "./base.js";

export class UnionOperator extends Operator {
  readonly operands: Operator[];

  constructor(operands: Operator[]) {
    super();
    this.operands = operands;
  }

  execute(context: ExecutionContext): PostingList {
    const results = this.operands.map((op) => op.execute(context));
    let acc = new PostingList();
    for (const r of results) {
      acc = acc.union(r);
    }
    return acc;
  }

  costEstimate(stats: IndexStats): number {
    let sum = 0;
    for (const op of this.operands) {
      sum += op.costEstimate(stats);
    }
    return sum;
  }
}

export class IntersectOperator extends Operator {
  readonly operands: Operator[];

  constructor(operands: Operator[]) {
    super();
    this.operands = operands;
  }

  execute(context: ExecutionContext): PostingList {
    if (this.operands.length === 0) return new PostingList();

    let acc = this.operands[0]!.execute(context);
    for (let i = 1; i < this.operands.length; i++) {
      if (acc.length === 0) return acc;
      acc = acc.intersect(this.operands[i]!.execute(context));
    }
    return acc;
  }

  costEstimate(stats: IndexStats): number {
    if (this.operands.length === 0) return 0;
    let min = Infinity;
    for (const op of this.operands) {
      const c = op.costEstimate(stats);
      if (c < min) min = c;
    }
    return min;
  }
}

export class ComplementOperator extends Operator {
  readonly operand: Operator;

  constructor(operand: Operator) {
    super();
    this.operand = operand;
  }

  execute(context: ExecutionContext): PostingList {
    const result = this.operand.execute(context);
    const docStore = context.documentStore;
    if (!docStore) return new PostingList();

    const universal = PostingList.fromSorted(
      [...docStore.docIds]
        .sort((a, b) => a - b)
        .map((docId) => ({ docId, payload: createPayload({ score: 0.0 }) })),
    );
    return result.complement(universal);
  }

  costEstimate(stats: IndexStats): number {
    return stats.totalDocs;
  }
}
