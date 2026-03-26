//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- semi and anti joins
// 1:1 port of uqa/joins/semi.py

import type { IndexStats, PostingEntry } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "../operators/base.js";
import { Operator } from "../operators/base.js";

export type JoinConditionFn = (left: PostingEntry, right: PostingEntry) => boolean;

export class SemiJoinOperator extends Operator {
  private readonly _left: Operator;
  private readonly _right: Operator;
  private readonly _condition: JoinConditionFn | null;

  constructor(left: Operator, right: Operator, condition?: JoinConditionFn | null) {
    super();
    this._left = left;
    this._right = right;
    this._condition = condition ?? null;
  }

  execute(context: ExecutionContext): PostingList {
    const leftPl = this._left.execute(context);
    const rightPl = this._right.execute(context);

    if (this._condition === null) {
      const rightIds = rightPl.docIds;
      const result = leftPl.entries.filter((e) => rightIds.has(e.docId));
      return PostingList.fromSorted(result);
    }

    const rightEntries = rightPl.entries;
    const result: PostingEntry[] = [];
    for (const leftEntry of leftPl) {
      let matched = false;
      for (const rightEntry of rightEntries) {
        if (this._condition(leftEntry, rightEntry)) {
          matched = true;
          break;
        }
      }
      if (matched) result.push(leftEntry);
    }
    return PostingList.fromSorted(result);
  }

  costEstimate(stats: IndexStats): number {
    return this._left.costEstimate(stats) + this._right.costEstimate(stats);
  }
}

export class AntiJoinOperator extends Operator {
  private readonly _left: Operator;
  private readonly _right: Operator;
  private readonly _condition: JoinConditionFn | null;

  constructor(left: Operator, right: Operator, condition?: JoinConditionFn | null) {
    super();
    this._left = left;
    this._right = right;
    this._condition = condition ?? null;
  }

  execute(context: ExecutionContext): PostingList {
    const leftPl = this._left.execute(context);
    const rightPl = this._right.execute(context);

    if (this._condition === null) {
      const rightIds = rightPl.docIds;
      const result = leftPl.entries.filter((e) => !rightIds.has(e.docId));
      return PostingList.fromSorted(result);
    }

    const rightEntries = rightPl.entries;
    const result: PostingEntry[] = [];
    for (const leftEntry of leftPl) {
      let matched = false;
      for (const rightEntry of rightEntries) {
        if (this._condition(leftEntry, rightEntry)) {
          matched = true;
          break;
        }
      }
      if (!matched) result.push(leftEntry);
    }
    return PostingList.fromSorted(result);
  }

  costEstimate(stats: IndexStats): number {
    return this._left.costEstimate(stats) + this._right.costEstimate(stats);
  }
}
