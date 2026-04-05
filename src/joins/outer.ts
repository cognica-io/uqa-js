//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- outer joins
// 1:1 port of uqa/joins/outer.py

import type { GeneralizedPostingEntry } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { GeneralizedPostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "../operators/base.js";
import type { JoinCondition, JoinEntry } from "./base.js";
import { JoinOperator, entryDocId } from "./base.js";

// -- LeftOuterJoinOperator ---------------------------------------------------

export class LeftOuterJoinOperator extends JoinOperator {
  private readonly _rightColumns: string[] | null;

  constructor(
    left: unknown,
    right: unknown,
    condition: JoinCondition,
    rightColumns?: string[] | null,
  ) {
    super(left, right, condition);
    this._rightColumns = rightColumns ?? null;
  }

  execute(context: ExecutionContext): GeneralizedPostingList {
    const leftEntries = this._getEntries(this.left, context);
    const rightEntries = this._getEntries(this.right, context);

    // Collect right-side field names for NULL padding on unmatched rows
    const rightFieldNames = new Set<string>();
    if (this._rightColumns) {
      for (const c of this._rightColumns) rightFieldNames.add(c);
    }
    if (rightEntries.length > 0) {
      for (const k of Object.keys(
        rightEntries[0]!.payload.fields as Record<string, unknown>,
      )) {
        rightFieldNames.add(k);
      }
    }

    // Build index on right
    const rightIndex = new Map<unknown, JoinEntry[]>();
    for (const entry of rightEntries) {
      const key = (entry.payload.fields as Record<string, unknown>)[
        this.condition.rightField
      ];
      if (key === undefined || key === null) continue;
      let bucket = rightIndex.get(key);
      if (!bucket) {
        bucket = [];
        rightIndex.set(key, bucket);
      }
      bucket.push(entry);
    }

    this.checkCancelled();
    const result: GeneralizedPostingEntry[] = [];
    for (const leftEntry of leftEntries) {
      const leftKey = (leftEntry.payload.fields as Record<string, unknown>)[
        this.condition.leftField
      ];
      const matches = leftKey != null ? rightIndex.get(leftKey) : undefined;

      if (matches && matches.length > 0) {
        for (const rightEntry of matches) {
          result.push({
            docIds: [entryDocId(leftEntry), entryDocId(rightEntry)],
            payload: createPayload({
              score: leftEntry.payload.score + rightEntry.payload.score,
              fields: {
                ...(leftEntry.payload.fields as Record<string, unknown>),
                ...(rightEntry.payload.fields as Record<string, unknown>),
              },
            }),
          });
        }
      } else {
        // No match: preserve left with right-side columns as null
        const fields: Record<string, unknown> = {
          ...(leftEntry.payload.fields as Record<string, unknown>),
        };
        for (const rk of rightFieldNames) {
          if (!(rk in fields)) {
            fields[rk] = null;
          }
        }
        result.push({
          docIds: [entryDocId(leftEntry)],
          payload: createPayload({
            score: leftEntry.payload.score,
            fields,
          }),
        });
      }
    }

    return new GeneralizedPostingList(result);
  }
}

// -- RightOuterJoinOperator --------------------------------------------------

export class RightOuterJoinOperator extends JoinOperator {
  private readonly _inner: LeftOuterJoinOperator;

  constructor(
    left: unknown,
    right: unknown,
    condition: JoinCondition,
    leftColumns?: string[] | null,
  ) {
    super(left, right, condition);
    const swapped: JoinCondition = {
      leftField: condition.rightField,
      rightField: condition.leftField,
    };
    this._inner = new LeftOuterJoinOperator(right, left, swapped, leftColumns);
  }

  execute(context: ExecutionContext): GeneralizedPostingList {
    return this._inner.execute(context);
  }
}

// -- FullOuterJoinOperator ---------------------------------------------------

export class FullOuterJoinOperator extends JoinOperator {
  private readonly _leftColumns: string[] | null;
  private readonly _rightColumns: string[] | null;

  constructor(
    left: unknown,
    right: unknown,
    condition: JoinCondition,
    leftColumns?: string[] | null,
    rightColumns?: string[] | null,
  ) {
    super(left, right, condition);
    this._leftColumns = leftColumns ?? null;
    this._rightColumns = rightColumns ?? null;
  }

  execute(context: ExecutionContext): GeneralizedPostingList {
    const leftEntries = this._getEntries(this.left, context);
    const rightEntries = this._getEntries(this.right, context);

    // Collect field names for NULL padding
    const rightFieldNames = new Set<string>();
    if (this._rightColumns) {
      for (const c of this._rightColumns) rightFieldNames.add(c);
    }
    if (rightEntries.length > 0) {
      for (const k of Object.keys(
        rightEntries[0]!.payload.fields as Record<string, unknown>,
      )) {
        rightFieldNames.add(k);
      }
    }

    const leftFieldNames = new Set<string>();
    if (this._leftColumns) {
      for (const c of this._leftColumns) leftFieldNames.add(c);
    }
    if (leftEntries.length > 0) {
      for (const k of Object.keys(
        leftEntries[0]!.payload.fields as Record<string, unknown>,
      )) {
        leftFieldNames.add(k);
      }
    }

    // Build index on right
    const rightIndex = new Map<unknown, JoinEntry[]>();
    const rightById = new Map<number, JoinEntry>();
    for (const entry of rightEntries) {
      rightById.set(entryDocId(entry), entry);
      const key = (entry.payload.fields as Record<string, unknown>)[
        this.condition.rightField
      ];
      if (key === undefined || key === null) continue;
      let bucket = rightIndex.get(key);
      if (!bucket) {
        bucket = [];
        rightIndex.set(key, bucket);
      }
      bucket.push(entry);
    }

    this.checkCancelled();
    const matchedRightIds = new Set<number>();
    const result: GeneralizedPostingEntry[] = [];

    // Left join pass
    for (const leftEntry of leftEntries) {
      const leftKey = (leftEntry.payload.fields as Record<string, unknown>)[
        this.condition.leftField
      ];
      const matches = leftKey != null ? rightIndex.get(leftKey) : undefined;

      if (matches && matches.length > 0) {
        for (const rightEntry of matches) {
          const rightId = entryDocId(rightEntry);
          matchedRightIds.add(rightId);
          result.push({
            docIds: [entryDocId(leftEntry), rightId],
            payload: createPayload({
              score: leftEntry.payload.score + rightEntry.payload.score,
              fields: {
                ...(leftEntry.payload.fields as Record<string, unknown>),
                ...(rightEntry.payload.fields as Record<string, unknown>),
              },
            }),
          });
        }
      } else {
        // Unmatched left: right-side columns as null
        const fields: Record<string, unknown> = {
          ...(leftEntry.payload.fields as Record<string, unknown>),
        };
        for (const rk of rightFieldNames) {
          if (!(rk in fields)) {
            fields[rk] = null;
          }
        }
        result.push({
          docIds: [entryDocId(leftEntry)],
          payload: createPayload({
            score: leftEntry.payload.score,
            fields,
          }),
        });
      }
    }

    // Unmatched right entries: left-side columns as null
    for (const rightEntry of rightEntries) {
      const rightId = entryDocId(rightEntry);
      if (!matchedRightIds.has(rightId)) {
        const fields: Record<string, unknown> = {
          ...(rightEntry.payload.fields as Record<string, unknown>),
        };
        for (const lk of leftFieldNames) {
          if (!(lk in fields)) {
            fields[lk] = null;
          }
        }
        result.push({
          docIds: [rightId],
          payload: createPayload({
            score: rightEntry.payload.score,
            fields,
          }),
        });
      }
    }

    return new GeneralizedPostingList(result);
  }
}
