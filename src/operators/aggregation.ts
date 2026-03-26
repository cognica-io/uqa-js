//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- aggregation operators
// 1:1 port of uqa/operators/aggregation.py

import type { DocId, IndexStats } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "./base.js";
import { Operator } from "./base.js";

// -- Aggregation Monoids -----------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-unnecessary-type-parameters
export abstract class AggregationMonoid<S, V, R> {
  abstract identity(): S;
  abstract accumulate(state: S, value: V): S;
  abstract combine(stateA: S, stateB: S): S;
  abstract finalize(state: S): R;
}

export class CountMonoid extends AggregationMonoid<number, unknown, number> {
  identity(): number {
    return 0;
  }
  accumulate(state: number, _value: unknown): number {
    return state + 1;
  }
  combine(a: number, b: number): number {
    return a + b;
  }
  finalize(state: number): number {
    return state;
  }
}

export class SumMonoid extends AggregationMonoid<number, number, number> {
  identity(): number {
    return 0;
  }
  accumulate(state: number, value: number): number {
    return state + value;
  }
  combine(a: number, b: number): number {
    return a + b;
  }
  finalize(state: number): number {
    return state;
  }
}

export class AvgMonoid extends AggregationMonoid<[number, number], number, number> {
  identity(): [number, number] {
    return [0, 0];
  }
  accumulate(state: [number, number], value: number): [number, number] {
    return [state[0] + value, state[1] + 1];
  }
  combine(a: [number, number], b: [number, number]): [number, number] {
    return [a[0] + b[0], a[1] + b[1]];
  }
  finalize(state: [number, number]): number {
    return state[1] === 0 ? 0 : state[0] / state[1];
  }
}

export class MinMonoid extends AggregationMonoid<number, number, number> {
  identity(): number {
    return Infinity;
  }
  accumulate(state: number, value: number): number {
    return Math.min(state, value);
  }
  combine(a: number, b: number): number {
    return Math.min(a, b);
  }
  finalize(state: number): number {
    return state;
  }
}

export class MaxMonoid extends AggregationMonoid<number, number, number> {
  identity(): number {
    return -Infinity;
  }
  accumulate(state: number, value: number): number {
    return Math.max(state, value);
  }
  combine(a: number, b: number): number {
    return Math.max(a, b);
  }
  finalize(state: number): number {
    return state;
  }
}

export class QuantileMonoid extends AggregationMonoid<number[], number, number> {
  readonly quantile: number;

  constructor(quantile = 0.5) {
    super();
    if (quantile < 0 || quantile > 1) {
      throw new Error("quantile must be in [0, 1]");
    }
    this.quantile = quantile;
  }

  identity(): number[] {
    return [];
  }
  accumulate(state: number[], value: number): number[] {
    state.push(value);
    return state;
  }
  combine(a: number[], b: number[]): number[] {
    return [...a, ...b];
  }
  finalize(state: number[]): number {
    if (state.length === 0) return 0;
    const sorted = state.slice().sort((a, b) => a - b);
    const n = sorted.length;
    const idx = this.quantile * (n - 1);
    const lower = Math.floor(idx);
    const upper = Math.min(lower + 1, n - 1);
    const frac = idx - lower;
    return sorted[lower]! * (1 - frac) + sorted[upper]! * frac;
  }
}

// -- AggregateOperator -------------------------------------------------------

export class AggregateOperator extends Operator {
  readonly source: Operator | null;
  readonly field: string;
  readonly monoid: AggregationMonoid<unknown, unknown, unknown>;

  constructor(
    source: Operator | null,
    field: string,
    monoid: AggregationMonoid<unknown, unknown, unknown>,
  ) {
    super();
    this.source = source;
    this.field = field;
    this.monoid = monoid;
  }

  execute(context: ExecutionContext): PostingList {
    const docStore = context.documentStore;

    let docIds: DocId[];
    if (this.source) {
      docIds = this.source.execute(context).entries.map((e) => e.docId);
    } else {
      docIds = docStore ? [...docStore.docIds].sort((a, b) => a - b) : [];
    }

    let state = this.monoid.identity();
    for (const docId of docIds) {
      const value = docStore ? docStore.getField(docId, this.field) : undefined;
      if (value !== null && value !== undefined) {
        state = this.monoid.accumulate(state, value);
      }
    }

    const resultValue = this.monoid.finalize(state);
    const score = typeof resultValue === "number" ? resultValue : 0;

    return PostingList.fromSorted([
      {
        docId: 0,
        payload: createPayload({
          score,
          fields: { _aggregate_field: this.field, _aggregate: resultValue },
        }),
      },
    ]);
  }

  costEstimate(stats: IndexStats): number {
    return stats.totalDocs;
  }
}

// -- GroupByOperator ---------------------------------------------------------

export class GroupByOperator extends Operator {
  readonly source: Operator;
  readonly groupField: string;
  readonly aggField: string;
  readonly monoid: AggregationMonoid<unknown, unknown, unknown>;

  constructor(
    source: Operator,
    groupField: string,
    aggField: string,
    monoid: AggregationMonoid<unknown, unknown, unknown>,
  ) {
    super();
    this.source = source;
    this.groupField = groupField;
    this.aggField = aggField;
    this.monoid = monoid;
  }

  execute(context: ExecutionContext): PostingList {
    const sourcePl = this.source.execute(context);
    const docStore = context.documentStore;
    if (!docStore) return new PostingList();

    const groups = new Map<string, unknown>();

    for (const entry of sourcePl) {
      const groupValue = docStore.getField(entry.docId, this.groupField);
      if (groupValue === null || groupValue === undefined) continue;
      const groupKey = String(groupValue as string | number);

      if (!groups.has(groupKey)) {
        groups.set(groupKey, this.monoid.identity());
      }

      const aggValue = docStore.getField(entry.docId, this.aggField);
      if (aggValue !== null && aggValue !== undefined) {
        groups.set(groupKey, this.monoid.accumulate(groups.get(groupKey), aggValue));
      }
    }

    const sorted = [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]));
    const entries = sorted.map(([groupKey, state], idx) => {
      const resultValue = this.monoid.finalize(state);
      const score = typeof resultValue === "number" ? resultValue : 0;
      return {
        docId: idx,
        payload: createPayload({
          score,
          fields: {
            _group_key: groupKey,
            _group_field: this.groupField,
            _aggregate_result: resultValue,
          },
        }),
      };
    });

    return PostingList.fromSorted(entries);
  }
}
