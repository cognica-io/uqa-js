//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- hierarchical operators
// 1:1 port of uqa/operators/hierarchical.py

import type { DocId, IndexStats, PathExpr, Predicate } from "../core/types.js";
import { createPayload, isNullPredicate } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "./base.js";
import { Operator } from "./base.js";
import { FilterOperator } from "./primitive.js";
import type { AggregationMonoid } from "./aggregation.js";

// -- PathFilterOperator ------------------------------------------------------

export class PathFilterOperator extends Operator {
  readonly path: PathExpr;
  readonly predicate: Predicate;
  readonly source: Operator | null;

  constructor(path: PathExpr, predicate: Predicate, source?: Operator | null) {
    super();
    this.path = path;
    this.predicate = predicate;
    this.source = source ?? null;
  }

  execute(context: ExecutionContext): PostingList {
    const docStore = context.documentStore;
    if (!docStore) return new PostingList();

    const isNullAware = isNullPredicate(this.predicate);
    let docIds: DocId[];
    if (this.source) {
      docIds = this.source.execute(context).entries.map((e) => e.docId);
    } else {
      docIds = [...docStore.docIds].sort((a, b) => a - b);
    }

    const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
    for (const docId of docIds) {
      const value = docStore.evalPath(docId, this.path);
      if (Array.isArray(value)) {
        if ((value as unknown[]).some((v) => this.predicate.evaluate(v))) {
          entries.push({ docId, payload: createPayload({ score: 0.0 }) });
        }
      } else {
        if (!isNullAware && (value === null || value === undefined)) continue;
        if (this.predicate.evaluate(value)) {
          entries.push({ docId, payload: createPayload({ score: 0.0 }) });
        }
      }
    }

    return PostingList.fromSorted(entries);
  }

  costEstimate(stats: IndexStats): number {
    return this.source ? this.source.costEstimate(stats) : stats.totalDocs;
  }
}

// -- PathProjectOperator -----------------------------------------------------

export class PathProjectOperator extends Operator {
  readonly paths: PathExpr[];
  readonly source: Operator;

  constructor(paths: PathExpr[], source: Operator) {
    super();
    this.paths = paths;
    this.source = source;
  }

  execute(context: ExecutionContext): PostingList {
    const sourcePl = this.source.execute(context);
    const docStore = context.documentStore;
    if (!docStore) return sourcePl;

    const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
    for (const entry of sourcePl) {
      const projected: Record<string, unknown> = {};
      for (const path of this.paths) {
        const value = docStore.evalPath(entry.docId, path);
        const key = path.join(".");
        projected[key] = value;
      }
      entries.push({
        docId: entry.docId,
        payload: createPayload({
          positions: entry.payload.positions as number[],
          score: entry.payload.score,
          fields: {
            ...(entry.payload.fields as Record<string, unknown>),
            ...projected,
          },
        }),
      });
    }

    return PostingList.fromSorted(entries);
  }

  costEstimate(stats: IndexStats): number {
    return this.source.costEstimate(stats);
  }
}

// -- PathUnnestOperator ------------------------------------------------------

export class PathUnnestOperator extends Operator {
  readonly path: PathExpr;
  readonly source: Operator;

  constructor(path: PathExpr, source: Operator) {
    super();
    this.path = path;
    this.source = source;
  }

  execute(context: ExecutionContext): PostingList {
    const sourcePl = this.source.execute(context);
    const docStore = context.documentStore;
    if (!docStore) return sourcePl;

    const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
    for (const entry of sourcePl) {
      const arr = docStore.evalPath(entry.docId, this.path);
      if (Array.isArray(arr)) {
        for (const item of arr as unknown[]) {
          entries.push({
            docId: entry.docId,
            payload: createPayload({
              positions: entry.payload.positions as number[],
              score: entry.payload.score,
              fields: {
                ...(entry.payload.fields as Record<string, unknown>),
                _unnested_data: item,
              },
            }),
          });
        }
      } else {
        entries.push(entry);
      }
    }

    return new PostingList(entries);
  }

  costEstimate(stats: IndexStats): number {
    return this.source.costEstimate(stats) * 2.0;
  }
}

// -- PathAggregateOperator ---------------------------------------------------

export class PathAggregateOperator extends Operator {
  readonly path: PathExpr;
  readonly monoid: AggregationMonoid<unknown, unknown, unknown>;
  readonly source: Operator | null;

  constructor(
    path: PathExpr,
    monoid: AggregationMonoid<unknown, unknown, unknown>,
    source?: Operator | null,
  ) {
    super();
    this.path = path;
    this.monoid = monoid;
    this.source = source ?? null;
  }

  execute(context: ExecutionContext): PostingList {
    const docStore = context.documentStore;
    if (!docStore) return new PostingList();

    let docIds: DocId[];
    if (this.source) {
      docIds = this.source.execute(context).entries.map((e) => e.docId);
    } else {
      docIds = [...docStore.docIds].sort((a, b) => a - b);
    }

    const entries: { docId: DocId; payload: ReturnType<typeof createPayload> }[] = [];
    for (const docId of docIds) {
      const value = docStore.evalPath(docId, this.path);
      let state = this.monoid.identity();

      if (Array.isArray(value)) {
        for (const v of value as unknown[]) {
          if (v !== null && v !== undefined) {
            state = this.monoid.accumulate(state, v);
          }
        }
      } else if (value !== null && value !== undefined) {
        state = this.monoid.accumulate(state, value);
      }

      const result = this.monoid.finalize(state);
      const score = typeof result === "number" ? result : 0;
      entries.push({
        docId,
        payload: createPayload({
          score,
          fields: {
            _path_agg_result: result,
            _path_agg_path: this.path.join("."),
          },
        }),
      });
    }

    return PostingList.fromSorted(entries);
  }

  costEstimate(stats: IndexStats): number {
    return this.source ? this.source.costEstimate(stats) : stats.totalDocs;
  }
}

// -- UnifiedFilterOperator ---------------------------------------------------

export class UnifiedFilterOperator extends Operator {
  readonly fieldExpr: string;
  readonly predicate: Predicate;
  readonly source: Operator | null;

  constructor(fieldExpr: string, predicate: Predicate, source?: Operator | null) {
    super();
    this.fieldExpr = fieldExpr;
    this.predicate = predicate;
    this.source = source ?? null;
  }

  execute(context: ExecutionContext): PostingList {
    if (this.fieldExpr.includes(".")) {
      const path = this.fieldExpr.split(".");
      return new PathFilterOperator(path, this.predicate, this.source).execute(context);
    }
    return new FilterOperator(this.fieldExpr, this.predicate, this.source).execute(
      context,
    );
  }

  costEstimate(stats: IndexStats): number {
    return this.source ? this.source.costEstimate(stats) : stats.totalDocs;
  }
}
