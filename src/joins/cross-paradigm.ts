//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- cross-paradigm joins
// 1:1 port of uqa/joins/cross_paradigm.py

import type { DocId, GeneralizedPostingEntry, PostingEntry } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { GeneralizedPostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "../operators/base.js";
import type { GraphStore } from "../storage/abc/graph-store.js";
import { DEFAULT_ANALYZER } from "../analysis/analyzer.js";
import { cosine } from "../math/linalg.js";

function getEntries(source: unknown, context: ExecutionContext): PostingEntry[] {
  if (
    source !== null &&
    typeof source === "object" &&
    "execute" in source &&
    typeof (source as { execute: unknown }).execute === "function"
  ) {
    const result = (
      source as { execute(ctx: ExecutionContext): { entries: PostingEntry[] } }
    ).execute(context);
    return [...result.entries];
  }
  return source as PostingEntry[];
}

// -- TextSimilarityJoinOperator ----------------------------------------------

export class TextSimilarityJoinOperator {
  readonly left: unknown;
  readonly right: unknown;
  readonly leftField: string;
  readonly rightField: string;
  readonly threshold: number;

  constructor(
    left: unknown,
    right: unknown,
    leftField: string,
    rightField: string,
    threshold = 0.5,
  ) {
    this.left = left;
    this.right = right;
    this.leftField = leftField;
    this.rightField = rightField;
    this.threshold = threshold;
  }

  execute(context: ExecutionContext): GeneralizedPostingList {
    const leftEntries = getEntries(this.left, context);
    const rightEntries = getEntries(this.right, context);
    const result: GeneralizedPostingEntry[] = [];

    for (const le of leftEntries) {
      const leftRaw = (le.payload.fields as Record<string, unknown>)[this.leftField];
      const leftText = typeof leftRaw === "string" ? leftRaw : "";
      const leftTokens = new Set(DEFAULT_ANALYZER.analyze(leftText));
      if (leftTokens.size === 0) continue;

      for (const re of rightEntries) {
        const rightRaw = (re.payload.fields as Record<string, unknown>)[
          this.rightField
        ];
        const rightText = typeof rightRaw === "string" ? rightRaw : "";
        const rightTokens = new Set(DEFAULT_ANALYZER.analyze(rightText));
        if (rightTokens.size === 0) continue;

        let intersection = 0;
        for (const t of leftTokens) {
          if (rightTokens.has(t)) intersection++;
        }
        const union = leftTokens.size + rightTokens.size - intersection;
        const jaccard = union > 0 ? intersection / union : 0;

        if (jaccard >= this.threshold) {
          result.push({
            docIds: [le.docId, re.docId],
            payload: createPayload({
              score: jaccard,
              fields: {
                ...(le.payload.fields as Record<string, unknown>),
                ...(re.payload.fields as Record<string, unknown>),
              },
            }),
          });
        }
      }
    }

    return new GeneralizedPostingList(result);
  }
}

// -- VectorSimilarityJoinOperator --------------------------------------------

export class VectorSimilarityJoinOperator {
  readonly left: unknown;
  readonly right: unknown;
  readonly leftField: string;
  readonly rightField: string;
  readonly threshold: number;

  constructor(
    left: unknown,
    right: unknown,
    leftField: string,
    rightField: string,
    threshold = 0.5,
  ) {
    this.left = left;
    this.right = right;
    this.leftField = leftField;
    this.rightField = rightField;
    this.threshold = threshold;
  }

  execute(context: ExecutionContext): GeneralizedPostingList {
    const leftEntries = getEntries(this.left, context);
    const rightEntries = getEntries(this.right, context);
    const result: GeneralizedPostingEntry[] = [];

    for (const le of leftEntries) {
      const leftVec = (le.payload.fields as Record<string, unknown>)[
        this.leftField
      ] as Float64Array | null;
      if (!leftVec) continue;

      for (const re of rightEntries) {
        const rightVec = (re.payload.fields as Record<string, unknown>)[
          this.rightField
        ] as Float64Array | null;
        if (!rightVec) continue;

        const sim = cosine(leftVec, rightVec);
        if (sim >= this.threshold) {
          result.push({
            docIds: [le.docId, re.docId],
            payload: createPayload({
              score: sim,
              fields: {
                ...(le.payload.fields as Record<string, unknown>),
                ...(re.payload.fields as Record<string, unknown>),
              },
            }),
          });
        }
      }
    }

    return new GeneralizedPostingList(result);
  }
}

// -- HybridJoinOperator ------------------------------------------------------

export class HybridJoinOperator {
  readonly left: unknown;
  readonly right: unknown;
  readonly structuredField: string;
  readonly vectorField: string;
  readonly threshold: number;

  constructor(
    left: unknown,
    right: unknown,
    structuredField: string,
    vectorField: string,
    threshold = 0.5,
  ) {
    this.left = left;
    this.right = right;
    this.structuredField = structuredField;
    this.vectorField = vectorField;
    this.threshold = threshold;
  }

  execute(context: ExecutionContext): GeneralizedPostingList {
    const leftEntries = getEntries(this.left, context);
    const rightEntries = getEntries(this.right, context);

    // Build index on right by structured field
    const rightIndex = new Map<unknown, PostingEntry[]>();
    for (const re of rightEntries) {
      const key = (re.payload.fields as Record<string, unknown>)[this.structuredField];
      if (key == null) continue;
      let bucket = rightIndex.get(key);
      if (!bucket) {
        bucket = [];
        rightIndex.set(key, bucket);
      }
      bucket.push(re);
    }

    const result: GeneralizedPostingEntry[] = [];
    for (const le of leftEntries) {
      const leftKey = (le.payload.fields as Record<string, unknown>)[
        this.structuredField
      ];
      if (leftKey == null) continue;
      const leftVec = (le.payload.fields as Record<string, unknown>)[
        this.vectorField
      ] as Float64Array | null;
      if (!leftVec) continue;

      const matches = rightIndex.get(leftKey);
      if (!matches) continue;

      for (const re of matches) {
        const rightVec = (re.payload.fields as Record<string, unknown>)[
          this.vectorField
        ] as Float64Array | null;
        if (!rightVec) continue;

        const sim = cosine(leftVec, rightVec);
        if (sim >= this.threshold) {
          result.push({
            docIds: [le.docId, re.docId],
            payload: createPayload({
              score: sim,
              fields: {
                ...(le.payload.fields as Record<string, unknown>),
                ...(re.payload.fields as Record<string, unknown>),
              },
            }),
          });
        }
      }
    }

    return new GeneralizedPostingList(result);
  }
}

// -- GraphJoinOperator -------------------------------------------------------

export class GraphJoinOperator {
  readonly left: unknown;
  readonly right: unknown;
  readonly label: string | null;
  readonly graphName: string;

  constructor(
    left: unknown,
    right: unknown,
    label?: string | null,
    graphName = "test",
  ) {
    this.left = left;
    this.right = right;
    this.label = label ?? null;
    this.graphName = graphName;
  }

  execute(context: ExecutionContext): GeneralizedPostingList {
    const graph = context.graphStore as GraphStore;
    const leftEntries = getEntries(this.left, context);
    const rightEntries = getEntries(this.right, context);

    const rightSet = new Map<DocId, PostingEntry>();
    for (const re of rightEntries) {
      rightSet.set(re.docId, re);
    }

    const result: GeneralizedPostingEntry[] = [];
    for (const le of leftEntries) {
      const neighbors = graph.neighbors(le.docId, this.graphName, this.label, "out");
      for (const neighborId of neighbors) {
        const re = rightSet.get(neighborId);
        if (re) {
          result.push({
            docIds: [le.docId, re.docId],
            payload: createPayload({
              score: le.payload.score + re.payload.score,
              fields: {
                ...(le.payload.fields as Record<string, unknown>),
                ...(re.payload.fields as Record<string, unknown>),
              },
            }),
          });
        }
      }
    }

    return new GeneralizedPostingList(result);
  }
}

// -- CrossParadigmJoinOperator -----------------------------------------------

export class CrossParadigmJoinOperator {
  readonly left: unknown;
  readonly right: unknown;
  readonly vertexField: string;
  readonly docField: string;

  constructor(left: unknown, right: unknown, vertexField: string, docField: string) {
    this.left = left;
    this.right = right;
    this.vertexField = vertexField;
    this.docField = docField;
  }

  execute(context: ExecutionContext): GeneralizedPostingList {
    const graph = context.graphStore as GraphStore;
    const leftEntries = getEntries(this.left, context);
    const rightEntries = getEntries(this.right, context);

    // Build right index by docField
    const rightIndex = new Map<unknown, PostingEntry[]>();
    for (const re of rightEntries) {
      const key = (re.payload.fields as Record<string, unknown>)[this.docField];
      if (key == null) continue;
      let bucket = rightIndex.get(key);
      if (!bucket) {
        bucket = [];
        rightIndex.set(key, bucket);
      }
      bucket.push(re);
    }

    const result: GeneralizedPostingEntry[] = [];
    for (const le of leftEntries) {
      const vertex = graph.getVertex(le.docId);
      let vertexKey: unknown;
      if (vertex) {
        vertexKey = vertex.properties[this.vertexField];
      } else {
        vertexKey = (le.payload.fields as Record<string, unknown>)[this.vertexField];
      }
      if (vertexKey == null) continue;

      const matches = rightIndex.get(vertexKey);
      if (!matches) continue;

      for (const re of matches) {
        const fields: Record<string, unknown> = {};
        if (vertex) Object.assign(fields, vertex.properties);
        Object.assign(fields, le.payload.fields);
        Object.assign(fields, re.payload.fields);

        result.push({
          docIds: [le.docId, re.docId],
          payload: createPayload({
            score: le.payload.score + re.payload.score,
            fields,
          }),
        });
      }
    }

    return new GeneralizedPostingList(result);
  }
}
