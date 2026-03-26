//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Graph join operators
// 1:1 port of uqa/graph/join.py

import type { PostingEntry } from "../core/types.js";
import { createPostingEntry } from "../core/types.js";
import type { PostingList } from "../core/posting-list.js";
import { Operator } from "../operators/base.js";
import type { ExecutionContext } from "../operators/base.js";
import { GraphPostingList, createGraphPayload } from "./posting-list.js";
import type { GraphPayload } from "./posting-list.js";

// -- GraphGraphJoinOperator ---------------------------------------------------

export class GraphGraphJoinOperator extends Operator {
  readonly left: Operator;
  readonly right: Operator;
  readonly joinVariable: string;

  constructor(left: Operator, right: Operator, joinVariable: string) {
    super();
    this.left = left;
    this.right = right;
    this.joinVariable = joinVariable;
  }

  execute(context: ExecutionContext): PostingList {
    const leftResult = this.left.execute(context);
    const rightResult = this.right.execute(context);

    // Build hash index on right by join variable value
    const rightIndex = new Map<number, PostingEntry[]>();
    for (const entry of rightResult) {
      const fields = entry.payload.fields as Record<string, unknown>;
      const joinVal = fields[this.joinVariable];
      if (typeof joinVal === "number") {
        let bucket = rightIndex.get(joinVal);
        if (!bucket) {
          bucket = [];
          rightIndex.set(joinVal, bucket);
        }
        bucket.push(entry);
      }
    }

    const entries: PostingEntry[] = [];
    const graphPayloads = new Map<number, GraphPayload>();
    let docId = 0;

    const leftGpl = leftResult instanceof GraphPostingList ? leftResult : null;
    const rightGpl = rightResult instanceof GraphPostingList ? rightResult : null;

    for (const leftEntry of leftResult) {
      const leftFields = leftEntry.payload.fields as Record<string, unknown>;
      const joinVal = leftFields[this.joinVariable];
      if (typeof joinVal !== "number") continue;

      const matches = rightIndex.get(joinVal);
      if (!matches) continue;

      for (const rightEntry of matches) {
        const rightFields = rightEntry.payload.fields as Record<string, unknown>;
        const combinedFields: Record<string, unknown> = {
          ...leftFields,
          ...rightFields,
        };

        const score = leftEntry.payload.score + rightEntry.payload.score;
        entries.push(
          createPostingEntry(docId, {
            score,
            fields: combinedFields,
          }),
        );

        // Merge graph payloads
        const leftGp = leftGpl?.getGraphPayload(leftEntry.docId);
        const rightGp = rightGpl?.getGraphPayload(rightEntry.docId);
        if (leftGp || rightGp) {
          const mergedVertices = new Set<number>();
          const mergedEdges = new Set<number>();
          if (leftGp) {
            for (const v of leftGp.subgraphVertices) mergedVertices.add(v);
            for (const e of leftGp.subgraphEdges) mergedEdges.add(e);
          }
          if (rightGp) {
            for (const v of rightGp.subgraphVertices) mergedVertices.add(v);
            for (const e of rightGp.subgraphEdges) mergedEdges.add(e);
          }
          graphPayloads.set(
            docId,
            createGraphPayload({
              subgraphVertices: mergedVertices,
              subgraphEdges: mergedEdges,
              score,
              graphName: leftGp?.graphName ?? rightGp?.graphName ?? "",
            }),
          );
        }

        docId++;
      }
    }

    const gpl = new GraphPostingList(entries);
    for (const [id, gp] of graphPayloads) {
      gpl.setGraphPayload(id, gp);
    }
    return gpl;
  }
}

// -- CrossParadigmGraphJoinOperator -------------------------------------------

export class CrossParadigmGraphJoinOperator extends Operator {
  readonly graphSource: Operator;
  readonly relationalSource: Operator;
  readonly joinField: string;

  constructor(graphSource: Operator, relationalSource: Operator, joinField: string) {
    super();
    this.graphSource = graphSource;
    this.relationalSource = relationalSource;
    this.joinField = joinField;
  }

  execute(context: ExecutionContext): PostingList {
    const graphResult = this.graphSource.execute(context);
    const relResult = this.relationalSource.execute(context);

    // Build hash index on relational result by join field
    const relIndex = new Map<unknown, PostingEntry[]>();
    for (const entry of relResult) {
      const fields = entry.payload.fields as Record<string, unknown>;
      const joinVal = fields[this.joinField];
      if (joinVal === undefined || joinVal === null) continue;
      let bucket = relIndex.get(joinVal);
      if (!bucket) {
        bucket = [];
        relIndex.set(joinVal, bucket);
      }
      bucket.push(entry);
    }

    const entries: PostingEntry[] = [];
    const graphPayloads = new Map<number, GraphPayload>();
    let docId = 0;
    const graphGpl = graphResult instanceof GraphPostingList ? graphResult : null;

    for (const graphEntry of graphResult) {
      const graphFields = graphEntry.payload.fields as Record<string, unknown>;

      // Try matching on docId (vertex id) or on the join field value
      const joinVal = graphFields[this.joinField] ?? graphEntry.docId;

      const matches = relIndex.get(joinVal);
      if (!matches) continue;

      for (const relEntry of matches) {
        const relFields = relEntry.payload.fields as Record<string, unknown>;
        const combinedFields: Record<string, unknown> = {
          ...graphFields,
          ...relFields,
        };

        const score = graphEntry.payload.score + relEntry.payload.score;
        entries.push(
          createPostingEntry(docId, {
            score,
            fields: combinedFields,
          }),
        );

        const graphGp = graphGpl?.getGraphPayload(graphEntry.docId);
        if (graphGp) {
          graphPayloads.set(
            docId,
            createGraphPayload({
              subgraphVertices: graphGp.subgraphVertices,
              subgraphEdges: graphGp.subgraphEdges,
              score,
              graphName: graphGp.graphName,
            }),
          );
        }

        docId++;
      }
    }

    const gpl = new GraphPostingList(entries);
    for (const [id, gp] of graphPayloads) {
      gpl.setGraphPayload(id, gp);
    }
    return gpl;
  }
}
