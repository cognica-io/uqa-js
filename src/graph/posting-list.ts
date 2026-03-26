//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- GraphPostingList
// 1:1 port of uqa/graph/posting_list.py

import type { PostingEntry } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";

// -- GraphPayload -------------------------------------------------------------

export interface GraphPayload {
  readonly subgraphVertices: ReadonlySet<number>;
  readonly subgraphEdges: ReadonlySet<number>;
  readonly score: number;
  readonly graphName: string;
}

export function createGraphPayload(
  opts?: Partial<{
    subgraphVertices: ReadonlySet<number>;
    subgraphEdges: ReadonlySet<number>;
    score: number;
    graphName: string;
  }>,
): GraphPayload {
  return {
    subgraphVertices: opts?.subgraphVertices ?? new Set<number>(),
    subgraphEdges: opts?.subgraphEdges ?? new Set<number>(),
    score: opts?.score ?? 0.0,
    graphName: opts?.graphName ?? "",
  };
}

// -- GraphPostingList ---------------------------------------------------------

export class GraphPostingList extends PostingList {
  private _graphPayloads: Map<number, GraphPayload>;

  constructor(entries?: PostingEntry[], graphPayloads?: Map<number, GraphPayload>) {
    super(entries);
    this._graphPayloads = new Map(graphPayloads ?? []);
  }

  setGraphPayload(docId: number, payload: GraphPayload): void {
    this._graphPayloads.set(docId, payload);
  }

  getGraphPayload(docId: number): GraphPayload | null {
    return this._graphPayloads.get(docId) ?? null;
  }

  get graphPayloads(): Map<number, GraphPayload> {
    return this._graphPayloads;
  }

  // Isomorphism Phi: GraphPostingList -> PostingList
  // Embeds graph-specific payload into the standard PostingList fields.
  toPostingList(): PostingList {
    const entries: PostingEntry[] = [];
    for (const entry of this) {
      const gp = this._graphPayloads.get(entry.docId);
      const fields: Record<string, unknown> = {
        ...(entry.payload.fields as Record<string, unknown>),
      };
      if (gp) {
        fields["_subgraph_vertices"] = [...gp.subgraphVertices];
        fields["_subgraph_edges"] = [...gp.subgraphEdges];
        fields["_graph_score"] = gp.score;
        fields["_graph_name"] = gp.graphName;
      }
      entries.push({
        docId: entry.docId,
        payload: createPayload({
          positions: entry.payload.positions,
          score: gp ? gp.score : entry.payload.score,
          fields,
        }),
      });
    }
    return new PostingList(entries);
  }

  // Inverse: PostingList -> GraphPostingList
  static fromPostingList(pl: PostingList): GraphPostingList {
    const entries: PostingEntry[] = [];
    const graphPayloads = new Map<number, GraphPayload>();

    for (const entry of pl) {
      entries.push(entry);
      const fields = entry.payload.fields as Record<string, unknown>;
      const rawVertices = fields["_subgraph_vertices"];
      const rawEdges = fields["_subgraph_edges"];
      const rawScore = fields["_graph_score"];
      const rawName = fields["_graph_name"];

      if (rawVertices !== undefined || rawEdges !== undefined) {
        graphPayloads.set(
          entry.docId,
          createGraphPayload({
            subgraphVertices: new Set(
              Array.isArray(rawVertices) ? (rawVertices as number[]) : [],
            ),
            subgraphEdges: new Set(
              Array.isArray(rawEdges) ? (rawEdges as number[]) : [],
            ),
            score: typeof rawScore === "number" ? rawScore : entry.payload.score,
            graphName: typeof rawName === "string" ? rawName : "",
          }),
        );
      }
    }

    return new GraphPostingList(entries, graphPayloads);
  }
}
