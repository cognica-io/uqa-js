//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Temporal pattern match operator
// 1:1 port of uqa/graph/temporal_pattern_match.py

import type { PostingEntry } from "../core/types.js";
import { createPostingEntry } from "../core/types.js";
import type { PostingList } from "../core/posting-list.js";
import { Operator } from "../operators/base.js";
import type { ExecutionContext } from "../operators/base.js";
import type { GraphStore } from "../storage/abc/graph-store.js";
import { GraphPostingList, createGraphPayload } from "./posting-list.js";
import type { GraphPattern } from "./pattern.js";
import type { TemporalFilter } from "./temporal-filter.js";

// -- TemporalPatternMatchOperator ---------------------------------------------

export class TemporalPatternMatchOperator extends Operator {
  readonly pattern: GraphPattern;
  readonly graph: string;
  readonly temporalFilter: TemporalFilter;
  readonly score: number;

  constructor(opts: {
    pattern: GraphPattern;
    graph: string;
    temporalFilter: TemporalFilter;
    score?: number;
  }) {
    super();
    this.pattern = opts.pattern;
    this.graph = opts.graph;
    this.temporalFilter = opts.temporalFilter;
    this.score = opts.score ?? 1.0;
  }

  execute(context: ExecutionContext): PostingList {
    const store = context.graphStore as GraphStore;
    const results = this._backtrackSearch(store);
    return this._buildPostingList(store, results);
  }

  private _backtrackSearch(store: GraphStore): Array<Map<string, number>> {
    const candidates = this._computeCandidates(store);
    const results: Array<Map<string, number>> = [];
    const assignment = new Map<string, number>();
    const variables = this.pattern.vertexPatterns.map((vp) => vp.variable);

    this._backtrack(store, variables, 0, assignment, candidates, results);
    return results;
  }

  private _computeCandidates(store: GraphStore): Map<string, Set<number>> {
    const candidates = new Map<string, Set<number>>();
    const allVertexIds = store.vertexIdsInGraph(this.graph);

    for (const vp of this.pattern.vertexPatterns) {
      let candidateSet = new Set(allVertexIds);

      if (vp.constraints.length > 0) {
        const filtered = new Set<number>();
        for (const vid of candidateSet) {
          const vertex = store.getVertex(vid);
          if (vertex) {
            let valid = true;
            for (const constraint of vp.constraints) {
              if (!constraint(vertex)) {
                valid = false;
                break;
              }
            }
            if (valid) filtered.add(vid);
          }
        }
        candidateSet = filtered;
      }

      candidates.set(vp.variable, candidateSet);
    }

    return candidates;
  }

  private _backtrack(
    store: GraphStore,
    variables: string[],
    index: number,
    assignment: Map<string, number>,
    candidates: Map<string, Set<number>>,
    results: Array<Map<string, number>>,
  ): void {
    if (index === variables.length) {
      if (this._validateAllEdges(store, assignment)) {
        results.push(new Map(assignment));
      }
      return;
    }

    const variable = variables[index]!;
    const candidateSet = candidates.get(variable) ?? new Set<number>();
    const usedVertices = new Set(assignment.values());

    for (const vid of candidateSet) {
      if (usedVertices.has(vid)) continue;
      assignment.set(variable, vid);

      if (this._validateEdgesFor(store, assignment, variable)) {
        this._backtrack(store, variables, index + 1, assignment, candidates, results);
      }

      assignment.delete(variable);
    }
  }

  private _validateEdgesFor(
    store: GraphStore,
    assignment: Map<string, number>,
    variable: string,
  ): boolean {
    for (const ep of this.pattern.edgePatterns) {
      if (ep.negated) continue;
      if (ep.sourceVar !== variable && ep.targetVar !== variable) continue;

      const srcId = assignment.get(ep.sourceVar);
      const tgtId = assignment.get(ep.targetVar);
      if (srcId === undefined || tgtId === undefined) continue;

      // Check for edge with temporal validity
      const outEdges = store.outEdgeIds(srcId, this.graph);
      let found = false;
      for (const eid of outEdges) {
        const edge = store.getEdge(eid);
        if (!edge) continue;
        if (edge.targetId !== tgtId) continue;
        if (ep.label !== null && edge.label !== ep.label) continue;
        if (!this.temporalFilter.isValid(edge)) continue;

        // Check edge constraints
        let valid = true;
        for (const constraint of ep.constraints) {
          if (!constraint(edge)) {
            valid = false;
            break;
          }
        }
        if (valid) {
          found = true;
          break;
        }
      }
      if (!found) return false;
    }
    return true;
  }

  private _validateAllEdges(
    store: GraphStore,
    assignment: Map<string, number>,
  ): boolean {
    for (const ep of this.pattern.edgePatterns) {
      const srcId = assignment.get(ep.sourceVar);
      const tgtId = assignment.get(ep.targetVar);
      if (srcId === undefined || tgtId === undefined) return false;

      if (ep.negated) {
        // Check no temporally valid edge exists
        const outEdges = store.outEdgeIds(srcId, this.graph);
        for (const eid of outEdges) {
          const edge = store.getEdge(eid);
          if (
            edge &&
            edge.targetId === tgtId &&
            (ep.label === null || edge.label === ep.label) &&
            this.temporalFilter.isValid(edge)
          ) {
            return false;
          }
        }
      } else {
        const outEdges = store.outEdgeIds(srcId, this.graph);
        let found = false;
        for (const eid of outEdges) {
          const edge = store.getEdge(eid);
          if (!edge) continue;
          if (edge.targetId !== tgtId) continue;
          if (ep.label !== null && edge.label !== ep.label) continue;
          if (!this.temporalFilter.isValid(edge)) continue;
          found = true;
          break;
        }
        if (!found) return false;
      }
    }
    return true;
  }

  private _collectMatchEdges(
    store: GraphStore,
    assignment: Map<string, number>,
  ): Set<number> {
    const edgeIds = new Set<number>();
    for (const ep of this.pattern.edgePatterns) {
      if (ep.negated) continue;
      const srcId = assignment.get(ep.sourceVar);
      const tgtId = assignment.get(ep.targetVar);
      if (srcId === undefined || tgtId === undefined) continue;

      const outEdges = store.outEdgeIds(srcId, this.graph);
      for (const eid of outEdges) {
        const edge = store.getEdge(eid);
        if (
          edge &&
          edge.targetId === tgtId &&
          (ep.label === null || edge.label === ep.label) &&
          this.temporalFilter.isValid(edge)
        ) {
          edgeIds.add(eid);
        }
      }
    }
    return edgeIds;
  }

  private _buildPostingList(
    store: GraphStore,
    results: Array<Map<string, number>>,
  ): GraphPostingList {
    const entries: PostingEntry[] = [];
    const graphPayloads = new Map<number, ReturnType<typeof createGraphPayload>>();

    for (let i = 0; i < results.length; i++) {
      const assignment = results[i]!;
      const vertexIds = new Set(assignment.values());
      const edgeIds = this._collectMatchEdges(store, assignment);

      const docId = i;
      const fields: Record<string, unknown> = {};
      for (const [variable, vid] of assignment) {
        fields[variable] = vid;
      }

      entries.push(
        createPostingEntry(docId, {
          score: this.score,
          fields,
        }),
      );

      graphPayloads.set(
        docId,
        createGraphPayload({
          subgraphVertices: vertexIds,
          subgraphEdges: edgeIds,
          score: this.score,
          graphName: this.graph,
        }),
      );
    }

    const gpl = new GraphPostingList(entries);
    for (const [docId, gp] of graphPayloads) {
      gpl.setGraphPayload(docId, gp);
    }
    return gpl;
  }
}
