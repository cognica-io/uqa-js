//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Graph operators
// 1:1 port of uqa/graph/operators.py

import type { PostingEntry } from "../core/types.js";
import { createPayload, createPostingEntry } from "../core/types.js";
import type { Vertex } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import { Operator } from "../operators/base.js";
import type { ExecutionContext } from "../operators/base.js";
import type { GraphStore } from "../storage/abc/graph-store.js";
import { GraphPostingList, createGraphPayload } from "./posting-list.js";
import type { GraphPattern, RegularPathExpr } from "./pattern.js";
import { Alternation, BoundedLabel, Concat, KleeneStar, Label } from "./pattern.js";
import type { NFATransition } from "./rpq-optimizer.js";
import { subsetConstruction } from "./rpq-optimizer.js";
import { CypherCompiler } from "./cypher/compiler.js";

// -- TraverseOperator ---------------------------------------------------------

export class TraverseOperator extends Operator {
  readonly startVertex: number;
  readonly graph: string;
  readonly label: string | null;
  readonly maxHops: number;
  readonly vertexPredicate: ((v: Vertex) => boolean) | null;
  readonly score: number;

  constructor(
    startVertex: number,
    graph: string,
    label?: string | null,
    maxHops?: number,
    vertexPredicate?: ((v: Vertex) => boolean) | null,
    score?: number,
  ) {
    super();
    this.startVertex = startVertex;
    this.graph = graph;
    this.label = label ?? null;
    this.maxHops = maxHops ?? Infinity;
    this.vertexPredicate = vertexPredicate ?? null;
    this.score = score ?? 1.0;
  }

  execute(context: ExecutionContext): PostingList {
    const store = context.graphStore as GraphStore;
    const graphName = this.graph;

    const visited = new Set<number>();
    const entries: PostingEntry[] = [];

    // BFS
    const queue: Array<[number, number]> = [[this.startVertex, 0]];
    visited.add(this.startVertex);

    const allVertices = new Set<number>();
    const allEdges = new Set<number>();

    while (queue.length > 0) {
      const [current, depth] = queue.shift()!;

      const vertex = store.getVertex(current);
      if (!vertex) continue;

      if (this.vertexPredicate && !this.vertexPredicate(vertex)) {
        continue;
      }

      allVertices.add(current);

      // Score decays with depth
      const depthScore = this.score / (1 + depth);
      entries.push(
        createPostingEntry(current, {
          score: depthScore,
          fields: { ...vertex.properties, _depth: depth },
        }),
      );

      if (depth < this.maxHops) {
        const neighbors = store.neighbors(current, graphName, this.label, "out");
        for (const neighborId of neighbors) {
          if (!visited.has(neighborId)) {
            visited.add(neighborId);
            queue.push([neighborId, depth + 1]);
          }
        }
        // Collect edges
        const outEdges = store.outEdgeIds(current, graphName);
        for (const eid of outEdges) {
          const edge = store.getEdge(eid);
          if (edge && (this.label === null || edge.label === this.label)) {
            allEdges.add(eid);
          }
        }
      }
    }

    const gpl = new GraphPostingList(entries);
    for (const entry of entries) {
      gpl.setGraphPayload(
        entry.docId,
        createGraphPayload({
          subgraphVertices: allVertices,
          subgraphEdges: allEdges,
          score: entry.payload.score,
          graphName,
        }),
      );
    }
    return gpl;
  }
}

// -- PatternMatchOperator -----------------------------------------------------

export class PatternMatchOperator extends Operator {
  readonly pattern: GraphPattern;
  readonly graph: string;
  readonly score: number;

  constructor(pattern: GraphPattern, graph: string, score?: number) {
    super();
    this.pattern = pattern;
    this.graph = graph;
    this.score = score ?? 1.0;
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

      // Apply vertex constraints to filter candidates
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
            if (valid) {
              filtered.add(vid);
            }
          }
        }
        candidateSet = filtered;
      }

      candidates.set(vp.variable, candidateSet);
    }

    // Arc consistency: prune candidates based on edge patterns
    let changed = true;
    while (changed) {
      changed = false;
      for (const ep of this.pattern.edgePatterns) {
        if (ep.negated) continue;
        const srcCandidates = candidates.get(ep.sourceVar);
        const tgtCandidates = candidates.get(ep.targetVar);
        if (!srcCandidates || !tgtCandidates) continue;

        // Filter source candidates
        const validSources = new Set<number>();
        for (const srcId of srcCandidates) {
          const neighbors = store.neighbors(srcId, this.graph, ep.label, "out");
          let hasValid = false;
          for (const nid of neighbors) {
            if (tgtCandidates.has(nid)) {
              hasValid = true;
              break;
            }
          }
          if (hasValid) {
            validSources.add(srcId);
          }
        }
        if (validSources.size < srcCandidates.size) {
          candidates.set(ep.sourceVar, validSources);
          changed = true;
        }

        // Filter target candidates
        const validTargets = new Set<number>();
        for (const tgtId of tgtCandidates) {
          const neighbors = store.neighbors(tgtId, this.graph, ep.label, "in");
          let hasValid = false;
          for (const nid of neighbors) {
            if (validSources.has(nid)) {
              hasValid = true;
              break;
            }
          }
          if (hasValid) {
            validTargets.add(tgtId);
          }
        }
        if (validTargets.size < tgtCandidates.size) {
          candidates.set(ep.targetVar, validTargets);
          changed = true;
        }
      }
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
      // Validate all edges
      if (
        this._validateAllEdges(store, assignment) &&
        this._checkNegatedEdges(store, assignment)
      ) {
        results.push(new Map(assignment));
      }
      return;
    }

    // MRV: pick variable with smallest remaining domain
    let bestIdx = index;
    let bestSize = Infinity;
    for (let i = index; i < variables.length; i++) {
      const v = variables[i]!;
      const cands = candidates.get(v);
      const size = cands ? cands.size : 0;
      if (size < bestSize) {
        bestSize = size;
        bestIdx = i;
      }
    }

    // Swap
    if (bestIdx !== index) {
      const tmp = variables[index]!;
      variables[index] = variables[bestIdx]!;
      variables[bestIdx] = tmp;
    }

    const variable = variables[index]!;
    const candidateSet = candidates.get(variable) ?? new Set<number>();
    const usedVertices = new Set(assignment.values());

    for (const vid of candidateSet) {
      // Ensure injective mapping (no two pattern variables map to same vertex)
      if (usedVertices.has(vid)) continue;

      assignment.set(variable, vid);

      // Check edges for assigned variables only
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

      // Check that an edge with the given label exists
      const neighbors = store.neighbors(srcId, this.graph, ep.label, "out");
      if (!neighbors.includes(tgtId)) {
        // Also check edge constraints
        return false;
      }

      // Check edge constraints
      if (ep.constraints.length > 0) {
        const outEdges = store.outEdgeIds(srcId, this.graph);
        let found = false;
        for (const eid of outEdges) {
          const edge = store.getEdge(eid);
          if (
            edge &&
            edge.targetId === tgtId &&
            (ep.label === null || edge.label === ep.label)
          ) {
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
        }
        if (!found) return false;
      }
    }
    return true;
  }

  private _validateAllEdges(
    store: GraphStore,
    assignment: Map<string, number>,
  ): boolean {
    for (const ep of this.pattern.edgePatterns) {
      if (ep.negated) continue;
      const srcId = assignment.get(ep.sourceVar);
      const tgtId = assignment.get(ep.targetVar);
      if (srcId === undefined || tgtId === undefined) return false;

      const neighbors = store.neighbors(srcId, this.graph, ep.label, "out");
      if (!neighbors.includes(tgtId)) return false;

      if (ep.constraints.length > 0) {
        const outEdges = store.outEdgeIds(srcId, this.graph);
        let found = false;
        for (const eid of outEdges) {
          const edge = store.getEdge(eid);
          if (
            edge &&
            edge.targetId === tgtId &&
            (ep.label === null || edge.label === ep.label)
          ) {
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
        }
        if (!found) return false;
      }
    }
    return true;
  }

  private _checkNegatedEdges(
    store: GraphStore,
    assignment: Map<string, number>,
  ): boolean {
    for (const ep of this.pattern.edgePatterns) {
      if (!ep.negated) continue;
      const srcId = assignment.get(ep.sourceVar);
      const tgtId = assignment.get(ep.targetVar);
      if (srcId === undefined || tgtId === undefined) continue;

      const neighbors = store.neighbors(srcId, this.graph, ep.label, "out");
      if (neighbors.includes(tgtId)) {
        return false;
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
          (ep.label === null || edge.label === ep.label)
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

// -- RegularPathQueryOperator -------------------------------------------------

interface _NFAState {
  readonly id: number;
}

interface _NFA {
  readonly states: _NFAState[];
  readonly transitions: NFATransition[];
  readonly startState: number;
  readonly acceptState: number;
}

let _nfaStateCounter = 0;

function _newState(): _NFAState {
  return { id: _nfaStateCounter++ };
}

function _resetNfaCounter(): void {
  _nfaStateCounter = 0;
}

function _buildNfa(expr: RegularPathExpr): _NFA {
  if (expr instanceof Label) {
    const start = _newState();
    const accept = _newState();
    return {
      states: [start, accept],
      transitions: [{ from: start.id, to: accept.id, label: expr.name }],
      startState: start.id,
      acceptState: accept.id,
    };
  }

  if (expr instanceof Concat) {
    const left = _buildNfa(expr.left);
    const right = _buildNfa(expr.right);
    // Connect left accept to right start with epsilon
    const transitions = [
      ...left.transitions,
      ...right.transitions,
      { from: left.acceptState, to: right.startState, label: null as string | null },
    ];
    return {
      states: [...left.states, ...right.states],
      transitions,
      startState: left.startState,
      acceptState: right.acceptState,
    };
  }

  if (expr instanceof Alternation) {
    const start = _newState();
    const accept = _newState();
    const left = _buildNfa(expr.left);
    const right = _buildNfa(expr.right);
    const transitions: NFATransition[] = [
      ...left.transitions,
      ...right.transitions,
      { from: start.id, to: left.startState, label: null },
      { from: start.id, to: right.startState, label: null },
      { from: left.acceptState, to: accept.id, label: null },
      { from: right.acceptState, to: accept.id, label: null },
    ];
    return {
      states: [start, accept, ...left.states, ...right.states],
      transitions,
      startState: start.id,
      acceptState: accept.id,
    };
  }

  if (expr instanceof KleeneStar) {
    const start = _newState();
    const accept = _newState();
    const inner = _buildNfa(expr.inner);
    const transitions: NFATransition[] = [
      ...inner.transitions,
      { from: start.id, to: inner.startState, label: null },
      { from: start.id, to: accept.id, label: null },
      { from: inner.acceptState, to: inner.startState, label: null },
      { from: inner.acceptState, to: accept.id, label: null },
    ];
    return {
      states: [start, accept, ...inner.states],
      transitions,
      startState: start.id,
      acceptState: accept.id,
    };
  }

  if (expr instanceof BoundedLabel) {
    // Expand bounded into concat of copies
    // a{m,n} = a/a/.../a (m times) / (epsilon | a) / ... (n-m times)
    if (expr.minCount === 0 && expr.maxCount === 0) {
      // epsilon
      const start = _newState();
      return {
        states: [start],
        transitions: [],
        startState: start.id,
        acceptState: start.id,
      };
    }

    let current: _NFA | null = null;

    // Build required repetitions
    for (let i = 0; i < expr.minCount; i++) {
      const copy = _buildNfa(new Label(expr.name));
      if (current === null) {
        current = copy;
      } else {
        const transitions: NFATransition[] = [
          ...current.transitions,
          ...copy.transitions,
          { from: current.acceptState, to: copy.startState, label: null },
        ];
        current = {
          states: [...current.states, ...copy.states],
          transitions,
          startState: current.startState,
          acceptState: copy.acceptState,
        };
      }
    }

    // Build optional repetitions
    for (let i = expr.minCount; i < expr.maxCount; i++) {
      const copy = _buildNfa(new Label(expr.name));
      if (current === null) {
        // min = 0, first optional
        const start = _newState();
        const accept = _newState();
        const transitions: NFATransition[] = [
          ...copy.transitions,
          { from: start.id, to: copy.startState, label: null },
          { from: start.id, to: accept.id, label: null },
          { from: copy.acceptState, to: accept.id, label: null },
        ];
        current = {
          states: [start, accept, ...copy.states],
          transitions,
          startState: start.id,
          acceptState: accept.id,
        };
      } else {
        // Add optional step
        const accept = _newState();
        const transitions: NFATransition[] = [
          ...current.transitions,
          ...copy.transitions,
          { from: current.acceptState, to: copy.startState, label: null },
          { from: current.acceptState, to: accept.id, label: null },
          { from: copy.acceptState, to: accept.id, label: null },
        ];
        current = {
          states: [...current.states, accept, ...copy.states],
          transitions,
          startState: current.startState,
          acceptState: accept.id,
        };
      }
    }

    if (current === null) {
      const start = _newState();
      return {
        states: [start],
        transitions: [],
        startState: start.id,
        acceptState: start.id,
      };
    }

    return current;
  }

  throw new Error(`Unsupported RPQ expression type: ${String(expr)}`);
}

export class RegularPathQueryOperator extends Operator {
  readonly pathExpr: RegularPathExpr;
  readonly graph: string;
  readonly startVertex: number | null;
  readonly score: number;

  constructor(
    pathExpr: RegularPathExpr,
    graph: string,
    startVertex?: number | null,
    score?: number,
  ) {
    super();
    this.pathExpr = pathExpr;
    this.graph = graph;
    this.startVertex = startVertex ?? null;
    this.score = score ?? 1.0;
  }

  execute(context: ExecutionContext): PostingList {
    const store = context.graphStore as GraphStore;

    // Try index lookup first
    const indexResult = this._tryIndexLookup(context, store);
    if (indexResult !== null) return indexResult;

    // Build NFA and convert to DFA
    _resetNfaCounter();
    const nfa = _buildNfa(this.pathExpr);
    const dfa = subsetConstruction(nfa);

    const entries: PostingEntry[] = [];
    const graphPayloads = new Map<number, ReturnType<typeof createGraphPayload>>();
    const seenPairs = new Set<string>();

    const startVertices: number[] = [];
    if (this.startVertex !== null) {
      startVertices.push(this.startVertex);
    } else {
      const allVids = store.vertexIdsInGraph(this.graph);
      for (const vid of allVids) {
        startVertices.push(vid);
      }
    }

    let docIdCounter = 0;

    for (const startVid of startVertices) {
      const reachable = this._simulateDfa(store, dfa, startVid);

      for (const [endVid, pathEdges] of reachable) {
        const pairKey = `${String(startVid)}-${String(endVid)}`;
        if (seenPairs.has(pairKey)) continue;
        seenPairs.add(pairKey);

        const docId = docIdCounter++;
        entries.push(
          createPostingEntry(docId, {
            score: this.score,
            fields: {
              start: startVid,
              end: endVid,
            },
          }),
        );

        graphPayloads.set(
          docId,
          createGraphPayload({
            subgraphVertices: new Set([startVid, endVid]),
            subgraphEdges: pathEdges,
            score: this.score,
            graphName: this.graph,
          }),
        );
      }
    }

    const gpl = new GraphPostingList(entries);
    for (const [docId, gp] of graphPayloads) {
      gpl.setGraphPayload(docId, gp);
    }
    return gpl;
  }

  private _simulateDfa(
    store: GraphStore,
    dfa: {
      transitions: Map<string, Map<string, string>>;
      startState: string;
      acceptStates: Set<string>;
    },
    startVid: number,
  ): Array<[number, Set<number>]> {
    const results: Array<[number, Set<number>]> = [];

    // BFS over (vertex, dfaState) pairs
    type SearchState = [number, string, Set<number>]; // [vertexId, dfaState, edgeIds]
    const queue: SearchState[] = [[startVid, dfa.startState, new Set()]];
    const visited = new Set<string>();
    visited.add(`${String(startVid)}:${dfa.startState}`);

    // Check if start state is accepting (epsilon path)
    if (dfa.acceptStates.has(dfa.startState)) {
      results.push([startVid, new Set()]);
    }

    while (queue.length > 0) {
      const [vid, dfaState, edgesSoFar] = queue.shift()!;
      const stateTransitions = dfa.transitions.get(dfaState);
      if (!stateTransitions) continue;

      for (const [label, nextDfaState] of stateTransitions) {
        // Find edges with this label from current vertex
        const outEdges = store.outEdgeIds(vid, this.graph);
        for (const eid of outEdges) {
          const edge = store.getEdge(eid);
          if (!edge || edge.label !== label) continue;

          const targetVid = edge.targetId;
          const visitKey = `${String(targetVid)}:${nextDfaState}`;
          if (visited.has(visitKey)) continue;
          visited.add(visitKey);

          const newEdges = new Set(edgesSoFar);
          newEdges.add(eid);

          if (dfa.acceptStates.has(nextDfaState)) {
            results.push([targetVid, newEdges]);
          }

          queue.push([targetVid, nextDfaState, newEdges]);
        }
      }
    }

    return results;
  }

  private _tryIndexLookup(
    context: ExecutionContext,
    _store: GraphStore,
  ): GraphPostingList | null {
    const pathIndex = context.pathIndex as
      | {
          hasPath?(labels: string[], graphName: string): boolean;
          lookup?(labels: string[], graphName: string): Array<[number, number]>;
        }
      | null
      | undefined;

    if (!pathIndex || !pathIndex.hasPath || !pathIndex.lookup) return null;

    const labels = this._extractLabelSequence(this.pathExpr);
    if (labels === null) return null;

    if (!pathIndex.hasPath(labels, this.graph)) return null;

    const pairs = pathIndex.lookup(labels, this.graph);
    const entries: PostingEntry[] = [];
    let docId = 0;

    for (const [startVid, endVid] of pairs) {
      if (this.startVertex !== null && startVid !== this.startVertex) continue;
      entries.push(
        createPostingEntry(docId, {
          score: this.score,
          fields: { start: startVid, end: endVid },
        }),
      );
      docId++;
    }

    return new GraphPostingList(entries);
  }

  private _extractLabelSequence(expr: RegularPathExpr): string[] | null {
    if (expr instanceof Label) {
      return [expr.name];
    }
    if (expr instanceof Concat) {
      const left = this._extractLabelSequence(expr.left);
      const right = this._extractLabelSequence(expr.right);
      if (left !== null && right !== null) {
        return [...left, ...right];
      }
    }
    return null;
  }
}

// -- VertexAggregationOperator ------------------------------------------------

export class VertexAggregationOperator extends Operator {
  readonly source: Operator;
  readonly propertyName: string;
  readonly aggFn: (values: number[]) => number;

  constructor(
    source: Operator,
    propertyName: string,
    aggFn?: (values: number[]) => number,
  ) {
    super();
    this.source = source;
    this.propertyName = propertyName;
    this.aggFn =
      aggFn ??
      ((vals: number[]) => {
        if (vals.length === 0) return 0;
        let s = 0;
        for (let i = 0; i < vals.length; i++) {
          s += vals[i]!;
        }
        return s / vals.length;
      });
  }

  execute(context: ExecutionContext): PostingList {
    const store = context.graphStore as GraphStore;
    const sourceResult = this.source.execute(context);

    const values: number[] = [];
    for (const entry of sourceResult) {
      const vertex = store.getVertex(entry.docId);
      if (vertex) {
        const prop = vertex.properties[this.propertyName];
        if (typeof prop === "number") {
          values.push(prop);
        }
      }
    }

    const aggregated = this.aggFn(values);

    const entries: PostingEntry[] = [];
    for (const entry of sourceResult) {
      entries.push({
        docId: entry.docId,
        payload: createPayload({
          positions: entry.payload.positions,
          score: aggregated,
          fields: {
            ...(entry.payload.fields as Record<string, unknown>),
            _aggregated: aggregated,
            _property: this.propertyName,
          },
        }),
      });
    }

    return new PostingList(entries);
  }
}

// -- WeightedPathQueryOperator ------------------------------------------------

/**
 * Weighted regular path query: finds paths matching a regular expression
 * with associated edge weights. The score of each result path is the
 * aggregation (sum/min/max/product) of edge weights along the path.
 */
export class WeightedPathQueryOperator extends Operator {
  readonly pathExpr: RegularPathExpr;
  readonly graph: string;
  readonly startVertex: number | null;
  readonly weightProperty: string;
  readonly aggregation: "sum" | "min" | "max" | "product";
  readonly weightThreshold: number | null;
  readonly score: number;

  constructor(
    pathExpr: RegularPathExpr,
    graph: string,
    opts?: {
      startVertex?: number | null;
      weightProperty?: string;
      aggregation?: "sum" | "min" | "max" | "product";
      weightThreshold?: number | null;
      score?: number;
    },
  ) {
    super();
    this.pathExpr = pathExpr;
    this.graph = graph;
    this.startVertex = opts?.startVertex ?? null;
    this.weightProperty = opts?.weightProperty ?? "weight";
    this.aggregation = opts?.aggregation ?? "sum";
    this.weightThreshold = opts?.weightThreshold ?? null;
    this.score = opts?.score ?? 1.0;
  }

  execute(context: ExecutionContext): PostingList {
    const store = context.graphStore as GraphStore;

    // Build NFA and convert to DFA
    _resetNfaCounter();
    const nfa = _buildNfa(this.pathExpr);
    const dfa = subsetConstruction(nfa);

    const entries: PostingEntry[] = [];
    const graphPayloads = new Map<number, ReturnType<typeof createGraphPayload>>();
    const seenPairs = new Set<string>();

    const startVertices: number[] = [];
    if (this.startVertex !== null) {
      startVertices.push(this.startVertex);
    } else {
      const allVids = store.vertexIdsInGraph(this.graph);
      for (const vid of allVids) {
        startVertices.push(vid);
      }
    }

    let docIdCounter = 0;

    for (const startVid of startVertices) {
      const reachable = this._simulateWeightedDfa(store, dfa, startVid);

      for (const [endVid, pathEdges, pathWeight] of reachable) {
        // Apply weight threshold
        if (this.weightThreshold !== null && pathWeight < this.weightThreshold) {
          continue;
        }

        const pairKey = `${String(startVid)}-${String(endVid)}`;
        if (seenPairs.has(pairKey)) continue;
        seenPairs.add(pairKey);

        const docId = docIdCounter++;
        entries.push(
          createPostingEntry(docId, {
            score: pathWeight * this.score,
            fields: {
              start: startVid,
              end: endVid,
              weight: pathWeight,
              path_length: pathEdges.size,
            },
          }),
        );

        graphPayloads.set(
          docId,
          createGraphPayload({
            subgraphVertices: new Set([startVid, endVid]),
            subgraphEdges: pathEdges,
            score: pathWeight * this.score,
            graphName: this.graph,
          }),
        );
      }
    }

    const gpl = new GraphPostingList(entries);
    for (const [docId, gp] of graphPayloads) {
      gpl.setGraphPayload(docId, gp);
    }
    return gpl;
  }

  private _simulateWeightedDfa(
    store: GraphStore,
    dfa: {
      transitions: Map<string, Map<string, string>>;
      startState: string;
      acceptStates: Set<string>;
    },
    startVid: number,
  ): Array<[number, Set<number>, number]> {
    const results: Array<[number, Set<number>, number]> = [];

    // BFS over (vertex, dfaState, edges, weight) tuples
    type SearchState = [number, string, Set<number>, number];
    const initWeight = this.aggregation === "product" ? 1.0 : 0.0;
    const queue: SearchState[] = [[startVid, dfa.startState, new Set(), initWeight]];
    const visited = new Set<string>();
    visited.add(`${String(startVid)}:${dfa.startState}`);

    // Check if start state is accepting
    if (dfa.acceptStates.has(dfa.startState)) {
      results.push([startVid, new Set(), initWeight]);
    }

    while (queue.length > 0) {
      const [vid, dfaState, edgesSoFar, weightSoFar] = queue.shift()!;
      const stateTransitions = dfa.transitions.get(dfaState);
      if (!stateTransitions) continue;

      for (const [label, nextDfaState] of stateTransitions) {
        const outEdges = store.outEdgeIds(vid, this.graph);
        for (const eid of outEdges) {
          const edge = store.getEdge(eid);
          if (!edge || edge.label !== label) continue;

          const targetVid = edge.targetId;
          const visitKey = `${String(targetVid)}:${nextDfaState}`;
          if (visited.has(visitKey)) continue;
          visited.add(visitKey);

          const newEdges = new Set(edgesSoFar);
          newEdges.add(eid);

          // Get edge weight
          const edgeWeight =
            typeof edge.properties[this.weightProperty] === "number"
              ? (edge.properties[this.weightProperty] as number)
              : 1.0;

          // Aggregate weight
          let newWeight: number;
          switch (this.aggregation) {
            case "sum":
              newWeight = weightSoFar + edgeWeight;
              break;
            case "min":
              newWeight =
                edgesSoFar.size === 0 ? edgeWeight : Math.min(weightSoFar, edgeWeight);
              break;
            case "max":
              newWeight = Math.max(weightSoFar, edgeWeight);
              break;
            case "product":
              newWeight = weightSoFar * edgeWeight;
              break;
          }

          if (dfa.acceptStates.has(nextDfaState)) {
            results.push([targetVid, newEdges, newWeight]);
          }

          queue.push([targetVid, nextDfaState, newEdges, newWeight]);
        }
      }
    }

    return results;
  }
}

// -- CypherQueryOperator ------------------------------------------------------

export class CypherQueryOperator extends Operator {
  readonly graph: string;
  readonly query: string;
  readonly graphName: string;
  readonly params: Record<string, unknown>;
  readonly colNames: string[];

  constructor(
    graph: string,
    query: string,
    graphName: string,
    params?: Record<string, unknown>,
    colNames?: string[],
  ) {
    super();
    this.graph = graph;
    this.query = query;
    this.graphName = graphName;
    this.params = params ?? {};
    this.colNames = colNames ?? [];
  }

  execute(context: ExecutionContext): PostingList {
    const store = context.graphStore as GraphStore;
    const compiler = new CypherCompiler(store, this.graphName, this.params);
    const gpl = compiler.executePostingList(this.query, this.params);
    return gpl.toPostingList();
  }
}
