//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Cypher-to-PostingList compiler
// 1:1 port of uqa/graph/cypher/compiler.py
//
// Cypher evaluation is inherently dynamically typed (unknown values flowing
// through expression evaluation), so we disable some strict lint rules here.
/* eslint-disable @typescript-eslint/no-base-to-string, @typescript-eslint/restrict-template-expressions, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unnecessary-type-conversion */
//
// Translates a CypherQuery AST into posting list operations on
// GraphStore.  This preserves UQA's core thesis that ALL paradigms
// -- relational, text, vector, AND graph -- flow through the
// posting list abstraction.
//
// Execution model:
//   Each Cypher clause transforms a binding posting list -- a
//   GraphPostingList where every PostingEntry represents one
//   binding row.  payload.fields carries the variable-to-value
//   assignments (vertex IDs, edge IDs, scalars).
//
//   MATCH  -> pattern matching produces a new GraphPostingList
//   WHERE  -> filters entries whose fields fail the predicate
//   CREATE -> mutates the GraphStore, extends fields with new bindings
//   SET    -> mutates vertex/edge properties in-place
//   DELETE -> removes vertices/edges from the GraphStore
//   RETURN -> projects fields into result dicts
//   WITH   -> projects and reshapes the binding posting list
//   UNWIND -> expands list-valued fields into multiple entries

import type { PostingEntry, Vertex, Edge } from "../../core/types.js";
import { createPayload, createVertex, createEdge } from "../../core/types.js";
import type { PostingList } from "../../core/posting-list.js";
import { GraphPostingList, createGraphPayload } from "../posting-list.js";
import type { GraphPayload } from "../posting-list.js";
import type { GraphStore } from "../../storage/abc/graph-store.js";
import type {
  CypherQuery,
  CypherExpr,
  MatchClause,
  CreateClause,
  MergeClause,
  SetClause,
  SetItem,
  DeleteClause,
  ReturnClause,
  WithClause,
  UnwindClause,
  OrderByItem,
  NodePattern,
  RelationshipPattern,
  PathPattern,
  PropertyAccessExpr,
  BinaryOpExpr,
  UnaryOpExpr,
  FunctionCallExpr,
  InExpr,
  IsNullExpr,
  CaseExpr,
} from "./ast.js";
import { createNodePattern, createRelationshipPattern } from "./ast.js";
import { parseCypher } from "./parser.js";

// -- Binding fields type alias ------------------------------------------------
// Keys are Cypher variable names, values are vertex IDs, edge IDs, or scalars.
type BindingFields = Record<string, unknown>;

// -- Marker classes for vertex/edge references --------------------------------
// We use branded types to distinguish vertex IDs from edge IDs from plain ints.

class _VertexRef {
  readonly __brand = "vertex" as const;
  readonly value: number;
  constructor(value: number) {
    this.value = value;
  }
}

class _EdgeRef {
  readonly __brand = "edge" as const;
  readonly value: number;
  constructor(value: number) {
    this.value = value;
  }
}

function isVertexRef(v: unknown): v is _VertexRef {
  return v instanceof _VertexRef;
}

function isEdgeRef(v: unknown): v is _EdgeRef {
  return v instanceof _EdgeRef;
}

function refValue(v: unknown): number | null {
  if (v instanceof _VertexRef) return v.value;
  if (v instanceof _EdgeRef) return v.value;
  if (typeof v === "number") return v;
  return null;
}

// -- CypherCompiler -----------------------------------------------------------

export class CypherCompiler {
  private readonly _graph: GraphStore;
  private readonly _graphName: string;
  private readonly _params: Record<string, unknown>;
  private _nextDocId: number;

  constructor(graph: GraphStore, graphName: string, params?: Record<string, unknown>) {
    this._graph = graph;
    this._graphName = graphName;
    this._params = params ?? {};
    this._nextDocId = 1;
  }

  get store(): GraphStore {
    return this._graph;
  }

  get graphName(): string {
    return this._graphName;
  }

  // -- Top-level execution ----------------------------------------------------

  execute(query: string, params?: Record<string, unknown>): PostingList {
    const ast = parseCypher(query);
    return this.executeAST(ast, params);
  }

  executePostingList(
    query: string,
    params?: Record<string, unknown>,
  ): GraphPostingList {
    const ast = parseCypher(query);
    return this.executeASTPostingList(ast, params);
  }

  executeAST(ast: CypherQuery, params?: Record<string, unknown>): PostingList {
    const gpl = this.executeASTPostingList(ast, params);
    return gpl.toPostingList();
  }

  executeASTPostingList(
    ast: CypherQuery,
    params?: Record<string, unknown>,
  ): GraphPostingList {
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        this._params[k] = v;
      }
    }

    // Start with a single empty binding (one entry, no fields)
    let gpl = this._emptyBinding();

    for (const clause of ast.clauses) {
      switch (clause.kind) {
        case "match":
          gpl = this._execMatch(clause, gpl);
          break;
        case "create":
          gpl = this._execCreate(clause, gpl);
          break;
        case "merge":
          gpl = this._execMerge(clause, gpl);
          break;
        case "set":
          gpl = this._execSet(clause, gpl);
          break;
        case "delete":
          gpl = this._execDelete(clause, gpl);
          break;
        case "return":
          gpl = this._execReturnPostingList(clause, gpl);
          break;
        case "with":
          gpl = this._execWith(clause, gpl);
          break;
        case "unwind":
          gpl = this._execUnwind(clause, gpl);
          break;
        // "remove" and "call" are recognized but not mutating the binding list
        default:
          break;
      }
    }

    return gpl;
  }

  // -- Convenience: execute and return plain result dicts ---------------------

  executeRows(
    query: string,
    params?: Record<string, unknown>,
  ): Array<Record<string, unknown>> {
    const ast = parseCypher(query);
    return this.executeASTRows(ast, params);
  }

  executeASTRows(
    ast: CypherQuery,
    params?: Record<string, unknown>,
  ): Array<Record<string, unknown>> {
    const gpl = this.executeASTPostingList(ast, params);
    const rows: Array<Record<string, unknown>> = [];
    for (const entry of gpl) {
      const row: Record<string, unknown> = {};
      const fields = entry.payload.fields as Record<string, unknown>;
      for (const k of Object.keys(fields)) {
        row[k] = fields[k];
      }
      rows.push(row);
    }
    return rows;
  }

  // -- Binding posting list construction --------------------------------------

  private _emptyBinding(): GraphPostingList {
    const docId = this._allocDocId();
    const entry: PostingEntry = {
      docId,
      payload: createPayload({ score: 1.0, fields: {} }),
    };
    const gpl = new GraphPostingList([entry]);
    gpl.setGraphPayload(docId, createGraphPayload());
    return gpl;
  }

  private _allocDocId(): number {
    const did = this._nextDocId;
    this._nextDocId += 1;
    return did;
  }

  private _makeBindingEntry(
    fields: BindingFields,
    _vertices: ReadonlySet<number>,
    _edges: ReadonlySet<number>,
  ): [PostingEntry, number] {
    const docId = this._allocDocId();
    const entry: PostingEntry = {
      docId,
      payload: createPayload({ score: 1.0, fields: { ...fields } }),
    };
    return [entry, docId];
  }

  // -- MATCH ------------------------------------------------------------------

  private _execMatch(
    clause: MatchClause,
    bindings: GraphPostingList,
  ): GraphPostingList {
    const entries: PostingEntry[] = [];
    const payloads = new Map<number, GraphPayload>();

    for (const bindingEntry of bindings) {
      const existingFields = {
        ...(bindingEntry.payload.fields as Record<string, unknown>),
      };
      let matched = this._matchPatterns(clause.patterns, existingFields);

      // Apply WHERE filter
      if (clause.where !== null) {
        matched = matched.filter((m) => this._eval(clause.where!, m));
      }

      if (clause.optional && matched.length === 0) {
        // OPTIONAL MATCH: keep existing binding with NULLs
        const nullFields: BindingFields = { ...existingFields };
        for (const pat of clause.patterns) {
          for (const elem of pat.elements) {
            if (_isNodePattern(elem) && elem.variable !== null) {
              if (!(elem.variable in nullFields)) {
                nullFields[elem.variable] = null;
              }
            } else if (_isRelPattern(elem) && elem.variable !== null) {
              if (!(elem.variable in nullFields)) {
                nullFields[elem.variable] = null;
              }
            }
          }
        }
        const [entry, docId] = this._makeBindingEntry(
          nullFields,
          new Set<number>(),
          new Set<number>(),
        );
        entries.push(entry);
        payloads.set(docId, createGraphPayload());
      } else {
        for (const mFields of matched) {
          const vtxIds = new Set<number>();
          const edgeIds = new Set<number>();
          for (const v of Object.values(mFields)) {
            if (isVertexRef(v)) {
              vtxIds.add(v.value);
            } else if (isEdgeRef(v)) {
              edgeIds.add(v.value);
            } else if (typeof v === "number") {
              if (this._graph.getVertex(v) !== null) {
                vtxIds.add(v);
              } else if (this._graph.getEdge(v) !== null) {
                edgeIds.add(v);
              }
            }
          }
          const [entry, docId] = this._makeBindingEntry(mFields, vtxIds, edgeIds);
          entries.push(entry);
          payloads.set(
            docId,
            createGraphPayload({
              subgraphVertices: vtxIds,
              subgraphEdges: edgeIds,
            }),
          );
        }
      }
    }

    return new GraphPostingList(entries, payloads);
  }

  private _matchPatterns(
    patterns: readonly PathPattern[],
    initialFields: BindingFields,
  ): BindingFields[] {
    let results: BindingFields[] = [{ ...initialFields }];
    for (const pattern of patterns) {
      const nextResults: BindingFields[] = [];
      for (const fields of results) {
        nextResults.push(...this._matchPath(pattern, fields));
      }
      results = nextResults;
    }
    return results;
  }

  private _matchPath(pattern: PathPattern, fields: BindingFields): BindingFields[] {
    const elements = pattern.elements;
    if (elements.length === 0) {
      return [{ ...fields }];
    }

    // Assign synthetic variable names to anonymous nodes/rels
    const anonElements = this._assignAnonVars(elements);

    const first = anonElements[0];
    if (!first || !_isNodePattern(first)) {
      return [{ ...fields }];
    }

    const candidates = this._nodeCandidates(first, fields);
    let current: BindingFields[] = [];
    for (const vtx of candidates) {
      const newFields: BindingFields = { ...fields };
      const varName = first.variable;
      if (varName === null) continue;
      if (varName in fields) {
        const bound = fields[varName];
        const boundId = refValue(bound);
        if (boundId !== null && boundId !== vtx.vertexId) continue;
      }
      newFields[varName] = new _VertexRef(vtx.vertexId);
      current.push(newFields);
    }

    // Process (rel, node) pairs
    let idx = 1;
    while (idx < anonElements.length) {
      const relPat = anonElements[idx];
      const nodePat = anonElements[idx + 1];
      if (!relPat || !nodePat || !_isRelPattern(relPat) || !_isNodePattern(nodePat)) {
        break;
      }

      const nextCurrent: BindingFields[] = [];
      for (const f of current) {
        nextCurrent.push(...this._expandRel(f, relPat, nodePat, anonElements, idx));
      }
      current = nextCurrent;
      idx += 2;
    }

    // Strip synthetic anonymous variables from results
    const anonVars = new Set<string>();
    for (const e of anonElements) {
      if (_isNodePattern(e) || _isRelPattern(e)) {
        const v = _isNodePattern(e) ? e.variable : e.variable;
        if (v !== null && v.startsWith("_anon_")) {
          anonVars.add(v);
        }
      }
    }
    if (anonVars.size > 0) {
      current = current.map((f) => {
        const cleaned: BindingFields = {};
        for (const k of Object.keys(f)) {
          if (!anonVars.has(k)) {
            cleaned[k] = f[k];
          }
        }
        return cleaned;
      });
    }

    return current;
  }

  private _assignAnonVars(
    elements: ReadonlyArray<NodePattern | RelationshipPattern>,
  ): Array<NodePattern | RelationshipPattern> {
    const result: Array<NodePattern | RelationshipPattern> = [];
    for (const elem of elements) {
      if (_isNodePattern(elem) && elem.variable === null) {
        const anonVar = `_anon_${this._allocDocId()}`;
        result.push(
          createNodePattern(anonVar, [...elem.labels], new Map(elem.properties)),
        );
      } else if (_isRelPattern(elem) && elem.variable === null) {
        const anonVar = `_anon_${this._allocDocId()}`;
        result.push(
          createRelationshipPattern({
            variable: anonVar,
            types: [...elem.types],
            properties: new Map(elem.properties),
            direction: elem.direction,
            minHops: elem.minHops,
            maxHops: elem.maxHops,
          }),
        );
      } else {
        result.push(elem);
      }
    }
    return result;
  }

  private _nodeCandidates(pat: NodePattern, fields: BindingFields): Vertex[] {
    if (pat.variable !== null && pat.variable in fields) {
      const val = fields[pat.variable];
      if (val === null || val === undefined) return [];
      const vid = refValue(val);
      if (vid === null) return [];
      const vtx = this._graph.getVertex(vid);
      if (vtx !== null && this._vertexMatches(vtx, pat, fields)) {
        return [vtx];
      }
      return [];
    }

    if (!this._graph.hasGraph(this._graphName)) {
      return [];
    }

    let candidates: Vertex[];
    if (pat.labels.length > 0) {
      const seen = new Set<number>();
      candidates = [];
      for (const label of pat.labels) {
        for (const v of this._graph.verticesByLabel(label, this._graphName)) {
          if (!seen.has(v.vertexId)) {
            seen.add(v.vertexId);
            candidates.push(v);
          }
        }
      }
    } else {
      candidates = this._graph.verticesInGraph(this._graphName);
    }

    return candidates.filter((v) => this._vertexMatches(v, pat, fields));
  }

  private _vertexMatches(
    vertex: Vertex,
    pat: NodePattern,
    fields?: BindingFields | null,
  ): boolean {
    if (pat.labels.length > 0 && !pat.labels.includes(vertex.label)) {
      return false;
    }
    if (pat.properties.size > 0) {
      const ctx = fields ?? {};
      for (const [key, valExpr] of pat.properties) {
        const expected = this._eval(valExpr, ctx);
        const actual = vertex.properties[key];
        if (actual !== expected) return false;
      }
    }
    return true;
  }

  private _expandRel(
    fields: BindingFields,
    relPat: RelationshipPattern,
    nextNode: NodePattern,
    elements: ReadonlyArray<NodePattern | RelationshipPattern>,
    idx: number,
  ): BindingFields[] {
    // Find the previous node's vertex ID from the binding
    const prevNode = elements[idx - 1];
    if (!prevNode || !_isNodePattern(prevNode)) return [];

    const prevVarName = prevNode.variable;
    const prevVal = prevVarName !== null ? fields[prevVarName] : undefined;
    const prevVid = refValue(prevVal);
    if (prevVid === null) return [];

    if (relPat.minHops !== null || relPat.maxHops !== null) {
      return this._expandVarLength(fields, prevVid, relPat, nextNode);
    }

    return this._expandSingleHop(fields, prevVid, relPat, nextNode);
  }

  private _expandSingleHop(
    fields: BindingFields,
    srcVid: number,
    relPat: RelationshipPattern,
    nextNode: NodePattern,
  ): BindingFields[] {
    const results: BindingFields[] = [];

    for (const [edge, neighborId] of this._getEdges(srcVid, relPat)) {
      const neighbor = this._graph.getVertex(neighborId);
      if (neighbor === null) continue;
      if (!this._vertexMatches(neighbor, nextNode, fields)) continue;
      if (!this._edgeMatches(edge, relPat, fields)) continue;

      // Check consistency with already-bound variables
      const newFields: BindingFields = { ...fields };
      if (relPat.variable !== null) {
        if (relPat.variable in fields) {
          const boundId = refValue(fields[relPat.variable]);
          if (boundId !== null && boundId !== edge.edgeId) continue;
        }
        newFields[relPat.variable] = new _EdgeRef(edge.edgeId);
      }
      if (nextNode.variable !== null) {
        if (nextNode.variable in fields) {
          const boundId = refValue(fields[nextNode.variable]);
          if (boundId !== null && boundId !== neighbor.vertexId) continue;
        }
        newFields[nextNode.variable] = new _VertexRef(neighbor.vertexId);
      }
      results.push(newFields);
    }

    return results;
  }

  private _expandVarLength(
    fields: BindingFields,
    srcVid: number,
    relPat: RelationshipPattern,
    nextNode: NodePattern,
  ): BindingFields[] {
    const minHops = relPat.minHops !== null ? relPat.minHops : 1;
    const maxHops = relPat.maxHops;

    const results: BindingFields[] = [];
    // BFS: [vertexId, depth, pathEdgeIds]
    const frontier: Array<[number, number, number[]]> = [[srcVid, 0, []]];

    while (frontier.length > 0) {
      const item = frontier.shift()!;
      const vid = item[0];
      const depth = item[1];
      const pathEids = item[2];

      if (depth >= minHops) {
        const vtx = this._graph.getVertex(vid);
        if (vtx !== null && this._vertexMatches(vtx, nextNode, fields)) {
          const newFields: BindingFields = { ...fields };
          if (relPat.variable !== null) {
            newFields[relPat.variable] = [...pathEids];
          }
          if (nextNode.variable !== null) {
            if (nextNode.variable in fields) {
              const boundId = refValue(fields[nextNode.variable]);
              if (boundId !== null && boundId !== vid) continue;
            }
            newFields[nextNode.variable] = new _VertexRef(vid);
          }
          results.push(newFields);
        }
      }

      if (maxHops !== null && depth >= maxHops) continue;

      for (const [edge, neighborId] of this._getEdges(vid, relPat)) {
        if (pathEids.includes(edge.edgeId)) continue;
        if (!this._edgeMatches(edge, relPat, fields)) continue;
        frontier.push([neighborId, depth + 1, [...pathEids, edge.edgeId]]);
      }
    }

    return results;
  }

  private _getEdges(
    vertexId: number,
    relPat: RelationshipPattern,
  ): Array<[Edge, number]> {
    let results: Array<[Edge, number]> = [];
    const direction = relPat.direction;

    if (direction === "out" || direction === "both") {
      for (const eid of this._graph.outEdgeIds(vertexId, this._graphName)) {
        const edge = this._graph.getEdge(eid);
        if (edge !== null) {
          results.push([edge, edge.targetId]);
        }
      }
    }
    if (direction === "in" || direction === "both") {
      for (const eid of this._graph.inEdgeIds(vertexId, this._graphName)) {
        const edge = this._graph.getEdge(eid);
        if (edge !== null) {
          results.push([edge, edge.sourceId]);
        }
      }
    }

    if (relPat.types.length > 0) {
      results = results.filter(([e]) => relPat.types.includes(e.label));
    }
    return results;
  }

  private _edgeMatches(
    edge: Edge,
    relPat: RelationshipPattern,
    fields?: BindingFields | null,
  ): boolean {
    if (relPat.properties.size > 0) {
      const ctx = fields ?? {};
      for (const [key, valExpr] of relPat.properties) {
        const expected = this._eval(valExpr, ctx);
        const actual = edge.properties[key];
        if (actual !== expected) return false;
      }
    }
    return true;
  }

  // -- CREATE -----------------------------------------------------------------

  private _execCreate(
    clause: CreateClause,
    bindings: GraphPostingList,
  ): GraphPostingList {
    const entries: PostingEntry[] = [];
    const payloads = new Map<number, GraphPayload>();

    for (const bindingEntry of bindings) {
      const fields: BindingFields = {
        ...(bindingEntry.payload.fields as Record<string, unknown>),
      };
      const createdVids = new Set<number>();
      const createdEids = new Set<number>();

      for (const pattern of clause.patterns) {
        this._createPath(pattern, fields, createdVids, createdEids);
      }

      const [entry, docId] = this._makeBindingEntry(fields, createdVids, createdEids);
      entries.push(entry);
      payloads.set(
        docId,
        createGraphPayload({
          subgraphVertices: createdVids,
          subgraphEdges: createdEids,
        }),
      );
    }

    return new GraphPostingList(entries, payloads);
  }

  private _createPath(
    pattern: PathPattern,
    fields: BindingFields,
    createdVids: Set<number>,
    createdEids: Set<number>,
  ): void {
    const elements = pattern.elements;
    for (let i = 0; i < elements.length; i++) {
      const elem = elements[i]!;
      if (_isNodePattern(elem)) {
        if (elem.variable !== null && elem.variable in fields) {
          continue;
        }
        const vtx = this._createVertex(elem, fields);
        createdVids.add(vtx.vertexId);
        if (elem.variable !== null) {
          fields[elem.variable] = new _VertexRef(vtx.vertexId);
        }
      } else if (_isRelPattern(elem)) {
        const nextNode = elements[i + 1];
        if (!nextNode || !_isNodePattern(nextNode)) continue;

        // Create target if not yet bound
        if (nextNode.variable !== null && !(nextNode.variable in fields)) {
          const vtx = this._createVertex(nextNode, fields);
          createdVids.add(vtx.vertexId);
          fields[nextNode.variable] = new _VertexRef(vtx.vertexId);
        }

        const prevNode = elements[i - 1];
        if (!prevNode || !_isNodePattern(prevNode)) continue;

        const srcVal =
          prevNode.variable !== null ? fields[prevNode.variable] : undefined;
        const tgtVal =
          nextNode.variable !== null ? fields[nextNode.variable] : undefined;
        const srcVid = refValue(srcVal);
        const tgtVid = refValue(tgtVal);
        if (srcVid === null || tgtVid === null) continue;

        const edge = this._createEdge(elem, srcVid, tgtVid, fields);
        createdEids.add(edge.edgeId);
        if (elem.variable !== null) {
          fields[elem.variable] = new _EdgeRef(edge.edgeId);
        }
      }
    }
  }

  private _createVertex(pat: NodePattern, fields: BindingFields): Vertex {
    const vid = this._graph.nextVertexId();
    const label = pat.labels.length > 0 ? pat.labels[0]! : "";
    const props: Record<string, unknown> = {};
    for (const [k, vExpr] of pat.properties) {
      props[k] = this._eval(vExpr, fields);
    }
    const vtx = createVertex(vid, label, props);
    this._graph.addVertex(vtx, this._graphName);
    return vtx;
  }

  private _createEdge(
    relPat: RelationshipPattern,
    srcVid: number,
    tgtVid: number,
    fields: BindingFields,
  ): Edge {
    const eid = this._graph.nextEdgeId();
    const label = relPat.types.length > 0 ? relPat.types[0]! : "";
    const props: Record<string, unknown> = {};
    for (const [k, vExpr] of relPat.properties) {
      props[k] = this._eval(vExpr, fields);
    }

    let actualSrc = srcVid;
    let actualTgt = tgtVid;
    if (relPat.direction === "in") {
      actualSrc = tgtVid;
      actualTgt = srcVid;
    }

    const edge = createEdge(eid, actualSrc, actualTgt, label, props);
    this._graph.addEdge(edge, this._graphName);
    return edge;
  }

  // -- MERGE ------------------------------------------------------------------

  private _execMerge(
    clause: MergeClause,
    bindings: GraphPostingList,
  ): GraphPostingList {
    const entries: PostingEntry[] = [];
    const payloads = new Map<number, GraphPayload>();

    for (const bindingEntry of bindings) {
      const fields: BindingFields = {
        ...(bindingEntry.payload.fields as Record<string, unknown>),
      };
      const matched = this._matchPath(clause.pattern, fields);

      if (matched.length > 0) {
        for (const mFields of matched) {
          if (clause.onMatch !== null) {
            for (const item of clause.onMatch.items) {
              this._applySetItem(item, mFields);
            }
          }
          const [entry, docId] = this._makeBindingEntry(
            mFields,
            new Set<number>(),
            new Set<number>(),
          );
          entries.push(entry);
          payloads.set(docId, createGraphPayload());
        }
      } else {
        const createdVids = new Set<number>();
        const createdEids = new Set<number>();
        this._createPath(clause.pattern, fields, createdVids, createdEids);
        if (clause.onCreate !== null) {
          for (const item of clause.onCreate.items) {
            this._applySetItem(item, fields);
          }
        }
        const [entry, docId] = this._makeBindingEntry(fields, createdVids, createdEids);
        entries.push(entry);
        payloads.set(
          docId,
          createGraphPayload({
            subgraphVertices: createdVids,
            subgraphEdges: createdEids,
          }),
        );
      }
    }

    return new GraphPostingList(entries, payloads);
  }

  // -- SET --------------------------------------------------------------------

  private _execSet(clause: SetClause, bindings: GraphPostingList): GraphPostingList {
    const entries: PostingEntry[] = [];
    const payloads = new Map<number, GraphPayload>();

    for (const bindingEntry of bindings) {
      const fields: BindingFields = {
        ...(bindingEntry.payload.fields as Record<string, unknown>),
      };
      for (const item of clause.items) {
        this._applySetItem(item, fields);
      }
      const [entry, docId] = this._makeBindingEntry(
        fields,
        new Set<number>(),
        new Set<number>(),
      );
      entries.push(entry);
      payloads.set(docId, createGraphPayload());
    }

    return new GraphPostingList(entries, payloads);
  }

  private _applySetItem(item: SetItem, fields: BindingFields): void {
    const value = this._eval(item.value, fields);
    const propAccess = item.property;

    // Resolve the object variable from PropertyAccessExpr
    // In TS AST: property_access has .object (CypherExpr) and .property (string)
    // The object is typically an identifier for the bound variable.
    const objExpr = propAccess.object;
    if (objExpr.kind !== "identifier") return;

    const varName = objExpr.name;
    const propKey = propAccess.property;
    const vidOrEid = fields[varName];
    if (vidOrEid === null || vidOrEid === undefined) return;

    // Try as vertex first (unless it is definitively an edge ref)
    if (!isEdgeRef(vidOrEid)) {
      const rawId = refValue(vidOrEid);
      if (rawId !== null) {
        const vtx = this._graph.getVertex(rawId);
        if (vtx !== null) {
          const newProps: Record<string, unknown> = { ...vtx.properties };
          newProps[propKey] = value;
          const newVtx = createVertex(vtx.vertexId, vtx.label, newProps);
          this._graph.addVertex(newVtx, this._graphName);
          return;
        }
      }
    }

    // Try as edge
    if (!isVertexRef(vidOrEid)) {
      const rawId = refValue(vidOrEid);
      if (rawId !== null) {
        const edge = this._graph.getEdge(rawId);
        if (edge !== null) {
          const newProps: Record<string, unknown> = { ...edge.properties };
          newProps[propKey] = value;
          const newEdge = createEdge(
            edge.edgeId,
            edge.sourceId,
            edge.targetId,
            edge.label,
            newProps,
          );
          this._graph.addEdge(newEdge, this._graphName);
          return;
        }
      }
    }
  }

  // -- DELETE -----------------------------------------------------------------

  private _execDelete(
    clause: DeleteClause,
    bindings: GraphPostingList,
  ): GraphPostingList {
    const toDeleteVertices: number[] = [];
    const toDeleteEdges: number[] = [];

    for (const bindingEntry of bindings) {
      const fields = bindingEntry.payload.fields as Record<string, unknown>;
      for (const expr of clause.exprs) {
        const val = this._eval(expr, fields);
        if (val === null || val === undefined) continue;

        if (isVertexRef(val)) {
          toDeleteVertices.push(val.value);
        } else if (isEdgeRef(val)) {
          toDeleteEdges.push(val.value);
        } else if (typeof val === "number") {
          if (this._graph.getVertex(val) !== null) {
            toDeleteVertices.push(val);
          } else if (this._graph.getEdge(val) !== null) {
            toDeleteEdges.push(val);
          }
        }
      }
    }

    for (const eid of toDeleteEdges) {
      this._graph.removeEdge(eid, this._graphName);
    }

    for (const vid of toDeleteVertices) {
      if (!clause.detach) {
        const hasOut = this._graph.outEdgeIds(vid, this._graphName).size > 0;
        const hasIn = this._graph.inEdgeIds(vid, this._graphName).size > 0;
        if (hasOut || hasIn) {
          throw new Error(
            `Cannot delete vertex ${String(vid)}: has incident edges. Use DETACH DELETE.`,
          );
        }
      }
      this._graph.removeVertex(vid, this._graphName);
    }

    return bindings;
  }

  // -- RETURN -----------------------------------------------------------------

  private _execReturnPostingList(
    clause: ReturnClause,
    bindings: GraphPostingList,
  ): GraphPostingList {
    const bindingList = [...bindings];

    // Check for aggregation functions
    const hasAggregation = clause.items.some((item) => _containsAggregation(item.expr));

    if (hasAggregation) {
      return this._execReturnWithAggregation(clause, bindingList);
    }

    // Sort using original binding fields before projection
    if (clause.orderBy !== null) {
      for (let oi = clause.orderBy.length - 1; oi >= 0; oi--) {
        const item = clause.orderBy[oi]!;
        bindingList.sort((a, b) => {
          const aVal = this._eval(
            item.expr,
            a.payload.fields as Record<string, unknown>,
          );
          const bVal = this._eval(
            item.expr,
            b.payload.fields as Record<string, unknown>,
          );
          const cmp = _compareSortKeys(_sortKey(aVal), _sortKey(bVal));
          return item.ascending ? cmp : -cmp;
        });
      }
    }

    // Project after sorting
    const entries: PostingEntry[] = [];
    const payloads = new Map<number, GraphPayload>();

    for (const bindingEntry of bindingList) {
      const fields = bindingEntry.payload.fields as Record<string, unknown>;
      const projected: BindingFields = {};
      for (const item of clause.items) {
        if (item.expr.kind === "identifier" && item.expr.name === "*") {
          for (const k of Object.keys(fields)) {
            projected[k] = this._toAgtype(fields[k]);
          }
          continue;
        }
        let val = this._eval(item.expr, fields);
        const key = item.alias ?? _exprName(item.expr);
        if (item.expr.kind === "identifier") {
          val = this._toAgtype(val);
        }
        projected[key] = val;
      }

      const [entry, docId] = this._makeBindingEntry(
        projected,
        new Set<number>(),
        new Set<number>(),
      );
      entries.push(entry);
      payloads.set(docId, createGraphPayload());
    }

    let gpl = new GraphPostingList(entries, payloads);

    if (clause.distinct) {
      gpl = this._distinctGpl(gpl);
    }

    if (clause.skip !== null) {
      const skipN = this._eval(clause.skip, {});
      gpl = this._sliceGpl(gpl, typeof skipN === "number" ? skipN : 0, null);
    }

    if (clause.limit !== null) {
      const limitN = this._eval(clause.limit, {});
      gpl = this._sliceGpl(gpl, 0, typeof limitN === "number" ? limitN : null);
    }

    return gpl;
  }

  private _execReturnWithAggregation(
    clause: ReturnClause,
    bindingList: PostingEntry[],
  ): GraphPostingList {
    // Determine grouping keys (non-aggregate expressions)
    // and aggregate expressions
    const groupKeyIndices: number[] = [];
    const aggIndices: number[] = [];

    for (let i = 0; i < clause.items.length; i++) {
      const item = clause.items[i]!;
      if (_containsAggregation(item.expr)) {
        aggIndices.push(i);
      } else {
        groupKeyIndices.push(i);
      }
    }

    // Group rows by non-aggregate values
    const groups = new Map<string, PostingEntry[]>();
    const groupOrder: string[] = [];

    for (const bindingEntry of bindingList) {
      const fields = bindingEntry.payload.fields as Record<string, unknown>;
      const keyParts: string[] = [];
      for (const idx of groupKeyIndices) {
        const item = clause.items[idx]!;
        const val = this._eval(item.expr, fields);
        keyParts.push(_stableStringify(val));
      }
      const groupKey = keyParts.join("\0");
      const existing = groups.get(groupKey);
      if (existing !== undefined) {
        existing.push(bindingEntry);
      } else {
        groups.set(groupKey, [bindingEntry]);
        groupOrder.push(groupKey);
      }
    }

    // If there are no grouping keys and no rows, produce a single row with default agg values
    if (groupKeyIndices.length === 0 && bindingList.length === 0) {
      groupOrder.push("");
      groups.set("", []);
    }

    // Produce one output row per group
    const entries: PostingEntry[] = [];
    const payloads = new Map<number, GraphPayload>();

    for (const groupKey of groupOrder) {
      const groupEntries = groups.get(groupKey) ?? [];
      const projected: BindingFields = {};

      // Evaluate non-aggregate items from first entry in group (or empty fields)
      const sampleFields =
        groupEntries.length > 0
          ? (groupEntries[0]!.payload.fields as Record<string, unknown>)
          : {};

      for (const idx of groupKeyIndices) {
        const item = clause.items[idx]!;
        let val = this._eval(item.expr, sampleFields);
        const key = item.alias ?? _exprName(item.expr);
        if (item.expr.kind === "identifier") {
          val = this._toAgtype(val);
        }
        projected[key] = val;
      }

      // Evaluate aggregate items
      for (const idx of aggIndices) {
        const item = clause.items[idx]!;
        const key = item.alias ?? _exprName(item.expr);
        projected[key] = this._evalAggregate(
          item.expr,
          groupEntries.map((e) => e.payload.fields as Record<string, unknown>),
        );
      }

      const [entry, docId] = this._makeBindingEntry(
        projected,
        new Set<number>(),
        new Set<number>(),
      );
      entries.push(entry);
      payloads.set(docId, createGraphPayload());
    }

    let gpl = new GraphPostingList(entries, payloads);

    if (clause.distinct) {
      gpl = this._distinctGpl(gpl);
    }

    if (clause.orderBy !== null) {
      gpl = this._orderByGpl(clause.orderBy, gpl);
    }

    if (clause.skip !== null) {
      const skipN = this._eval(clause.skip, {});
      gpl = this._sliceGpl(gpl, typeof skipN === "number" ? skipN : 0, null);
    }

    if (clause.limit !== null) {
      const limitN = this._eval(clause.limit, {});
      gpl = this._sliceGpl(gpl, 0, typeof limitN === "number" ? limitN : null);
    }

    return gpl;
  }

  private _evalAggregate(
    expr: CypherExpr,
    groupFields: Array<Record<string, unknown>>,
  ): unknown {
    if (expr.kind === "function_call") {
      const fc = expr;
      const name = fc.name.toLowerCase();

      if (name === "count") {
        if (fc.args.length === 0) return groupFields.length;
        const arg = fc.args[0]!;
        if (arg.kind === "identifier" && arg.name === "*") {
          return groupFields.length;
        }
        let count = 0;
        for (const f of groupFields) {
          const v = this._eval(arg, f);
          if (v !== null && v !== undefined) count++;
        }
        return count;
      }

      if (name === "sum") {
        const arg = fc.args[0]!;
        let total = 0;
        for (const f of groupFields) {
          const v = this._eval(arg, f);
          if (typeof v === "number") total += v;
        }
        return total;
      }

      if (name === "avg") {
        const arg = fc.args[0]!;
        let total = 0;
        let count = 0;
        for (const f of groupFields) {
          const v = this._eval(arg, f);
          if (typeof v === "number") {
            total += v;
            count++;
          }
        }
        return count > 0 ? total / count : null;
      }

      if (name === "min") {
        const arg = fc.args[0]!;
        let best: unknown = null;
        for (const f of groupFields) {
          const v = this._eval(arg, f);
          if (v === null || v === undefined) continue;
          if (best === null) {
            best = v;
          } else {
            const cmp = _compareSortKeys(_sortKey(v), _sortKey(best));
            if (cmp < 0) best = v;
          }
        }
        return best;
      }

      if (name === "max") {
        const arg = fc.args[0]!;
        let best: unknown = null;
        for (const f of groupFields) {
          const v = this._eval(arg, f);
          if (v === null || v === undefined) continue;
          if (best === null) {
            best = v;
          } else {
            const cmp = _compareSortKeys(_sortKey(v), _sortKey(best));
            if (cmp > 0) best = v;
          }
        }
        return best;
      }

      if (name === "collect") {
        const arg = fc.args[0]!;
        const collected: unknown[] = [];
        for (const f of groupFields) {
          const v = this._eval(arg, f);
          if (v !== null && v !== undefined) {
            collected.push(v);
          }
        }
        return collected;
      }
    }

    // Not an aggregate at the top level; evaluate against sample
    if (groupFields.length > 0) {
      return this._eval(expr, groupFields[0]!);
    }
    return null;
  }

  // -- WITH -------------------------------------------------------------------

  private _execWith(clause: WithClause, bindings: GraphPostingList): GraphPostingList {
    // Check for aggregation
    const hasAggregation = clause.items.some((item) => _containsAggregation(item.expr));

    if (hasAggregation) {
      return this._execWithAggregation(clause, bindings);
    }

    const entries: PostingEntry[] = [];
    const payloads = new Map<number, GraphPayload>();

    for (const bindingEntry of bindings) {
      const oldFields = bindingEntry.payload.fields as Record<string, unknown>;
      const newFields: BindingFields = {};
      for (const item of clause.items) {
        if (item.expr.kind === "identifier" && item.expr.name === "*") {
          Object.assign(newFields, oldFields);
          continue;
        }
        const val = this._eval(item.expr, oldFields);
        const key = item.alias ?? _exprName(item.expr);
        newFields[key] = val;
      }

      const [entry, docId] = this._makeBindingEntry(
        newFields,
        new Set<number>(),
        new Set<number>(),
      );
      entries.push(entry);
      payloads.set(docId, createGraphPayload());
    }

    let gpl = new GraphPostingList(entries, payloads);

    if (clause.distinct) {
      gpl = this._distinctGpl(gpl);
    }

    if (clause.orderBy !== null) {
      gpl = this._orderByGpl(clause.orderBy, gpl);
    }

    if (clause.skip !== null) {
      const skipN = this._eval(clause.skip, {});
      gpl = this._sliceGpl(gpl, typeof skipN === "number" ? skipN : 0, null);
    }

    if (clause.limit !== null) {
      const limitN = this._eval(clause.limit, {});
      gpl = this._sliceGpl(gpl, 0, typeof limitN === "number" ? limitN : null);
    }

    if (clause.where !== null) {
      const entriesOut: PostingEntry[] = [];
      const payloadsOut = new Map<number, GraphPayload>();
      for (const e of gpl) {
        if (this._eval(clause.where, e.payload.fields as Record<string, unknown>)) {
          entriesOut.push(e);
          const gp = gpl.getGraphPayload(e.docId);
          if (gp !== null) {
            payloadsOut.set(e.docId, gp);
          }
        }
      }
      gpl = new GraphPostingList(entriesOut, payloadsOut);
    }

    return gpl;
  }

  private _execWithAggregation(
    clause: WithClause,
    bindings: GraphPostingList,
  ): GraphPostingList {
    const bindingList = [...bindings];

    const groupKeyIndices: number[] = [];
    const aggIndices: number[] = [];

    for (let i = 0; i < clause.items.length; i++) {
      const item = clause.items[i]!;
      if (_containsAggregation(item.expr)) {
        aggIndices.push(i);
      } else {
        groupKeyIndices.push(i);
      }
    }

    const groups = new Map<string, PostingEntry[]>();
    const groupOrder: string[] = [];

    for (const bindingEntry of bindingList) {
      const fields = bindingEntry.payload.fields as Record<string, unknown>;
      const keyParts: string[] = [];
      for (const idx of groupKeyIndices) {
        const item = clause.items[idx]!;
        const val = this._eval(item.expr, fields);
        keyParts.push(_stableStringify(val));
      }
      const groupKey = keyParts.join("\0");
      const existing = groups.get(groupKey);
      if (existing !== undefined) {
        existing.push(bindingEntry);
      } else {
        groups.set(groupKey, [bindingEntry]);
        groupOrder.push(groupKey);
      }
    }

    if (groupKeyIndices.length === 0 && bindingList.length === 0) {
      groupOrder.push("");
      groups.set("", []);
    }

    const entries: PostingEntry[] = [];
    const payloads = new Map<number, GraphPayload>();

    for (const groupKey of groupOrder) {
      const groupEntries = groups.get(groupKey) ?? [];
      const newFields: BindingFields = {};

      const sampleFields =
        groupEntries.length > 0
          ? (groupEntries[0]!.payload.fields as Record<string, unknown>)
          : {};

      for (const idx of groupKeyIndices) {
        const item = clause.items[idx]!;
        const val = this._eval(item.expr, sampleFields);
        const key = item.alias ?? _exprName(item.expr);
        newFields[key] = val;
      }

      for (const idx of aggIndices) {
        const item = clause.items[idx]!;
        const key = item.alias ?? _exprName(item.expr);
        newFields[key] = this._evalAggregate(
          item.expr,
          groupEntries.map((e) => e.payload.fields as Record<string, unknown>),
        );
      }

      const [entry, docId] = this._makeBindingEntry(
        newFields,
        new Set<number>(),
        new Set<number>(),
      );
      entries.push(entry);
      payloads.set(docId, createGraphPayload());
    }

    let gpl = new GraphPostingList(entries, payloads);

    if (clause.distinct) {
      gpl = this._distinctGpl(gpl);
    }

    if (clause.orderBy !== null) {
      gpl = this._orderByGpl(clause.orderBy, gpl);
    }

    if (clause.skip !== null) {
      const skipN = this._eval(clause.skip, {});
      gpl = this._sliceGpl(gpl, typeof skipN === "number" ? skipN : 0, null);
    }

    if (clause.limit !== null) {
      const limitN = this._eval(clause.limit, {});
      gpl = this._sliceGpl(gpl, 0, typeof limitN === "number" ? limitN : null);
    }

    if (clause.where !== null) {
      const entriesOut: PostingEntry[] = [];
      const payloadsOut = new Map<number, GraphPayload>();
      for (const e of gpl) {
        if (this._eval(clause.where, e.payload.fields as Record<string, unknown>)) {
          entriesOut.push(e);
          const gp = gpl.getGraphPayload(e.docId);
          if (gp !== null) {
            payloadsOut.set(e.docId, gp);
          }
        }
      }
      gpl = new GraphPostingList(entriesOut, payloadsOut);
    }

    return gpl;
  }

  // -- UNWIND -----------------------------------------------------------------

  private _execUnwind(
    clause: UnwindClause,
    bindings: GraphPostingList,
  ): GraphPostingList {
    const entries: PostingEntry[] = [];
    const payloads = new Map<number, GraphPayload>();

    for (const bindingEntry of bindings) {
      const fields = bindingEntry.payload.fields as Record<string, unknown>;
      const collection = this._eval(clause.expr, fields);
      if (collection === null || collection === undefined) continue;
      if (!Array.isArray(collection)) continue;

      for (const item of collection) {
        const newFields: BindingFields = { ...fields };
        newFields[clause.alias] = item;
        const [entry, docId] = this._makeBindingEntry(
          newFields,
          new Set<number>(),
          new Set<number>(),
        );
        entries.push(entry);
        payloads.set(docId, createGraphPayload());
      }
    }

    return new GraphPostingList(entries, payloads);
  }

  // -- AGType conversion ------------------------------------------------------

  private _toAgtype(val: unknown): unknown {
    if (val === null || val === undefined) return null;
    if (typeof val === "boolean") return val;

    if (isVertexRef(val)) {
      const vtx = this._graph.getVertex(val.value);
      if (vtx !== null) {
        return {
          id: vtx.vertexId,
          label: vtx.label,
          properties: { ...vtx.properties },
        };
      }
      return val.value;
    }

    if (isEdgeRef(val)) {
      const edge = this._graph.getEdge(val.value);
      if (edge !== null) {
        return {
          id: edge.edgeId,
          label: edge.label,
          start: edge.sourceId,
          end: edge.targetId,
          properties: { ...edge.properties },
        };
      }
      return val.value;
    }

    if (typeof val === "number") {
      const vtx = this._graph.getVertex(val);
      if (vtx !== null) {
        return {
          id: vtx.vertexId,
          label: vtx.label,
          properties: { ...vtx.properties },
        };
      }
      const edge = this._graph.getEdge(val);
      if (edge !== null) {
        return {
          id: edge.edgeId,
          label: edge.label,
          start: edge.sourceId,
          end: edge.targetId,
          properties: { ...edge.properties },
        };
      }
    }

    if (Array.isArray(val)) {
      return val.map((v) => this._toAgtype(v));
    }

    return val;
  }

  // -- Posting list utilities -------------------------------------------------

  private _distinctGpl(gpl: GraphPostingList): GraphPostingList {
    const seen = new Set<string>();
    const entries: PostingEntry[] = [];
    const payloads = new Map<number, GraphPayload>();
    for (const e of gpl) {
      const fields = e.payload.fields as Record<string, unknown>;
      const keyParts: Array<[string, string]> = [];
      for (const k of Object.keys(fields).sort()) {
        keyParts.push([k, _stableStringify(fields[k])]);
      }
      const key = JSON.stringify(keyParts);
      if (!seen.has(key)) {
        seen.add(key);
        entries.push(e);
        const gp = gpl.getGraphPayload(e.docId);
        if (gp !== null) {
          payloads.set(e.docId, gp);
        }
      }
    }
    return new GraphPostingList(entries, payloads);
  }

  private _orderByGpl(
    orderItems: readonly OrderByItem[],
    gpl: GraphPostingList,
  ): GraphPostingList {
    const entryList = [...gpl];
    for (let i = orderItems.length - 1; i >= 0; i--) {
      const item = orderItems[i]!;
      entryList.sort((a, b) => {
        const aVal = this._eval(item.expr, a.payload.fields as Record<string, unknown>);
        const bVal = this._eval(item.expr, b.payload.fields as Record<string, unknown>);
        const cmp = _compareSortKeys(_sortKey(aVal), _sortKey(bVal));
        return item.ascending ? cmp : -cmp;
      });
    }
    const payloads = new Map<number, GraphPayload>();
    for (const e of entryList) {
      const gp = gpl.getGraphPayload(e.docId);
      if (gp !== null) {
        payloads.set(e.docId, gp);
      }
    }
    return new GraphPostingList(entryList, payloads);
  }

  private _sliceGpl(
    gpl: GraphPostingList,
    start: number,
    end: number | null,
  ): GraphPostingList {
    const entryList = [...gpl];
    const sliced =
      end !== null ? entryList.slice(start, start + end) : entryList.slice(start);
    const payloads = new Map<number, GraphPayload>();
    for (const e of sliced) {
      const gp = gpl.getGraphPayload(e.docId);
      if (gp !== null) {
        payloads.set(e.docId, gp);
      }
    }
    return new GraphPostingList(sliced, payloads);
  }

  // -- Expression evaluator ---------------------------------------------------

  private _eval(expr: CypherExpr, fields: Record<string, unknown>): unknown {
    switch (expr.kind) {
      case "literal":
        return expr.value;

      case "identifier":
        return fields[expr.name];

      case "parameter":
        return this._params[expr.name];

      case "property_access":
        return this._evalPropertyAccess(expr, fields);

      case "binary_op":
        return this._evalBinary(expr, fields);

      case "unary_op":
        return this._evalUnary(expr, fields);

      case "function_call":
        return this._evalFunction(expr, fields);

      case "in":
        return this._evalIn(expr, fields);

      case "is_null":
        return this._evalIsNull(expr, fields);

      case "list":
        return expr.elements.map((e) => this._eval(e, fields));

      case "case":
        return this._evalCase(expr, fields);

      case "exists":
        // exists() requires pattern matching - simplified support
        return false;

      case "index": {
        const collection = this._eval(expr.value, fields);
        const idx = this._eval(expr.index, fields);
        if (Array.isArray(collection) && typeof idx === "number") {
          return collection[idx];
        }
        return null;
      }

      default: {
        const _exhaustive: never = expr;
        throw new Error(
          `Cannot evaluate expression kind: ${String((_exhaustive as CypherExpr).kind)}`,
        );
      }
    }
  }

  private _evalPropertyAccess(
    expr: PropertyAccessExpr,
    fields: Record<string, unknown>,
  ): unknown {
    let val = this._eval(expr.object, fields);
    if (val === null || val === undefined) return null;

    // Resolve through graph objects if val is a ref or ID
    if (isVertexRef(val)) {
      const obj = this._graph.getVertex(val.value);
      if (obj !== null) {
        val = obj as unknown;
      }
    } else if (isEdgeRef(val)) {
      const obj = this._graph.getEdge(val.value);
      if (obj !== null) {
        val = obj as unknown;
      }
    } else if (typeof val === "number") {
      const vtxObj = this._graph.getVertex(val);
      if (vtxObj !== null) {
        val = vtxObj as unknown;
      } else {
        const edgeObj = this._graph.getEdge(val);
        if (edgeObj !== null) {
          val = edgeObj as unknown;
        }
      }
    }

    // Now access the property
    const key = expr.property;
    if (_isVertexLike(val)) {
      return val.properties[key] ?? null;
    }
    if (_isEdgeLike(val)) {
      return val.properties[key] ?? null;
    }
    if (typeof val === "object" && val !== null && !Array.isArray(val)) {
      const obj = val as Record<string, unknown>;
      return obj[key] ?? null;
    }
    return null;
  }

  private _evalBinary(expr: BinaryOpExpr, fields: Record<string, unknown>): unknown {
    const op = expr.op;

    // Short-circuit for AND/OR
    if (op === "AND") {
      const left = this._eval(expr.left, fields);
      if (!left) return false;
      return Boolean(this._eval(expr.right, fields));
    }
    if (op === "OR") {
      const left = this._eval(expr.left, fields);
      if (left) return true;
      return Boolean(this._eval(expr.right, fields));
    }
    if (op === "XOR") {
      return (
        Boolean(this._eval(expr.left, fields)) !==
        Boolean(this._eval(expr.right, fields))
      );
    }

    const left = this._eval(expr.left, fields);
    const right = this._eval(expr.right, fields);

    // NULL propagation
    if (left === null || left === undefined || right === null || right === undefined) {
      if (op === "=") {
        return (
          (left === null || left === undefined) &&
          (right === null || right === undefined)
        );
      }
      if (op === "<>") {
        return !(
          (left === null || left === undefined) &&
          (right === null || right === undefined)
        );
      }
      return null;
    }

    switch (op) {
      case "=":
        return _cypherEquals(left, right);
      case "<>":
        return !_cypherEquals(left, right);
      case "<":
        return _cypherCompare(left, right) < 0;
      case ">":
        return _cypherCompare(left, right) > 0;
      case "<=":
        return _cypherCompare(left, right) <= 0;
      case ">=":
        return _cypherCompare(left, right) >= 0;
      case "+":
        if (typeof left === "string" || typeof right === "string") {
          return String(left) + String(right);
        }
        if (Array.isArray(left)) {
          return [...left, ...(Array.isArray(right) ? right : [right])];
        }
        return (left as number) + (right as number);
      case "-":
        return (left as number) - (right as number);
      case "*":
        return (left as number) * (right as number);
      case "/":
        if (right === 0) return null;
        if (typeof left === "number" && typeof right === "number") {
          if (Number.isInteger(left) && Number.isInteger(right)) {
            return Math.trunc(left / right);
          }
          return left / right;
        }
        return null;
      case "%":
        return (left as number) % (right as number);
      case "^":
        return Math.pow(left as number, right as number);
      case "STARTS WITH":
        return String(left).startsWith(String(right));
      case "ENDS WITH":
        return String(left).endsWith(String(right));
      case "CONTAINS":
        return String(left).includes(String(right));
      default:
        throw new Error(`Unknown binary operator: ${op}`);
    }
  }

  private _evalUnary(expr: UnaryOpExpr, fields: Record<string, unknown>): unknown {
    const val = this._eval(expr.operand, fields);
    if (expr.op === "NOT") return !val;
    if (expr.op === "-") {
      return val !== null && val !== undefined ? -(val as number) : null;
    }
    throw new Error(`Unknown unary operator: ${expr.op}`);
  }

  private _evalFunction(
    expr: FunctionCallExpr,
    fields: Record<string, unknown>,
  ): unknown {
    const name = expr.name.toLowerCase();
    const args = expr.args;

    if (name === "id") {
      const argExpr = args[0];
      if (!argExpr) return null;
      const val =
        argExpr.kind === "identifier"
          ? fields[argExpr.name]
          : this._eval(argExpr, fields);
      const id = refValue(val);
      return id !== null ? id : typeof val === "number" ? val : null;
    }

    if (name === "type") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      const id = refValue(val);
      if (id !== null) {
        const edge = this._graph.getEdge(id);
        if (edge !== null) return edge.label;
      }
      return null;
    }

    if (name === "labels") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      const id = refValue(val);
      if (id !== null) {
        const vtx = this._graph.getVertex(id);
        if (vtx !== null) return vtx.label ? [vtx.label] : [];
      }
      return null;
    }

    if (name === "label") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      const id = refValue(val);
      if (id !== null) {
        const vtx = this._graph.getVertex(id);
        if (vtx !== null) return vtx.label;
      }
      return null;
    }

    if (name === "properties") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      const id = refValue(val);
      if (id !== null) {
        const vtx = this._graph.getVertex(id);
        if (vtx !== null) return { ...vtx.properties };
        const edge = this._graph.getEdge(id);
        if (edge !== null) return { ...edge.properties };
      }
      return null;
    }

    if (name === "keys") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      const id = refValue(val);
      if (id !== null) {
        const vtx = this._graph.getVertex(id);
        if (vtx !== null) return Object.keys(vtx.properties);
        const edge = this._graph.getEdge(id);
        if (edge !== null) return Object.keys(edge.properties);
      }
      if (typeof val === "object" && val !== null && !Array.isArray(val)) {
        return Object.keys(val as Record<string, unknown>);
      }
      return null;
    }

    if (name === "tostring" || name === "tostr") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      return val !== null && val !== undefined ? String(val) : null;
    }

    if (name === "tointeger" || name === "toint") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      if (val === null || val === undefined) return null;
      const n = parseInt(String(val), 10);
      return Number.isNaN(n) ? null : n;
    }

    if (name === "tofloat") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      if (val === null || val === undefined) return null;
      const n = parseFloat(String(val));
      return Number.isNaN(n) ? null : n;
    }

    if (name === "size" || name === "length") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      if (val === null || val === undefined) return null;
      if (typeof val === "string") return val.length;
      if (Array.isArray(val)) return val.length;
      return null;
    }

    if (name === "head") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      if (Array.isArray(val) && val.length > 0) return val[0];
      return null;
    }

    if (name === "last") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      if (Array.isArray(val) && val.length > 0) return val[val.length - 1];
      return null;
    }

    if (name === "tail") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      if (Array.isArray(val)) return val.slice(1);
      return [];
    }

    if (name === "reverse") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      if (val === null || val === undefined) return null;
      if (typeof val === "string") return val.split("").reverse().join("");
      if (Array.isArray(val)) return [...val].reverse();
      return null;
    }

    if (name === "range") {
      const start = args[0] ? this._eval(args[0], fields) : 0;
      const end = args[1] ? this._eval(args[1], fields) : 0;
      const step = args[2] ? this._eval(args[2], fields) : 1;
      if (
        typeof start !== "number" ||
        typeof end !== "number" ||
        typeof step !== "number" ||
        step === 0
      ) {
        return [];
      }
      const result: number[] = [];
      if (step > 0) {
        for (let i = start; i <= end; i += step) result.push(i);
      } else {
        for (let i = start; i >= end; i += step) result.push(i);
      }
      return result;
    }

    if (name === "abs") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      return typeof val === "number" ? Math.abs(val) : null;
    }

    if (name === "coalesce") {
      for (const a of args) {
        const val = this._eval(a, fields);
        if (val !== null && val !== undefined) return val;
      }
      return null;
    }

    if (name === "tolower" || name === "lower") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      return typeof val === "string" ? val.toLowerCase() : null;
    }

    if (name === "toupper" || name === "upper") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      return typeof val === "string" ? val.toUpperCase() : null;
    }

    if (name === "trim") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      return typeof val === "string" ? val.trim() : null;
    }

    if (name === "replace") {
      const s = args[0] ? this._eval(args[0], fields) : null;
      const oldStr = args[1] ? this._eval(args[1], fields) : null;
      const newStr = args[2] ? this._eval(args[2], fields) : null;
      if (
        typeof s === "string" &&
        typeof oldStr === "string" &&
        typeof newStr === "string"
      ) {
        return s.split(oldStr).join(newStr);
      }
      return null;
    }

    if (name === "substring") {
      const s = args[0] ? this._eval(args[0], fields) : null;
      const startIdx = args[1] ? this._eval(args[1], fields) : null;
      if (typeof s !== "string" || typeof startIdx !== "number") return null;
      if (args[2]) {
        const length = this._eval(args[2], fields);
        if (typeof length === "number") {
          return s.substring(startIdx, startIdx + length);
        }
      }
      return s.substring(startIdx);
    }

    if (name === "split") {
      const s = args[0] ? this._eval(args[0], fields) : null;
      const delim = args[1] ? this._eval(args[1], fields) : null;
      if (typeof s === "string" && typeof delim === "string") {
        return s.split(delim);
      }
      return null;
    }

    if (name === "startnode") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      const id = refValue(val);
      if (id !== null) {
        const edge = this._graph.getEdge(id);
        if (edge !== null) return edge.sourceId;
      }
      return null;
    }

    if (name === "endnode") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      const id = refValue(val);
      if (id !== null) {
        const edge = this._graph.getEdge(id);
        if (edge !== null) return edge.targetId;
      }
      return null;
    }

    // Aggregation functions in scalar context
    if (name === "count") {
      return 1;
    }

    if (name === "collect") {
      const val = args[0] ? this._eval(args[0], fields) : null;
      return [val];
    }

    if (name === "sum" || name === "avg" || name === "min" || name === "max") {
      // In scalar context, just return the value
      const val = args[0] ? this._eval(args[0], fields) : null;
      return val;
    }

    throw new Error(`Unknown function: ${expr.name}`);
  }

  private _evalIn(expr: InExpr, fields: Record<string, unknown>): unknown {
    const val = this._eval(expr.value, fields);
    const lst = this._eval(expr.list, fields);
    if (lst === null || lst === undefined) return null;
    if (!Array.isArray(lst)) return false;
    return lst.some((item) => _cypherEquals(item, val));
  }

  private _evalIsNull(expr: IsNullExpr, fields: Record<string, unknown>): boolean {
    const val = this._eval(expr.value, fields);
    const isNull = val === null || val === undefined;
    return expr.negated ? !isNull : isNull;
  }

  private _evalCase(expr: CaseExpr, fields: Record<string, unknown>): unknown {
    if (expr.operand !== null) {
      const val = this._eval(expr.operand, fields);
      for (const w of expr.whens) {
        if (_cypherEquals(val, this._eval(w.when, fields))) {
          return this._eval(w.then, fields);
        }
      }
    } else {
      for (const w of expr.whens) {
        if (this._eval(w.when, fields)) {
          return this._eval(w.then, fields);
        }
      }
    }
    if (expr.elseExpr !== null) {
      return this._eval(expr.elseExpr, fields);
    }
    return null;
  }
}

// -- Module-level helpers -----------------------------------------------------

function _isNodePattern(elem: NodePattern | RelationshipPattern): elem is NodePattern {
  return "labels" in elem && !("types" in elem);
}

function _isRelPattern(
  elem: NodePattern | RelationshipPattern,
): elem is RelationshipPattern {
  return "types" in elem;
}

function _isVertexLike(val: unknown): val is Vertex {
  if (typeof val !== "object" || val === null) return false;
  return "vertexId" in val && "label" in val && "properties" in val;
}

function _isEdgeLike(val: unknown): val is Edge {
  if (typeof val !== "object" || val === null) return false;
  return "edgeId" in val && "sourceId" in val && "targetId" in val;
}

function _exprName(expr: CypherExpr): string {
  switch (expr.kind) {
    case "identifier":
      return expr.name;
    case "property_access": {
      const pa = expr;
      const objName = _exprName(pa.object);
      return `${objName}.${pa.property}`;
    }
    case "function_call":
      return expr.name;
    default:
      return String(expr);
  }
}

function _sortKey(val: unknown): [number, unknown] {
  if (val === null || val === undefined) return [0, null];
  if (typeof val === "boolean") return [1, val ? 1 : 0];
  if (isVertexRef(val)) return [2, val.value];
  if (isEdgeRef(val)) return [2, val.value];
  if (typeof val === "number") return [2, val];
  if (typeof val === "string") return [3, val];
  return [4, String(val)];
}

function _compareSortKeys(a: [number, unknown], b: [number, unknown]): number {
  if (a[0] !== b[0]) return a[0] - b[0];
  const av = a[1];
  const bv = b[1];
  if (av === bv) return 0;
  if (av === null) return -1;
  if (bv === null) return 1;
  if (typeof av === "number" && typeof bv === "number") {
    return av < bv ? -1 : av > bv ? 1 : 0;
  }
  if (typeof av === "string" && typeof bv === "string") {
    return av < bv ? -1 : av > bv ? 1 : 0;
  }
  return 0;
}

function _cypherEquals(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a === null || a === undefined) return b === null || b === undefined;
  if (b === null || b === undefined) return false;
  if (isVertexRef(a))
    return isVertexRef(b)
      ? a.value === b.value
      : typeof b === "number" && a.value === b;
  if (isEdgeRef(a))
    return isEdgeRef(b) ? a.value === b.value : typeof b === "number" && a.value === b;
  if (typeof a === "number" && isVertexRef(b)) return a === b.value;
  if (typeof a === "number" && isEdgeRef(b)) return a === b.value;
  return a === b;
}

function _cypherCompare(a: unknown, b: unknown): number {
  if (typeof a === "number" && typeof b === "number") {
    return a < b ? -1 : a > b ? 1 : 0;
  }
  if (typeof a === "string" && typeof b === "string") {
    return a < b ? -1 : a > b ? 1 : 0;
  }
  // Fall back to string comparison
  const sa = String(a);
  const sb = String(b);
  return sa < sb ? -1 : sa > sb ? 1 : 0;
}

function _containsAggregation(expr: CypherExpr): boolean {
  if (expr.kind === "function_call") {
    const fc = expr;
    const name = fc.name.toLowerCase();
    if (
      name === "count" ||
      name === "sum" ||
      name === "avg" ||
      name === "min" ||
      name === "max" ||
      name === "collect"
    ) {
      return true;
    }
    // Check nested args
    for (const arg of fc.args) {
      if (_containsAggregation(arg)) return true;
    }
  }
  if (expr.kind === "binary_op") {
    const bo = expr;
    return _containsAggregation(bo.left) || _containsAggregation(bo.right);
  }
  if (expr.kind === "unary_op") {
    return _containsAggregation(expr.operand);
  }
  return false;
}

function _stableStringify(val: unknown): string {
  if (val === null || val === undefined) return "null";
  if (isVertexRef(val)) return `vref:${String(val.value)}`;
  if (isEdgeRef(val)) return `eref:${String(val.value)}`;
  if (typeof val === "object") {
    return JSON.stringify(val);
  }
  return String(val);
}
