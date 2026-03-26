//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Graph delta tracking
// 1:1 port of uqa/graph/delta.py

import type { Vertex, Edge } from "../core/types.js";

// -- DeltaOp ------------------------------------------------------------------

export type DeltaKind = "add_vertex" | "remove_vertex" | "add_edge" | "remove_edge";

export interface DeltaOp {
  readonly kind: DeltaKind;
  readonly vertex?: Vertex;
  readonly edge?: Edge;
  readonly vertexId?: number;
  readonly edgeId?: number;
}

export function createDeltaOp(
  kind: DeltaKind,
  opts?: {
    vertex?: Vertex;
    edge?: Edge;
    vertexId?: number;
    edgeId?: number;
  },
): DeltaOp {
  return {
    kind,
    vertex: opts?.vertex,
    edge: opts?.edge,
    vertexId: opts?.vertexId,
    edgeId: opts?.edgeId,
  };
}

// -- GraphDelta ---------------------------------------------------------------

export class GraphDelta {
  private _ops: DeltaOp[] = [];
  private _affectedVertices: Set<number> = new Set();
  private _affectedLabels: Set<string> = new Set();

  get ops(): ReadonlyArray<DeltaOp> {
    return this._ops;
  }

  get affectedVertices(): ReadonlySet<number> {
    return this._affectedVertices;
  }

  get affectedLabels(): ReadonlySet<string> {
    return this._affectedLabels;
  }

  get isEmpty(): boolean {
    return this._ops.length === 0;
  }

  addVertex(vertex: Vertex): void {
    this._ops.push(createDeltaOp("add_vertex", { vertex }));
    this._affectedVertices.add(vertex.vertexId);
    this._affectedLabels.add(vertex.label);
  }

  removeVertex(vertexId: number, label?: string): void {
    this._ops.push(createDeltaOp("remove_vertex", { vertexId }));
    this._affectedVertices.add(vertexId);
    if (label) {
      this._affectedLabels.add(label);
    }
  }

  addEdge(edge: Edge): void {
    this._ops.push(createDeltaOp("add_edge", { edge }));
    this._affectedVertices.add(edge.sourceId);
    this._affectedVertices.add(edge.targetId);
    this._affectedLabels.add(edge.label);
  }

  removeEdge(edgeId: number, edge?: Edge): void {
    this._ops.push(createDeltaOp("remove_edge", { edgeId, edge }));
    if (edge) {
      this._affectedVertices.add(edge.sourceId);
      this._affectedVertices.add(edge.targetId);
      this._affectedLabels.add(edge.label);
    }
  }

  clear(): void {
    this._ops = [];
    this._affectedVertices = new Set();
    this._affectedLabels = new Set();
  }

  merge(other: GraphDelta): GraphDelta {
    const merged = new GraphDelta();
    for (const op of this._ops) {
      merged._ops.push(op);
    }
    for (const op of other._ops) {
      merged._ops.push(op);
    }
    for (const vid of this._affectedVertices) {
      merged._affectedVertices.add(vid);
    }
    for (const vid of other._affectedVertices) {
      merged._affectedVertices.add(vid);
    }
    for (const label of this._affectedLabels) {
      merged._affectedLabels.add(label);
    }
    for (const label of other._affectedLabels) {
      merged._affectedLabels.add(label);
    }
    return merged;
  }
}
