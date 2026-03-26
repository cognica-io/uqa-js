//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- join graph data structures
// 1:1 port of uqa/planner/join_graph.py

import type { Operator } from "../operators/base.js";
import type { ColumnStats } from "../sql/table.js";

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

export interface JoinEdge {
  leftNode: number;
  rightNode: number;
  leftField: string;
  rightField: string;
  selectivity: number;
}

export interface JoinNode {
  index: number;
  alias: string;
  operator: Operator | null;
  table: unknown;
  cardinality: number;
  columnStats: Map<string, ColumnStats> | null;
}

export interface JoinPlan {
  relations: Set<number>;
  cardinality: number;
  cost: number;
  left: JoinPlan | null;
  right: JoinPlan | null;
  joinEdge: JoinEdge | null;
}

// ---------------------------------------------------------------------------
// JoinGraph
// ---------------------------------------------------------------------------

export class JoinGraph {
  private _nodes: JoinNode[] = [];
  private _edges: JoinEdge[] = [];
  private _adjacency: Map<number, Set<number>> = new Map();

  /**
   * Add a relation node to the graph.
   * @returns The index of the new node.
   */
  addNode(
    alias: string,
    operator: Operator | null,
    table: unknown,
    cardinality: number,
    columnStats?: Map<string, ColumnStats> | null,
  ): number {
    const index = this._nodes.length;
    this._nodes.push({
      index,
      alias,
      operator,
      table,
      cardinality,
      columnStats: columnStats ?? null,
    });
    this._adjacency.set(index, new Set());
    return index;
  }

  /**
   * Add a join edge between two nodes.
   * @param selectivity  Fraction of the cross product retained (default 0.1).
   */
  addEdge(
    leftNode: number,
    rightNode: number,
    leftField: string,
    rightField: string,
    selectivity = 0.1,
  ): void {
    const edge: JoinEdge = { leftNode, rightNode, leftField, rightField, selectivity };
    this._edges.push(edge);

    let leftAdj = this._adjacency.get(leftNode);
    if (leftAdj === undefined) {
      leftAdj = new Set();
      this._adjacency.set(leftNode, leftAdj);
    }
    leftAdj.add(rightNode);

    let rightAdj = this._adjacency.get(rightNode);
    if (rightAdj === undefined) {
      rightAdj = new Set();
      this._adjacency.set(rightNode, rightAdj);
    }
    rightAdj.add(leftNode);
  }

  /** Return the neighbor node indices for the given node. */
  neighbors(node: number): number[] {
    const adj = this._adjacency.get(node);
    if (adj === undefined) return [];
    return [...adj];
  }

  /**
   * Return all edges that connect a node in setA to a node in setB.
   */
  edgesBetween(setA: Set<number>, setB: Set<number>): JoinEdge[] {
    const result: JoinEdge[] = [];
    for (const edge of this._edges) {
      const leftInA = setA.has(edge.leftNode);
      const leftInB = setB.has(edge.leftNode);
      const rightInA = setA.has(edge.rightNode);
      const rightInB = setB.has(edge.rightNode);

      if ((leftInA && rightInB) || (leftInB && rightInA)) {
        result.push(edge);
      }
    }
    return result;
  }

  /** Total number of nodes in the graph. */
  get length(): number {
    return this._nodes.length;
  }

  /** Access nodes by index. */
  getNode(index: number): JoinNode {
    const node = this._nodes[index];
    if (node === undefined) {
      throw new Error(`JoinGraph: node ${String(index)} does not exist`);
    }
    return node;
  }

  /** Return all nodes. */
  get nodes(): readonly JoinNode[] {
    return this._nodes;
  }

  /** Return all edges. */
  get edges(): readonly JoinEdge[] {
    return this._edges;
  }

  /**
   * Estimate join selectivity between two nodes using column statistics.
   * Returns 1/max(distinct_count_left, distinct_count_right) when stats are
   * available, otherwise falls back to the default selectivity of 0.01.
   */
  estimateJoinSelectivity(
    leftNode: number,
    rightNode: number,
    leftField: string,
    rightField: string,
  ): number {
    const leftStats = this._nodes[leftNode]?.columnStats;
    const rightStats = this._nodes[rightNode]?.columnStats;

    const leftDistinct = leftStats?.get(leftField)?.distinctCount;
    const rightDistinct = rightStats?.get(rightField)?.distinctCount;

    if (leftDistinct !== undefined && leftDistinct > 0) {
      if (rightDistinct !== undefined && rightDistinct > 0) {
        return 1.0 / Math.max(leftDistinct, rightDistinct);
      }
      return 1.0 / leftDistinct;
    }
    if (rightDistinct !== undefined && rightDistinct > 0) {
      return 1.0 / rightDistinct;
    }
    return 0.01;
  }
}
