//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- greedy join-order optimizer
// 1:1 port of uqa/planner/join_order.py

import type { JoinEdge, JoinNode, JoinPlan } from "./join-graph.js";

// ---------------------------------------------------------------------------
// JoinOrderOptimizer -- greedy heuristic (for small join counts)
// ---------------------------------------------------------------------------

/**
 * Greedy join-order optimizer.
 *
 * For each step, picks the cheapest pair (lowest estimated intermediate
 * cardinality) from the remaining relations and fuses them.
 * Works well for star schemas and small numbers of relations (< ~8).
 * For larger join graphs, use the DPccp enumerator instead.
 */
export class JoinOrderOptimizer {
  /**
   * Find a join ordering for the given relations and join predicates.
   *
   * @param relations  The base relation nodes.
   * @param predicates The join predicates (edges).
   * @returns A JoinPlan tree, or null if no valid plan can be formed.
   */
  optimize(relations: JoinNode[], predicates: JoinEdge[]): JoinPlan | null {
    if (relations.length === 0) return null;

    // Build initial leaf plans keyed by node index
    const plans = new Map<number, JoinPlan>();
    for (const rel of relations) {
      plans.set(rel.index, {
        relations: new Set([rel.index]),
        cardinality: rel.cardinality,
        cost: 0,
        left: null,
        right: null,
        joinEdge: null,
      });
    }

    if (relations.length === 1) {
      return plans.values().next().value as JoinPlan;
    }

    // Build an index of predicates by node pair
    const edgeMap = new Map<string, JoinEdge>();
    for (const pred of predicates) {
      const key =
        pred.leftNode < pred.rightNode
          ? `${String(pred.leftNode)},${String(pred.rightNode)}`
          : `${String(pred.rightNode)},${String(pred.leftNode)}`;
      edgeMap.set(key, pred);
    }

    // Greedy loop: merge the cheapest pair at each step
    while (plans.size > 1) {
      let bestCost = Infinity;
      let bestLeft: number | null = null;
      let bestRight: number | null = null;
      let bestEdge: JoinEdge | null = null;

      const nodeIds = [...plans.keys()];

      for (let i = 0; i < nodeIds.length; i++) {
        for (let j = i + 1; j < nodeIds.length; j++) {
          const leftId = nodeIds[i]!;
          const rightId = nodeIds[j]!;
          const leftPlan = plans.get(leftId)!;
          const rightPlan = plans.get(rightId)!;

          // Find edges between the two plan's relation sets
          const edge = this._findEdge(
            leftPlan.relations,
            rightPlan.relations,
            predicates,
          );
          const selectivity = edge !== null ? edge.selectivity : 1.0;
          const joinCard = leftPlan.cardinality * rightPlan.cardinality * selectivity;
          const cost = leftPlan.cost + rightPlan.cost + joinCard;

          if (cost < bestCost) {
            bestCost = cost;
            bestLeft = leftId;
            bestRight = rightId;
            bestEdge = edge;
          }
        }
      }

      if (bestLeft === null || bestRight === null) {
        // No valid pair found -- should not happen with >= 2 plans
        break;
      }

      const leftPlan = plans.get(bestLeft)!;
      const rightPlan = plans.get(bestRight)!;
      const selectivity = bestEdge !== null ? bestEdge.selectivity : 1.0;

      const mergedRelations = new Set<number>(leftPlan.relations);
      for (const r of rightPlan.relations) {
        mergedRelations.add(r);
      }

      const merged: JoinPlan = {
        relations: mergedRelations,
        cardinality: leftPlan.cardinality * rightPlan.cardinality * selectivity,
        cost: bestCost,
        left: leftPlan,
        right: rightPlan,
        joinEdge: bestEdge,
      };

      plans.delete(bestLeft);
      plans.delete(bestRight);
      // Use the smaller index as key for the merged plan
      const mergedKey = Math.min(bestLeft, bestRight);
      plans.set(mergedKey, merged);
    }

    // Return the single remaining plan
    const remaining = plans.values().next().value;
    return remaining ?? null;
  }

  /**
   * Find the first edge connecting any node in setA to any node in setB.
   */
  private _findEdge(
    setA: Set<number>,
    setB: Set<number>,
    predicates: JoinEdge[],
  ): JoinEdge | null {
    for (const pred of predicates) {
      const leftInA = setA.has(pred.leftNode);
      const leftInB = setB.has(pred.leftNode);
      const rightInA = setA.has(pred.rightNode);
      const rightInB = setB.has(pred.rightNode);
      if ((leftInA && rightInB) || (leftInB && rightInA)) {
        return pred;
      }
    }
    return null;
  }
}
