//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- DPccp join enumeration
// 1:1 port of uqa/planner/join_enumerator.py
//
// DPccp (Dynamic Programming connected subgraph Complement Pair)
// algorithm from Moerkotte & Neumann, VLDB 2006.
//
// Enumerates connected subgraph complement pairs of the join graph to
// find the optimal join order via dynamic programming.  Complexity is
// O(3^n) where n is the number of relations, compared to O(n!) for
// exhaustive enumeration.
//
// Internally, relation subsets are represented as integer bitmasks for
// O(1) hash lookup and set operations.

import type { JoinGraph, JoinEdge, JoinPlan } from "./join-graph.js";

// Use index join when the smaller side has fewer rows than this threshold.
const INDEX_JOIN_THRESHOLD = 100;

// Maximum number of relations for exact DP enumeration.
// Beyond this threshold, use greedy heuristic to avoid
// exponential planning time.
const MAX_DP_RELATIONS = 16;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function frozensetToMask(fs: Set<number>): number {
  let mask = 0;
  for (const i of fs) {
    mask |= 1 << i;
  }
  return mask;
}

// ---------------------------------------------------------------------------
// DPccp
// ---------------------------------------------------------------------------

export class DPccp {
  private readonly _graph: JoinGraph;
  private _dp: Map<number, JoinPlan>;
  private readonly _allMask: number;

  constructor(graph: JoinGraph) {
    this._graph = graph;
    this._dp = new Map();
    this._allMask = (1 << graph.length) - 1;
  }

  /**
   * Find the optimal join plan for all relations in the graph.
   *
   * Returns the JoinPlan covering all relations with minimum cost.
   * Falls back to greedy for large queries.
   */
  optimize(): JoinPlan {
    const n = this._graph.length;
    if (n === 0) {
      throw new Error("Join graph has no relations");
    }
    if (n === 1) {
      const node = this._graph.getNode(0);
      return {
        relations: new Set([0]),
        cardinality: node.cardinality,
        cost: node.cardinality,
        left: null,
        right: null,
        joinEdge: null,
      };
    }

    // Initialize base relations
    for (let i = 0; i < n; i++) {
      const node = this._graph.getNode(i);
      this._dp.set(1 << i, {
        relations: new Set([i]),
        cardinality: node.cardinality,
        cost: node.cardinality,
        left: null,
        right: null,
        joinEdge: null,
      });
    }

    if (n > MAX_DP_RELATIONS) {
      return this._greedyOptimize();
    }

    // Enumerate connected subgraph complement pairs
    this._enumerateCsgCmpPairs();

    const result = this._dp.get(this._allMask);
    if (result === undefined) {
      // Graph is disconnected; join connected components
      return this._joinDisconnectedComponents();
    }

    return result;
  }

  /**
   * Core DPccp: enumerate all connected subgraph complement pairs.
   *
   * Builds connected subgraphs incrementally via BFS extension
   * instead of generating all C(n,k) subsets and filtering.
   * Each connected subgraph S is formed by extending a smaller
   * connected subgraph with an adjacent vertex whose index exceeds
   * min(S), ensuring each subgraph is generated exactly once.
   *
   * Uses a Uint8Array lookup table (indexed by bitmask) for O(1)
   * connectivity checks instead of hash-based set lookups.
   */
  private _enumerateCsgCmpPairs(): void {
    const n = this._graph.length;

    // Pre-compute neighbor lists for faster iteration.
    const neighbors: number[][] = [];
    for (let i = 0; i < n; i++) {
      neighbors.push(this._graph.neighbors(i));
    }

    // Uint8Array lookup table: connected[mask] == 1 iff the subgraph
    // represented by mask is connected.  At most 2^16 = 64 KB for
    // MAX_DP_RELATIONS = 16.
    const connected = new Uint8Array(1 << n);
    let prevLayer: number[] = [];
    for (let i = 0; i < n; i++) {
      const mask = 1 << i;
      connected[mask] = 1;
      prevLayer.push(mask);
    }

    for (let _size = 2; _size <= n; _size++) {
      const curLayer: number[] = [];
      for (const sMask of prevLayer) {
        // Find lowest set bit position (= min node in subset).
        const minNode = trailingZeros(sMask);
        // Try extending with each neighbor of each node in s.
        let node = 0;
        let tmp = sMask;
        while (tmp) {
          if (tmp & 1) {
            for (const nb of neighbors[node]!) {
              if (nb > minNode && !(sMask & (1 << nb))) {
                const newMask = sMask | (1 << nb);
                if (!connected[newMask]) {
                  connected[newMask] = 1;
                  curLayer.push(newMask);
                }
              }
            }
          }
          tmp >>>= 1;
          node++;
        }
      }

      // Phase 2: Enumerate valid splits for each new subgraph.
      for (const subsetMask of curLayer) {
        this._enumerateSplits(subsetMask, connected);
      }

      prevLayer = curLayer;
    }
  }

  /**
   * Try all valid splits of subsetMask into (s1, s2).
   *
   * Enumerates only submasks that contain the lowest set bit
   * (canonical half) using the identity:
   * rest = subsetMask ^ lowestBit; iterate submasks of
   * rest and OR in lowestBit.  This skips the entire
   * non-canonical half without branch checks.
   *
   * Connectivity is checked via O(1) Uint8Array indexing.
   */
  private _enumerateSplits(subsetMask: number, connected: Uint8Array): void {
    const dp = this._dp;
    const graph = this._graph;
    const lowestBit = subsetMask & -subsetMask;
    const rest = subsetMask ^ lowestBit;

    // Enumerate submasks of rest; each | lowestBit gives a
    // canonical submask of subsetMask (containing the min element).
    let subRest = (rest - 1) & rest;
    while (subRest) {
      const sub = subRest | lowestBit;
      const comp = subsetMask ^ sub;
      if (connected[sub] && connected[comp]) {
        const plan1 = dp.get(sub);
        const plan2 = dp.get(comp);
        if (plan1 !== undefined && plan2 !== undefined) {
          const edges = graph.edgesBetween(plan1.relations, plan2.relations);
          if (edges.length > 0) {
            this._emitCsgCmpPair(plan1, plan2, edges, subsetMask);
          }
        }
      }
      subRest = (subRest - 1) & rest;
    }

    // subRest == 0: s1 = {min element}, s2 = rest of subset.
    // Singletons are always connected, so only check s2.
    if (connected[rest]) {
      const plan1 = dp.get(lowestBit);
      const plan2 = dp.get(rest);
      if (plan1 !== undefined && plan2 !== undefined) {
        const edges = graph.edgesBetween(plan1.relations, plan2.relations);
        if (edges.length > 0) {
          this._emitCsgCmpPair(plan1, plan2, edges, subsetMask);
        }
      }
    }
  }

  /**
   * Consider joining plan1 and plan2 via the given edges.
   */
  private _emitCsgCmpPair(
    plan1: JoinPlan,
    plan2: JoinPlan,
    edges: JoinEdge[],
    combinedMask: number,
  ): void {
    // Compute join cardinality: product * selectivity for each edge
    let cardinality = plan1.cardinality * plan2.cardinality;
    for (const edge of edges) {
      cardinality *= edge.selectivity;
    }

    // Cost = C_out + C_left + C_right
    // Use index join cost when the smaller side fits within threshold;
    // otherwise use hash join cost.
    const c1 = plan1.cardinality;
    const c2 = plan2.cardinality;
    let joinCost: number;
    if (c1 <= c2) {
      if (c1 <= INDEX_JOIN_THRESHOLD) {
        joinCost = c1 * Math.log2(c2 + 1);
      } else {
        joinCost = c1 + c2;
      }
    } else if (c2 <= INDEX_JOIN_THRESHOLD) {
      joinCost = c2 * Math.log2(c1 + 1);
    } else {
      joinCost = c1 + c2;
    }
    const totalCost = joinCost + plan1.cost + plan2.cost;

    const existing = this._dp.get(combinedMask);
    if (existing === undefined || totalCost < existing.cost) {
      const combined = new Set<number>();
      for (const r of plan1.relations) combined.add(r);
      for (const r of plan2.relations) combined.add(r);
      this._dp.set(combinedMask, {
        relations: combined,
        cardinality,
        cost: totalCost,
        left: plan1,
        right: plan2,
        joinEdge: edges[0]!, // Primary join edge
      });
    }
  }

  /**
   * Check if the subgraph induced by subset is connected.
   */
  private _isConnected(subset: Set<number>): boolean {
    if (subset.size <= 1) return true;

    const start = subset.values().next().value!;
    const visited = new Set<number>([start]);
    const stack = [start];

    while (stack.length > 0) {
      const node = stack.pop()!;
      for (const neighbor of this._graph.neighbors(node)) {
        if (subset.has(neighbor) && !visited.has(neighbor)) {
          visited.add(neighbor);
          stack.push(neighbor);
        }
      }
    }

    return visited.size === subset.size;
  }

  /**
   * Handle disconnected join graphs by cross-joining components.
   *
   * When the join graph is not fully connected (e.g. FROM a, b
   * with no join predicate), identify connected components and
   * join them with cross products.
   */
  private _joinDisconnectedComponents(): JoinPlan {
    const components = this._findConnectedComponents();

    // Solve each component independently
    const componentPlans: JoinPlan[] = [];
    for (const component of components) {
      if (component.size === 1) {
        const nodeIdx = component.values().next().value!;
        const plan = this._dp.get(1 << nodeIdx);
        if (plan !== undefined) {
          componentPlans.push(plan);
        }
      } else {
        const compMask = frozensetToMask(component);
        let plan = this._dp.get(compMask);
        if (plan === undefined) {
          // Component was not solved; build subgraph and solve
          const subGraph = this._buildSubgraph(component);
          const subSolver = new DPccp(subGraph);
          const subPlan = subSolver.optimize();
          plan = this._remapPlan(
            subPlan,
            [...component].sort((a, b) => a - b),
          );
        }
        componentPlans.push(plan);
      }
    }

    // Cross-join components in order of ascending cardinality
    componentPlans.sort((a, b) => a.cardinality - b.cardinality);
    let result = componentPlans[0]!;
    for (let i = 1; i < componentPlans.length; i++) {
      const plan = componentPlans[i]!;
      const combined = new Set<number>();
      for (const r of result.relations) combined.add(r);
      for (const r of plan.relations) combined.add(r);
      const cardinality = result.cardinality * plan.cardinality;
      const cost = cardinality + result.cost + plan.cost;
      result = {
        relations: combined,
        cardinality,
        cost,
        left: result,
        right: plan,
        joinEdge: null, // Cross join
      };
    }
    return result;
  }

  /**
   * Find all connected components of the join graph.
   */
  private _findConnectedComponents(): Set<number>[] {
    const n = this._graph.length;
    const remaining = new Set<number>();
    for (let i = 0; i < n; i++) remaining.add(i);
    const components: Set<number>[] = [];

    while (remaining.size > 0) {
      const start = remaining.values().next().value!;
      const visited = new Set<number>([start]);
      const stack = [start];

      while (stack.length > 0) {
        const node = stack.pop()!;
        for (const neighbor of this._graph.neighbors(node)) {
          if (remaining.has(neighbor) && !visited.has(neighbor)) {
            visited.add(neighbor);
            stack.push(neighbor);
          }
        }
      }

      components.push(visited);
      for (const v of visited) remaining.delete(v);
    }

    return components;
  }

  /**
   * Build a JoinGraph containing only the given nodes.
   */
  private _buildSubgraph(nodes: Set<number>): JoinGraph {
    // Dynamic import would be circular; construct inline
    // We need the JoinGraph class -- import it from the same module
    const GraphCtor = this._graph.constructor as {
      new (): JoinGraph;
    };
    const sub = new GraphCtor();
    const indexMap = new Map<number, number>();

    for (const oldIdx of [...nodes].sort((a, b) => a - b)) {
      const node = this._graph.getNode(oldIdx);
      const newIdx = sub.addNode(
        node.alias,
        node.operator,
        node.table,
        node.cardinality,
      );
      indexMap.set(oldIdx, newIdx);
    }

    for (const edge of this._graph.edges) {
      if (nodes.has(edge.leftNode) && nodes.has(edge.rightNode)) {
        sub.addEdge(
          indexMap.get(edge.leftNode)!,
          indexMap.get(edge.rightNode)!,
          edge.leftField,
          edge.rightField,
          edge.selectivity,
        );
      }
    }

    return sub;
  }

  /**
   * Remap plan relation indices from subgraph back to original graph.
   */
  private _remapPlan(plan: JoinPlan, originalIndices: number[]): JoinPlan {
    const newRels = new Set<number>();
    for (const i of plan.relations) {
      newRels.add(originalIndices[i]!);
    }
    return {
      relations: newRels,
      cardinality: plan.cardinality,
      cost: plan.cost,
      left: plan.left !== null ? this._remapPlan(plan.left, originalIndices) : null,
      right: plan.right !== null ? this._remapPlan(plan.right, originalIndices) : null,
      joinEdge: plan.joinEdge,
    };
  }

  /**
   * Greedy join ordering for large queries (> MAX_DP_RELATIONS).
   *
   * At each step, pick the pair of existing plans with the lowest
   * join cost and merge them.  This is O(n^3) in the number of
   * relations.
   */
  private _greedyOptimize(): JoinPlan {
    const active = new Map<number, JoinPlan>(this._dp);

    while (active.size > 1) {
      let bestCost = Infinity;
      let bestCombinedMask = 0;
      let bestPlan: JoinPlan | null = null;

      const items = [...active.entries()];
      for (let i = 0; i < items.length; i++) {
        const [m1, p1] = items[i]!;
        for (let j = i + 1; j < items.length; j++) {
          const [m2, p2] = items[j]!;
          const edges = this._graph.edgesBetween(p1.relations, p2.relations);
          if (edges.length === 0) continue;

          let cardinality = p1.cardinality * p2.cardinality;
          for (const edge of edges) {
            cardinality *= edge.selectivity;
          }

          const gc1 = p1.cardinality;
          const gc2 = p2.cardinality;
          let greedyJoinCost: number;
          if (gc1 <= gc2) {
            if (gc1 <= INDEX_JOIN_THRESHOLD) {
              greedyJoinCost = gc1 * Math.log2(gc2 + 1);
            } else {
              greedyJoinCost = gc1 + gc2;
            }
          } else if (gc2 <= INDEX_JOIN_THRESHOLD) {
            greedyJoinCost = gc2 * Math.log2(gc1 + 1);
          } else {
            greedyJoinCost = gc1 + gc2;
          }
          const cost = greedyJoinCost + p1.cost + p2.cost;

          if (cost < bestCost) {
            bestCost = cost;
            bestCombinedMask = m1 | m2;
            const combined = new Set<number>();
            for (const r of p1.relations) combined.add(r);
            for (const r of p2.relations) combined.add(r);
            bestPlan = {
              relations: combined,
              cardinality,
              cost,
              left: p1,
              right: p2,
              joinEdge: edges[0]!,
            };
          }
        }
      }

      if (bestPlan === null) {
        // No more edges; cross-join remaining
        const remaining = [...active.values()];
        remaining.sort((a, b) => a.cardinality - b.cardinality);
        let result = remaining[0]!;
        for (let i = 1; i < remaining.length; i++) {
          const plan = remaining[i]!;
          const combined = new Set<number>();
          for (const r of result.relations) combined.add(r);
          for (const r of plan.relations) combined.add(r);
          const cardinality = result.cardinality * plan.cardinality;
          const cost = cardinality + result.cost + plan.cost;
          result = {
            relations: combined,
            cardinality,
            cost,
            left: result,
            right: plan,
            joinEdge: null,
          };
        }
        return result;
      }

      // Merge the best pair
      for (const relMask of [...active.keys()]) {
        if ((relMask & bestCombinedMask) === relMask) {
          active.delete(relMask);
        }
      }
      active.set(bestCombinedMask, bestPlan);
    }

    return active.values().next().value!;
  }
}

// ---------------------------------------------------------------------------
// Bit manipulation helper
// ---------------------------------------------------------------------------

/** Count trailing zeros in a positive integer (position of lowest set bit). */
function trailingZeros(n: number): number {
  if (n === 0) return 32;
  let count = 0;
  let v = n;
  while ((v & 1) === 0) {
    count++;
    v >>>= 1;
  }
  return count;
}
