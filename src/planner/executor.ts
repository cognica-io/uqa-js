//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- plan executor
// 1:1 port of uqa/planner/executor.py

import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "../operators/base.js";
import type { Operator } from "../operators/base.js";
import {
  UnionOperator,
  IntersectOperator,
  ComplementOperator,
} from "../operators/boolean.js";
import {
  TermOperator,
  VectorSimilarityOperator,
  KNNOperator,
  FilterOperator,
  ScoreOperator,
  IndexScanOperator,
} from "../operators/primitive.js";
import { SparseThresholdOperator } from "../operators/sparse.js";
import { MultiStageOperator } from "../operators/multi-stage.js";

// ---------------------------------------------------------------------------
// ExecutionStats
// ---------------------------------------------------------------------------

export interface ExecutionStats {
  operatorName: string;
  elapsedMs: number;
  resultCount: number;
  children: ExecutionStats[];
}

// ---------------------------------------------------------------------------
// PlanExecutor
// ---------------------------------------------------------------------------

/**
 * Executes an operator tree against an ExecutionContext, collecting
 * per-operator timing statistics.
 */
export class PlanExecutor {
  private readonly _context: ExecutionContext;
  private _lastStats: ExecutionStats | null = null;

  constructor(context: ExecutionContext) {
    this._context = context;
  }

  /**
   * Execute the operator tree and return the resulting PostingList.
   */
  execute(op: Operator): PostingList {
    const [result, stats] = this._executeWithStats(op);
    this._lastStats = stats;
    return result;
  }

  /**
   * Return a human-readable EXPLAIN string for the operator tree.
   */
  explain(op: Operator): string {
    return this._explainNode(op, 0);
  }

  /**
   * Return the execution statistics from the last execute() call.
   */
  get lastStats(): ExecutionStats | null {
    return this._lastStats;
  }

  // -----------------------------------------------------------------------
  // Internal: execute with statistics collection
  // -----------------------------------------------------------------------

  private _executeWithStats(op: Operator): [PostingList, ExecutionStats] {
    const name = this._operatorName(op);
    const childStats: ExecutionStats[] = [];

    const start = performance.now();
    let result: PostingList;

    if (op instanceof UnionOperator) {
      const childResults: PostingList[] = [];
      for (const child of op.operands) {
        const [r, s] = this._executeWithStats(child);
        childResults.push(r);
        childStats.push(s);
      }
      result = new PostingList();
      for (const r of childResults) {
        result = result.union(r);
      }
    } else if (op instanceof IntersectOperator) {
      if (op.operands.length === 0) {
        result = new PostingList();
      } else {
        const [first, firstStats] = this._executeWithStats(op.operands[0]!);
        childStats.push(firstStats);
        result = first;
        for (let i = 1; i < op.operands.length; i++) {
          if (result.length === 0) {
            // Short-circuit: no need to evaluate remaining operands
            break;
          }
          const [r, s] = this._executeWithStats(op.operands[i]!);
          childStats.push(s);
          result = result.intersect(r);
        }
      }
    } else if (op instanceof ComplementOperator) {
      const [, s] = this._executeWithStats(op.operand);
      childStats.push(s);
      result = op.execute(this._context); // Need full universal set
    } else if (op instanceof SparseThresholdOperator) {
      const [, s] = this._executeWithStats(op.source);
      childStats.push(s);
      // Re-execute to apply threshold (the source result was already computed
      // but we need the threshold logic -- just call execute directly)
      result = op.execute(this._context);
    } else if (op instanceof ScoreOperator) {
      const [, s] = this._executeWithStats(op.source);
      childStats.push(s);
      result = op.execute(this._context);
    } else if (op instanceof FilterOperator && op.source !== null) {
      const [, s] = this._executeWithStats(op.source);
      childStats.push(s);
      result = op.execute(this._context);
    } else if (op instanceof MultiStageOperator) {
      for (const [stageOp] of op.stages) {
        const [, s] = this._executeWithStats(stageOp);
        childStats.push(s);
      }
      result = op.execute(this._context);
    } else {
      // Leaf operator (TermOperator, VectorSimilarityOperator, etc.)
      result = op.execute(this._context);
    }

    const elapsed = performance.now() - start;

    const stats: ExecutionStats = {
      operatorName: name,
      elapsedMs: elapsed,
      resultCount: result.length,
      children: childStats,
    };

    return [result, stats];
  }

  // -----------------------------------------------------------------------
  // EXPLAIN
  // -----------------------------------------------------------------------

  private _explainNode(op: Operator, depth: number): string {
    const indent = "  ".repeat(depth);
    const name = this._operatorName(op);
    const details = this._operatorDetails(op);
    const line =
      details.length > 0 ? `${indent}${name} (${details})\n` : `${indent}${name}\n`;

    let result = line;

    // Recurse into children
    if (op instanceof UnionOperator) {
      for (const child of op.operands) {
        result += this._explainNode(child, depth + 1);
      }
    } else if (op instanceof IntersectOperator) {
      for (const child of op.operands) {
        result += this._explainNode(child, depth + 1);
      }
    } else if (op instanceof ComplementOperator) {
      result += this._explainNode(op.operand, depth + 1);
    } else if (op instanceof SparseThresholdOperator) {
      result += this._explainNode(op.source, depth + 1);
    } else if (op instanceof ScoreOperator) {
      result += this._explainNode(op.source, depth + 1);
    } else if (op instanceof FilterOperator && op.source !== null) {
      result += this._explainNode(op.source, depth + 1);
    } else if (op instanceof MultiStageOperator) {
      for (const [stageOp, cutoff] of op.stages) {
        result += `${indent}  [cutoff=${String(cutoff)}]\n`;
        result += this._explainNode(stageOp, depth + 2);
      }
    }

    return result;
  }

  // -----------------------------------------------------------------------
  // Operator metadata
  // -----------------------------------------------------------------------

  private _operatorName(op: Operator): string {
    if (op instanceof TermOperator) return "TermScan";
    if (op instanceof VectorSimilarityOperator) return "VectorThreshold";
    if (op instanceof KNNOperator) return "KNN";
    if (op instanceof FilterOperator) return "Filter";
    if (op instanceof IndexScanOperator) return "IndexScan";
    if (op instanceof ScoreOperator) return "Score";
    if (op instanceof UnionOperator) return "Union";
    if (op instanceof IntersectOperator) return "Intersect";
    if (op instanceof ComplementOperator) return "Complement";
    if (op instanceof SparseThresholdOperator) return "SparseThreshold";
    if (op instanceof MultiStageOperator) return "MultiStage";
    return op.constructor.name;
  }

  private _operatorDetails(op: Operator): string {
    if (op instanceof TermOperator) {
      const field = op.field !== null ? `field=${op.field}, ` : "";
      return `${field}term="${op.term}"`;
    }
    if (op instanceof VectorSimilarityOperator) {
      return `field=${op.field}, threshold=${String(op.threshold)}`;
    }
    if (op instanceof KNNOperator) {
      return `field=${op.field}, k=${String(op.k)}`;
    }
    if (op instanceof FilterOperator) {
      return `field=${op.field}`;
    }
    if (op instanceof IndexScanOperator) {
      return `field=${op.field}`;
    }
    if (op instanceof SparseThresholdOperator) {
      return `threshold=${String(op.threshold)}`;
    }
    if (op instanceof UnionOperator) {
      return `operands=${String(op.operands.length)}`;
    }
    if (op instanceof IntersectOperator) {
      return `operands=${String(op.operands.length)}`;
    }
    if (op instanceof ScoreOperator) {
      return `terms=${String(op.queryTerms.length)}`;
    }
    if (op instanceof MultiStageOperator) {
      return `stages=${String(op.stages.length)}`;
    }

    // Dynamic operator detail extraction (for graph and fusion operators)
    const typeName = op.constructor.name;
    if (typeName === "TraverseOperator") {
      const t = op as unknown as {
        graph: string;
        label: string | null;
        maxHops: number;
      };
      const label = t.label !== null ? `, label=${t.label}` : "";
      return `graph=${t.graph}${label}, maxHops=${String(t.maxHops)}`;
    }
    if (typeName === "PatternMatchOperator") {
      const p = op as unknown as {
        graph: string;
        pattern: { vertexPatterns: unknown[] };
      };
      return `graph=${p.graph}, vertices=${String(p.pattern.vertexPatterns.length)}`;
    }
    if (typeName === "RegularPathQueryOperator") {
      const r = op as unknown as { graph: string; startVertex: number | null };
      const start = r.startVertex !== null ? `, start=${String(r.startVertex)}` : "";
      return `graph=${r.graph}${start}`;
    }
    if (typeName === "WeightedPathQueryOperator") {
      const w = op as unknown as {
        graph: string;
        aggregation: string;
        weightProperty: string;
      };
      return `graph=${w.graph}, agg=${w.aggregation}, weight=${w.weightProperty}`;
    }
    if (typeName === "CypherQueryOperator") {
      const c = op as unknown as { graphName: string; query: string };
      const shortQuery = c.query.length > 40 ? c.query.slice(0, 40) + "..." : c.query;
      return `graph=${c.graphName}, query="${shortQuery}"`;
    }
    if (typeName === "LogOddsFusionOperator") {
      const f = op as unknown as { signals: Operator[]; alpha: number };
      return `signals=${String(f.signals.length)}, alpha=${String(f.alpha)}`;
    }
    if (typeName === "ProbBoolFusionOperator") {
      const f = op as unknown as { signals: Operator[]; mode: string };
      return `signals=${String(f.signals.length)}, mode=${f.mode}`;
    }
    if (typeName === "AttentionFusionOperator") {
      const f = op as unknown as { signals: Operator[] };
      return `signals=${String(f.signals.length)}`;
    }
    if (typeName === "LearnedFusionOperator") {
      const f = op as unknown as { signals: Operator[] };
      return `signals=${String(f.signals.length)}`;
    }
    if (typeName === "VertexAggregationOperator") {
      const a = op as unknown as { propertyName: string };
      return `property=${a.propertyName}`;
    }

    return "";
  }

  /**
   * Recursive EXPLAIN with full cost and cardinality estimation.
   * Produces a detailed plan tree with estimated rows and cost per node.
   */
  explainAnalyze(op: Operator, opts?: { verbose?: boolean }): string {
    const [, stats] = this._executeWithStats(op);
    return this._explainAnalyzeNode(stats, 0, opts?.verbose ?? false);
  }

  private _explainAnalyzeNode(
    stats: ExecutionStats,
    depth: number,
    verbose: boolean,
  ): string {
    const indent = "  ".repeat(depth);
    let line = `${indent}${stats.operatorName}`;
    line += ` (rows=${String(stats.resultCount)}, time=${stats.elapsedMs.toFixed(3)}ms)`;
    line += "\n";

    if (verbose && stats.children.length > 0) {
      line += `${indent}  children: ${String(stats.children.length)}\n`;
    }

    for (const child of stats.children) {
      line += this._explainAnalyzeNode(child, depth + 1, verbose);
    }

    return line;
  }

  /**
   * Return a structured EXPLAIN output as a tree of objects.
   */
  explainTree(op: Operator): Record<string, unknown> {
    return this._explainTreeNode(op);
  }

  private _explainTreeNode(op: Operator): Record<string, unknown> {
    const node: Record<string, unknown> = {
      operator: this._operatorName(op),
      details: this._operatorDetails(op),
    };

    const children: Record<string, unknown>[] = [];
    if (op instanceof UnionOperator) {
      for (const child of op.operands) {
        children.push(this._explainTreeNode(child));
      }
    } else if (op instanceof IntersectOperator) {
      for (const child of op.operands) {
        children.push(this._explainTreeNode(child));
      }
    } else if (op instanceof ComplementOperator) {
      children.push(this._explainTreeNode(op.operand));
    } else if (op instanceof SparseThresholdOperator) {
      children.push(this._explainTreeNode(op.source));
    } else if (op instanceof ScoreOperator) {
      children.push(this._explainTreeNode(op.source));
    } else if (op instanceof FilterOperator && op.source !== null) {
      children.push(this._explainTreeNode(op.source));
    } else if (op instanceof MultiStageOperator) {
      for (const [stageOp, cutoff] of op.stages) {
        const stageNode = this._explainTreeNode(stageOp);
        stageNode["cutoff"] = cutoff;
        children.push(stageNode);
      }
    } else {
      // Check for dynamic operator children
      const typeName = op.constructor.name;
      if (
        typeName === "LogOddsFusionOperator" ||
        typeName === "ProbBoolFusionOperator" ||
        typeName === "AttentionFusionOperator" ||
        typeName === "LearnedFusionOperator"
      ) {
        const f = op as unknown as { signals: Operator[] };
        for (const sig of f.signals) {
          children.push(this._explainTreeNode(sig));
        }
      }
    }

    if (children.length > 0) {
      node["children"] = children;
    }

    return node;
  }
}
