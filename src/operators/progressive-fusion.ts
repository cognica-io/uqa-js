//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- progressive fusion operator
// 1:1 port of uqa/operators/progressive_fusion.py

import type { DocId, IndexStats } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "./base.js";
import { Operator } from "./base.js";
import { FusionWANDScorer } from "../scoring/fusion-wand.js";

export class ProgressiveFusionOperator extends Operator {
  readonly stages: [Operator[], number][];
  readonly alpha: number;
  readonly gating: string | null;

  constructor(stages: [Operator[], number][], alpha = 0.5, gating?: string | null) {
    super();
    this.stages = stages;
    this.alpha = alpha;
    this.gating = gating ?? null;
  }

  execute(context: ExecutionContext): PostingList {
    const accumulatedPls: PostingList[] = [];
    let candidateIds: Set<DocId> | null = null;
    let result = new PostingList();

    for (const [newSignals, k] of this.stages) {
      // Execute new signal operators
      const newPls = newSignals.map((s) => s.execute(context));

      // Filter to candidate set if available
      if (candidateIds !== null) {
        for (let i = 0; i < newPls.length; i++) {
          const filtered = newPls[i]!.entries.filter((e) => candidateIds!.has(e.docId));
          newPls[i] = PostingList.fromSorted(filtered);
        }
      }

      accumulatedPls.push(...newPls);

      // Compute upper bounds
      const upperBounds = accumulatedPls.map((pl) => {
        let max = 0;
        for (const e of pl) {
          if (e.payload.score > max) max = e.payload.score;
        }
        return Math.max(max, 0.01);
      });

      // Fuse with WAND
      const fwand = new FusionWANDScorer(
        accumulatedPls,
        upperBounds,
        this.alpha,
        k,
        this.gating,
      );
      result = fwand.scoreTopK();

      // Update candidate set for next stage
      candidateIds = new Set(result.entries.map((e) => e.docId));
    }

    return result;
  }

  costEstimate(stats: IndexStats): number {
    let total = 0;
    let cardinality = stats.totalDocs;
    for (const [signals, k] of this.stages) {
      for (const s of signals) {
        total += s.costEstimate(stats) * (cardinality / Math.max(stats.totalDocs, 1));
      }
      cardinality = Math.min(k, cardinality);
    }
    return total;
  }
}
