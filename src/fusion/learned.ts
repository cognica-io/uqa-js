//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- learned fusion
// 1:1 port of uqa/fusion/learned.py

import { LearnableLogOddsWeights, logOddsConjunction } from "bayesian-bm25";

export class LearnedFusion {
  private _learnable: LearnableLogOddsWeights;

  constructor(nSignals: number, alpha = 0.5) {
    this._learnable = new LearnableLogOddsWeights(nSignals, alpha);
  }

  get nSignals(): number {
    return this._learnable.nSignals;
  }

  fuse(probabilities: number[]): number {
    const weights = [...this._learnable.weights];
    return logOddsConjunction(probabilities, this._learnable.alpha, weights);
  }

  update(probs: number[], label: number, options?: { learningRate?: number }): void {
    // Compute logits for Hebbian update
    const logits = probs.map((p) => {
      const c = Math.max(1e-10, Math.min(1 - 1e-10, p));
      return Math.log(c / (1 - c));
    });
    const error = label - this.fuse(probs);
    this._learnable.update(logits, error, options);
  }

  stateDict(): Record<string, unknown> {
    return {
      weights: [...this._learnable.weights],
      alpha: this._learnable.alpha,
      n_signals: this._learnable.nSignals,
    };
  }

  loadStateDict(state: Record<string, unknown>): void {
    this._learnable = new LearnableLogOddsWeights(
      state["n_signals"] as number,
      state["alpha"] as number,
    );
    const w = state["weights"] as number[];
    for (let i = 0; i < w.length; i++) {
      this._learnable.weights[i] = w[i]!;
    }
  }
}
