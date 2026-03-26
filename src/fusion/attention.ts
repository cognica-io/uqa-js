//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- attention fusion
// 1:1 port of uqa/fusion/attention.py

import {
  AttentionLogOddsWeights,
  logOddsConjunction,
  MultiHeadAttentionLogOddsWeights,
} from "bayesian-bm25";

export class AttentionFusion {
  private _attn: AttentionLogOddsWeights;

  constructor(
    nSignals: number,
    nQueryFeatures = 6,
    alpha = 0.5,
    normalize = false,
    baseRate?: number | null,
  ) {
    this._attn = new AttentionLogOddsWeights(
      nSignals,
      nQueryFeatures,
      alpha,
      normalize,
      0,
      baseRate ?? null,
    );
  }

  get nSignals(): number {
    return this._attn.nSignals;
  }

  get nQueryFeatures(): number {
    return this._attn.nQueryFeatures;
  }

  fuse(probabilities: number[], queryFeatures: number[]): number {
    // Compute attention weights via matrix-vector product
    const wm = this._attn.weightsMatrix;
    const nSig = probabilities.length;
    const nFeat = queryFeatures.length;
    const weights: number[] = [];

    for (let i = 0; i < nSig; i++) {
      let dot = 0;
      const row = wm[i];
      if (row) {
        for (let j = 0; j < nFeat; j++) {
          dot += (row[j] ?? 0) * (queryFeatures[j] ?? 0);
        }
      }
      weights.push(dot);
    }

    // Softmax normalization
    let maxW = -Infinity;
    for (const w of weights) if (w > maxW) maxW = w;
    let sumExp = 0;
    for (let i = 0; i < weights.length; i++) {
      weights[i] = Math.exp(weights[i]! - maxW);
      sumExp += weights[i]!;
    }
    for (let i = 0; i < weights.length; i++) {
      weights[i] = weights[i]! / sumExp;
    }

    return logOddsConjunction(probabilities, this._attn.alpha, weights);
  }

  fit(probs: number[][], labels: number[], queryFeatures: number[][]): void {
    this._attn.fit(probs, labels, queryFeatures);
  }

  update(probs: number[], label: number, queryFeatures: number[]): void {
    this._attn.update(probs, label, queryFeatures);
  }

  stateDict(): Record<string, unknown> {
    return {
      weights_matrix: this._attn.weightsMatrix,
      alpha: this._attn.alpha,
      n_signals: this._attn.nSignals,
      n_query_features: this._attn.nQueryFeatures,
    };
  }

  loadStateDict(state: Record<string, unknown>): void {
    this._attn = new AttentionLogOddsWeights(
      state["n_signals"] as number,
      state["n_query_features"] as number,
      state["alpha"] as number,
    );
    // Restore weights matrix
    const wm = state["weights_matrix"];
    if (wm !== null && wm !== undefined) {
      const matrix = wm as number[][];
      for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[i]!.length; j++) {
          this._attn.weightsMatrix[i]![j] = matrix[i]![j]!;
        }
      }
    }
  }
}

export class MultiHeadAttentionFusion {
  private _mh: MultiHeadAttentionLogOddsWeights;
  private _nQueryFeatures: number;

  constructor(
    nSignals: number,
    nHeads = 4,
    nQueryFeatures = 6,
    alpha = 0.5,
    normalize = false,
  ) {
    this._mh = new MultiHeadAttentionLogOddsWeights(
      nHeads,
      nSignals,
      nQueryFeatures,
      alpha,
      normalize,
    );
    this._nQueryFeatures = nQueryFeatures;
  }

  get nSignals(): number {
    return this._mh.heads[0]?.nSignals ?? 0;
  }

  get nQueryFeatures(): number {
    return this._nQueryFeatures;
  }

  fuse(probabilities: number[], queryFeatures: number[]): number {
    return this._mh.combine(probabilities, queryFeatures, false);
  }

  fit(probs: number[][], labels: number[], queryFeatures: number[][]): void {
    this._mh.fit(probs, labels, queryFeatures);
  }

  stateDict(): Record<string, unknown> {
    return {
      n_heads: this._mh.nHeads,
      n_signals: this.nSignals,
      n_query_features: this._nQueryFeatures,
      alpha: this._mh.heads[0]?.alpha ?? 0.5,
    };
  }

  loadStateDict(state: Record<string, unknown>): void {
    this._mh = new MultiHeadAttentionLogOddsWeights(
      state["n_heads"] as number,
      state["n_signals"] as number,
      state["n_query_features"] as number,
      state["alpha"] as number,
    );
    this._nQueryFeatures = state["n_query_features"] as number;
  }
}
