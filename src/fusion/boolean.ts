//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- probabilistic boolean fusion
// 1:1 port of uqa/fusion/boolean.py

import { probAnd, probNot, probOr } from "bayesian-bm25";

// eslint-disable-next-line @typescript-eslint/no-extraneous-class
export class ProbabilisticBoolean {
  static probAnd(probabilities: number[]): number {
    return probAnd(probabilities);
  }

  static probOr(probabilities: number[]): number {
    return probOr(probabilities);
  }

  static probNot(p: number): number {
    return probNot(p);
  }
}
