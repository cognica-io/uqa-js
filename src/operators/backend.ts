//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- operator backend utilities
// 1:1 port of uqa/operators/_backend.py
// Browser version: no PyTorch, pure TypeScript/Float64Array

import * as linalg from "../math/linalg.js";

const PROB_FLOOR = 1e-15;
const PROB_CEIL = 1.0 - 1e-15;

export function deviceName(): string {
  return "cpu (js)";
}

export function safeLogit(p: number): number {
  const c = Math.max(PROB_FLOOR, Math.min(PROB_CEIL, p));
  return Math.log(c / (1.0 - c));
}

export function sigmoidStable(x: number): number {
  if (x >= 0) return 1.0 / (1.0 + Math.exp(-x));
  const ex = Math.exp(x);
  return ex / (1.0 + ex);
}

export function applyGating(logitVal: number, gating: string): number {
  if (gating === "relu") return Math.max(0, logitVal);
  if (gating === "swish") return logitVal * sigmoidStable(logitVal);
  return logitVal;
}

export function sigmoidVec(x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = sigmoidStable(x[i]!);
  }
  return out;
}

export function applyGatingVec(vec: Float64Array, gating: string): Float64Array {
  if (gating === "relu") {
    const out = new Float64Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
      out[i] = Math.max(0.0, vec[i]!);
    }
    return out;
  }
  if (gating === "swish") {
    const out = new Float64Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
      out[i] = vec[i]! * sigmoidStable(vec[i]!);
    }
    return out;
  }
  return new Float64Array(vec);
}

// -- Kernel builder --------------------------------------------------------

function buildKernelNp(hopWeights: number[]): Float64Array {
  const total = hopWeights.reduce((a, b) => a + b, 0);
  const k = new Float64Array(9); // 3x3
  if (total <= 0) return k;
  const wS = hopWeights[0]! / total;
  const wN = (hopWeights.length > 1 ? hopWeights[1]! : 0.0) / total;
  k[4] = wS; // center
  k[1] = wN / 4; // top
  k[7] = wN / 4; // bottom
  k[3] = wN / 4; // left
  k[5] = wN / 4; // right
  return k;
}

export function hopWeightsToKernel(hopWeights: number[]): Float64Array {
  // Convert [w_self, w_neighbor] to (1, 1, 3, 3) kernel array.
  // For the JS version we return a flat 9-element array representing the 3x3 kernel.
  // Multi-channel kernels would need (out_ch * in_ch * 3 * 3) elements.
  return buildKernelNp(hopWeights);
}

// -- Ridge regression ------------------------------------------------------

export function ridgeSolve(
  X: Float64Array,
  shapeX: linalg.Shape2D,
  Y: Float64Array,
  shapeY: linalg.Shape2D,
  lam: number,
): { weights: Float64Array; bias: Float64Array } {
  const [n, p] = shapeX;
  const [, nClasses] = shapeY;

  // W = (X^T X + lam I)^{-1} X^T Y via normal equations
  const Xt = linalg.transpose(X, n, p);
  const XtX = linalg.matmul(Xt, [p, n], X, [n, p]);

  // Add lambda * I
  for (let i = 0; i < p; i++) {
    XtX.data[i * p + i]! += lam;
  }

  const XtY = linalg.matmul(Xt, [p, n], Y, [n, nClasses]);

  // Solve via iterative refinement (Gauss-Seidel)
  const W = new Float64Array(p * nClasses);

  // Use gradient descent to solve (XtX)W = XtY
  const lr = 0.01;
  for (let iter = 0; iter < 100; iter++) {
    const residual = linalg.matmul(XtX.data, [p, p], W, [p, nClasses]);
    for (let i = 0; i < p * nClasses; i++) {
      W[i]! -= lr * (residual.data[i]! - XtY.data[i]!);
    }
  }

  // Bias = meanY - W^T meanX
  const meanX = new Float64Array(p);
  const meanY = new Float64Array(nClasses);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) meanX[j]! += X[i * p + j]! / n;
    for (let j = 0; j < nClasses; j++) meanY[j]! += Y[i * nClasses + j]! / n;
  }

  const bias = new Float64Array(nClasses);
  for (let j = 0; j < nClasses; j++) {
    bias[j] = meanY[j]!;
    for (let i = 0; i < p; i++) {
      bias[j]! -= W[i * nClasses + j]! * meanX[i]!;
    }
  }

  return { weights: W, bias };
}

// -- Elastic net solve -----------------------------------------------------

export function elasticNetSolve(
  X: Float64Array,
  shapeX: linalg.Shape2D,
  Y: Float64Array,
  shapeY: linalg.Shape2D,
  lam: number,
  l1Ratio: number,
  maxIter = 200,
  tol = 1e-4,
): { weights: Float64Array; bias: Float64Array } {
  const l2Pen = lam * (1.0 - l1Ratio);
  const l1Pen = lam * l1Ratio;

  const [n, p] = shapeX;
  const [, nClasses] = shapeY;

  // Warm start from ridge
  const { weights: W, bias } = ridgeSolve(X, shapeX, Y, shapeY, l2Pen);

  // Compute XtX / n
  const Xt = linalg.transpose(X, n, p);
  const XtXn = linalg.matmul(Xt, [p, n], X, [n, p]);
  for (let i = 0; i < p * p; i++) XtXn.data[i]! /= n;

  // Lipschitz constant
  let lip = 0;
  for (let i = 0; i < p * p; i++) lip += XtXn.data[i]! * XtXn.data[i]!;
  lip = Math.sqrt(lip) + l2Pen;
  const step = 1.0 / lip;
  const threshold = (l1Pen * step) / n;

  for (let iter = 0; iter < maxIter; iter++) {
    const WOld = new Float64Array(W);

    // residual = Y - X @ W^T - bias
    // grad = -(X^T @ residual / n)^T + l2Pen * W
    // W = W - step * grad, then soft threshold
    const residual = new Float64Array(n * nClasses);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < nClasses; j++) {
        let pred = bias[j]!;
        for (let k = 0; k < p; k++) {
          pred += X[i * p + k]! * W[j * p + k]!;
        }
        residual[i * nClasses + j] = Y[i * nClasses + j]! - pred;
      }
    }

    // grad = -(Xt @ residual / n)^T + l2Pen * W
    const XtRes = linalg.matmul(Xt, [p, n], residual, [n, nClasses]);
    for (let j = 0; j < nClasses; j++) {
      for (let k = 0; k < p; k++) {
        const grad = -(XtRes.data[k * nClasses + j]! / n) + l2Pen * W[j * p + k]!;
        let val = W[j * p + k]! - step * grad;
        // Soft threshold (proximal L1)
        val = Math.sign(val) * Math.max(Math.abs(val) - threshold, 0);
        W[j * p + k] = val;
      }
    }

    // Update bias
    for (let j = 0; j < nClasses; j++) {
      let total = 0;
      for (let i = 0; i < n; i++) {
        let pred = 0;
        for (let k = 0; k < p; k++) {
          pred += X[i * p + k]! * W[j * p + k]!;
        }
        total += Y[i * nClasses + j]! - pred;
      }
      bias[j] = total / n;
    }

    // Check convergence
    let maxDiff = 0;
    for (let i = 0; i < W.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(W[i]! - WOld[i]!));
    }
    if (maxDiff < tol) break;
  }

  return { weights: W, bias };
}

// -- Magnitude pruning -------------------------------------------------------

export function magnitudePrune(W: Float64Array, pruneRatio: number): Float64Array {
  if (pruneRatio <= 0) return new Float64Array(W);
  const absVals = Array.from(W, (v) => Math.abs(v));
  const sorted = absVals.slice().sort((a, b) => a - b);
  const cutoffIdx = Math.floor(pruneRatio * sorted.length);
  const cutoff = sorted[cutoffIdx] ?? 0;
  const out = new Float64Array(W.length);
  for (let i = 0; i < W.length; i++) {
    out[i] = Math.abs(W[i]!) < cutoff ? 0 : W[i]!;
  }
  return out;
}

// -- WAND predict ------------------------------------------------------------

export function wandPredict(
  W: Float64Array,
  shapeW: linalg.Shape2D,
  bias: Float64Array,
  x: Float64Array,
): Float64Array {
  const [nClasses, nFeatures] = shapeW;
  const logits = new Float64Array(bias);

  for (let c = 0; c < nClasses; c++) {
    for (let f = 0; f < nFeatures; f++) {
      const w = W[c * nFeatures + f]!;
      if (w !== 0) {
        logits[c]! += w * x[f]!;
      }
    }
  }

  return logits;
}

// -- Grid forward pass -------------------------------------------------------

export function gridForward(
  embeddings: Float64Array,
  shapeE: linalg.Shape2D,
  gridH: number,
  gridW: number,
  stages: {
    kernel: Float64Array;
    kernelShape: number[];
    poolSize: number;
    poolMethod: string;
  }[],
  gating = "none",
): { data: Float64Array; shape: linalg.Shape2D } {
  const [batch] = shapeE;
  let inCh = Math.floor(shapeE[1] / (gridH * gridW));
  if (inCh < 1) inCh = 1;
  let h = gridH;
  let w = gridW;

  // current: (batch, inCh, h, w) stored flat
  let current = new Float64Array(embeddings);

  for (const stage of stages) {
    const ks = stage.kernelShape;
    // kernel: (outCh, inCurCh, kH, kW) flattened
    let outCh: number;
    let inCurCh: number;
    if (ks.length === 4) {
      outCh = ks[0]!;
      inCurCh = ks[1]!;
    } else {
      // Simple 3x3 kernel
      outCh = inCh;
      inCurCh = inCh;
    }

    const kH = ks.length === 4 ? ks[2]! : 3;
    const kW = ks.length === 4 ? ks[3]! : 3;
    const padH = Math.floor(kH / 2);
    const padW = Math.floor(kW / 2);

    // Convolution: output (batch, outCh, h, w)
    const convOut = new Float64Array(batch * outCh * h * w);
    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < outCh; oc++) {
        for (let i = 0; i < h; i++) {
          for (let j = 0; j < w; j++) {
            let val = 0;
            for (let ic = 0; ic < inCurCh; ic++) {
              for (let ki = 0; ki < kH; ki++) {
                for (let kj = 0; kj < kW; kj++) {
                  const ni = i + ki - padH;
                  const nj = j + kj - padW;
                  if (ni >= 0 && ni < h && nj >= 0 && nj < w) {
                    const srcIdx = b * inCurCh * h * w + ic * h * w + ni * w + nj;
                    const kIdx = oc * inCurCh * kH * kW + ic * kH * kW + ki * kW + kj;
                    const coeff = stage.kernel[kIdx] ?? 0;
                    if (coeff !== 0) {
                      val += (current[srcIdx] ?? 0) * coeff;
                    }
                  }
                }
              }
            }
            const dstIdx = b * outCh * h * w + oc * h * w + i * w + j;
            convOut[dstIdx] = applyGating(val, gating);
          }
        }
      }
    }
    current = convOut;
    inCh = outCh;

    // Apply pooling
    if (stage.poolSize > 1) {
      const newH = Math.floor(h / stage.poolSize);
      const newW = Math.floor(w / stage.poolSize);
      const pooled = new Float64Array(batch * outCh * newH * newW);
      for (let b = 0; b < batch; b++) {
        for (let c = 0; c < outCh; c++) {
          for (let pi = 0; pi < newH; pi++) {
            for (let pj = 0; pj < newW; pj++) {
              let poolVal = stage.poolMethod === "max" ? -Infinity : 0;
              let count = 0;
              for (let ki = 0; ki < stage.poolSize; ki++) {
                for (let kj = 0; kj < stage.poolSize; kj++) {
                  const si = pi * stage.poolSize + ki;
                  const sj = pj * stage.poolSize + kj;
                  if (si < h && sj < w) {
                    const idx = b * outCh * h * w + c * h * w + si * w + sj;
                    const v = current[idx] ?? 0;
                    if (stage.poolMethod === "max") {
                      poolVal = Math.max(poolVal, v);
                    } else {
                      poolVal += v;
                    }
                    count++;
                  }
                }
              }
              if (stage.poolMethod === "avg" && count > 0) poolVal /= count;
              const dstIdx = b * outCh * newH * newW + c * newH * newW + pi * newW + pj;
              pooled[dstIdx] = poolVal;
            }
          }
        }
      }
      current = pooled;
      h = newH;
      w = newW;
    }
  }

  const outFeatures = inCh * h * w;
  return { data: current, shape: [batch, outFeatures] };
}

// -- Batch dense / softmax / batchnorm ------------------------------------

export function batchDense(
  X: Float64Array,
  shapeX: linalg.Shape2D,
  weights: Float64Array,
  shapeW: linalg.Shape2D,
  bias: Float64Array,
  gating = "none",
): { data: Float64Array; shape: linalg.Shape2D } {
  // out = X @ W^T + bias with gating
  const [n] = shapeX;
  const [nOut] = shapeW;
  const Wt = linalg.transpose(weights, shapeW[0], shapeW[1]);
  const result = linalg.matmul(X, shapeX, Wt, [shapeW[1], shapeW[0]]);

  // Add bias and apply gating
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < nOut; j++) {
      const idx = i * nOut + j;
      result.data[idx] = applyGating(result.data[idx]! + bias[j]!, gating);
    }
  }

  return { data: result.data, shape: [n, nOut] };
}

export function batchSoftmax(X: Float64Array, shape: linalg.Shape2D): Float64Array {
  const [n, c] = shape;
  const out = new Float64Array(X.length);
  for (let i = 0; i < n; i++) {
    const row = X.subarray(i * c, (i + 1) * c);
    const sm = linalg.softmax(row);
    out.set(sm, i * c);
  }
  return out;
}

export function batchBatchnorm(
  X: Float64Array,
  shape: linalg.Shape2D,
  epsilon = 1e-5,
): Float64Array {
  const [n, c] = shape;
  if (n < 2) return new Float64Array(X);

  const means = new Float64Array(c);
  const vars = new Float64Array(c);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < c; j++) {
      means[j]! += X[i * c + j]! / n;
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < c; j++) {
      const diff = X[i * c + j]! - means[j]!;
      vars[j]! += (diff * diff) / n;
    }
  }

  const out = new Float64Array(X.length);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < c; j++) {
      out[i * c + j] = (X[i * c + j]! - means[j]!) / Math.sqrt(vars[j]! + epsilon);
    }
  }
  return out;
}

// -- Self-attention (Theorem 8.3, Paper 4) --------------------------------

export function generateQkProjections(
  dModel: number,
  seed = 42,
): { wQ: Float64Array; wK: Float64Array } {
  // Generate random Q, K projection matrices (Kaiming init).
  // Returns (wQ, wK) each of shape (dModel, dModel).
  const std = Math.sqrt(2.0 / dModel);
  const wQ = new Float64Array(dModel * dModel);
  const wK = new Float64Array(dModel * dModel);

  // Simple seeded PRNG (xorshift32)
  let s = seed;
  function nextRand(): number {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    // Box-Muller approximation for normal distribution
    const u1 = (s >>> 0) / 0xffffffff;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    const u2 = (s >>> 0) / 0xffffffff;
    return Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
  }

  for (let i = 0; i < dModel * dModel; i++) {
    wQ[i] = nextRand() * std;
  }
  for (let i = 0; i < dModel * dModel; i++) {
    wK[i] = nextRand() * std;
  }

  return { wQ, wK };
}

export function batchSelfAttention(
  X: Float64Array,
  shapeX: readonly [number, number, number], // (batch, seqLen, dModel)
  nHeads = 1,
  wQ: Float64Array | null = null,
  wK: Float64Array | null = null,
  wV: Float64Array | null = null,
  gating = "none",
): Float64Array {
  const [nTotal, seqLen, dModel] = shapeX;

  let effectiveHeads = nHeads;
  if (dModel % effectiveHeads !== 0) effectiveHeads = 1;
  const dHead = Math.floor(dModel / effectiveHeads);

  // Q = X @ W_q or X, K = X @ W_k or X, V = X @ W_v or X
  const Q =
    wQ !== null
      ? matmul3d(X, nTotal, seqLen, dModel, wQ, dModel, dModel)
      : new Float64Array(X);
  const K =
    wK !== null
      ? matmul3d(X, nTotal, seqLen, dModel, wK, dModel, dModel)
      : new Float64Array(X);
  const V =
    wV !== null
      ? matmul3d(X, nTotal, seqLen, dModel, wV, dModel, dModel)
      : new Float64Array(X);

  // Reshape to (nTotal, nHeads, seqLen, dHead) and compute attention
  const scale = 1.0 / Math.sqrt(dHead);
  const out = new Float64Array(nTotal * seqLen * dModel);

  for (let b = 0; b < nTotal; b++) {
    for (let h = 0; h < effectiveHeads; h++) {
      // Compute attention scores for this head
      const scores = new Float64Array(seqLen * seqLen);
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          let dot = 0;
          for (let d = 0; d < dHead; d++) {
            const qIdx = b * seqLen * dModel + i * dModel + h * dHead + d;
            const kIdx = b * seqLen * dModel + j * dModel + h * dHead + d;
            dot += Q[qIdx]! * K[kIdx]!;
          }
          scores[i * seqLen + j] = dot * scale;
        }
      }

      // Softmax per row
      for (let i = 0; i < seqLen; i++) {
        let maxVal = -Infinity;
        for (let j = 0; j < seqLen; j++) {
          maxVal = Math.max(maxVal, scores[i * seqLen + j]!);
        }
        let sumExp = 0;
        for (let j = 0; j < seqLen; j++) {
          scores[i * seqLen + j] = Math.exp(scores[i * seqLen + j]! - maxVal);
          sumExp += scores[i * seqLen + j]!;
        }
        for (let j = 0; j < seqLen; j++) {
          scores[i * seqLen + j]! /= sumExp;
        }
      }

      // Apply attention: out = scores @ V
      for (let i = 0; i < seqLen; i++) {
        for (let d = 0; d < dHead; d++) {
          let val = 0;
          for (let j = 0; j < seqLen; j++) {
            const vIdx = b * seqLen * dModel + j * dModel + h * dHead + d;
            val += scores[i * seqLen + j]! * V[vIdx]!;
          }
          const outIdx = b * seqLen * dModel + i * dModel + h * dHead + d;
          out[outIdx] = val;
        }
      }
    }
  }

  // Apply gating
  if (gating === "relu") {
    for (let i = 0; i < out.length; i++) {
      out[i] = Math.max(0.0, out[i]!);
    }
  } else if (gating === "swish") {
    for (let i = 0; i < out.length; i++) {
      out[i] = out[i]! * sigmoidStable(out[i]!);
    }
  }

  return out;
}

// -- Global pooling --------------------------------------------------------

export function gridGlobalPool(
  features: Float64Array,
  shape: linalg.Shape2D,
  gridH: number,
  gridW: number,
  method = "avg",
): { data: Float64Array; shape: linalg.Shape2D } {
  const [batch, totalFeatures] = shape;
  let channels = Math.floor(totalFeatures / (gridH * gridW));
  if (channels < 1) channels = 1;

  const outChannels = method === "avg_max" ? channels * 2 : channels;
  const out = new Float64Array(batch * outChannels);

  for (let b = 0; b < batch; b++) {
    for (let c = 0; c < channels; c++) {
      let sumVal = 0;
      let maxVal = -Infinity;
      for (let i = 0; i < gridH * gridW; i++) {
        const v = features[b * totalFeatures + c * gridH * gridW + i] ?? 0;
        sumVal += v;
        maxVal = Math.max(maxVal, v);
      }
      const avgVal = sumVal / (gridH * gridW);

      if (method === "avg") {
        out[b * outChannels + c] = avgVal;
      } else if (method === "max") {
        out[b * outChannels + c] = maxVal;
      } else {
        // avg_max
        out[b * outChannels + c] = avgVal;
        out[b * outChannels + channels + c] = maxVal;
      }
    }
  }

  return { data: out, shape: [batch, outChannels] };
}

// -- Kernel initialization -------------------------------------------------

export function generateOrthogonalKernels(
  nChannels: number,
  inChannels: number,
  seed = 42,
): Float64Array {
  // Orthogonal conv kernels via QR decomposition (Saxe et al. 2014).
  // Returns (nChannels * inChannels * 3 * 3) float64 array.
  const fanIn = inChannels * 9;
  const rows = Math.max(nChannels, fanIn);
  const cols = Math.max(nChannels, fanIn);

  // Generate random matrix
  let s = seed;
  function nextRand(): number {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    const u1 = (s >>> 0) / 0xffffffff;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    const u2 = (s >>> 0) / 0xffffffff;
    return Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
  }

  // Simple QR via Gram-Schmidt (sufficient for kernel init)
  const flat = new Float64Array(rows * cols);
  for (let i = 0; i < rows * cols; i++) flat[i] = nextRand();

  // Gram-Schmidt orthogonalization (on rows)
  const Q = new Float64Array(rows * cols);
  for (let i = 0; i < Math.min(rows, cols); i++) {
    // Copy row i
    for (let j = 0; j < cols; j++) Q[i * cols + j] = flat[i * cols + j]!;
    // Subtract projections
    for (let k = 0; k < i; k++) {
      let dot = 0;
      let normK = 0;
      for (let j = 0; j < cols; j++) {
        dot += Q[i * cols + j]! * Q[k * cols + j]!;
        normK += Q[k * cols + j]! * Q[k * cols + j]!;
      }
      if (normK > 1e-12) {
        const scale = dot / normK;
        for (let j = 0; j < cols; j++) {
          Q[i * cols + j]! -= scale * Q[k * cols + j]!;
        }
      }
    }
    // Normalize
    let norm = 0;
    for (let j = 0; j < cols; j++) norm += Q[i * cols + j]! * Q[i * cols + j]!;
    norm = Math.sqrt(norm);
    if (norm > 1e-12) {
      for (let j = 0; j < cols; j++) Q[i * cols + j]! /= norm;
    }
  }

  // Take first nChannels rows, first fanIn columns
  const kernels = new Float64Array(nChannels * inChannels * 9);
  const gain = Math.sqrt(2.0 / fanIn) * Math.sqrt(fanIn);
  for (let c = 0; c < nChannels; c++) {
    for (let f = 0; f < fanIn && f < cols; f++) {
      kernels[c * fanIn + f] = Q[c * cols + f]! * gain;
    }
  }

  return kernels;
}

export function generateGaborKernels(
  nChannels: number,
  inChannels: number,
  seed = 42,
): Float64Array {
  // Gabor filter bank with varied orientations, frequencies, and phases.
  // Returns (nChannels * inChannels * 3 * 3) float64 array.
  const ks = 3;
  const half = Math.floor(ks / 2);
  const nOrientations = 8;
  const frequencies = [0.5, 1.0, 1.5];
  const phases = [0.0, Math.PI / 2];
  const sigma = 1.0;
  const gamma = 0.5;

  const gaborFilters: Float64Array[] = [];
  for (let thetaIdx = 0; thetaIdx < nOrientations; thetaIdx++) {
    const theta = (thetaIdx * Math.PI) / nOrientations;
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);
    for (const freq of frequencies) {
      for (const phase of phases) {
        const g = new Float64Array(9);
        for (let yi = 0; yi < ks; yi++) {
          for (let xi = 0; xi < ks; xi++) {
            const x = xi - half;
            const y = yi - half;
            const xRot = x * cosT + y * sinT;
            const yRot = -x * sinT + y * cosT;
            const envelope = Math.exp(
              -(xRot * xRot + gamma * gamma * yRot * yRot) / (2 * sigma * sigma),
            );
            const sinusoid = Math.cos(2 * Math.PI * freq * xRot + phase);
            g[yi * ks + xi] = envelope * sinusoid;
          }
        }
        // Normalize to zero mean, unit norm
        let mean = 0;
        for (let i = 0; i < 9; i++) mean += g[i]! / 9;
        for (let i = 0; i < 9; i++) g[i]! -= mean;
        let norm = 0;
        for (let i = 0; i < 9; i++) norm += g[i]! * g[i]!;
        norm = Math.sqrt(norm);
        if (norm > 1e-6) {
          for (let i = 0; i < 9; i++) g[i]! /= norm;
        }
        gaborFilters.push(g);
      }
    }
  }

  const kernels = new Float64Array(nChannels * inChannels * 9);
  const nGabor = gaborFilters.length;

  let s = seed;
  function nextRand(): number {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    const u1 = (s >>> 0) / 0xffffffff;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    const u2 = (s >>> 0) / 0xffffffff;
    return Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
  }

  for (let c = 0; c < nChannels; c++) {
    if (c < nGabor) {
      for (let ic = 0; ic < inChannels; ic++) {
        const filter = gaborFilters[c]!;
        for (let k = 0; k < 9; k++) {
          kernels[c * inChannels * 9 + ic * 9 + k] = filter[k]!;
        }
      }
    } else {
      // Fill remaining with random Kaiming
      const fanIn = inChannels * 9;
      const std = Math.sqrt(2.0 / fanIn);
      for (let ic = 0; ic < inChannels; ic++) {
        for (let k = 0; k < 9; k++) {
          kernels[c * inChannels * 9 + ic * 9 + k] = nextRand() * std;
        }
      }
    }
  }

  return kernels;
}

export function generateKmeansKernels(
  nChannels: number,
  inChannels: number,
  trainingData: Float64Array,
  shapeData: linalg.Shape2D,
  gridH: number,
  gridW: number,
  seed = 42,
  nPatches = 10000,
  maxIter = 50,
): Float64Array {
  // Data-dependent conv kernels via k-means on image patches.
  // Returns (nChannels * inChannels * 3 * 3) float64 array.
  const nSamples = shapeData[0];
  const ks = 3;
  const half = Math.floor(ks / 2);
  const patchDim = inChannels * ks * ks;

  let s = seed;
  function nextInt(max: number): number {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (s >>> 0) % max;
  }

  // Extract random patches
  const patches = new Float64Array(nPatches * patchDim);
  for (let i = 0; i < nPatches; i++) {
    const imgIdx = nextInt(nSamples);
    const row = half + nextInt(gridH - ks + 1);
    const col = half + nextInt(gridW - ks + 1);
    for (let ic = 0; ic < inChannels; ic++) {
      for (let ki = 0; ki < ks; ki++) {
        for (let kj = 0; kj < ks; kj++) {
          const srcIdx =
            imgIdx * inChannels * gridH * gridW +
            ic * gridH * gridW +
            (row - half + ki) * gridW +
            (col - half + kj);
          patches[i * patchDim + ic * ks * ks + ki * ks + kj] =
            trainingData[srcIdx] ?? 0;
        }
      }
    }
  }

  // Normalize patches: zero mean, unit norm
  for (let i = 0; i < nPatches; i++) {
    let mean = 0;
    for (let j = 0; j < patchDim; j++) mean += patches[i * patchDim + j]! / patchDim;
    for (let j = 0; j < patchDim; j++) patches[i * patchDim + j]! -= mean;
    let norm = 0;
    for (let j = 0; j < patchDim; j++)
      norm += patches[i * patchDim + j]! * patches[i * patchDim + j]!;
    norm = Math.sqrt(norm);
    if (norm < 1e-6) norm = 1.0;
    for (let j = 0; j < patchDim; j++) patches[i * patchDim + j]! /= norm;
  }

  // K-means++ initialization
  const centroids = new Float64Array(nChannels * patchDim);
  const firstIdx = nextInt(nPatches);
  for (let j = 0; j < patchDim; j++) {
    centroids[j] = patches[firstIdx * patchDim + j]!;
  }

  for (let c = 1; c < nChannels; c++) {
    const dists = new Float64Array(nPatches);
    let totalDist = 0;
    for (let i = 0; i < nPatches; i++) {
      let minDist = Infinity;
      for (let cc = 0; cc < c; cc++) {
        let dist = 0;
        for (let j = 0; j < patchDim; j++) {
          const diff = patches[i * patchDim + j]! - centroids[cc * patchDim + j]!;
          dist += diff * diff;
        }
        minDist = Math.min(minDist, dist);
      }
      dists[i] = minDist;
      totalDist += minDist;
    }
    // Choose next centroid proportional to distance
    let target = (nextInt(1000000) / 1000000) * totalDist;
    let chosen = 0;
    for (let i = 0; i < nPatches; i++) {
      target -= dists[i]!;
      if (target <= 0) {
        chosen = i;
        break;
      }
    }
    for (let j = 0; j < patchDim; j++) {
      centroids[c * patchDim + j] = patches[chosen * patchDim + j]!;
    }
  }

  // Lloyd's algorithm
  const labels = new Int32Array(nPatches);
  for (let iter = 0; iter < maxIter; iter++) {
    // Assign patches to nearest centroid
    for (let i = 0; i < nPatches; i++) {
      let bestDist = Infinity;
      let bestC = 0;
      for (let c = 0; c < nChannels; c++) {
        let dist = 0;
        for (let j = 0; j < patchDim; j++) {
          const diff = patches[i * patchDim + j]! - centroids[c * patchDim + j]!;
          dist += diff * diff;
        }
        if (dist < bestDist) {
          bestDist = dist;
          bestC = c;
        }
      }
      labels[i] = bestC;
    }

    // Update centroids
    const newCentroids = new Float64Array(nChannels * patchDim);
    const counts = new Int32Array(nChannels);
    for (let i = 0; i < nPatches; i++) {
      const c = labels[i]!;
      counts[c]!++;
      for (let j = 0; j < patchDim; j++) {
        newCentroids[c * patchDim + j]! += patches[i * patchDim + j]!;
      }
    }
    for (let c = 0; c < nChannels; c++) {
      if (counts[c]! > 0) {
        for (let j = 0; j < patchDim; j++) {
          newCentroids[c * patchDim + j]! /= counts[c]!;
        }
      } else {
        for (let j = 0; j < patchDim; j++) {
          newCentroids[c * patchDim + j] = centroids[c * patchDim + j]!;
        }
      }
    }

    // Check convergence
    let maxDiff = 0;
    for (let i = 0; i < nChannels * patchDim; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(newCentroids[i]! - centroids[i]!));
    }
    for (let i = 0; i < nChannels * patchDim; i++) {
      centroids[i] = newCentroids[i]!;
    }
    if (maxDiff < 1e-6) break;
  }

  return centroids;
}

// -- V projection search ---------------------------------------------------

export function searchVProjection(
  X: Float64Array,
  shapeX: readonly [number, number, number], // (nTotal, seqLen, dModel)
  nHeads: number,
  wQ: Float64Array | null,
  wK: Float64Array | null,
  Y: Float64Array,
  shapeY: linalg.Shape2D,
  lam: number,
  gating = "none",
  nCandidates = 20,
  seed = 42,
): { bestWV: Float64Array; bestOutFlat: Float64Array } {
  const [nTotal, seqLen, dModel] = shapeX;
  let effectiveHeads = nHeads;
  if (dModel % effectiveHeads !== 0) effectiveHeads = 1;
  const nFlat = dModel * seqLen;

  // Generate candidate V projections
  let s = seed + 1000;
  function nextRand(): number {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    const u1 = (s >>> 0) / 0xffffffff;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    const u2 = (s >>> 0) / 0xffffffff;
    return Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
  }

  // Identity matrix as first candidate
  const candidates: Float64Array[] = [];
  const eye = new Float64Array(dModel * dModel);
  for (let i = 0; i < dModel; i++) eye[i * dModel + i] = 1.0;
  candidates.push(eye);

  // Random orthogonal candidates
  for (let c = 1; c < nCandidates; c++) {
    const M = new Float64Array(dModel * dModel);
    for (let i = 0; i < dModel * dModel; i++) M[i] = nextRand();
    candidates.push(M);
  }

  let bestAcc = -1.0;
  let bestWV = candidates[0]!;
  let bestOutFlat = new Float64Array(nTotal * nFlat);

  for (const wVCand of candidates) {
    // Forward pass with this V projection
    const outFlat = batchSelfAttention(
      X,
      shapeX,
      effectiveHeads,
      wQ,
      wK,
      wVCand,
      gating,
    );

    // Flatten to (nTotal, nFlat)
    // Ridge regression
    const { weights: Wt, bias: bt } = ridgeSolve(
      outFlat,
      [nTotal, nFlat],
      Y,
      shapeY,
      lam,
    );

    // Compute accuracy
    let correct = 0;
    const nClasses = shapeY[1];
    for (let i = 0; i < nTotal; i++) {
      let bestClass = 0;
      let bestScore = -Infinity;
      for (let j = 0; j < nClasses; j++) {
        let score = bt[j]!;
        for (let k = 0; k < nFlat; k++) {
          score += outFlat[i * nFlat + k]! * Wt[k * nClasses + j]!;
        }
        if (score > bestScore) {
          bestScore = score;
          bestClass = j;
        }
      }
      // Find true class
      let trueClass = 0;
      let trueMax = -Infinity;
      for (let j = 0; j < nClasses; j++) {
        if (Y[i * nClasses + j]! > trueMax) {
          trueMax = Y[i * nClasses + j]!;
          trueClass = j;
        }
      }
      if (bestClass === trueClass) correct++;
    }

    const acc = correct / nTotal;
    if (acc > bestAcc) {
      bestAcc = acc;
      bestWV = wVCand;
      bestOutFlat = new Float64Array(outFlat);
    }
  }

  return { bestWV, bestOutFlat };
}

// -- Helper for 3D matrix multiplication -----------------------------------

function matmul3d(
  X: Float64Array,
  batch: number,
  seqLen: number,
  dIn: number,
  W: Float64Array,
  _wRows: number,
  wCols: number,
): Float64Array {
  // X: (batch, seqLen, dIn), W: (dIn, wCols)
  // Out: (batch, seqLen, wCols)
  const out = new Float64Array(batch * seqLen * wCols);
  for (let b = 0; b < batch; b++) {
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < wCols; j++) {
        let val = 0;
        for (let k = 0; k < dIn; k++) {
          val += X[b * seqLen * dIn + i * dIn + k]! * W[k * wCols + j]!;
        }
        out[b * seqLen * wCols + i * wCols + j] = val;
      }
    }
  }
  return out;
}
