//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- linear algebra utilities (replaces numpy)
// All operations use Float64Array. No mutation of input arrays.

export type Shape2D = readonly [number, number];

export interface MatrixResult {
  readonly data: Float64Array;
  readonly shape: Shape2D;
}

// -- Scalar results ----------------------------------------------------------

export function dot(a: Float64Array, b: Float64Array): number {
  if (a.length !== b.length) {
    throw new Error(
      `dot: length mismatch (${String(a.length)} vs ${String(b.length)})`,
    );
  }
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s += a[i]! * b[i]!;
  }
  return s;
}

export function norm(v: Float64Array): number {
  return Math.sqrt(dot(v, v));
}

export function cosine(a: Float64Array, b: Float64Array): number {
  const na = norm(a);
  const nb = norm(b);
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / (na * nb);
}

export function sum(a: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s += a[i]!;
  }
  return s;
}

export function mean(a: Float64Array): number {
  if (a.length === 0) {
    throw new Error("mean: empty array");
  }
  return sum(a) / a.length;
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function argmax(a: Float64Array): number {
  if (a.length === 0) return 0;
  let best = 0;
  let bestVal = a[0]!;
  for (let i = 1; i < a.length; i++) {
    if (a[i]! > bestVal) {
      bestVal = a[i]!;
      best = i;
    }
  }
  return best;
}

// -- Vector -> Vector --------------------------------------------------------

function assertSameLength(op: string, a: Float64Array, b: Float64Array): void {
  if (a.length !== b.length) {
    throw new Error(
      `${op}: length mismatch (${String(a.length)} vs ${String(b.length)})`,
    );
  }
}

export function add(a: Float64Array, b: Float64Array): Float64Array {
  assertSameLength("add", a, b);
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i]! + b[i]!;
  }
  return out;
}

export function sub(a: Float64Array, b: Float64Array): Float64Array {
  assertSameLength("sub", a, b);
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i]! - b[i]!;
  }
  return out;
}

export function mul(a: Float64Array, b: Float64Array): Float64Array {
  assertSameLength("mul", a, b);
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i]! * b[i]!;
  }
  return out;
}

export function div(a: Float64Array, b: Float64Array): Float64Array {
  assertSameLength("div", a, b);
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i]! / b[i]!;
  }
  return out;
}

export function scale(a: Float64Array, scalar: number): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i]! * scalar;
  }
  return out;
}

export function exp(a: Float64Array): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = Math.exp(a[i]!);
  }
  return out;
}

export function log(a: Float64Array): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = Math.log(a[i]!);
  }
  return out;
}

export function softmax(a: Float64Array): Float64Array {
  if (a.length === 0) return new Float64Array(0);
  // Subtract max for numerical stability
  let mx = a[0]!;
  for (let i = 1; i < a.length; i++) {
    if (a[i]! > mx) mx = a[i]!;
  }
  const out = new Float64Array(a.length);
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    out[i] = Math.exp(a[i]! - mx);
    total += out[i]!;
  }
  for (let i = 0; i < out.length; i++) {
    out[i] = out[i]! / total;
  }
  return out;
}

export function clip(a: Float64Array, lo: number, hi: number): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = Math.max(lo, Math.min(hi, a[i]!));
  }
  return out;
}

// -- Index operations --------------------------------------------------------

export function argsort(a: Float64Array): number[] {
  const indices = Array.from({ length: a.length }, (_, i) => i);
  indices.sort((x, y) => a[x]! - a[y]!);
  return indices;
}

// -- Constructors ------------------------------------------------------------

export function ones(n: number): Float64Array {
  const out = new Float64Array(n);
  out.fill(1.0);
  return out;
}

export function zeros(n: number): Float64Array {
  return new Float64Array(n);
}

// -- Matrix operations -------------------------------------------------------

export function matmul(
  a: Float64Array,
  shapeA: Shape2D,
  b: Float64Array,
  shapeB: Shape2D,
): MatrixResult {
  const [rowsA, colsA] = shapeA;
  const [rowsB, colsB] = shapeB;
  if (colsA !== rowsB) {
    throw new Error(
      `matmul: inner dimension mismatch (${String(colsA)} vs ${String(rowsB)})`,
    );
  }
  if (a.length !== rowsA * colsA) {
    throw new Error(
      `matmul: array a length (${String(a.length)}) does not match shape [${String(rowsA)}, ${String(colsA)}]`,
    );
  }
  if (b.length !== rowsB * colsB) {
    throw new Error(
      `matmul: array b length (${String(b.length)}) does not match shape [${String(rowsB)}, ${String(colsB)}]`,
    );
  }
  const out = new Float64Array(rowsA * colsB);
  for (let i = 0; i < rowsA; i++) {
    for (let k = 0; k < colsA; k++) {
      const aik = a[i * colsA + k]!;
      for (let j = 0; j < colsB; j++) {
        out[i * colsB + j] = out[i * colsB + j]! + aik * b[k * colsB + j]!;
      }
    }
  }
  return { data: out, shape: [rowsA, colsB] };
}

export function transpose(a: Float64Array, rows: number, cols: number): Float64Array {
  if (a.length !== rows * cols) {
    throw new Error(
      `transpose: array length (${String(a.length)}) does not match shape [${String(rows)}, ${String(cols)}]`,
    );
  }
  const out = new Float64Array(a.length);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[j * rows + i] = a[i * cols + j]!;
    }
  }
  return out;
}
