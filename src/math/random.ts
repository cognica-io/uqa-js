//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- seedable PRNG (replaces numpy.random)
// xoshiro256** with SplitMix64 seed expansion

// SplitMix64: expand a single seed into four 64-bit state values
function splitmix64(seed: bigint): bigint {
  let z = seed + 0x9e3779b97f4a7c15n;
  z = (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n;
  z = (z ^ (z >> 27n)) * 0x94d049bb133111ebn;
  return (z ^ (z >> 31n)) & 0xffffffffffffffffn;
}

function rotl(x: bigint, k: bigint): bigint {
  return ((x << k) | (x >> (64n - k))) & 0xffffffffffffffffn;
}

export class SeededRandom {
  private s0: bigint;
  private s1: bigint;
  private s2: bigint;
  private s3: bigint;
  private cachedNormal: number | null = null;

  constructor(seed: number) {
    // Expand seed via SplitMix64
    let sm = BigInt(Math.floor(seed)) & 0xffffffffffffffffn;
    sm = splitmix64(sm);
    this.s0 = splitmix64(sm);
    sm = splitmix64(sm + 1n);
    this.s1 = splitmix64(sm);
    sm = splitmix64(sm + 1n);
    this.s2 = splitmix64(sm);
    sm = splitmix64(sm + 1n);
    this.s3 = splitmix64(sm);
  }

  // xoshiro256** core step, returns u64
  private next(): bigint {
    const result = (rotl(this.s1 * 5n, 7n) * 9n) & 0xffffffffffffffffn;
    const t = (this.s1 << 17n) & 0xffffffffffffffffn;
    this.s2 ^= this.s0;
    this.s3 ^= this.s1;
    this.s1 ^= this.s2;
    this.s0 ^= this.s3;
    this.s2 ^= t;
    this.s3 = rotl(this.s3, 45n);
    return result;
  }

  // Uniform [0, 1) with 53 bits of precision
  random(): number {
    const bits = this.next() >> 11n; // top 53 bits
    return Number(bits) / 9007199254740992; // 2^53
  }

  // Standard normal via Box-Muller transform
  randn(): number {
    if (this.cachedNormal !== null) {
      const val = this.cachedNormal;
      this.cachedNormal = null;
      return val;
    }
    let u1: number;
    do {
      u1 = this.random();
    } while (u1 === 0); // avoid log(0)
    const u2 = this.random();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    this.cachedNormal = r * Math.sin(theta);
    return r * Math.cos(theta);
  }

  randnArray(n: number): Float64Array {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = this.randn();
    }
    return out;
  }

  randomArray(n: number): Float64Array {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = this.random();
    }
    return out;
  }

  choice<T>(arr: readonly T[], n: number): T[] {
    if (n > arr.length) {
      throw new Error(
        `choice: n (${String(n)}) > array length (${String(arr.length)})`,
      );
    }
    // Fisher-Yates partial shuffle on a copy
    const copy = arr.slice();
    for (let i = 0; i < n; i++) {
      const j = i + Math.floor(this.random() * (copy.length - i));
      const tmp = copy[i]!;
      copy[i] = copy[j]!;
      copy[j] = tmp;
    }
    return copy.slice(0, n);
  }
}
