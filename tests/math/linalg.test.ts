import { describe, expect, it } from "vitest";
import {
  add,
  argmax,
  argsort,
  clip,
  cosine,
  div,
  dot,
  exp,
  log,
  matmul,
  mean,
  mul,
  norm,
  ones,
  scale,
  sigmoid,
  softmax,
  sub,
  sum,
  transpose,
  zeros,
} from "../../src/math/linalg.js";

describe("dot", () => {
  it("computes dot product of known vectors", () => {
    const a = new Float64Array([1, 2, 3]);
    const b = new Float64Array([4, 5, 6]);
    expect(dot(a, b)).toBe(32);
  });

  it("returns 0 for orthogonal vectors", () => {
    const a = new Float64Array([1, 0]);
    const b = new Float64Array([0, 1]);
    expect(dot(a, b)).toBe(0);
  });

  it("throws on length mismatch", () => {
    expect(() => dot(new Float64Array([1]), new Float64Array([1, 2]))).toThrow(
      "length mismatch",
    );
  });
});

describe("norm", () => {
  it("computes L2 norm", () => {
    expect(norm(new Float64Array([3, 4]))).toBe(5);
  });

  it("returns 0 for zero vector", () => {
    expect(norm(new Float64Array([0, 0, 0]))).toBe(0);
  });
});

describe("cosine", () => {
  it("returns 1 for identical vectors", () => {
    const v = new Float64Array([1, 2, 3]);
    expect(cosine(v, v)).toBeCloseTo(1.0, 10);
  });

  it("returns 0 for orthogonal vectors", () => {
    const a = new Float64Array([1, 0]);
    const b = new Float64Array([0, 1]);
    expect(cosine(a, b)).toBeCloseTo(0.0, 10);
  });

  it("returns 0 when either vector is zero", () => {
    const z = new Float64Array([0, 0]);
    const v = new Float64Array([1, 2]);
    expect(cosine(z, v)).toBe(0);
    expect(cosine(v, z)).toBe(0);
  });
});

describe("sum", () => {
  it("sums elements", () => {
    expect(sum(new Float64Array([1, 2, 3, 4]))).toBe(10);
  });

  it("returns 0 for empty array", () => {
    expect(sum(new Float64Array(0))).toBe(0);
  });
});

describe("mean", () => {
  it("computes mean", () => {
    expect(mean(new Float64Array([2, 4, 6]))).toBeCloseTo(4.0, 10);
  });

  it("throws on empty array", () => {
    expect(() => mean(new Float64Array(0))).toThrow("empty");
  });
});

describe("sigmoid", () => {
  it("sigmoid(0) = 0.5", () => {
    expect(sigmoid(0)).toBe(0.5);
  });

  it("large positive -> ~1", () => {
    expect(sigmoid(100)).toBeCloseTo(1.0, 10);
  });

  it("large negative -> ~0", () => {
    expect(sigmoid(-100)).toBeCloseTo(0.0, 10);
  });
});

describe("argmax", () => {
  it("returns index of max element", () => {
    expect(argmax(new Float64Array([1, 5, 3, 2]))).toBe(1);
  });

  it("returns first index on tie", () => {
    expect(argmax(new Float64Array([5, 5, 3]))).toBe(0);
  });

  it("returns 0 for empty array", () => {
    expect(argmax(new Float64Array(0))).toBe(0);
  });
});

describe("add", () => {
  it("adds element-wise", () => {
    const r = add(new Float64Array([1, 2]), new Float64Array([3, 4]));
    expect([...r]).toEqual([4, 6]);
  });

  it("throws on length mismatch", () => {
    expect(() => add(new Float64Array([1]), new Float64Array([1, 2]))).toThrow(
      "length mismatch",
    );
  });
});

describe("sub", () => {
  it("subtracts element-wise", () => {
    const r = sub(new Float64Array([5, 3]), new Float64Array([1, 2]));
    expect([...r]).toEqual([4, 1]);
  });
});

describe("mul", () => {
  it("multiplies element-wise", () => {
    const r = mul(new Float64Array([2, 3]), new Float64Array([4, 5]));
    expect([...r]).toEqual([8, 15]);
  });
});

describe("div", () => {
  it("divides element-wise", () => {
    const r = div(new Float64Array([6, 10]), new Float64Array([2, 5]));
    expect([...r]).toEqual([3, 2]);
  });

  it("produces Infinity for division by zero", () => {
    const r = div(new Float64Array([1]), new Float64Array([0]));
    expect(r[0]).toBe(Infinity);
  });
});

describe("scale", () => {
  it("scales by scalar", () => {
    const r = scale(new Float64Array([1, 2, 3]), 2);
    expect([...r]).toEqual([2, 4, 6]);
  });
});

describe("exp", () => {
  it("applies Math.exp element-wise", () => {
    const r = exp(new Float64Array([0, 1]));
    expect(r[0]).toBeCloseTo(1.0, 10);
    expect(r[1]).toBeCloseTo(Math.E, 10);
  });
});

describe("log", () => {
  it("applies Math.log element-wise", () => {
    const r = log(new Float64Array([1, Math.E]));
    expect(r[0]).toBeCloseTo(0.0, 10);
    expect(r[1]).toBeCloseTo(1.0, 10);
  });
});

describe("softmax", () => {
  it("uniform for equal inputs", () => {
    const r = softmax(new Float64Array([1, 1, 1]));
    for (let i = 0; i < 3; i++) {
      expect(r[i]).toBeCloseTo(1 / 3, 10);
    }
  });

  it("sums to 1", () => {
    const r = softmax(new Float64Array([1, 2, 3, 4]));
    expect(sum(r)).toBeCloseTo(1.0, 10);
  });

  it("handles empty array", () => {
    const r = softmax(new Float64Array(0));
    expect(r.length).toBe(0);
  });

  it("handles large values without overflow", () => {
    const r = softmax(new Float64Array([1000, 1001, 1002]));
    expect(sum(r)).toBeCloseTo(1.0, 10);
    expect(r[2]).toBeGreaterThan(r[1]!);
    expect(r[1]).toBeGreaterThan(r[0]!);
  });
});

describe("clip", () => {
  it("clips values to range", () => {
    const r = clip(new Float64Array([-1, 0.5, 2]), 0, 1);
    expect([...r]).toEqual([0, 0.5, 1]);
  });
});

describe("argsort", () => {
  it("returns sorted indices ascending", () => {
    expect(argsort(new Float64Array([3, 1, 2]))).toEqual([1, 2, 0]);
  });

  it("handles ties", () => {
    const result = argsort(new Float64Array([2, 1, 2]));
    expect(result[0]).toBe(1); // smallest is index 1
  });
});

describe("ones", () => {
  it("creates array of ones", () => {
    const r = ones(3);
    expect([...r]).toEqual([1, 1, 1]);
  });
});

describe("zeros", () => {
  it("creates array of zeros", () => {
    const r = zeros(3);
    expect([...r]).toEqual([0, 0, 0]);
  });
});

describe("matmul", () => {
  it("multiplies 2x3 by 3x2", () => {
    // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    const a = new Float64Array([1, 2, 3, 4, 5, 6]);
    const b = new Float64Array([7, 8, 9, 10, 11, 12]);
    const r = matmul(a, [2, 3], b, [3, 2]);
    expect(r.shape).toEqual([2, 2]);
    expect(r.data[0]).toBeCloseTo(58, 10);
    expect(r.data[1]).toBeCloseTo(64, 10);
    expect(r.data[2]).toBeCloseTo(139, 10);
    expect(r.data[3]).toBeCloseTo(154, 10);
  });

  it("identity matrix preserves input", () => {
    const a = new Float64Array([1, 2, 3, 4]);
    const eye = new Float64Array([1, 0, 0, 1]);
    const r = matmul(a, [2, 2], eye, [2, 2]);
    expect([...r.data]).toEqual([1, 2, 3, 4]);
  });

  it("throws on dimension mismatch", () => {
    expect(() =>
      matmul(new Float64Array([1, 2]), [1, 2], new Float64Array([1, 2]), [1, 2]),
    ).toThrow("inner dimension mismatch");
  });
});

describe("transpose", () => {
  it("transposes 2x3 matrix", () => {
    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    const a = new Float64Array([1, 2, 3, 4, 5, 6]);
    const r = transpose(a, 2, 3);
    expect([...r]).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it("transposes 1xN", () => {
    const a = new Float64Array([1, 2, 3]);
    const r = transpose(a, 1, 3);
    expect([...r]).toEqual([1, 2, 3]);
  });

  it("throws on shape mismatch", () => {
    expect(() => transpose(new Float64Array([1, 2]), 2, 2)).toThrow(
      "does not match shape",
    );
  });
});
