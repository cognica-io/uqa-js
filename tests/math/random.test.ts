import { describe, expect, it } from "vitest";
import { SeededRandom } from "../../src/math/random.js";

describe("SeededRandom", () => {
  describe("determinism", () => {
    it("same seed produces same sequence", () => {
      const a = new SeededRandom(42);
      const b = new SeededRandom(42);
      for (let i = 0; i < 100; i++) {
        expect(a.random()).toBe(b.random());
      }
    });

    it("different seeds produce different sequences", () => {
      const a = new SeededRandom(42);
      const b = new SeededRandom(43);
      let allSame = true;
      for (let i = 0; i < 10; i++) {
        if (a.random() !== b.random()) allSame = false;
      }
      expect(allSame).toBe(false);
    });
  });

  describe("random()", () => {
    it("values are in [0, 1)", () => {
      const rng = new SeededRandom(123);
      for (let i = 0; i < 1000; i++) {
        const v = rng.random();
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(1);
      }
    });
  });

  describe("randn()", () => {
    it("mean is approximately 0 and variance approximately 1", () => {
      const rng = new SeededRandom(42);
      const n = 10000;
      let sumVal = 0;
      let sumSq = 0;
      for (let i = 0; i < n; i++) {
        const v = rng.randn();
        sumVal += v;
        sumSq += v * v;
      }
      const meanVal = sumVal / n;
      const variance = sumSq / n - meanVal * meanVal;
      expect(meanVal).toBeCloseTo(0, 1); // within 0.05
      expect(variance).toBeCloseTo(1, 0); // within 0.5
    });
  });

  describe("randnArray()", () => {
    it("returns correct length", () => {
      const rng = new SeededRandom(42);
      const arr = rng.randnArray(50);
      expect(arr.length).toBe(50);
      expect(arr).toBeInstanceOf(Float64Array);
    });
  });

  describe("randomArray()", () => {
    it("returns correct length with values in [0, 1)", () => {
      const rng = new SeededRandom(42);
      const arr = rng.randomArray(100);
      expect(arr.length).toBe(100);
      for (let i = 0; i < arr.length; i++) {
        expect(arr[i]).toBeGreaterThanOrEqual(0);
        expect(arr[i]).toBeLessThan(1);
      }
    });
  });

  describe("choice()", () => {
    it("returns correct number of elements", () => {
      const rng = new SeededRandom(42);
      const result = rng.choice([1, 2, 3, 4, 5], 3);
      expect(result.length).toBe(3);
    });

    it("returns unique elements (without replacement)", () => {
      const rng = new SeededRandom(42);
      const result = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5);
      expect(new Set(result).size).toBe(5);
    });

    it("throws when n > array length", () => {
      const rng = new SeededRandom(42);
      expect(() => rng.choice([1, 2], 3)).toThrow("n (3) > array length (2)");
    });
  });
});
