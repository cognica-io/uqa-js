import { describe, expect, it } from "vitest";
import { SeededRandom } from "../../src/math/random.js";
import type { PostingEntry } from "../../src/core/types.js";
import { createPayload } from "../../src/core/types.js";
import { GeneralizedPostingList, PostingList } from "../../src/core/posting-list.js";

// -- Helpers -----------------------------------------------------------------

function pl(docIds: number[], scores?: number[]): PostingList {
  const entries: PostingEntry[] = docIds.map((id, i) => ({
    docId: id,
    payload: createPayload({ score: scores?.[i] ?? 0 }),
  }));
  return new PostingList(entries);
}

function ids(posting: PostingList): number[] {
  return posting.entries.map((e) => e.docId);
}

function isSorted(posting: PostingList): boolean {
  const e = posting.entries;
  for (let i = 1; i < e.length; i++) {
    if (e[i]!.docId <= e[i - 1]!.docId) return false;
  }
  return true;
}

function makeUniversal(maxDocId: number): PostingList {
  const entries: PostingEntry[] = [];
  for (let i = 0; i <= maxDocId; i++) {
    entries.push({ docId: i, payload: createPayload() });
  }
  return new PostingList(entries);
}

function makeRandomPostingList(
  rng: SeededRandom,
  maxDocId: number,
  maxSize: number,
): PostingList {
  const size = Math.floor(rng.random() * maxSize);
  const entries: PostingEntry[] = [];
  for (let i = 0; i < size; i++) {
    const docId = Math.floor(rng.random() * (maxDocId + 1));
    entries.push({
      docId,
      payload: createPayload({ score: rng.random() * 10 }),
    });
  }
  return new PostingList(entries);
}

// -- Constructor & basics ----------------------------------------------------

describe("PostingList", () => {
  describe("constructor", () => {
    it("sorts entries by docId", () => {
      const p = pl([3, 1, 2]);
      expect(ids(p)).toEqual([1, 2, 3]);
    });

    it("deduplicates keeping first occurrence", () => {
      const p = pl([1, 2, 2, 3]);
      expect(ids(p)).toEqual([1, 2, 3]);
    });

    it("creates empty list with no args", () => {
      const p = new PostingList();
      expect(p.length).toBe(0);
    });
  });

  describe("fromSorted", () => {
    it("bypasses sort", () => {
      const entries: PostingEntry[] = [
        { docId: 1, payload: createPayload() },
        { docId: 5, payload: createPayload() },
        { docId: 10, payload: createPayload() },
      ];
      const p = PostingList.fromSorted(entries);
      expect(ids(p)).toEqual([1, 5, 10]);
    });
  });

  // -- Boolean Algebra Axioms ------------------------------------------------

  describe("Boolean Algebra Axioms (100 random trials each)", () => {
    const MAX_DOC_ID = 30;
    const MAX_SIZE = 15;
    const TRIALS = 100;
    const U = makeUniversal(MAX_DOC_ID);

    describe("A1: Commutativity", () => {
      it("union: A | B = B | A", () => {
        for (let t = 0; t < TRIALS; t++) {
          const rng = new SeededRandom(1000 + t);
          const a = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          const b = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          expect(a.union(b).equals(b.union(a))).toBe(true);
        }
      });

      it("intersect: A & B = B & A", () => {
        for (let t = 0; t < TRIALS; t++) {
          const rng = new SeededRandom(2000 + t);
          const a = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          const b = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          expect(a.intersect(b).equals(b.intersect(a))).toBe(true);
        }
      });
    });

    describe("A2: Associativity", () => {
      it("union: (A | B) | C = A | (B | C)", () => {
        for (let t = 0; t < TRIALS; t++) {
          const rng = new SeededRandom(3000 + t);
          const a = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          const b = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          const c = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          expect(
            a
              .union(b)
              .union(c)
              .equals(a.union(b.union(c))),
          ).toBe(true);
        }
      });

      it("intersect: (A & B) & C = A & (B & C)", () => {
        for (let t = 0; t < TRIALS; t++) {
          const rng = new SeededRandom(4000 + t);
          const a = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          const b = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          const c = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          expect(
            a
              .intersect(b)
              .intersect(c)
              .equals(a.intersect(b.intersect(c))),
          ).toBe(true);
        }
      });
    });

    describe("A3: Distributivity", () => {
      it("intersect over union: A & (B | C) = (A & B) | (A & C)", () => {
        for (let t = 0; t < TRIALS; t++) {
          const rng = new SeededRandom(5000 + t);
          const a = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          const b = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          const c = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          const lhs = a.intersect(b.union(c));
          const rhs = a.intersect(b).union(a.intersect(c));
          expect(lhs.equals(rhs)).toBe(true);
        }
      });

      it("union over intersect: A | (B & C) = (A | B) & (A | C)", () => {
        for (let t = 0; t < TRIALS; t++) {
          const rng = new SeededRandom(6000 + t);
          const a = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          const b = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          const c = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          const lhs = a.union(b.intersect(c));
          const rhs = a.union(b).intersect(a.union(c));
          expect(lhs.equals(rhs)).toBe(true);
        }
      });
    });

    describe("A4: Identity", () => {
      it("union with empty: A | {} = A", () => {
        const empty = new PostingList();
        for (let t = 0; t < TRIALS; t++) {
          const rng = new SeededRandom(7000 + t);
          const a = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          expect(a.union(empty).equals(a)).toBe(true);
        }
      });

      it("intersect with universal: A & U = A", () => {
        for (let t = 0; t < TRIALS; t++) {
          const rng = new SeededRandom(8000 + t);
          const a = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          expect(a.intersect(U).equals(a)).toBe(true);
        }
      });
    });

    describe("A5: Complement", () => {
      it("A | ~A = U", () => {
        for (let t = 0; t < TRIALS; t++) {
          const rng = new SeededRandom(9000 + t);
          const a = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          const notA = a.complement(U);
          expect(a.union(notA).equals(U)).toBe(true);
        }
      });

      it("A & ~A = empty", () => {
        const empty = new PostingList();
        for (let t = 0; t < TRIALS; t++) {
          const rng = new SeededRandom(10000 + t);
          const a = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
          const notA = a.complement(U);
          expect(a.intersect(notA).equals(empty)).toBe(true);
        }
      });
    });
  });

  // -- De Morgan's Laws ------------------------------------------------------

  describe("De Morgan's Laws", () => {
    const MAX_DOC_ID = 30;
    const MAX_SIZE = 15;
    const U = makeUniversal(MAX_DOC_ID);

    it("~(A & B) = ~A | ~B", () => {
      for (let t = 0; t < 50; t++) {
        const rng = new SeededRandom(11000 + t);
        const a = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
        const b = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
        const lhs = a.intersect(b).complement(U);
        const rhs = a.complement(U).union(b.complement(U));
        expect(lhs.equals(rhs)).toBe(true);
      }
    });

    it("~(A | B) = ~A & ~B", () => {
      for (let t = 0; t < 50; t++) {
        const rng = new SeededRandom(12000 + t);
        const a = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
        const b = makeRandomPostingList(rng, MAX_DOC_ID, MAX_SIZE);
        const lhs = a.union(b).complement(U);
        const rhs = a.complement(U).intersect(b.complement(U));
        expect(lhs.equals(rhs)).toBe(true);
      }
    });
  });

  // -- Sorted invariant ------------------------------------------------------

  describe("sorted invariant", () => {
    it("after union", () => {
      expect(isSorted(pl([1, 5, 9]).union(pl([2, 5, 8])))).toBe(true);
    });

    it("after intersect", () => {
      expect(isSorted(pl([1, 3, 5, 7]).intersect(pl([2, 3, 5, 9])))).toBe(true);
    });

    it("after difference", () => {
      expect(isSorted(pl([1, 2, 3, 4, 5]).difference(pl([2, 4])))).toBe(true);
    });

    it("after complement", () => {
      expect(isSorted(pl([1, 3]).complement(makeUniversal(5)))).toBe(true);
    });
  });

  // -- mergePayloads ---------------------------------------------------------

  describe("mergePayloads", () => {
    it("unions and sorts positions", () => {
      const a = createPayload({ positions: [1, 3, 5] });
      const b = createPayload({ positions: [2, 3, 6] });
      const merged = PostingList.mergePayloads(a, b);
      expect(merged.positions).toEqual([1, 2, 3, 5, 6]);
    });

    it("sums scores", () => {
      const a = createPayload({ score: 1.5 });
      const b = createPayload({ score: 2.5 });
      const merged = PostingList.mergePayloads(a, b);
      expect(merged.score).toBe(4.0);
    });

    it("merges fields (b overwrites a)", () => {
      const a = createPayload({ fields: { x: 1, y: 2 } });
      const b = createPayload({ fields: { y: 3, z: 4 } });
      const merged = PostingList.mergePayloads(a, b);
      expect(merged.fields).toEqual({ x: 1, y: 3, z: 4 });
    });
  });

  // -- topK ------------------------------------------------------------------

  describe("topK", () => {
    it("returns k highest-scored entries", () => {
      const p = pl([1, 2, 3, 4, 5], [10, 50, 30, 20, 40]);
      const top = p.topK(3);
      expect(top.length).toBe(3);
      // Should contain docIds 2 (50), 5 (40), 3 (30)
      const topIds = new Set(ids(top));
      expect(topIds.has(2)).toBe(true);
      expect(topIds.has(5)).toBe(true);
      expect(topIds.has(3)).toBe(true);
    });

    it("k >= length returns all", () => {
      const p = pl([1, 2, 3]);
      const top = p.topK(10);
      expect(top.length).toBe(3);
    });
  });

  // -- getEntry (binary search) ----------------------------------------------

  describe("getEntry", () => {
    it("finds existing entry", () => {
      const p = pl([1, 5, 10, 20, 50]);
      const entry = p.getEntry(10);
      expect(entry).not.toBeNull();
      expect(entry!.docId).toBe(10);
    });

    it("returns null for missing entry", () => {
      const p = pl([1, 5, 10, 20, 50]);
      expect(p.getEntry(7)).toBeNull();
    });

    it("works on empty list", () => {
      const p = new PostingList();
      expect(p.getEntry(1)).toBeNull();
    });

    it("finds first and last elements", () => {
      const p = pl([1, 5, 10]);
      expect(p.getEntry(1)!.docId).toBe(1);
      expect(p.getEntry(10)!.docId).toBe(10);
    });
  });

  // -- withScores ------------------------------------------------------------

  describe("withScores", () => {
    it("applies score function", () => {
      const p = pl([1, 2, 3], [10, 20, 30]);
      const doubled = p.withScores((e) => e.payload.score * 2);
      const scores = doubled.entries.map((e) => e.payload.score);
      expect(scores).toEqual([20, 40, 60]);
    });
  });

  // -- difference ------------------------------------------------------------

  describe("difference", () => {
    it("removes entries in other", () => {
      const a = pl([1, 2, 3, 4, 5]);
      const b = pl([2, 4]);
      expect(ids(a.difference(b))).toEqual([1, 3, 5]);
    });

    it("returns empty when subtracting self", () => {
      const a = pl([1, 2, 3]);
      expect(a.difference(a).length).toBe(0);
    });
  });

  // -- iteration and misc ----------------------------------------------------

  describe("iteration", () => {
    it("Symbol.iterator yields all entries", () => {
      const p = pl([3, 1, 2]);
      const collected: number[] = [];
      for (const e of p) {
        collected.push(e.docId);
      }
      expect(collected).toEqual([1, 2, 3]);
    });
  });

  describe("equals", () => {
    it("same doc ids are equal", () => {
      expect(pl([1, 2, 3]).equals(pl([1, 2, 3]))).toBe(true);
    });

    it("different doc ids are not equal", () => {
      expect(pl([1, 2, 3]).equals(pl([1, 2, 4]))).toBe(false);
    });

    it("different lengths are not equal", () => {
      expect(pl([1, 2]).equals(pl([1, 2, 3]))).toBe(false);
    });
  });

  describe("toString", () => {
    it("produces readable representation", () => {
      expect(pl([1, 2, 3]).toString()).toBe("PostingList([1, 2, 3])");
    });
  });

  describe("aliases", () => {
    it("and() is intersect()", () => {
      const a = pl([1, 2, 3]);
      const b = pl([2, 3, 4]);
      expect(a.and(b).equals(a.intersect(b))).toBe(true);
    });

    it("or() is union()", () => {
      const a = pl([1, 2]);
      const b = pl([3, 4]);
      expect(a.or(b).equals(a.union(b))).toBe(true);
    });

    it("sub() is difference()", () => {
      const a = pl([1, 2, 3]);
      const b = pl([2]);
      expect(a.sub(b).equals(a.difference(b))).toBe(true);
    });
  });
});

// -- GeneralizedPostingList --------------------------------------------------

describe("GeneralizedPostingList", () => {
  function gpl(tuples: number[][]): GeneralizedPostingList {
    return new GeneralizedPostingList(
      tuples.map((t) => ({
        docIds: t,
        payload: createPayload(),
      })),
    );
  }

  function gplIds(g: GeneralizedPostingList): number[][] {
    return g.entries.map((e) => [...e.docIds]);
  }

  describe("constructor", () => {
    it("sorts by docIds lexicographically", () => {
      const g = gpl([
        [2, 1],
        [1, 2],
        [1, 1],
      ]);
      expect(gplIds(g)).toEqual([
        [1, 1],
        [1, 2],
        [2, 1],
      ]);
    });
  });

  describe("union", () => {
    it("merges and deduplicates", () => {
      const a = gpl([
        [1, 2],
        [3, 4],
      ]);
      const b = gpl([
        [2, 3],
        [3, 4],
      ]);
      const result = a.union(b);
      expect(gplIds(result)).toEqual([
        [1, 2],
        [2, 3],
        [3, 4],
      ]);
    });
  });

  describe("intersect", () => {
    it("keeps only shared tuples", () => {
      const a = gpl([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const b = gpl([
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      expect(gplIds(a.intersect(b))).toEqual([
        [3, 4],
        [5, 6],
      ]);
    });
  });

  describe("difference", () => {
    it("removes tuples in other", () => {
      const a = gpl([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const b = gpl([[3, 4]]);
      expect(gplIds(a.difference(b))).toEqual([
        [1, 2],
        [5, 6],
      ]);
    });
  });

  describe("complement", () => {
    it("U - A", () => {
      const u = gpl([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const a = gpl([[3, 4]]);
      expect(gplIds(a.complement(u))).toEqual([
        [1, 2],
        [5, 6],
      ]);
    });
  });

  describe("docIdsSet", () => {
    it("returns set of all docIds keys", () => {
      const g = gpl([
        [1, 2],
        [3, 4],
      ]);
      const s = g.docIdsSet;
      expect(s.size).toBe(2);
      expect(s.has("1\x002")).toBe(true);
      expect(s.has("3\x004")).toBe(true);
    });
  });

  describe("equals", () => {
    it("same tuples are equal", () => {
      expect(
        gpl([
          [1, 2],
          [3, 4],
        ]).equals(
          gpl([
            [1, 2],
            [3, 4],
          ]),
        ),
      ).toBe(true);
    });

    it("different tuples are not equal", () => {
      expect(
        gpl([
          [1, 2],
          [3, 4],
        ]).equals(
          gpl([
            [1, 2],
            [3, 5],
          ]),
        ),
      ).toBe(false);
    });
  });

  describe("iteration", () => {
    it("Symbol.iterator yields all entries", () => {
      const g = gpl([
        [1, 2],
        [3, 4],
      ]);
      const collected: number[][] = [];
      for (const e of g) {
        collected.push([...e.docIds]);
      }
      expect(collected).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });
  });

  describe("aliases", () => {
    it("and/or/sub work correctly", () => {
      const a = gpl([
        [1, 2],
        [3, 4],
      ]);
      const b = gpl([
        [3, 4],
        [5, 6],
      ]);
      expect(a.and(b).equals(a.intersect(b))).toBe(true);
      expect(a.or(b).equals(a.union(b))).toBe(true);
      expect(a.sub(b).equals(a.difference(b))).toBe(true);
    });
  });
});
