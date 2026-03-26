import { describe, expect, it } from "vitest";
import { PostingList, GeneralizedPostingList } from "../../src/core/posting-list.js";
import {
  createPayload,
  createPostingEntry,
  createGeneralizedPostingEntry,
} from "../../src/core/types.js";
import type {
  PostingEntry,
  GeneralizedPostingEntry,
  Payload,
} from "../../src/core/types.js";

// -- SeededRandom for property-based tests ------------------------------------

class SeededRandom {
  private _state: number;

  constructor(seed: number) {
    this._state = seed;
  }

  next(): number {
    // xorshift32
    this._state ^= this._state << 13;
    this._state ^= this._state >> 17;
    this._state ^= this._state << 5;
    return (this._state >>> 0) / 0xffffffff;
  }

  nextInt(min: number, max: number): number {
    return min + Math.floor(this.next() * (max - min + 1));
  }

  nextFloat(min: number, max: number): number {
    return min + this.next() * (max - min);
  }
}

// -- Helpers ------------------------------------------------------------------

function makePayload(opts?: {
  positions?: number[];
  score?: number;
  fields?: Record<string, unknown>;
}): Payload {
  return createPayload({
    positions: opts?.positions ?? [],
    score: opts?.score ?? 0.0,
    fields: opts?.fields ?? {},
  });
}
void makePayload;

function makeEntry(docId: number, score = 0.0, positions: number[] = []): PostingEntry {
  return createPostingEntry(docId, { score, positions });
}

function postingListDocIdsEqual(a: PostingList, b: PostingList): boolean {
  const aIds = a.docIds;
  const bIds = b.docIds;
  if (aIds.size !== bIds.size) return false;
  for (const id of aIds) {
    if (!bIds.has(id)) return false;
  }
  return true;
}

function randomPayload(rng: SeededRandom): Payload {
  const p1 = rng.nextInt(0, 100);
  const p2 = rng.nextInt(0, 100);
  const p3 = rng.nextInt(0, 100);
  const posSet = new Set([p1, p2, p3]);
  const positions = [...posSet].sort((a, b) => a - b);
  const score = rng.nextFloat(0, 10);
  return createPayload({ positions, score });
}

function randomPostingList(rng: SeededRandom, maxSize = 15): PostingList {
  const size = rng.nextInt(0, maxSize);
  const entries: PostingEntry[] = [];
  for (let i = 0; i < size; i++) {
    const docId = rng.nextInt(0, 50);
    entries.push({ docId, payload: randomPayload(rng) });
  }
  return new PostingList(entries);
}

const UNIVERSAL_DOC_IDS = Array.from({ length: 51 }, (_, i) => i);
const UNIVERSAL_POSTING_LIST = new PostingList(
  UNIVERSAL_DOC_IDS.map((id) => makeEntry(id, 0.0)),
);

// =============================================================================
// Axiom A1: Commutativity
// =============================================================================

describe("PostingList Boolean Algebra - Axiom A1: Commutativity", () => {
  it("union is commutative (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(42);
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      const b = randomPostingList(rng);
      expect(postingListDocIdsEqual(a.union(b), b.union(a))).toBe(true);
    }
  });

  it("intersect is commutative (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(123);
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      const b = randomPostingList(rng);
      expect(postingListDocIdsEqual(a.intersect(b), b.intersect(a))).toBe(true);
    }
  });
});

// =============================================================================
// Axiom A2: Associativity
// =============================================================================

describe("PostingList Boolean Algebra - Axiom A2: Associativity", () => {
  it("union is associative (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(200);
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      const b = randomPostingList(rng);
      const c = randomPostingList(rng);
      const lhs = a.union(b.union(c));
      const rhs = a.union(b).union(c);
      expect(postingListDocIdsEqual(lhs, rhs)).toBe(true);
    }
  });

  it("intersect is associative (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(300);
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      const b = randomPostingList(rng);
      const c = randomPostingList(rng);
      const lhs = a.intersect(b.intersect(c));
      const rhs = a.intersect(b).intersect(c);
      expect(postingListDocIdsEqual(lhs, rhs)).toBe(true);
    }
  });
});

// =============================================================================
// Axiom A3: Distributivity
// =============================================================================

describe("PostingList Boolean Algebra - Axiom A3: Distributivity", () => {
  it("intersect distributes over union (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(400);
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      const b = randomPostingList(rng);
      const c = randomPostingList(rng);
      const lhs = a.intersect(b.union(c));
      const rhs = a.intersect(b).union(a.intersect(c));
      expect(postingListDocIdsEqual(lhs, rhs)).toBe(true);
    }
  });

  it("union distributes over intersect (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(500);
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      const b = randomPostingList(rng);
      const c = randomPostingList(rng);
      const lhs = a.union(b.intersect(c));
      const rhs = a.union(b).intersect(a.union(c));
      expect(postingListDocIdsEqual(lhs, rhs)).toBe(true);
    }
  });
});

// =============================================================================
// Axiom A4: Identity
// =============================================================================

describe("PostingList Boolean Algebra - Axiom A4: Identity", () => {
  it("union with empty is identity (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(600);
    const empty = new PostingList();
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      expect(postingListDocIdsEqual(a.union(empty), a)).toBe(true);
      expect(postingListDocIdsEqual(empty.union(a), a)).toBe(true);
    }
  });

  it("intersect with universal set is identity (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(700);
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      // Build universal set containing all doc_ids from a plus universal range
      const allIds = new Set([...a.docIds, ...UNIVERSAL_DOC_IDS]);
      const u = new PostingList([...allIds].map((id) => makeEntry(id, 0.0)));
      const result = a.intersect(u);
      expect(postingListDocIdsEqual(result, a)).toBe(true);
    }
  });
});

// =============================================================================
// Axiom A5: Complement
// =============================================================================

describe("PostingList Boolean Algebra - Axiom A5: Complement", () => {
  it("A union complement(A) equals universal (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(800);
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      const complementA = a.complement(UNIVERSAL_POSTING_LIST);
      const result = a.union(complementA);
      expect(result.docIds).toEqual(UNIVERSAL_POSTING_LIST.docIds);
    }
  });

  it("A intersect complement(A) is empty (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(900);
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      const complementA = a.complement(UNIVERSAL_POSTING_LIST);
      const result = a.intersect(complementA);
      expect(result.length).toBe(0);
    }
  });
});

// =============================================================================
// Sorted invariant
// =============================================================================

describe("PostingList sorted invariant", () => {
  it("entries sorted by docId after union (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(1000);
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      const b = randomPostingList(rng);
      const result = a.union(b);
      const entries = result.entries;
      for (let i = 0; i < entries.length - 1; i++) {
        expect(entries[i]!.docId).toBeLessThan(entries[i + 1]!.docId);
      }
    }
  });

  it("entries sorted by docId after intersect (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(1100);
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      const b = randomPostingList(rng);
      const result = a.intersect(b);
      const entries = result.entries;
      for (let i = 0; i < entries.length - 1; i++) {
        expect(entries[i]!.docId).toBeLessThan(entries[i + 1]!.docId);
      }
    }
  });

  it("entries sorted by docId after complement (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(1200);
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      const result = a.complement(UNIVERSAL_POSTING_LIST);
      const entries = result.entries;
      for (let i = 0; i < entries.length - 1; i++) {
        expect(entries[i]!.docId).toBeLessThan(entries[i + 1]!.docId);
      }
    }
  });

  it("entries sorted by docId after difference (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(1300);
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      const b = randomPostingList(rng);
      const result = a.difference(b);
      const entries = result.entries;
      for (let i = 0; i < entries.length - 1; i++) {
        expect(entries[i]!.docId).toBeLessThan(entries[i + 1]!.docId);
      }
    }
  });
});

// =============================================================================
// Difference correctness
// =============================================================================

describe("PostingList difference correctness", () => {
  it("difference returns entries in A but not in B (property-based, 100 iterations)", () => {
    const rng = new SeededRandom(1400);
    for (let iter = 0; iter < 100; iter++) {
      const a = randomPostingList(rng);
      const b = randomPostingList(rng);
      const result = a.difference(b);
      const resultIds = new Set([...result].map((e) => e.docId));
      const expectedIds = new Set<number>();
      for (const id of a.docIds) {
        if (!b.docIds.has(id)) expectedIds.add(id);
      }
      expect(resultIds).toEqual(expectedIds);
    }
  });
});

// =============================================================================
// Merge payloads
// =============================================================================

describe("PostingList merge payloads", () => {
  it("merged positions are the sorted union of both positions", () => {
    const a = new PostingList([
      createPostingEntry(1, { positions: [0, 2], score: 1.0 }),
    ]);
    const b = new PostingList([
      createPostingEntry(1, { positions: [1, 3], score: 0.5 }),
    ]);
    const result = a.union(b);
    const entry = result.getEntry(1);
    expect(entry).not.toBeNull();
    expect([...entry!.payload.positions]).toEqual([0, 1, 2, 3]);
  });

  it("merged score is the sum of both scores", () => {
    const a = new PostingList([createPostingEntry(1, { score: 1.0 })]);
    const b = new PostingList([createPostingEntry(1, { score: 0.5 })]);
    const result = a.union(b);
    const entry = result.getEntry(1);
    expect(entry).not.toBeNull();
    expect(entry!.payload.score).toBe(1.5);
  });

  it("merged fields combine both field dicts", () => {
    const a = new PostingList([createPostingEntry(1, { fields: { a: 1 } })]);
    const b = new PostingList([createPostingEntry(1, { fields: { b: 2 } })]);
    const result = a.union(b);
    const entry = result.getEntry(1);
    expect(entry).not.toBeNull();
    expect(entry!.payload.fields).toEqual({ a: 1, b: 2 });
  });
});

// =============================================================================
// GeneralizedPostingList
// =============================================================================

describe("GeneralizedPostingList", () => {
  function makeGPL(tuples: number[][]): GeneralizedPostingList {
    const entries: GeneralizedPostingEntry[] = tuples.map((t) =>
      createGeneralizedPostingEntry(t, {
        score: t.reduce((sum, v) => sum + v, 0),
      }),
    );
    return new GeneralizedPostingList(entries);
  }

  it("union deduplicates by doc_ids tuple", () => {
    const gplA = new GeneralizedPostingList([
      createGeneralizedPostingEntry([1, 2], { score: 1.0 }),
      createGeneralizedPostingEntry([3, 4], { score: 0.5 }),
    ]);
    const gplB = new GeneralizedPostingList([
      createGeneralizedPostingEntry([1, 2], { score: 0.8 }),
      createGeneralizedPostingEntry([5, 6], { score: 0.3 }),
    ]);
    const result = gplA.union(gplB);
    expect(result.length).toBe(3);
    const idsSet = new Set([...result].map((e) => e.docIds.join(",")));
    expect(idsSet).toEqual(new Set(["1,2", "3,4", "5,6"]));
  });

  it("entries are sorted by doc_ids", () => {
    const gpl = new GeneralizedPostingList([
      createGeneralizedPostingEntry([5, 6], { score: 0.3 }),
      createGeneralizedPostingEntry([1, 2], { score: 1.0 }),
      createGeneralizedPostingEntry([3, 4], { score: 0.5 }),
    ]);
    const entries = gpl.entries;
    for (let i = 0; i < entries.length - 1; i++) {
      const a = entries[i]!.docIds;
      const b = entries[i + 1]!.docIds;
      const cmp = a[0]! < b[0]! || (a[0] === b[0] && a[1]! <= b[1]!);
      expect(cmp).toBe(true);
    }
  });

  // -- intersect --

  describe("intersect", () => {
    it("keeps only shared tuples", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const b = makeGPL([
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const result = a.intersect(b);
      const idsSet = new Set([...result].map((e) => e.docIds.join(",")));
      expect(idsSet).toEqual(new Set(["3,4", "5,6"]));
    });

    it("preserves left payload", () => {
      const a = new GeneralizedPostingList([
        createGeneralizedPostingEntry([1, 2], { score: 9.0 }),
      ]);
      const b = new GeneralizedPostingList([
        createGeneralizedPostingEntry([1, 2], { score: 1.0 }),
      ]);
      const result = a.intersect(b);
      expect(result.length).toBe(1);
      const first = [...result][0]!;
      expect(first.payload.score).toBe(9.0);
    });

    it("result is sorted", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const b = makeGPL([
        [5, 6],
        [1, 2],
      ]);
      const result = a.intersect(b);
      const entries = result.entries;
      for (let i = 0; i < entries.length - 1; i++) {
        const cmpA = entries[i]!.docIds;
        const cmpB = entries[i + 1]!.docIds;
        expect(
          cmpA[0]! < cmpB[0]! || (cmpA[0] === cmpB[0] && cmpA[1]! <= cmpB[1]!),
        ).toBe(true);
      }
    });
  });

  // -- difference --

  describe("difference", () => {
    it("removes tuples present in other", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const b = makeGPL([[3, 4]]);
      const result = a.difference(b);
      const idsSet = new Set([...result].map((e) => e.docIds.join(",")));
      expect(idsSet).toEqual(new Set(["1,2", "5,6"]));
    });

    it("preserves payload from self", () => {
      const a = new GeneralizedPostingList([
        createGeneralizedPostingEntry([1, 2], { score: 5.0 }),
        createGeneralizedPostingEntry([3, 4], { score: 7.0 }),
      ]);
      const b = makeGPL([[3, 4]]);
      const result = a.difference(b);
      expect(result.length).toBe(1);
      expect([...result][0]!.payload.score).toBe(5.0);
    });

    it("result is sorted", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const b = makeGPL([[3, 4]]);
      const result = a.difference(b);
      const entries = result.entries;
      for (let i = 0; i < entries.length - 1; i++) {
        const cmpA = entries[i]!.docIds;
        const cmpB = entries[i + 1]!.docIds;
        expect(
          cmpA[0]! < cmpB[0]! || (cmpA[0] === cmpB[0] && cmpA[1]! <= cmpB[1]!),
        ).toBe(true);
      }
    });
  });

  // -- complement --

  describe("complement", () => {
    it("returns universal minus self", () => {
      const universal = makeGPL([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const a = makeGPL([
        [3, 4],
        [7, 8],
      ]);
      const result = a.complement(universal);
      const idsSet = new Set([...result].map((e) => e.docIds.join(",")));
      expect(idsSet).toEqual(new Set(["1,2", "5,6"]));
    });

    it("complement of empty is universal", () => {
      const universal = makeGPL([
        [1, 2],
        [3, 4],
      ]);
      const empty = new GeneralizedPostingList();
      const result = empty.complement(universal);
      expect(result.equals(universal)).toBe(true);
    });

    it("complement of universal is empty", () => {
      const universal = makeGPL([
        [1, 2],
        [3, 4],
      ]);
      const result = universal.complement(universal);
      expect(result.length).toBe(0);
    });
  });

  // -- docIdsSet --

  describe("docIdsSet", () => {
    it("returns set of all doc_ids tuple keys", () => {
      const gpl = makeGPL([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const idsSet = gpl.docIdsSet;
      expect(idsSet.size).toBe(3);
    });

    it("empty list returns empty set", () => {
      const gpl = new GeneralizedPostingList();
      expect(gpl.docIdsSet.size).toBe(0);
    });
  });

  // -- operator aliases --

  describe("operator aliases", () => {
    it("and() delegates to intersect()", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const b = makeGPL([
        [3, 4],
        [7, 8],
      ]);
      const result = a.and(b);
      const idsSet = new Set([...result].map((e) => e.docIds.join(",")));
      expect(idsSet).toEqual(new Set(["3,4"]));
    });

    it("or() delegates to union()", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
      ]);
      const b = makeGPL([
        [3, 4],
        [5, 6],
      ]);
      const result = a.or(b);
      const idsSet = new Set([...result].map((e) => e.docIds.join(",")));
      expect(idsSet).toEqual(new Set(["1,2", "3,4", "5,6"]));
    });

    it("sub() delegates to difference()", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const b = makeGPL([[3, 4]]);
      const result = a.sub(b);
      const idsSet = new Set([...result].map((e) => e.docIds.join(",")));
      expect(idsSet).toEqual(new Set(["1,2", "5,6"]));
    });
  });

  // -- equals --

  describe("equals", () => {
    it("same entries are equal", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
      ]);
      const b = makeGPL([
        [1, 2],
        [3, 4],
      ]);
      expect(a.equals(b)).toBe(true);
    });

    it("different entries are not equal", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
      ]);
      const b = makeGPL([
        [1, 2],
        [5, 6],
      ]);
      expect(a.equals(b)).toBe(false);
    });

    it("different lengths are not equal", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
      ]);
      const b = makeGPL([[1, 2]]);
      expect(a.equals(b)).toBe(false);
    });

    it("ignores payload differences", () => {
      const a = new GeneralizedPostingList([
        createGeneralizedPostingEntry([1, 2], { score: 1.0 }),
      ]);
      const b = new GeneralizedPostingList([
        createGeneralizedPostingEntry([1, 2], { score: 99.0 }),
      ]);
      expect(a.equals(b)).toBe(true);
    });
  });

  // -- empty operands --

  describe("empty operands", () => {
    it("intersect with empty returns empty", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
      ]);
      const empty = new GeneralizedPostingList();
      expect(a.intersect(empty).length).toBe(0);
      expect(empty.intersect(a).length).toBe(0);
    });

    it("difference with empty returns self", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
      ]);
      const empty = new GeneralizedPostingList();
      expect(a.difference(empty).equals(a)).toBe(true);
      expect(empty.difference(a).length).toBe(0);
    });

    it("union with empty returns self", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
      ]);
      const empty = new GeneralizedPostingList();
      expect(a.union(empty).equals(a)).toBe(true);
      expect(empty.union(a).equals(a)).toBe(true);
    });
  });

  // -- no-overlap cases --

  describe("no-overlap cases", () => {
    it("intersect of disjoint sets is empty", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
      ]);
      const b = makeGPL([
        [5, 6],
        [7, 8],
      ]);
      expect(a.intersect(b).length).toBe(0);
    });

    it("difference of disjoint sets returns self", () => {
      const a = makeGPL([
        [1, 2],
        [3, 4],
      ]);
      const b = makeGPL([
        [5, 6],
        [7, 8],
      ]);
      expect(a.difference(b).equals(a)).toBe(true);
    });
  });
});

// =============================================================================
// De Morgan property tests
// =============================================================================

describe("De Morgan properties", () => {
  function ids(pl: PostingList): Set<number> {
    return new Set([...pl].map((e) => e.docId));
  }

  it("NOT (A AND B) == (NOT A) OR (NOT B) (property-based, 50 iterations)", () => {
    const rng = new SeededRandom(2000);
    for (let iter = 0; iter < 50; iter++) {
      const a = randomPostingList(rng);
      const b = randomPostingList(rng);
      const u = a
        .union(b)
        .union(new PostingList(Array.from({ length: 51 }, (_, i) => makeEntry(i))));
      const lhs = a.intersect(b).complement(u);
      const rhs = a.complement(u).union(b.complement(u));
      expect(ids(lhs)).toEqual(ids(rhs));
    }
  });

  it("NOT (A OR B) == (NOT A) AND (NOT B) (property-based, 50 iterations)", () => {
    const rng = new SeededRandom(3000);
    for (let iter = 0; iter < 50; iter++) {
      const a = randomPostingList(rng);
      const b = randomPostingList(rng);
      const u = a
        .union(b)
        .union(new PostingList(Array.from({ length: 51 }, (_, i) => makeEntry(i))));
      const lhs = a.union(b).complement(u);
      const rhs = a.complement(u).intersect(b.complement(u));
      expect(ids(lhs)).toEqual(ids(rhs));
    }
  });

  it("distributivity: A AND (B OR C) == (A AND B) OR (A AND C) (property-based, 30 iterations)", () => {
    const rng = new SeededRandom(4000);
    for (let iter = 0; iter < 30; iter++) {
      const a = randomPostingList(rng);
      const b = randomPostingList(rng);
      const c = randomPostingList(rng);
      const lhs = a.intersect(b.union(c));
      const rhs = a.intersect(b).union(a.intersect(c));
      expect(ids(lhs)).toEqual(ids(rhs));
    }
  });
});

// =============================================================================
// GeneralizedPostingList algebra property tests
// =============================================================================

describe("GPL algebra properties", () => {
  function randomGPL(rng: SeededRandom, maxSize = 8): GeneralizedPostingList {
    const size = rng.nextInt(0, maxSize);
    const entries: GeneralizedPostingEntry[] = [];
    for (let i = 0; i < size; i++) {
      const docIds = [rng.nextInt(0, 10), rng.nextInt(0, 10)];
      entries.push(createGeneralizedPostingEntry(docIds, { score: 0.0 }));
    }
    return new GeneralizedPostingList(entries);
  }

  function tuples(gpl: GeneralizedPostingList): Set<string> {
    return new Set([...gpl].map((e) => e.docIds.join(",")));
  }

  it("GPL union is commutative (property-based, 30 iterations)", () => {
    const rng = new SeededRandom(5000);
    for (let iter = 0; iter < 30; iter++) {
      const a = randomGPL(rng);
      const b = randomGPL(rng);
      expect(tuples(a.union(b))).toEqual(tuples(b.union(a)));
    }
  });

  it("GPL intersect is commutative (property-based, 30 iterations)", () => {
    const rng = new SeededRandom(6000);
    for (let iter = 0; iter < 30; iter++) {
      const a = randomGPL(rng);
      const b = randomGPL(rng);
      expect(tuples(a.intersect(b))).toEqual(tuples(b.intersect(a)));
    }
  });

  it("GPL De Morgan: NOT (A AND B) == (NOT A) OR (NOT B) (property-based, 30 iterations)", () => {
    const rng = new SeededRandom(7000);
    for (let iter = 0; iter < 30; iter++) {
      const a = randomGPL(rng);
      const b = randomGPL(rng);
      const u = a.union(b);
      const lhs = tuples(a.intersect(b).complement(u));
      const rhs = tuples(a.complement(u).union(b.complement(u)));
      expect(lhs).toEqual(rhs);
    }
  });
});
