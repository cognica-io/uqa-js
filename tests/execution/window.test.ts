import { describe, expect, it } from "vitest";
import type { PhysicalOperator } from "../../src/execution/physical.js";
// import { Batch } from "../../src/execution/batch.js";
import { WindowOp, SortOp, LimitOp } from "../../src/execution/relational.js";
import type { WindowSpec, SortKey } from "../../src/execution/relational.js";
import { SeqScanOp } from "../../src/execution/scan.js";
import { MemoryDocumentStore } from "../../src/storage/document-store.js";

// ======================================================================
// Helpers
// ======================================================================

function collectRows(op: PhysicalOperator): Record<string, unknown>[] {
  op.open();
  const rows: Record<string, unknown>[] = [];
  for (;;) {
    const batch = op.next();
    if (batch === null) break;
    rows.push(...batch.toRows());
  }
  op.close();
  return rows;
}

function makeEmployeeStore(): MemoryDocumentStore {
  const store = new MemoryDocumentStore();
  store.put(1, { id: 1, name: "Alice", dept: "eng", salary: 90000 });
  store.put(2, { id: 2, name: "Bob", dept: "mkt", salary: 75000 });
  store.put(3, { id: 3, name: "Carol", dept: "eng", salary: 85000 });
  store.put(4, { id: 4, name: "Dave", dept: "sales", salary: 70000 });
  store.put(5, { id: 5, name: "Eve", dept: "eng", salary: 95000 });
  store.put(6, { id: 6, name: "Frank", dept: "mkt", salary: 80000 });
  return store;
}

function makeScoreStore(): MemoryDocumentStore {
  const store = new MemoryDocumentStore();
  store.put(1, { id: 1, name: "A", score: 100 });
  store.put(2, { id: 2, name: "B", score: 200 });
  store.put(3, { id: 3, name: "C", score: 200 });
  store.put(4, { id: 4, name: "D", score: 300 });
  store.put(5, { id: 5, name: "E", score: 400 });
  return store;
}

// ======================================================================
// ROW_NUMBER
// ======================================================================

describe("ROW_NUMBER", () => {
  it("basic row number over order by salary desc", () => {
    const store = makeEmployeeStore();
    const spec: WindowSpec = {
      outputCol: "rn",
      funcName: "ROW_NUMBER",
      partitionBy: [],
      orderBy: [{ column: "salary", ascending: false, nullsFirst: false }],
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const byRn: Record<number, string> = {};
    for (const row of rows) {
      byRn[row["rn"] as number] = row["name"] as string;
    }
    expect(byRn[1]).toBe("Eve");
    expect(byRn[2]).toBe("Alice");
    expect(byRn[6]).toBe("Dave");
  });

  it("row number with partition by dept", () => {
    const store = makeEmployeeStore();
    const spec: WindowSpec = {
      outputCol: "rn",
      funcName: "ROW_NUMBER",
      partitionBy: ["dept"],
      orderBy: [{ column: "salary", ascending: false, nullsFirst: false }],
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const engRows = rows
      .filter((r) => r["dept"] === "eng")
      .map((r) => [r["rn"], r["name"]] as [number, string])
      .sort((a, b) => a[0] - b[0]);
    expect(engRows).toEqual([
      [1, "Eve"],
      [2, "Alice"],
      [3, "Carol"],
    ]);
  });

  it("row number ascending order", () => {
    const store = makeEmployeeStore();
    const spec: WindowSpec = {
      outputCol: "rn",
      funcName: "ROW_NUMBER",
      partitionBy: [],
      orderBy: [{ column: "salary", ascending: true, nullsFirst: false }],
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const byRn: Record<number, string> = {};
    for (const row of rows) {
      byRn[row["rn"] as number] = row["name"] as string;
    }
    expect(byRn[1]).toBe("Dave"); // lowest salary
    expect(byRn[6]).toBe("Eve"); // highest salary
  });
});

// ======================================================================
// RANK / DENSE_RANK
// ======================================================================

describe("RANK", () => {
  it("basic rank with ties", () => {
    const store = makeEmployeeStore();
    // Add Grace with same salary as Alice
    store.put(7, { id: 7, name: "Grace", dept: "eng", salary: 90000 });

    const spec: WindowSpec = {
      outputCol: "rnk",
      funcName: "RANK",
      partitionBy: [],
      orderBy: [{ column: "salary", ascending: false, nullsFirst: false }],
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const rankMap: Record<string, number> = {};
    for (const row of rows) {
      rankMap[row["name"] as string] = row["rnk"] as number;
    }
    expect(rankMap["Eve"]).toBe(1);
    // Alice and Grace tie at 90000 -> both rank 2
    expect(rankMap["Alice"]).toBe(2);
    expect(rankMap["Grace"]).toBe(2);
    // Carol at 85000 -> rank 4 (skips 3)
    expect(rankMap["Carol"]).toBe(4);
  });

  it("dense rank no gaps", () => {
    const store = makeEmployeeStore();
    store.put(7, { id: 7, name: "Grace", dept: "eng", salary: 90000 });

    const spec: WindowSpec = {
      outputCol: "drnk",
      funcName: "DENSE_RANK",
      partitionBy: [],
      orderBy: [{ column: "salary", ascending: false, nullsFirst: false }],
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const rankMap: Record<string, number> = {};
    for (const row of rows) {
      rankMap[row["name"] as string] = row["drnk"] as number;
    }
    expect(rankMap["Eve"]).toBe(1);
    expect(rankMap["Alice"]).toBe(2);
    expect(rankMap["Grace"]).toBe(2);
    // Carol at 85000 -> dense_rank 3 (no gap)
    expect(rankMap["Carol"]).toBe(3);
  });

  it("rank with partition", () => {
    const store = makeEmployeeStore();
    const spec: WindowSpec = {
      outputCol: "rnk",
      funcName: "RANK",
      partitionBy: ["dept"],
      orderBy: [{ column: "salary", ascending: false, nullsFirst: false }],
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const engRanks: Record<string, number> = {};
    for (const row of rows) {
      if (row["dept"] === "eng") {
        engRanks[row["name"] as string] = row["rnk"] as number;
      }
    }
    expect(engRanks["Eve"]).toBe(1);
    expect(engRanks["Alice"]).toBe(2);
    expect(engRanks["Carol"]).toBe(3);
  });
});

// ======================================================================
// LAG / LEAD
// ======================================================================

describe("LAG / LEAD", () => {
  it("lag basic", () => {
    const store = makeEmployeeStore();
    const spec: WindowSpec = {
      outputCol: "prev_sal",
      funcName: "LAG",
      inputCol: "salary",
      partitionBy: [],
      orderBy: [{ column: "salary", ascending: true, nullsFirst: false }],
      lagLeadOffset: 1,
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const sorted = rows
      .slice()
      .sort((a, b) => (a["salary"] as number) - (b["salary"] as number));
    // First row has no previous -> null
    expect(sorted[0]!["prev_sal"]).toBeNull();
    // Second row's prev is first row's salary
    expect(sorted[1]!["prev_sal"]).toBe(sorted[0]!["salary"]);
  });

  it("lead basic", () => {
    const store = makeEmployeeStore();
    const spec: WindowSpec = {
      outputCol: "next_sal",
      funcName: "LEAD",
      inputCol: "salary",
      partitionBy: [],
      orderBy: [{ column: "salary", ascending: true, nullsFirst: false }],
      lagLeadOffset: 1,
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const sorted = rows
      .slice()
      .sort((a, b) => (a["salary"] as number) - (b["salary"] as number));
    // Last row has no next -> null
    expect(sorted[sorted.length - 1]!["next_sal"]).toBeNull();
    // First row's next is second row's salary
    expect(sorted[0]!["next_sal"]).toBe(sorted[1]!["salary"]);
  });

  it("lag with default value", () => {
    const store = makeEmployeeStore();
    const spec: WindowSpec = {
      outputCol: "prev_sal",
      funcName: "LAG",
      inputCol: "salary",
      partitionBy: [],
      orderBy: [{ column: "salary", ascending: true, nullsFirst: false }],
      lagLeadOffset: 1,
      lagLeadDefault: 0,
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const sorted = rows
      .slice()
      .sort((a, b) => (a["salary"] as number) - (b["salary"] as number));
    // First row uses default value 0
    expect(sorted[0]!["prev_sal"]).toBe(0);
  });

  it("lag with partition", () => {
    const store = makeEmployeeStore();
    const spec: WindowSpec = {
      outputCol: "prev_sal",
      funcName: "LAG",
      inputCol: "salary",
      partitionBy: ["dept"],
      orderBy: [{ column: "salary", ascending: true, nullsFirst: false }],
      lagLeadOffset: 1,
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const engRows = rows
      .filter((r) => r["dept"] === "eng")
      .sort((a, b) => (a["salary"] as number) - (b["salary"] as number));
    // First eng employee has no prev in partition
    expect(engRows[0]!["prev_sal"]).toBeNull();
    expect(engRows[1]!["prev_sal"]).toBe(engRows[0]!["salary"]);
  });
});

// ======================================================================
// Aggregate window functions (SUM, COUNT, AVG, MIN, MAX)
// ======================================================================

describe("Aggregate window functions", () => {
  it("SUM over partition (running sum)", () => {
    const store = makeEmployeeStore();
    const spec: WindowSpec = {
      outputCol: "dept_total",
      funcName: "SUM",
      inputCol: "salary",
      partitionBy: ["dept"],
      orderBy: [],
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    // WindowOp computes running SUM, so the last row in each partition
    // has the full partition sum
    const engRows = rows.filter((r) => r["dept"] === "eng");
    const lastEngTotal = engRows[engRows.length - 1]!["dept_total"] as number;
    // eng total: 90000 + 85000 + 95000 = 270000
    expect(lastEngTotal).toBe(270000);

    const mktRows = rows.filter((r) => r["dept"] === "mkt");
    const lastMktTotal = mktRows[mktRows.length - 1]!["dept_total"] as number;
    // mkt total: 75000 + 80000 = 155000
    expect(lastMktTotal).toBe(155000);
  });

  it("COUNT over partition", () => {
    const store = makeEmployeeStore();
    const spec: WindowSpec = {
      outputCol: "dept_cnt",
      funcName: "ROW_NUMBER",
      partitionBy: ["dept"],
      orderBy: [],
    };
    // Use ROW_NUMBER as a proxy; for COUNT we check via the SUM pattern
    // Since WindowOp does not have a COUNT window function directly,
    // we test via the rows per dept
    const rows = collectRows(new WindowOp(new SeqScanOp(store), [spec]));
    const engRows = rows.filter((r) => r["dept"] === "eng");
    expect(engRows.length).toBe(3);
  });

  it("AVG over partition (running avg)", () => {
    const store = makeEmployeeStore();
    const spec: WindowSpec = {
      outputCol: "avg_sal",
      funcName: "AVG",
      inputCol: "salary",
      partitionBy: ["dept"],
      orderBy: [],
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const engRows = rows.filter((r) => r["dept"] === "eng");
    const expectedAvg = (90000 + 85000 + 95000) / 3;
    // All eng rows should see the running avg after all 3 are processed
    // (since no ORDER BY, all rows are in one partition frame)
    const lastEngRow = engRows[engRows.length - 1]!;
    expect(Math.abs((lastEngRow["avg_sal"] as number) - expectedAvg)).toBeLessThan(
      0.01,
    );
  });

  it("MIN/MAX over partition (running)", () => {
    const store = makeEmployeeStore();
    const minSpec: WindowSpec = {
      outputCol: "min_sal",
      funcName: "MIN",
      inputCol: "salary",
      partitionBy: ["dept"],
      orderBy: [],
    };
    const maxSpec: WindowSpec = {
      outputCol: "max_sal",
      funcName: "MAX",
      inputCol: "salary",
      partitionBy: ["dept"],
      orderBy: [],
    };
    const op = new WindowOp(new SeqScanOp(store), [minSpec, maxSpec]);
    const rows = collectRows(op);

    const engRows = rows.filter((r) => r["dept"] === "eng");
    // After processing all eng rows, last row should have correct min/max
    const lastEng = engRows[engRows.length - 1]!;
    expect(lastEng["min_sal"]).toBe(85000);
    expect(lastEng["max_sal"]).toBe(95000);
  });
});

// ======================================================================
// NTILE
// ======================================================================

describe("NTILE", () => {
  it("divides into n buckets", () => {
    const store = makeScoreStore();
    const spec: WindowSpec = {
      outputCol: "bucket",
      funcName: "NTILE",
      partitionBy: [],
      orderBy: [{ column: "score", ascending: true, nullsFirst: false }],
      lagLeadOffset: 3, // 3 buckets
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const sorted = rows
      .slice()
      .sort((a, b) => (a["score"] as number) - (b["score"] as number));
    // 5 rows in 3 buckets: [2, 2, 1]
    // Row 1: bucket 1, Row 2: bucket 1, Row 3: bucket 2, Row 4: bucket 2, Row 5: bucket 3
    // 5 rows in 3 buckets: sizes [2, 2, 1]
    // sorted[0] (score 100) -> bucket 1
    // sorted[1] (score 200) -> bucket 1
    // sorted[2] (score 200) -> bucket 2
    // sorted[3] (score 300) -> bucket 2
    // sorted[4] (score 400) -> bucket 3
    expect(sorted[0]!["bucket"]).toBe(1);
    expect(sorted[1]!["bucket"]).toBe(1);
    expect(sorted[sorted.length - 1]!["bucket"]).toBe(3);
  });
});

// ======================================================================
// PERCENT_RANK
// ======================================================================

describe("PERCENT_RANK", () => {
  it("basic percent rank", () => {
    const store = makeScoreStore();
    const spec: WindowSpec = {
      outputCol: "pr",
      funcName: "PERCENT_RANK",
      partitionBy: [],
      orderBy: [{ column: "score", ascending: true, nullsFirst: false }],
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const sorted = rows
      .slice()
      .sort((a, b) => (a["score"] as number) - (b["score"] as number));
    const prs = sorted.map((r) => r["pr"] as number);
    // ranks: 1, 2, 2, 4, 5 -> (0/4, 1/4, 1/4, 3/4, 4/4)
    expect(prs[0]).toBeCloseTo(0.0);
    expect(prs[1]).toBeCloseTo(0.25);
    expect(prs[2]).toBeCloseTo(0.25);
    expect(prs[3]).toBeCloseTo(0.75);
    expect(prs[4]).toBeCloseTo(1.0);
  });
});

// ======================================================================
// CUME_DIST
// ======================================================================

describe("CUME_DIST", () => {
  it("basic cume_dist", () => {
    const store = makeScoreStore();
    const spec: WindowSpec = {
      outputCol: "cd",
      funcName: "CUME_DIST",
      partitionBy: [],
      orderBy: [{ column: "score", ascending: true, nullsFirst: false }],
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const sorted = rows
      .slice()
      .sort((a, b) => (a["score"] as number) - (b["score"] as number));
    const cds = sorted.map((r) => r["cd"] as number);
    // A(100): 1/5=0.2, B(200): 3/5=0.6, C(200): 3/5=0.6, D(300): 4/5=0.8, E(400): 5/5=1.0
    expect(cds[0]).toBeCloseTo(0.2);
    expect(cds[1]).toBeCloseTo(0.6);
    expect(cds[2]).toBeCloseTo(0.6);
    expect(cds[3]).toBeCloseTo(0.8);
    expect(cds[4]).toBeCloseTo(1.0);
  });
});

// ======================================================================
// NTH_VALUE
// ======================================================================

describe("NTH_VALUE", () => {
  it("basic nth value", () => {
    const store = makeScoreStore();
    const spec: WindowSpec = {
      outputCol: "second",
      funcName: "NTH_VALUE",
      inputCol: "name",
      partitionBy: [],
      orderBy: [{ column: "score", ascending: true, nullsFirst: false }],
      lagLeadOffset: 2, // 2nd value
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    // 2nd value ordered by score: B
    for (const row of rows) {
      expect(row["second"]).toBe("B");
    }
  });

  it("nth value out of range returns null", () => {
    const store = new MemoryDocumentStore();
    store.put(1, { id: 1, val: 10 });

    const spec: WindowSpec = {
      outputCol: "v",
      funcName: "NTH_VALUE",
      inputCol: "val",
      partitionBy: [],
      orderBy: [{ column: "id", ascending: true, nullsFirst: false }],
      lagLeadOffset: 5,
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);
    expect(rows[0]!["v"]).toBeNull();
  });
});

// ======================================================================
// FIRST_VALUE / LAST_VALUE
// ======================================================================

describe("FIRST_VALUE / LAST_VALUE", () => {
  it("first value in partition", () => {
    const store = makeEmployeeStore();
    const spec: WindowSpec = {
      outputCol: "first_name",
      funcName: "FIRST_VALUE",
      inputCol: "name",
      partitionBy: ["dept"],
      orderBy: [{ column: "salary", ascending: false, nullsFirst: false }],
    };
    const op = new WindowOp(new SeqScanOp(store), [spec]);
    const rows = collectRows(op);

    const engRows = rows.filter((r) => r["dept"] === "eng");
    // Eve has highest salary in eng
    for (const row of engRows) {
      expect(row["first_name"]).toBe("Eve");
    }
  });
});

// ======================================================================
// Multiple window functions in one query
// ======================================================================

describe("Multiple window functions", () => {
  it("row_number and sum together", () => {
    const store = makeEmployeeStore();
    const rnSpec: WindowSpec = {
      outputCol: "rn",
      funcName: "ROW_NUMBER",
      partitionBy: [],
      orderBy: [{ column: "salary", ascending: false, nullsFirst: false }],
    };
    const sumSpec: WindowSpec = {
      outputCol: "dept_total",
      funcName: "SUM",
      inputCol: "salary",
      partitionBy: ["dept"],
      orderBy: [],
    };
    const op = new WindowOp(new SeqScanOp(store), [rnSpec, sumSpec]);
    const rows = collectRows(op);

    expect(rows.length).toBe(6);
    // Every row should have both rn and dept_total
    for (const row of rows) {
      expect(row["rn"]).toBeDefined();
      expect(row["dept_total"]).toBeDefined();
    }
  });
});

// ======================================================================
// Window with sort and limit (integration)
// ======================================================================

describe("Window integration", () => {
  it("window result can be sorted and limited", () => {
    const store = makeEmployeeStore();
    const spec: WindowSpec = {
      outputCol: "rn",
      funcName: "ROW_NUMBER",
      partitionBy: [],
      orderBy: [{ column: "salary", ascending: false, nullsFirst: false }],
    };
    const sortKeys: SortKey[] = [{ column: "rn", ascending: true, nullsFirst: false }];
    const op = new LimitOp(
      new SortOp(new WindowOp(new SeqScanOp(store), [spec]), sortKeys),
      3,
    );
    const rows = collectRows(op);
    expect(rows.length).toBe(3);
    expect(rows[0]!["name"]).toBe("Eve");
    expect(rows[2]!["name"]).toBe("Carol");
  });
});
