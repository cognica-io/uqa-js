import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// generate_series
// =============================================================================

describe("GenerateSeries", () => {
  it("basic", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT n FROM generate_series(1, 5) AS t(n)");
    const values = result!.rows.map((r) => r["n"]);
    expect(values).toEqual([1, 2, 3, 4, 5]);
  });

  it("with step", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT n FROM generate_series(0, 10, 3) AS t(n)");
    const values = result!.rows.map((r) => r["n"]);
    expect(values).toEqual([0, 3, 6, 9]);
  });

  it("descending", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT n FROM generate_series(5, 1, -1) AS t(n)");
    const values = result!.rows.map((r) => r["n"]);
    expect(values).toEqual([5, 4, 3, 2, 1]);
  });

  it("single value", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT n FROM generate_series(1, 1) AS t(n)");
    expect(result!.rows.length).toBe(1);
    expect(result!.rows[0]!["n"]).toBe(1);
  });

  it("empty range", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT n FROM generate_series(5, 1) AS t(n)");
    expect(result!.rows.length).toBe(0);
  });
});

// =============================================================================
// UNNEST
// =============================================================================

describe("Unnest", () => {
  it("basic", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT val FROM unnest(ARRAY[10, 20, 30]) AS t(val)");
    expect(result!.rows.length).toBe(3);
    const vals = result!.rows.map((r) => r["val"]);
    expect(vals).toEqual([10, 20, 30]);
  });

  it("text array", async () => {
    const e = new Engine();
    const result = await e.sql(
      "SELECT val FROM unnest(ARRAY['a', 'b', 'c']) AS t(val)",
    );
    expect(result!.rows.length).toBe(3);
    const vals = result!.rows.map((r) => r["val"]);
    expect(vals).toEqual(["a", "b", "c"]);
  });
});
