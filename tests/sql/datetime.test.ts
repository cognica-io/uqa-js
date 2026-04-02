import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// DATE / TIMESTAMP type support
// =============================================================================

describe("DateTimeTypes", () => {
  it("create table with date", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE events (id INTEGER, event_date DATE)");
    const result = await e.sql(
      "INSERT INTO events (id, event_date) VALUES (1, '2024-01-15')",
    );
    expect(result!.rows[0]!["inserted"]).toBe(1);
  });

  it("insert date values", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE log (id INTEGER, ts TIMESTAMP)");
    await e.sql("INSERT INTO log (id, ts) VALUES (1, '2024-06-15T10:30:00')");
    await e.sql("INSERT INTO log (id, ts) VALUES (2, '2024-06-16T14:00:00')");
    const result = await e.sql("SELECT COUNT(*) AS cnt FROM log");
    expect(result!.rows[0]!["cnt"]).toBe(2);
  });

  it("date comparison", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE events (id INTEGER, event_date DATE)");
    await e.sql("INSERT INTO events (id, event_date) VALUES (1, '2024-01-01')");
    await e.sql("INSERT INTO events (id, event_date) VALUES (2, '2024-06-15')");
    await e.sql("INSERT INTO events (id, event_date) VALUES (3, '2024-12-31')");
    const result = await e.sql(
      "SELECT id FROM events WHERE event_date > '2024-03-01' ORDER BY id",
    );
    const ids = result!.rows.map((r) => r["id"]);
    expect(ids).toEqual([2, 3]);
  });

  it("date ordering", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE events (id INTEGER, event_date DATE)");
    await e.sql("INSERT INTO events (id, event_date) VALUES (1, '2024-12-31')");
    await e.sql("INSERT INTO events (id, event_date) VALUES (2, '2024-01-01')");
    await e.sql("INSERT INTO events (id, event_date) VALUES (3, '2024-06-15')");
    const result = await e.sql(
      "SELECT id, event_date FROM events ORDER BY event_date ASC",
    );
    const ids = result!.rows.map((r) => r["id"]);
    expect(ids).toEqual([2, 3, 1]);
  });
});

// =============================================================================
// NOW() / CURRENT_DATE / CURRENT_TIMESTAMP
// =============================================================================

describe("DateTimeFunctions", () => {
  it("now", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT NOW() AS ts");
    const ts = result!.rows[0]!["ts"];
    expect(ts).toBeInstanceOf(Date);
  });

  it("current date", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT CURRENT_DATE AS d");
    const d = result!.rows[0]!["d"];
    expect(d).toBeInstanceOf(Date);
    // Date-only: hours/minutes/seconds should be 0
    expect((d as Date).getHours()).toBe(0);
    expect((d as Date).getMinutes()).toBe(0);
  });

  it("current timestamp", async () => {
    const e = new Engine();
    const result = await e.sql("SELECT CURRENT_TIMESTAMP AS ts");
    const ts = result!.rows[0]!["ts"];
    expect(ts).toBeInstanceOf(Date);
  });
});

// =============================================================================
// EXTRACT / DATE_PART / DATE_TRUNC
// =============================================================================

describe("ExtractDatePartDateTrunc", () => {
  async function makeTSEngine(): Promise<Engine> {
    const e = new Engine();
    await e.sql("CREATE TABLE log (id INTEGER, ts TIMESTAMP)");
    await e.sql("INSERT INTO log (id, ts) VALUES (1, '2024-06-15T10:30:45')");
    return e;
  }

  it("extract year", async () => {
    const e = await makeTSEngine();
    const result = await e.sql("SELECT EXTRACT(year FROM ts) AS y FROM log");
    expect(result!.rows[0]!["y"]).toBe(2024);
  });

  it("extract month", async () => {
    const e = await makeTSEngine();
    const result = await e.sql("SELECT EXTRACT(month FROM ts) AS m FROM log");
    expect(result!.rows[0]!["m"]).toBe(6);
  });

  it("extract day", async () => {
    const e = await makeTSEngine();
    const result = await e.sql("SELECT EXTRACT(day FROM ts) AS d FROM log");
    expect(result!.rows[0]!["d"]).toBe(15);
  });

  it("extract hour", async () => {
    const e = await makeTSEngine();
    const result = await e.sql("SELECT EXTRACT(hour FROM ts) AS h FROM log");
    expect(result!.rows[0]!["h"]).toBe(10);
  });

  it("extract dow", async () => {
    const e = await makeTSEngine();
    const result = await e.sql("SELECT EXTRACT(dow FROM ts) AS dow FROM log");
    // 2024-06-15 is a Saturday -> PostgreSQL dow=6
    expect(result!.rows[0]!["dow"]).toBe(6);
  });

  it("extract epoch", async () => {
    const e = await makeTSEngine();
    const result = await e.sql("SELECT EXTRACT(epoch FROM ts) AS e FROM log");
    expect(typeof result!.rows[0]!["e"]).toBe("number");
  });

  it("date part", async () => {
    const e = await makeTSEngine();
    const result = await e.sql("SELECT DATE_PART('year', ts) AS y FROM log");
    expect(result!.rows[0]!["y"]).toBe(2024);
  });

  it("date trunc year", async () => {
    const e = await makeTSEngine();
    const result = await e.sql("SELECT DATE_TRUNC('year', ts) AS t FROM log");
    const t = result!.rows[0]!["t"] as Date;
    expect(t).toBeInstanceOf(Date);
    expect(t.getFullYear()).toBe(2024);
    expect(t.getMonth()).toBe(0); // January
    expect(t.getDate()).toBe(1);
  });

  it("date trunc month", async () => {
    const e = await makeTSEngine();
    const result = await e.sql("SELECT DATE_TRUNC('month', ts) AS t FROM log");
    const t = result!.rows[0]!["t"] as Date;
    expect(t).toBeInstanceOf(Date);
    expect(t.getFullYear()).toBe(2024);
    expect(t.getMonth()).toBe(5); // June
    expect(t.getDate()).toBe(1);
  });

  it("date trunc day", async () => {
    const e = await makeTSEngine();
    const result = await e.sql("SELECT DATE_TRUNC('day', ts) AS t FROM log");
    const t = result!.rows[0]!["t"] as Date;
    expect(t).toBeInstanceOf(Date);
    expect(t.getFullYear()).toBe(2024);
    expect(t.getMonth()).toBe(5);
    expect(t.getDate()).toBe(15);
    expect(t.getHours()).toBe(0);
  });

  it("extract quarter", async () => {
    const e = await makeTSEngine();
    const result = await e.sql("SELECT EXTRACT(quarter FROM ts) AS q FROM log");
    expect(result!.rows[0]!["q"]).toBe(2);
  });

  it("extract week", async () => {
    const e = await makeTSEngine();
    const result = await e.sql("SELECT EXTRACT(week FROM ts) AS w FROM log");
    const w = result!.rows[0]!["w"] as number;
    expect(typeof w).toBe("number");
    expect(w).toBeGreaterThanOrEqual(1);
    expect(w).toBeLessThanOrEqual(53);
  });
});

// =============================================================================
// MAKE_TIMESTAMP
// =============================================================================

describe("MakeTimestamp", () => {
  it("basic", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT make_timestamp(2024, 3, 15, 10, 30, 0) AS ts");
    const ts = r!.rows[0]!["ts"] as Date;
    expect(ts).toBeInstanceOf(Date);
    expect(ts.getFullYear()).toBe(2024);
    expect(ts.getMonth()).toBe(2); // March
    expect(ts.getDate()).toBe(15);
    expect(ts.getHours()).toBe(10);
    expect(ts.getMinutes()).toBe(30);
  });

  it("with fractional seconds", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT make_timestamp(2024, 1, 1, 0, 0, 30.5) AS ts");
    const ts = r!.rows[0]!["ts"] as Date;
    expect(ts).toBeInstanceOf(Date);
    expect(ts.getFullYear()).toBe(2024);
    expect(ts.getSeconds()).toBe(30);
  });

  it("midnight", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT make_timestamp(2024, 12, 31, 0, 0, 0) AS ts");
    const ts = r!.rows[0]!["ts"] as Date;
    expect(ts).toBeInstanceOf(Date);
    expect(ts.getFullYear()).toBe(2024);
    expect(ts.getMonth()).toBe(11); // December
    expect(ts.getDate()).toBe(31);
    expect(ts.getHours()).toBe(0);
  });

  it("end of day", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT make_timestamp(2024, 6, 15, 23, 59, 59) AS ts");
    const ts = r!.rows[0]!["ts"] as Date;
    expect(ts).toBeInstanceOf(Date);
    expect(ts.getFullYear()).toBe(2024);
    expect(ts.getHours()).toBe(23);
    expect(ts.getMinutes()).toBe(59);
    expect(ts.getSeconds()).toBe(59);
  });
});

// =============================================================================
// MAKE_INTERVAL
// =============================================================================

describe("MakeInterval", () => {
  it("days hours minutes", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT make_interval(0, 0, 0, 1, 2, 30, 0) AS iv");
    const iv = r!.rows[0]!["iv"] as string;
    expect(iv).not.toBeNull();
    // Output: "1 day 02:30:00" (days separate from time)
    expect(iv).toContain("1 day");
    expect(iv).toContain("02:30:00");
  });

  it("hours minutes only", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT make_interval(0, 0, 0, 0, 1, 30, 0) AS iv");
    expect(r!.rows[0]!["iv"] as string).toContain("01:30:00");
  });

  it("zero interval", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT make_interval(0, 0, 0, 0, 0, 0, 0) AS iv");
    expect(r!.rows[0]!["iv"] as string).toContain("00:00:00");
  });
});

// =============================================================================
// TO_NUMBER
// =============================================================================

describe("ToNumber", () => {
  it("with currency and commas", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT to_number('$1,234.56', '9999.99') AS n");
    expect(Math.abs((r!.rows[0]!["n"] as number) - 1234.56)).toBeLessThan(0.01);
  });

  it("plain integer", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT to_number('42', '99') AS n");
    expect(r!.rows[0]!["n"]).toBe(42.0);
  });

  it("negative number", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT to_number('-99.5', '999.9') AS n");
    expect(Math.abs((r!.rows[0]!["n"] as number) - -99.5)).toBeLessThan(0.01);
  });

  it("with spaces", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT to_number('  100  ', '999') AS n");
    expect(Math.abs((r!.rows[0]!["n"] as number) - 100.0)).toBeLessThan(0.01);
  });
});

// =============================================================================
// OVERLAPS operator
// =============================================================================

describe("Overlaps", () => {
  it("overlapping ranges", async () => {
    const e = new Engine();
    const r = await e.sql(
      "SELECT overlaps(" +
        "'2024-01-01'::timestamp, '2024-06-01'::timestamp, " +
        "'2024-03-01'::timestamp, '2024-09-01'::timestamp) AS ov",
    );
    expect(r!.rows[0]!["ov"]).toBe(true);
  });

  it("non overlapping ranges", async () => {
    const e = new Engine();
    const r = await e.sql(
      "SELECT overlaps(" +
        "'2024-01-01'::timestamp, '2024-03-01'::timestamp, " +
        "'2024-06-01'::timestamp, '2024-09-01'::timestamp) AS ov",
    );
    expect(r!.rows[0]!["ov"]).toBe(false);
  });

  it("adjacent ranges do not overlap", async () => {
    const e = new Engine();
    const r = await e.sql(
      "SELECT overlaps(" +
        "'2024-01-01'::timestamp, '2024-03-01'::timestamp, " +
        "'2024-03-01'::timestamp, '2024-06-01'::timestamp) AS ov",
    );
    expect(r!.rows[0]!["ov"]).toBe(false);
  });

  it("function form", async () => {
    const e = new Engine();
    const r = await e.sql(
      "SELECT overlaps(" +
        "'2024-01-01'::timestamp, '2024-06-01'::timestamp, " +
        "'2024-03-01'::timestamp, '2024-09-01'::timestamp) AS ov",
    );
    expect(r!.rows[0]!["ov"]).toBe(true);
  });

  it("function form non overlapping", async () => {
    const e = new Engine();
    const r = await e.sql(
      "SELECT overlaps(" +
        "'2024-01-01'::timestamp, '2024-02-01'::timestamp, " +
        "'2024-06-01'::timestamp, '2024-07-01'::timestamp) AS ov",
    );
    expect(r!.rows[0]!["ov"]).toBe(false);
  });

  it("one range within another", async () => {
    const e = new Engine();
    const r = await e.sql(
      "SELECT overlaps(" +
        "'2024-01-01'::timestamp, '2024-12-31'::timestamp, " +
        "'2024-03-01'::timestamp, '2024-06-01'::timestamp) AS ov",
    );
    expect(r!.rows[0]!["ov"]).toBe(true);
  });
});
