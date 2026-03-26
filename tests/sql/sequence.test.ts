import { describe, expect, it, beforeEach } from "vitest";
import { Engine } from "../../src/engine.js";

let engine: Engine;

beforeEach(() => {
  engine = new Engine();
});

describe("Sequence", () => {
  it("create and nextval", async () => {
    await engine.sql("CREATE SEQUENCE myseq START 1");
    const result = await engine.sql("SELECT nextval('myseq') AS v");
    expect(result!.rows[0]!["v"]).toBe(1);
    const result2 = await engine.sql("SELECT nextval('myseq') AS v");
    expect(result2!.rows[0]!["v"]).toBe(2);
  });

  it("currval", async () => {
    await engine.sql("CREATE SEQUENCE s2 START 10");
    await engine.sql("SELECT nextval('s2') AS v");
    const result = await engine.sql("SELECT currval('s2') AS v");
    expect(result!.rows[0]!["v"]).toBe(10);
  });

  it("setval", async () => {
    await engine.sql("CREATE SEQUENCE s3 START 1");
    await engine.sql("SELECT nextval('s3') AS v");
    await engine.sql("SELECT setval('s3', 100) AS v");
    const result = await engine.sql("SELECT currval('s3') AS v");
    expect(result!.rows[0]!["v"]).toBe(100);
  });

  it("increment", async () => {
    await engine.sql("CREATE SEQUENCE s4 START 1 INCREMENT 5");
    const result = await engine.sql("SELECT nextval('s4') AS v");
    expect(result!.rows[0]!["v"]).toBe(1);
    const result2 = await engine.sql("SELECT nextval('s4') AS v");
    expect(result2!.rows[0]!["v"]).toBe(6);
  });
});
