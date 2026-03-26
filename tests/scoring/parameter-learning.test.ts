import { describe, expect, it } from "vitest";
import { ParameterLearner } from "../../src/scoring/parameter-learner.js";
import { Engine } from "../../src/engine.js";

// -- TestParameterLearner --

describe("TestParameterLearner", () => {
  it("init default params", () => {
    const learner = new ParameterLearner();
    const params = learner.params();
    expect(params["alpha"]).toBeCloseTo(1.0);
    expect(params["beta"]).toBeCloseTo(0.0);
    expect(params["baseRate"]).toBeCloseTo(0.5);
  });

  it("init custom params", () => {
    const learner = new ParameterLearner(2.0, 0.5, 0.3);
    const params = learner.params();
    expect(params["alpha"]).toBeCloseTo(2.0);
    expect(params["beta"]).toBeCloseTo(0.5);
    expect(params["baseRate"]).toBeCloseTo(0.3);
  });

  it("fit returns dict", () => {
    const learner = new ParameterLearner();
    const scores = [0.1, 0.3, 0.5, 0.7, 0.9];
    const labels = [0, 0, 0, 1, 1];
    const result = learner.fit(scores, labels);
    expect(typeof result).toBe("object");
    expect("alpha" in result).toBe(true);
    expect("beta" in result).toBe(true);
    expect("baseRate" in result).toBe(true);
  });

  it("fit changes params", () => {
    const learner = new ParameterLearner();
    const initial = learner.params();
    const scores = [0.1, 0.2, 0.8, 0.9];
    const labels = [0, 0, 1, 1];
    const learned = learner.fit(scores, labels);
    // At least one parameter should differ after fitting
    const changed =
      Math.abs(learned["alpha"]! - initial["alpha"]!) > 1e-6 ||
      Math.abs(learned["beta"]! - initial["beta"]!) > 1e-6 ||
      Math.abs(learned["baseRate"]! - initial["baseRate"]!) > 1e-6;
    expect(changed).toBe(true);
  });

  it("fit with mode", () => {
    const learner = new ParameterLearner();
    const scores = [0.1, 0.3, 0.7, 0.9];
    const labels = [0, 0, 1, 1];
    const result = learner.fit(scores, labels, { mode: "balanced" });
    expect(typeof result).toBe("object");
  });

  it("fit with tfs and doc len ratios", () => {
    const learner = new ParameterLearner();
    const scores = [0.1, 0.3, 0.7, 0.9];
    const labels = [0, 0, 1, 1];
    const tfs = [1, 2, 3, 4];
    const ratios = [0.8, 1.0, 1.2, 0.9];
    const result = learner.fit(scores, labels, { tfs, docLenRatios: ratios });
    expect(typeof result).toBe("object");
  });

  it("update modifies params", () => {
    const learner = new ParameterLearner();
    const initial = learner.params();
    for (let i = 0; i < 100; i++) {
      learner.update(0.9, 1);
      learner.update(0.1, 0);
    }
    const updated = learner.params();
    const changed =
      Math.abs(updated["alpha"]! - initial["alpha"]!) > 1e-6 ||
      Math.abs(updated["beta"]! - initial["beta"]!) > 1e-6;
    expect(changed).toBe(true);
  });

  it("update with tf and doc len", () => {
    const learner = new ParameterLearner();
    learner.update(0.5, 1, { tf: 3, docLenRatio: 1.2 });
    const params = learner.params();
    expect(typeof params).toBe("object");
  });

  it("params returns numbers", () => {
    const learner = new ParameterLearner();
    const params = learner.params();
    for (const key of ["alpha", "beta", "baseRate"]) {
      expect(typeof params[key]).toBe("number");
    }
  });
});

// -- TestEngineParameterLearning --

describe("TestEngineParameterLearning", () => {
  async function makeEngine(): Promise<Engine> {
    const e = new Engine();
    await e.sql("CREATE TABLE docs (id SERIAL PRIMARY KEY, content TEXT)");
    await e.sql("INSERT INTO docs (content) VALUES ('machine learning algorithms')");
    await e.sql("INSERT INTO docs (content) VALUES ('deep learning neural networks')");
    await e.sql("INSERT INTO docs (content) VALUES ('database indexing structures')");
    await e.sql("INSERT INTO docs (content) VALUES ('search engine optimization')");
    // Ensure table is registered in _tables by calling getTable
    e.getTable("docs");
    return e;
  }

  it("learn returns dict", async () => {
    const e = await makeEngine();
    const labels = [1, 1, 0, 0];
    const result = e.learnScoringParams("docs", "content", "learning", labels);
    expect(typeof result).toBe("object");
    expect("alpha" in result).toBe(true);
    expect("beta" in result).toBe(true);
    expect("baseRate" in result || "base_rate" in result).toBe(true);
  });

  it("learn wrong label count", async () => {
    const e = await makeEngine();
    expect(() => e.learnScoringParams("docs", "content", "learning", [1, 0])).toThrow(
      /labels length/,
    );
  });

  it("learn nonexistent table", async () => {
    const e = await makeEngine();
    expect(() =>
      e.learnScoringParams("nonexistent", "content", "learning", [1]),
    ).toThrow(/does not exist/);
  });

  it("update scoring params", async () => {
    const e = await makeEngine();
    // Should not throw
    e.updateScoringParams("docs", "content", 0.8, 1);
  });

  it("query builder learn params", async () => {
    const e = await makeEngine();
    const labels = [1, 1, 0, 0];
    const result = e
      .query("docs")
      .learnParams("learning", labels, { field: "content" });
    expect(typeof result).toBe("object");
    expect("alpha" in result).toBe(true);
  });
});
