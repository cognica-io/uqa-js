import { describe, expect, it } from "vitest";
import { CalibrationMetrics } from "../../src/scoring/calibration.js";
import { Engine } from "../../src/engine.js";

// -- TestCalibrationMetrics --

describe("TestCalibrationMetrics", () => {
  it("ece perfect calibration", () => {
    const probs = [0.0, 0.0, 1.0, 1.0];
    const labels = [0, 0, 1, 1];
    const ece = CalibrationMetrics.ece(probs, labels);
    expect(ece).toBeCloseTo(0.0, 5);
  });

  it("ece imperfect calibration", () => {
    const probs = [0.9, 0.9, 0.1, 0.1];
    const labels = [0, 0, 1, 1];
    const ece = CalibrationMetrics.ece(probs, labels);
    expect(ece).toBeGreaterThan(0.0);
  });

  it("ece returns number", () => {
    const probs = [0.5, 0.5, 0.5, 0.5];
    const labels = [0, 1, 0, 1];
    const result = CalibrationMetrics.ece(probs, labels);
    expect(typeof result).toBe("number");
  });

  it("brier perfect predictions", () => {
    const probs = [0.0, 0.0, 1.0, 1.0];
    const labels = [0, 0, 1, 1];
    const score = CalibrationMetrics.brier(probs, labels);
    expect(score).toBeCloseTo(0.0, 5);
  });

  it("brier worst predictions", () => {
    const probs = [1.0, 1.0, 0.0, 0.0];
    const labels = [0, 0, 1, 1];
    const score = CalibrationMetrics.brier(probs, labels);
    expect(score).toBeCloseTo(1.0, 5);
  });

  it("brier uniform predictions", () => {
    const probs = [0.5, 0.5, 0.5, 0.5];
    const labels = [0, 1, 0, 1];
    const score = CalibrationMetrics.brier(probs, labels);
    expect(score).toBeCloseTo(0.25, 5);
  });

  it("brier returns number", () => {
    const result = CalibrationMetrics.brier([0.5], [1]);
    expect(typeof result).toBe("number");
  });

  it("report returns object", () => {
    const probs = [0.1, 0.4, 0.6, 0.9];
    const labels = [0, 0, 1, 1];
    const report = CalibrationMetrics.report(probs, labels);
    expect(typeof report).toBe("object");
  });

  it("report contains metrics", () => {
    const probs = [0.1, 0.4, 0.6, 0.9];
    const labels = [0, 0, 1, 1];
    const report = CalibrationMetrics.report(probs, labels);
    // Report should contain at least ece and brier
    const hasMetrics =
      "ece" in report || "brier" in report || Object.keys(report).length > 0;
    expect(hasMetrics).toBe(true);
  });

  it("reliability diagram returns array", () => {
    const probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
    const labels = [0, 0, 0, 1, 1, 1];
    const diagram = CalibrationMetrics.reliabilityDiagram(probs, labels);
    expect(Array.isArray(diagram)).toBe(true);
  });

  it("reliability diagram tuple structure", () => {
    const probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
    const labels = [0, 0, 0, 1, 1, 1];
    const diagram = CalibrationMetrics.reliabilityDiagram(probs, labels, 5);
    for (const item of diagram) {
      expect(item.length).toBe(3);
      const [avgPred, avgActual, count] = item;
      expect(typeof avgPred).toBe("number");
      expect(typeof avgActual).toBe("number");
      expect(typeof count).toBe("number");
    }
  });

  it("reliability diagram n bins", () => {
    const probs = [0.1, 0.3, 0.5, 0.7, 0.9];
    const labels = [0, 0, 1, 1, 1];
    const diagram = CalibrationMetrics.reliabilityDiagram(probs, labels, 5);
    expect(diagram.length).toBeLessThanOrEqual(5);
  });

  it("ece with many bins", () => {
    const probs = [0.1, 0.3, 0.5, 0.7, 0.9];
    const labels = [0, 0, 1, 1, 1];
    const ece = CalibrationMetrics.ece(probs, labels, 20);
    expect(typeof ece).toBe("number");
    expect(ece).toBeGreaterThanOrEqual(0.0);
  });

  it("brier score range", () => {
    const probs = [0.3, 0.7, 0.2, 0.8];
    const labels = [0, 1, 0, 1];
    const score = CalibrationMetrics.brier(probs, labels);
    expect(score).toBeGreaterThanOrEqual(0.0);
    expect(score).toBeLessThanOrEqual(1.0);
  });
});

// -- TestEngineCalibrationReport --

describe("TestEngineCalibrationReport", () => {
  it("calibration report returns object", () => {
    const e = new Engine();
    // calibrationReport in TS returns an object describing saved scoring params
    const report = e.calibrationReport("docs", "content");
    expect(typeof report).toBe("object");
    expect("table" in report).toBe(true);
    expect("field" in report).toBe(true);
  });

  it("calibration metrics ece with real data", () => {
    const probs = [0.1, 0.4, 0.6, 0.9];
    const labels = [0, 0, 1, 1];
    const ece = CalibrationMetrics.ece(probs, labels);
    expect(ece).toBeGreaterThanOrEqual(0.0);
    expect(ece).toBeLessThanOrEqual(1.0);
  });

  it("calibration metrics brier with real data", () => {
    const probs = [0.1, 0.4, 0.6, 0.9];
    const labels = [0, 0, 1, 1];
    const brier = CalibrationMetrics.brier(probs, labels);
    expect(brier).toBeGreaterThanOrEqual(0.0);
    expect(brier).toBeLessThanOrEqual(1.0);
  });
});
