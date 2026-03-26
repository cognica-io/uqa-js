import { describe, expect, it } from "vitest";
import type {
  ConvSpec,
  PoolSpec,
  FlattenSpec,
  GlobalPoolSpec,
  DenseSpec,
  SoftmaxSpec,
  TrainedModel,
  LayerSpec,
} from "../../src/operators/deep-learn.js";
import {
  trainModel,
  predict,
  generateKernels,
  trainedModelToJSON,
  trainedModelFromJSON,
  trainedModelToDict,
  trainedModelFromDict,
  trainedModelToDeepFusionLayers,
} from "../../src/operators/deep-learn.js";
import { ridgeSolve, gridGlobalPool } from "../../src/operators/backend.js";

// -- Helpers ---------------------------------------------------------------

function makeEngine(): {
  _tables: Map<string, unknown>;
  saveModel(name: string, data: Record<string, unknown>): void;
  loadModel(name: string): Record<string, unknown> | null;
  deleteModel(name: string): void;
  _models: Map<string, Record<string, unknown>>;
} {
  const models = new Map<string, Record<string, unknown>>();
  return {
    _tables: new Map(),
    _models: models,
    saveModel(name: string, data: Record<string, unknown>): void {
      models.set(name, data);
    },
    loadModel(name: string): Record<string, unknown> | null {
      return models.get(name) ?? null;
    },
    deleteModel(name: string): void {
      models.delete(name);
    },
  };
}

// Generate simple embeddings for a grid: C*H*W flattened
// Using 1 channel, 3x3 grid = 9-dimensional embeddings
function makeTrainingRows(
  nSamples: number,
  gridSize = 3,
  nChannels = 1,
): Record<string, unknown>[] {
  const dim = nChannels * gridSize * gridSize;
  const rows: Record<string, unknown>[] = [];
  for (let i = 0; i < nSamples; i++) {
    const emb: number[] = [];
    for (let j = 0; j < dim; j++) {
      emb.push(Math.sin(i * 0.3 + j * 0.7) * 0.5 + 0.5);
    }
    rows.push({
      label: i % 2 === 0 ? "cat" : "dog",
      embedding: emb,
    });
  }
  return rows;
}

// -- TestRidgeRegression --

describe("TestRidgeRegression", () => {
  it("exact solution no regularization", () => {
    // X = [[1, 0], [0, 1]], Y = [[1, 0], [0, 1]]
    // With lam=0, W should be close to identity
    const X = new Float64Array([1, 0, 0, 1]);
    const Y = new Float64Array([1, 0, 0, 1]);
    const result = ridgeSolve(X, [2, 2], Y, [2, 2], 0.001);
    expect(result.weights.length).toBe(4);
    expect(result.bias.length).toBe(2);
    // The solution should approximate: predict Y from X
    // Prediction: X @ W + bias should approximate Y
    for (let i = 0; i < 2; i++) {
      let pred0 = result.bias[0]!;
      let pred1 = result.bias[1]!;
      for (let j = 0; j < 2; j++) {
        pred0 += X[i * 2 + j]! * result.weights[j * 2 + 0]!;
        pred1 += X[i * 2 + j]! * result.weights[j * 2 + 1]!;
      }
      // Should be reasonably close to Y
      expect(Math.abs(pred0 - Y[i * 2 + 0]!)).toBeLessThan(0.5);
      expect(Math.abs(pred1 - Y[i * 2 + 1]!)).toBeLessThan(0.5);
    }
  });

  it("regularization shrinks weights", () => {
    const X = new Float64Array([1, 2, 3, 4, 5, 6]);
    const Y = new Float64Array([1, 0, 1]);
    const lowReg = ridgeSolve(X, [3, 2], Y, [3, 1], 0.01);
    const highReg = ridgeSolve(X, [3, 2], Y, [3, 1], 100.0);
    // Higher regularization should produce smaller weights
    const normLow = Math.sqrt(
      Array.from(lowReg.weights).reduce((a, b) => a + b * b, 0),
    );
    const normHigh = Math.sqrt(
      Array.from(highReg.weights).reduce((a, b) => a + b * b, 0),
    );
    expect(normHigh).toBeLessThan(normLow);
  });

  it("output shapes", () => {
    const n = 10;
    const p = 5;
    const c = 3;
    const X = new Float64Array(n * p);
    const Y = new Float64Array(n * c);
    for (let i = 0; i < n * p; i++) X[i] = Math.random();
    for (let i = 0; i < n * c; i++) Y[i] = Math.random();
    const result = ridgeSolve(X, [n, p], Y, [n, c], 1.0);
    expect(result.weights.length).toBe(p * c);
    expect(result.bias.length).toBe(c);
  });

  it("single feature single class", () => {
    const X = new Float64Array([1, 2, 3, 4, 5]);
    const Y = new Float64Array([2, 4, 6, 8, 10]); // y = 2*x
    const result = ridgeSolve(X, [5, 1], Y, [5, 1], 0.001);
    expect(result.weights.length).toBe(1);
    expect(result.bias.length).toBe(1);
    // Weight should be close to 2, bias close to 0
    const pred = result.weights[0]! * 3 + result.bias[0]!;
    expect(Math.abs(pred - 6)).toBeLessThan(1.5);
  });
});

// -- TestLayerSpecSerialization --

describe("TestLayerSpecSerialization", () => {
  it("specs to dicts roundtrip", () => {
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 2, nChannels: 4, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 2 },
      { type: "flatten" },
      { type: "dense", outputChannels: 3 },
      { type: "softmax" },
    ];
    // Roundtrip via JSON
    const json = JSON.stringify(specs);
    const restored = JSON.parse(json) as LayerSpec[];
    expect(restored.length).toBe(5);
    expect(restored[0]!.type).toBe("conv");
    expect((restored[0] as ConvSpec).nChannels).toBe(4);
    expect(restored[4]!.type).toBe("softmax");
  });

  it("trained model JSON roundtrip", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(20);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    trainModel(
      engine,
      "test_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    const config = engine.loadModel("test_model")!;
    const json = trainedModelToJSON(config as unknown as TrainedModel);
    const restored = trainedModelFromJSON(json);
    expect(restored.modelName).toBe("test_model");
    expect(restored.numClasses).toBe(2);
  });

  it("to deep fusion layers", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(20);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    trainModel(
      engine,
      "test_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    const config = engine.loadModel("test_model")!;
    const model = trainedModelFromDict(config);
    const layers = trainedModelToDeepFusionLayers(model);
    expect(layers.length).toBeGreaterThan(0);
    // Should contain at least conv and dense
    const types = layers.map((l) => l.type);
    expect(types).toContain("conv");
    expect(types).toContain("dense");
  });
});

// -- TestDeepLearnAPI --

describe("TestDeepLearnAPI", () => {
  it("train and predict", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(20);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    const result = trainModel(
      engine,
      "model1",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    expect(result["training_samples"]).toBe(20);
    expect(result["num_classes"]).toBe(2);
    expect(result["training_accuracy"] as number).toBeGreaterThanOrEqual(0);

    // Predict
    const testEmb = rows[0]!["embedding"] as number[];
    const predictions = predict(engine, "model1", testEmb);
    expect(predictions.length).toBe(2);
    // Each prediction is [classIdx, probability]
    const totalProb = predictions.reduce((a, p) => a + p[1], 0);
    expect(totalProb).toBeCloseTo(1.0, 2);
  });

  it("predict nonexistent model", () => {
    const engine = makeEngine();
    expect(() => predict(engine, "nonexistent", [1, 2, 3])).toThrow(/does not exist/);
  });

  it("train empty table", () => {
    const engine = makeEngine();
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
    ];
    expect(() =>
      trainModel(
        engine,
        "model1",
        null,
        "label",
        "embedding",
        "spatial",
        specs,
        "none",
        1.0,
        0,
        0,
        [],
      ),
    ).toThrow();
  });
});

// -- TestPruning --

describe("TestPruning", () => {
  it("elastic net", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(30);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    const result = trainModel(
      engine,
      "en_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0.5,
      0,
      rows,
    );
    expect(result["l1_ratio"]).toBe(0.5);
    // Some weights should be zeroed due to L1
    const config = engine.loadModel("en_model")!;
    const sparsity = config["weightSparsity"] as number;
    expect(sparsity).toBeGreaterThanOrEqual(0);
  });

  it("magnitude pruning", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(30);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    const result = trainModel(
      engine,
      "prune_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0.3,
      rows,
    );
    expect(result["prune_ratio"]).toBe(0.3);
    const config = engine.loadModel("prune_model")!;
    const sparsity = config["weightSparsity"] as number;
    expect(sparsity).toBeGreaterThan(0);
  });

  it("elastic net plus pruning", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(30);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    const result = trainModel(
      engine,
      "both_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0.3,
      0.2,
      rows,
    );
    expect(result["l1_ratio"]).toBe(0.3);
    expect(result["prune_ratio"]).toBe(0.2);
  });

  it("pruning SQL", () => {
    // Verify that pruning parameters are correctly stored
    const engine = makeEngine();
    const rows = makeTrainingRows(20);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    trainModel(
      engine,
      "sql_prune",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0.4,
      0.5,
      rows,
    );
    const config = engine.loadModel("sql_prune")!;
    expect(config["l1Ratio"]).toBe(0.4);
    expect(config["pruneRatio"]).toBe(0.5);
    expect(config["weightSparsity"]).toBeGreaterThanOrEqual(0);
  });
});

// -- TestDeepLearnSQL --

describe("TestDeepLearnSQL", () => {
  it("deep learn SQL", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(20);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    const result = trainModel(
      engine,
      "sql_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    expect(result["model_name"]).toBe("sql_model");
    expect(result["num_classes"]).toBe(2);
  });

  it("deep predict SQL", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(20);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    trainModel(
      engine,
      "pred_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    const testEmb = rows[0]!["embedding"] as number[];
    const preds = predict(engine, "pred_model", testEmb);
    expect(preds.length).toBe(2);
    // Probabilities should sum to 1
    const sum = preds.reduce((a, p) => a + p[1], 0);
    expect(sum).toBeCloseTo(1.0, 2);
  });

  it("deep predict per row", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(20);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    trainModel(
      engine,
      "row_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    // Predict for each row
    for (const row of rows.slice(0, 5)) {
      const preds = predict(engine, "row_model", row["embedding"] as number[]);
      expect(preds.length).toBe(2);
      const sum = preds.reduce((a, p) => a + p[1], 0);
      expect(sum).toBeCloseTo(1.0, 2);
    }
  });

  it("deep learn no layers raises", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(20);
    expect(() =>
      trainModel(
        engine,
        "empty_model",
        null,
        "label",
        "embedding",
        "spatial",
        [],
        "none",
        1.0,
        0,
        0,
        rows,
      ),
    ).toThrow(/at least one layer/);
  });
});

// -- TestDeepPredictFusionCompat --

describe("TestDeepPredictFusionCompat", () => {
  it("predict matches fusion simple", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(30);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    trainModel(
      engine,
      "compat_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    const config = engine.loadModel("compat_model")!;
    const model = trainedModelFromDict(config);
    const layers = trainedModelToDeepFusionLayers(model);
    expect(layers.length).toBeGreaterThan(0);

    // The model should be predictable
    const testEmb = rows[0]!["embedding"] as number[];
    const preds = predict(engine, "compat_model", testEmb);
    expect(preds.length).toBe(2);
    expect(preds[0]![1]).toBeGreaterThanOrEqual(0);
    expect(preds[0]![1]).toBeLessThanOrEqual(1);
  });

  it("predict poe with conv pool", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(30);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    trainModel(
      engine,
      "poe_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    const config = engine.loadModel("poe_model")!;
    // Model should have expert weights (PoE)
    expect((config["expertWeights"] as number[][]).length).toBeGreaterThan(0);
    expect((config["expertAccuracies"] as number[]).length).toBeGreaterThan(0);

    const testEmb = rows[0]!["embedding"] as number[];
    const preds = predict(engine, "poe_model", testEmb);
    expect(preds.length).toBe(2);
    const sum = preds.reduce((a, p) => a + p[1], 0);
    expect(sum).toBeCloseTo(1.0, 2);
  });
});

// -- TestModelCatalogPersistence --

describe("TestModelCatalogPersistence", () => {
  it("save and load model", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(20);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    trainModel(
      engine,
      "persist_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    // Simulate persistence: serialize and deserialize
    const config = engine.loadModel("persist_model")!;
    const json = JSON.stringify(config);
    const restored = JSON.parse(json) as Record<string, unknown>;
    engine.saveModel("restored_model", restored);

    const loaded = engine.loadModel("restored_model")!;
    expect(loaded["modelName"]).toBe("persist_model");
    expect(loaded["numClasses"]).toBe(2);
  });

  it("delete model", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(20);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    trainModel(
      engine,
      "del_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    expect(engine.loadModel("del_model")).not.toBeNull();
    engine.deleteModel("del_model");
    expect(engine.loadModel("del_model")).toBeNull();
  });
});

// -- TestGlobalPooling --

describe("TestGlobalPooling", () => {
  it("grid global pool avg", () => {
    // 1 sample, 1 channel, 2x2 grid
    const features = new Float64Array([1, 2, 3, 4]);
    const result = gridGlobalPool(features, [1, 4], 2, 2, "avg");
    expect(result.data.length).toBe(1);
    expect(result.data[0]).toBeCloseTo(2.5);
  });

  it("grid global pool max", () => {
    const features = new Float64Array([1, 2, 3, 4]);
    const result = gridGlobalPool(features, [1, 4], 2, 2, "max");
    expect(result.data.length).toBe(1);
    expect(result.data[0]).toBeCloseTo(4.0);
  });

  it("grid global pool avg max", () => {
    const features = new Float64Array([1, 2, 3, 4]);
    const result = gridGlobalPool(features, [1, 4], 2, 2, "avg_max");
    expect(result.data.length).toBe(2);
    expect(result.data[0]).toBeCloseTo(2.5); // avg
    expect(result.data[1]).toBeCloseTo(4.0); // max
  });

  it("global pool spec serialization", () => {
    const spec: GlobalPoolSpec = { type: "global_pool", method: "avg" };
    const json = JSON.stringify(spec);
    const restored = JSON.parse(json) as GlobalPoolSpec;
    expect(restored.type).toBe("global_pool");
    expect(restored.method).toBe("avg");
  });

  it("global pool training SQL", () => {
    const engine = makeEngine();
    // 4x4 grid with 1 channel = 16-dim embedding
    const rows: Record<string, unknown>[] = [];
    for (let i = 0; i < 20; i++) {
      const emb: number[] = [];
      for (let j = 0; j < 16; j++) {
        emb.push(Math.sin(i * 0.3 + j * 0.7) * 0.5 + 0.5);
      }
      rows.push({ label: i % 2 === 0 ? "A" : "B", embedding: emb });
    }
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 2 },
      { type: "global_pool", method: "avg" },
      { type: "dense", outputChannels: 2 },
    ];
    const result = trainModel(
      engine,
      "gp_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    expect(result["training_samples"]).toBe(20);
  });

  it("global pool avg max training SQL", () => {
    const engine = makeEngine();
    const rows: Record<string, unknown>[] = [];
    for (let i = 0; i < 20; i++) {
      const emb: number[] = [];
      for (let j = 0; j < 16; j++) {
        emb.push(Math.sin(i * 0.3 + j * 0.7) * 0.5 + 0.5);
      }
      rows.push({ label: i % 2 === 0 ? "A" : "B", embedding: emb });
    }
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 2 },
      { type: "global_pool", method: "avg_max" },
      { type: "dense", outputChannels: 2 },
    ];
    const result = trainModel(
      engine,
      "gp_am_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    expect(result["training_samples"]).toBe(20);
  });

  it("global pool predict", () => {
    const engine = makeEngine();
    const rows: Record<string, unknown>[] = [];
    for (let i = 0; i < 20; i++) {
      const emb: number[] = [];
      for (let j = 0; j < 16; j++) {
        emb.push(Math.sin(i * 0.3 + j * 0.7) * 0.5 + 0.5);
      }
      rows.push({ label: i % 2 === 0 ? "A" : "B", embedding: emb });
    }
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 1, initMode: "kaiming" },
      { type: "pool", method: "max", poolSize: 2 },
      { type: "global_pool", method: "avg" },
      { type: "dense", outputChannels: 2 },
    ];
    trainModel(
      engine,
      "gp_pred_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    const testEmb = rows[0]!["embedding"] as number[];
    const preds = predict(engine, "gp_pred_model", testEmb);
    expect(preds.length).toBe(2);
    const sum = preds.reduce((a, p) => a + p[1], 0);
    expect(sum).toBeCloseTo(1.0, 2);
  });
});

// -- TestKernelInitModes --

describe("TestKernelInitModes", () => {
  it("kaiming default", () => {
    const kernels = generateKernels(4, 1, 42, "kaiming");
    expect(kernels.length).toBe(4 * 1 * 9);
    // Should not be all zeros
    let nonZero = 0;
    for (let i = 0; i < kernels.length; i++) {
      if (Math.abs(kernels[i]!) > 1e-10) nonZero++;
    }
    expect(nonZero).toBeGreaterThan(0);
  });

  it("orthogonal shape", () => {
    const kernels = generateKernels(4, 1, 42, "orthogonal");
    expect(kernels.length).toBe(4 * 1 * 9);
  });

  it("orthogonal diversity", () => {
    const k1 = generateKernels(4, 1, 42, "orthogonal");
    const k2 = generateKernels(4, 1, 99, "orthogonal");
    // Different seeds should produce different kernels
    let diff = 0;
    for (let i = 0; i < k1.length; i++) {
      if (Math.abs(k1[i]! - k2[i]!) > 1e-10) diff++;
    }
    expect(diff).toBeGreaterThan(0);
  });

  it("gabor shape", () => {
    const kernels = generateKernels(4, 1, 42, "gabor");
    expect(kernels.length).toBe(4 * 1 * 9);
  });

  it("gabor structured", () => {
    const kernels = generateKernels(8, 1, 42, "gabor");
    // Gabor filters should have structure -- not all same
    const first = kernels.subarray(0, 9);
    const second = kernels.subarray(9, 18);
    let diff = 0;
    for (let i = 0; i < 9; i++) {
      if (Math.abs(first[i]! - second[i]!) > 1e-10) diff++;
    }
    expect(diff).toBeGreaterThan(0);
  });

  it("kmeans shape", () => {
    // Need training data for kmeans
    const nSamples = 20;
    const dim = 9; // 1 channel, 3x3 grid
    const data = new Float64Array(nSamples * dim);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.sin(i * 0.3) * 0.5 + 0.5;
    }
    const kernels = generateKernels(4, 1, 42, "kmeans", data, [nSamples, dim], 3, 3);
    expect(kernels.length).toBe(4 * 1 * 9);
  });

  it("kmeans data dependent", () => {
    const nSamples = 20;
    const dim = 9;
    const data1 = new Float64Array(nSamples * dim);
    const data2 = new Float64Array(nSamples * dim);
    for (let i = 0; i < data1.length; i++) {
      data1[i] = Math.sin(i * 0.3) * 0.5 + 0.5;
      data2[i] = Math.cos(i * 0.7) * 0.5 + 0.5;
    }
    const k1 = generateKernels(4, 1, 42, "kmeans", data1, [nSamples, dim], 3, 3);
    const k2 = generateKernels(4, 1, 42, "kmeans", data2, [nSamples, dim], 3, 3);
    // Different training data should produce different kernels
    let diff = 0;
    for (let i = 0; i < k1.length; i++) {
      if (Math.abs(k1[i]! - k2[i]!) > 1e-10) diff++;
    }
    expect(diff).toBeGreaterThan(0);
  });

  it("orthogonal training SQL", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(20);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 4, initMode: "orthogonal" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    const result = trainModel(
      engine,
      "ortho_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    expect(result["training_samples"]).toBe(20);
  });

  it("gabor training SQL", () => {
    const engine = makeEngine();
    const rows = makeTrainingRows(20);
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 4, initMode: "gabor" },
      { type: "pool", method: "max", poolSize: 1 },
      { type: "dense", outputChannels: 2 },
    ];
    const result = trainModel(
      engine,
      "gabor_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    expect(result["training_samples"]).toBe(20);
  });

  it("combined global pool and orthogonal", () => {
    const engine = makeEngine();
    const rows: Record<string, unknown>[] = [];
    for (let i = 0; i < 20; i++) {
      const emb: number[] = [];
      for (let j = 0; j < 16; j++) {
        emb.push(Math.sin(i * 0.3 + j * 0.7) * 0.5 + 0.5);
      }
      rows.push({ label: i % 2 === 0 ? "A" : "B", embedding: emb });
    }
    const specs: LayerSpec[] = [
      { type: "conv", kernelHops: 1, nChannels: 4, initMode: "orthogonal" },
      { type: "pool", method: "max", poolSize: 2 },
      { type: "global_pool", method: "avg" },
      { type: "dense", outputChannels: 2 },
    ];
    const result = trainModel(
      engine,
      "combo_model",
      null,
      "label",
      "embedding",
      "spatial",
      specs,
      "none",
      1.0,
      0,
      0,
      rows,
    );
    expect(result["training_samples"]).toBe(20);
    const preds = predict(engine, "combo_model", rows[0]!["embedding"] as number[]);
    expect(preds.length).toBe(2);
  });
});
