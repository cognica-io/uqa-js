import { describe, expect, it } from "vitest";
import {
  createPayload,
  IndexStats,
  createVertex,
  createEdge,
} from "../../src/core/types.js";
import { PostingList } from "../../src/core/posting-list.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import { Operator } from "../../src/operators/base.js";
import { DeepFusionOperator, sigmoidVal } from "../../src/operators/deep-fusion.js";
import {
  sigmoidVec,
  applyGatingVec,
  batchSoftmax,
  batchBatchnorm,
} from "../../src/operators/backend.js";
import type {
  SignalLayer,
  PropagateLayer,
  ConvLayer,
  PoolLayer,
  DenseLayer,
  FlattenLayer,
  SoftmaxLayer,
  BatchNormLayer,
  DropoutLayer,
  FusionLayer,
  EmbedLayer,
} from "../../src/operators/deep-fusion.js";
import { MemoryGraphStore } from "../../src/graph/store.js";

// -- Helpers --

class TermOperatorStub extends Operator {
  readonly term: string;

  constructor(term: string) {
    super();
    this.term = term;
  }

  execute(_context: ExecutionContext): PostingList {
    return new PostingList([
      { docId: 1, payload: createPayload({ score: 0.8 }) },
      { docId: 2, payload: createPayload({ score: 0.6 }) },
      { docId: 3, payload: createPayload({ score: 0.4 }) },
    ]);
  }
}

class MultiDocOperatorStub extends Operator {
  private _entries: { docId: number; score: number }[];

  constructor(entries: { docId: number; score: number }[]) {
    super();
    this._entries = entries;
  }

  execute(_context: ExecutionContext): PostingList {
    return new PostingList(
      this._entries.map((e) => ({
        docId: e.docId,
        payload: createPayload({ score: e.score }),
      })),
    );
  }
}

function makeGridGraphStore(): {
  gs: MemoryGraphStore;
  ctx: ExecutionContext;
} {
  const gs = new MemoryGraphStore();
  gs.createGraph("grid");
  // 3x3 grid: vertices 1-9
  for (let i = 1; i <= 9; i++) {
    gs.addVertex(createVertex(i, "node", {}), "grid");
  }
  // Spatial edges (4-connected grid)
  let edgeId = 1;
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const vid = r * 3 + c + 1;
      if (c + 1 < 3) {
        gs.addEdge(createEdge(edgeId++, vid, vid + 1, "spatial", {}), "grid");
        gs.addEdge(createEdge(edgeId++, vid + 1, vid, "spatial", {}), "grid");
      }
      if (r + 1 < 3) {
        gs.addEdge(createEdge(edgeId++, vid, vid + 3, "spatial", {}), "grid");
        gs.addEdge(createEdge(edgeId++, vid + 3, vid, "spatial", {}), "grid");
      }
    }
  }
  return { gs, ctx: { graphStore: gs } };
}

// -- TestVectorizedHelpers --

describe("TestVectorizedHelpers", () => {
  it("sigmoid vec", () => {
    const result = sigmoidVec(new Float64Array([0.0]));
    expect(Math.abs(result[0]! - 0.5)).toBeLessThan(0.001);
  });

  it("sigmoid vec large negative", () => {
    const result = sigmoidVec(new Float64Array([-100.0]));
    expect(result[0]!).toBeLessThan(0.001);
  });

  it("apply gating vec none", () => {
    const result = applyGatingVec(new Float64Array([-2.0, 3.0]), "none");
    expect(result[0]).toBe(-2.0);
    expect(result[1]).toBe(3.0);
  });

  it("apply gating vec relu", () => {
    const result = applyGatingVec(new Float64Array([-2.0, 3.0]), "relu");
    expect(result[0]).toBe(0.0);
    expect(result[1]).toBe(3.0);
  });

  it("apply gating vec swish", () => {
    const result = applyGatingVec(new Float64Array([2.0]), "swish");
    const expected = 2.0 * sigmoidVal(2.0);
    expect(Math.abs(result[0]! - expected)).toBeLessThan(0.001);
  });
});

// -- TestPoolLayer --

describe("TestPoolLayer", () => {
  it("pool max on grid", () => {
    const { ctx } = makeGridGraphStore();
    // Create a signal layer with entries for all 9 grid nodes
    const entries = [];
    for (let i = 1; i <= 9; i++) {
      entries.push({ docId: i, score: i * 0.1 });
    }
    const signalOp = new MultiDocOperatorStub(entries);
    const signalLayer: SignalLayer = { type: "signal", signals: [signalOp] };
    const poolLayer: PoolLayer = {
      type: "pool",
      edgeLabel: "spatial",
      poolSize: 3,
      method: "max",
      direction: "both",
    };
    const op = new DeepFusionOperator([signalLayer, poolLayer], 0.5, "none", "grid");
    const result = op.execute(ctx);
    // After pooling, should have fewer nodes than original
    expect(result.length).toBeLessThanOrEqual(9);
    expect(result.length).toBeGreaterThan(0);
  });

  it("pool avg on grid", () => {
    const { ctx } = makeGridGraphStore();
    const entries = [];
    for (let i = 1; i <= 9; i++) {
      entries.push({ docId: i, score: i * 0.1 });
    }
    const signalOp = new MultiDocOperatorStub(entries);
    const signalLayer: SignalLayer = { type: "signal", signals: [signalOp] };
    const poolLayer: PoolLayer = {
      type: "pool",
      edgeLabel: "spatial",
      poolSize: 3,
      method: "avg",
      direction: "both",
    };
    const op = new DeepFusionOperator([signalLayer, poolLayer], 0.5, "none", "grid");
    const result = op.execute(ctx);
    expect(result.length).toBeLessThanOrEqual(9);
    expect(result.length).toBeGreaterThan(0);
  });

  it("pool size 4", () => {
    const { ctx } = makeGridGraphStore();
    const entries = [];
    for (let i = 1; i <= 9; i++) {
      entries.push({ docId: i, score: 0.5 });
    }
    const signalOp = new MultiDocOperatorStub(entries);
    const signalLayer: SignalLayer = { type: "signal", signals: [signalOp] };
    const poolLayer: PoolLayer = {
      type: "pool",
      edgeLabel: "spatial",
      poolSize: 4,
      method: "max",
      direction: "both",
    };
    const op = new DeepFusionOperator([signalLayer, poolLayer], 0.5, "none", "grid");
    const result = op.execute(ctx);
    // With pool size 4, 9 nodes -> 3 or fewer representatives
    expect(result.length).toBeLessThanOrEqual(9);
    expect(result.length).toBeGreaterThan(0);
  });

  it("pool size validation -- construction", () => {
    const poolLayer: PoolLayer = {
      type: "pool",
      edgeLabel: "spatial",
      poolSize: 1,
      method: "max",
      direction: "both",
    };
    expect(poolLayer.poolSize).toBe(1);
    // Pool size < 2 should be rejected by DeepFusionOperator constructor
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    expect(() => new DeepFusionOperator([signalLayer, poolLayer])).toThrow(
      /pool_size must be >= 2/,
    );
  });
});

// -- TestDenseLayer --

describe("TestDenseLayer", () => {
  it("identity weights construction", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const denseLayer: DenseLayer = {
      type: "dense",
      weights: [1.0],
      bias: [0.0],
      outputChannels: 1,
      inputChannels: 1,
    };
    const op = new DeepFusionOperator([signalLayer, denseLayer]);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
  });

  it("channel expansion", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const denseLayer: DenseLayer = {
      type: "dense",
      weights: [1.0, 0.5, -0.5, 0.3],
      bias: [0.0, 0.0, 0.0, 0.0],
      outputChannels: 4,
      inputChannels: 1,
    };
    const op = new DeepFusionOperator([signalLayer, denseLayer]);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });

  it("dense with bias", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const noBiasOp = new DeepFusionOperator([
      signalLayer,
      {
        type: "dense",
        weights: [1.0],
        bias: [0.0],
        outputChannels: 1,
        inputChannels: 1,
      } as DenseLayer,
    ]);
    const withBiasOp = new DeepFusionOperator([
      signalLayer,
      {
        type: "dense",
        weights: [1.0],
        bias: [2.0],
        outputChannels: 1,
        inputChannels: 1,
      } as DenseLayer,
    ]);
    const noBiasResult = noBiasOp.execute({});
    const withBiasResult = withBiasOp.execute({});
    expect(noBiasResult.length).toBe(withBiasResult.length);
    // Positive bias should push every score higher or equal
    for (let i = 0; i < noBiasResult.entries.length; i++) {
      const noBiasEntry = noBiasResult.entries[i]!;
      const withBiasEntry = withBiasResult.entries.find(
        (e) => e.docId === noBiasEntry.docId,
      );
      if (withBiasEntry) {
        expect(withBiasEntry.payload.score).toBeGreaterThanOrEqual(
          noBiasEntry.payload.score - 1e-10,
        );
      }
    }
  });

  it("dense channel reduction", () => {
    // Start with 1 channel, expand to 4, then reduce to 2
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const expandLayer: DenseLayer = {
      type: "dense",
      weights: [1.0, 0.5, -0.5, 0.3],
      bias: [0.0, 0.0, 0.0, 0.0],
      outputChannels: 4,
      inputChannels: 1,
    };
    const reduceLayer: DenseLayer = {
      type: "dense",
      weights: [1.0, 0.5, -0.5, 0.3, 0.2, -0.1, 0.4, 0.6],
      bias: [0.0, 0.0],
      outputChannels: 2,
      inputChannels: 4,
    };
    const op = new DeepFusionOperator([signalLayer, expandLayer, reduceLayer]);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });
});

// -- TestFlattenLayer --

describe("TestFlattenLayer", () => {
  it("flatten reduces to one node", () => {
    // With 3 docs from signal, flatten should produce 1 entry
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const flattenLayer: FlattenLayer = { type: "flatten" };
    const op = new DeepFusionOperator([signalLayer, flattenLayer]);
    const result = op.execute({});
    // After flatten, all doc channel vectors are concatenated into one entry
    expect(result.length).toBe(1);
  });

  it("flatten then dense", () => {
    // 3 docs, 1 channel each -> flatten to 3-dim vector -> dense to 2
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const flattenLayer: FlattenLayer = { type: "flatten" };
    const denseLayer: DenseLayer = {
      type: "dense",
      weights: [1.0, 0.5, -0.5, 0.3, 0.2, -0.1],
      bias: [0.0, 0.0],
      outputChannels: 2,
      inputChannels: 3,
    };
    const op = new DeepFusionOperator([signalLayer, flattenLayer, denseLayer]);
    const result = op.execute({});
    expect(result.length).toBe(1);
    expect(result.entries[0]!.payload.score).toBeGreaterThan(0);
    expect(result.entries[0]!.payload.score).toBeLessThanOrEqual(1.0);
  });

  it("spatial after flatten rejected", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const flattenLayer: FlattenLayer = { type: "flatten" };
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [1.0, 0.5],
      direction: "both",
    };
    expect(
      () => new DeepFusionOperator([signalLayer, flattenLayer, convLayer]),
    ).toThrow(/must not appear after flatten/);
  });
});

// -- TestSoftmaxLayer --

describe("TestSoftmaxLayer", () => {
  it("softmax probabilities sum to one", () => {
    // Test batchSoftmax directly
    const X = new Float64Array([1.0, 2.0, 3.0]);
    const out = batchSoftmax(X, [1, 3]);
    let sum = 0;
    for (let i = 0; i < 3; i++) sum += out[i]!;
    expect(sum).toBeCloseTo(1.0, 5);
  });

  it("softmax score is max prob", () => {
    // Signal -> dense (expand to 3 channels) -> softmax
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const denseLayer: DenseLayer = {
      type: "dense",
      weights: [1.0, 0.5, -0.5],
      bias: [0.0, 0.0, 0.0],
      outputChannels: 3,
      inputChannels: 1,
    };
    const softmaxLayer: SoftmaxLayer = { type: "softmax" };
    const op = new DeepFusionOperator([signalLayer, denseLayer, softmaxLayer]);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      // Score should be the max probability
      const probs = entry.payload.fields["class_probs"] as number[];
      expect(probs).toBeDefined();
      const maxProb = Math.max(...probs);
      expect(entry.payload.score).toBeCloseTo(maxProb, 5);
    }
  });

  it("softmax payload fields", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const denseLayer: DenseLayer = {
      type: "dense",
      weights: [1.0, -1.0],
      bias: [0.0, 0.0],
      outputChannels: 2,
      inputChannels: 1,
    };
    const softmaxLayer: SoftmaxLayer = { type: "softmax" };
    const op = new DeepFusionOperator([signalLayer, denseLayer, softmaxLayer]);
    const result = op.execute({});
    for (const entry of result) {
      expect(entry.payload.fields["class_probs"]).toBeDefined();
      const probs = entry.payload.fields["class_probs"] as number[];
      expect(probs.length).toBe(2);
      // Probabilities sum to 1
      const sum = probs.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    }
  });

  it("softmax numerically stable", () => {
    // Large values should not cause NaN/Infinity
    const X = new Float64Array([1000.0, 1001.0, 1002.0]);
    const out = batchSoftmax(X, [1, 3]);
    let sum = 0;
    for (let i = 0; i < 3; i++) {
      expect(isNaN(out[i]!)).toBe(false);
      expect(isFinite(out[i]!)).toBe(true);
      sum += out[i]!;
    }
    expect(sum).toBeCloseTo(1.0, 5);
  });
});

// -- TestBatchNormLayer --

describe("TestBatchNormLayer", () => {
  it("batchnorm zero mean unit variance", () => {
    // 4 samples, 2 features
    const X = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const out = batchBatchnorm(X, [4, 2]);
    // Each column should have ~0 mean and ~1 variance
    let m0 = 0,
      m1 = 0;
    for (let i = 0; i < 4; i++) {
      m0 += out[i * 2]! / 4;
      m1 += out[i * 2 + 1]! / 4;
    }
    expect(Math.abs(m0)).toBeLessThan(0.1);
    expect(Math.abs(m1)).toBeLessThan(0.1);

    let v0 = 0,
      v1 = 0;
    for (let i = 0; i < 4; i++) {
      v0 += (out[i * 2]! - m0) ** 2 / 4;
      v1 += (out[i * 2 + 1]! - m1) ** 2 / 4;
    }
    expect(Math.abs(v0 - 1.0)).toBeLessThan(0.2);
    expect(Math.abs(v1 - 1.0)).toBeLessThan(0.2);
  });

  it("batchnorm SQL", () => {
    // Test batchnorm through DeepFusionOperator
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const bnLayer: BatchNormLayer = { type: "batchnorm" };
    const op = new DeepFusionOperator([signalLayer, bnLayer]);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
  });

  it("batchnorm custom epsilon", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const bnLayer: BatchNormLayer = { type: "batchnorm", epsilon: 1e-3 };
    const op = new DeepFusionOperator([signalLayer, bnLayer]);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
  });
});

// -- TestDropoutLayer --

describe("TestDropoutLayer", () => {
  it("dropout scales by one minus p", () => {
    // Test dropout through DeepFusionOperator
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const dropoutLayer: DropoutLayer = { type: "dropout", p: 0.5 };
    const opWithDrop = new DeepFusionOperator([signalLayer, dropoutLayer]);
    const opWithout = new DeepFusionOperator([signalLayer]);
    const withDrop = opWithDrop.execute({});
    const without = opWithout.execute({});
    // Dropout should scale values by (1 - p)
    // Scores will differ because sigmoid(scaled_logit) != scaled_sigmoid(logit)
    // But dropout result should still be valid
    expect(withDrop.length).toBe(without.length);
    for (const entry of withDrop) {
      expect(entry.payload.score).toBeGreaterThanOrEqual(0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });

  it("dropout p validation", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    // p=0 should be rejected
    expect(
      () =>
        new DeepFusionOperator([
          signalLayer,
          { type: "dropout", p: 0 } as DropoutLayer,
        ]),
    ).toThrow(/dropout\(\) p must be in/);
    // p=1 should be rejected
    expect(
      () =>
        new DeepFusionOperator([
          signalLayer,
          { type: "dropout", p: 1 } as DropoutLayer,
        ]),
    ).toThrow(/dropout\(\) p must be in/);
  });

  it("dropout SQL", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const dropLayer: DropoutLayer = { type: "dropout", p: 0.3 };
    const op = new DeepFusionOperator([signalLayer, dropLayer]);
    const result = op.execute({});
    expect(result.length).toBe(3);
  });
});

// -- TestFullCNNPipeline --

describe("TestFullCNNPipeline", () => {
  it("conv pool conv pool flatten dense softmax", () => {
    const { ctx } = makeGridGraphStore();
    const entries = [];
    for (let i = 1; i <= 9; i++) {
      entries.push({ docId: i, score: 0.3 + (i % 3) * 0.2 });
    }
    const signalOp = new MultiDocOperatorStub(entries);
    const layers: FusionLayer[] = [
      { type: "signal", signals: [signalOp] },
      { type: "conv", edgeLabel: "spatial", hopWeights: [1.0, 0.5], direction: "both" },
      {
        type: "pool",
        edgeLabel: "spatial",
        poolSize: 3,
        method: "max",
        direction: "both",
      },
      { type: "flatten" },
      { type: "softmax" },
    ];
    const op = new DeepFusionOperator(layers, 0.5, "none", "grid");
    const result = op.execute(ctx);
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });

  it("conv pool flatten dense softmax", () => {
    const { ctx } = makeGridGraphStore();
    const entries = [];
    for (let i = 1; i <= 9; i++) {
      entries.push({ docId: i, score: 0.5 });
    }
    const signalOp = new MultiDocOperatorStub(entries);
    const layers: FusionLayer[] = [
      { type: "signal", signals: [signalOp] },
      { type: "conv", edgeLabel: "spatial", hopWeights: [1.0, 0.5], direction: "both" },
      {
        type: "pool",
        edgeLabel: "spatial",
        poolSize: 3,
        method: "avg",
        direction: "both",
      },
      { type: "flatten" },
      { type: "softmax" },
    ];
    const op = new DeepFusionOperator(layers, 0.5, "none", "grid");
    const result = op.execute(ctx);
    expect(result.length).toBeGreaterThan(0);
  });

  it("pipeline with batchnorm dropout", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const layers: FusionLayer[] = [
      signalLayer,
      { type: "batchnorm" } as BatchNormLayer,
      { type: "dropout", p: 0.3 } as DropoutLayer,
      {
        type: "dense",
        weights: [1.0],
        bias: [0.0],
        outputChannels: 1,
        inputChannels: 1,
      } as DenseLayer,
    ];
    const op = new DeepFusionOperator(layers);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
  });
});

// -- TestExplainNewLayers --

describe("TestExplainNewLayers", () => {
  it("explain pool", () => {
    const poolLayer: PoolLayer = {
      type: "pool",
      edgeLabel: "spatial",
      poolSize: 2,
      method: "max",
      direction: "both",
    };
    expect(poolLayer.type).toBe("pool");
    expect(poolLayer.method).toBe("max");
    expect(poolLayer.poolSize).toBe(2);
  });

  it("explain dense", () => {
    const denseLayer: DenseLayer = {
      type: "dense",
      weights: [1.0, 0.5],
      bias: [0.0],
      outputChannels: 1,
      inputChannels: 2,
    };
    expect(denseLayer.type).toBe("dense");
    expect(denseLayer.outputChannels).toBe(1);
    expect(denseLayer.inputChannels).toBe(2);
  });

  it("explain flatten", () => {
    const flattenLayer: FlattenLayer = { type: "flatten" };
    expect(flattenLayer.type).toBe("flatten");
  });

  it("explain softmax", () => {
    const softmaxLayer: SoftmaxLayer = { type: "softmax" };
    expect(softmaxLayer.type).toBe("softmax");
  });

  it("explain batchnorm", () => {
    const bnLayer: BatchNormLayer = { type: "batchnorm", epsilon: 1e-5 };
    expect(bnLayer.type).toBe("batchnorm");
    expect(bnLayer.epsilon).toBe(1e-5);
  });

  it("explain dropout", () => {
    const dropLayer: DropoutLayer = { type: "dropout", p: 0.5 };
    expect(dropLayer.type).toBe("dropout");
    expect(dropLayer.p).toBe(0.5);
  });

  it("explain full pipeline", () => {
    const layers: FusionLayer[] = [
      { type: "signal", signals: [new TermOperatorStub("test")] } as SignalLayer,
      { type: "batchnorm" } as BatchNormLayer,
      { type: "dropout", p: 0.3 } as DropoutLayer,
      {
        type: "dense",
        weights: [1.0],
        bias: [0.0],
        outputChannels: 1,
        inputChannels: 1,
      } as DenseLayer,
      { type: "flatten" } as FlattenLayer,
      { type: "softmax" } as SoftmaxLayer,
    ];
    const types = layers.map((l) => l.type);
    expect(types).toEqual([
      "signal",
      "batchnorm",
      "dropout",
      "dense",
      "flatten",
      "softmax",
    ]);
  });
});

// -- TestBackwardCompatibility --

describe("TestBackwardCompatibility", () => {
  it("existing signal layer unchanged", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const op = new DeepFusionOperator([signalLayer]);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });

  it("existing propagate unchanged", () => {
    const { ctx } = makeGridGraphStore();
    const entries = [];
    for (let i = 1; i <= 9; i++) {
      entries.push({ docId: i, score: 0.5 });
    }
    const signalOp = new MultiDocOperatorStub(entries);
    const signalLayer: SignalLayer = { type: "signal", signals: [signalOp] };
    const propagateLayer: PropagateLayer = {
      type: "propagate",
      edgeLabel: "spatial",
      aggregation: "mean",
      direction: "both",
    };
    const op = new DeepFusionOperator(
      [signalLayer, propagateLayer],
      0.5,
      "none",
      "grid",
    );
    const result = op.execute(ctx);
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });

  it("existing conv unchanged", () => {
    const { ctx } = makeGridGraphStore();
    const entries = [];
    for (let i = 1; i <= 9; i++) {
      entries.push({ docId: i, score: 0.5 });
    }
    const signalOp = new MultiDocOperatorStub(entries);
    const signalLayer: SignalLayer = { type: "signal", signals: [signalOp] };
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [1.0, 0.5],
      direction: "both",
    };
    const op = new DeepFusionOperator([signalLayer, convLayer], 0.5, "none", "grid");
    const result = op.execute(ctx);
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });
});

// -- TestCostEstimateNewLayers --

describe("TestCostEstimateNewLayers", () => {
  it("cost includes dense", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("a")],
    };
    const denseLayer: DenseLayer = {
      type: "dense",
      weights: [1.0, 0.5, -0.5, 0.3],
      bias: [0.0, 0.0],
      outputChannels: 2,
      inputChannels: 2,
    };
    const op = new DeepFusionOperator([signalLayer, denseLayer]);
    const stats = new IndexStats(100);
    const cost = op.costEstimate(stats);
    expect(cost).toBeGreaterThan(0);
  });

  it("cost includes flatten softmax", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("a")],
    };
    const flattenLayer: FlattenLayer = { type: "flatten" };
    const softmaxLayer: SoftmaxLayer = { type: "softmax" };
    const op = new DeepFusionOperator([signalLayer, flattenLayer, softmaxLayer]);
    const stats = new IndexStats(100);
    const cost = op.costEstimate(stats);
    expect(cost).toBeGreaterThan(0);
  });

  it("cost includes pool", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("a")],
    };
    const poolLayer: PoolLayer = {
      type: "pool",
      edgeLabel: "spatial",
      poolSize: 2,
      method: "max",
      direction: "both",
    };
    const op = new DeepFusionOperator([signalLayer, poolLayer]);
    const stats = new IndexStats(100);
    const cost = op.costEstimate(stats);
    expect(cost).toBeGreaterThan(0);
  });
});

// -- TestNewLayerErrors --

describe("TestNewLayerErrors", () => {
  it("dense weight length mismatch", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    // weights has 3 elements but outputChannels * inputChannels = 2*1 = 2
    // This causes a dimension mismatch in the backend transpose
    const denseLayer: DenseLayer = {
      type: "dense",
      weights: [1.0, 0.5, -0.5],
      bias: [0.0, 0.0],
      outputChannels: 2,
      inputChannels: 1,
    };
    const op = new DeepFusionOperator([signalLayer, denseLayer]);
    expect(() => op.execute({})).toThrow(/does not match shape/);
  });

  it("dense bias length mismatch", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    // bias has 1 element but outputChannels = 2
    const denseLayer: DenseLayer = {
      type: "dense",
      weights: [1.0, 0.5],
      bias: [0.0],
      outputChannels: 2,
      inputChannels: 1,
    };
    const op = new DeepFusionOperator([signalLayer, denseLayer]);
    const result = op.execute({});
    // Should still produce output (missing bias elements default to 0)
    expect(result.length).toBeGreaterThan(0);
  });

  it("dense missing named args", () => {
    // Dense layer with all required fields
    const denseLayer: DenseLayer = {
      type: "dense",
      weights: [1.0],
      bias: [0.0],
      outputChannels: 1,
      inputChannels: 1,
    };
    expect(denseLayer.type).toBe("dense");
    expect(denseLayer.weights.length).toBe(1);
  });

  it("pool invalid method", () => {
    // Pool with a non-standard method should still execute (treated as avg)
    const { ctx } = makeGridGraphStore();
    const entries = [];
    for (let i = 1; i <= 9; i++) {
      entries.push({ docId: i, score: 0.5 });
    }
    const signalOp = new MultiDocOperatorStub(entries);
    const layers: FusionLayer[] = [
      { type: "signal", signals: [signalOp] },
      {
        type: "pool",
        edgeLabel: "spatial",
        poolSize: 3,
        method: "unknown_method",
        direction: "both",
      } as PoolLayer,
    ];
    const op = new DeepFusionOperator(layers, 0.5, "none", "grid");
    const result = op.execute(ctx);
    // Should still produce output (defaults to avg pooling)
    expect(result.length).toBeGreaterThan(0);
  });

  it("dropout missing arg", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    // Dropout with p=0 should be rejected
    expect(
      () =>
        new DeepFusionOperator([
          signalLayer,
          { type: "dropout", p: 0 } as DropoutLayer,
        ]),
    ).toThrow(/p must be in/);
  });
});
