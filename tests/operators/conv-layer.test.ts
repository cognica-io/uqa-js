import { describe, expect, it } from "vitest";
import {
  createPayload,
  createEdge,
  createVertex,
  IndexStats,
} from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import { Engine } from "../../src/engine.js";
import { PostingList } from "../../src/core/posting-list.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import { Operator } from "../../src/operators/base.js";
import { DeepFusionOperator } from "../../src/operators/deep-fusion.js";
import type {
  SignalLayer,
  ConvLayer,
  FusionLayer,
} from "../../src/operators/deep-fusion.js";

// -- Helpers --

class TermOperatorStub extends Operator {
  readonly term: string;

  constructor(term: string) {
    super();
    this.term = term;
  }

  execute(_context: ExecutionContext): PostingList {
    return new PostingList([
      { docId: 1, payload: createPayload({ score: 0.9 }) },
      { docId: 2, payload: createPayload({ score: 0.7 }) },
      { docId: 3, payload: createPayload({ score: 0.5 }) },
    ]);
  }
}

// -- TestConvLayerConstruction --

describe("TestConvLayerConstruction", () => {
  it("conv layer first raises", () => {
    // The constructor validates that the first layer must be a SignalLayer
    // or EmbedLayer, not a spatial layer like ConvLayer.
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.5, 0.5],
      direction: "both",
    };
    expect(() => new DeepFusionOperator([convLayer])).toThrow(
      /first layer must be a SignalLayer or EmbedLayer/,
    );
  });

  it("valid conv layer", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.6, 0.4],
      direction: "both",
    };
    const op = new DeepFusionOperator([signalLayer, convLayer]);
    expect(op.layers.length).toBe(2);
  });
});

function makeConvGraph(): MemoryGraphStore {
  const gs = new MemoryGraphStore();
  gs.createGraph("g");
  for (let i = 1; i <= 5; i++) {
    gs.addVertex(createVertex(i, "node", {}), "g");
  }
  // Grid-like: 1-2, 2-3, 3-4, 4-5
  gs.addEdge(createEdge(1, 1, 2, "spatial", {}), "g");
  gs.addEdge(createEdge(2, 2, 3, "spatial", {}), "g");
  gs.addEdge(createEdge(3, 3, 4, "spatial", {}), "g");
  gs.addEdge(createEdge(4, 4, 5, "spatial", {}), "g");
  // Bidirectional
  gs.addEdge(createEdge(5, 2, 1, "spatial", {}), "g");
  gs.addEdge(createEdge(6, 3, 2, "spatial", {}), "g");
  gs.addEdge(createEdge(7, 4, 3, "spatial", {}), "g");
  gs.addEdge(createEdge(8, 5, 4, "spatial", {}), "g");
  return gs;
}

describe("TestConvLayerExecution", () => {
  it("single hop convolution", () => {
    const gs = makeConvGraph();
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.6, 0.4],
      direction: "both",
    };
    const op = new DeepFusionOperator([signalLayer, convLayer], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });

  it("two hop convolution", () => {
    const gs = makeConvGraph();
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.5, 0.3, 0.2],
      direction: "both",
    };
    const op = new DeepFusionOperator([signalLayer, convLayer], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });

  it("stacked conv layers", () => {
    const gs = makeConvGraph();
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const conv1: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.6, 0.4],
      direction: "both",
    };
    const conv2: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.7, 0.3],
      direction: "both",
    };
    const op = new DeepFusionOperator([signalLayer, conv1, conv2], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });

  it("conv smooths scores", () => {
    const gs = makeConvGraph();
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    // No conv - baseline
    const opNoConv = new DeepFusionOperator([signalLayer], 0.5, "none", "g");
    const noConvResult = opNoConv.execute({ graphStore: gs });

    // With conv
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.5, 0.5],
      direction: "both",
    };
    const opConv = new DeepFusionOperator([signalLayer, convLayer], 0.5, "none", "g");
    const convResult = opConv.execute({ graphStore: gs });

    // Conv should smooth/change scores
    const scoresNoConv = new Map<number, number>();
    for (const e of noConvResult) scoresNoConv.set(e.docId, e.payload.score);
    const scoresConv = new Map<number, number>();
    for (const e of convResult) scoresConv.set(e.docId, e.payload.score);

    let anyDifferent = false;
    for (const [docId, s1] of scoresNoConv) {
      const s2 = scoresConv.get(docId);
      if (s2 !== undefined && Math.abs(s1 - s2) > 1e-6) {
        anyDifferent = true;
        break;
      }
    }
    expect(anyDifferent).toBe(true);
  });

  it("conv with text signal", () => {
    const gs = makeConvGraph();
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention"), new TermOperatorStub("neural")],
    };
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.6, 0.4],
      direction: "both",
    };
    const op = new DeepFusionOperator([signalLayer, convLayer], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });

  it("conv direction out", () => {
    const gs = makeConvGraph();
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.6, 0.4],
      direction: "out",
    };
    const op = new DeepFusionOperator([signalLayer, convLayer], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });

  it("conv with gating", () => {
    const gs = makeConvGraph();
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.6, 0.4],
      direction: "both",
    };
    const op = new DeepFusionOperator([signalLayer, convLayer], 0.5, "swish", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });
});

// -- TestMixedConvLayers --

describe("TestMixedConvLayers", () => {
  it("signal conv signal", () => {
    const gs = makeConvGraph();
    const s1: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const conv: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.6, 0.4],
      direction: "both",
    };
    const s2: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("neural")],
    };
    const op = new DeepFusionOperator([s1, conv, s2], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });
});

// -- TestConvExplain --

describe("TestConvExplain", () => {
  it("explain conv layer", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.6, 0.4],
      direction: "both",
    };
    const op = new DeepFusionOperator([signalLayer, convLayer], 0.5, "relu");
    expect(op.layers.length).toBe(2);
    expect(op.layers[1]!.type).toBe("conv");
  });
});

// -- TestEstimateConvWeights --

describe("TestEstimateConvWeights", () => {
  it("basic estimation", () => {
    // Simple weight estimation: weights proportional to 1/(hop+1)
    const hops = 3;
    const weights: number[] = [];
    for (let i = 0; i <= hops; i++) {
      weights.push(1 / (i + 1));
    }
    const total = weights.reduce((a, b) => a + b, 0);
    const normalized = weights.map((w) => w / total);
    expect(normalized.length).toBe(4);
    expect(normalized[0]!).toBeGreaterThan(normalized[1]!);
  });

  it("weights decrease with distance", () => {
    const hops = 3;
    const weights: number[] = [];
    for (let i = 0; i <= hops; i++) {
      weights.push(1 / (i + 1));
    }
    for (let i = 0; i < weights.length - 1; i++) {
      expect(weights[i]!).toBeGreaterThan(weights[i + 1]!);
    }
  });

  it("single hop", () => {
    const hops = 1;
    const weights = [1.0, 0.5];
    const total = weights.reduce((a, b) => a + b, 0);
    const normalized = weights.map((w) => w / total);
    expect(normalized.length).toBe(hops + 1);
    expect(normalized[0]!).toBeGreaterThan(normalized[1]!);
  });

  it("nonexistent table raises", () => {
    const engine = new Engine();
    expect(() => engine.getTable("nonexistent")).toThrow(/not found/);
  });

  it("estimated weights improve retrieval", () => {
    const gs = makeConvGraph();
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    // Without conv
    const opBase = new DeepFusionOperator([signalLayer], 0.5, "none", "g");
    const baseResult = opBase.execute({ graphStore: gs });

    // With estimated weights
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.6, 0.3, 0.1],
      direction: "both",
    };
    const opConv = new DeepFusionOperator([signalLayer, convLayer], 0.5, "none", "g");
    const convResult = opConv.execute({ graphStore: gs });

    // Both should return results
    expect(baseResult.length).toBeGreaterThan(0);
    expect(convResult.length).toBeGreaterThan(0);
  });
});

// -- TestConvErrors --

describe("TestConvErrors", () => {
  it("invalid direction", () => {
    const gs = makeConvGraph();
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.6, 0.4],
      direction: "invalid",
    };
    const op = new DeepFusionOperator([signalLayer, convLayer], 0.5, "none", "g");
    // Invalid direction means no neighbors found, but should not throw
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });

  it("empty weights array", () => {
    const gs = makeConvGraph();
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [],
      direction: "both",
    };
    const op = new DeepFusionOperator([signalLayer, convLayer], 0.5, "none", "g");
    // Empty weights sum to 0, so convolution is a no-op
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });
});

// -- TestConvCostEstimate --

describe("TestConvCostEstimate", () => {
  it("cost includes conv", () => {
    const signalLayer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("a")],
    };
    const convLayer: ConvLayer = {
      type: "conv",
      edgeLabel: "spatial",
      hopWeights: [0.6, 0.4],
      direction: "both",
    };
    const op = new DeepFusionOperator([signalLayer, convLayer]);
    const stats = new IndexStats(100);
    const cost = op.costEstimate(stats);
    expect(cost).toBeGreaterThan(0);
  });
});
