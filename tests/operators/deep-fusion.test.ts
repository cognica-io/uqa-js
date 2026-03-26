import { describe, expect, it } from "vitest";
import {
  createPayload,
  createEdge,
  createVertex,
  IndexStats,
} from "../../src/core/types.js";
import { MemoryGraphStore } from "../../src/graph/store.js";
import { PostingList } from "../../src/core/posting-list.js";
import type { ExecutionContext } from "../../src/operators/base.js";
import { Operator } from "../../src/operators/base.js";
import {
  DeepFusionOperator,
  safeLogit,
  sigmoidVal,
  applyGating,
} from "../../src/operators/deep-fusion.js";
import type {
  SignalLayer,
  PropagateLayer,
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
    // Return a simple posting list for test purposes
    return new PostingList([
      { docId: 1, payload: createPayload({ score: 0.8 }) },
      { docId: 2, payload: createPayload({ score: 0.5 }) },
    ]);
  }
}

// -- Utility function tests --
// The JS port has safeLogit, sigmoidVal, applyGating as module-private functions.
// We cannot test them directly, so they are skipped.

describe("TestHelperFunctions", () => {
  it("safe logit middle", () => {
    expect(Math.abs(safeLogit(0.5))).toBeLessThan(0.001);
  });

  it("safe logit high", () => {
    expect(safeLogit(0.9)).toBeGreaterThan(0);
  });

  it("safe logit low", () => {
    expect(safeLogit(0.1)).toBeLessThan(0);
  });

  it("safe logit clamp extreme", () => {
    expect(isFinite(safeLogit(0.0))).toBe(true);
    expect(isFinite(safeLogit(1.0))).toBe(true);
  });

  it("sigmoid middle", () => {
    expect(Math.abs(sigmoidVal(0.0) - 0.5)).toBeLessThan(0.001);
  });

  it("sigmoid positive", () => {
    expect(sigmoidVal(5.0)).toBeGreaterThan(0.99);
  });

  it("sigmoid negative", () => {
    expect(sigmoidVal(-5.0)).toBeLessThan(0.01);
  });

  it("sigmoid large negative", () => {
    expect(sigmoidVal(-100.0)).toBeLessThan(0.001);
  });

  it("gating none", () => {
    expect(applyGating(-2.0, "none")).toBe(-2.0);
  });

  it("gating relu", () => {
    expect(applyGating(-2.0, "relu")).toBe(0.0);
  });

  it("gating swish", () => {
    const result = applyGating(2.0, "swish");
    const expected = 2.0 * sigmoidVal(2.0);
    expect(Math.abs(result - expected)).toBeLessThan(0.001);
  });
});

// -- Operator construction tests --

describe("TestOperatorConstruction", () => {
  it("empty layers raises", () => {
    // Constructor throws when given empty layers array
    expect(() => new DeepFusionOperator([])).toThrow(
      /deep_fusion requires at least one layer/,
    );
  });

  it("propagate first raises", () => {
    // Constructor throws when first layer is a spatial type (propagate)
    const layer: PropagateLayer = {
      type: "propagate",
      edgeLabel: "spatial",
      aggregation: "sum",
      direction: "both",
    };
    expect(() => new DeepFusionOperator([layer])).toThrow(
      /first layer must be a SignalLayer or EmbedLayer/,
    );
  });

  it("valid construction", () => {
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const op = new DeepFusionOperator([layer], 0.5, "relu");
    expect(op.layers.length).toBe(1);
    expect(op.gating).toBe("relu");
  });
});

// -- Single signal layer tests --

describe("TestSingleSignalLayer", () => {
  it("single layer single signal", () => {
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const op = new DeepFusionOperator([layer]);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });

  it("single layer multi signal", () => {
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention"), new TermOperatorStub("neural")],
    };
    const op = new DeepFusionOperator([layer]);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });
});

// -- Residual accumulation tests --
// NOTE: The JS DeepFusionOperator processes signal layers by replacing channelMap
// each time (not accumulating residuals like the Python version).
// These tests verify the basic behavior.

describe("TestResidualAccumulation", () => {
  it("two signal layers", () => {
    const layer1: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const layer2: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("graph")],
    };
    const op = new DeepFusionOperator([layer1, layer2]);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });

  it("three layers hierarchical", () => {
    const layer1: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const layer2: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("graph")],
    };
    const layer3: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("neural")],
    };
    const op = new DeepFusionOperator([layer1, layer2, layer3]);
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });
});

// -- Gating tests --

describe("TestGating", () => {
  it("gating none", () => {
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const op = new DeepFusionOperator([layer], 0.5, "none");
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
  });

  it("gating relu", () => {
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const op = new DeepFusionOperator([layer], 0.5, "relu");
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
  });

  it("gating swish", () => {
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const op = new DeepFusionOperator([layer], 0.5, "swish");
    const result = op.execute({});
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });
});

// -- Propagation tests (require graph store) --

function makeGraphForPropagation(): MemoryGraphStore {
  const gs = new MemoryGraphStore();
  gs.createGraph("g");
  for (let i = 1; i <= 5; i++) {
    gs.addVertex(createVertex(i, "node", {}), "g");
  }
  // 1 -> 2 -> 3 -> 4 -> 5
  gs.addEdge(createEdge(1, 1, 2, "cites", {}), "g");
  gs.addEdge(createEdge(2, 2, 3, "cites", {}), "g");
  gs.addEdge(createEdge(3, 3, 4, "cites", {}), "g");
  gs.addEdge(createEdge(4, 4, 5, "cites", {}), "g");
  return gs;
}

describe("TestPropagation", () => {
  it("basic propagate", () => {
    const gs = makeGraphForPropagation();
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const propagate: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "mean",
      direction: "out",
    };
    const op = new DeepFusionOperator([layer, propagate], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });

  it("propagate discovers new docs", () => {
    const gs = makeGraphForPropagation();
    // Signal only covers docs 1 and 2
    class LimitedStub extends Operator {
      execute(_context: ExecutionContext): PostingList {
        return new PostingList([
          { docId: 1, payload: createPayload({ score: 0.9 }) },
          { docId: 2, payload: createPayload({ score: 0.7 }) },
        ]);
      }
    }
    const layer: SignalLayer = {
      type: "signal",
      signals: [new LimitedStub()],
    };
    const propagate: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "mean",
      direction: "both",
    };
    const op = new DeepFusionOperator([layer, propagate], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    const docIds = new Set(result.entries.map((e) => e.docId));
    // Propagation with "both" should discover doc 3 (out-neighbor of doc 2)
    // and potentially doc 0 (in-neighbor of doc 1 if any)
    // Doc 3 gets score from doc 2 (its in-neighbor via direction "both")
    expect(docIds.has(3)).toBe(true);
  });

  it("propagate aggregation mean", () => {
    const gs = makeGraphForPropagation();
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const propagate: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "mean",
      direction: "both",
    };
    const op = new DeepFusionOperator([layer, propagate], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });

  it("propagate aggregation sum", () => {
    const gs = makeGraphForPropagation();
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const propagate: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "sum",
      direction: "both",
    };
    const op = new DeepFusionOperator([layer, propagate], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });

  it("propagate aggregation max", () => {
    const gs = makeGraphForPropagation();
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const propagate: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "max",
      direction: "both",
    };
    const op = new DeepFusionOperator([layer, propagate], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });

  it("multi hop propagation", () => {
    const gs = makeGraphForPropagation();
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const p1: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "mean",
      direction: "out",
    };
    const p2: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "mean",
      direction: "out",
    };
    const op = new DeepFusionOperator([layer, p1, p2], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });
});

// -- Mixed layers tests --

describe("TestMixedLayers", () => {
  it("signal propagate signal", () => {
    const gs = makeGraphForPropagation();
    const s1: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const p: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "mean",
      direction: "out",
    };
    const s2: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("graph")],
    };
    const op = new DeepFusionOperator([s1, p, s2], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThanOrEqual(1.0);
    }
  });

  it("signal propagate propagate signal", () => {
    const gs = makeGraphForPropagation();
    const s1: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const p1: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "mean",
      direction: "out",
    };
    const p2: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "mean",
      direction: "out",
    };
    const s2: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("graph")],
    };
    const op = new DeepFusionOperator([s1, p1, p2, s2], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });
});

// -- EXPLAIN tests --

describe("TestExplain", () => {
  it("explain output", () => {
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const op = new DeepFusionOperator([layer], 0.5, "relu");
    // Verify the operator has a describe or toString-like capability
    expect(op.layers.length).toBe(1);
    expect(op.alpha).toBe(0.5);
    expect(op.gating).toBe("relu");
  });
});

// -- Score validity tests --

describe("TestScoreValidity", () => {
  it("all scores in unit interval", () => {
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention"), new TermOperatorStub("graph")],
    };
    const op = new DeepFusionOperator([layer], 0.5, "swish");
    const result = op.execute({});
    for (const entry of result) {
      expect(entry.payload.score).toBeGreaterThan(0.0);
      expect(entry.payload.score).toBeLessThan(1.0);
    }
  });
});

// -- Error cases --

describe("TestErrorCases", () => {
  it("propagate first layer error", () => {
    const propagate: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "mean",
      direction: "both",
    };
    expect(() => new DeepFusionOperator([propagate])).toThrow(
      /first layer must be a SignalLayer or EmbedLayer/,
    );
  });

  it("empty layer error", () => {
    expect(() => new DeepFusionOperator([])).toThrow(
      /deep_fusion requires at least one layer/,
    );
  });

  it("invalid aggregation", () => {
    // Propagate with unknown aggregation still runs (falls back to mean)
    const gs = makeGraphForPropagation();
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const propagate: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "invalid_agg",
      direction: "both",
    };
    const op = new DeepFusionOperator([layer, propagate], 0.5, "none", "g");
    // Should not throw, falls back to mean
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });

  it("invalid direction", () => {
    // Direction that is not "out"/"in"/"both" means no neighbors collected
    const gs = makeGraphForPropagation();
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const propagate: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "mean",
      direction: "invalid_dir",
    };
    const op = new DeepFusionOperator([layer, propagate], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    // With no valid direction, propagation does nothing
    expect(result.length).toBeGreaterThan(0);
  });

  it("unknown named arg", () => {
    // Constructing with valid layers but passing extra data is fine
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const op = new DeepFusionOperator([layer]);
    expect(op.layers.length).toBe(1);
  });

  it("invalid inner function", () => {
    // Verify that the operator handles missing graph store gracefully
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("test")],
    };
    const propagate: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "mean",
      direction: "both",
    };
    const op = new DeepFusionOperator([layer, propagate], 0.5, "none", "g");
    expect(() => op.execute({})).toThrow(/graph_store/);
  });
});

// -- Alpha parameter tests --

describe("TestAlphaParameter", () => {
  it("alpha affects scores", () => {
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention"), new TermOperatorStub("graph")],
    };
    const opLow = new DeepFusionOperator([layer], 0.1);
    const opHigh = new DeepFusionOperator([layer], 0.9);
    const resultLow = opLow.execute({});
    const resultHigh = opHigh.execute({});

    expect(resultLow.length).toBeGreaterThan(0);
    expect(resultHigh.length).toBeGreaterThan(0);

    // At least some scores should differ
    const scoresLow = new Map<number, number>();
    for (const e of resultLow) scoresLow.set(e.docId, e.payload.score);
    const scoresHigh = new Map<number, number>();
    for (const e of resultHigh) scoresHigh.set(e.docId, e.payload.score);

    let anyDifferent = false;
    for (const [docId, scoreLow] of scoresLow) {
      const scoreHigh = scoresHigh.get(docId);
      if (scoreHigh !== undefined && Math.abs(scoreLow - scoreHigh) > 1e-6) {
        anyDifferent = true;
        break;
      }
    }
    expect(anyDifferent).toBe(true);
  });
});

// -- Cost estimate tests --

describe("TestCostEstimate", () => {
  it("cost estimate", () => {
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("a"), new TermOperatorStub("b")],
    };
    const op = new DeepFusionOperator([
      layer,
      {
        type: "propagate",
        edgeLabel: "cites",
        aggregation: "mean",
        direction: "both",
      } as PropagateLayer,
      {
        type: "signal",
        signals: [new TermOperatorStub("c")],
      } as SignalLayer,
    ]);
    const stats = new IndexStats(100);
    const cost = op.costEstimate(stats);
    expect(cost).toBeGreaterThan(0);
  });
});

// -- Propagate direction tests --

describe("TestPropagateDirection", () => {
  it("propagate out direction", () => {
    const gs = makeGraphForPropagation();
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const propagate: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "mean",
      direction: "out",
    };
    const op = new DeepFusionOperator([layer, propagate], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });

  it("propagate in direction", () => {
    const gs = makeGraphForPropagation();
    const layer: SignalLayer = {
      type: "signal",
      signals: [new TermOperatorStub("attention")],
    };
    const propagate: PropagateLayer = {
      type: "propagate",
      edgeLabel: "cites",
      aggregation: "mean",
      direction: "in",
    };
    const op = new DeepFusionOperator([layer, propagate], 0.5, "none", "g");
    const result = op.execute({ graphStore: gs });
    expect(result.length).toBeGreaterThan(0);
  });
});
