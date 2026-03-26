//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- deep fusion operator
// 1:1 port of uqa/operators/deep_fusion.py
// Multi-layer fusion operator (Paper 4, Section 7)

import { logOddsConjunction } from "bayesian-bm25";
import type { DocId, IndexStats } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { ExecutionContext } from "./base.js";
import { Operator } from "./base.js";
import { coverageBasedDefault } from "./hybrid.js";
import {
  applyGating as backendApplyGating,
  batchDense,
  batchSoftmax,
  batchBatchnorm,
  batchSelfAttention,
  gridForward as backendGridForward,
  gridGlobalPool as backendGridGlobalPool,
  hopWeightsToKernel,
  sigmoidStable,
  sigmoidVec,
} from "./backend.js";

// -- Constants ---------------------------------------------------------------

const PROB_FLOOR = 1e-15;
const PROB_CEIL = 1.0 - 1e-15;

// -- Layer definitions -------------------------------------------------------

export interface SignalLayer {
  readonly type: "signal";
  readonly signals: Operator[];
}

export interface PropagateLayer {
  readonly type: "propagate";
  readonly edgeLabel: string;
  readonly aggregation: string;
  readonly direction: string;
}

export interface ConvLayer {
  readonly type: "conv";
  readonly edgeLabel: string;
  readonly hopWeights: readonly number[];
  readonly direction: string;
  readonly kernel?: readonly number[] | null;
  readonly kernelShape?: readonly number[] | null;
}

export interface PoolLayer {
  readonly type: "pool";
  readonly edgeLabel: string;
  readonly poolSize: number;
  readonly method: string;
  readonly direction: string;
}

export interface DenseLayer {
  readonly type: "dense";
  readonly weights: readonly number[];
  readonly bias: readonly number[];
  readonly outputChannels: number;
  readonly inputChannels: number;
}

export interface FlattenLayer {
  readonly type: "flatten";
}

export interface SoftmaxLayer {
  readonly type: "softmax";
}

export interface BatchNormLayer {
  readonly type: "batchnorm";
  readonly epsilon?: number;
}

export interface DropoutLayer {
  readonly type: "dropout";
  readonly p: number;
}

export interface GlobalPoolLayer {
  readonly type: "global_pool";
  readonly method: string;
}

export interface AttentionLayer {
  readonly type: "attention";
  readonly nHeads: number;
  readonly mode: string;
  readonly qWeights?: readonly number[] | null;
  readonly qShape?: readonly number[] | null;
  readonly kWeights?: readonly number[] | null;
  readonly kShape?: readonly number[] | null;
  readonly vWeights?: readonly number[] | null;
  readonly vShape?: readonly number[] | null;
}

export interface EmbedLayer {
  readonly type: "embed";
  readonly embedding?: readonly number[];
  readonly vectorField?: string;
  readonly gridH: number;
  readonly gridW: number;
  readonly inChannels?: number;
}

export type FusionLayer =
  | SignalLayer
  | PropagateLayer
  | ConvLayer
  | PoolLayer
  | DenseLayer
  | FlattenLayer
  | SoftmaxLayer
  | BatchNormLayer
  | DropoutLayer
  | GlobalPoolLayer
  | AttentionLayer
  | EmbedLayer;

// -- Utility -----------------------------------------------------------------

export function safeLogit(p: number): number {
  const c = Math.max(PROB_FLOOR, Math.min(PROB_CEIL, p));
  return Math.log(c / (1 - c));
}

export function sigmoidVal(x: number): number {
  return sigmoidStable(x);
}

export function applyGating(logitVal: number, gating: string): number {
  return backendApplyGating(logitVal, gating);
}

const SPATIAL_TYPES = new Set(["propagate", "conv", "pool"]);

// -- Graph neighbor helper ---------------------------------------------------

function graphNeighbors(
  gs: unknown,
  vid: number,
  edgeLabel: string,
  direction: string,
  graphName: string,
): number[] {
  const store = gs as {
    neighbors: (
      vid: number,
      graph: string,
      label?: string | null,
      dir?: "out" | "in",
    ) => number[];
  };
  const result: number[] = [];
  if (direction === "out" || direction === "both") {
    result.push(...store.neighbors(vid, graphName, edgeLabel || null, "out"));
  }
  if (direction === "in" || direction === "both") {
    result.push(...store.neighbors(vid, graphName, edgeLabel || null, "in"));
  }
  return result;
}

// -- DeepFusionOperator ------------------------------------------------------

export class DeepFusionOperator extends Operator {
  readonly layers: FusionLayer[];
  readonly alpha: number;
  readonly gating: string;
  readonly graphName: string;
  readonly embedMode: boolean;
  private readonly _gridShape: [number, number] | null;

  constructor(layers: FusionLayer[], alpha = 0.5, gating = "none", graphName = "") {
    super();
    if (layers.length === 0) {
      throw new Error("deep_fusion requires at least one layer");
    }
    if (SPATIAL_TYPES.has(layers[0]!.type)) {
      throw new Error(
        "deep_fusion: first layer must be a SignalLayer or EmbedLayer " +
          "(no scores to propagate or convolve)",
      );
    }

    // Validate layer ordering and parameters
    let flattened = false;
    for (const layer of layers) {
      if (SPATIAL_TYPES.has(layer.type) && flattened) {
        throw new Error(
          "deep_fusion: spatial layers (propagate, convolve, pool) " +
            "must not appear after flatten() or global_pool()",
        );
      }
      if (layer.type === "flatten" || layer.type === "global_pool") {
        flattened = true;
      }
      if (layer.type === "pool" && layer.poolSize < 2) {
        throw new Error("deep_fusion: pool() pool_size must be >= 2");
      }
      if (layer.type === "dropout") {
        const p = layer.p;
        if (p <= 0 || p >= 1) {
          throw new Error("deep_fusion: dropout() p must be in (0, 1)");
        }
      }
    }

    this.layers = layers;
    this.alpha = alpha;
    this.gating = gating;
    this.graphName = graphName;
    this.embedMode = layers[0]!.type === "embed";

    // Grid acceleration
    if (this.embedMode) {
      const el = layers[0] as EmbedLayer;
      if (el.gridH > 0 && el.gridW > 0) {
        this._gridShape = [el.gridH, el.gridW];
      } else {
        this._gridShape = null;
      }
    } else {
      this._gridShape = null;
    }
  }

  execute(context: ExecutionContext): PostingList {
    let channelMap = new Map<DocId, Float64Array>();
    let numChannels = 1;
    let softmaxApplied = false;

    // Grid acceleration: batch conv+pool sequences via backend
    if (this._gridShape !== null) {
      const result = this._executeGrid(
        channelMap,
        numChannels,
        softmaxApplied,
        context,
      );
      channelMap = result.channelMap;
      numChannels = result.numChannels;
      softmaxApplied = result.softmaxApplied;
    } else {
      for (const layer of this.layers) {
        switch (layer.type) {
          case "embed":
            DeepFusionOperator._executeEmbedLayer(layer, channelMap);
            break;
          case "signal":
            this._executeSignalLayer(layer, context, channelMap, numChannels);
            break;
          case "propagate":
            this._executePropagateLayer(layer, context, channelMap, numChannels);
            break;
          case "conv":
            this._executeConvLayer(layer, context, channelMap);
            break;
          case "pool":
            this._executePoolLayer(layer, context, channelMap);
            break;
          case "dense":
            this._executeDenseLayer(layer, channelMap);
            numChannels = layer.outputChannels;
            break;
          case "flatten": {
            const r = DeepFusionOperator._executeFlattenLayer(channelMap);
            channelMap = r.channelMap;
            numChannels = r.numChannels;
            break;
          }
          case "global_pool": {
            const r = DeepFusionOperator._executeGlobalPoolLayer(layer, channelMap);
            channelMap = r.channelMap;
            numChannels = r.numChannels;
            break;
          }
          case "softmax":
            DeepFusionOperator._executeSoftmaxLayer(channelMap);
            softmaxApplied = true;
            break;
          case "batchnorm":
            DeepFusionOperator._executeBatchNormLayer(layer, channelMap);
            break;
          case "dropout":
            DeepFusionOperator._executeDropoutLayer(layer, channelMap);
            break;
          case "attention":
            this._executeAttentionLayer(layer, channelMap);
            break;
        }
      }
    }

    return DeepFusionOperator._buildResult(channelMap, numChannels, softmaxApplied);
  }

  // -- Result builder --

  private static _buildResult(
    channelMap: Map<DocId, Float64Array>,
    numChannels: number,
    softmaxApplied: boolean,
  ): PostingList {
    if (channelMap.size === 0) return new PostingList();

    const sortedIds = [...channelMap.keys()].sort((a, b) => a - b);
    const entries = sortedIds.map((docId) => {
      const vec = channelMap.get(docId)!;
      if (softmaxApplied) {
        let maxProb = -Infinity;
        for (let i = 0; i < vec.length; i++) {
          if (vec[i]! > maxProb) maxProb = vec[i]!;
        }
        const classProbs = Array.from(vec);
        return {
          docId,
          payload: createPayload({
            score: maxProb,
            fields: { class_probs: classProbs },
          }),
        };
      } else if (numChannels === 1) {
        const score = sigmoidVal(vec[0] ?? 0);
        return { docId, payload: createPayload({ score }) };
      } else {
        const sigVec = sigmoidVec(vec);
        let maxSig = -Infinity;
        for (let i = 0; i < sigVec.length; i++) {
          if (sigVec[i]! > maxSig) maxSig = sigVec[i]!;
        }
        return { docId, payload: createPayload({ score: maxSig }) };
      }
    });
    return PostingList.fromSorted(entries);
  }

  // -- Signal layer --

  private _executeSignalLayer(
    layer: SignalLayer,
    context: ExecutionContext,
    channelMap: Map<DocId, Float64Array>,
    numChannels: number,
  ): void {
    const signals = layer.signals;
    const postingLists = signals.map((s) => s.execute(context));

    const allDocIds = new Set<DocId>();
    const scoreMaps: Map<DocId, number>[] = [];
    for (const pl of postingLists) {
      const smap = new Map<DocId, number>();
      for (const entry of pl) {
        smap.set(entry.docId, entry.payload.score);
        allDocIds.add(entry.docId);
      }
      scoreMaps.push(smap);
    }

    if (allDocIds.size === 0) return;

    const numDocs = allDocIds.size;
    const defaults = scoreMaps.map((m) => coverageBasedDefault(m.size, numDocs));
    const alpha = this.alpha;
    const gating = this.gating;

    if (signals.length === 1) {
      const smap = scoreMaps[0]!;
      const def = defaults[0]!;
      for (const docId of allDocIds) {
        const p = smap.get(docId) ?? def;
        let layerLogit = safeLogit(p);
        layerLogit = applyGating(layerLogit, gating);
        if (!channelMap.has(docId)) {
          channelMap.set(docId, new Float64Array(numChannels));
        }
        channelMap.get(docId)![0]! += layerLogit;
      }
    } else {
      for (const docId of allDocIds) {
        const probs: number[] = [];
        for (let j = 0; j < scoreMaps.length; j++) {
          probs.push(scoreMaps[j]!.get(docId) ?? defaults[j]!);
        }
        const fusedP = logOddsConjunction(probs, alpha, undefined, "none");
        let layerLogit = safeLogit(fusedP);
        layerLogit = applyGating(layerLogit, gating);
        if (!channelMap.has(docId)) {
          channelMap.set(docId, new Float64Array(numChannels));
        }
        channelMap.get(docId)![0]! += layerLogit;
      }
    }
  }

  // -- Propagate layer --

  private _executePropagateLayer(
    layer: PropagateLayer,
    context: ExecutionContext,
    channelMap: Map<DocId, Float64Array>,
    numChannels: number,
  ): void {
    const gs = context.graphStore;
    if (gs == null) {
      throw new Error(
        "deep_fusion propagate layer requires a graph_store in ExecutionContext",
      );
    }

    // Convert channel 0 to probabilities
    const probMap = new Map<DocId, number>();
    for (const [docId, vec] of channelMap) {
      probMap.set(docId, sigmoidVal(vec[0]!));
    }

    const newMap = new Map<DocId, Float64Array>();
    const { direction, edgeLabel, aggregation } = layer;
    const gName = this.graphName;
    const gating = this.gating;

    // Collect all vertices that could be affected
    const allVertexIds = new Set(channelMap.keys());
    for (const docId of [...allVertexIds]) {
      const nbs = graphNeighbors(gs, docId, edgeLabel, direction, gName);
      for (const nb of nbs) allVertexIds.add(nb);
    }

    for (const vid of allVertexIds) {
      const neighborProbs: number[] = [];
      const nbs = graphNeighbors(gs, vid, edgeLabel, direction, gName);
      for (const nb of nbs) {
        const p = probMap.get(nb);
        if (p !== undefined) neighborProbs.push(p);
      }

      if (neighborProbs.length === 0) {
        if (channelMap.has(vid)) {
          newMap.set(vid, new Float64Array(channelMap.get(vid)!));
        }
        continue;
      }

      let aggProb: number;
      if (aggregation === "mean") {
        aggProb = neighborProbs.reduce((a, b) => a + b, 0) / neighborProbs.length;
      } else if (aggregation === "sum") {
        aggProb = Math.min(
          PROB_CEIL,
          neighborProbs.reduce((a, b) => a + b, 0),
        );
      } else if (aggregation === "max") {
        aggProb = Math.max(...neighborProbs);
      } else {
        aggProb = neighborProbs.reduce((a, b) => a + b, 0) / neighborProbs.length;
      }

      let propagatedLogit = safeLogit(aggProb);
      propagatedLogit = applyGating(propagatedLogit, gating);

      const existing = channelMap.get(vid);
      let newVec: Float64Array;
      if (existing !== undefined) {
        newVec = new Float64Array(existing);
        newVec[0] = existing[0]! + propagatedLogit;
      } else {
        newVec = new Float64Array(numChannels);
        newVec[0] = propagatedLogit;
      }
      newMap.set(vid, newVec);
    }

    channelMap.clear();
    for (const [k, v] of newMap) channelMap.set(k, v);
  }

  // -- Conv layer --

  private _executeConvLayer(
    layer: ConvLayer,
    context: ExecutionContext,
    channelMap: Map<DocId, Float64Array>,
  ): void {
    const gs = context.graphStore;
    if (gs == null) {
      throw new Error(
        "deep_fusion convolve layer requires a graph_store in ExecutionContext",
      );
    }

    const embedMode = this.embedMode;

    // Build value map
    const valMap = new Map<DocId, number>();
    for (const [docId, vec] of channelMap) {
      valMap.set(docId, embedMode ? vec[0]! : sigmoidVal(vec[0]!));
    }

    // Normalize hop weights
    const totalW = layer.hopWeights.reduce((a, b) => a + b, 0);
    if (totalW <= 0) return;
    const normWeights = layer.hopWeights.map((w) => w / totalW);

    const newMap = new Map<DocId, Float64Array>();
    const { edgeLabel, direction } = layer;
    const gName = this.graphName;
    const gating = this.gating;
    const kernelHops = layer.hopWeights.length - 1;

    for (const vid of [...channelMap.keys()]) {
      let weightedVal = 0.0;

      // Hop 0: self
      const selfVal = valMap.get(vid);
      if (selfVal !== undefined) {
        weightedVal += normWeights[0]! * selfVal;
      }

      // Hop 1..kernelHops: BFS rings
      let currentFrontier = new Set([vid]);
      const visited = new Set([vid]);
      for (let h = 1; h <= kernelHops; h++) {
        const nextFrontier = new Set<number>();
        for (const fv of currentFrontier) {
          for (const nb of graphNeighbors(gs, fv, edgeLabel, direction, gName)) {
            if (!visited.has(nb)) {
              nextFrontier.add(nb);
              visited.add(nb);
            }
          }
        }

        if (nextFrontier.size > 0) {
          const hopVals: number[] = [];
          for (const nb of nextFrontier) {
            const v = valMap.get(nb);
            if (v !== undefined) hopVals.push(v);
          }
          if (hopVals.length > 0) {
            const hopMean = hopVals.reduce((a, b) => a + b, 0) / hopVals.length;
            weightedVal += normWeights[h]! * hopMean;
          }
        }

        currentFrontier = nextFrontier;
      }

      const newVec = new Float64Array(channelMap.get(vid)!);
      if (embedMode) {
        newVec[0] = applyGating(weightedVal, gating);
      } else {
        let convLogit = safeLogit(
          Math.max(PROB_FLOOR, Math.min(PROB_CEIL, weightedVal)),
        );
        convLogit = applyGating(convLogit, gating);
        newVec[0] = channelMap.get(vid)![0]! + convLogit;
      }
      newMap.set(vid, newVec);
    }

    channelMap.clear();
    for (const [k, v] of newMap) channelMap.set(k, v);
  }

  // -- Pool layer --

  private _executePoolLayer(
    layer: PoolLayer,
    context: ExecutionContext,
    channelMap: Map<DocId, Float64Array>,
  ): void {
    const gs = context.graphStore;
    if (gs == null) {
      throw new Error(
        "deep_fusion pool layer requires a graph_store in ExecutionContext",
      );
    }

    const { edgeLabel, direction, poolSize, method } = layer;
    const gName = this.graphName;

    const remaining = new Set(channelMap.keys());
    const pooled = new Map<DocId, Float64Array>();

    while (remaining.size > 0) {
      const seed = Math.min(...remaining);
      remaining.delete(seed);

      // BFS to collect poolSize - 1 more neighbors
      const group = [seed];
      let frontier = new Set([seed]);
      const visitedBfs = new Set([seed]);

      while (group.length < poolSize && frontier.size > 0) {
        const nextFrontier = new Set<number>();
        for (const fv of frontier) {
          for (const nb of graphNeighbors(gs, fv, edgeLabel, direction, gName)) {
            if (!visitedBfs.has(nb)) {
              visitedBfs.add(nb);
              nextFrontier.add(nb);
              if (remaining.has(nb)) {
                group.push(nb);
                remaining.delete(nb);
                if (group.length >= poolSize) break;
              }
            }
          }
          if (group.length >= poolSize) break;
        }
        frontier = nextFrontier;
      }

      // Aggregate channel vectors element-wise
      const nCh = channelMap.get(group[0]!)!.length;
      const agg = new Float64Array(nCh);
      if (method === "max") {
        agg.fill(-Infinity);
        for (const g of group) {
          const vec = channelMap.get(g)!;
          for (let c = 0; c < nCh; c++) {
            agg[c] = Math.max(agg[c]!, vec[c]!);
          }
        }
      } else {
        for (const g of group) {
          const vec = channelMap.get(g)!;
          for (let c = 0; c < nCh; c++) {
            agg[c]! += vec[c]! / group.length;
          }
        }
      }

      const rep = Math.min(...group);
      pooled.set(rep, agg);
    }

    channelMap.clear();
    for (const [k, v] of pooled) channelMap.set(k, v);
  }

  // -- Dense layer --

  private _executeDenseLayer(
    layer: DenseLayer,
    channelMap: Map<DocId, Float64Array>,
  ): void {
    const docIds = [...channelMap.keys()].sort((a, b) => a - b);
    if (docIds.length === 0) return;
    const nCh = channelMap.get(docIds[0]!)!.length;
    const n = docIds.length;

    // Stack into matrix
    const X = new Float64Array(n * nCh);
    for (let i = 0; i < n; i++) {
      X.set(channelMap.get(docIds[i]!)!, i * nCh);
    }

    const W = new Float64Array(layer.weights);
    const bias = new Float64Array(layer.bias);
    const result = batchDense(
      X,
      [n, nCh],
      W,
      [layer.outputChannels, layer.inputChannels],
      bias,
      this.gating,
    );

    for (let i = 0; i < n; i++) {
      channelMap.set(
        docIds[i]!,
        result.data.slice(i * layer.outputChannels, (i + 1) * layer.outputChannels),
      );
    }
  }

  // -- Flatten layer --

  private static _executeFlattenLayer(channelMap: Map<DocId, Float64Array>): {
    channelMap: Map<DocId, Float64Array>;
    numChannels: number;
  } {
    if (channelMap.size === 0) return { channelMap: new Map(), numChannels: 0 };

    const sortedIds = [...channelMap.keys()].sort((a, b) => a - b);
    const parts: Float64Array[] = [];
    for (const did of sortedIds) parts.push(channelMap.get(did)!);

    let totalLen = 0;
    for (const p of parts) totalLen += p.length;
    const flatVec = new Float64Array(totalLen);
    let offset = 0;
    for (const p of parts) {
      flatVec.set(p, offset);
      offset += p.length;
    }

    const repId = sortedIds[0]!;
    const newMap = new Map<DocId, Float64Array>();
    newMap.set(repId, flatVec);
    return { channelMap: newMap, numChannels: totalLen };
  }

  // -- Global pool layer --

  private static _executeGlobalPoolLayer(
    layer: GlobalPoolLayer,
    channelMap: Map<DocId, Float64Array>,
  ): { channelMap: Map<DocId, Float64Array>; numChannels: number } {
    if (channelMap.size === 0) return { channelMap: new Map(), numChannels: 0 };

    const sortedIds = [...channelMap.keys()].sort((a, b) => a - b);
    const nCh = channelMap.get(sortedIds[0]!)!.length;
    const n = sortedIds.length;

    if (layer.method === "avg") {
      const pooled = new Float64Array(nCh);
      for (const did of sortedIds) {
        const vec = channelMap.get(did)!;
        for (let c = 0; c < nCh; c++) pooled[c]! += vec[c]! / n;
      }
      const newMap = new Map<DocId, Float64Array>();
      newMap.set(sortedIds[0]!, pooled);
      return { channelMap: newMap, numChannels: nCh };
    }

    if (layer.method === "max") {
      const pooled = new Float64Array(nCh).fill(-Infinity);
      for (const did of sortedIds) {
        const vec = channelMap.get(did)!;
        for (let c = 0; c < nCh; c++) pooled[c] = Math.max(pooled[c]!, vec[c]!);
      }
      const newMap = new Map<DocId, Float64Array>();
      newMap.set(sortedIds[0]!, pooled);
      return { channelMap: newMap, numChannels: nCh };
    }

    // avg_max
    const avgPooled = new Float64Array(nCh);
    const maxPooled = new Float64Array(nCh).fill(-Infinity);
    for (const did of sortedIds) {
      const vec = channelMap.get(did)!;
      for (let c = 0; c < nCh; c++) {
        avgPooled[c]! += vec[c]! / n;
        maxPooled[c] = Math.max(maxPooled[c]!, vec[c]!);
      }
    }
    const combined = new Float64Array(nCh * 2);
    combined.set(avgPooled, 0);
    combined.set(maxPooled, nCh);
    const newMap = new Map<DocId, Float64Array>();
    newMap.set(sortedIds[0]!, combined);
    return { channelMap: newMap, numChannels: nCh * 2 };
  }

  // -- Softmax layer --

  private static _executeSoftmaxLayer(channelMap: Map<DocId, Float64Array>): void {
    const docIds = [...channelMap.keys()].sort((a, b) => a - b);
    if (docIds.length === 0) return;
    const nCh = channelMap.get(docIds[0]!)!.length;
    const n = docIds.length;

    const X = new Float64Array(n * nCh);
    for (let i = 0; i < n; i++) {
      X.set(channelMap.get(docIds[i]!)!, i * nCh);
    }
    const out = batchSoftmax(X, [n, nCh]);
    for (let i = 0; i < n; i++) {
      channelMap.set(docIds[i]!, out.slice(i * nCh, (i + 1) * nCh));
    }
  }

  // -- Batchnorm layer --

  private static _executeBatchNormLayer(
    layer: BatchNormLayer,
    channelMap: Map<DocId, Float64Array>,
  ): void {
    if (channelMap.size < 2) return;

    const docIds = [...channelMap.keys()].sort((a, b) => a - b);
    const nCh = channelMap.get(docIds[0]!)!.length;
    const n = docIds.length;
    const eps = layer.epsilon ?? 1e-5;

    const X = new Float64Array(n * nCh);
    for (let i = 0; i < n; i++) {
      X.set(channelMap.get(docIds[i]!)!, i * nCh);
    }
    const out = batchBatchnorm(X, [n, nCh], eps);
    for (let i = 0; i < n; i++) {
      channelMap.set(docIds[i]!, out.slice(i * nCh, (i + 1) * nCh));
    }
  }

  // -- Dropout layer --

  private static _executeDropoutLayer(
    layer: DropoutLayer,
    channelMap: Map<DocId, Float64Array>,
  ): void {
    const scale = 1.0 - layer.p;
    for (const [docId, vec] of channelMap) {
      const scaled = new Float64Array(vec.length);
      for (let i = 0; i < vec.length; i++) scaled[i] = vec[i]! * scale;
      channelMap.set(docId, scaled);
    }
  }

  // -- Attention layer --

  private _executeAttentionLayer(
    layer: AttentionLayer,
    channelMap: Map<DocId, Float64Array>,
  ): void {
    if (channelMap.size === 0) return;

    const docIds = [...channelMap.keys()].sort((a, b) => a - b);
    const nCh = channelMap.get(docIds[0]!)!.length;
    const seqLen = docIds.length;

    // Stack into (1, seqLen, nCh) for batch_self_attention
    const X = new Float64Array(seqLen * nCh);
    for (let i = 0; i < seqLen; i++) {
      X.set(channelMap.get(docIds[i]!)!, i * nCh);
    }

    const wQ = layer.qWeights && layer.qShape ? new Float64Array(layer.qWeights) : null;
    const wK = layer.kWeights && layer.kShape ? new Float64Array(layer.kWeights) : null;
    const wV = layer.vWeights && layer.vShape ? new Float64Array(layer.vWeights) : null;

    const out = batchSelfAttention(
      X,
      [1, seqLen, nCh],
      layer.nHeads,
      wQ,
      wK,
      wV,
      this.gating,
    );

    for (let i = 0; i < seqLen; i++) {
      channelMap.set(docIds[i]!, out.slice(i * nCh, (i + 1) * nCh));
    }
  }

  // -- Embed layer --

  private static _executeEmbedLayer(
    layer: EmbedLayer,
    channelMap: Map<DocId, Float64Array>,
  ): void {
    if (layer.embedding) {
      for (let i = 0; i < layer.embedding.length; i++) {
        channelMap.set(i + 1, new Float64Array([layer.embedding[i]!]));
      }
    }
  }

  // -- Grid-accelerated execution --

  private _executeGrid(
    channelMap: Map<DocId, Float64Array>,
    numChannels: number,
    softmaxApplied: boolean,
    _context: ExecutionContext,
  ): {
    channelMap: Map<DocId, Float64Array>;
    numChannels: number;
    softmaxApplied: boolean;
  } {
    const [gridH, gridW] = this._gridShape!;

    type SegType = "conv_pool" | "attention" | "global_pool";
    type ConvPoolStage = {
      kernel: Float64Array;
      kernelShape: number[];
      poolSize: number;
      poolMethod: string;
    };
    const segments: [SegType, unknown][] = [];
    let currentConvPool: ConvPoolStage[] = [];
    const remainingLayers: FusionLayer[] = [];
    const nonEmbed = this.layers.filter((la) => la.type !== "embed");

    let i = 0;
    while (i < nonEmbed.length) {
      const la = nonEmbed[i]!;
      if (la.type === "conv") {
        const conv = la;
        let poolSize = 2;
        let poolMethod = "max";
        if (i + 1 < nonEmbed.length && nonEmbed[i + 1]!.type === "pool") {
          const pl = nonEmbed[i + 1] as PoolLayer;
          poolSize = pl.poolSize;
          poolMethod = pl.method;
          i += 2;
        } else {
          i += 1;
        }
        let k: Float64Array;
        let kShape: number[];
        if (conv.kernel != null && conv.kernelShape != null) {
          k = new Float64Array(conv.kernel);
          kShape = [...conv.kernelShape];
        } else {
          k = hopWeightsToKernel([...conv.hopWeights]);
          kShape = [1, 1, 3, 3];
        }
        currentConvPool.push({ kernel: k, kernelShape: kShape, poolSize, poolMethod });
      } else if (la.type === "attention") {
        if (currentConvPool.length > 0) {
          segments.push(["conv_pool", currentConvPool]);
          currentConvPool = [];
        }
        segments.push(["attention", la]);
        i += 1;
      } else if (la.type === "global_pool") {
        if (currentConvPool.length > 0) {
          segments.push(["conv_pool", currentConvPool]);
          currentConvPool = [];
        }
        segments.push(["global_pool", la]);
        i += 1;
      } else {
        if (currentConvPool.length > 0) {
          segments.push(["conv_pool", currentConvPool]);
          currentConvPool = [];
        }
        remainingLayers.push(la);
        i += 1;
      }
    }
    if (currentConvPool.length > 0) {
      segments.push(["conv_pool", currentConvPool]);
    }

    // Process segments sequentially
    const embedLayer = this.layers[0] as EmbedLayer;
    const embedding = embedLayer.embedding ?? [];
    let currentFlat: Float64Array = new Float64Array(embedding);
    let curH = gridH;
    let curW = gridW;
    const batch = 1;

    for (const [segType, segData] of segments) {
      if (segType === "conv_pool") {
        const stagesList = segData as ConvPoolStage[];
        const result = backendGridForward(
          currentFlat,
          [batch, currentFlat.length],
          curH,
          curW,
          stagesList,
          this.gating,
        );
        currentFlat = result.data;
        for (const stage of stagesList) {
          curH = Math.floor(curH / stage.poolSize);
          curW = Math.floor(curW / stage.poolSize);
        }
      } else if (segType === "attention") {
        const attnLayer = segData as AttentionLayer;
        const nCh = Math.floor(currentFlat.length / (curH * curW));
        const seqLen = curH * curW;

        // Reshape (1, C*H*W) -> (1, H*W, C)
        const X3d = new Float64Array(seqLen * nCh);
        for (let s = 0; s < seqLen; s++) {
          for (let c = 0; c < nCh; c++) {
            X3d[s * nCh + c] = currentFlat[c * seqLen + s] ?? 0;
          }
        }

        const wQ =
          attnLayer.qWeights && attnLayer.qShape
            ? new Float64Array(attnLayer.qWeights)
            : null;
        const wK =
          attnLayer.kWeights && attnLayer.kShape
            ? new Float64Array(attnLayer.kWeights)
            : null;
        const wV =
          attnLayer.vWeights && attnLayer.vShape
            ? new Float64Array(attnLayer.vWeights)
            : null;

        const out3d = batchSelfAttention(
          X3d,
          [1, seqLen, nCh],
          attnLayer.nHeads,
          wQ,
          wK,
          wV,
          this.gating,
        );

        // Reshape back to (1, C*H*W)
        currentFlat = new Float64Array(nCh * seqLen);
        for (let s = 0; s < seqLen; s++) {
          for (let c = 0; c < nCh; c++) {
            currentFlat[c * seqLen + s] = out3d[s * nCh + c]!;
          }
        }
      } else {
        const gpLayer = segData as GlobalPoolLayer;
        const result = backendGridGlobalPool(
          currentFlat,
          [batch, currentFlat.length],
          curH,
          curW,
          gpLayer.method,
        );
        currentFlat = result.data;
        curH = 1;
        curW = 1;
      }
    }

    // Rebuild channel_map from final features
    const newChannelMap = new Map<DocId, Float64Array>();
    const hasGlobalPool = segments.some(([st]) => st === "global_pool");
    if (hasGlobalPool) {
      newChannelMap.set(1, currentFlat);
      numChannels = currentFlat.length;
    } else {
      for (let idx = 0; idx < currentFlat.length; idx++) {
        newChannelMap.set(idx + 1, new Float64Array([currentFlat[idx]!]));
      }
      numChannels = 1;
    }

    // Process remaining layers
    let cm = newChannelMap;
    let nc = numChannels;
    let sa = softmaxApplied;
    for (const layer of remainingLayers) {
      switch (layer.type) {
        case "flatten": {
          const r = DeepFusionOperator._executeFlattenLayer(cm);
          cm = r.channelMap;
          nc = r.numChannels;
          break;
        }
        case "global_pool": {
          const r = DeepFusionOperator._executeGlobalPoolLayer(layer, cm);
          cm = r.channelMap;
          nc = r.numChannels;
          break;
        }
        case "dense":
          this._executeDenseLayer(layer, cm);
          nc = layer.outputChannels;
          break;
        case "softmax":
          DeepFusionOperator._executeSoftmaxLayer(cm);
          sa = true;
          break;
        case "batchnorm":
          DeepFusionOperator._executeBatchNormLayer(layer, cm);
          break;
        case "dropout":
          DeepFusionOperator._executeDropoutLayer(layer, cm);
          break;
      }
    }

    return { channelMap: cm, numChannels: nc, softmaxApplied: sa };
  }

  // -- Cost estimation --

  costEstimate(stats: IndexStats): number {
    let total = 0;
    for (const layer of this.layers) {
      switch (layer.type) {
        case "signal":
          for (const s of layer.signals) {
            total += s.costEstimate(stats);
          }
          break;
        case "embed":
          total += (layer.embedding ?? []).length;
          break;
        case "propagate":
        case "conv":
        case "pool":
          total += stats.totalDocs;
          break;
        case "dense":
          total += layer.inputChannels * layer.outputChannels;
          break;
        case "flatten":
        case "global_pool":
        case "softmax":
        case "batchnorm":
        case "dropout":
          total += stats.totalDocs;
          break;
        case "attention":
          total += stats.totalDocs ** 2;
          break;
      }
    }
    return total;
  }
}

// -- estimate_conv_weights --------------------------------------------------

export function estimateConvWeights(
  graphStore: unknown,
  docStore: unknown,
  edgeLabel: string,
  kernelHops: number,
  graphName: string,
  embeddingField = "embedding",
): number[] {
  const store = docStore as {
    docIds: number[];
    getField: (docId: number, field: string) => number[] | Float64Array | null;
  };
  const gs = graphStore;

  // Collect all doc_ids and their embeddings
  const embeddings = new Map<number, Float64Array>();
  for (const docId of store.docIds) {
    const vec = store.getField(docId, embeddingField);
    if (vec != null) {
      const arr = vec instanceof Float64Array ? vec : new Float64Array(vec);
      let norm = 0;
      for (let i = 0; i < arr.length; i++) norm += arr[i]! * arr[i]!;
      norm = Math.sqrt(norm);
      if (norm > 0) {
        const normalized = new Float64Array(arr.length);
        for (let i = 0; i < arr.length; i++) normalized[i] = arr[i]! / norm;
        embeddings.set(docId, normalized);
      }
    }
  }

  if (embeddings.size < 2) {
    const w = 1.0 / (kernelHops + 1);
    return new Array<number>(kernelHops + 1).fill(w);
  }

  // For each hop distance, compute average cosine similarity
  const hopSimilarities: number[][] = [];
  for (let h = 0; h <= kernelHops; h++) hopSimilarities.push([]);

  for (const [vid, vecV] of embeddings) {
    hopSimilarities[0]!.push(1.0);

    let currentFrontier = new Set([vid]);
    const visited = new Set([vid]);
    for (let h = 1; h <= kernelHops; h++) {
      const nextFrontier = new Set<number>();
      for (const fv of currentFrontier) {
        for (const nb of graphNeighbors(gs, fv, edgeLabel, "both", graphName)) {
          if (!visited.has(nb)) {
            nextFrontier.add(nb);
            visited.add(nb);
          }
        }
      }
      for (const nb of nextFrontier) {
        const vecNb = embeddings.get(nb);
        if (vecNb !== undefined) {
          let dot = 0;
          for (let i = 0; i < vecV.length; i++) dot += vecV[i]! * vecNb[i]!;
          hopSimilarities[h]!.push(dot);
        }
      }
      currentFrontier = nextFrontier;
    }
  }

  // Compute mean similarity per hop
  const rawWeights: number[] = [];
  for (let h = 0; h <= kernelHops; h++) {
    const sims = hopSimilarities[h]!;
    if (sims.length > 0) {
      const meanSim = sims.reduce((a, b) => a + b, 0) / sims.length;
      rawWeights.push(Math.max(0.0, meanSim));
    } else {
      rawWeights.push(0.0);
    }
  }

  // Normalize to sum to 1
  const total = rawWeights.reduce((a, b) => a + b, 0);
  if (total <= 0) {
    const w = 1.0 / (kernelHops + 1);
    return new Array<number>(kernelHops + 1).fill(w);
  }

  return rawWeights.map((w) => w / total);
}
