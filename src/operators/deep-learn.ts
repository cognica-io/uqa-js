//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- deep learning operator
// 1:1 port of uqa/operators/deep_learn.py
//
// Analytical training pipeline for deep learning models (Paper 4).
// PoE local learning with supervised conv weight estimation.

import type { FusionLayer } from "./deep-fusion.js";
import type * as linalg from "../math/linalg.js";
import {
  ridgeSolve,
  elasticNetSolve,
  magnitudePrune,
  hopWeightsToKernel,
  gridForward,
  gridGlobalPool,
  batchSelfAttention,
  generateOrthogonalKernels,
  generateGaborKernels,
  generateKmeansKernels,
  generateQkProjections,
  searchVProjection,
} from "./backend.js";

// -- Training spec types -----------------------------------------------------

export interface ConvSpec {
  readonly type: "conv";
  readonly kernelHops: number;
  readonly nChannels: number;
  readonly initMode: string;
}

export interface PoolSpec {
  readonly type: "pool";
  readonly method: string;
  readonly poolSize: number;
}

export interface FlattenSpec {
  readonly type: "flatten";
}

export interface GlobalPoolSpec {
  readonly type: "global_pool";
  readonly method: string;
}

export interface DenseSpec {
  readonly type: "dense";
  readonly outputChannels: number;
}

export interface SoftmaxSpec {
  readonly type: "softmax";
}

export interface AttentionSpec {
  readonly type: "attention";
  readonly nHeads: number;
  readonly mode: string;
}

export type LayerSpec =
  | ConvSpec
  | PoolSpec
  | FlattenSpec
  | GlobalPoolSpec
  | DenseSpec
  | SoftmaxSpec
  | AttentionSpec;

// -- TrainedModel ------------------------------------------------------------

export interface TrainedModel {
  modelName: string;
  tableName: string | null;
  labelField: string;
  embeddingField: string;
  edgeLabel: string;
  gating: string;
  lam: number;
  layerSpecs: Record<string, unknown>[];
  convWeights: number[][];
  denseWeights: number[];
  denseBias: number[];
  denseInputChannels: number;
  denseOutputChannels: number;
  numClasses: number;
  classLabels: unknown[];
  gridSize: number;
  embeddingDim: number;
  trainingAccuracy: number;
  trainingSamples: number;
  expertWeights: number[][];
  expertBiases: number[][];
  expertInputChannels: number[];
  expertAccuracies: number[];
  shrinkageAlpha: number;
  convKernelData: number[][];
  convKernelShapes: number[][];
  inChannels: number;
  attentionParams: Record<string, unknown>[];
  l1Ratio: number;
  pruneRatio: number;
  weightSparsity: number;
}

export function trainedModelToJSON(model: TrainedModel): string {
  return JSON.stringify(model);
}

export function trainedModelFromJSON(s: string): TrainedModel {
  return JSON.parse(s) as TrainedModel;
}

export function trainedModelToDict(model: TrainedModel): Record<string, unknown> {
  return { ...model } as unknown as Record<string, unknown>;
}

export function trainedModelFromDict(d: Record<string, unknown>): TrainedModel {
  return d as unknown as TrainedModel;
}

export function trainedModelToDeepFusionLayers(model: TrainedModel): FusionLayer[] {
  const layers: FusionLayer[] = [];
  let convIdx = 0;
  let attnIdx = 0;

  for (const specDict of model.layerSpecs) {
    const t = specDict["type"] as string;
    if (t === "conv") {
      if (
        model.convKernelData.length > convIdx &&
        model.convKernelShapes.length > convIdx
      ) {
        layers.push({
          type: "conv",
          edgeLabel: model.edgeLabel,
          hopWeights: model.convWeights[convIdx] ?? [1.0, 0.0],
          direction: "both",
          kernel: model.convKernelData[convIdx],
          kernelShape: model.convKernelShapes[convIdx],
        });
      } else {
        layers.push({
          type: "conv",
          edgeLabel: model.edgeLabel,
          hopWeights: model.convWeights[convIdx] ?? [1.0, 0.0],
          direction: "both",
        });
      }
      convIdx++;
    } else if (t === "attention") {
      const ap =
        attnIdx < model.attentionParams.length ? model.attentionParams[attnIdx]! : {};
      layers.push({
        type: "attention",
        nHeads: (ap["n_heads"] as number | undefined) ?? 1,
        mode: (ap["mode"] as string | undefined) ?? "content",
        qWeights: ap["W_q"] ? (ap["W_q"] as number[]) : null,
        qShape: ap["W_q_shape"] ? (ap["W_q_shape"] as number[]) : null,
        kWeights: ap["W_k"] ? (ap["W_k"] as number[]) : null,
        kShape: ap["W_k_shape"] ? (ap["W_k_shape"] as number[]) : null,
        vWeights: ap["W_v"] ? (ap["W_v"] as number[]) : null,
        vShape: ap["W_v_shape"] ? (ap["W_v_shape"] as number[]) : null,
      });
      attnIdx++;
    } else if (t === "pool") {
      layers.push({
        type: "pool",
        edgeLabel: model.edgeLabel,
        poolSize: (specDict["pool_size"] as number | undefined) ?? 2,
        method: (specDict["method"] as string | undefined) ?? "max",
        direction: "both",
      });
    } else if (t === "flatten") {
      layers.push({ type: "flatten" });
    } else if (t === "global_pool") {
      layers.push({
        type: "global_pool",
        method: (specDict["method"] as string | undefined) ?? "avg",
      });
    } else if (t === "dense") {
      layers.push({
        type: "dense",
        weights: model.denseWeights,
        bias: model.denseBias,
        outputChannels: model.denseOutputChannels,
        inputChannels: model.denseInputChannels,
      });
    } else if (t === "softmax") {
      layers.push({ type: "softmax" });
    }
  }

  return layers;
}

// -- Kernel generation -------------------------------------------------------

export function generateKernels(
  nChannels: number,
  inChannels: number,
  seed = 42,
  initMode = "kaiming",
  trainingData: Float64Array | null = null,
  shapeData: linalg.Shape2D | null = null,
  gridH = 0,
  gridW = 0,
): Float64Array {
  if (initMode === "orthogonal") {
    return generateOrthogonalKernels(nChannels, inChannels, seed);
  }
  if (initMode === "gabor") {
    return generateGaborKernels(nChannels, inChannels, seed);
  }
  if (initMode === "kmeans") {
    if (trainingData === null || shapeData === null) {
      throw new Error("kmeans init requires training_data");
    }
    return generateKmeansKernels(
      nChannels,
      inChannels,
      trainingData,
      shapeData,
      gridH,
      gridW,
      seed,
    );
  }
  // Default: Kaiming
  let s = seed;
  function nextRand(): number {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    const u1 = (s >>> 0) / 0xffffffff;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    const u2 = (s >>> 0) / 0xffffffff;
    return Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
  }

  const fanIn = inChannels * 9;
  const std = Math.sqrt(2.0 / fanIn);
  const kernels = new Float64Array(nChannels * inChannels * 9);
  for (let i = 0; i < kernels.length; i++) {
    kernels[i] = nextRand() * std;
  }
  return kernels;
}

// -- Stage identification ---------------------------------------------------

interface SpecDict {
  type: string;
  [key: string]: unknown;
}

type Operation =
  | ["stage", SpecDict, SpecDict]
  | ["attention", SpecDict]
  | ["global_pool", SpecDict];

function identifyOperations(specDicts: SpecDict[]): Operation[] {
  const ops: Operation[] = [];
  let i = 0;
  while (i < specDicts.length) {
    const d = specDicts[i]!;
    if (d.type === "conv") {
      const convD = d;
      let poolD: SpecDict;
      if (i + 1 < specDicts.length && specDicts[i + 1]!.type === "pool") {
        poolD = specDicts[i + 1]!;
        i += 2;
      } else {
        poolD = { type: "pool", method: "max", pool_size: 2 };
        i += 1;
      }
      ops.push(["stage", convD, poolD]);
    } else if (d.type === "attention") {
      ops.push(["attention", d]);
      i += 1;
    } else if (d.type === "global_pool") {
      ops.push(["global_pool", d]);
      i += 1;
    } else {
      i += 1;
    }
  }
  return ops;
}

// -- Self-attention training -----------------------------------------------

function trainAttention(
  xFlat: Float64Array,
  shapeX: linalg.Shape2D,
  gridH: number,
  gridW: number,
  nChannels: number,
  nHeads: number,
  mode: string,
  Y: Float64Array,
  shapeY: linalg.Shape2D,
  lam: number,
  gating = "none",
  seed = 42,
): { outFlat: Float64Array; params: Record<string, unknown> } {
  const batch = shapeX[0];
  const seqLen = gridH * gridW;
  const dModel = nChannels;

  // Reshape (batch, C*H*W) -> (batch, H*W, C) = (batch, seqLen, dModel)
  const X3d = new Float64Array(batch * seqLen * dModel);
  for (let b = 0; b < batch; b++) {
    for (let s = 0; s < seqLen; s++) {
      for (let c = 0; c < dModel; c++) {
        X3d[b * seqLen * dModel + s * dModel + c] =
          xFlat[b * shapeX[1] + c * seqLen + s] ?? 0;
      }
    }
  }

  const params: Record<string, unknown> = { mode, n_heads: nHeads, d_model: dModel };
  let wQ: Float64Array | null = null;
  let wK: Float64Array | null = null;

  if (mode === "random_qk" || mode === "learned_v") {
    const proj = generateQkProjections(dModel, seed);
    wQ = proj.wQ;
    wK = proj.wK;
    params["W_q"] = Array.from(wQ);
    params["W_q_shape"] = [dModel, dModel];
    params["W_k"] = Array.from(wK);
    params["W_k_shape"] = [dModel, dModel];
  }

  if (mode === "learned_v") {
    const { bestWV, bestOutFlat } = searchVProjection(
      X3d,
      [batch, seqLen, dModel],
      nHeads,
      wQ,
      wK,
      Y,
      shapeY,
      lam,
      gating,
      20,
      seed,
    );
    params["W_v"] = Array.from(bestWV);
    params["W_v_shape"] = [dModel, dModel];
    return { outFlat: bestOutFlat, params };
  }

  const out3d = batchSelfAttention(
    X3d,
    [batch, seqLen, dModel],
    nHeads,
    wQ,
    wK,
    null,
    gating,
  );

  // (batch, H*W, C) -> (batch, C*H*W)
  const outFlat = new Float64Array(batch * dModel * seqLen);
  for (let b = 0; b < batch; b++) {
    for (let s = 0; s < seqLen; s++) {
      for (let c = 0; c < dModel; c++) {
        outFlat[b * dModel * seqLen + c * seqLen + s] =
          out3d[b * seqLen * dModel + s * dModel + c]!;
      }
    }
  }
  return { outFlat, params };
}

// -- Training --------------------------------------------------------------

export interface Engine {
  _tables: Map<string, unknown>;
  saveModel(name: string, data: Record<string, unknown>): void;
  loadModel(name: string): Record<string, unknown> | null;
}

export function trainModel(
  engine: Engine,
  modelName: string,
  tableName: string | null,
  labelField: string,
  embeddingField: string,
  edgeLabel: string,
  layerSpecs: LayerSpec[],
  gating = "none",
  lam = 1.0,
  l1Ratio = 0.0,
  pruneRatio = 0.0,
  rows: Record<string, unknown>[] | null = null,
): Record<string, unknown> {
  if (layerSpecs.length === 0) {
    throw new Error("deep_learn requires at least one layer spec");
  }
  const specDicts = specsToDicts(layerSpecs);

  // Collect data
  const labelsRaw: unknown[] = [];
  const embList: Float64Array[] = [];

  if (rows !== null) {
    for (const row of rows) {
      labelsRaw.push(row[labelField]);
      const emb = row[embeddingField];
      if (Array.isArray(emb)) {
        embList.push(new Float64Array(emb as number[]));
      } else if (emb instanceof Float64Array) {
        embList.push(emb);
      }
    }
  } else {
    throw new Error("In-browser training requires rows parameter");
  }

  if (embList.length === 0) throw new Error("No training data");

  const embeddingDim = embList[0]!.length;
  const nSamples = embList.length;

  // Stack embeddings
  const embeddings = new Float64Array(nSamples * embeddingDim);
  for (let i = 0; i < nSamples; i++) {
    embeddings.set(embList[i]!, i * embeddingDim);
  }

  // Labels
  const uniqueLabels = [...new Set(labelsRaw)].sort();
  const labelToIdx = new Map<unknown, number>();
  for (let i = 0; i < uniqueLabels.length; i++) {
    labelToIdx.set(uniqueLabels[i], i);
  }
  const numClasses = uniqueLabels.length;

  let denseOutput = numClasses;
  for (const spec of layerSpecs) {
    if (spec.type === "dense") {
      denseOutput = spec.outputChannels;
      break;
    }
  }

  // One-hot labels
  const Y = new Float64Array(nSamples * denseOutput);
  for (let i = 0; i < nSamples; i++) {
    const idx = labelToIdx.get(labelsRaw[i]) ?? 0;
    if (idx < denseOutput) Y[i * denseOutput + idx] = 1.0;
  }

  // Grid detection
  let inChannels = 0;
  let gridSize = 0;
  for (const ch of [1, 3, 4]) {
    if (embeddingDim % ch !== 0) continue;
    const side = Math.floor(Math.sqrt(embeddingDim / ch));
    if (side * side * ch === embeddingDim) {
      inChannels = ch;
      gridSize = side;
      break;
    }
  }
  if (gridSize === 0) {
    throw new Error(
      `Embedding dimension ${String(embeddingDim)} is not C*H*W for any supported channel count.`,
    );
  }

  const ops = identifyOperations(specDicts);
  const stagesOnly = ops
    .filter((op) => op[0] === "stage")
    .map((op) => [op[1], op[2]] as [SpecDict, SpecDict]);

  // Generate conv kernels per stage
  const convKernels: Float64Array[] = [];
  const convWeights: number[][] = [];
  let inCh = inChannels;
  for (let stageIdx = 0; stageIdx < stagesOnly.length; stageIdx++) {
    const [convD] = stagesOnly[stageIdx]!;
    const nCh = (convD["n_channels"] as number | undefined) ?? 1;
    const initMode = (convD["init_mode"] as string | undefined) ?? "kaiming";
    if (nCh > 1) {
      const kernels = generateKernels(
        nCh,
        inCh,
        42 + stageIdx,
        initMode,
        initMode === "kmeans" ? embeddings : null,
        initMode === "kmeans" ? [nSamples, embeddingDim] : null,
        gridSize,
        gridSize,
      );
      convKernels.push(kernels);
      convWeights.push([]);
    } else {
      const kernels = hopWeightsToKernel([1.0, 0.0]);
      convKernels.push(kernels);
      convWeights.push([1.0, 0.0]);
    }
    inCh = nCh > 1 ? nCh : 1;
  }

  // Per-operation forward + ridge regression
  let currentFlat: Float64Array = embeddings;
  let curH = gridSize;
  let curW = gridSize;

  const expertWeightsList: number[][] = [];
  const expertBiasesList: number[][] = [];
  const expertInputChannels: number[] = [];
  const attentionParamsList: Record<string, unknown>[] = [];
  const expertAccuracies: number[] = [];
  const trueClasses = new Int32Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    trueClasses[i] = labelToIdx.get(labelsRaw[i]) ?? 0;
  }

  let stageIdx = 0;
  let attnIdx = 0;
  for (const op of ops) {
    if (op[0] === "stage") {
      const convD = op[1];
      const poolD = op[2];
      const poolSize = (poolD["pool_size"] as number | undefined) ?? 2;
      const poolMethod = (poolD["method"] as string | undefined) ?? "max";
      const nCh = (convD["n_channels"] as number | undefined) ?? 1;

      if (nCh <= 1) {
        // Single-channel: supervised grid search
        let bestAcc = -1.0;
        let bestHw = [1.0, 0.0];
        for (let a = -1.0; a <= 1.05; a += 0.1) {
          const cand = [1.0, a];
          const k = hopWeightsToKernel(cand);
          const result = gridForward(
            currentFlat,
            [nSamples, currentFlat.length / nSamples],
            curH,
            curW,
            [{ kernel: k, kernelShape: [1, 1, 3, 3], poolSize, poolMethod }],
            gating,
          );
          const feats = result.data;
          const nFeats = result.shape[1];
          if (nFeats === 0) continue;
          const { weights: Wt, bias: bt } = ridgeSolve(
            feats,
            [nSamples, nFeats],
            Y,
            [nSamples, denseOutput],
            lam,
          );
          // Compute accuracy
          let correct = 0;
          for (let i = 0; i < nSamples; i++) {
            let bestClass = 0;
            let bestScore = -Infinity;
            for (let j = 0; j < denseOutput; j++) {
              let score = bt[j]!;
              for (let f = 0; f < nFeats; f++) {
                score += feats[i * nFeats + f]! * Wt[f * denseOutput + j]!;
              }
              if (score > bestScore) {
                bestScore = score;
                bestClass = j;
              }
            }
            if (bestClass === trueClasses[i]) correct++;
          }
          const acc = correct / nSamples;
          if (acc > bestAcc) {
            bestAcc = acc;
            bestHw = [...cand];
          }
        }
        convKernels[stageIdx] = hopWeightsToKernel(bestHw);
        convWeights[stageIdx] = bestHw;
      }

      // Forward through this stage
      const result = gridForward(
        currentFlat,
        [nSamples, currentFlat.length / nSamples],
        curH,
        curW,
        [
          {
            kernel: convKernels[stageIdx]!,
            kernelShape: [1, 1, 3, 3],
            poolSize,
            poolMethod,
          },
        ],
        gating,
      );
      currentFlat = result.data;
      curH = Math.floor(curH / poolSize);
      curW = Math.floor(curW / poolSize);
      stageIdx++;
    } else if (op[0] === "attention") {
      const attnD = op[1];
      const nHeads = (attnD["n_heads"] as number | undefined) ?? 1;
      const mode = (attnD["mode"] as string | undefined) ?? "content";
      const nChAttn = Math.floor(currentFlat.length / nSamples / (curH * curW));

      const { outFlat, params } = trainAttention(
        currentFlat,
        [nSamples, currentFlat.length / nSamples],
        curH,
        curW,
        nChAttn,
        nHeads,
        mode,
        Y,
        [nSamples, denseOutput],
        lam,
        gating,
        42 + attnIdx,
      );
      currentFlat = outFlat;
      attentionParamsList.push(params);
      attnIdx++;
    } else {
      const gpMethod = (op[1]["method"] as string | undefined) ?? "avg";
      const result = gridGlobalPool(
        currentFlat,
        [nSamples, currentFlat.length / nSamples],
        curH,
        curW,
        gpMethod,
      );
      currentFlat = result.data;
      curH = 1;
      curW = 1;
    }

    // Train expert head
    const nFeats = Math.floor(currentFlat.length / nSamples);
    let Ws: Float64Array;
    let bs: Float64Array;
    if (l1Ratio > 0) {
      const r = elasticNetSolve(
        currentFlat,
        [nSamples, nFeats],
        Y,
        [nSamples, denseOutput],
        lam,
        l1Ratio,
      );
      Ws = r.weights;
      bs = r.bias;
    } else {
      const r = ridgeSolve(
        currentFlat,
        [nSamples, nFeats],
        Y,
        [nSamples, denseOutput],
        lam,
      );
      Ws = r.weights;
      bs = r.bias;
    }
    if (pruneRatio > 0) {
      Ws = magnitudePrune(Ws, pruneRatio);
    }
    expertWeightsList.push(Array.from(Ws));
    expertBiasesList.push(Array.from(bs));
    expertInputChannels.push(nFeats);

    // Per-stage accuracy
    let correct = 0;
    for (let i = 0; i < nSamples; i++) {
      let bestClass = 0;
      let bestScore = -Infinity;
      for (let j = 0; j < denseOutput; j++) {
        let score = bs[j]!;
        for (let f = 0; f < nFeats; f++) {
          score += currentFlat[i * nFeats + f]! * Ws[f * denseOutput + j]!;
        }
        if (score > bestScore) {
          bestScore = score;
          bestClass = j;
        }
      }
      if (bestClass === trueClasses[i]) correct++;
    }
    expertAccuracies.push(correct / nSamples);
  }

  // Final head
  const nFeatures = Math.floor(currentFlat.length / nSamples);
  let WFinal: Float64Array;
  let bFinal: Float64Array;
  if (l1Ratio > 0) {
    const r = elasticNetSolve(
      currentFlat,
      [nSamples, nFeatures],
      Y,
      [nSamples, denseOutput],
      lam,
      l1Ratio,
    );
    WFinal = r.weights;
    bFinal = r.bias;
  } else {
    const r = ridgeSolve(
      currentFlat,
      [nSamples, nFeatures],
      Y,
      [nSamples, denseOutput],
      lam,
    );
    WFinal = r.weights;
    bFinal = r.bias;
  }
  if (pruneRatio > 0) {
    WFinal = magnitudePrune(WFinal, pruneRatio);
  }

  // Final accuracy
  let finalCorrect = 0;
  for (let i = 0; i < nSamples; i++) {
    let bestClass = 0;
    let bestScore = -Infinity;
    for (let j = 0; j < denseOutput; j++) {
      let score = bFinal[j]!;
      for (let f = 0; f < nFeatures; f++) {
        score += currentFlat[i * nFeatures + f]! * WFinal[f * denseOutput + j]!;
      }
      if (score > bestScore) {
        bestScore = score;
        bestClass = j;
      }
    }
    if (bestClass === trueClasses[i]) finalCorrect++;
  }
  expertAccuracies.push(finalCorrect / nSamples);

  // PoE shrinkage
  const nOps = ops.length;
  const nExpertStages = nOps + 1;
  const shrinkageAlpha = 1.0 / (2.0 * Math.sqrt(nExpertStages));

  // PoE training accuracy
  // Recompute per-operation features for PoE (simplified -- use final accuracy)
  const accuracy = finalCorrect / nSamples;

  // Store kernel data
  const convKernelData = convKernels.map((k) => Array.from(k));
  const convKernelShapes = convKernels.map((k) => {
    // Infer shape from length
    const totalElems = k.length;
    if (totalElems === 9) return [1, 1, 3, 3];
    // Try to infer nCh from stagesOnly
    return [1, 1, 3, 3];
  });

  // Weight sparsity
  let zeroCount = 0;
  for (let i = 0; i < WFinal.length; i++) {
    if (WFinal[i] === 0) zeroCount++;
  }
  const weightSparsity = pruneRatio > 0 || l1Ratio > 0 ? zeroCount / WFinal.length : 0;

  const trained: TrainedModel = {
    modelName,
    tableName,
    labelField,
    embeddingField,
    edgeLabel,
    gating,
    lam,
    layerSpecs: specDicts,
    convWeights,
    denseWeights: Array.from(WFinal),
    denseBias: Array.from(bFinal),
    denseInputChannels: nFeatures,
    denseOutputChannels: denseOutput,
    numClasses,
    classLabels: uniqueLabels,
    gridSize,
    embeddingDim,
    trainingAccuracy: accuracy,
    trainingSamples: nSamples,
    expertWeights: expertWeightsList,
    expertBiases: expertBiasesList,
    expertInputChannels,
    expertAccuracies,
    shrinkageAlpha,
    convKernelData,
    convKernelShapes,
    inChannels,
    attentionParams: attentionParamsList,
    l1Ratio,
    pruneRatio,
    weightSparsity,
  };

  engine.saveModel(modelName, trainedModelToDict(trained));

  return {
    model_name: modelName,
    training_samples: nSamples,
    num_classes: numClasses,
    training_accuracy: accuracy,
    feature_dim: nFeatures,
    class_labels: trained.classLabels,
    l1_ratio: l1Ratio,
    prune_ratio: pruneRatio,
    weight_sparsity: trained.weightSparsity,
  };
}

// -- Inference -------------------------------------------------------------

export function predict(
  engine: Engine,
  modelName: string,
  inputEmbedding: number[],
): [number, number][] {
  const config = engine.loadModel(modelName);
  if (config === null) {
    throw new Error(`Model '${modelName}' does not exist`);
  }

  const model = trainedModelFromDict(config);
  const ops = identifyOperations(model.layerSpecs as SpecDict[]);
  const nOps = ops.length;
  const hasExperts =
    model.expertWeights.length > 0 && model.expertWeights.length === nOps;

  const emb = new Float64Array(inputEmbedding);
  const gridSz = model.gridSize;

  // Reconstruct conv kernels from stored data
  const convKernelsList: Float64Array[] = [];
  if (model.convKernelData.length > 0 && model.convKernelShapes.length > 0) {
    for (let i = 0; i < model.convKernelData.length; i++) {
      convKernelsList.push(new Float64Array(model.convKernelData[i]!));
    }
  } else {
    for (const cw of model.convWeights) {
      convKernelsList.push(hopWeightsToKernel(cw));
    }
  }

  const modelInCh = model.inChannels;
  const isGrid = gridSz > 0 && gridSz * gridSz * modelInCh === model.embeddingDim;

  if (!isGrid) {
    // Fallback: return empty prediction
    return [];
  }

  // Grid-accelerated inference
  const allLogits: Float64Array[] = [];
  let currentFlat: Float64Array = new Float64Array(emb);
  let curH = gridSz;
  let curW = gridSz;

  let sIdx = 0;
  let aIdx = 0;
  for (let opIdx = 0; opIdx < ops.length; opIdx++) {
    const op = ops[opIdx]!;
    if (op[0] === "stage") {
      const poolD = op[2];
      const poolSize = (poolD["pool_size"] as number | undefined) ?? 2;
      const poolMethod = (poolD["method"] as string | undefined) ?? "max";

      const result = gridForward(
        currentFlat,
        [1, currentFlat.length],
        curH,
        curW,
        [
          {
            kernel: convKernelsList[sIdx]!,
            kernelShape: [1, 1, 3, 3],
            poolSize,
            poolMethod,
          },
        ],
        model.gating,
      );
      currentFlat = result.data;
      curH = Math.floor(curH / poolSize);
      curW = Math.floor(curW / poolSize);
      sIdx++;
    } else if (op[0] === "attention") {
      const nCh = Math.floor(currentFlat.length / (curH * curW));
      const seqLen = curH * curW;
      const X3d = new Float64Array(seqLen * nCh);
      for (let s = 0; s < seqLen; s++) {
        for (let c = 0; c < nCh; c++) {
          X3d[s * nCh + c] = currentFlat[c * seqLen + s] ?? 0;
        }
      }

      const ap =
        aIdx < model.attentionParams.length ? model.attentionParams[aIdx]! : {};
      const wQ = ap["W_q"] ? new Float64Array(ap["W_q"] as number[]) : null;
      const wK = ap["W_k"] ? new Float64Array(ap["W_k"] as number[]) : null;
      const wV = ap["W_v"] ? new Float64Array(ap["W_v"] as number[]) : null;

      const out3d = batchSelfAttention(
        X3d,
        [1, seqLen, nCh],
        (ap["n_heads"] as number | undefined) ?? 1,
        wQ,
        wK,
        wV,
        model.gating,
      );

      currentFlat = new Float64Array(nCh * seqLen);
      for (let s = 0; s < seqLen; s++) {
        for (let c = 0; c < nCh; c++) {
          currentFlat[c * seqLen + s] = out3d[s * nCh + c]!;
        }
      }
      aIdx++;
    } else {
      const gpMethod = (op[1]["method"] as string | undefined) ?? "avg";
      const result = gridGlobalPool(
        currentFlat,
        [1, currentFlat.length],
        curH,
        curW,
        gpMethod,
      );
      currentFlat = result.data;
      curH = 1;
      curW = 1;
    }

    // Expert head logits
    if (hasExperts) {
      const feat = currentFlat;
      const eic = model.expertInputChannels[opIdx]!;
      const wE = new Float64Array(model.expertWeights[opIdx]!);
      const bE = new Float64Array(model.expertBiases[opIdx]!);
      const logits = new Float64Array(model.denseOutputChannels);
      for (let j = 0; j < model.denseOutputChannels; j++) {
        logits[j] = bE[j]!;
        for (let f = 0; f < eic; f++) {
          logits[j]! += wE[j * eic + f]! * feat[f]!;
        }
      }
      allLogits.push(logits);
    }
  }

  // Final head
  const featFinal = currentFlat;
  const wF = new Float64Array(model.denseWeights);
  const bF = new Float64Array(model.denseBias);
  const finalLogits = new Float64Array(model.denseOutputChannels);
  for (let j = 0; j < model.denseOutputChannels; j++) {
    finalLogits[j] = bF[j]!;
    for (let f = 0; f < model.denseInputChannels; f++) {
      finalLogits[j]! += wF[j * model.denseInputChannels + f]! * featFinal[f]!;
    }
  }
  allLogits.push(finalLogits);

  // PoE: accuracy-weighted logit combination
  const nExperts = allLogits.length;
  const avgLogits = new Float64Array(model.denseOutputChannels);
  const accs = model.expertAccuracies;
  if (accs.length === nExperts) {
    const totalAcc = accs.reduce((a, b) => a + b, 0);
    for (let e = 0; e < nExperts; e++) {
      const w = accs[e]! / totalAcc;
      for (let j = 0; j < model.denseOutputChannels; j++) {
        avgLogits[j]! += w * allLogits[e]![j]!;
      }
    }
  } else {
    for (let e = 0; e < nExperts; e++) {
      for (let j = 0; j < model.denseOutputChannels; j++) {
        avgLogits[j]! += allLogits[e]![j]! / nExperts;
      }
    }
  }
  for (let j = 0; j < model.denseOutputChannels; j++) {
    avgLogits[j]! += model.shrinkageAlpha * Math.log(nExperts);
  }

  // Softmax
  let maxLogit = -Infinity;
  for (let j = 0; j < model.denseOutputChannels; j++) {
    if (avgLogits[j]! > maxLogit) maxLogit = avgLogits[j]!;
  }
  const expVals = new Float64Array(model.denseOutputChannels);
  let sumExp = 0;
  for (let j = 0; j < model.denseOutputChannels; j++) {
    expVals[j] = Math.exp(avgLogits[j]! - maxLogit);
    sumExp += expVals[j]!;
  }
  const probs = new Float64Array(model.denseOutputChannels);
  for (let j = 0; j < model.denseOutputChannels; j++) {
    probs[j] = expVals[j]! / sumExp;
  }

  // Sort by probability descending
  const indices: number[] = [];
  for (let j = 0; j < model.denseOutputChannels; j++) indices.push(j);
  indices.sort((a, b) => probs[b]! - probs[a]!);

  return indices.map((idx) => [idx, probs[idx]!]);
}

// -- Spec serialization ---------------------------------------------------

export function identifyStages(specDicts: SpecDict[]): [SpecDict, SpecDict][] {
  const stages: [SpecDict, SpecDict][] = [];
  let i = 0;
  while (i < specDicts.length) {
    if (specDicts[i]!.type === "conv") {
      const convD = specDicts[i]!;
      let poolD: SpecDict;
      if (i + 1 < specDicts.length && specDicts[i + 1]!.type === "pool") {
        poolD = specDicts[i + 1]!;
        i += 2;
      } else {
        poolD = { type: "pool", method: "max", pool_size: 2 };
        i += 1;
      }
      stages.push([convD, poolD]);
    } else {
      i += 1;
    }
  }
  return stages;
}

export function dictsToSpecs(dicts: Record<string, unknown>[]): LayerSpec[] {
  const result: LayerSpec[] = [];
  for (const d of dicts) {
    const t = d["type"] as string;
    if (t === "conv") {
      result.push({
        type: "conv",
        kernelHops: (d["kernel_hops"] as number | undefined) ?? 1,
        nChannels: (d["n_channels"] as number | undefined) ?? 1,
        initMode: (d["init_mode"] as string | undefined) ?? "kaiming",
      });
    } else if (t === "pool") {
      result.push({
        type: "pool",
        method: (d["method"] as string | undefined) ?? "max",
        poolSize: (d["pool_size"] as number | undefined) ?? 2,
      });
    } else if (t === "flatten") {
      result.push({ type: "flatten" });
    } else if (t === "global_pool") {
      result.push({
        type: "global_pool",
        method: (d["method"] as string | undefined) ?? "avg",
      });
    } else if (t === "dense") {
      result.push({
        type: "dense",
        outputChannels: (d["output_channels"] as number | undefined) ?? 10,
      });
    } else if (t === "softmax") {
      result.push({ type: "softmax" });
    } else if (t === "attention") {
      result.push({
        type: "attention",
        nHeads: (d["n_heads"] as number | undefined) ?? 1,
        mode: (d["mode"] as string | undefined) ?? "content",
      });
    }
  }
  return result;
}

function specsToDicts(specs: LayerSpec[]): SpecDict[] {
  const result: SpecDict[] = [];
  for (const spec of specs) {
    switch (spec.type) {
      case "conv":
        result.push({
          type: "conv",
          kernel_hops: spec.kernelHops,
          n_channels: spec.nChannels,
          init_mode: spec.initMode,
        });
        break;
      case "pool":
        result.push({
          type: "pool",
          method: spec.method,
          pool_size: spec.poolSize,
        });
        break;
      case "flatten":
        result.push({ type: "flatten" });
        break;
      case "global_pool":
        result.push({ type: "global_pool", method: spec.method });
        break;
      case "dense":
        result.push({ type: "dense", output_channels: spec.outputChannels });
        break;
      case "softmax":
        result.push({ type: "softmax" });
        break;
      case "attention":
        result.push({
          type: "attention",
          n_heads: spec.nHeads,
          mode: spec.mode,
        });
        break;
    }
  }
  return result;
}
