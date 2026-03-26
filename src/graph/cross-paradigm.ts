//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Cross-paradigm graph operators
// 1:1 port of uqa/graph/cross_paradigm.py

import type { PostingEntry } from "../core/types.js";
import { createPostingEntry, createVertex, createEdge } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import { Operator } from "../operators/base.js";
import type { ExecutionContext } from "../operators/base.js";
import type { GraphStore } from "../storage/abc/graph-store.js";
import { MemoryGraphStore } from "./store.js";
import { GraphPostingList, createGraphPayload } from "./posting-list.js";
import { cosine } from "../math/linalg.js";
import { DEFAULT_ANALYZER } from "../analysis/analyzer.js";

// -- ToGraphOperator ----------------------------------------------------------

export class ToGraphOperator extends Operator {
  readonly source: Operator;
  readonly graphName: string;
  readonly labelField: string;

  constructor(source: Operator, graphName: string, labelField?: string) {
    super();
    this.source = source;
    this.graphName = graphName;
    this.labelField = labelField ?? "_label";
  }

  execute(context: ExecutionContext): PostingList {
    const sourceResult = this.source.execute(context);
    const store = (context.graphStore as GraphStore | null) ?? new MemoryGraphStore();

    if (!store.hasGraph(this.graphName)) {
      store.createGraph(this.graphName);
    }

    // Convert each document to a vertex
    for (const entry of sourceResult) {
      const fields = entry.payload.fields as Record<string, unknown>;
      const label =
        typeof fields[this.labelField] === "string"
          ? (fields[this.labelField] as string)
          : "document";

      const vertex = createVertex(entry.docId, label, {
        ...fields,
        _score: entry.payload.score,
      });
      store.addVertex(vertex, this.graphName);
    }

    // Return the source result as-is (the graph store is modified in context)
    return sourceResult;
  }
}

// -- FromGraphOperator --------------------------------------------------------

export class FromGraphOperator extends Operator {
  readonly graph: string;

  constructor(graph: string) {
    super();
    this.graph = graph;
  }

  execute(context: ExecutionContext): PostingList {
    const store = context.graphStore as GraphStore;
    const vertices = store.verticesInGraph(this.graph);

    const entries: PostingEntry[] = [];
    for (const vertex of vertices) {
      entries.push(
        createPostingEntry(vertex.vertexId, {
          score:
            typeof vertex.properties["_score"] === "number"
              ? vertex.properties["_score"]
              : 0.0,
          fields: { ...vertex.properties, _label: vertex.label },
        }),
      );
    }

    return new PostingList(entries);
  }
}

// -- SemanticGraphSearchOperator ----------------------------------------------

export class SemanticGraphSearchOperator extends Operator {
  readonly startVertex: number;
  readonly graph: string;
  readonly queryVector: Float64Array;
  readonly vectorField: string;
  readonly threshold: number;
  readonly maxHops: number;

  constructor(opts: {
    startVertex: number;
    graph: string;
    queryVector: Float64Array;
    vectorField?: string;
    threshold?: number;
    maxHops?: number;
  }) {
    super();
    this.startVertex = opts.startVertex;
    this.graph = opts.graph;
    this.queryVector = opts.queryVector;
    this.vectorField = opts.vectorField ?? "embedding";
    this.threshold = opts.threshold ?? 0.5;
    this.maxHops = opts.maxHops ?? 3;
  }

  execute(context: ExecutionContext): PostingList {
    const store = context.graphStore as GraphStore;

    // Traverse and filter by vector similarity
    const visited = new Set<number>();
    const entries: PostingEntry[] = [];
    const queue: Array<[number, number]> = [[this.startVertex, 0]];
    visited.add(this.startVertex);

    const allVertices = new Set<number>();
    const allEdges = new Set<number>();

    while (queue.length > 0) {
      const [current, depth] = queue.shift()!;
      const vertex = store.getVertex(current);
      if (!vertex) continue;

      // Check vector similarity
      const embedding = vertex.properties[this.vectorField] as Float64Array | undefined;
      let sim = 0.0;
      if (embedding && embedding instanceof Float64Array) {
        sim = cosine(this.queryVector, embedding);
      }

      if (sim >= this.threshold || depth === 0) {
        allVertices.add(current);
        entries.push(
          createPostingEntry(current, {
            score: sim,
            fields: { ...vertex.properties, _depth: depth, _similarity: sim },
          }),
        );
      }

      if (depth < this.maxHops) {
        const neighbors = store.neighbors(current, this.graph, null, "out");
        for (const nid of neighbors) {
          if (!visited.has(nid)) {
            visited.add(nid);
            queue.push([nid, depth + 1]);
          }
        }
        const outEdges = store.outEdgeIds(current, this.graph);
        for (const eid of outEdges) {
          allEdges.add(eid);
        }
      }
    }

    const gpl = new GraphPostingList(entries);
    for (const entry of entries) {
      gpl.setGraphPayload(
        entry.docId,
        createGraphPayload({
          subgraphVertices: allVertices,
          subgraphEdges: allEdges,
          score: entry.payload.score,
          graphName: this.graph,
        }),
      );
    }
    return gpl;
  }
}

// -- VertexEmbeddingOperator --------------------------------------------------

export class VertexEmbeddingOperator extends Operator {
  readonly graph: string;
  readonly queryVector: Float64Array;
  readonly vectorField: string;

  constructor(opts: {
    graph: string;
    queryVector: Float64Array;
    vectorField?: string;
  }) {
    super();
    this.graph = opts.graph;
    this.queryVector = opts.queryVector;
    this.vectorField = opts.vectorField ?? "embedding";
  }

  execute(context: ExecutionContext): PostingList {
    const store = context.graphStore as GraphStore;
    const vertices = store.verticesInGraph(this.graph);

    const entries: PostingEntry[] = [];
    for (const vertex of vertices) {
      const embedding = vertex.properties[this.vectorField] as Float64Array | undefined;
      let sim = 0.0;
      if (embedding && embedding instanceof Float64Array) {
        sim = cosine(this.queryVector, embedding);
      }

      entries.push(
        createPostingEntry(vertex.vertexId, {
          score: sim,
          fields: { ...vertex.properties, _similarity: sim },
        }),
      );
    }

    const gpl = new GraphPostingList(entries);
    for (const entry of entries) {
      gpl.setGraphPayload(
        entry.docId,
        createGraphPayload({
          subgraphVertices: new Set([entry.docId]),
          subgraphEdges: new Set(),
          score: entry.payload.score,
          graphName: this.graph,
        }),
      );
    }
    return gpl;
  }
}

// -- VectorEnhancedMatchOperator ----------------------------------------------

export class VectorEnhancedMatchOperator extends Operator {
  readonly pattern: unknown;
  readonly queryVector: Float64Array;
  readonly scoreVariable: string;
  readonly vectorField: string;
  readonly threshold: number;
  readonly graphName: string;

  constructor(opts: {
    pattern: unknown;
    queryVector: Float64Array;
    scoreVariable: string;
    vectorField?: string;
    threshold?: number;
    graph: string;
  }) {
    super();
    this.pattern = opts.pattern;
    this.queryVector = opts.queryVector;
    this.scoreVariable = opts.scoreVariable;
    this.vectorField = opts.vectorField ?? "embedding";
    this.threshold = opts.threshold ?? 0.0;
    this.graphName = opts.graph;
  }

  execute(context: ExecutionContext): PostingList {
    const graphStore = context.graphStore as GraphStore | undefined;
    if (!graphStore) return new PostingList();

    // Import pattern match operator dynamically to avoid circular deps
    let PatternMatchOperator: new (
      pattern: unknown,
      opts: { graph: string },
    ) => Operator;
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const mod = require("./operators.js") as {
        PatternMatchOperator: typeof PatternMatchOperator;
      };
      PatternMatchOperator = mod.PatternMatchOperator;
    } catch {
      return new PostingList();
    }

    const matchOp = new PatternMatchOperator(this.pattern, {
      graph: this.graphName,
    });
    const matchResult = matchOp.execute(context);

    const entries: PostingEntry[] = [];
    const graphPayloads = new Map<number, ReturnType<typeof createGraphPayload>>();

    for (const entry of matchResult) {
      const fields = entry.payload.fields as Record<string, unknown> | undefined;
      const vid = fields?.[this.scoreVariable] as number | undefined;
      if (vid === undefined) continue;

      const vertex = graphStore.getVertex(vid);
      if (!vertex) continue;

      const vec = vertex.properties[this.vectorField];
      if (!vec || !(vec instanceof Float64Array)) continue;

      const sim = cosine(this.queryVector, vec);
      if (sim >= this.threshold) {
        entries.push(
          createPostingEntry(entry.docId, {
            score: sim,
            fields: fields,
          }),
        );
        graphPayloads.set(
          entry.docId,
          createGraphPayload({
            subgraphVertices: new Set([vid]),
            subgraphEdges: new Set<number>(),
            score: sim,
            graphName: this.graphName,
          }),
        );
      }
    }

    const gpl = new GraphPostingList(entries, graphPayloads);
    return gpl;
  }
}

// -- TextToGraphOperator ------------------------------------------------------

export class TextToGraphOperator extends Operator {
  readonly source: Operator;
  readonly graphName: string;
  readonly windowSize: number;

  constructor(source: Operator, graphName: string, windowSize?: number) {
    super();
    this.source = source;
    this.graphName = graphName;
    this.windowSize = windowSize ?? 3;
  }

  execute(context: ExecutionContext): PostingList {
    const sourceResult = this.source.execute(context);
    const store = (context.graphStore as GraphStore | null) ?? new MemoryGraphStore();

    if (!store.hasGraph(this.graphName)) {
      store.createGraph(this.graphName);
    }

    // Track token -> vertexId mapping
    const tokenVertexMap = new Map<string, number>();
    // Track co-occurrence edges
    const edgeSet = new Set<string>();

    for (const entry of sourceResult) {
      const fields = entry.payload.fields as Record<string, unknown>;
      const textField = fields["text"] ?? fields["content"] ?? fields["body"];
      if (typeof textField !== "string") continue;

      const tokens = DEFAULT_ANALYZER.analyze(textField);

      // Create vertices for tokens
      for (const token of tokens) {
        if (!tokenVertexMap.has(token)) {
          const vid = store.nextVertexId();
          const vertex = createVertex(vid, "token", { text: token });
          store.addVertex(vertex, this.graphName);
          tokenVertexMap.set(token, vid);
        }
      }

      // Create co-occurrence edges within window
      for (let i = 0; i < tokens.length; i++) {
        for (let j = i + 1; j < Math.min(i + this.windowSize, tokens.length); j++) {
          const srcToken = tokens[i]!;
          const tgtToken = tokens[j]!;
          if (srcToken === tgtToken) continue;

          const edgeKey = `${srcToken}\0${tgtToken}`;
          if (edgeSet.has(edgeKey)) continue;
          edgeSet.add(edgeKey);

          const srcVid = tokenVertexMap.get(srcToken)!;
          const tgtVid = tokenVertexMap.get(tgtToken)!;
          const eid = store.nextEdgeId();
          const edge = createEdge(eid, srcVid, tgtVid, "co_occurs", {
            distance: j - i,
          });
          store.addEdge(edge, this.graphName);
        }
      }
    }

    // Return vertices as posting list
    const entries: PostingEntry[] = [];
    for (const [token, vid] of tokenVertexMap) {
      entries.push(
        createPostingEntry(vid, {
          score: 1.0,
          fields: { text: token },
        }),
      );
    }

    return new GraphPostingList(entries);
  }
}
