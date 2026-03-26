//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Category-theoretic functors
// 1:1 port of uqa/core/functor.py
//
// Functors map objects and morphisms between categories (e.g. graph -> relational).

export abstract class Functor {
  abstract mapObject(obj: unknown): unknown;
  abstract mapMorphism(morphism: unknown): unknown;
}

// -- GraphToRelationalFunctor -------------------------------------------------

/**
 * Maps graph objects (vertices/edges) into relational tuples.
 *
 * Vertices become rows with (vertexId, label, ...properties).
 * Edges become rows with (edgeId, sourceId, targetId, label, ...properties).
 */
export class GraphToRelationalFunctor extends Functor {
  mapObject(obj: unknown): unknown {
    if (obj === null || obj === undefined) return null;
    const record = obj as Record<string, unknown>;

    // Vertex: has vertexId
    if ("vertexId" in record) {
      const props = (record["properties"] ?? {}) as Record<string, unknown>;
      return {
        vertex_id: record["vertexId"],
        label: record["label"],
        ...props,
      };
    }

    // Edge: has edgeId
    if ("edgeId" in record) {
      const props = (record["properties"] ?? {}) as Record<string, unknown>;
      return {
        edge_id: record["edgeId"],
        source_id: record["sourceId"],
        target_id: record["targetId"],
        label: record["label"],
        ...props,
      };
    }

    return obj;
  }

  mapMorphism(morphism: unknown): unknown {
    // A morphism in the graph category is an edge;
    // map it to a foreign key relationship in the relational category.
    if (morphism === null || morphism === undefined) return null;
    const edge = morphism as Record<string, unknown>;
    return {
      from_table: "vertices",
      from_column: "vertex_id",
      to_table: "vertices",
      to_column: "vertex_id",
      source_id: edge["sourceId"],
      target_id: edge["targetId"],
      label: edge["label"],
    };
  }
}

// -- RelationalToGraphFunctor -------------------------------------------------

/**
 * Maps relational rows into graph edges. Each row becomes an edge
 * connecting the row's primary key to each foreign-key-referenced row.
 */
export class RelationalToGraphFunctor extends Functor {
  private _edgeLabel: string;

  constructor(edgeLabel?: string) {
    super();
    this._edgeLabel = edgeLabel ?? "has";
  }

  get edgeLabel(): string {
    return this._edgeLabel;
  }

  mapObject(obj: unknown): unknown {
    // A relational row becomes a graph vertex
    if (obj === null || obj === undefined) return null;
    const row = obj as Record<string, unknown>;
    const vertexId = row["id"] ?? row["_doc_id"] ?? 0;
    const props: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(row)) {
      if (key !== "id" && key !== "_doc_id") {
        props[key] = value;
      }
    }
    return {
      vertexId,
      label: "row",
      properties: props,
    };
  }

  mapMorphism(morphism: unknown): unknown {
    // A foreign-key relationship becomes an edge
    if (morphism === null || morphism === undefined) return null;
    const fk = morphism as Record<string, unknown>;
    return {
      edgeId: 0,
      sourceId: fk["source_id"],
      targetId: fk["target_id"],
      label: this._edgeLabel,
      properties: {},
    };
  }
}

// -- TextToVectorFunctor ------------------------------------------------------

/**
 * Placeholder functor that represents the mapping from text documents
 * to vector embeddings. The actual embedding computation requires an
 * external model; this functor defines the categorical structure.
 */
export class TextToVectorFunctor extends Functor {
  private _dimensions: number;

  constructor(dimensions?: number) {
    super();
    this._dimensions = dimensions ?? 768;
  }

  get dimensions(): number {
    return this._dimensions;
  }

  mapObject(obj: unknown): unknown {
    // Map a text string to a zero-vector placeholder of the correct dimension.
    // In production, this would invoke an embedding model.
    if (typeof obj === "string") {
      return new Float64Array(this._dimensions);
    }
    return obj;
  }

  mapMorphism(morphism: unknown): unknown {
    // Morphisms in the text category (e.g. substring relationships)
    // map to cosine similarity in the vector category.
    if (morphism === null || morphism === undefined) return null;
    return {
      type: "cosine_similarity",
      source: this.mapObject((morphism as Record<string, unknown>)["source"]),
      target: this.mapObject((morphism as Record<string, unknown>)["target"]),
    };
  }
}
