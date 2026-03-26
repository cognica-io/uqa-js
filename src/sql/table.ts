//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- SQL Table
// 1:1 port of uqa/sql/table.py

import type { IndexedTerms } from "../storage/abc/inverted-index.js";
import type { DocumentStore } from "../storage/abc/document-store.js";
import type { InvertedIndex } from "../storage/abc/inverted-index.js";
import { MemoryDocumentStore } from "../storage/document-store.js";
import { MemoryInvertedIndex } from "../storage/inverted-index.js";
import type { VectorIndex } from "../storage/vector-index.js";
import { FlatVectorIndex } from "../storage/vector-index.js";
import { SpatialIndex } from "../storage/spatial-index.js";

// -- ColumnStats ----------------------------------------------------------------

export interface ColumnStats {
  distinctCount: number;
  nullCount: number;
  minValue: unknown;
  maxValue: unknown;
  rowCount: number;
  histogram: unknown[];
  mcvValues: unknown[];
  mcvFrequencies: number[];
}

export function columnStatsSelectivity(stats: ColumnStats): number {
  if (stats.distinctCount <= 0) return 1.0;
  return 1.0 / stats.distinctCount;
}

// -- ColumnDef ------------------------------------------------------------------

export interface ColumnDef {
  readonly name: string;
  readonly typeName: string;
  readonly pythonType: string; // "number" | "string" | "boolean" | "object" | "array"
  readonly primaryKey: boolean;
  readonly notNull: boolean;
  readonly autoIncrement: boolean;
  readonly defaultValue: unknown;
  readonly vectorDimensions: number | null;
  readonly unique: boolean;
  readonly numericPrecision: number | null;
  readonly numericScale: number | null;
}

export function createColumnDef(
  name: string,
  typeName: string,
  opts?: Partial<Omit<ColumnDef, "name" | "typeName">>,
): ColumnDef {
  return {
    name,
    typeName,
    pythonType: opts?.pythonType ?? "string",
    primaryKey: opts?.primaryKey ?? false,
    notNull: opts?.notNull ?? false,
    autoIncrement: opts?.autoIncrement ?? false,
    defaultValue: opts?.defaultValue ?? null,
    vectorDimensions: opts?.vectorDimensions ?? null,
    unique: opts?.unique ?? false,
    numericPrecision: opts?.numericPrecision ?? null,
    numericScale: opts?.numericScale ?? null,
  };
}

// -- ForeignKeyDef --------------------------------------------------------------

export interface ForeignKeyDef {
  readonly column: string;
  readonly refTable: string;
  readonly refColumn: string;
}

// -- Type map -------------------------------------------------------------------

const TYPE_MAP: ReadonlyMap<string, [string, string]> = new Map([
  ["INT", ["integer", "number"]],
  ["INTEGER", ["integer", "number"]],
  ["SERIAL", ["integer", "number"]],
  ["BIGINT", ["integer", "number"]],
  ["BIGSERIAL", ["integer", "number"]],
  ["SMALLINT", ["integer", "number"]],
  ["SMALLSERIAL", ["integer", "number"]],
  ["INT2", ["integer", "number"]],
  ["INT4", ["integer", "number"]],
  ["INT8", ["integer", "number"]],

  ["FLOAT", ["float", "number"]],
  ["FLOAT4", ["float", "number"]],
  ["FLOAT8", ["float", "number"]],
  ["DOUBLE", ["float", "number"]],
  ["DOUBLE PRECISION", ["float", "number"]],
  ["REAL", ["float", "number"]],
  ["NUMERIC", ["float", "number"]],
  ["DECIMAL", ["float", "number"]],

  ["TEXT", ["text", "string"]],
  ["VARCHAR", ["text", "string"]],
  ["CHAR", ["text", "string"]],
  ["CHARACTER", ["text", "string"]],
  ["CHARACTER VARYING", ["text", "string"]],
  ["NAME", ["text", "string"]],

  ["BOOLEAN", ["boolean", "boolean"]],
  ["BOOL", ["boolean", "boolean"]],

  ["DATE", ["date", "string"]],
  ["TIME", ["time", "string"]],
  ["TIMESTAMP", ["timestamp", "string"]],
  ["TIMESTAMPTZ", ["timestamp", "string"]],
  ["TIMESTAMP WITH TIME ZONE", ["timestamp", "string"]],
  ["TIMESTAMP WITHOUT TIME ZONE", ["timestamp", "string"]],
  ["INTERVAL", ["interval", "string"]],

  ["VECTOR", ["vector", "object"]],
  ["POINT", ["point", "object"]],

  ["JSON", ["json", "object"]],
  ["JSONB", ["json", "object"]],

  ["BYTEA", ["bytea", "object"]],

  ["ARRAY", ["array", "array"]],
  ["UUID", ["text", "string"]],
]);

/**
 * Resolve an array of SQL type name tokens into a canonical [typeName, jsTypeCategory] pair.
 */
export function resolveType(
  typeNames: string[],
  arrayBounds?: unknown[] | null,
): [string, string] {
  const raw = typeNames[typeNames.length - 1]!.toLowerCase();

  if (raw === "vector") return ["vector", "object"];
  if (raw === "point") return ["point", "object"];

  if (arrayBounds !== null && arrayBounds !== undefined && arrayBounds.length > 0) {
    // Array column: e.g. TEXT[] -> element type is "text"
    const combined = typeNames.map((t) => t.toUpperCase()).join(" ");
    const match = TYPE_MAP.get(combined) ?? TYPE_MAP.get(typeNames[0]!.toUpperCase());
    if (match === undefined) {
      throw new Error(`Unsupported array element type: ${raw}`);
    }
    return [`${raw}[]`, "array"];
  }

  // Try the full combined name first (e.g. "DOUBLE PRECISION")
  const combined = typeNames.map((t) => t.toUpperCase()).join(" ");
  const fullMatch = TYPE_MAP.get(combined);
  if (fullMatch) return [fullMatch[0], fullMatch[1]];

  // Try the first token alone
  const first = typeNames[0];
  if (first !== undefined) {
    const singleMatch = TYPE_MAP.get(first.toUpperCase());
    if (singleMatch) return [singleMatch[0], singleMatch[1]];
  }

  // Unknown type -- fall back to text/string
  return ["text", "string"];
}

// -- Type coercion helpers ------------------------------------------------------

function coerceJSON(value: unknown): unknown {
  if (typeof value === "object" && value !== null) return value;
  if (typeof value === "string") return JSON.parse(value);
  return value;
}

function coerceBytea(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) return value;
  if (typeof value === "string") return new TextEncoder().encode(value);
  return new TextEncoder().encode(String(value));
}

function coerceArray(value: unknown): unknown[] {
  if (Array.isArray(value)) return value;
  if (typeof value === "string") return JSON.parse(value) as unknown[];
  return [value];
}

function coerceNumeric(value: unknown, scale: number): number {
  const num = typeof value === "number" ? value : Number(value);
  const factor = Math.pow(10, scale);
  return Math.round(num * factor) / factor;
}

// -- Table ----------------------------------------------------------------------

const HISTOGRAM_BUCKETS = 100;
const MCV_COUNT = 10;

export class Table {
  readonly name: string;
  readonly columns: Map<string, ColumnDef>;
  readonly primaryKey: string | null;
  checkConstraints: [string, (row: Record<string, unknown>) => boolean][];
  foreignKeys: ForeignKeyDef[];
  fkInsertValidators: ((row: Record<string, unknown>) => void)[];
  fkDeleteValidators: ((docId: number) => void)[];
  fkUpdateValidators: ((
    oldDoc: Record<string, unknown>,
    newDoc: Record<string, unknown>,
  ) => void)[];
  documentStore: DocumentStore;
  invertedIndex: InvertedIndex;
  vectorIndexes: Map<string, VectorIndex>;
  spatialIndexes: Map<string, SpatialIndex>;
  private _stats: Map<string, ColumnStats>;
  _nextDocId: number;
  private _uniqueIndexes: Map<string, Map<unknown, number>>;
  private _uniqueIndexesBuilt: boolean;

  constructor(name: string, columns: ColumnDef[], conn?: unknown) {
    this.name = name;
    this.columns = new Map<string, ColumnDef>();
    this.checkConstraints = [];
    this.foreignKeys = [];
    this.fkInsertValidators = [];
    this.fkDeleteValidators = [];
    this.fkUpdateValidators = [];
    this._stats = new Map();
    this._nextDocId = 1;
    this.vectorIndexes = new Map();
    this.spatialIndexes = new Map();
    this._uniqueIndexes = new Map();
    this._uniqueIndexesBuilt = false;

    // Determine primary key and populate column map
    let pk: string | null = null;
    for (const col of columns) {
      this.columns.set(col.name, col);
      if (col.primaryKey) {
        pk = col.name;
      }
    }
    this.primaryKey = pk;

    // Set up storage backends -- in-memory when conn is null/undefined
    void conn; // reserved for future backend selection
    this.documentStore = new MemoryDocumentStore();
    this.invertedIndex = new MemoryInvertedIndex();

    // Create vector indexes for VECTOR columns
    for (const col of columns) {
      if (col.typeName === "vector" && col.vectorDimensions !== null) {
        this.vectorIndexes.set(col.name, new FlatVectorIndex(col.vectorDimensions));
      }
      if (col.typeName === "point") {
        this.spatialIndexes.set(col.name, new SpatialIndex(this.name, col.name));
      }
    }
  }

  get columnNames(): string[] {
    return [...this.columns.keys()];
  }

  get rowCount(): number {
    return this.documentStore.length;
  }

  /**
   * Insert a row into the table. Returns [docId, indexedTerms].
   */
  insert(row: Record<string, unknown>): [number, IndexedTerms | null] {
    // -- primary key / doc_id resolution --
    let docId: number;
    if (this.primaryKey !== null) {
      const pkCol = this.columns.get(this.primaryKey)!;
      if (pkCol.autoIncrement) {
        if (
          !(this.primaryKey in row) ||
          row[this.primaryKey] === null ||
          row[this.primaryKey] === undefined
        ) {
          row[this.primaryKey] = this._nextDocId;
        }
        docId = row[this.primaryKey] as number;
        this._nextDocId = Math.max(this._nextDocId, docId + 1);
      } else {
        if (
          !(this.primaryKey in row) ||
          row[this.primaryKey] === null ||
          row[this.primaryKey] === undefined
        ) {
          throw new Error(
            `Missing primary key '${this.primaryKey}' for table '${this.name}'`,
          );
        }
        const pkVal = row[this.primaryKey];
        if (typeof pkVal === "number") {
          docId = pkVal;
          this._nextDocId = Math.max(this._nextDocId, docId + 1);
        } else {
          // Non-integer PK: auto-generate doc_id
          docId = this._nextDocId;
          this._nextDocId++;
        }
      }
    } else {
      docId = this._nextDocId;
      this._nextDocId++;
    }

    // -- NOT NULL validation --
    for (const [colName, colDef] of this.columns) {
      if (colDef.notNull && !colDef.autoIncrement) {
        const value = row[colName];
        if (value === null || value === undefined) {
          if (colDef.defaultValue !== null && colDef.defaultValue !== undefined) {
            row[colName] = colDef.defaultValue;
          } else {
            throw new Error(
              `NOT NULL constraint violated: column '${colName}' in table '${this.name}'`,
            );
          }
        }
      }
    }

    // -- UNIQUE constraint validation --
    this._buildUniqueIndexes();
    for (const [colName, colDef] of this.columns) {
      if (!(colDef.unique || colDef.primaryKey)) continue;
      if (colDef.autoIncrement) continue;
      const value = row[colName];
      if (value === null || value === undefined) continue; // NULL is allowed in UNIQUE
      const idx = this._uniqueIndexes.get(colName);
      if (idx !== undefined && idx.has(value)) {
        throw new Error(
          `UNIQUE constraint violated: duplicate value '${String(value as string | number)}' for column '${colName}' in table '${this.name}'`,
        );
      }
    }

    // -- CHECK constraint validation --
    for (const [constraintName, checkFn] of this.checkConstraints) {
      if (!checkFn(row)) {
        throw new Error(
          `CHECK constraint '${constraintName}' violated in table '${this.name}'`,
        );
      }
    }

    // -- FOREIGN KEY constraint validation --
    for (const fkValidator of this.fkInsertValidators) {
      fkValidator(row);
    }

    // -- unknown column check --
    for (const colName of Object.keys(row)) {
      if (!this.columns.has(colName)) {
        throw new Error(`Unknown column '${colName}' for table '${this.name}'`);
      }
    }

    // -- type coercion + defaults --
    const coerced: Record<string, unknown> = {};
    const vectors: Record<string, Float64Array> = {};
    const points: Record<string, [number, number]> = {};

    for (const [colName, colDef] of this.columns) {
      const rawValue = row[colName];
      if (rawValue !== null && rawValue !== undefined) {
        if (colDef.vectorDimensions !== null) {
          const vec =
            rawValue instanceof Float64Array
              ? rawValue
              : new Float64Array(rawValue as number[]);
          coerced[colName] = Array.from(vec);
          vectors[colName] = vec;
        } else if (colDef.typeName === "point") {
          const pt = rawValue as [number, number] | { x: number; y: number };
          let x: number, y: number;
          if (Array.isArray(pt)) {
            x = pt[0];
            y = pt[1];
          } else if (typeof pt === "object" && "x" in pt) {
            x = pt.x;
            y = pt.y;
          } else {
            throw new Error(
              `POINT column '${colName}' requires [x, y] (2 elements), got ${JSON.stringify(pt)}`,
            );
          }
          coerced[colName] = [x, y];
          points[colName] = [x, y];
        } else if (colDef.typeName === "json" || colDef.typeName === "jsonb") {
          coerced[colName] = coerceJSON(rawValue);
        } else if (colDef.typeName.endsWith("[]")) {
          coerced[colName] = coerceArray(rawValue);
        } else if (colDef.typeName === "bytea") {
          coerced[colName] = coerceBytea(rawValue);
        } else if (colDef.numericScale !== null) {
          coerced[colName] = coerceNumeric(rawValue, colDef.numericScale);
        } else if (colDef.pythonType === "number") {
          coerced[colName] = Number(rawValue);
        } else if (colDef.pythonType === "boolean") {
          coerced[colName] = Boolean(rawValue);
        } else if (colDef.pythonType === "string") {
          coerced[colName] = String(rawValue as string | number);
        } else {
          coerced[colName] = rawValue;
        }
      } else if (colDef.defaultValue !== null && colDef.defaultValue !== undefined) {
        coerced[colName] = colDef.defaultValue;
      }
      // else: column absent -> not stored (sparse document)
    }

    // -- persist --
    this.documentStore.put(docId, coerced);

    let indexed: IndexedTerms | null = null;
    const textFields: Record<string, string> = {};
    for (const [k, v] of Object.entries(coerced)) {
      if (typeof v === "string") {
        textFields[k] = v;
      }
    }
    if (Object.keys(textFields).length > 0) {
      indexed = this.invertedIndex.addDocument(docId, textFields);
    }

    for (const [fieldName, vec] of Object.entries(vectors)) {
      const vecIdx = this.vectorIndexes.get(fieldName);
      if (vecIdx !== undefined) {
        vecIdx.add(docId, vec);
      }
    }

    for (const [fieldName, [px, py]] of Object.entries(points)) {
      const spIdx = this.spatialIndexes.get(fieldName);
      if (spIdx !== undefined) {
        spIdx.add(docId, px, py);
      }
    }

    // Maintain unique indexes.
    for (const [colName, idx] of this._uniqueIndexes) {
      const val = coerced[colName];
      if (val !== null && val !== undefined) {
        idx.set(val, docId);
      }
    }

    return [docId, indexed];
  }

  private _buildUniqueIndexes(): void {
    if (this._uniqueIndexesBuilt) return;
    this._uniqueIndexesBuilt = true;
    const uniqueCols: string[] = [];
    for (const [colName, colDef] of this.columns) {
      if ((colDef.unique || colDef.primaryKey) && !colDef.autoIncrement) {
        uniqueCols.push(colName);
        this._uniqueIndexes.set(colName, new Map());
      }
    }
    if (uniqueCols.length === 0) return;
    for (const docId of this.documentStore.docIds) {
      for (const colName of uniqueCols) {
        const val = this.documentStore.getField(docId, colName);
        if (val !== null && val !== undefined) {
          this._uniqueIndexes.get(colName)!.set(val, docId);
        }
      }
    }
  }

  removeFromUniqueIndexes(docId: number): void {
    for (const [colName, idx] of this._uniqueIndexes) {
      const val = this.documentStore.getField(docId, colName);
      if (val !== null && val !== undefined) {
        idx.delete(val);
      }
    }
  }

  /**
   * Compute column statistics for the query optimizer.
   */
  analyze(): Map<string, ColumnStats> {
    const docIds = [...this.documentStore.docIds].sort((a, b) => a - b);
    const n = docIds.length;
    const colNames = [...this.columns.keys()];

    // Single-pass: collect all column values per doc.
    const colValues: Map<string, unknown[]> = new Map();
    const colNulls: Map<string, number> = new Map();
    for (const c of colNames) {
      colValues.set(c, []);
      colNulls.set(c, 0);
    }

    for (const docId of docIds) {
      const doc = this.documentStore.get(docId);
      for (const colName of colNames) {
        const val = doc !== null ? (doc[colName] ?? null) : null;
        if (val === null) {
          colNulls.set(colName, (colNulls.get(colName) ?? 0) + 1);
        } else {
          colValues.get(colName)!.push(val);
        }
      }
    }

    const stats = new Map<string, ColumnStats>();
    for (const colName of colNames) {
      const values = colValues.get(colName)!;
      const nullCount = colNulls.get(colName)!;
      const distinct = new Set(
        values.map((v) =>
          typeof v === "object"
            ? JSON.stringify(v)
            : String(v as string | number | boolean),
        ),
      ).size;

      const comparable = values.filter(
        (v) => typeof v === "number" || typeof v === "string",
      );

      let minVal: unknown = null;
      let maxVal: unknown = null;
      if (comparable.length > 0) {
        minVal = comparable.reduce((a, b) => (a < b ? a : b));
        maxVal = comparable.reduce((a, b) => (a > b ? a : b));
      }

      const histogram = _buildHistogram(comparable);
      const [mcvValues, mcvFrequencies] = _buildMcv(values, n);

      stats.set(colName, {
        distinctCount: distinct,
        nullCount,
        minValue: minVal,
        maxValue: maxVal,
        rowCount: n,
        histogram,
        mcvValues,
        mcvFrequencies,
      });
    }

    this._stats = stats;
    return new Map(stats);
  }

  getColumnStats(colName: string): ColumnStats | null {
    return this._stats.get(colName) ?? null;
  }
}

// -- Histogram and MCV helpers --------------------------------------------------

function _buildHistogram(values: (number | string)[]): unknown[] {
  if (values.length === 0) return [];
  try {
    const sorted = [...values].sort((a, b) => {
      if (typeof a === "number" && typeof b === "number") return a - b;
      return String(a).localeCompare(String(b));
    });

    const n = sorted.length;
    const numBuckets = Math.min(HISTOGRAM_BUCKETS, n);
    if (numBuckets <= 1) {
      return [sorted[0], sorted[n - 1]];
    }

    const boundaries: unknown[] = [sorted[0]];
    for (let i = 1; i < numBuckets; i++) {
      const idx = Math.floor((i * n) / numBuckets);
      const val = sorted[idx];
      if (val !== boundaries[boundaries.length - 1]) {
        boundaries.push(val);
      }
    }
    if (boundaries[boundaries.length - 1] !== sorted[n - 1]) {
      boundaries.push(sorted[n - 1]);
    }
    return boundaries;
  } catch {
    return [];
  }
}

function _buildMcv(values: unknown[], total: number): [unknown[], number[]] {
  if (values.length === 0 || total <= 0) return [[], []];

  const counts = new Map<string, { value: unknown; count: number }>();
  for (const v of values) {
    const key =
      typeof v === "object"
        ? JSON.stringify(v)
        : String(v as string | number | boolean);
    const entry = counts.get(key);
    if (entry !== undefined) {
      entry.count++;
    } else {
      counts.set(key, { value: v, count: 1 });
    }
  }

  const ndv = counts.size;
  if (ndv <= 0) return [[], []];

  const avgFreq = 1.0 / ndv;
  const sorted = [...counts.values()]
    .sort((a, b) => b.count - a.count)
    .slice(0, MCV_COUNT);

  const aboveAvg = sorted.filter((e) => e.count / total > avgFreq);
  if (aboveAvg.length === 0) return [[], []];

  return [aboveAvg.map((e) => e.value), aboveAvg.map((e) => e.count / total)];
}
