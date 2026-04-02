//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Columnar batch (pure JS arrays, no PyArrow)
// 1:1 port of uqa/execution/batch.py

export type DataType =
  | "integer"
  | "float"
  | "text"
  | "boolean"
  | "bytes"
  | "timestamp"
  | "date"
  | "time"
  | "interval"
  | "unknown";

// ---------------------------------------------------------------------------
// Arrow-compatible schema
// ---------------------------------------------------------------------------

export interface ArrowField {
  readonly name: string;
  readonly type: DataType;
  readonly nullable: boolean;
}

export interface ArrowSchema {
  readonly fields: readonly ArrowField[];
}

export function arrowSchema(fields: ArrowField[]): ArrowSchema {
  return { fields: Object.freeze([...fields]) };
}

export function arrowField(name: string, type: DataType, nullable = true): ArrowField {
  return { name, type, nullable };
}

// ---------------------------------------------------------------------------
// ColumnVector -- typed column with optional selection vector
// ---------------------------------------------------------------------------

/**
 * ColumnVector wraps a plain JS array with an optional selection vector.
 * When a selection vector is present, only indices listed in the selection
 * vector are logically valid. This avoids copying data during filter operations.
 */
export class ColumnVector {
  readonly name: string;
  readonly type: DataType;
  private _data: unknown[];
  private _selection: number[] | null;
  private _logicalLength: number;

  constructor(
    name: string,
    type: DataType,
    data: unknown[],
    selection?: number[] | null,
  ) {
    this.name = name;
    this.type = type;
    this._data = data;
    this._selection = selection ?? null;
    this._logicalLength =
      this._selection !== null ? this._selection.length : data.length;
  }

  /** Physical length of the underlying data array. */
  get physicalLength(): number {
    return this._data.length;
  }

  /** Logical length respecting the selection vector. */
  get length(): number {
    return this._logicalLength;
  }

  /** Whether a selection vector is active. */
  get hasSelection(): boolean {
    return this._selection !== null;
  }

  /** Access the raw data array (bypasses selection). */
  get rawData(): unknown[] {
    return this._data;
  }

  /** Access the selection vector, or null if none. */
  get selection(): number[] | null {
    return this._selection;
  }

  /**
   * Get the value at logical index i.
   * If a selection vector is present, maps through it.
   */
  get(i: number): unknown {
    if (this._selection !== null) {
      const physIdx = this._selection[i];
      if (physIdx === undefined) return null;
      return this._data[physIdx] ?? null;
    }
    return this._data[i] ?? null;
  }

  /**
   * Return a new ColumnVector with an applied selection vector.
   * The resulting selection vector is composed with any existing selection.
   */
  withSelection(indices: number[]): ColumnVector {
    if (this._selection !== null) {
      // Compose: map through existing selection
      const composed: number[] = [];
      for (const idx of indices) {
        if (idx < this._selection.length) {
          composed.push(this._selection[idx]!);
        }
      }
      return new ColumnVector(this.name, this.type, this._data, composed);
    }
    return new ColumnVector(this.name, this.type, this._data, indices);
  }

  /**
   * Compact the column vector by materializing only the selected rows
   * into a new contiguous array. Removes the selection vector.
   */
  compact(): ColumnVector {
    if (this._selection === null) return this;
    const compacted: unknown[] = [];
    for (const idx of this._selection) {
      compacted.push(this._data[idx] ?? null);
    }
    return new ColumnVector(this.name, this.type, compacted, null);
  }

  /**
   * Convert to a plain array respecting selection.
   */
  toArray(): unknown[] {
    if (this._selection === null) return [...this._data];
    const result: unknown[] = [];
    for (const idx of this._selection) {
      result.push(this._data[idx] ?? null);
    }
    return result;
  }

  /**
   * Create a ColumnVector from a typed Float64Array (for numeric columns).
   */
  static fromFloat64Array(name: string, arr: Float64Array): ColumnVector {
    const data: unknown[] = new Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      data[i] = arr[i];
    }
    return new ColumnVector(name, "float", data);
  }

  /**
   * Create a ColumnVector from a string array.
   */
  static fromStrings(name: string, arr: string[]): ColumnVector {
    return new ColumnVector(name, "text", arr);
  }
}

/**
 * Columnar batch: a collection of named columns, each represented as a plain
 * JS array. All columns share the same logical length.
 */
export class Batch {
  private _columns: Map<string, unknown[]>;
  private _length: number;

  constructor(data: Map<string, unknown[]>, length: number) {
    this._columns = data;
    this._length = length;
  }

  /**
   * Build a Batch from an array of row objects.
   * If schema is provided, columns are created in schema order; otherwise
   * columns are derived from the union of all row keys.
   */
  static fromRows(
    rows: Record<string, unknown>[],
    schema?: Map<string, DataType>,
  ): Batch {
    if (rows.length === 0) {
      const cols = new Map<string, unknown[]>();
      if (schema) {
        for (const colName of schema.keys()) {
          cols.set(colName, []);
        }
      }
      return new Batch(cols, 0);
    }

    // Determine column names: prefer schema order, fall back to row-key union
    let columnNames: string[];
    if (schema) {
      columnNames = [...schema.keys()];
    } else {
      const nameSet = new Set<string>();
      for (const row of rows) {
        for (const key of Object.keys(row)) {
          nameSet.add(key);
        }
      }
      columnNames = [...nameSet];
    }

    const cols = new Map<string, unknown[]>();
    for (const name of columnNames) {
      const arr: unknown[] = [];
      for (const row of rows) {
        arr.push(row[name] ?? null);
      }
      cols.set(name, arr);
    }

    return new Batch(cols, rows.length);
  }

  /** Number of rows in this batch. */
  get length(): number {
    return this._length;
  }

  /** Ordered column names. */
  get columnNames(): string[] {
    return [...this._columns.keys()];
  }

  /**
   * Return the column array for the given name.
   * Throws if the column does not exist.
   */
  column(name: string): unknown[] {
    const col = this._columns.get(name);
    if (col === undefined) {
      throw new Error(`Column "${name}" not found in batch`);
    }
    return col;
  }

  /**
   * Return the column array for the given name, or null if missing.
   */
  getColumn(name: string): unknown[] | null {
    return this._columns.get(name) ?? null;
  }

  /**
   * Convert the columnar batch back into an array of row objects.
   */
  toRows(): Record<string, unknown>[] {
    const rows: Record<string, unknown>[] = [];
    const names = this.columnNames;
    for (let i = 0; i < this._length; i++) {
      const row: Record<string, unknown> = {};
      for (const name of names) {
        const col = this._columns.get(name)!;
        row[name] = col[i] ?? null;
      }
      rows.push(row);
    }
    return rows;
  }

  /**
   * Return a new Batch containing rows [offset, offset + length).
   */
  slice(offset: number, length: number): Batch {
    const start = Math.max(0, offset);
    const end = Math.min(this._length, start + length);
    const actualLength = Math.max(0, end - start);

    const cols = new Map<string, unknown[]>();
    for (const [name, col] of this._columns) {
      cols.set(name, col.slice(start, start + actualLength));
    }

    return new Batch(cols, actualLength);
  }

  /**
   * Return a new Batch with only the specified columns, optionally renaming
   * them via the aliases map (original -> alias).
   */
  selectColumns(columns: string[], aliases?: Map<string, string>): Batch {
    const cols = new Map<string, unknown[]>();
    for (const name of columns) {
      const col = this._columns.get(name);
      if (col === undefined) {
        throw new Error(`Column "${name}" not found in batch`);
      }
      const outputName = aliases?.get(name) ?? name;
      cols.set(outputName, col);
    }
    return new Batch(cols, this._length);
  }

  /**
   * Return a new Batch containing only the rows at the given indices.
   */
  take(indices: number[]): Batch {
    const cols = new Map<string, unknown[]>();
    for (const [name, col] of this._columns) {
      const taken: unknown[] = [];
      for (const idx of indices) {
        taken.push(col[idx] ?? null);
      }
      cols.set(name, taken);
    }
    return new Batch(cols, indices.length);
  }

  /**
   * Add a new column to this batch (mutating).
   * The array must have the same length as existing columns.
   */
  addColumn(name: string, data: unknown[]): void {
    if (data.length !== this._length && this._length > 0) {
      throw new Error(
        `Column "${name}" length (${String(data.length)}) does not match batch length (${String(this._length)})`,
      );
    }
    this._columns.set(name, data);
    if (this._length === 0 && data.length > 0) {
      this._length = data.length;
    }
  }

  /**
   * Remove a column from this batch (mutating).
   */
  removeColumn(name: string): void {
    this._columns.delete(name);
  }

  /**
   * Rename a column (mutating).
   */
  renameColumn(oldName: string, newName: string): void {
    const col = this._columns.get(oldName);
    if (col === undefined) {
      throw new Error(`Column "${oldName}" not found in batch`);
    }
    // Preserve insertion order by rebuilding the map
    const newCols = new Map<string, unknown[]>();
    for (const [name, data] of this._columns) {
      if (name === oldName) {
        newCols.set(newName, data);
      } else {
        newCols.set(name, data);
      }
    }
    this._columns = newCols;
  }

  /**
   * Return true if this batch has a column with the given name.
   */
  hasColumn(name: string): boolean {
    return this._columns.has(name);
  }

  /**
   * Concatenate rows from another batch (same schema assumed).
   */
  concat(other: Batch): Batch {
    const cols = new Map<string, unknown[]>();
    const allNames = new Set([...this.columnNames, ...other.columnNames]);
    for (const name of allNames) {
      const thisCol =
        this._columns.get(name) ?? new Array<unknown>(this._length).fill(null);
      const otherCol =
        other.getColumn(name) ?? new Array<unknown>(other.length).fill(null);
      cols.set(name, [...thisCol, ...otherCol]);
    }
    return new Batch(cols, this._length + other.length);
  }

  /**
   * Return a new empty Batch with the same column names.
   */
  empty(): Batch {
    const cols = new Map<string, unknown[]>();
    for (const name of this._columns.keys()) {
      cols.set(name, []);
    }
    return new Batch(cols, 0);
  }

  /**
   * Build a Batch from ColumnVector instances.
   */
  static fromColumnVectors(vectors: ColumnVector[]): Batch {
    if (vectors.length === 0) return new Batch(new Map(), 0);
    const length = vectors[0]!.length;
    const cols = new Map<string, unknown[]>();
    for (const vec of vectors) {
      cols.set(vec.name, vec.toArray());
    }
    return new Batch(cols, length);
  }

  /**
   * Convert this Batch to an array of ColumnVector instances.
   */
  toColumnVectors(): ColumnVector[] {
    const vectors: ColumnVector[] = [];
    for (const [name, data] of this._columns) {
      // Infer data type from the first non-null value
      let type: DataType = "unknown";
      for (const v of data) {
        if (v !== null && v !== undefined) {
          if (typeof v === "number") {
            type = Number.isInteger(v) ? "integer" : "float";
          } else if (typeof v === "string") {
            type = "text";
          } else if (typeof v === "boolean") {
            type = "boolean";
          } else if (v instanceof Date) {
            type = "timestamp";
          }
          break;
        }
      }
      vectors.push(new ColumnVector(name, type, data));
    }
    return vectors;
  }

  /**
   * Build an ArrowSchema from this Batch's column types.
   */
  inferSchema(): ArrowSchema {
    const fields: ArrowField[] = [];
    for (const [name, data] of this._columns) {
      let type: DataType = "unknown";
      let nullable = false;
      for (const v of data) {
        if (v === null || v === undefined) {
          nullable = true;
        } else if (type === "unknown") {
          if (typeof v === "number") {
            type = Number.isInteger(v) ? "integer" : "float";
          } else if (typeof v === "string") {
            type = "text";
          } else if (typeof v === "boolean") {
            type = "boolean";
          } else if (v instanceof Date) {
            type = "timestamp";
          }
        }
      }
      fields.push(arrowField(name, type, nullable));
    }
    return arrowSchema(fields);
  }
}
