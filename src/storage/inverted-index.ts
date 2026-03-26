//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- MemoryInvertedIndex
// 1:1 port of uqa/storage/inverted_index.py

import type { DocId, FieldName, PostingEntry } from "../core/types.js";
import { IndexStats, createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";
import type { AnalyzerLike, IndexedTerms } from "./abc/inverted-index.js";
import { InvertedIndex } from "./abc/inverted-index.js";

// Default analyzer: simple whitespace + lowercase split
const DEFAULT_ANALYZER: AnalyzerLike = {
  analyze(text: string): string[] {
    return text
      .toLowerCase()
      .split(/\s+/)
      .filter((t) => t.length > 0);
  },
};

export class MemoryInvertedIndex extends InvertedIndex {
  private _analyzer: AnalyzerLike;
  private _indexFieldAnalyzers: Map<string, AnalyzerLike>;
  private _searchFieldAnalyzers: Map<string, AnalyzerLike>;
  // (field, term) -> (docId -> PostingEntry)
  private _index: Map<string, Map<DocId, PostingEntry>>;
  // docId -> set of keys in _index
  private _docTerms: Map<DocId, Set<string>>;
  // docId -> { field -> length }
  private _docLengths: Map<DocId, Map<FieldName, number>>;
  private _docCount: number;
  private _totalLength: Map<FieldName, number>;
  private _cachedStats: IndexStats | null;
  // term -> list of keys (field\0term format)
  private _termToKeys: Map<string, string[]>;

  constructor(
    analyzer?: AnalyzerLike | null,
    fieldAnalyzers?: Record<string, AnalyzerLike> | null,
  ) {
    super();
    this._analyzer = analyzer ?? DEFAULT_ANALYZER;
    this._indexFieldAnalyzers = new Map();
    this._searchFieldAnalyzers = new Map();
    if (fieldAnalyzers) {
      for (const [field, a] of Object.entries(fieldAnalyzers)) {
        this._indexFieldAnalyzers.set(field, a);
      }
    }
    this._index = new Map();
    this._docTerms = new Map();
    this._docLengths = new Map();
    this._docCount = 0;
    this._totalLength = new Map();
    this._cachedStats = null;
    this._termToKeys = new Map();
  }

  private _key(field: string, term: string): string {
    return `${field}\0${term}`;
  }

  private _parseKey(key: string): [string, string] {
    const idx = key.indexOf("\0");
    return [key.substring(0, idx), key.substring(idx + 1)];
  }

  // -- Analyzer methods -------------------------------------------------------

  get analyzer(): AnalyzerLike {
    return this._analyzer;
  }

  get fieldAnalyzers(): Record<string, AnalyzerLike> {
    const result: Record<string, AnalyzerLike> = {};
    for (const [k, v] of this._indexFieldAnalyzers) {
      result[k] = v;
    }
    return result;
  }

  setFieldAnalyzer(
    field: string,
    analyzer: AnalyzerLike,
    phase: "index" | "search" | "both" = "both",
  ): void {
    if (phase === "index" || phase === "both") {
      this._indexFieldAnalyzers.set(field, analyzer);
    }
    if (phase === "search" || phase === "both") {
      this._searchFieldAnalyzers.set(field, analyzer);
    }
  }

  getFieldAnalyzer(field: string): AnalyzerLike {
    return this._indexFieldAnalyzers.get(field) ?? this._analyzer;
  }

  getSearchAnalyzer(field: string): AnalyzerLike {
    return (
      this._searchFieldAnalyzers.get(field) ??
      this._indexFieldAnalyzers.get(field) ??
      this._analyzer
    );
  }

  // -- Indexing ----------------------------------------------------------------

  addDocument(docId: DocId, fields: Record<FieldName, string>): IndexedTerms {
    this._cachedStats = null;
    this._docCount++;
    this._docLengths.set(docId, new Map());

    let docTermSet = this._docTerms.get(docId);
    if (!docTermSet) {
      docTermSet = new Set();
      this._docTerms.set(docId, docTermSet);
    }

    const resultFieldLengths: Record<string, number> = {};
    const resultPostings = new Map<string, readonly number[]>();

    for (const [fieldName, text] of Object.entries(fields)) {
      const fieldAnalyzer = this.getFieldAnalyzer(fieldName);
      const tokens = fieldAnalyzer.analyze(text);

      const lengths = this._docLengths.get(docId)!;
      lengths.set(fieldName, tokens.length);
      resultFieldLengths[fieldName] = tokens.length;

      const prev = this._totalLength.get(fieldName) ?? 0;
      this._totalLength.set(fieldName, prev + tokens.length);

      // Build term -> positions map
      const termPositions = new Map<string, number[]>();
      for (let pos = 0; pos < tokens.length; pos++) {
        const token = tokens[pos]!;
        let positions = termPositions.get(token);
        if (!positions) {
          positions = [];
          termPositions.set(token, positions);
        }
        positions.push(pos);
      }

      for (const [term, positions] of termPositions) {
        const key = this._key(fieldName, term);

        let inner = this._index.get(key);
        if (!inner) {
          inner = new Map();
          this._index.set(key, inner);
          let termKeys = this._termToKeys.get(term);
          if (!termKeys) {
            termKeys = [];
            this._termToKeys.set(term, termKeys);
          }
          termKeys.push(key);
        }

        const entry: PostingEntry = {
          docId,
          payload: createPayload({ positions, score: 0.0 }),
        };
        inner.set(docId, entry);
        docTermSet.add(key);

        resultPostings.set(key, positions);
      }
    }

    return { fieldLengths: resultFieldLengths, postings: resultPostings };
  }

  addPosting(field: string, term: string, entry: PostingEntry): void {
    this._cachedStats = null;
    const key = this._key(field, term);

    let inner = this._index.get(key);
    if (!inner) {
      inner = new Map();
      this._index.set(key, inner);
      let termKeys = this._termToKeys.get(term);
      if (!termKeys) {
        termKeys = [];
        this._termToKeys.set(term, termKeys);
      }
      termKeys.push(key);
    }

    inner.set(entry.docId, entry);

    let docTermSet = this._docTerms.get(entry.docId);
    if (!docTermSet) {
      docTermSet = new Set();
      this._docTerms.set(entry.docId, docTermSet);
    }
    docTermSet.add(key);
  }

  setDocLength(docId: DocId, lengths: Record<FieldName, number>): void {
    const m = new Map<FieldName, number>();
    for (const [k, v] of Object.entries(lengths)) {
      m.set(k, v);
    }
    this._docLengths.set(docId, m);
  }

  setDocCount(count: number): void {
    this._cachedStats = null;
    this._docCount = count;
  }

  addTotalLength(field: FieldName, length: number): void {
    this._cachedStats = null;
    const prev = this._totalLength.get(field) ?? 0;
    this._totalLength.set(field, prev + length);
  }

  removeDocument(docId: DocId): void {
    this._cachedStats = null;
    const keys = this._docTerms.get(docId);
    if (!keys) return;

    for (const key of keys) {
      const inner = this._index.get(key);
      if (inner) {
        inner.delete(docId);
        if (inner.size === 0) {
          this._index.delete(key);
          const [, term] = this._parseKey(key);
          const termKeys = this._termToKeys.get(term);
          if (termKeys) {
            const idx = termKeys.indexOf(key);
            if (idx !== -1) termKeys.splice(idx, 1);
            if (termKeys.length === 0) this._termToKeys.delete(term);
          }
        }
      }
    }

    this._docTerms.delete(docId);

    const lengths = this._docLengths.get(docId);
    if (lengths) {
      for (const [fld, length] of lengths) {
        const prev = this._totalLength.get(fld) ?? 0;
        this._totalLength.set(fld, prev - length);
      }
      this._docLengths.delete(docId);
      this._docCount--;
    }
  }

  clear(): void {
    this._index.clear();
    this._docTerms.clear();
    this._docLengths.clear();
    this._termToKeys.clear();
    this._cachedStats = null;
    this._docCount = 0;
    this._totalLength.clear();
  }

  // -- Querying ---------------------------------------------------------------

  getPostingList(field: string, term: string): PostingList {
    const inner = this._index.get(this._key(field, term));
    if (!inner) return new PostingList();
    const entries = [...inner.values()].sort((a, b) => a.docId - b.docId);
    return PostingList.fromSorted(entries);
  }

  getPostingListAnyField(term: string): PostingList {
    const keys = this._termToKeys.get(term);
    if (!keys || keys.length === 0) return new PostingList();

    const seen = new Set<DocId>();
    const allEntries: PostingEntry[] = [];
    for (const key of keys) {
      const inner = this._index.get(key);
      if (!inner) continue;
      for (const [docId, entry] of inner) {
        if (!seen.has(docId)) {
          seen.add(docId);
          allEntries.push(entry);
        }
      }
    }
    allEntries.sort((a, b) => a.docId - b.docId);
    return PostingList.fromSorted(allEntries);
  }

  docFreq(field: string, term: string): number {
    const inner = this._index.get(this._key(field, term));
    return inner ? inner.size : 0;
  }

  docFreqAnyField(term: string): number {
    const keys = this._termToKeys.get(term);
    if (!keys || keys.length === 0) return 0;
    const docIds = new Set<DocId>();
    for (const key of keys) {
      const inner = this._index.get(key);
      if (inner) {
        for (const docId of inner.keys()) {
          docIds.add(docId);
        }
      }
    }
    return docIds.size;
  }

  getDocLength(docId: DocId, field: FieldName): number {
    const lengths = this._docLengths.get(docId);
    if (!lengths) return 0;
    return lengths.get(field) ?? 0;
  }

  getDocLengthsBulk(docIds: DocId[], field: FieldName): Map<DocId, number> {
    const result = new Map<DocId, number>();
    for (const docId of docIds) {
      result.set(docId, this.getDocLength(docId, field));
    }
    return result;
  }

  getTotalDocLength(docId: DocId): number {
    const lengths = this._docLengths.get(docId);
    if (!lengths) return 0;
    let total = 0;
    for (const v of lengths.values()) {
      total += v;
    }
    return total;
  }

  getTermFreq(docId: DocId, field: string, term: string): number {
    const inner = this._index.get(this._key(field, term));
    if (!inner) return 0;
    const entry = inner.get(docId);
    if (!entry) return 0;
    return entry.payload.positions.length;
  }

  getTermFreqsBulk(docIds: DocId[], field: string, term: string): Map<DocId, number> {
    const inner = this._index.get(this._key(field, term));
    const result = new Map<DocId, number>();
    for (const docId of docIds) {
      if (inner) {
        const entry = inner.get(docId);
        result.set(docId, entry ? entry.payload.positions.length : 0);
      } else {
        result.set(docId, 0);
      }
    }
    return result;
  }

  getTotalTermFreq(docId: DocId, term: string): number {
    const keys = this._termToKeys.get(term);
    if (!keys) return 0;
    let total = 0;
    for (const key of keys) {
      const inner = this._index.get(key);
      if (inner) {
        const entry = inner.get(docId);
        if (entry) {
          total += entry.payload.positions.length;
        }
      }
    }
    return total;
  }

  // -- Bulk operations -------------------------------------------------------

  addDocuments(docs: Array<[DocId, Record<FieldName, string>]>): void {
    for (const [docId, fields] of docs) {
      this.addDocument(docId, fields);
    }
  }

  removeDocuments(docIds: DocId[]): void {
    for (const docId of docIds) {
      this.removeDocument(docId);
    }
  }

  // -- Term enumeration ------------------------------------------------------

  *terms(field: string): Generator<string> {
    const prefix = this._key(field, "");
    for (const key of this._index.keys()) {
      if (key.startsWith(prefix)) {
        const [, term] = this._parseKey(key);
        yield term;
      }
    }
  }

  *allTerms(): Generator<[string, string]> {
    for (const key of this._index.keys()) {
      yield this._parseKey(key);
    }
  }

  *fieldNames(): Generator<string> {
    const fields = new Set<string>();
    for (const key of this._index.keys()) {
      const [field] = this._parseKey(key);
      if (!fields.has(field)) {
        fields.add(field);
        yield field;
      }
    }
  }

  // -- Existence checks ------------------------------------------------------

  hasTerm(field: string, term: string): boolean {
    return this._index.has(this._key(field, term));
  }

  hasDoc(docId: DocId): boolean {
    return this._docTerms.has(docId);
  }

  // -- Document length statistics -------------------------------------------

  avgDocLength(field: FieldName): number {
    if (this._docCount === 0) return 0;
    const total = this._totalLength.get(field) ?? 0;
    return total / this._docCount;
  }

  totalDocCount(): number {
    return this._docCount;
  }

  totalFieldLength(field: FieldName): number {
    return this._totalLength.get(field) ?? 0;
  }

  // -- Position access -------------------------------------------------------

  getPositions(docId: DocId, field: string, term: string): readonly number[] {
    const inner = this._index.get(this._key(field, term));
    if (!inner) return [];
    const entry = inner.get(docId);
    if (!entry) return [];
    return entry.payload.positions;
  }

  // -- Statistics -------------------------------------------------------------

  get stats(): IndexStats {
    if (this._cachedStats !== null) return this._cachedStats;

    let totalLen = 0;
    for (const v of this._totalLength.values()) {
      totalLen += v;
    }
    const avgDocLength = this._docCount > 0 ? totalLen / this._docCount : 0.0;

    const s = new IndexStats(this._docCount, avgDocLength);

    for (const [key, inner] of this._index) {
      const [field, term] = this._parseKey(key);
      s.setDocFreq(field, term, inner.size);
    }

    this._cachedStats = s;
    return s;
  }
}
