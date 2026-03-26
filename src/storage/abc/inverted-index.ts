//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- InvertedIndex abstract interface
// 1:1 port of uqa/storage/abc/inverted_index.py
//
// An inverted index maps (field, term) pairs to posting lists and
// maintains per-document token lengths and corpus statistics for scoring.
// Concrete implementations include in-memory and SQLite-backed stores.

import type { DocId, FieldName, IndexStats } from "../../core/types.js";
import type { PostingEntry } from "../../core/types.js";
import type { PostingList } from "../../core/posting-list.js";

/**
 * Minimal analyzer interface -- full implementation in analysis module.
 * Analyzers tokenize text into a list of terms for indexing and search.
 */
export interface AnalyzerLike {
  analyze(text: string): string[];
}

/**
 * Metadata returned from indexing a document.
 *
 * Used by the persistence layer to store posting entries and per-field
 * token lengths without duplicating tokenization logic.
 */
export interface IndexedTerms {
  /** Per-field token counts after analysis. */
  readonly fieldLengths: Record<string, number>;
  /**
   * Posting data keyed by "${field}\0${term}" -> positions.
   * Each entry maps a (field, term) pair to the list of token positions
   * where the term occurred in the document.
   */
  readonly postings: Map<string, readonly number[]>;
}

/**
 * Abstract interface for inverted index backends.
 *
 * An inverted index maps (field, term) pairs to posting lists and
 * maintains per-document token lengths and corpus statistics for scoring.
 * Concrete implementations include in-memory and SQLite-backed stores.
 */
export abstract class InvertedIndex {
  // -- Indexing ---------------------------------------------------------------

  /**
   * Index a document by tokenizing each field.
   *
   * Returns an IndexedTerms with per-field lengths and posting data
   * so the caller can persist them without re-tokenizing.
   */
  abstract addDocument(docId: DocId, fields: Record<FieldName, string>): IndexedTerms;

  /** Add a single posting entry directly (for catalog restore). */
  abstract addPosting(field: string, term: string, entry: PostingEntry): void;

  /** Set per-field token lengths for a document (for catalog restore). */
  abstract setDocLength(docId: DocId, lengths: Record<FieldName, number>): void;

  /** Set the indexed document count (for catalog restore). */
  abstract setDocCount(count: number): void;

  /** Accumulate total token length for a field (for catalog restore). */
  abstract addTotalLength(field: FieldName, length: number): void;

  /** Remove all entries for a document from the index. */
  abstract removeDocument(docId: DocId): void;

  /** Remove all indexed data. */
  abstract clear(): void;

  // -- Querying ---------------------------------------------------------------

  /** Return the posting list for a specific (field, term) pair. */
  abstract getPostingList(field: string, term: string): PostingList;

  /** Return the posting list matching term across any field. */
  abstract getPostingListAnyField(term: string): PostingList;

  /** Return the document frequency for a (field, term) pair. */
  abstract docFreq(field: string, term: string): number;

  /** Return the document frequency across all fields. */
  abstract docFreqAnyField(term: string): number;

  /** Return the token count for docId in field. */
  abstract getDocLength(docId: DocId, field: FieldName): number;

  /** Return doc lengths for multiple docIds in a single call. */
  abstract getDocLengthsBulk(docIds: DocId[], field: FieldName): Map<DocId, number>;

  /** Return the total document length across all fields. */
  abstract getTotalDocLength(docId: DocId): number;

  /** Return term frequency for a specific doc in a specific field. */
  abstract getTermFreq(docId: DocId, field: string, term: string): number;

  /** Return term frequencies for multiple docIds in a single call. */
  abstract getTermFreqsBulk(
    docIds: DocId[],
    field: string,
    term: string,
  ): Map<DocId, number>;

  /** Return total term frequency for a doc across all fields. */
  abstract getTotalTermFreq(docId: DocId, term: string): number;

  // -- Analyzers --------------------------------------------------------------

  /** Return the default analyzer. */
  abstract get analyzer(): AnalyzerLike;

  /** Return the per-field index-time analyzer overrides. */
  abstract get fieldAnalyzers(): Record<string, AnalyzerLike>;

  /**
   * Set a per-field analyzer override.
   *
   * phase controls which phase the analyzer applies to:
   * "index" for indexing only, "search" for search only,
   * or "both" (default) for both phases.
   */
  abstract setFieldAnalyzer(
    field: string,
    analyzer: AnalyzerLike,
    phase?: "index" | "search" | "both",
  ): void;

  /** Return the index-time analyzer for a specific field. */
  abstract getFieldAnalyzer(field: string): AnalyzerLike;

  /**
   * Return the search-time analyzer for a specific field.
   * Falls back to the index-time analyzer, then the default analyzer.
   */
  abstract getSearchAnalyzer(field: string): AnalyzerLike;

  // -- Bulk operations -------------------------------------------------------

  /** Add multiple documents in a single batch. */
  abstract addDocuments(docs: Array<[DocId, Record<FieldName, string>]>): void;

  /** Remove multiple documents in a single batch. */
  abstract removeDocuments(docIds: DocId[]): void;

  // -- Term enumeration ------------------------------------------------------

  /** Return all distinct terms indexed under a specific field. */
  abstract terms(field: string): Iterable<string>;

  /** Return all (field, term) pairs in the index. */
  abstract allTerms(): Iterable<[string, string]>;

  /** Return all field names that have been indexed. */
  abstract fieldNames(): Iterable<string>;

  // -- Existence checks ------------------------------------------------------

  /** Return true if the given (field, term) pair has any postings. */
  abstract hasTerm(field: string, term: string): boolean;

  /** Return true if the given document ID is indexed. */
  abstract hasDoc(docId: DocId): boolean;

  // -- Document length statistics -------------------------------------------

  /** Return the average document length for a field. */
  abstract avgDocLength(field: FieldName): number;

  /** Return the total number of indexed documents. */
  abstract totalDocCount(): number;

  /** Return the total token length across all documents for a field. */
  abstract totalFieldLength(field: FieldName): number;

  // -- Position access -------------------------------------------------------

  /** Return the token positions for a term in a specific document and field. */
  abstract getPositions(docId: DocId, field: string, term: string): readonly number[];

  // -- Statistics -------------------------------------------------------------

  /** Return corpus-level statistics for scoring. */
  abstract get stats(): IndexStats;
}
