# History

## 0.2.0 (2026-04-01)

SQL faceted search and search result highlighting.

### Features
- **Search result highlighting (`uqa_highlight()`)**: New SQL scalar function that highlights matched query terms in text fields. Supports custom markup tags, analyzer-aware stemming match, and fragment extraction with word-boundary snapping.
  - `uqa_highlight(field, 'query')` -- highlight with default `<b>` tags
  - `uqa_highlight(field, 'query', '<em>', '</em>')` -- custom tags
  - `uqa_highlight(field, 'query', '<b>', '</b>', max_fragments, fragment_size)` -- snippet extraction
  - Analyzer-aware matching: uses the table's GIN index analyzer for stemming-consistent highlighting (e.g., searching for "run" highlights "running")
  - Fragment extraction clusters nearby matches, selects densest clusters, and snaps to word boundaries with ellipsis markers
- **Faceted search (`uqa_facets()`)**: New SQL function that computes facet value counts over search results.
  - `SELECT uqa_facets(field) FROM table` -- returns `facet_value | facet_count` rows
  - `SELECT uqa_facets(field1, field2) FROM table` -- multi-field facets with `facet_field | facet_value | facet_count`
  - Respects WHERE clause filtering (including `@@` and `text_match` predicates)
  - Results sorted alphabetically by facet value
- **Search highlighting module** (`src/search/highlight.ts`): Standalone `highlight()` and `extractQueryTerms()` utilities exported from the public API for programmatic use outside SQL.

### Tests
- 2,912 tests across 111 test files
- Added 35 tests in `tests/sql/facets-highlight.test.ts` covering highlight utility, query term extraction, SQL `uqa_highlight()`, and SQL `uqa_facets()`

## 0.1.5 (2026-03-29)

Foreign Data Wrapper integration with the SQL execution engine.

### Features
- **FDW handler dispatch in SQL compiler**: `SELECT` queries on foreign tables now route through `DuckDBFDWHandler` or `ArrowFDWHandler` via the SQL compiler's FROM-clause resolution. Previously, `CREATE FOREIGN SERVER` and `CREATE FOREIGN TABLE` stored metadata only; `SELECT * FROM foreign_table` raised "Table does not exist".
- **FDW handler lifecycle management**: Handlers are cached per server name and reused across queries. `DROP SERVER` closes the cached handler. `Engine.close()` releases all FDW handlers.
- **DuckDB FDW**: Inline JSON data path fully functional through SQL. Predicate pushdown, column projection, and LIMIT work end-to-end.
- **Arrow FDW**: Inline JSON data and base64 Arrow IPC buffer paths fully functional through SQL. Predicate pushdown, column projection, and LIMIT work end-to-end.
- **Mixed queries**: Foreign tables can be joined with local tables, used in subqueries, and combined across different FDW types (DuckDB + Arrow).

- **Explicit FTS index via CREATE INDEX USING gin**: Full-text search inverted indexes are no longer created automatically on every TEXT column. FTS indexing now requires an explicit `CREATE INDEX ... USING gin (column)` statement, matching PostgreSQL semantics. Supports multi-column indexes and optional analyzer specification via `WITH (analyzer = 'name')`. Existing rows are backfilled at index creation time. `DROP INDEX` removes the FTS index and clears indexed data.

### Bug Fixes
- **DROP FOREIGN TABLE**: Fixed AST parsing that silently failed to extract the table name. libpg-query wraps `DROP FOREIGN TABLE` objects in a `List` node (`{"List": {"items": [...]}}`) unlike `DROP SERVER` which uses flat `String` nodes. The previous code did not unwrap the `List`, causing `DROP FOREIGN TABLE` to be a no-op.
- **Removed automatic FTS indexing**: Previously all TEXT/VARCHAR columns were indexed into the inverted index on every INSERT/UPDATE, wasting memory and CPU for columns that were never searched. Now only explicitly indexed columns are maintained.

### Tests
- 2,877 tests across 110 test files
- Added 33 FDW integration tests (`tests/fdw/fdw-integration.test.ts`) exercising the full `Engine.sql()` -> `SQLCompiler` path
- Updated 7 test files to use explicit `CREATE INDEX ... USING gin` before FTS queries

## 0.1.4 (2026-03-29)

### Bug Fixes
- **StandardTokenizer Unicode support**: Fixed `StandardTokenizer` regex from `/\w+/gu` to `/[\p{L}\p{N}_]+/gu` so it matches Unicode word characters (Hangul, CJK, etc.). JavaScript's `\w` only matches ASCII `[a-zA-Z0-9_]` even with the `u` flag, unlike Python's `\w` which matches all Unicode letters. This caused Korean text like "검색 기능" to produce zero tokens instead of `["검색", "기능"]`.

### Tests
- 2,844 tests across 109 test files
- Added Korean and CJK tokenization tests for `standardCJKAnalyzer`

## 0.1.3 (2026-03-28)

### Bug Fixes
- **set_table_analyzer phase argument**: Fixed SQL `set_table_analyzer()` passing `{ phase: "both" }` object instead of plain string `"both"` to `Engine.setTableAnalyzer()`.
- **Engine.setTableAnalyzer table lookup**: Fixed `setTableAnalyzer` to use `getTable()` which checks both `Engine._tables` and `SQLCompiler.tables`, so tables created via `CREATE TABLE` SQL are found.

## 0.1.2 (2026-03-28)

Bug fix: UQA extension functions in SQL WHERE clause now work correctly.

### Bug Fixes
- **WHERE clause UQA function dispatch**: `text_match`, `multi_field_match`, `fuse_log_odds`, `fuse_prob_and`, `fuse_prob_or`, `fuse_prob_not`, `knn_match`, `bayesian_knn_match`, `traverse_match`, `sparse_threshold`, and all other UQA extension functions now correctly compile to posting-list operators when used in WHERE clauses. Previously, these functions fell through to the ExprEvaluator scalar path, which does not know how to evaluate them, resulting in empty results or errors.
- **A_Const value extraction**: Fixed extraction of string and integer values from libpg-query v17 AST nodes. The parser produces nested structures like `{sval: {sval: "text"}}` and `{ival: {ival: 42}}`, but `extractConstValue()` was returning the inner object instead of unwrapping to the scalar value.

### New Tests
- `tests/sql/uqa-functions.test.ts`: 9 tests covering `text_match`, `multi_field_match`, `fuse_log_odds`, `fuse_prob_and`, `fuse_prob_or`, `fuse_prob_not`, `sparse_threshold`, `create_analyzer`/`list_analyzers`, `create_graph`/`drop_graph`.

## 0.1.1 (2026-03-28)

SQL FROM-clause table functions for graph and analyzer management.

### SQL Functions Added
- `SELECT * FROM create_graph('name')` — create a named graph
- `SELECT * FROM drop_graph('name')` — drop a named graph
- `SELECT * FROM create_analyzer('name', 'config_json')` — create a custom text analyzer
- `SELECT * FROM drop_analyzer('name')` — drop a custom analyzer
- `SELECT * FROM list_analyzers()` — list all registered analyzers
- `SELECT * FROM set_table_analyzer('table', 'field', 'analyzer'[, 'phase'])` — assign analyzer to table field
- `SELECT * FROM graph_add_vertex(id, 'label', 'table'[, 'key=val,...'])` — add graph vertex via SQL
- `SELECT * FROM graph_add_edge(eid, src, tgt, 'label', 'table'[, 'key=val,...'])` — add graph edge via SQL
- `SELECT * FROM build_grid_graph('table', rows, cols, 'label')` — build 4-connected grid graph
- `SELECT * FROM cypher('graph', $$ MATCH ... RETURN ... $$) AS (col agtype)` — Apache AGE compatible Cypher execution

## 0.1.0 (2026-03-28)

Initial release. Complete 1:1 port of Python UQA v0.22.1 to TypeScript for browser execution.

### Core
- PostingList and GeneralizedPostingList with full Boolean algebra (5 axioms verified)
- 14 predicate types (Equals, NotEquals, GreaterThan, LessThan, Between, Like, ILike, etc.)
- HierarchicalDocument with path evaluation and implicit array wildcards
- Cross-paradigm functors (Graph-to-Relational, Relational-to-Graph, Text-to-Vector)

### Storage
- MemoryDocumentStore, MemoryInvertedIndex, FlatVectorIndex, IVFIndex (HNSW)
- SpatialIndex (Haversine + bounding box)
- BlockMaxIndex for WAND optimization
- SQLite persistence via sql.js WASM (Catalog, SQLiteDocumentStore, SQLiteInvertedIndex, SQLiteGraphStore)
- BTreeIndex, IndexManager

### Text Analysis
- 6 tokenizers (Whitespace, Standard, Letter, NGram, Pattern, Keyword)
- 8 token filters (LowerCase, StopWord, PorterStem, ASCIIFolding, Synonym, NGram, EdgeNGram, Length)
- 3 char filters (HTMLStrip, Mapping, PatternReplace)
- Composable Analyzer pipeline with serialization

### Scoring
- BM25 with numerically stable formulation
- Bayesian BM25 (three-term log-odds decomposition)
- Vector calibration (likelihood ratio framework)
- WAND, BlockMax WAND, Adaptive WAND
- Fusion WAND, Tightened Fusion WAND
- Multi-field Bayesian scorer
- External prior scorer (recency, authority)
- Parameter learner (online Bayesian calibration)

### Operators
- Primitive: Term, VectorSimilarity, KNN, Filter, SpatialWithin, Facet, Score, IndexScan
- Boolean: Union, Intersect, Complement
- Aggregation: Count, Sum, Avg, Min, Max, Quantile monoids + GroupBy
- Hybrid: HybridTextVector, SemanticFilter, LogOddsFusion, ProbBoolFusion, VectorExclusion, ProbNot, FacetVector, AdaptiveLogOddsFusion
- Hierarchical: PathFilter, PathProject, PathUnnest, PathAggregate, UnifiedFilter
- Advanced: CalibratedVector, MultiField, MultiStage, SparseThreshold, ProgressiveFusion
- Deep: DeepFusion (signal/propagate/conv/pool/dense/flatten/softmax/batchnorm/dropout/attention/embed/global_pool layers)
- Deep Learn: train_model, predict with PoE local learning

### Fusion
- ProbabilisticBoolean (prob_and, prob_or, prob_not)
- LogOddsFusion with scale neutrality
- AttentionFusion, MultiHeadAttentionFusion
- LearnedFusion with state persistence
- AdaptiveLogOddsFusion with signal quality metrics
- QueryFeatureExtractor

### Joins
- Inner, Left/Right/Full Outer, Semi, Anti
- SortMerge, Index (binary search), Cross (Cartesian)
- Cross-paradigm: TextSimilarity, VectorSimilarity, Hybrid, Graph, CrossParadigm

### Graph
- MemoryGraphStore with named graph partitioning and graph algebra (union/intersect/difference)
- GraphPostingList with isomorphism Phi: L_G -> L
- TraverseOperator (BFS with label filtering)
- PatternMatchOperator (backtracking + arc consistency + MRV)
- RegularPathQueryOperator (NFA/DFA simulation)
- WeightedPathQueryOperator
- VertexAggregationOperator
- Centrality: PageRank, HITS, Betweenness (Brandes)
- MessagePassingOperator (k-layer GNN)
- GraphEmbeddingOperator
- Temporal: TemporalTraverseOperator, TemporalFilter, TemporalPatternMatch
- Delta, IncrementalPatternMatcher, VersionedGraphStore
- Cypher: lexer, parser (recursive descent), compiler (MATCH/CREATE/MERGE/SET/DELETE/RETURN/WITH/UNWIND)

### SQL
- SQL compiler via libpg-query WASM (PostgreSQL 17 parser)
- DDL: CREATE/DROP TABLE, VIEW, SEQUENCE, INDEX, ALTER TABLE, FOREIGN SERVER/TABLE
- DML: INSERT (VALUES, SELECT, ON CONFLICT), UPDATE (SET, FROM), DELETE (WHERE, USING)
- DQL: SELECT with JOIN, GROUP BY, HAVING, ORDER BY, LIMIT/OFFSET, DISTINCT, WITH (CTE), RECURSIVE CTE, subqueries, window functions, set operations (UNION/INTERSECT/EXCEPT)
- 50+ UQA extension functions (text_match, knn_match, fuse_log_odds, traverse_match, etc.)
- ExprEvaluator with 80+ scalar SQL functions
- FTS query parser (boolean, phrase, vector, field-scoped)
- Constant folding optimization
- EXPLAIN support

### Execution
- Volcano model (open/next/close) with Batch columnar format
- Physical operators: SeqScan, PostingListScan, Filter, ExprFilter, Project, ExprProject, Sort (external merge), Limit, HashAgg (16-partition spill), Distinct, Window
- All window functions: ROW_NUMBER, RANK, DENSE_RANK, NTILE, LAG, LEAD, PERCENT_RANK, CUME_DIST, NTH_VALUE, FIRST_VALUE, LAST_VALUE

### Planner
- QueryOptimizer with 10 rewrite rules
- CostModel for all operator types
- CardinalityEstimator with histogram/MCV/entropy
- DPccp join enumeration
- PlanExecutor with timing stats and EXPLAIN

### Engine
- Main Engine class with SQL, QueryBuilder, graph management
- Deep learning integration (train/predict)
- Path index management
- Analyzer and scoring parameter management
- Transaction support

### Build
- ESM + UMD browser bundles via Vite
- Full TypeScript type declarations
- Zero-dependency core (runtime deps: bayesian-bm25, libpg-query, sql.js, apache-arrow, @duckdb/duckdb-wasm)

### Tests
- 2,832 tests across 108 test files
- Boolean algebra axioms verified with 100 random trials each
- De Morgan's laws, sorted invariants
- Complete operator, storage, scoring, graph, SQL, execution, planner test coverage
