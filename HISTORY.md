# History

## 0.4.2 (2026-04-09)

Fix DROP TABLE / DROP SCHEMA CASCADE index cleanup, fix DROP TABLE / DROP INDEX List-node unwrapping, and add CREATE INDEX IF NOT EXISTS. DROP TABLE now removes in-memory BTree index metadata, GIN index metadata, and stale foreign key validators from parent tables. DROP SCHEMA CASCADE now performs full per-table cleanup (IndexManager, GIN, BTree, FK validators, sequences, catalog) instead of only deleting the schema entry. CREATE INDEX IF NOT EXISTS is supported for all index types (BTree, GIN). All 3,027 tests pass across 112 test files.

### Bug Fixes

- **DROP TABLE / DROP INDEX List-node unwrapping** (`sql/compiler.ts`): Fixed `_compileDropTable` and `_compileDropIndex` failing to unwrap the `List` AST node from libpg-query's parsed output. The parser produces `{ List: { items: [{ String: { sval: "name" } }] } }` for DROP TABLE/INDEX objects, but the code used `asList(obj)` which only handles plain arrays. This caused DROP TABLE and DROP INDEX to silently skip all cleanup. Added the same `List` unwrapping logic already used by `_compileDropView`.
- **DROP TABLE BTree index cleanup** (`sql/compiler.ts`): DROP TABLE now removes orphaned entries from `engine._btreeIndexes` for the dropped table. Previously, in-memory BTree index metadata survived table deletion.
- **DROP TABLE GIN index cleanup** (`sql/compiler.ts`): DROP TABLE now removes GIN index metadata entries from the compiler's `_indexes` map for the dropped table.
- **DROP TABLE FK validator cleanup** (`sql/compiler.ts`): When a child table with FOREIGN KEY constraints is dropped, stale delete and update validators are now removed from parent tables. FK validators registered on parent tables are tagged with a `_childTable` property for identification during cleanup.
- **DROP SCHEMA CASCADE full cleanup** (`sql/compiler.ts`): DROP SCHEMA CASCADE now iterates all tables in the schema and performs the same cleanup as individual DROP TABLE -- IndexManager (physical BTree indexes), GIN indexes, in-memory BTree metadata, FK validators, sequences, and catalog rows. Previously, only the schema entry was deleted, leaving all per-table resources orphaned. Validation (non-empty schema check, public schema protection) is now performed in the compiler rather than delegated to the store.

### Enhancements

- **CREATE INDEX IF NOT EXISTS** (`sql/compiler.ts`): All index types now support the `IF NOT EXISTS` clause. When the named index already exists, the statement is a silent no-op. The existence check inspects IndexManager, GIN index tracking (`_indexes`), and in-memory BTree metadata (`_btreeIndexes`).

### Internal

- **`_dropTableByName` method** (`sql/compiler.ts`): Extracted from `_compileDropTable` as a shared cleanup method. Used by both `_compileDropTable` and `_compileDropSchema` to ensure identical cleanup behavior.
- **`_indexExists` method** (`sql/compiler.ts`): Checks all index stores (compiler `_indexes`, `IndexManager`, and `_btreeIndexes`) for existence of a named index.
- **`_removeFkValidatorsForChild` / `_validatorReferencesTable` methods** (`sql/compiler.ts`): Walk all remaining tables and purge FK validators that belong to the dropped child table, identified by the `_childTable` tag.

### Tests

- **Total**: 3,027 tests across 112 test files
- Added 12 tests in `tests/sql/pg-compat-bugs.test.ts`:
  - `DropTableCascadeCleanup`: BTree index cleanup, GIN index cleanup, FK validator cleanup on parent
  - `DropSchemaCascadeCleanup`: BTree/GIN/FK cleanup on CASCADE, non-empty schema error, IF EXISTS, nonexistent schema error
  - `CreateIndexIfNotExists`: BTree IF NOT EXISTS, GIN IF NOT EXISTS, duplicate index error

## 0.4.1 (2026-04-09)

Fix named graph vertex and edge persistence. The engine-level graph store was initialized as a pure `MemoryGraphStore` even when backed by SQLite, causing all vertex and edge data added to named graphs to be lost on process exit. Named graph metadata was persisted correctly, creating the illusion that graphs existed while their data was empty. All 3,015 tests pass across 112 test files.

### Bug Fix

- **Named graph store persistence** (`engine.ts`): When `dbPath` is provided, `Engine._graphStore` is now initialized as `SQLiteGraphStore(conn)` instead of `MemoryGraphStore()`. This enables write-through persistence for all named graph mutations (vertices, edges, graph lifecycle). Previously, only per-table graph stores used `SQLiteGraphStore`; the engine-level store for named graphs was always in-memory, losing all vertex and edge data on process termination while the graph name survived -- giving the appearance of an empty graph on restart.

### Tests

- **Total**: 3,015 tests across 112 test files.

## 0.4.0 (2026-04-08)

Standalone property graph SQL functions and per-field analyzer fix.

### Features
- **Standalone property graph SQL functions**: 8 new FROM-clause table functions for graph CRUD and traversal on named graphs: `graph_create_node(graph, label[, props])` creates a vertex with auto-generated ID; `graph_nodes(graph[, label][, filter])` queries vertices with optional label and JSON property filtering; `graph_delete_node(graph, id)` removes a vertex and all incident edges; `graph_create_edge(graph, type, src, tgt[, props])` creates a directed edge with auto-generated ID; `graph_edges(graph, ...)` queries edges in two modes -- graph-wide with optional type/property filters, or per-vertex with direction control (`outgoing`/`incoming`/`both`); `graph_delete_edge(graph, id)` removes an edge; `graph_neighbors(graph, id[, type][, dir][, depth])` performs multi-hop BFS neighbor discovery with depth and path tracking; `graph_traverse(graph, id[, types][, dir][, depth][, strategy])` performs advanced graph traversal with multiple edge types (comma-separated, `ARRAY[...]`, or `NULL` for all), configurable direction, and BFS/DFS strategy selection.
- **`graph_create` / `graph_drop` aliases**: `graph_create('name')` and `graph_drop('name')` as aliases for `create_graph` / `drop_graph`.
- **LATERAL support for table functions**: `_resolveLateralRangeFunction()` enables LATERAL joins with FROM-clause table functions. Correlated column references from the outer query (e.g., `n.node_id`) are resolved via `_resolveCorrelatedColumnRef()` when processing graph table function arguments.
- **1-based auto-increment IDs**: `MemoryGraphStore.nextVertexId()` and `nextEdgeId()` now start at 1 (matching the Python reference implementation).

### Fixes
- **Per-field analyzer in all-field FTS search**: When no field is specified in a full-text search, the engine now iterates each indexed field's own search analyzer instead of using a single default analyzer. This ensures fields indexed with different analyzers (e.g., `standard_cjk` for CJK text) produce correct tokens at search time. Fixed in `TermOperator.execute()`, `_makeTextSearchOp()`, `_makeBayesianWithPriorOp()`, and `compilePhrase()`.

### Tests
- 3,015 tests across 112 test files
- Added 63 tests in `tests/sql/graph-standalone.test.ts` covering node CRUD, edge CRUD, graph-wide and per-vertex edge queries, BFS/DFS traversal, LATERAL correlated column references, `ARRAY[...]` and `NULL` edge type arguments, multi-graph isolation, and full lifecycle workflows

## 0.3.7 (2026-04-05)

Query cancellation mechanism for in-flight queries.

### Features
- **Query cancellation**: New `CancellationToken` class and `QueryCancelled` error (`src/cancel.ts`) provide cooperative cancellation for long-running queries. `Engine.cancel()` triggers cancellation; `Engine.cancelToken` exposes the token for external coordination. `PhysicalOperator` base class gains `cancelToken`, `checkCancelled()`, and `propagateCancelToken()`. `JoinOperator` and `CrossJoinOperator` gain `cancelToken` and `checkCancelled()`. Operators check cancellation in hot loops: SeqScan, PostingListScan, Filter, ExprFilter, Sort, HashAgg, Distinct, Window, InnerJoin, CrossJoin, LeftOuterJoin, FullOuterJoin. SQLCompiler propagates cancel tokens to operator trees before execution. `QueryCancelled` and `CancellationToken` are exported from the package index.

### Tests
- 2,952 tests across 111 test files

## 0.3.6 (2026-04-04)

Register `bayesian_match_with_prior` as a calibrated signal for fusion operators.

### Fixes
- **`bayesian_match_with_prior` in fusion operators**: `bayesian_match_with_prior()` can now be used as a signal inside `fuse_attention()`, `fuse_multihead()`, and `fuse_learned()`. Previously, the calibrated signal dispatcher (`_compileCalibratedSignal`) did not recognize `bayesian_match_with_prior`, causing an "Unknown signal function for fusion" error when composing it with other signals in any fusion meta-function. The function already produced calibrated (0, 1) output via `ExternalPriorScorer`, so this was purely a registration gap.

### Tests
- 2,952 tests across 111 test files
- Added 5 tests in `tests/sql/uqa-functions.test.ts` covering `bayesian_match_with_prior` as a calibrated signal in `fuse_attention` (recency and authority modes), `fuse_multihead`, `fuse_learned`, and dual `bayesian_match_with_prior` signal fusion

## 0.3.5 (2026-04-03)

Primary key point lookup for O(1) single-row access.

### Features
- **PK point lookup**: `WHERE pk = expr` on single-table SELECT now does an O(1) document store lookup instead of a full table scan. Detects integer primary key equality against constants, parameters (`$1`), and scalar subqueries. Also works when the equality is one conjunct of an AND. Correlated column references (`WHERE e.dept_id = d.id`) are correctly excluded. Eliminates the major bottleneck in correlated subqueries where the inner query re-scans the full table for each outer row.

### Tests
- 2,947 tests across 111 test files

## 0.3.4 (2026-04-03)

Hash join for equi-join conditions and EXPLAIN improvements.

### Features
- **Hash join for equi-join conditions**: `_resolveJoin()` now detects equi-join ON conditions (single `=` or AND of `=` with ColumnRef on both sides) and uses an O(n+m) hash join instead of the O(n*m) nested loop. The hash table is built on the right side and probed by the left side, with automatic detection of which equality operand belongs to which side of the JOIN. Supports INNER, LEFT, RIGHT, and FULL joins. Non-equi conditions (e.g., `ON a.x > b.y`) fall back to the nested loop.
- **EXPLAIN reports scan strategy**: `EXPLAIN SELECT` now inspects the WHERE clause for UQA posting-list functions and the FROM clause for JoinExpr structure. Reports `GIN Index Scan using {function}` for tables with UQA WHERE functions, `Hash Join` / `Hash Left Join` for equi-join conditions, and `Nested Loop` for non-equi or cross joins.

### Tests
- 2,947 tests across 111 test files

## 0.3.3 (2026-04-03)

UQA posting-list scan pushed below JOIN for index-driven performance.

### Fixes
- **UQA + JOIN predicate pushdown**: The 0.3.2 fix for UQA WHERE functions with JOIN materialized the full join first (N x M rows), then filtered via a score map. This was correct but O(N*M) -- for large tables the full cartesian product was built before any filtering. Restructured to push the GIN index scan into the FROM phase: `_resolveFromItem()` now checks `_uqaFromFilter` and emits only posting-list matches (with scores) for the target table before the join executes, so the join operates on the small result set instead of the full table.

### Tests
- 2,947 tests across 111 test files

## 0.3.2 (2026-04-03)

UQA WHERE functions (fusion, scoring, retrieval) now work correctly in JOIN queries.

### Fixes
- **UQA WHERE functions with JOIN**: `fuse_log_odds`, `fuse_prob_and`, `fuse_prob_or`, `fuse_attention`, `fuse_learned`, `bayesian_match`, `knn_match`, `staged_retrieval`, `progressive_fusion`, `deep_fusion`, and all other posting-list WHERE functions now work in queries that include `JOIN`. Previously, `_resolveFromTableName()` returned `null` for `JoinExpr` FROM clauses, causing UQA functions to fall through to the generic expression evaluator which threw `Unknown SQL function`.

### Tests
- 2,947 tests across 111 test files
- Added 4 tests in `tests/sql/uqa-functions.test.ts` covering `fuse_log_odds` with INNER JOIN, `bayesian_match` with INNER JOIN, `fuse_prob_or` with LEFT JOIN, and single-table regression check

## 0.3.1 (2026-04-02)

SQLite persistence lifecycle: save, load, and restore engine state from disk.

### Features
- **Async engine initialization (`init()`)**: New `Engine.init()` method handles sql.js WASM initialization, opens or creates the SQLite database file, and restores persisted state. Called automatically by `Engine.create()` when `dbPath` is provided. Idempotent -- safe to call multiple times.
- **Full catalog restore on load**: `loadFromCatalog()` now reconstructs complete `Table` objects with column definitions, documents, sequences, named graphs, and trained models from the SQLite catalog. Previously loaded schemas were discarded (`void name`) and tables were not reconstructed.
- **Document persistence**: `saveToCatalog()` now persists all documents for each non-temporary table alongside the table schema. Previously only schemas were saved.
- **Sequence persistence**: Sequences (`SERIAL` / `NEXTVAL`) are saved to and restored from catalog metadata, preserving auto-increment state across engine restarts.
- **Model persistence**: Trained deep learning models are restored from the `_models` catalog table on load.
- **Disk flush on close**: `Engine.close()` now calls `saveToCatalog()` and writes the in-memory SQLite database to disk via `fs.writeFileSync()` before releasing resources. Previously the in-memory database was discarded on close.

### Fixes
- **Idempotent schema save**: `Catalog.saveTableSchema()` uses `INSERT OR REPLACE` instead of `INSERT`, preventing unique constraint violations when saving the same table schema multiple times.
- **Rollup externals**: Added `fs` and `node:fs` to Vite rollup externals so the Node.js `fs` module is not bundled into browser builds.

## 0.3.0 (2026-04-02)

PostgreSQL compatibility: schemas, sessions, transactions, DDL, query fixes.

### Features
- **Schema namespaces** (`CREATE SCHEMA` / `DROP SCHEMA`): `SchemaAwareTableStore` provides real schema namespaces with `search_path` resolution. Schema-qualified table names (`schema.table`) work throughout DDL and DML. `information_schema.tables` and `pg_catalog.pg_tables` report correct schema names.
- **Session variables** (`SET` / `SHOW` / `RESET` / `DISCARD`): Session variable storage with PostgreSQL-compatible default values (`server_version`, `client_encoding`, `timezone`, etc.). `SET search_path` propagates to the table store for name resolution.
- **In-memory transactions** (`BEGIN` / `COMMIT` / `ROLLBACK`): Real rollback support via document store snapshot/restore. Previously transactions required a persistent engine (`dbPath`). Savepoint support included.
- **`ALTER TABLE ADD CONSTRAINT`**: Supports `CHECK`, `UNIQUE`, `PRIMARY KEY`, and `FOREIGN KEY` constraints added after table creation. CHECK constraints validate existing data.
- **`ON CONFLICT DO NOTHING` without explicit columns**: INSERT with `ON CONFLICT DO NOTHING` now works without specifying conflict target columns -- catches UNIQUE/PK violations at insert time and silently skips.
- **In-memory BTREE index tracking**: `CREATE INDEX` / `DROP INDEX` works in in-memory engines (metadata tracking for optimizer use).
- **Deferred `DEFAULT` evaluation**: `DEFAULT CURRENT_TIMESTAMP`, `DEFAULT CURRENT_DATE`, and similar SQL function defaults are evaluated at insert time, not at table creation time.

### Fixes
- **Date/time functions return Date objects**: `NOW()`, `CURRENT_DATE`, `CURRENT_TIMESTAMP`, `DATE_TRUNC`, `MAKE_TIMESTAMP`, `TO_DATE`, `TO_TIMESTAMP`, `MAKE_DATE` now return JavaScript `Date` objects instead of ISO strings. Date/string comparison coercion added throughout the expression evaluator and predicate system.
- **Filter `_doc_id`/`_score` from `SELECT *`**: Internal columns injected by scan operators are no longer exposed in single-table `SELECT *` results. They remain available for graph traversal and search queries.
- **LEFT JOIN NULL padding**: Unmatched rows in LEFT/RIGHT/FULL outer joins now explicitly set opposite-side columns to `null`, matching PostgreSQL semantics. Previously unmatched rows had missing fields instead of `null`.
- **TIMESTAMP/DATE/TIME/INTERVAL DataType**: Added to the batch type system with proper inference from Date objects.

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
