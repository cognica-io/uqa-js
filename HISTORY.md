# History

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
