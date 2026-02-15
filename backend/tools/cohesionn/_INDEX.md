# backend/tools/cohesionn/

ARCA's RAG (Retrieval-Augmented Generation) pipeline. Implements a 12-step hybrid retrieval flow combining dense semantic search (Qdrant), sparse lexical search (BM25), hierarchical summaries (RAPTOR), knowledge graph traversal (GraphRAG), and corrective web fallback (CRAG). All domain knowledge flows through lexicon pipeline config.

| File | Purpose | Key Exports |
|------|---------|-------------|
| __init__.py | Public API, model warm-up, stats | `search_knowledge()`, `warm_models()`, `get_stats()` |
| __main__.py | `python -m tools.cohesionn` CLI entry point | -- |
| retriever.py | Main 12-step retrieval pipeline with RRF fusion, quality gates | `CohesionnRetriever`, `RetrievalResult`, `Citation` |
| embeddings.py | Universal embedder (Qwen3/BGE/Nomic), ONNX-first, LRU cache | `UniversalEmbedder`, `get_embedder()` |
| reranker.py | Cross-encoder reranking, MMR diversity, topic routing, domain boost | `BGEReranker`, `DiversityReranker`, `TopicRouter`, `apply_domain_boost()` |
| chunker.py | Structure-aware semantic chunking with contextual prefixes | `SemanticChunker`, `Chunk`, `ContentType` |
| vectorstore.py | Qdrant storage (single "cohesionn" collection, topic-filtered views) | `TopicStore`, `KnowledgeBase` |
| sparse_retrieval.py | BM25 index with engineering-aware tokenization, RRF fusion | `BM25Index`, `BM25Manager`, `reciprocal_rank_fusion()` |
| query_expansion.py | Synonym expansion from lexicon `rag_synonyms` | `expand_query()`, `QueryExpander` |
| hyde.py | Hypothetical Document Embeddings via fast LLM | `HyDEGenerator`, `generate_hypothetical()`, `expand_query_hyde()` |
| crag.py | Corrective RAG -- web search fallback on low confidence | `CRAGEvaluator` |
| query_classifier.py | GLOBAL/LOCAL/HYBRID query classification (regex, no LLM) | `QueryClassifier`, `QueryType` |
| graph_extraction.py | Entity extraction (pattern + optional SpaCy NER) | `EntityExtractor`, `ExtractionResult` |
| graph_builder.py | Neo4j graph construction with auto-constraint creation | `GraphBuilder`, `GraphBuildResult` |
| graph_retriever.py | Entity-based graph traversal (0-2 hops) | `GraphRetriever`, `get_graph_retriever()` |
| global_retriever.py | Community summary search for broad/thematic queries | `GlobalRetriever` |
| community_detection.py | Leiden algorithm clustering on Neo4j graph | `CommunityDetector`, `DetectionResult` |
| community_summarizer.py | LLM-generated community summaries | `CommunitySummarizer` |
| community_cli.py | CLI for community detection and summarization | -- |
| session.py | Per-session ephemeral Qdrant storage (24h TTL) | `SessionKnowledge` |
| corpus_profiler.py | Post-ingest term extraction, saves corpus_profile.json | `extract_terms()`, `profile_after_ingest()` |
| ingest.py | Document ingestion pipeline (SESSION/KNOWLEDGE_BASE modes) | `DocumentIngester`, `IngestResult`, `IngestMode` |
| autoingest.py | Filesystem-watched auto-ingestion with manifest tracking | `AutoIngestService` |
| manifest.py | Ingestion manifest (MD5 content hash change detection) | `IngestManifest` |
| cli.py | Full CLI interface for cohesionn operations | -- |
| raptor_cli.py | CLI for RAPTOR tree building | -- |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| raptor/ | RAPTOR hierarchical summarization (UMAP clustering, level-aware LLM summaries, tree traversal) |
| benchmark/ | Model shootout harness (7 phases: ingestion, reranker, embedding, cross-matrix, LLM, param sweep, ablation) |
