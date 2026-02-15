"""
Cohesionn - Technical Knowledge RAG Pipeline for ARCA

Provides intelligent retrieval over domain-specific knowledge bases
configured via domain pack lexicon.

Usage:
    from tools.cohesionn import search_knowledge

    # Search all topics
    result = search_knowledge("what methodology was used in this analysis")
    print(result["context"])
    print(result["references"])

    # Search specific topic
    result = search_knowledge(
        "material strength specifications",
        topics=["general"]
    )

Ingestion:
    from tools.cohesionn import ingest_file, ingest_directory

    # Single file
    result = ingest_file("/path/to/textbook.pdf", "my_topic")

    # Directory
    results = ingest_directory("/path/to/docs/", "general")

CLI:
    python -m tools.cohesionn.cli ingest --source /data --all
    python -m tools.cohesionn.cli search "summarize key findings"
    python -m tools.cohesionn.cli stats
"""

from .retriever import (
    search_knowledge,
    CohesionnRetriever,
    RetrievalResult,
    Citation,
)
from .vectorstore import (
    get_knowledge_base,
    KnowledgeBase,
    TopicStore,
)
from .embeddings import (
    get_embedder,
    BGEEmbedder,
)
from .reranker import (
    get_reranker,
    get_router,
    BGEReranker,
    TopicRouter,
)
from .sparse_retrieval import (
    get_bm25_manager,
    BM25Manager,
    BM25Index,
    reciprocal_rank_fusion,
)
from .query_expansion import (
    get_query_expander,
    QueryExpander,
    expand_query,
)
from .hyde import (
    get_hyde_generator,
    HyDEGenerator,
    generate_hypothetical,
)
from .crag import (
    get_crag_evaluator,
    CRAGEvaluator,
    evaluate_and_correct,
)
from .chunker import (
    SemanticChunker,
    Chunk,
    chunk_text,
)
from .ingest import (
    DocumentIngester,
    IngestResult,
    ingest_file,
    ingest_directory,
)
from .manifest import (
    IngestManifest,
    IngestedFile,
)
from .autoingest import (
    AutoIngestService,
    run_auto_ingestion,
)
from typing import Dict, Any
from .session import (
    SessionKnowledge,
    SessionDocument,
    SessionSearchResult,
    get_session,
    clear_session,
    list_sessions,
)

# Phase 3a: RAPTOR hierarchical summarization
from .raptor import (
    RaptorClusterer,
    RaptorSummarizer,
    RaptorTreeBuilder,
    RaptorNode,
    RaptorRetrieverMixin,
)

# Phase 3b: GraphRAG with Neo4j
from .graph_extraction import (
    EntityExtractor,
    Entity,
    Relationship,
    ExtractionResult,
)
from .graph_builder import GraphBuilder, GraphBuildResult
from .graph_retriever import get_graph_retriever, GraphRetriever

# Phase 3c: Community Summaries for Global Search
from .community_detection import CommunityDetector, Community, DetectionResult
from .community_summarizer import CommunitySummarizer, CommunitySummary
from .query_classifier import QueryClassifier, QueryType, get_query_classifier
from .global_retriever import (
    GlobalRetriever,
    CommunityStore,
    get_global_retriever,
    get_community_store,
)


def warm_models():
    """
    Pre-load embedding and reranker models on startup.

    This eliminates cold start latency (35-50s) by loading models
    into memory before the first search request.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Force embedder model load
    embedder = get_embedder()
    _ = embedder.dimension  # Trigger load (works for both proxy and direct)
    logger.info(f"Embedder loaded: {embedder.dimension} dimensions")

    # Force reranker model load
    reranker = get_reranker()
    _ = reranker.model  # Trigger lazy load
    logger.info("Reranker loaded")

    # Pre-warm router topic embeddings
    router = get_router()
    router.warm_embeddings()
    logger.info("Router topic embeddings pre-computed")

    # Pre-load BM25 indices from disk (avoids cold start on first hybrid search)
    try:
        bm25 = get_bm25_manager()
        kb = get_knowledge_base()
        for topic in kb.discover_topics():
            idx = bm25.get_index(topic)
            if idx.count > 0:
                logger.info(f"BM25 index loaded: {topic} ({idx.count} docs)")
    except Exception as e:
        logger.warning(f"BM25 pre-load failed: {e}")


def unload_onnx_models():
    """
    Unload ONNX embedder + reranker from GPU to free VRAM.

    Called before batch ingest Phase 2 (vision extraction) so the vision
    llama-server gets exclusive GPU access. Phase 3 reloads them fresh
    via get_embedder()/get_reranker() since singletons are cleared.

    Token-aware batching queries free VRAM dynamically, so it adapts
    automatically after unload — no batch state to reset.
    """
    import logging
    import gc

    logger = logging.getLogger(__name__)

    from . import embeddings as _emb_mod
    from . import reranker as _rr_mod

    unloaded = []

    # Unload embedder singleton
    if _emb_mod._embedder is not None:
        _emb_mod._embedder.unload()
        _emb_mod._embedder = None
        unloaded.append("embedder")

    # Unload reranker singleton
    if _rr_mod._reranker is not None:
        _rr_mod._reranker.unload()
        _rr_mod._reranker = None
        unloaded.append("reranker")

    # No embed batch state to reset — token-aware batching queries
    # free VRAM dynamically, so it adapts automatically after unload.

    # Force garbage collection + CUDA cache clear
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    logger.info(f"ONNX models unloaded from GPU: {unloaded or 'none loaded'}")


def get_stats() -> Dict[str, Any]:
    """
    Get knowledge base statistics.

    Returns:
        Dict with 'topics' list and 'total_chunks' count.
    """
    try:
        retriever = CohesionnRetriever()
        return retriever.get_stats()
    except Exception:
        return {"topics": [], "total_chunks": 0}


__all__ = [
    # Main API
    "search_knowledge",
    # Retriever
    "CohesionnRetriever",
    "RetrievalResult",
    "Citation",
    # Vector store
    "get_knowledge_base",
    "KnowledgeBase",
    "TopicStore",
    # Embeddings
    "get_embedder",
    "BGEEmbedder",
    # Reranker
    "get_reranker",
    "get_router",
    "BGEReranker",
    "TopicRouter",
    # BM25 Sparse Retrieval (Phase 1)
    "get_bm25_manager",
    "BM25Manager",
    "BM25Index",
    "reciprocal_rank_fusion",
    # Query Expansion (Phase 1)
    "get_query_expander",
    "QueryExpander",
    "expand_query",
    # HyDE (Phase 2)
    "get_hyde_generator",
    "HyDEGenerator",
    "generate_hypothetical",
    # CRAG (Phase 2)
    "get_crag_evaluator",
    "CRAGEvaluator",
    "evaluate_and_correct",
    # Chunker
    "SemanticChunker",
    "Chunk",
    "chunk_text",
    # Ingestion
    "DocumentIngester",
    "IngestResult",
    "ingest_file",
    "ingest_directory",
    # Auto-ingestion
    "AutoIngestService",
    "IngestManifest",
    "IngestedFile",
    "run_auto_ingestion",
    # Session Knowledge
    "SessionKnowledge",
    "SessionDocument",
    "SessionSearchResult",
    "get_session",
    "clear_session",
    "list_sessions",
    # Warmup / GPU management
    "warm_models",
    "unload_onnx_models",
    # Stats
    "get_stats",
    # Phase 3a: RAPTOR
    "RaptorClusterer",
    "RaptorSummarizer",
    "RaptorTreeBuilder",
    "RaptorNode",
    "RaptorRetrieverMixin",
    # Phase 3b: GraphRAG
    "EntityExtractor",
    "Entity",
    "Relationship",
    "ExtractionResult",
    "GraphBuilder",
    "GraphBuildResult",
    "get_graph_retriever",
    "GraphRetriever",
    # Phase 3c: Community/Global Search
    "CommunityDetector",
    "Community",
    "DetectionResult",
    "CommunitySummarizer",
    "CommunitySummary",
    "QueryClassifier",
    "QueryType",
    "get_query_classifier",
    "GlobalRetriever",
    "CommunityStore",
    "get_global_retriever",
    "get_community_store",
]
