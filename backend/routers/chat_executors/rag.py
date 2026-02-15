"""
ARCA Chat Executors - RAG

Knowledge search (Cohesionn) and session document search.
"""

import logging
from typing import Dict, Any, List

from errors import (
    handle_tool_errors,
    NotFoundError,
    DependencyError,
)

logger = logging.getLogger(__name__)


@handle_tool_errors("cohesionn")
def execute_search_knowledge(query: str, topics: List[str] = None, rerank: bool = None, profile: str = None) -> Dict[str, Any]:
    """Search Cohesionn technical knowledge bases.

    Args:
        query: Search query
        topics: Optional list of topics to search (defaults to all enabled)
        rerank: Override reranker setting. None=use config, True=force rerank, False=skip rerank
        profile: Optional retrieval profile override ("fast", "deep"). Temporarily applies
                 profile settings for this query without changing the global config.
    """
    try:
        from tools.cohesionn import CohesionnRetriever
        from config import runtime_config
    except ImportError:
        raise DependencyError(
            "Knowledge base not available",
            details="Run ingestion first to set up the knowledge base",
            package="cohesionn",
        )

    # Get enabled topics from runtime config
    enabled_topics = runtime_config.get_enabled_topics()

    # Filter requested topics to only include enabled ones
    if topics:
        requested = topics[:]
        topics = [t for t in topics if t in enabled_topics]
        if not topics:
            raise NotFoundError(
                f"Requested topics {requested} are not enabled (enabled: {enabled_topics or 'none'})",
                details="Enable topics in the admin panel (Ctrl+Shift+D)",
                resource_type="topic",
            )
    else:
        topics = enabled_topics

    if not topics:
        raise NotFoundError(
            "No knowledge topics are enabled",
            details="Enable topics in the admin panel (Ctrl+Shift+D)",
            resource_type="topic",
        )

    # Get reranker settings - allow override for fast path
    rerank_enabled = rerank if rerank is not None else runtime_config.reranker_enabled
    initial_k = runtime_config.reranker_candidates if rerank_enabled else 5
    top_k = runtime_config.rag_top_k

    # Create retriever and search
    retriever = CohesionnRetriever()
    result = retriever.retrieve(
        query=query,
        topics=topics,
        top_k=top_k,
        initial_k=initial_k,
        rerank=rerank_enabled,
    )

    # Format result for tool response
    chunks = []
    for i, chunk in enumerate(result.chunks):
        citation = result.citations[i] if i < len(result.citations) else None
        chunks.append(
            {
                "content": chunk.get("content", ""),
                "source": citation.source if citation else "Unknown",
                "title": citation.title if citation else "Unknown",
                "page": citation.page if citation else None,
                "section": citation.section if citation else None,
                "topic": citation.topic if citation else "unknown",
                "score": result._normalize_score(chunk.get("rerank_score") or chunk.get("score", 0)),
            }
        )

    # Extract citations
    citations = []
    total_score = 0
    for c in chunks:
        citation = {
            "source": c["title"],
            "title": c["title"],
            "page": c.get("page"),
            "section": c.get("section"),
            "topic": c.get("topic"),
            "score": c.get("score", 0),
        }
        citations.append(citation)
        total_score += citation["score"]

    return {
        "success": True,
        "chunks": chunks,
        "context": result.get_context(),
        "citations": citations,
        "avg_confidence": total_score / len(citations) if citations else 0,
        "topics_searched": result.topics_searched,
        "reranker_used": rerank,
    }


@handle_tool_errors("cohesionn")
def execute_search_session(query: str, search_session_docs_fn=None) -> Dict[str, Any]:
    """Search user's uploaded session documents.

    Args:
        query: Search query
        search_session_docs_fn: Optional function to search session docs (injected from upload router)
    """
    # Try the new RAM-based search first
    if search_session_docs_fn is not None:
        result = search_session_docs_fn(query)
        # Add citations from session search results
        # Note: search_session_docs now returns cleaned filenames and normalized scores
        if result.get("success") and result.get("results"):
            citations = []
            total_score = 0
            for r in result.get("results", []):
                # filename is already cleaned by search_session_docs
                clean_name = r.get("filename", "Uploaded Document")
                citation = {
                    "source": clean_name,
                    "title": clean_name,
                    "section": r.get("chunk_index", None),
                    "topic": "Session Documents",
                    "score": r.get("score", 0),  # Already normalized
                }
                citations.append(citation)
                total_score += citation["score"]

            result["citations"] = citations
            result["avg_confidence"] = total_score / len(citations) if citations else 0
        return result

    # Fallback: search via cohesionn session module
    try:
        from tools.cohesionn.session import get_session
    except ImportError:
        raise DependencyError(
            "Session search not available",
            details="Session document search module is not installed",
            package="cohesionn.session",
        )

    session = get_session("default")

    if session.chunk_count == 0:
        raise NotFoundError(
            "No documents uploaded yet", details="Upload a PDF, Word doc, or text file first", resource_type="file"
        )

    context = session.get_context(query)
    results = session.search(query, top_k=5)

    # Build citations
    citations = []
    total_score = 0
    documents = []
    for r in results:
        doc = {
            "filename": r.filename,
            "doc_type": r.doc_type,
            "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
            "score": round(r.score, 3),
        }
        documents.append(doc)

        citation = {
            "source": r.filename,
            "title": r.filename,
            "topic": "Session Documents",
            "score": r.score,
        }
        citations.append(citation)
        total_score += r.score

    return {
        "success": True,
        "query": query,
        "context": context,
        "num_results": len(results),
        "documents": documents,
        "citations": citations,
        "avg_confidence": total_score / len(citations) if citations else 0,
    }
