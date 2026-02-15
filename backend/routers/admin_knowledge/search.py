import logging
import time
from typing import Dict, Any

from fastapi import Depends, HTTPException

from services.admin_auth import verify_admin
from . import router, COHESIONN_DB_DIR
from .models import SearchTestRequest

logger = logging.getLogger(__name__)


@router.post("/search")
async def test_search(
    request: SearchTestRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Test search with detailed scoring information.

    Args:
        query: Search query
        topics: Optional topic filter
        top_k: Number of results
        include_routing: Include topic routing decision
        include_raw_scores: Include raw scores before normalization

    Returns:
        Search results with detailed scoring
    """


    try:
        from tools.cohesionn.retriever import CohesionnRetriever, RetrievalResult
        from tools.cohesionn.reranker import get_router

        start_time = time.time()

        retriever = CohesionnRetriever(db_path=COHESIONN_DB_DIR)

        # Get routing decision first if requested
        routing_info = None
        if request.include_routing and request.topics is None:
            router = get_router()
            routed_topics = router.route(request.query)
            routing_info = {
                "query": request.query,
                "routed_to": routed_topics,
            }

        # Run retrieval
        result = retriever.retrieve(
            query=request.query,
            topics=request.topics,
            top_k=request.top_k,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Format results with optional raw scores
        formatted_results = []
        for i, chunk in enumerate(result.chunks):
            citation = result.citations[i] if i < len(result.citations) else None

            item = {
                "content": chunk["content"][:500] + "..." if len(chunk["content"]) > 500 else chunk["content"],
                "source": citation.title if citation else "Unknown",
                "page": chunk["metadata"].get("page"),
                "topic": chunk.get("topic", "unknown"),
                "normalized_score": RetrievalResult._normalize_score(chunk.get("rerank_score", chunk.get("score", 0))),
            }

            if request.include_raw_scores:
                item["raw_score"] = chunk.get("score", 0)
                item["rerank_score"] = chunk.get("rerank_score")

            formatted_results.append(item)

        return {
            "success": True,
            "query": request.query,
            "topics_searched": result.topics_searched,
            "num_results": len(formatted_results),
            "results": formatted_results,
            "routing": routing_info,
            "processing_ms": elapsed_ms,
        }

    except Exception as e:
        logger.error(f"Search test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
