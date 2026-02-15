"""
Global Retriever - Community-based retrieval for broad questions

Searches community summaries for theme-level context.
"""

import logging
from typing import List, Dict, Any, Optional

from .query_classifier import get_query_classifier, QueryType

logger = logging.getLogger(__name__)


# Qdrant collection for community summaries
COMMUNITY_COLLECTION = "community_summaries"


class GlobalRetriever:
    """
    Retrieve context from community summaries.

    Used for broad/conceptual questions where theme-level
    context is more useful than individual chunks.

    Process:
    1. Classify query to confirm global search is appropriate
    2. Search community summaries by semantic similarity
    3. Return top-k community summaries as context
    """

    def __init__(
        self,
        top_k: int = 3,
        min_score: float = 0.3,
    ):
        """
        Args:
            top_k: Number of community summaries to return
            min_score: Minimum similarity score threshold
        """
        self.top_k = top_k
        self.min_score = min_score
        self.classifier = get_query_classifier()

    def retrieve(
        self,
        query: str,
        level: str = "medium",
        topic: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant community summaries.

        Args:
            query: Search query
            level: Community resolution level ("coarse", "medium", "fine")
            topic: Optional topic filter

        Returns:
            List of community summary results
        """
        # Check if global search is appropriate
        strategy = self.classifier.get_search_strategy(query)

        if not strategy["use_community_search"]:
            logger.debug(f"Query classified as {strategy['query_type']}, skipping global search")
            return []

        try:
            from .embeddings import get_embedder
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            import os

            # Get embedder and client
            embedder = get_embedder()
            qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
            client = QdrantClient(url=qdrant_url)

            # Check if collection exists
            collections = client.get_collections().collections
            if not any(c.name == COMMUNITY_COLLECTION for c in collections):
                logger.warning(f"Community collection '{COMMUNITY_COLLECTION}' not found")
                return []

            # Embed query
            query_embedding = embedder.embed_query(query)

            # Build filter
            must_conditions = []
            if level:
                must_conditions.append(
                    FieldCondition(key="level", match=MatchValue(value=level))
                )
            if topic:
                must_conditions.append(
                    FieldCondition(key="topic", match=MatchValue(value=topic))
                )

            query_filter = Filter(must=must_conditions) if must_conditions else None

            # Search
            response = client.query_points(
                collection_name=COMMUNITY_COLLECTION,
                query=query_embedding,
                limit=self.top_k,
                query_filter=query_filter,
                with_payload=True,
            )

            # Format results
            results = []
            for hit in response.points:
                if hit.score >= self.min_score:
                    payload = hit.payload or {}
                    results.append({
                        "community_id": payload.get("community_id", ""),
                        "level": payload.get("level", ""),
                        "summary": payload.get("summary", ""),
                        "themes": payload.get("themes", []),
                        "key_entities": payload.get("key_entities", []),
                        "score": hit.score,
                        "node_count": payload.get("node_count", 0),
                        "is_community": True,
                    })

            logger.debug(f"Global retrieval: {len(results)} community summaries")
            return results

        except Exception as e:
            logger.error(f"Global retrieval failed: {e}")
            return []

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format community summaries for LLM context.

        Args:
            results: List of community summary results

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        parts = ["## Overview Context (from knowledge themes)\n"]

        for i, result in enumerate(results, 1):
            themes_str = ", ".join(result.get("themes", [])[:3])
            parts.append(f"### Theme {i}: {themes_str}")
            parts.append(result.get("summary", ""))
            parts.append("")

        return "\n".join(parts)

    def should_use(self, query: str) -> bool:
        """Check if global retrieval should be used for this query."""
        return self.classifier.should_use_global_search(query)


class CommunityStore:
    """
    Manages storage of community summaries in Qdrant.

    Creates and populates the community_summaries collection.
    """

    def __init__(self):
        self._client = None
        self._embedder = None

    @property
    def client(self):
        """Lazy-load Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient
            import os

            qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
            self._client = QdrantClient(url=qdrant_url)
        return self._client

    @property
    def embedder(self):
        """Lazy-load embedder."""
        if self._embedder is None:
            from .embeddings import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def ensure_collection(self):
        """Ensure community summaries collection exists."""
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

        collections = self.client.get_collections().collections
        if any(c.name == COMMUNITY_COLLECTION for c in collections):
            return

        logger.info(f"Creating collection: {COMMUNITY_COLLECTION}")

        self.client.create_collection(
            collection_name=COMMUNITY_COLLECTION,
            vectors_config=VectorParams(
                size=1024,  # Match Qwen3-Embedding dimension
                distance=Distance.COSINE,
            ),
        )

        # Create payload indexes
        for field_name, field_type in [
            ("community_id", PayloadSchemaType.KEYWORD),
            ("level", PayloadSchemaType.KEYWORD),
            ("topic", PayloadSchemaType.KEYWORD),
        ]:
            self.client.create_payload_index(
                collection_name=COMMUNITY_COLLECTION,
                field_name=field_name,
                field_schema=field_type,
            )

    def store_summaries(self, summaries: List[Any], topic: str = None) -> int:
        """
        Store community summaries in Qdrant.

        Args:
            summaries: List of CommunitySummary objects
            topic: Optional topic label

        Returns:
            Number of summaries stored
        """
        from qdrant_client.models import PointStruct
        import hashlib

        self.ensure_collection()

        points = []
        for summary in summaries:
            # Generate embedding if not present
            if summary.embedding is None:
                summary.embedding = self.embedder.embed_document(summary.summary)

            # Generate point ID from community_id
            point_id = int(hashlib.md5(summary.community_id.encode()).hexdigest()[:16], 16)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=summary.embedding,
                    payload={
                        "community_id": summary.community_id,
                        "level": summary.level,
                        "summary": summary.summary,
                        "themes": summary.themes,
                        "key_entities": summary.key_entities,
                        "node_count": summary.node_count,
                        "topic": topic or "",
                        **summary.metadata,
                    },
                )
            )

        if points:
            self.client.upsert(
                collection_name=COMMUNITY_COLLECTION,
                points=points,
                wait=True,
            )

        logger.info(f"Stored {len(points)} community summaries")
        return len(points)

    def delete_level(self, level: str):
        """Delete all summaries for a resolution level."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self.client.delete(
            collection_name=COMMUNITY_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="level", match=MatchValue(value=level))]
            ),
        )
        logger.info(f"Deleted summaries for level: {level}")

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(COMMUNITY_COLLECTION)
            return {
                "collection": COMMUNITY_COLLECTION,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
            }
        except Exception:
            return {"collection": COMMUNITY_COLLECTION, "points_count": 0}


# Singletons
_global_retriever = None
_community_store = None


def get_global_retriever() -> GlobalRetriever:
    """Get or create singleton global retriever."""
    global _global_retriever

    if _global_retriever is None:
        from config import runtime_config
        _global_retriever = GlobalRetriever(top_k=runtime_config.global_search_top_k)

    return _global_retriever


def get_community_store() -> CommunityStore:
    """Get or create singleton community store."""
    global _community_store

    if _community_store is None:
        _community_store = CommunityStore()

    return _community_store
