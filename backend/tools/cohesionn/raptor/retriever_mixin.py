"""
RAPTOR Retriever Mixin - RAPTOR-aware retrieval integration

Provides methods to retrieve from RAPTOR tree and merge with
standard dense retrieval results.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class RaptorRetrieverMixin:
    """
    Mixin class for RAPTOR-aware retrieval.

    Adds methods to retrieve hierarchical summaries and merge
    them with standard retrieval results.

    Retrieval strategies:
    - collapsed: Search all levels simultaneously, merge by score
    - tree_traversal: Start at top level, traverse down to relevant branches
    """

    def retrieve_raptor_nodes(
        self,
        query: str,
        topics: List[str],
        n_results: int = 10,
        levels: Optional[List[int]] = None,
        strategy: str = "collapsed",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve RAPTOR summary nodes.

        Args:
            query: Search query
            topics: Topics to search
            n_results: Number of results
            levels: Specific levels to search (default: [1, 2, 3])
            strategy: "collapsed" or "tree_traversal"

        Returns:
            List of RAPTOR nodes with content and scores
        """
        if levels is None:
            levels = [1, 2, 3]  # Search summary levels by default

        if strategy == "collapsed":
            return self._retrieve_collapsed(query, topics, n_results, levels)
        elif strategy == "tree_traversal":
            return self._retrieve_tree_traversal(query, topics, n_results)
        else:
            logger.warning(f"Unknown RAPTOR strategy '{strategy}', using collapsed")
            return self._retrieve_collapsed(query, topics, n_results, levels)

    def _retrieve_collapsed(
        self,
        query: str,
        topics: List[str],
        n_results: int,
        levels: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Collapsed retrieval: search all levels simultaneously.

        This is the default strategy - fast and effective for most queries.
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
            from ..vectorstore import get_knowledge_base
            from ..embeddings import get_embedder

            kb = get_knowledge_base()
            embedder = get_embedder()

            # Embed query
            query_embedding = embedder.embed_query(query)

            # Build filter for topics and levels
            client = kb.client

            results = []
            for topic in topics:
                response = client.query_points(
                    collection_name="cohesionn",
                    query=query_embedding,
                    limit=n_results,
                    query_filter=Filter(
                        must=[
                            FieldCondition(key="topic", match=MatchValue(value=topic)),
                            FieldCondition(key="raptor_level", match=MatchAny(any=levels)),
                        ],
                    ),
                    with_payload=True,
                )

                for hit in response.points:
                    payload = hit.payload or {}
                    results.append({
                        "id": payload.get("chunk_id", str(hit.id)),
                        "content": payload.get("content", ""),
                        "score": hit.score,
                        "raptor_level": payload.get("raptor_level", 0),
                        "topic": topic,
                        "metadata": {
                            k: v for k, v in payload.items()
                            if k not in ("content", "chunk_id", "raptor_level")
                        },
                        "is_raptor": True,
                    })

            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:n_results]

        except Exception as e:
            logger.error(f"RAPTOR collapsed retrieval failed: {e}")
            return []

    def _retrieve_tree_traversal(
        self,
        query: str,
        topics: List[str],
        n_results: int,
    ) -> List[Dict[str, Any]]:
        """
        Tree traversal retrieval: start at top, descend to relevant branches.

        More expensive but can provide better context for complex queries.
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            from ..vectorstore import get_knowledge_base
            from ..embeddings import get_embedder

            kb = get_knowledge_base()
            embedder = get_embedder()
            client = kb.client

            query_embedding = embedder.embed_query(query)

            all_results = []

            for topic in topics:
                # Start at highest level available
                for level in [3, 2, 1]:
                    response = client.query_points(
                        collection_name="cohesionn",
                        query=query_embedding,
                        limit=3,  # Top 3 at each level
                        query_filter=Filter(
                            must=[
                                FieldCondition(key="topic", match=MatchValue(value=topic)),
                                FieldCondition(key="raptor_level", match=MatchValue(value=level)),
                            ],
                        ),
                        with_payload=True,
                    )

                    if response.points:
                        for hit in response.points:
                            payload = hit.payload or {}
                            all_results.append({
                                "id": payload.get("chunk_id", str(hit.id)),
                                "content": payload.get("content", ""),
                                "score": hit.score,
                                "raptor_level": level,
                                "topic": topic,
                                "metadata": {
                                    k: v for k, v in payload.items()
                                    if k not in ("content", "chunk_id", "raptor_level")
                                },
                                "is_raptor": True,
                            })
                        break  # Found results at this level, stop descending

            # Sort by score
            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:n_results]

        except Exception as e:
            logger.error(f"RAPTOR tree traversal failed: {e}")
            return []

    def merge_raptor_results(
        self,
        dense_results: List[Dict[str, Any]],
        raptor_results: List[Dict[str, Any]],
        raptor_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Merge dense retrieval results with RAPTOR results.

        Uses weighted combination to balance chunk-level detail
        with hierarchical summaries.

        Args:
            dense_results: Results from standard dense retrieval
            raptor_results: Results from RAPTOR retrieval
            raptor_weight: Weight for RAPTOR results (0-1)

        Returns:
            Merged and re-scored results
        """
        if not raptor_results:
            return dense_results

        if not dense_results:
            return raptor_results

        dense_weight = 1.0 - raptor_weight

        # Build score map
        merged_scores: Dict[str, float] = {}
        result_map: Dict[str, Dict[str, Any]] = {}

        # Process dense results
        for i, result in enumerate(dense_results):
            doc_id = result.get("id", str(i))
            score = result.get("score", 0)
            merged_scores[doc_id] = score * dense_weight
            result_map[doc_id] = result

        # Process RAPTOR results
        for i, result in enumerate(raptor_results):
            doc_id = result.get("id", f"raptor_{i}")
            score = result.get("score", 0)

            if doc_id in merged_scores:
                # Combine scores if same document
                merged_scores[doc_id] += score * raptor_weight
            else:
                merged_scores[doc_id] = score * raptor_weight
                result_map[doc_id] = result

        # Sort by merged score
        sorted_ids = sorted(merged_scores.keys(), key=lambda x: merged_scores[x], reverse=True)

        # Build output
        merged = []
        for doc_id in sorted_ids:
            result = result_map[doc_id].copy()
            result["merged_score"] = merged_scores[doc_id]
            merged.append(result)

        return merged

    def should_use_raptor(
        self,
        query: str,
        config: Optional[Any] = None,
    ) -> bool:
        """
        Determine if RAPTOR should be used for this query.

        Heuristics:
        - Broad/conceptual queries benefit from RAPTOR
        - Specific/factual queries use standard retrieval
        """
        # Check config first
        if config:
            if hasattr(config, "raptor_enabled") and not config.raptor_enabled:
                return False

        # Heuristics for broad queries
        broad_indicators = [
            "overview",
            "summary",
            "introduction",
            "main",
            "general",
            "concepts",
            "principles",
            "fundamentals",
            "types of",
            "categories",
            "classification",
            "compare",
            "difference between",
            "what are the",
            "explain",
        ]

        query_lower = query.lower()
        for indicator in broad_indicators:
            if indicator in query_lower:
                return True

        # Short queries are often broad
        if len(query.split()) <= 4:
            return True

        return False
