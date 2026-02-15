"""
Cohesionn Sparse Retrieval - BM25 for exact term matching

Complements dense (semantic) retrieval with sparse (lexical) matching.
# Domain-aware tokenizer. Preserves technical terms loaded from lexicon (e.g., abbreviations, standards codes)

Uses rank_bm25 library with domain-aware preprocessing.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

# BM25-preserved terms loaded from lexicon pipeline config
_preserve_terms_cache = None


def _get_preserve_terms():
    """Load BM25-preserved terms from lexicon pipeline config."""
    global _preserve_terms_cache
    if _preserve_terms_cache is not None:
        return _preserve_terms_cache
    try:
        from domain_loader import get_pipeline_config
        pipeline = get_pipeline_config()
        terms = pipeline.get("rag_preserve_terms", [])
        _preserve_terms_cache = set(terms) if terms else set()
    except Exception:
        _preserve_terms_cache = set()
    return _preserve_terms_cache


def clear_preserve_terms_cache():
    """Clear preserve terms cache (call after domain/lexicon change)."""
    global _preserve_terms_cache
    _preserve_terms_cache = None

# Standard stopwords to filter
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall", "can",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how",
}


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25 with engineering awareness.

    Preserves technical terms, splits on whitespace/punctuation,
    lowercases, filters standard stopwords.
    """
    # Lowercase
    text = text.lower()

    # Split on whitespace and most punctuation, but preserve technical notation
    # Keep: alphanumeric, hyphen (for compound terms), period (for decimals/abbreviations)
    tokens = re.findall(r"[a-z0-9]+(?:[-.]?[a-z0-9]+)*", text)

    # Filter stopwords but keep engineering terms
    tokens = [
        t for t in tokens
        if t in _get_preserve_terms() or (t not in STOPWORDS and len(t) > 1)
    ]

    return tokens


class BM25Index:
    """
    BM25 sparse index for a topic collection.

    Maintains parallel index with Qdrant for hybrid retrieval.
    Persists to disk for fast startup.
    """

    def __init__(self, topic: str, persist_dir: Optional[Path] = None):
        """
        Args:
            topic: Topic name (matches Qdrant collection)
            persist_dir: Directory for persisting index
        """
        self.topic = topic
        self.persist_dir = persist_dir or Path("/app/data/bm25_index")
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._bm25 = None
        self._doc_ids: List[str] = []
        self._doc_metadata: List[Dict[str, Any]] = []
        self._corpus: List[List[str]] = []

        # Try to load existing index
        self._load_index()

    @property
    def index_path(self) -> Path:
        return self.persist_dir / f"bm25_{self.topic}.pkl"

    def _load_index(self):
        """Load persisted index if available"""
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self._doc_ids = data["doc_ids"]
                    self._doc_metadata = data["doc_metadata"]
                    self._corpus = data["corpus"]
                    self._rebuild_bm25()
                logger.info(f"Loaded BM25 index for {self.topic}: {len(self._doc_ids)} documents")
            except Exception as e:
                logger.warning(f"Failed to load BM25 index for {self.topic}: {e}")

    def _save_index(self):
        """Persist index to disk"""
        try:
            data = {
                "doc_ids": self._doc_ids,
                "doc_metadata": self._doc_metadata,
                "corpus": self._corpus,
            }
            with open(self.index_path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Saved BM25 index for {self.topic}")
        except Exception as e:
            logger.warning(f"Failed to save BM25 index for {self.topic}: {e}")

    def _rebuild_bm25(self):
        """Rebuild BM25 model from corpus"""
        if not self._corpus:
            self._bm25 = None
            return

        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._corpus)
        except ImportError:
            logger.warning("rank_bm25 not installed, BM25 disabled")
            self._bm25 = None

    def add_documents(
        self,
        doc_ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Add documents to the BM25 index.

        Args:
            doc_ids: Unique IDs (should match Qdrant chunk IDs)
            documents: Document texts
            metadatas: Optional metadata per document
        """
        if not documents:
            return

        metadatas = metadatas or [{} for _ in documents]

        # Tokenize and add
        for doc_id, doc, meta in zip(doc_ids, documents, metadatas):
            # Skip duplicates
            if doc_id in self._doc_ids:
                continue

            tokens = tokenize(doc)
            self._doc_ids.append(doc_id)
            self._doc_metadata.append(meta)
            self._corpus.append(tokens)

        # Rebuild BM25 model
        self._rebuild_bm25()
        self._save_index()

        logger.info(f"BM25 index {self.topic}: added {len(documents)} docs, total {len(self._doc_ids)}")

    def search(
        self,
        query: str,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents matching query.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of results with id, score, metadata
        """
        if self._bm25 is None or not self._doc_ids:
            return []

        # Tokenize query
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)

        # Get top results
        scored_docs = list(zip(self._doc_ids, scores, self._doc_metadata))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score, metadata in scored_docs[:n_results]:
            if score > 0:  # Only return matches
                results.append({
                    "id": doc_id,
                    "bm25_score": float(score),
                    "metadata": metadata,
                    "topic": self.topic,
                })

        return results

    def delete_by_source(self, source: str):
        """Delete all documents from a source file"""
        indices_to_remove = [
            i for i, meta in enumerate(self._doc_metadata)
            if meta.get("source") == source
        ]

        if not indices_to_remove:
            return

        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            del self._doc_ids[i]
            del self._doc_metadata[i]
            del self._corpus[i]

        self._rebuild_bm25()
        self._save_index()
        logger.info(f"BM25 index {self.topic}: deleted {len(indices_to_remove)} docs from {source}")

    def clear(self):
        """Clear the entire index"""
        self._doc_ids = []
        self._doc_metadata = []
        self._corpus = []
        self._bm25 = None

        if self.index_path.exists():
            self.index_path.unlink()
        logger.info(f"Cleared BM25 index for {self.topic}")

    @property
    def count(self) -> int:
        return len(self._doc_ids)


class BM25Manager:
    """
    Manages BM25 indices across all topics.

    Singleton that provides topic-specific indices on demand.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or Path("/app/data/bm25_index")
        self._indices: Dict[str, BM25Index] = {}

    def get_index(self, topic: str) -> BM25Index:
        """Get or create BM25 index for topic"""
        if topic not in self._indices:
            self._indices[topic] = BM25Index(topic, self.persist_dir)
        return self._indices[topic]

    def search(
        self,
        query: str,
        topics: List[str],
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search across multiple topic indices.

        Args:
            query: Search query
            topics: Topics to search
            n_results: Results per topic

        Returns:
            Combined results sorted by BM25 score
        """
        all_results = []

        for topic in topics:
            index = self.get_index(topic)
            results = index.search(query, n_results=n_results)
            all_results.extend(results)

        # Sort by score
        all_results.sort(key=lambda x: x["bm25_score"], reverse=True)

        return all_results[:n_results]

    def get_stats(self) -> Dict[str, int]:
        """Get document counts per topic"""
        return {topic: idx.count for topic, idx in self._indices.items()}


# Singleton
_bm25_manager: Optional[BM25Manager] = None


def get_bm25_manager(persist_dir: Optional[Path] = None) -> BM25Manager:
    """Get or create singleton BM25 manager"""
    global _bm25_manager

    if _bm25_manager is None:
        _bm25_manager = BM25Manager(persist_dir)

    return _bm25_manager


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    graph_results: List[Dict[str, Any]] = None,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.3,
    graph_weight: float = 0.2,
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Combine dense, sparse, and graph results using Reciprocal Rank Fusion.

    RRF formula: score = sum(weight / (k + rank))

    Args:
        dense_results: Results from dense (semantic) retrieval
        sparse_results: Results from sparse (BM25) retrieval
        graph_results: Results from graph (Neo4j) retrieval (optional)
        dense_weight: Weight for dense results (default 0.5)
        sparse_weight: Weight for sparse results (default 0.3)
        graph_weight: Weight for graph results (default 0.2)
        k: RRF constant (default 60, standard value)

    Returns:
        Fused results sorted by combined RRF score
    """
    # Normalize weights if graph not provided
    if graph_results is None:
        graph_results = []
        # Redistribute graph weight proportionally to dense and sparse
        total = dense_weight + sparse_weight
        if total > 0:
            scale = (dense_weight + sparse_weight + graph_weight) / total
            dense_weight = dense_weight * scale
            sparse_weight = sparse_weight * scale

    scores: Dict[str, float] = {}
    result_map: Dict[str, Dict[str, Any]] = {}

    # Process dense results
    for rank, result in enumerate(dense_results, 1):
        doc_id = result.get("id", str(rank))
        rrf_score = dense_weight / (k + rank)
        scores[doc_id] = scores.get(doc_id, 0) + rrf_score
        result_map[doc_id] = result

    # Process sparse results
    for rank, result in enumerate(sparse_results, 1):
        doc_id = result.get("id", str(rank))
        rrf_score = sparse_weight / (k + rank)
        scores[doc_id] = scores.get(doc_id, 0) + rrf_score

        # Merge metadata if not already present
        if doc_id not in result_map:
            result_map[doc_id] = result
        else:
            # Add BM25 score to existing result
            result_map[doc_id]["bm25_score"] = result.get("bm25_score", 0)

    # Process graph results
    for rank, result in enumerate(graph_results, 1):
        doc_id = result.get("id") or result.get("chunk_id", str(rank))
        rrf_score = graph_weight / (k + rank)
        scores[doc_id] = scores.get(doc_id, 0) + rrf_score

        # Merge metadata if not already present
        if doc_id not in result_map:
            result_map[doc_id] = result
        else:
            # Add graph metadata to existing result
            result_map[doc_id]["graph_score"] = result.get("graph_score", 0)
            result_map[doc_id]["matched_entities"] = result.get("matched_entities", [])
            result_map[doc_id]["is_graph_result"] = True

    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Build output
    fused = []
    for doc_id in sorted_ids:
        result = result_map[doc_id].copy()
        result["rrf_score"] = scores[doc_id]
        fused.append(result)

    return fused
