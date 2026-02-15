"""
Cohesionn Vector Store - Qdrant-based knowledge storage

Single collection with topic as payload field for efficient multi-topic retrieval.
Optimized for 1024-dim embeddings (Qwen3-Embedding-0.6B).
"""

import logging
import math
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    PayloadSchemaType,
    HnswConfigDiff,
    OptimizersConfigDiff,
)

from .embeddings import get_embedder, EMBEDDING_MODEL
from .chunker import Chunk
from .sparse_retrieval import get_bm25_manager

logger = logging.getLogger(__name__)

# Token-aware embed batching.
# VRAM for self-attention scales as O(batch × max_seq²). Rather than fix
# the batch count and react to OOM, we derive a VRAM budget from the actual
# free GPU memory and let batch size adapt to content length automatically.
# Short chunks → big batches (fast). Long chunks → small batches (safe).
# No OOM, no corrupted CUDA contexts, no retry latency.

# Bytes of VRAM per (batch × token²) unit.
# Covers attention scores, K/V matrices, and ONNX intermediate buffers.
_BYTES_PER_TOKEN_SQ = 128


def _get_vram_budget() -> int:
    """Derive token² budget from actual free VRAM.

    Queries the GPU for real free memory, applies a 50% safety margin,
    and converts to token² units. Automatically adapts to:
    - GPU size (4GB, 8GB, 24GB, 48GB)
    - Other GPU consumers (vision server, chat model)
    - Model weight variations
    - Multi-GPU: queries the device ONNX is actually using

    Returns a conservative budget in token² units.
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Query the device the embedder is actually on (from device map).
            try:
                from services.hardware import get_gpu_for_component
                device_idx = get_gpu_for_component("embedder")
            except Exception:
                device_idx = 0
            device_idx = min(device_idx, torch.cuda.device_count() - 1)
            free, _ = torch.cuda.mem_get_info(device_idx)
            budget = int(free * 0.5 / _BYTES_PER_TOKEN_SQ)
            return max(budget, 1_000_000)  # Floor: 1M (always embed at least 1 doc)
    except (ImportError, RuntimeError):
        pass
    # CPU: no VRAM constraint, RAM is abundant
    return 100_000_000


def _token_aware_batches(documents: List[str]) -> List[List[str]]:
    """Group documents into VRAM-safe sub-batches for embedding.

    Queries actual free VRAM and bins documents so each sub-batch's
    attention cost (batch × max_seq²) stays under budget:

        24GB GPU, vision up (8GB free after model):
          500-tok chunks  → batch ≈ 100+ (short docs, fast)
          1000-tok chunks → batch ≈ 30   (medium)
          2000-tok chunks → batch ≈ 8    (long docs from vision)
          4000-tok chunks → batch ≈ 2    (very long, still safe)

    Hardware-agnostic: works on any GPU, any VRAM state.
    """
    if not documents:
        return []

    budget = _get_vram_budget()
    batches: List[List[str]] = []
    current: List[str] = []
    current_max_tok = 0

    for doc in documents:
        tok = max(1, len(doc) // 4)  # rough chars→tokens
        new_max = max(current_max_tok, tok)
        cost = (len(current) + 1) * new_max * new_max

        if current and cost > budget:
            batches.append(current)
            current = [doc]
            current_max_tok = tok
        else:
            current.append(doc)
            current_max_tok = new_max

    if current:
        batches.append(current)

    return batches


class _LazyEmbedder:
    """
    Defers ONNX embedder loading until first use, and always follows the singleton.

    During three-phase batch ingest:
    - Phase 1: KnowledgeBase created, but embedder not loaded (no GPU)
    - Phase 2: unload_onnx_models() clears the singleton (frees VRAM for vision)
    - Phase 3: First embed call triggers get_embedder() → fresh load on clean GPU

    Always delegates to get_embedder() so singleton resets (from unload_onnx_models)
    are respected automatically.
    """

    def embed_documents(self, *args, **kwargs):
        return get_embedder().embed_documents(*args, **kwargs)

    def embed_query(self, *args, **kwargs):
        return get_embedder().embed_query(*args, **kwargs)


def chunk_id_to_int(chunk_id: str) -> int:
    """Convert hex chunk_id to integer for Qdrant point ID.

    Qdrant requires point IDs to be unsigned integers or UUIDs.
    Our chunk_ids are 16-char hex strings (64 bits) from MD5 hash.
    """
    return int(chunk_id, 16)

# Collection configuration
COLLECTION_NAME = "cohesionn"
VECTOR_DIM = 1024

# HNSW parameters optimized for 1024-dim embeddings, ~50K-500K chunks
HNSW_CONFIG = HnswConfigDiff(
    m=16,              # Connections per node (balance for 1024-dim)
    ef_construct=128,  # Build quality (higher = better recall)
)
SEARCH_EF = 64  # Runtime search depth

# Quantization for 4x compression with ~1-2% recall loss
QUANTIZATION_CONFIG = ScalarQuantization(
    scalar=ScalarQuantizationConfig(
        type=ScalarType.INT8,
        quantile=0.99,
        always_ram=True,  # 96GB RAM available
    )
)

# Payload indexes for fast filtering
PAYLOAD_INDEXES = {
    "topic": PayloadSchemaType.KEYWORD,
    "source": PayloadSchemaType.KEYWORD,
    "session_id": PayloadSchemaType.KEYWORD,
    "doc_id": PayloadSchemaType.KEYWORD,
    "raptor_level": PayloadSchemaType.INTEGER,  # RAPTOR tree level (0=leaf, 1-3=summaries)
}


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client from environment or default."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=qdrant_url)


class TopicStore:
    """Qdrant-backed store for a single topic (filtered view of shared collection)"""

    def __init__(self, topic: str, client: QdrantClient, embedder):
        self.topic = topic
        self.client = client
        self.embedder = embedder
        self._ensure_collection()

        logger.info(f"TopicStore '{topic}' initialized: {self.count} chunks")

    def _ensure_collection(self):
        """Ensure collection exists with proper configuration."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if COLLECTION_NAME in collection_names:
            # Check vector dimension matches current embedding model
            try:
                info = self.client.get_collection(COLLECTION_NAME)
                existing_dim = info.config.params.vectors.size
                if existing_dim != VECTOR_DIM:
                    logger.warning(
                        f"EMBEDDING MODEL MISMATCH: Collection has {existing_dim}-dim vectors "
                        f"but current model ({EMBEDDING_MODEL}) produces {VECTOR_DIM}-dim. "
                        f"Old vectors will be incompatible! Recreate collection or revert model."
                    )
            except Exception as e:
                logger.debug(f"Could not check collection dimensions: {e}")
            return

        logger.info(f"Creating collection {COLLECTION_NAME}")
        self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_DIM,
                    distance=Distance.COSINE,
                    hnsw_config=HNSW_CONFIG,
                ),
                quantization_config=QUANTIZATION_CONFIG,
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=20000,  # Start indexing after 20K points
                ),
            )

        # Create payload indexes for fast filtering
        for field_name, field_type in PAYLOAD_INDEXES.items():
            try:
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=field_type,
                )
                logger.info(f"Created payload index: {field_name}")
            except Exception as e:
                # Index may already exist
                logger.debug(f"Payload index {field_name} may exist: {e}")

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 0) -> Tuple[int, int]:
        """
        Add chunks to the collection with token-aware embedding.

        Derives a VRAM budget from actual free GPU memory and batches
        documents by token content, not count. Short chunks get big
        batches (fast), long chunks get small batches (safe). No OOM,
        no corrupted CUDA contexts, no retry overhead.

        Also syncs to BM25 index for hybrid retrieval.

        Args:
            chunks: List of Chunk objects
            batch_size: Qdrant upsert batch size (0 = auto). Embed sub-batching
                       is managed separately via token-aware sizing.

        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not chunks:
            return (0, 0)

        # Qdrant upsert batch size (how many points per upsert call)
        if batch_size <= 0:
            try:
                from services.hardware import get_hardware_specs
                profile = get_hardware_specs().get("profile", "medium")
                batch_size = {"cpu": 32, "small": 64, "medium": 128, "large": 256}.get(profile, 128)
            except Exception:
                batch_size = 128

        total = len(chunks)
        successful = 0
        failed = 0

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            documents = [c.content for c in batch]

            # Token-aware embedding: batch by content size, not count
            embeddings = []
            embed_ok = True
            sub_batches = _token_aware_batches(documents)

            for sb_idx, sub_docs in enumerate(sub_batches):
                try:
                    embs = self.embedder.embed_documents(
                        sub_docs, batch_size=len(sub_docs)
                    )
                    embeddings.extend(embs)
                except Exception as e:
                    logger.error(
                        f"Embed failed for {self.topic} "
                        f"sub-batch {sb_idx}/{len(sub_batches)} "
                        f"({len(sub_docs)} docs): {e}"
                    )
                    embed_ok = False
                    break

            if not embed_ok or len(embeddings) != len(documents):
                failed += len(batch)
                logger.error(
                    f"Batch failed for {self.topic} "
                    f"(chunks {i}-{i+len(batch)}): embedding failed"
                )
                continue

            try:
                # Build Qdrant points, filtering out malformed vectors
                points = []
                valid_chunks = []
                skipped = 0
                for c, emb in zip(batch, embeddings):
                    # Validate: must be a list of finite floats with correct dim
                    if (not isinstance(emb, list) or len(emb) != VECTOR_DIM
                            or not all(isinstance(v, (int, float)) for v in emb[:3])
                            or any(math.isnan(v) or math.isinf(v) for v in emb)):
                        logger.warning(
                            f"Skipping chunk {c.chunk_id}: bad vector "
                            f"(type={type(emb).__name__}, len={len(emb) if isinstance(emb, list) else 'N/A'}, "
                            f"content_len={len(c.content)})"
                        )
                        skipped += 1
                        continue
                    points.append(PointStruct(
                        id=chunk_id_to_int(c.chunk_id),
                        vector=emb,
                        payload={
                            "content": c.content,
                            "topic": self.topic,
                            "chunk_id": c.chunk_id,
                            **c.metadata,
                        },
                    ))
                    valid_chunks.append(c)

                if skipped:
                    logger.warning(f"Filtered {skipped}/{len(batch)} chunks with bad vectors")
                    failed += skipped

                if points:
                    self.client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points,
                        wait=True,
                    )

                    # Sync to BM25 index for hybrid retrieval
                    try:
                        bm25_manager = get_bm25_manager()
                        bm25_index = bm25_manager.get_index(self.topic)
                        bm25_index.add_documents(
                            [c.chunk_id for c in valid_chunks],
                            [c.content for c in valid_chunks],
                            [c.metadata for c in valid_chunks],
                        )
                    except Exception as bm25_err:
                        logger.warning(f"BM25 sync failed for batch: {bm25_err}")

                successful += len(points)
                logger.info(f"Added {successful}/{total} chunks to {self.topic}")

            except Exception as e:
                failed += len(batch)
                logger.error(
                    f"Upsert failed for {self.topic} "
                    f"(chunks {i}-{i+len(batch)}): {e}"
                )

        logger.info(
            f"Collection {self.topic}: {successful} added, {failed} failed, "
            f"total now {self.count}"
        )
        return (successful, failed)

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query: Search query
            n_results: Number of results
            where: Optional metadata filter

        Returns:
            List of results with content, metadata, and scores
        """
        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Build filter - always filter by topic
        must_conditions = [
            FieldCondition(key="topic", match=MatchValue(value=self.topic))
        ]

        # Add additional filters from where clause
        if where:
            for key, value in where.items():
                if isinstance(value, dict) and "$in" in value:
                    # Handle $in operator for filter compatibility
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value["$in"]))
                    )
                else:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

        # Search with filter - qdrant-client 1.16+ uses query_points
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=n_results,
            query_filter=Filter(must=must_conditions),
            with_payload=True,
        )

        # Format results - Qdrant returns similarity scores directly (higher = more similar)
        formatted = []
        for hit in response.points:
            payload = hit.payload or {}
            # Exclude content and internal fields from metadata
            metadata = {k: v for k, v in payload.items() if k not in ("content", "chunk_id")}
            # Use original chunk_id from payload if available
            chunk_id = payload.get("chunk_id", str(hit.id))

            formatted.append({
                "id": chunk_id,
                "content": payload.get("content", ""),
                "metadata": metadata,
                "score": hit.score,  # Already 0-1 similarity for cosine
                "distance": 1 - hit.score,  # Cosine distance (1 - similarity)
                "topic": self.topic,
            })

        return formatted

    def delete_by_source(self, source: str):
        """Delete all chunks from a source file"""
        # Qdrant filter-based deletion
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(key="topic", match=MatchValue(value=self.topic)),
                    FieldCondition(key="source", match=MatchValue(value=source)),
                ]
            ),
        )

        # Sync BM25 index
        try:
            bm25_manager = get_bm25_manager()
            bm25_index = bm25_manager.get_index(self.topic)
            bm25_index.delete_by_source(source)
        except Exception as e:
            logger.warning(f"BM25 delete sync failed: {e}")

        logger.info(f"Deleted chunks from source: {source}")

    def clear(self):
        """Clear all chunks for this topic"""
        # Delete all points matching this topic
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(key="topic", match=MatchValue(value=self.topic))
                ]
            ),
        )
        logger.info(f"Cleared {self.topic}")

    @property
    def count(self) -> int:
        """Count chunks for this topic"""
        try:
            result = self.client.count(
                collection_name=COLLECTION_NAME,
                count_filter=Filter(
                    must=[
                        FieldCondition(key="topic", match=MatchValue(value=self.topic))
                    ]
                ),
            )
            return result.count
        except Exception:
            return 0


class KnowledgeBase:
    """
    Multi-topic knowledge base.

    Manages a single Qdrant collection with topic filtering.
    Supports dynamic topic discovery from filesystem.
    """

    # Default topics (used when knowledge_dir not provided)
    @staticmethod
    def _get_default_topics():
        try:
            from domain_loader import get_pipeline_config
            return get_pipeline_config().get("default_topics", ["general"])
        except Exception:
            return ["general"]

    def __init__(self, persist_dir: Path = None, knowledge_dir: Path = None):
        """
        Args:
            persist_dir: Ignored for Qdrant (persistence handled by Qdrant server)
            knowledge_dir: Optional directory containing topic subdirectories.
                          If provided, topics are discovered dynamically.
        """
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.knowledge_dir = Path(knowledge_dir) if knowledge_dir else None

        # Initialize Qdrant client
        self.client = get_qdrant_client()

        # Lazy embedder: defers ONNX CUDA load until first embed call.
        # Critical for three-phase ingest — vision server needs exclusive GPU in phase 2.
        self.embedder = _LazyEmbedder()

        # Lazy-loaded topic stores
        self.stores: Dict[str, TopicStore] = {}

        # Pre-load existing topics from collection
        self._load_existing_stores()

        logger.info("KnowledgeBase initialized with Qdrant")

    def _load_existing_stores(self):
        """Load stores for existing topics in the collection"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if COLLECTION_NAME not in collection_names:
                return

            # Scroll through collection to find unique topics
            # Use scroll with limit to get sample of points
            scroll_result = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=["topic"],
            )

            topics = set()
            for point in scroll_result[0]:
                if point.payload and "topic" in point.payload:
                    topics.add(point.payload["topic"])

            for topic in topics:
                self.stores[topic] = TopicStore(topic, self.client, self.embedder)

        except Exception as e:
            logger.warning(f"Failed to load existing topics: {e}")

    def discover_topics(self) -> List[str]:
        """
        Discover available topics.

        Priority:
        1. Filesystem directories (if knowledge_dir provided)
        2. Existing topics in Qdrant collection
        3. Default topics
        """
        topics = set()

        # From filesystem
        if self.knowledge_dir and self.knowledge_dir.exists():
            for d in self.knowledge_dir.iterdir():
                if d.is_dir():
                    topics.add(d.name)

        # From existing stores
        topics.update(self.stores.keys())

        # Fallback to defaults if nothing found
        if not topics:
            topics = set(self._get_default_topics())

        return sorted(list(topics))

    @property
    def TOPICS(self) -> List[str]:
        """Property for backward compatibility"""
        return self.discover_topics()

    def get_store(self, topic: str) -> TopicStore:
        """
        Get or create store for a topic.

        Supports dynamic topic creation - new topics are created on demand.
        """
        if topic not in self.stores:
            # Create new topic store on demand
            logger.info(f"Creating new topic store: {topic}")
            self.stores[topic] = TopicStore(topic, self.client, self.embedder)
        return self.stores[topic]

    def add_chunks(self, topic: str, chunks: List[Chunk]):
        """Add chunks to a specific topic"""
        store = self.get_store(topic)
        store.add_chunks(chunks)

    def search(
        self,
        query: str,
        topics: Optional[List[str]] = None,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search across topics.

        Args:
            query: Search query
            topics: Topics to search (default: all)
            n_results: Results per topic

        Returns:
            Combined results sorted by score
        """
        topics = topics or self.TOPICS

        all_results = []
        for topic in topics:
            if topic in self.stores:
                results = self.stores[topic].search(query, n_results=n_results)
                all_results.extend(results)

        # Sort by score (highest first)
        all_results.sort(key=lambda x: x["score"], reverse=True)

        return all_results[:n_results]

    def get_stats(self) -> Dict[str, int]:
        """Get chunk counts per topic"""
        return {topic: store.count for topic, store in self.stores.items()}

    def clear_topic(self, topic: str):
        """Clear a single topic"""
        self.get_store(topic).clear()

    def clear_all(self):
        """Clear all topics"""
        for store in self.stores.values():
            store.clear()


# Singleton
_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base(persist_dir: Path = None) -> KnowledgeBase:
    """Get or create singleton knowledge base"""
    global _knowledge_base

    if _knowledge_base is None:
        if persist_dir is None:
            persist_dir = Path("/app/data/cohesionn_db")
        _knowledge_base = KnowledgeBase(persist_dir)

    return _knowledge_base
