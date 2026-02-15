"""
Phase 2: Embedding Shootout
============================
Re-embed existing chunks into separate Qdrant collections per embedder.

Flow:
  1. Stop chat LLM to free VRAM (biggest consumer, ~17 GB)
  2. Extract chunks from main 'cohesionn' collection (content + metadata, no vectors)
  3. Per embedder: load model → embed → upsert → query → score → UNLOAD model
  4. Restart chat LLM when done
  5. Defer collection cleanup until cross-matrix phase completes (or --keep-collections)

Memory safety:
  - Only ONE embedding model loaded at a time
  - Shared reranker instance across all models (~1 GB)
  - VRAM budget check before each model load
  - Explicit unload() + gc + torch.cuda.empty_cache() between swaps
  - Batch size capped at 32 to limit peak memory
"""

import time
import uuid
from typing import Any, Dict, List

from ..config import ModelSpec
from .base import BasePhase, PhaseResult

EMBED_BATCH_SIZE = 32
VRAM_SAFETY_MARGIN_MB = 512


class EmbeddingPhase(BasePhase):
    """Evaluate embedding models by creating separate Qdrant collections."""

    phase_name = "embedding"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._created_collections: List[str] = []

    def run(self) -> List[PhaseResult]:
        self._begin()
        results: List[PhaseResult] = []

        # Step 1: Extract chunks from main collection (no GPU needed)
        self._log("Extracting chunks from main collection...")
        chunks = self._extract_chunks()
        self._log(f"Extracted {len(chunks)} chunks")

        if not chunks:
            self._log("No chunks found in main collection. Aborting.")
            return results

        # Step 2: Stop chat LLM to free VRAM for embedding models
        llm_was_running = self._stop_llm_server()

        # Step 3: Create a single shared reranker (~1 GB, stays loaded the whole phase)
        self._log("Loading shared reranker...")
        from tools.cohesionn.reranker import BGEReranker
        shared_reranker = BGEReranker()
        self._log_vram("after reranker load")

        # Step 4: Test each embedder ONE AT A TIME
        embedders = self.config.filter_models("embedding")
        for i, model in enumerate(embedders):
            self._log(f"[{i + 1}/{len(embedders)}] Testing {model.name}...")
            self._log_vram("before model load")
            t0 = time.time()

            try:
                result = self._test_embedder(model, chunks, shared_reranker)
                results.append(result)
                self._log(
                    f"  → composite={result.aggregate.get('avg_composite', 0):.3f}  "
                    f"keywords={result.aggregate.get('avg_keyword_hits', 0):.1%}  "
                    f"{time.time() - t0:.1f}s"
                )
            except Exception as e:
                self._log(f"  → ERROR: {e}")
                results.append(PhaseResult(
                    phase=self.phase_name,
                    variant_name=model.short_name,
                    model_spec=model.to_dict(),
                    error=str(e),
                    duration_s=time.time() - t0,
                ))
            finally:
                # Always clean up GPU between models
                self._gpu_cleanup()
                self._log_vram("after cleanup")

        # Step 5: Cleanup shared reranker
        del shared_reranker
        self._gpu_cleanup()

        # Step 6: Restart chat LLM if it was running before
        if llm_was_running:
            self._restart_llm_server()

        return results

    def _stop_llm_server(self) -> bool:
        """Stop the chat LLM server to free VRAM. Returns True if it was running."""
        try:
            from utils.llm import get_server_manager
            mgr = get_server_manager()

            # Check if running (sync — just check if process exists)
            proc = mgr._processes.get("chat")
            if proc is None or proc.poll() is not None:
                self._log("Chat LLM not running, no need to stop")
                return False

            self._log("Stopping chat LLM to free VRAM...")
            self._log_vram("before LLM stop")
            mgr.stop("chat")

            # Wait for GPU memory to actually release
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            time.sleep(3)  # llama-server needs a moment to release VRAM

            self._log_vram("after LLM stop")
            return True
        except Exception as e:
            self._log(f"Warning: Could not stop LLM server: {e}")
            return False

    def _restart_llm_server(self):
        """Restart the chat LLM server after embedding phase."""
        try:
            from utils.llm import get_server_manager
            mgr = get_server_manager()

            self._log("Restarting chat LLM...")
            mgr.start("chat")

            # Poll for health synchronously (we're in a sync context)
            import httpx
            from services.llm_config import SLOTS
            port = SLOTS["chat"].port
            url = f"http://localhost:{port}/health"

            for attempt in range(120):  # 2 minutes max
                try:
                    r = httpx.get(url, timeout=3.0)
                    if r.status_code == 200:
                        self._log(f"Chat LLM healthy after {attempt}s")
                        return
                except Exception:
                    pass
                time.sleep(1)

            self._log("Warning: Chat LLM did not become healthy within 120s")
        except Exception as e:
            self._log(f"Warning: Could not restart LLM server: {e}")

    def _extract_chunks(self) -> List[Dict[str, Any]]:
        """Extract all chunks from the main Qdrant collection (no vectors)."""
        from qdrant_client import QdrantClient
        import os

        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url)

        collection_name = "cohesionn"
        chunks = []
        offset = None

        try:
            while True:
                result = client.scroll(
                    collection_name=collection_name,
                    limit=500,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points, next_offset = result
                for point in points:
                    payload = point.payload or {}
                    chunks.append({
                        "id": str(point.id),
                        "content": payload.get("content", ""),
                        "metadata": {
                            k: v for k, v in payload.items()
                            if k != "content"
                        },
                    })
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as e:
            self._log(f"Error extracting chunks: {e}")

        return chunks

    def _test_embedder(
        self,
        model: ModelSpec,
        chunks: List[Dict[str, Any]],
        shared_reranker,
    ) -> PhaseResult:
        """Create collection, embed chunks, query, score. One model at a time."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance,
            HnswConfigDiff,
            PointStruct,
            VectorParams,
        )
        from tools.cohesionn.embeddings import UniversalEmbedder
        import os

        t0 = time.time()

        # VRAM budget check — estimate model size from dimension (rough heuristic)
        estimated_mb = (model.dimension or 1024) * 2  # very rough: 2 MB per 1024 dims
        if not self._check_vram_budget(estimated_mb + VRAM_SAFETY_MARGIN_MB):
            raise RuntimeError(
                f"Insufficient VRAM for {model.name}. "
                f"Need ~{estimated_mb + VRAM_SAFETY_MARGIN_MB} MB free."
            )

        # Create fresh embedder
        embedder = UniversalEmbedder(
            model_name=model.hf_id,
            trust_remote_code=model.trust_remote_code,
        )
        dimension = model.dimension or embedder.dimension
        self._log_vram("after embedder load")

        # Create shootout collection
        collection_name = f"shootout_{model.short_name}"
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url)

        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
        )
        self._created_collections.append(collection_name)
        self._log(f"  Created collection {collection_name} (dim={dimension})")

        # Batch embed and upsert — small batches to limit peak memory
        contents = [c["content"] for c in chunks]
        for batch_start in range(0, len(contents), EMBED_BATCH_SIZE):
            batch_end = min(batch_start + EMBED_BATCH_SIZE, len(contents))
            batch_contents = contents[batch_start:batch_end]
            batch_chunks = chunks[batch_start:batch_end]

            vectors = embedder.embed_documents(batch_contents, batch_size=EMBED_BATCH_SIZE)

            points = []
            for chunk, vector in zip(batch_chunks, vectors):
                payload = {"content": chunk["content"]}
                payload.update(chunk.get("metadata", {}))
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload,
                ))

            client.upsert(collection_name=collection_name, points=points)

            if batch_start % (EMBED_BATCH_SIZE * 10) == 0:
                self._log(f"  Embedded {batch_end}/{len(contents)} chunks")

        self._log(f"  Upserted {len(contents)} chunks")

        # Query each benchmark query against this collection
        retrieval_results: Dict[str, Any] = {}

        for q in self.queries:
            start = time.time()
            try:
                query_vector = embedder.embed_query(q.query)
                search_results = client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=self.config.initial_k,
                    with_payload=True,
                ).points

                search_chunks = []
                for sr in search_results:
                    payload = sr.payload or {}
                    search_chunks.append({
                        "content": payload.get("content", ""),
                        "score": sr.score,
                        "source": payload.get("source", "unknown"),
                        "page": payload.get("page"),
                        "topic": payload.get("topic", ""),
                        "is_raptor": payload.get("raptor_level", 0) > 0,
                        "is_graph_result": payload.get("is_graph_result", False),
                    })

                # Rerank using SHARED reranker (no new allocation)
                ranked = shared_reranker.rerank(
                    query=q.query,
                    results=search_chunks,
                    top_k=self.config.top_k,
                )
                elapsed_ms = (time.time() - start) * 1000
                retrieval_results[q.id] = {
                    "chunks": ranked,
                    "latency_ms": elapsed_ms,
                }
            except Exception as e:
                retrieval_results[q.id] = {
                    "chunks": [],
                    "latency_ms": (time.time() - start) * 1000,
                    "error": str(e),
                }

        # Score
        aggregate, per_query = self._score_all(model.short_name, retrieval_results)

        # CRITICAL: Properly unload embedder to free GPU memory
        embedder.unload()
        self._log_vram("after embedder unload")

        duration = time.time() - t0

        return self._make_result(
            variant_name=model.short_name,
            model_spec=model,
            aggregate=aggregate,
            per_query=per_query,
            duration_s=duration,
            metadata={"collection": collection_name, "n_chunks": len(chunks)},
        )

    def get_created_collections(self) -> List[str]:
        """Return list of shootout collections created (for cross-matrix)."""
        return list(self._created_collections)

    def cleanup_collections(self):
        """Delete all shootout collections."""
        from qdrant_client import QdrantClient
        import os

        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url)

        for name in self._created_collections:
            try:
                client.delete_collection(name)
                self._log(f"Deleted collection {name}")
            except Exception as e:
                self._log(f"Failed to delete {name}: {e}")
        self._created_collections.clear()
