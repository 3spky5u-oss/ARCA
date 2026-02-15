"""
Layer E: Embedding Model Shootout
=================================
Swap embedding models while holding chunking + reranker constant.
Adapted from tools.cohesionn.benchmark.phases.embedding.

Each model gets its own Qdrant collection (dimensions may differ).
Only ONE embedding model loaded at a time — explicit unload() between swaps.

Flow:
  1. Load optimal chunking config from L0
  2. Chunk corpus (no GPU)
  3. Stop chat LLM to free VRAM
  4. Load shared reranker (~1 GB, constant across all tests)
  5. Per embedding model:
     a. Load embedder
     b. Embed all chunks → create Qdrant collection → upsert
     c. Run 40 queries (embed_query → search → rerank with shared reranker)
     d. Score with ScoringEngine
     e. Unload embedder, delete collection
  6. Restart chat LLM
  7. Rank models, save optimal
"""
import gc
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from benchmark.layers.base import BaseLayer, LayerResult

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 32
VRAM_SAFETY_MARGIN_MB = 512


class EmbeddingShootoutLayer(BaseLayer):
    """Benchmark embedding models on the synthetic corpus."""

    LAYER_NAME = "layer_embed"

    def execute(self, result: LayerResult) -> LayerResult:
        from tools.cohesionn.benchmark.config import EMBEDDING_CANDIDATES, ModelSpec
        from tools.cohesionn.benchmark.metrics import ScoringEngine
        from tools.cohesionn.chunker import SemanticChunker
        from benchmark.config import ChunkingConfig
        from benchmark.queries.battery import BenchmarkQuery

        # Step 1: Load L0 optimal chunking config + queries + corpus
        optimal_chunking = self.load_optimal_config("layer0_chunking")
        if not optimal_chunking:
            result.errors.append("No optimal chunking config from L0. Run layer0 first.")
            return result

        chunk_cfg = ChunkingConfig.from_dict(optimal_chunking)
        logger.info(f"Using chunking config: {chunk_cfg.config_id}")

        queries_path = Path(self.config.output_dir) / "layer0_chunking" / "queries.json"
        if not queries_path.exists():
            result.errors.append("No queries from L0. Run layer0 first.")
            return result

        raw_queries = json.loads(queries_path.read_text(encoding="utf-8"))
        queries = [BenchmarkQuery.from_dict(q) for q in raw_queries]
        logger.info(f"Loaded {len(queries)} benchmark queries")

        corpus_dir = Path(self.config.output_dir) / "layer0_chunking" / "corpus_md"
        md_files = sorted(str(f) for f in corpus_dir.glob("*.md"))
        if not md_files:
            result.errors.append("No corpus markdown files from L0.")
            return result

        # Step 2: Chunk corpus (no GPU — same chunking for all models)
        logger.info(f"Chunking {len(md_files)} files with {chunk_cfg.config_id}...")
        chunker = SemanticChunker(
            chunk_size=chunk_cfg.chunk_size,
            chunk_overlap=chunk_cfg.chunk_overlap,
            context_prefix_enabled=chunk_cfg.context_prefix,
        )

        all_chunks = []
        for md_path in md_files:
            path = Path(md_path)
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue
            metadata = {
                "source": str(path),
                "file_name": path.name,
                "title": path.stem,
            }
            chunks = chunker.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks")
        # Chunk is a dataclass with .content and .metadata
        chunk_contents = [c.content for c in all_chunks]
        chunk_metadata = [{"content": c.content, **c.metadata} for c in all_chunks]

        # Step 3: Stop chat LLM to free VRAM
        llm_was_running = self._stop_llm_server()

        # Step 4: Load shared reranker (constant across all embedding tests)
        logger.info("Loading shared reranker...")
        from tools.cohesionn.reranker import BGEReranker
        shared_reranker = BGEReranker()

        # Step 5: Test each embedding model
        models = list(EMBEDDING_CANDIDATES)
        result.configs_total = len(models)
        scorer = ScoringEngine()
        all_scores = []

        for i, model in enumerate(models):
            if self.checkpoint.is_completed(self.LAYER_NAME, model.short_name):
                result.configs_skipped += 1
                saved = self.checkpoint.get_result(self.LAYER_NAME, model.short_name)
                if saved:
                    all_scores.append(saved)
                logger.info(f"[{i+1}/{len(models)}] Skipping {model.name} (checkpointed)")
                continue

            logger.info(f"[{i+1}/{len(models)}] Testing {model.name}...")
            t0 = time.time()

            try:
                model_result = self._test_embedder(
                    model, chunk_contents, chunk_metadata, queries,
                    shared_reranker, scorer,
                )
                all_scores.append(model_result)
                self.checkpoint.mark_completed(self.LAYER_NAME, model.short_name, model_result)
                result.configs_completed += 1

                composite = model_result.get("aggregate", {}).get("avg_composite", 0)
                logger.info(f"  {model.name}: composite={composite:.4f} ({time.time()-t0:.1f}s)")

            except Exception as e:
                result.errors.append(f"{model.name}: {e}")
                logger.error(f"  {model.name} FAILED: {e}", exc_info=True)
            finally:
                self._gpu_cleanup()

        # Step 6: Cleanup shared reranker
        shared_reranker.unload()
        del shared_reranker
        self._gpu_cleanup()

        # Step 7: Restart chat LLM
        if llm_was_running:
            if not self._restart_llm_server():
                result.errors.append(
                    "Chat LLM failed to restart after embedding shootout. "
                    "May need manual restart via admin panel."
                )

        # Step 8: Rank and save
        all_scores.sort(
            key=lambda x: x.get("aggregate", {}).get("avg_composite", 0),
            reverse=True,
        )

        summary = {
            "total_models": len(models),
            "completed": result.configs_completed,
            "skipped": result.configs_skipped,
            "chunking_config": chunk_cfg.to_dict(),
            "ranking": [
                {
                    "rank": i + 1,
                    "model_name": s["model"]["name"],
                    "hf_id": s["model"]["hf_id"],
                    "short_name": s["model"]["short_name"],
                    "composite": s["aggregate"]["avg_composite"],
                    "keyword_hits": s["aggregate"].get("avg_keyword_hits", 0),
                    "entity_hits": s["aggregate"].get("avg_entity_hits", 0),
                    "mrr": s["aggregate"].get("avg_mrr", 0),
                    "duration_s": s.get("duration_s", 0),
                }
                for i, s in enumerate(all_scores)
            ],
        }

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if all_scores:
            best = all_scores[0]
            optimal_path = self.output_dir / "optimal_config.json"
            optimal_path.write_text(
                json.dumps(best["model"], indent=2), encoding="utf-8"
            )
            result.best_config_id = best["model"]["short_name"]
            result.best_score = best["aggregate"]["avg_composite"]

        result.summary = summary
        return result

    def _test_embedder(
        self,
        model,
        chunk_contents: List[str],
        chunk_metadata: List[Dict],
        queries,
        shared_reranker,
        scorer,
    ) -> Dict[str, Any]:
        """Test a single embedding model: embed, ingest, query, score."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams
        from tools.cohesionn.embeddings import UniversalEmbedder

        t0 = time.time()

        # Create embedder
        embedder = UniversalEmbedder(
            model_name=model.hf_id,
            trust_remote_code=model.trust_remote_code,
        )
        dimension = model.dimension or embedder.dimension
        logger.info(f"  Loaded {model.name} (dim={dimension})")

        # Create Qdrant collection
        collection_name = f"bench_embed_{model.short_name}"
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url)

        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
        )

        # Batch embed + upsert
        for batch_start in range(0, len(chunk_contents), EMBED_BATCH_SIZE):
            batch_end = min(batch_start + EMBED_BATCH_SIZE, len(chunk_contents))
            batch_texts = chunk_contents[batch_start:batch_end]
            batch_meta = chunk_metadata[batch_start:batch_end]

            vectors = embedder.embed_documents(batch_texts, batch_size=EMBED_BATCH_SIZE)

            points = []
            for meta, vector in zip(batch_meta, vectors):
                payload = {
                    "content": meta.get("content", ""),
                    "source": meta.get("source", ""),
                    "file_name": meta.get("file_name", ""),
                    "title": meta.get("title", ""),
                }
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload,
                ))
            client.upsert(collection_name=collection_name, points=points)

            if batch_start % (EMBED_BATCH_SIZE * 5) == 0 and batch_start > 0:
                logger.info(f"  Embedded {batch_end}/{len(chunk_contents)} chunks")

        logger.info(f"  Upserted {len(chunk_contents)} chunks")

        # Query + score
        per_query_metrics = []
        for q in queries:
            q_start = time.time()
            try:
                query_vector = embedder.embed_query(q.query)
                search_results = client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=15,  # retrieve more, let reranker pick top_k
                    with_payload=True,
                ).points

                search_chunks = []
                for sr in search_results:
                    payload = sr.payload or {}
                    search_chunks.append({
                        "content": payload.get("content", ""),
                        "score": sr.score,
                        "source": payload.get("source", "unknown"),
                        "topic": "",
                        "is_raptor": False,
                        "is_graph_result": False,
                    })

                # Rerank with shared reranker
                ranked = shared_reranker.rerank(
                    query=q.query,
                    results=search_chunks,
                    top_k=self.config.top_k,
                )
                latency = (time.time() - q_start) * 1000

                metrics = scorer.score_retrieval(
                    query_id=q.id,
                    tier=q.tier,
                    chunks=ranked,
                    latency_ms=latency,
                    expect_keywords=q.expect_keywords,
                    expect_entities=q.expect_entities,
                )
                per_query_metrics.append(metrics)

            except Exception as e:
                logger.warning(f"  Query {q.id} failed: {e}")

        # Aggregate
        agg = scorer.aggregate(model.short_name, per_query_metrics)

        # Cleanup
        embedder.unload()
        try:
            client.delete_collection(collection_name)
            logger.info(f"  Deleted collection {collection_name}")
        except Exception:
            pass

        duration = time.time() - t0

        return {
            "model": model.to_dict(),
            "aggregate": agg.to_dict(),
            "per_query": [m.to_dict() for m in per_query_metrics],
            "duration_s": round(duration, 1),
            "n_chunks": len(chunk_contents),
        }

    def _gpu_cleanup(self):
        """Release GPU memory between model swaps."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def _stop_llm_server(self) -> bool:
        """Stop the chat LLM server to free VRAM."""
        try:
            from services.llm_server_manager import get_server_manager
            mgr = get_server_manager()
            proc = mgr._processes.get("chat")
            if proc is None or proc.poll() is not None:
                logger.info("Chat LLM not running, no need to stop")
                return False

            logger.info("Stopping chat LLM to free VRAM...")
            mgr.stop("chat")
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass

            # Wait for port release before returning
            port = int(os.environ.get("LLM_CHAT_PORT", "8081"))
            import socket
            for wait in range(30):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    result = sock.connect_ex(("127.0.0.1", port))
                    sock.close()
                    if result != 0:
                        break
                except Exception:
                    break
                time.sleep(1)
            else:
                logger.warning(f"Port {port} still in use after 30s wait")

            time.sleep(2)
            return True
        except Exception as e:
            logger.warning(f"Could not stop LLM server: {e}")
            return False

    def _restart_llm_server(self) -> bool:
        """Restart the chat LLM server. Returns True if healthy."""
        try:
            from services.llm_server_manager import get_server_manager
            mgr = get_server_manager()
            logger.info("Restarting chat LLM...")
            mgr.start("chat")

            import httpx
            port = os.environ.get("LLM_CHAT_PORT", "8081")
            url = f"http://localhost:{port}/health"

            for attempt in range(120):
                try:
                    r = httpx.get(url, timeout=3.0)
                    if r.status_code == 200:
                        logger.info(f"Chat LLM healthy after {attempt}s")
                        return True
                except Exception as e:
                    if attempt % 15 == 0 and attempt > 0:
                        logger.debug(f"Chat LLM health check attempt {attempt}: {e}")
                time.sleep(1)

            logger.error("Chat LLM did not become healthy within 120s")
            return False
        except Exception as e:
            logger.error(f"Could not restart LLM server: {e}", exc_info=True)
            return False
