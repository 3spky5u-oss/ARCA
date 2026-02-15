"""
Layer X: Cross-Model Sweep
===========================
Top-N chunking configs x top-N embedding models x top-N reranker models.

Reads top-3 results from L0, layer_embed, and layer_rerank summary files,
then tests every combination (up to 27) end-to-end.

Only ONE embedding model + ONE reranker loaded at a time.
GPU cleanup between every model swap.

Flow:
  1. Load top-3 chunking configs from L0 summary.json ranking
  2. Load top-3 embedding models from layer_embed summary.json ranking
  3. Load top-3 reranker models from layer_rerank summary.json ranking
  4. Load queries from L0's queries.json
  5. Load corpus markdown from L0's corpus_md/
  6. Stop chat LLM to free VRAM
  7. Per combination (chunk_cfg x embed_model x reranker_model):
     a. Chunk corpus with the chunking config
     b. Create UniversalEmbedder with the embedding model
     c. Embed chunks, create Qdrant collection, upsert
     d. Create BGEReranker with the reranker model
     e. Run 40 queries: embed_query -> search -> rerank
     f. Score with ScoringEngine
     g. Unload embedder & reranker, delete collection
     h. Checkpoint
  8. Restart chat LLM
  9. Rank all combos, save optimal
"""
import gc
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.layers.base import BaseLayer, LayerResult

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 32
TOP_N = 3


class CrossModelSweepLayer(BaseLayer):
    """Benchmark all combinations of top chunking x embedding x reranker."""

    LAYER_NAME = "layer_cross"

    def execute(self, result: LayerResult) -> LayerResult:
        from tools.cohesionn.benchmark.config import (
            EMBEDDING_CANDIDATES,
            RERANKER_CANDIDATES,
            ModelSpec,
        )
        from tools.cohesionn.benchmark.metrics import ScoringEngine
        from tools.cohesionn.chunker import SemanticChunker
        from benchmark.config import ChunkingConfig
        from benchmark.queries.battery import BenchmarkQuery

        # ── Step 1: Load top-3 chunking configs from L0 ──────────────────
        l0_summary = self._load_layer_summary("layer0_chunking")
        if not l0_summary:
            result.errors.append("No L0 summary.json found. Run layer0 first.")
            return result

        l0_ranking = l0_summary.get("ranking", [])
        if not l0_ranking:
            result.errors.append("L0 summary has no ranking.")
            return result

        chunk_cfgs: List[ChunkingConfig] = []
        for entry in l0_ranking[:TOP_N]:
            chunk_cfgs.append(ChunkingConfig(
                config_id=entry["config_id"],
                chunk_size=entry["chunk_size"],
                chunk_overlap=entry["chunk_overlap"],
                context_prefix=entry["context_prefix"],
            ))
        logger.info(f"Loaded {len(chunk_cfgs)} chunking configs from L0")

        # ── Step 2: Load top-3 embedding models from layer_embed ──────────
        embed_summary = self._load_layer_summary("layer_embed")
        if not embed_summary:
            result.errors.append("No layer_embed summary.json found. Run embed layer first.")
            return result

        embed_ranking = embed_summary.get("ranking", [])
        if not embed_ranking:
            result.errors.append("layer_embed summary has no ranking.")
            return result

        embed_models: List[ModelSpec] = []
        all_embed_by_short = {m.short_name: m for m in EMBEDDING_CANDIDATES}
        for entry in embed_ranking[:TOP_N]:
            short = entry.get("short_name", "")
            if short in all_embed_by_short:
                embed_models.append(all_embed_by_short[short])
            else:
                # Reconstruct from summary data
                embed_models.append(ModelSpec(
                    name=entry.get("model_name", short),
                    hf_id=entry.get("hf_id", short),
                    short_name=short,
                    model_type="embedding",
                    trust_remote_code=entry.get("trust_remote_code", False),
                ))
        logger.info(f"Loaded {len(embed_models)} embedding models from layer_embed")

        # ── Step 3: Load top-3 reranker models from layer_rerank ──────────
        rerank_summary = self._load_layer_summary("layer_rerank")
        if not rerank_summary:
            result.errors.append("No layer_rerank summary.json found. Run rerank layer first.")
            return result

        rerank_ranking = rerank_summary.get("ranking", [])
        if not rerank_ranking:
            result.errors.append("layer_rerank summary has no ranking.")
            return result

        rerank_models: List[ModelSpec] = []
        all_rerank_by_short = {m.short_name: m for m in RERANKER_CANDIDATES}
        for entry in rerank_ranking[:TOP_N]:
            short = entry.get("short_name", "")
            if short in all_rerank_by_short:
                rerank_models.append(all_rerank_by_short[short])
            else:
                rerank_models.append(ModelSpec(
                    name=entry.get("model_name", short),
                    hf_id=entry.get("hf_id", short),
                    short_name=short,
                    model_type="reranker",
                    trust_remote_code=entry.get("trust_remote_code", False),
                    max_length=entry.get("max_length"),
                ))
        logger.info(f"Loaded {len(rerank_models)} reranker models from layer_rerank")

        # ── Step 4: Load queries from L0 ──────────────────────────────────
        queries_path = Path(self.config.output_dir) / "layer0_chunking" / "queries.json"
        if not queries_path.exists():
            result.errors.append("No queries from L0. Run layer0 first.")
            return result

        raw_queries = json.loads(queries_path.read_text(encoding="utf-8"))
        queries = [BenchmarkQuery.from_dict(q) for q in raw_queries]
        logger.info(f"Loaded {len(queries)} benchmark queries")

        # ── Step 5: Load corpus markdown from L0 ─────────────────────────
        corpus_dir = Path(self.config.output_dir) / "layer0_chunking" / "corpus_md"
        md_files = sorted(str(f) for f in corpus_dir.glob("*.md"))
        if not md_files:
            result.errors.append("No corpus markdown files from L0.")
            return result

        logger.info(f"Found {len(md_files)} corpus markdown files")

        # ── Step 6: Stop chat LLM to free VRAM ───────────────────────────
        llm_was_running = self._stop_llm_server()

        # ── Step 7: Run all combinations ──────────────────────────────────
        combos = []
        for chunk_cfg in chunk_cfgs:
            for embed_model in embed_models:
                for rerank_model in rerank_models:
                    combos.append((chunk_cfg, embed_model, rerank_model))

        result.configs_total = len(combos)
        logger.info(
            f"Cross-model sweep: {len(chunk_cfgs)} chunking x "
            f"{len(embed_models)} embed x {len(rerank_models)} rerank = "
            f"{len(combos)} combinations"
        )

        scorer = ScoringEngine()
        all_scores: List[Dict[str, Any]] = []

        for i, (chunk_cfg, embed_model, rerank_model) in enumerate(combos):
            combo_key = f"{chunk_cfg.config_id}_{embed_model.short_name}_{rerank_model.short_name}"

            # Check checkpoint
            if self.checkpoint.is_completed(self.LAYER_NAME, combo_key):
                result.configs_skipped += 1
                saved = self.checkpoint.get_result(self.LAYER_NAME, combo_key)
                if saved:
                    all_scores.append(saved)
                logger.info(f"[{i+1}/{len(combos)}] Skipping {combo_key} (checkpointed)")
                continue

            logger.info(
                f"[{i+1}/{len(combos)}] Testing {combo_key} "
                f"(chunk={chunk_cfg.config_id}, embed={embed_model.short_name}, "
                f"rerank={rerank_model.short_name})"
            )
            t0 = time.time()

            try:
                combo_result = self._test_combo(
                    chunk_cfg=chunk_cfg,
                    embed_model=embed_model,
                    rerank_model=rerank_model,
                    md_files=md_files,
                    queries=queries,
                    scorer=scorer,
                    combo_key=combo_key,
                )
                all_scores.append(combo_result)
                self.checkpoint.mark_completed(self.LAYER_NAME, combo_key, combo_result)
                result.configs_completed += 1

                composite = combo_result.get("aggregate", {}).get("avg_composite", 0)
                logger.info(f"  {combo_key}: composite={composite:.4f} ({time.time()-t0:.1f}s)")

            except Exception as e:
                result.errors.append(f"{combo_key}: {e}")
                logger.error(f"  {combo_key} FAILED: {e}", exc_info=True)
            finally:
                self._gpu_cleanup()

        # ── Step 8: Restart chat LLM ─────────────────────────────────────
        if llm_was_running:
            if not self._restart_llm_server():
                result.errors.append(
                    "Chat LLM failed to restart after cross-model sweep. "
                    "May need manual restart via admin panel."
                )

        # ── Step 9: Rank and save ─────────────────────────────────────────
        all_scores.sort(
            key=lambda x: x.get("aggregate", {}).get("avg_composite", 0),
            reverse=True,
        )

        summary = {
            "total_combos": len(combos),
            "completed": result.configs_completed,
            "skipped": result.configs_skipped,
            "n_chunking_configs": len(chunk_cfgs),
            "n_embed_models": len(embed_models),
            "n_rerank_models": len(rerank_models),
            "ranking": [
                {
                    "rank": i + 1,
                    "combo_key": s["combo_key"],
                    "chunk_config_id": s["chunk_config"]["config_id"],
                    "embed_model_name": s["embed_model"]["name"],
                    "embed_short": s["embed_model"]["short_name"],
                    "rerank_model_name": s["rerank_model"]["name"],
                    "rerank_short": s["rerank_model"]["short_name"],
                    "composite": s["aggregate"]["avg_composite"],
                    "keyword_hits": s["aggregate"].get("avg_keyword_hits", 0),
                    "entity_hits": s["aggregate"].get("avg_entity_hits", 0),
                    "mrr": s["aggregate"].get("avg_mrr", 0),
                    "duration_s": s.get("duration_s", 0),
                    "n_chunks": s.get("n_chunks", 0),
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
                json.dumps({
                    "combo_key": best["combo_key"],
                    "chunk_config": best["chunk_config"],
                    "embed_model": best["embed_model"],
                    "rerank_model": best["rerank_model"],
                }, indent=2),
                encoding="utf-8",
            )
            result.best_config_id = best["combo_key"]
            result.best_score = best["aggregate"]["avg_composite"]

        result.summary = summary
        return result

    def _test_combo(
        self,
        chunk_cfg,
        embed_model,
        rerank_model,
        md_files: List[str],
        queries,
        scorer,
        combo_key: str,
    ) -> Dict[str, Any]:
        """Test a single (chunking, embedding, reranker) combination."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams
        from tools.cohesionn.embeddings import UniversalEmbedder
        from tools.cohesionn.reranker import BGEReranker, RERANKER_MAX_LENGTH
        from tools.cohesionn.chunker import SemanticChunker

        t0 = time.time()

        # ── 1. Chunk corpus ───────────────────────────────────────────────
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

        # Chunk is a dataclass with .content and .metadata
        chunk_contents = [c.content for c in all_chunks]
        chunk_metadata = [{"content": c.content, **c.metadata} for c in all_chunks]

        logger.info(f"  Chunked {len(chunk_contents)} chunks with {chunk_cfg.config_id}")

        # ── 2. Create embedder ────────────────────────────────────────────
        embedder = UniversalEmbedder(
            model_name=embed_model.hf_id,
            trust_remote_code=embed_model.trust_remote_code,
        )
        dimension = embed_model.dimension or embedder.dimension
        logger.info(f"  Loaded embedder {embed_model.name} (dim={dimension})")

        # ── 3. Embed + create Qdrant collection ──────────────────────────
        collection_name = f"bench_cross_{combo_key}"
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

        # Wait for collection to be ready (avoid 404 race condition)
        for _retry in range(10):
            try:
                client.get_collection(collection_name)
                break
            except Exception:
                time.sleep(0.5)

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

        # ── 4. Create reranker ────────────────────────────────────────────
        reranker = BGEReranker(
            model_name=rerank_model.hf_id,
            max_length=rerank_model.max_length or RERANKER_MAX_LENGTH,
            trust_remote_code=rerank_model.trust_remote_code,
        )
        logger.info(f"  Loaded reranker {rerank_model.name}")

        # ── 5. Query + score ──────────────────────────────────────────────
        per_query_metrics = []
        for q in queries:
            q_start = time.time()
            try:
                query_vector = embedder.embed_query(q.query)
                search_results = client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=15,
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

                ranked = reranker.rerank(
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
        agg = scorer.aggregate(combo_key, per_query_metrics)

        # ── 6. Cleanup embedder, reranker, collection ─────────────────────
        embedder.unload()
        reranker.unload()
        del embedder
        del reranker
        self._gpu_cleanup()

        try:
            client.delete_collection(collection_name)
            logger.info(f"  Deleted collection {collection_name}")
        except Exception:
            pass

        duration = time.time() - t0

        return {
            "combo_key": combo_key,
            "chunk_config": chunk_cfg.to_dict(),
            "embed_model": embed_model.to_dict(),
            "rerank_model": rerank_model.to_dict(),
            "aggregate": agg.to_dict(),
            "per_query": [m.to_dict() for m in per_query_metrics],
            "duration_s": round(duration, 1),
            "n_chunks": len(chunk_contents),
        }

    def _load_layer_summary(self, layer_name: str) -> Optional[Dict[str, Any]]:
        """Load summary.json from a previous layer's output directory."""
        path = Path(self.config.output_dir) / layer_name / "summary.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

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
