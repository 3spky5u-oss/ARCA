"""
Layer R: Reranker Model Shootout
================================
Swap reranker models without re-embedding. Zero-cost on Qdrant.
Adapted from tools.cohesionn.benchmark.phases.reranker.

Flow:
  1. Load optimal chunking config from L0
  2. Ingest corpus into benchmark topic (or reuse if exists)
  3. Retrieve candidates for all 40 queries WITHOUT reranking (cached)
  4. Per reranker model:
     a. Create BGEReranker with model_name
     b. Re-rank all cached candidates
     c. Score with ScoringEngine
     d. Unload reranker
  5. Rank models, save optimal
"""
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from benchmark.layers.base import BaseLayer, LayerResult

logger = logging.getLogger(__name__)


class RerankerShootoutLayer(BaseLayer):
    """Benchmark reranker models on cached retrieval candidates."""

    LAYER_NAME = "layer_rerank"

    def execute(self, result: LayerResult) -> LayerResult:
        from tools.cohesionn.benchmark.config import RERANKER_CANDIDATES
        from tools.cohesionn.benchmark.metrics import ScoringEngine
        from tools.cohesionn.retriever import CohesionnRetriever
        from benchmark.config import ChunkingConfig
        from benchmark.collection_manager import BenchmarkCollectionManager
        from benchmark.queries.battery import BenchmarkQuery

        # Step 1: Load L0 optimal chunking + queries
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

        # Step 2: Ingest corpus with optimal chunking
        corpus_dir = Path(self.config.output_dir) / "layer0_chunking" / "corpus_md"
        md_files = sorted(str(f) for f in corpus_dir.glob("*.md"))
        if not md_files:
            result.errors.append("No corpus markdown files from L0.")
            return result

        collection_mgr = BenchmarkCollectionManager()
        ingest_config_id = f"rerank_{chunk_cfg.config_id}"
        topic = collection_mgr.get_topic(ingest_config_id)

        n_chunks = collection_mgr.ingest_corpus(
            config_id=ingest_config_id,
            md_files=md_files,
            chunk_size=chunk_cfg.chunk_size,
            chunk_overlap=chunk_cfg.chunk_overlap,
            context_prefix=chunk_cfg.context_prefix,
        )
        logger.info(f"Ingested {n_chunks} chunks into topic {topic}")

        # Step 3: Retrieve candidates for all queries WITHOUT reranking
        logger.info("Retrieving candidates (rerank=False)...")
        retriever = CohesionnRetriever()
        cached: Dict[str, Any] = {}

        for q in queries:
            q_start = time.time()
            try:
                retrieval = retriever.retrieve(
                    query=q.query,
                    topics=[topic],
                    top_k=15,  # retrieve more candidates for rerankers to work with
                    rerank=False,
                    use_hybrid=False,
                    use_expansion=False,
                    use_hyde=False,
                    use_raptor=False,
                    use_graph=False,
                    use_global=False,
                )
                cached[q.id] = {
                    "chunks": retrieval.chunks,
                    "latency_ms": (time.time() - q_start) * 1000,
                }
            except Exception as e:
                cached[q.id] = {
                    "chunks": [],
                    "latency_ms": (time.time() - q_start) * 1000,
                    "error": str(e),
                }

        logger.info(f"Cached {len(cached)} query results")

        # Step 4: Test each reranker
        models = list(RERANKER_CANDIDATES)
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
                model_result = self._test_reranker(model, cached, queries, scorer)
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

        # Step 5: Cleanup ingested corpus
        collection_mgr.cleanup_topic(ingest_config_id)

        # Step 6: Rank and save
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
                    "avg_latency_ms": s["aggregate"].get("avg_latency_ms", 0),
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

    def _test_reranker(
        self,
        model,
        cached: Dict[str, Any],
        queries,
        scorer,
    ) -> Dict[str, Any]:
        """Test a single reranker model on cached candidates."""
        from tools.cohesionn.reranker import BGEReranker, RERANKER_MAX_LENGTH

        t0 = time.time()

        reranker = BGEReranker(
            model_name=model.hf_id,
            max_length=model.max_length or RERANKER_MAX_LENGTH,
            trust_remote_code=model.trust_remote_code,
        )

        per_query_metrics = []
        for q in queries:
            qr = cached.get(q.id, {})
            if "error" in qr or not qr.get("chunks"):
                continue

            q_start = time.time()
            try:
                ranked_chunks = reranker.rerank(
                    query=q.query,
                    results=qr["chunks"],
                    top_k=self.config.top_k,
                )
                latency = (time.time() - q_start) * 1000

                metrics = scorer.score_retrieval(
                    query_id=q.id,
                    tier=q.tier,
                    chunks=ranked_chunks,
                    latency_ms=latency,
                    expect_keywords=q.expect_keywords,
                    expect_entities=q.expect_entities,
                )
                per_query_metrics.append(metrics)

            except Exception as e:
                logger.warning(f"  Query {q.id} failed with {model.name}: {e}")

        agg = scorer.aggregate(model.short_name, per_query_metrics)

        # Cleanup
        reranker.unload()
        duration = time.time() - t0

        return {
            "model": model.to_dict(),
            "aggregate": agg.to_dict(),
            "per_query": [m.to_dict() for m in per_query_metrics],
            "duration_s": round(duration, 1),
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
