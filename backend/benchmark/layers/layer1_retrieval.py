"""
Layer 1: Retrieval Configuration Sweep

Uses the best chunking config from L0 (re-ingests once), then tests ~15 named
retrieval toggle configurations. Measures both quality and latency.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from benchmark.layers.base import BaseLayer, LayerResult

logger = logging.getLogger(__name__)


class RetrievalSweepLayer(BaseLayer):
    """Layer 1: Retrieval toggle configuration sweep."""

    LAYER_NAME = "layer1_retrieval"

    def execute(self, result: LayerResult) -> LayerResult:
        from benchmark.collection_manager import BenchmarkCollectionManager
        from benchmark.config import ChunkingConfig, RETRIEVAL_CONFIGS
        from tools.cohesionn.retriever import CohesionnRetriever
        from tools.cohesionn.benchmark.metrics import ScoringEngine
        from config import runtime_config

        # Load optimal chunking config from L0
        optimal = self.load_optimal_config("layer0_chunking")
        if not optimal:
            result.errors.append("No optimal chunking config found from Layer 0")
            return result

        chunk_cfg = ChunkingConfig.from_dict(optimal)
        logger.info(f"Using optimal chunking config: {chunk_cfg.config_id}")

        # Load queries from L0
        queries_path = (
            Path(self.config.output_dir) / "layer0_chunking" / "queries.json"
        )
        if not queries_path.exists():
            result.errors.append("No queries found from Layer 0")
            return result

        from benchmark.queries.battery import BenchmarkQuery

        raw_queries = json.loads(queries_path.read_text(encoding="utf-8"))
        queries = [BenchmarkQuery.from_dict(q) for q in raw_queries]

        # Ingest corpus with optimal chunking
        collection_mgr = BenchmarkCollectionManager()
        l0_corpus_dir = (
            Path(self.config.output_dir) / "layer0_chunking" / "corpus_md"
        )
        md_files = sorted(str(f) for f in l0_corpus_dir.glob("*.md"))

        if not md_files:
            result.errors.append("No markdown files found from L0 corpus conversion")
            return result

        ingest_config_id = f"l1_{chunk_cfg.config_id}"
        topic = collection_mgr.get_topic(ingest_config_id)

        n_chunks = collection_mgr.ingest_corpus(
            config_id=ingest_config_id,
            md_files=md_files,
            chunk_size=chunk_cfg.chunk_size,
            chunk_overlap=chunk_cfg.chunk_overlap,
            context_prefix=chunk_cfg.context_prefix,
        )
        logger.info(f"Ingested {n_chunks} chunks for retrieval sweep")

        # Test each retrieval config
        retrieval_configs = self.config.retrieval_configs or RETRIEVAL_CONFIGS
        result.configs_total = len(retrieval_configs)

        retriever = CohesionnRetriever()
        scorer = ScoringEngine()
        all_results = []

        for config_name, toggles in retrieval_configs.items():
            if self.checkpoint.is_completed(self.LAYER_NAME, config_name):
                result.configs_skipped += 1
                saved = self.checkpoint.get_result(self.LAYER_NAME, config_name)
                if saved:
                    all_results.append(saved)
                continue

            logger.info(f"Testing retrieval config: {config_name}")

            # Handle domain_boost toggle via runtime_config
            original_boost = runtime_config.domain_boost_enabled
            if config_name == "deep_no_domain_boost":
                runtime_config.update(domain_boost_enabled=False)

            try:
                per_query_metrics = []

                for q in queries:
                    q_start = time.time()
                    try:
                        retrieval = retriever.retrieve(
                            query=q.query,
                            topics=[topic],
                            top_k=self.config.top_k,
                            **toggles,
                        )
                        latency = (time.time() - q_start) * 1000

                        metrics = scorer.score_retrieval(
                            query_id=q.id,
                            tier=q.tier,
                            chunks=retrieval.chunks,
                            latency_ms=latency,
                            expect_keywords=q.expect_keywords,
                            expect_entities=q.expect_entities,
                        )
                        per_query_metrics.append(metrics)
                    except Exception as e:
                        logger.warning(
                            f"Query {q.id} failed for {config_name}: {e}"
                        )

                agg = scorer.aggregate(config_name, per_query_metrics)

                config_result = {
                    "config_name": config_name,
                    "toggles": toggles,
                    "aggregate": agg.to_dict(),
                    "per_query": [m.to_dict() for m in per_query_metrics],
                }

                all_results.append(config_result)
                self.checkpoint.mark_completed(
                    self.LAYER_NAME, config_name, config_result
                )
                result.configs_completed += 1

                logger.info(
                    f"  {config_name}: composite={agg.avg_composite:.4f}, "
                    f"latency={agg.avg_latency_ms:.0f}ms"
                )

            except Exception as e:
                result.errors.append(f"{config_name}: {e}")
                logger.error(f"Retrieval config {config_name} failed: {e}")
            finally:
                # Restore domain boost
                runtime_config.update(domain_boost_enabled=original_boost)

        # Cleanup
        collection_mgr.cleanup_topic(ingest_config_id)

        # Rank results
        all_results.sort(
            key=lambda x: x.get("aggregate", {}).get("avg_composite", 0),
            reverse=True,
        )

        # Save summary
        summary = {
            "total_configs": len(retrieval_configs),
            "completed": result.configs_completed,
            "chunking_config": optimal,
            "ranking": [
                {
                    "rank": i + 1,
                    "config_name": r["config_name"],
                    "composite": r["aggregate"]["avg_composite"],
                    "latency_ms": r["aggregate"]["avg_latency_ms"],
                }
                for i, r in enumerate(all_results)
            ],
        }

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Save optimal retrieval config
        if all_results:
            best = all_results[0]
            optimal_path = self.output_dir / "optimal_config.json"
            optimal_path.write_text(
                json.dumps(
                    {
                        "config_name": best["config_name"],
                        "toggles": best["toggles"],
                        "composite": best["aggregate"]["avg_composite"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            result.best_config_id = best["config_name"]
            result.best_score = best["aggregate"]["avg_composite"]

        result.summary = summary
        return result
