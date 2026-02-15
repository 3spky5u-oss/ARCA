"""
Layer 2: Continuous Parameter Sweep (One-at-a-Time)

Uses the best toggle config from L1. Sweeps continuous parameters one at a time
via runtime_config.update(). Measures impact of each parameter on retrieval quality.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from benchmark.layers.base import BaseLayer, LayerResult

logger = logging.getLogger(__name__)


class ParamSweepLayer(BaseLayer):
    """Layer 2: One-at-a-time continuous parameter sweep."""

    LAYER_NAME = "layer2_params"

    def execute(self, result: LayerResult) -> LayerResult:
        from benchmark.collection_manager import BenchmarkCollectionManager
        from benchmark.config import ChunkingConfig, PARAM_SWEEP_RANGES
        from tools.cohesionn.retriever import CohesionnRetriever
        from tools.cohesionn.benchmark.metrics import ScoringEngine
        from config import runtime_config

        # Load optimal configs from L0 and L1
        optimal_chunking = self.load_optimal_config("layer0_chunking")
        optimal_retrieval = self.load_optimal_config("layer1_retrieval")

        if not optimal_chunking:
            result.errors.append("No optimal chunking config from L0")
            return result
        if not optimal_retrieval:
            result.errors.append("No optimal retrieval config from L1")
            return result

        chunk_cfg = ChunkingConfig.from_dict(optimal_chunking)
        retrieval_toggles = optimal_retrieval.get("toggles", {})

        logger.info(
            f"Using chunking: {chunk_cfg.config_id}, "
            f"retrieval: {optimal_retrieval.get('config_name')}"
        )

        # Load queries
        queries_path = (
            Path(self.config.output_dir) / "layer0_chunking" / "queries.json"
        )
        from benchmark.queries.battery import BenchmarkQuery

        raw_queries = json.loads(queries_path.read_text(encoding="utf-8"))
        queries = [BenchmarkQuery.from_dict(q) for q in raw_queries]

        # Ingest corpus with optimal chunking
        collection_mgr = BenchmarkCollectionManager()
        l0_corpus_dir = (
            Path(self.config.output_dir) / "layer0_chunking" / "corpus_md"
        )
        md_files = sorted(str(f) for f in l0_corpus_dir.glob("*.md"))

        ingest_config_id = f"l2_{chunk_cfg.config_id}"
        topic = collection_mgr.get_topic(ingest_config_id)

        n_chunks = collection_mgr.ingest_corpus(
            config_id=ingest_config_id,
            md_files=md_files,
            chunk_size=chunk_cfg.chunk_size,
            chunk_overlap=chunk_cfg.chunk_overlap,
            context_prefix=chunk_cfg.context_prefix,
        )
        logger.info(f"Ingested {n_chunks} chunks for param sweep")

        # Sweep each parameter one at a time
        sweep_ranges = self.config.param_sweep_ranges or PARAM_SWEEP_RANGES

        # Count total configs
        total = sum(len(values) for values in sweep_ranges.values())
        result.configs_total = total

        retriever = CohesionnRetriever()
        scorer = ScoringEngine()
        all_results = {}

        for param_name, values in sweep_ranges.items():
            param_results = []
            original_value = getattr(runtime_config, param_name, None)

            for value in values:
                config_id = f"{param_name}_{value}"

                if self.checkpoint.is_completed(self.LAYER_NAME, config_id):
                    result.configs_skipped += 1
                    saved = self.checkpoint.get_result(self.LAYER_NAME, config_id)
                    if saved:
                        param_results.append(saved)
                    continue

                logger.info(f"  Testing {param_name}={value}")

                # Update runtime config
                runtime_config.update(**{param_name: value})

                try:
                    per_query_metrics = []

                    for q in queries:
                        q_start = time.time()
                        try:
                            retrieval = retriever.retrieve(
                                query=q.query,
                                topics=[topic],
                                top_k=self.config.top_k,
                                **retrieval_toggles,
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
                                f"Query {q.id} failed for {config_id}: {e}"
                            )

                    agg = scorer.aggregate(config_id, per_query_metrics)

                    sweep_result = {
                        "param": param_name,
                        "value": value,
                        "config_id": config_id,
                        "aggregate": agg.to_dict(),
                    }

                    param_results.append(sweep_result)
                    self.checkpoint.mark_completed(
                        self.LAYER_NAME, config_id, sweep_result
                    )
                    result.configs_completed += 1

                    logger.info(f"    composite={agg.avg_composite:.4f}")

                except Exception as e:
                    result.errors.append(f"{config_id}: {e}")
                    logger.error(f"Param {config_id} failed: {e}")

            # Restore original value
            if original_value is not None:
                runtime_config.update(**{param_name: original_value})

            # Find best value for this param
            if param_results:
                param_results.sort(
                    key=lambda x: x.get("aggregate", {}).get("avg_composite", 0),
                    reverse=True,
                )
                all_results[param_name] = {
                    "best_value": param_results[0]["value"],
                    "best_composite": param_results[0]["aggregate"]["avg_composite"],
                    "all_values": [
                        {
                            "value": r["value"],
                            "composite": r["aggregate"]["avg_composite"],
                        }
                        for r in param_results
                    ],
                }

        # Cleanup
        collection_mgr.cleanup_topic(ingest_config_id)

        # Build optimal param set
        optimal_params = {}
        for param_name, data in all_results.items():
            optimal_params[param_name] = data["best_value"]

        # Save results
        summary = {
            "chunking_config": optimal_chunking,
            "retrieval_config": optimal_retrieval,
            "param_sweeps": all_results,
            "optimal_params": optimal_params,
        }

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

        optimal_path = self.output_dir / "optimal_config.json"
        optimal_path.write_text(
            json.dumps(
                {
                    "chunking": optimal_chunking,
                    "retrieval": optimal_retrieval,
                    "params": optimal_params,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        result.summary = summary
        if optimal_params:
            result.best_config_id = str(optimal_params)

        return result
