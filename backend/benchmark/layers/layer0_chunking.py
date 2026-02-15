"""
Layer 0: Chunking Configuration Sweep

Tests ~90 chunking configs (chunk_size x overlap x context_prefix) using dense-only
retrieval to isolate chunking quality from pipeline variables.

Flow per config:
  1. Check checkpoint -- skip if completed
  2. Ingest corpus into bench_{config_id} topic
  3. Run all queries with dense-only retrieval
  4. Score with ScoringEngine
  5. Save result + checkpoint
  6. Cleanup Qdrant topic (if configured)
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from benchmark.layers.base import BaseLayer, LayerResult

logger = logging.getLogger(__name__)


class ChunkingSweepLayer(BaseLayer):
    """Layer 0: Chunking configuration sweep."""

    LAYER_NAME = "layer0_chunking"

    def execute(self, result: LayerResult) -> LayerResult:
        from benchmark.config import generate_chunking_matrix
        from benchmark.corpus import DocxConverter
        from benchmark.collection_manager import BenchmarkCollectionManager
        from benchmark.queries.loader import QueryBatteryLoader
        from tools.cohesionn.retriever import CohesionnRetriever
        from tools.cohesionn.benchmark.metrics import ScoringEngine

        # Step 1: Convert corpus to markdown
        corpus_dir = self.config.corpus_dir
        md_output = self.output_dir / "corpus_md"

        converter = DocxConverter()
        md_files = converter.convert_directory(corpus_dir, str(md_output))
        if not md_files:
            result.errors.append("No .docx files converted")
            return result

        logger.info(f"Converted {len(md_files)} .docx files to markdown")

        # Step 2: Load benchmark queries (domain battery > auto-gen > generic)
        loader = QueryBatteryLoader()
        queries = loader.load(corpus_dir, self.config)

        if not queries:
            result.errors.append("No benchmark queries available")
            return result

        # Save queries + source metadata
        queries_path = self.output_dir / "queries.json"
        queries_path.write_text(
            json.dumps([q.to_dict() for q in queries], indent=2),
            encoding="utf-8",
        )
        query_source = loader.source
        logger.info(f"Loaded {len(queries)} benchmark queries (source: {query_source})")

        # Persist query source for CLI / downstream layers
        meta_path = self.output_dir / "query_meta.json"
        meta_path.write_text(
            json.dumps({"source": query_source, "count": len(queries)}, indent=2),
            encoding="utf-8",
        )

        # Step 3: Generate chunking matrix
        configs = self.config.chunking_configs or generate_chunking_matrix()
        if self.config.max_configs > 0:
            configs = configs[: self.config.max_configs]

        result.configs_total = len(configs)
        logger.info(f"Testing {len(configs)} chunking configurations")

        # Step 4: Score each config
        collection_mgr = BenchmarkCollectionManager()
        retriever = CohesionnRetriever()
        scorer = ScoringEngine()

        all_scores = []

        for i, cfg in enumerate(configs):
            # Check checkpoint
            if self.checkpoint.is_completed(self.LAYER_NAME, cfg.config_id):
                result.configs_skipped += 1
                saved = self.checkpoint.get_result(self.LAYER_NAME, cfg.config_id)
                if saved:
                    all_scores.append(saved)
                logger.info(
                    f"[{i + 1}/{len(configs)}] Skipping {cfg.config_id} (checkpointed)"
                )
                continue

            logger.info(f"[{i + 1}/{len(configs)}] Testing config: {cfg.config_id}")
            config_start = time.time()

            try:
                # Ingest corpus with this chunking config
                n_chunks = collection_mgr.ingest_corpus(
                    config_id=cfg.config_id,
                    md_files=md_files,
                    chunk_size=cfg.chunk_size,
                    chunk_overlap=cfg.chunk_overlap,
                    context_prefix=cfg.context_prefix,
                )

                if n_chunks == 0:
                    logger.warning(f"No chunks created for {cfg.config_id}")
                    continue

                # Run all queries with dense-only retrieval
                topic = collection_mgr.get_topic(cfg.config_id)
                per_query_metrics = []

                for q in queries:
                    q_start = time.time()
                    try:
                        retrieval = retriever.retrieve(
                            query=q.query,
                            topics=[topic],
                            top_k=self.config.top_k,
                            rerank=False,
                            use_hybrid=False,
                            use_expansion=False,
                            use_hyde=False,
                            use_raptor=False,
                            use_graph=False,
                            use_global=False,
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
                            f"Query {q.id} failed for {cfg.config_id}: {e}"
                        )

                # Aggregate metrics
                agg = scorer.aggregate(cfg.config_id, per_query_metrics)

                config_result = {
                    "config": cfg.to_dict(),
                    "n_chunks": n_chunks,
                    "aggregate": agg.to_dict(),
                    "per_query": [m.to_dict() for m in per_query_metrics],
                    "duration_seconds": round(time.time() - config_start, 1),
                }

                all_scores.append(config_result)
                self.checkpoint.mark_completed(
                    self.LAYER_NAME, cfg.config_id, config_result
                )
                result.configs_completed += 1

                logger.info(
                    f"  {cfg.config_id}: composite={agg.avg_composite:.4f} "
                    f"({n_chunks} chunks, {agg.avg_latency_ms:.0f}ms avg)"
                )

            except Exception as e:
                result.errors.append(f"{cfg.config_id}: {e}")
                logger.error(f"Config {cfg.config_id} failed: {e}")

            finally:
                # Cleanup Qdrant topic
                if self.config.cleanup_after_score:
                    collection_mgr.cleanup_topic(cfg.config_id)

        # Step 5: Rank and save results
        all_scores.sort(
            key=lambda x: x.get("aggregate", {}).get("avg_composite", 0),
            reverse=True,
        )

        # Save ranked summary
        summary_path = self.output_dir / "summary.json"
        summary = {
            "total_configs": len(configs),
            "completed": result.configs_completed,
            "skipped": result.configs_skipped,
            "ranking": [
                {
                    "rank": i + 1,
                    "config_id": s["config"]["config_id"],
                    "chunk_size": s["config"]["chunk_size"],
                    "chunk_overlap": s["config"]["chunk_overlap"],
                    "context_prefix": s["config"]["context_prefix"],
                    "composite": s["aggregate"]["avg_composite"],
                    "n_chunks": s["n_chunks"],
                }
                for i, s in enumerate(all_scores[:20])  # Top 20
            ],
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Save optimal config
        if all_scores:
            best = all_scores[0]
            optimal_path = self.output_dir / "optimal_config.json"
            optimal_path.write_text(
                json.dumps(best["config"], indent=2), encoding="utf-8"
            )

            result.best_config_id = best["config"]["config_id"]
            result.best_score = best["aggregate"]["avg_composite"]
            result.summary = summary

        return result
