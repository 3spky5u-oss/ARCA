"""
Layer LLM: Chat LLM Comparison
===============================
Compare chat LLM models by swapping the llama-server chat slot
and generating answers for all benchmark queries with each model.

Flow:
  1. Load optimal retrieval config from L0/L1/L2 (same as layer3_answers)
  2. Load queries from L0's queries.json
  3. Ingest corpus with optimal chunking config
  4. For each LLM candidate:
     a. Swap the chat model via LLMServerManager
     b. Wait for health check (up to 120s)
     c. For each query: retrieve context + generate answer
     d. Checkpoint each model's results
     e. Stop the model, clean up
  5. Compute basic stats (answer length, timeout rate, response time)
  6. Save per-model answers + summary.json
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from benchmark.layers.base import BaseLayer, LayerResult

logger = logging.getLogger(__name__)

# LLM models to compare — name is human-readable, gguf is the file in MODELS_DIR
LLM_CANDIDATES = [
    {"name": "GLM-4.7-Flash", "gguf": "GLM-4.7-Flash-Q4_K_M.gguf"},
    {"name": "Qwen3-30B-A3B", "gguf": "Qwen3-30B-A3B-Q4_K_M.gguf"},
]

# Per-query timeout for LLM generation
QUERY_TIMEOUT = 60.0
# Max time to wait for model health after swap
HEALTH_TIMEOUT = 120.0
# Pause after stopping a model before starting next
STOP_PAUSE = 3.0


class LLMComparisonLayer(BaseLayer):
    """Compare chat LLM models on the same retrieval pipeline."""

    LAYER_NAME = "layer_llm"

    def execute(self, result: LayerResult) -> LayerResult:
        from benchmark.collection_manager import BenchmarkCollectionManager
        from benchmark.config import ChunkingConfig
        from benchmark.queries.battery import BenchmarkQuery
        from tools.cohesionn.retriever import CohesionnRetriever
        from services.llm_server_manager import get_server_manager
        from config import runtime_config
        import httpx
        import os

        # ── Load optimal configs from previous layers ──────────────────
        optimal_l2 = self.load_optimal_config("layer2_params")
        if not optimal_l2:
            # Fall back to L1 if L2 hasn't run
            optimal_l2 = {
                "chunking": self.load_optimal_config("layer0_chunking"),
                "retrieval": self.load_optimal_config("layer1_retrieval"),
                "params": {},
            }

        chunking = optimal_l2.get("chunking", {})
        retrieval = optimal_l2.get("retrieval", {})
        optimal_params = optimal_l2.get("params", {})

        if not chunking:
            result.errors.append("No optimal chunking config found. Run L0 first.")
            return result

        chunk_cfg = ChunkingConfig.from_dict(chunking)
        retrieval_toggles = retrieval.get("toggles", {}) if retrieval else {}

        # Apply optimal params to runtime config
        if optimal_params:
            runtime_config.update(**optimal_params)
            logger.info(f"Applied optimal params: {optimal_params}")

        # ── Load queries ───────────────────────────────────────────────
        queries_path = Path(self.config.output_dir) / "layer0_chunking" / "queries.json"
        raw_queries = json.loads(queries_path.read_text(encoding="utf-8"))
        queries = [BenchmarkQuery.from_dict(q) for q in raw_queries]
        logger.info(f"Loaded {len(queries)} queries for LLM comparison")

        # ── Ingest corpus ──────────────────────────────────────────────
        collection_mgr = BenchmarkCollectionManager()
        l0_corpus_dir = Path(self.config.output_dir) / "layer0_chunking" / "corpus_md"
        md_files = sorted(str(f) for f in l0_corpus_dir.glob("*.md"))

        ingest_config_id = f"llm_{chunk_cfg.config_id}"
        topic = collection_mgr.get_topic(ingest_config_id)

        n_chunks = collection_mgr.ingest_corpus(
            config_id=ingest_config_id,
            md_files=md_files,
            chunk_size=chunk_cfg.chunk_size,
            chunk_overlap=chunk_cfg.chunk_overlap,
            context_prefix=chunk_cfg.context_prefix,
        )
        logger.info(f"Ingested {n_chunks} chunks for LLM comparison")

        # ── Run each LLM candidate ────────────────────────────────────
        retriever = CohesionnRetriever()
        llm_port = os.environ.get("LLM_CHAT_PORT", "8081")
        llm_url = f"http://localhost:{llm_port}/v1/chat/completions"
        health_url = f"http://localhost:{llm_port}/health"
        mgr = get_server_manager()

        all_model_results: Dict[str, List[Dict[str, Any]]] = {}
        model_stats: List[Dict[str, Any]] = []

        result.configs_total = len(LLM_CANDIDATES)

        for candidate in LLM_CANDIDATES:
            model_name = candidate["name"]
            gguf_file = candidate["gguf"]
            checkpoint_key = f"model_{model_name}"

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Testing LLM: {model_name} ({gguf_file})")
            logger.info(f"{'=' * 60}")

            # Check if entire model already checkpointed
            if self.checkpoint.is_completed(self.LAYER_NAME, checkpoint_key):
                logger.info(f"  Skipping {model_name} (checkpointed)")
                saved = self.checkpoint.get_result(self.LAYER_NAME, checkpoint_key)
                if saved:
                    all_model_results[model_name] = saved.get("answers", [])
                    model_stats.append(saved.get("stats", {}))
                result.configs_skipped += 1
                continue

            # ── Swap model ─────────────────────────────────────────────
            try:
                mgr.stop("chat")
                time.sleep(STOP_PAUSE)
                started = mgr.start("chat", gguf_override=gguf_file)
                if not started:
                    msg = f"Failed to start {model_name} ({gguf_file})"
                    logger.error(msg)
                    result.errors.append(msg)
                    continue
            except Exception as e:
                msg = f"Model swap failed for {model_name}: {e}"
                logger.error(msg)
                result.errors.append(msg)
                continue

            # ── Wait for healthy ───────────────────────────────────────
            healthy = False
            health_start = time.time()
            while time.time() - health_start < HEALTH_TIMEOUT:
                try:
                    with httpx.Client(timeout=5.0) as client:
                        resp = client.get(health_url)
                        if resp.status_code == 200:
                            healthy = True
                            break
                except Exception:
                    pass
                time.sleep(1.0)

            if not healthy:
                msg = f"{model_name} did not become healthy within {HEALTH_TIMEOUT}s"
                logger.error(msg)
                result.errors.append(msg)
                mgr.stop("chat")
                time.sleep(STOP_PAUSE)
                continue

            elapsed_startup = time.time() - health_start
            logger.info(f"  {model_name} healthy after {elapsed_startup:.1f}s")

            # ── Warmup: send a short prompt and retry until model is truly ready
            warmup_ok = False
            for _warmup_attempt in range(15):
                try:
                    with httpx.Client(timeout=30.0) as client:
                        warmup_resp = client.post(
                            llm_url,
                            json={
                                "model": gguf_file,
                                "messages": [{"role": "user", "content": "Hello"}],
                                "max_tokens": 8,
                                "temperature": 0.0,
                            },
                        )
                        if warmup_resp.status_code == 200:
                            warmup_ok = True
                            logger.info(f"  {model_name} warmup OK (attempt {_warmup_attempt + 1})")
                            break
                        else:
                            logger.info(f"  {model_name} warmup got {warmup_resp.status_code}, retrying...")
                except Exception:
                    pass
                time.sleep(2.0)

            if not warmup_ok:
                logger.warning(f"  {model_name} warmup failed after 15 attempts, proceeding anyway")

            # ── Generate answers for all queries ───────────────────────
            answers: List[Dict[str, Any]] = []
            total_response_time = 0.0
            timeout_count = 0

            for q in queries:
                per_query_key = f"{model_name}_{q.id}"

                # Per-query checkpoint
                if self.checkpoint.is_completed(self.LAYER_NAME, per_query_key):
                    saved = self.checkpoint.get_result(self.LAYER_NAME, per_query_key)
                    if saved:
                        answers.append(saved)
                    continue

                try:
                    # Retrieve context
                    retrieval_result = retriever.retrieve(
                        query=q.query,
                        topics=[topic],
                        top_k=self.config.top_k,
                        **retrieval_toggles,
                    )
                    context = retrieval_result.get_context(max_chunks=5)

                    # Generate answer via local LLM
                    answer_text = ""
                    query_start = time.time()
                    timed_out = False

                    try:
                        llm_prompt = (
                            f"Based on the following context, answer the question.\n\n"
                            f"Context:\n{context}\n\n"
                            f"Question: {q.query}\n\n"
                            f"Answer:"
                        )

                        with httpx.Client(timeout=QUERY_TIMEOUT) as client:
                            # Retry up to 3 times for 503 (model busy/loading)
                            for _attempt in range(3):
                                resp = client.post(
                                    llm_url,
                                    json={
                                        "model": gguf_file,
                                        "messages": [{"role": "user", "content": llm_prompt}],
                                        "max_tokens": runtime_config.max_output_tokens,
                                        "temperature": 0.3,
                                    },
                                )
                                if resp.status_code == 200:
                                    data = resp.json()
                                    answer_text = data["choices"][0]["message"]["content"]
                                    break
                                elif resp.status_code == 503 and _attempt < 2:
                                    logger.info(f"  {q.id}: 503, retry {_attempt + 2}/3 in 3s")
                                    time.sleep(3.0)
                                else:
                                    logger.warning(
                                        f"  LLM returned {resp.status_code} for {q.id}"
                                    )
                                    answer_text = f"[LLM error: {resp.status_code}]"
                                    break
                    except httpx.TimeoutException:
                        timed_out = True
                        timeout_count += 1
                        answer_text = "[TIMEOUT]"
                        logger.warning(f"  Timeout for {q.id} with {model_name}")
                    except Exception as e:
                        logger.warning(f"  LLM call failed for {q.id}: {e}")
                        answer_text = f"[LLM error: {e}]"

                    query_elapsed = time.time() - query_start
                    total_response_time += query_elapsed

                    answer_data = {
                        "query_id": q.id,
                        "query": q.query,
                        "tier": q.tier,
                        "context": context,
                        "answer": answer_text,
                        "ground_truth": q.ground_truth_answer,
                        "source_projects": q.source_projects,
                        "n_chunks_retrieved": len(retrieval_result.chunks),
                        "max_score": retrieval_result.max_score,
                        "confidence": retrieval_result.confidence,
                        "response_time_s": round(query_elapsed, 2),
                        "timed_out": timed_out,
                        "model_name": model_name,
                    }

                    answers.append(answer_data)
                    self.checkpoint.mark_completed(
                        self.LAYER_NAME, per_query_key, answer_data
                    )

                    logger.info(
                        f"  {q.id}: {len(answer_text)} chars, "
                        f"{query_elapsed:.1f}s, "
                        f"{retrieval_result.confidence} confidence"
                    )

                except Exception as e:
                    result.errors.append(f"{model_name}/{q.id}: {e}")
                    logger.error(f"  Answer generation failed for {q.id}: {e}")

            # ── Per-model stats ────────────────────────────────────────
            answered = [a for a in answers if not a.get("timed_out", False)]
            answer_lengths = [len(a.get("answer", "")) for a in answered]
            response_times = [a.get("response_time_s", 0) for a in answers]

            stats = {
                "model_name": model_name,
                "gguf": gguf_file,
                "total_queries": len(queries),
                "answers_generated": len(answered),
                "timeout_count": timeout_count,
                "timeout_rate": round(timeout_count / max(len(queries), 1), 3),
                "avg_answer_length": round(
                    sum(answer_lengths) / max(len(answer_lengths), 1), 1
                ),
                "avg_response_time_s": round(
                    sum(response_times) / max(len(response_times), 1), 2
                ),
                "total_response_time_s": round(total_response_time, 1),
                "startup_time_s": round(elapsed_startup, 1),
            }

            model_stats.append(stats)
            all_model_results[model_name] = answers

            # Checkpoint entire model result
            self.checkpoint.mark_completed(
                self.LAYER_NAME,
                checkpoint_key,
                {"answers": answers, "stats": stats},
            )

            # Save per-model answers file
            model_answers_path = self.output_dir / f"answers_{model_name.replace(' ', '_').replace('/', '_')}.json"
            model_answers_path.write_text(
                json.dumps(answers, indent=2, default=str), encoding="utf-8"
            )

            result.configs_completed += 1
            logger.info(
                f"  {model_name}: {len(answered)}/{len(queries)} answers, "
                f"avg {stats['avg_response_time_s']}s/query, "
                f"{timeout_count} timeouts"
            )

        # ── Stop chat model after all candidates ──────────────────────
        try:
            mgr.stop("chat")
            time.sleep(STOP_PAUSE)
        except Exception:
            pass

        # ── Cleanup ingested corpus ───────────────────────────────────
        collection_mgr.cleanup_topic(ingest_config_id)

        # ── Save summary ──────────────────────────────────────────────
        # Sort by avg response time (lower is better) for a basic ranking
        model_stats.sort(key=lambda s: s.get("avg_response_time_s", 999))
        for i, s in enumerate(model_stats, 1):
            s["rank"] = i

        summary = {
            "total_models": len(LLM_CANDIDATES),
            "models_tested": result.configs_completed + result.configs_skipped,
            "total_queries": len(queries),
            "ranking": model_stats,
        }

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

        # Pick "best" by lowest avg response time among models with <10% timeout
        viable = [s for s in model_stats if s.get("timeout_rate", 1) < 0.10]
        if viable:
            best = viable[0]
            result.best_config_id = best["model_name"]
            result.best_score = best.get("avg_response_time_s", 0)

        result.summary = {
            "total_models": len(LLM_CANDIDATES),
            "models_tested": result.configs_completed + result.configs_skipped,
            "total_queries": len(queries),
            "model_stats": model_stats,
        }

        return result
