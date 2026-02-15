"""Internal helper functions for admin benchmark router."""

import importlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Import constants directly (immutable — safe to copy)
from . import ALL_LAYERS, QUICK_LAYERS, LAYER_REGISTRY, BENCHMARKS_DIR, _LLM_CHAT_PORT, _job_lock

logger = logging.getLogger(__name__)


def _resolve_layers(phases_str: str) -> List[str]:
    """Resolve phase/layer string to list of layer names."""
    if phases_str == "quick":
        return list(QUICK_LAYERS)
    if phases_str == "full":
        return list(ALL_LAYERS)
    return [p.strip() for p in phases_str.split(",") if p.strip() in ALL_LAYERS]


def _get_layer_class(layer_name: str):
    """Import and return the layer class for a given layer name."""
    if layer_name not in LAYER_REGISTRY:
        return None
    module_path, class_name = LAYER_REGISTRY[layer_name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _format_layer_for_frontend(layer_name: str, result, output_dir: str) -> Dict[str, Any]:
    """Convert v2 LayerResult to frontend PhaseData format.

    Frontend expects: {duration_s, n_variants, winner, ranking: [{rank, variant, composite}]}
    """
    summary_path = Path(output_dir) / layer_name / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    phase_data: Dict[str, Any] = {
        "duration_s": result.duration_seconds if result else 0,
        "n_variants": result.configs_total if result else 0,
        "winner": None,
        "ranking": [],
    }

    if layer_name == "layer0_chunking":
        ranking = summary.get("ranking", [])
        if ranking:
            phase_data["winner"] = ranking[0].get("config_id", "unknown")
            phase_data["n_variants"] = summary.get("total_configs", len(ranking))
            phase_data["ranking"] = [
                {
                    "rank": r.get("rank", i + 1),
                    "variant": r.get("config_id", f"config_{i}"),
                    "composite": r.get("composite", 0),
                }
                for i, r in enumerate(ranking[:20])
            ]

    elif layer_name == "layer1_retrieval":
        ranking = summary.get("ranking", [])
        if ranking:
            phase_data["winner"] = ranking[0].get("config_name", "unknown")
            phase_data["n_variants"] = len(ranking)
            phase_data["ranking"] = [
                {
                    "rank": r.get("rank", i + 1),
                    "variant": r.get("config_name", f"config_{i}"),
                    "composite": r.get("composite", 0),
                }
                for i, r in enumerate(ranking)
            ]

    elif layer_name == "layer2_params":
        sweeps = summary.get("param_sweeps", {})
        rankings = []
        for i, (param, data) in enumerate(sweeps.items(), 1):
            rankings.append({
                "rank": i,
                "variant": f"{param} = {data.get('best_value', '?')}",
                "composite": data.get("best_composite", 0),
            })
        phase_data["ranking"] = rankings
        phase_data["n_variants"] = sum(
            len(d.get("all_values", [])) for d in sweeps.values()
        )
        phase_data["winner"] = f"{len(sweeps)} params optimized"

    elif layer_name == "layer_embed":
        ranking = summary.get("ranking", [])
        if ranking:
            phase_data["winner"] = ranking[0].get("model_name", "unknown")
            phase_data["n_variants"] = summary.get("total_models", len(ranking))
            phase_data["ranking"] = [
                {
                    "rank": r.get("rank", i + 1),
                    "variant": r.get("model_name", f"model_{i}"),
                    "composite": r.get("composite", 0),
                }
                for i, r in enumerate(ranking)
            ]

    elif layer_name == "layer_rerank":
        ranking = summary.get("ranking", [])
        if ranking:
            phase_data["winner"] = ranking[0].get("model_name", "unknown")
            phase_data["n_variants"] = summary.get("total_models", len(ranking))
            phase_data["ranking"] = [
                {
                    "rank": r.get("rank", i + 1),
                    "variant": r.get("model_name", f"model_{i}"),
                    "composite": r.get("composite", 0),
                }
                for i, r in enumerate(ranking)
            ]

    elif layer_name == "layer_cross":
        ranking = summary.get("ranking", [])
        if ranking:
            phase_data["winner"] = ranking[0].get("combo_key", "unknown")
            phase_data["n_variants"] = summary.get("total_combos", len(ranking))
            phase_data["ranking"] = [
                {
                    "rank": r.get("rank", i + 1),
                    "variant": r.get("combo_key", f"combo_{i}"),
                    "composite": r.get("composite", 0),
                }
                for i, r in enumerate(ranking)
            ]

    elif layer_name == "layer_llm":
        ranking = summary.get("ranking", [])
        if ranking:
            phase_data["winner"] = ranking[0].get("model_name", "unknown")
            phase_data["n_variants"] = summary.get("total_models", len(ranking))
            phase_data["ranking"] = [
                {
                    "rank": r.get("rank", i + 1),
                    "variant": r.get("model_name", f"model_{i}"),
                    "composite": r.get("avg_response_time_s", 0),
                }
                for i, r in enumerate(ranking)
            ]

    elif layer_name == "layer3_answers":
        phase_data["winner"] = f"{result.configs_completed if result else 0} answers"
        phase_data["n_variants"] = result.configs_total if result else 0

    elif layer_name == "layer4_judge":
        avg = summary.get("avg_overall", 0)
        phase_data["winner"] = f"{avg:.2f}/5 avg"
        phase_data["n_variants"] = summary.get("total_judged", result.configs_total if result else 0)
        by_tier = summary.get("by_tier", {})
        for i, (tier, data) in enumerate(by_tier.items(), 1):
            phase_data["ranking"].append({
                "rank": i,
                "variant": tier,
                "composite": data.get("avg_overall", 0),
            })

    elif layer_name == "layer5_analysis":
        charts = summary.get("charts_generated", [])
        phase_data["winner"] = f"{len(charts)} charts"
        phase_data["n_variants"] = len(charts) + 1

    elif layer_name == "layer6_failures":
        total = summary.get("total_failures", 0)
        phase_data["winner"] = f"{total} failures found"
        phase_data["n_variants"] = total
        by_cat = summary.get("by_category", {})
        for i, (cat, count) in enumerate(by_cat.items(), 1):
            phase_data["ranking"].append({
                "rank": i,
                "variant": cat.replace("_", " ").title(),
                "composite": count,
            })

    elif layer_name == "layer_live":
        queries_tested = summary.get("benchmark_queries_tested", 0)
        phase_data["winner"] = f"{queries_tested} queries tested"
        phase_data["n_variants"] = queries_tested
        phase_data["ranking"] = [
            {"rank": 1, "variant": "Avg Latency", "composite": summary.get("avg_latency_ms", 0)},
            {"rank": 2, "variant": "Avg Confidence", "composite": summary.get("avg_confidence", 0)},
            {"rank": 3, "variant": "Keyword Hit Rate", "composite": summary.get("keyword_hit_rate", 0)},
            {"rank": 4, "variant": "Adversarial Pass Rate", "composite": summary.get("adversarial_pass_rate", 0)},
        ]

    elif layer_name == "layer_ceiling":
        ceiling_avg = summary.get("ceiling_avg_composite", 0)
        local_avg = summary.get("local_avg_composite", 0)
        delta = summary.get("model_quality_delta", 0)
        ceiling_wins = summary.get("ceiling_wins", 0)
        phase_data["winner"] = f"delta={delta:.3f} ({ceiling_wins} ceiling wins)"
        phase_data["n_variants"] = 2
        phase_data["ranking"] = [
            {"rank": 1, "variant": "Ceiling Avg Score", "composite": ceiling_avg},
            {"rank": 2, "variant": "Local Avg Score", "composite": local_avg},
            {"rank": 3, "variant": "Quality Delta", "composite": delta},
            {"rank": 4, "variant": "Ceiling Wins", "composite": ceiling_wins},
        ]

    return phase_data


def _extract_winners(output_dir: str) -> Dict[str, Any]:
    """Extract optimal configuration from v2 benchmark results for RuntimeConfig."""
    winners = {}

    # Benchmark toggle names -> RuntimeConfig field names
    toggle_map = {
        "rerank": "reranker_enabled",
        "use_hybrid": "bm25_enabled",
        "use_expansion": "query_expansion_enabled",
        "use_hyde": "hyde_enabled",
        "use_raptor": "raptor_enabled",
        "use_graph": "graph_rag_enabled",
        "use_global": "global_search_enabled",
        "apply_diversity": "rag_diversity_enabled",
    }

    # Continuous params (names already match RuntimeConfig)
    continuous_params = [
        "bm25_weight", "rag_diversity_lambda", "rag_top_k",
        "reranker_candidates", "rag_min_score", "domain_boost_factor",
    ]

    def _extract_toggles(source: Dict[str, Any]) -> None:
        """Extract toggle values from a dict using the toggle_map."""
        for src_key, dst_key in toggle_map.items():
            if src_key in source:
                winners[dst_key] = source[src_key]

    def _extract_continuous(source: Dict[str, Any]) -> None:
        """Extract continuous parameter values from a dict."""
        for param in continuous_params:
            if param in source:
                winners[param] = source[param]

    # Read the most complete optimal config (L2 > L1 > L0)
    for layer in ["layer2_params", "layer1_retrieval", "layer0_chunking"]:
        optimal_path = Path(output_dir) / layer / "optimal_config.json"
        if not optimal_path.exists():
            continue
        try:
            optimal = json.loads(optimal_path.read_text(encoding="utf-8"))

            # L0: top-level chunk_size, chunk_overlap
            if "chunk_size" in optimal:
                winners["chunk_size"] = optimal["chunk_size"]
            if "chunk_overlap" in optimal:
                winners["chunk_overlap"] = optimal["chunk_overlap"]

            # L1: toggles nested under "toggles" key
            if "toggles" in optimal and isinstance(optimal["toggles"], dict):
                _extract_toggles(optimal["toggles"])

            # L2: params nested under "params" key, retrieval under "retrieval"
            if "params" in optimal and isinstance(optimal["params"], dict):
                _extract_continuous(optimal["params"])
            if "retrieval" in optimal and isinstance(optimal["retrieval"], dict):
                retrieval = optimal["retrieval"]
                # L2 may nest retrieval toggles
                if "toggles" in retrieval:
                    _extract_toggles(retrieval["toggles"])
                else:
                    _extract_toggles(retrieval)

            # Also check top-level (in case any layer writes flat)
            _extract_toggles(optimal)
            _extract_continuous(optimal)

        except Exception as e:
            logger.warning("Could not read optimal config from " + layer + ": " + str(e))

    return winners


def _auto_apply_winners(winners: Dict[str, Any]) -> Dict[str, Any]:
    """Apply winners to RuntimeConfig and update the 'auto' retrieval profile.

    Called automatically after auto-tune completes.
    Returns summary of what was applied.
    """
    if not winners:
        return {"applied": False, "reason": "No winners to apply"}

    from config import runtime_config
    from profile_loader import PROFILE_TOGGLE_KEYS

    # Apply to RuntimeConfig
    updates = {
        k: v for k, v in winners.items()
        if hasattr(runtime_config, k) and not k.startswith("_")
    }
    result = runtime_config.update(**updates) if updates else {"updated": [], "ignored": []}

    if result.get("updated"):
        runtime_config.save_overrides()

    # Update the "auto" retrieval profile with toggle winners
    toggle_winners = {
        k: v for k, v in updates.items()
        if k in PROFILE_TOGGLE_KEYS and isinstance(v, bool)
    }
    auto_profile_changes = {}
    if toggle_winners:
        try:
            from profile_loader import get_profile_manager
            pm = get_profile_manager()
            auto_profile_changes = pm.update_profile_toggles("auto", toggle_winners)
        except Exception as e:
            logger.warning("Could not update auto profile: " + str(e))

    logger.info(
        "Auto-applied " + str(len(result.get("updated", []))) + " winners to RuntimeConfig, "
        + str(len(auto_profile_changes)) + " to auto profile"
    )

    return {
        "applied": True,
        "updated": result.get("updated", []),
        "ignored": result.get("ignored", []),
        "auto_profile_updated": auto_profile_changes,
    }


def _generate_llm_analysis(output_dir: str) -> Optional[str]:
    """Generate plain-English analysis of v2 benchmark results via local LLM."""
    try:
        summary_lines = []

        for layer_name in ALL_LAYERS:
            summary_path = Path(output_dir) / layer_name / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            if layer_name == "layer0_chunking":
                ranking = summary.get("ranking", [])
                if ranking:
                    top = ranking[0]
                    summary_lines.append(
                        f"- Chunking: Best = {top.get('config_id')} "
                        f"(size={top.get('chunk_size')}, overlap={top.get('chunk_overlap')}, "
                        f"prefix={top.get('context_prefix')}) score {top.get('composite', 0):.4f}"
                    )
            elif layer_name == "layer1_retrieval":
                ranking = summary.get("ranking", [])
                if ranking:
                    summary_lines.append(
                        f"- Retrieval: Best = {ranking[0].get('config_name')} "
                        f"score {ranking[0].get('composite', 0):.4f}"
                    )
            elif layer_name == "layer2_params":
                sweeps = summary.get("param_sweeps", {})
                for param, data in sweeps.items():
                    summary_lines.append(
                        f"- {param}: best={data.get('best_value')} "
                        f"(score {data.get('best_composite', 0):.4f})"
                    )
            elif layer_name in ("layer_embed", "layer_rerank"):
                ranking = summary.get("ranking", [])
                kind = "Embedding" if layer_name == "layer_embed" else "Reranker"
                if ranking:
                    top = ranking[0]
                    summary_lines.append(
                        f"- {kind} Shootout: Best = {top.get('model_name')} "
                        f"({top.get('hf_id', '')}) score {top.get('composite', 0):.4f}"
                    )
            elif layer_name == "layer_cross":
                ranking = summary.get("ranking", [])
                if ranking:
                    top = ranking[0]
                    summary_lines.append(
                        f"- Cross-Model Sweep: Best = {top.get('combo_key')} "
                        f"(embed={top.get('embed_short', '')}, rerank={top.get('rerank_short', '')}) "
                        f"score {top.get('composite', 0):.4f}"
                    )
            elif layer_name == "layer_llm":
                ranking = summary.get("ranking", [])
                if ranking:
                    top = ranking[0]
                    summary_lines.append(
                        f"- Chat LLM Comparison: Best = {top.get('model_name')} "
                        f"avg {top.get('avg_response_time_s', 0):.2f}s/query, "
                        f"{top.get('timeout_rate', 0)*100:.0f}% timeout rate, "
                        f"avg answer {top.get('avg_answer_length', 0):.0f} chars"
                    )
            elif layer_name == "layer4_judge":
                summary_lines.append(
                    f"- LLM Judge: relevance={summary.get('avg_relevance', 0):.2f}, "
                    f"accuracy={summary.get('avg_accuracy', 0):.2f}, "
                    f"completeness={summary.get('avg_completeness', 0):.2f}"
                )
            elif layer_name == "layer6_failures":
                total = summary.get("total_failures", 0)
                if total > 0:
                    cats = summary.get("by_category", {})
                    cat_str = ", ".join(f"{k}={v}" for k, v in cats.items())
                    summary_lines.append(f"- Failures: {total} total ({cat_str})")
            elif layer_name == "layer_live":
                summary_lines.append(
                    f"- Live Pipeline: {summary.get('benchmark_queries_tested', 0)} queries, "
                    f"avg latency {summary.get('avg_latency_ms', 0):.0f}ms, "
                    f"avg confidence {summary.get('avg_confidence', 0):.3f}, "
                    f"keyword hit rate {summary.get('keyword_hit_rate', 0):.3f}, "
                    f"adversarial pass rate {summary.get('adversarial_pass_rate', 0):.1%}"
                )
            elif layer_name == "layer_ceiling":
                summary_lines.append(
                    f"- Ceiling Comparison: ceiling_avg={summary.get('ceiling_avg_composite', 0):.3f}, "
                    f"local_avg={summary.get('local_avg_composite', 0):.3f}, "
                    f"delta={summary.get('model_quality_delta', 0):.3f}, "
                    f"ceiling wins {summary.get('ceiling_wins', 0)} queries"
                )

        if not summary_lines:
            return None

        user_prompt = "Here are the RAG pipeline benchmark results:\n" + "\n".join(summary_lines)

        resp = httpx.post(
            f"http://localhost:{_LLM_CHAT_PORT}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are analyzing benchmark results for a RAG system. "
                            "Explain the results in plain English for a non-ML-expert. "
                            "Be concise — 3-5 bullet points on what these results mean "
                            "and what actions to take."
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 512,
                "temperature": 0.4,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        logger.info("LLM auto-analysis generated successfully")
        return content

    except Exception as e:
        logger.warning(f"LLM auto-analysis failed (non-blocking): {e}")
        return None


def _run_benchmark(job_id: str, layers: List[str], topic: str, corpus_path: Optional[str]):
    """Execute v2 benchmark in background thread. Updates _current_job in place."""
    # Lazy import to avoid circular dependency during module init.
    # Safe here because this runs in a background thread after startup.
    import routers.admin_benchmark as _pkg

    try:
        from benchmark.config import BenchmarkConfig
        from benchmark.checkpoint import CheckpointManager

        config = BenchmarkConfig(
            corpus_dir=corpus_path or "/app/data/Synthetic Reports",
            topic=topic,
        )

        # Load provider config
        providers_path = Path("/app/data/config/benchmark_providers.json")
        if providers_path.exists():
            try:
                prov_data = json.loads(providers_path.read_text(encoding="utf-8"))
                judge_cfg = prov_data.get("judge", {})
                ceiling_cfg = prov_data.get("ceiling", {})
                config.judge_provider = judge_cfg.get("provider", "local")
                config.judge_model = judge_cfg.get("model", "")
                config.ceiling_provider = ceiling_cfg.get("provider", "local")
                config.ceiling_model = ceiling_cfg.get("model", "")
            except Exception as e:
                logger.warning("Could not load benchmark_providers.json: %s", e)

        output_dir = config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save config for history
        config_path = Path(output_dir) / "benchmark_config.json"
        config_path.write_text(
            json.dumps(config.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        checkpoint = CheckpointManager(Path(output_dir) / "checkpoints")

        with _job_lock:
            _pkg._current_job["status"] = "running"
            _pkg._current_job["output_dir"] = output_dir
            _pkg._current_job["phases_total"] = len(layers)

        layer_results = {}
        phase_durations: Dict[str, float] = {}

        for idx, layer_name in enumerate(layers):
            layer_cls = _get_layer_class(layer_name)
            if layer_cls is None:
                logger.warning(f"Unknown layer: {layer_name}")
                continue

            with _job_lock:
                _pkg._current_job["current_phase"] = layer_name
                _pkg._current_job["phases_completed"] = idx
                _pkg._current_job["progress_pct"] = round(idx / len(layers) * 100)

            try:
                layer = layer_cls(config, checkpoint)
                result = layer.run()

                phase_data = _format_layer_for_frontend(layer_name, result, output_dir)
                layer_results[layer_name] = phase_data
                phase_durations[layer_name] = result.duration_seconds

                # Time estimation
                phases_remaining = len(layers) - (idx + 1)
                avg_duration = sum(phase_durations.values()) / len(phase_durations)
                estimated_remaining = avg_duration * phases_remaining

                with _job_lock:
                    _pkg._current_job["results_so_far"] = dict(layer_results)
                    _pkg._current_job["phase_durations"] = dict(phase_durations)
                    _pkg._current_job["estimated_remaining_s"] = round(estimated_remaining, 1)

            except Exception as e:
                logger.error(f"Layer {layer_name} failed: {e}", exc_info=True)
                layer_results[layer_name] = {
                    "duration_s": 0,
                    "n_variants": 0,
                    "winner": f"FAILED: {e}",
                    "ranking": [],
                }

        # Extract winners from optimal configs
        winners = _extract_winners(output_dir)

        # Auto-apply winners for auto-tune jobs
        apply_result = {}
        if job_id.startswith("autotune_") and winners:
            try:
                apply_result = _auto_apply_winners(winners)
            except Exception as e:
                logger.error("Auto-apply failed: " + str(e), exc_info=True)

        # Build overall results in the shape the frontend expects
        results_data: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phases": layer_results,
            "config": config.to_dict(),
        }

        # Find overall best across L0/L1
        best_score = 0.0
        best_variant = None
        best_phase = None
        for lname in ["layer1_retrieval", "layer0_chunking"]:
            if lname in layer_results:
                ranking = layer_results[lname].get("ranking", [])
                if ranking and ranking[0].get("composite", 0) > best_score:
                    best_score = ranking[0]["composite"]
                    best_variant = ranking[0]["variant"]
                    best_phase = lname

        if best_variant:
            results_data["overall_winner"] = {
                "phase": best_phase,
                "variant": best_variant,
                "composite": best_score,
            }

        # Save results for history
        results_path = Path(output_dir) / "results.json"
        results_path.write_text(
            json.dumps(results_data, indent=2, default=str),
            encoding="utf-8",
        )

        # LLM analysis
        llm_analysis = _generate_llm_analysis(output_dir)
        if llm_analysis:
            results_data["llm_analysis"] = llm_analysis
            results_path.write_text(
                json.dumps(results_data, indent=2, default=str),
                encoding="utf-8",
            )

        # Collect chart paths from L5
        charts_dir = Path(output_dir) / "layer5_analysis" / "charts"
        chart_paths = []
        if charts_dir.exists():
            chart_paths = [str(p) for p in sorted(charts_dir.glob("*.png"))]

        # Verify chat server health after benchmark completes
        chat_healthy = False
        try:
            health_resp = httpx.get(
                "http://localhost:" + str(_LLM_CHAT_PORT) + "/health",
                timeout=3.0,
            )
            chat_healthy = health_resp.status_code == 200
        except Exception:
            pass

        if not chat_healthy:
            logger.warning(
                "Chat LLM not healthy after benchmark completion — "
                "attempting final restart"
            )
            try:
                from services.llm_server_manager import get_server_manager
                mgr = get_server_manager()
                mgr.start("chat")
                # Brief wait for startup
                for attempt in range(60):
                    try:
                        r = httpx.get(
                            "http://localhost:" + str(_LLM_CHAT_PORT) + "/health",
                            timeout=3.0,
                        )
                        if r.status_code == 200:
                            chat_healthy = True
                            logger.info("Chat LLM recovered after final restart")
                            break
                    except Exception:
                        pass
                    import time as _time
                    _time.sleep(1)
            except Exception as e:
                logger.error("Final chat LLM restart failed: %s", e)

        with _job_lock:
            _pkg._current_job["status"] = "completed"
            _pkg._current_job["current_phase"] = None
            _pkg._current_job["phases_completed"] = len(layers)
            _pkg._current_job["progress_pct"] = 100
            _pkg._current_job["results"] = results_data
            _pkg._current_job["charts"] = chart_paths
            _pkg._current_job["winners"] = winners
            _pkg._current_job["llm_analysis"] = llm_analysis
            _pkg._current_job["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _pkg._current_job["auto_applied"] = apply_result
            _pkg._current_job["chat_server_healthy"] = chat_healthy

    except Exception as e:
        logger.error(f"Benchmark job {job_id} failed: {e}", exc_info=True)
        with _job_lock:
            if _pkg._current_job and _pkg._current_job.get("job_id") == job_id:
                _pkg._current_job["status"] = "failed"
                _pkg._current_job["error"] = str(e)
