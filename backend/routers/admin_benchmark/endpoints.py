"""HTTP endpoint handlers for admin benchmark router."""

import json
import logging
import re
import time
import threading
from pathlib import Path
from typing import Any, Dict

import httpx
from fastapi import Depends, Request, UploadFile, File
from fastapi.responses import JSONResponse

from services.admin_auth import verify_admin

logger = logging.getLogger(__name__)

from . import router, _job_lock, _LLM_CHAT_PORT, ALL_LAYERS, QUICK_LAYERS, BENCHMARKS_DIR
from .models import (
    BenchmarkStartRequest, ApplyWinnersRequest,
    ProviderConfigRequest, ProviderTestRequest, AutoTuneRequest,
)
from .helpers import (
    _resolve_layers, _extract_winners, _auto_apply_winners, _run_benchmark,
)


@router.post("/start")
async def start_benchmark(
    request: BenchmarkStartRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Start a benchmark run.

    Phases:
      - "quick" = L0 chunking sweep + L1 retrieval sweep
      - "full" = all 7 layers (L0-L6)
      - Comma-separated layer names for custom selection
    """
    import routers.admin_benchmark as _pkg

    with _job_lock:
        if _pkg._current_job and _pkg._current_job.get("status") in ("starting", "running"):
            return {
                "error": "A benchmark is already running",
                "job_id": _pkg._current_job["job_id"],
                "status": _pkg._current_job["status"],
            }

    layers = _resolve_layers(request.phases)
    if not layers:
        return {"error": f"No valid layers in: {request.phases}"}

    # Pull default topic from active domain lexicon pipeline config
    from domain_loader import get_pipeline_config
    pipeline_cfg = get_pipeline_config()
    topic = request.topic or pipeline_cfg.get("default_topic", "benchmark")
    job_id = f"bench_{int(time.time())}"

    with _job_lock:
        _pkg._current_job = {
            "job_id": job_id,
            "status": "starting",
            "phases": layers,
            "topic": topic,
            "corpus_path": request.corpus_path,
            "current_phase": None,
            "phases_completed": 0,
            "phases_total": len(layers),
            "progress_pct": 0,
            "estimated_remaining_s": None,
            "phase_durations": {},
            "results_so_far": {},
            "results": None,
            "charts": [],
            "winners": {},
            "llm_analysis": None,
            "error": None,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_at": None,
            "output_dir": None,
        }

    thread = threading.Thread(
        target=_run_benchmark,
        args=(job_id, layers, topic, request.corpus_path),
        daemon=True,
        name=f"benchmark-{job_id}",
    )
    thread.start()

    return {"job_id": job_id, "status": "starting", "phases": layers}


@router.get("/providers")
async def get_providers(
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Get current provider config. API keys are masked."""
    providers_path = Path("/app/data/config/benchmark_providers.json")
    if not providers_path.exists():
        return {
            "judge": {"provider": "local", "model": "", "api_key": "", "api_key_env": "", "rate_limit": 1.0, "base_url": ""},
            "ceiling": {"provider": "local", "model": "", "api_key": "", "api_key_env": "", "rate_limit": 1.0, "base_url": ""},
        }

    data = json.loads(providers_path.read_text(encoding="utf-8"))

    # Mask API keys
    for role in ("judge", "ceiling"):
        cfg = data.get(role, {})
        raw_key = cfg.get("api_key", "")
        if raw_key and len(raw_key) > 4:
            cfg["api_key"] = "..." + raw_key[-4:]
        elif raw_key:
            cfg["api_key"] = "...masked"

    return data


@router.put("/providers")
async def update_providers(
    request: ProviderConfigRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Update provider config. Writes to benchmark_providers.json."""
    providers_path = Path("/app/data/config/benchmark_providers.json")

    # Load existing
    if providers_path.exists():
        data = json.loads(providers_path.read_text(encoding="utf-8"))
    else:
        data = {
            "judge": {"provider": "local", "model": "", "api_key": "", "api_key_env": "", "rate_limit": 1.0, "base_url": ""},
            "ceiling": {"provider": "local", "model": "", "api_key": "", "api_key_env": "", "rate_limit": 1.0, "base_url": ""},
        }

    # Update
    for role, incoming in [("judge", request.judge), ("ceiling", request.ceiling)]:
        if incoming is None:
            continue
        cfg = data.get(role, {})
        for key in ("provider", "model", "api_key", "api_key_env", "rate_limit", "base_url"):
            if key in incoming:
                # Don't overwrite with masked key
                if key == "api_key" and str(incoming[key]).startswith("..."):
                    continue
                cfg[key] = incoming[key]
        data[role] = cfg

    providers_path.parent.mkdir(parents=True, exist_ok=True)
    providers_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Updated benchmark provider config")

    return {"success": True}


@router.post("/providers/test")
async def test_provider(
    request: ProviderTestRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Test a provider connection."""
    try:
        from benchmark.providers import get_provider

        provider = get_provider(request.provider, {
            "model": request.model,
            "api_key": request.api_key,
            "base_url": request.base_url,
        })

        success, message = provider.test_connection()
        return {"success": success, "message": message}

    except Exception as e:
        return {"success": False, "message": str(e)}


@router.post("/auto-tune")
async def start_auto_tune(
    request: AutoTuneRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Run auto-tune: L0+L1+L2, optionally L3+L4 if judge is configured.

    Always uses local LLM for retrieval optimization. If include_judge=True
    and a judge provider is configured, also runs answer generation + judging.
    """
    import routers.admin_benchmark as _pkg

    with _job_lock:
        if _pkg._current_job and _pkg._current_job.get("status") in ("starting", "running"):
            return {
                "error": "A benchmark is already running",
                "job_id": _pkg._current_job["job_id"],
                "status": _pkg._current_job["status"],
            }

    layers = ["layer0_chunking", "layer1_retrieval", "layer2_params"]

    if request.include_judge:
        layers.extend(["layer3_answers", "layer4_judge"])

    from domain_loader import get_pipeline_config
    pipeline_cfg = get_pipeline_config()
    topic = request.topic or pipeline_cfg.get("default_topic", "benchmark")
    job_id = "autotune_" + str(int(time.time()))

    with _job_lock:
        _pkg._current_job = {
            "job_id": job_id,
            "status": "starting",
            "phases": layers,
            "topic": topic,
            "corpus_path": None,
            "current_phase": None,
            "phases_completed": 0,
            "phases_total": len(layers),
            "progress_pct": 0,
            "estimated_remaining_s": None,
            "phase_durations": {},
            "results_so_far": {},
            "results": None,
            "charts": [],
            "winners": {},
            "llm_analysis": None,
            "error": None,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_at": None,
            "output_dir": None,
        }

    thread = threading.Thread(
        target=_run_benchmark,
        args=(job_id, layers, topic, None),
        daemon=True,
        name="autotune-" + job_id,
    )
    thread.start()

    return {"job_id": job_id, "status": "starting", "phases": layers}


@router.get("/status")
async def get_benchmark_status(
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Get current benchmark run status."""
    import routers.admin_benchmark as _pkg

    with _job_lock:
        if not _pkg._current_job:
            return {"status": "idle", "message": "No benchmark running or completed"}

        return {
            "job_id": _pkg._current_job["job_id"],
            "status": _pkg._current_job["status"],
            "current_phase": _pkg._current_job.get("current_phase"),
            "phases": _pkg._current_job.get("phases", []),
            "phases_completed": _pkg._current_job.get("phases_completed", 0),
            "phases_total": _pkg._current_job.get("phases_total", 0),
            "progress_pct": _pkg._current_job.get("progress_pct", 0),
            "estimated_remaining_s": _pkg._current_job.get("estimated_remaining_s"),
            "phase_durations": _pkg._current_job.get("phase_durations", {}),
            "results_so_far": _pkg._current_job.get("results_so_far", {}),
            "winners": _pkg._current_job.get("winners", {}),
            "error": _pkg._current_job.get("error"),
            "auto_applied": _pkg._current_job.get("auto_applied"),
            "started_at": _pkg._current_job.get("started_at"),
            "completed_at": _pkg._current_job.get("completed_at"),
        }


@router.get("/results")
async def get_benchmark_results(
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Get completed benchmark results, charts, and winners."""
    import routers.admin_benchmark as _pkg

    with _job_lock:
        if not _pkg._current_job:
            return {"error": "No benchmark results available"}

        if _pkg._current_job["status"] != "completed":
            return {
                "error": "Benchmark not yet completed",
                "status": _pkg._current_job["status"],
            }

        output_dir = _pkg._current_job.get("output_dir", "")

        # Build chart URLs from L5 output
        chart_urls = []
        charts_dir = Path(output_dir) / "layer5_analysis" / "charts" if output_dir else None
        if charts_dir and charts_dir.exists():
            run_id = Path(output_dir).name
            for chart_path in sorted(charts_dir.glob("*.png")):
                chart_urls.append({
                    "name": chart_path.stem.replace("_", " ").title(),
                    "filename": chart_path.name,
                    "url": f"/api/admin/benchmark/chart/{run_id}/{chart_path.name}",
                })

        return {
            "results": _pkg._current_job.get("results", {}),
            "charts": chart_urls,
            "winners": _pkg._current_job.get("winners", {}),
            "llm_analysis": _pkg._current_job.get("llm_analysis"),
            "output_dir": output_dir,
            "started_at": _pkg._current_job.get("started_at"),
            "completed_at": _pkg._current_job.get("completed_at"),
        }


@router.post("/apply")
async def apply_winners(
    request: ApplyWinnersRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Apply benchmark winners to RuntimeConfig and update the auto profile.

    Body: { winners: { bm25_weight: 0.3, reranker_candidates: 10, ... } }
    Supported keys: Any RuntimeConfig field.
    """
    result = _auto_apply_winners(request.winners)
    return {"success": result.get("applied", False), **result}


@router.get("/history")
async def get_benchmark_history(
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """List previous benchmark runs from data/benchmarks/v2/ directory."""
    runs = []

    if not BENCHMARKS_DIR.exists():
        return {"runs": []}

    for run_dir in sorted(BENCHMARKS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue

        results_path = run_dir / "results.json"
        report_path = run_dir / "layer5_analysis" / "report.md"

        entry: Dict[str, Any] = {
            "id": run_dir.name,
            "timestamp": run_dir.name,
            "has_results": results_path.exists(),
            "has_report": report_path.exists(),
        }

        if results_path.exists():
            try:
                data = json.loads(results_path.read_text(encoding="utf-8"))
                entry["phases"] = list(data.get("phases", {}).keys())
                entry["overall_winner"] = data.get("overall_winner")
                entry["timestamp_str"] = data.get("timestamp", run_dir.name)

                # Count charts
                charts_dir = run_dir / "layer5_analysis" / "charts"
                if charts_dir.exists():
                    entry["chart_count"] = len(list(charts_dir.glob("*.png")))
                else:
                    entry["chart_count"] = 0
            except Exception:
                pass

        runs.append(entry)

    return {"runs": runs}


@router.get("/history/{run_id}")
async def get_history_run(
    run_id: str,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Get results for a specific historical benchmark run."""
    run_dir = BENCHMARKS_DIR / run_id
    results_path = run_dir / "results.json"

    if not results_path.exists():
        return {"error": f"Run not found: {run_id}"}

    data = json.loads(results_path.read_text(encoding="utf-8"))

    # Build chart list from L5 output
    charts = []
    charts_dir = run_dir / "layer5_analysis" / "charts"
    if charts_dir.exists():
        for chart_path in sorted(charts_dir.glob("*.png")):
            charts.append({
                "name": chart_path.stem.replace("_", " ").title(),
                "filename": chart_path.name,
                "url": f"/api/admin/benchmark/chart/{run_id}/{chart_path.name}",
            })

    # Extract winners from optimal configs
    winners = _extract_winners(str(run_dir))

    return {
        "results": data,
        "charts": charts,
        "winners": winners,
        "llm_analysis": data.get("llm_analysis"),
    }


@router.get("/chart/{run_id}/{filename}")
async def get_chart(
    run_id: str,
    filename: str,
    _: bool = Depends(verify_admin),
):
    """Serve a benchmark chart image from L5 output."""
    from fastapi.responses import FileResponse

    chart_path = (BENCHMARKS_DIR / run_id / "layer5_analysis" / "charts" / filename).resolve()

    # Security: prevent path traversal
    if not chart_path.is_relative_to(BENCHMARKS_DIR.resolve()):
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Access denied")

    if not chart_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Chart not found")

    return FileResponse(
        path=chart_path,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@router.get("/report/{run_id}")
async def get_report(
    run_id: str,
    _: bool = Depends(verify_admin),
):
    """Serve the L5 analysis markdown report."""
    from fastapi import HTTPException

    report_path = (BENCHMARKS_DIR / run_id / "layer5_analysis" / "report.md").resolve()

    if not report_path.is_relative_to(BENCHMARKS_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not generated yet. Run Layer 5 first.")

    content = report_path.read_text(encoding="utf-8")
    return {"report": content, "run_id": run_id}


@router.get("/layers")
async def list_layers(
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """List all available benchmark layers with metadata."""
    layer_info = {
        "layer0_chunking": {"label": "Chunking Sweep", "group": "retrieval", "description": "Sweep chunk size, overlap, and context prefix configurations"},
        "layer1_retrieval": {"label": "Retrieval Config", "group": "retrieval", "description": "Toggle retrieval features (BM25, HyDE, RAPTOR, GraphRAG, etc.)"},
        "layer2_params": {"label": "Parameter Tuning", "group": "retrieval", "description": "Continuous parameter sweep (weights, thresholds, top_k)"},
        "layer_embed": {"label": "Embedding Shootout", "group": "model", "description": "Compare embedding models (requires L0 results)"},
        "layer_rerank": {"label": "Reranker Shootout", "group": "model", "description": "Compare reranker models (requires L0 results)"},
        "layer_cross": {"label": "Cross-Model Sweep", "group": "model", "description": "Top-N chunking x embedding x reranker combinations (requires L0, embed, rerank results)"},
        "layer_llm": {"label": "Chat LLM Comparison", "group": "model", "description": "Compare chat LLM models by generating answers with each (requires L0 results)"},
        "layer3_answers": {"label": "Answer Generation", "group": "evaluation", "description": "Generate answers with optimal retrieval config"},
        "layer4_judge": {"label": "LLM-as-Judge", "group": "evaluation", "description": "Score answers with configured judge provider (requires L3 results)"},
        "layer5_analysis": {"label": "Analysis & Charts", "group": "evaluation", "description": "Statistical analysis, heatmaps, and markdown report"},
        "layer6_failures": {"label": "Failure Analysis", "group": "evaluation", "description": "Categorize retrieval and answer failures"},
        "layer_live": {"label": "Live Pipeline Test", "group": "cloud", "description": "End-to-end pipeline test via MCP API"},
        "layer_ceiling": {"label": "Frontier vs Local LLM Ceiling", "group": "cloud", "description": "Compare frontier/cloud LLM answer quality against local LLM"},
    }
    return {"layers": layer_info, "presets": {"quick": QUICK_LAYERS, "full": ALL_LAYERS}}


@router.post("/upload-corpus")
async def upload_benchmark_corpus(
    file: UploadFile = File(...),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Upload a document to use as benchmark corpus."""
    corpus_dir = Path("data/benchmark/corpus")
    corpus_dir.mkdir(parents=True, exist_ok=True)

    dest = corpus_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)

    return {
        "filename": file.filename,
        "size_mb": round(len(content) / (1024 * 1024), 2),
        "path": str(dest),
    }


@router.get("/corpus")
async def list_benchmark_corpus(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """List available benchmark corpus files."""
    corpus_dir = Path("data/benchmark/corpus")
    if not corpus_dir.exists():
        return {"files": [], "corpus_dir": str(corpus_dir)}

    files = []
    for f in corpus_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ('.pdf', '.docx', '.txt', '.md'):
            files.append({
                "name": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                "path": str(f),
            })
    return {"files": files, "corpus_dir": str(corpus_dir)}


@router.delete("/corpus/{filename}")
async def delete_benchmark_corpus(
    filename: str,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Delete a benchmark corpus file."""
    corpus_dir = Path("data/benchmark/corpus")
    target = corpus_dir / filename
    if not target.exists():
        return {"error": f"File not found: {filename}"}
    target.unlink()
    return {"deleted": filename}


@router.post("/discuss")
async def discuss_benchmark(
    request: Request,
    _: bool = Depends(verify_admin),
):
    """Discuss benchmark results with LLM in a conversational manner."""
    import routers.admin_benchmark as _pkg

    body = await request.json()
    question = body.get("question", "")
    if not question:
        return JSONResponse({"error": "No question provided"}, status_code=400)

    # Get current results
    with _job_lock:
        results = _pkg._current_job.get("results") if _pkg._current_job else None
        prev_analysis = _pkg._current_job.get("llm_analysis", "") if _pkg._current_job else ""

    if not results:
        return JSONResponse({"error": "No benchmark results available"}, status_code=404)

    # Build context from results (generic — works with both old and v2 shapes)
    context_parts = ["## Benchmark Results Summary\n"]
    for phase_name, phase_data in results.get("phases", {}).items():
        context_parts.append(f"### {phase_name}")
        context_parts.append(f"Winner: {phase_data.get('winner', 'N/A')}")
        context_parts.append(f"Duration: {phase_data.get('duration_s', 0):.1f}s")
        for r in phase_data.get("ranking", []):
            context_parts.append(
                f"  #{r['rank']} {r['variant']} — score: {r.get('composite', 0):.4f}"
            )
        context_parts.append("")

    overall = results.get("overall_winner")
    if overall:
        context_parts.append(
            f"Overall Winner: {overall['variant']} ({overall['phase']}) "
            f"— {overall['composite']:.4f}"
        )

    if prev_analysis:
        context_parts.append(f"\n## Previous AI Analysis\n{prev_analysis}")

    results_context = "\n".join(context_parts)

    system_prompt = (
        "You are an AI model evaluation expert analyzing benchmark results "
        "for the ARCA RAG platform.\nHere are the benchmark results:\n\n"
        + results_context
        + "\n\nAnswer the user's question about these results. Be specific "
        "— reference actual scores, model names, and phase names. "
        "If suggesting changes, explain why based on the data. "
        "Keep responses concise (3-5 sentences unless detail is requested)."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"http://localhost:{_LLM_CHAT_PORT}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse({"error": f"LLM call failed: {e}"}, status_code=500)


@router.post("/generate-queries")
async def generate_benchmark_queries(
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Auto-generate tiered benchmark queries from uploaded corpus using LLM.

    Uses QueryAutoGenerator which resolves the LLM provider from
    benchmark_providers.json (judge section), falling back to local LLM.
    Returns structured queries with tier breakdown.
    """
    corpus_dir = Path("data/benchmark/corpus")
    if not corpus_dir.exists() or not any(corpus_dir.iterdir()):
        return {"error": "No corpus files uploaded. Upload documents first."}

    corpus_files = [f for f in corpus_dir.iterdir() if f.suffix.lower() in ('.pdf', '.docx', '.txt', '.md')]
    if not corpus_files:
        return {"error": "No supported corpus files found."}

    try:
        from benchmark.queries.auto_generator import QueryAutoGenerator

        generator = QueryAutoGenerator()
        queries = generator.generate(str(corpus_dir))

        if not queries:
            return {"error": "Query generation returned no results. Check LLM availability."}

        # Build tier breakdown
        tier_counts: Dict[str, int] = {}
        for q in queries:
            tier_counts[q.tier] = tier_counts.get(q.tier, 0) + 1

        # Save for later use
        queries_path = corpus_dir / "queries.json"
        queries_path.write_text(json.dumps({
            "queries": [q.to_dict() for q in queries],
            "source": "auto_generated",
            "tier_breakdown": tier_counts,
        }, indent=2), encoding="utf-8")

        return {
            "queries": [q.query for q in queries],
            "count": len(queries),
            "source": "auto_generated",
            "tier_breakdown": tier_counts,
        }

    except Exception as e:
        logger.warning("Query generation failed: %s", e)
        return {"error": "Query generation failed: " + str(e), "queries": []}
