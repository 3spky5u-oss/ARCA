"""
Hardware, status, config, and recalibration endpoints.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException

from . import router
from .models import ConfigUpdate
from config import runtime_config
from services.admin_auth import verify_admin
from utils.llm import get_llm_client, get_server_manager

logger = logging.getLogger(__name__)


@router.get("/hardware")
async def get_hardware(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """
    Get hardware specs, profile, and model recommendations.

    Returns GPU/CPU/RAM info, auto-assigned profile, VRAM availability,
    and model recommendations for the detected hardware tier.
    """
    try:
        from services.hardware import get_hardware_info, get_vram_available_mb

        hw = get_hardware_info()
        result = hw.to_dict()

        # Add live VRAM availability
        result["gpu"]["vram_available_mb"] = get_vram_available_mb()
        result["gpu"]["vram_available_gb"] = round(get_vram_available_mb() / 1024, 1)

        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Hardware info failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Hardware detection failed: {e}")


@router.get("/status")
async def get_system_status(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """
    Get comprehensive system status.

    Returns:
        - llama.cpp server status and loaded models
        - Storage usage (uploads, reports)
        - Active sessions count
        - Runtime config values
        - Uptime info
    """

    status = {
        "timestamp": datetime.now().isoformat(),
        "llm": {"status": "unknown", "models": []},
        "storage": {"uploads_mb": 0, "reports_mb": 0, "total_mb": 0},
        "sessions": {"active": 0},
        "config": {},
        "rag": {"knowledge_chunks": 0, "session_chunks": 0},
    }

    # Check LLM servers
    try:
        from services.llm_config import get_model_path
        mgr = get_server_manager()
        running = mgr.list_running()
        available = mgr.get_available_models()

        loaded = []
        for slot_name, info in running.items():
            if info.get("alive"):
                model_name = info.get("model", "unknown")
                # Get file size for VRAM estimate
                model_path = get_model_path(model_name) if model_name != "unknown" else None
                size_gb = round(model_path.stat().st_size / (1024**3), 1) if model_path and model_path.exists() else 0
                # Parse quantization from filename (e.g., "Q4_K_M" from "GLM-4.7-Flash-Q4_K_M.gguf")
                quant = ""
                name_stem = model_name.replace(".gguf", "")
                for q in ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K", "IQ4_XS", "IQ3_XXS"]:
                    if q.lower() in name_stem.lower() or q in name_stem:
                        quant = q
                        break
                parallel = info.get("parallel", 1)
                loaded.append({
                    "name": model_name,
                    "slot": slot_name,
                    "port": info.get("port"),
                    "pid": info.get("pid"),
                    "parallel": parallel,
                    # Fields expected by frontend StatusTab
                    "size_gb": size_gb,
                    "vram_gb": size_gb,  # n_gpu_layers=-1 means fully offloaded
                    "cpu_gb": 0,
                    "gpu_pct": 100,
                    "quantization": quant,
                    "parameter_size": "",
                })

        status["llm"] = {
            "status": "connected" if any(s.get("alive") for s in running.values()) else "stopped",
            "models": available,
            "loaded_models": loaded,
        }

    except Exception as e:
        logger.error(f"LLM server check failed: {e}")
        status["llm"] = {
            "status": "error",
            "error": str(e),
        }

    # Storage stats
    try:
        from main import get_storage_stats, UPLOAD_DIR, REPORTS_DIR

        # Ensure directories exist before getting stats
        UPLOAD_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)

        stats = get_storage_stats()
        status["storage"] = {
            "uploads_count": stats["uploads"]["count"],
            "uploads_mb": round(stats["uploads"]["bytes"] / (1024 * 1024), 2),
            "reports_count": stats["reports"]["count"],
            "reports_mb": round(stats["reports"]["bytes"] / (1024 * 1024), 2),
            "total_mb": stats["total_mb"],
        }
    except Exception as e:
        logger.warning(f"Could not get storage stats: {e}")
        status["storage"] = {
            "uploads_count": 0,
            "uploads_mb": 0,
            "reports_count": 0,
            "reports_mb": 0,
            "total_mb": 0,
            "note": "Storage directories not available",
        }

    # Session info
    try:
        from routers.upload import files_db

        status["sessions"]["active_files"] = len(files_db)
    except Exception as e:
        logger.warning(f"Could not get session info: {e}")

    # Runtime config
    try:
        from config import runtime_config

        status["config"] = runtime_config.to_dict()
    except Exception as e:
        status["config"] = {"error": str(e)}

    # RAG stats
    try:
        from tools.cohesionn import get_stats

        rag_stats = get_stats()
        # topics is a dict like {"topic_name": 1000}, convert to list for frontend
        topics_dict = rag_stats.get("topics", {})
        topics_list = list(topics_dict.keys()) if isinstance(topics_dict, dict) else []
        status["rag"] = {
            "knowledge_chunks": rag_stats.get("total_chunks", 0),
            "topics": topics_list,
        }
    except Exception as e:
        logger.warning(f"Could not get RAG stats: {e}")

    # Embedder health (subprocess isolation status)
    try:
        from tools.cohesionn.embeddings import get_embedder, EmbeddingProxy
        embedder = get_embedder()
        status["embedder"] = {
            "type": "subprocess" if isinstance(embedder, EmbeddingProxy) else "in_process",
            "worker_alive": getattr(embedder, "_process", None) is not None and embedder._process.is_alive(),
            "dimension": embedder.dimension,
            "cpu_fallback": getattr(embedder, "_cpu_fallback", False),
            "crash_count": len(getattr(embedder, "_crash_times", [])),
        }
    except Exception as e:
        status["embedder"] = {"type": "unknown", "error": str(e)}

    # Service health checks
    services = {}

    # Redis health
    try:
        from services.redis_client import get_redis
        redis = await get_redis()
        services["redis"] = await redis.health_check()
    except Exception as e:
        services["redis"] = {"status": "error", "error": str(e)}

    # PostgreSQL health
    try:
        from services.database import get_database
        db = await get_database()
        services["postgres"] = await db.health_check()
    except Exception as e:
        services["postgres"] = {"status": "error", "error": str(e)}

    # Qdrant health
    try:
        import os
        import httpx
        qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
        async with httpx.AsyncClient(timeout=3.0) as client:
            start_t = time.time()
            resp = await client.get(f"{qdrant_url}/collections")
            latency_ms = (time.time() - start_t) * 1000
            if resp.status_code == 200:
                services["qdrant"] = {"status": "connected", "latency_ms": round(latency_ms, 1)}
            else:
                services["qdrant"] = {"status": "error", "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        services["qdrant"] = {"status": "error", "error": str(e)}

    # Neo4j health
    try:
        from services.neo4j_client import get_neo4j_client
        neo4j = get_neo4j_client()
        if neo4j:
            services["neo4j"] = neo4j.health_check()
        else:
            services["neo4j"] = {"status": "disabled"}
    except Exception as e:
        services["neo4j"] = {"status": "error", "error": str(e)}

    # SearXNG health
    try:
        searxng_url = (runtime_config.searxng_url or "http://searxng:8080").rstrip("/")
        if not runtime_config.searxng_enabled:
            services["searxng"] = {"status": "disabled", "url": searxng_url}
        else:
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                start_t = time.time()
                # Use healthz or base URL instead of a full search query
                resp = await client.get(f"{searxng_url}/healthz")
                latency_ms = (time.time() - start_t) * 1000
                if resp.status_code == 200:
                    services["searxng"] = {"status": "connected", "latency_ms": round(latency_ms, 1), "url": searxng_url}
                elif resp.status_code == 404:
                    # /healthz not available, fall back to base URL check
                    resp2 = await client.get(searxng_url)
                    latency_ms = (time.time() - start_t) * 1000
                    if resp2.status_code in (200, 301, 302):
                        services["searxng"] = {"status": "connected", "latency_ms": round(latency_ms, 1), "url": searxng_url}
                    else:
                        services["searxng"] = {"status": "error", "error": f"HTTP {resp2.status_code}", "url": searxng_url}
                else:
                    services["searxng"] = {"status": "error", "error": f"HTTP {resp.status_code}", "url": searxng_url}
    except Exception as e:
        services["searxng"] = {
            "status": "error",
            "error": str(e),
            "url": (runtime_config.searxng_url or "http://searxng:8080"),
        }

    status["services"] = services

    # System metrics (CPU, RAM, Disk, GPU)
    try:
        from services.system_metrics import get_system_metrics
        status["system"] = await get_system_metrics()
    except Exception as e:
        logger.warning(f"Could not get system metrics: {e}")
        status["system"] = {"error": str(e)}

    return status


@router.get("/config")
async def get_config(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Get current runtime configuration."""

    try:
        from config import runtime_config

        return {
            "success": True,
            "config": runtime_config.to_dict(),
        }
    except ImportError as e:
        logger.error(f"Config import failed: {e}")
        raise HTTPException(status_code=503, detail="RuntimeConfig not available")
    except Exception as e:
        logger.error(f"Config fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Config error: {str(e)}")


@router.put("/config")
async def update_config(update: ConfigUpdate, _: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """
    Update runtime configuration.

    Changes take effect immediately without restart.
    """

    try:
        from config import runtime_config

        # Filter out None values
        updates = {k: v for k, v in update.dict().items() if v is not None}

        if not updates:
            return {"success": True, "updated": [], "message": "No changes"}

        # Validate model names (basic format check - GGUF availability checked at startup)
        model_fields = {
            "model_chat", "model_code", "model_expert",
            "model_vision", "model_vision_heavy", "model_vision_structured",
            "model_chat_finetuned", "raptor_summary_model",
            "hyde_model", "community_summary_model",
        }
        model_updates = {k: v for k, v in updates.items() if k in model_fields}
        if model_updates:
            # Just validate format - actual GGUF availability checked at server start
            import re as _re
            for k, v in model_updates.items():
                if v and not _re.match(r'^[a-zA-Z0-9._:\-/]+$', v):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid model name format for {k}: '{v}'"
                    )

        # Sanitize string config values
        for k, v in updates.items():
            if isinstance(v, str):
                updates[k] = v.strip()

        result = runtime_config.update(**updates)

        # Persist overrides so they survive container restarts
        if result["updated"]:
            runtime_config.save_overrides()

        return {
            "success": True,
            "updated": result["updated"],
            "ignored": result["ignored"],
            "update_count": result["update_count"],
        }
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=503, detail="RuntimeConfig not available")


@router.post("/config/reset")
async def reset_config(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Reset configuration to environment defaults."""

    try:
        from config import runtime_config

        result = runtime_config.reset_to_defaults()
        return {
            "success": True,
            "changes": result["changes"],
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="RuntimeConfig not available")


@router.post("/recalibrate")
async def recalibrate(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """
    Trigger warmup calibration.

    Runs a test extraction to calibrate model parameters.
    """

    results = {
        "llm_warmup": False,
        "rag_warmup": False,
        "extraction_test": None,
    }

    # Warm up LLM
    try:
        from config import runtime_config

        model = runtime_config.model_chat
        client = get_llm_client("chat")

        start = time.time()
        client.chat(model=model, messages=[{"role": "user", "content": "Hi"}], options={"num_predict": 1})
        results["llm_warmup"] = True
        results["llm_warmup_ms"] = int((time.time() - start) * 1000)
    except Exception as e:
        results["llm_warmup_error"] = str(e)

    # Warm up RAG
    try:
        from tools.cohesionn import warm_models

        start = time.time()
        warm_models()
        results["rag_warmup"] = True
        results["rag_warmup_ms"] = int((time.time() - start) * 1000)
    except Exception as e:
        results["rag_warmup_error"] = str(e)

    # Run test extraction if reference docs exist
    try:
        qa_dir = Path(__file__).parent.parent.parent / "data" / "qa_reference" / "docs"
        test_pdf = qa_dir / "clean_digital.pdf"

        if test_pdf.exists():
            from tools.readd.pipeline import ReaddPipeline

            pipeline = ReaddPipeline(auto_escalate=False)
            start = time.time()
            result = pipeline.process(test_pdf)
            elapsed = int((time.time() - start) * 1000)

            results["extraction_test"] = {
                "success": result.success,
                "extractor": result.extractors_tried[-1] if result.extractors_tried else None,
                "text_length": len(result.text) if result.text else 0,
                "processing_ms": elapsed,
            }
    except Exception as e:
        results["extraction_test_error"] = str(e)

    return {"success": True, "results": results}
