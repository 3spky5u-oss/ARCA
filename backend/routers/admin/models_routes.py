"""
Model management endpoints.
"""

import logging
import os
import time
import urllib.parse
from typing import Dict, Any

from fastapi import Depends, HTTPException

from . import router
from .models import ModelPullRequest, ModelAssignRequest
from services.admin_auth import verify_admin
from utils.llm import get_server_manager

logger = logging.getLogger(__name__)


@router.get("/models")
async def list_models(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """
    List available GGUF models and running server slots.

    Returns model files, sizes, running slots, and config assignments.
    """

    try:
        from config import runtime_config
        from services.llm_config import MODELS_DIR, SLOTS, get_model_path

        mgr = get_server_manager()

        # List GGUF files in models directory
        models = []
        models_dir = {
            "path": str(MODELS_DIR),
            "exists": MODELS_DIR.exists(),
            "readable": os.access(MODELS_DIR, os.R_OK),
            "writable": os.access(MODELS_DIR, os.W_OK),
            "error": None,
        }
        if MODELS_DIR.exists():
            try:
                for f in sorted(MODELS_DIR.glob("*.gguf")):
                    size_gb = round(f.stat().st_size / (1024**3), 2)
                    model_info = {
                        "name": f.name,
                        "size_gb": size_gb,
                        "path": str(f),
                    }

                    # Check which slots use this model
                    assigned_to = []
                    for slot_name, slot in SLOTS.items():
                        if slot.gguf_filename == f.name:
                            assigned_to.append(slot_name)
                    model_info["assigned_to"] = assigned_to
                    models.append(model_info)
            except Exception as e:
                models_dir["error"] = str(e)
                logger.warning(f"Failed to read models directory {MODELS_DIR}: {e}")
        else:
            models_dir["error"] = "models directory not found"

        # Sort by size descending
        models.sort(key=lambda x: x["size_gb"], reverse=True)

        # Running slots
        running = mgr.list_running()

        # Config assignments
        config_assignments = {
            "chat": runtime_config.model_chat,
            "code": runtime_config.model_code,
            "expert": runtime_config.model_expert,
            "vision": runtime_config.model_vision,
            "vision_structured": runtime_config.model_vision_structured,
        }

        return {
            "success": True,
            "models": models,
            "running_slots": running,
            "config_assignments": config_assignments,
            "total_size_gb": round(sum(m["size_gb"] for m in models), 2),
            "models_dir": models_dir,
        }

    except Exception as e:
        logger.error(f"Failed to list models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/assign")
async def assign_model(
    request: ModelAssignRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Assign a GGUF model to a slot and hot-swap the running server.

    Updates config, restarts llama-server with the new model, and waits for health.
    """
    from config import runtime_config
    from services.llm_config import MODELS_DIR

    slot = request.slot.strip()
    model = request.model.strip()

    config_key = f"model_{slot}"
    if not hasattr(runtime_config, config_key):
        raise HTTPException(status_code=400, detail=f"Unknown slot: {slot}")

    # Verify GGUF file exists
    model_path = MODELS_DIR / model
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model}")

    old_model = getattr(runtime_config, config_key)

    # Update config
    setattr(runtime_config, config_key, model)
    runtime_config.save_overrides()
    logger.info(f"Model assignment: {slot} = {model} (was {old_model})")

    # Hot-swap the running server (if this slot has one)
    swapped = False
    try:
        mgr = get_server_manager()
        swapped = await mgr.swap_model(slot, model)
    except Exception as e:
        logger.warning(f"Model swap for {slot} failed: {e}")

    return {
        "success": True,
        "slot": slot,
        "model": model,
        "old_model": old_model,
        "server_swapped": swapped,
    }


@router.post("/models/pull")
async def pull_model(
    request: ModelPullRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Model pull is not supported with llama-server.

    Download GGUF files manually to the ./models/ directory instead.
    This endpoint is kept for backward compatibility.
    """

    model_name = request.name.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name required")

    try:
        return {
            "success": False,
            "message": (
                f"Model pulling is not supported with llama-server. "
                f"Download '{model_name}' GGUF file manually to the models/ directory. "
                f"Use: huggingface-cli download <repo> {model_name} --local-dir ./models"
            ),
        }

    except Exception as e:
        logger.error(f"Failed to start model pull: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# In-memory pull status tracking with TTL cleanup
_pull_status: Dict[str, Dict[str, Any]] = {}
_PULL_STATUS_TTL = 3600  # 1 hour


def _cleanup_pull_status():
    """Remove stale pull status entries older than TTL."""
    now = time.time()
    stale = [k for k, v in _pull_status.items() if now - v.get("_ts", 0) > _PULL_STATUS_TTL]
    for k in stale:
        del _pull_status[k]


@router.get("/models/pull/{model_name}/status")
async def get_pull_status(
    model_name: str,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Get the status of a model pull operation."""

    # URL decode the model name (colons get encoded)
    model_name = urllib.parse.unquote(model_name)

    status = _pull_status.get(model_name)
    if not status:
        return {"success": False, "error": "No pull in progress for this model"}

    # Parse progress format
    result = {
        "success": True,
        "model": model_name,
        "status": status.get("status", "unknown"),
    }

    # If pulling, calculate progress percentage
    if "total" in status and "completed" in status:
        total = status["total"]
        completed = status["completed"]
        if total > 0:
            result["progress_percent"] = round((completed / total) * 100, 1)
            result["completed_gb"] = round(completed / (1024**3), 2)
            result["total_gb"] = round(total / (1024**3), 2)

    # Check if done
    if status.get("status") == "success":
        result["done"] = True
        # Clean up status after reporting success
        del _pull_status[model_name]

    return result
