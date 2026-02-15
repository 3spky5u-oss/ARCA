import asyncio
import logging
from typing import Dict, Any

from fastapi import Depends, HTTPException, Query

from config import runtime_config
from services.admin_auth import verify_admin
from . import router

logger = logging.getLogger(__name__)


@router.post("/core-knowledge/toggle")
async def toggle_core_knowledge(
    enabled: bool = Query(..., description="Enable or disable core knowledge"),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Enable or disable the built-in ARCA core knowledge corpus.

    When disabled, the arca_core topic remains in the collection but won't be searched.
    """
    runtime_config.core_knowledge_enabled = enabled
    runtime_config.save_overrides()

    return {
        "success": True,
        "core_knowledge_enabled": enabled,
    }


@router.get("/core-knowledge/status")
async def core_knowledge_status(
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Get status of the core knowledge corpus (stored in main cohesionn collection)."""
    vectors_count = 0
    try:
        from tools.cohesionn.vectorstore import get_knowledge_base
        kb = get_knowledge_base()
        store = kb.get_store("arca_core")
        vectors_count = store.count
    except Exception:
        pass

    return {
        "enabled": runtime_config.core_knowledge_enabled,
        "collection": "cohesionn",
        "topic": "arca_core",
        "vectors_count": vectors_count,
    }


@router.post("/core-knowledge/reingest")
async def reingest_core_knowledge(
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Force re-ingestion of core knowledge by clearing arca_core topic and re-running.
    """
    # Clear existing arca_core chunks from main collection
    try:
        from tools.cohesionn.vectorstore import get_knowledge_base
        kb = get_knowledge_base()
        store = kb.get_store("arca_core")
        store.clear()
        logger.info("Cleared arca_core topic from cohesionn collection for re-ingestion")
    except Exception as e:
        logger.warning(f"Failed to clear arca_core topic: {e}")

    # Trigger re-ingestion in background
    try:
        from main import _ingest_core_knowledge
        asyncio.create_task(_ingest_core_knowledge())
        return {
            "success": True,
            "message": "Core knowledge re-ingestion started in background",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start re-ingestion: {e}")
