"""
ARCA MCP API — Backend endpoints for the MCP server adapter.

Thin HTTP layer that forwards calls to existing executor functions.
Gated by a static API key (MCP_API_KEY env var). If unset, all
endpoints return 503 (disabled by default).

Auth: X-MCP-Key header checked against MCP_API_KEY.
"""

import os
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile, File
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp")

# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def verify_mcp_key(x_mcp_key: str = Header(..., alias="X-MCP-Key")) -> str:
    """Validate the MCP API key from the request header."""
    configured_key = os.environ.get("MCP_API_KEY", "")
    if not configured_key:
        raise HTTPException(status_code=503, detail="MCP API is disabled (MCP_API_KEY not set)")
    if x_mcp_key != configured_key:
        raise HTTPException(status_code=401, detail="Invalid MCP API key")
    return x_mcp_key


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    topics: Optional[List[str]] = None
    profile: Optional[str] = None  # per-query profile override ("fast", "deep")

class WebSearchRequest(BaseModel):
    query: str

class UnitConvertRequest(BaseModel):
    value: float
    from_unit: str
    to_unit: str

class ExecuteRequest(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Endpoints (sync — FastAPI runs them in a thread pool automatically)
# ---------------------------------------------------------------------------

@router.post("/search", dependencies=[Depends(verify_mcp_key)])
def mcp_search(req: SearchRequest):
    """Search the ingested knowledge base via the RAG pipeline."""
    from routers.chat_executors.rag import execute_search_knowledge
    return execute_search_knowledge(query=req.query, topics=req.topics, profile=req.profile)


@router.post("/upload", dependencies=[Depends(verify_mcp_key)])
async def mcp_upload(file: UploadFile = File(...)):
    """Upload a file for MCP tool processing (stored in RAM).

    Reuses the standard upload pipeline — file type detection, size
    validation, lab data parsing for Excel files, RAG indexing for
    text/PDF/Word.  After uploading, tools like ``analyze_files`` can
    access the file via the shared ``files_db``.
    """
    from routers.upload import _process_upload
    return await _process_upload([file])


@router.post("/web-search", dependencies=[Depends(verify_mcp_key)])
def mcp_web_search(req: WebSearchRequest):
    """Web search via SearXNG."""
    from routers.chat_executors.search import execute_web_search
    return execute_web_search(query=req.query)


@router.post("/unit-convert", dependencies=[Depends(verify_mcp_key)])
def mcp_unit_convert(req: UnitConvertRequest):
    """Convert between engineering units."""
    from routers.chat_executors.calculations import execute_unit_convert
    return execute_unit_convert(value=req.value, from_unit=req.from_unit, to_unit=req.to_unit)


@router.get("/topics", dependencies=[Depends(verify_mcp_key)])
def mcp_topics():
    """List available knowledge topics and their enabled status."""
    from config import runtime_config
    from pathlib import Path

    enabled = set(runtime_config.get_enabled_topics())

    # Discover all topics from Qdrant collections
    all_topics = set(enabled)
    try:
        from tools.cohesionn.vectorstore import get_knowledge_base
        kb = get_knowledge_base()
        for name in kb.discover_topics():
            all_topics.add(name)
    except Exception as e:
        logger.warning(f"MCP topics: Qdrant discovery failed: {e}")

    # Build topic list with enabled flag
    topics = []
    for name in sorted(all_topics):
        topics.append({"name": name, "enabled": name in enabled})

    return {"topics": topics, "enabled_count": len(enabled), "total_count": len(topics)}


@router.get("/stats", dependencies=[Depends(verify_mcp_key)])
def mcp_stats():
    """Knowledge base statistics — chunk counts, file counts, collections."""
    stats = {"collections": [], "total_chunks": 0, "total_files": 0}

    try:
        from tools.cohesionn.vectorstore import get_knowledge_base
        kb = get_knowledge_base()
        for name in kb.discover_topics():
            store = kb.get_store(name)
            count = store.count
            stats["collections"].append({"name": name, "chunks": count})
            stats["total_chunks"] += count
    except Exception as e:
        logger.warning(f"MCP stats: Qdrant query failed: {e}")
        stats["error"] = str(e)

    # Count ingested files from manifest
    try:
        from tools.cohesionn.autoingest import IngestManifest
        from pathlib import Path

        knowledge_dir = Path(os.environ.get("KNOWLEDGE_DIR", "/app/data/technical_knowledge"))
        db_dir = Path(os.environ.get("COHESIONN_DB_DIR", "/app/data/cohesionn_db"))
        manifest = IngestManifest(db_dir / "manifest.json")
        stats["total_files"] = len(manifest.entries)
    except Exception as e:
        logger.debug(f"MCP stats: manifest read failed: {e}")

    return stats


@router.get("/health", dependencies=[Depends(verify_mcp_key)])
async def mcp_health():
    """Component health checks — LLM, Redis, Qdrant, Postgres."""
    import httpx
    checks = {}

    # LLM
    try:
        from utils.llm import get_server_manager
        mgr = get_server_manager()
        chat_health = await mgr.health_check("chat")
        checks["llm"] = "ok" if chat_health.get("status") == "healthy" else "down"
    except Exception:
        checks["llm"] = "down"

    # Redis
    try:
        from services.redis_client import get_redis
        redis = await get_redis()
        redis_health = await redis.health_check()
        checks["redis"] = "ok" if redis_health.get("status") in ("connected", "fallback") else "down"
    except Exception:
        checks["redis"] = "down"

    # Qdrant
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
            resp = await client.get(f"{qdrant_url}/collections")
            checks["qdrant"] = "ok" if resp.status_code == 200 else "down"
    except Exception:
        checks["qdrant"] = "down"

    # Postgres
    try:
        from services.database import get_database
        db = await get_database()
        db_health = await db.health_check()
        checks["postgres"] = "ok" if db_health.get("status") in ("connected", "fallback") else "down"
    except Exception:
        checks["postgres"] = "down"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "healthy" if all_ok else "degraded", "checks": checks}


# ---------------------------------------------------------------------------
# Dynamic tool discovery + generic executor
# ---------------------------------------------------------------------------

@router.get("/tools", dependencies=[Depends(verify_mcp_key)])
def mcp_tools():
    """Return all registered tools formatted for MCP tool discovery.

    The MCP server polls this endpoint at startup (and periodically) to
    discover which tools are available, including domain-specific ones.
    """
    from tools.registry import ToolRegistry, register_all_tools
    from domain_loader import get_domain_config

    register_all_tools()

    tools_list = []
    for tool in ToolRegistry.get_all_tools().values():
        tools_list.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "required": tool.required_params,
        })

    return {
        "domain": get_domain_config().name,
        "tools": tools_list,
    }


@router.post("/execute", dependencies=[Depends(verify_mcp_key)])
def mcp_execute(req: ExecuteRequest):
    """Generic tool executor — MCP server calls this instead of per-tool endpoints.

    Dispatches to ToolRegistry.execute() which handles all registered tools
    (core + domain). Old per-tool endpoints remain for backward compatibility.
    """
    import json as _json
    from tools.registry import ToolRegistry, register_all_tools

    register_all_tools()

    args = req.args

    # Safety net: MCP clients with stale tool schemas may send a single "kwargs"
    # key containing a JSON string of the actual parameters. Detect and unpack.
    if list(args.keys()) == ["kwargs"]:
        raw = args["kwargs"]
        if isinstance(raw, str):
            try:
                parsed = _json.loads(raw)
                if isinstance(parsed, dict):
                    args = parsed
                    logger.info(f"MCP execute: unpacked legacy kwargs string for {req.tool}")
            except (ValueError, TypeError):
                pass
        elif isinstance(raw, dict):
            args = raw
            logger.info(f"MCP execute: unpacked legacy kwargs dict for {req.tool}")

    # Inject file context so file-dependent tools (analyze_files, etc.) work
    from routers.upload import files_db, get_file_data, update_file_analysis

    result = ToolRegistry.execute(
        req.tool, args,
        files_db=files_db,
        get_file_data_fn=get_file_data,
        update_file_analysis_fn=update_file_analysis,
    )

    if not result.success:
        raise HTTPException(status_code=400, detail=result.error or "Tool execution failed")

    return result.data
