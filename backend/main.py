"""
ARCA - Agentic Retrieval-augmented Corpus Architecture
FastAPI Backend with LLM + Tools
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
import logging
import asyncio
import os
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware

import importlib
from routers import chat, upload, sessions, admin, admin_knowledge, admin_phii, admin_graph, admin_training, admin_benchmark, admin_lexicon, visualize, mcp_api
from domain_loader import get_domain_config
from middleware.rate_limit import RateLimitMiddleware
from logging_config import setup_logging
from utils.llm import get_llm_client, get_server_manager
from config import runtime_config

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class StartupHealth:
    """Tracks component health through startup phases."""
    phase: str = "initializing"
    llm: str = "pending"
    rag: str = "pending"
    redis: str = "pending"
    qdrant: str = "pending"
    postgres: str = "pending"
    background_tasks: dict = field(default_factory=dict)
    startup_complete: bool = False


_startup_health = StartupHealth()

# Instance ID - changes on every startup, used by frontend to detect restarts
INSTANCE_ID = str(uuid.uuid4())

# Directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
GUIDELINES_DIR = BASE_DIR / "guidelines"
SESSIONS_DIR = BASE_DIR / "data" / "sessions"
KNOWLEDGE_DIR = BASE_DIR / "data" / "technical_knowledge"
COHESIONN_DB_DIR = BASE_DIR / "data" / "cohesionn_db"

# Config from environment
CLEANUP_MAX_AGE_HOURS = int(os.environ.get("CLEANUP_MAX_AGE_HOURS", "24"))
CLEANUP_INTERVAL_HOURS = int(os.environ.get("CLEANUP_INTERVAL_HOURS", "6"))
AUTO_INGEST_ON_STARTUP = os.environ.get("AUTO_INGEST_ON_STARTUP", "false").lower() == "true"
CLEANUP_ON_STARTUP = os.environ.get("CLEANUP_ON_STARTUP", "false").lower() == "true"


def get_storage_stats():
    """Get storage statistics"""
    stats = {"uploads": {"count": 0, "bytes": 0}, "reports": {"count": 0, "bytes": 0}, "total_mb": 0}

    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            if f.is_file():
                stats["uploads"]["count"] += 1
                stats["uploads"]["bytes"] += f.stat().st_size

    if REPORTS_DIR.exists():
        for f in REPORTS_DIR.iterdir():
            if f.is_file():
                stats["reports"]["count"] += 1
                stats["reports"]["bytes"] += f.stat().st_size

    stats["total_mb"] = round((stats["uploads"]["bytes"] + stats["reports"]["bytes"]) / (1024 * 1024), 2)
    return stats


def cleanup_old_files(max_age_hours: int = 24):
    """Delete files older than max_age_hours"""
    from datetime import datetime, timedelta

    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    deleted = {"uploads": 0, "reports": 0, "bytes": 0}

    for dir_name, dir_path in [("uploads", UPLOAD_DIR), ("reports", REPORTS_DIR)]:
        if dir_path.exists():
            for f in dir_path.iterdir():
                if f.is_file():
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    if mtime < cutoff:
                        try:
                            size = f.stat().st_size
                            f.unlink()
                            deleted[dir_name] += 1
                            deleted["bytes"] += size
                            logger.info(f"Deleted old file: {f.name}")
                        except Exception as e:
                            logger.warning(f"Failed to delete {f.name}: {e}")

    return deleted


def run_startup_cleanup(max_age_hours: int = 24, security_wipe: bool = False):
    """
    Unified startup cleanup.

    Args:
        max_age_hours: Delete files older than this (age-based mode)
        security_wipe: If True, delete ALL sensitive data regardless of age
                       (sessions, uploads, reports, in-memory stores)
    """
    cleanup_stats = {"sessions": 0, "uploads": 0, "reports": 0, "memory": 0, "aged_out": 0}

    if security_wipe:
        logger.info("Running security cleanup on startup...")

        # Clear session files
        if SESSIONS_DIR.exists():
            for session_file in SESSIONS_DIR.glob("*.json"):
                try:
                    session_file.unlink()
                    cleanup_stats["sessions"] += 1
                except Exception as e:
                    logger.warning(f"Failed to delete session file {session_file}: {e}")

        # Clear upload files
        if UPLOAD_DIR.exists():
            for upload_file in UPLOAD_DIR.iterdir():
                if upload_file.is_file():
                    try:
                        upload_file.unlink()
                        cleanup_stats["uploads"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete upload file {upload_file}: {e}")

        # Clear report files
        if REPORTS_DIR.exists():
            for report_file in REPORTS_DIR.iterdir():
                if report_file.is_file():
                    try:
                        report_file.unlink()
                        cleanup_stats["reports"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete report file {report_file}: {e}")

        # Clear in-memory file storage (files_db)
        try:
            from routers.upload import files_db
            if files_db:
                cleanup_stats["memory"] = len(files_db)
                files_db.clear()
        except Exception as e:
            logger.warning(f"Failed to clear in-memory files: {e}")

        # Clear session knowledge stores
        try:
            from tools.cohesionn.session import cleanup_stale_sessions
            cleanup_stale_sessions(0)  # 0 hours = clear all
        except Exception as e:
            logger.warning(f"Failed to clear session knowledge stores: {e}")

    # Age-based cleanup (runs regardless of security_wipe)
    result = cleanup_old_files(max_age_hours)
    cleanup_stats["aged_out"] = result["uploads"] + result["reports"]

    total = sum(cleanup_stats.values())
    if total > 0:
        logger.info(f"Startup cleanup complete: {cleanup_stats}")
    else:
        logger.info("Startup cleanup: no data to clear")


async def periodic_cleanup(interval_hours: int = 6, max_age_hours: int = 24):
    """Periodically clean up old files and stale sessions"""
    while True:
        await asyncio.sleep(interval_hours * 3600)
        try:
            logger.info("Running periodic cleanup")
            cleanup_old_files(max_age_hours)

            # Clean up stale session knowledge stores
            try:
                from tools.cohesionn.session import cleanup_stale_sessions

                session_result = cleanup_stale_sessions(max_age_hours)
                if session_result["sessions_cleaned"] > 0:
                    logger.info(f"Session cleanup: {session_result['sessions_cleaned']} stale sessions removed")
            except Exception as e:
                logger.warning(f"Session cleanup failed: {e}")
        except Exception as e:
            logger.error(f"Periodic cleanup error: {e}", exc_info=True)


def check_models_available():
    """
    Check that required GGUF model files exist in the models directory.

    Users download GGUFs to ./models/ which is mounted into the container.
    """
    from services.llm_config import SLOTS, MODELS_DIR, get_model_path

    if not MODELS_DIR.exists():
        logger.warning(f"Models directory not found: {MODELS_DIR}")
        return

    available = [f.name for f in MODELS_DIR.glob("*.gguf")]
    logger.info(f"Available GGUF models: {available}")

    for name, slot in SLOTS.items():
        model_path = get_model_path(slot.gguf_filename)
        if model_path.exists():
            size_gb = model_path.stat().st_size / (1024**3)
            logger.info(f"Model {slot.gguf_filename} ({name}): {size_gb:.1f} GB")
        else:
            logger.warning(
                f"Model {slot.gguf_filename} ({name}) not found at {model_path}. "
                f"Download it to ./models/ before starting."
            )


async def run_auto_ingestion_task():
    """
    Run auto-ingestion in background thread (CPU-bound work).

    Scans the technical_knowledge directory for new/changed PDFs
    and ingests them into Qdrant automatically.
    """

    def _ingest():
        try:
            from tools.cohesionn.autoingest import AutoIngestService

            service = AutoIngestService(KNOWLEDGE_DIR, COHESIONN_DB_DIR)
            return service.run()
        except Exception as e:
            logger.error(f"Auto-ingestion failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _ingest)

    if result:
        if result.get("new_files", 0) > 0:
            logger.info(
                f"Auto-ingestion complete: {result.get('successful', 0)} files processed, "
                f"{result.get('failed', 0)} failed"
            )
        else:
            logger.info("Auto-ingestion: knowledge base up to date")


async def _ingest_core_knowledge():
    """
    Ingest ARCA's built-in knowledge corpus into the main cohesionn collection
    with topic="arca_core".

    Checks if arca_core chunks already exist in the main collection. If not,
    reads all .md files from the core_knowledge directory, chunks them, and
    stores via the TopicStore API. Runs as a background task during startup.
    """
    from config import runtime_config

    if not runtime_config.core_knowledge_enabled:
        logger.info("Core knowledge ingestion disabled (CORE_KNOWLEDGE_ENABLED=false)")
        return

    knowledge_dir = Path(runtime_config.core_knowledge_dir)

    if not knowledge_dir.exists() or not any(knowledge_dir.glob("*.md")):
        logger.info(f"Core knowledge directory empty or missing: {knowledge_dir}")
        return

    def _do_ingest():
        import hashlib
        import json
        from tools.cohesionn.vectorstore import get_knowledge_base
        from tools.cohesionn.chunker import SemanticChunker

        kb = get_knowledge_base()
        store = kb.get_store("arca_core")

        md_files = sorted(knowledge_dir.glob("*.md"))
        if not md_files:
            logger.info("Core knowledge: no markdown files found")
            return 0

        # Build a content fingerprint so doc updates auto-trigger re-ingestion.
        digest = hashlib.sha256()
        for md_file in md_files:
            digest.update(md_file.name.encode("utf-8"))
            digest.update(md_file.read_bytes())
        current_hash = digest.hexdigest()

        state_path = Path("/app/data/config/core_knowledge_state.json")
        previous_hash = ""
        if state_path.exists():
            try:
                previous_hash = json.loads(state_path.read_text(encoding="utf-8")).get("hash", "")
            except Exception:
                previous_hash = ""

        existing_count = store.count
        should_reingest = existing_count == 0 or current_hash != previous_hash
        if not should_reingest:
            logger.info(
                f"Core knowledge unchanged ({existing_count} chunks in cohesionn/arca_core) - skipping ingestion"
            )
            return 0

        if existing_count > 0:
            logger.info("Core knowledge changed - clearing arca_core for re-ingestion")
            store.clear()

        # Chunk all .md files
        chunker = SemanticChunker(chunk_size=800, chunk_overlap=150)
        all_chunks = []
        for md_file in md_files:
            try:
                text = md_file.read_text(encoding="utf-8")
                if not text.strip():
                    continue
                chunks = chunker.chunk_text(text, metadata={"source": md_file.name})
                all_chunks.extend(chunks)
                logger.info(f"Core knowledge: chunked {md_file.name} -> {len(chunks)} chunks")
            except Exception as e:
                logger.warning(f"Core knowledge: failed to read {md_file.name}: {e}")

        if not all_chunks:
            logger.warning("Core knowledge: no chunks produced from .md files")
            return 0

        # Add via TopicStore (handles embedding, Qdrant upsert, and BM25 sync)
        successful, failed = store.add_chunks(all_chunks)
        logger.info(
            f"Core knowledge ingestion complete: {successful} chunks from "
            f"{len(md_files)} files ({failed} failed)"
        )

        # Persist successful ingest fingerprint.
        if successful > 0 and failed == 0:
            try:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(
                    json.dumps({"hash": current_hash, "files": len(md_files)}, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.warning(f"Core knowledge: could not persist ingest state: {e}")

        return successful

    try:
        result = await asyncio.to_thread(_do_ingest)
        if result:
            logger.info(f"Core knowledge ready ({result} vectors in cohesionn/arca_core)")
    except Exception as e:
        logger.warning(f"Core knowledge ingestion failed: {e} — ARCA will work without it")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events"""
    # Startup
    UPLOAD_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    COHESIONN_DB_DIR.mkdir(parents=True, exist_ok=True)

    # Unified cleanup: security wipe + age-based cleanup
    # Set CLEANUP_ON_STARTUP=false for dev to preserve sessions across restarts
    if CLEANUP_ON_STARTUP:
        run_startup_cleanup(max_age_hours=CLEANUP_MAX_AGE_HOURS, security_wipe=True)
    else:
        logger.info("Security cleanup skipped (CLEANUP_ON_STARTUP=false)")
        run_startup_cleanup(max_age_hours=CLEANUP_MAX_AGE_HOURS, security_wipe=False)

    # Auto-download missing GGUF models when enabled.
    # This keeps first-run setup smoother for users with an empty ./models directory.
    try:
        from services.model_bootstrap import bootstrap_models_from_env

        bootstrap = await asyncio.to_thread(bootstrap_models_from_env)
        if bootstrap.get("enabled"):
            downloaded = bootstrap.get("downloaded", [])
            missing_repo = bootstrap.get("missing_repo", [])
            failed = bootstrap.get("failed", [])

            if downloaded:
                logger.info(f"Model bootstrap downloaded {len(downloaded)} model(s)")
            if missing_repo:
                logger.warning(
                    f"Model bootstrap skipped {len(missing_repo)} model(s): "
                    "repo mapping missing (set LLM_<SLOT>_MODEL_REPO in .env)"
                )
            if failed:
                logger.warning(
                    f"Model bootstrap failed for {len(failed)} model(s). "
                    "ARCA will continue startup and report remaining model issues."
                )
        else:
            logger.info("Model auto-download disabled (ARCA_AUTO_DOWNLOAD_MODELS=false)")
    except Exception as e:
        logger.warning(f"Model bootstrap could not run: {e}")

    # Initialize admin authentication (multi-user accounts)
    try:
        from services.admin_auth import get_auth_manager
        auth_manager = get_auth_manager()
        auth_manager.initialize()
        await auth_manager.initialize_async()
        if auth_manager.setup_required:
            logger.info("Admin auth: no users found (visit /admin to create your account)")
        else:
            logger.info("Admin auth: ready")
    except Exception as e:
        logger.warning(f"Admin auth initialization failed: {e}")

    # Validate startup configuration and wiring
    # This catches missing executors, orphan tools, bad config early
    try:
        from validation import validate_startup

        skip_llm_validation = runtime_config.mcp_mode
        if skip_llm_validation:
            logger.info("Startup validation: skipping LLM checks (MCP_MODE=true)")
        validation_result = validate_startup(skip_llm=skip_llm_validation)
        logger.info(f"Startup validation passed ({validation_result['warning_count']} warnings)")
    except RuntimeError as e:
        # Critical validation failure - abort startup
        logger.error(f"Startup validation failed: {e}")
        raise
    except Exception as e:
        # Non-critical validation error - warn but continue
        logger.warning(f"Startup validation could not complete: {e}")

    # Start periodic cleanup task
    cleanup_task = asyncio.create_task(
        periodic_cleanup(interval_hours=CLEANUP_INTERVAL_HOURS, max_age_hours=CLEANUP_MAX_AGE_HOURS)
    )

    # Start auto-ingestion as background task (non-blocking) - if enabled
    # Disabled by default to avoid processing large PDFs on every restart
    # Enable with AUTO_INGEST_ON_STARTUP=true
    ingestion_task = None
    if AUTO_INGEST_ON_STARTUP:
        logger.info("Starting knowledge base auto-ingestion (background)...")
        ingestion_task = asyncio.create_task(run_auto_ingestion_task())
    else:
        logger.info("Auto-ingestion disabled (set AUTO_INGEST_ON_STARTUP=true to enable)")

    # --- Phase 1: Infrastructure ---
    _startup_health.phase = "infrastructure"

    # Hardware detection and profiling
    try:
        from services.hardware import get_hardware_info, validate_model_files
        hw = get_hardware_info()
        logger.info(f"Hardware profile: {hw.profile}")
    except Exception as e:
        logger.warning(f"Hardware detection failed: {e}")

    # Validate model files (2D — config-filesystem mismatch check)
    if runtime_config.mcp_mode:
        logger.info("MCP mode active: skipping local GGUF model validation")
    else:
        logger.info("Checking model files...")
        try:
            validation = validate_model_files()
            if not validation["valid"]:
                logger.error(
                "Model validation failed — some always-on models are missing. "
                "Check the log above for details and fix .env or download models."
            )
        except Exception as e:
            logger.warning(f"Model availability check failed: {e}")

    # --- Phase 2: Services ---
    _startup_health.phase = "services"

    # Initialize Redis connection for session cache
    try:
        from services.redis_client import get_redis

        redis = await get_redis()
        health = await redis.health_check()
        if health.get("status") == "connected":
            logger.info(f"Redis connected (latency: {health.get('latency_ms', '?')}ms)")
            _startup_health.redis = "healthy"
        elif health.get("status") == "fallback":
            logger.warning("Redis unavailable, using in-memory fallback (sessions won't persist)")
            _startup_health.redis = "fallback"
        else:
            logger.warning(f"Redis status: {health.get('status')}")
            _startup_health.redis = "failed"
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}, sessions won't persist")
        _startup_health.redis = "failed"

    # Initialize retrieval profile
    try:
        from profile_loader import get_profile_manager
        pm = get_profile_manager()
        if runtime_config.retrieval_profile != "custom":
            pm.set_active(runtime_config.retrieval_profile)
            logger.info(f"Retrieval profile: {runtime_config.retrieval_profile}")
        else:
            logger.info("Retrieval profile: custom (migrated config, toggle overrides preserved)")
    except Exception as e:
        logger.warning(f"Profile initialization failed: {e}")

    # --- Phase 3: Models (sequential — VRAM budget awareness) ---
    _startup_health.phase = "models"

    # Timeout warmup to prevent startup from hanging if LLM loading is slow
    # Uses self-calibrating timeout when baseline exists, else env var / 300s cap
    try:
        from services.hardware import get_calibrated_timeout
        from services.llm_config import SLOTS
        chat_model = SLOTS.get("chat", None)
        model_name = chat_model.gguf_filename if chat_model else "unknown"
        startup_timeout = get_calibrated_timeout("chat", model_name)
    except Exception:
        startup_timeout = float(os.environ.get("LLM_STARTUP_TIMEOUT", "300"))

    async def _warm_llm():
        if runtime_config.mcp_mode:
            logger.info("MCP mode active: skipping local llama-server startup")
            _startup_health.llm = "disabled"
            return
        logger.info("Starting LLM servers...")
        _startup_health.llm = "starting"
        try:
            mgr = get_server_manager()
            results = await mgr.startup_sequence()
            all_healthy = True
            for slot_name, healthy in results.items():
                if healthy:
                    logger.info(f"  {slot_name} server started and healthy")
                else:
                    logger.warning(f"  {slot_name} server failed to start")
                    all_healthy = False
            _startup_health.llm = "healthy" if all_healthy else "failed"
        except Exception as e:
            logger.warning(f"LLM server startup failed: {e}")
            _startup_health.llm = "failed"

    async def _warm_rag():
        logger.info("Warming up RAG models...")
        _startup_health.rag = "starting"
        try:
            from tools.cohesionn import warm_models
            await asyncio.to_thread(warm_models)
            logger.info("  RAG models ready")
            _startup_health.rag = "healthy"
        except Exception as e:
            logger.warning(f"RAG warmup failed: {e}")
            _startup_health.rag = "failed"

    async def _seed_phii():
        try:
            from tools.phii import seed_if_empty
            if seed_if_empty():
                logger.info("  Phii workspace defaults seeded")
            else:
                logger.debug("Phii workspace defaults already present")
        except Exception as e:
            logger.warning(f"Phii seed failed: {e}")

    async def _precache_ondemand_models():
        if runtime_config.mcp_mode:
            logger.info("MCP mode active: skipping model pre-cache")
            return
        if os.environ.get("LLM_PRECACHE_MODELS", "true").lower() != "true":
            logger.info("Model pre-caching disabled (LLM_PRECACHE_MODELS=false)")
            return
        try:
            from utils.llm import precache_model_file
            from services.llm_config import SLOTS
            for slot in SLOTS.values():
                if not slot.always_running:
                    await asyncio.to_thread(precache_model_file, slot.gguf_filename)
        except Exception as e:
            logger.warning(f"Model pre-cache failed: {e}")

    try:
        async def _sequential_model_warmup():
            # LLM first (needs VRAM)
            await _warm_llm()
            # RAG second (needs VRAM for embedder)
            await _warm_rag()
            # Lightweight tasks can overlap
            await asyncio.gather(_seed_phii(), _precache_ondemand_models())

        await asyncio.wait_for(_sequential_model_warmup(), timeout=startup_timeout)
    except asyncio.TimeoutError:
        logger.warning(
            f"Model warmup timed out after {startup_timeout:.0f}s — continuing startup. "
            f"Try: increase LLM_STARTUP_TIMEOUT or use a smaller model."
        )
        if _startup_health.llm == "starting":
            _startup_health.llm = "failed"
        if _startup_health.rag == "starting":
            _startup_health.rag = "failed"

    # --- Phase 4: Background tasks ---
    _startup_health.phase = "background"

    # Ingest core knowledge corpus (non-blocking background task)
    # Runs after RAG warmup so embedder is ready
    core_knowledge_task = asyncio.create_task(_ingest_core_knowledge())
    _startup_health.background_tasks["core_knowledge"] = "running"

    # Mark startup complete if critical components are healthy
    llm_ready = _startup_health.llm in ("healthy", "disabled")
    critical_ok = llm_ready and _startup_health.rag == "healthy"
    _startup_health.startup_complete = critical_ok
    _startup_health.phase = "ready" if critical_ok else "degraded"

    domain = get_domain_config()
    if critical_ok:
        logger.info(f"{domain.app_name} is ready to help")
    else:
        logger.warning(
            f"{domain.app_name} started in degraded mode "
            f"(llm={_startup_health.llm}, rag={_startup_health.rag})"
        )
    yield

    # Shutdown
    cleanup_task.cancel()
    if ingestion_task:
        ingestion_task.cancel()
    if core_knowledge_task and not core_knowledge_task.done():
        core_knowledge_task.cancel()

    # Stop all llama-server instances
    try:
        mgr = get_server_manager()
        mgr.shutdown_sequence()
        logger.info("LLM servers stopped")
    except Exception as e:
        logger.debug(f"LLM server shutdown error: {e}")

    # Close Redis connection
    try:
        from services.redis_client import close_redis
        await close_redis()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.debug(f"Redis close error: {e}")

    # Close Neo4j connection
    try:
        from services.neo4j_client import close_neo4j_client
        close_neo4j_client()
        logger.info("Neo4j connection closed")
    except Exception as e:
        logger.debug(f"Neo4j close error: {e}")

    logger.info(f"{get_domain_config().app_name} signing off")


_domain = get_domain_config()
app = FastAPI(
    title=_domain.app_name,
    description=_domain.tagline or "Tools and knowledge at your service",
    version="1.0.0",
    lifespan=lifespan,
)


# Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Enable XSS filter in older browsers
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Referrer policy for privacy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


# Request body size limit middleware
MAX_BODY_SIZE_UPLOAD = 50 * 1024 * 1024  # 50MB for file uploads
MAX_BODY_SIZE_API = 1 * 1024 * 1024  # 1MB for regular API calls


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with bodies exceeding size limits."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            size = int(content_length)
            is_upload = request.url.path.startswith("/api/upload")
            limit = MAX_BODY_SIZE_UPLOAD if is_upload else MAX_BODY_SIZE_API
            if size > limit:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body too large ({size} bytes, limit {limit} bytes)"},
                )
        return await call_next(request)


app.add_middleware(SecurityHeadersMiddleware)

# Request body size limit
app.add_middleware(RequestSizeLimitMiddleware)

# Rate limiting middleware (Redis-backed)
app.add_middleware(RateLimitMiddleware)

# CORS - restrict to localhost and private network IPs on port 3000
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|100\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|[a-zA-Z][a-zA-Z0-9\-]*):3000$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers
# Chat router is mounted WITHOUT /api prefix so WebSocket is at /ws/chat
app.include_router(chat.router, tags=["chat"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(sessions.router, prefix="/api", tags=["sessions"])
# Admin router (already has /api/admin prefix)
app.include_router(admin.router, tags=["admin"])
# Admin knowledge router (already has /api/admin/knowledge prefix)
app.include_router(admin_knowledge.router, tags=["admin-knowledge"])
# Admin phii router (already has /api/admin/phii prefix)
app.include_router(admin_phii.router, tags=["admin-phii"])
# Admin graph router (already has /api/admin/graph prefix)
app.include_router(admin_graph.router, tags=["admin-graph"])
# Admin training router (already has /api/admin/training prefix)
app.include_router(admin_training.router, tags=["admin-training"])
# Admin benchmark router (already has /api/admin/benchmark prefix)
app.include_router(admin_benchmark.router, tags=["admin-benchmark"])
# Admin lexicon router (already has /api/admin/lexicon prefix)
app.include_router(admin_lexicon.router, tags=["admin-lexicon"])

# Domain-specific routes (loaded dynamically from domain pack)
if _domain.has_route("admin_exceedee"):
    try:
        _admin_exceedee = importlib.import_module(f"domains.{_domain.name}.routes.admin_exceedee")
        app.include_router(_admin_exceedee.router, tags=["admin-exceedee"])
    except ImportError as e:
        logger.warning(f"Route admin_exceedee unavailable: {e}")

if _domain.has_route("logg"):
    try:
        _logg = importlib.import_module(f"domains.{_domain.name}.routes.logg")
        app.include_router(_logg.router, prefix="/api", tags=["logg"])
    except ImportError as e:
        logger.warning(f"Route logg unavailable: {e}")
# Visualization router (3D viz data serving)
app.include_router(visualize.router, tags=["visualize"])
# MCP API router (already has /api/mcp prefix)
app.include_router(mcp_api.router, tags=["mcp"])

# Ensure reports directory exists
REPORTS_DIR.mkdir(exist_ok=True)


@app.get("/api/download/{filename:path}")
async def download_file(filename: str, download: bool = False):
    """Serve reports. Images open in browser by default, use ?download=true to force download."""
    file_path = (REPORTS_DIR / filename).resolve()
    # Security: Prevent path traversal attacks
    if not file_path.is_relative_to(REPORTS_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type and disposition
    suffix = file_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".pdf": "application/pdf",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    # Images and PDFs open inline by default, others download
    inline_types = {".png", ".jpg", ".jpeg", ".pdf"}
    disposition = "attachment" if download or suffix not in inline_types else "inline"

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type=media_type,
        headers={"Content-Disposition": f'{disposition}; filename="{file_path.name}"'},
    )


# Groundidd template endpoint (domain-specific)
if _domain.has_tool("groundidd"):
    @app.get("/api/groundidd/template")
    async def get_groundidd_template():
        """Generate and serve a blank Groundidd Lab Template for download."""
        try:
            from groundidd import generate_template

            # Generate template to reports directory (persistent)
            template_path = REPORTS_DIR / "Groundidd_Lab_Template.xlsx"

            # Generate fresh template if not already cached
            if not template_path.exists():
                generate_template(REPORTS_DIR)

            return FileResponse(
                path=template_path,
                filename="Groundidd_Lab_Template.xlsx",
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": 'attachment; filename="Groundidd_Lab_Template.xlsx"'},
            )
        except ImportError:
            raise HTTPException(status_code=500, detail="Groundidd tool not available")
        except Exception as e:
            logger.error(f"Failed to generate Groundidd template: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check - pings critical dependencies."""
    checks = {}

    # Check LLM servers
    if runtime_config.mcp_mode:
        checks["llm"] = "disabled"
    else:
        try:
            mgr = get_server_manager()
            chat_health = await mgr.health_check("chat")
            checks["llm"] = "ok" if chat_health.get("status") == "healthy" else "down"
        except Exception:
            checks["llm"] = "down"

    # Check Redis
    try:
        from services.redis_client import get_redis
        redis = await get_redis()
        redis_health = await redis.health_check()
        checks["redis"] = "ok" if redis_health.get("status") in ("connected", "fallback") else "down"
    except Exception:
        checks["redis"] = "down"

    # Check Qdrant
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
            resp = await client.get(f"{qdrant_url}/collections")
            checks["qdrant"] = "ok" if resp.status_code == 200 else "down"
    except Exception:
        checks["qdrant"] = "down"

    # Check PostgreSQL
    try:
        from services.database import get_database
        db = await get_database()
        db_health = await db.health_check()
        checks["postgres"] = "ok" if db_health.get("status") in ("connected", "fallback") else "down"
    except Exception:
        checks["postgres"] = "down"

    all_ok = all(v in ("ok", "disabled") for v in checks.values())
    return {
        "status": "healthy" if all_ok else "degraded",
        "service": get_domain_config().name,
        "checks": checks,
        "mcp_mode": runtime_config.mcp_mode,
        "startup_phase": _startup_health.phase,
        "startup_complete": _startup_health.startup_complete,
        "components": {
            "llm": _startup_health.llm,
            "rag": _startup_health.rag,
            "redis": _startup_health.redis,
            "qdrant": checks.get("qdrant", "unknown"),
            "postgres": checks.get("postgres", "unknown"),
        },
    }


@app.get("/api/instance")
async def get_instance():
    """Return instance ID - changes on each startup."""
    return {"instance_id": INSTANCE_ID}


@app.get("/api/domain")
async def get_domain():
    """Return domain configuration for frontend theming."""
    from domain_loader import get_lexicon

    domain = get_domain_config()
    lexicon = get_lexicon()
    identity = lexicon.get("identity", {})

    return {
        "name": domain.name,
        "display_name": domain.display_name,
        "app_name": domain.app_name,
        "tagline": domain.tagline,
        "primary_color": domain.primary_color,
        "tools": domain.tools,
        "routes": domain.routes,
        "admin_visible": domain.admin_visible,
        "thinking_messages": lexicon.get("thinking_messages", [
            "Pondering", "Cogitating", "Ruminating", "Mulling it over",
            "Working on it", "Crunching the numbers",
        ]),
        "welcome_message": identity.get("welcome_message", "Hello! How can I help?"),
    }


@app.get("/api/domain/logo")
async def get_domain_logo():
    """Serve domain-specific logo, or default ARCA logo fallback."""
    domain = get_domain_config()
    # Check for domain-specific logo
    for ext in ("png", "svg", "ico"):
        logo_path = domain.domain_dir / f"logo.{ext}"
        if logo_path.exists():
            media = {"png": "image/png", "svg": "image/svg+xml", "ico": "image/x-icon"}
            return FileResponse(logo_path, media_type=media[ext])
    # Fallback to default logo in frontend/public
    default_path = Path(__file__).resolve().parent.parent / "frontend" / "public" / "logo.png"
    if default_path.exists():
        return FileResponse(default_path, media_type="image/png")
    from fastapi.responses import JSONResponse
    return JSONResponse({"error": "No logo available"}, status_code=404)


@app.get("/api/status")
async def status():
    """System status"""
    # Check LLM servers
    llm_ok = False
    model_name = runtime_config.model_chat
    llm_status = {}
    try:
        mgr = get_server_manager()
        running = mgr.list_running()
        llm_ok = any(s.get("alive") for s in running.values())
        llm_status = {
            "status": "connected" if llm_ok else "unavailable",
            "models": [s.get("model", "") for s in running.values() if s.get("alive")],
        }
    except Exception as e:
        model_name = "Offline"
        llm_status = {"status": "unavailable", "error": str(e)}

    # Check guidelines (domain-specific)
    guidelines_path = GUIDELINES_DIR / "table1_guidelines.pkl"
    active_domain = get_domain_config()
    guidelines_ok = guidelines_path.exists() if active_domain.has_tool("exceedee") else True

    # Storage stats
    storage = get_storage_stats()

    components = {
        "llm": llm_status.get("status", "unavailable"),
    }
    if active_domain.has_tool("exceedee"):
        components["guidelines"] = "loaded" if guidelines_ok else "missing"

    return {
        "status": "healthy" if (llm_ok and guidelines_ok) else "degraded",
        "model": model_name if llm_ok else "Offline",
        "components": components,
        "llm": llm_status,
        "storage": {
            "uploads": storage["uploads"]["count"],
            "reports": storage["reports"]["count"],
            "total_mb": storage["total_mb"],
        },
    }


@app.post("/api/cleanup")
async def manual_cleanup(max_age_hours: int = 24):
    """Manually trigger cleanup"""
    result = cleanup_old_files(max_age_hours=max_age_hours)
    return {
        "success": True,
        "deleted": {
            "uploads": result["uploads"],
            "reports": result["reports"],
        },
        "mb_freed": round(result["bytes"] / (1024 * 1024), 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, ws_max_size=1048576)  # 1MB WS frame limit
