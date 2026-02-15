import logging
import time
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException, UploadFile, File, Form, Query

from config import runtime_config
from services.admin_auth import verify_admin
from . import router, KNOWLEDGE_DIR, COHESIONN_DB_DIR, MANIFEST_PATH, SUPPORTED_EXTENSIONS, _glob_supported_files
from .llm_management import unload_llm_models, preload_chat_model
from .models import CreateTopicRequest
from .reprocess import _reprocess_jobs

logger = logging.getLogger(__name__)


@router.post("/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    topic: str = Form(...),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(150),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Upload and ingest a document into the knowledge base.

    Args:
        file: Document file to ingest (PDF, DOCX, TXT, MD)
        topic: Target topic name
        chunk_size: Target chunk size (default 800)
        chunk_overlap: Overlap between chunks (default 150)

    Returns:
        Ingestion result with chunk count and timing
    """


    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type: " + file_ext + ". Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )

    # Ensure topic directory exists
    topic_dir = KNOWLEDGE_DIR / topic
    topic_dir.mkdir(parents=True, exist_ok=True)

    # Save to topic directory (not temp)
    target_path = topic_dir / file.filename
    content = await file.read()

    # Check for existing file
    if target_path.exists():
        raise HTTPException(
            status_code=400, detail=f"File already exists: {file.filename}. Delete it first to re-ingest."
        )

    target_path.write_bytes(content)

    try:
        from tools.cohesionn.autoingest import AutoIngestService
        from tools.cohesionn.manifest import IngestManifest

        start_time = time.time()

        service = AutoIngestService(
            knowledge_dir=KNOWLEDGE_DIR,
            db_dir=COHESIONN_DB_DIR,
            mode="knowledge_base",
        )

        result = service.ingest_single_file(
            file_path=target_path,
            topic=topic,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "success": result["success"],
            "file": file.filename,
            "topic": topic,
            "chunks_created": result.get("chunks_created", 0),
            "chunks_failed": result.get("chunks_failed", 0),
            "extractor_used": result.get("extractor_used"),
            "warnings": result.get("warnings", []),
            "error": result.get("error"),
            "processing_ms": elapsed_ms,
        }

    except Exception as e:
        # Clean up file on failure
        if target_path.exists():
            target_path.unlink()
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/topic")
async def create_topic(
    request: CreateTopicRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Create a new topic folder.

    Args:
        name: Topic name (will be used as folder name)

    Returns:
        Success status and path
    """


    # Validate topic name
    name = request.name.strip().lower()
    if not name:
        raise HTTPException(status_code=400, detail="Topic name required")

    # Only allow alphanumeric and underscores
    import re

    if not re.match(r"^[a-z0-9_]+$", name):
        raise HTTPException(
            status_code=400, detail="Topic name must contain only lowercase letters, numbers, and underscores"
        )

    topic_dir = KNOWLEDGE_DIR / name

    if topic_dir.exists():
        return {
            "success": True,
            "message": "Topic already exists",
            "path": str(topic_dir),
            "created": False,
        }

    topic_dir.mkdir(parents=True, exist_ok=True)

    return {
        "success": True,
        "message": f"Topic '{name}' created",
        "path": str(topic_dir),
        "created": True,
    }


@router.post("/auto-ingest")
async def run_auto_ingest(
    _: bool = Depends(verify_admin),
    skip_vision: bool = Query(False, description="Skip Phase 2 vision extraction (text-only, faster)"),
) -> Dict[str, Any]:
    """
    Run five-phase auto-ingestion as a background job.

    Returns immediately with a job_id. Poll /reprocess-status/{job_id} for progress.

    Phases:
    1. Text extraction (CPU — Docling/PyMuPDF4LLM)
    2. Vision extraction (GPU — llama.cpp vision server)
    3. Chunk + Embed (GPU — ONNX embedder)
    4. RAPTOR tree building
    5. GraphRAG entity extraction
    """
    import uuid

    # Check if another job is already running
    for existing_id, job in _reprocess_jobs.items():
        if job.get("status") in ("running", "starting"):
            raise HTTPException(
                status_code=409,
                detail=f"Another job is already running: {existing_id}",
            )

    job_id = str(uuid.uuid4())[:8]

    # Create job entry immediately — scanning + ingest happen in background thread.
    # scan_new_files() hashes every file (slow for large PDFs), so we can't block the request.
    _reprocess_jobs[job_id] = {
        "status": "starting",
        "mode": "auto-ingest",
        "topics": [],
        "total_files": 0,
        "processed_files": 0,
        "successful": 0,
        "failed": 0,
        "current_file": None,
        "current_topic": None,
        "current_page": 0,
        "total_pages": 0,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "results": [],
        "extraction_stats": {"text_pages": 0, "vision_pages": 0},
        "phase": "scanning",
        "phase_progress": 0,
        "phases": {
            "purge": {"status": "skipped"},
            "ingest": {"status": "pending", "files": 0, "chunks": 0},
            "raptor": {"status": "pending", "nodes": 0, "levels": {}},
            "graph": {"status": "pending", "entities": 0, "relationships": 0},
        },
        "options": {
            "purge_qdrant": False,
            "clear_bm25": False,
            "clear_neo4j": False,
            "skip_ingest": False,
            "build_raptor": True,
            "build_graph": True,
        },
    }

    def run_ingest():
        _run_auto_ingest_job(job_id, skip_vision=skip_vision)

    thread = Thread(target=run_ingest, daemon=True)
    thread.start()

    return {
        "success": True,
        "job_id": job_id,
        "status": "starting",
        "message": f"Auto-ingest started. Poll /api/admin/knowledge/reprocess-status/{job_id} for progress.",
    }


def _run_auto_ingest_job(job_id: str, skip_vision: bool = False):
    """Background job for five-phase auto-ingest."""
    import asyncio

    job = _reprocess_jobs[job_id]
    job["status"] = "running"

    runtime_config.set_ingest_active(True)

    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        unloaded = loop.run_until_complete(unload_llm_models())
        if unloaded:
            logger.info(f"Unloaded {len(unloaded)} models for auto-ingest")
    except Exception as e:
        logger.warning(f"Failed to unload models: {e}")

    try:
        from tools.cohesionn.autoingest import AutoIngestService

        # Scan files (hashes every file — this is why we do it in background)
        job["phase"] = "scanning"
        job["current_file"] = "Scanning for new files..."
        service = AutoIngestService(KNOWLEDGE_DIR, COHESIONN_DB_DIR, mode="knowledge_base",
                                    skip_vision=skip_vision)
        new_files = service.scan_new_files()
        total_files = sum(len(files) for files in new_files.values())
        topics = list(new_files.keys())

        job["total_files"] = total_files
        job["topics"] = topics

        if total_files == 0:
            job["status"] = "completed"
            job["phase"] = "complete"
            job["phase_progress"] = 100
            job["completed_at"] = datetime.now().isoformat()
            job["current_file"] = None
            logger.info(f"Auto-ingest job {job_id}: no new files found")
            return

        logger.info(f"Auto-ingest job {job_id}: found {total_files} files across {len(topics)} topics")

        job["phase"] = "ingest"
        job["phases"]["ingest"]["status"] = "running"
        job["current_file"] = "Running five-phase pipeline..."

        result = service.run_phased()

        # Update job with results
        job["successful"] = result.get("successful", 0)
        job["failed"] = result.get("failed", 0)
        job["processed_files"] = job["successful"] + job["failed"]

        phases = result.get("phases", {})
        job["phases"]["ingest"]["status"] = "completed"
        job["phases"]["ingest"]["files"] = job["successful"]
        job["phases"]["ingest"]["chunks"] = phases.get("embed", 0)
        job["phases"]["raptor"]["nodes"] = phases.get("raptor", 0)
        job["phases"]["raptor"]["status"] = "completed"
        job["phases"]["graph"]["entities"] = phases.get("graph_entities", 0)
        job["phases"]["graph"]["relationships"] = phases.get("graph_relationships", 0)
        job["phases"]["graph"]["status"] = "completed"

        job["extraction_stats"]["text_pages"] = phases.get("text", 0)
        job["extraction_stats"]["vision_pages"] = phases.get("vision", 0)

        # Flatten results for job tracking
        for topic_name, topic_results in result.get("results", {}).items():
            if isinstance(topic_results, list):
                for r in topic_results:
                    job["results"].append({
                        "file": r.get("file", "unknown"),
                        "topic": topic_name,
                        "success": r.get("success", False),
                        "chunks": r.get("chunks", 0),
                        "error": r.get("error"),
                    })

        job["phase"] = "complete"
        job["phase_progress"] = 100
        job["status"] = "completed"
        job["completed_at"] = datetime.now().isoformat()
        job["current_file"] = None

        logger.info(
            f"Auto-ingest job {job_id} completed: "
            f"{job['successful']} files, {phases.get('embed', 0)} chunks, "
            f"{phases.get('raptor', 0)} RAPTOR nodes, "
            f"{phases.get('graph_entities', 0)} graph entities"
        )

    except Exception as e:
        logger.error(f"Auto-ingest job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()
        job["phase"] = "failed"

    finally:
        runtime_config.set_ingest_active(False)
        try:
            if loop is None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(preload_chat_model())
            loop.close()
        except Exception:
            pass


@router.post("/ingest-existing")
async def ingest_existing_file(
    path: str = Query(..., description="Full path to the existing file"),
    topic: str = Query(..., description="Topic to ingest into"),
    extractor: Optional[str] = Query(None, description="Specific extractor to use"),
    chunk_size: int = Query(800, description="Target chunk size"),
    chunk_overlap: int = Query(150, description="Chunk overlap"),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Ingest an existing file from the knowledge folder.

    Use this for files that failed auto-ingestion or haven't been ingested yet.

    Args:
        path: Full path to the file (must be in knowledge directory)
        topic: Topic to ingest into
        extractor: Specific extractor to use (or None for auto)
        chunk_size: Target chunk size (default 800)
        chunk_overlap: Chunk overlap (default 150)

    Returns:
        Ingestion result with chunk count and timing
    """


    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    # Security check: ensure file is within knowledge directory
    try:
        file_path.resolve().relative_to(KNOWLEDGE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="File must be within the knowledge directory")

    try:
        from tools.cohesionn.autoingest import AutoIngestService

        start_time = time.time()

        service = AutoIngestService(
            knowledge_dir=KNOWLEDGE_DIR,
            db_dir=COHESIONN_DB_DIR,
            mode="knowledge_base",
        )

        result = service.ingest_single_file(
            file_path=file_path,
            topic=topic,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "success": result["success"],
            "file": file_path.name,
            "path": str(file_path),
            "topic": topic,
            "extractor_used": result.get("extractor_used") or extractor,
            "chunks_created": result.get("chunks_created", 0),
            "chunks_failed": result.get("chunks_failed", 0),
            "warnings": result.get("warnings", []),
            "error": result.get("error"),
            "processing_ms": elapsed_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest existing file failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex-file")
async def reindex_file(
    path: str = Query(..., description="Full path to the file"),
    extractor: Optional[str] = Query(
        None, description="Specific extractor to use (pymupdf_text, pymupdf4llm, marker, vision_ocr)"
    ),
    chunk_size: int = Query(800, description="Target chunk size"),
    chunk_overlap: int = Query(150, description="Chunk overlap"),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Re-index a single file with optional extractor override.

    Args:
        path: Full path to the file
        extractor: Specific extractor to use (or None for auto)
        chunk_size: Target chunk size (default 800)
        chunk_overlap: Chunk overlap (default 150)

    Returns:
        Re-index result with chunk count and timing
    """


    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        from tools.cohesionn import get_knowledge_base
        from tools.cohesionn.manifest import IngestManifest

        start_time = time.time()

        # Find existing entry in manifest to get topic
        manifest = IngestManifest(MANIFEST_PATH)
        file_entry = None
        file_hash = None
        topic = None

        for h, entry in manifest.files.items():
            if entry.file_path == path:
                file_hash = h
                file_entry = entry
                topic = entry.topic
                break

        if not file_entry:
            raise HTTPException(
                status_code=404, detail=f"File not found in manifest: {path}. Use ingest endpoint for new files."
            )

        # Delete existing chunks
        kb = get_knowledge_base(COHESIONN_DB_DIR)
        store = kb.get_store(topic)
        store.delete_by_source(path)

        # Remove from manifest
        del manifest.files[file_hash]

        # Re-ingest with Docling pipeline
        from tools.cohesionn.autoingest import AutoIngestService

        service = AutoIngestService(
            knowledge_dir=KNOWLEDGE_DIR,
            db_dir=COHESIONN_DB_DIR,
            mode="knowledge_base",
        )

        result = service.ingest_single_file(
            file_path=file_path,
            topic=topic,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "success": result["success"],
            "file": file_path.name,
            "path": str(file_path),
            "topic": topic,
            "extractor_used": result.get("extractor_used") or extractor,
            "chunks_created": result.get("chunks_created", 0),
            "chunks_failed": result.get("chunks_failed", 0),
            "warnings": result.get("warnings", []),
            "error": result.get("error"),
            "processing_ms": elapsed_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Re-index file failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex/{topic}")
async def reindex_topic(
    topic: str,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Force re-index all files in a topic.

    Clears existing chunks and re-ingests all files.
    """


    topic_dir = KNOWLEDGE_DIR / topic
    if not topic_dir.exists():
        raise HTTPException(status_code=404, detail=f"Topic not found: {topic}")

    try:
        from tools.cohesionn import get_knowledge_base
        from tools.cohesionn.manifest import IngestManifest

        start_time = time.time()

        # Clear existing topic data
        kb = get_knowledge_base(COHESIONN_DB_DIR)
        kb.clear_topic(topic)

        # Remove topic entries from manifest
        manifest = IngestManifest(MANIFEST_PATH)
        hashes_to_remove = [h for h, entry in manifest.files.items() if entry.topic == topic]
        for h in hashes_to_remove:
            del manifest.files[h]
        manifest.save()

        # Re-ingest all files with Docling pipeline
        from tools.cohesionn.autoingest import AutoIngestService

        service = AutoIngestService(
            knowledge_dir=KNOWLEDGE_DIR,
            db_dir=COHESIONN_DB_DIR,
            mode="knowledge_base",
        )

        results = []
        successful = 0
        failed = 0

        for file_path in _glob_supported_files(topic_dir):
            result = service.ingest_single_file(file_path, topic)
            results.append(
                {
                    "file": file_path.name,
                    "success": result["success"],
                    "chunks": result.get("chunks_created", 0),
                    "error": result.get("error"),
                }
            )

            if result["success"]:
                successful += 1
            else:
                failed += 1
        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "topic": topic,
            "files_processed": successful + failed,
            "successful": successful,
            "failed": failed,
            "results": results,
            "processing_ms": elapsed_ms,
        }

    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
