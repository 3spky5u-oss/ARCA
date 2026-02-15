import asyncio
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Dict, Any, List, Optional

from fastapi import Depends, HTTPException
from pydantic import BaseModel

from config import runtime_config
from services.admin_auth import verify_admin
from . import router, KNOWLEDGE_DIR, COHESIONN_DB_DIR, MANIFEST_PATH, _glob_supported_files
from .llm_management import unload_llm_models, preload_chat_model

logger = logging.getLogger(__name__)

# In-memory job storage (for simplicity - production would use Redis/DB)
_reprocess_jobs: Dict[str, Dict[str, Any]] = {}


class ReprocessRequest(BaseModel):
    """Request to reprocess knowledge base."""

    mode: str = "knowledge_base"  # knowledge_base or session
    topics: Optional[List[str]] = None  # None = all topics
    # Purge options
    purge_qdrant: bool = True  # Delete old Qdrant vectors
    clear_bm25: bool = True  # Delete BM25 pkl files
    clear_neo4j: bool = True  # Clear GraphRAG entities
    # Build options
    skip_ingest: bool = False  # Skip re-extraction, jump to RAPTOR/GraphRAG
    skip_vision: bool = False  # Skip Phase 2 vision extraction (text-only, faster)
    build_raptor: bool = True  # Build RAPTOR trees after ingestion
    build_graph: bool = True  # Build GraphRAG entities after ingestion


@router.post("/reprocess")
async def start_reprocess(
    request: ReprocessRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Start a background job to reprocess all documents using hybrid extraction.

    This re-extracts all documents using the per-page vision/text routing
    pipeline for maximum quality. The process runs in the background.

    Args:
        mode: Extraction mode (knowledge_base for hybrid vision, session for fast)
        topics: Optional list of topics to reprocess (None = all)

    Returns:
        Job ID and initial status
    """
    import uuid



    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Check if another job is already running
    for existing_id, job in _reprocess_jobs.items():
        if job.get("status") == "running":
            raise HTTPException(status_code=409, detail=f"Another reprocess job is already running: {existing_id}")

    # Get topics to process
    topics_to_process = request.topics
    if not topics_to_process:
        # Discover all topics
        topics_to_process = []
        if KNOWLEDGE_DIR.exists():
            topics_to_process = [d.name for d in KNOWLEDGE_DIR.iterdir() if d.is_dir()]

    # Count total files
    total_files = 0
    for topic in topics_to_process:
        topic_dir = KNOWLEDGE_DIR / topic
        if topic_dir.exists():
            total_files += len(_glob_supported_files(topic_dir))

    # Initialize job state with phase tracking
    _reprocess_jobs[job_id] = {
        "status": "starting",
        "mode": request.mode,
        "topics": topics_to_process,
        "total_files": total_files,
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
        "extraction_stats": {
            "text_pages": 0,
            "vision_pages": 0,
        },
        # Phase tracking for full pipeline rebuild
        "phase": "purge",  # purge -> ingest -> raptor -> graph
        "phase_progress": 0,
        "phases": {
            "purge": {"status": "pending", "deleted_vectors": 0, "deleted_bm25": 0, "deleted_neo4j": 0},
            "ingest": {"status": "pending", "files": 0, "chunks": 0},
            "raptor": {"status": "pending", "nodes": 0, "levels": {}},
            "graph": {"status": "pending", "entities": 0, "relationships": 0},
        },
        # Pipeline options from request
        "options": {
            "purge_qdrant": request.purge_qdrant,
            "clear_bm25": request.clear_bm25,
            "clear_neo4j": request.clear_neo4j,
            "skip_ingest": request.skip_ingest,
            "skip_vision": request.skip_vision,
            "build_raptor": request.build_raptor,
            "build_graph": request.build_graph,
        },
    }

    # Start background thread
    def run_reprocess():
        _run_reprocess_job(job_id, request.mode, topics_to_process, request)

    thread = Thread(target=run_reprocess, daemon=True)
    thread.start()

    return {
        "success": True,
        "job_id": job_id,
        "status": "starting",
        "total_files": total_files,
        "topics": topics_to_process,
        "message": f"Reprocessing started. Use GET /api/admin/knowledge/reprocess-status/{job_id} to check progress.",
    }


def _run_reprocess_job(job_id: str, mode: str, topics: List[str], request: ReprocessRequest):
    """
    Background job to reprocess knowledge base with full pipeline rebuild.

    Phases:
    1. Purge: Clear old data (Qdrant vectors, BM25 indices, Neo4j graph)
    2. Ingest: Re-extract and embed all documents
    3. RAPTOR: Build hierarchical summaries
    4. GraphRAG: Extract entities and relationships
    """
    from tools.cohesionn import get_knowledge_base
    from tools.cohesionn.manifest import IngestManifest

    job = _reprocess_jobs[job_id]
    job["status"] = "running"
    options = job["options"]

    # Activate ingest lock — blocks chat/tool models from loading
    runtime_config.set_ingest_active(True)

    # Unload models to free VRAM for ingestion
    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        unloaded = loop.run_until_complete(unload_llm_models())
        if unloaded:
            logger.info(f"Unloaded {len(unloaded)} models before ingestion: {unloaded}")

        # Three-phase ingest handles ONNX unload and vision server lifecycle
        # internally — no need to manage them here.
    except Exception as e:
        logger.warning(f"Failed to unload models: {e}")

    try:
        # =============================================================
        # PHASE 1: PURGE
        # =============================================================
        job["phase"] = "purge"
        job["phases"]["purge"]["status"] = "running"
        logger.info(f"Phase 1: Purge (qdrant={options['purge_qdrant']}, bm25={options['clear_bm25']}, neo4j={options['clear_neo4j']})")

        # Purge Qdrant vectors
        if options["purge_qdrant"]:
            try:
                kb = get_knowledge_base(COHESIONN_DB_DIR)
                for topic in topics:
                    kb.clear_topic(topic)
                    job["phases"]["purge"]["deleted_vectors"] += 1
                logger.info(f"Cleared Qdrant vectors for {len(topics)} topics")
            except Exception as e:
                logger.error(f"Failed to clear Qdrant: {e}")

        # Clear BM25 index files
        if options["clear_bm25"]:
            bm25_dir = COHESIONN_DB_DIR / "bm25_index"
            if bm25_dir.exists():
                try:
                    pkl_files = list(bm25_dir.glob("*.pkl"))
                    for pkl_file in pkl_files:
                        pkl_file.unlink()
                        job["phases"]["purge"]["deleted_bm25"] += 1
                    logger.info(f"Deleted {len(pkl_files)} BM25 index files")
                except Exception as e:
                    logger.error(f"Failed to clear BM25 indices: {e}")

        # Clear manifest
        if MANIFEST_PATH.exists():
            try:
                MANIFEST_PATH.unlink()
                logger.info("Deleted manifest.json")
            except Exception as e:
                logger.error(f"Failed to delete manifest: {e}")

        # Clear Neo4j graph
        if options["clear_neo4j"]:
            try:
                from services.neo4j_client import get_neo4j_client
                neo4j = get_neo4j_client()
                # Delete all nodes and relationships
                result = neo4j.run_write_query("MATCH (n) DETACH DELETE n")
                deleted_count = result.get("nodes_deleted", 0)
                job["phases"]["purge"]["deleted_neo4j"] = deleted_count
                logger.info(f"Cleared Neo4j: {deleted_count} nodes deleted")
            except Exception as e:
                logger.warning(f"Failed to clear Neo4j (may not be available): {e}")

        job["phases"]["purge"]["status"] = "completed"
        job["phase_progress"] = 25

        # =============================================================
        # PHASE 2: INGEST (skippable)
        # =============================================================
        if options.get("skip_ingest"):
            job["phase"] = "ingest"
            job["phases"]["ingest"]["status"] = "skipped"
            job["phase_progress"] = 50
            logger.info("Phase 2: Ingest SKIPPED (skip_ingest=True)")
        else:
            job["phase"] = "ingest"
            job["phases"]["ingest"]["status"] = "running"
            logger.info(f"Phase 2: Ingest ({job['total_files']} files)")

            from tools.cohesionn.autoingest import AutoIngestService

            service = AutoIngestService(KNOWLEDGE_DIR, COHESIONN_DB_DIR, mode=mode,
                                        skip_vision=options.get("skip_vision", False))

            # Build file map for specified topics
            file_map: Dict[str, List[Path]] = {}
            for topic in topics:
                topic_dir = KNOWLEDGE_DIR / topic
                if topic_dir.exists():
                    doc_files = _glob_supported_files(topic_dir)
                    if doc_files:
                        file_map[topic] = doc_files

            temp_dir = Path(tempfile.mkdtemp(prefix="arca_reprocess_"))
            file_states = []

            try:
                # Three-phase ingest: text → vision → embed
                # Each phase uses GPU exclusively — no CUDA conflicts.

                # Phase 2a: Text extraction (CPU only)
                job["current_file"] = "Phase 2a: Text extraction (CPU)"
                file_states = service._phase1_extract_all(file_map, temp_dir)

                # Phase 2b: Vision extraction (GPU: vision server only)
                # Handles ONNX unload + vision start/stop internally
                job["current_file"] = "Phase 2b: Vision extraction (GPU)"
                vision_count = service._phase2_extract_vision(file_states)
                job["extraction_stats"]["vision_pages"] = vision_count
                job["extraction_stats"]["text_pages"] = sum(
                    len(s.text_pages) for s in file_states
                )

                # Phase 2c: Chunk + embed (GPU: ONNX embedder only)
                job["current_file"] = "Phase 2c: Embedding (GPU)"
                results_dict, successful_count, failed_count, total_chunks = (
                    service._phase3_chunk_and_embed(file_states)
                )

                service.manifest.save()

                job["successful"] = successful_count
                job["failed"] = failed_count
                job["processed_files"] = successful_count + failed_count

                # Flatten results for job tracking
                for topic_name, topic_results in results_dict.items():
                    for r in topic_results:
                        job["results"].append({
                            "file": r["file"],
                            "topic": topic_name,
                            "success": r["success"],
                            "chunks": r["chunks"],
                            "error": r.get("error"),
                        })

                job["phases"]["ingest"]["status"] = "completed"
                job["phases"]["ingest"]["files"] = successful_count
                job["phases"]["ingest"]["chunks"] = total_chunks

            finally:
                for state in file_states:
                    state.cleanup()
                try:
                    temp_dir.rmdir()
                except OSError:
                    pass

            job["phase_progress"] = 50
            job["current_file"] = None

        # =============================================================
        # PHASE 3: RAPTOR
        # =============================================================
        if options["build_raptor"]:
            job["phase"] = "raptor"
            job["phases"]["raptor"]["status"] = "running"
            logger.info(f"Phase 3: RAPTOR tree building for {len(topics)} topics")

            try:
                from tools.cohesionn.raptor import RaptorTreeBuilder

                builder = RaptorTreeBuilder(max_levels=3)
                total_raptor_nodes = 0

                for i, topic in enumerate(topics):
                    if job["status"] == "cancelling":
                        break

                    job["current_topic"] = topic
                    job["current_file"] = f"Building RAPTOR tree ({i+1}/{len(topics)})"

                    try:
                        result = builder.build_tree(topic=topic, rebuild=True)
                        total_raptor_nodes += result.total_nodes

                        # Update level stats
                        for level, count in result.nodes_per_level.items():
                            level_key = str(level)
                            if level_key not in job["phases"]["raptor"]["levels"]:
                                job["phases"]["raptor"]["levels"][level_key] = 0
                            job["phases"]["raptor"]["levels"][level_key] += count

                        logger.info(f"RAPTOR for {topic}: {result.total_nodes} nodes, {result.levels_built} levels")
                    except Exception as e:
                        logger.error(f"RAPTOR build failed for {topic}: {e}")

                job["phases"]["raptor"]["nodes"] = total_raptor_nodes
                job["phases"]["raptor"]["status"] = "completed"
                job["phase_progress"] = 75

            except ImportError as e:
                logger.warning(f"RAPTOR module not available: {e}")
                job["phases"]["raptor"]["status"] = "skipped"
        else:
            job["phases"]["raptor"]["status"] = "skipped"
            job["phase_progress"] = 75

        # =============================================================
        # PHASE 4: GRAPHRAG
        # =============================================================
        if options["build_graph"]:
            job["phase"] = "graph"
            job["phases"]["graph"]["status"] = "running"
            logger.info(f"Phase 4: GraphRAG building for {len(topics)} topics")

            try:
                from tools.cohesionn.graph_builder import GraphBuilder

                builder = GraphBuilder()
                total_entities = 0
                total_relationships = 0

                for i, topic in enumerate(topics):
                    if job["status"] == "cancelling":
                        break

                    job["current_topic"] = topic
                    job["current_file"] = f"Building knowledge graph ({i+1}/{len(topics)})"

                    try:
                        result = builder.build_graph(topic=topic, incremental=False)
                        total_entities += result.entities_created
                        total_relationships += result.relationships_created

                        logger.info(
                            f"GraphRAG for {topic}: {result.entities_created} entities, "
                            f"{result.relationships_created} relationships"
                        )
                    except Exception as e:
                        logger.error(f"GraphRAG build failed for {topic}: {e}")

                job["phases"]["graph"]["entities"] = total_entities
                job["phases"]["graph"]["relationships"] = total_relationships
                job["phases"]["graph"]["status"] = "completed"
                job["phase_progress"] = 100

            except ImportError as e:
                logger.warning(f"GraphRAG module not available: {e}")
                job["phases"]["graph"]["status"] = "skipped"
        else:
            job["phases"]["graph"]["status"] = "skipped"
            job["phase_progress"] = 100

        # =============================================================
        # COMPLETE
        # =============================================================
        job["phase"] = "complete"
        job["status"] = "completed"
        job["completed_at"] = datetime.now().isoformat()
        job["current_file"] = None
        job["current_topic"] = None

        logger.info(
            f"Reprocess job {job_id} completed: "
            f"{job['successful']} files, {job['phases']['ingest'].get('chunks', 0)} chunks, "
            f"RAPTOR: {job['phases']['raptor'].get('nodes', 0)} nodes, "
            f"GraphRAG: {job['phases']['graph'].get('entities', 0)} entities"
        )

        # Release ingest lock and reload the default model
        runtime_config.set_ingest_active(False)
        if loop:
            try:
                loop.run_until_complete(preload_chat_model())
            except Exception as e:
                logger.warning(f"Failed to preload model after ingestion: {e}")
            finally:
                loop.close()

    except Exception as e:
        logger.error(f"Reprocess job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()

        # Mark current phase as failed
        current_phase = job.get("phase", "ingest")
        if current_phase in job["phases"]:
            job["phases"][current_phase]["status"] = "failed"

        # Release ingest lock and reload model even on failure
        runtime_config.set_ingest_active(False)
        try:
            if loop is None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(preload_chat_model())
            loop.close()
        except Exception:
            pass


@router.get("/reprocess-status/{job_id}")
async def get_reprocess_status(
    job_id: str,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Get status of a reprocess job.

    Args:
        job_id: The job ID returned from POST /reprocess

    Returns:
        Current job status including progress
    """


    if job_id not in _reprocess_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = _reprocess_jobs[job_id]

    # Calculate progress percentage
    progress = 0
    if job["total_files"] > 0:
        progress = int((job["processed_files"] / job["total_files"]) * 100)

    # Calculate elapsed time
    elapsed_seconds = 0
    if job["started_at"]:
        started = datetime.fromisoformat(job["started_at"])
        if job["completed_at"]:
            ended = datetime.fromisoformat(job["completed_at"])
            elapsed_seconds = (ended - started).total_seconds()
        else:
            elapsed_seconds = (datetime.now() - started).total_seconds()

    # Estimate remaining time
    remaining_seconds = 0
    if job["processed_files"] > 0 and job["status"] == "running":
        seconds_per_file = elapsed_seconds / job["processed_files"]
        remaining_files = job["total_files"] - job["processed_files"]
        remaining_seconds = int(seconds_per_file * remaining_files)

    return {
        "job_id": job_id,
        "status": job["status"],
        "mode": job["mode"],
        "progress": progress,
        "total_files": job["total_files"],
        "processed_files": job["processed_files"],
        "successful": job["successful"],
        "failed": job["failed"],
        "current_file": job["current_file"],
        "current_topic": job["current_topic"],
        "topics": job["topics"],
        "elapsed_seconds": int(elapsed_seconds),
        "remaining_seconds": remaining_seconds,
        "extraction_stats": job["extraction_stats"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "error": job["error"],
        # Only include last 10 results to avoid huge responses
        "recent_results": job["results"][-10:] if job["results"] else [],
        # Phase tracking for full pipeline rebuild
        "phase": job.get("phase", "ingest"),
        "phase_progress": job.get("phase_progress", 0),
        "phases": job.get("phases", {}),
        # Ingest lock status
        "ingest_lock_active": runtime_config.ingest_active,
    }


@router.post("/reprocess-cancel/{job_id}")
async def cancel_reprocess(
    job_id: str,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Cancel a running reprocess job.

    Note: This marks the job for cancellation. The job will stop
    after completing the current file.
    """


    if job_id not in _reprocess_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = _reprocess_jobs[job_id]

    if job["status"] != "running":
        raise HTTPException(status_code=400, detail=f"Job is not running (status: {job['status']})")

    # Mark for cancellation (the job loop should check this)
    job["status"] = "cancelling"

    return {
        "success": True,
        "job_id": job_id,
        "message": "Job marked for cancellation. Will stop after current file.",
    }


@router.get("/reprocess-jobs")
async def list_reprocess_jobs(
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    List all reprocess jobs (recent history).
    """


    jobs = []
    for job_id, job in _reprocess_jobs.items():
        jobs.append(
            {
                "job_id": job_id,
                "status": job["status"],
                "mode": job["mode"],
                "total_files": job["total_files"],
                "processed_files": job["processed_files"],
                "successful": job["successful"],
                "failed": job["failed"],
                "started_at": job["started_at"],
                "completed_at": job["completed_at"],
            }
        )

    return {
        "jobs": sorted(jobs, key=lambda x: x["started_at"] or "", reverse=True),
    }
