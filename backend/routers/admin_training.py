"""
Admin Training Router - Fine-Tuning Pipeline Management

Provides endpoints for managing the training pipeline:
- Pipeline status overview
- PDF parsing, QA generation, filtering, formatting jobs
- Dataset and golden set stats
- Shadow evaluation
- Candidate buffer management
- GGUF deployment to llama-server

Protected by admin JWT authentication.
"""

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from services.admin_auth import verify_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/training", tags=["admin-training"])

# Base directories
BASE_DIR = Path(__file__).parent.parent
TRAINING_DIR = Path(os.environ.get("TRAINING_DIR", str(BASE_DIR / "training")))
TRAINING_DATA_DIR = TRAINING_DIR / "data"

# In-memory job storage (same pattern as admin_knowledge.py)
_training_jobs: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# REQUEST MODELS
# =============================================================================


class ApproveRequest(BaseModel):
    """Approve pending QA pairs from buffer."""
    count: int = 10
    min_score: float = 7.5


class DeployRequest(BaseModel):
    """Deploy a fine-tuned GGUF model to llama-server."""
    gguf_path: Optional[str] = None  # Auto-detect if not specified
    model_name: str = "arca-finetuned"


class EvaluateRequest(BaseModel):
    """Run shadow evaluation."""
    base_model: Optional[str] = None
    finetuned_model: Optional[str] = None


# =============================================================================
# JOB HELPERS
# =============================================================================


def _create_job(job_type: str) -> str:
    """Create a new training job entry. Returns job_id."""
    # Check for existing running job of same type
    for jid, job in _training_jobs.items():
        if job["type"] == job_type and job["status"] in ("starting", "running"):
            raise HTTPException(
                status_code=409,
                detail=f"A {job_type} job is already running: {jid}",
            )

    job_id = str(uuid.uuid4())[:8]
    _training_jobs[job_id] = {
        "id": job_id,
        "type": job_type,
        "status": "starting",
        "phase": "",
        "progress": {"current": 0, "total": 0, "percent": 0},
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "result": {},
    }
    return job_id


def _run_in_thread(job_id: str, target_fn, *args, **kwargs) -> None:
    """Run a function in a daemon thread, updating job status."""

    def wrapper():
        job = _training_jobs[job_id]
        job["status"] = "running"
        try:
            result = target_fn(*args, **kwargs)
            job["status"] = "completed"
            job["result"] = result or {}
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
            job["status"] = "failed"
            job["error"] = str(e)
        finally:
            job["completed_at"] = datetime.now().isoformat()

    thread = Thread(target=wrapper, daemon=True)
    thread.start()


# =============================================================================
# STATUS ENDPOINT
# =============================================================================


@router.get("/status")
async def get_training_status(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """
    Get pipeline status overview.

    Returns source PDF count, parsed/generated/curated counts,
    active jobs, and dataset stats.
    """
    status: Dict[str, Any] = {
        "pipeline_available": TRAINING_DIR.exists(),
        "source_pdfs": 0,
        "parsed_files": 0,
        "generated_files": 0,
        "curated_files": 0,
        "dataset_ready": False,
        "active_jobs": [],
        "finetuned_model": "",
    }

    if not TRAINING_DIR.exists():
        return status

    # Count source PDFs
    source_dir = TRAINING_DATA_DIR
    if source_dir.exists():
        status["source_pdfs"] = len(list(source_dir.glob("*.pdf")))

    # Count parsed output
    parsed_dir = TRAINING_DIR / "data" / "parsed"
    if parsed_dir.exists():
        status["parsed_files"] = len(list(parsed_dir.glob("*.txt"))) + len(list(parsed_dir.glob("*.md")))

    # Count generated QA pairs
    generated_dir = TRAINING_DIR / "data" / "generated"
    if generated_dir.exists():
        status["generated_files"] = len(list(generated_dir.glob("*.jsonl")))

    # Count curated pairs
    curated_dir = TRAINING_DIR / "data" / "curated"
    if curated_dir.exists():
        status["curated_files"] = len(list(curated_dir.glob("*.jsonl")))

    # Check if dataset is ready
    dataset_dir = TRAINING_DIR / "data" / "training_ready"
    if dataset_dir.exists():
        train_files = list(dataset_dir.glob("train*.jsonl"))
        status["dataset_ready"] = len(train_files) > 0

    # Active jobs
    for jid, job in _training_jobs.items():
        if job["status"] in ("starting", "running"):
            status["active_jobs"].append({
                "id": jid,
                "type": job["type"],
                "status": job["status"],
                "phase": job["phase"],
                "progress": job["progress"],
            })

    # Current finetuned model
    try:
        from config import runtime_config
        status["finetuned_model"] = runtime_config.model_chat_finetuned
    except Exception:
        pass

    return status


# =============================================================================
# PIPELINE STAGE ENDPOINTS
# =============================================================================


@router.post("/parse")
async def start_parse(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Start PDF parsing job."""
    job_id = _create_job("parse")

    def do_parse():
        import sys
        sys.path.insert(0, str(TRAINING_DIR))
        from data_pipeline.pdf_parser import parse_all
        return parse_all()

    _run_in_thread(job_id, do_parse)

    return {
        "success": True,
        "job_id": job_id,
        "message": "PDF parsing started",
    }


@router.post("/generate")
async def start_generate(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Start QA generation job."""
    job_id = _create_job("generate")

    def do_generate():
        import sys
        sys.path.insert(0, str(TRAINING_DIR))
        from data_pipeline.qa_generator import generate_all
        return generate_all()

    _run_in_thread(job_id, do_generate)

    return {
        "success": True,
        "job_id": job_id,
        "message": "QA generation started",
    }


@router.post("/filter")
async def start_filter(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Start QA filtering/curation job."""
    job_id = _create_job("filter")

    def do_filter():
        import sys
        sys.path.insert(0, str(TRAINING_DIR))
        from data_pipeline.qa_filter import filter_and_curate
        return filter_and_curate()

    _run_in_thread(job_id, do_filter)

    return {
        "success": True,
        "job_id": job_id,
        "message": "QA filtering started",
    }


@router.post("/format")
async def start_format(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Format curated QA pairs into training dataset."""
    job_id = _create_job("format")

    def do_format():
        import sys
        sys.path.insert(0, str(TRAINING_DIR))
        from data_pipeline.format_dataset import format_all
        return format_all()

    _run_in_thread(job_id, do_format)

    return {
        "success": True,
        "job_id": job_id,
        "message": "Dataset formatting started",
    }


# =============================================================================
# JOB MANAGEMENT
# =============================================================================


@router.get("/jobs")
async def list_jobs(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """List all training pipeline jobs."""
    jobs = []
    for jid, job in _training_jobs.items():
        jobs.append({
            "id": job["id"],
            "type": job["type"],
            "status": job["status"],
            "phase": job["phase"],
            "progress": job["progress"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "error": job["error"],
        })

    # Most recent first
    jobs.sort(key=lambda x: x["started_at"] or "", reverse=True)
    return {"jobs": jobs}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, _: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Get detailed job status."""
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = _training_jobs[job_id]
    return {
        "id": job["id"],
        "type": job["type"],
        "status": job["status"],
        "phase": job["phase"],
        "progress": job["progress"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "error": job["error"],
        "result": job["result"],
    }


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, _: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Cancel a running job."""
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = _training_jobs[job_id]
    if job["status"] not in ("starting", "running"):
        raise HTTPException(status_code=400, detail=f"Job is not running (status: {job['status']})")

    job["status"] = "cancelled"
    job["completed_at"] = datetime.now().isoformat()

    return {"success": True, "message": f"Job {job_id} cancelled"}


# =============================================================================
# DATASET & EVALUATION
# =============================================================================


@router.get("/dataset")
async def get_dataset_stats(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Get dataset statistics."""
    stats: Dict[str, Any] = {
        "train_count": 0,
        "val_count": 0,
        "total_pairs": 0,
        "ready": False,
    }

    dataset_dir = TRAINING_DIR / "data" / "training_ready"
    if not dataset_dir.exists():
        return stats

    # Count lines in train/val files
    for jsonl_file in dataset_dir.glob("train*.jsonl"):
        with open(jsonl_file) as f:
            stats["train_count"] = sum(1 for _ in f)
        stats["ready"] = True

    for jsonl_file in dataset_dir.glob("val*.jsonl"):
        with open(jsonl_file) as f:
            stats["val_count"] = sum(1 for _ in f)

    stats["total_pairs"] = stats["train_count"] + stats["val_count"]

    return stats


@router.get("/golden-set")
async def get_golden_set_stats(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Get golden validation set statistics."""
    golden_path = TRAINING_DIR / "evaluation" / "golden_set.jsonl"

    if not golden_path.exists():
        return {"exists": False, "count": 0}

    with open(golden_path) as f:
        count = sum(1 for _ in f)

    return {"exists": True, "count": count, "path": str(golden_path)}


@router.get("/evaluation")
async def get_evaluation_results(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Get last evaluation results."""
    eval_dir = TRAINING_DIR / "evaluation" / "results"

    if not eval_dir.exists():
        return {"has_results": False}

    # Find most recent result file
    result_files = sorted(eval_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not result_files:
        return {"has_results": False}

    import json
    try:
        latest = json.loads(result_files[0].read_text(encoding="utf-8"))
        return {"has_results": True, "latest": latest, "file": result_files[0].name}
    except Exception as e:
        return {"has_results": False, "error": str(e)}


@router.post("/evaluate")
async def start_evaluation(
    request: EvaluateRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Run shadow evaluation (A/B test base vs finetuned)."""
    job_id = _create_job("evaluate")

    def do_evaluate():
        import sys
        sys.path.insert(0, str(TRAINING_DIR))
        from evaluation.shadow_eval import run_full_evaluation

        kwargs = {}
        if request.base_model:
            kwargs["base_model"] = request.base_model
        if request.finetuned_model:
            kwargs["finetuned_model"] = request.finetuned_model

        return run_full_evaluation(**kwargs)

    _run_in_thread(job_id, do_evaluate)

    return {
        "success": True,
        "job_id": job_id,
        "message": "Shadow evaluation started",
    }


# =============================================================================
# CANDIDATE BUFFER
# =============================================================================


@router.get("/buffer")
async def get_buffer_stats(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Get candidate buffer statistics."""
    try:
        import sys
        sys.path.insert(0, str(TRAINING_DIR))
        from guardrails.candidate_buffer import CandidateBuffer

        buffer = CandidateBuffer()
        return buffer.get_status()
    except ImportError:
        return {"error": "Candidate buffer module not available"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/buffer/approve")
async def approve_buffer(
    request: ApproveRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Approve pending QA pairs from the candidate buffer."""
    try:
        import sys
        sys.path.insert(0, str(TRAINING_DIR))
        from guardrails.candidate_buffer import CandidateBuffer

        buffer = CandidateBuffer()
        approved_count = buffer.approve_batch(
            count=request.count,
            min_score=request.min_score,
            reviewer="admin",
        )

        return {
            "success": True,
            "approved": approved_count,
            "status": buffer.get_status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DEPLOYMENT
# =============================================================================


@router.post("/deploy")
async def deploy_model(
    request: DeployRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Deploy a fine-tuned GGUF to llama-server.

    Copies the GGUF into the models directory and updates runtime config
    so the chat slot can swap to it on demand.
    """
    job_id = _create_job("deploy")

    def do_deploy():
        import shutil
        from services.llm_config import MODELS_DIR

        # Find GGUF file
        gguf_path = request.gguf_path
        if not gguf_path:
            output_dir = TRAINING_DIR / "output"
            if output_dir.exists():
                gguf_files = list(output_dir.glob("**/*.gguf"))
                if not gguf_files:
                    raise FileNotFoundError("No GGUF files found in training/output/")
                gguf_path = str(sorted(gguf_files, key=lambda f: f.stat().st_mtime, reverse=True)[0])
            else:
                raise FileNotFoundError("training/output/ directory not found")

        gguf_file = Path(gguf_path)
        if not gguf_file.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        # Copy GGUF to models directory so llama-server can load it
        dest = MODELS_DIR / gguf_file.name
        if not dest.exists() or dest.stat().st_mtime < gguf_file.stat().st_mtime:
            shutil.copy2(gguf_file, dest)
            logger.info(f"Copied {gguf_file.name} to {MODELS_DIR}")

        # Update runtime config to use the new model
        from config import runtime_config
        runtime_config.update(model_chat_finetuned=gguf_file.name)

        return {
            "model_name": request.model_name,
            "gguf_path": str(gguf_path),
            "deployed_to": str(dest),
        }

    _run_in_thread(job_id, do_deploy)

    return {
        "success": True,
        "job_id": job_id,
        "message": f"Deploying {request.model_name}",
    }


@router.post("/rollback")
async def rollback_model(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Rollback to base model (disable finetuned model)."""
    from config import runtime_config

    old_model = runtime_config.model_chat_finetuned
    runtime_config.update(model_chat_finetuned="")

    return {
        "success": True,
        "message": "Rolled back to base model",
        "previous_finetuned": old_model,
        "current_chat_model": runtime_config.model_chat,
    }
