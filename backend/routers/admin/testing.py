"""
Test extraction and QA test suite endpoints.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException, UploadFile, File

from . import router
from services.admin_auth import verify_admin

logger = logging.getLogger(__name__)


@router.post("/test-extraction")
async def test_extraction(
    file: UploadFile = File(...),
    extractor: Optional[str] = None,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Test document extraction on uploaded file.

    Args:
        file: PDF file to extract
        extractor: Specific extractor to use (or None for auto)

    Returns:
        Extraction result with QA metrics
    """

    import tempfile

    # Save uploaded file temporarily
    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        from tools.readd.scout import DocumentScout
        from tools.readd.extractors import get_extractor, EXTRACTORS
        from tools.readd.pipeline import ReaddPipeline

        # Scout the document
        scout = DocumentScout()
        scout_report = scout.scout(tmp_path)

        result = {
            "filename": file.filename,
            "scout": {
                "quality_tier": scout_report.quality_tier.value,
                "recommended_extractor": scout_report.recommended_extractor.value,
                "text_density": scout_report.text_density,
                "warnings": scout_report.warnings,
            },
        }

        # Run extraction
        if extractor:
            if extractor not in EXTRACTORS:
                raise HTTPException(
                    status_code=400, detail=f"Unknown extractor: {extractor}. Available: {list(EXTRACTORS.keys())}"
                )
            ext = get_extractor(extractor)
            start = time.time()
            extraction = ext.extract(tmp_path)
            elapsed = int((time.time() - start) * 1000)
        else:
            # Use pipeline with auto-selection
            pipeline = ReaddPipeline(auto_escalate=True)
            start = time.time()
            pipeline_result = pipeline.process(tmp_path)
            elapsed = int((time.time() - start) * 1000)

            result["extraction"] = {
                "success": pipeline_result.success,
                "extractor_used": pipeline_result.extractors_tried[-1] if pipeline_result.extractors_tried else None,
                "text_preview": pipeline_result.text[:1000] if pipeline_result.text else None,
                "text_length": len(pipeline_result.text) if pipeline_result.text else 0,
                "processing_ms": elapsed,
                "warnings": pipeline_result.warnings,
            }
            return result

        result["extraction"] = {
            "success": True,
            "extractor_used": extractor,
            "text_preview": extraction.full_text[:1000] if extraction.full_text else None,
            "text_length": len(extraction.full_text) if extraction.full_text else 0,
            "page_count": extraction.page_count,
            "processing_ms": elapsed,
            "warnings": extraction.warnings,
        }

        return result

    finally:
        # Cleanup temp file
        tmp_path.unlink(missing_ok=True)


@router.get("/test-suite")
async def get_test_suite(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Get available QA test suite info."""

    qa_dir = Path(__file__).parent.parent.parent / "data" / "qa_reference"
    manifest_path = qa_dir / "manifest.json"

    if not manifest_path.exists():
        return {
            "available": False,
            "message": "QA reference manifest not found. Run generate_test_docs.py first.",
        }

    manifest = json.loads(manifest_path.read_text())
    docs_dir = qa_dir / "docs"

    # Check which documents exist
    docs_status = []
    for doc in manifest.get("documents", []):
        pdf_path = docs_dir / doc["filename"]
        docs_status.append(
            {
                "filename": doc["filename"],
                "exists": pdf_path.exists(),
                "expected_tier": doc["expected_tier"],
                "expected_extractor": doc["expected_extractor"],
                "min_accuracy": doc["min_accuracy"],
            }
        )

    return {
        "available": True,
        "version": manifest.get("version"),
        "documents": docs_status,
    }


@router.post("/test-suite/run")
async def run_test_suite(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """
    Run full QA test suite.

    Tests all reference documents against expected accuracy thresholds.
    """

    qa_dir = Path(__file__).parent.parent.parent / "data" / "qa_reference"
    manifest_path = qa_dir / "manifest.json"
    docs_dir = qa_dir / "docs"
    expected_dir = qa_dir / "expected"

    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="QA manifest not found")

    manifest = json.loads(manifest_path.read_text())

    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "passed": 0,
        "failed": 0,
        "skipped": 0,
    }

    # Import accuracy helper
    def levenshtein_ratio(s1: str, s2: str) -> float:
        """Compute Levenshtein ratio."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        s1 = " ".join(s1.split())
        s2 = " ".join(s2.split())

        len1, len2 = len(s1), len(s2)
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1

        prev_row = list(range(len2 + 1))
        curr_row = [0] * (len2 + 1)

        for i, c1 in enumerate(s1, 1):
            curr_row[0] = i
            for j, c2 in enumerate(s2, 1):
                cost = 0 if c1 == c2 else 1
                curr_row[j] = min(curr_row[j - 1] + 1, prev_row[j] + 1, prev_row[j - 1] + cost)
            prev_row, curr_row = curr_row, prev_row

        distance = prev_row[len2]
        return 1 - (distance / max(len1, len2))

    from tools.readd.extractors import PyMuPDFTextExtractor

    for doc in manifest.get("documents", []):
        pdf_path = docs_dir / doc["filename"]
        expected_path = expected_dir / doc["filename"].replace(".pdf", ".txt")

        test_result = {
            "filename": doc["filename"],
            "expected_tier": doc["expected_tier"],
            "min_accuracy": doc["min_accuracy"],
        }

        if not pdf_path.exists():
            test_result["status"] = "skipped"
            test_result["reason"] = "PDF not found"
            results["skipped"] += 1
            results["tests"].append(test_result)
            continue

        if not expected_path.exists():
            test_result["status"] = "skipped"
            test_result["reason"] = "Ground truth not found"
            results["skipped"] += 1
            results["tests"].append(test_result)
            continue

        try:
            expected_text = expected_path.read_text(encoding="utf-8")
            extractor = PyMuPDFTextExtractor()

            start = time.time()
            extraction = extractor.extract(pdf_path)
            elapsed = int((time.time() - start) * 1000)

            accuracy = levenshtein_ratio(extraction.full_text, expected_text)
            passed = accuracy >= doc["min_accuracy"]

            test_result["status"] = "passed" if passed else "failed"
            test_result["accuracy"] = round(accuracy, 4)
            test_result["processing_ms"] = elapsed

            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1

        except Exception as e:
            test_result["status"] = "error"
            test_result["error"] = str(e)
            results["failed"] += 1

        results["tests"].append(test_result)

    return results
