"""
ARCA Chat Executors - Documents (Core)

Document processing: PII redaction.
Domain-specific document executors moved to domains/{domain}/executors/.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Callable

from errors import (
    handle_tool_errors,
    NotFoundError,
    ValidationError,
    DependencyError,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports"


@handle_tool_errors("redactrr")
def execute_redact_document(
    file_id: str = None, files_db: Dict = None, get_file_data_fn: Callable = None
) -> Dict[str, Any]:
    """Redact PII from a document.

    Args:
        file_id: ID of file to redact (uses most recent if not specified)
        files_db: Files database dict (injected from upload router)
        get_file_data_fn: Function to get file data (injected from upload router)
    """
    # Get most recent file if not specified
    if not file_id and files_db:
        file_id = list(files_db.keys())[-1]

    if not file_id:
        raise NotFoundError("No file to redact", details="Upload a document (PDF or Word) first", resource_type="file")

    file_data = get_file_data_fn(file_id) if get_file_data_fn else None
    if not file_data:
        raise NotFoundError(
            "File not found", details="The specified file could not be found", resource_type="file", resource_id=file_id
        )

    # Get filename
    filename = file_data.filename if hasattr(file_data, "filename") else file_data.get("filename", "document")

    # Check if it's a supported type
    ext = Path(filename).suffix.lower()
    if ext not in [".pdf", ".docx", ".doc"]:
        raise ValidationError(
            f"Unsupported file type: {ext}",
            details="Redaction only supports PDF and Word files (.pdf, .docx, .doc)",
            parameter="file_type",
            received=ext,
        )

    try:
        from tools.redactrr import Redactrr
    except ImportError:
        raise DependencyError(
            "Redactrr not installed", details="The PII redaction module is not available", package="redactrr"
        )

    # Get file bytes
    if hasattr(file_data, "data"):
        file_bytes = file_data.data
    elif hasattr(file_data, "get_bytesio"):
        file_bytes = file_data.get_bytesio().read()
    else:
        # Old format - read from disk path
        paths = file_data.get("paths", [])
        if not paths:
            raise NotFoundError("Cannot access file data", details="File data is not accessible", resource_type="file")
        with open(paths[0], "rb") as f:
            file_bytes = f.read()

    # Write to temp file (redactrr needs file paths)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        redactor = Redactrr(use_llm=True)

        # Output to temp file first
        output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        output_path = output_tmp.name
        output_tmp.close()

        result = redactor.redact(tmp_path, output_path)

        if result.success:
            # Read redacted file
            with open(output_path, "rb") as f:
                redacted_bytes = f.read()

            # Save to disk in REPORTS_DIR (like analyze_files does)
            # Use simple filename: Redacted.docx or Redacted.pdf
            REPORTS_DIR.mkdir(exist_ok=True)
            output_name = f"Redacted{ext}"
            final_path = REPORTS_DIR / output_name
            final_path.write_bytes(redacted_bytes)

            download_path = f"/api/download/{output_name}"
            logger.info(f"Saved redacted file to {final_path}")

            return {
                "success": True,
                "original_file": filename,
                "redacted_file": download_path,
                "entities_found": result.entities_found,
                "entities_redacted": result.entities_redacted,
                "entity_types": result.entity_types,
            }
        else:
            # Redaction failed internally
            raise ValidationError("Redaction failed", details=result.error or "Unknown error during redaction")

    finally:
        # Cleanup temp files
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        try:
            os.unlink(output_path)
        except (OSError, NameError):
            pass


