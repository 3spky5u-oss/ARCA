import logging
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/knowledge", tags=["admin-knowledge"])

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
KNOWLEDGE_DIR = BASE_DIR / "data" / "technical_knowledge"
COHESIONN_DB_DIR = BASE_DIR / "data" / "cohesionn_db"
MANIFEST_PATH = COHESIONN_DB_DIR / "manifest.json"

# Supported document extensions for knowledge ingestion
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")


def _glob_supported_files(directory: Path) -> List[Path]:
    """Glob all supported document files in a directory (non-recursive)."""
    files = []
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(f)
    return sorted(files, key=lambda p: p.name)


__all__ = ["router"]

from . import stats, ingest, deletion, search, topics, settings, reprocess, core_knowledge, qdrant
