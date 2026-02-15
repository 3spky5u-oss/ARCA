import json
import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import Depends

from services.admin_auth import verify_admin
from . import router, KNOWLEDGE_DIR, COHESIONN_DB_DIR, MANIFEST_PATH, _glob_supported_files

logger = logging.getLogger(__name__)


@router.get("/stats")
async def get_knowledge_stats(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """
    Get comprehensive knowledge base statistics.

    Returns:
        - Total chunks across all topics
        - Per-topic breakdown: files, chunks
        - Manifest health info
        - Last update timestamp
    """


    stats = {
        "total_chunks": 0,
        "total_files": 0,
        "topics": {},
        "manifest": {
            "tracked_files": 0,
            "stale_entries": 0,
            "last_updated": None,
        },
    }

    try:
        # Get vector store stats
        from tools.cohesionn import get_knowledge_base

        kb = get_knowledge_base(COHESIONN_DB_DIR)
        topic_stats = kb.get_stats()

        for topic, chunk_count in topic_stats.items():
            stats["topics"][topic] = {
                "chunks": chunk_count,
                "files": 0,  # Will be filled from manifest
            }
            stats["total_chunks"] += chunk_count

        # Get manifest stats
        from tools.cohesionn.manifest import IngestManifest

        manifest = IngestManifest(MANIFEST_PATH)
        manifest_stats = manifest.get_stats()

        stats["total_files"] = manifest_stats["total_files"]
        stats["manifest"]["tracked_files"] = manifest_stats["total_files"]

        # Update file counts per topic from manifest
        for topic, topic_data in manifest_stats.get("by_topic", {}).items():
            if topic in stats["topics"]:
                stats["topics"][topic]["files"] = topic_data["files"]
            else:
                stats["topics"][topic] = {
                    "files": topic_data["files"],
                    "chunks": topic_data["chunks"],
                }

        # Core knowledge lives in a dedicated directory and does not use ingest manifest.
        # Surface file counts so admin UI doesn't show "0 files" for arca_core.
        try:
            from config import runtime_config

            core_topic = "arca_core"
            core_dir = Path(runtime_config.core_knowledge_dir)
            if core_topic in stats["topics"] and core_dir.exists():
                core_files = list(core_dir.glob("*.md"))
                if core_files and stats["topics"][core_topic].get("files", 0) == 0:
                    stats["topics"][core_topic]["files"] = len(core_files)
                    stats["total_files"] += len(core_files)
        except Exception:
            pass

        # Check for stale entries
        stale_hashes = manifest.get_stale_hashes(KNOWLEDGE_DIR)
        stats["manifest"]["stale_entries"] = len(stale_hashes)

        # Also include filesystem-only topics (empty folders not yet in Qdrant/manifest)
        if KNOWLEDGE_DIR.exists():
            for d in KNOWLEDGE_DIR.iterdir():
                if d.is_dir() and d.name not in stats["topics"]:
                    stats["topics"][d.name] = {"files": 0, "chunks": 0}

        # Get last updated from manifest file
        if MANIFEST_PATH.exists():
            manifest_data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            stats["manifest"]["last_updated"] = manifest_data.get("updated_at")

    except Exception as e:
        logger.error(f"Failed to get knowledge stats: {e}")
        stats["error"] = str(e)

    return stats


@router.get("/topic/{topic}")
async def get_topic_files(topic: str, _: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """
    Get detailed information about files in a topic.

    Returns:
        - List of ingested files with metadata
        - List of untracked files (in folder but not ingested)
        - Chunk counts per file
        - File sizes and ingestion dates
    """


    result = {
        "topic": topic,
        "files": [],
        "untracked": [],  # Files in folder but not in manifest
    }

    try:
        # arca_core is built-in docs, not manifest-tracked.
        if topic == "arca_core":
            from config import runtime_config
            from tools.cohesionn.chunker import SemanticChunker

            core_dir = Path(runtime_config.core_knowledge_dir)
            chunker = SemanticChunker(chunk_size=800, chunk_overlap=150)

            if core_dir.exists():
                for md_file in sorted(core_dir.glob("*.md")):
                    try:
                        text = md_file.read_text(encoding="utf-8")
                        chunks = len(chunker.chunk_text(text, metadata={"source": md_file.name})) if text.strip() else 0
                    except Exception:
                        chunks = 0
                    stat = md_file.stat()
                    result["files"].append(
                        {
                            "filename": md_file.name,
                            "path": str(md_file),
                            "chunks": chunks,
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "ingested_at": None,
                            "file_hash": None,
                            "exists": True,
                            "status": "ingested",
                        }
                    )
            return result

        from tools.cohesionn.manifest import IngestManifest

        manifest = IngestManifest(MANIFEST_PATH)

        # Get all ingested file paths for this topic
        ingested_paths = set()

        # Find files for this topic from manifest
        for file_hash, entry in manifest.files.items():
            if entry.topic == topic:
                file_path = Path(entry.file_path)
                ingested_paths.add(str(file_path.resolve()))
                result["files"].append(
                    {
                        "filename": file_path.name,
                        "path": entry.file_path,
                        "chunks": entry.chunks_created,
                        "size_mb": round(entry.file_size / (1024 * 1024), 2),
                        "ingested_at": entry.ingested_at,
                        "file_hash": file_hash,
                        "exists": file_path.exists(),
                        "status": "ingested",
                    }
                )

        # Find untracked files in the topic folder
        topic_dir = KNOWLEDGE_DIR / topic
        if topic_dir.exists():
            for doc_file in _glob_supported_files(topic_dir):
                resolved_path = str(doc_file.resolve())
                if resolved_path not in ingested_paths:
                    stat = doc_file.stat()
                    result["untracked"].append(
                        {
                            "filename": doc_file.name,
                            "path": str(doc_file),
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "status": "untracked",
                        }
                    )

        # Sort both lists by filename
        result["files"].sort(key=lambda x: x["filename"])
        result["untracked"].sort(key=lambda x: x["filename"])

    except Exception as e:
        logger.error(f"Failed to get topic files: {e}")
        result["error"] = str(e)

    return result
