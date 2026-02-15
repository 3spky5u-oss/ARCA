import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import Depends, HTTPException, Query

from services.admin_auth import verify_admin
from . import router, KNOWLEDGE_DIR, COHESIONN_DB_DIR, MANIFEST_PATH

logger = logging.getLogger(__name__)


@router.delete("/file")
async def delete_file(
    path: str = Query(..., description="Full path to the file"),
    remove_source: bool = Query(False, description="Also delete the source PDF"),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Delete a file's chunks from the knowledge base.

    Args:
        path: Full path to the file
        remove_source: Whether to also delete the source PDF file

    Returns:
        Deletion status
    """


    file_path = Path(path)

    try:
        from tools.cohesionn import get_knowledge_base
        from tools.cohesionn.manifest import IngestManifest

        # Find file in manifest
        manifest = IngestManifest(MANIFEST_PATH)
        file_hash = None
        file_entry = None

        for h, entry in manifest.files.items():
            if entry.file_path == path:
                file_hash = h
                file_entry = entry
                break

        if not file_entry:
            raise HTTPException(status_code=404, detail=f"File not found in manifest: {path}")

        # Delete chunks from vector store
        kb = get_knowledge_base(COHESIONN_DB_DIR)
        store = kb.get_store(file_entry.topic)
        store.delete_by_source(path)

        # Remove from manifest
        del manifest.files[file_hash]
        manifest.save()

        result = {
            "success": True,
            "file": file_path.name,
            "topic": file_entry.topic,
            "chunks_deleted": file_entry.chunks_created,
            "source_deleted": False,
        }

        # Optionally delete source file
        if remove_source and file_path.exists():
            file_path.unlink()
            result["source_deleted"] = True

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/topic/{topic}")
async def delete_topic(
    topic: str,
    remove_sources: bool = Query(False, description="Also delete source PDF files"),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Clear all chunks for a topic.

    Args:
        topic: Topic name
        remove_sources: Whether to also delete source PDF files

    Returns:
        Deletion status
    """


    try:
        from tools.cohesionn import get_knowledge_base
        from tools.cohesionn.manifest import IngestManifest

        # Get current counts
        kb = get_knowledge_base(COHESIONN_DB_DIR)
        topic_stats = kb.get_stats()
        chunks_before = topic_stats.get(topic, 0)

        # Clear topic in Qdrant
        kb.clear_topic(topic)

        # Find and remove manifest entries
        manifest = IngestManifest(MANIFEST_PATH)
        files_deleted = []
        hashes_to_remove = []

        for h, entry in manifest.files.items():
            if entry.topic == topic:
                hashes_to_remove.append(h)
                files_deleted.append(entry.file_path)

        for h in hashes_to_remove:
            del manifest.files[h]
        manifest.save()

        result = {
            "success": True,
            "topic": topic,
            "chunks_deleted": chunks_before,
            "files_affected": len(files_deleted),
            "sources_deleted": 0,
        }

        # Optionally delete source files
        if remove_sources:
            for file_path_str in files_deleted:
                file_path = Path(file_path_str)
                if file_path.exists():
                    file_path.unlink()
                    result["sources_deleted"] += 1

        return result

    except Exception as e:
        logger.error(f"Topic delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup-stale")
async def cleanup_stale_entries(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """
    Remove stale manifest entries for files that no longer exist.

    Returns:
        Count of removed entries
    """


    try:
        from tools.cohesionn.manifest import IngestManifest

        manifest = IngestManifest(MANIFEST_PATH)
        stale_hashes = manifest.get_stale_hashes(KNOWLEDGE_DIR)

        if not stale_hashes:
            return {
                "success": True,
                "removed": 0,
                "message": "No stale entries found",
            }

        # Collect info before removal
        stale_files = [manifest.files[h].file_path for h in stale_hashes]

        manifest.remove_stale(stale_hashes)
        manifest.save()

        return {
            "success": True,
            "removed": len(stale_hashes),
            "files": stale_files,
        }

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
