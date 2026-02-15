"""
Logs and sessions endpoints.
"""

from typing import Dict, Any, Optional

from fastapi import Depends

from . import router
from .log_capture import _log_buffer
from services.admin_auth import verify_admin


@router.get("/logs")
async def get_logs(
    limit: int = 100,
    level: Optional[str] = None,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Get recent log entries.

    Args:
        limit: Max entries to return (default 100)
        level: Filter by log level (INFO, WARNING, ERROR)
    """

    logs = list(_log_buffer)

    # Filter by level if specified
    if level:
        level_upper = level.upper()
        logs = [l for l in logs if l["level"] == level_upper]

    # Return most recent first, limited
    logs = logs[-limit:][::-1]

    return {
        "count": len(logs),
        "total_buffered": len(_log_buffer),
        "logs": logs,
    }


@router.get("/sessions")
async def get_sessions(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Get active session information."""

    sessions = []

    try:
        from routers.upload import files_db

        for file_id, stored in files_db.items():
            sessions.append(
                {
                    "file_id": file_id,
                    "filename": stored.original_filename,
                    "type": stored.file_type,
                    "rag_chunks": stored.rag_chunks,
                }
            )
    except Exception as e:
        return {"error": str(e), "sessions": []}

    return {
        "count": len(sessions),
        "sessions": sessions,
    }
