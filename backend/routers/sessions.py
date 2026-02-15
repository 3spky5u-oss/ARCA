"""
ARCA Sessions Router
Manages chat session persistence (optional backend sync)

Sessions are stored as JSON files in /data/sessions/
Primary storage is localStorage; backend is for optional persistence
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Session ID validation pattern: alphanumeric, hyphens, underscores, max 64 chars
_SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

logger = logging.getLogger(__name__)

router = APIRouter()

# Sessions directory (mounted volume in Docker)
BASE_DIR = Path(__file__).parent.parent
SESSIONS_DIR = BASE_DIR / "data" / "sessions"


class ChatMessage(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str
    analysisResult: Optional[Dict[str, Any]] = None
    citations: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None


class Chat(BaseModel):
    id: str
    title: str
    messages: List[ChatMessage]
    createdAt: str
    updatedAt: str


class SessionData(BaseModel):
    chats: List[Chat]


@dataclass
class StoredSession:
    session_id: str
    created_at: str
    updated_at: str
    chats: List[Dict[str, Any]]


def ensure_sessions_dir():
    """Ensure sessions directory exists"""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def get_session_path(session_id: str) -> Path:
    """Get path to session file"""
    # Sanitize session_id to prevent path traversal
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "_-")
    return SESSIONS_DIR / f"{safe_id}.json"


def load_session(session_id: str) -> Optional[StoredSession]:
    """Load session from disk"""
    path = get_session_path(session_id)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return StoredSession(**data)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to load session {session_id}: {e}")
        return None


def save_session(session: StoredSession) -> bool:
    """Save session to disk"""
    ensure_sessions_dir()
    path = get_session_path(session.session_id)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(session), f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save session {session.session_id}: {e}")
        return False


def delete_session(session_id: str) -> bool:
    """Delete session from disk"""
    path = get_session_path(session_id)
    if path.exists():
        try:
            path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    return True


def cleanup_all_sessions():
    """Delete all session files (for security cleanup on startup)"""
    ensure_sessions_dir()
    deleted = 0

    for session_file in SESSIONS_DIR.glob("*.json"):
        try:
            session_file.unlink()
            deleted += 1
        except Exception as e:
            logger.warning(f"Failed to delete session file {session_file}: {e}")

    return deleted


# =============================================================================
# API ENDPOINTS
# =============================================================================


def _validate_session_id(session_id: str) -> None:
    """Validate session ID format to prevent path traversal."""
    if not _SESSION_ID_PATTERN.match(session_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid session ID. Must be alphanumeric/hyphens/underscores, max 64 chars."
        )


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieve session data"""
    _validate_session_id(session_id)
    session = load_session(session_id)

    if not session:
        return {
            "session_id": session_id,
            "chats": [],
            "created_at": None,
            "updated_at": None,
        }

    return {
        "session_id": session.session_id,
        "chats": session.chats,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


@router.put("/sessions/{session_id}")
async def update_session(session_id: str, data: SessionData):
    """Update or create session"""
    _validate_session_id(session_id)
    now = datetime.now().isoformat()

    # Load existing session or create new
    existing = load_session(session_id)
    created_at = existing.created_at if existing else now

    session = StoredSession(
        session_id=session_id,
        created_at=created_at,
        updated_at=now,
        chats=[chat.model_dump() for chat in data.chats],
    )

    if save_session(session):
        return {
            "success": True,
            "session_id": session_id,
            "updated_at": now,
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to save session")


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Delete a session"""
    _validate_session_id(session_id)
    if delete_session(session_id):
        return {"success": True, "session_id": session_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete session")


@router.delete("/sessions")
async def clear_all_sessions():
    """Delete all sessions (admin endpoint)"""
    deleted = cleanup_all_sessions()
    return {
        "success": True,
        "sessions_deleted": deleted,
    }
