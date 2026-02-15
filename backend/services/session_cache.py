"""
Session Cache - Redis-backed session persistence.

Provides session state persistence across WebSocket disconnects
and backend restarts. Sessions are stored as Redis hashes with
automatic TTL expiration.

Key pattern: arca:session:{session_id}
TTL: 24 hours (configurable)

Usage:
    from services.session_cache import SessionCache

    cache = SessionCache()
    await cache.save(session)
    session = await cache.load(session_id)
"""

import json
import logging
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from routers.chat_orchestration.session import ChatSession

logger = logging.getLogger(__name__)

# Key prefix for session data
SESSION_PREFIX = "arca:session:"


class SessionCache:
    """
    Redis-backed session persistence.

    Handles serialization/deserialization of ChatSession objects
    to Redis hashes with automatic TTL management.
    """

    def __init__(self, ttl_seconds: int = 86400):
        """
        Initialize session cache.

        Args:
            ttl_seconds: Session TTL in seconds (default 24 hours)
        """
        self.ttl_seconds = ttl_seconds
        self._redis = None

    async def _get_redis(self):
        """Lazy load Redis manager."""
        if self._redis is None:
            from .redis_client import get_redis
            self._redis = await get_redis()
        return self._redis

    def _make_key(self, session_id: str) -> str:
        """Create Redis key for session."""
        return f"{SESSION_PREFIX}{session_id}"

    async def save(self, session: "ChatSession") -> bool:
        """
        Save session to Redis.

        Args:
            session: ChatSession object to persist

        Returns:
            True if saved successfully
        """
        redis = await self._get_redis()
        key = self._make_key(session.session_id)

        try:
            # Serialize session to hash fields
            data = session.to_dict()

            # Convert complex fields to JSON strings
            mapping = {
                "session_id": data["session_id"],
                "conversation_history": json.dumps(data["conversation_history"]),
                "project_name": data.get("project_name") or "",
                "site_name": data.get("site_name") or "",
                "last_analysis_category": data.get("last_analysis_category") or "",
                "last_analysis_context": data.get("last_analysis_context") or "",
                "notes": json.dumps(data.get("notes", [])),
                "last_user_message": data.get("last_user_message") or "",
                "last_assistant_response": data.get("last_assistant_response") or "",
                "last_message_id": data.get("last_message_id") or "",
                "last_tools_used": json.dumps(data.get("last_tools_used", [])),
            }

            await redis.hset(key, mapping)
            await redis.expire(key, self.ttl_seconds)

            logger.debug(f"Session saved: {session.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False

    async def load(self, session_id: str) -> Optional["ChatSession"]:
        """
        Load session from Redis.

        Args:
            session_id: Session ID to load

        Returns:
            ChatSession if found, None otherwise
        """
        redis = await self._get_redis()
        key = self._make_key(session_id)

        try:
            data = await redis.hgetall(key)
            if not data:
                return None

            # Import here to avoid circular imports
            from routers.chat_orchestration.session import ChatSession

            # Deserialize from hash fields
            session_data = {
                "session_id": data.get("session_id", session_id),
                "conversation_history": json.loads(data.get("conversation_history", "[]")),
                "project_name": data.get("project_name") or None,
                "site_name": data.get("site_name") or None,
                "last_analysis_category": data.get("last_analysis_category") or None,
                "last_analysis_context": data.get("last_analysis_context") or None,
                "notes": json.loads(data.get("notes", "[]")),
                "last_user_message": data.get("last_user_message") or "",
                "last_assistant_response": data.get("last_assistant_response") or "",
                "last_message_id": data.get("last_message_id") or "",
                "last_tools_used": json.loads(data.get("last_tools_used", "[]")),
            }

            session = ChatSession.from_dict(session_data)

            # Refresh TTL on load
            await redis.expire(key, self.ttl_seconds)

            logger.debug(f"Session loaded: {session_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    async def exists(self, session_id: str) -> bool:
        """Check if session exists in cache."""
        redis = await self._get_redis()
        key = self._make_key(session_id)
        return await redis.exists(key)

    async def delete(self, session_id: str) -> bool:
        """
        Delete session from Redis.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted successfully
        """
        redis = await self._get_redis()
        key = self._make_key(session_id)

        try:
            await redis.delete(key)
            logger.debug(f"Session deleted: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    async def touch(self, session_id: str) -> bool:
        """
        Refresh session TTL without loading full data.

        Args:
            session_id: Session ID to refresh

        Returns:
            True if refreshed successfully
        """
        redis = await self._get_redis()
        key = self._make_key(session_id)

        try:
            if await redis.exists(key):
                await redis.expire(key, self.ttl_seconds)
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to touch session {session_id}: {e}")
            return False

    async def get_session_count(self) -> int:
        """
        Get count of active sessions (cached for 60s to avoid expensive SCAN).
        """
        redis = await self._get_redis()

        # Try cached count first
        cache_key = "arca:meta:session_count"
        try:
            cached = await redis.get(cache_key)
            if cached is not None:
                return int(cached)
        except Exception:
            pass

        # Compute and cache
        if redis.fallback_mode:
            count = sum(
                1 for k in (redis._local_cache or {})
                if k.startswith(SESSION_PREFIX)
            )
        else:
            try:
                count = 0
                cursor = 0
                pattern = f"{SESSION_PREFIX}*"
                while True:
                    cursor, keys = await redis._client.scan(cursor, match=pattern, count=100)
                    count += len(keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"Failed to count sessions: {e}")
                return -1

        # Cache for 60 seconds
        try:
            await redis.set(cache_key, str(count), ttl=60)
        except Exception:
            pass

        return count


# Singleton instance
_session_cache: Optional[SessionCache] = None


async def get_session_cache() -> SessionCache:
    """Get session cache singleton."""
    global _session_cache
    if _session_cache is None:
        from config import runtime_config
        _session_cache = SessionCache(ttl_seconds=runtime_config.redis_session_ttl)
    return _session_cache
