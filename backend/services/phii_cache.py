"""
PHII Profile Cache - Redis-backed personality profile persistence.

Persists PHII profiles (energy, specialty, expertise) across sessions.
Enables user preferences to survive backend restarts.

Key pattern: arca:phii:{session_id}
TTL: 24 hours (linked to session TTL)

Usage:
    from services.phii_cache import PhiiCache

    cache = PhiiCache()
    await cache.save_profiles(session_id, energy, specialty, expertise, message_count)
    profiles = await cache.load_profiles(session_id)
"""

import json
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import asdict

logger = logging.getLogger(__name__)

# Key prefix for PHII profile data
PHII_PREFIX = "arca:phii:"


class PhiiCache:
    """
    Redis-backed PHII profile persistence.

    Stores energy, specialty, and expertise profiles as a Redis hash
    with automatic TTL linked to session lifetime.
    """

    def __init__(self, ttl_seconds: int = 86400):
        """
        Initialize PHII cache.

        Args:
            ttl_seconds: Profile TTL in seconds (default 24 hours)
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
        """Create Redis key for PHII profiles."""
        return f"{PHII_PREFIX}{session_id}"

    async def save_profiles(
        self,
        session_id: str,
        energy_profile: Any,
        specialty_profile: Any,
        expertise_profile: Any,
        message_count: int = 0,
    ) -> bool:
        """
        Save PHII profiles to Redis.

        Args:
            session_id: Session identifier
            energy_profile: EnergyProfile object
            specialty_profile: SpecialtyProfile object
            expertise_profile: ExpertiseProfile object
            message_count: Current message count

        Returns:
            True if saved successfully
        """
        redis = await self._get_redis()
        key = self._make_key(session_id)

        try:
            # Convert profiles to dicts
            energy_dict = energy_profile.to_dict() if hasattr(energy_profile, 'to_dict') else asdict(energy_profile)
            specialty_dict = specialty_profile.to_dict() if hasattr(specialty_profile, 'to_dict') else asdict(specialty_profile)
            expertise_dict = expertise_profile.to_dict() if hasattr(expertise_profile, 'to_dict') else self._expertise_to_dict(expertise_profile)

            mapping = {
                "energy_profile": json.dumps(energy_dict),
                "specialty_profile": json.dumps(specialty_dict),
                "expertise_profile": json.dumps(expertise_dict),
                "message_count": str(message_count),
            }

            await redis.hset(key, mapping)
            await redis.expire(key, self.ttl_seconds)

            logger.debug(f"PHII profiles saved: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save PHII profiles {session_id}: {e}")
            return False

    def _expertise_to_dict(self, expertise_profile: Any) -> Dict[str, Any]:
        """Convert ExpertiseProfile to dict, handling enum."""
        return {
            "level": expertise_profile.level.value if hasattr(expertise_profile.level, 'value') else str(expertise_profile.level),
            "confidence": expertise_profile.confidence,
            "signals": expertise_profile.signals if hasattr(expertise_profile, 'signals') else [],
        }

    async def load_profiles(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load PHII profiles from Redis.

        Args:
            session_id: Session identifier

        Returns:
            Dict with energy_profile, specialty_profile, expertise_profile, message_count
            or None if not found
        """
        redis = await self._get_redis()
        key = self._make_key(session_id)

        try:
            data = await redis.hgetall(key)
            if not data:
                return None

            result = {
                "energy_profile": json.loads(data.get("energy_profile", "{}")),
                "specialty_profile": json.loads(data.get("specialty_profile", "{}")),
                "expertise_profile": json.loads(data.get("expertise_profile", "{}")),
                "message_count": int(data.get("message_count", "0")),
            }

            # Refresh TTL on load
            await redis.expire(key, self.ttl_seconds)

            logger.debug(f"PHII profiles loaded: {session_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to load PHII profiles {session_id}: {e}")
            return None

    async def save_energy_profile(self, session_id: str, energy_profile: Any) -> bool:
        """Update just the energy profile."""
        redis = await self._get_redis()
        key = self._make_key(session_id)

        try:
            energy_dict = energy_profile.to_dict() if hasattr(energy_profile, 'to_dict') else asdict(energy_profile)
            await redis.hset(key, {"energy_profile": json.dumps(energy_dict)})
            return True
        except Exception as e:
            logger.warning(f"Failed to save energy profile: {e}")
            return False

    async def save_specialty_profile(self, session_id: str, specialty_profile: Any) -> bool:
        """Update just the specialty profile."""
        redis = await self._get_redis()
        key = self._make_key(session_id)

        try:
            specialty_dict = specialty_profile.to_dict() if hasattr(specialty_profile, 'to_dict') else asdict(specialty_profile)
            await redis.hset(key, {"specialty_profile": json.dumps(specialty_dict)})
            return True
        except Exception as e:
            logger.warning(f"Failed to save specialty profile: {e}")
            return False

    async def save_expertise_profile(self, session_id: str, expertise_profile: Any) -> bool:
        """Update just the expertise profile."""
        redis = await self._get_redis()
        key = self._make_key(session_id)

        try:
            expertise_dict = self._expertise_to_dict(expertise_profile)
            await redis.hset(key, {"expertise_profile": json.dumps(expertise_dict)})
            return True
        except Exception as e:
            logger.warning(f"Failed to save expertise profile: {e}")
            return False

    async def increment_message_count(self, session_id: str) -> int:
        """Increment message count and return new value."""
        redis = await self._get_redis()
        key = self._make_key(session_id)

        try:
            # Get current count
            data = await redis.hgetall(key)
            current = int(data.get("message_count", "0"))
            new_count = current + 1

            await redis.hset(key, {"message_count": str(new_count)})
            return new_count
        except Exception as e:
            logger.warning(f"Failed to increment message count: {e}")
            return 0

    async def exists(self, session_id: str) -> bool:
        """Check if PHII profiles exist for session."""
        redis = await self._get_redis()
        key = self._make_key(session_id)
        return await redis.exists(key)

    async def delete(self, session_id: str) -> bool:
        """Delete PHII profiles for session."""
        redis = await self._get_redis()
        key = self._make_key(session_id)

        try:
            await redis.delete(key)
            logger.debug(f"PHII profiles deleted: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete PHII profiles {session_id}: {e}")
            return False


# Singleton instance
_phii_cache: Optional[PhiiCache] = None


async def get_phii_cache() -> PhiiCache:
    """Get PHII cache singleton."""
    global _phii_cache
    if _phii_cache is None:
        from config import runtime_config
        _phii_cache = PhiiCache(ttl_seconds=runtime_config.redis_session_ttl)
    return _phii_cache
