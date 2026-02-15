"""
Redis Connection Manager - Session cache infrastructure.

Provides:
- Connection pooling with async support
- Health checks and reconnection
- Graceful fallback to in-memory when Redis unavailable
- Thread-safe singleton pattern

Usage:
    from services.redis_client import get_redis

    redis = await get_redis()
    if redis.available:
        await redis.set("key", "value")
"""

import logging
import asyncio
import time
from collections import OrderedDict
from typing import Optional, Any, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Lazy import redis to avoid startup errors if not installed
_redis_module = None


def _get_redis_module():
    """Lazy import redis module."""
    global _redis_module
    if _redis_module is None:
        try:
            import redis.asyncio as redis_async
            _redis_module = redis_async
        except ImportError:
            logger.warning("redis package not installed, Redis features disabled")
            _redis_module = False
    return _redis_module if _redis_module else None


@dataclass
class RedisManager:
    """
    Redis connection manager with fallback support.

    Maintains connection state and provides graceful degradation
    when Redis is unavailable.
    """

    url: str = "redis://localhost:6379/0"
    enabled: bool = True

    # Fallback cache limits
    _fallback_max_entries: int = 1000
    _fallback_sweep_interval: float = 60.0

    # Connection state
    _client: Any = field(default=None, repr=False)
    _available: bool = field(default=False, repr=False)
    _fallback_mode: bool = field(default=False, repr=False)
    _local_cache: Any = field(default=None, repr=False)  # OrderedDict for LRU
    _local_cache_ttl: Dict[str, float] = field(default_factory=dict, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _initialized: bool = field(default=False, repr=False)
    _sweep_task: Any = field(default=None, repr=False)

    @property
    def available(self) -> bool:
        """Check if Redis is available."""
        return self._available and not self._fallback_mode

    @property
    def fallback_mode(self) -> bool:
        """Check if operating in fallback mode."""
        return self._fallback_mode

    async def connect(self) -> bool:
        """
        Establish Redis connection.

        Returns:
            True if connected, False if fallback mode activated
        """
        if not self.enabled:
            logger.info("Redis disabled by config, using fallback mode")
            self._fallback_mode = True
            self._initialized = True
            return False

        redis_module = _get_redis_module()
        if not redis_module:
            logger.warning("Redis module not available, using fallback mode")
            self._fallback_mode = True
            self._initialized = True
            return False

        async with self._lock:
            if self._initialized and self._available:
                return True

            try:
                self._client = redis_module.from_url(
                    self.url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5.0,
                    socket_timeout=5.0,
                )
                # Test connection
                await self._client.ping()
                self._available = True
                self._fallback_mode = False
                self._initialized = True
                logger.info(f"Redis connected: {self.url}")
                return True
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using fallback mode")
                self._fallback_mode = True
                self._available = False
                self._initialized = True
                return False

    async def disconnect(self) -> None:
        """Close Redis connection."""
        async with self._lock:
            if self._client:
                try:
                    await self._client.close()
                except Exception as e:
                    logger.warning(f"Error closing Redis: {e}")
                finally:
                    self._client = None
                    self._available = False

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Redis health status.

        Returns:
            Dict with status, mode, and latency info
        """
        if self._fallback_mode:
            return {
                "status": "fallback",
                "mode": "in-memory",
                "cache_size": len(self._local_cache) if self._local_cache is not None else 0,
            }

        if not self._client:
            return {"status": "disconnected", "mode": "none"}

        try:
            start = asyncio.get_event_loop().time()
            await self._client.ping()
            latency_ms = (asyncio.get_event_loop().time() - start) * 1000

            info = await self._client.info("memory")
            return {
                "status": "connected",
                "mode": "redis",
                "latency_ms": round(latency_ms, 2),
                "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
            }
        except Exception as e:
            # Connection lost, switch to fallback
            self._fallback_mode = True
            self._available = False
            logger.warning(f"Redis health check failed: {e}, switching to fallback")
            return {"status": "error", "mode": "fallback", "error": str(e)}

    # === Key-Value Operations ===

    async def get(self, key: str) -> Optional[str]:
        """Get a value by key."""
        if self._fallback_mode:
            return self._fallback_get(key)

        try:
            return await self._client.get(key)
        except Exception as e:
            logger.warning(f"Redis GET failed for {key}: {e}")
            self._enter_fallback()
            return self._fallback_get(key)

    def _fallback_set_with_ttl(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in fallback cache with optional TTL and LRU eviction."""
        # Initialize OrderedDict on first use (can't do in dataclass default)
        if self._local_cache is None:
            self._local_cache = OrderedDict()

        # Enforce max entries - LRU eviction
        if len(self._local_cache) >= self._fallback_max_entries and key not in self._local_cache:
            # Remove expired entries first
            self._sweep_expired()
            # If still over limit, evict least recently used entries
            while len(self._local_cache) >= self._fallback_max_entries:
                evicted_key, _ = self._local_cache.popitem(last=False)  # FIFO=LRU since we move_to_end on access
                self._local_cache_ttl.pop(evicted_key, None)

        self._local_cache[key] = value
        self._local_cache.move_to_end(key)  # Mark as recently used
        if ttl is not None:
            self._local_cache_ttl[key] = time.time() + ttl
        elif key in self._local_cache_ttl:
            del self._local_cache_ttl[key]

    def _fallback_get(self, key: str) -> Optional[Any]:
        """Get a value from fallback cache, respecting TTL. Marks LRU access."""
        if self._local_cache is None:
            self._local_cache = OrderedDict()
        expiry = self._local_cache_ttl.get(key)
        if expiry is not None and time.time() > expiry:
            self._local_cache.pop(key, None)
            self._local_cache_ttl.pop(key, None)
            return None
        value = self._local_cache.get(key)
        if value is not None:
            self._local_cache.move_to_end(key)  # Mark as recently used
        return value

    def _sweep_expired(self) -> None:
        """Remove expired entries from fallback cache."""
        now = time.time()
        expired = [k for k, exp in self._local_cache_ttl.items() if now > exp]
        for k in expired:
            self._local_cache.pop(k, None)
            self._local_cache_ttl.pop(k, None)

    async def set(
        self, key: str, value: str, ttl: Optional[int] = None
    ) -> bool:
        """Set a value with optional TTL in seconds."""
        if self._fallback_mode:
            self._fallback_set_with_ttl(key, value, ttl)
            return True

        try:
            if ttl:
                await self._client.setex(key, ttl, value)
            else:
                await self._client.set(key, value)
            return True
        except Exception as e:
            logger.warning(f"Redis SET failed for {key}: {e}")
            self._enter_fallback()
            self._fallback_set_with_ttl(key, value, ttl)
            return True

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        if self._fallback_mode:
            if self._local_cache is not None:
                self._local_cache.pop(key, None)
            return True

        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis DELETE failed for {key}: {e}")
            self._enter_fallback()
            self._local_cache.pop(key, None)
            return True

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if self._fallback_mode:
            return key in self._local_cache

        try:
            return await self._client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Redis EXISTS failed for {key}: {e}")
            self._enter_fallback()
            return key in self._local_cache

    # === Hash Operations (for complex objects) ===

    async def hset(self, key: str, mapping: Dict[str, str]) -> bool:
        """Set multiple hash fields."""
        if self._fallback_mode:
            if key not in self._local_cache:
                self._local_cache[key] = {}
            self._local_cache[key].update(mapping)
            return True

        try:
            await self._client.hset(key, mapping=mapping)
            return True
        except Exception as e:
            logger.warning(f"Redis HSET failed for {key}: {e}")
            self._enter_fallback()
            if key not in self._local_cache:
                self._local_cache[key] = {}
            self._local_cache[key].update(mapping)
            return True

    async def hgetall(self, key: str) -> Dict[str, str]:
        """Get all hash fields."""
        if self._fallback_mode:
            return self._local_cache.get(key, {})

        try:
            return await self._client.hgetall(key)
        except Exception as e:
            logger.warning(f"Redis HGETALL failed for {key}: {e}")
            self._enter_fallback()
            return self._local_cache.get(key, {})

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on a key."""
        if self._fallback_mode:
            if key in self._local_cache:
                self._local_cache_ttl[key] = time.time() + ttl
            return True

        try:
            await self._client.expire(key, ttl)
            return True
        except Exception as e:
            logger.warning(f"Redis EXPIRE failed for {key}: {e}")
            return False

    # === Rate Limiting Operations ===

    async def incr(self, key: str) -> int:
        """Increment a counter."""
        if self._fallback_mode:
            current = int(self._local_cache.get(key, 0))
            self._local_cache[key] = str(current + 1)
            return current + 1

        try:
            return await self._client.incr(key)
        except Exception as e:
            logger.warning(f"Redis INCR failed for {key}: {e}")
            self._enter_fallback()
            current = int(self._local_cache.get(key, 0))
            self._local_cache[key] = str(current + 1)
            return current + 1

    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL for a key."""
        if self._fallback_mode:
            return -1  # No TTL in fallback mode

        try:
            return await self._client.ttl(key)
        except Exception as e:
            logger.warning(f"Redis TTL failed for {key}: {e}")
            return -1

    # === Internal ===

    def _enter_fallback(self) -> None:
        """Switch to fallback mode."""
        if not self._fallback_mode:
            logger.warning("Redis unavailable, switching to fallback mode")
            self._fallback_mode = True
            self._available = False
            if self._local_cache is None:
                from collections import OrderedDict
                self._local_cache = OrderedDict()

    async def try_reconnect(self) -> bool:
        """Attempt to reconnect to Redis."""
        if not self._fallback_mode:
            return True

        logger.info("Attempting Redis reconnection...")
        self._fallback_mode = False
        self._initialized = False
        return await self.connect()


# Singleton instance
_redis_manager: Optional[RedisManager] = None
_init_lock = asyncio.Lock()


async def get_redis() -> RedisManager:
    """
    Get the Redis manager singleton.

    Lazily initializes connection on first call.
    """
    global _redis_manager

    if _redis_manager is None:
        async with _init_lock:
            if _redis_manager is None:
                from config import runtime_config

                _redis_manager = RedisManager(
                    url=runtime_config.redis_url,
                    enabled=runtime_config.redis_enabled,
                )
                await _redis_manager.connect()

    return _redis_manager


async def close_redis() -> None:
    """Close the Redis connection (call on shutdown)."""
    global _redis_manager
    if _redis_manager:
        await _redis_manager.disconnect()
        _redis_manager = None
