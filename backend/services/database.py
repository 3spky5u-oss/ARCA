"""
PostgreSQL Connection Manager - Async database infrastructure.

Provides:
- Connection pooling with asyncpg
- Health checks and reconnection
- Graceful fallback to SQLite when PostgreSQL unavailable
- Thread-safe singleton pattern

Usage:
    from services.database import get_database

    db = await get_database()
    if db.available:
        rows = await db.fetch("SELECT * FROM feedback LIMIT 10")
"""

import logging
import asyncio
import os
import time
from typing import Optional, Any, Dict, List
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy import asyncpg to avoid startup errors if not installed
_asyncpg_module = None


def _looks_like_dns_resolution_error(error_text: str) -> bool:
    text = (error_text or "").lower()
    patterns = (
        "name or service not known",
        "temporary failure in name resolution",
        "nodename nor servname provided",
        "getaddrinfo failed",
    )
    return any(p in text for p in patterns)


def _looks_like_auth_error(error_text: str) -> bool:
    text = (error_text or "").lower()
    patterns = (
        "password authentication failed",
        "authentication failed",
        "invalid password",
        "role \"arca\" does not exist",
    )
    return any(p in text for p in patterns)


def _looks_like_transient_connect_error(error_text: str) -> bool:
    text = (error_text or "").lower()
    patterns = (
        "connection refused",
        "the database system is starting up",
        "server closed the connection unexpectedly",
        "connection reset by peer",
        "connection timed out",
        "timeout expired",
        "could not connect to server",
    )
    return any(p in text for p in patterns)


def _get_asyncpg_module():
    """Lazy import asyncpg module."""
    global _asyncpg_module
    if _asyncpg_module is None:
        try:
            import asyncpg
            _asyncpg_module = asyncpg
        except ImportError:
            logger.warning("asyncpg package not installed, PostgreSQL features disabled")
            _asyncpg_module = False
    return _asyncpg_module if _asyncpg_module else None


@dataclass
class DatabaseManager:
    """
    PostgreSQL connection manager with SQLite fallback support.

    Maintains connection state and provides graceful degradation
    when PostgreSQL is unavailable.
    """

    url: str = "postgresql://arca:arca_dev@localhost:5432/arca"
    enabled: bool = True
    pool_size: int = 10

    # SQLite fallback paths
    phii_db_path: Path = field(default_factory=lambda: Path("data/phii/reinforcement.sqlite"))
    logg_db_path: Path = field(default_factory=lambda: Path("data/logg/corrections.db"))

    # Connection state
    _pool: Any = field(default=None, repr=False)
    _available: bool = field(default=False, repr=False)
    _fallback_mode: bool = field(default=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _initialized: bool = field(default=False, repr=False)
    _last_reconnect_attempt: float = field(default=0.0, repr=False)
    _last_reconnect_error: Optional[str] = field(default=None, repr=False)
    _reconnect_interval_s: float = field(default=30.0, repr=False)

    @property
    def available(self) -> bool:
        """Check if PostgreSQL is available."""
        return self._available and not self._fallback_mode

    @property
    def fallback_mode(self) -> bool:
        """Check if operating in SQLite fallback mode."""
        return self._fallback_mode

    async def connect(self) -> bool:
        """
        Establish PostgreSQL connection pool.

        Returns:
            True if connected, False if fallback mode activated
        """
        if not self.enabled:
            logger.info("PostgreSQL disabled by config, using SQLite fallback mode")
            self._fallback_mode = True
            self._initialized = True
            return False

        asyncpg = _get_asyncpg_module()
        if not asyncpg:
            logger.warning("asyncpg module not available, using SQLite fallback mode")
            self._fallback_mode = True
            self._initialized = True
            return False

        async with self._lock:
            if self._initialized and self._available:
                return True

            max_retries = int(os.environ.get("POSTGRES_CONNECT_RETRIES", "15"))
            retry_delay_s = float(os.environ.get("POSTGRES_CONNECT_RETRY_DELAY_S", "2.0"))
            last_error: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    self._pool = await asyncpg.create_pool(
                        self.url,
                        min_size=2,
                        max_size=self.pool_size,
                        command_timeout=30.0,
                        statement_cache_size=100,
                    )
                    # Test connection
                    async with self._pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")

                    # Ensure critical tables exist (handles pre-existing volumes
                    # where docker-entrypoint-initdb.d didn't re-run init.sql)
                    await self._ensure_schema()

                    self._available = True
                    self._fallback_mode = False
                    self._initialized = True
                    self._last_reconnect_error = None
                    logger.info(f"PostgreSQL connected: pool_size={self.pool_size}")
                    return True
                except Exception as e:
                    last_error = e
                    # Retry transient startup races (e.g., postgres not yet ready).
                    if (
                        attempt < max_retries
                        and _looks_like_transient_connect_error(str(e))
                        and not _looks_like_auth_error(str(e))
                        and not _looks_like_dns_resolution_error(str(e))
                    ):
                        if attempt == 0:
                            logger.info(
                                "PostgreSQL not ready yet; retrying startup connection "
                                f"(max_retries={max_retries}, delay={retry_delay_s:.1f}s)"
                            )
                        await asyncio.sleep(retry_delay_s)
                        continue
                    break

            if last_error is not None:
                e = last_error
                if _looks_like_dns_resolution_error(str(e)):
                    logger.info(
                        "PostgreSQL host not resolvable from backend container; "
                        "using SQLite fallback mode. "
                        "If you expect PostgreSQL, ensure services are started from the same compose project."
                    )
                elif _looks_like_auth_error(str(e)):
                    logger.warning(
                        "PostgreSQL authentication failed; using SQLite fallback mode. "
                        "Fix: ensure backend and postgres use the same POSTGRES_* values. "
                        "If this is a disposable local stack, reset DB state by deleting ./data/postgres "
                        "and restarting docker compose."
                    )
                else:
                    logger.warning(f"PostgreSQL connection failed: {e}, using SQLite fallback mode")
                self._fallback_mode = True
                self._available = False
                self._initialized = True
                return False
            # Should not happen, but preserve fallback behavior.
            self._fallback_mode = True
            self._available = False
            self._initialized = True
            return False

    async def _ensure_schema(self) -> None:
        """Create critical tables if they don't exist.

        This handles the case where a PostgreSQL volume already exists
        but init.sql didn't run (Docker only runs entrypoint scripts
        on first volume creation).
        """
        try:
            init_sql = Path(__file__).parent.parent / "migrations" / "init.sql"
            if init_sql.exists():
                sql = init_sql.read_text()
                async with self._pool.acquire() as conn:
                    await conn.execute(sql)
                logger.info("PostgreSQL schema verified (init.sql applied)")
            else:
                logger.warning("migrations/init.sql not found, skipping schema check")
        except Exception as e:
            logger.warning(f"Schema verification warning (non-fatal): {e}")

    async def disconnect(self) -> None:
        """Close PostgreSQL connection pool."""
        async with self._lock:
            if self._pool:
                try:
                    await self._pool.close()
                except Exception as e:
                    logger.warning(f"Error closing PostgreSQL pool: {e}")
                finally:
                    self._pool = None
                    self._available = False

    async def health_check(self) -> Dict[str, Any]:
        """
        Check PostgreSQL health status.

        Returns:
            Dict with status, mode, and pool info
        """
        if self._fallback_mode:
            now = time.monotonic()
            if self.enabled and (now - self._last_reconnect_attempt) >= self._reconnect_interval_s:
                self._last_reconnect_attempt = now
                try:
                    if await self.try_reconnect():
                        # Reconnected successfully; continue to regular connected health check below.
                        pass
                except Exception as e:
                    self._last_reconnect_error = str(e)
                    logger.info(f"PostgreSQL reconnect attempt failed: {e}")

        if self._fallback_mode:
            return {
                "status": "fallback",
                "mode": "sqlite",
                "phii_db": str(self.phii_db_path),
                "logg_db": str(self.logg_db_path),
                "reconnect_error": self._last_reconnect_error,
            }

        if not self._pool:
            return {"status": "disconnected", "mode": "none"}

        try:
            start = asyncio.get_event_loop().time()
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            latency_ms = (asyncio.get_event_loop().time() - start) * 1000

            return {
                "status": "connected",
                "mode": "postgresql",
                "latency_ms": round(latency_ms, 2),
                "pool_size": self._pool.get_size(),
                "pool_free": self._pool.get_idle_size(),
            }
        except Exception as e:
            # Connection lost, switch to fallback
            self._fallback_mode = True
            self._available = False
            self._last_reconnect_error = str(e)
            logger.warning(f"PostgreSQL health check failed: {e}, switching to fallback")
            return {"status": "error", "mode": "fallback", "error": str(e)}

    # === Query Operations ===

    async def execute(self, query: str, *args) -> str:
        """Execute a query that doesn't return results (INSERT, UPDATE, DELETE).

        Returns:
            Status string (e.g., "INSERT 0 1")
        """
        if self._fallback_mode:
            raise RuntimeError("Cannot execute PostgreSQL query in fallback mode")

        try:
            async with self._pool.acquire() as conn:
                return await conn.execute(query, *args)
        except Exception as e:
            logger.error(f"PostgreSQL execute failed: {e}")
            self._enter_fallback()
            raise

    async def executemany(self, query: str, args: List[tuple]) -> None:
        """Execute a query multiple times with different arguments."""
        if self._fallback_mode:
            raise RuntimeError("Cannot execute PostgreSQL query in fallback mode")

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(query, args)
        except Exception as e:
            logger.error(f"PostgreSQL executemany failed: {e}")
            self._enter_fallback()
            raise

    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch multiple rows as list of dicts."""
        if self._fallback_mode:
            raise RuntimeError("Cannot execute PostgreSQL query in fallback mode")

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"PostgreSQL fetch failed: {e}")
            self._enter_fallback()
            raise

    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch a single row as dict."""
        if self._fallback_mode:
            raise RuntimeError("Cannot execute PostgreSQL query in fallback mode")

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query, *args)
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"PostgreSQL fetchrow failed: {e}")
            self._enter_fallback()
            raise

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value."""
        if self._fallback_mode:
            raise RuntimeError("Cannot execute PostgreSQL query in fallback mode")

        try:
            async with self._pool.acquire() as conn:
                return await conn.fetchval(query, *args)
        except Exception as e:
            logger.error(f"PostgreSQL fetchval failed: {e}")
            self._enter_fallback()
            raise

    async def fetch_with_returning(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Execute INSERT/UPDATE with RETURNING and fetch the result."""
        if self._fallback_mode:
            raise RuntimeError("Cannot execute PostgreSQL query in fallback mode")

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query, *args)
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"PostgreSQL fetch_with_returning failed: {e}")
            self._enter_fallback()
            raise

    # === Transaction Support ===

    async def transaction(self):
        """Get a transaction context manager.

        Usage:
            async with await db.transaction() as conn:
                await conn.execute(...)
                await conn.execute(...)
        """
        if self._fallback_mode:
            raise RuntimeError("Cannot start PostgreSQL transaction in fallback mode")

        return self._pool.acquire()

    # SQLite fallback helpers removed — no callers in codebase.
    # ReinforcementStore and logg/learning.py handle their own SQLite fallback internally.

    # === Internal ===

    def _enter_fallback(self) -> None:
        """Switch to fallback mode."""
        if not self._fallback_mode:
            logger.warning("PostgreSQL unavailable, switching to SQLite fallback mode")
            self._fallback_mode = True
            self._available = False

    async def try_reconnect(self) -> bool:
        """Attempt to reconnect to PostgreSQL."""
        if not self._fallback_mode:
            return True

        logger.info("Attempting PostgreSQL reconnection...")
        self._fallback_mode = False
        self._initialized = False
        return await self.connect()


# Singleton instance
_database_manager: Optional[DatabaseManager] = None
_init_lock = asyncio.Lock()


async def get_database() -> DatabaseManager:
    """
    Get the database manager singleton.

    Lazily initializes connection on first call.
    """
    global _database_manager

    if _database_manager is None:
        async with _init_lock:
            if _database_manager is None:
                from config import runtime_config

                _database_manager = DatabaseManager(
                    url=runtime_config.database_url,
                    enabled=runtime_config.database_enabled,
                    pool_size=runtime_config.database_pool_size,
                )
                await _database_manager.connect()

    return _database_manager


async def close_database() -> None:
    """Close the database connection (call on shutdown)."""
    global _database_manager
    if _database_manager:
        await _database_manager.disconnect()
        _database_manager = None
