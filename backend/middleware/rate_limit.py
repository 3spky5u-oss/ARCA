"""
Rate Limiting Middleware - Redis-backed request throttling.

Provides rate limiting for:
- WebSocket connections (per IP)
- WebSocket messages (per session)
- File uploads (per IP)

Uses Redis INCR with EXPIRE for efficient token bucket implementation.
Gracefully disables when Redis is unavailable (logs warning).

Usage:
    # REST middleware
    app.add_middleware(RateLimitMiddleware)

    # WebSocket (manual check)
    if not await check_rate_limit(RateLimitType.WS_MESSAGE, session_id):
        await websocket.send_json({"type": "error", "content": "Rate limit exceeded"})
        return
"""

import logging
from enum import Enum
from typing import Optional, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Rate limit types with their Redis key patterns."""
    WS_CONNECTION = "arca:rl:conn"    # Per IP
    WS_MESSAGE = "arca:rl:msg"        # Per session
    FILE_UPLOAD = "arca:rl:upload"    # Per IP
    ADMIN_LOGIN = "arca:rl:admin"     # Per IP, stricter


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies."""
    # Check X-Forwarded-For header (set by nginx/load balancer)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take first IP in chain (original client)
        return forwarded.split(",")[0].strip()

    # Check X-Real-IP header (nginx)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct connection IP
    if request.client:
        return request.client.host

    return "unknown"


async def check_rate_limit(
    limit_type: RateLimitType,
    identifier: str,
    limit: Optional[int] = None,
    window_seconds: int = 60,
) -> Tuple[bool, int, int]:
    """
    Check if request is within rate limit.

    Args:
        limit_type: Type of rate limit to check
        identifier: Unique identifier (IP, session_id, etc.)
        limit: Max requests per window (uses config default if None)
        window_seconds: Time window in seconds

    Returns:
        Tuple of (allowed, current_count, limit)
    """
    from config import runtime_config

    # Get limit from config if not specified
    if limit is None:
        if limit_type == RateLimitType.WS_CONNECTION:
            limit = runtime_config.rate_limit_ws_conn
        elif limit_type == RateLimitType.WS_MESSAGE:
            limit = runtime_config.rate_limit_ws_msg
        elif limit_type == RateLimitType.FILE_UPLOAD:
            limit = runtime_config.rate_limit_upload
        elif limit_type == RateLimitType.ADMIN_LOGIN:
            limit = runtime_config.rate_limit_admin_login
        else:
            limit = 30  # Default

    # In production, deny requests when Redis is unavailable (fail-closed)
    fail_closed = getattr(runtime_config, 'arca_env', 'development') == 'production'

    try:
        from services.redis_client import get_redis
        redis = await get_redis()

        if redis.fallback_mode:
            if fail_closed:
                logger.warning("Rate limiting fail-closed (Redis unavailable in production)")
                return (False, 0, limit)
            logger.debug("Rate limiting disabled (Redis fallback mode)")
            return (True, 0, limit)

        key = f"{limit_type.value}:{identifier}"

        # Increment counter
        count = await redis.incr(key)

        # Set TTL on first request in window
        if count == 1:
            await redis.expire(key, window_seconds)

        allowed = count <= limit

        if not allowed:
            logger.warning(
                f"Rate limit exceeded: {limit_type.name} for {identifier} "
                f"({count}/{limit} in {window_seconds}s)"
            )

        return (allowed, count, limit)

    except Exception as e:
        if fail_closed:
            logger.error(f"Rate limit check failed (fail-closed): {e}")
            return (False, 0, limit)
        logger.warning(f"Rate limit check failed: {e}, allowing request")
        return (True, 0, limit)


async def get_rate_limit_remaining(
    limit_type: RateLimitType,
    identifier: str,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Get remaining requests in current window.

    Args:
        limit_type: Type of rate limit
        identifier: Unique identifier

    Returns:
        Tuple of (remaining, reset_in_seconds)
    """
    from config import runtime_config

    if limit is None:
        if limit_type == RateLimitType.WS_CONNECTION:
            limit = runtime_config.rate_limit_ws_conn
        elif limit_type == RateLimitType.WS_MESSAGE:
            limit = runtime_config.rate_limit_ws_msg
        elif limit_type == RateLimitType.FILE_UPLOAD:
            limit = runtime_config.rate_limit_upload
        else:
            limit = 30

    try:
        from services.redis_client import get_redis
        redis = await get_redis()

        if redis.fallback_mode:
            return (limit, 0)

        key = f"{limit_type.value}:{identifier}"

        current = await redis.get(key)
        count = int(current) if current else 0
        remaining = max(0, limit - count)

        ttl = await redis.get_ttl(key)
        reset_in = max(0, ttl) if ttl > 0 else 0

        return (remaining, reset_in)

    except Exception as e:
        logger.warning(f"Failed to get rate limit info: {e}")
        return (limit, 0)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for REST endpoints.

    Applies rate limiting to file upload endpoints.
    WebSocket rate limiting is handled separately in the WebSocket handler.
    """

    # Endpoints to rate limit: path -> (type, window_seconds)
    RATE_LIMITED_PATHS = {
        "/api/upload": (RateLimitType.FILE_UPLOAD, 60),
        "/api/admin/login": (RateLimitType.ADMIN_LOGIN, 900),  # 5 attempts per 15 min
    }

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        path = request.url.path

        # Check if path needs rate limiting
        for limited_path, (limit_type, window) in self.RATE_LIMITED_PATHS.items():
            if path.startswith(limited_path):
                client_ip = _get_client_ip(request)
                allowed, count, limit = await check_rate_limit(limit_type, client_ip, window_seconds=window)

                if not allowed:
                    remaining, reset_in = await get_rate_limit_remaining(limit_type, client_ip)
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": "Rate limit exceeded",
                            "retry_after": reset_in,
                        },
                        headers={
                            "X-RateLimit-Limit": str(limit),
                            "X-RateLimit-Remaining": str(remaining),
                            "X-RateLimit-Reset": str(reset_in),
                            "Retry-After": str(reset_in),
                        },
                    )

                # Add rate limit headers to successful response
                response = await call_next(request)
                remaining, reset_in = await get_rate_limit_remaining(limit_type, client_ip)
                response.headers["X-RateLimit-Limit"] = str(limit)
                response.headers["X-RateLimit-Remaining"] = str(remaining)
                response.headers["X-RateLimit-Reset"] = str(reset_in)
                return response

        # No rate limiting for this path
        return await call_next(request)


async def check_ws_connection_limit(client_ip: str) -> Tuple[bool, str]:
    """
    Check WebSocket connection rate limit.

    Args:
        client_ip: Client IP address

    Returns:
        Tuple of (allowed, error_message)
    """
    allowed, count, limit = await check_rate_limit(
        RateLimitType.WS_CONNECTION,
        client_ip,
    )

    if not allowed:
        return (False, f"Connection rate limit exceeded ({count}/{limit}/min)")

    return (True, "")


async def check_ws_message_limit(session_id: str) -> Tuple[bool, str]:
    """
    Check WebSocket message rate limit.

    Args:
        session_id: Session identifier

    Returns:
        Tuple of (allowed, error_message)
    """
    allowed, count, limit = await check_rate_limit(
        RateLimitType.WS_MESSAGE,
        session_id,
    )

    if not allowed:
        remaining, reset_in = await get_rate_limit_remaining(
            RateLimitType.WS_MESSAGE,
            session_id,
        )
        return (
            False,
            f"Message rate limit exceeded ({count}/{limit}/min). "
            f"Try again in {reset_in} seconds.",
        )

    return (True, "")
