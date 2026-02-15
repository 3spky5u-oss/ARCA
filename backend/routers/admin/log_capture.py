"""
Log capture utilities for admin API.

Provides a buffered log handler with credential scrubbing.
"""

import logging
import re as _re
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

# Log buffer for recent logs (in-memory ring buffer)
LOG_BUFFER_SIZE = 500
_log_buffer: deque = deque(maxlen=LOG_BUFFER_SIZE)

# Patterns to scrub from log messages before exposing via admin API
_SCRUB_PATTERNS = _re.compile(
    r'(?i)'
    r'(password|passwd|secret|token|api_key|apikey|admin_key|authorization|credential)'
    r'[\s]*[=:]\s*'
    r'["\']?([^\s"\',;}{]{3,})["\']?'
)


def _scrub_log_message(message: str) -> str:
    """Redact passwords, keys, and tokens from log messages."""
    return _SCRUB_PATTERNS.sub(lambda m: f"{m.group(1)}=***REDACTED***", message)


class BufferedLogHandler(logging.Handler):
    """Custom log handler that captures logs to buffer with credential scrubbing."""

    def emit(self, record):
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "message": _scrub_log_message(record.getMessage()),
                "module": record.module,
            }
            _log_buffer.append(entry)
        except Exception:
            pass


def setup_log_capture():
    """Add buffered handler to root logger."""
    handler = BufferedLogHandler()
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logger.info("Admin log capture initialized")
