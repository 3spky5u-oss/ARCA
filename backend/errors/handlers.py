"""
Error handling decorators and utilities for ARCA.

Provides decorators for consistent error handling across tool functions.
"""

import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from .exceptions import ARCAError
from .response import error_response

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


def handle_tool_errors(tool_name: str, logger: Optional[logging.Logger] = None):
    """Decorator that catches exceptions and returns standard error responses.

    Wraps a function to catch all exceptions, log them with stack traces,
    and return a standardized error response dictionary.

    Args:
        tool_name: Name of the tool for error response context
        logger: Optional logger instance (defaults to tool-specific logger)

    Returns:
        Decorated function that returns error_response on exception

    Example:
        >>> @handle_tool_errors("exceedee")
        ... def analyze_files(...):
        ...     if not files:
        ...         raise NotFoundError("No files uploaded")
        ...     # ... analysis logic ...
        ...     return {"success": True, "results": results}
    """

    def decorator(func: F) -> F:
        # Use provided logger or create one based on tool name
        log = logger or logging.getLogger(f"arca.{tool_name}")

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> dict:
            try:
                return func(*args, **kwargs)
            except ARCAError as e:
                log.error(f"[{tool_name}] {e.code.value}: {e.message}", exc_info=True)
                return error_response(e, tool=tool_name)
            except Exception as e:
                log.error(f"[{tool_name}] Unexpected error: {e}", exc_info=True)
                return error_response(e, tool=tool_name)

        return wrapper  # type: ignore

    return decorator


def handle_async_tool_errors(tool_name: str, logger: Optional[logging.Logger] = None):
    """Async version of handle_tool_errors decorator.

    Same behavior as handle_tool_errors but for async functions.

    Args:
        tool_name: Name of the tool for error response context
        logger: Optional logger instance (defaults to tool-specific logger)

    Returns:
        Decorated async function that returns error_response on exception
    """

    def decorator(func: F) -> F:
        log = logger or logging.getLogger(f"arca.{tool_name}")

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> dict:
            try:
                return await func(*args, **kwargs)
            except ARCAError as e:
                log.error(f"[{tool_name}] {e.code.value}: {e.message}", exc_info=True)
                return error_response(e, tool=tool_name)
            except Exception as e:
                log.error(f"[{tool_name}] Unexpected error: {e}", exc_info=True)
                return error_response(e, tool=tool_name)

        return wrapper  # type: ignore

    return decorator


def log_error(
    logger: logging.Logger, error: Exception, context: Optional[str] = None, include_traceback: bool = True
) -> None:
    """Log an error with consistent formatting.

    Args:
        logger: Logger instance to use
        error: The exception to log
        context: Optional context string to prefix the message
        include_traceback: Whether to include the full stack trace

    Example:
        >>> log_error(logger, err, context="Analysis")
        # Logs: "[Analysis] PARSE_EXCEL_FAILED: Could not parse file"
    """
    if isinstance(error, ARCAError):
        message = f"{error.code.value}: {error.message}"
    else:
        message = str(error)

    if context:
        message = f"[{context}] {message}"

    logger.error(message, exc_info=include_traceback)
