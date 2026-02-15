"""
Standard error response builders for ARCA.

Provides consistent response formats for error handling across all tools.
"""

from typing import Any, Optional
from .codes import ErrorCode
from .exceptions import ARCAError


def error_response(error: ARCAError | Exception, tool: Optional[str] = None, include_context: bool = True) -> dict:
    """Build a standard error response dictionary.

    Args:
        error: The exception to convert to a response
        tool: Optional tool name for context
        include_context: Whether to include the context dict (disable for privacy)

    Returns:
        Standard error response dict with success=False

    Example:
        >>> from errors import ValidationError, error_response
        >>> err = ValidationError("Missing parameter", parameter="file_id")
        >>> error_response(err, tool="exceedee")
        {
            "success": False,
            "error": {
                "code": "VALIDATION_MISSING_PARAM",
                "message": "Missing parameter",
                "details": None,
                "tool": "exceedee",
                "recoverable": True,
                "context": {"parameter": "file_id"}
            }
        }
    """
    if isinstance(error, ARCAError):
        return {
            "success": False,
            "error": {
                "code": error.code.value,
                "message": error.message,
                "details": error.details,
                "tool": tool,
                "recoverable": error.recoverable,
                "context": error.context if include_context else None,
            },
        }

    # Fallback for non-ARCA exceptions
    return {
        "success": False,
        "error": {
            "code": ErrorCode.INTERNAL_UNEXPECTED.value,
            "message": str(error),
            "details": None,
            "tool": tool,
            "recoverable": False,
            "context": None,
        },
    }


def success_response(data: Optional[dict] = None, **kwargs: Any) -> dict:
    """Build a standard success response dictionary.

    Args:
        data: Optional data dict to include in response
        **kwargs: Additional key-value pairs to include at top level

    Returns:
        Standard success response dict with success=True

    Example:
        >>> success_response(result=42)
        {"success": True, "result": 42}

        >>> success_response({"items": [1, 2, 3]})
        {"success": True, "items": [1, 2, 3]}
    """
    response = {"success": True}

    if data:
        response.update(data)
    if kwargs:
        response.update(kwargs)

    return response


def format_error_for_llm(error: ARCAError | Exception, tool: Optional[str] = None) -> str:
    """Format an error for inclusion in LLM context.

    Creates a concise, readable error message suitable for the LLM to
    understand and communicate to the user.

    Args:
        error: The exception to format
        tool: Optional tool name for context

    Returns:
        Formatted error string
    """
    if isinstance(error, ARCAError):
        parts = [f"Error: {error.message}"]
        if error.details:
            parts.append(f"Details: {error.details}")
        if error.recoverable:
            parts.append("This error may be recoverable by the user.")
        return " ".join(parts)

    return f"Error: {str(error)}"
