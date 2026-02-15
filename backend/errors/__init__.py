"""
ARCA Error Handling Module

Provides standardized error codes, exceptions, and response builders
for consistent error handling across the application.

Usage:
    from errors import (
        # Error codes
        ErrorCode,

        # Exceptions
        ARCAError,
        ParseError,
        ValidationError,
        NotFoundError,
        LLMError,
        ExternalServiceError,
        DependencyError,

        # Response builders
        error_response,
        success_response,
        format_error_for_llm,

        # Decorators
        handle_tool_errors,
        handle_async_tool_errors,
        log_error,
    )

Example:
    from errors import handle_tool_errors, ValidationError, NotFoundError

    @handle_tool_errors("exceedee")
    def execute_analyze_files(files_db, params):
        if not files_db:
            raise NotFoundError(
                "No files uploaded",
                details="Upload lab data files to analyze",
                resource_type="file"
            )

        category = params.get("category")
        if category not in ["A", "B"]:
            raise ValidationError(
                "Invalid category",
                details="Must be 'A' or 'B'",
                parameter="category",
                received=category
            )

        # ... analysis logic ...
        return {"success": True, "results": results}
"""

from .codes import ErrorCode
from .exceptions import (
    ARCAError,
    ParseError,
    ValidationError,
    NotFoundError,
    LLMError,
    ExternalServiceError,
    DependencyError,
)
from .response import (
    error_response,
    success_response,
    format_error_for_llm,
)
from .handlers import (
    handle_tool_errors,
    handle_async_tool_errors,
    log_error,
)

__all__ = [
    # Error codes
    "ErrorCode",
    # Exceptions
    "ARCAError",
    "ParseError",
    "ValidationError",
    "NotFoundError",
    "LLMError",
    "ExternalServiceError",
    "DependencyError",
    # Response builders
    "error_response",
    "success_response",
    "format_error_for_llm",
    # Decorators
    "handle_tool_errors",
    "handle_async_tool_errors",
    "log_error",
]
