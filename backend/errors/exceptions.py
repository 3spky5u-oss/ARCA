"""
Custom exception hierarchy for ARCA.

All exceptions inherit from ARCAError (ARCAError) and include:
- code: ErrorCode for categorization
- message: Human-readable error message
- details: Optional additional context
- recoverable: Whether the user can retry/fix the issue
- context: Additional key-value pairs for debugging
"""

from typing import Any, Optional
from .codes import ErrorCode


class ARCAError(Exception):
    """Base exception for all ARCA errors.

    Attributes:
        code: The ErrorCode categorizing this error
        message: Human-readable error message
        details: Optional additional context for the user
        recoverable: Whether the error can be resolved by user action
        context: Additional debugging information
    """

    code: ErrorCode = ErrorCode.INTERNAL_UNEXPECTED
    recoverable: bool = False

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        code: Optional[ErrorCode] = None,
        recoverable: Optional[bool] = None,
        **context: Any,
    ):
        self.message = message
        self.details = details
        self.context = context if context else None

        # Allow overriding class defaults
        if code is not None:
            self.code = code
        if recoverable is not None:
            self.recoverable = recoverable

        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - {self.details}"
        return self.message

    def to_dict(self) -> dict:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
            "context": self.context,
        }


class ParseError(ARCAError):
    """Error during file parsing (Excel, PDF, Word)."""

    code = ErrorCode.PARSE_EXCEL_FAILED
    recoverable = True

    def __init__(self, message: str, details: Optional[str] = None, file_type: Optional[str] = None, **context: Any):
        # Set appropriate code based on file type
        if file_type == "pdf":
            code = ErrorCode.PARSE_PDF_FAILED
        elif file_type == "word":
            code = ErrorCode.PARSE_WORD_FAILED
        elif file_type == "unknown":
            code = ErrorCode.PARSE_FORMAT_UNKNOWN
        else:
            code = ErrorCode.PARSE_EXCEL_FAILED

        super().__init__(message, details, code=code, **context)


class ValidationError(ARCAError):
    """Error during input validation."""

    code = ErrorCode.VALIDATION_MISSING_PARAM
    recoverable = True

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        parameter: Optional[str] = None,
        expected: Optional[str] = None,
        received: Optional[str] = None,
        **context: Any,
    ):
        ctx = {**context}
        if parameter:
            ctx["parameter"] = parameter
        if expected:
            ctx["expected"] = expected
        if received:
            ctx["received"] = received
        super().__init__(message, details, **ctx)


class NotFoundError(ARCAError):
    """Error when a required resource is not found."""

    code = ErrorCode.NOT_FOUND_FILE
    recoverable = True

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **context: Any,
    ):
        # Set appropriate code based on resource type
        if resource_type == "guideline":
            code = ErrorCode.NOT_FOUND_GUIDELINE
        elif resource_type == "calculation":
            code = ErrorCode.NOT_FOUND_CALCULATION
        elif resource_type == "topic":
            code = ErrorCode.NOT_FOUND_TOPIC
        elif resource_type == "template":
            code = ErrorCode.NOT_FOUND_TEMPLATE
        else:
            code = ErrorCode.NOT_FOUND_FILE

        ctx = {**context}
        if resource_type:
            ctx["resource_type"] = resource_type
        if resource_id:
            ctx["resource_id"] = resource_id
        super().__init__(message, details, code=code, **ctx)


class LLMError(ARCAError):
    """Error during LLM interactions."""

    code = ErrorCode.LLM_UNAVAILABLE
    recoverable = False

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        model: Optional[str] = None,
        error_type: Optional[str] = None,
        **context: Any,
    ):
        # Set appropriate code based on error type
        if error_type == "timeout":
            code = ErrorCode.LLM_TIMEOUT
        elif error_type == "parse":
            code = ErrorCode.LLM_PARSE_FAILED
        elif error_type == "invalid":
            code = ErrorCode.LLM_RESPONSE_INVALID
        else:
            code = ErrorCode.LLM_UNAVAILABLE

        ctx = {**context}
        if model:
            ctx["model"] = model
        super().__init__(message, details, code=code, **ctx)


class ExternalServiceError(ARCAError):
    """Error with external services (SearXNG, LLM server, etc.)."""

    code = ErrorCode.EXTERNAL_NETWORK_ERROR
    recoverable = True

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        service: Optional[str] = None,
        status_code: Optional[int] = None,
        **context: Any,
    ):
        # Set appropriate code based on service
        if service == "searxng":
            code = ErrorCode.EXTERNAL_SEARXNG_FAILED
        elif service == "llm":
            code = ErrorCode.EXTERNAL_LLM_FAILED
        else:
            code = ErrorCode.EXTERNAL_NETWORK_ERROR

        ctx = {**context}
        if service:
            ctx["service"] = service
        if status_code:
            ctx["status_code"] = status_code
        super().__init__(message, details, code=code, **ctx)


class DependencyError(ARCAError):
    """Error with missing or failed dependencies."""

    code = ErrorCode.DEPENDENCY_MISSING
    recoverable = False

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        package: Optional[str] = None,
        install_hint: Optional[str] = None,
        **context: Any,
    ):
        ctx = {**context}
        if package:
            ctx["package"] = package
        if install_hint:
            ctx["install_hint"] = install_hint
        super().__init__(message, details, **ctx)
