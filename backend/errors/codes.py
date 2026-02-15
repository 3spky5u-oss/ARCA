"""
Error codes for ARCA application.

Provides a standardized taxonomy of error codes organized by category.
Use these codes consistently across all error responses.
"""

from enum import Enum


class ErrorCode(str, Enum):
    """Standardized error codes for ARCA.

    Categories:
    - PARSE_*: File parsing and data extraction errors
    - VALIDATION_*: Input validation errors
    - NOT_FOUND_*: Resource not found errors
    - LLM_*: Language model errors
    - EXTERNAL_*: External service errors
    - DEPENDENCY_*: Missing dependency errors
    - INTERNAL_*: Internal/unexpected errors
    """

    # Parse errors (file processing)
    PARSE_EXCEL_FAILED = "PARSE_EXCEL_FAILED"
    PARSE_PDF_FAILED = "PARSE_PDF_FAILED"
    PARSE_WORD_FAILED = "PARSE_WORD_FAILED"
    PARSE_FORMAT_UNKNOWN = "PARSE_FORMAT_UNKNOWN"
    PARSE_DATA_INVALID = "PARSE_DATA_INVALID"

    # Validation errors (input checking)
    VALIDATION_MISSING_PARAM = "VALIDATION_MISSING_PARAM"
    VALIDATION_INVALID_TYPE = "VALIDATION_INVALID_TYPE"
    VALIDATION_OUT_OF_RANGE = "VALIDATION_OUT_OF_RANGE"
    VALIDATION_INVALID_FORMAT = "VALIDATION_INVALID_FORMAT"

    # Not found errors (missing resources)
    NOT_FOUND_FILE = "NOT_FOUND_FILE"
    NOT_FOUND_GUIDELINE = "NOT_FOUND_GUIDELINE"
    NOT_FOUND_CALCULATION = "NOT_FOUND_CALCULATION"
    NOT_FOUND_TOPIC = "NOT_FOUND_TOPIC"
    NOT_FOUND_TEMPLATE = "NOT_FOUND_TEMPLATE"

    # LLM errors (model interactions)
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_PARSE_FAILED = "LLM_PARSE_FAILED"
    LLM_RESPONSE_INVALID = "LLM_RESPONSE_INVALID"

    # External service errors
    EXTERNAL_SEARXNG_FAILED = "EXTERNAL_SEARXNG_FAILED"
    EXTERNAL_LLM_FAILED = "EXTERNAL_LLM_FAILED"
    EXTERNAL_NETWORK_ERROR = "EXTERNAL_NETWORK_ERROR"

    # Dependency errors (missing packages/modules)
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    DEPENDENCY_INIT_FAILED = "DEPENDENCY_INIT_FAILED"
    DEPENDENCY_VERSION_MISMATCH = "DEPENDENCY_VERSION_MISMATCH"

    # Internal errors (unexpected failures)
    INTERNAL_UNEXPECTED = "INTERNAL_UNEXPECTED"
    INTERNAL_CONFIG_ERROR = "INTERNAL_CONFIG_ERROR"
    INTERNAL_STATE_ERROR = "INTERNAL_STATE_ERROR"
