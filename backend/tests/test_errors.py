"""
Tests for the ARCA error handling module.
"""

import logging
from errors import (
    ErrorCode,
    ARCAError,
    ParseError,
    ValidationError,
    NotFoundError,
    LLMError,
    ExternalServiceError,
    DependencyError,
    error_response,
    success_response,
    format_error_for_llm,
    handle_tool_errors,
)


class TestErrorCodes:
    """Test error code enum."""

    def test_error_codes_are_strings(self):
        """Error codes should be string values."""
        assert ErrorCode.PARSE_EXCEL_FAILED.value == "PARSE_EXCEL_FAILED"
        assert ErrorCode.NOT_FOUND_FILE.value == "NOT_FOUND_FILE"

    def test_error_codes_have_categories(self):
        """Error codes should follow category naming convention."""
        parse_codes = [c for c in ErrorCode if c.value.startswith("PARSE_")]
        assert len(parse_codes) >= 4

        validation_codes = [c for c in ErrorCode if c.value.startswith("VALIDATION_")]
        assert len(validation_codes) >= 3


class TestARCAError:
    """Test base ARCAError exception."""

    def test_basic_creation(self):
        """Create basic error with message."""
        err = ARCAError("Test error")
        assert err.message == "Test error"
        assert err.details is None
        assert err.code == ErrorCode.INTERNAL_UNEXPECTED
        assert err.recoverable is False

    def test_with_details(self):
        """Create error with details."""
        err = ARCAError("Test error", details="More info")
        assert err.message == "Test error"
        assert err.details == "More info"

    def test_with_context(self):
        """Create error with additional context."""
        err = ARCAError("Test error", foo="bar", count=42)
        assert err.context == {"foo": "bar", "count": 42}

    def test_str_representation(self):
        """String representation includes message and details."""
        err = ARCAError("Test error", details="More info")
        assert str(err) == "Test error - More info"

        err_no_details = ARCAError("Test error")
        assert str(err_no_details) == "Test error"

    def test_to_dict(self):
        """Convert error to dictionary."""
        err = ARCAError("Test error", details="More info", key="value")
        d = err.to_dict()
        assert d["code"] == ErrorCode.INTERNAL_UNEXPECTED.value
        assert d["message"] == "Test error"
        assert d["details"] == "More info"
        assert d["recoverable"] is False
        assert d["context"] == {"key": "value"}


class TestParseError:
    """Test ParseError exception."""

    def test_default_code(self):
        """Default code is PARSE_EXCEL_FAILED."""
        err = ParseError("Failed to parse")
        assert err.code == ErrorCode.PARSE_EXCEL_FAILED
        assert err.recoverable is True

    def test_pdf_file_type(self):
        """PDF file type sets appropriate code."""
        err = ParseError("Failed to parse", file_type="pdf")
        assert err.code == ErrorCode.PARSE_PDF_FAILED

    def test_word_file_type(self):
        """Word file type sets appropriate code."""
        err = ParseError("Failed to parse", file_type="word")
        assert err.code == ErrorCode.PARSE_WORD_FAILED

    def test_unknown_file_type(self):
        """Unknown file type sets appropriate code."""
        err = ParseError("Failed to parse", file_type="unknown")
        assert err.code == ErrorCode.PARSE_FORMAT_UNKNOWN


class TestValidationError:
    """Test ValidationError exception."""

    def test_default_code(self):
        """Default code is VALIDATION_MISSING_PARAM."""
        err = ValidationError("Missing param")
        assert err.code == ErrorCode.VALIDATION_MISSING_PARAM
        assert err.recoverable is True

    def test_with_parameter_info(self):
        """Include parameter context."""
        err = ValidationError("Invalid value", parameter="soil_type", expected="fine or coarse", received="sandy")
        assert err.context["parameter"] == "soil_type"
        assert err.context["expected"] == "fine or coarse"
        assert err.context["received"] == "sandy"


class TestNotFoundError:
    """Test NotFoundError exception."""

    def test_default_code(self):
        """Default code is NOT_FOUND_FILE."""
        err = NotFoundError("Not found")
        assert err.code == ErrorCode.NOT_FOUND_FILE
        assert err.recoverable is True

    def test_guideline_resource_type(self):
        """Guideline resource type sets appropriate code."""
        err = NotFoundError("Not found", resource_type="guideline")
        assert err.code == ErrorCode.NOT_FOUND_GUIDELINE

    def test_calculation_resource_type(self):
        """Calculation resource type sets appropriate code."""
        err = NotFoundError("Not found", resource_type="calculation")
        assert err.code == ErrorCode.NOT_FOUND_CALCULATION

    def test_with_resource_id(self):
        """Include resource ID in context."""
        err = NotFoundError("Not found", resource_type="file", resource_id="abc123")
        assert err.context["resource_type"] == "file"
        assert err.context["resource_id"] == "abc123"


class TestLLMError:
    """Test LLMError exception."""

    def test_default_code(self):
        """Default code is LLM_UNAVAILABLE."""
        err = LLMError("LLM error")
        assert err.code == ErrorCode.LLM_UNAVAILABLE
        assert err.recoverable is False

    def test_timeout_error_type(self):
        """Timeout error type sets appropriate code."""
        err = LLMError("Timed out", error_type="timeout")
        assert err.code == ErrorCode.LLM_TIMEOUT

    def test_with_model(self):
        """Include model in context."""
        err = LLMError("Error", model="qwen3:32b")
        assert err.context["model"] == "qwen3:32b"


class TestExternalServiceError:
    """Test ExternalServiceError exception."""

    def test_default_code(self):
        """Default code is EXTERNAL_NETWORK_ERROR."""
        err = ExternalServiceError("Network error")
        assert err.code == ErrorCode.EXTERNAL_NETWORK_ERROR
        assert err.recoverable is True

    def test_searxng_service(self):
        """SearXNG service sets appropriate code."""
        err = ExternalServiceError("Failed", service="searxng")
        assert err.code == ErrorCode.EXTERNAL_SEARXNG_FAILED

    def test_llm_service(self):
        """LLM service sets appropriate code."""
        err = ExternalServiceError("Failed", service="llm")
        assert err.code == ErrorCode.EXTERNAL_LLM_FAILED

    def test_with_status_code(self):
        """Include status code in context."""
        err = ExternalServiceError("Failed", service="searxng", status_code=500)
        assert err.context["service"] == "searxng"
        assert err.context["status_code"] == 500


class TestDependencyError:
    """Test DependencyError exception."""

    def test_default_code(self):
        """Default code is DEPENDENCY_MISSING."""
        err = DependencyError("Missing package")
        assert err.code == ErrorCode.DEPENDENCY_MISSING
        assert err.recoverable is False

    def test_with_package(self):
        """Include package name in context."""
        err = DependencyError("Missing", package="cohesionn")
        assert err.context["package"] == "cohesionn"

    def test_with_install_hint(self):
        """Include install hint in context."""
        err = DependencyError("Missing", package="cohesionn", install_hint="pip install cohesionn")
        assert err.context["install_hint"] == "pip install cohesionn"


class TestErrorResponse:
    """Test error_response function."""

    def test_arca_error_response(self):
        """Convert ARCAError to response dict."""
        err = NotFoundError("No files", details="Upload first")
        resp = error_response(err, tool="exceedee")

        assert resp["success"] is False
        assert resp["error"]["code"] == "NOT_FOUND_FILE"
        assert resp["error"]["message"] == "No files"
        assert resp["error"]["details"] == "Upload first"
        assert resp["error"]["tool"] == "exceedee"
        assert resp["error"]["recoverable"] is True

    def test_generic_exception_response(self):
        """Convert generic Exception to response dict."""
        err = ValueError("Bad value")
        resp = error_response(err, tool="test")

        assert resp["success"] is False
        assert resp["error"]["code"] == "INTERNAL_UNEXPECTED"
        assert resp["error"]["message"] == "Bad value"
        assert resp["error"]["recoverable"] is False

    def test_without_context(self):
        """Exclude context when requested."""
        err = NotFoundError("No files", resource_id="abc123")
        resp = error_response(err, include_context=False)

        assert resp["error"]["context"] is None


class TestSuccessResponse:
    """Test success_response function."""

    def test_basic_success(self):
        """Create basic success response."""
        resp = success_response()
        assert resp == {"success": True}

    def test_with_kwargs(self):
        """Include additional kwargs."""
        resp = success_response(result=42, items=[1, 2, 3])
        assert resp["success"] is True
        assert resp["result"] == 42
        assert resp["items"] == [1, 2, 3]

    def test_with_data_dict(self):
        """Include data dictionary."""
        resp = success_response({"items": [1, 2], "count": 2})
        assert resp["success"] is True
        assert resp["items"] == [1, 2]
        assert resp["count"] == 2


class TestFormatErrorForLLM:
    """Test format_error_for_llm function."""

    def test_arca_error(self):
        """Format ARCAError for LLM."""
        err = NotFoundError("No files", details="Upload files first")
        formatted = format_error_for_llm(err)

        assert "Error: No files" in formatted
        assert "Details: Upload files first" in formatted
        assert "recoverable" in formatted.lower()

    def test_generic_exception(self):
        """Format generic Exception for LLM."""
        err = ValueError("Bad value")
        formatted = format_error_for_llm(err)

        assert "Error: Bad value" in formatted


class TestHandleToolErrors:
    """Test handle_tool_errors decorator."""

    def test_success_passthrough(self):
        """Successful function returns normally."""

        @handle_tool_errors("test")
        def my_func():
            return {"success": True, "result": 42}

        result = my_func()
        assert result["success"] is True
        assert result["result"] == 42

    def test_arca_error_handling(self):
        """ARCAError is caught and converted."""

        @handle_tool_errors("test")
        def my_func():
            raise NotFoundError("Not found")

        result = my_func()
        assert result["success"] is False
        assert result["error"]["code"] == "NOT_FOUND_FILE"
        assert result["error"]["tool"] == "test"

    def test_generic_exception_handling(self):
        """Generic Exception is caught and converted."""

        @handle_tool_errors("test")
        def my_func():
            raise ValueError("Bad value")

        result = my_func()
        assert result["success"] is False
        assert result["error"]["code"] == "INTERNAL_UNEXPECTED"

    def test_logging(self, caplog):
        """Errors are logged with stack trace."""

        @handle_tool_errors("test")
        def my_func():
            raise NotFoundError("Not found")

        with caplog.at_level(logging.ERROR):
            my_func()

        assert "NOT_FOUND_FILE" in caplog.text
        assert "Not found" in caplog.text

    def test_preserves_function_metadata(self):
        """Decorator preserves function name and docstring."""

        @handle_tool_errors("test")
        def my_func():
            """My docstring."""
            return {"success": True}

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."


class TestAsyncHandleToolErrors:
    """Test handle_async_tool_errors decorator.

    Note: Async tests require pytest-asyncio to be installed and configured.
    These tests verify the sync wrapper behavior instead.
    """

    def test_decorator_exists(self):
        """Verify async decorator is available."""
        from errors import handle_async_tool_errors

        assert callable(handle_async_tool_errors)

    def test_decorator_creates_wrapper(self):
        """Verify decorator wraps async functions."""
        from errors import handle_async_tool_errors
        import asyncio

        @handle_async_tool_errors("test")
        async def my_func():
            return {"success": True}

        # Verify the wrapped function is still async
        assert asyncio.iscoroutinefunction(my_func)
        assert my_func.__name__ == "my_func"
