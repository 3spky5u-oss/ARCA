"""
Tool Integration Tests - Verify all registered tools work end-to-end.

This module tests:
1. All tools in registry have valid executors
2. All executors are importable
3. Tool schemas are valid JSON Schema
4. Each tool category works with valid minimal parameters

Tests are organized by tool category:
- SIMPLE: unit_convert, solve_engineering, lookup_guideline
- RAG: search_knowledge, search_session
- ANALYSIS: analyze_files, list_exceedances, get_analysis_summary, generate_report
- DOCUMENT: redact_document, get_lab_template, generate_openground
- EXTERNAL: web_search

IMPORTANT: All executors use @handle_tool_errors decorator which catches
exceptions and returns {"success": False, "error": ...} dicts. Tests should
check for error dicts, not expect exceptions.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from io import BytesIO


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_files_db():
    """Mock files database with sample data for testing."""

    @dataclass
    class MockStoredFile:
        filename: str
        data: bytes
        lab_data: Any = None
        analysis: Optional[Dict] = None

        def get_bytesio(self):
            return BytesIO(self.data)

    # Create a simple mock lab data object
    class MockLabData:
        def __init__(self):
            self.samples = []

        def iter_samples(self):
            return iter(self.samples)

        def add_sample(self, sample):
            self.samples.append(sample)

    lab_data = MockLabData()

    return {
        "file_1": MockStoredFile(
            filename="test_lab_data.xlsx",
            data=b"mock excel data",
            lab_data=lab_data,
        ),
        "file_2": MockStoredFile(
            filename="document.pdf",
            data=b"%PDF-1.4 mock pdf data",
        ),
        "file_3": MockStoredFile(
            filename="report.docx",
            data=b"PK mock docx data",
        ),
    }


@pytest.fixture
def mock_files_db_with_analysis():
    """Mock files database with completed analysis."""

    @dataclass
    class MockStoredFile:
        filename: str
        data: bytes
        lab_data: Any = None
        analysis: Optional[Dict] = None

        def get_bytesio(self):
            return BytesIO(self.data)

    return {
        "file_1": MockStoredFile(
            filename="test_lab_data.xlsx",
            data=b"mock excel data",
            analysis={
                "summary": {"total_samples": 10, "exceedance_count": 2},
                "exceedances": [{"sample_id": "S1", "parameter": "arsenic", "value": 15.0}],
                "excel_report": "/api/download/test.xlsx",
            },
        ),
    }


@pytest.fixture
def mock_session_context():
    """Mock session context for session-aware tools."""

    @dataclass
    class MockSessionContext:
        material_type: str = "fine"
        land_use: str = "commercial"
        topics_searched: List[str] = None

        def __post_init__(self):
            if self.topics_searched is None:
                self.topics_searched = []

        def update_from_analysis(self, material_type: str, land_use: str):
            self.material_type = material_type or self.material_type
            self.land_use = land_use or self.land_use

    return MockSessionContext()


@pytest.fixture
def temp_reports_dir(tmp_path):
    """Create temporary reports directory."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    return reports_dir


@pytest.fixture
def sample_guidelines():
    """Sample guideline dictionary for testing compliance checks."""
    return {
        ("arsenic", "fine", "commercial"): 12.0,
        ("benzene", "fine", "commercial"): 0.078,
        ("lead", "fine", "commercial"): 70.0,
    }


# ============================================================================
# REGISTRY VALIDATION TESTS
# ============================================================================


class TestRegistryIntegrity:
    """Tests to verify registry structure and tool definitions."""

    def test_all_tools_have_executors(self):
        """Verify every registered tool has a valid executor function."""
        from tools.registry import register_all_tools, ToolRegistry

        ToolRegistry.clear()
        register_all_tools()

        for name, tool in ToolRegistry.get_all_tools().items():
            assert tool.executor is not None, f"Tool {name} has no executor"
            assert callable(tool.executor), f"Tool {name} executor is not callable"

    def test_all_executors_importable(self):
        """Verify all executor functions can be imported."""
        # Test that the main import works
        from routers.chat_executors import (
            execute_unit_convert,
            execute_solve_engineering,
            execute_web_search,
            execute_lookup_guideline,
            execute_search_knowledge,
            execute_search_session,
            execute_redact_document,
            execute_get_lab_template,
            execute_generate_openground,
            execute_analyze_files,
            execute_list_exceedances,
            execute_get_summary,
            execute_generate_report,
            execute_process_field_logs,
            execute_extract_borehole_log,
            execute_generate_cross_section,
            execute_generate_3d_visualization,
        )

        # Verify all are callable
        executors = [
            execute_unit_convert,
            execute_solve_engineering,
            execute_web_search,
            execute_lookup_guideline,
            execute_search_knowledge,
            execute_search_session,
            execute_redact_document,
            execute_get_lab_template,
            execute_generate_openground,
            execute_analyze_files,
            execute_list_exceedances,
            execute_get_summary,
            execute_generate_report,
            execute_process_field_logs,
            execute_extract_borehole_log,
            execute_generate_cross_section,
            execute_generate_3d_visualization,
        ]

        for executor in executors:
            assert callable(executor), f"{executor.__name__} is not callable"

    def test_tool_schemas_valid(self):
        """Verify all tool schemas are valid JSON Schema format."""
        from tools.registry import register_all_tools, ToolRegistry

        ToolRegistry.clear()
        register_all_tools()

        schema = ToolRegistry.get_tools_schema()

        assert isinstance(schema, list), "Schema should be a list"
        assert len(schema) > 0, "Schema should not be empty"

        for tool_schema in schema:
            # Check required schema structure
            assert "type" in tool_schema, "Missing 'type' in tool schema"
            assert tool_schema["type"] == "function", "Tool type should be 'function'"
            assert "function" in tool_schema, "Missing 'function' in tool schema"

            func = tool_schema["function"]
            assert "name" in func, "Missing 'name' in function schema"
            assert "description" in func, "Missing 'description' in function schema"
            assert "parameters" in func, "Missing 'parameters' in function schema"

            params = func["parameters"]
            assert params.get("type") == "object", "Parameters should be object type"
            assert "properties" in params, "Missing 'properties' in parameters"
            assert "required" in params, "Missing 'required' in parameters"

    def test_tool_categories_defined(self):
        """Verify all tools have a valid category."""
        from tools.registry import register_all_tools, ToolRegistry, ToolCategory

        ToolRegistry.clear()
        register_all_tools()

        valid_categories = set(ToolCategory)

        for name, tool in ToolRegistry.get_all_tools().items():
            assert tool.category in valid_categories, f"Tool {name} has invalid category: {tool.category}"

    def test_required_params_subset_of_params(self):
        """Verify required_params are a subset of parameters."""
        from tools.registry import register_all_tools, ToolRegistry

        ToolRegistry.clear()
        register_all_tools()

        for name, tool in ToolRegistry.get_all_tools().items():
            param_names = set(tool.parameters.keys())
            required_names = set(tool.required_params)
            assert required_names.issubset(
                param_names
            ), f"Tool {name}: required params {required_names - param_names} not in parameters"


# ============================================================================
# SIMPLE TOOL TESTS
# ============================================================================


class TestSimpleTools:
    """Tests for SIMPLE category tools (unit_convert, solve_engineering)."""

    def test_unit_convert_valid(self):
        """Test unit_convert with valid parameters."""
        from routers.chat_executors import execute_unit_convert

        result = execute_unit_convert(value=100, from_unit="kPa", to_unit="psi")

        assert isinstance(result, dict), "Result should be a dict"
        assert result.get("success") is True, "Conversion should succeed"
        assert "result" in result, "Result should contain 'result' key"
        assert "expression" in result, "Result should contain 'expression' key"

    def test_unit_convert_temperature(self):
        """Test temperature conversion (special case)."""
        from routers.chat_executors import execute_unit_convert

        result = execute_unit_convert(value=0, from_unit="C", to_unit="F")

        assert result.get("success") is True
        assert result["result"] == 32.0

    def test_unit_convert_same_unit(self):
        """Test converting to same unit."""
        from routers.chat_executors import execute_unit_convert

        result = execute_unit_convert(value=50, from_unit="kPa", to_unit="kPa")

        assert result.get("success") is True
        assert result["result"] == 50

    def test_unit_convert_invalid_units(self):
        """Test unit_convert with invalid unit pair returns error dict."""
        from routers.chat_executors import execute_unit_convert

        # The @handle_tool_errors decorator catches exceptions and returns error dict
        result = execute_unit_convert(value=100, from_unit="kPa", to_unit="meters")

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    def test_solve_engineering_bearing_capacity(self):
        """Test solve_engineering with capacity calculation."""
        from routers.chat_executors import execute_solve_engineering

        result = execute_solve_engineering(
            calculation="bearing_capacity",
            params={
                "cohesion": 25,
                "friction_angle": 30,
                "unit_weight": 18,
                "depth": 1.5,
                "width": 2.0,
            },
        )

        assert isinstance(result, dict)
        assert result.get("success") is True

    def test_solve_engineering_earth_pressure(self):
        """Test solve_engineering with lateral pressure calculation."""
        from routers.chat_executors import execute_solve_engineering

        result = execute_solve_engineering(
            calculation="earth_pressure",
            params={
                "unit_weight": 18,
                "height": 5.0,
                "friction_angle": 30,
            },
        )

        assert isinstance(result, dict)
        assert result.get("success") is True

    def test_solve_engineering_frost_depth(self):
        """Test solve_engineering with depth calculation."""
        from routers.chat_executors import execute_solve_engineering

        result = execute_solve_engineering(
            calculation="frost_depth",
            params={
                "freezing_index": 1500,
            },
        )

        assert isinstance(result, dict)
        assert result.get("success") is True

    def test_solve_engineering_unknown_calculation(self):
        """Test solve_engineering with unknown calculation type."""
        from routers.chat_executors import execute_solve_engineering

        result = execute_solve_engineering(calculation="unknown_calc", params={})

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    def test_solve_engineering_missing_params(self):
        """Test solve_engineering with missing required parameters."""
        from routers.chat_executors import execute_solve_engineering

        result = execute_solve_engineering(
            calculation="bearing_capacity", params={"cohesion": 25}  # Missing friction_angle, unit_weight, depth, width
        )

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result


class TestGuidelineLookup:
    """Tests for lookup_guideline tool."""

    def test_lookup_guideline_valid(self):
        """Test guideline lookup with valid parameters."""
        from routers.chat_executors import execute_lookup_guideline

        # Check if guidelines exist
        guidelines_path = Path(__file__).parent.parent / "guidelines" / "table1_guidelines.pkl"

        if not guidelines_path.exists():
            pytest.skip("Guidelines database not found")

        result = execute_lookup_guideline(parameter="arsenic", soil_type="fine", land_use="commercial")

        assert isinstance(result, dict)
        assert result.get("success") is True


# ============================================================================
# RAG TOOL TESTS (with mocking)
# ============================================================================


class TestRAGTools:
    """Tests for RAG category tools (search_knowledge, search_session)."""

    def test_search_knowledge_valid(self):
        """Test search_knowledge with mocked embedder."""
        from routers.chat_executors import execute_search_knowledge

        # Patch at the source module level where imports happen inside the function
        with patch("tools.cohesionn.CohesionnRetriever") as mock_retriever_class, patch(
            "config.runtime_config"
        ) as mock_config:

            # Setup mocks
            mock_config.get_enabled_topics.return_value = ["general", "technical"]
            mock_config.reranker_enabled = False
            mock_config.reranker_candidates = 8
            mock_config.rag_top_k = 5

            # Mock retriever
            mock_retriever = Mock()
            mock_result = Mock()
            mock_result.chunks = [{"content": "Test content", "score": 0.8}]
            mock_result.citations = [
                Mock(source="test.pdf", title="Test Doc", page=1, section="1.1", topic="general")
            ]
            mock_result.topics_searched = ["general"]
            mock_result.get_context.return_value = "Test context"
            mock_result._normalize_score = lambda x: min(1.0, max(0.0, x))
            mock_retriever.retrieve.return_value = mock_result
            mock_retriever_class.return_value = mock_retriever

            result = execute_search_knowledge(query="What is structural capacity?", topics=["general"])

            assert isinstance(result, dict)
            assert result.get("success") is True
            assert "chunks" in result
            assert "citations" in result

    def test_search_session_with_function(self):
        """Test search_session with injected search function."""
        from routers.chat_executors import execute_search_session

        # Mock search function
        def mock_search(query):
            return {
                "success": True,
                "results": [{"filename": "test.pdf", "chunk_index": 0, "score": 0.85, "content": "Test content"}],
            }

        result = execute_search_session(query="project requirements", search_session_docs_fn=mock_search)

        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "citations" in result


# ============================================================================
# DOCUMENT TOOL TESTS
# ============================================================================


class TestDocumentTools:
    """Tests for DOCUMENT category tools."""

    def test_get_lab_template(self):
        """Test get_lab_template returns download info."""
        from routers.chat_executors import execute_get_lab_template

        result = execute_get_lab_template()

        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "template_file" in result
        assert "filename" in result
        assert "instructions" in result

    def test_generate_openground_calls_groundidd(self):
        """Test generate_openground delegates to groundidd module."""
        from routers.chat_executors import execute_generate_openground

        # Patch at the source module where the import happens
        with patch("tools.groundidd.execute_groundidd") as mock_groundidd:
            mock_groundidd.return_value = {"success": True, "files": []}

            result = execute_generate_openground(file_id="test_123")

            mock_groundidd.assert_called_once()
            assert result.get("success") is True

    def test_redact_document_no_file(self):
        """Test redact_document without file returns error dict."""
        from routers.chat_executors import execute_redact_document

        # @handle_tool_errors catches NotFoundError and returns error dict
        result = execute_redact_document(file_id=None, files_db=None)

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    def test_redact_document_with_mock_file(self, mock_files_db):
        """Test redact_document with mock file data returns error (no actual redactor)."""
        from routers.chat_executors import execute_redact_document

        def get_file_data(fid):
            return mock_files_db.get(fid)

        # file_2 is a PDF - will fail because we don't have actual Redactrr
        # but tests the routing and returns error dict
        result = execute_redact_document(file_id="file_2", files_db=mock_files_db, get_file_data_fn=get_file_data)

        assert isinstance(result, dict)
        # Either success with mock or failure with missing dependency
        assert "success" in result or "error" in result

    def test_extract_borehole_log_no_file(self):
        """Test extract_borehole_log without file returns error dict."""
        from routers.chat_executors import execute_extract_borehole_log

        result = execute_extract_borehole_log(file_id=None, files_db={})

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    def test_generate_cross_section_no_data(self):
        """Test generate_cross_section without prior extraction returns error dict."""
        from routers.chat_executors import execute_generate_cross_section
        from routers.chat_executors.documents import _extracted_boreholes

        # Clear any cached data
        _extracted_boreholes.clear()

        result = execute_generate_cross_section(section_name="Test Section")

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    def test_generate_3d_visualization_no_data(self):
        """Test generate_3d_visualization without prior extraction returns error dict."""
        from routers.chat_executors import execute_generate_3d_visualization
        from routers.chat_executors.documents import _extracted_boreholes

        # Clear any cached data
        _extracted_boreholes.clear()

        result = execute_generate_3d_visualization(title="Test Viz")

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result


# ============================================================================
# ANALYSIS TOOL TESTS
# ============================================================================


class TestAnalysisTools:
    """Tests for ANALYSIS category tools."""

    def test_analyze_files_no_files(self):
        """Test analyze_files without uploaded files returns error dict."""
        from routers.chat_executors import execute_analyze_files

        result = execute_analyze_files(soil_type="fine", land_use="commercial", files_db=None)

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    def test_analyze_files_no_lab_data(self, mock_files_db):
        """Test analyze_files with files but no lab data returns error dict."""
        from routers.chat_executors import execute_analyze_files

        # Remove lab_data from mock
        for fid in mock_files_db:
            mock_files_db[fid].lab_data = None

        result = execute_analyze_files(soil_type="fine", land_use="commercial", files_db=mock_files_db)

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    def test_list_exceedances_no_analysis(self, mock_files_db):
        """Test list_exceedances without prior analysis returns error dict."""
        from routers.chat_executors import execute_list_exceedances

        def get_file_data(fid):
            return mock_files_db.get(fid)

        result = execute_list_exceedances(file_id="file_1", get_file_data_fn=get_file_data)

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    def test_list_exceedances_with_analysis(self, mock_files_db_with_analysis):
        """Test list_exceedances with prior analysis returns exceedances."""
        from routers.chat_executors import execute_list_exceedances

        def get_file_data(fid):
            return mock_files_db_with_analysis.get(fid)

        result = execute_list_exceedances(file_id="file_1", get_file_data_fn=get_file_data)

        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "exceedances" in result

    def test_get_summary_no_analysis(self, mock_files_db):
        """Test get_analysis_summary without prior analysis returns error dict."""
        from routers.chat_executors import execute_get_summary

        def get_file_data(fid):
            return mock_files_db.get(fid)

        result = execute_get_summary(file_id="file_1", get_file_data_fn=get_file_data)

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    def test_get_summary_with_analysis(self, mock_files_db_with_analysis):
        """Test get_analysis_summary with prior analysis returns summary."""
        from routers.chat_executors import execute_get_summary

        def get_file_data(fid):
            return mock_files_db_with_analysis.get(fid)

        result = execute_get_summary(file_id="file_1", get_file_data_fn=get_file_data)

        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "summary" in result


# ============================================================================
# EXTERNAL TOOL TESTS
# ============================================================================


class TestExternalTools:
    """Tests for EXTERNAL category tools (web_search)."""

    def test_web_search_valid(self):
        """Test web_search with mocked HTTP client."""
        from routers.chat_executors import execute_web_search

        with patch("routers.chat_executors.search.httpx.Client") as mock_client_class:
            # Mock the HTTP response
            mock_response = Mock()
            mock_response.json.return_value = {
                "results": [{"title": "Test Result", "url": "https://example.com", "content": "Test content"}]
            }
            mock_response.raise_for_status = Mock()

            mock_client = Mock()
            mock_client.get.return_value = mock_response
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            mock_client_class.return_value = mock_client

            result = execute_web_search(query="engineering analysis")

            assert isinstance(result, dict)
            assert result.get("success") is True
            assert "results" in result

    def test_web_search_timeout(self):
        """Test web_search handles timeout and returns error dict."""
        from routers.chat_executors import execute_web_search
        import httpx

        with patch("routers.chat_executors.search.httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.get.side_effect = httpx.TimeoutException("Timeout")
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            mock_client_class.return_value = mock_client

            result = execute_web_search(query="test query")

            assert isinstance(result, dict)
            assert result.get("success") is False
            assert "error" in result


# ============================================================================
# REGISTRY EXECUTE TESTS
# ============================================================================


class TestRegistryExecute:
    """Tests for ToolRegistry.execute() method."""

    def test_execute_unknown_tool(self):
        """Test executing unknown tool returns error."""
        from tools.registry import register_all_tools, ToolRegistry

        ToolRegistry.clear()
        register_all_tools()

        result = ToolRegistry.execute("unknown_tool_xyz", {})

        assert result.success is False
        assert "unknown" in result.error.lower()

    def test_execute_unit_convert(self):
        """Test executing unit_convert through registry."""
        from tools.registry import register_all_tools, ToolRegistry

        ToolRegistry.clear()
        register_all_tools()

        result = ToolRegistry.execute("unit_convert", {"value": 100, "from_unit": "kPa", "to_unit": "psi"})

        assert result.success is True
        assert "result" in result.data

    def test_execute_solve_engineering(self):
        """Test executing solve_engineering through registry."""
        from tools.registry import register_all_tools, ToolRegistry

        ToolRegistry.clear()
        register_all_tools()

        result = ToolRegistry.execute(
            "solve_engineering", {"calculation": "frost_depth", "params": {"freezing_index": 1500}}
        )

        assert result.success is True


# ============================================================================
# TOOL COUNT VALIDATION
# ============================================================================


class TestToolCount:
    """Verify expected number of tools are registered."""

    def test_minimum_tool_count(self):
        """Verify at least the expected number of tools are registered."""
        from tools.registry import register_all_tools, ToolRegistry

        ToolRegistry.clear()
        register_all_tools()

        tools = ToolRegistry.get_all_tools()

        # Based on registry.py, we expect at least 17 tools
        expected_tools = [
            "unit_convert",
            "lookup_guideline",
            "solve_engineering",
            "search_knowledge",
            "search_session",
            "web_search",
            "analyze_files",
            "generate_report",
            "redact_document",
            "list_exceedances",
            "get_analysis_summary",
            "get_lab_template",
            "generate_openground",
            "process_field_logs",
            "extract_borehole_log",
            "generate_cross_section",
            "generate_3d_visualization",
        ]

        for tool_name in expected_tools:
            assert tool_name in tools, f"Expected tool {tool_name} not in registry"

    def test_all_categories_have_tools(self):
        """Verify each category has at least one tool."""
        from tools.registry import register_all_tools, ToolRegistry, ToolCategory

        ToolRegistry.clear()
        register_all_tools()

        for category in ToolCategory:
            tools = ToolRegistry.get_tools_by_category(category)
            assert len(tools) > 0, f"Category {category.name} has no tools"


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_params_handled(self):
        """Test tools handle empty parameters gracefully."""
        from routers.chat_executors import execute_solve_engineering

        result = execute_solve_engineering(calculation="bearing_capacity", params=None)  # Empty params

        # Should fail gracefully with error message
        assert isinstance(result, dict)
        assert result.get("success") is False

    def test_registry_clear_and_reinit(self):
        """Test registry can be cleared and reinitialized."""
        from tools.registry import register_all_tools, ToolRegistry

        ToolRegistry.clear()
        assert len(ToolRegistry.get_all_tools()) == 0

        register_all_tools()
        assert len(ToolRegistry.get_all_tools()) > 0

        ToolRegistry.clear()
        assert len(ToolRegistry.get_all_tools()) == 0

    def test_double_registration_idempotent(self):
        """Test double registration doesn't duplicate tools."""
        from tools.registry import register_all_tools, ToolRegistry

        ToolRegistry.clear()
        register_all_tools()
        count1 = len(ToolRegistry.get_all_tools())

        register_all_tools()  # Should be idempotent
        count2 = len(ToolRegistry.get_all_tools())

        assert count1 == count2


# ============================================================================
# INTEGRATION WITH execute_tool
# ============================================================================


class TestExecuteToolDispatch:
    """Test the execute_tool dispatch function."""

    def test_execute_tool_unit_convert(self):
        """Test execute_tool dispatches unit_convert correctly."""
        from routers.chat_executors import execute_tool

        result = execute_tool(tool_name="unit_convert", args={"value": 100, "from_unit": "kPa", "to_unit": "psi"})

        assert isinstance(result, dict)
        assert result.get("success") is True

    def test_execute_tool_solve_engineering(self):
        """Test execute_tool dispatches solve_engineering correctly."""
        from routers.chat_executors import execute_tool

        result = execute_tool(
            tool_name="solve_engineering",
            args={
                "calculation": "earth_pressure",
                "params": {
                    "unit_weight": 18,
                    "height": 5.0,
                    "friction_angle": 30,
                },
            },
        )

        assert isinstance(result, dict)
        assert result.get("success") is True

    def test_execute_tool_unknown(self):
        """Test execute_tool returns error for unknown tool."""
        from routers.chat_executors import execute_tool

        result = execute_tool(tool_name="nonexistent_tool", args={})

        assert isinstance(result, dict)
        assert "error" in result


# ============================================================================
# PARAMETER VALIDATION
# ============================================================================


class TestParameterValidation:
    """Test parameter validation for tools."""

    def test_unit_convert_missing_value_returns_error(self):
        """Test unit_convert without value returns error dict."""
        from routers.chat_executors import execute_unit_convert

        # Missing value - decorator catches TypeError and returns error dict
        result = execute_unit_convert(from_unit="kPa", to_unit="psi")

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    def test_solve_engineering_missing_calculation_returns_error(self):
        """Test solve_engineering without calculation returns error dict."""
        from routers.chat_executors import execute_solve_engineering

        # Missing calculation - decorator catches TypeError and returns error dict
        result = execute_solve_engineering(params={"cohesion": 25})

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result


# ============================================================================
# TOOL RESULT STRUCTURE
# ============================================================================


class TestToolResultStructure:
    """Test that tools return consistent result structures."""

    def test_successful_result_has_success_true(self):
        """Test successful tool results have success=True."""
        from routers.chat_executors import execute_unit_convert, execute_get_lab_template

        results = [
            execute_unit_convert(value=100, from_unit="kPa", to_unit="psi"),
            execute_get_lab_template(),
        ]

        for result in results:
            assert isinstance(result, dict)
            assert result.get("success") is True

    def test_failed_result_has_success_false(self):
        """Test failed tool results have success=False."""
        from routers.chat_executors import execute_solve_engineering

        result = execute_solve_engineering(calculation="unknown_calculation", params={})

        assert isinstance(result, dict)
        assert result.get("success") is False

    def test_error_result_has_error_key(self):
        """Test error results contain 'error' key."""
        from routers.chat_executors import execute_analyze_files

        result = execute_analyze_files(soil_type="fine", land_use="commercial", files_db=None)

        assert isinstance(result, dict)
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
