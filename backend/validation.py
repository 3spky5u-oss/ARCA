"""
ARCA Startup Validation - Configuration and Wiring Checks

Validates critical system configuration at startup to catch wiring issues early.
Runs quickly (~350ms) and provides clear diagnostics for any problems found.

Usage:
    from validation import validate_startup
    result = validate_startup()
"""

import os
import re
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RESULT TYPES
# =============================================================================


@dataclass
class ValidationIssue:
    """A single validation issue."""

    category: str
    severity: str  # "critical" or "warning"
    message: str
    details: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of startup validation."""

    success: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    checks_performed: Dict[str, bool] = field(default_factory=dict)
    duration_ms: float = 0.0

    def add_issue(self, category: str, severity: str, message: str, details: Optional[str] = None) -> None:
        """Add a validation issue."""
        self.issues.append(ValidationIssue(category=category, severity=severity, message=message, details=details))

    def has_critical_issues(self) -> bool:
        """Check if any critical issues exist."""
        return any(i.severity == "critical" for i in self.issues)

    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues."""
        return [i for i in self.issues if i.severity == "warning"]

    def get_critical(self) -> List[ValidationIssue]:
        """Get all critical issues."""
        return [i for i in self.issues if i.severity == "critical"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "duration_ms": round(self.duration_ms, 2),
            "checks_performed": self.checks_performed,
            "critical_count": len(self.get_critical()),
            "warning_count": len(self.get_warnings()),
            "issues": [
                {"category": i.category, "severity": i.severity, "message": i.message, "details": i.details}
                for i in self.issues
            ],
        }


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_tool_registry(result: ValidationResult) -> None:
    """
    Validate tool registry consistency.

    Checks:
    - Every registered tool has a matching executor in chat_executors
    - Executor functions exist and are callable
    - Tool parameter schemas are valid JSON schema
    """
    try:
        from tools.registry import ToolRegistry, register_all_tools

        # Ensure tools are registered
        register_all_tools()

        tools = ToolRegistry.get_all_tools()
        if not tools:
            result.add_issue("tool_registry", "critical", "No tools registered in ToolRegistry")
            result.checks_performed["tool_registry"] = False
            return

        # Check each tool
        for name, tool_def in tools.items():
            # Check executor exists and is callable
            if not tool_def.executor:
                result.add_issue("tool_registry", "critical", f"Tool '{name}' has no executor defined")
            elif not callable(tool_def.executor):
                result.add_issue(
                    "tool_registry", "critical", f"Tool '{name}' executor is not callable: {type(tool_def.executor)}"
                )

            # Check parameter schema is valid
            if tool_def.parameters:
                for param_name, param_schema in tool_def.parameters.items():
                    if not isinstance(param_schema, dict):
                        result.add_issue(
                            "tool_registry",
                            "warning",
                            f"Tool '{name}' parameter '{param_name}' has invalid schema",
                            details=f"Expected dict, got {type(param_schema)}",
                        )
                    elif "type" not in param_schema and "enum" not in param_schema:
                        result.add_issue(
                            "tool_registry",
                            "warning",
                            f"Tool '{name}' parameter '{param_name}' missing 'type' in schema",
                        )

            # Check required params are defined in parameters
            for req_param in tool_def.required_params:
                if req_param not in tool_def.parameters:
                    result.add_issue(
                        "tool_registry",
                        "warning",
                        f"Tool '{name}' requires '{req_param}' but it's not in parameters",
                        details=f"Required: {tool_def.required_params}, Defined: {list(tool_def.parameters.keys())}",
                    )

        result.checks_performed["tool_registry"] = True
        logger.debug(f"Tool registry validated: {len(tools)} tools")

    except ImportError as e:
        result.add_issue("tool_registry", "critical", f"Failed to import tool registry: {e}")
        result.checks_performed["tool_registry"] = False
    except Exception as e:
        result.add_issue("tool_registry", "critical", f"Tool registry validation failed: {e}")
        result.checks_performed["tool_registry"] = False


def validate_executor_dispatch(result: ValidationResult) -> None:
    """
    Validate that execute_tool() can dispatch to all registered tools.

    Since TD-012, dispatch is registry-based via ToolRegistry.get_tool().
    This validates that:
    1. execute_tool uses registry dispatch pattern
    2. Each registered tool has a callable executor
    """
    try:
        from tools.registry import ToolRegistry, register_all_tools
        from routers.chat_executors import execute_tool

        register_all_tools()
        tools = ToolRegistry.get_all_tools()

        # Verify execute_tool uses registry-based dispatch
        import inspect

        source = inspect.getsource(execute_tool)

        # Check for registry dispatch pattern
        if "ToolRegistry.get_tool" not in source:
            result.add_issue(
                "executor_dispatch",
                "critical",
                "execute_tool() does not use ToolRegistry dispatch",
                details="Expected ToolRegistry.get_tool(tool_name) pattern",
            )

        # Verify each tool has a callable executor (already checked in registry validation,
        # but double-check the dispatch path works)
        for name, tool_def in tools.items():
            if not tool_def.executor or not callable(tool_def.executor):
                result.add_issue(
                    "executor_dispatch",
                    "critical",
                    f"Tool '{name}' has no callable executor for dispatch",
                    details="Check tools/registry.py executor registration",
                )

        result.checks_performed["executor_dispatch"] = True

    except Exception as e:
        result.add_issue("executor_dispatch", "warning", f"Executor dispatch validation failed: {e}")
        result.checks_performed["executor_dispatch"] = False


def validate_prompt_consistency(result: ValidationResult) -> None:
    """
    Validate that tools mentioned in prompts exist in registry.

    Checks:
    - Tools mentioned in TOOLS_SECTION exist
    - No orphan tools (registered but not mentioned in prompts)

    Note: Secondary tools (no friendly_name) are intentionally excluded from TOOLS_SECTION.
    """
    try:
        from tools.registry import ToolRegistry, register_all_tools
        from routers.chat_prompts import TOOLS_SECTION

        register_all_tools()
        all_tools = ToolRegistry.get_all_tools()

        # Only check primary tools (those with friendly_name set)
        # Secondary tools like list_exceedances, get_analysis_summary are helper tools
        # that don't need to be mentioned in the main prompt
        primary_tools = {name for name, defn in all_tools.items() if defn.friendly_name}

        # Tool names that should be mentioned in prompts
        # Map from tool name to expected mention patterns
        # Only check tools that are actually registered (domain tools may not be present)
        _all_tool_patterns = {
            "search_knowledge": ["knowledge", "cohesionn", "search_knowledge"],
            "search_session": ["session", "uploaded", "search_session"],
            "analyze_files": ["exceedee", "compliance", "analyze", "tier 1"],
            "web_search": ["web search", "web_search"],
            "unit_convert": ["unit", "convert", "unit_convert"],
            "solve_engineering": ["solverr", "calculation", "solve_engineering"],
            "redact_document": ["redact", "pii", "redact_document"],
            "lookup_guideline": ["guideline", "lookup_guideline"],
            "get_lab_template": ["template", "groundidd", "get_lab_template"],
            "generate_openground": ["openground", "generate_openground"],
            "process_field_logs": ["field log", "process_field_logs"],
            "extract_borehole_log": ["borehole", "extract_borehole_log", "loggview"],
            "generate_cross_section": ["cross-section", "cross section", "generate_cross_section"],
            "generate_3d_visualization": ["3d", "visualization", "generate_3d_visualization"],
            "generate_report": ["report", "generate_report"],
        }
        registered = set(all_tools.keys())
        tool_patterns = {name: patterns for name, patterns in _all_tool_patterns.items() if name in registered}

        tools_section_lower = TOOLS_SECTION.lower()

        for tool_name, patterns in tool_patterns.items():
            # Only check primary tools
            if tool_name in primary_tools:
                # Check if any pattern matches
                found = any(p.lower() in tools_section_lower for p in patterns)
                if not found:
                    result.add_issue(
                        "prompt_consistency",
                        "warning",
                        f"Registered tool '{tool_name}' not mentioned in TOOLS_SECTION",
                        details="Consider adding to chat_prompts.py TOOLS_SECTION",
                    )

        result.checks_performed["prompt_consistency"] = True

    except Exception as e:
        result.add_issue("prompt_consistency", "warning", f"Prompt consistency validation failed: {e}")
        result.checks_performed["prompt_consistency"] = False


def validate_config(result: ValidationResult) -> None:
    """
    Validate configuration settings.

    Checks:
    - Required paths exist (REPORTS_DIR, etc.)
    - Model names are reasonable (not empty)
    - Timeout values are sensible
    """
    try:
        from config import runtime_config
        from services.llm_config import SLOTS

        # Check model names are not empty
        models_to_check = [
            ("model_chat", runtime_config.model_chat),
            ("model_code", runtime_config.model_code),
            ("model_expert", runtime_config.model_expert),
        ]

        for name, value in models_to_check:
            if not value or value.strip() == "":
                result.add_issue(
                    "config",
                    "warning",
                    f"Model '{name}' is not configured",
                    details="Set via environment variable or runtime config",
                )

        # Validate optional multi-GPU tensor split settings
        try:
            from services.hardware import get_gpu_count

            gpu_count = get_gpu_count()
        except Exception:
            gpu_count = 0

        for slot_name, slot in SLOTS.items():
            if not slot.tensor_split:
                continue
            ratios = [p.strip() for p in slot.tensor_split.split(",") if p.strip()]
            if len(ratios) < 2:
                result.add_issue(
                    "config",
                    "warning",
                    f"{slot_name} tensor split has fewer than 2 entries: '{slot.tensor_split}'",
                    details="Use comma-separated GPU ratios, e.g. 1,1 or disable tensor split.",
                )
            if gpu_count and len(ratios) > gpu_count:
                result.add_issue(
                    "config",
                    "warning",
                    f"{slot_name} tensor split defines {len(ratios)} GPUs but only {gpu_count} detected",
                    details="Adjust LLM_<SLOT>_TENSOR_SPLIT to match available GPUs.",
                )
            if slot.split_mode == "none":
                result.add_issue(
                    "config",
                    "warning",
                    f"{slot_name} tensor split is set but split mode is 'none'",
                    details="Set LLM_<SLOT>_SPLIT_MODE to 'layer' or 'row'.",
                )

        # Check timeout values are sensible
        if runtime_config.llm_timeout < 30:
            result.add_issue(
                "config",
                "warning",
                f"LLM timeout ({runtime_config.llm_timeout}s) is very short",
                details="Consider increasing LLM_TIMEOUT for complex queries",
            )

        # Check RAG settings
        if runtime_config.rag_top_k < 1:
            result.add_issue("config", "warning", f"RAG_TOP_K ({runtime_config.rag_top_k}) should be at least 1")

        # Check paths
        base_dir = Path(__file__).parent
        paths_to_check = [
            ("REPORTS_DIR", base_dir / "reports"),
            ("UPLOAD_DIR", base_dir / "uploads"),
            ("GUIDELINES_DIR", base_dir / "guidelines"),
        ]

        for name, path in paths_to_check:
            # Don't fail if they don't exist - they're created at startup
            # Just check parent exists
            if not path.parent.exists():
                result.add_issue("config", "warning", f"Parent directory for {name} doesn't exist: {path.parent}")

        result.checks_performed["config"] = True

    except Exception as e:
        result.add_issue("config", "warning", f"Config validation failed: {e}")
        result.checks_performed["config"] = False


def validate_dependencies(result: ValidationResult) -> None:
    """
    Validate that required dependencies can be imported.

    Uses importlib.util.find_spec for fast checking without actually
    importing heavy modules like sentence_transformers.
    """
    import importlib.util

    # List of (module_name, severity, description)
    # Note: Uses Qdrant for vector storage (migrated Feb 2026)
    # Note: uses llama.cpp server + OpenAI SDK
    dependencies = [
        ("openai", "critical", "OpenAI SDK for llama-server communication"),
        ("qdrant_client", "warning", "Qdrant for vector storage"),
        ("fastapi", "critical", "FastAPI web framework"),
        ("openpyxl", "warning", "Excel file processing"),
        ("docx", "warning", "Word document processing"),
        ("fitz", "warning", "PDF processing (PyMuPDF)"),
        ("sentence_transformers", "warning", "Embedding models"),
    ]

    for module, severity, description in dependencies:
        # Use find_spec for fast check without import
        spec = importlib.util.find_spec(module)
        if spec is None:
            result.add_issue("dependencies", severity, f"Module '{module}' not installed", details=description)

    result.checks_performed["dependencies"] = True


def validate_llm_connection(result: ValidationResult) -> None:
    """
    Validate LLM server connection and model availability.

    Checks that llama-server instances are running and healthy.
    Also verifies that GGUF model files exist in the models directory.
    """
    try:
        from services.llm_config import SLOTS, MODELS_DIR, get_model_path

        # Check models directory exists
        if not MODELS_DIR.exists():
            result.add_issue(
                "llm",
                "warning",
                f"Models directory not found: {MODELS_DIR}",
                details="Mount GGUF models to /models in docker-compose.yml",
            )
        else:
            # Check that required GGUF files exist
            for name, slot in SLOTS.items():
                model_path = get_model_path(slot.gguf_filename)
                if not model_path.exists():
                    # Only warn for always-running slots.
                    # On-demand slots are allowed to be absent at startup.
                    if slot.always_running:
                        result.add_issue(
                            "llm",
                            "warning",
                            f"GGUF model not found for {name} slot: {slot.gguf_filename}",
                            details=f"Download to {MODELS_DIR}/",
                        )

        # Optional live server health checks.
        # Disabled by default during startup because llama-server processes are not
        # started yet and would produce noisy false warnings.
        if os.environ.get("ARCA_VALIDATE_LLM_HEALTH", "false").lower() == "true":
            try:
                import httpx

                for name, slot in SLOTS.items():
                    if not slot.always_running:
                        continue
                    try:
                        resp = httpx.get(
                            f"http://localhost:{slot.port}/health",
                            timeout=3.0,
                        )
                        if resp.status_code != 200:
                            result.add_issue(
                                "llm",
                                "warning",
                                f"LLM server for {name} slot returned HTTP {resp.status_code}",
                            )
                    except Exception:
                        result.add_issue(
                            "llm",
                            "warning",
                            f"LLM server for {name} slot not reachable on port {slot.port}",
                            details="Server may not have started yet",
                        )
            except ImportError:
                pass  # httpx not available for health check

        result.checks_performed["llm_connection"] = True

    except ImportError as e:
        result.add_issue("llm", "warning", f"LLM config not available: {e}")
        result.checks_performed["llm_connection"] = False
    except Exception as e:
        result.add_issue("llm", "warning", f"LLM connection check failed: {e}")
        result.checks_performed["llm_connection"] = False


def validate_guidelines(result: ValidationResult) -> None:
    """
    Validate that guidelines data is available.

    Checks for the pickled guidelines file needed for compliance checks.
    """
    # Guidelines are only required when guideline tools are registered.
    # Vanilla ARCA (example domain) does not ship compliance guideline tools.
    try:
        from tools.registry import ToolRegistry
        tools = ToolRegistry.get_all_tools()
        guidelines_required = "lookup_guideline" in tools
    except Exception:
        # If registry state is unavailable, keep prior behavior and validate.
        guidelines_required = True

    if not guidelines_required:
        result.checks_performed["guidelines"] = True
        return

    base_dir = Path(__file__).parent
    guidelines_path = base_dir / "guidelines" / "table1_guidelines.pkl"

    if not guidelines_path.exists():
        result.add_issue(
            "guidelines", "warning", "Guidelines file not found", details=f"Expected at: {guidelines_path}"
        )
    else:
        # Quick validation that it's readable
        # SECURITY: Guidelines pickle is trusted internal data from official
        # Alberta government PDFs, not user-supplied data.
        try:
            import pickle

            with open(guidelines_path, "rb") as f:
                data = pickle.load(f)
            if not data:
                result.add_issue("guidelines", "warning", "Guidelines file is empty")
        except Exception as e:
            result.add_issue("guidelines", "warning", f"Guidelines file unreadable: {e}")

    result.checks_performed["guidelines"] = True


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================


def validate_startup(skip_llm: bool = False) -> Dict[str, Any]:
    """
    Run all startup validation checks.

    Args:
        skip_llm: If True, skip LLM connection check (for faster tests)

    Returns:
        Dict with validation results summary

    Raises:
        RuntimeError: If any critical issues are found
    """
    start_time = time.perf_counter()
    result = ValidationResult(success=True)

    logger.info("Running startup validation...")

    # Run all validation checks
    validate_tool_registry(result)
    validate_executor_dispatch(result)
    validate_prompt_consistency(result)
    validate_config(result)
    validate_dependencies(result)
    validate_guidelines(result)

    if not skip_llm:
        validate_llm_connection(result)

    # Calculate duration
    result.duration_ms = (time.perf_counter() - start_time) * 1000

    # Log warnings
    warnings = result.get_warnings()
    if warnings:
        for w in warnings:
            logger.warning(f"Validation warning [{w.category}]: {w.message}")
            if w.details:
                logger.warning(f"  Details: {w.details}")

    # Check for critical issues
    critical = result.get_critical()
    if critical:
        result.success = False
        error_msgs = []
        for c in critical:
            error_msgs.append(f"[{c.category}] {c.message}")
            logger.error(f"Validation CRITICAL [{c.category}]: {c.message}")
            if c.details:
                logger.error(f"  Details: {c.details}")

        raise RuntimeError(
            f"Startup validation failed with {len(critical)} critical issue(s):\n"
            + "\n".join(f"  - {m}" for m in error_msgs)
        )

    # Log success
    checks_passed = sum(1 for v in result.checks_performed.values() if v)
    total_checks = len(result.checks_performed)
    logger.info(
        f"Startup validation complete: {checks_passed}/{total_checks} checks passed, "
        f"{len(warnings)} warnings in {result.duration_ms:.1f}ms"
    )

    return result.to_dict()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_validate() -> bool:
    """
    Quick validation check - returns True if no critical issues.

    Useful for health checks that need to be fast.
    """
    try:
        validate_startup(skip_llm=True)
        return True
    except RuntimeError:
        return False
