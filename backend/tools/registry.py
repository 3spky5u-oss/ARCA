"""
Tool Registry - Unified tool dispatch pattern for ARCA.

Replaces the monolithic if/elif dispatch chain in chat.py with a clean,
extensible registry pattern. Each tool is a self-contained definition
that registers itself with the registry.

Core tools are always registered. Domain tools are loaded conditionally
based on the active domain pack (ARCA_DOMAIN env var).
"""

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categories for grouping and processing."""

    SIMPLE = "simple"  # Quick calculations, lookups
    RAG = "rag"  # Knowledge/document search
    ANALYSIS = "analysis"  # File analysis, report generation
    EXTERNAL = "external"  # Web search, external APIs
    DOCUMENT = "document"  # Document processing (redaction, etc.)


@dataclass
class ToolDefinition:
    """Definition of a tool for the registry."""

    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]
    executor: Callable[..., Dict]
    category: ToolCategory
    friendly_name: str = ""  # Human-readable name (e.g., "Mapperr", "Cohesionn")
    brief: str = ""  # One-line summary for tools list
    provides_citations: bool = False
    updates_session: bool = False
    triggers_analysis_result: bool = False  # For UI download buttons
    extracts_nested_result: bool = False  # Result has nested analysis_result to extract
    requires_config: Optional[str] = None  # Only register if this runtime_config flag is truthy


@dataclass
class ToolResult:
    """Standardized result from tool execution."""

    success: bool
    data: Dict[str, Any]
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Optional[float] = None
    analysis_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ToolRegistry:
    """
    Central registry for all ARCA tools.

    Usage:
        # Register a tool
        ToolRegistry.register(ToolDefinition(...))

        # Get LLM-compatible schema
        tools_schema = ToolRegistry.get_tools_schema()

        # Execute a tool
        result = await ToolRegistry.execute("tool_name", {"arg": "value"})
    """

    _tools: Dict[str, ToolDefinition] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        cls._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    @classmethod
    def get_tool(cls, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return cls._tools.get(name)

    @classmethod
    def get_tools_schema(cls) -> List[Dict[str, Any]]:
        """Generate OpenAI-compatible tools schema for function calling.

        Tools with requires_config are only included if the corresponding
        runtime_config flag is truthy.
        """
        schema = []
        for tool in cls._tools.values():
            # Skip tools gated by a config flag that is currently disabled
            if tool.requires_config:
                try:
                    from config import runtime_config
                    if not getattr(runtime_config, tool.requires_config, False):
                        continue
                except ImportError:
                    continue

            schema.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": tool.parameters,
                            "required": tool.required_params,
                        },
                    },
                }
            )
        return schema

    @classmethod
    def execute(cls, name: str, args: Dict[str, Any], **context) -> ToolResult:
        """
        Execute a tool by name with given arguments.

        Args:
            name: Tool name
            args: Tool arguments
            context: Additional context (file_id, session, etc.)

        Returns:
            ToolResult with success status, data, and optional citations
        """
        tool = cls._tools.get(name)
        if not tool:
            return ToolResult(success=False, data={}, error=f"Unknown tool: {name}")

        try:
            # Filter kwargs to only those the executor accepts
            all_kwargs = {**args, **context}
            sig = inspect.signature(tool.executor)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if has_var_kw:
                filtered = all_kwargs
            else:
                accepted = set(sig.parameters.keys())
                filtered = {k: v for k, v in all_kwargs.items() if k in accepted}
            result = tool.executor(**filtered)

            # Build standardized result
            tool_result = ToolResult(
                success=result.get("success", not result.get("error")), data=result, error=result.get("error")
            )

            # Extract citations if tool provides them
            if tool.provides_citations and result.get("citations"):
                tool_result.citations = result["citations"]
                tool_result.confidence = result.get("avg_confidence")

            # Mark analysis results for UI
            if tool.triggers_analysis_result and result.get("success"):
                tool_result.analysis_result = result

            return tool_result

        except Exception as e:
            logger.error(f"Tool {name} failed: {e}", exc_info=True)
            return ToolResult(success=False, data={}, error=str(e))

    @classmethod
    def get_all_tools(cls) -> Dict[str, ToolDefinition]:
        """Get all registered tools."""
        return cls._tools.copy()

    @classmethod
    def get_tools_by_category(cls, category: ToolCategory) -> List[ToolDefinition]:
        """Get all tools in a category."""
        return [t for t in cls._tools.values() if t.category == category]

    @classmethod
    def generate_tools_section(cls) -> str:
        """Generate TOOLS_SECTION for system prompt from registry.

        Single source of truth - tool descriptions live in registry only.
        """
        lines = ["TOOLS YOU HAVE:"]

        # Build numbered list from tools with friendly names
        tools_with_names = [t for t in cls._tools.values() if t.friendly_name and t.brief]

        for i, tool in enumerate(tools_with_names, 1):
            lines.append(f"{i}. {tool.friendly_name}: {tool.brief}")

        lines.append("")
        lines.append("Tool descriptions contain full usage guidance. Key reminders:")
        lines.append("- NEVER cite sources - the UI auto-displays citations with confidence badges")
        lines.append("- NEVER include References/Sources sections - auto-cite handles this")
        lines.append("- If unsure whether to search knowledge, SEARCH")
        lines.append("- For compliance analysis: ask for required parameters if not provided")
        lines.append("")

        return "\n".join(lines)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools (for testing)."""
        cls._tools.clear()
        cls._initialized = False

    @classmethod
    def reinitialize(cls) -> int:
        """Clear and re-register all tools (used after domain switch). Returns tool count."""
        cls._tools.clear()
        cls._initialized = False
        register_all_tools()
        return len(cls._tools)


def _get_pipeline_specialty() -> str:
    """Get domain specialty from lexicon pipeline config."""
    try:
        from domain_loader import get_pipeline_config
        return get_pipeline_config()["specialty"]
    except Exception:
        return "scientific and engineering disciplines"


def _register_core_tools() -> None:
    """Register ARCA core tools (always available regardless of domain)."""
    from routers.chat_executors import (
        execute_unit_convert,
        execute_web_search,
        execute_search_knowledge,
        execute_search_session,
        execute_redact_document,
    )

    # unit_convert — generic engineering unit conversions
    ToolRegistry.register(
        ToolDefinition(
            name="unit_convert",
            friendly_name="Unit Convert",
            brief="Convert between unit systems (metric, imperial, SI)",
            description=f"""Convert between engineering units. Comprehensive support for {_get_pipeline_specialty()}.

Categories:
- Length: m, ft, in, mm, cm, km, mi
- Area: m², ft², acres, ha
- Volume: m³, L, gal, ft³
- Pressure/Stress: kPa, psi, psf, ksf, tsf, MPa, ksi, bar
- Unit Weight: kN/m³, pcf, kg/m³, g/cm³
- Permeability: m/s, cm/s, ft/day, ft/min
- Force: kN, kip, lbf, MN, ton
- Moment: kN-m, kip-ft, lb-ft
- Thermal: W/m-K, BTU/hr-ft-F, °C-days, °F-days
- Flow: m³/s, cfs, L/s, gpm
- Velocity: m/s, ft/s, km/h, mph
- Temperature: °C, °F
- Mass: kg, lb, tonne, ton

Examples: "20 kN/m³ to pcf", "100 kPa to psf", "1500 C-days to F-days" """,
            parameters={
                "value": {"type": "number", "description": "Value to convert"},
                "from_unit": {"type": "string", "description": "Source unit"},
                "to_unit": {"type": "string", "description": "Target unit"},
            },
            required_params=["value", "from_unit", "to_unit"],
            executor=execute_unit_convert,
            category=ToolCategory.SIMPLE,
        )
    )

    # search_knowledge — RAG search (Cohesionn)
    ToolRegistry.register(
        ToolDefinition(
            name="search_knowledge",
            friendly_name="Cohesionn",
            brief="Technical knowledge search - textbooks, design guides, engineering theory",
            description="""Search technical knowledge bases for information.

DO NOT answer technical questions from memory - ALWAYS search first.
If unsure whether to search, SEARCH.""",
            parameters={
                "query": {"type": "string", "description": "Technical question or topic to search for"},
            },
            required_params=["query"],
            executor=execute_search_knowledge,
            category=ToolCategory.RAG,
            provides_citations=True,
        )
    )

    # search_session — session document search
    ToolRegistry.register(
        ToolDefinition(
            name="search_session",
            friendly_name="Session Docs",
            brief="Search uploaded documents (RFPs, specs, reports, PDFs, Word docs)",
            description="""Search documents the user has uploaded in this session (RFPs, specs, reports, PDFs, Word docs).""",
            parameters={
                "query": {"type": "string", "description": "What to search for"},
            },
            required_params=["query"],
            executor=execute_search_session,
            category=ToolCategory.RAG,
            provides_citations=True,
        )
    )

    # web_search — web search
    ToolRegistry.register(
        ToolDefinition(
            name="web_search",
            friendly_name="Web Search",
            brief="Current regulations, news, technical info from the web",
            description="""Search the web for current information, news, regulations.

Use for anything not in the knowledge base - current events, recent publications, live data.""",
            parameters={
                "query": {"type": "string", "description": "Search query"},
            },
            required_params=["query"],
            executor=execute_web_search,
            category=ToolCategory.EXTERNAL,
            provides_citations=True,
        )
    )

    # web_fallback_search — CRAG web fallback (only when crag_enabled)
    ToolRegistry.register(
        ToolDefinition(
            name="web_fallback_search",
            friendly_name="Web Fallback",
            brief="Search the web when knowledge base results are insufficient",
            description="Search the web when knowledge base results are insufficient. Use when retrieval confidence is low or the query is about very recent information not in the corpus.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The search query to send to the web",
                },
            },
            required_params=["query"],
            executor=execute_web_search,
            category=ToolCategory.EXTERNAL,
            provides_citations=True,
            requires_config="crag_enabled",
        )
    )

    # redact_document — PII redaction (Redactrr)
    ToolRegistry.register(
        ToolDefinition(
            name="redact_document",
            friendly_name="Redactrr",
            brief="Remove PII from documents before external sharing",
            description="Remove personally identifiable information (PII) from a document.",
            parameters={
                "file_id": {"type": "string", "description": "File ID (optional - uses most recent)"},
            },
            required_params=[],
            executor=execute_redact_document,
            category=ToolCategory.DOCUMENT,
            triggers_analysis_result=True,
        )
    )

    logger.info(f"Registered {len(ToolRegistry._tools)} core tools")


def register_all_tools() -> None:
    """
    Register all tools with the registry.

    Core tools are always registered. Domain tools are loaded dynamically
    from the active domain pack's register_tools module.
    """
    if ToolRegistry._initialized:
        return

    # 1. Always register ARCA core tools
    _register_core_tools()

    # 2. Register domain-specific tools via domain pack module
    import importlib
    from domain_loader import get_domain_config

    domain = get_domain_config()
    if domain.tools:
        try:
            mod = importlib.import_module(f"domains.{domain.name}.register_tools")
            before = len(ToolRegistry._tools)
            mod.register_domain_tools()
            added = len(ToolRegistry._tools) - before
            logger.info(f"Domain '{domain.name}': {added} tools registered")
        except ImportError as e:
            logger.warning(f"No tool module for domain '{domain.name}': {e}")

    # 3. Register optional admin-generated custom tools (if present)
    try:
        custom_mod = importlib.import_module(f"domains.{domain.name}.custom_tools")
        if hasattr(custom_mod, "register_custom_tools"):
            before = len(ToolRegistry._tools)
            custom_mod.register_custom_tools()
            added = len(ToolRegistry._tools) - before
            if added:
                logger.info(f"Domain '{domain.name}': {added} custom tools registered")
    except ImportError:
        # Custom tools are optional; missing package is expected for most domains.
        pass
    except Exception as e:
        logger.warning(f"Custom tools load failed for domain '{domain.name}': {e}")

    ToolRegistry._initialized = True
    logger.info(f"Total tools: {len(ToolRegistry._tools)}")
