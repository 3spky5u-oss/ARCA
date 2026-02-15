"""
Base Handler - Abstract base class for query handlers.

Each handler knows how to:
1. Detect if it should handle a query (should_handle)
2. Pre-process the query (execute tools, modify context)
3. Configure the LLM call (model, context size, options)
4. Post-process the response
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..tool_router import ToolRoutingDecision


@dataclass
class LLMConfig:
    """Configuration for an LLM call."""

    model: Optional[str] = None  # None = use default
    context_size: Optional[int] = None  # None = auto-select
    temperature: Optional[float] = None
    tools_enabled: bool = True  # Whether to pass tools schema
    timeout: Optional[int] = None  # Override timeout


@dataclass
class HandlerContext:
    """
    Context passed through the handler pipeline.

    Accumulates state as handlers pre-process the query.
    """

    # Input
    message: str
    session_id: str
    router_decision: Optional[ToolRoutingDecision] = None

    # Mode flags (from client or auto-detected)
    search_mode: bool = False
    deep_search: bool = False
    think_mode: bool = False
    calculate_mode: bool = False
    phii_enabled: bool = True

    # Auto-detection flags
    auto_think: bool = False
    auto_calculate: bool = False

    # Handler state (populated by handlers)
    handler_name: str = ""
    forced_tool: Optional[str] = None
    forced_tool_result: Optional[Dict[str, Any]] = None
    forced_tool_args: Dict[str, Any] = field(default_factory=dict)

    # Context injection
    injected_context: str = ""
    mode_hints: str = ""

    # Citations and analysis
    citations: List[Dict[str, Any]] = field(default_factory=list)
    analysis_result: Optional[Dict[str, Any]] = None
    tools_used: List[str] = field(default_factory=list)

    # Vision mode
    pending_images: List[Dict[str, Any]] = field(default_factory=list)
    vision_mode: bool = False

    # LLM config override
    llm_config: Optional[LLMConfig] = None

    def add_citation(self, citation: Dict[str, Any]) -> None:
        """Add a citation to the context."""
        self.citations.append(citation)

    def add_tool_used(self, tool_name: str) -> None:
        """Record a tool that was used."""
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)

    def inject_context(self, context: str) -> None:
        """Add context to be injected into system prompt."""
        if context:
            self.injected_context += "\n" + context


class QueryHandler(ABC):
    """
    Abstract base class for query handlers.

    Handlers are checked in priority order (lowest first).
    First handler where should_handle() returns True wins.
    """

    # Lower = higher priority. Default handler has priority 1000.
    priority: int = 100
    name: str = "base"

    @abstractmethod
    def should_handle(self, ctx: HandlerContext) -> bool:
        """
        Check if this handler should process the query.

        Can use:
        - ctx.router_decision (if router selected a tool)
        - ctx.message (pattern matching fallback)
        - ctx.* (any other context)

        Returns:
            True if this handler should process the query
        """
        pass

    async def pre_process(self, ctx: HandlerContext) -> None:
        """
        Pre-process the query before LLM call.

        This is where handlers:
        - Execute forced tools
        - Inject context into system prompt
        - Set up analysis_result for UI

        Modifies ctx in-place.
        """
        pass

    def get_llm_config(self, ctx: HandlerContext) -> LLMConfig:
        """
        Get LLM configuration for this handler.

        Override to customize model, context size, etc.
        """
        return LLMConfig()

    def build_mode_hints(self, ctx: HandlerContext) -> str:
        """
        Build mode hints to add to system prompt.

        Override to add handler-specific instructions.
        """
        return ""

    def post_process(self, ctx: HandlerContext, response: Dict[str, Any]) -> None:
        """
        Post-process after LLM response.

        Can modify ctx.analysis_result, ctx.citations, etc.
        """
        pass
