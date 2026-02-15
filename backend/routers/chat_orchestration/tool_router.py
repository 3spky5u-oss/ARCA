"""
Tool Router - Stub module (tool routing handled natively by LLM).

GLM-4.7-Flash handles tool routing natively via function calling,
so this module is a no-op. The ToolRoutingDecision dataclass and
ToolRouter class are preserved for interface compatibility.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolRoutingDecision:
    """Result from tool router."""

    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    fallback: bool = False  # True = let main LLM decide
    latency_ms: float = 0.0
    error: Optional[str] = None


class ToolRouter:
    """
    No-op tool router stub.

    Tool routing is now handled natively by the main LLM via function calling.
    This class exists for interface compatibility only.
    """

    def __init__(self, runtime_config=None):
        self.runtime_config = runtime_config
        logger.info("ToolRouter initialized (no-op — LLM handles tool routing natively)")

    async def route(
        self,
        message: str,
        session=None,
        files_db: Optional[Dict] = None,
    ) -> ToolRoutingDecision:
        """Always falls back to main LLM for tool routing."""
        return ToolRoutingDecision(fallback=True)

    async def close(self):
        """No-op cleanup."""
        pass


# Singleton instance for easy import
_router_instance: Optional[ToolRouter] = None


def get_tool_router(runtime_config=None) -> ToolRouter:
    """Get or create the singleton router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = ToolRouter(runtime_config)
    return _router_instance
