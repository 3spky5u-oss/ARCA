"""
Default Handler - Fallback for queries that don't match other handlers.

This is the catch-all handler with lowest priority (1000).
It passes the query to the LLM with full tools schema.
"""

from .base import QueryHandler, HandlerContext, LLMConfig


class DefaultHandler(QueryHandler):
    """
    Default handler for general queries.

    Always matches (lowest priority), passes to LLM with tools.
    """

    priority = 1000
    name = "default"

    def should_handle(self, ctx: HandlerContext) -> bool:
        """Always matches as fallback."""
        return True

    def get_llm_config(self, ctx: HandlerContext) -> LLMConfig:
        """Standard LLM config with tools enabled."""
        return LLMConfig(
            tools_enabled=True,
        )
