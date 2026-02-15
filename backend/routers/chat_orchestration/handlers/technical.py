"""
Technical Handler - Auto-search for engineering/technical questions.

This handler automatically searches the knowledge base for technical
questions, injecting context before the LLM responds.
"""

import logging

from .base import QueryHandler, HandlerContext, LLMConfig

logger = logging.getLogger(__name__)


class TechnicalHandler(QueryHandler):
    """
    Handler for technical engineering questions.

    Triggers auto-search via Cohesionn before LLM response.

    Detection:
    1. Primary: ToolRouter selected search_knowledge
    2. Fallback: is_technical_question() pattern matching
    """

    priority = 70  # After geology, vision, code, calculate, think
    name = "technical"

    def should_handle(self, ctx: HandlerContext) -> bool:
        """
        Check if this is a technical question.

        Skips if search_mode is already active (user explicitly requested search).
        """
        # Don't double-search if search mode already active
        if ctx.search_mode:
            return False

        # Primary: Router selected search_knowledge
        if ctx.router_decision and ctx.router_decision.tool == "search_knowledge":
            logger.info("TechnicalHandler: Router selected search_knowledge")
            return True

        # Fallback: Pattern matching via is_technical_question
        from ..auto_search import AutoSearchManager

        manager = AutoSearchManager()
        if manager.should_auto_search(ctx.message, ctx.search_mode):
            logger.info("TechnicalHandler: Pattern match for technical question")
            return True

        return False

    async def pre_process(self, ctx: HandlerContext) -> None:
        """
        Execute auto-search before LLM.

        Note: This piggybacks on existing AutoSearchManager logic.
        The actual search is done in chat.py via AutoSearchManager.
        This handler just marks the intent.
        """
        # Mark that technical handler was triggered
        # Actual search happens in chat.py via AutoSearchManager
        ctx.forced_tool = "search_knowledge"
        logger.info("TechnicalHandler: Marked for auto-search")

    def get_llm_config(self, ctx: HandlerContext) -> LLMConfig:
        """Config for technical queries - needs RAG context size."""
        return LLMConfig(
            tools_enabled=True,  # Still allow other tools
        )
