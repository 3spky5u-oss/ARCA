"""
Think Handler - Enhanced reasoning with prompt modifications.

This handler enhances qwen3's reasoning by adding detailed prompts
for thorough analysis. Unlike calculate mode (which uses qwq),
think mode uses qwen3 with enhanced prompts.
"""

import logging

from .base import QueryHandler, HandlerContext, LLMConfig

logger = logging.getLogger(__name__)

# Keywords that trigger think mode
THINK_KEYWORDS = [
    "think through",
    "think about this",
    "think carefully",
    "think deeply",
    "think hard",
    "think about",
    "reason through",
    "reason about",
    "cogitate",
    "ponder",
    "analyze deeply",
    "walk me through",
    "explain your thinking",
    "thorough analysis",
    "detailed analysis",
    "work through this",
    "break down",
]


class ThinkHandler(QueryHandler):
    """
    Handler for thorough analysis requests.

    Uses qwen3:32b with enhanced reasoning prompts.
    Triggers on /think command or think keywords.
    """

    priority = 50  # After calculate
    name = "think"

    def should_handle(self, ctx: HandlerContext) -> bool:
        """
        Check if this is a think/reasoning request.

        Triggers on:
        1. ctx.think_mode already set (from /think command)
        2. Think keywords in message
        """
        # Already in think mode
        if ctx.think_mode:
            return True

        # Don't auto-detect if already in calculate mode
        if ctx.calculate_mode:
            return False

        # Keyword detection
        msg_lower = ctx.message.lower()
        for keyword in THINK_KEYWORDS:
            if keyword in msg_lower:
                ctx.think_mode = True
                ctx.auto_think = True
                logger.info(f"ThinkHandler: Auto-triggered by keyword '{keyword}'")
                return True

        return False

    def get_llm_config(self, ctx: HandlerContext) -> LLMConfig:
        """Config for think mode - uses default model with extended context."""
        return LLMConfig(
            # Uses default model (qwen3:32b) with enhanced prompts
            tools_enabled=True,
            timeout=300,  # 5 min timeout for thinking
        )

    def build_mode_hints(self, ctx: HandlerContext) -> str:
        """Add think mode instructions to system prompt."""
        return """

THINK MODE ACTIVE: The user wants thorough, detailed analysis and reasoning. Provide comprehensive technical analysis:

1. **Problem Decomposition**: Break the problem into clear components and state any assumptions
2. **Analysis**: Work through each component methodically, showing your reasoning
3. **Relevant Theory**: Reference applicable principles or standards
4. **Trade-offs**: Consider different approaches, their pros/cons, and practical implications
5. **Edge Cases**: What could go wrong? Limiting factors? Special conditions?
6. **Practical Considerations**: Constructability, cost, common pitfalls
7. **Recommendations**: Clear conclusions with confidence levels

Format with headers and clear organization. Aim for thoroughness - this is think mode."""
