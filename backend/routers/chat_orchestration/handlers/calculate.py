"""
Calculate Handler - Mathematical calculations with qwq:32b.

This handler routes calculation requests to the qwq model which has
native mathematical reasoning capabilities.
"""

import logging
import re

from config import runtime_config
from .base import QueryHandler, HandlerContext, LLMConfig

logger = logging.getLogger(__name__)

# Load domain-specific validation examples from lexicon pipeline config
try:
    from domain_loader import get_pipeline_config
    _calc_examples = get_pipeline_config().get("calculate_validation_examples", {})
    _units_example = _calc_examples.get("units", "values in correct units for the quantity")
    _magnitude_example = _calc_examples.get("magnitude", "result in a reasonable range for this type of calculation")
    _calc_assumptions = get_pipeline_config().get("calculate_assumptions", [])
except Exception:
    _units_example = "values in correct units for the quantity"
    _magnitude_example = "result in a reasonable range for this type of calculation"
    _calc_assumptions = []

# Generic fallback assumptions when no domain-specific ones are configured
_DEFAULT_ASSUMPTIONS = [
    "Input parameters",
    "Calculation method and basis",
    "Safety factors",
]


def _build_assumptions_text() -> str:
    """Build assumption category bullet list from pipeline config or defaults."""
    categories = _calc_assumptions if _calc_assumptions else _DEFAULT_ASSUMPTIONS
    return "\n".join(f"   - **{cat}**: Basis and rationale" for cat in categories)


def has_numbers(text: str) -> bool:
    """Check if text contains actual numbers (not just number words).

    Returns True if text contains digits that look like engineering values,
    e.g. "phi=30", "1500 C-days", "gamma=18", "2.5m", "150 kPa".
    """
    # Pattern for numbers that look like values (not page numbers, list items, etc.)
    # Matches: 30, 1.5, 150, 0.3, etc. when followed/preceded by engineering context
    number_pattern = r'\b\d+\.?\d*\b'
    matches = re.findall(number_pattern, text)
    # Need at least one number that's likely a parameter value
    return len(matches) > 0

# Keywords that trigger calculate mode
CALCULATE_KEYWORDS = [
    "calculate",
    "compute",
    "derive",
    "derivation",
    "show the math",
    "show your work",
    "show calculations",
    "step by step calculation",
    "work out the numbers",
    "solve for",
    "find the value",
    "what is the value of",
    "how much is",
    "determine the",
]


class CalculateHandler(QueryHandler):
    """
    Handler for mathematical calculations.

    Uses qwq:32b which has native mathematical reasoning.
    Triggers on /calc command or calculate keywords.
    """

    priority = 40  # After geology, vision, code
    name = "calculate"

    def should_handle(self, ctx: HandlerContext) -> bool:
        """
        Check if this is a calculation request.

        Triggers on:
        1. ctx.calculate_mode already set (from /calc command)
        2. Calculate keywords in message
        3. Router selected solve_engineering (with numbers)
        """
        # Already in calculate mode
        if ctx.calculate_mode:
            return True

        # Don't auto-detect if already in think mode
        if ctx.think_mode:
            return False

        # Router selected solve_engineering - but validate it has numbers
        if ctx.router_decision and ctx.router_decision.tool == "solve_engineering":
            if has_numbers(ctx.message):
                ctx.calculate_mode = True
                ctx.auto_calculate = True
                logger.info("CalculateHandler: Router selected solve_engineering (validated: has numbers)")
                return True
            else:
                logger.info("CalculateHandler: Router said solve_engineering but no numbers found, skipping")
                return False

        # Keyword detection
        msg_lower = ctx.message.lower()
        for keyword in CALCULATE_KEYWORDS:
            if keyword in msg_lower:
                ctx.calculate_mode = True
                ctx.auto_calculate = True
                logger.info(f"CalculateHandler: Auto-triggered by keyword '{keyword}'")
                return True

        return False

    def get_llm_config(self, ctx: HandlerContext) -> LLMConfig:
        """Config for calculate mode - use qwq model."""
        return LLMConfig(
            model=runtime_config.model_expert,
            tools_enabled=True,  # qwq supports tools
            timeout=runtime_config.llm_timeout_think,
        )

    def build_mode_hints(self, ctx: HandlerContext) -> str:
        """Add calculate mode instructions to system prompt."""
        return """

CALCULATE MODE ACTIVE: The user wants detailed mathematical derivations and step-by-step calculations. Show your work:

1. **Given Values**: List all input parameters with units
2. **Equations**: Write out the governing equations you'll use
3. **Substitution**: Show each step of plugging in values
4. **Intermediate Results**: Show intermediate calculations, don't skip steps
5. **Units**: Track units through calculations, verify dimensional consistency
6. **Final Answer**: Box or highlight the final result with appropriate significant figures
7. **Verification Checklist** (run these checks before finalizing):
   - **Units**: Are final units correct for this quantity? (e.g., """ + _units_example + """)
   - **Magnitude**: Is the result in a reasonable range? (e.g., """ + _magnitude_example + """)
   - **Limiting cases**: Does result make sense at extremes?
   - **Common errors**: Unit conversions correct? Safety factors applied where required?

   If ANY check fails, state: "⚠️ VERIFICATION WARNING: [specific issue]" and explain the concern.

8. **Assumptions** (REQUIRED - list each assumption with its basis):
   Format as a bulleted list with categories:
""" + _build_assumptions_text() + """

   Example: "- **Assumption**: Using standard empirical correlation for the given conditions"

Use LaTeX formatting for equations. Be meticulous with the math - this is calculate mode."""
