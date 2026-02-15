"""Phii Reinforcement types — enums, dataclasses, constants, and pattern helpers."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback signals."""

    FLAG = "flag"  # Explicit flag from user
    POSITIVE = "positive"  # Implicit positive (thanks, great, etc.)
    NEGATIVE = "negative"  # Implicit negative (wrong, incorrect, etc.)


@dataclass
class ActionSuggestion:
    """A suggested next action based on pattern prediction."""

    action: str
    message: str
    confidence: float


# Action classifications - maps tools to canonical action names
ACTION_CLASSIFICATIONS = {
    # Tool-based
    "analyze_files": "compliance_check",
    "search_knowledge": "knowledge_search",
    "solve_engineering": "calculation",
    "unit_convert": "unit_conversion",
    "redact_document": "redaction",
    "get_lab_template": "template_request",
    "generate_openground": "openground_export",
}

# Generic message patterns (always active)
_GENERIC_MESSAGE_PATTERNS = [
    (r"upload|attached|here'?s?\s+(?:the|a|my)", "file_upload"),
    (r"(?:\d+\s*[\+\-\*\/]\s*\d+|calculate|compute|formula)", "math_interest"),
    (r"(?:code|script|function|class|import|def\s)", "code_interest"),
    (r"\?$|^(?:what|how|why|when|where|who|is|are|can|could|would|should)\b", "question"),
]


def _build_message_patterns():
    """Build message patterns combining generic + domain-specific patterns."""
    try:
        from domain_loader import get_pipeline_config
        domain_patterns = get_pipeline_config().get("phii_message_patterns", [])
    except Exception:
        domain_patterns = []
    # Domain patterns come first (higher priority), then generic fallbacks
    return [(p, a) for p, a in domain_patterns] + list(_GENERIC_MESSAGE_PATTERNS)


MESSAGE_PATTERNS = _build_message_patterns()

_action_suggestions_cache = None


def _get_action_suggestions():
    """Get action suggestions, with domain-specific messages from lexicon."""
    global _action_suggestions_cache
    if _action_suggestions_cache is not None:
        return _action_suggestions_cache
    suggestions = {
        "knowledge_search": ActionSuggestion("calculation", "Want me to run calculations based on these values?", 0.4),
    }
    # Load domain-specific action suggestions
    try:
        from domain_loader import get_pipeline_config
        domain_suggestions = get_pipeline_config().get("phii_action_suggestions", {})
        for key, entry in domain_suggestions.items():
            suggestions[key] = ActionSuggestion(
                entry.get("action", "knowledge_search"),
                entry.get("message", ""),
                entry.get("confidence", 0.5),
            )
    except Exception:
        pass
    # Add domain-specific file_upload suggestion if configured
    try:
        from domain_loader import get_pipeline_config
        msg = get_pipeline_config().get("phii_compliance_suggestion", "")
        if msg:
            suggestions["file_upload"] = ActionSuggestion("compliance_check", msg, 0.7)
    except Exception:
        pass
    _action_suggestions_cache = suggestions
    return _action_suggestions_cache


# For backward compatibility
ACTION_SUGGESTIONS = None  # Use _get_action_suggestions() instead
