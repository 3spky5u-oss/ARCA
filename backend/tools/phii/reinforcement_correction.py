"""Phii Reinforcement correction — Correction dataclass and CorrectionDetector."""

import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Optional, Tuple


@dataclass
class Correction:
    """A learned correction from user feedback.

    Corrections capture when the user corrects ARCA's behavior,
    storing what was wrong and what the right behavior should be.
    """

    id: Optional[int] = None
    timestamp: str = ""
    session_id: str = ""
    ai_message_excerpt: str = ""  # What the AI said (truncated)
    user_correction: str = ""  # The user's correction message
    wrong_behavior: str = ""  # What was wrong
    right_behavior: str = ""  # What the right behavior is
    context_keywords: List[str] = field(default_factory=list)  # Keywords for relevance matching
    confidence: float = 0.8  # Confidence in this correction
    times_applied: int = 0  # How many times this correction was applied
    is_active: bool = True  # Whether this correction is active

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["context_keywords"] = self.context_keywords  # Already a list
        return d


class CorrectionDetector:
    """Detect and parse corrections from user messages.

    Recognizes patterns like:
    - "No, I meant X"
    - "Actually, it should be X"
    - "That's wrong, X"
    - "Don't use X, use Y instead"
    - "Always use X"
    - "Too long/short/detailed"
    """

    # Correction patterns with confidence scores
    # Each pattern extracts the correction content in group 1
    CORRECTION_PATTERNS: List[Tuple[str, float]] = [
        # Direct corrections
        (r"no[,.]?\s*(?:i\s+)?meant?\s+(.+)", 0.9),
        (r"actually[,.]?\s*(?:it'?s?|i\s+want(?:ed)?|it\s+should\s+be)\s+(.+)", 0.85),
        (r"(?:that'?s?\s+)?(?:wrong|incorrect)[,.]?\s*(.+)?", 0.9),
        # Behavioral instructions (supports both "use" and "using")
        (r"(?:don'?t|stop|never)\s+(?:us(?:e|ing)|say(?:ing)?|call(?:ing)?|assum(?:e|ing)|do(?:ing)?)\s+(.+)", 0.95),
        (r"(?:always|whenever)\s+(?:us(?:e|ing)|say(?:ing)?|assum(?:e|ing)|call(?:ing)?|do(?:ing)?)\s+(.+)", 0.9),
        # Style corrections
        (r"(?:too|way\s+too)\s+(long|short|detailed|brief|verbose|technical|simple)", 0.8),
        # Clarifications that imply correction
        (r"i\s+(?:said|asked\s+for|wanted|meant)\s+(.+?)(?:,|\s+not\s+|\s*$)", 0.85),
        # Preference declarations
        (r"(?:i\s+)?prefer\s+(.+?)(?:\s+(?:instead|over|rather)\s+|$)", 0.75),
        (r"(?:please\s+)?use\s+(.+?)\s+instead(?:\s+of\s+)?", 0.85),
    ]

    # Generic context keywords (always active)
    _GENERIC_CONTEXT_KEYWORDS = {
        # Units and formats
        "metric",
        "imperial",
        "si",
        "units",
        "meters",
        "feet",
        "inches",
        "mm",
        # Style terms
        "brief",
        "detailed",
        "summary",
        "bullets",
        "list",
        "table",
    }

    @staticmethod
    def _load_context_keywords():
        """Load context keywords combining generic + domain-specific."""
        try:
            from domain_loader import get_pipeline_config
            domain_terms = get_pipeline_config().get("phii_common_terms", [])
        except Exception:
            domain_terms = []
        return CorrectionDetector._GENERIC_CONTEXT_KEYWORDS | set(domain_terms)

    def __init__(self):
        """Initialize detector with compiled patterns."""
        self._patterns = [(re.compile(p, re.IGNORECASE), conf) for p, conf in self.CORRECTION_PATTERNS]
        self._context_keywords = self._load_context_keywords()

    def detect(self, user_message: str, ai_message: str = "") -> Optional[Correction]:
        """Detect if a user message contains a correction.

        Args:
            user_message: The user's message to analyze
            ai_message: The AI's previous message (for context)

        Returns:
            Correction object if detected, None otherwise
        """
        user_lower = user_message.lower().strip()

        # Skip very short messages
        if len(user_lower) < 5:
            return None

        for pattern, base_confidence in self._patterns:
            match = pattern.search(user_lower)
            if match:
                correction_text = match.group(1) if match.lastindex else ""

                # Extract context keywords from both messages
                context_keywords = self._extract_keywords(user_message + " " + ai_message)

                # Parse wrong/right behavior from the correction
                wrong, right = self._parse_correction(user_message, correction_text, ai_message)

                if not wrong and not right:
                    # Couldn't parse meaningful correction
                    continue

                return Correction(
                    timestamp=datetime.now().isoformat(),
                    ai_message_excerpt=ai_message[:200] if ai_message else "",
                    user_correction=user_message[:500],
                    wrong_behavior=wrong,
                    right_behavior=right,
                    context_keywords=context_keywords,
                    confidence=base_confidence,
                )

        return None

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant context keywords from text."""
        words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
        return list(words & self._context_keywords)[:10]

    def _parse_correction(self, user_message: str, correction_text: str, ai_message: str) -> Tuple[str, str]:
        """Parse correction into wrong behavior and right behavior.

        Args:
            user_message: Full user message
            correction_text: Extracted correction portion
            ai_message: AI's previous message

        Returns:
            Tuple of (wrong_behavior, right_behavior)
        """
        user_lower = user_message.lower()

        # Check for "don't X, use Y instead" or "don't X" patterns
        # Two-part pattern: "don't use X" + optional ", use Y instead"
        full_pattern = re.search(
            r"(?:don'?t|stop|never)\s+(?:us(?:e|ing)|say(?:ing)?|call(?:ing)?|assum(?:e|ing)|do(?:ing)?)\s+(.+?)"
            r"(?:,\s*(?:use|say)\s+(.+?)\s+instead|,\s*instead\s+(?:use|say)\s+(.+?)|$)",
            user_lower,
        )
        if full_pattern:
            wrong = full_pattern.group(1).strip()
            right = (full_pattern.group(2) or full_pattern.group(3) or "").strip()
            return wrong, right

        # Simpler "don't X" without "instead" clause
        dont_match = re.search(
            r"(?:don'?t|stop|never)\s+(?:us(?:e|ing)|say(?:ing)?|call(?:ing)?|assum(?:e|ing)|do(?:ing)?)\s+(.+?)(?:\s*[,.]|$)",
            user_lower,
        )
        if dont_match:
            wrong = dont_match.group(1).strip()
            return wrong, ""

        # Check for "always X" pattern -> right = X
        always_match = re.search(r"(?:always|whenever)\s+(?:use|say|assume|call|do)\s+(.+?)(?:\s*[,.]|$)", user_lower)
        if always_match:
            right = always_match.group(1).strip()
            return "", right

        # Check for style corrections
        style_match = re.search(r"(?:too|way\s+too)\s+(long|short|detailed|brief|verbose|technical|simple)", user_lower)
        if style_match:
            style = style_match.group(1)
            style_opposites = {
                "long": ("long responses", "shorter responses"),
                "short": ("short responses", "more detailed responses"),
                "detailed": ("overly detailed responses", "concise responses"),
                "brief": ("too brief responses", "more complete responses"),
                "verbose": ("verbose responses", "concise responses"),
                "technical": ("overly technical language", "simpler explanations"),
                "simple": ("oversimplified responses", "more technical detail"),
            }
            wrong, right = style_opposites.get(style, ("", ""))
            return wrong, right

        # Check for "X not Y" or "X instead of Y" pattern
        instead_match = re.search(r"(.+?)\s+(?:not|instead\s+of)\s+(.+?)(?:\s*[,.]|$)", user_lower)
        if instead_match:
            right = instead_match.group(1).strip()
            wrong = instead_match.group(2).strip()
            return wrong, right

        # Fallback: use the correction text as "right behavior"
        if correction_text:
            return "", correction_text.strip()

        # If user message is short enough, use it as the right behavior
        if len(user_message) < 100:
            return "", user_message.strip()

        return "", ""
