"""Phii Reinforcement cue detection — implicit feedback cues in messages."""

import re
from typing import Optional

from .reinforcement_types import FeedbackType


class CueDetector:
    """Detect implicit feedback cues in messages."""

    # Positive cue patterns
    POSITIVE_PATTERNS = [
        r"\b(thanks|thank you|thx|ty)\b",
        r"\b(great|awesome|perfect|excellent|wonderful)\b",
        r"\b(helpful|useful|exactly|spot on)\b",
        r"\b(good job|well done|nice work)\b",
        r"\b(that helps|makes sense|got it)\b",
    ]

    # Negative cue patterns
    NEGATIVE_PATTERNS = [
        r"\b(wrong|incorrect|not right|mistake)\b",
        r"\b(no|nope),?\s+(that|the|it)",  # "no, that's not..."
        r"\bactually,?\s*(it|the|that|you)",  # "actually, it should be..."
        r"\b(doesn\'t|didn\'t|isn\'t|wasn\'t)\s+work",
        r"\b(not what|that\'s not)\s+",
        r"\b(error|problem|issue)\s+with",
    ]

    def __init__(self):
        """Initialize detector with compiled patterns."""
        self._positive_re = [re.compile(p, re.IGNORECASE) for p in self.POSITIVE_PATTERNS]
        self._negative_re = [re.compile(p, re.IGNORECASE) for p in self.NEGATIVE_PATTERNS]

    def detect(self, message: str) -> Optional[FeedbackType]:
        """Detect implicit feedback cue in a message.

        Args:
            message: User message to analyze

        Returns:
            FeedbackType if cue detected, None otherwise
        """
        # Check for negative cues first (more important to catch)
        for pattern in self._negative_re:
            if pattern.search(message):
                return FeedbackType.NEGATIVE

        # Check for positive cues
        for pattern in self._positive_re:
            if pattern.search(message):
                return FeedbackType.POSITIVE

        return None
