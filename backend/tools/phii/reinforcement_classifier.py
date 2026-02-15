"""Phii Reinforcement classifier — ActionClassifier for pattern tracking."""

import re
from typing import List, Optional

from .reinforcement_types import ACTION_CLASSIFICATIONS, MESSAGE_PATTERNS


class ActionClassifier:
    """Classify user actions for pattern tracking."""

    def __init__(self):
        """Initialize classifier with compiled patterns."""
        self._message_patterns = [(re.compile(p, re.IGNORECASE), action) for p, action in MESSAGE_PATTERNS]

    def classify(self, message: str, tools_used: List[str]) -> Optional[str]:
        """Classify an exchange into a canonical action.

        Args:
            message: User message
            tools_used: Tools invoked in response

        Returns:
            Action string or None if no classification
        """
        # Tool-based classification takes priority
        for tool in tools_used or []:
            if tool in ACTION_CLASSIFICATIONS:
                return ACTION_CLASSIFICATIONS[tool]

        # Fall back to message pattern matching
        for pattern, action in self._message_patterns:
            if pattern.search(message):
                return action

        return None
