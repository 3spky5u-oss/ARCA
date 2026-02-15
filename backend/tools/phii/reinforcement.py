"""Phii Reinforcement — re-export barrel for backward compatibility."""

from .reinforcement_types import (
    FeedbackType,
    ActionSuggestion,
    ACTION_CLASSIFICATIONS,
    _GENERIC_MESSAGE_PATTERNS,
    _build_message_patterns,
    MESSAGE_PATTERNS,
    _get_action_suggestions,
    ACTION_SUGGESTIONS,
)
from .reinforcement_classifier import ActionClassifier
from .reinforcement_cue import CueDetector
from .reinforcement_correction import Correction, CorrectionDetector
from .reinforcement_store import FeedbackRecord, ReinforcementStore

__all__ = [
    "FeedbackType",
    "ActionSuggestion",
    "ACTION_CLASSIFICATIONS",
    "ActionClassifier",
    "CueDetector",
    "Correction",
    "CorrectionDetector",
    "FeedbackRecord",
    "ReinforcementStore",
]
