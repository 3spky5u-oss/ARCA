"""
Phii Expertise Detection - User expertise level detection and response adaptation.

Detects user expertise through:
- Base layer: technical_depth from EnergyProfile
- Explicit signals: Regex patterns for strong level indicators
- Session accumulation: Signals accumulate across conversation

Expertise Levels:
- JUNIOR: EIT, learning, needs explanation
- INTERMEDIATE: Competent engineer (default)
- SENIOR: Specialist, peer consultation style
- MANAGEMENT: Executive, decision support focus
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class Expertise(Enum):
    """User expertise level."""

    JUNIOR = "junior"  # EIT, learning
    INTERMEDIATE = "intermediate"  # Competent, default
    SENIOR = "senior"  # Specialist, peer consultation
    MANAGEMENT = "management"  # Executive, decision support


@dataclass
class ExpertiseProfile:
    """User expertise profile with detection metadata."""

    level: Expertise = Expertise.INTERMEDIATE
    confidence: float = 0.5
    signals: List[str] = field(default_factory=list)  # What triggered detection

    # Running signal counts for session stability
    junior_signals: int = 0
    senior_signals: int = 0
    management_signals: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "level": self.level.value,
            "confidence": self.confidence,
            "signals": self.signals[-5:],  # Keep last 5 signals
            "junior_signals": self.junior_signals,
            "senior_signals": self.senior_signals,
            "management_signals": self.management_signals,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExpertiseProfile":
        """Create from dictionary."""
        profile = cls(
            level=Expertise(data.get("level", "intermediate")),
            confidence=data.get("confidence", 0.5),
            signals=data.get("signals", []),
        )
        profile.junior_signals = data.get("junior_signals", 0)
        profile.senior_signals = data.get("senior_signals", 0)
        profile.management_signals = data.get("management_signals", 0)
        return profile


# Explicit signal patterns: (pattern, level, signal_description)
EXPLICIT_SIGNALS: Dict[str, List[Tuple[str, str]]] = {
    "junior": [
        (r"\b(I'?m (an? )?(EIT|intern|student|learning|new to))\b", "self-identified as learning"),
        (r"\bcan you explain\b.*\bto me\b", "asked for explanation"),
        (r"\bwhat (is|are|does) (a |an |the )?\w+\b", "definitional question"),
        (r"\bhow do (I|you|we) (calculate|determine|find)\b", "process question"),
        (r"\bi('?m| am) (not sure|confused|unsure)\b", "expressed uncertainty"),
        (r"\b(never done|first time|haven'?t done)\b", "indicated inexperience"),
    ],
    "senior": [
        (r"\b(verify|check|review) (this|my|these)\b", "verification request"),
        (r"\b(edge case|typical practice|your (take|read|opinion))\b", "seeking judgment"),
        (r"\b(assuming .{15,})\b", "stated detailed assumptions"),
        (r"\b(in my experience|typically I|usually we)\b", "referenced own experience"),
        (r"\b(sanity check|gut check|double.?check)\b", "peer review language"),
        (r"\b(code (says|requires|specifies)|per (the|CFEM|NBC))\b", "code reference"),
        (r"\b(second opinion|another perspective)\b", "colleague consultation"),
    ],
    "management": [
        (r"\b(executive |high-level )?(summary|overview)\b", "requested summary"),
        (r"\b(bottom line|net-net|upshot)\b", "bottom-line language"),
        (r"\b(risk|cost|schedule|budget) (impact|implication)\b", "business impact focus"),
        (r"\b(client|stakeholder) (meeting|presentation|call)\b", "client context"),
        (r"\bfor (the proposal|our bid|the client)\b", "business deliverable"),
        (r"\b(decision|go.?no.?go|recommendation)\b", "decision-making context"),
        (r"\bkeep it (brief|short|simple)\b", "brevity preference"),
    ],
}


# Guidance templates for each expertise level
EXPERTISE_GUIDANCE: Dict[Expertise, str] = {
    Expertise.JUNIOR: """USER LEVEL: Learning/EIT
- Explain reasoning and show calculation steps
- Define technical terms on first use
- Cite references for deeper learning
- Be encouraging, not condescending""",
    Expertise.INTERMEDIATE: """USER LEVEL: Competent Engineer
- Balance thoroughness with efficiency
- Explain non-obvious considerations
- Skip basic definitions unless asked
- Highlight key decision points""",
    Expertise.SENIOR: """USER LEVEL: Senior/Specialist
- Be concise and direct
- Skip fundamentals - they know them
- Focus on edge cases, judgment calls
- Use technical terminology freely
- Treat as peer consultation""",
    Expertise.MANAGEMENT: """USER LEVEL: Management/Executive
- Lead with the bottom line
- Summarize in 2-3 sentences first
- Highlight risks, costs, schedule impacts
- Keep technical details secondary
- Focus on decisions and recommendations""",
}


class ExpertiseDetector:
    """Detect user expertise level from messages."""

    # Minimum signal count to override base level with high confidence
    MIN_OVERRIDE_SIGNALS = 3

    def __init__(self):
        """Initialize detector with compiled patterns."""
        self._patterns: Dict[str, List[Tuple[re.Pattern, str]]] = {}

        for level, patterns in EXPLICIT_SIGNALS.items():
            self._patterns[level] = [(re.compile(p, re.IGNORECASE), desc) for p, desc in patterns]

    def analyze(
        self, message: str, technical_depth: float, existing_profile: Optional[ExpertiseProfile] = None
    ) -> ExpertiseProfile:
        """Analyze a message and update expertise profile.

        Uses hybrid detection:
        1. Base layer from technical_depth
        2. Explicit signals override with high confidence
        3. Signals accumulate across session

        Args:
            message: User message to analyze
            technical_depth: Technical depth from EnergyProfile (0-1)
            existing_profile: Existing profile to update (creates new if None)

        Returns:
            Updated ExpertiseProfile
        """
        if existing_profile is None:
            profile = ExpertiseProfile()
        else:
            profile = existing_profile

        # Detect explicit signals
        detected_signals = self._detect_signals(message)

        # Update signal counts
        for level, signal in detected_signals:
            if level == "junior":
                profile.junior_signals += 1
            elif level == "senior":
                profile.senior_signals += 1
            elif level == "management":
                profile.management_signals += 1

            # Track signal descriptions (keep last 5)
            profile.signals.append(signal)
            if len(profile.signals) > 5:
                profile.signals = profile.signals[-5:]

        # Determine level using hybrid logic
        profile.level, profile.confidence = self._determine_level(
            technical_depth,
            profile.junior_signals,
            profile.senior_signals,
            profile.management_signals,
        )

        return profile

    def _detect_signals(self, message: str) -> List[Tuple[str, str]]:
        """Detect explicit expertise signals in a message.

        Args:
            message: Message to analyze

        Returns:
            List of (level, signal_description) tuples
        """
        detected = []

        for level, patterns in self._patterns.items():
            for pattern, description in patterns:
                if pattern.search(message):
                    detected.append((level, description))
                    # Only count first match per level per message
                    break

        return detected

    def _determine_level(
        self,
        technical_depth: float,
        junior_signals: int,
        senior_signals: int,
        management_signals: int,
    ) -> Tuple[Expertise, float]:
        """Determine expertise level from accumulated signals.

        Args:
            technical_depth: Technical depth from energy profile
            junior_signals: Count of junior-level signals
            senior_signals: Count of senior-level signals
            management_signals: Count of management-level signals

        Returns:
            Tuple of (Expertise level, confidence)
        """
        # Start with base level from technical_depth
        if technical_depth < 0.05:
            base_level = Expertise.JUNIOR
            base_confidence = 0.3
        elif technical_depth > 0.5:
            base_level = Expertise.SENIOR
            base_confidence = 0.4
        else:
            base_level = Expertise.INTERMEDIATE
            base_confidence = 0.5

        # Weight senior signals 2x when both senior and junior signals are present.
        # One "can you explain?" shouldn't override 5 technical queries.
        if senior_signals > 0 and junior_signals > 0:
            senior_signals = senior_signals * 2

        # Check for explicit overrides
        max_signals = max(junior_signals, senior_signals, management_signals)

        if max_signals >= self.MIN_OVERRIDE_SIGNALS:
            # Strong explicit signal - override with high confidence
            if management_signals == max_signals:
                return Expertise.MANAGEMENT, 0.8
            elif senior_signals == max_signals:
                return Expertise.SENIOR, 0.8
            elif junior_signals == max_signals:
                return Expertise.JUNIOR, 0.8

        # Single signal can nudge confidence (but junior needs base_level == JUNIOR)
        if max_signals == 1:
            if management_signals == 1:
                return Expertise.MANAGEMENT, 0.6
            elif senior_signals == 1 and base_level != Expertise.JUNIOR:
                return Expertise.SENIOR, 0.6
            elif junior_signals == 1 and base_level == Expertise.JUNIOR:
                return Expertise.JUNIOR, 0.6

        return base_level, base_confidence

    def get_guidance(self, profile: ExpertiseProfile) -> str:
        """Get response guidance for the detected expertise level.

        Args:
            profile: User's expertise profile

        Returns:
            Guidance string to inject into system prompt
        """
        return EXPERTISE_GUIDANCE.get(profile.level, "")
