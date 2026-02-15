"""
Phii Energy Detection - User style analysis for response adaptation.

Analyzes user messages to detect:
- Brevity: terse vs verbose
- Formality: casual vs formal
- Urgency: relaxed vs time-pressured
- Technical depth: casual vs technical
- Explicit preferences: verbosity, format, units, reasoning
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


# Preference signal patterns: (regex, preference_key, value)
PREFERENCE_SIGNALS: List[Tuple[str, str, any]] = [
    # Length preferences
    (r"\b(too long|too detailed|tl;?dr|be concise|shorter)\b", "verbosity", -0.3),
    (r"\b(too short|more detail|elaborate|be thorough|longer)\b", "verbosity", 0.3),
    # Format preferences
    (r"\b(use bullets|bullet points|make a list|bulleted)\b", "format_preference", "bullets"),
    (r"\b(no bullets|in prose|paragraph form|no lists)\b", "format_preference", "prose"),
    (r"\b(use a table|in table format|tabular)\b", "format_preference", "table"),
    # Unit preferences
    (r"\b(in metric|use metric|metres|meters|kPa\b|kN\b)", "unit_system", "metric"),
    (r"\b(in imperial|use imperial|feet|psf\b|psi\b)", "unit_system", "imperial"),
    # Reasoning preferences
    (r"\b(show your work|explain.{0,10}reasoning|step by step)\b", "show_reasoning", True),
    (r"\b(just the answer|skip the explanation|bottom line)\b", "show_reasoning", False),
]


class Brevity(Enum):
    """Message brevity level."""

    TERSE = "terse"  # < 10 words
    NORMAL = "normal"  # 10-50 words
    VERBOSE = "verbose"  # > 50 words


class Formality(Enum):
    """Message formality level."""

    CASUAL = "casual"  # hey, thanks, cool
    NEUTRAL = "neutral"  # Standard
    FORMAL = "formal"  # please, would you, kindly


class Urgency(Enum):
    """Message urgency level."""

    RELAXED = "relaxed"
    NORMAL = "normal"
    URGENT = "urgent"


@dataclass
class EnergyProfile:
    """User communication style profile."""

    brevity: Brevity = Brevity.NORMAL
    formality: Formality = Formality.NEUTRAL
    urgency: Urgency = Urgency.NORMAL
    technical_depth: float = 0.5  # 0=casual, 1=highly technical

    # Running averages for stability
    brevity_samples: List[int] = field(default_factory=list)
    formality_samples: List[float] = field(default_factory=list)

    # Accumulated preferences (session-scoped)
    verbosity_preference: float = 0.0  # -1 = concise, +1 = detailed
    format_preference: Optional[str] = None  # "bullets", "prose", "table"
    unit_system: Optional[str] = None  # "metric", "imperial"
    show_reasoning: Optional[bool] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "brevity": self.brevity.value,
            "formality": self.formality.value,
            "urgency": self.urgency.value,
            "technical_depth": self.technical_depth,
            "verbosity_preference": self.verbosity_preference,
            "format_preference": self.format_preference,
            "unit_system": self.unit_system,
            "show_reasoning": self.show_reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnergyProfile":
        """Create from dictionary."""
        profile = cls(
            brevity=Brevity(data.get("brevity", "normal")),
            formality=Formality(data.get("formality", "neutral")),
            urgency=Urgency(data.get("urgency", "normal")),
            technical_depth=data.get("technical_depth", 0.5),
        )
        profile.verbosity_preference = data.get("verbosity_preference", 0.0)
        profile.format_preference = data.get("format_preference")
        profile.unit_system = data.get("unit_system")
        profile.show_reasoning = data.get("show_reasoning")
        return profile


# Core patterns (always active — generic technical writing signals)
_CORE_TECHNICAL_PATTERNS = [
    r"\d+\.?\d*\s*(kpa|psi|mpa|kn|psf|ksf|m|ft|mm|kg|g|mol|hz|ghz)",  # Values with units
    r"\b(c|phi|gamma|sigma|tau|epsilon|delta|alpha|beta|lambda|mu)\b",  # Greek letters
    r"\d+\s*[x×]\s*\d+",  # Dimensions
    r"=\s*\d+",  # Equations
    r"\b\d+(\.\d+)?%\b",  # Percentages
]

_technical_patterns_cache = None


def _get_technical_patterns():
    """Load technical patterns: core + domain-specific from lexicon."""
    global _technical_patterns_cache
    if _technical_patterns_cache is not None:
        return _technical_patterns_cache
    patterns = list(_CORE_TECHNICAL_PATTERNS)
    try:
        from domain_loader import get_pipeline_config
        pipeline = get_pipeline_config()
        domain_patterns = pipeline.get("phii_technical_patterns", [])
        patterns.extend(domain_patterns)
    except Exception:
        pass
    _technical_patterns_cache = [re.compile(p, re.IGNORECASE) for p in patterns]
    return _technical_patterns_cache


def clear_technical_patterns_cache():
    global _technical_patterns_cache
    _technical_patterns_cache = None


class EnergyDetector:
    """Detect user communication style from messages."""

    # Casual language markers
    CASUAL_PATTERNS = [
        r"\bhey\b",
        r"\bthanks\b",
        r"\bthx\b",
        r"\bcool\b",
        r"\bawesome\b",
        r"\bgreat\b",
        r"\bnice\b",
        r"\byeah\b",
        r"\byep\b",
        r"\bnope\b",
        r"\bgotcha\b",
        r"\bsure\b",
        r"\bbtw\b",
        r"\bfyi\b",
        r"\blol\b",
        r"\bhaha\b",
        r"!+",
        r"\bk\b",
        r"\bok\b",
    ]

    # Formal language markers
    FORMAL_PATTERNS = [
        r"\bplease\b",
        r"\bkindly\b",
        r"\bwould you\b",
        r"\bcould you\b",
        r"\bi would appreciate\b",
        r"\bthank you\b",
        r"\bregards\b",
        r"\brespe[c]?tfully\b",
        r"\baccordingly\b",
        r"\bfurthermore\b",
        r"\bpursuant\b",
        r"\bhereby\b",
    ]

    # Urgency markers
    URGENCY_PATTERNS = [
        r"\basap\b",
        r"\burgent\b",
        r"\bdeadline\b",
        r"\bimmediately\b",
        r"\bright away\b",
        r"\bquick\b",
        r"\bhurry\b",
        r"\beod\b",
        r"\btoday\b",
        r"\bnow\b",
        r"\bpriority\b",
        r"\bcritical\b",
        r"\btime.?sensitive\b",
        r"\brushed\b",
    ]

    def __init__(self):
        """Initialize detector with compiled patterns."""
        self._casual_re = [re.compile(p, re.IGNORECASE) for p in self.CASUAL_PATTERNS]
        self._formal_re = [re.compile(p, re.IGNORECASE) for p in self.FORMAL_PATTERNS]
        self._urgency_re = [re.compile(p, re.IGNORECASE) for p in self.URGENCY_PATTERNS]
        self._preference_re = [(re.compile(p, re.IGNORECASE), k, v) for p, k, v in PREFERENCE_SIGNALS]

    def _detect_preference_signals(self, message: str, profile: EnergyProfile) -> None:
        """Detect explicit preference signals and accumulate.

        Args:
            message: User message to scan
            profile: Profile to update with detected preferences
        """
        for pattern, pref_key, value in self._preference_re:
            if pattern.search(message):
                if pref_key == "verbosity":
                    # Accumulate with clamp to [-1, 1]
                    current = profile.verbosity_preference
                    profile.verbosity_preference = max(-1.0, min(1.0, current + value))
                elif pref_key == "format_preference":
                    profile.format_preference = value
                elif pref_key == "unit_system":
                    profile.unit_system = value
                elif pref_key == "show_reasoning":
                    profile.show_reasoning = value

    def analyze(self, message: str, profile: Optional[EnergyProfile] = None) -> EnergyProfile:
        """Analyze a message and update or create an energy profile.

        Args:
            message: User message to analyze
            profile: Existing profile to update (creates new if None)

        Returns:
            Updated EnergyProfile
        """
        if profile is None:
            profile = EnergyProfile()

        words = message.split()
        word_count = len(words)
        message_lower = message.lower()

        # Brevity detection
        profile.brevity_samples.append(word_count)
        if len(profile.brevity_samples) > 10:
            profile.brevity_samples = profile.brevity_samples[-10:]

        avg_words = sum(profile.brevity_samples) / len(profile.brevity_samples)
        if avg_words < 10:
            profile.brevity = Brevity.TERSE
        elif avg_words > 50:
            profile.brevity = Brevity.VERBOSE
        else:
            profile.brevity = Brevity.NORMAL

        # Formality detection
        casual_count = sum(1 for p in self._casual_re if p.search(message_lower))
        formal_count = sum(1 for p in self._formal_re if p.search(message_lower))

        # Normalize to -1 (casual) to +1 (formal)
        if casual_count + formal_count > 0:
            formality_score = (formal_count - casual_count) / (casual_count + formal_count)
        else:
            formality_score = 0.0

        profile.formality_samples.append(formality_score)
        if len(profile.formality_samples) > 10:
            profile.formality_samples = profile.formality_samples[-10:]

        avg_formality = sum(profile.formality_samples) / len(profile.formality_samples)
        if avg_formality < -0.3:
            profile.formality = Formality.CASUAL
        elif avg_formality > 0.3:
            profile.formality = Formality.FORMAL
        else:
            profile.formality = Formality.NEUTRAL

        # Urgency detection (instantaneous, not averaged)
        urgency_count = sum(1 for p in self._urgency_re if p.search(message_lower))
        if urgency_count >= 2:
            profile.urgency = Urgency.URGENT
        elif urgency_count == 1:
            profile.urgency = Urgency.NORMAL
        else:
            profile.urgency = Urgency.RELAXED

        # Technical depth detection
        technical_count = sum(1 for p in _get_technical_patterns() if p.search(message))
        # Normalize: 0-5+ matches -> 0-1 score
        profile.technical_depth = min(1.0, technical_count / 5.0)

        # Preference signal detection
        self._detect_preference_signals(message, profile)

        return profile

    def _build_preference_guidance(self, profile: EnergyProfile) -> List[str]:
        """Build preference hints from accumulated preferences.

        Args:
            profile: User's energy profile

        Returns:
            List of guidance strings for detected preferences
        """
        guidance = []

        if profile.verbosity_preference < -0.2:
            guidance.append("User prefers concise responses - be brief.")
        elif profile.verbosity_preference > 0.2:
            guidance.append("User prefers detailed responses - be thorough.")

        if profile.format_preference == "bullets":
            guidance.append("User prefers bullet point format.")
        elif profile.format_preference == "prose":
            guidance.append("User prefers prose/paragraph format.")
        elif profile.format_preference == "table":
            guidance.append("User prefers tabular format when applicable.")

        if profile.unit_system == "metric":
            guidance.append("User prefers metric units (m, kPa, kN).")
        elif profile.unit_system == "imperial":
            guidance.append("User prefers imperial units (ft, psf, psi).")

        if profile.show_reasoning is True:
            guidance.append("Show your reasoning - user wants to see the work.")
        elif profile.show_reasoning is False:
            guidance.append("Give direct answers - skip lengthy reasoning.")

        return guidance

    def get_guidance(self, profile: EnergyProfile) -> str:
        """Generate response guidance based on energy profile.

        Args:
            profile: User's energy profile

        Returns:
            Guidance string to inject into system prompt
        """
        guidance_parts = []

        # Brevity guidance
        if profile.brevity == Brevity.TERSE:
            guidance_parts.append("Keep response brief and direct - user prefers concise answers.")
        elif profile.brevity == Brevity.VERBOSE:
            guidance_parts.append("User provides detailed context - feel free to elaborate in response.")

        # Formality guidance
        if profile.formality == Formality.CASUAL:
            guidance_parts.append("User is casual - match their relaxed tone.")
        elif profile.formality == Formality.FORMAL:
            guidance_parts.append("User is formal - maintain professional language.")

        # Urgency guidance
        if profile.urgency == Urgency.URGENT:
            guidance_parts.append("User seems time-pressured - prioritize actionable info first.")

        # Technical depth guidance
        if profile.technical_depth > 0.7:
            guidance_parts.append("User is technically detailed - include specific values and formulas.")
        elif profile.technical_depth < 0.3:
            guidance_parts.append("User is asking generally - keep technical jargon minimal.")

        # Accumulated preference guidance
        preference_guidance = self._build_preference_guidance(profile)
        guidance_parts.extend(preference_guidance)

        if not guidance_parts:
            return ""

        return "USER STYLE:\n- " + "\n- ".join(guidance_parts)
