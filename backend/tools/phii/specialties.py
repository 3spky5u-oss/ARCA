"""
Phii Specialty Detection - Focus area identification.

Detects user's specialty based on keyword patterns loaded from the
active domain's lexicon. Vanilla ARCA (no domain pack) disables
specialty detection entirely.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Specialty(Enum):
    """Specialty areas loaded from domain lexicon.
    These enum values are the defaults for the example domain pack.
    Domain packs define their own specialties in lexicon.json."""

    DOMAIN_PRIMARY = "domain_primary"
    DOMAIN_SECONDARY = "domain_secondary"
    DOMAIN_TERTIARY = "domain_tertiary"
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass
class SpecialtyProfile:
    """User's engineering specialty profile."""

    # Keyword hit counts per specialty
    hits: Dict[str, int] = field(
        default_factory=lambda: {
            "domain_primary": 0,
            "domain_secondary": 0,
            "domain_tertiary": 0,
            "general": 0,
        }
    )

    # Primary detected specialty
    primary: Specialty = Specialty.UNKNOWN

    # Confidence (0-1)
    confidence: float = 0.0

    # Terminology preferences (term -> usage count)
    terminology: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "hits": self.hits.copy(),
            "primary": self.primary.value,
            "confidence": self.confidence,
            "terminology": self.terminology.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpecialtyProfile":
        """Create from dictionary."""
        profile = cls()
        profile.hits = data.get("hits", profile.hits)
        profile.primary = Specialty(data.get("primary", "unknown"))
        profile.confidence = data.get("confidence", 0.0)
        profile.terminology = data.get("terminology", {})
        return profile

    def get_distribution(self) -> Dict[str, float]:
        """Get normalized distribution of specialty focus."""
        total = sum(self.hits.values())
        if total == 0:
            return {k: 0.25 for k in self.hits}
        return {k: v / total for k, v in self.hits.items()}


class SpecialtyDetector:
    """Detect user's engineering specialty from messages."""

    def __init__(self):
        """Initialize detector with keyword sets from lexicon.

        If the domain lexicon has no specialties configured, detection is
        disabled (empty keyword sets).
        """
        from domain_loader import get_lexicon

        lexicon = get_lexicon()
        specialties = lexicon.get("specialties", {})

        # Map lexicon specialty keys to generic slots (up to 4 specialties)
        # Domain packs define their own key names in lexicon.json
        slot_names = ["domain_primary", "domain_secondary", "domain_tertiary", "general"]
        self._lexicon_key_map = {}  # lexicon_key -> generic slot name

        if specialties:
            spec_keys = list(specialties.keys())
            for i, key in enumerate(spec_keys[:4]):
                slot = slot_names[i]
                self._lexicon_key_map[key] = slot
            kw_lists = [
                specialties.get(k, {}).get("keywords", [])
                for k in spec_keys[:4]
            ]
        else:
            kw_lists = []

        # Pad to 4 slots
        while len(kw_lists) < 4:
            kw_lists.append([])

        self._primary_set = set(k.lower() for k in kw_lists[0])
        self._secondary_set = set(k.lower() for k in kw_lists[1])
        self._tertiary_set = set(k.lower() for k in kw_lists[2])
        self._general_set = set(k.lower() for k in kw_lists[3])

        # Load terminology variants from lexicon (empty = disabled)
        term_variants = lexicon.get("terminology_variants", {})
        if term_variants:
            # Convert list format [[pattern, display], ...] to tuple format
            self.TERMINOLOGY_VARIANTS = {
                k: [(p, d) for p, d in v] for k, v in term_variants.items()
            }
        else:
            self.TERMINOLOGY_VARIANTS = {}

    def _count_hits(self, message: str, keywords: set) -> int:
        """Count keyword hits in message."""
        message_lower = message.lower()
        count = 0
        for keyword in keywords:
            if keyword in message_lower:
                count += 1
        return count

    def _analyze_terminology(self, message: str, profile: SpecialtyProfile) -> None:
        """Analyze message for terminology preferences.

        Tracks which variant of terms the user prefers (e.g., abbreviation vs
        full name). Updates the profile's terminology dict with usage counts.

        Args:
            message: User message to analyze
            profile: Profile to update with terminology preferences
        """
        for canonical, variants in self.TERMINOLOGY_VARIANTS.items():
            for pattern, display_form in variants:
                if re.search(pattern, message, re.IGNORECASE):
                    # Increment count for this specific variant
                    profile.terminology[display_form] = profile.terminology.get(display_form, 0) + 1
                    break  # Only count once per group per message

    def _build_terminology_hint(self, profile: SpecialtyProfile) -> str:
        """Build terminology mirroring hint from profile.

        Args:
            profile: User's specialty profile with terminology preferences

        Returns:
            Hint string for system prompt, or empty string if no strong preferences
        """
        if not profile.terminology:
            return ""

        # Filter to terms with 2+ uses (indicates consistent preference)
        strong_prefs = {term: count for term, count in profile.terminology.items() if count >= 2}

        if not strong_prefs:
            return ""

        # Sort by frequency, take top 5
        top_terms = sorted(strong_prefs.items(), key=lambda x: -x[1])[:5]
        terms_list = ", ".join(f'"{term}"' for term, _ in top_terms)

        return f"TERMINOLOGY: Mirror user's vocabulary - they prefer: {terms_list}"

    def analyze(self, message: str, profile: Optional[SpecialtyProfile] = None) -> SpecialtyProfile:
        """Analyze a message and update or create a specialty profile.

        Args:
            message: User message to analyze
            profile: Existing profile to update (creates new if None)

        Returns:
            Updated SpecialtyProfile
        """
        if profile is None:
            profile = SpecialtyProfile()

        # Count hits for each specialty
        primary_hits = self._count_hits(message, self._primary_set)
        secondary_hits = self._count_hits(message, self._secondary_set)
        tertiary_hits = self._count_hits(message, self._tertiary_set)
        general_hits = self._count_hits(message, self._general_set)

        # Update cumulative hits
        profile.hits["domain_primary"] += primary_hits
        profile.hits["domain_secondary"] += secondary_hits
        profile.hits["domain_tertiary"] += tertiary_hits
        profile.hits["general"] += general_hits

        # Determine primary specialty
        total = sum(profile.hits.values())
        if total == 0:
            profile.primary = Specialty.UNKNOWN
            profile.confidence = 0.0
        else:
            max_specialty = max(profile.hits, key=profile.hits.get)
            max_hits = profile.hits[max_specialty]

            profile.primary = Specialty(max_specialty)
            profile.confidence = max_hits / total

        # Analyze terminology preferences
        self._analyze_terminology(message, profile)

        return profile

    def get_hint(self, profile: SpecialtyProfile) -> str:
        """Generate specialty hint for system prompt.

        Combines specialty detection and terminology mirroring hints.

        Args:
            profile: User's specialty profile

        Returns:
            Hint string to inject into system prompt
        """
        hints = []

        # Specialty hint
        if profile.primary != Specialty.UNKNOWN and profile.confidence >= 0.3:
            from domain_loader import get_pipeline_config
            config_descs = get_pipeline_config().get("phii_specialty_descriptions", {})
            # Map lexicon description keys to generic slots via _lexicon_key_map
            specialty_descriptions = {}
            for lexicon_key, description in config_descs.items():
                slot = self._lexicon_key_map.get(lexicon_key)
                if slot:
                    try:
                        specialty_descriptions[Specialty(slot)] = description
                    except ValueError:
                        pass

            desc = specialty_descriptions.get(profile.primary, "engineering")
            confidence_pct = int(profile.confidence * 100)

            hints.append(
                f"SPECIALTY DETECTED:\nUser focus appears to be {desc} ({confidence_pct}% confidence). "
                "Bias knowledge search and responses toward this area."
            )

        # Terminology hint
        terminology_hint = self._build_terminology_hint(profile)
        if terminology_hint:
            hints.append(terminology_hint)

        return "\n\n".join(hints)

    def get_topic_weights(self, profile: SpecialtyProfile) -> Dict[str, float]:
        """Get knowledge topic weights based on specialty.

        Returns weights for routing knowledge base queries.

        Args:
            profile: User's specialty profile

        Returns:
            Dict mapping topic names to weights (0-1)
        """
        from domain_loader import get_pipeline_config
        topic_weight_config = get_pipeline_config().get("phii_topic_weights", {})

        if topic_weight_config:
            # Domain provides topic weight formulas as {topic: {lexicon_key: multiplier, ...}}
            # Translate lexicon keys to generic slot names for distribution lookup
            dist = profile.get_distribution()
            weights = {}
            for topic, spec_map in topic_weight_config.items():
                w = 0.0
                for lexicon_key, multiplier in spec_map.items():
                    slot = self._lexicon_key_map.get(lexicon_key, lexicon_key)
                    w += dist.get(slot, 0.0) * multiplier
                weights[topic] = w
        else:
            # No topic weights configured — return empty
            return {}

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights
