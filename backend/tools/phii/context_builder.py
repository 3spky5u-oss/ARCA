"""
Phii Context Builder - Dynamic prompt assembly.

ARCHITECTURE:
- Shared sections (tools, instructions, rules) are imported from chat_prompts.py
- Only personality differs between vanilla and Phii modes
- This ensures single source of truth for tool definitions

Combines:
- Base/Phii personality (from chat_prompts.py or personalities.py)
- Shared sections (from chat_prompts.py)
- Energy guidance (user style adaptation)
- Specialty hints (engineering focus)
- Session context (files, notes)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from .personalities import PHII_PERSONALITY
from .energy import EnergyDetector, EnergyProfile
from .specialties import SpecialtyDetector, SpecialtyProfile
from .expertise import ExpertiseDetector, ExpertiseProfile
from .reinforcement import ReinforcementStore, CueDetector, FeedbackType, Correction

logger = logging.getLogger(__name__)


@dataclass
class PhiiContext:
    """Result of context building."""

    system_prompt: str
    energy_profile: EnergyProfile
    specialty_profile: SpecialtyProfile
    expertise_profile: ExpertiseProfile = None
    phii_enabled: bool = True
    # Explainability metadata
    corrections_applied: List[str] = None  # List of correction reasons
    expertise_signals: List[str] = None  # What triggered expertise detection

    def __post_init__(self):
        """Ensure expertise_profile is initialized."""
        if self.expertise_profile is None:
            self.expertise_profile = ExpertiseProfile()
        if self.corrections_applied is None:
            self.corrections_applied = []
        if self.expertise_signals is None:
            self.expertise_signals = []


class PhiiContextBuilder:
    """Builds dynamic system prompts with Phii enhancements."""

    def __init__(self, reinforcement_store: Optional[ReinforcementStore] = None):
        """Initialize builder.

        Args:
            reinforcement_store: Store for feedback signals (creates default if None)
        """
        self.energy_detector = EnergyDetector()
        self.specialty_detector = SpecialtyDetector()
        self.expertise_detector = ExpertiseDetector()
        self.cue_detector = CueDetector()
        self.reinforcement_store = reinforcement_store or ReinforcementStore()

        # Per-session profiles (keyed by session_id)
        # These are the primary in-memory store for fast sync access
        # Redis serves as write-through persistence layer
        self._energy_profiles: Dict[str, EnergyProfile] = {}
        self._specialty_profiles: Dict[str, SpecialtyProfile] = {}
        self._expertise_profiles: Dict[str, ExpertiseProfile] = {}
        self._message_counts: Dict[str, int] = {}  # Track message count per session
        self._user_id_map: Dict[str, str] = {}  # session_id -> user_id mapping
        self._loaded_from_cache: set = set()  # Track which sessions were loaded from Redis

        # Cached corpus profile terms (lazy-loaded once)
        self._corpus_terms_text: Optional[str] = None
        self._corpus_terms_loaded: bool = False

    def _get_corpus_context(self) -> str:
        """Get corpus profile terms for system prompt injection.

        Lazily loads and caches the corpus profile on first call.
        Returns empty string if no profile exists or profiling is disabled.
        """
        if not self._corpus_terms_loaded:
            self._corpus_terms_loaded = True
            try:
                from config import runtime_config
                if runtime_config.corpus_profiling_enabled:
                    from tools.cohesionn.corpus_profiler import get_top_terms_text
                    self._corpus_terms_text = get_top_terms_text(top_n=20)
            except Exception as e:
                logger.debug(f"Corpus profile not available: {e}")
                self._corpus_terms_text = None

        if self._corpus_terms_text:
            # Escape curly braces so .format() calls downstream don't choke
            safe = self._corpus_terms_text.replace("{", "{{").replace("}", "}}")
            return f"\nCORPUS CONTEXT: The user's document corpus focuses on: {safe}\n"
        return ""

    def reload_corpus_profile(self) -> None:
        """Force reload of corpus profile (call after new ingest)."""
        self._corpus_terms_loaded = False
        self._corpus_terms_text = None

    async def load_from_cache(self, session_id: str) -> bool:
        """Load profiles from Redis cache.

        Call this at session start to restore profiles from previous session.
        Profiles are loaded into in-memory dicts for fast sync access.

        Args:
            session_id: Session identifier

        Returns:
            True if profiles were loaded, False if not found or error
        """
        if session_id in self._loaded_from_cache:
            return True  # Already loaded

        try:
            from services.phii_cache import get_phii_cache
            cache = await get_phii_cache()

            data = await cache.load_profiles(session_id)
            if not data:
                self._loaded_from_cache.add(session_id)
                return False

            # Restore energy profile
            energy_data = data.get("energy_profile", {})
            if energy_data:
                energy = EnergyProfile()
                energy.verbosity_preference = energy_data.get("verbosity_preference", 0.0)
                energy.technical_depth = energy_data.get("technical_depth", 0.5)
                energy.format_preference = energy_data.get("format_preference", "balanced")
                self._energy_profiles[session_id] = energy

            # Restore specialty profile
            specialty_data = data.get("specialty_profile", {})
            if specialty_data:
                specialty = SpecialtyProfile()
                if "hits" in specialty_data:
                    specialty.hits.update(specialty_data["hits"])
                self._specialty_profiles[session_id] = specialty

            # Restore expertise profile
            expertise_data = data.get("expertise_profile", {})
            if expertise_data:
                from .expertise import Expertise
                level_str = expertise_data.get("level", "intermediate")
                try:
                    level = Expertise(level_str)
                except ValueError:
                    level = Expertise.INTERMEDIATE
                expertise = ExpertiseProfile(
                    level=level,
                    confidence=expertise_data.get("confidence", 0.5),
                    signals=expertise_data.get("signals", []),
                )
                self._expertise_profiles[session_id] = expertise

            # Restore message count
            self._message_counts[session_id] = data.get("message_count", 0)

            self._loaded_from_cache.add(session_id)
            logger.info(f"PHII profiles loaded from cache: {session_id}")
            return True

        except Exception as e:
            logger.debug(f"Failed to load PHII profiles from cache: {e}")
            self._loaded_from_cache.add(session_id)
            return False

    async def save_to_cache(self, session_id: str) -> bool:
        """Save profiles to Redis cache.

        Call this periodically or on session end to persist profiles.

        Args:
            session_id: Session identifier

        Returns:
            True if saved successfully
        """
        try:
            from services.phii_cache import get_phii_cache
            cache = await get_phii_cache()

            energy = self._energy_profiles.get(session_id, EnergyProfile())
            specialty = self._specialty_profiles.get(session_id, SpecialtyProfile())
            expertise = self._expertise_profiles.get(session_id, ExpertiseProfile())
            message_count = self._message_counts.get(session_id, 0)

            return await cache.save_profiles(
                session_id, energy, specialty, expertise, message_count
            )
        except Exception as e:
            logger.debug(f"Failed to save PHII profiles to cache: {e}")
            return False

    def build_context(
        self,
        current_message: str,
        files_context: str,
        session_notes: str,
        search_hint: str = "",
        session_id: str = "default",
        phii_enabled: bool = True,
        energy_matching: bool = True,
        specialty_detection: bool = True,
    ) -> PhiiContext:
        """Build a complete system prompt with Phii enhancements.

        Args:
            current_message: User's current message
            files_context: Context string describing uploaded files
            session_notes: Session notes from SessionContext
            search_hint: Optional search mode hint to append
            session_id: Session identifier for profile tracking
            phii_enabled: Whether Phii enhancements are enabled
            energy_matching: Whether to apply energy matching
            specialty_detection: Whether to apply specialty detection

        Returns:
            PhiiContext with assembled prompt and profiles
        """
        # Import shared sections from chat_prompts.py (single source of truth)
        from routers.chat_prompts import (
            get_base_personality,
            TOOLS_SECTION,
            get_instructions_section,
            get_rules_section,
        )

        # Get or create profiles for this session
        energy_profile = self._energy_profiles.get(session_id, EnergyProfile())
        specialty_profile = self._specialty_profiles.get(session_id, SpecialtyProfile())
        expertise_profile = self._expertise_profiles.get(session_id, ExpertiseProfile())

        # Get corpus context (cached, non-blocking)
        corpus_context = self._get_corpus_context()

        if not phii_enabled:
            # Vanilla mode - use domain personality + shared sections
            # Use .replace() instead of .format() to avoid KeyError on curly braces
            # in corpus context, corrections, or other dynamic text
            system_prompt = get_base_personality() + corpus_context + TOOLS_SECTION + get_instructions_section() + get_rules_section()
            system_prompt = system_prompt.replace("{context}", files_context).replace("{session_notes}", session_notes)
            if search_hint:
                system_prompt += search_hint

            return PhiiContext(
                system_prompt=system_prompt,
                energy_profile=energy_profile,
                specialty_profile=specialty_profile,
                expertise_profile=expertise_profile,
                phii_enabled=False,
            )

        # Phii enhanced mode

        # Track message count for feature gating
        message_count = self._message_counts.get(session_id, 0) + 1
        self._message_counts[session_id] = message_count
        is_first_message = message_count == 1

        # Update profiles from current message
        # Energy detection runs on all messages (fast, helps calibrate early)
        if energy_matching:
            energy_profile = self.energy_detector.analyze(current_message, energy_profile)
            self._energy_profiles[session_id] = energy_profile

        # Feature gating: Skip specialty/expertise detection on first message
        # These are slower operations and need accumulated context to be useful
        if specialty_detection and not is_first_message:
            specialty_profile = self.specialty_detector.analyze(current_message, specialty_profile)
            self._specialty_profiles[session_id] = specialty_profile

        # Detect expertise using hybrid approach (skip on first message)
        if not is_first_message:
            expertise_profile = self.expertise_detector.analyze(
                current_message, energy_profile.technical_depth, expertise_profile
            )
            self._expertise_profiles[session_id] = expertise_profile

        # Get guidance strings
        # Only inject expertise guidance if we have reasonable confidence
        # Default ExpertiseProfile has confidence=0.5 and level=INTERMEDIATE — safe
        if expertise_profile.confidence >= 0.7:
            expertise_guidance = self.expertise_detector.get_guidance(expertise_profile)
        else:
            expertise_guidance = self.expertise_detector.get_guidance(
                ExpertiseProfile()  # Default = INTERMEDIATE, safe neutral
            )

        energy_guidance = ""
        if energy_matching:
            energy_guidance = self.energy_detector.get_guidance(energy_profile)

        specialty_hint = ""
        if specialty_detection and not is_first_message:
            specialty_hint = self.specialty_detector.get_hint(specialty_profile)

        # Get relevant corrections for this message (skip on first message)
        corrections = []
        corrections_hint = ""
        corrections_applied = []  # For explainability
        if not is_first_message:
            corrections = self.reinforcement_store.get_relevant_corrections(current_message, top_k=3)
            corrections_hint = self._build_corrections_hint(corrections)

            # Track which corrections were applied and collect reasons
            for correction in corrections:
                if correction.id:
                    self.reinforcement_store.increment_applied(correction.id, session_id=session_id)
                # Collect reason for explainability
                if correction.right_behavior:
                    corrections_applied.append(correction.right_behavior[:50])  # Truncate long reasons
                elif correction.wrong_behavior:
                    corrections_applied.append(f"avoid: {correction.wrong_behavior[:40]}")

        # Get proactive hint from pattern prediction (skip on first message)
        proactive_hint = ""
        if not is_first_message:
            suggestion = self.reinforcement_store.predict_next_action(session_id, min_confidence=0.4)
            if suggestion:
                proactive_hint = f"PROACTIVE TIP: {suggestion.message}"

        # Build enhanced prompt: PHII_PERSONALITY + shared sections
        from domain_loader import get_branding
        app_name = get_branding().get("app_name", "ARCA")

        personality = PHII_PERSONALITY.format(
            app_name=app_name,
            expertise_guidance=expertise_guidance,
            energy_guidance=energy_guidance,
            specialty_hint=specialty_hint,
            corrections_hint=corrections_hint,
            proactive_hint=proactive_hint,
        )

        # Use .replace() instead of .format() to avoid KeyError on curly braces
        # in corpus context, corrections, expertise guidance, or other dynamic text
        system_prompt = personality + corpus_context + TOOLS_SECTION + get_instructions_section() + get_rules_section()
        system_prompt = system_prompt.replace("{context}", files_context).replace("{session_notes}", session_notes)

        if search_hint:
            system_prompt += search_hint

        return PhiiContext(
            system_prompt=system_prompt,
            energy_profile=energy_profile,
            specialty_profile=specialty_profile,
            expertise_profile=expertise_profile,
            phii_enabled=True,
            corrections_applied=corrections_applied,
            expertise_signals=expertise_profile.signals[:3] if expertise_profile.signals else [],
        )

    async def process_implicit_feedback(
        self,
        current_message: str,
        session_id: str,
        previous_message_id: str,
        previous_user_message: str,
        previous_assistant_response: str,
        tools_used: List[str] = None,
    ) -> Optional[FeedbackType]:
        """Check for implicit feedback in a message and store if found.

        Also processes feedback to adjust correction confidence (verification loop).

        Args:
            current_message: Current user message to check for cues
            session_id: Session identifier
            previous_message_id: ID of the previous assistant message
            previous_user_message: The user message that prompted previous response
            previous_assistant_response: The assistant's previous response
            tools_used: Tools used in the previous response

        Returns:
            FeedbackType if implicit cue detected, None otherwise
        """
        cue = self.cue_detector.detect(current_message)

        if cue is not None:
            # Get current profiles for context
            energy_profile = self._energy_profiles.get(session_id, EnergyProfile())
            specialty_profile = self._specialty_profiles.get(session_id, SpecialtyProfile())

            # Store the feedback (async PostgreSQL, falls back to sync SQLite)
            try:
                await self.reinforcement_store.add_feedback_async(
                    session_id=session_id,
                    message_id=previous_message_id,
                    feedback_type=cue,
                    user_message=previous_user_message,
                    assistant_response=previous_assistant_response,
                    tools_used=tools_used,
                    personality="enhanced",
                    energy_profile=energy_profile.to_dict(),
                    specialty_profile=specialty_profile.to_dict(),
                )
                logger.info(f"Implicit {cue.value} feedback recorded for message {previous_message_id}")
            except Exception as e:
                logger.warning(f"Failed to store implicit feedback: {e}")

            # Process correction verification loop - adjust confidence based on feedback
            try:
                adjusted = await self.reinforcement_store.process_correction_feedback_async(
                    session_id=session_id,
                    feedback_type=cue,
                    lookback_seconds=300,  # 5 minute window
                )
                if adjusted > 0:
                    logger.info(f"Adjusted confidence for {adjusted} corrections based on {cue.value} feedback")
            except Exception as e:
                logger.warning(f"Failed to process correction feedback: {e}")

        return cue

    def add_flag(
        self,
        session_id: str,
        message_id: str,
        user_message: str,
        assistant_response: str,
        tools_used: List[str] = None,
    ) -> int:
        """Add an explicit flag for a response.

        Args:
            session_id: Session identifier
            message_id: ID of the flagged message
            user_message: The user message that prompted the response
            assistant_response: The flagged response
            tools_used: Tools used in the response

        Returns:
            ID of the created flag record
        """
        energy_profile = self._energy_profiles.get(session_id, EnergyProfile())
        specialty_profile = self._specialty_profiles.get(session_id, SpecialtyProfile())

        return self.reinforcement_store.add_feedback(
            session_id=session_id,
            message_id=message_id,
            feedback_type=FeedbackType.FLAG,
            user_message=user_message,
            assistant_response=assistant_response,
            tools_used=tools_used,
            personality="enhanced",
            energy_profile=energy_profile.to_dict(),
            specialty_profile=specialty_profile.to_dict(),
        )

    def get_session_profile(self, session_id: str) -> Dict[str, Any]:
        """Get current profiles for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict with energy, specialty, and expertise profiles
        """
        energy = self._energy_profiles.get(session_id, EnergyProfile())
        specialty = self._specialty_profiles.get(session_id, SpecialtyProfile())
        expertise = self._expertise_profiles.get(session_id, ExpertiseProfile())

        return {
            "energy": energy.to_dict(),
            "specialty": specialty.to_dict(),
            "expertise": expertise.to_dict(),
        }

    def clear_session(self, session_id: str):
        """Clear profiles for a session.

        Args:
            session_id: Session identifier to clear
        """
        self._energy_profiles.pop(session_id, None)
        self._specialty_profiles.pop(session_id, None)
        self._expertise_profiles.pop(session_id, None)
        self._message_counts.pop(session_id, None)
        self._user_id_map.pop(session_id, None)
        self._loaded_from_cache.discard(session_id)
        self.reinforcement_store.clear_session_actions(session_id)

    def load_user_profile(self, session_id: str, user_id: str) -> bool:
        """Load persistent user profile into session.

        Call this at session start to restore user preferences.
        Profiles are initialized from persistent storage if available.

        Args:
            session_id: Current session identifier
            user_id: User identifier for profile lookup

        Returns:
            True if profile was loaded, False if no profile exists
        """
        from .expertise import Expertise

        profile = self.reinforcement_store.get_user_profile(user_id)
        if profile is None:
            return False

        self._user_id_map[session_id] = user_id

        # Initialize energy profile from stored preferences
        energy = EnergyProfile()
        energy.verbosity_preference = profile.get("verbosity_preference", 0.0)
        energy.technical_depth = profile.get("technical_depth", 0.5)
        # Map preferred_format to format_preference
        pref_format = profile.get("preferred_format")
        if pref_format:
            energy.format_preference = pref_format
        self._energy_profiles[session_id] = energy

        # Initialize expertise profile
        from .expertise import ExpertiseProfile

        expertise_level = profile.get("expertise_level", "intermediate")
        try:
            level = Expertise(expertise_level)
        except ValueError:
            level = Expertise.INTERMEDIATE

        expertise = ExpertiseProfile(
            level=level,
            confidence=profile.get("expertise_confidence", 0.5),
            signals=["persistent profile"],
        )
        self._expertise_profiles[session_id] = expertise

        # Initialize specialty profile from stored specialties
        stored_specialties = profile.get("specialties", [])
        if stored_specialties:
            specialty = SpecialtyProfile()
            for spec in stored_specialties:
                if spec in specialty.hits:
                    specialty.hits[spec] = 5  # Give stored specialties a head start
            self._specialty_profiles[session_id] = specialty

        logger.info(f"Loaded persistent profile for user {user_id}: {expertise_level}, {len(stored_specialties)} specialties")
        return True

    def save_session_to_profile(self, session_id: str, user_id: str = None) -> bool:
        """Save session data to persistent user profile.

        Call this at session end to persist learned preferences.
        Uses exponential smoothing to blend with existing profile.

        Args:
            session_id: Session identifier
            user_id: User identifier (uses mapped user_id if not provided)

        Returns:
            True if profile was saved, False if no session data
        """
        # Get user_id from map if not provided
        if user_id is None:
            user_id = self._user_id_map.get(session_id)
        if user_id is None:
            logger.warning(f"No user_id for session {session_id}, cannot save profile")
            return False

        # Get session profiles
        energy = self._energy_profiles.get(session_id)
        expertise = self._expertise_profiles.get(session_id)
        specialty = self._specialty_profiles.get(session_id)

        if not any([energy, expertise, specialty]):
            return False

        # Extract data for persistence
        expertise_level = expertise.level.value if expertise else "intermediate"
        expertise_confidence = expertise.confidence if expertise else 0.5
        verbosity = energy.verbosity_preference if energy else 0.0
        technical_depth = energy.technical_depth if energy else 0.5

        # Get top specialties
        specialties = []
        if specialty:
            sorted_specs = sorted(specialty.hits.items(), key=lambda x: x[1], reverse=True)
            specialties = [s[0] for s in sorted_specs[:5] if s[1] > 0]

        # Merge into persistent profile
        self.reinforcement_store.merge_session_to_profile(
            user_id=user_id,
            expertise_level=expertise_level,
            expertise_confidence=expertise_confidence,
            verbosity_preference=verbosity,
            technical_depth=technical_depth,
            specialties=specialties,
        )

        logger.info(f"Saved session {session_id} to profile for user {user_id}")
        return True

    async def observe_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        tools_used: List[str] = None,
    ) -> Optional[Correction]:
        """Observe a complete exchange for learning opportunities.

        Called after each assistant response to detect if the user's message
        contains a correction of previous behavior. If detected, the correction
        is stored for future reference.

        Also tracks actions for pattern learning to enable proactive suggestions.

        Args:
            session_id: Session identifier
            user_message: The user's message
            assistant_response: The assistant's response to analyze
            tools_used: List of tools used in the response

        Returns:
            Correction if one was detected and stored, None otherwise
        """
        # Track action for pattern learning (classify is pure logic, record is DB)
        action = self.reinforcement_store.classify_action(user_message, tools_used or [])
        if action:
            await self.reinforcement_store.record_action_async(session_id, action)
            logger.debug(f"Recorded action '{action}' for session {session_id}")

        # Check for correction in the user's message (pure logic, no DB)
        correction = self.reinforcement_store.detect_correction(user_message, assistant_response)

        if correction:
            # Store the correction (async PostgreSQL, falls back to sync SQLite)
            correction_id = await self.reinforcement_store.store_correction_async(correction, session_id)
            correction.id = correction_id
            logger.info(
                f"Learned correction #{correction_id}: "
                f"'{correction.wrong_behavior}' -> '{correction.right_behavior}'"
            )
            return correction

        return None

    def _build_corrections_hint(self, corrections: List[Correction]) -> str:
        """Build a hint string from relevant corrections.

        Args:
            corrections: List of relevant corrections

        Returns:
            Formatted hint string, or empty string if no corrections
        """
        if not corrections:
            return ""

        lines = ["LEARNED CORRECTIONS (from previous interactions):"]
        for c in corrections:
            if c.wrong_behavior and c.right_behavior:
                lines.append(f"• Avoid: {c.wrong_behavior}")
                lines.append(f"  Instead: {c.right_behavior}")
            elif c.wrong_behavior:
                lines.append(f"• Avoid: {c.wrong_behavior}")
            elif c.right_behavior:
                lines.append(f"• Remember: {c.right_behavior}")

        return "\n".join(lines)
