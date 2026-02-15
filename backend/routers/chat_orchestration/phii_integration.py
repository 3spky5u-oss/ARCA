"""
ARCA Phii Integration - Behavior enhancement facade

Provides a clean interface to the Phii behavior enhancement module.
Consolidates:
- Escape command handling (/reset learning, /learning status, etc.)
- Implicit feedback processing
- Context building with personalization
- Exchange observation for learning
"""

import logging
import re
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Try to import Phii module
try:
    from tools.phii import PhiiContextBuilder, ReinforcementStore

    PHII_AVAILABLE = True
except ImportError:
    PHII_AVAILABLE = False
    PhiiContextBuilder = None
    ReinforcementStore = None
    logger.warning("Phii module not available - using vanilla prompts")


# Escape hatch command patterns
LEARNING_COMMANDS = [
    (re.compile(r"^/reset\s*learning\s*$", re.IGNORECASE), "reset_learning"),
    (re.compile(r"^/learning\s*status\s*$", re.IGNORECASE), "show_status"),
    (re.compile(r"^/be\s+(concise|detailed|technical|simple)\s*$", re.IGNORECASE), "set_style"),
    (re.compile(r"^/expertise\s+(junior|senior|intermediate|management)\s*$", re.IGNORECASE), "set_expertise"),
]


class PhiiIntegration:
    """Facade for Phii behavior enhancement module.

    Provides a unified interface for:
    - Checking if Phii is available
    - Handling escape commands
    - Processing implicit feedback
    - Building personalized context
    - Observing exchanges for learning
    """

    def __init__(self, runtime_config=None):
        """Initialize Phii integration.

        Args:
            runtime_config: RuntimeConfig for feature flags
        """
        self.runtime_config = runtime_config
        self._builder: Optional["PhiiContextBuilder"] = None
        self._store: Optional["ReinforcementStore"] = None

    @property
    def available(self) -> bool:
        """Check if Phii module is available."""
        return PHII_AVAILABLE

    def _get_builder(self) -> Optional["PhiiContextBuilder"]:
        """Get or create singleton PhiiContextBuilder."""
        if not PHII_AVAILABLE:
            return None
        if self._builder is None:
            self._store = ReinforcementStore()
            self._builder = PhiiContextBuilder(self._store)
        return self._builder

    async def handle_escape_command(
        self,
        message: str,
        session_id: str,
    ) -> Optional[str]:
        """Handle /commands for learning control.

        Args:
            message: User message (potentially a command)
            session_id: Session identifier

        Returns:
            Response string if command matched, None otherwise
        """
        if not PHII_AVAILABLE:
            return None

        builder = self._get_builder()
        if not builder:
            return None

        msg = message.strip()

        for pattern, cmd in LEARNING_COMMANDS:
            match = pattern.match(msg)
            if not match:
                continue

            if cmd == "reset_learning":
                builder.clear_session(session_id)
                return "Learning reset for this session. Profiles cleared."

            elif cmd == "show_status":
                profile = builder.get_session_profile(session_id)
                energy = profile.get("energy", {})
                expertise = profile.get("expertise", {})
                return (
                    f"**Learning Status**\n"
                    f"- Expertise: {expertise.get('level', 'intermediate')} "
                    f"(confidence: {expertise.get('confidence', 0.5):.0%})\n"
                    f"- Energy: brevity={energy.get('brevity', 'normal')}, "
                    f"formality={energy.get('formality', 'neutral')}\n"
                    f"- Verbosity preference: {energy.get('verbosity_preference', 0):.1f}\n"
                    f"- Format preference: {energy.get('format_preference', 'none')}"
                )

            elif cmd == "set_style":
                style = match.group(1).lower()
                energy_profile = builder._energy_profiles.get(session_id)
                if energy_profile is None:
                    from tools.phii.energy import EnergyProfile

                    energy_profile = EnergyProfile()
                    builder._energy_profiles[session_id] = energy_profile

                if style == "concise":
                    energy_profile.verbosity_preference = -0.5
                elif style == "detailed":
                    energy_profile.verbosity_preference = 0.5
                elif style == "technical":
                    energy_profile.technical_depth = 0.8
                elif style == "simple":
                    energy_profile.technical_depth = 0.2

                return f"Style set to **{style}**."

            elif cmd == "set_expertise":
                level = match.group(1).lower()
                from tools.phii.expertise import Expertise, ExpertiseProfile

                expertise_profile = ExpertiseProfile(
                    level=Expertise(level),
                    confidence=0.9,
                    signals=["explicit command"],
                )
                builder._expertise_profiles[session_id] = expertise_profile
                return f"Expertise level set to **{level}**."

        return None

    async def process_implicit_feedback(
        self,
        current_message: str,
        session_id: str,
        previous_message_id: str,
        previous_user_message: str,
        previous_assistant_response: str,
        tools_used: List[str],
    ) -> None:
        """Process implicit feedback from user's follow-up message.

        Args:
            current_message: Current user message
            session_id: Session identifier
            previous_message_id: ID of previous exchange
            previous_user_message: Previous user message
            previous_assistant_response: Previous assistant response
            tools_used: Tools used in previous response
        """
        if not PHII_AVAILABLE:
            return

        if not self.runtime_config or not self.runtime_config.phii_implicit_feedback:
            return

        builder = self._get_builder()
        if builder:
            await builder.process_implicit_feedback(
                current_message=current_message,
                session_id=session_id,
                previous_message_id=previous_message_id,
                previous_user_message=previous_user_message,
                previous_assistant_response=previous_assistant_response,
                tools_used=tools_used,
            )

    def build_system_prompt(
        self,
        message: str,
        files_context: str,
        session_notes: str,
        search_hint: str,
        session_id: str,
        phii_enabled: bool,
        base_prompt: str,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Build system prompt with optional Phii personalization.

        Args:
            message: Current user message
            files_context: File context string
            session_notes: Session notes string
            search_hint: Search mode hint string
            session_id: Session identifier
            phii_enabled: Whether Phii is enabled for this request
            base_prompt: Base system prompt template

        Returns:
            Tuple of (system_prompt, phii_metadata)
        """
        if not PHII_AVAILABLE or not phii_enabled or not self.runtime_config:
            # Return vanilla prompt
            prompt = base_prompt.format(context=files_context, session_notes=session_notes) + search_hint
            return prompt, None

        builder = self._get_builder()
        if not builder:
            prompt = base_prompt.format(context=files_context, session_notes=session_notes) + search_hint
            return prompt, None

        phii_ctx = builder.build_context(
            current_message=message,
            files_context=files_context,
            session_notes=session_notes,
            search_hint=search_hint,
            session_id=session_id,
            phii_enabled=True,
            energy_matching=self.runtime_config.phii_energy_matching,
            specialty_detection=self.runtime_config.phii_specialty_detection,
        )

        # Build explainability metadata
        corrections_count = len(phii_ctx.corrections_applied)
        is_personalized = corrections_count > 0 or phii_ctx.expertise_profile.level.value != "intermediate"

        phii_metadata = {
            "corrections_applied": corrections_count,
            "corrections_reasons": phii_ctx.corrections_applied,  # List of what was applied
            "expertise_level": phii_ctx.expertise_profile.level.value,
            "expertise_signals": phii_ctx.expertise_signals,  # What triggered expertise detection
            "personalized": is_personalized,
        }

        return phii_ctx.system_prompt, phii_metadata

    async def observe_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        tools_used: List[str],
    ) -> None:
        """Observe a message exchange for Phii learning.

        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_response: Assistant's response
            tools_used: Tools that were used
        """
        if not PHII_AVAILABLE:
            return

        builder = self._get_builder()
        if builder:
            try:
                await builder.observe_exchange(
                    session_id=session_id,
                    user_message=user_message,
                    assistant_response=assistant_response,
                    tools_used=tools_used,
                )
            except Exception as e:
                logger.warning(f"Phii observe_exchange error: {e}")

    def load_user_profile(self, session_id: str, user_id: str) -> bool:
        """Load persistent user profile for session.

        Call at session start to restore user preferences.

        Args:
            session_id: Session identifier
            user_id: User identifier for profile lookup

        Returns:
            True if profile was loaded
        """
        if not PHII_AVAILABLE:
            return False

        builder = self._get_builder()
        if builder:
            try:
                return builder.load_user_profile(session_id, user_id)
            except Exception as e:
                logger.warning(f"Failed to load user profile: {e}")
        return False

    def save_user_profile(self, session_id: str, user_id: str = None) -> bool:
        """Save session data to persistent user profile.

        Call at session end to persist learned preferences.

        Args:
            session_id: Session identifier
            user_id: User identifier (optional if already mapped)

        Returns:
            True if profile was saved
        """
        if not PHII_AVAILABLE:
            return False

        builder = self._get_builder()
        if builder:
            try:
                return builder.save_session_to_profile(session_id, user_id)
            except Exception as e:
                logger.warning(f"Failed to save user profile: {e}")
        return False
