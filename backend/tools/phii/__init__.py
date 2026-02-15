"""
Phii - Behavior Enhancement Module for ARCA

Provides adaptive, context-aware responses through:
- Energy matching (user style detection)
- Specialty detection (engineering focus areas)
- Expertise tiering (junior/intermediate/senior/management)
- Firm-wide standards (unit conventions, regional rules, professional practices)
- Reinforcement signals (feedback tracking)
- Dynamic prompt assembly

Usage:
    from tools.phii import PhiiContextBuilder, ReinforcementStore

    store = ReinforcementStore()
    builder = PhiiContextBuilder(store)

    # Build enhanced system prompt
    context = builder.build_context(
        current_message="What's the standard approach?",
        files_context="No files uploaded",
        session_notes="",
    )
    system_prompt = context.system_prompt
"""

from .personalities import PHII_PERSONALITY, get_prompt
from .energy import EnergyDetector, EnergyProfile
from .specialties import Specialty, SpecialtyDetector, SpecialtyProfile
from .expertise import Expertise, ExpertiseProfile, ExpertiseDetector
from .reinforcement import (
    ReinforcementStore,
    FeedbackRecord,
    CueDetector,
    Correction,
    CorrectionDetector,
)
from .context_builder import PhiiContextBuilder, PhiiContext
from .seed import seed_if_empty, get_all_firm_corrections, get_firm_terminology

__all__ = [
    "PHII_PERSONALITY",
    "get_prompt",
    "EnergyDetector",
    "EnergyProfile",
    "Specialty",
    "SpecialtyDetector",
    "SpecialtyProfile",
    "Expertise",
    "ExpertiseProfile",
    "ExpertiseDetector",
    "ReinforcementStore",
    "FeedbackRecord",
    "CueDetector",
    "Correction",
    "CorrectionDetector",
    "PhiiContextBuilder",
    "PhiiContext",
    "seed_if_empty",
    "get_all_firm_corrections",
    "get_firm_terminology",
]
