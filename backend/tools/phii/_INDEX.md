# backend/tools/phii/

Behavior enhancement system that dynamically adapts response style, technical depth, and personality based on real-time user analysis. Has 5 subsystems: context building, energy detection, specialty detection, expertise detection, and reinforcement learning. All domain knowledge loads from lexicon pipeline config.

| File | Purpose | Key Exports |
|------|---------|-------------|
| __init__.py | Public API | `PhiiContextBuilder`, `get_prompt()` |
| context_builder.py | Main orchestrator: builds system prompt with all Phii guidance injected | `PhiiContextBuilder`, `PhiiContext` |
| energy.py | Analyzes communication style (brevity, formality, urgency, technical depth, verbosity) | `EnergyDetector`, `EnergyProfile` |
| expertise.py | Detects expertise level (junior/intermediate/senior/management) via signals | `ExpertiseDetector`, `ExpertiseLevel` |
| specialties.py | Detects domain specialty from lexicon | `SpecialtyDetector`, `SpecialtyProfile` |
| reinforcement.py | PostgreSQL-backed learning: corrections, action patterns, feedback loops (~1050 lines) | `ReinforcementStore`, `CueDetector`, `CorrectionDetector` |
| seed.py | Seeds firm corrections and terminology tables from lexicon | `seed_if_empty()` |
| personalities.py | Personality template with 6 injection points | `PHII_PERSONALITY`, `get_prompt()` |
