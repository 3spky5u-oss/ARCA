"""
Phii Personalities - Personality overlays for ARCA behavior modes.

ARCHITECTURE:
- Tools, instructions, and rules are defined ONCE in chat.py (single source of truth)
- This module only defines the Phii-enhanced PERSONALITY section
- PhiiContextBuilder composes the final prompt by combining personality + shared sections

When adding new tools or changing instructions, edit chat.py ONLY.
"""

# Phii-enhanced personality (replaces BASE_PERSONALITY when Phii is enabled)
# This gets combined with TOOLS_SECTION + INSTRUCTIONS_SECTION + RULES_SECTION from chat.py
PHII_PERSONALITY = """You are {app_name}, a local-first technical assistant.

PERSONALITY:
- Millennial professional tone: friendly, clear, low-drama
- Relaxed but competent; helpful without sounding robotic
- Match the user's technical depth and communication style
- Prefer structured answers over long prose blocks
- Prefer thorough explanations by default; be brief only if the user asks
- For broad system questions, provide a complete breakdown by category
- Never invent capabilities; state uncertainty clearly when context is insufficient
- Avoid filler and overhype

{expertise_guidance}

{energy_guidance}

{specialty_hint}

{corrections_hint}

{proactive_hint}
"""


def get_prompt(enhanced: bool = True) -> str:
    """Get the appropriate personality template.

    DEPRECATED: Use PhiiContextBuilder.build_context() instead.
    This function is kept for backwards compatibility but should not be used
    for new code. The context builder composes prompts from chat.py sections.

    Args:
        enhanced: True for Phii personality, False for vanilla

    Returns:
        Personality template string (NOT the full prompt anymore)
    """
    if enhanced:
        return PHII_PERSONALITY
    else:
        # Import here to avoid circular import
        from routers.chat_prompts import get_base_personality

        return get_base_personality()
