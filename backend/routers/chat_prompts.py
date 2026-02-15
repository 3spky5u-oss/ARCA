"""
ARCA Chat Prompts - System prompts and technical detection

Contains:
- get_base_personality(): Domain-aware personality from lexicon
- get_welcome_message(): Domain-aware welcome message from lexicon
- TOOLS_SECTION: Available tools description (auto-generated from registry)
- INSTRUCTIONS_SECTION: Tool usage instructions
- RULES_SECTION: Response style and context (template)
- SYSTEM_PROMPT: Complete vanilla prompt
- get_technical_patterns(): Domain-aware technical patterns from lexicon
- is_technical_question(): Detect technical engineering questions
- is_geology_query(): Detect geology queries (domain-aware)
- cleanup_response_text(): Clean LLM response of artifacts
"""

import re
from typing import List


def cleanup_response_text(text: str) -> str:
    """Clean LLM response of tool call artifacts and think tags.

    Removes:
    - Complete <think>...</think> blocks
    - Orphaned </think> or <think> tags
    - Inline JSON tool calls ({"name": "...", "arguments": ...})
    - Tool-prefixed lines (e.g., "search_knowledge: ..." from DeepSeek)
    - Excessive whitespace
    """
    if not text:
        return text

    # Remove all think tags (complete and orphaned)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</think>", "", text)
    text = re.sub(r"<think>", "", text)

    # Remove tool-prefixed lines that DeepSeek sometimes outputs
    # Matches lines starting with "tool_name: " pattern
    # Get tool names dynamically from registry
    from tools.registry import ToolRegistry, register_all_tools

    register_all_tools()  # Ensure tools are registered
    tool_names = list(ToolRegistry.get_all_tools().keys())
    for tool in tool_names:
        # Remove "tool_name: content" at line start, keeping content on new line if substantive
        text = re.sub(rf"^{tool}:\s*", "", text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove inline JSON tool calls - patterns like {"name": "tool_name", "arguments": {...}}
    # This catches tool call JSON that the LLM outputs as text instead of using native tool calls
    def remove_tool_json(content: str) -> str:
        """Remove JSON objects that look like tool calls."""
        result = content.strip()

        # Special case: if entire response is just a JSON tool call, return empty
        # This catches LLM outputting only tool call JSON with no human text
        if result.startswith("{") and '"name"' in result[:100] and '"arguments"' in result:
            # Looks like the entire response is a tool call - strip it
            # Return a helpful message instead
            tool_match = re.search(r'["\']name["\']\s*:\s*["\'](\w+)["\']', result)
            if tool_match:
                tool_name = tool_match.group(1)
                return f"I've processed the data. You can use the buttons above to generate a {tool_name.replace('_', ' ')}."

        # Normal case: remove embedded tool call JSON
        # Keep removing until no more matches (handles multiple tool calls)
        while True:
            # Find potential tool call JSON start
            match = re.search(r'\{\s*["\']name["\']\s*:\s*["\'](\w+)["\']', result)
            if not match:
                break

            start = match.start()
            # Find matching closing brace - but be careful of braces inside strings
            brace_count = 0
            in_string = False
            escape_next = False
            end = start

            for i, c in enumerate(result[start:], start):
                if escape_next:
                    escape_next = False
                    continue
                if c == "\\":
                    escape_next = True
                    continue
                if c == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == "{":
                    brace_count += 1
                elif c == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break

            if end > start:
                result = result[:start] + result[end:]
            else:
                break

        return result

    text = remove_tool_json(text)

    # Remove markdown links, keeping just the link text
    # Matches [link text](url) and replaces with just "link text"
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove LLM-generated reference/source sections (UI handles citations)
    # Matches: "## References", "**References**", "6. References", "References:", etc.
    # Removes from the header to end of text (references are usually at the end)
    reference_pattern = (
        r"(?:^|\n)(?:\d+\.\s*|\##+\s*|\*\*)?(?:References?|Sources?|Bibliography|Citations?)(?:\*\*)?:?\s*\n[\s\S]*$"
    )
    text = re.sub(reference_pattern, "", text, flags=re.IGNORECASE)

    # Remove inline one-line source/reference labels that slip through.
    # Examples:
    # - "Sources: retrieval_pipeline.md"
    # - "**References:** [1] ..."
    text = re.sub(
        r"(?im)^[ \t>]*\*{0,2}(?:references?|sources?|bibliography|citations?)\*{0,2}\s*:\s*.*$",
        "",
        text,
    )

    # Clean up multiple newlines and spaces
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
    text = re.sub(r"  +", " ", text)

    return text.strip()


# =============================================================================
# TECHNICAL QUESTION DETECTION (Pre-query search forcing)
# =============================================================================

def get_technical_patterns() -> List[str]:
    """Get technical patterns from domain lexicon, empty list if none defined."""
    from domain_loader import get_lexicon
    lexicon = get_lexicon()
    return lexicon.get("technical_patterns", [])


# Keep module-level reference for backward compatibility
TECHNICAL_PATTERNS = None  # Lazy-loaded via get_technical_patterns()


def is_geology_query(message: str) -> tuple[bool, str]:
    """
    Detect if message is a geology query that should force Mapperr tool.

    Uses advanced_triggers (or legacy geology_triggers) from domain lexicon.
    Returns (False, "") if no triggers are defined (non-geology domain).

    Returns:
        (is_geology_query, extracted_location)
    """
    from domain_loader import get_lexicon

    lexicon = get_lexicon()
    geology_keywords = lexicon.get("advanced_triggers", []) or lexicon.get("geology_triggers", [])

    # No geology triggers = no geology detection
    if not geology_keywords:
        return False, ""

    message_lower = message.lower()

    # Must have geology intent
    if not any(geo in message_lower for geo in geology_keywords):
        return False, ""

    # Extract location using patterns from pipeline config
    from domain_loader import get_pipeline_config
    pipeline = get_pipeline_config()
    location_patterns = pipeline.get("geology_location_patterns", [])

    if not location_patterns:
        # No location patterns defined — no location extraction
        return False, ""

    for pattern in location_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            skip_words = ["the", "this", "that", "these", "my", "our", "your"]
            if location.lower() not in skip_words:
                return True, location

    return False, ""


def is_technical_question(message: str) -> bool:
    """Detect if a message is a technical question that should trigger auto-search.

    Two modes:
    1. Domain with technical_patterns: regex match (domain pack patterns)
    2. Generic domain (no patterns): auto-search if enabled topics have data,
       unless the message is clearly casual (greeting, short command, etc.)
    """
    from domain_loader import get_lexicon

    lexicon = get_lexicon()
    message_lower = message.lower().strip()

    # Skip conversational follow-ups — bad RAG queries, let LLM reformulate
    _CONTINUATION_PHRASES = [
        "continue", "keep going", "go on", "more of", "pull more",
        "show more", "what else", "any more", "any other", "next ones",
    ]
    if any(phrase in message_lower for phrase in _CONTINUATION_PHRASES):
        # Guard against false positives like "continue pulling bearing capacity data"
        # If the message also contains a technical noun (>60 chars usually), let it through
        if len(message_lower) < 60:
            return False

    _REFERENTIAL_STARTS = [
        "what about", "tell me more", "can you also", "how about", "and the",
    ]
    if len(message_lower) < 50 and any(message_lower.startswith(phrase) for phrase in _REFERENTIAL_STARTS):
        return False

    # Skip very short messages (greetings, commands)
    if len(message_lower) < 15:
        return False

    # Skip if clearly about uploaded files, analysis, or visualization tasks
    skip_list = lexicon.get("skip_patterns", [])
    if skip_list and any(x in message_lower for x in skip_list):
        return False

    # Skip if geological map query (Mapperr will be force-called)
    is_geo, _ = is_geology_query(message)
    if is_geo:
        return False

    # Mode 1: Domain-specific patterns
    patterns = lexicon.get("technical_patterns", [])
    if patterns:
        for pattern in patterns:
            if re.search(pattern, message_lower):
                return True
        return False

    # Mode 2: Generic domain — auto-search if knowledge topics are enabled
    # Skip casual/meta messages
    _CASUAL = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye",
               "good morning", "good evening", "help", "what can you do"}
    if message_lower in _CASUAL:
        return False

    try:
        from config import runtime_config
        enabled = runtime_config.get_enabled_topics()
        if enabled:
            return True
    except Exception:
        pass

    return False


# =============================================================================
# SYSTEM PROMPT - Composable Sections (Single Source of Truth)
# =============================================================================
# These sections are exported for use by Phii module. When adding tools or
# changing instructions, edit here ONLY - Phii will inherit automatically.

_GENERIC_PERSONALITY = """You are ARCA, a helpful AI assistant. You help users with technical questions, document processing, and knowledge search.

PERSONALITY:
- Professional but approachable
- Direct and concise
- Helpful without being overly verbose
- Honest about limitations

"""


ARCA_IDENTITY_GUARD = """ARCA IDENTITY (MANDATORY):
- In this application, "ARCA" is the product/platform name for this local RAG + MCP system.
- Treat "ARCA" as a name, not an acronym expansion.
- Do NOT redefine ARCA as unrelated acronyms (for example: racing organizations or "advanced research computing" expansions).
- If asked what ARCA means, answer that it is this platform's name in context.

"""


def get_base_personality() -> str:
    """Get personality from domain lexicon, generic fallback."""
    from domain_loader import get_lexicon
    lexicon = get_lexicon()
    identity = lexicon.get("identity", {})
    return identity.get("personality", _GENERIC_PERSONALITY)


# Keep module-level for backward compatibility (lazy proxy)
BASE_PERSONALITY = None  # Use get_base_personality() instead


# Shared tools section - AUTO-GENERATED from registry
# Single source of truth: tool descriptions live in backend/tools/registry.py
def get_tools_section() -> str:
    """Generate TOOLS_SECTION from registry at runtime.

    This ensures the system prompt always reflects the current tool registry.
    """
    from tools.registry import ToolRegistry, register_all_tools

    register_all_tools()  # Ensure tools are registered
    return ToolRegistry.generate_tools_section()


# Lazy-loaded TOOLS_SECTION (generated on first access)
_tools_section_cache: str = ""


def _get_tools_section_cached() -> str:
    """Get cached TOOLS_SECTION, generating if needed."""
    global _tools_section_cache
    if not _tools_section_cache:
        _tools_section_cache = get_tools_section()
    return _tools_section_cache


def clear_prompt_caches() -> None:
    """Clear cached prompt sections (call after domain switch)."""
    global _tools_section_cache
    _tools_section_cache = ""
    # Also clear graph extraction domain pattern cache
    try:
        from tools.cohesionn.graph_extraction import clear_domain_patterns_cache
        clear_domain_patterns_cache()
    except ImportError:
        pass
    # Clear topic router cache
    try:
        from tools.cohesionn.reranker import clear_topic_cache
        clear_topic_cache()
    except ImportError:
        pass
    # Clear synonym expansion cache
    try:
        from tools.cohesionn.query_expansion import clear_synonym_cache
        clear_synonym_cache()
    except ImportError:
        pass
    # Clear BM25 preserve terms cache
    try:
        from tools.cohesionn.sparse_retrieval import clear_preserve_terms_cache
        clear_preserve_terms_cache()
    except ImportError:
        pass
    # Clear Phii energy technical patterns cache
    try:
        from tools.phii.energy import clear_technical_patterns_cache
        clear_technical_patterns_cache()
    except ImportError:
        pass


# For backwards compatibility, expose as TOOLS_SECTION property
# Code that imports TOOLS_SECTION will get the cached generated version
class _ToolsSectionProxy:
    """Proxy that generates TOOLS_SECTION on first access."""

    def __str__(self) -> str:
        return _get_tools_section_cached()

    def __add__(self, other: str) -> str:
        return _get_tools_section_cached() + other

    def __radd__(self, other: str) -> str:
        return other + _get_tools_section_cached()

    def lower(self) -> str:
        return _get_tools_section_cached().lower()

    def upper(self) -> str:
        return _get_tools_section_cached().upper()

    def __contains__(self, item: str) -> bool:
        return item in _get_tools_section_cached()


TOOLS_SECTION = _ToolsSectionProxy()


# Shared instructions section - cross-cutting guidance only
# Tool-specific usage guidance lives in registry.py descriptions (single source of truth)
def get_instructions_section() -> str:
    """Build instructions section with domain-aware confidence example."""
    from domain_loader import get_pipeline_config
    pipeline = get_pipeline_config()
    confidence_example = pipeline["confidence_example"]
    return f"""CALCULATION RESULTS:
- Show inputs, results, and formula used
- Ask for missing required parameters before calculating

RETRIEVED CONTEXT HANDLING:
- If retrieved context does not directly answer the specific question, say so
- Do NOT extrapolate from neighboring projects or related-but-different data
- Do NOT fabricate numerical values not explicitly stated in retrieved context
- "The available documents do not contain this specific information" is a valid answer

CONFIDENCE LEVELS:
For technical recommendations, indicate your confidence:
- **HIGH**: Well-constrained problem with direct code/standard guidance, sufficient data
- **MEDIUM**: Requires engineering judgment, typical ranges exist, standard correlations used
- **LOW**: Limited data, novel situation, multiple valid interpretations - recommend verification

State confidence after recommendations. Example: "Confidence: MEDIUM - based on {confidence_example}."

CITATION HANDLING:
- NEVER cite sources in response text - the UI auto-cites with confidence badges
- NEVER include: "References:", "Sources:", "Bibliography:", author names like "Das (2016)", numbered citations like "[1]"

FEATURE EXPLANATIONS:
- If user asks for a full/system overview (e.g., "all features", "how ARCA works"), provide a complete breakdown by category
- Use markdown headings and bullet lists for capability inventories
- Do not claim docs are missing unless required details are truly absent from retrieved context

"""

# Keep module-level reference for backward compatibility
INSTRUCTIONS_SECTION = ""  # Use get_instructions_section() instead


def get_rules_section() -> str:
    """Build rules section with domain-aware equation examples."""
    from domain_loader import get_pipeline_config
    pipeline = get_pipeline_config()
    equation_example = pipeline["equation_example"]
    # Note: using concatenation to avoid raw string issues with f-string
    return (
        "RESPONSE STYLE:\n"
        "- Default to complete, high-utility explanations for technical/system questions\n"
        "- Use concise responses only when the user asks for brief output\n"
        "- NO markdown tables - UI renders them\n"
        "- NO file paths, source citations, or download URLs - the UI adds these automatically\n"
        "- NEVER add References/Sources/Bibliography sections - auto-cite handles this\n"
        '- NEVER cite authors like "Das (2016)" or "[1] Smith" - UI shows sources automatically\n'
        "- For feature/capability overviews, use short headings + bullet lists (one capability per bullet)\n"
        '- Good: "Done. 14 exceedances in 64 samples - all Chloride, 130-200 vs 120 limit."\n'
        '- Bad: "I have completed the analysis. Here is a table of results..."\n'
        '- Bad: "6. References\\nDas (2016)..." (UI handles citations automatically)\n'
        "\n"
        "FORMATTING:\n"
        "- NO emojis - this is professional software\n"
        "- For math/equations: use $$...$$ for block math, $...$ for inline math (KaTeX format)\n"
        f"- Example: {equation_example}\n"
        "- NEVER use \\[ ... \\] or \\( ... \\) delimiters - they won't render\n"
        "\n"
        "WHAT YOU DON'T DO:\n"
        "- Give recommendations - state facts, the user decides\n"
        '- Say "I\'d be happy to help" or "Great question!" - just help\n'
        "- Over-explain unless asked\n"
        "- Use emojis\n"
        "\n"
        "CURRENT SESSION:\n"
        "{context}\n"
        "\n"
        "{session_notes}"
    )

# Keep module-level reference for backward compatibility
RULES_SECTION = ""  # Use get_rules_section() instead


# Full vanilla prompt - composed from sections at runtime to avoid circular imports
def get_system_prompt() -> str:
    """Get the full system prompt with tools section generated from registry.

    Called at runtime, not import time, to avoid circular imports.
    All sections are domain-aware via lexicon pipeline config.
    """
    return (
        get_base_personality()
        + ARCA_IDENTITY_GUARD
        + str(TOOLS_SECTION)
        + get_instructions_section()
        + get_rules_section()
    )


# For backwards compatibility - but prefer using get_system_prompt()
# This will be evaluated lazily on first string operation
class _SystemPromptProxy:
    """Proxy that generates SYSTEM_PROMPT on first access."""

    def __str__(self) -> str:
        return get_system_prompt()

    def __add__(self, other: str) -> str:
        return get_system_prompt() + other

    def __radd__(self, other: str) -> str:
        return other + get_system_prompt()

    def format(self, **kwargs) -> str:
        return get_system_prompt().format(**kwargs)


SYSTEM_PROMPT = _SystemPromptProxy()


_GENERIC_WELCOME = """Hello! I'm ARCA, your AI assistant.

What I can help with:
- **Knowledge search**: Ask questions about your document library
- Unit conversions and quick calculations
- Document processing and redaction
- Web search for current information

How can I help?"""


def get_welcome_message() -> str:
    """Get welcome message from domain lexicon, generic fallback."""
    from domain_loader import get_lexicon
    lexicon = get_lexicon()
    identity = lexicon.get("identity", {})
    return identity.get("welcome_message", _GENERIC_WELCOME)


# Keep module-level for backward compatibility
WELCOME_MESSAGE = None  # Use get_welcome_message() instead
