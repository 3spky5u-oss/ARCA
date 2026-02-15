"""
Geology Handler - Domain-specific location queries.

Forces the query_geological_map tool BEFORE the LLM sees the message to prevent
hallucination of location-specific data. Only active when a domain pack provides Mapperr.
"""

import logging
from typing import Optional, Tuple

from .base import QueryHandler, HandlerContext, LLMConfig

logger = logging.getLogger(__name__)

# Domain-gated import: mapperr is only available with a domain pack that provides it
try:
    from tools.mapperr import query_geological_map
    MAPPERR_AVAILABLE = True
except ImportError:
    MAPPERR_AVAILABLE = False

# Load domain-specific location and geology keywords from pipeline config.
# These are only populated when a domain pack defines them in its lexicon.
try:
    from domain_loader import get_pipeline_config
    _pipeline = get_pipeline_config()
    COMMUNITY_KEYWORDS = _pipeline.get("geology_community_keywords", [])
    GEOLOGY_KEYWORDS = _pipeline.get("geology_keywords", [])
except Exception:
    COMMUNITY_KEYWORDS = []
    GEOLOGY_KEYWORDS = []


def is_geology_query(message: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if message is asking about domain-specific location data.

    Uses geology_keywords and geology_community_keywords from pipeline config.
    Returns (False, None) if no keywords are configured.

    Returns:
        (is_geology, location) - location is extracted if found
    """
    if not GEOLOGY_KEYWORDS or not COMMUNITY_KEYWORDS:
        return False, None

    msg_lower = message.lower()

    # Must have a domain-specific keyword
    has_geology = any(kw in msg_lower for kw in GEOLOGY_KEYWORDS)
    if not has_geology:
        return False, None

    # Must mention a known location
    for loc in COMMUNITY_KEYWORDS:
        if loc in msg_lower:
            # Extract the location (capitalize for API)
            loc_start = msg_lower.find(loc)
            loc_end = loc_start + len(loc)
            location = message[loc_start:loc_end].strip()

            # Title case for proper nouns (short abbreviations stay uppercase)
            if len(location) <= 2:
                location = location.upper()
            else:
                location = location.title()

            return True, location

    return False, None


class GeologyHandler(QueryHandler):
    """
    Handler for domain-specific location queries.

    Forces query_geological_map BEFORE LLM to prevent hallucination.

    Detection:
    1. Primary: ToolRouter selected query_geological_map
    2. Fallback: Pattern matching (domain keywords + known locations from pipeline config)
    """

    priority = 10  # High priority
    name = "geology"

    def __init__(self):
        self._location: Optional[str] = None

    def should_handle(self, ctx: HandlerContext) -> bool:
        """
        Check if this is a geology query.

        Primary: Use router decision if available
        Fallback: Pattern matching
        """
        if not MAPPERR_AVAILABLE:
            return False

        # Primary: Router selected this tool
        if ctx.router_decision and ctx.router_decision.tool == "query_geological_map":
            self._location = ctx.router_decision.args.get("location")
            if self._location:
                logger.info(f"GeologyHandler: Router selected geology for '{self._location}'")
                return True

        # Fallback: Pattern matching
        is_geo, location = is_geology_query(ctx.message)
        if is_geo and location:
            self._location = location
            logger.info(f"GeologyHandler: Pattern match for '{location}'")
            return True

        return False

    async def pre_process(self, ctx: HandlerContext) -> None:
        """Execute geology lookup before LLM."""
        if not self._location:
            return

        logger.info(f"GeologyHandler: Forcing Mapperr for '{self._location}'")

        try:
            result = query_geological_map(self._location)

            if result.get("success"):
                # Store result for final response
                ctx.forced_tool = "query_geological_map"
                ctx.forced_tool_result = result
                ctx.add_tool_used("query_geological_map")

                # Build analysis_result for frontend geology card
                ctx.analysis_result = {
                    "type": "geology",
                    "unit_code": result.get("unit_code"),
                    "unit_name": result.get("unit_name"),
                    "location": result.get("location"),
                    "lithology": result.get("lithology"),
                    "lithogenesis": result.get("lithogenesis"),
                    "morphology": result.get("morphology"),
                    "soil_types": result.get("soil_types", []),
                    "expected_conditions": result.get("expected_conditions", {}),
                    "map_segment": result.get("map_segment"),
                    "citation": result.get("citation"),
                }

                # Inject geology context into system prompt
                geo_context = "\n\nGEOLOGY DATA (from AGS maps - use this, don't make up geology):\n"
                geo_context += f"Location: {result.get('location')}\n"
                geo_context += f"Unit: {result.get('unit_code')} - {result.get('unit_name')}\n"
                geo_context += f"Lithology: {result.get('lithology', 'N/A')}\n"
                geo_context += f"Lithogenesis: {result.get('lithogenesis', 'N/A')}\n"
                geo_context += f"Soil Types: {', '.join(result.get('soil_types', []))}\n"

                ctx.inject_context(geo_context)

                logger.info(f"GeologyHandler: Injected {result.get('unit_code')} context")

        except Exception as e:
            logger.warning(f"GeologyHandler: Mapperr failed: {e}")

    def get_llm_config(self, ctx: HandlerContext) -> LLMConfig:
        """
        LLM config for geology queries.

        If we already have geology data, disable tools (just need formatting).
        """
        if ctx.forced_tool_result:
            return LLMConfig(
                tools_enabled=False,  # Already have the data
            )
        return LLMConfig(tools_enabled=True)

    def build_mode_hints(self, ctx: HandlerContext) -> str:
        """No special mode hints for geology."""
        return ""
