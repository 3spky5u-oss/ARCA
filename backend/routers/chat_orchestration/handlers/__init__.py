"""
Query Handlers - Clean routing for different query types.

This module replaces ad-hoc if/else patterns in chat.py with a unified
handler pattern. Each handler knows how to detect and process its query type.

Architecture:
    QueryClassifier iterates handlers by priority, finds first match.
    Handler provides: pre_process, build_context, get_llm_config, post_process

Handler Priority (lower = higher priority):
    10  - GeologyHandler: Domain-specific location queries
    20  - VisionHandler: Pending image uploads
    30  - CodeHandler: Code generation requests
    40  - CalculateHandler: /calc or calculate keywords
    50  - ThinkHandler: /think or think keywords
    70  - TechnicalHandler: Auto-search for technical questions
    1000 - DefaultHandler: Everything else

FALLBACK: If ToolRouter selected a tool, handler uses that.
          If ToolRouter returned null/failed, handler uses pattern matching.
"""

from .base import QueryHandler, HandlerContext, LLMConfig
from .classifier import QueryClassifier, get_classifier
from .default import DefaultHandler
from .technical import TechnicalHandler
from .calculate import CalculateHandler
from .think import ThinkHandler

# Domain-gated: GeologyHandler requires mapperr (domain pack with mapping support)
try:
    from .geology import GeologyHandler
    _HAS_GEOLOGY = True
except ImportError:
    _HAS_GEOLOGY = False

__all__ = [
    "QueryHandler",
    "HandlerContext",
    "LLMConfig",
    "QueryClassifier",
    "get_classifier",
    "DefaultHandler",
    "TechnicalHandler",
    "CalculateHandler",
    "ThinkHandler",
]

if _HAS_GEOLOGY:
    __all__.append("GeologyHandler")
