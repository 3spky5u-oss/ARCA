"""
ARCA Chat Orchestration - Modular chat handler components

This module provides the orchestration layer for the chat WebSocket handler.
It breaks down the monolithic chat.py into focused, testable components.

Components:
- ChatSession: Conversation state management
- CitationCollector: Citation accumulation from tools
- AutoSearchManager: Technical question detection and auto-search
- ToolDispatcher: Tool parsing and execution coordination
- LLMOrchestrator: Two-call LLM pattern (tool call -> execute -> response)
- PhiiIntegration: Behavior enhancement facade
- ToolRouter: Fast tool routing using a small model (qwen2.5:1.5b)

Tool Router Architecture:
    The ToolRouter uses a small ~1.5B model to make fast (<200ms) decisions
    about which tool to call. This addresses the problem of qwen3:32b sometimes
    ignoring tools and hallucinating answers.

    FALLBACK LOGIC (important!):
    1. Router confidence < 0.7 → Let main LLM decide via tools schema
    2. Router timeout (>2s)    → Let main LLM decide via tools schema
    3. Router error            → Let main LLM decide via tools schema

    This ensures we never fail on the user - the small model tries first,
    but the smart model (qwen3:32b) is always there as a safety net.
"""

from .session import ChatSession
from .citations import CitationCollector
from .auto_search import AutoSearchManager
from .tool_dispatch import ToolDispatcher
from .orchestrator import LLMOrchestrator
from .phii_integration import PhiiIntegration
from .tool_router import ToolRouter, ToolRoutingDecision, get_tool_router

__all__ = [
    "ChatSession",
    "CitationCollector",
    "AutoSearchManager",
    "ToolDispatcher",
    "LLMOrchestrator",
    "PhiiIntegration",
    "ToolRouter",
    "ToolRoutingDecision",
    "get_tool_router",
]
