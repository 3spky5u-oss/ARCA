"""
ARCA Auto-Search Manager - Technical question detection and auto-search

Handles automatic knowledge base search for technical engineering questions.
Manages caching for N+1 optimization when LLM later calls search_knowledge.
"""

import logging
from typing import Dict, Any, Optional

from fastapi import WebSocket

from ..chat_prompts import is_technical_question
from ..chat_executors import execute_search_knowledge
from .citations import CitationCollector
from logging_config import log_tool

logger = logging.getLogger(__name__)


class AutoSearchManager:
    """Manages automatic knowledge base search for technical questions.

    Features:
    - Detects technical engineering questions using pattern matching
    - Executes search_knowledge with rerank=False for fast path
    - Caches results for N+1 optimization (skip duplicate RAG call)
    - Injects context into system prompt
    """

    def __init__(self):
        self._cached_results: Optional[Dict[str, Any]] = None
        self._cached_query: Optional[str] = None

    def should_auto_search(self, message: str, search_mode: bool) -> bool:
        """Determine if message should trigger auto-search.

        Args:
            message: User message
            search_mode: Whether search mode is enabled (skips auto-search)

        Returns:
            True if auto-search should be performed
        """
        if search_mode:
            return False
        return is_technical_question(message)

    async def execute_auto_search(
        self,
        query: str,
        websocket: WebSocket,
        citation_collector: CitationCollector,
    ) -> Optional[Dict[str, Any]]:
        """Execute auto-search and collect citations.

        Args:
            query: The search query (user message)
            websocket: WebSocket for progress updates
            citation_collector: Collector to add citations to

        Returns:
            Search results dict, or None if error
        """
        log_tool(logger, "search_knowledge", "start", mode="auto")
        await websocket.send_json({"type": "tool_start", "tool": "search_knowledge"})

        # Fast path: skip reranking for auto-search
        results = execute_search_knowledge(query=query, topics=None, rerank=False)

        results_count = len(results.get("chunks", []))
        log_tool(logger, "search_knowledge", "end", results=results_count)

        await websocket.send_json({"type": "tool_end", "tool": "search_knowledge", "success": not results.get("error")})

        # Collect citations
        citation_collector.add_from_rag(results)

        # Cache for N+1 optimization
        self._cached_results = results
        self._cached_query = query

        return results

    def get_cached_results(self) -> Optional[Dict[str, Any]]:
        """Get cached auto-search results for N+1 optimization.

        Returns:
            Cached results if available, None otherwise
        """
        return self._cached_results

    def get_cached_query(self) -> Optional[str]:
        """Get the query used for cached auto-search results.

        Returns:
            Cached query string if available, None otherwise
        """
        return self._cached_query

    def has_valid_cache(self) -> bool:
        """Check if there are valid cached results.

        Returns:
            True if cache has usable results
        """
        return bool(self._cached_results and self._cached_results.get("chunks"))

    def clear_cache(self) -> None:
        """Clear the cached results."""
        self._cached_results = None
        self._cached_query = None

    def inject_context(self, system_prompt: str, results: Optional[Dict[str, Any]] = None) -> str:
        """Inject auto-search results into system prompt.

        Args:
            system_prompt: Original system prompt
            results: Search results (uses cached if None)

        Returns:
            System prompt with knowledge context appended
        """
        results = results or self._cached_results
        if not results or not results.get("chunks"):
            return system_prompt

        knowledge_context = "\n\nKNOWLEDGE BASE RESULTS (use this to answer):\n"
        for r in results["chunks"][:5]:
            source = r.get("source", "Unknown")
            content = r.get("content", "")[:500]
            knowledge_context += f"- [{source}]: {content}\n"
        knowledge_context += "\nUse this information to answer. Cite the sources."

        return system_prompt + knowledge_context
