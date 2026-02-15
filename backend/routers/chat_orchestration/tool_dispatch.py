"""
ARCA Tool Dispatcher - Tool parsing and execution coordination

Handles:
- Parsing tool calls from LLM response (native and inline JSON fallback)
- N+1 cache optimization (reuse auto-search results)
- Tool execution coordination with progress updates
- Uses ToolRegistry metadata for dispatch decisions
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from fastapi import WebSocket

from ..chat_executors import execute_tool
from .citations import CitationCollector
from .auto_search import AutoSearchManager
from logging_config import log_tool
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _queries_similar(q1: str, q2: str, threshold: float = 0.6) -> bool:
    """Check if two queries are similar enough to reuse cached results.

    Uses Jaccard word overlap on normalized tokens.

    Args:
        q1: First query string
        q2: Second query string
        threshold: Minimum Jaccard similarity (0.0-1.0)

    Returns:
        True if queries have sufficient overlap
    """
    if not q1 or not q2:
        return False
    words1 = set(re.sub(r'[^\w\s]', '', q1.lower()).split())
    words2 = set(re.sub(r'[^\w\s]', '', q2.lower()).split())
    if not words1 or not words2:
        return False
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union) >= threshold


class ToolDispatcher:
    """Coordinates tool parsing and execution.

    Features:
    - Parses native LLM tool calls
    - Falls back to inline JSON parsing for models that emit JSON as text
    - Reuses auto-search results for N+1 optimization
    - Sends WebSocket progress updates
    - Collects citations from tool results
    """

    def __init__(
        self,
        files_db: Optional[Dict] = None,
        get_file_data_fn=None,
        update_file_analysis_fn=None,
        search_session_docs_fn=None,
    ):
        """Initialize dispatcher with dependency functions.

        Args:
            files_db: Files database dict
            get_file_data_fn: Function to get file data
            update_file_analysis_fn: Function to update file analysis
            search_session_docs_fn: Function to search session documents
        """
        self.files_db = files_db
        self.get_file_data_fn = get_file_data_fn
        self.update_file_analysis_fn = update_file_analysis_fn
        self.search_session_docs_fn = search_session_docs_fn

    def parse_tool_calls(self, response: Dict[str, Any]) -> Optional[List[Dict]]:
        """Parse tool calls from LLM response.

        First checks for native tool_calls, then falls back to inline JSON parsing.

        Args:
            response: LLM response dict

        Returns:
            List of tool call dicts, or None if no tool calls
        """
        message = response.get("message", {})
        tool_calls = message.get("tool_calls")

        if tool_calls:
            logger.debug(f"Found {len(tool_calls)} native tool calls")
            return tool_calls

        # Check for inline JSON tool calls
        content = message.get("content", "")
        logger.debug("No native tool_calls, checking content for inline JSON...")

        json_match = re.search(r'\{["\']name["\']\s*:\s*["\'](\w+)["\']', content, re.DOTALL)
        if json_match:
            logger.debug("Found JSON pattern, attempting to parse...")
            try:
                parsed = self._extract_json_object(content)
                if parsed and "name" in parsed:
                    tool_calls = [
                        {
                            "function": {
                                "name": parsed["name"],
                                "arguments": parsed.get("arguments", parsed.get("parameters", {})),
                            }
                        }
                    ]
                    logger.debug(f"Parsed inline tool call: {parsed['name']}")
                    return tool_calls
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Could not parse inline JSON: {e}")

        return None

    def _extract_json_object(self, content: str) -> Optional[Dict]:
        """Extract first complete JSON object from content.

        Args:
            content: Text potentially containing JSON

        Returns:
            Parsed JSON dict, or None if parsing fails
        """
        start = content.find("{")
        if start == -1:
            return None

        brace_count = 0
        end = start
        for i, c in enumerate(content[start:], start):
            if c == "{":
                brace_count += 1
            elif c == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        json_str = content[start:end]
        return json.loads(json_str)

    async def execute_tools(
        self,
        tool_calls: List[Dict],
        websocket: WebSocket,
        file_id: Optional[str],
        session,  # ChatSession (avoid circular import)
        citation_collector: CitationCollector,
        auto_search_manager: AutoSearchManager,
        deep_search: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict], List[str]]:
        """Execute tool calls and collect results.

        Args:
            tool_calls: List of tool call dicts from LLM
            websocket: WebSocket for progress updates
            file_id: Current file ID
            session: ChatSession for session context
            citation_collector: Collector for citations
            auto_search_manager: For N+1 cache check
            deep_search: Whether deep search mode is enabled

        Returns:
            Tuple of:
            - List of message dicts to append to conversation
            - analysis_result if any (for download buttons)
            - List of tool names used
        """
        messages_to_add = []
        analysis_result = None
        tools_used = []

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]

            # Parse args if string
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            # Log tool start with context
            log_ctx = self._build_log_context(tool_name, args)
            log_tool(logger, tool_name, "start", **log_ctx)
            await websocket.send_json({"type": "tool_start", "tool": tool_name})

            # Handle deep_search flag for web_search
            if tool_name == "web_search" and "deep" not in args and deep_search:
                args["deep"] = True

            # Pass profile="deep" to search_knowledge when deep_search is active
            if tool_name == "search_knowledge" and "profile" not in args and deep_search:
                args["profile"] = "deep"

            # N+1 optimization: reuse auto-search results if query matches
            if tool_name == "search_knowledge" and auto_search_manager.has_valid_cache():
                llm_query = args.get("query", "")
                cached_query = auto_search_manager.get_cached_query()
                if _queries_similar(llm_query, cached_query):
                    logger.info("Returning cached auto-search results (query match, skipping duplicate RAG call)")
                    result = auto_search_manager.get_cached_results()
                else:
                    logger.info(f"LLM reformulated query, executing new search (cached='{cached_query}', new='{llm_query}')")
                    result = execute_tool(
                        tool_name,
                        args,
                        file_id=file_id,
                        session=session,
                        files_db=self.files_db,
                        get_file_data_fn=self.get_file_data_fn,
                        update_file_analysis_fn=self.update_file_analysis_fn,
                        search_session_docs_fn=self.search_session_docs_fn,
                    )
            else:
                result = execute_tool(
                    tool_name,
                    args,
                    file_id=file_id,
                    session=session,
                    files_db=self.files_db,
                    get_file_data_fn=self.get_file_data_fn,
                    update_file_analysis_fn=self.update_file_analysis_fn,
                    search_session_docs_fn=self.search_session_docs_fn,
                )

            # Capture special results using registry metadata
            tool_def = ToolRegistry.get_tool(tool_name)

            if tool_def and result.get("success"):
                # Handle analysis results based on metadata
                if tool_def.triggers_analysis_result:
                    if tool_def.extracts_nested_result:
                        # Extract nested analysis_result (e.g., query_geological_map)
                        analysis_result = result.get("analysis_result", result)
                        logger.info(
                            f"[tool_dispatch] Captured nested analysis_result: type={analysis_result.get('type')}"
                        )
                    else:
                        analysis_result = result

                    # Handle session updates
                    if tool_def.updates_session and session:
                        session.update_from_analysis(args.get("soil_type", ""), args.get("land_use", ""))

            # Handle citations (web_search has special format)
            if tool_name == "web_search":
                citation_collector.add_from_web_search(result)
            elif tool_def and tool_def.provides_citations:
                citation_collector.add_from_rag(result)

            # Log tool completion
            result_ctx = self._build_result_context(result)
            log_tool(logger, tool_name, "end", **result_ctx)

            await websocket.send_json(
                {
                    "type": "tool_end",
                    "tool": tool_name,
                    "success": result.get("success", not result.get("error")),
                }
            )

            # Add to messages for LLM
            messages_to_add.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
            messages_to_add.append({"role": "tool", "content": json.dumps(result)})
            tools_used.append(tool_name)

        return messages_to_add, analysis_result, tools_used

    def _build_log_context(self, tool_name: str, args: Dict) -> Dict[str, str]:
        """Build context dict for tool start logging."""
        ctx = {}
        if "query" in args:
            query = str(args.get("query", ""))
            ctx["query"] = f'"{query[:40]}..."' if len(query) > 40 else f'"{query}"'
        elif "calculation" in args:
            ctx["calc"] = args["calculation"]
        elif "value" in args:
            ctx["value"] = f"{args['value']} {args.get('from_unit', '')} -> {args.get('to_unit', '')}"
        return ctx

    def _build_result_context(self, result: Dict) -> Dict[str, str]:
        """Build context dict for tool end logging."""
        ctx = {}
        if result.get("success"):
            if "results" in result:
                ctx["results"] = len(result["results"])
            elif "chunks" in result:
                ctx["results"] = len(result["chunks"])
            elif "download_url" in result:
                ctx["file"] = "ready"
        else:
            ctx["error"] = "true"
        return ctx
