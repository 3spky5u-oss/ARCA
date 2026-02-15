"""
ARCA Chat Executors - Unified Tool Dispatch

This module provides a unified interface for executing all chat tools.
It re-exports core executors and provides the main execute_tool() function.

Domain-specific executors are loaded dynamically via the domain pack's
register_tools module — they don't need to be imported here.

The execute_tool() function uses ToolRegistry for dispatch, eliminating the
need for a large if/elif chain. Tools register their executors via registry.py.
"""

import inspect
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

from tools.registry import ToolRegistry

# Re-export CORE executors only (domain executors loaded via domain pack)
from .calculations import execute_unit_convert
from .search import execute_web_search, execute_deep_web_search
from .rag import execute_search_knowledge, execute_search_session
from .documents import execute_redact_document
from .common import clean_citation_source

if TYPE_CHECKING:
    from ..chat import SessionContext

logger = logging.getLogger(__name__)


def _filter_kwargs_for_executor(executor, all_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to only those accepted by the executor function.

    This allows us to pass a unified context to all executors, and each
    executor only receives the parameters it actually accepts.
    """
    sig = inspect.signature(executor)
    accepts_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    if accepts_var_keyword:
        # Executor has **kwargs, pass everything
        return all_kwargs

    # Filter to only params the executor declares
    accepted_params = set(sig.parameters.keys())
    return {k: v for k, v in all_kwargs.items() if k in accepted_params}


def execute_tool(
    tool_name: str,
    args: Dict[str, Any],
    file_id: Optional[str] = None,
    session: Optional["SessionContext"] = None,
    files_db: Optional[Dict] = None,
    get_file_data_fn=None,
    update_file_analysis_fn=None,
    search_session_docs_fn=None,
) -> Dict[str, Any]:
    """
    Unified tool dispatch function using ToolRegistry.

    Routes tool calls through the registry pattern. Each executor receives
    only the kwargs it accepts (filtered via inspect).

    Args:
        tool_name: Name of the tool to execute
        args: Tool arguments
        file_id: Current file ID (for file-based tools)
        session: Session context (for session-aware tools)
        files_db: Files database dict (for file-based tools)
        get_file_data_fn: Function to get file data
        update_file_analysis_fn: Function to update file analysis
        search_session_docs_fn: Function to search session documents

    Returns:
        Tool result dictionary
    """
    tool_def = ToolRegistry.get_tool(tool_name)
    if not tool_def:
        return {"error": f"Unknown tool: {tool_name}"}

    # Build full context
    context = {
        "file_id": file_id,
        "session": session,
        "files_db": files_db,
        "get_file_data_fn": get_file_data_fn,
        "update_file_analysis_fn": update_file_analysis_fn,
        "search_session_docs_fn": search_session_docs_fn,
    }

    # Merge args and context, then filter to what executor accepts
    all_kwargs = {**args, **context}
    filtered_kwargs = _filter_kwargs_for_executor(tool_def.executor, all_kwargs)

    result = tool_def.executor(**filtered_kwargs)

    # Handle session updates via metadata
    if tool_def.updates_session and result.get("success") and session:
        session.update_from_analysis(args.get("soil_type", ""), args.get("land_use", ""))

    return result


__all__ = [
    # Main dispatch
    "execute_tool",
    # Core Calculations
    "execute_unit_convert",
    # Core Search
    "execute_web_search",
    "execute_deep_web_search",
    # Core RAG
    "execute_search_knowledge",
    "execute_search_session",
    # Core Documents
    "execute_redact_document",
    # Utilities
    "clean_citation_source",
]
