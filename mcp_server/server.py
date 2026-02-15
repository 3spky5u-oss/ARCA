"""
ARCA MCP Server — Model Context Protocol adapter for external AI clients.

Runs on the host machine as a subprocess spawned by an MCP-compatible client
(Claude Desktop, GPT desktop apps, custom agents, etc.).
Communicates over STDIO transport. Forwards tool calls to the ARCA
backend via HTTP (localhost:8000).

Tool discovery:
    At startup, queries GET /api/mcp/tools to discover all registered tools
    (core + domain-specific). Falls back to 6 hardcoded tools if the backend
    is unreachable. Polls every 60s for domain changes.

Environment variables:
    ARCA_URL      — Backend base URL (default: http://localhost:8000)
    ARCA_MCP_KEY  — API key matching MCP_API_KEY in ARCA's .env

CRITICAL: Never print() to stdout — it corrupts the STDIO protocol.
          Use logging (routes to stderr) for all debug output.
"""

import inspect
import os
import sys
import json
import logging
import threading
import time
from typing import Annotated, Literal

import httpx
from mcp.server.fastmcp import FastMCP

try:
    from pydantic import Field
    _HAS_PYDANTIC_FIELD = True
except ImportError:
    _HAS_PYDANTIC_FIELD = False

# Logging goes to stderr only (stdout is the MCP transport)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("arca-mcp")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARCA_URL = os.environ.get("ARCA_URL", "http://localhost:8000")
ARCA_MCP_KEY = os.environ.get("ARCA_MCP_KEY", "")

HEADERS = {"X-MCP-Key": ARCA_MCP_KEY, "Content-Type": "application/json"}
TIMEOUT = 30.0

# Track current domain so we can detect changes on poll
_current_domain: str = ""

mcp = FastMCP(
    "ARCA",
    instructions="Domain-aware AI RAG platform — search ingested knowledge, web, convert units, inspect corpus.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(method: str, path: str, body: dict | None = None) -> dict:
    """Make an HTTP call to the ARCA backend. Returns parsed JSON or error dict."""
    url = f"{ARCA_URL}{path}"
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            if method == "GET":
                resp = client.get(url, headers=HEADERS)
            else:
                resp = client.post(url, headers=HEADERS, json=body or {})
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            detail = e.response.text[:200]
        logger.error(f"ARCA {method} {path} -> {e.response.status_code}: {detail}")
        return {"error": f"ARCA returned {e.response.status_code}: {detail}"}
    except httpx.ConnectError:
        logger.error(f"Cannot connect to ARCA at {ARCA_URL}")
        return {"error": f"Cannot connect to ARCA backend at {ARCA_URL}. Is it running?"}
    except Exception as e:
        logger.error(f"ARCA call failed: {e}")
        return {"error": str(e)}


def _format_chunks(chunks: list) -> str:
    """Format RAG result chunks into readable text for the AI client."""
    if not chunks:
        return "No results found."
    lines = []
    for i, c in enumerate(chunks, 1):
        source = c.get("title") or c.get("source", "Unknown")
        page = c.get("page")
        score = c.get("score", 0)
        loc = f" (p. {page})" if page else ""
        lines.append(f"[{i}] {source}{loc}  (score: {score:.2f})")
        lines.append(c.get("content", "").strip())
        lines.append("")
    return "\n".join(lines)


def _format_tool_result(tool_name: str, data: dict) -> str:
    """Format a generic tool result into readable text.

    For tools that return chunks (RAG-style results), uses _format_chunks().
    For everything else, returns a JSON string.
    """
    if "error" in data:
        return f"Error: {data['error']}"

    # RAG-style results with chunks
    if "chunks" in data:
        chunks = data["chunks"]
        topics_searched = data.get("topics_searched", [])
        confidence = data.get("avg_confidence", 0)
        header = f"Found {len(chunks)} results"
        if topics_searched:
            header += f" across topics: {', '.join(topics_searched)}"
        if confidence:
            header += f" (avg confidence: {confidence:.2f})"
        return header + "\n\n" + _format_chunks(chunks)

    # Simple results with an expression (unit conversion)
    if "expression" in data:
        return data["expression"]

    # Fallback: JSON dump
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# JSON schema type -> Python type mapping for MCP tool parameters
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _param_type(param_def: dict):
    """Build Python type annotation from a JSON schema parameter definition.

    Handles:
    - Basic types via _TYPE_MAP (string->str, number->float, etc.)
    - Enum constraints via typing.Literal
    - Descriptions via typing.Annotated + pydantic.Field
    """
    if "enum" in param_def:
        base = Literal.__getitem__(tuple(param_def["enum"]))
    else:
        json_type = param_def.get("type", "string")
        base = _TYPE_MAP.get(json_type, str)

    desc = param_def.get("description")
    if desc and _HAS_PYDANTIC_FIELD:
        return Annotated[base, Field(description=desc)]
    return base


# ---------------------------------------------------------------------------
# Dynamic tool discovery
# ---------------------------------------------------------------------------

def _discover_tools() -> tuple[list[dict], str] | None:
    """Query backend for registered tools. Returns (tools, domain) or None on failure."""
    result = _call("GET", "/api/mcp/tools")
    if "error" in result:
        logger.warning(f"Tool discovery failed: {result['error']}")
        return None
    return result.get("tools", []), result.get("domain", "")


def _execute_via_backend(tool_name: str, **kwargs) -> str:
    """Generic executor: calls POST /api/mcp/execute with tool name and args."""
    # Strip None values so the backend uses its own defaults for omitted params
    clean_args = {k: v for k, v in kwargs.items() if v is not None}
    result = _call("POST", "/api/mcp/execute", {"tool": tool_name, "args": clean_args})
    return _format_tool_result(tool_name, result)


def _register_dynamic_tools() -> bool:
    """Discover tools from backend and register them with FastMCP.

    Returns True if dynamic registration succeeded, False if fallback needed.
    """
    global _current_domain

    discovery = _discover_tools()
    if discovery is None:
        return False

    tools, domain = discovery
    _current_domain = domain
    logger.info(f"Discovered {len(tools)} tools from backend (domain: {domain})")

    registered = 0
    for tool_def in tools:
        name = tool_def["name"]
        description = tool_def.get("description", "")
        params = tool_def.get("parameters", {})
        required = tool_def.get("required", [])

        try:
            _register_one_tool(name, description, params, required)
            registered += 1
        except Exception as e:
            logger.error(f"  Failed to register tool '{name}': {e}")

    logger.info(f"Successfully registered {registered}/{len(tools)} dynamic tools")
    return registered > 0


def _register_one_tool(name: str, description: str, params: dict, required: list) -> None:
    """Register a single tool with FastMCP using a closure over POST /api/mcp/execute.

    Builds a proper inspect.Signature so FastMCP generates correct MCP tool
    schemas with individual named parameters, types, enum constraints, and
    descriptions — instead of a single **kwargs parameter.
    """
    tool_name = name

    def _executor(**kwargs) -> str:
        return _execute_via_backend(tool_name, **kwargs)

    _executor.__name__ = tool_name
    _executor.__doc__ = description

    # Build inspect.Signature with proper Parameter objects.
    # FastMCP uses inspect.signature() to determine tool input schema —
    # __annotations__ alone is NOT sufficient (it sees **kwargs from the code).
    sig_params = []
    annotations = {}

    # Required parameters first (no default value)
    for param_name in required:
        if param_name not in params:
            continue
        py_type = _param_type(params[param_name])
        sig_params.append(
            inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=py_type)
        )
        annotations[param_name] = py_type

    # Optional parameters (default=None)
    for param_name, param_def in params.items():
        if param_name in required:
            continue
        py_type = _param_type(param_def)
        opt_type = py_type | None
        sig_params.append(
            inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default=None, annotation=opt_type)
        )
        annotations[param_name] = opt_type

    annotations["return"] = str
    _executor.__signature__ = inspect.Signature(sig_params, return_annotation=str)
    _executor.__annotations__ = annotations

    mcp.tool()(_executor)
    logger.info(f"  Registered dynamic tool: {tool_name}")


# ---------------------------------------------------------------------------
# Fallback: hardcoded tools (used when backend is unreachable at startup)
# ---------------------------------------------------------------------------

def _register_fallback_tools() -> None:
    """Register the 6 original hardcoded tools as fallback."""
    logger.warning("Backend unreachable — registering fallback hardcoded tools")

    @mcp.tool()
    def search_knowledge(query: str, topics: list[str] | None = None) -> str:
        """Search ARCA's ingested knowledge base using the hybrid RAG pipeline.

        This searches through all ingested technical documents (PDFs, textbooks, etc.)
        using dense embeddings, BM25 sparse retrieval, RAPTOR summaries, and GraphRAG,
        then reranks results for relevance.

        Args:
            query: Natural language search query (e.g. "bearing capacity of shallow foundations")
            topics: Optional list of topic names to restrict search (omit to search all enabled topics)
        """
        body: dict = {"query": query}
        if topics:
            body["topics"] = topics
        result = _call("POST", "/api/mcp/search", body)
        if "error" in result:
            return f"Search error: {result['error']}"
        chunks = result.get("chunks", [])
        topics_searched = result.get("topics_searched", [])
        confidence = result.get("avg_confidence", 0)
        header = f"Found {len(chunks)} results across topics: {', '.join(topics_searched)} (avg confidence: {confidence:.2f})\n\n"
        return header + _format_chunks(chunks)

    @mcp.tool()
    def web_search(query: str) -> str:
        """Search the web via SearXNG (privacy-respecting meta-search engine).

        Returns up to 5 results ranked by source quality, with academic and
        government sources prioritized.

        Args:
            query: Web search query
        """
        result = _call("POST", "/api/mcp/web-search", {"query": query})
        if "error" in result:
            return f"Web search error: {result['error']}"
        results = result.get("results", [])
        if not results:
            return "No web results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.get('title', 'Untitled')}")
            lines.append(f"    {r.get('url', '')}")
            content = r.get("content", "").strip()
            if content:
                lines.append(f"    {content}")
            lines.append("")
        return "\n".join(lines)

    @mcp.tool()
    def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
        """Convert between engineering and scientific units.

        Supports length, area, volume, pressure/stress, force, mass, density,
        permeability, flow rate, velocity, thermal, and temperature conversions.

        Args:
            value: Numeric value to convert
            from_unit: Source unit (e.g. "kpa", "ft", "psi", "kg/m3")
            to_unit: Target unit (e.g. "psf", "m", "bar", "pcf")
        """
        result = _call("POST", "/api/mcp/unit-convert", {
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
        })
        if "error" in result:
            return f"Conversion error: {result['error']}"
        return result.get("expression", f"{value} {from_unit} = {result.get('result', '?')} {to_unit}")

    @mcp.tool()
    def list_topics() -> str:
        """List all knowledge base topics and their enabled/disabled status.

        Topics correspond to ingested document collections (e.g. textbooks, manuals).
        Only enabled topics are searched by default.
        """
        result = _call("GET", "/api/mcp/topics")
        if "error" in result:
            return f"Error: {result['error']}"
        topics = result.get("topics", [])
        if not topics:
            return "No topics found. Ingest documents first."
        lines = [f"Knowledge base: {result.get('enabled_count', 0)} enabled / {result.get('total_count', 0)} total\n"]
        for t in topics:
            status = "enabled" if t.get("enabled") else "disabled"
            lines.append(f"  [{status:>8}] {t['name']}")
        return "\n".join(lines)

    @mcp.tool()
    def corpus_stats() -> str:
        """Get knowledge base statistics — collection sizes, chunk counts, file counts.

        Useful for understanding how much content has been ingested and where it lives.
        """
        result = _call("GET", "/api/mcp/stats")
        if "error" in result:
            return f"Error: {result['error']}"
        lines = [f"Total: {result.get('total_chunks', 0)} chunks from {result.get('total_files', 0)} files\n"]
        for coll in result.get("collections", []):
            lines.append(f"  {coll['name']}: {coll['chunks']} chunks")
        return "\n".join(lines)

    @mcp.tool()
    def system_status() -> str:
        """Check ARCA system health — LLM server, Redis, Qdrant, PostgreSQL status."""
        result = _call("GET", "/api/mcp/health")
        if "error" in result:
            return f"Error: {result['error']}"
        status = result.get("status", "unknown")
        checks = result.get("checks", {})
        lines = [f"ARCA status: {status}\n"]
        for component, state in checks.items():
            icon = "OK" if state == "ok" else "DOWN"
            lines.append(f"  {component}: {icon}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Domain change polling (background thread)
# ---------------------------------------------------------------------------

_POLL_INTERVAL = 60  # seconds

def _poll_domain_changes() -> None:
    """Background thread: polls /api/mcp/tools every 60s to detect domain changes.

    NOTE: FastMCP does not currently expose `notifications/tools/list_changed`
    from the MCP protocol, so we cannot push tool list updates to connected
    clients mid-session. When a domain change is detected, we log it and
    update the arca://domain resource. The client will see new tools on
    next session reconnect.
    """
    global _current_domain

    while True:
        time.sleep(_POLL_INTERVAL)
        try:
            result = _call("GET", "/api/mcp/tools")
            if "error" in result:
                continue
            new_domain = result.get("domain", "")
            if new_domain and new_domain != _current_domain:
                logger.info(
                    f"Domain changed: {_current_domain} -> {new_domain}. "
                    f"New tools will appear on next MCP session reconnect."
                )
                _current_domain = new_domain
        except Exception as e:
            logger.debug(f"Domain poll error: {e}")


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("arca://topics")
def resource_topics() -> str:
    """Knowledge base topic list."""
    result = _call("GET", "/api/mcp/topics")
    return json.dumps(result, indent=2)


@mcp.resource("arca://domain")
def resource_domain() -> str:
    """Active domain pack configuration."""
    result = _call("GET", "/api/domain")
    return json.dumps(result, indent=2)


@mcp.resource("arca://status")
def resource_status() -> str:
    """System health status."""
    # Use the public health endpoint (no MCP key needed)
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{ARCA_URL}/health")
            resp.raise_for_status()
            return json.dumps(resp.json(), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Infrastructure tools (always registered, independent of backend discovery)
# ---------------------------------------------------------------------------

def _register_infrastructure_tools() -> None:
    """Register MCP-specific infrastructure tools that don't come from the backend registry."""

    @mcp.tool()
    def upload_file(
        filename: str,
        file_content: str,
        file_path: str | None = None,
    ) -> str:
        """Upload a file to ARCA for analysis by other tools.

        Call this BEFORE tools that need files (analyze_files, process_field_logs,
        extract_borehole_log, redact_document, generate_openground).

        Supports: Excel (.xlsx/.xls/.csv), PDF, Word (.docx), text (.txt/.md),
        images (.png/.jpg). Max 50 MB.

        Provide EITHER:
        - file_content: base64-encoded file bytes (preferred for cross-container access)
        - file_path: absolute filesystem path (only works when ARCA can access the path)

        Args:
            filename: Original filename with extension (e.g. "lab_results.xlsx")
            file_content: Base64-encoded file content
            file_path: Absolute filesystem path (fallback, used only if file_content is empty)
        """
        import base64
        import mimetypes

        file_data = None

        # Primary: decode base64 content
        if file_content:
            try:
                file_data = base64.b64decode(file_content)
            except Exception as e:
                return f"Error: Invalid base64 content: {e}"

        # Fallback: read from filesystem path
        if file_data is None and file_path:
            path = os.path.expanduser(file_path)
            if not os.path.isfile(path):
                return f"Error: File not found: {path}"
            with open(path, "rb") as f:
                file_data = f.read()
            if not filename:
                filename = os.path.basename(path)

        if file_data is None:
            return "Error: Provide either file_content (base64) or file_path"

        if len(file_data) > 50 * 1024 * 1024:
            return f"Error: File too large ({len(file_data) / 1024 / 1024:.1f} MB, max 50 MB)"

        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        try:
            url = f"{ARCA_URL}/api/mcp/upload"
            with httpx.Client(timeout=TIMEOUT) as client:
                resp = client.post(
                    url,
                    headers={"X-MCP-Key": ARCA_MCP_KEY},
                    files={"file": (filename, file_data, content_type)},
                )
                resp.raise_for_status()
                result = resp.json()

            parts = [f"Uploaded: {result.get('filename', filename)}"]
            parts.append(f"File ID: {result.get('file_id', 'unknown')}")
            parts.append(f"Type: {result.get('file_type', 'unknown')}")
            parts.append(f"Size: {result.get('size', len(file_data))} bytes")

            samples = result.get("samples", 0)
            params = result.get("parameters", 0)
            if samples:
                parts.append(f"Lab data detected: {samples} samples, {params} parameters")

            return "\n".join(parts)

        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                detail = e.response.text[:200]
            return f"Upload failed ({e.response.status_code}): {detail}"
        except Exception as e:
            return f"Upload failed: {e}"

    logger.info("Registered infrastructure tool: upload_file")


# ---------------------------------------------------------------------------
# Startup: discover tools dynamically, fall back to hardcoded
# ---------------------------------------------------------------------------

try:
    if not _register_dynamic_tools():
        _register_fallback_tools()
except Exception as e:
    logger.error(f"Dynamic tool registration crashed: {e}")
    _register_fallback_tools()

# Always register MCP-specific infrastructure tools
_register_infrastructure_tools()

# Start background domain-change poller (daemon thread — dies with process)
_poll_thread = threading.Thread(target=_poll_domain_changes, daemon=True)
_poll_thread.start()
