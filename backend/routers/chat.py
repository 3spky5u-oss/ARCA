"""
ARCA Chat Router - WebSocket Handler

LLM-powered junior engineer with tools. This module handles the WebSocket
endpoint and delegates orchestration to chat_orchestration/ modules.

Architecture:
- chat.py: WebSocket endpoint and main loop (~150 lines)
- chat_orchestration/: Modular components
  - session.py: ChatSession state management
  - citations.py: CitationCollector
  - auto_search.py: AutoSearchManager
  - tool_dispatch.py: ToolDispatcher
  - orchestrator.py: LLMOrchestrator
  - phii_integration.py: PhiiIntegration facade
- chat_prompts.py: System prompts, technical detection
- chat_streaming.py: Response streaming utilities
- chat_executors/: Tool execution modules
"""

import re
import secrets

import logging
import asyncio
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from errors.exceptions import LLMError
from errors.codes import ErrorCode
from logging_config import log_message_in, log_message_out, log_thinking

# Import prompts
from .chat_prompts import SYSTEM_PROMPT, cleanup_response_text, is_geology_query

# Import streaming helpers
from .chat_streaming import stream_response, build_final_response

# Import orchestration modules
from .chat_orchestration import (
    ChatSession,
    CitationCollector,
    AutoSearchManager,
    ToolDispatcher,
    LLMOrchestrator,
    PhiiIntegration,
    ToolRouter,
)

# Import handler system
from .chat_orchestration.handlers import (
    HandlerContext,
    get_classifier,
)

# Import from upload router
try:
    from .upload import get_file_data, files_db, update_file_analysis, search_session_docs, FileType
except ImportError:
    from .upload import get_file_data, files_db, update_file_analysis, FileType

    search_session_docs = None

# Import tool registry
from tools.registry import ToolRegistry, register_all_tools

import base64

# Runtime configuration
try:
    from config import runtime_config

    RUNTIME_CONFIG_AVAILABLE = True
except ImportError:
    runtime_config = None
    RUNTIME_CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter()

# Session save debounce tracking
_session_save_tasks: dict = {}


def _get_client_ip(websocket: WebSocket) -> str:
    """Extract client IP from WebSocket, handling proxies."""
    # Check X-Forwarded-For header
    forwarded = websocket.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    # Check X-Real-IP header
    real_ip = websocket.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct connection
    if websocket.client:
        return websocket.client.host

    return "unknown"

# Initialize tool registry at module load
register_all_tools()

BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "reports"
GUIDELINES_DIR = BASE_DIR / "guidelines"

# Security
MAX_MESSAGE_LENGTH = 4000  # ~1000 tokens, sufficient for engineering queries


def _sanitize_filename_for_prompt(name: str) -> str:
    """Sanitize user-controlled filename before injection into system prompt."""
    # Strip instruction-like patterns that could manipulate LLM behavior
    name = re.sub(
        r'\b(IGNORE|DISREGARD|FORGET)\s+(PREVIOUS|ALL|ABOVE)\s+\w+',
        '[REDACTED]', name, flags=re.IGNORECASE,
    )
    name = re.sub(
        r'\b(YOU ARE|ACT AS|BEHAVE LIKE|PRETEND|SYSTEM)\s+',
        '', name, flags=re.IGNORECASE,
    )
    # Cap display length
    return name[:120]


# Vision model from runtime config (imported in functions that need it)


def get_pending_images() -> list:
    """Get base64-encoded images from recently uploaded files.

    Returns list of base64 strings for images that haven't been processed yet.
    """
    images = []
    for file_id, file_data in files_db.items():
        if hasattr(file_data, "file_type") and file_data.file_type == FileType.IMAGE:
            # Check if this image has been processed (simple flag)
            if not file_data.metadata.get("processed"):
                try:
                    image_b64 = base64.b64encode(file_data.data).decode("utf-8")
                    images.append(
                        {
                            "file_id": file_id,
                            "filename": file_data.filename,
                            "data": image_b64,
                        }
                    )
                    # Mark as processed
                    file_data.metadata["processed"] = True
                    logger.info(f"Including image in message: {file_data.filename}")
                except Exception as e:
                    logger.error(f"Failed to encode image {file_data.filename}: {e}")
    return images


def build_context() -> str:
    """Build file context string for system prompt."""
    context_parts = []
    logger.info(f"build_context: files_db has {len(files_db)} entries")

    if files_db:
        context_parts.append(f"UPLOADED FILES ({len(files_db)}):")
        for file_id, file_data in files_db.items():
            if hasattr(file_data, "filename"):
                filename = _sanitize_filename_for_prompt(file_data.filename)
                samples = file_data.samples
                params = file_data.parameters if hasattr(file_data, "parameters") else 0
                file_type = file_data.file_type.value if hasattr(file_data, "file_type") else "unknown"
                rag_chunks = file_data.rag_chunks if hasattr(file_data, "rag_chunks") else 0
            else:
                filename = _sanitize_filename_for_prompt(file_data.get("filename", "Unknown"))
                samples = file_data.get("samples", 0)
                params = file_data.get("parameters", 0)
                file_type = "unknown"
                rag_chunks = 0
                if samples == 0 and file_data.get("lab_data"):
                    samples = len(file_data["lab_data"]) if file_data.get("lab_data") else 0

            logger.info(f"  File: {filename}, samples={samples}, type={file_type}")

            if samples > 0:
                context_parts.append(f"  - {filename}: {samples} samples, {params} parameters (lab data)")
            elif rag_chunks > 0:
                context_parts.append(f"  - {filename}: indexed for search ({rag_chunks} chunks)")
            else:
                context_parts.append(f"  - {filename}: {file_type}")

            analysis = file_data.analysis if hasattr(file_data, "analysis") else file_data.get("analysis")
            if analysis:
                exc_count = analysis.get("summary", {}).get("exceedance_count", 0)
                context_parts.append(f"    -> Analysis done: {exc_count} exceedances found")

        from domain_loader import get_pipeline_config
        _analysis_hint = get_pipeline_config().get("chat_analysis_hint", "use analyze_files tool")
        context_parts.append(f"\nTo analyze lab data: {_analysis_hint}")
        context_parts.append("To search uploaded docs: use search_session tool")
    else:
        context_parts.append("No files uploaded yet")

    return "\n".join(context_parts)


async def _debounced_session_save(session: "ChatSession", delay: float = 5.0):
    """Save session to Redis after a delay (debounced)."""
    try:
        await asyncio.sleep(delay)
        from services.session_cache import get_session_cache
        cache = await get_session_cache()
        await cache.save(session)
        logger.debug(f"Session {session.session_id} saved (debounced)")
    except asyncio.CancelledError:
        pass  # Normal cancellation, another save is coming
    except Exception as e:
        logger.warning(f"Debounced session save failed: {e}")


def _schedule_session_save(session: "ChatSession"):
    """Schedule a debounced session save."""
    global _session_save_tasks
    session_id = session.session_id

    # Cancel existing save task if any
    if session_id in _session_save_tasks:
        _session_save_tasks[session_id].cancel()

    # Schedule new save
    task = asyncio.create_task(_debounced_session_save(session))
    _session_save_tasks[session_id] = task

    # Auto-cleanup when task completes to prevent memory leak
    def _cleanup_task(t, sid=session_id):
        _session_save_tasks.pop(sid, None)

    task.add_done_callback(_cleanup_task)


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for chat."""
    await websocket.accept()

    # Rate limit check for WebSocket connections
    client_ip = _get_client_ip(websocket)
    try:
        from middleware.rate_limit import check_ws_connection_limit
        allowed, error_msg = await check_ws_connection_limit(client_ip)
        if not allowed:
            logger.warning(f"WS connection rate limited: {client_ip}")
            await websocket.send_json({"type": "error", "content": error_msg})
            await websocket.close(code=1008, reason="Rate limit exceeded")
            return
    except Exception as e:
        logger.debug(f"Rate limit check skipped: {e}")

    # Block chat when MCP mode is active
    if RUNTIME_CONFIG_AVAILABLE and runtime_config.mcp_mode:
        await websocket.send_json({
            "type": "error",
            "content": "ARCA is in MCP mode — serving as a tool backend for cloud AI models. "
            "Connect via Claude Desktop, GPT, or any MCP-compatible client. "
            "Disable MCP mode in the Admin panel to return to local chat.",
        })
        await websocket.close(code=1000, reason="MCP mode active")
        logger.info(f"Chat rejected (MCP mode): {client_ip}")
        return

    logger.info(f"Chat connected from {client_ip}")

    session_id = f"ws_{secrets.token_urlsafe(16)}"

    # Try to load existing session from Redis
    session = None
    try:
        from services.session_cache import get_session_cache
        cache = await get_session_cache()
        session = await cache.load(session_id)
        if session:
            logger.info(f"Session {session_id} restored from cache")
    except Exception as e:
        logger.debug(f"Session cache load skipped: {e}")

    # Create new session if not restored
    if session is None:
        session = ChatSession(session_id=session_id)

    # Initialize orchestration components
    phii = PhiiIntegration(runtime_config=runtime_config if RUNTIME_CONFIG_AVAILABLE else None)
    orchestrator = LLMOrchestrator(runtime_config=runtime_config if RUNTIME_CONFIG_AVAILABLE else None)
    auto_search = AutoSearchManager()
    tool_dispatcher = ToolDispatcher(
        files_db=files_db,
        get_file_data_fn=get_file_data,
        update_file_analysis_fn=update_file_analysis,
        search_session_docs_fn=search_session_docs,
    )

    # Initialize tool router (small model for fast routing decisions)
    # FALLBACK: If router fails/times out/low confidence, main LLM decides via tools schema
    tool_router = ToolRouter(runtime_config=runtime_config if RUNTIME_CONFIG_AVAILABLE else None)
    query_classifier = get_classifier()

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            search_mode = data.get("search_mode", False)
            deep_search = data.get("deep_search", False)
            think_mode = data.get("think_mode", False)
            calculate_mode = data.get("calculate_mode", False)  # qwq for math
            phii_enabled = data.get("phii_enabled", True)

            if not message.strip():
                continue

            # Block chat during active ingestion to protect VRAM
            if RUNTIME_CONFIG_AVAILABLE and runtime_config.ingest_active:
                await websocket.send_json({
                    "type": "response",
                    "content": "The system is currently ingesting documents into the knowledge base. "
                    "Chat is temporarily paused to dedicate GPU memory to the extraction pipeline. "
                    "Please try again in a few minutes — you can check progress in the Admin panel.",
                })
                await websocket.send_json({"type": "done"})
                continue

            if len(message) > MAX_MESSAGE_LENGTH:
                await websocket.send_json(
                    {"type": "error", "content": f"Message too long (max {MAX_MESSAGE_LENGTH:,} characters)"}
                )
                continue

            # Rate limit check for messages
            try:
                from middleware.rate_limit import check_ws_message_limit
                allowed, error_msg = await check_ws_message_limit(session_id)
                if not allowed:
                    await websocket.send_json({"type": "error", "content": error_msg})
                    continue
            except Exception as e:
                logger.debug(f"Message rate limit check skipped: {e}")

            # Detect mode prefixes in message (alternative to JSON flags)
            auto_think = False  # Track if think was auto-triggered by keywords
            auto_calculate = False  # Track if calculate was auto-triggered
            if message.strip().lower().startswith("/think "):
                think_mode = True
                message = message.strip()[7:]  # Remove /think prefix
            elif message.strip().lower().startswith("/calc ") or message.strip().lower().startswith("/calculate "):
                calculate_mode = True
                prefix_len = 6 if message.strip().lower().startswith("/calc ") else 11
                message = message.strip()[prefix_len:]
            elif message.strip().lower().startswith("/search "):
                search_mode = True
                message = message.strip()[8:]  # Remove /search prefix
            elif message.strip().lower().startswith("/deep "):
                deep_search = True
                message = message.strip()[6:]  # Remove /deep prefix

            # Auto-detect calculate mode from keywords (math-heavy requests)
            if not calculate_mode and not think_mode:
                calculate_keywords = [
                    "calculate",
                    "compute",
                    "derive",
                    "derivation",
                    "show the math",
                    "show your work",
                    "show calculations",
                    "step by step calculation",
                    "work out the numbers",
                    "solve for",
                    "find the value",
                    "what is the value of",
                    "how much is",
                    "determine the",
                ]
                message_lower = message.lower()
                for keyword in calculate_keywords:
                    if keyword in message_lower:
                        calculate_mode = True
                        auto_calculate = True
                        logger.info(f"Auto-calculate triggered by keyword: '{keyword}'")
                        break

            # Auto-detect think mode from keywords (reasoning/judgement)
            if not think_mode and not calculate_mode:
                think_keywords = [
                    "think through",
                    "think about this",
                    "think carefully",
                    "think deeply",
                    "think hard",
                    "think about",
                    "reason through",
                    "reason about",
                    "cogitate",
                    "ponder",
                    "analyze deeply",
                    "walk me through",
                    "explain your thinking",
                    "thorough analysis",
                    "detailed analysis",
                    "work through this",
                    "break down",
                ]
                message_lower = message.lower()
                for keyword in think_keywords:
                    if keyword in message_lower:
                        think_mode = True
                        auto_think = True
                        logger.info(f"Auto-think triggered by keyword: '{keyword}'")
                        break

            log_message_in(
                logger,
                message,
                search=search_mode,
                deep=deep_search,
                think=think_mode,
                calculate=calculate_mode,
                phii=phii_enabled,
            )

            is_intro_message = message.startswith("[INTRO")

            # Handle Phii escape commands
            if phii.available and phii_enabled and message.strip().startswith("/"):
                cmd_response = await phii.handle_escape_command(message, session_id)
                if cmd_response:
                    await websocket.send_json(
                        {
                            "type": "stream",
                            "content": cmd_response,
                            "done": True,
                            "tools_used": [],
                            "phii": {"command": True},
                        }
                    )
                    continue

            # Process implicit feedback from previous exchange
            if phii.available and phii_enabled and session.last_assistant_response:
                await phii.process_implicit_feedback(
                    current_message=message,
                    session_id=session_id,
                    previous_message_id=session.last_message_id,
                    previous_user_message=session.last_user_message,
                    previous_assistant_response=session.last_assistant_response,
                    tools_used=session.last_tools_used,
                )

            # === TOOL ROUTER + HANDLER CLASSIFICATION ===
            # Step 1: Get routing decision from small model (fast, ~150ms)
            # FALLBACK: If router fails/times out/low confidence, handlers use pattern matching
            router_decision = await tool_router.route(message, session, files_db)

            # Step 2: Build handler context
            handler_ctx = HandlerContext(
                message=message,
                session_id=session_id,
                router_decision=router_decision,
                search_mode=search_mode,
                deep_search=deep_search,
                think_mode=think_mode,
                calculate_mode=calculate_mode,
                phii_enabled=phii_enabled,
                auto_think=auto_think,
                auto_calculate=auto_calculate,
            )

            # Step 3: Classify query and get handler
            handler = query_classifier.classify(handler_ctx)

            # Step 4: Pre-process (handler may execute tools, inject context)
            await handler.pre_process(handler_ctx)

            # Step 5: Get mode hints from handler
            handler_mode_hints = handler.build_mode_hints(handler_ctx)

            # Update mode flags from handler context (handlers may auto-detect)
            think_mode = handler_ctx.think_mode
            calculate_mode = handler_ctx.calculate_mode
            auto_think = handler_ctx.auto_think
            auto_calculate = handler_ctx.auto_calculate

            # Build context and system prompt
            context = build_context()
            session_notes = session.get_notes_string()
            search_hint = ""
            if search_mode:
                search_hint = (
                    "\n\nDEEP SEARCH MODE ACTIVE: User wants comprehensive web search. Use the web_search tool with deep=true to find current information with expanded queries."
                    if deep_search
                    else "\n\nSEARCH MODE ACTIVE: User wants you to search the web. Use the web_search tool to find current information."
                )

            # Think mode: enhanced reasoning via prompt (uses qwen3 with prompting)
            think_hint = ""
            if think_mode:
                think_hint = """

THINK MODE ACTIVE: The user wants thorough, detailed analysis and reasoning. Provide comprehensive engineering judgement:

1. **Problem Decomposition**: Break the problem into clear components and state any assumptions
2. **Analysis**: Work through each component methodically, showing your reasoning
3. **Relevant Theory**: Reference applicable engineering principles or standards
4. **Trade-offs**: Consider different approaches, their pros/cons, and practical implications
5. **Edge Cases**: What could go wrong? Limiting factors? Special conditions?
6. **Practical Considerations**: Constructability, cost, common pitfalls
7. **Recommendations**: Clear conclusions with confidence levels

Format with headers and clear organization. Aim for thoroughness - this is think mode."""

            # Calculate mode: mathematical derivations (uses qwq for enhanced math reasoning)
            calculate_hint = ""
            if calculate_mode:
                calculate_hint = """

CALCULATE MODE ACTIVE: The user wants detailed mathematical derivations and step-by-step calculations. Show your work:

1. **Given Values**: List all input parameters with units
2. **Equations**: Write out the governing equations you'll use
3. **Substitution**: Show each step of plugging in values
4. **Intermediate Results**: Show intermediate calculations, don't skip steps
5. **Units**: Track units through calculations, verify dimensional consistency
6. **Final Answer**: Box or highlight the final result with appropriate significant figures
7. **Verification**: Quick sanity check - is the answer reasonable?

Use LaTeX formatting for equations. Be meticulous with the math - this is calculate mode."""

            # Combine mode hints for system prompt (include handler hints)
            mode_hints = search_hint + think_hint + calculate_hint + handler_mode_hints

            system_prompt, phii_metadata = phii.build_system_prompt(
                message=message,
                files_context=context,
                session_notes=session_notes,
                search_hint=mode_hints,  # Combined hints
                session_id=session_id,
                phii_enabled=phii_enabled,
                base_prompt=SYSTEM_PROMPT,
            )

            # Initialize citation tracking
            citation_collector = CitationCollector()

            # Auto-search for technical questions
            if auto_search.should_auto_search(message, search_mode):
                await auto_search.execute_auto_search(message, websocket, citation_collector)
                system_prompt = auto_search.inject_context(system_prompt)

            # Force Mapperr for geology queries - NOW HANDLED BY GeologyHandler
            # Check if handler already executed the tool
            forced_geology_result = None
            if handler_ctx.forced_tool == "query_geological_map" and handler_ctx.forced_tool_result:
                # Handler already did geology lookup
                forced_geology_result = handler_ctx.forced_tool_result
                # Inject handler's context into system prompt
                if handler_ctx.injected_context:
                    system_prompt = system_prompt.replace("{context}", handler_ctx.injected_context + "\n{context}")
                logger.info(f"Using handler's geology result: {forced_geology_result.get('unit_code')}")
            else:
                # Fallback: Old pattern-matching code (for backward compatibility)
                is_geo, geo_location = is_geology_query(message)
                if is_geo and geo_location:
                    logger.info(f"Forcing geology tool for location: {geo_location}")
                    try:
                        from tools.mapperr import query_geological_map

                        forced_geology_result = query_geological_map(geo_location)
                        if forced_geology_result.get("success"):
                            # Inject geology context into system prompt
                            geo_context = "\n\nGEOLOGY DATA (from AGS maps - use this, don't make up geology):\n"
                            geo_context += f"Location: {forced_geology_result.get('location')}\n"
                            geo_context += f"Unit: {forced_geology_result.get('unit_code')} - {forced_geology_result.get('unit_name')}\n"
                            geo_context += f"Lithology: {forced_geology_result.get('lithology', 'N/A')}\n"
                            geo_context += f"Lithogenesis: {forced_geology_result.get('lithogenesis', 'N/A')}\n"
                            geo_context += f"Soil Types: {', '.join(forced_geology_result.get('soil_types', []))}\n"
                            system_prompt = system_prompt.replace("{context}", geo_context + "\n{context}")
                            logger.info(f"Injected geology context: {forced_geology_result.get('unit_code')}")
                    except ImportError:
                        logger.debug("Mapperr not available (no domain pack with mapping support)")
                    except Exception as e:
                        logger.warning(f"Forced geology lookup failed: {e}")

            # Build messages for LLM
            messages = session.get_messages_for_llm(system_prompt, message)
            messages[0] = {"role": "system", "content": system_prompt}  # Update with potentially modified prompt

            # Check for pending images and use vision model
            pending_images = get_pending_images()
            vision_mode = len(pending_images) > 0

            if vision_mode:
                # Add images to the last user message
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        msg["images"] = [img["data"] for img in pending_images]
                        logger.info(f"Vision mode: Added {len(pending_images)} images to message")
                        break

            file_id = list(files_db.keys())[-1] if files_db else None
            code_mode = orchestrator.detect_code_request(message)
            # Allow safe tools in vision mode (search, guidelines, calculations)
            _VISION_SAFE_TOOLS = {
                "search_knowledge", "lookup_guideline", "unit_convert",
                "solve_engineering", "web_search", "query_geological_map",
            }
            if code_mode:
                use_tools = None
            elif vision_mode:
                all_tools = ToolRegistry.get_tools_schema()
                use_tools = [t for t in all_tools if t["function"]["name"] in _VISION_SAFE_TOOLS] or None
            else:
                use_tools = ToolRegistry.get_tools_schema()

            try:
                # Initial LLM call
                response = await orchestrator.initial_call(
                    messages=messages,
                    tools=use_tools,
                    websocket=websocket,
                    think_mode=think_mode,
                    calculate_mode=calculate_mode,
                    code_mode=code_mode,
                    vision_mode=vision_mode,
                    has_rag_context=auto_search.has_valid_cache(),
                )

                # Use handler's analysis_result and tools_used if available
                analysis_result = handler_ctx.analysis_result
                tools_used = handler_ctx.tools_used.copy()

                # Fallback: Build analysis_result from forced_geology_result (legacy path)
                if not analysis_result and forced_geology_result and forced_geology_result.get("success"):
                    try:
                        # Build analysis_result for frontend geology card
                        analysis_result = {
                            "type": "geology",
                            "unit_code": forced_geology_result.get("unit_code"),
                            "unit_name": forced_geology_result.get("unit_name"),
                            "location": forced_geology_result.get("location"),
                            "lithology": forced_geology_result.get("lithology"),
                            "lithogenesis": forced_geology_result.get("lithogenesis"),
                            "morphology": forced_geology_result.get("morphology"),
                            "soil_types": forced_geology_result.get("soil_types", []),
                            "expected_conditions": forced_geology_result.get("expected_conditions", {}),
                            "map_segment": forced_geology_result.get("map_segment"),
                            "citation": forced_geology_result.get("citation"),
                        }
                        if "query_geological_map" not in tools_used:
                            tools_used.append("query_geological_map")
                        logger.info(f"Using forced geology result: {analysis_result.get('unit_code')}")
                    except Exception as e:
                        logger.warning(f"Failed to build geology analysis result: {e}")

                # Parse and execute tool calls
                tool_calls = tool_dispatcher.parse_tool_calls(response)
                if tool_calls:
                    tool_messages, analysis_result, tools_used = await tool_dispatcher.execute_tools(
                        tool_calls=tool_calls,
                        websocket=websocket,
                        file_id=file_id,
                        session=session,
                        citation_collector=citation_collector,
                        auto_search_manager=auto_search,
                        deep_search=deep_search,
                    )
                    messages.extend(tool_messages)

                    # Final LLM call after tools
                    has_rag = any(t in {"search_knowledge", "search_session"} for t in tools_used)
                    response = await orchestrator.final_call(
                        messages=messages,
                        websocket=websocket,
                        think_mode=think_mode,
                        calculate_mode=calculate_mode,
                        code_mode=code_mode,
                        has_rag_context=has_rag or auto_search.has_valid_cache(),
                        tools_called=tools_used,
                    )

                # Extract and clean response
                response_text = response["message"].get("content", "")
                thinking_text = response["message"].get("thinking", "")
                thinking_for_display = None  # Separate thinking content for expandable UI

                if thinking_text:
                    log_thinking(logger, "end", chars=len(thinking_text))

                # Fallback: if content empty but thinking present, use thinking
                # (qwen3 with think=True may put response in thinking field)
                if not response_text and thinking_text:
                    logger.info("Content empty, falling back to thinking field")
                    response_text = thinking_text
                elif response_text and thinking_text:
                    # Content AND thinking both present - send thinking for expandable display
                    thinking_for_display = thinking_text

                # Extract <think> tags from qwen3 response BEFORE cleanup strips them
                # This lets us show qwen3's reasoning in the expandable view
                if not thinking_for_display and response_text:
                    think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
                    if think_match:
                        thinking_for_display = think_match.group(1).strip()
                        logger.info(f"Extracted <think> content: {len(thinking_for_display)} chars")

                response_text = cleanup_response_text(response_text)

                # Stream response
                if response_text:
                    delay_ms = (
                        runtime_config.stream_token_delay_ms if RUNTIME_CONFIG_AVAILABLE and runtime_config else 50
                    )
                    await stream_response(websocket, response_text, is_intro=is_intro_message, delay_ms=delay_ms)

                # Build and send final response
                final_response = build_final_response(
                    text=response_text,
                    tools_used=tools_used,
                    analysis_result=analysis_result,
                    citations=citation_collector.get_all(),
                    confidence=citation_collector.get_confidence(),
                    think_mode=think_mode,
                    auto_think=auto_think,
                    calculate_mode=calculate_mode,
                    auto_calculate=auto_calculate,
                    phii_metadata=phii_metadata,
                    thinking_content=thinking_for_display,
                )

                log_message_out(logger, tools_used=tools_used, citations=len(citation_collector))

                # Log full exchange for debugging/testing (tools already logged above)
                logger.info(f"[{session_id}] QUERY: {message}")
                logger.info(f"[{session_id}] RESPONSE: {response_text}")

                await websocket.send_json(final_response)

                # Record exchange for history and Phii tracking
                session.add_exchange(message, response_text, tools_used)

                # Observe for Phii learning
                if phii.available and phii_enabled:
                    await phii.observe_exchange(session_id, message, response_text, tools_used)

                # Schedule debounced session save to Redis
                _schedule_session_save(session)

                # Clear auto-search cache for next message
                auto_search.clear_cache()

            except WebSocketDisconnect:
                raise
            except LLMError as e:
                if e.code == ErrorCode.LLM_TIMEOUT:
                    logger.warning(f"LLM timeout: {e.message}")
                    error_msg = "The model is taking longer than expected. Please try a simpler query or try again."
                else:
                    logger.error(f"LLM error: {e}", exc_info=True)
                    error_msg = f"Model error: {str(e)}"
                try:
                    await websocket.send_json({"type": "error", "content": error_msg})
                except Exception as send_err:
                    logger.error(f"Failed to send LLM error to WebSocket: {send_err}", exc_info=True)
            except Exception as e:
                # Handle 503 "Loading model" gracefully — LLM still warming up
                err_str = str(e)
                if "503" in err_str and "Loading model" in err_str:
                    logger.warning("LLM still loading model — returning friendly message")
                    error_msg = "Still warming up — the model is loading into VRAM. Try again in a moment."
                else:
                    logger.error(f"Chat error: {e}", exc_info=True)
                    error_msg = f"Hit a snag: {err_str}"
                try:
                    await websocket.send_json({"type": "error", "content": error_msg})
                except Exception as send_err:
                    logger.error(f"Failed to send error to WebSocket: {send_err}", exc_info=True)

    except WebSocketDisconnect:
        logger.info(f"Chat disconnected: {session_id}")

        # Cancel pending debounced save
        if session_id in _session_save_tasks:
            _session_save_tasks[session_id].cancel()
            del _session_save_tasks[session_id]

        # Final session save to Redis
        try:
            from services.session_cache import get_session_cache
            cache = await get_session_cache()
            await cache.save(session)
            logger.debug(f"Session {session_id} saved on disconnect")
        except Exception as e:
            logger.warning(f"Session save on disconnect failed: {e}")
