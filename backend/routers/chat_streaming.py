"""
ARCA Chat Streaming - Response streaming utilities

Helper functions for streaming responses and building final messages.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


async def stream_response(
    websocket: WebSocket,
    text: str,
    is_intro: bool = False,
    delay_ms: int = 50,
    min_delay_ms: int = 50
) -> None:
    """Stream text word-by-word with optional delay for typewriter effect.

    Args:
        websocket: WebSocket connection to stream to
        text: Full text to stream
        is_intro: If True, applies delay for typewriter effect
        delay_ms: Delay between tokens for intro/regular messages (consistent rate)
        min_delay_ms: Minimum delay for regular messages (matches intro for consistency)
    """
    if not text:
        return

    tokens = text.split()
    total_chars = len(text)
    start_time = time.time()

    for i, token in enumerate(tokens):
        # Add space after token (except for last token which will be in final message)
        await websocket.send_json(
            {"type": "stream", "content": token + (" " if i < len(tokens) - 1 else ""), "done": False}
        )
        # Apply streaming delay
        if is_intro:
            await asyncio.sleep(delay_ms / 1000.0)
        elif min_delay_ms > 0:
            # Small delay for regular streaming to help frontend chunk animation
            await asyncio.sleep(min_delay_ms / 1000.0)

    # Log streaming stats
    elapsed = time.time() - start_time
    if elapsed > 0:
        tok_per_sec = len(tokens) / elapsed
        char_per_sec = total_chars / elapsed
        logger.info(
            f"[STREAM] {len(tokens)} tokens, {total_chars} chars in {elapsed:.2f}s "
            f"({tok_per_sec:.0f} tok/s, {char_per_sec:.0f} char/s)"
        )


def build_final_response(
    text: str,
    tools_used: List[str],
    analysis_result: Optional[Dict] = None,
    citations: Optional[List[Dict]] = None,
    confidence: Optional[float] = None,
    think_mode: bool = False,
    auto_think: bool = False,
    calculate_mode: bool = False,
    auto_calculate: bool = False,
    phii_metadata: Optional[Dict] = None,
    thinking_content: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the final response message with all metadata.

    Args:
        text: Full response text
        tools_used: List of tool names that were called
        analysis_result: Optional analysis result (for download buttons)
        citations: Optional list of citations from RAG tools
        confidence: Optional confidence score
        think_mode: Whether think mode was enabled (qwen3 + prompts)
        auto_think: Whether think was auto-triggered by keywords
        calculate_mode: Whether calculate mode was enabled (qwq for math)
        auto_calculate: Whether calculate was auto-triggered by keywords
        phii_metadata: Optional Phii learning metadata (corrections, expertise, etc.)
        thinking_content: Optional LLM thinking/reasoning tokens for expandable display

    Returns:
        Final response dict ready to send via WebSocket
    """
    response = {
        "type": "stream",
        "content": text,  # Full content for final markdown render
        "done": True,
        "tools_used": tools_used,
        "analysis_result": analysis_result,
        "think_mode": think_mode,  # For purple styling in frontend
        "auto_think": auto_think,  # For "auto-enabled" notice in frontend
        "calculate_mode": calculate_mode,  # For green styling in frontend
        "auto_calculate": auto_calculate,  # For "auto-enabled" notice
    }

    # Debug logging for analysis_result
    if analysis_result:
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"[build_final_response] Sending analysis_result with type={analysis_result.get('type')}")

    # Include citations if any were collected from RAG tools
    if citations:
        response["citations"] = citations
    if confidence is not None:
        response["confidence"] = confidence

    # Include Phii learning metadata if available
    if phii_metadata:
        response["phii"] = phii_metadata

    # Include thinking content for expandable display
    if thinking_content:
        response["thinking_content"] = thinking_content

    return response
