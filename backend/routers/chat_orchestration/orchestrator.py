"""
ARCA LLM Orchestrator - Two-call LLM pattern management

Handles the orchestration of LLM calls:
1. Initial call with tools -> get tool calls
2. Execute tools
3. Final call -> get response

Also manages:
- Model selection (code mode, think mode)
- LLM options (context size, output tokens)
- Timeouts and error handling
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

from fastapi import WebSocket

from config import runtime_config
from errors.exceptions import LLMError
from logging_config import log_llm, log_thinking
from utils.llm import get_llm_client

logger = logging.getLogger(__name__)

# Retry settings for transient model loading errors
MODEL_RETRY_MAX = 2
MODEL_RETRY_DELAY = 3.0  # seconds


_PERMANENT_ERROR_PATTERNS = [
    "model not found",
    "does not exist",
    "invalid model",
]

_TRANSIENT_ERROR_PATTERNS = [
    "model is loading",
    "connection refused",
    "connection reset",
    "temporarily unavailable",
]


def is_retryable_error(error: Exception) -> bool:
    """Check if error is transient and worth retrying (not permanent failures)."""
    error_str = str(error).lower()
    # Never retry permanent errors
    if any(p in error_str for p in _PERMANENT_ERROR_PATTERNS):
        return False
    # Retry known transient errors
    return any(p in error_str for p in _TRANSIENT_ERROR_PATTERNS)


class _CircuitBreaker:
    """Prevents cascading failures when LLM service is down."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failures = 0
        self.threshold = failure_threshold
        self.timeout = recovery_timeout
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half_open

    def is_open(self) -> bool:
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
                return False
            return True
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.state = "closed"

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.threshold:
            self.state = "open"
            logger.error("Circuit breaker OPEN — LLM service unavailable")


_circuit_breaker = _CircuitBreaker()

def _get_chat_client():
    """Get LLM client for chat slot."""
    return get_llm_client("chat")


class LLMOrchestrator:
    """Orchestrates LLM calls with tool execution.

    Handles:
    - Model selection based on code/think mode
    - Context size calculation
    - Two-call pattern (tool call -> execute -> response)
    - Timeout management
    """

    def __init__(self, runtime_config=None):
        """Initialize orchestrator.

        Args:
            runtime_config: RuntimeConfig instance for dynamic settings
        """
        self.runtime_config = runtime_config
        self.client = _get_chat_client()

    def detect_code_request(self, message: str) -> bool:
        """Detect if message is a code generation request.

        Args:
            message: User message

        Returns:
            True if code generation is requested
        """
        if message.startswith("[SYSTEM:") or message.startswith("[INTRO"):
            return False
        code_indicators = [
            "write a script",
            "python code",
            "code for",
            "write code",
            "javascript",
            "typescript",
            "function to",
            "class for",
            "implement",
            "algorithm",
            "snippet",
            "program that",
        ]
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in code_indicators)

    def select_model(self, message: str, think_mode: bool = False) -> str:
        """Select appropriate model based on request type.

        Priority: code mode > think mode > default

        Models:
            - qwen3:32b: General chat (default)
            - qwen3-coder: Code generation
            - qwq:32b: Think mode (enhanced math/reasoning, supports tools)

        Args:
            message: User message
            think_mode: Whether think mode is enabled

        Returns:
            Model name to use
        """
        if self.detect_code_request(message):
            logger.info(f"Code request detected - using {runtime_config.model_code}")
            return runtime_config.model_code
        if think_mode:
            logger.info(f"Think mode enabled - using {runtime_config.model_expert} (expert - enhanced reasoning)")
            return runtime_config.model_expert
        return runtime_config.model_chat

    @staticmethod
    def estimate_tokens(messages: List[Dict], tools_schema: Optional[List] = None) -> int:
        """Estimate token count for messages + tools using char/token heuristic.

        Uses ~3.5 chars per token for English technical text (conservative).
        This helps detect when total prompt is close to num_ctx and
        prevents silent truncation.

        Args:
            messages: Chat messages list
            tools_schema: Optional tools JSON schema

        Returns:
            Estimated token count
        """
        CHARS_PER_TOKEN = 3.5
        total_chars = 0
        for msg in messages:
            total_chars += len(msg.get("content", "") or "")
            # Tool call results can be large
            if "tool_calls" in msg:
                total_chars += len(str(msg["tool_calls"]))
        if tools_schema:
            total_chars += len(str(tools_schema))
        return int(total_chars / CHARS_PER_TOKEN)

    def get_context_size(
        self,
        think_mode: bool = False,
        code_mode: bool = False,
        has_rag_context: bool = False,
        tool_calls: Optional[List[str]] = None,
    ) -> int:
        """Determine optimal context size based on task.

        Args:
            think_mode: Whether think mode is enabled
            code_mode: Whether code mode is enabled
            has_rag_context: Whether RAG context is present
            tool_calls: List of tools that were called

        Returns:
            Context size in tokens
        """
        if self.runtime_config:
            simple_tools_only = tool_calls and all(t in {"unit_convert", "lookup_guideline"} for t in tool_calls)
            return self.runtime_config.get_context_size(
                think_mode=think_mode,
                code_mode=code_mode,
                has_rag_context=has_rag_context,
                simple_tools_only=simple_tools_only,
            )

        # Fallback defaults (use runtime_config which is always available)
        if think_mode or code_mode:
            return runtime_config.ctx_xlarge
        if has_rag_context:
            return runtime_config.ctx_large
        simple_tools = {"unit_convert", "lookup_guideline"}
        if tool_calls and all(t in simple_tools for t in tool_calls):
            return runtime_config.ctx_small
        return runtime_config.ctx_medium

    def get_llm_options(
        self,
        think_mode: bool = False,
        code_mode: bool = False,
        has_rag_context: bool = False,
        tool_calls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get LLM options dict.

        Args:
            think_mode: Whether think mode is enabled
            code_mode: Whether code mode is enabled
            has_rag_context: Whether RAG context is present
            tool_calls: List of tools that were called

        Returns:
            Options dict with num_ctx, num_predict, and model params
        """
        ctx_size = self.get_context_size(
            think_mode=think_mode,
            code_mode=code_mode,
            has_rag_context=has_rag_context,
            tool_calls=tool_calls,
        )
        max_output = 4096
        if self.runtime_config:
            max_output = self.runtime_config.max_output_tokens

        options: Dict[str, Any] = {"num_ctx": ctx_size, "num_predict": max_output}

        # Merge model parameters from runtime config
        if self.runtime_config:
            options["temperature"] = self.runtime_config.temperature
            options["top_p"] = self.runtime_config.top_p
            options["top_k"] = self.runtime_config.top_k

        return options

    def get_timeout(self, think_mode: bool = False) -> int:
        """Get appropriate LLM timeout.

        Args:
            think_mode: Whether think mode is enabled

        Returns:
            Timeout in seconds
        """
        if self.runtime_config:
            return self.runtime_config.llm_timeout_think if think_mode else self.runtime_config.llm_timeout
        return 300 if think_mode else 180

    async def call_with_timeout(self, timeout_seconds: int, **kwargs) -> Dict[str, Any]:
        """Call LLM with timeout and retry on transient errors.

        Args:
            timeout_seconds: Maximum wait time
            **kwargs: Arguments for client.chat()

        Returns:
            Response dict

        Raises:
            LLMError: On timeout, ingest lock active, or other error
        """
        # Block model loading during active ingestion
        if runtime_config.ingest_active:
            raise LLMError(
                message="System is ingesting documents — chat model loading blocked to protect VRAM",
                error_type="ingest_lock",
                model=kwargs.get("model", "unknown"),
            )

        # Circuit breaker: fail fast if LLM is down
        if _circuit_breaker.is_open():
            raise LLMError(
                message="LLM service temporarily unavailable (circuit breaker open, retrying in 30s)",
                error_type="circuit_open",
                model=kwargs.get("model", "unknown"),
            )

        loop = asyncio.get_event_loop()
        model = kwargs.get("model", "unknown")
        last_error = None

        # Warn if estimated prompt tokens approach context limit
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools")
        est_tokens = self.estimate_tokens(messages, tools)
        num_ctx = kwargs.get("options", {}).get("num_ctx", 0)
        if num_ctx and est_tokens > num_ctx * 0.85:
            logger.warning(
                f"Prompt token estimate ({est_tokens}) is ≥85% of num_ctx ({num_ctx}). "
                "Response may be truncated."
            )

        for attempt in range(MODEL_RETRY_MAX + 1):
            start_time = time.time()

            if attempt > 0:
                delay = MODEL_RETRY_DELAY * (2 ** (attempt - 1))
                logger.info(f"Retry {attempt}/{MODEL_RETRY_MAX} for {model} after {delay:.1f}s")
                await asyncio.sleep(delay)

            log_llm(logger, "start", model=model)

            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self.client.chat(**kwargs)), timeout=timeout_seconds
                )
                duration = time.time() - start_time
                log_llm(logger, "end", model=model, duration=duration)
                _circuit_breaker.record_success()
                return response
            except asyncio.TimeoutError:
                duration = time.time() - start_time
                logger.warning(f"LLM call timed out after {duration:.2f}s (limit={timeout_seconds}s, model={model})")
                _circuit_breaker.record_failure()
                raise LLMError(
                    message=f"Model response timed out after {timeout_seconds}s",
                    error_type="timeout",
                    model=model,
                ) from None
            except Exception as e:
                last_error = e
                _circuit_breaker.record_failure()
                if is_retryable_error(e) and attempt < MODEL_RETRY_MAX:
                    logger.warning(f"Retryable error on {model}: {e}")
                    continue
                raise

        # Should not reach here, but safety fallback
        if last_error:
            raise last_error

    async def call_with_streaming(self, websocket: WebSocket, timeout_seconds: int, **kwargs) -> Dict[str, Any]:
        """Call LLM with streaming, sending thinking tokens via WebSocket.

        Includes retry logic for transient model loading errors on initial connection.

        Args:
            websocket: WebSocket for streaming thinking tokens
            timeout_seconds: Maximum wait time
            **kwargs: Arguments for client.chat()

        Returns:
            Assembled response dict with content and thinking

        Raises:
            LLMError: On timeout, ingest lock active, or other error
        """
        # Block model loading during active ingestion
        if runtime_config.ingest_active:
            raise LLMError(
                message="System is ingesting documents — chat model loading blocked to protect VRAM",
                error_type="ingest_lock",
                model=kwargs.get("model", "unknown"),
            )

        loop = asyncio.get_event_loop()
        model = kwargs.get("model", "unknown")

        # Force streaming on
        kwargs["stream"] = True

        accumulated_content = ""
        accumulated_thinking = ""
        last_error = None

        for attempt in range(MODEL_RETRY_MAX + 1):
            start_time = time.time()

            if attempt > 0:
                delay = MODEL_RETRY_DELAY * (2 ** (attempt - 1))
                logger.info(f"Retry {attempt}/{MODEL_RETRY_MAX} for streaming {model} after {delay:.1f}s")
                await asyncio.sleep(delay)
                # Reset accumulators on retry
                accumulated_content = ""
                accumulated_thinking = ""

            log_llm(logger, "start", model=model)

            try:
                # Run streaming in executor
                def stream_generator():
                    return self.client.chat(**kwargs)

                stream = await asyncio.wait_for(loop.run_in_executor(None, stream_generator), timeout=timeout_seconds)

                # Process streaming chunks
                for chunk in stream:
                    # Check timeout during streaming
                    if time.time() - start_time > timeout_seconds:
                        raise asyncio.TimeoutError()

                    content_chunk = chunk.get("message", {}).get("content", "")
                    thinking_chunk = chunk.get("message", {}).get("thinking", "")

                    # Accumulate content
                    if content_chunk:
                        accumulated_content += content_chunk

                    # Stream thinking tokens to frontend (filter system prompt leaks)
                    if thinking_chunk:
                        chunk_lower = thinking_chunk.lower()
                        if any(
                            ind in chunk_lower
                            for ind in ("system prompt", "you are arca", "you are an ai", "personality:", "instructions section")
                        ):
                            thinking_chunk = "[Internal reasoning...]"
                        accumulated_thinking += thinking_chunk
                        # Send thinking chunk via WebSocket
                        await websocket.send_json(
                            {
                                "type": "thinking_stream",
                                "content": thinking_chunk,
                            }
                        )

                duration = time.time() - start_time
                log_llm(logger, "end", model=model, duration=duration)

                # Return assembled response in same format as non-streaming
                return {
                    "message": {
                        "content": accumulated_content,
                        "thinking": accumulated_thinking,
                        "role": "assistant",
                    }
                }

            except asyncio.TimeoutError:
                duration = time.time() - start_time
                logger.warning(f"LLM streaming timed out after {duration:.2f}s (limit={timeout_seconds}s, model={model})")
                raise LLMError(
                    message=f"Model response timed out after {timeout_seconds}s",
                    error_type="timeout",
                    model=model,
                )
            except Exception as e:
                last_error = e
                if is_retryable_error(e) and attempt < MODEL_RETRY_MAX:
                    logger.warning(f"Retryable error on streaming {model}: {e}")
                    continue
                raise

        # Safety fallback
        if last_error:
            raise last_error

    async def initial_call(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        websocket: WebSocket,
        think_mode: bool = False,
        calculate_mode: bool = False,
        code_mode: bool = False,
        vision_mode: bool = False,
        has_rag_context: bool = False,
    ) -> Dict[str, Any]:
        """Make initial LLM call (may return tool calls).

        Args:
            messages: Message list for LLM
            tools: Tool definitions (None for code mode)
            websocket: WebSocket for progress updates
            think_mode: Whether think mode is enabled (qwen3 + prompts)
            calculate_mode: Whether calculate mode is enabled (qwq for math)
            code_mode: Whether code mode is enabled
            vision_mode: Whether vision model is needed (images attached)
            has_rag_context: Whether RAG context is present

        Returns:
            LLM response dict
        """
        # Model selection: vision > code > calculate > think/default
        # Vision mode uses qwen3-vl for image analysis
        # Think mode uses qwen3 with prompt enhancement
        # Calculate mode uses qwq for math reasoning
        if vision_mode:
            model = runtime_config.model_vision
            logger.info(f"Vision mode: using {model} for image analysis")
        elif code_mode:
            model = runtime_config.model_code
        elif calculate_mode:
            model = runtime_config.model_expert  # expert model for math
        else:
            model = runtime_config.model_chat  # qwen3 for think mode (with prompts) and default

        if calculate_mode:
            log_thinking(logger, "start")
            logger.info(f"Calculate mode: using {model} (qwq - math reasoning)")
        elif think_mode:
            log_thinking(logger, "start")
            logger.info(f"Think mode: using {model} (prompt-enhanced reasoning)")
        await websocket.send_json({"type": "tool_start", "tool": "thinking"})

        options = self.get_llm_options(
            think_mode=think_mode,
            code_mode=code_mode,
            has_rag_context=has_rag_context,
        )
        timeout = self.get_timeout(think_mode)

        # Use streaming for calculate mode to show live thinking tokens
        if calculate_mode:
            response = await self.call_with_streaming(
                websocket=websocket,
                timeout_seconds=timeout,
                model=model,
                messages=messages,
                tools=tools,
                options=options,
            )
        else:
            # BENCHMARK FINDING (2026-02-03):
            # Prompt-based thinking for think_mode quality
            response = await self.call_with_timeout(
                timeout_seconds=timeout,
                model=model,
                messages=messages,
                tools=tools,
                stream=False,
                options=options,
            )

        await websocket.send_json({"type": "tool_end", "tool": "thinking"})
        return response

    async def final_call(
        self,
        messages: List[Dict],
        websocket: WebSocket,
        think_mode: bool = False,
        calculate_mode: bool = False,
        code_mode: bool = False,
        vision_mode: bool = False,
        has_rag_context: bool = False,
        tools_called: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Make final LLM call after tool execution.

        Args:
            messages: Message list including tool results
            websocket: WebSocket for progress updates
            think_mode: Whether think mode is enabled (qwen3 + prompts)
            calculate_mode: Whether calculate mode is enabled (qwq)
            code_mode: Whether code mode is enabled
            vision_mode: Whether vision model is needed (images attached)
            has_rag_context: Whether RAG context is present
            tools_called: List of tools that were called

        Returns:
            LLM response dict
        """
        # Model selection: vision > code > calculate > think/default
        if vision_mode:
            model = runtime_config.model_vision
        elif code_mode:
            model = runtime_config.model_code
        elif calculate_mode:
            model = runtime_config.model_expert  # expert model for math
        else:
            model = runtime_config.model_chat  # qwen3 for think (with prompts) and default

        await websocket.send_json({"type": "tool_start", "tool": "thinking"})

        options = self.get_llm_options(
            think_mode=think_mode,
            code_mode=code_mode,
            has_rag_context=has_rag_context,
            tool_calls=tools_called,
        )
        timeout = self.get_timeout(think_mode)

        # Use streaming for calculate mode to show live thinking tokens
        if calculate_mode:
            response = await self.call_with_streaming(
                websocket=websocket,
                timeout_seconds=timeout,
                model=model,
                messages=messages,
                options=options,
            )
        else:
            # Prompt-based thinking for think_mode quality
            # <think> tags are extracted in chat.py for expandable view
            response = await self.call_with_timeout(
                timeout_seconds=timeout,
                model=model,
                messages=messages,
                stream=False,
                options=options,
            )

        await websocket.send_json({"type": "tool_end", "tool": "thinking"})
        return response
