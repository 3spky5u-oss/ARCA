"""
LLM Client — wraps OpenAI SDK to talk to llama-server.

Response format:
    {"message": {"content": "...", "thinking": "...", "tool_calls": [...]}}

Key translations:
- Vision: images:[b64] → OpenAI content:[{type:"image_url",...}]
- Streaming: ChatCompletionChunk → {"message": {"content": chunk}}
- Thinking: <think>...</think> inline tags → separate "thinking" field
- Tool calls: OpenAI objects → simplified dicts
- Options: num_predict→max_tokens, num_ctx→ignored (set at server startup)
"""

import json
import logging
import re
from typing import Any, Dict, Generator, List, Optional

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)


def _translate_messages_for_openai(messages: List[Dict]) -> List[Dict]:
    """Translate internal message format to OpenAI API format.

    Handles vision images, tool call results, and regular messages.
    """
    translated = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        images = msg.get("images", [])

        # Vision messages with base64 images
        if images:
            parts = []
            if content:
                parts.append({"type": "text", "text": content})
            for img_b64 in images:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
            translated.append({"role": role, "content": parts})

        # Tool call results
        elif role == "tool":
            translated.append({
                "role": "tool",
                "content": content if isinstance(content, str) else json.dumps(content),
                "tool_call_id": msg.get("tool_call_id", "call_0"),
            })

        # Regular messages
        else:
            new_msg = {"role": role, "content": content}

            # Forward tool_calls from assistant messages
            if role == "assistant" and "tool_calls" in msg:
                openai_tool_calls = []
                for i, tc in enumerate(msg["tool_calls"]):
                    fn = tc.get("function", tc)
                    openai_tool_calls.append({
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": (
                                json.dumps(fn["arguments"])
                                if isinstance(fn.get("arguments"), dict)
                                else fn.get("arguments", "{}")
                            ),
                        },
                    })
                new_msg["tool_calls"] = openai_tool_calls
                # OpenAI requires content to be None when tool_calls present
                if not content:
                    new_msg["content"] = None

            translated.append(new_msg)

    return translated


def _translate_tools_for_openai(tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """Pass through tool definitions (already OpenAI-compatible)."""
    if not tools:
        return None
    return tools


def _extract_thinking(content: str) -> tuple:
    """Extract <think>...</think> tags from content.

    llama-server returns thinking inline in content as <think> tags.

    Returns:
        (clean_content, thinking_text)
    """
    if not content:
        return "", ""

    # Find all think blocks
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinking_parts = think_pattern.findall(content)
    thinking = "\n".join(thinking_parts).strip()

    # Remove think tags from content
    clean = think_pattern.sub("", content).strip()
    return clean, thinking


def _translate_tool_calls_from_openai(choices) -> Optional[List[Dict]]:
    """Translate OpenAI tool call objects to simplified dicts.

    OpenAI: choice.message.tool_calls[i].function.{name, arguments(str)}
    Internal: [{"function": {"name": ..., "arguments": {dict}}}]
    """
    if not choices:
        return None

    message = choices[0].message
    if not message.tool_calls:
        return None

    result = []
    for tc in message.tool_calls:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call arguments: {tc.function.arguments}")
            args = {}

        result.append({
            "function": {
                "name": tc.function.name,
                "arguments": args,
            },
            "id": tc.id,
        })

    return result if result else None


class LLMClient:
    """Wraps OpenAI SDK pointing at a llama-server instance."""

    def __init__(self, base_url: str, timeout: float = 180.0):
        """
        Args:
            base_url: llama-server URL (e.g., "http://localhost:8081")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._openai = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="not-needed",  # llama-server doesn't require auth
            timeout=timeout,
        )

    def is_healthy(self, timeout: float = 3.0) -> bool:
        """Sync health check against llama-server /health endpoint."""
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def chat(
        self,
        model: str = "",
        messages: List[Dict] = None,
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        options: Optional[Dict] = None,
        think: bool = False,
        keep_alive: Any = None,
        format: str = None,
    ) -> Any:
        """Call LLM chat endpoint.

        Args:
            model: Model name (ignored - slot determines model)
            messages: List of message dicts
            tools: Tool definitions
            stream: Whether to stream response
            options: Generation options (num_predict, temperature, etc.)
            think: Ignored (thinking handled by server config)
            keep_alive: Ignored (handled by process manager)
            format: Response format ("json" for JSON mode)

        Returns:
            If stream=False: dict with "message" key
            If stream=True: generator yielding chunk dicts
        """
        messages = messages or []
        options = options or {}

        # Translate messages and tools
        openai_messages = _translate_messages_for_openai(messages)
        openai_tools = _translate_tools_for_openai(tools)

        # Translate options
        kwargs = {
            "model": model or "default",
            "messages": openai_messages,
        }

        # Map options to OpenAI params
        if "temperature" in options:
            kwargs["temperature"] = options["temperature"]
        if "top_p" in options:
            kwargs["top_p"] = options["top_p"]
        if "num_predict" in options:
            kwargs["max_tokens"] = options["num_predict"]
        elif "max_tokens" in options:
            kwargs["max_tokens"] = options["max_tokens"]

        # num_ctx is ignored (set at server startup)
        # top_k is not supported by OpenAI API

        if openai_tools:
            kwargs["tools"] = openai_tools

        if format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        if stream:
            return self._stream_chat(**kwargs)

        # Non-streaming call
        response = self._openai.chat.completions.create(stream=False, **kwargs)

        # Extract content and thinking
        raw_content = response.choices[0].message.content or ""
        content, thinking = _extract_thinking(raw_content)

        # Extract tool calls
        tool_calls = _translate_tool_calls_from_openai(response.choices)

        result = {
            "message": {
                "role": "assistant",
                "content": content,
            }
        }

        if thinking:
            result["message"]["thinking"] = thinking
        if tool_calls:
            result["message"]["tool_calls"] = tool_calls

        return result

    def _stream_chat(self, **kwargs) -> Generator[Dict, None, None]:
        """Stream chat response, yielding chunk dicts.

        Handles <think> tag extraction from streaming content with a
        state machine that tracks whether we're inside a think block.
        """
        stream = self._openai.chat.completions.create(stream=True, **kwargs)

        # State machine for think tag extraction
        in_think = False
        buffer = ""  # Buffer for partial tag detection
        tag_open = "<think>"
        tag_close = "</think>"

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            content_chunk = delta.content or ""

            if not content_chunk:
                # Check for tool calls in stream
                if delta.tool_calls:
                    # Tool calls come at end of stream, yield as final chunk
                    tool_calls = []
                    for tc in delta.tool_calls:
                        if tc.function:
                            try:
                                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                            except json.JSONDecodeError:
                                args = {}
                            tool_calls.append({
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": args,
                                },
                                "id": tc.id,
                            })
                    if tool_calls:
                        yield {"message": {"tool_calls": tool_calls}, "done": True}
                continue

            # Buffer content for tag detection
            buffer += content_chunk

            # Process buffer
            while buffer:
                if in_think:
                    # Look for closing tag
                    close_idx = buffer.find(tag_close)
                    if close_idx >= 0:
                        # Found closing tag - emit thinking up to it
                        thinking_text = buffer[:close_idx]
                        if thinking_text:
                            yield {
                                "message": {"thinking": thinking_text, "content": ""},
                                "done": False,
                            }
                        buffer = buffer[close_idx + len(tag_close):]
                        in_think = False
                    elif len(buffer) > len(tag_close):
                        # No closing tag yet, emit buffered thinking (keep tail for partial match)
                        safe = buffer[:-(len(tag_close) - 1)]
                        buffer = buffer[len(safe):]
                        if safe:
                            yield {
                                "message": {"thinking": safe, "content": ""},
                                "done": False,
                            }
                    else:
                        break  # Wait for more data
                else:
                    # Look for opening tag
                    open_idx = buffer.find(tag_open)
                    if open_idx >= 0:
                        # Found opening tag - emit content before it
                        before = buffer[:open_idx]
                        if before:
                            yield {
                                "message": {"content": before},
                                "done": False,
                            }
                        buffer = buffer[open_idx + len(tag_open):]
                        in_think = True
                    elif len(buffer) > len(tag_open):
                        # No opening tag, emit safe content
                        safe = buffer[:-(len(tag_open) - 1)]
                        buffer = buffer[len(safe):]
                        if safe:
                            yield {
                                "message": {"content": safe},
                                "done": False,
                            }
                    else:
                        break  # Wait for more data

        # Flush remaining buffer
        if buffer:
            if in_think:
                yield {"message": {"thinking": buffer, "content": ""}, "done": False}
            else:
                yield {"message": {"content": buffer}, "done": False}

        # Final done signal
        yield {"message": {"content": ""}, "done": True}

    def generate(
        self,
        model: str = "",
        prompt: str = "",
        stream: bool = False,
        options: Optional[Dict] = None,
        keep_alive: Any = None,
    ) -> Dict[str, Any]:
        """Simple text generation (used by HyDE, warmup, etc.).

        Uses /v1/completions (text completion) instead of /v1/chat/completions
        to avoid thinking-mode models putting all output into reasoning_content
        and returning empty content.

        Args:
            model: Model name (ignored - slot determines model)
            prompt: Text prompt
            stream: Streaming (not supported for generate)
            options: Generation options
            keep_alive: Ignored

        Returns:
            Dict with "response" key containing generated text
        """
        options = options or {}

        payload: Dict[str, Any] = {
            "model": model or "default",
            "prompt": prompt,
        }

        if "temperature" in options:
            payload["temperature"] = options["temperature"]
        if "num_predict" in options:
            payload["max_tokens"] = options["num_predict"]
        if "stop" in options:
            payload["stop"] = options["stop"]

        resp = httpx.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0].get("text", "")

        return {"response": content}

    def list(self) -> Dict[str, Any]:
        """List available models (compatibility shim).

        Returns empty list - use server_manager.list_running() instead.
        """
        return {"models": []}

    def ps(self) -> Dict[str, Any]:
        """List running models (compatibility shim).

        Returns empty list - use server_manager.list_running() instead.
        """
        return {"models": []}
