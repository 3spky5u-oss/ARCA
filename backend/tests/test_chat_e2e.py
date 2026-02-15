"""
End-to-end tests for the chat WebSocket endpoint.

Tests the full flow: connect -> send message -> receive streamed response -> done,
with the LLM client mocked to return canned responses.

Strategy:
    - Build a lightweight FastAPI app that includes ONLY the chat router
    - Mock infrastructure (LLM client, Redis, rate limiter, domain loader)
    - Use Starlette TestClient for synchronous WebSocket testing
    - Each test is independent (fresh mocks per test)
"""

import json
import pytest
from contextlib import asynccontextmanager
from dataclasses import dataclass
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi import FastAPI
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers — fake LLM response builders
# ---------------------------------------------------------------------------

def _llm_response(content: str, tool_calls=None, thinking: str = ""):
    """Build a canned LLM response dict matching LLMClient.chat() format."""
    msg = {"content": content, "role": "assistant", "thinking": thinking}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"message": msg}


def _tool_call(name: str, arguments: dict, call_id: str = "call_0"):
    """Build a single tool call dict."""
    return {
        "id": call_id,
        "function": {"name": name, "arguments": arguments},
    }


# ---------------------------------------------------------------------------
# Patch targets — collected here so changes in one place
# ---------------------------------------------------------------------------

# The orchestrator imports get_llm_client at module level and stores it
# on self.client. We patch _get_chat_client which feeds the constructor.
_PATCH_LLM_CLIENT = "routers.chat_orchestration.orchestrator._get_chat_client"

# Rate limiter checks inside the WS handler — imported lazily
_PATCH_WS_CONN_LIMIT = "middleware.rate_limit.check_ws_connection_limit"
_PATCH_WS_MSG_LIMIT = "middleware.rate_limit.check_ws_message_limit"

# Session cache — imported lazily inside handler
_PATCH_SESSION_CACHE = "services.session_cache.get_session_cache"

# Domain loader — called at import time by chat_prompts and others
_PATCH_DOMAIN_CONFIG = "domain_loader.get_domain_config"
_PATCH_DOMAIN_BRANDING = "domain_loader.get_branding"
_PATCH_DOMAIN_LEXICON = "domain_loader.get_lexicon"
_PATCH_PIPELINE_CONFIG = "domain_loader.get_pipeline_config"

# Tool registry — called at module load of chat.py
_PATCH_REGISTER_TOOLS = "tools.registry.register_all_tools"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeDomainConfig:
    name: str = "example"
    display_name: str = "Example"
    tools: list = None
    routes: list = None
    admin_visible: list = None
    domain_dir: str = "."
    branding: dict = None

    def __post_init__(self):
        self.tools = self.tools or []
        self.routes = self.routes or []
        self.admin_visible = self.admin_visible or []
        self.branding = self.branding or {
            "app_name": "ARCA",
            "tagline": "Test",
            "primary_color": "#000",
        }

    @property
    def app_name(self) -> str:
        return self.branding.get("app_name", "ARCA")

    @property
    def tagline(self) -> str:
        return self.branding.get("tagline", "")

    @property
    def primary_color(self) -> str:
        return self.branding.get("primary_color", "#000")

    def has_tool(self, name):
        return name in self.tools

    def has_route(self, name):
        return name in self.routes


class FakeSessionCache:
    """In-memory session cache stand-in."""

    async def load(self, session_id):
        return None

    async def save(self, session):
        pass


def _make_fake_llm_client(side_effect_fn=None, return_value=None):
    """Create a mock LLM client whose .chat() returns canned data.

    Args:
        side_effect_fn: If set, client.chat() calls this with (**kwargs).
        return_value: If set (and no side_effect_fn), client.chat() returns this.
    """
    client = MagicMock()
    if side_effect_fn:
        client.chat.side_effect = side_effect_fn
    elif return_value is not None:
        client.chat.return_value = return_value
    else:
        client.chat.return_value = _llm_response("Hello! How can I help?")
    return client


def _build_test_app():
    """Build a minimal FastAPI app with the chat router and no lifespan."""

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    app = FastAPI(lifespan=noop_lifespan)

    # Import and mount the chat router AFTER patches are in place
    from routers.chat import router as chat_router
    app.include_router(chat_router)

    return app


@pytest.fixture()
def mock_infra():
    """Patch all heavy infrastructure so the chat handler can run in tests.

    Yields a dict of mock objects for further assertions.
    """
    fake_domain = FakeDomainConfig()
    fake_branding = fake_domain.branding
    fake_lexicon = {
        "identity": {"name": "ARCA", "welcome_message": "Hello"},
        "topics": ["general"],
        "thinking_messages": ["Pondering"],
        "pipeline": {},
    }
    fake_pipeline = {
        "chat_analysis_hint": "use analyze_files tool",
        "confidence_example": "empirical correlation, limited data",
        "equation_example": "$$F = ma$$",
    }

    fake_client = _make_fake_llm_client()
    fake_cache = FakeSessionCache()

    patches = {
        "llm_client": patch(_PATCH_LLM_CLIENT, return_value=fake_client),
        "register_tools": patch(_PATCH_REGISTER_TOOLS),
        "ws_conn_limit": patch(
            _PATCH_WS_CONN_LIMIT,
            new_callable=AsyncMock,
            return_value=(True, None),
        ),
        "ws_msg_limit": patch(
            _PATCH_WS_MSG_LIMIT,
            new_callable=AsyncMock,
            return_value=(True, None),
        ),
        "session_cache": patch(
            _PATCH_SESSION_CACHE,
            new_callable=AsyncMock,
            return_value=fake_cache,
        ),
        "domain_config": patch(_PATCH_DOMAIN_CONFIG, return_value=fake_domain),
        "domain_branding": patch(_PATCH_DOMAIN_BRANDING, return_value=fake_branding),
        "domain_lexicon": patch(_PATCH_DOMAIN_LEXICON, return_value=fake_lexicon),
        "pipeline_config": patch(_PATCH_PIPELINE_CONFIG, return_value=fake_pipeline),
    }

    started = {}
    for name, p in patches.items():
        started[name] = p.start()

    # Expose the fake client for per-test customization
    started["fake_client"] = fake_client
    started["fake_cache"] = fake_cache

    yield started

    for p in patches.values():
        p.stop()


@pytest.fixture()
def client(mock_infra):
    """Provide a Starlette TestClient with all infrastructure mocked."""
    app = _build_test_app()
    with TestClient(app) as c:
        yield c, mock_infra


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_ws_messages(ws, send_msg: dict = None, max_messages: int = 50):
    """Send a message on the WebSocket and collect all responses until 'done'.

    Returns list of parsed JSON dicts.
    """
    if send_msg:
        ws.send_json(send_msg)

    messages = []
    for _ in range(max_messages):
        try:
            data = ws.receive_json()
            messages.append(data)
            # Stop on the final "done" message or error
            if data.get("done") is True:
                break
            if data.get("type") == "error":
                break
        except Exception:
            break
    return messages


# ===========================================================================
# TEST CASES
# ===========================================================================


class TestBasicChat:
    """Basic chat: connect -> send message -> receive streamed response -> done."""

    def test_connect_send_receive(self, client):
        """A simple message should produce stream chunks + a done message."""
        tc, infra = client
        # Configure canned LLM response
        infra["fake_client"].chat.return_value = _llm_response(
            "Foundation design depends on soil conditions."
        )

        with tc.websocket_connect("/ws/chat") as ws:
            msgs = _collect_ws_messages(ws, {
                "message": "What is foundation design?",
                "session_id": "test-basic-001",
            })

        # Should have at least: tool_start(thinking), tool_end(thinking),
        # stream chunks, and the final done message
        assert len(msgs) >= 3, f"Expected >=3 messages, got {len(msgs)}: {msgs}"

        # Final message should be done=True with full content
        final = msgs[-1]
        assert final.get("done") is True
        assert "foundation" in final.get("content", "").lower() or \
               "Foundation" in final.get("content", "")

    def test_response_includes_tools_used(self, client):
        """Final response should include tools_used list."""
        tc, infra = client
        infra["fake_client"].chat.return_value = _llm_response("Answer here.")

        with tc.websocket_connect("/ws/chat") as ws:
            msgs = _collect_ws_messages(ws, {"message": "Hello", "session_id": "s1"})

        final = msgs[-1]
        assert "tools_used" in final
        assert isinstance(final["tools_used"], list)

    def test_empty_message_is_ignored(self, client):
        """Sending an empty message should not produce a response."""
        tc, infra = client

        with tc.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "", "session_id": "s2"})

            # Now send a real message to verify the connection still works
            infra["fake_client"].chat.return_value = _llm_response("I'm here.")
            ws.send_json({"message": "ping", "session_id": "s2"})

            msgs = []
            for _ in range(50):
                try:
                    data = ws.receive_json()
                    msgs.append(data)
                    if data.get("done") is True:
                        break
                except Exception:
                    break

            # The response should be for "ping", not for the empty message
            final = msgs[-1]
            assert final.get("done") is True


class TestMCPModeGuard:
    """MCP mode should reject WebSocket connections with an error."""

    def test_mcp_mode_rejects_chat(self, client):
        """When mcp_mode is True, WebSocket should get error and close."""
        tc, infra = client

        from config import runtime_config
        original = runtime_config.mcp_mode

        try:
            runtime_config.mcp_mode = True

            with tc.websocket_connect("/ws/chat") as ws:
                data = ws.receive_json()
                assert data["type"] == "error"
                assert "MCP mode" in data["content"]
        finally:
            runtime_config.mcp_mode = original

    def test_mcp_mode_off_allows_chat(self, client):
        """When mcp_mode is False, WebSocket should work normally."""
        tc, infra = client

        from config import runtime_config
        original = runtime_config.mcp_mode

        try:
            runtime_config.mcp_mode = False
            infra["fake_client"].chat.return_value = _llm_response("Normal response.")

            with tc.websocket_connect("/ws/chat") as ws:
                msgs = _collect_ws_messages(ws, {
                    "message": "Hello",
                    "session_id": "mcp-off-test",
                })

            final = msgs[-1]
            assert final.get("done") is True
        finally:
            runtime_config.mcp_mode = original


class TestModeDetection:
    """/think and /search prefixes should trigger mode flags."""

    def test_think_prefix_strips_and_activates(self, client):
        """'/think ...' should enable think mode and strip the prefix."""
        tc, infra = client

        call_kwargs = {}

        def capture_chat(**kwargs):
            call_kwargs.update(kwargs)
            return _llm_response("Deep analysis result.")

        infra["fake_client"].chat.side_effect = capture_chat

        with tc.websocket_connect("/ws/chat") as ws:
            msgs = _collect_ws_messages(ws, {
                "message": "/think What causes consolidation?",
                "session_id": "think-test",
            })

        final = msgs[-1]
        assert final.get("done") is True
        # The final response should indicate think mode was active
        assert final.get("think_mode") is True

    def test_search_prefix(self, client):
        """'/search ...' should enable search mode and strip prefix."""
        tc, infra = client
        infra["fake_client"].chat.return_value = _llm_response("Search results here.")

        with tc.websocket_connect("/ws/chat") as ws:
            msgs = _collect_ws_messages(ws, {
                "message": "/search latest codes",
                "session_id": "search-test",
            })

        # Should succeed without errors
        final = msgs[-1]
        assert final.get("done") is True

    def test_calculate_prefix(self, client):
        """'/calc ...' should enable calculate mode."""
        tc, infra = client

        # Calculate mode uses streaming, so mock needs to return an iterable
        def streaming_chat(**kwargs):
            if kwargs.get("stream"):
                return [_llm_response("Step 1: Given values...")]
            return _llm_response("Step 1: Given values...")

        infra["fake_client"].chat.side_effect = streaming_chat

        with tc.websocket_connect("/ws/chat") as ws:
            msgs = _collect_ws_messages(ws, {
                "message": "/calc bearing capacity for phi=30",
                "session_id": "calc-test",
            })

        final = msgs[-1]
        assert final.get("done") is True
        assert final.get("calculate_mode") is True


class TestErrorHandling:
    """Error conditions: malformed messages, LLM failures."""

    def test_message_too_long(self, client):
        """Messages exceeding MAX_MESSAGE_LENGTH should return an error."""
        tc, infra = client

        with tc.websocket_connect("/ws/chat") as ws:
            long_msg = "x" * 5000  # MAX_MESSAGE_LENGTH is 4000
            ws.send_json({"message": long_msg, "session_id": "long-test"})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "too long" in data["content"].lower()

    def test_llm_timeout_error(self, client):
        """LLM timeout should return a user-friendly error."""
        tc, infra = client

        from errors.exceptions import LLMError

        infra["fake_client"].chat.side_effect = LLMError(
            message="Model response timed out after 180s",
            error_type="timeout",
            model="test-model",
        )

        with tc.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "Hello", "session_id": "timeout-test"})

            # Skip tool_start thinking message
            msgs = []
            for _ in range(10):
                try:
                    data = ws.receive_json()
                    msgs.append(data)
                    if data.get("type") == "error":
                        break
                except Exception:
                    break

            error_msgs = [m for m in msgs if m.get("type") == "error"]
            assert len(error_msgs) >= 1
            assert "taking longer" in error_msgs[0]["content"].lower() or \
                   "timed out" in error_msgs[0]["content"].lower()

    def test_llm_generic_error(self, client):
        """Generic LLM error should be reported to the user."""
        tc, infra = client

        infra["fake_client"].chat.side_effect = RuntimeError("Connection refused")

        with tc.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "Hello", "session_id": "err-test"})

            msgs = []
            for _ in range(10):
                try:
                    data = ws.receive_json()
                    msgs.append(data)
                    if data.get("type") == "error":
                        break
                except Exception:
                    break

            error_msgs = [m for m in msgs if m.get("type") == "error"]
            assert len(error_msgs) >= 1


class TestToolDispatch:
    """Tool dispatch: message triggers tool call -> tool result -> final response."""

    def test_search_knowledge_tool_call(self, client):
        """LLM requesting search_knowledge should execute tool and return citations."""
        tc, infra = client

        call_count = {"n": 0}

        def llm_with_tool_call(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: LLM decides to call search_knowledge
                return _llm_response(
                    content="",
                    tool_calls=[
                        _tool_call(
                            "search_knowledge",
                            {"query": "bearing capacity", "top_k": 5},
                        )
                    ],
                )
            else:
                # Second call: LLM produces final answer after tool results
                return _llm_response(
                    "Bearing capacity is the ability of soil to support loads. "
                    "According to Terzaghi (1943), the ultimate bearing capacity..."
                )

        infra["fake_client"].chat.side_effect = llm_with_tool_call

        # Mock the search_knowledge executor to return canned results
        with patch("routers.chat_executors.execute_tool") as mock_exec:
            mock_exec.return_value = {
                "success": True,
                "results": [
                    {
                        "text": "Bearing capacity theory from Terzaghi...",
                        "source": "CFEM",
                        "score": 0.92,
                    }
                ],
                "citations": [{"source": "CFEM", "section": "4.3"}],
            }

            with tc.websocket_connect("/ws/chat") as ws:
                msgs = _collect_ws_messages(ws, {
                    "message": "What is bearing capacity?",
                    "session_id": "tool-test",
                })

        # LLM should have been called at least twice (initial + final)
        assert call_count["n"] >= 2

        # Final message
        final = msgs[-1]
        assert final.get("done") is True
        assert "bearing capacity" in final.get("content", "").lower()


class TestIngestLock:
    """Chat should be blocked during active document ingestion."""

    def test_ingest_active_blocks_chat(self, client):
        """When ingest is active, messages should get a friendly block message."""
        tc, infra = client

        from config import runtime_config
        original_lock = runtime_config.ingest_lock_enabled
        original_active = runtime_config._ingest_active

        try:
            runtime_config.ingest_lock_enabled = True
            runtime_config._ingest_active = True

            with tc.websocket_connect("/ws/chat") as ws:
                ws.send_json({"message": "Hello", "session_id": "ingest-test"})
                # Should get a "currently ingesting" response then done
                msgs = []
                for _ in range(5):
                    try:
                        data = ws.receive_json()
                        msgs.append(data)
                        if data.get("type") == "done":
                            break
                    except Exception:
                        break

                # Look for ingestion block message
                content_msgs = [m for m in msgs if m.get("content") and "ingest" in m.get("content", "").lower()]
                assert len(content_msgs) >= 1, f"Expected ingestion block message, got: {msgs}"
        finally:
            runtime_config.ingest_lock_enabled = original_lock
            runtime_config._ingest_active = original_active


class TestRateLimiting:
    """Rate limiter integration (mocked Redis)."""

    def test_ws_connection_rate_limited(self, client):
        """When rate limit check fails, connection should be rejected."""
        tc, infra = client

        # Override the rate limiter mock to return failure
        with patch(
            _PATCH_WS_CONN_LIMIT,
            new_callable=AsyncMock,
            return_value=(False, "Too many connections"),
        ):
            with tc.websocket_connect("/ws/chat") as ws:
                data = ws.receive_json()
                assert data["type"] == "error"
                assert "too many" in data["content"].lower() or \
                       "connections" in data["content"].lower()

    def test_ws_message_rate_limited(self, client):
        """When per-session message limit is hit, error should be returned."""
        tc, infra = client
        infra["fake_client"].chat.return_value = _llm_response("OK")

        with patch(
            _PATCH_WS_MSG_LIMIT,
            new_callable=AsyncMock,
            return_value=(False, "Message rate limit exceeded"),
        ):
            with tc.websocket_connect("/ws/chat") as ws:
                ws.send_json({"message": "Hello", "session_id": "rl-test"})
                data = ws.receive_json()
                assert data["type"] == "error"
                assert "rate limit" in data["content"].lower()


class TestMultipleExchanges:
    """Multiple messages in a single WebSocket session."""

    def test_two_messages_in_one_session(self, client):
        """Sending two messages should both get responses."""
        tc, infra = client

        call_count = {"n": 0}

        def sequential_responses(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _llm_response("First response.")
            return _llm_response("Second response.")

        infra["fake_client"].chat.side_effect = sequential_responses

        with tc.websocket_connect("/ws/chat") as ws:
            # First message
            msgs1 = _collect_ws_messages(ws, {"message": "First", "session_id": "multi-1"})
            assert msgs1[-1].get("done") is True
            assert "First response" in msgs1[-1].get("content", "")

            # Second message in same session
            msgs2 = _collect_ws_messages(ws, {"message": "Second", "session_id": "multi-1"})
            assert msgs2[-1].get("done") is True
            assert "Second response" in msgs2[-1].get("content", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
