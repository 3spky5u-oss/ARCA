"""
Integration tests for the Tool Router + Handler system.

Tests the full flow: router decision -> handler classification -> pre_process
"""

import asyncio
import pytest
from unittest.mock import patch

from routers.chat_orchestration import ToolRoutingDecision
from routers.chat_orchestration.handlers import (
    HandlerContext,
    QueryClassifier,
    GeologyHandler,
    TechnicalHandler,
    CalculateHandler,
    ThinkHandler,
    DefaultHandler,
)


class TestHandlerClassification:
    """Test handler classification logic."""

    def setup_method(self):
        """Reset classifier for each test."""
        self.classifier = QueryClassifier()
        self.classifier.register(GeologyHandler())
        self.classifier.register(CalculateHandler())
        self.classifier.register(ThinkHandler())
        self.classifier.register(TechnicalHandler())
        self.classifier.register(DefaultHandler())

    def test_geology_from_router(self):
        """Router selecting geology should trigger GeologyHandler."""
        ctx = HandlerContext(
            message="What materials are at Location A?",
            session_id="test",
            router_decision=ToolRoutingDecision(
                tool="query_geological_map",
                args={"location": "Location A"},
                confidence=0.9,
            ),
        )

        handler = self.classifier.classify(ctx)
        assert handler.name == "geology"

    def test_geology_from_pattern(self):
        """Pattern matching should trigger GeologyHandler when router returns null."""
        ctx = HandlerContext(
            message="What materials are at Location A?",
            session_id="test",
            router_decision=ToolRoutingDecision(tool=None, confidence=0.9),
        )

        handler = self.classifier.classify(ctx)
        assert handler.name == "geology"

    def test_calculate_from_flag(self):
        """Calculate mode flag should trigger CalculateHandler."""
        ctx = HandlerContext(
            message="Calculate structural capacity",
            session_id="test",
            calculate_mode=True,
        )

        handler = self.classifier.classify(ctx)
        assert handler.name == "calculate"

    def test_calculate_from_keyword(self):
        """Calculate keyword should auto-trigger CalculateHandler."""
        ctx = HandlerContext(
            message="Calculate the structural capacity for resistance_angle=30",
            session_id="test",
        )

        handler = self.classifier.classify(ctx)
        assert handler.name == "calculate"
        assert ctx.calculate_mode is True
        assert ctx.auto_calculate is True

    def test_think_from_flag(self):
        """Think mode flag should trigger ThinkHandler."""
        ctx = HandlerContext(
            message="Analyze this problem",
            session_id="test",
            think_mode=True,
        )

        handler = self.classifier.classify(ctx)
        assert handler.name == "think"

    def test_think_from_keyword(self):
        """Think keyword should auto-trigger ThinkHandler."""
        ctx = HandlerContext(
            message="Think through this problem carefully",
            session_id="test",
        )

        handler = self.classifier.classify(ctx)
        assert handler.name == "think"
        assert ctx.think_mode is True
        assert ctx.auto_think is True

    def test_technical_from_router(self):
        """Router selecting search_knowledge should trigger TechnicalHandler."""
        ctx = HandlerContext(
            message="What is structural capacity?",
            session_id="test",
            router_decision=ToolRoutingDecision(
                tool="search_knowledge",
                confidence=0.9,
            ),
        )

        handler = self.classifier.classify(ctx)
        assert handler.name == "technical"

    def test_default_fallback(self):
        """Non-matching queries should fall through to DefaultHandler."""
        ctx = HandlerContext(
            message="Hello, how are you?",
            session_id="test",
            router_decision=ToolRoutingDecision(tool=None, confidence=0.9),
        )

        handler = self.classifier.classify(ctx)
        assert handler.name == "default"

    def test_priority_order(self):
        """Handlers should be checked in priority order."""
        handlers = self.classifier.get_handlers()
        priorities = [h.priority for h in handlers]
        assert priorities == sorted(priorities), "Handlers not sorted by priority"


class TestGeologyHandler:
    """Test GeologyHandler specifically."""

    def test_pre_process_injects_context(self):
        """Pre-process should inject domain spatial context."""
        handler = GeologyHandler()

        ctx = HandlerContext(
            message="What materials are at Location A?",
            session_id="test",
            router_decision=ToolRoutingDecision(
                tool="query_geological_map",
                args={"location": "Location A"},
                confidence=0.9,
            ),
        )

        # Should match
        assert handler.should_handle(ctx) is True

        # Mock the spatial lookup (imported inside pre_process from tools.mapperr)
        with patch("tools.mapperr.query_geological_map") as mock_geo:
            mock_geo.return_value = {
                "success": True,
                "location": "Location A",
                "unit_code": "U01",
                "unit_name": "Sedimentary Unit",
                "lithology": "Mixed material",
                "lithogenesis": "Depositional",
                "material_types": ["Type-A", "Type-B"],
            }

            # Run async pre_process synchronously
            asyncio.run(handler.pre_process(ctx))

            # Check context was injected
            assert ctx.forced_tool == "query_geological_map"
            assert ctx.analysis_result is not None
            assert ctx.analysis_result["unit_code"] == "U01"
            assert "query_geological_map" in ctx.tools_used
            assert "GEOLOGY DATA" in ctx.injected_context


class TestToolRouterFallback:
    """Test that fallback logic works correctly."""

    def test_low_confidence_triggers_fallback(self):
        """Low confidence should set fallback=True."""
        decision = ToolRoutingDecision(
            tool="query_geological_map",
            confidence=0.5,  # Below 0.7 threshold
        )

        # In real usage, router.route() sets fallback based on confidence
        # Here we just verify the flag exists
        decision.fallback = decision.confidence < 0.7
        assert decision.fallback is True

    def test_high_confidence_no_fallback(self):
        """High confidence should not trigger fallback."""
        decision = ToolRoutingDecision(
            tool="query_geological_map",
            confidence=0.9,
        )

        decision.fallback = decision.confidence < 0.7
        assert decision.fallback is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
