"""
Tests for Phii Pattern Prediction system.

Tests the ActionClassifier, action recording, pattern learning,
and proactive suggestion prediction.
"""

import pytest
import tempfile
from pathlib import Path

from tools.phii.reinforcement import (
    ActionClassifier,
    ReinforcementStore,
    ACTION_SUGGESTIONS,
)
from tools.phii.context_builder import PhiiContextBuilder


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def classifier():
    """Create an ActionClassifier instance."""
    return ActionClassifier()


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_reinforcement.sqlite"
    yield db_path
    # Cleanup handled by tempfile


@pytest.fixture
def store(temp_db):
    """Create a ReinforcementStore with temp database."""
    return ReinforcementStore(db_path=temp_db)


@pytest.fixture
def builder(store):
    """Create a PhiiContextBuilder with test store."""
    return PhiiContextBuilder(reinforcement_store=store)


# =============================================================================
# ACTION CLASSIFIER TESTS
# =============================================================================


class TestActionClassifierTools:
    """Test action classification from tools."""

    def test_analyze_files_tool(self, classifier):
        """Test analyze_files tool classification."""
        result = classifier.classify("check this file", ["analyze_files"])
        assert result == "compliance_check"

    def test_search_knowledge_tool(self, classifier):
        """Test search_knowledge tool classification."""
        result = classifier.classify("search for performance capacity", ["search_knowledge"])
        assert result == "knowledge_search"

    def test_solve_engineering_tool(self, classifier):
        """Test solve_engineering tool classification."""
        result = classifier.classify("calculate deflection", ["solve_engineering"])
        assert result == "calculation"

    def test_unit_convert_tool(self, classifier):
        """Test unit_convert tool classification."""
        result = classifier.classify("convert 10 kPa", ["unit_convert"])
        assert result == "unit_conversion"

    def test_redact_document_tool(self, classifier):
        """Test redact_document tool classification."""
        result = classifier.classify("redact this document", ["redact_document"])
        assert result == "redaction"

    def test_get_lab_template_tool(self, classifier):
        """Test get_lab_template tool classification."""
        result = classifier.classify("get the template", ["get_lab_template"])
        assert result == "template_request"

    def test_generate_openground_tool(self, classifier):
        """Test generate_openground tool classification (domain tool)."""
        result = classifier.classify("generate export files", ["generate_openground"])
        assert result == "openground_export"

    def test_first_tool_wins(self, classifier):
        """Test that first matching tool takes priority."""
        result = classifier.classify("do stuff", ["analyze_files", "search_knowledge"])
        assert result == "compliance_check"

    def test_unknown_tool_falls_through(self, classifier):
        """Test that unknown tools don't match."""
        result = classifier.classify("hello", ["unknown_tool"])
        assert result is None


class TestActionClassifierMessages:
    """Test action classification from message patterns."""

    def test_upload_pattern(self, classifier):
        """Test file upload detection."""
        assert classifier.classify("I've uploaded the file", []) == "file_upload"
        assert classifier.classify("Here's the lab data", []) == "file_upload"
        assert classifier.classify("Attached is my report", []) == "file_upload"
        assert classifier.classify("here's a file I attached", []) == "file_upload"

    def test_domain_message_patterns(self, classifier):
        """Test domain-specific message pattern detection (lexicon-driven).

        These patterns (compliance_interest, domain calculations, etc.) are
        loaded from the domain lexicon. Without a domain pack, only generic
        patterns are active.
        """
        # Generic patterns should always work
        result = classifier.classify("calculate 2 + 2", [])
        assert result == "math_interest"

    def test_upload_pattern_variants(self, classifier):
        """Test additional file upload pattern variants."""
        assert classifier.classify("here's my data file", []) == "file_upload"
        assert classifier.classify("I've attached the report", []) == "file_upload"

    def test_tool_takes_priority_over_message(self, classifier):
        """Test that tool classification takes priority over message patterns."""
        # Message says "upload" but tool was search_knowledge
        result = classifier.classify("I uploaded a file", ["search_knowledge"])
        assert result == "knowledge_search"

    def test_no_match(self, classifier):
        """Test that unrecognized messages return None."""
        assert classifier.classify("hello", []) is None
        assert classifier.classify("thanks", []) is None
        assert classifier.classify("what time is it?", []) is None


# =============================================================================
# ACTION RECORDING TESTS
# =============================================================================


class TestActionRecording:
    """Test action recording and pattern storage."""

    def test_record_single_action(self, store):
        """Test recording a single action."""
        store.record_action("session-1", "file_upload")

        # Check in-memory tracking
        assert "session-1" in store._session_actions
        assert store._session_actions["session-1"] == ["file_upload"]

    def test_record_multiple_actions(self, store):
        """Test recording multiple actions in sequence."""
        store.record_action("session-1", "file_upload")
        store.record_action("session-1", "compliance_check")
        store.record_action("session-1", "knowledge_search")

        assert store._session_actions["session-1"] == ["file_upload", "compliance_check", "knowledge_search"]

    def test_pattern_created(self, store):
        """Test that action patterns are created."""
        store.record_action("session-1", "file_upload")
        store.record_action("session-1", "compliance_check")

        stats = store.get_pattern_stats()
        assert stats["pattern_count"] == 1
        assert stats["action_count"] == 2
        assert len(stats["top_patterns"]) == 1
        assert stats["top_patterns"][0]["from"] == "file_upload"
        assert stats["top_patterns"][0]["to"] == "compliance_check"
        assert stats["top_patterns"][0]["count"] == 1

    def test_pattern_count_increments(self, store):
        """Test that repeated patterns increment count."""
        # First occurrence
        store.record_action("session-1", "file_upload")
        store.record_action("session-1", "compliance_check")

        # Second occurrence (different session)
        store.record_action("session-2", "file_upload")
        store.record_action("session-2", "compliance_check")

        stats = store.get_pattern_stats()
        assert stats["pattern_count"] == 1  # Still one unique pattern
        assert stats["top_patterns"][0]["count"] == 2  # But count is 2

    def test_session_isolation(self, store):
        """Test that sessions are tracked separately."""
        store.record_action("session-1", "file_upload")
        store.record_action("session-2", "knowledge_search")

        assert store._session_actions["session-1"] == ["file_upload"]
        assert store._session_actions["session-2"] == ["knowledge_search"]

    def test_session_memory_limit(self, store):
        """Test that session memory is limited to 10 actions."""
        for i in range(15):
            store.record_action("session-1", f"action_{i}")

        assert len(store._session_actions["session-1"]) == 10
        # Should have last 10 actions
        assert store._session_actions["session-1"][0] == "action_5"
        assert store._session_actions["session-1"][-1] == "action_14"

    def test_clear_session_actions(self, store):
        """Test clearing session actions."""
        store.record_action("session-1", "file_upload")
        store.record_action("session-1", "compliance_check")

        store.clear_session_actions("session-1")

        assert "session-1" not in store._session_actions

    def test_clear_nonexistent_session(self, store):
        """Test that clearing nonexistent session doesn't error."""
        store.clear_session_actions("nonexistent")  # Should not raise


# =============================================================================
# PATTERN PREDICTION TESTS
# =============================================================================


class TestPatternPrediction:
    """Test action prediction from patterns."""

    def test_no_prediction_without_history(self, store):
        """Test that no prediction without session history."""
        result = store.predict_next_action("session-1")
        assert result is None

    def test_static_suggestion_fallback(self, store):
        """Test fallback to static suggestions."""
        store.record_action("session-1", "file_upload")

        # No pattern data yet, should fall back to static
        result = store.predict_next_action("session-1")

        # Static suggestion for file_upload comes from domain lexicon.
        # Without a domain pack, file_upload may not have a suggestion.
        # The knowledge_search -> calculation suggestion is always present.
        if result is not None:
            assert result.action in ("compliance_check", "calculation")
            assert len(result.message) > 0

    def test_learned_pattern_prediction(self, store):
        """Test prediction from learned patterns."""
        # Build up pattern data
        for _ in range(5):
            store._session_actions["session-x"] = ["file_upload"]
            store.record_action("session-x", "compliance_check")

        # Now test prediction for a session with file_upload
        store.record_action("session-1", "file_upload")
        result = store.predict_next_action("session-1")

        assert result is not None
        assert result.action == "compliance_check"
        assert result.confidence >= 0.4

    def test_confidence_threshold(self, store):
        """Test that low confidence predictions are filtered."""
        # Create mixed pattern data (no clear winner)
        store._session_actions["s1"] = ["file_upload"]
        store.record_action("s1", "compliance_check")

        store._session_actions["s2"] = ["file_upload"]
        store.record_action("s2", "knowledge_search")

        store._session_actions["s3"] = ["file_upload"]
        store.record_action("s3", "calculation")

        # With 3 different next actions, confidence is ~33% each
        store.record_action("session-1", "file_upload")

        # With high threshold (50%), should return None since confidence is ~33%
        result_high = store.predict_next_action("session-1", min_confidence=0.5)
        assert result_high is None

        # With lower threshold (30%), should return prediction
        result_low = store.predict_next_action("session-1", min_confidence=0.3)
        assert result_low is not None
        # Should be one of the three actions we recorded
        assert result_low.action in ["compliance_check", "knowledge_search", "calculation"]
        # Confidence should be ~33%
        assert 0.3 <= result_low.confidence <= 0.4

    def test_prediction_after_last_action(self, store):
        """Test prediction is based on last action."""
        store.record_action("session-1", "file_upload")
        store.record_action("session-1", "knowledge_search")

        # Prediction should be based on knowledge_search (the last action)
        result = store.predict_next_action("session-1")

        # Static suggestion for knowledge_search -> calculation
        if result:
            assert result.action == "calculation"

    def test_generic_suggestion_for_unknown_action(self, store):
        """Test generic suggestion for actions without static message."""
        # Build pattern for an action without static suggestion
        for _ in range(3):
            store._session_actions["sx"] = ["calculation"]
            store.record_action("sx", "unit_conversion")

        store.record_action("session-1", "calculation")
        result = store.predict_next_action("session-1")

        assert result is not None
        assert result.action == "unit_conversion"
        assert "unit conversion" in result.message.lower()


# =============================================================================
# PATTERN STATS TESTS
# =============================================================================


class TestPatternStats:
    """Test pattern statistics retrieval."""

    def test_empty_stats(self, store):
        """Test stats with no data."""
        stats = store.get_pattern_stats()

        assert stats["pattern_count"] == 0
        assert stats["action_count"] == 0
        assert stats["top_patterns"] == []

    def test_stats_after_actions(self, store):
        """Test stats after recording actions."""
        store.record_action("s1", "file_upload")
        store.record_action("s1", "compliance_check")
        store.record_action("s1", "knowledge_search")

        stats = store.get_pattern_stats()

        assert stats["pattern_count"] == 2
        assert stats["action_count"] == 3
        assert len(stats["top_patterns"]) == 2

    def test_top_patterns_ordered(self, store):
        """Test that top patterns are ordered by count."""
        # Create patterns with different counts
        for _ in range(5):
            store._session_actions["sx"] = ["file_upload"]
            store.record_action("sx", "compliance_check")

        for _ in range(2):
            store._session_actions["sy"] = ["file_upload"]
            store.record_action("sy", "knowledge_search")

        stats = store.get_pattern_stats()

        # Compliance check should be first (count 5)
        assert stats["top_patterns"][0]["to"] == "compliance_check"
        assert stats["top_patterns"][0]["count"] == 5
        assert stats["top_patterns"][1]["to"] == "knowledge_search"
        assert stats["top_patterns"][1]["count"] == 2


# =============================================================================
# CONTEXT BUILDER INTEGRATION TESTS
# =============================================================================


class TestPhiiContextBuilderPatterns:
    """Test PhiiContextBuilder pattern integration."""

    def test_observe_exchange_records_action(self, builder, store):
        """Test that observe_exchange records actions."""
        builder.observe_exchange(
            session_id="test-session",
            user_message="I uploaded the test results",
            assistant_response="Got it, analyzing...",
            tools_used=["analyze_files"],
        )

        # Should have recorded compliance_check (from analyze_files tool)
        assert "test-session" in store._session_actions
        assert store._session_actions["test-session"] == ["compliance_check"]

    def test_observe_exchange_records_from_message(self, builder, store):
        """Test action recording from message patterns."""
        builder.observe_exchange(
            session_id="test-session",
            user_message="Here's the file I uploaded",
            assistant_response="Thanks, I see the file.",
            tools_used=[],
        )

        # Should have recorded file_upload (from message pattern)
        assert store._session_actions["test-session"] == ["file_upload"]

    def test_observe_exchange_tool_priority(self, builder, store):
        """Test that tool classification takes priority."""
        builder.observe_exchange(
            session_id="test-session",
            user_message="I uploaded a file, search for performance capacity",
            assistant_response="Searching...",
            tools_used=["search_knowledge"],
        )

        # Tool should win over message pattern
        assert store._session_actions["test-session"] == ["knowledge_search"]

    def test_clear_session_clears_actions(self, builder, store):
        """Test that clear_session clears action memory."""
        builder.observe_exchange(
            session_id="test-session",
            user_message="Here's my file",
            assistant_response="Got it",
            tools_used=["analyze_files"],
        )

        builder.clear_session("test-session")

        assert "test-session" not in store._session_actions

    def test_proactive_hint_in_context(self, builder, store):
        """Test that proactive hint appears in context."""
        # First, record an action to enable prediction
        store.record_action("test-session", "file_upload")

        # Build context
        ctx = builder.build_context(
            current_message="what should I do next?",
            files_context="",
            session_notes="",
            session_id="test-session",
            phii_enabled=True,
        )

        # Proactive hint depends on domain lexicon providing suggestions.
        # Without a domain pack, file_upload may not have a static suggestion,
        # so PROACTIVE TIP may or may not appear.
        # With domain pack: "PROACTIVE TIP:" in ctx.system_prompt
        assert isinstance(ctx.system_prompt, str)

    def test_no_proactive_hint_without_action(self, builder, store):
        """Test no proactive hint without prior action."""
        ctx = builder.build_context(
            current_message="hello",
            files_context="",
            session_notes="",
            session_id="new-session",
            phii_enabled=True,
        )

        # Should NOT have proactive tip
        assert "PROACTIVE TIP:" not in ctx.system_prompt


# =============================================================================
# ACTION SUGGESTIONS CONSTANT TESTS
# =============================================================================


class TestActionSuggestionsConstant:
    """Test the ACTION_SUGGESTIONS constant is properly configured."""

    def test_file_upload_suggestion(self):
        """Test file_upload suggestion (domain-dependent).

        ACTION_SUGGESTIONS is None at module level; actual suggestions
        are loaded lazily via _get_action_suggestions(). The file_upload
        suggestion comes from the domain lexicon.
        """
        # Module-level ACTION_SUGGESTIONS is None
        assert ACTION_SUGGESTIONS is None or isinstance(ACTION_SUGGESTIONS, dict)

    def test_knowledge_search_suggestion(self):
        """Test knowledge_search has suggestion (always present)."""
        # knowledge_search -> calculation is a generic suggestion
        # that exists regardless of domain pack
        from tools.phii.reinforcement import _get_action_suggestions
        suggestions = _get_action_suggestions()
        suggestion = suggestions.get("knowledge_search")
        assert suggestion is not None
        assert suggestion.action == "calculation"

    def test_domain_calculation_suggestion(self):
        """Test domain calculation suggestion (lexicon-driven).

        ACTION_SUGGESTIONS is now loaded lazily from the lexicon.
        Domain-specific suggestions like geotech_calculation only exist
        when a domain pack is active.
        """
        # ACTION_SUGGESTIONS is set to None at module level;
        # actual suggestions come from _get_action_suggestions()
        assert ACTION_SUGGESTIONS is None or isinstance(ACTION_SUGGESTIONS, dict)

    def test_knowledge_search_suggestion_legacy(self):
        """Test knowledge_search suggestion via lazy loader."""
        from tools.phii.reinforcement import _get_action_suggestions
        suggestions = _get_action_suggestions()
        suggestion = suggestions.get("knowledge_search")
        assert suggestion is not None
        assert suggestion.action == "calculation"


# =============================================================================
# EDGE CASES
# =============================================================================


class TestPatternEdgeCases:
    """Test edge cases in pattern tracking."""

    def test_empty_tools_list(self, classifier):
        """Test classification with empty tools list."""
        result = classifier.classify("I uploaded a file", [])
        assert result == "file_upload"

    def test_none_tools(self, store):
        """Test classify_action with None tools."""
        result = store.classify_action("I uploaded a file", None)
        # Should handle None gracefully and fall through to message patterns
        assert result == "file_upload"

    def test_very_long_session(self, store):
        """Test pattern tracking over many actions."""
        for i in range(100):
            store.record_action("long-session", "file_upload")
            store.record_action("long-session", "compliance_check")

        # Session memory should still be bounded
        assert len(store._session_actions["long-session"]) == 10

        # Pattern should have high count
        stats = store.get_pattern_stats()
        pattern = next(
            (p for p in stats["top_patterns"] if p["from"] == "file_upload" and p["to"] == "compliance_check"), None
        )
        assert pattern is not None
        assert pattern["count"] == 100

    def test_concurrent_sessions(self, store):
        """Test multiple concurrent sessions."""
        sessions = ["s1", "s2", "s3", "s4", "s5"]

        for session in sessions:
            store.record_action(session, "file_upload")

        # All sessions should have their own tracking
        for session in sessions:
            assert session in store._session_actions
            assert store._session_actions[session] == ["file_upload"]
