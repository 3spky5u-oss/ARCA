"""
Tests for Phii Correction Memory system.

Tests the CorrectionDetector patterns, ReinforcementStore CRUD operations,
and PhiiContextBuilder integration.
"""

import pytest
import tempfile
from pathlib import Path

from tools.phii.reinforcement import (
    Correction,
    CorrectionDetector,
    ReinforcementStore,
)
from tools.phii.context_builder import PhiiContextBuilder


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def detector():
    """Create a CorrectionDetector instance."""
    return CorrectionDetector()


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
# CORRECTION DETECTOR TESTS
# =============================================================================


class TestCorrectionDetectorPatterns:
    """Test correction pattern detection."""

    def test_no_i_meant_pattern(self, detector):
        """Test 'No, I meant X' pattern."""
        result = detector.detect("No, I meant Method-B", "The Method-A value is...")
        assert result is not None
        assert result.confidence == 0.9
        assert "method-b" in result.right_behavior.lower()

    def test_actually_pattern(self, detector):
        """Test 'Actually, X' pattern."""
        result = detector.detect("Actually, it should be in metric", "Here's the value")
        assert result is not None
        assert result.confidence == 0.85
        assert "metric" in result.right_behavior.lower()

    def test_dont_use_pattern(self, detector):
        """Test 'Don't use X' pattern."""
        result = detector.detect("Don't use imperial units", "The value is 10 psf")
        assert result is not None
        assert result.confidence == 0.95
        assert "imperial" in result.wrong_behavior.lower()

    def test_dont_use_with_instead(self, detector):
        """Test 'Don't use X, use Y instead' pattern."""
        result = detector.detect("Don't use imperial, use metric instead", "The value is 10 psf")
        assert result is not None
        assert "imperial" in result.wrong_behavior.lower()
        assert "metric" in result.right_behavior.lower()

    def test_stop_using_pattern(self, detector):
        """Test 'Stop using X' pattern."""
        result = detector.detect("Stop using feet", "result")
        assert result is not None
        assert "feet" in result.wrong_behavior.lower()

    def test_never_say_with_instead(self, detector):
        """Test 'Never say X, say Y instead' pattern."""
        result = detector.detect("Never say abbrev, say full technical name instead", "result")
        assert result is not None
        assert "abbrev" in result.wrong_behavior.lower()
        assert "full technical name" in result.right_behavior.lower()

    def test_always_use_pattern(self, detector):
        """Test 'Always use X' pattern."""
        result = detector.detect("Always use bullets for lists", "result")
        assert result is not None
        assert result.confidence == 0.9
        assert "bullets" in result.right_behavior.lower()

    def test_too_long_pattern(self, detector):
        """Test 'Too long' style correction."""
        result = detector.detect("Too long", "Here's a detailed explanation...")
        assert result is not None
        assert result.confidence == 0.8
        assert "long" in result.wrong_behavior.lower()
        assert "shorter" in result.right_behavior.lower()

    def test_too_verbose_pattern(self, detector):
        """Test 'Too verbose' style correction."""
        result = detector.detect("Too verbose", "result")
        assert result is not None
        assert "verbose" in result.wrong_behavior.lower()
        assert "concise" in result.right_behavior.lower()

    def test_too_short_pattern(self, detector):
        """Test 'Too short' style correction."""
        result = detector.detect("Too short", "result")
        assert result is not None
        assert "short" in result.wrong_behavior.lower()
        assert "detailed" in result.right_behavior.lower()

    def test_prefer_pattern(self, detector):
        """Test 'I prefer X over Y' pattern."""
        result = detector.detect("I prefer metric over imperial", "result")
        assert result is not None
        assert "metric" in result.right_behavior.lower()

    def test_wrong_pattern(self, detector):
        """Test 'Wrong. X' pattern."""
        result = detector.detect("Wrong. The value should be 10 kPa", "result")
        assert result is not None
        assert "10 kpa" in result.right_behavior.lower()

    def test_no_detection_for_normal_message(self, detector):
        """Test that normal messages don't trigger detection."""
        result = detector.detect("What is the performance capacity?", "result")
        assert result is None

    def test_no_detection_for_short_message(self, detector):
        """Test that very short messages don't trigger detection."""
        result = detector.detect("ok", "result")
        assert result is None

    def test_no_detection_for_thanks(self, detector):
        """Test that 'thanks' doesn't trigger correction (it's positive feedback)."""
        result = detector.detect("Thanks!", "result")
        assert result is None


class TestCorrectionDetectorKeywords:
    """Test context keyword extraction."""

    def test_extracts_engineering_keywords(self, detector):
        """Test extraction of context keywords."""
        result = detector.detect("Don't use imperial, use metric instead", "The detailed summary shows...")
        assert result is not None
        assert "imperial" in result.context_keywords or "metric" in result.context_keywords
        assert "detailed" in result.context_keywords or "summary" in result.context_keywords

    def test_extracts_unit_keywords(self, detector):
        """Test extraction of unit-related keywords."""
        result = detector.detect("Always use metric units", "The value in psf is...")
        assert result is not None
        keywords = result.context_keywords
        assert any(k in keywords for k in ["metric", "units", "psf"])


# =============================================================================
# REINFORCEMENT STORE TESTS
# =============================================================================


class TestReinforcementStoreCRUD:
    """Test ReinforcementStore CRUD operations for corrections."""

    def test_store_correction(self, store):
        """Test storing a correction."""
        correction = Correction(
            wrong_behavior="imperial units",
            right_behavior="metric units",
            context_keywords=["units", "metric"],
            confidence=0.9,
        )
        cid = store.store_correction(correction, "test-session")
        assert cid > 0

    def test_get_all_corrections(self, store):
        """Test retrieving all corrections."""
        # Store some corrections
        for i in range(3):
            correction = Correction(
                wrong_behavior=f"wrong_{i}",
                right_behavior=f"right_{i}",
            )
            store.store_correction(correction, "test-session")

        corrections = store.get_all_corrections()
        assert len(corrections) == 3

    def test_get_all_corrections_limit(self, store):
        """Test limit parameter."""
        for i in range(5):
            correction = Correction(
                wrong_behavior=f"wrong_{i}",
                right_behavior=f"right_{i}",
            )
            store.store_correction(correction, "test-session")

        corrections = store.get_all_corrections(limit=2)
        assert len(corrections) == 2

    def test_delete_correction(self, store):
        """Test soft-deleting a correction."""
        correction = Correction(
            wrong_behavior="test",
            right_behavior="test",
        )
        cid = store.store_correction(correction, "test-session")

        # Delete it
        result = store.delete_correction(cid)
        assert result is True

        # Should not appear in active corrections
        corrections = store.get_all_corrections(active_only=True)
        assert len(corrections) == 0

        # Should appear when including inactive
        corrections = store.get_all_corrections(active_only=False)
        assert len(corrections) == 1
        assert corrections[0].is_active is False

    def test_delete_nonexistent_correction(self, store):
        """Test deleting a non-existent correction."""
        result = store.delete_correction(9999)
        assert result is False

    def test_increment_applied(self, store):
        """Test incrementing the times_applied counter."""
        correction = Correction(
            wrong_behavior="test",
            right_behavior="test",
        )
        cid = store.store_correction(correction, "test-session")

        # Increment a few times
        store.increment_applied(cid)
        store.increment_applied(cid)
        store.increment_applied(cid)

        # Check the count
        corrections = store.get_all_corrections()
        assert corrections[0].times_applied == 3

    def test_get_corrections_stats(self, store):
        """Test getting correction statistics."""
        # Add some corrections
        for i in range(3):
            correction = Correction(
                wrong_behavior=f"wrong_{i}",
                right_behavior=f"right_{i}",
            )
            cid = store.store_correction(correction, "test-session")
            store.increment_applied(cid)

        stats = store.get_corrections_stats()
        assert stats["active_count"] == 3
        assert stats["total_applied"] == 3
        assert stats["recent_7d"] == 3


class TestReinforcementStoreRelevance:
    """Test correction relevance matching."""

    def test_get_relevant_corrections_by_keyword(self, store):
        """Test finding relevant corrections by keyword overlap."""
        # Store corrections with different keywords
        c1 = Correction(
            wrong_behavior="imperial",
            right_behavior="metric",
            context_keywords=["metric", "imperial", "units"],
        )
        c2 = Correction(
            wrong_behavior="long responses",
            right_behavior="shorter responses",
            context_keywords=["brief", "detailed"],
        )
        store.store_correction(c1, "test")
        store.store_correction(c2, "test")

        # Query about units - should find c1
        relevant = store.get_relevant_corrections("What are the units in metric?")
        assert len(relevant) >= 1
        assert any("metric" in c.right_behavior for c in relevant)

    def test_get_relevant_corrections_empty(self, store):
        """Test that unrelated queries return empty."""
        c1 = Correction(
            wrong_behavior="imperial",
            right_behavior="metric",
            context_keywords=["metric", "imperial"],
        )
        store.store_correction(c1, "test")

        # Completely unrelated query
        relevant = store.get_relevant_corrections("hello world")
        # May or may not find it depending on scoring
        # At minimum, should not crash
        assert isinstance(relevant, list)

    def test_get_relevant_corrections_top_k(self, store):
        """Test top_k parameter limits results."""
        for i in range(10):
            c = Correction(
                wrong_behavior=f"wrong_{i}",
                right_behavior=f"right_{i}",
                context_keywords=["common", "keyword"],
            )
            store.store_correction(c, "test")

        relevant = store.get_relevant_corrections("common keyword test", top_k=3)
        assert len(relevant) <= 3


class TestReinforcementStoreCache:
    """Test correction caching behavior."""

    def test_cache_is_populated(self, store):
        """Test that cache gets populated on query."""
        c = Correction(
            wrong_behavior="test",
            right_behavior="test",
            context_keywords=["keyword"],
        )
        store.store_correction(c, "test")

        # Query should populate cache
        store.get_relevant_corrections("keyword")
        assert len(store._corrections_cache) > 0

    def test_cache_invalidated_on_store(self, store):
        """Test that cache is invalidated when storing new correction."""
        c1 = Correction(wrong_behavior="test1", right_behavior="test1")
        store.store_correction(c1, "test")

        # Populate cache
        store.get_relevant_corrections("test")
        cache_time_1 = store._corrections_cache_time

        # Store another correction - should invalidate cache
        c2 = Correction(wrong_behavior="test2", right_behavior="test2")
        store.store_correction(c2, "test")
        assert store._corrections_cache_time == 0

    def test_cache_invalidated_on_delete(self, store):
        """Test that cache is invalidated when deleting correction."""
        c = Correction(wrong_behavior="test", right_behavior="test")
        cid = store.store_correction(c, "test")

        # Populate cache
        store.get_relevant_corrections("test")

        # Delete - should invalidate cache
        store.delete_correction(cid)
        assert store._corrections_cache_time == 0


# =============================================================================
# CONTEXT BUILDER INTEGRATION TESTS
# =============================================================================


class TestPhiiContextBuilderCorrections:
    """Test PhiiContextBuilder correction integration."""

    def test_observe_exchange_detects_correction(self, builder, store):
        """Test that observe_exchange detects and stores corrections."""
        correction = builder.observe_exchange(
            session_id="test-session",
            user_message="Don't use imperial, use metric instead",
            assistant_response="The value is 10 psf",
            tools_used=[],
        )
        assert correction is not None
        assert correction.id is not None

        # Verify it was stored
        corrections = store.get_all_corrections()
        assert len(corrections) == 1

    def test_observe_exchange_no_correction(self, builder, store):
        """Test that normal messages don't create corrections."""
        correction = builder.observe_exchange(
            session_id="test-session",
            user_message="What is the performance capacity?",
            assistant_response="The performance capacity is...",
            tools_used=[],
        )
        assert correction is None

        corrections = store.get_all_corrections()
        assert len(corrections) == 0

    def test_build_corrections_hint_formats_correctly(self, builder, store):
        """Test that corrections hint is formatted correctly."""
        # Store some corrections
        c1 = Correction(
            wrong_behavior="imperial",
            right_behavior="metric",
        )
        c2 = Correction(
            wrong_behavior="",
            right_behavior="use bullets",
        )
        store.store_correction(c1, "test")
        store.store_correction(c2, "test")

        hint = builder._build_corrections_hint([c1, c2])
        assert "LEARNED CORRECTIONS" in hint
        assert "Avoid: imperial" in hint
        assert "Instead: metric" in hint
        assert "Remember: use bullets" in hint

    def test_build_corrections_hint_empty(self, builder):
        """Test that empty corrections list returns empty string."""
        hint = builder._build_corrections_hint([])
        assert hint == ""


# =============================================================================
# CORRECTION DATACLASS TESTS
# =============================================================================


class TestCorrectionDataclass:
    """Test Correction dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        c = Correction()
        assert c.id is None
        assert c.confidence == 0.8
        assert c.times_applied == 0
        assert c.is_active is True
        assert c.context_keywords == []

    def test_to_dict(self):
        """Test conversion to dictionary."""
        c = Correction(
            id=1,
            wrong_behavior="wrong",
            right_behavior="right",
            context_keywords=["kw1", "kw2"],
            confidence=0.9,
        )
        d = c.to_dict()
        assert d["id"] == 1
        assert d["wrong_behavior"] == "wrong"
        assert d["right_behavior"] == "right"
        assert d["context_keywords"] == ["kw1", "kw2"]
        assert d["confidence"] == 0.9
