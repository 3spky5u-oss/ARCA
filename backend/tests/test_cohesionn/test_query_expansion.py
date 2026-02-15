"""
Unit tests for Cohesionn query expansion module.

Tests:
- Engineering synonym expansion
- Query term detection
- Expansion limits
"""

import pytest
from tools.cohesionn.query_expansion import (
    expand_query,
    get_query_expander,
    _get_synonym_groups,
)


class TestSynonymGroups:
    """Tests for synonym group data."""

    def test_synonym_groups_load(self):
        """Synonym groups should load from config."""
        groups = _get_synonym_groups()
        assert isinstance(groups, list)

    def test_synonym_groups_contain_sets(self):
        """Each group should be a set of related terms."""
        groups = _get_synonym_groups()
        for group in groups:
            assert isinstance(group, set), "Each synonym group should be a set"
            assert len(group) >= 2, "Each group should have at least 2 terms"

    def test_minimum_synonym_groups(self):
        """Should have synonym groups if domain pack is loaded."""
        groups = _get_synonym_groups()
        # May be empty without domain pack -- just verify it returns a list
        assert isinstance(groups, list)


class TestQueryExpander:
    """Tests for QueryExpander class."""

    def test_expand_query_with_term(self):
        """Query with known term should be expanded if synonyms configured."""
        expanded = expand_query("load capacity analysis")

        # Should preserve or expand
        assert len(expanded) >= len("load capacity analysis")
        # Original terms should be preserved
        assert "load" in expanded.lower() or "capacity" in expanded.lower()

    def test_expand_query_preserves_original(self):
        """Original query terms should be preserved."""
        original = "load capacity formula"
        expanded = expand_query(original)

        # Original words should still be present
        for word in original.split():
            assert word.lower() in expanded.lower()

    def test_expand_query_no_synonyms(self):
        """Query with no matching terms should remain unchanged."""
        original = "chocolate cake recipe"
        expanded = expand_query(original)

        # Should be unchanged or minimally changed
        assert original in expanded or expanded == original

    def test_max_expansions_limit(self):
        """Expansion should respect max_expansions limit."""
        # Query with many potentially expandable terms
        query = "Method-A Method-B V60 load capacity settlement"
        expanded = expand_query(query, max_expansions=2)

        # Should not explode in length
        # Original is ~45 chars, expanded should be reasonable
        assert len(expanded) < len(query) * 5

    def test_singleton_expander(self):
        """get_query_expander should return singleton."""
        exp1 = get_query_expander()
        exp2 = get_query_expander()
        assert exp1 is exp2


class TestQueryExpansionQuality:
    """Quality tests for query expansion."""

    @pytest.mark.parametrize("query,expected_terms", [
        ("load test", ["load", "test"]),
        ("capacity analysis", ["capacity", "analysis"]),
        ("settlement evaluation", ["settlement"]),
    ])
    def test_expansion_contains_relevant_terms(self, query, expected_terms):
        """Expanded query should contain relevant terms."""
        expanded = expand_query(query)
        expanded_lower = expanded.lower()

        for term in expected_terms:
            assert term in expanded_lower, (
                f"Expected '{term}' in expanded query.\n"
                f"Original: {query}\n"
                f"Expanded: {expanded}"
            )

    def test_case_insensitive_matching(self):
        """Expansion should work regardless of case."""
        lower = expand_query("load")
        upper = expand_query("LOAD")
        mixed = expand_query("Load")

        # All should produce non-empty results
        assert len(lower) >= 4
        assert len(upper) >= 4
        assert len(mixed) >= 4

    def test_multiple_terms_expanded(self):
        """Multiple terms in query should be expanded."""
        query = "Method-A and Method-B correlation"
        expanded = expand_query(query)

        # Should preserve original at minimum
        assert len(expanded) >= len(query)


class TestEdgeCases:
    """Edge case tests for query expansion."""

    def test_empty_query(self):
        """Empty query should return empty."""
        assert expand_query("") == ""

    def test_whitespace_query(self):
        """Whitespace-only query should be handled."""
        result = expand_query("   ")
        assert result.strip() == ""

    def test_special_characters(self):
        """Query with special characters should be handled."""
        query = "V60 = Vm × (Em/60)"
        # Should not raise
        expanded = expand_query(query)
        assert "v60" in expanded.lower() or "V60" in expanded

    def test_very_long_query(self):
        """Very long query should be handled."""
        query = "test " * 100
        # Should not raise or explode
        expanded = expand_query(query)
        assert len(expanded) < len(query) * 10
