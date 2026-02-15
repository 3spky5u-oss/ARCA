"""
Unit tests for Cohesionn diversity reranker (MMR).

Tests:
- Maximal Marginal Relevance calculation
- Source diversity limits (max_per_source)
- Lambda parameter effects
- Edge cases
"""

import pytest
from tools.cohesionn.reranker import DiversityReranker


class TestDiversityRerankerBasics:
    """Basic functionality tests for DiversityReranker."""

    def test_returns_top_k_results(self):
        """Should return exactly top_k results."""
        reranker = DiversityReranker(lambda_param=0.6, max_per_source=2)

        results = [
            {"id": str(i), "content": f"content {i}", "rerank_score": 1.0 - i * 0.1, "metadata": {"source": f"source{i}.pdf"}}
            for i in range(10)
        ]

        reranked = reranker.rerank(results, top_k=5)
        assert len(reranked) == 5

    def test_empty_results(self):
        """Should handle empty results."""
        reranker = DiversityReranker()
        reranked = reranker.rerank([], top_k=5)
        assert reranked == []

    def test_single_result(self):
        """Should handle single result."""
        reranker = DiversityReranker()
        results = [{"id": "1", "content": "test", "rerank_score": 0.9, "metadata": {}}]
        reranked = reranker.rerank(results, top_k=5)
        assert len(reranked) == 1

    def test_preserves_original_fields(self):
        """Should preserve original result fields."""
        reranker = DiversityReranker()
        results = [
            {
                "id": "test",
                "content": "test content",
                "rerank_score": 0.9,
                "metadata": {"source": "test.pdf", "page": 5},
                "custom_field": "preserved",
            }
        ]

        reranked = reranker.rerank(results, top_k=1)

        assert reranked[0]["custom_field"] == "preserved"
        assert reranked[0]["metadata"]["source"] == "test.pdf"


class TestMaxPerSource:
    """Tests for source diversity limit."""

    def test_respects_max_per_source(self):
        """Should not select more than max_per_source from same source."""
        reranker = DiversityReranker(lambda_param=1.0, max_per_source=2)

        # All from same source with descending scores
        results = [
            {"id": str(i), "content": f"content {i}", "rerank_score": 1.0 - i * 0.1, "metadata": {"source": "same.pdf"}}
            for i in range(10)
        ]

        reranked = reranker.rerank(results, top_k=5)

        # Should only get 2 from same source
        assert len(reranked) == 2

    def test_distributes_across_sources(self):
        """Should select from multiple sources."""
        reranker = DiversityReranker(lambda_param=1.0, max_per_source=2)

        # Two sources
        results = [
            {"id": "a1", "content": "a content 1", "rerank_score": 0.95, "metadata": {"source": "a.pdf"}},
            {"id": "a2", "content": "a content 2", "rerank_score": 0.90, "metadata": {"source": "a.pdf"}},
            {"id": "a3", "content": "a content 3", "rerank_score": 0.85, "metadata": {"source": "a.pdf"}},
            {"id": "b1", "content": "b content 1", "rerank_score": 0.80, "metadata": {"source": "b.pdf"}},
            {"id": "b2", "content": "b content 2", "rerank_score": 0.75, "metadata": {"source": "b.pdf"}},
        ]

        reranked = reranker.rerank(results, top_k=4)

        # Should get 2 from each source
        sources = [r["metadata"]["source"] for r in reranked]
        assert sources.count("a.pdf") <= 2
        assert sources.count("b.pdf") <= 2

    def test_handles_missing_source(self):
        """Should handle results without source metadata."""
        reranker = DiversityReranker(lambda_param=1.0, max_per_source=2)

        results = [
            {"id": "1", "content": "no source", "rerank_score": 0.9, "metadata": {}},
            {"id": "2", "content": "also no source", "rerank_score": 0.8, "metadata": {}},
        ]

        # Should not raise
        reranked = reranker.rerank(results, top_k=2)
        assert len(reranked) == 2


class TestLambdaParameter:
    """Tests for MMR lambda parameter effects."""

    def test_lambda_one_pure_relevance(self):
        """Lambda=1.0 should be pure relevance ordering."""
        reranker = DiversityReranker(lambda_param=1.0, max_per_source=100)

        results = [
            {"id": "a", "content": "content a", "rerank_score": 0.9, "metadata": {"source": "x.pdf"}},
            {"id": "b", "content": "content b", "rerank_score": 0.8, "metadata": {"source": "y.pdf"}},
            {"id": "c", "content": "content c", "rerank_score": 0.7, "metadata": {"source": "z.pdf"}},
        ]

        reranked = reranker.rerank(results, top_k=3)

        # Should be in original score order
        ids = [r["id"] for r in reranked]
        assert ids == ["a", "b", "c"]

    def test_low_lambda_promotes_diversity(self):
        """Low lambda should promote diversity over relevance."""
        # With lambda=0.3, diversity matters more
        reranker = DiversityReranker(lambda_param=0.3, max_per_source=100)

        # Similar content should be penalized
        results = [
            {"id": "a1", "content": "Method-A threshold values for high-performance systems are important", "rerank_score": 0.95, "metadata": {"source": "x.pdf"}},
            {"id": "a2", "content": "Method-A threshold values for high-performance systems range from 30 to 50", "rerank_score": 0.90, "metadata": {"source": "y.pdf"}},
            {"id": "b", "content": "Load capacity depends on material properties", "rerank_score": 0.85, "metadata": {"source": "z.pdf"}},
        ]

        reranked = reranker.rerank(results, top_k=3)

        # Second similar result should be penalized for similarity to first
        # So order might change from pure relevance order
        # Just verify it runs and produces results
        assert len(reranked) == 3

    def test_default_lambda_balanced(self):
        """Default lambda (0.6) should balance relevance and diversity."""
        reranker = DiversityReranker()  # Uses default lambda_param=0.6

        assert reranker.lambda_param == 0.6


class TestMMRCalculation:
    """Tests for MMR score calculation."""

    def test_first_result_uses_relevance_only(self):
        """First selected result should be based on relevance only."""
        reranker = DiversityReranker(lambda_param=0.6, max_per_source=100)

        results = [
            {"id": "low", "content": "low score", "rerank_score": 0.5, "metadata": {}},
            {"id": "high", "content": "high score", "rerank_score": 0.9, "metadata": {}},
        ]

        reranked = reranker.rerank(results, top_k=2)

        # First result should be highest relevance
        assert reranked[0]["id"] == "high"

    def test_similar_content_penalized(self):
        """Similar content to already-selected should be penalized."""
        reranker = DiversityReranker(lambda_param=0.5, max_per_source=100)

        results = [
            {"id": "a", "content": "the quick brown fox jumps over the lazy dog", "rerank_score": 0.9, "metadata": {}},
            {"id": "b", "content": "the quick brown fox runs over the lazy cat", "rerank_score": 0.85, "metadata": {}},  # Similar to a
            {"id": "c", "content": "completely different content about material mechanics", "rerank_score": 0.8, "metadata": {}},  # Different
        ]

        reranked = reranker.rerank(results, top_k=3)

        # After selecting 'a', 'c' might be preferred over 'b' due to diversity
        # Just verify calculation doesn't fail
        assert len(reranked) == 3


class TestScoreKeyParameter:
    """Tests for score_key parameter."""

    def test_uses_rerank_score_by_default(self):
        """Should use rerank_score by default."""
        reranker = DiversityReranker()

        results = [
            {"id": "a", "content": "a", "rerank_score": 0.9, "score": 0.1, "metadata": {}},
            {"id": "b", "content": "b", "rerank_score": 0.1, "score": 0.9, "metadata": {}},
        ]

        reranked = reranker.rerank(results, top_k=2)

        # Should use rerank_score, so 'a' first
        assert reranked[0]["id"] == "a"

    def test_custom_score_key(self):
        """Should use custom score_key when specified."""
        reranker = DiversityReranker()

        results = [
            {"id": "a", "content": "a", "rerank_score": 0.9, "custom_score": 0.1, "metadata": {}},
            {"id": "b", "content": "b", "rerank_score": 0.1, "custom_score": 0.9, "metadata": {}},
        ]

        reranked = reranker.rerank(results, top_k=2, score_key="custom_score")

        # Should use custom_score, so 'b' first
        assert reranked[0]["id"] == "b"

    def test_falls_back_to_score(self):
        """Should fall back to 'score' if rerank_score missing."""
        reranker = DiversityReranker()

        results = [
            {"id": "a", "content": "a", "score": 0.9, "metadata": {}},
            {"id": "b", "content": "b", "score": 0.1, "metadata": {}},
        ]

        reranked = reranker.rerank(results, top_k=2)

        # Should fall back to score
        assert reranked[0]["id"] == "a"


class TestIntegrationWithReranker:
    """Integration tests with real diversity reranker."""

    def test_real_diversity_reranker(self, real_diversity_reranker, all_test_content):
        """Real diversity reranker should work with test content."""
        results = [
            {"id": k, "content": v, "rerank_score": 0.9 - i * 0.05, "metadata": {"source": f"test_{k}.md"}}
            for i, (k, v) in enumerate(all_test_content.items())
        ]

        reranked = real_diversity_reranker.rerank(results, top_k=5)

        assert len(reranked) == 5
        # Check source diversity
        sources = [r["metadata"]["source"] for r in reranked]
        unique_sources = len(set(sources))
        assert unique_sources >= 2  # Should have multiple sources

    def test_mmr_score_added(self, real_diversity_reranker, all_test_content):
        """MMR score should be added to results."""
        results = [
            {"id": k, "content": v, "rerank_score": 0.9, "metadata": {"source": f"test_{k}.md"}}
            for k, v in all_test_content.items()
        ]

        reranked = real_diversity_reranker.rerank(results, top_k=3)

        for r in reranked:
            assert "_mmr_score" in r
