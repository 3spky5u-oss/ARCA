"""
Unit tests for Cohesionn reranker module.

Tests:
- BGEReranker cross-encoder scoring
- Score sorting and top-k selection
- Batch processing
- Fallback behavior
"""

import pytest
from typing import Dict, Any, List


class TestMockReranker:
    """Tests using mock reranker (fast, no GPU)."""

    def test_rerank_returns_top_k(self, mock_reranker, test_results):
        """Rerank should return exactly top_k results."""
        reranked = mock_reranker.rerank("test query", test_results, top_k=3)
        assert len(reranked) == 3

    def test_rerank_adds_scores(self, mock_reranker, test_results):
        """Rerank should add rerank_score to results."""
        reranked = mock_reranker.rerank("test query", test_results, top_k=5)

        for result in reranked:
            assert "rerank_score" in result

    def test_rerank_sorted_by_score(self, mock_reranker, test_results):
        """Results should be sorted by rerank_score descending."""
        reranked = mock_reranker.rerank("test query", test_results, top_k=5)

        scores = [r["rerank_score"] for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_empty_results(self, mock_reranker):
        """Reranking empty results should return empty list."""
        reranked = mock_reranker.rerank("query", [], top_k=5)
        assert reranked == []

    def test_rerank_fewer_than_top_k(self, mock_reranker):
        """Should handle fewer results than top_k."""
        results = [
            {"id": "1", "content": "only one result", "metadata": {}},
        ]
        reranked = mock_reranker.rerank("query", results, top_k=5)
        assert len(reranked) == 1


class TestRealReranker:
    """Tests using real reranker (requires models)."""

    def test_real_rerank_produces_scores(self, real_reranker, all_test_content):
        """Real reranker should produce meaningful scores."""
        results = [
            {"id": k, "content": v, "metadata": {}}
            for k, v in all_test_content.items()
        ]

        reranked = real_reranker.rerank("threshold values high-performance systems", results, top_k=5)

        assert len(reranked) == 5
        for r in reranked:
            assert "rerank_score" in r
            assert isinstance(r["rerank_score"], float)

    def test_real_rerank_score_range(self, real_reranker, all_test_content):
        """Scores should be in reasonable range."""
        results = [
            {"id": k, "content": v, "metadata": {}}
            for k, v in all_test_content.items()
        ]

        reranked = real_reranker.rerank("load capacity", results, top_k=len(results))

        scores = [r["rerank_score"] for r in reranked]
        # Scores typically range from negative to positive
        # Just verify they're numeric and sorted
        assert all(isinstance(s, float) for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_relevant_content_scores_higher(self, real_reranker):
        """Content matching query should score higher."""
        results = [
            {"id": "relevant", "content": "Method-A threshold values for high-performance systems are 30-50 units per cycle", "metadata": {}},
            {"id": "irrelevant", "content": "Recipe for chocolate cake requires flour and sugar", "metadata": {}},
        ]

        reranked = real_reranker.rerank("What are threshold values for high-performance systems?", results, top_k=2)

        # Relevant content should be first
        assert reranked[0]["id"] == "relevant"
        assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]

    def test_batch_processing(self, real_reranker, all_test_content):
        """Should handle batches correctly."""
        results = [
            {"id": k, "content": v, "metadata": {}}
            for k, v in all_test_content.items()
        ]

        # Process with different batch sizes
        reranked_small_batch = real_reranker.rerank(
            "test query", results, top_k=5, batch_size=2
        )
        reranked_large_batch = real_reranker.rerank(
            "test query", results, top_k=5, batch_size=32
        )

        # Should produce same results regardless of batch size
        assert len(reranked_small_batch) == len(reranked_large_batch)

        # Same ordering
        small_ids = [r["id"] for r in reranked_small_batch]
        large_ids = [r["id"] for r in reranked_large_batch]
        assert small_ids == large_ids

    def test_preserves_original_fields(self, real_reranker):
        """Original result fields should be preserved."""
        results = [
            {
                "id": "test",
                "content": "Test content",
                "metadata": {"source": "test.pdf", "page": 5},
                "original_score": 0.8,
            }
        ]

        reranked = real_reranker.rerank("test", results, top_k=1)

        assert reranked[0]["metadata"]["source"] == "test.pdf"
        assert reranked[0]["metadata"]["page"] == 5
        assert reranked[0]["original_score"] == 0.8


class TestRerankerDomainQuality:
    """Content-specific quality tests for reranker."""

    @pytest.mark.parametrize("query,preferred_id", [
        ("What are threshold values for high-performance systems?", "spt_n60"),
        ("load capacity safety factor", "bearing_capacity"),
        ("component testing settlement", "pile_load_test"),
        ("environmental factor depth seasonal", "frost_depth"),
    ])
    def test_domain_query_prefers_domain_content(
        self,
        real_reranker,
        all_test_content,
        query: str,
        preferred_id: str,
    ):
        """Specialized queries should rank matching content highest."""
        results = [
            {"id": k, "content": v, "metadata": {}}
            for k, v in all_test_content.items()
        ]

        reranked = real_reranker.rerank(query, results, top_k=len(results))

        # Check that preferred content is in top 2
        top_ids = [r["id"] for r in reranked[:2]]
        assert preferred_id in top_ids, (
            f"Query '{query}' should prefer '{preferred_id}'.\n"
            f"Got top 2: {top_ids}\n"
            f"Top scores: {[(r['id'], round(r['rerank_score'], 3)) for r in reranked[:3]]}"
        )

    def test_specialized_scores_above_generic(self, real_reranker, domain_content, generic_content):
        """Specialized content should score above generic for matching queries."""
        query = "What is the load testing methodology threshold?"

        results = [
            {"id": "specialized", "content": domain_content["spt_n60"], "metadata": {}},
            {"id": "generic", "content": generic_content["lrfd_factors"], "metadata": {}},
        ]

        reranked = real_reranker.rerank(query, results, top_k=2)

        specialized_score = next(r["rerank_score"] for r in reranked if r["id"] == "specialized")
        generic_score = next(r["rerank_score"] for r in reranked if r["id"] == "generic")

        assert specialized_score > generic_score, (
            f"Specialized should score higher than generic for matching query.\n"
            f"Specialized: {specialized_score:.3f}, Generic: {generic_score:.3f}"
        )


class TestRerankerEdgeCases:
    """Edge case tests for reranker."""

    def test_very_long_content(self, real_reranker):
        """Should handle content exceeding max_length."""
        long_content = "This is a test sentence. " * 500  # Very long

        results = [
            {"id": "long", "content": long_content, "metadata": {}},
            {"id": "short", "content": "Short content.", "metadata": {}},
        ]

        # Should not raise
        reranked = real_reranker.rerank("test", results, top_k=2)
        assert len(reranked) == 2

    def test_special_characters(self, real_reranker):
        """Should handle special characters in content."""
        results = [
            {"id": "special", "content": "V60 = Vm × (Em/60) where φ ≥ 30°", "metadata": {}},
            {"id": "normal", "content": "Normal content without special chars", "metadata": {}},
        ]

        # Should not raise
        reranked = real_reranker.rerank("V60 formula", results, top_k=2)
        assert len(reranked) == 2

    def test_unicode_content(self, real_reranker):
        """Should handle unicode in content."""
        results = [
            {"id": "unicode", "content": "材料力学 (material mechanics) γ = 18 kN/m³", "metadata": {}},
        ]

        # Should not raise
        reranked = real_reranker.rerank("material mechanics", results, top_k=1)
        assert len(reranked) == 1

    def test_empty_content(self, real_reranker):
        """Should handle empty content strings."""
        results = [
            {"id": "empty", "content": "", "metadata": {}},
            {"id": "normal", "content": "Normal content here", "metadata": {}},
        ]

        # Should handle gracefully
        reranked = real_reranker.rerank("test", results, top_k=2)
        assert len(reranked) == 2
