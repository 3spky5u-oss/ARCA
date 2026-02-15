"""
Tests for RAPTOR hierarchical summarization module.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from typing import List, Dict, Any


# =============================================================================
# Clusterer Tests
# =============================================================================

class TestRaptorClusterer:
    """Tests for RaptorClusterer."""

    def test_cluster_empty_input(self):
        """Empty input returns empty result."""
        from tools.cohesionn.raptor.clusterer import RaptorClusterer

        clusterer = RaptorClusterer()
        result = clusterer.cluster([])

        assert result.cluster_assignments == []
        assert result.n_clusters == 0
        assert result.cluster_sizes == {}

    def test_cluster_small_input(self):
        """Small input (< min_cluster_size) goes to single cluster."""
        from tools.cohesionn.raptor.clusterer import RaptorClusterer

        clusterer = RaptorClusterer(min_cluster_size=5)

        # Create 3 random embeddings (less than min_cluster_size)
        embeddings = np.random.randn(3, 1024).tolist()
        result = clusterer.cluster(embeddings)

        assert len(result.cluster_assignments) == 3
        assert result.n_clusters == 1
        assert all(c == 0 for c in result.cluster_assignments)

    def test_cluster_normal_input(self):
        """Normal input clusters into multiple groups."""
        from tools.cohesionn.raptor.clusterer import RaptorClusterer

        clusterer = RaptorClusterer(
            target_cluster_size=5,
            min_cluster_size=2,
        )

        # Create 20 embeddings that should form ~4 clusters
        np.random.seed(42)
        embeddings = np.random.randn(20, 1024).tolist()
        result = clusterer.cluster(embeddings)

        assert len(result.cluster_assignments) == 20
        assert result.n_clusters >= 2
        assert sum(result.cluster_sizes.values()) == 20

    def test_cluster_fixed_n_clusters(self):
        """Fixed n_clusters parameter is respected."""
        from tools.cohesionn.raptor.clusterer import RaptorClusterer

        clusterer = RaptorClusterer()

        np.random.seed(42)
        embeddings = np.random.randn(30, 1024).tolist()
        result = clusterer.cluster(embeddings, n_clusters=3)

        assert result.n_clusters == 3

    def test_cluster_by_similarity(self):
        """Similarity-based clustering works."""
        from tools.cohesionn.raptor.clusterer import RaptorClusterer

        clusterer = RaptorClusterer()

        # Create embeddings with clear clusters
        np.random.seed(42)
        cluster1 = np.random.randn(5, 1024) + np.array([1.0] + [0.0] * 1023)
        cluster2 = np.random.randn(5, 1024) + np.array([-1.0] + [0.0] * 1023)
        embeddings = np.vstack([cluster1, cluster2]).tolist()

        result = clusterer.cluster_by_similarity(embeddings, threshold=0.5)

        assert len(result.cluster_assignments) == 10
        assert result.n_clusters >= 1


# =============================================================================
# Summarizer Tests
# =============================================================================

class TestRaptorSummarizer:
    """Tests for RaptorSummarizer."""

    def test_summarize_empty_input(self):
        """Empty input returns empty summary."""
        from tools.cohesionn.raptor.summarizer import RaptorSummarizer

        summarizer = RaptorSummarizer()
        result = summarizer.summarize([])

        assert result.summary == ""
        assert result.source_count == 0

    @patch("tools.cohesionn.raptor.summarizer.RaptorSummarizer.client")
    def test_summarize_single_content(self, mock_client):
        """Single content item is summarized."""
        from tools.cohesionn.raptor.summarizer import RaptorSummarizer

        mock_client.chat.return_value = {
            "message": {"content": "Test summary of technical content."},
            "eval_count": 50,
            "prompt_eval_count": 100,
        }

        summarizer = RaptorSummarizer()
        summarizer._client = mock_client

        result = summarizer.summarize(["Method-A test procedures..."], level=1)

        assert result.summary == "Test summary of technical content."
        assert result.source_count == 1
        assert result.level == 1

    @patch("tools.cohesionn.raptor.summarizer.RaptorSummarizer.client")
    def test_summarize_multiple_contents(self, mock_client):
        """Multiple content items are combined and summarized."""
        from tools.cohesionn.raptor.summarizer import RaptorSummarizer

        mock_client.chat.return_value = {
            "message": {"content": "Combined summary of testing methods."},
        }

        summarizer = RaptorSummarizer()
        summarizer._client = mock_client

        result = summarizer.summarize(
            ["Method-A procedures...", "Method-B procedures...", "Classification limits..."],
            level=1,
        )

        assert result.source_count == 3

    def test_fallback_summary(self):
        """Fallback summary works without LLM."""
        from tools.cohesionn.raptor.summarizer import RaptorSummarizer

        summarizer = RaptorSummarizer()

        fallback = summarizer._fallback_summary(
            ["First content. Second sentence.", "Third content."],
            max_length=50,
        )

        assert len(fallback) <= 53  # 50 + "..."
        assert "First" in fallback

    @patch("tools.cohesionn.raptor.summarizer.RaptorSummarizer.client")
    def test_summarize_batch(self, mock_client):
        """Batch summarization works."""
        from tools.cohesionn.raptor.summarizer import RaptorSummarizer

        mock_client.chat.return_value = {
            "message": {"content": "Batch summary."},
        }

        summarizer = RaptorSummarizer()
        summarizer._client = mock_client

        content_groups = [
            ["Content A1", "Content A2"],
            ["Content B1", "Content B2"],
        ]

        results = summarizer.summarize_batch(content_groups, level=1)

        assert len(results) == 2
        assert all(r.source_count == 2 for r in results)


# =============================================================================
# Tree Builder Tests
# =============================================================================

class TestRaptorTreeBuilder:
    """Tests for RaptorTreeBuilder."""

    def test_node_dataclass(self):
        """RaptorNode dataclass works correctly."""
        from tools.cohesionn.raptor.tree_builder import RaptorNode

        node = RaptorNode(
            node_id="test_node_1",
            content="Test summary content",
            level=1,
            embedding=[0.1] * 1024,
            children=["child1", "child2"],
            metadata={"topic": "technical"},
        )

        assert node.node_id == "test_node_1"
        assert node.level == 1
        assert len(node.children) == 2

    def test_group_by_cluster(self):
        """Items are correctly grouped by cluster assignment."""
        from tools.cohesionn.raptor.tree_builder import RaptorTreeBuilder

        builder = RaptorTreeBuilder()

        items = [
            {"id": "a", "content": "A"},
            {"id": "b", "content": "B"},
            {"id": "c", "content": "C"},
        ]
        assignments = [0, 1, 0]

        groups = builder._group_by_cluster(items, assignments)

        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 1


# =============================================================================
# Retriever Mixin Tests
# =============================================================================

class TestRaptorRetrieverMixin:
    """Tests for RaptorRetrieverMixin."""

    def test_should_use_raptor_broad_query(self):
        """Broad queries should use RAPTOR."""
        from tools.cohesionn.raptor.retriever_mixin import RaptorRetrieverMixin

        mixin = RaptorRetrieverMixin()

        assert mixin.should_use_raptor("What are the main types of structural systems?")
        assert mixin.should_use_raptor("Overview of material testing methods")
        assert mixin.should_use_raptor("Compare Method-A and Method-B tests")
        assert mixin.should_use_raptor("Fundamentals of load capacity")

    def test_should_use_raptor_specific_query(self):
        """Specific queries may not need RAPTOR."""
        from tools.cohesionn.raptor.retriever_mixin import RaptorRetrieverMixin

        mixin = RaptorRetrieverMixin()

        # Short queries are considered broad by default
        assert mixin.should_use_raptor("Method-A V60")  # Short, uses RAPTOR

        # Specific detailed queries
        # These don't contain broad indicators but are short
        assert mixin.should_use_raptor("V60 value")  # Short

    def test_merge_raptor_results_empty(self):
        """Merging with empty RAPTOR results returns dense only."""
        from tools.cohesionn.raptor.retriever_mixin import RaptorRetrieverMixin

        mixin = RaptorRetrieverMixin()

        dense = [{"id": "1", "score": 0.8}]
        raptor = []

        merged = mixin.merge_raptor_results(dense, raptor)

        assert len(merged) == 1
        assert merged[0]["id"] == "1"

    def test_merge_raptor_results_weighted(self):
        """Merging applies weights correctly."""
        from tools.cohesionn.raptor.retriever_mixin import RaptorRetrieverMixin

        mixin = RaptorRetrieverMixin()

        dense = [{"id": "1", "score": 0.8, "content": "Dense content"}]
        raptor = [{"id": "2", "score": 0.9, "content": "RAPTOR summary", "is_raptor": True}]

        merged = mixin.merge_raptor_results(dense, raptor, raptor_weight=0.3)

        assert len(merged) == 2
        # Dense: 0.8 * 0.7 = 0.56
        # RAPTOR: 0.9 * 0.3 = 0.27
        assert merged[0]["merged_score"] > merged[1]["merged_score"]

    def test_merge_raptor_results_same_id(self):
        """Duplicate IDs are merged correctly."""
        from tools.cohesionn.raptor.retriever_mixin import RaptorRetrieverMixin

        mixin = RaptorRetrieverMixin()

        dense = [{"id": "1", "score": 0.8}]
        raptor = [{"id": "1", "score": 0.9}]

        merged = mixin.merge_raptor_results(dense, raptor, raptor_weight=0.3)

        assert len(merged) == 1
        # Combined: 0.8 * 0.7 + 0.9 * 0.3 = 0.56 + 0.27 = 0.83
        assert abs(merged[0]["merged_score"] - 0.83) < 0.01


# =============================================================================
# Integration Tests (require models/services)
# =============================================================================

@pytest.mark.integration
class TestRaptorIntegration:
    """Integration tests requiring full services."""

    @pytest.fixture
    def mock_kb(self):
        """Mock knowledge base."""
        kb = MagicMock()
        kb.discover_topics.return_value = ["technical"]
        return kb

    def test_tree_builder_integration(self, mock_kb):
        """Full tree build with mocked components."""
        # This test would run against real Qdrant in integration environment
        pass
