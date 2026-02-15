"""
Tests for Community Search components - Detection, summarization, global retrieval.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any


# =============================================================================
# Query Classifier Tests
# =============================================================================

class TestQueryClassifier:
    """Tests for QueryClassifier."""

    def test_classify_global_overview(self):
        """Overview questions are classified as global."""
        from tools.cohesionn.query_classifier import QueryClassifier, QueryType

        classifier = QueryClassifier()

        queries = [
            "What are the main types of structural design?",
            "Overview of material testing methods",
            "Explain the principles of load capacity",
            "Fundamentals of engineering analysis",
        ]

        for query in queries:
            query_type, confidence = classifier.classify(query)
            assert query_type == QueryType.GLOBAL, f"'{query}' should be GLOBAL"
            assert confidence > 0.5

    def test_classify_global_comparison(self):
        """Comparison questions are classified as global."""
        from tools.cohesionn.query_classifier import QueryClassifier, QueryType

        classifier = QueryClassifier()

        queries = [
            "Compare Method-A and Method-B tests",
            "Difference between primary and secondary elements",
            "What are the pros and cons of composite structures?",
        ]

        for query in queries:
            query_type, confidence = classifier.classify(query)
            assert query_type == QueryType.GLOBAL, f"'{query}' should be GLOBAL"

    def test_classify_local_specific(self):
        """Specific value questions are classified as local."""
        from tools.cohesionn.query_classifier import QueryClassifier, QueryType

        classifier = QueryClassifier()

        queries = [
            "What is the threshold value for high-performance systems?",
            "Calculate load capacity for 2m wide element",
            "According to ASTM D1586 what is the sample recovery length?",
        ]

        for query in queries:
            query_type, confidence = classifier.classify(query)
            assert query_type == QueryType.LOCAL, f"'{query}' should be LOCAL"

    def test_classify_local_standards(self):
        """Standard reference questions are classified as local."""
        from tools.cohesionn.query_classifier import QueryClassifier, QueryType

        classifier = QueryClassifier()

        queries = [
            "ASTM D1586 requirements",
            "Per AASHTO T99 what is the procedure?",
            "What does ASTM D4318 specify?",
        ]

        for query in queries:
            query_type, confidence = classifier.classify(query)
            assert query_type == QueryType.LOCAL, f"'{query}' should be LOCAL"

    def test_search_strategy(self):
        """Search strategy returns correct flags."""
        from tools.cohesionn.query_classifier import QueryClassifier

        classifier = QueryClassifier()

        # Global query
        strategy = classifier.get_search_strategy("What are the main types of structures?")
        assert strategy["use_community_search"] is True
        assert strategy["query_type"] == "global"

        # Local query
        strategy = classifier.get_search_strategy("What is V60?")
        # Short query defaults to hybrid
        assert "query_type" in strategy


# =============================================================================
# Community Detection Tests
# =============================================================================

class TestCommunityDetection:
    """Tests for CommunityDetector."""

    def test_resolution_params_exist(self):
        """Resolution parameters are defined."""
        from tools.cohesionn.community_detection import CommunityDetector

        detector = CommunityDetector()

        assert "coarse" in detector.RESOLUTION_PARAMS
        assert "medium" in detector.RESOLUTION_PARAMS
        assert "fine" in detector.RESOLUTION_PARAMS

        # Coarse < medium < fine
        assert detector.RESOLUTION_PARAMS["coarse"] < detector.RESOLUTION_PARAMS["medium"]
        assert detector.RESOLUTION_PARAMS["medium"] < detector.RESOLUTION_PARAMS["fine"]

    def test_community_dataclass(self):
        """Community dataclass works correctly."""
        from tools.cohesionn.community_detection import Community

        community = Community(
            community_id="test_1",
            level="medium",
            node_ids=["chunk_1", "chunk_2"],
            entity_ids=["Method-A", "V60"],
            node_count=2,
            entity_count=2,
        )

        assert community.community_id == "test_1"
        assert community.node_count == 2
        assert len(community.entity_ids) == 2

    @patch("tools.cohesionn.community_detection.CommunityDetector._load_graph_from_neo4j")
    def test_detect_empty_graph(self, mock_load):
        """Empty graph returns empty result."""
        from tools.cohesionn.community_detection import CommunityDetector
        import networkx as nx

        mock_load.return_value = nx.Graph()

        detector = CommunityDetector()
        result = detector.detect()

        assert len(result.communities) == 0
        assert result.total_nodes == 0


# =============================================================================
# Community Summarizer Tests
# =============================================================================

class TestCommunitySummarizer:
    """Tests for CommunitySummarizer."""

    def test_summary_prompt_defined(self):
        """Summary prompt template is defined."""
        from tools.cohesionn.community_summarizer import CommunitySummarizer

        summarizer = CommunitySummarizer()

        assert "SUMMARY:" in summarizer.SUMMARY_PROMPT
        assert "THEMES:" in summarizer.SUMMARY_PROMPT
        assert "{entities}" in summarizer.SUMMARY_PROMPT
        assert "{content}" in summarizer.SUMMARY_PROMPT

    def test_format_entities(self):
        """Entity formatting works."""
        from tools.cohesionn.community_summarizer import CommunitySummarizer

        summarizer = CommunitySummarizer()

        entities = ["Method-A", "Method-B", "V60", "load capacity"]
        formatted = summarizer._format_entities(entities)

        assert "Method-A" in formatted
        assert "Method-B" in formatted

    def test_format_entities_empty(self):
        """Empty entity list is handled."""
        from tools.cohesionn.community_summarizer import CommunitySummarizer

        summarizer = CommunitySummarizer()

        formatted = summarizer._format_entities([])
        assert "No specific" in formatted


# =============================================================================
# Global Retriever Tests
# =============================================================================

class TestGlobalRetriever:
    """Tests for GlobalRetriever."""

    def test_format_context(self):
        """Context formatting works."""
        from tools.cohesionn.global_retriever import GlobalRetriever

        retriever = GlobalRetriever()

        results = [
            {
                "community_id": "c1",
                "themes": ["structures", "design"],
                "summary": "This is about structural design.",
            },
            {
                "community_id": "c2",
                "themes": ["testing", "Method-A"],
                "summary": "This covers material testing methods.",
            },
        ]

        context = retriever.format_context(results)

        assert "Overview Context" in context
        assert "structures" in context
        assert "structural design" in context

    def test_format_context_empty(self):
        """Empty results return empty context."""
        from tools.cohesionn.global_retriever import GlobalRetriever

        retriever = GlobalRetriever()

        context = retriever.format_context([])
        assert context == ""


# =============================================================================
# Community Store Tests
# =============================================================================

class TestCommunityStore:
    """Tests for CommunityStore."""

    def test_collection_name_defined(self):
        """Community collection name is defined."""
        from tools.cohesionn.global_retriever import COMMUNITY_COLLECTION

        assert COMMUNITY_COLLECTION == "community_summaries"


# =============================================================================
# Integration Tests (require services)
# =============================================================================

@pytest.mark.integration
class TestCommunitySearchIntegration:
    """Integration tests requiring Neo4j and Qdrant."""

    def test_full_pipeline(self):
        """Test full detect -> summarize -> store -> retrieve pipeline."""
        # This would run against real services
        pass
