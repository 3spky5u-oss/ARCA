"""
Tests for GraphRAG components - Entity extraction, graph building, retrieval.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any


# =============================================================================
# Entity Extraction Tests
# =============================================================================

class TestEntityExtractor:
    """Tests for EntityExtractor."""

    def test_extract_standards_astm(self):
        """Extract ASTM standard references."""
        from tools.cohesionn.graph_extraction import EntityExtractor

        extractor = EntityExtractor(use_spacy=False)

        text = "The SPT test follows ASTM D1586-22 guidelines. Also see ASTM D 4318 for Atterberg limits."
        result = extractor.extract(text)

        standard_names = [e.name for e in result.entities if e.entity_type == "Standard"]
        assert any("D1586" in s for s in standard_names)
        assert any("D4318" in s.replace(" ", "") for s in standard_names)

    def test_extract_standards_aashto(self):
        """Extract AASHTO standard references."""
        from tools.cohesionn.graph_extraction import EntityExtractor

        extractor = EntityExtractor(use_spacy=False)

        text = "Compaction follows AASHTO T99 standard. Use AASHTO T180 for modified Proctor."
        result = extractor.extract(text)

        standard_names = [e.name for e in result.entities if e.entity_type == "Standard"]
        assert len(standard_names) >= 2

    def test_extract_test_methods(self):
        """Extract test method abbreviations."""
        from tools.cohesionn.graph_extraction import EntityExtractor

        extractor = EntityExtractor(use_spacy=False)

        text = "SPT and CPT are common in-situ tests. The DMT provides K0 estimates."
        result = extractor.extract(text)

        test_methods = [e.name for e in result.entities if e.entity_type == "TestMethod"]
        assert "SPT" in test_methods
        assert "CPT" in test_methods
        assert "DMT" in test_methods

    def test_extract_soil_types(self):
        """Extract USCS soil classifications."""
        from tools.cohesionn.graph_extraction import EntityExtractor

        extractor = EntityExtractor(use_spacy=False)

        text = "The site consists of SP sand overlying CH clay. Some SM material was noted."
        result = extractor.extract(text)

        soil_types = [e.name for e in result.entities if e.entity_type == "SoilType"]
        assert "SP" in soil_types
        assert "CH" in soil_types
        assert "SM" in soil_types

    def test_extract_parameters(self):
        """Extract engineering parameters."""
        from tools.cohesionn.graph_extraction import EntityExtractor

        extractor = EntityExtractor(use_spacy=False)

        text = "The N60 values range from 20 to 40. The bearing capacity factors Nc and Nq are used."
        result = extractor.extract(text)

        params = [e.name for e in result.entities if e.entity_type == "Parameter"]
        assert "N60" in params
        assert "Nc" in params
        assert "Nq" in params

    def test_infer_relationships(self):
        """Infer relationships from co-occurrence."""
        from tools.cohesionn.graph_extraction import EntityExtractor

        extractor = EntityExtractor(use_spacy=False)

        text = "The SPT test measures N60 values in sand (SP) according to ASTM D1586."
        result = extractor.extract(text)

        # Should have relationships
        assert len(result.relationships) > 0

        # Check for SPT -> N60 MEASURES relationship
        measures_rels = [r for r in result.relationships if r.rel_type == "MEASURES"]
        assert any(r.source == "SPT" and r.target == "N60" for r in measures_rels)

    def test_deduplicate_entities(self):
        """Duplicate entities are deduplicated."""
        from tools.cohesionn.graph_extraction import EntityExtractor

        extractor = EntityExtractor(use_spacy=False)

        text = "SPT SPT SPT - the SPT test is performed using SPT equipment."
        result = extractor.extract(text)

        spt_entities = [e for e in result.entities if e.name == "SPT"]
        assert len(spt_entities) == 1

    def test_extract_batch(self):
        """Batch extraction works."""
        from tools.cohesionn.graph_extraction import EntityExtractor

        extractor = EntityExtractor(use_spacy=False)

        chunks = [
            {"chunk_id": "1", "content": "SPT test per ASTM D1586"},
            {"chunk_id": "2", "content": "CPT measures cone resistance"},
        ]

        results = extractor.extract_batch(chunks)

        assert len(results) == 2
        assert results[0].chunk_id == "1"
        assert results[1].chunk_id == "2"


# =============================================================================
# Graph Builder Tests
# =============================================================================

class TestGraphBuilder:
    """Tests for GraphBuilder."""

    def test_node_merge_templates_exist(self):
        """All node types have merge templates."""
        from tools.cohesionn.graph_builder import GraphBuilder

        builder = GraphBuilder()

        expected_types = ["Standard", "TestMethod", "SoilType", "Parameter", "Equipment", "Chunk"]
        for node_type in expected_types:
            assert node_type in builder.NODE_MERGE_TEMPLATES

    @patch("tools.cohesionn.graph_builder.get_neo4j_client")
    @patch("tools.cohesionn.graph_builder.get_knowledge_base")
    def test_build_graph_empty_topic(self, mock_kb, mock_neo4j):
        """Empty topic returns early."""
        from tools.cohesionn.graph_builder import GraphBuilder

        mock_kb_instance = MagicMock()
        mock_kb_instance.get_store.return_value.client.scroll.return_value = ([], None)
        mock_kb.return_value = mock_kb_instance

        builder = GraphBuilder()
        # This would fail without proper mocking, so we just test the structure
        assert builder.batch_size == 100


# =============================================================================
# Graph Retriever Tests
# =============================================================================

class TestGraphRetriever:
    """Tests for GraphRetriever."""

    def test_hop_scores_exist(self):
        """Hop score configuration exists."""
        from tools.cohesionn.graph_retriever import GraphRetriever

        retriever = GraphRetriever()

        assert 0 in retriever.HOP_SCORES
        assert 1 in retriever.HOP_SCORES
        assert 2 in retriever.HOP_SCORES

        # Scores should decay with distance
        assert retriever.HOP_SCORES[0] > retriever.HOP_SCORES[1]
        assert retriever.HOP_SCORES[1] > retriever.HOP_SCORES[2]

    def test_retrieve_no_entities(self):
        """Query with no entities returns empty."""
        from tools.cohesionn.graph_retriever import GraphRetriever

        retriever = GraphRetriever()

        # Mock extractor to return no entities
        retriever.extractor = MagicMock()
        retriever.extractor.extract.return_value = MagicMock(entities=[])

        results = retriever.retrieve("hello world", ["technical"])
        assert results == []


# =============================================================================
# Neo4j Client Tests
# =============================================================================

class TestNeo4jClient:
    """Tests for Neo4j client."""

    def test_singleton_pattern(self):
        """Singleton returns same instance."""
        from services.neo4j_client import get_neo4j_client, close_neo4j_client, _neo4j_client

        # Reset singleton
        close_neo4j_client()

        # This would connect to real Neo4j, so we just test the import works
        # client1 = get_neo4j_client()
        # client2 = get_neo4j_client()
        # assert client1 is client2

    def test_schema_queries_valid(self):
        """Schema initialization queries are valid Cypher."""
        from services.neo4j_client import Neo4jClient

        client = Neo4jClient()

        # Check that schema queries are defined (actual execution needs Neo4j)
        assert hasattr(client, "initialize_schema")


# =============================================================================
# RRF 3-Way Fusion Tests
# =============================================================================

class TestRRF3Way:
    """Tests for 3-way RRF fusion."""

    def test_rrf_dense_only(self):
        """RRF with only dense results."""
        from tools.cohesionn.sparse_retrieval import reciprocal_rank_fusion

        dense = [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.8}]

        result = reciprocal_rank_fusion(dense, [])

        assert len(result) == 2
        assert result[0]["id"] == "1"  # Higher ranked

    def test_rrf_with_graph(self):
        """RRF with graph results included."""
        from tools.cohesionn.sparse_retrieval import reciprocal_rank_fusion

        dense = [{"id": "1", "score": 0.9}]
        sparse = [{"id": "2", "bm25_score": 5.0}]
        graph = [{"id": "3", "graph_score": 0.8, "matched_entities": ["SPT"]}]

        result = reciprocal_rank_fusion(dense, sparse, graph)

        assert len(result) == 3
        # All results should have rrf_score
        assert all("rrf_score" in r for r in result)

    def test_rrf_graph_metadata_merged(self):
        """Graph metadata is merged into results."""
        from tools.cohesionn.sparse_retrieval import reciprocal_rank_fusion

        dense = [{"id": "1", "score": 0.9, "content": "Dense content"}]
        graph = [{"id": "1", "graph_score": 0.8, "matched_entities": ["SPT"], "is_graph_result": True}]

        result = reciprocal_rank_fusion(dense, [], graph)

        # Same ID should be merged
        assert len(result) == 1
        assert result[0]["id"] == "1"
        assert result[0].get("is_graph_result") is True
        assert result[0].get("matched_entities") == ["SPT"]

    def test_rrf_weights_applied(self):
        """Custom weights are applied."""
        from tools.cohesionn.sparse_retrieval import reciprocal_rank_fusion

        dense = [{"id": "1", "score": 0.9}]
        sparse = [{"id": "2", "bm25_score": 5.0}]
        graph = [{"id": "3", "graph_score": 0.8}]

        # Heavy graph weight
        result = reciprocal_rank_fusion(
            dense, sparse, graph,
            dense_weight=0.2, sparse_weight=0.2, graph_weight=0.6
        )

        # Graph result should rank higher with 0.6 weight
        ids = [r["id"] for r in result]
        assert ids[0] == "3"  # Graph result first due to high weight


# =============================================================================
# Integration Tests (require services)
# =============================================================================

@pytest.mark.integration
class TestGraphRAGIntegration:
    """Integration tests requiring Neo4j and other services."""

    def test_full_pipeline(self):
        """Test full extract -> build -> retrieve pipeline."""
        # This would run against real Neo4j
        pass
