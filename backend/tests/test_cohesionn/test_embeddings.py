"""
Unit tests for Cohesionn embeddings module.

Tests:
- Embedding dimensions and normalization
- Query vs document embedding differences
- Cache behavior and hit rates
- Model family detection
"""

import pytest
import numpy as np


class TestMockEmbedder:
    """Tests using mock embedder (fast, no GPU)."""

    def test_embed_query_returns_correct_dimension(self, mock_embedder):
        """Query embedding should return correct dimension."""
        embedding = mock_embedder.embed_query("test query")
        assert len(embedding) == mock_embedder.dimension

    def test_embed_document_returns_correct_dimension(self, mock_embedder):
        """Document embedding should return correct dimension."""
        embedding = mock_embedder.embed_document("test document content")
        assert len(embedding) == mock_embedder.dimension

    def test_embeddings_are_normalized(self, mock_embedder):
        """Embeddings should be L2 normalized (unit vectors)."""
        embedding = mock_embedder.embed_query("test query")
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01, f"Embedding norm {norm} is not 1.0"

    def test_same_query_same_embedding(self, mock_embedder):
        """Same query should produce identical embedding."""
        emb1 = mock_embedder.embed_query("what is V60?")
        emb2 = mock_embedder.embed_query("what is V60?")
        assert emb1 == emb2

    def test_different_queries_different_embeddings(self, mock_embedder):
        """Different queries should produce different embeddings."""
        emb1 = mock_embedder.embed_query("what is V60?")
        emb2 = mock_embedder.embed_query("load capacity formula")
        assert emb1 != emb2

    def test_batch_embed_documents(self, mock_embedder):
        """Batch embedding should process multiple documents."""
        docs = [
            "First document about material mechanics",
            "Second document about structural design",
            "Third document about component assembly",
        ]
        embeddings = mock_embedder.embed_documents(docs)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == mock_embedder.dimension

    def test_cache_clear(self, mock_embedder):
        """Cache should be clearable."""
        mock_embedder.embed_query("test")
        mock_embedder.clear_cache()
        # Should not raise
        mock_embedder.embed_query("test")


class TestRealEmbedder:
    """Tests using real embedder (requires models)."""

    def test_real_embedding_dimension(self, real_embedder):
        """Real embedder should produce expected dimension."""
        embedding = real_embedder.embed_query("test query")
        assert len(embedding) == real_embedder.dimension
        # Qwen3 and BGE both use 1024 dimensions
        assert real_embedder.dimension == 1024

    def test_real_embeddings_normalized(self, real_embedder):
        """Real embeddings should be L2 normalized."""
        embedding = real_embedder.embed_query("test query")
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    def test_model_family_detection(self, real_embedder):
        """Model family should be correctly detected."""
        family = real_embedder.model_family
        assert family in ["qwen3", "bge", "nomic", "generic"]

    def test_query_cache_works(self, real_embedder):
        """Cache should provide hits on repeated queries."""
        # Clear any existing cache
        real_embedder.clear_cache()

        # First call - cache miss
        real_embedder.embed_query("test query for caching")
        stats1 = real_embedder.cache_stats

        # Second call - cache hit
        real_embedder.embed_query("test query for caching")
        stats2 = real_embedder.cache_stats

        assert stats2["hits"] > stats1["hits"], "Cache should record hit on repeated query"

    def test_similar_queries_similar_embeddings(self, real_embedder):
        """Semantically similar queries should have high cosine similarity."""
        emb1 = np.array(real_embedder.embed_query("What is the load capacity?"))
        emb2 = np.array(real_embedder.embed_query("How to calculate load capacity?"))

        # Cosine similarity (embeddings are normalized)
        similarity = float(np.dot(emb1, emb2))

        assert similarity > 0.7, f"Similar queries should have high similarity, got {similarity}"

    def test_dissimilar_queries_lower_similarity(self, real_embedder):
        """Dissimilar queries should have lower similarity."""
        emb1 = np.array(real_embedder.embed_query("load capacity formula"))
        emb2 = np.array(real_embedder.embed_query("recipe for chocolate cake"))

        similarity = float(np.dot(emb1, emb2))

        # Should still be somewhat positive but not high
        assert similarity < 0.5, f"Dissimilar queries should have low similarity, got {similarity}"

    def test_batch_queries_match_individual(self, real_embedder):
        """Batch query embeddings should be very similar to individual embeddings."""
        queries = ["query one", "query two"]

        # Individual
        emb1 = np.array(real_embedder.embed_query(queries[0]))
        emb2 = np.array(real_embedder.embed_query(queries[1]))

        # Batch
        batch_embs = real_embedder.embed_queries(queries)

        # Should be very similar (high cosine similarity)
        # Note: Small numerical differences can occur between individual and batch processing
        sim1 = float(np.dot(emb1, batch_embs[0]))
        sim2 = float(np.dot(emb2, batch_embs[1]))

        assert sim1 > 0.99, f"Batch embedding 1 too different: similarity={sim1}"
        assert sim2 > 0.99, f"Batch embedding 2 too different: similarity={sim2}"


class TestEmbeddingQuality:
    """Tests for embedding quality on domain content."""

    def test_domain_query_matches_domain_content(
        self,
        real_embedder,
        domain_content,
        generic_content,
    ):
        """Specialized queries should be closer to specialized content than generic."""
        query = "What are threshold values for high-performance systems?"

        query_emb = np.array(real_embedder.embed_query(query))

        # Get similarities to domain and generic content
        domain_sims = []
        for content in domain_content.values():
            doc_emb = np.array(real_embedder.embed_document(content))
            sim = float(np.dot(query_emb, doc_emb))
            domain_sims.append(sim)

        generic_sims = []
        for content in generic_content.values():
            doc_emb = np.array(real_embedder.embed_document(content))
            sim = float(np.dot(query_emb, doc_emb))
            generic_sims.append(sim)

        # Domain content should have higher average similarity
        avg_domain = sum(domain_sims) / len(domain_sims)
        avg_generic = sum(generic_sims) / len(generic_sims)

        assert avg_domain > avg_generic, (
            f"Domain content should have higher similarity to domain queries.\n"
            f"Avg domain: {avg_domain:.3f}, Avg generic: {avg_generic:.3f}"
        )

    def test_technical_term_discrimination(self, real_embedder):
        """Technical terms should have distinct embeddings."""
        terms = ["Method-A", "Method-B", "V60", "load capacity", "settlement", "LRFD"]

        embeddings = [real_embedder.embed_query(term) for term in terms]

        # Check pairwise similarities aren't too high (terms are distinct)
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                # Should be somewhat similar (all engineering) but not identical
                assert sim < 0.95, (
                    f"Terms '{terms[i]}' and '{terms[j]}' too similar: {sim:.3f}"
                )
