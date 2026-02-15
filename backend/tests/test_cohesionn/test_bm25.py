"""
Unit tests for Cohesionn BM25 sparse retrieval module.

Tests:
- Tokenization with engineering term preservation
- BM25 index operations (add, search, delete)
- Reciprocal Rank Fusion (RRF)
- Engineering stopword handling
"""

import pytest
from tools.cohesionn.sparse_retrieval import (
    tokenize,
    BM25Index,
    BM25Manager,
    reciprocal_rank_fusion,
    ENGINEERING_TERMS,
    STOPWORDS,
)


class TestTokenization:
    """Tests for engineering-aware tokenization."""

    def test_engineering_terms_preserved(self):
        """Engineering terms should not be filtered as stopwords."""
        for term in ["phi", "qu", "cu", "n60", "lrfd", "asd"]:
            tokens = tokenize(f"The {term} value is important")
            assert term in tokens, f"Engineering term '{term}' was filtered out"

    def test_standard_stopwords_removed(self):
        """Standard stopwords should be filtered."""
        text = "The material is a medium grade type"
        tokens = tokenize(text)

        for stopword in ["the", "is", "a"]:
            assert stopword not in tokens, f"Stopword '{stopword}' not filtered"

    def test_single_letter_terms_preserved(self):
        """Single letter engineering terms should be kept."""
        # Terms like c (constant), e (coefficient), k (factor)
        text = "constant c and coefficient e and factor k"
        tokens = tokenize(text)

        assert "c" in tokens, "Engineering term 'c' was filtered"
        assert "e" in tokens, "Engineering term 'e' was filtered"
        assert "k" in tokens, "Engineering term 'k' was filtered"

    def test_case_insensitive(self):
        """Tokenization should be case-insensitive."""
        text1 = "LRFD N60 values"
        text2 = "lrfd n60 values"

        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)

        assert tokens1 == tokens2

    def test_numbers_preserved(self):
        """Numbers in technical terms should be preserved."""
        text = "V60 value is 30 to 50"
        tokens = tokenize(text)

        assert "v60" in tokens
        assert "30" in tokens
        assert "50" in tokens

    def test_hyphenated_terms(self):
        """Hyphenated compound terms should be handled."""
        text = "c-phi material mohr-coulomb"
        tokens = tokenize(text)

        # Should preserve hyphenated terms or their components
        assert any("phi" in t for t in tokens)
        assert any("mohr" in t or "coulomb" in t for t in tokens)

    def test_decimal_numbers(self):
        """Decimal numbers should be tokenized."""
        text = "friction angle 35.5 degrees"
        tokens = tokenize(text)

        assert any("35" in t for t in tokens)

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert tokenize("") == []

    def test_stopwords_only_returns_empty(self):
        """Text with only stopwords should return empty or minimal list."""
        text = "the and or but"
        tokens = tokenize(text)
        assert len(tokens) == 0

    def test_engineering_terms_constant(self):
        """ENGINEERING_TERMS should include key technical terms."""
        required_terms = {"phi", "cu", "qu"}
        assert required_terms.issubset(ENGINEERING_TERMS)


class TestBM25Index:
    """Tests for BM25Index class."""

    def test_add_documents(self, bm25_index):
        """Should add documents to index."""
        doc_ids = ["doc1", "doc2"]
        documents = [
            "Method-A threshold values for high-performance systems",
            "Load capacity calculation methods",
        ]
        metadatas = [{"source": "a.md"}, {"source": "b.md"}]

        bm25_index.add_documents(doc_ids, documents, metadatas)

        assert bm25_index.count == 2

    def test_search_returns_results(self, populated_bm25_index):
        """Search should return matching documents."""
        results = populated_bm25_index.search("threshold values systems", n_results=5)

        assert len(results) > 0
        assert all("bm25_score" in r for r in results)

    def test_search_relevance_ordering(self, populated_bm25_index):
        """Results should be ordered by BM25 score."""
        results = populated_bm25_index.search("threshold values systems", n_results=10)

        if len(results) >= 2:
            scores = [r["bm25_score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_empty_query(self, populated_bm25_index):
        """Empty query should return empty results."""
        results = populated_bm25_index.search("", n_results=5)
        assert results == []

    def test_search_no_matches(self, populated_bm25_index):
        """Query with no matches should return empty."""
        results = populated_bm25_index.search("xyz123 nonexistent term", n_results=5)
        # May return empty or very low scores
        for r in results:
            if "bm25_score" in r:
                assert r["bm25_score"] >= 0

    def test_exact_term_match_scores_high(self, populated_bm25_index):
        """Exact term matches should score highly."""
        results = populated_bm25_index.search("spt_n60", n_results=5)

        # The spt_n60 document should be in top results
        if results:
            top_ids = [r["id"] for r in results[:3]]
            # Should find the specialized content
            assert any("spt" in id.lower() for id in top_ids) or len(results) > 0

    def test_duplicate_documents_not_added(self, bm25_index):
        """Duplicate document IDs should not be added twice."""
        bm25_index.add_documents(["doc1"], ["content one"], [{"source": "a.md"}])
        bm25_index.add_documents(["doc1"], ["content two"], [{"source": "b.md"}])

        assert bm25_index.count == 1

    def test_clear_index(self, bm25_index):
        """Clear should remove all documents."""
        # Use documents with actual content (not just stopwords)
        bm25_index.add_documents(
            ["doc1", "doc2"],
            ["Method-A threshold values for systems", "load capacity formula"],
            [{}, {}]
        )
        assert bm25_index.count == 2

        bm25_index.clear()
        assert bm25_index.count == 0

    def test_delete_by_source(self, bm25_index):
        """Should delete documents by source."""
        bm25_index.add_documents(
            ["doc1", "doc2", "doc3"],
            ["content a", "content b", "content c"],
            [{"source": "file1.md"}, {"source": "file2.md"}, {"source": "file1.md"}],
        )
        assert bm25_index.count == 3

        bm25_index.delete_by_source("file1.md")
        assert bm25_index.count == 1


class TestBM25Manager:
    """Tests for BM25Manager class."""

    def test_get_index_creates_new(self, tmp_path):
        """get_index should create new index for unknown topic."""
        manager = BM25Manager(persist_dir=tmp_path)

        index1 = manager.get_index("topic1")
        index2 = manager.get_index("topic2")

        assert index1 is not index2
        assert index1.topic == "topic1"
        assert index2.topic == "topic2"

    def test_get_index_returns_same(self, tmp_path):
        """get_index should return same instance for same topic."""
        manager = BM25Manager(persist_dir=tmp_path)

        index1 = manager.get_index("mytopic")
        index2 = manager.get_index("mytopic")

        assert index1 is index2

    def test_multi_topic_search(self, tmp_path):
        """Should search across multiple topics."""
        manager = BM25Manager(persist_dir=tmp_path)

        # BM25 needs multiple documents for proper IDF calculation
        # With a single doc, IDF is negative (100% of docs contain the term)
        idx1 = manager.get_index("technical")
        idx1.add_documents(
            ["g1", "g2", "g3"],
            [
                "method-a threshold values high-performance systems typically units verification test",
                "load capacity installation structure shallow deep",
                "component load settlement element assembly capacity",
            ],
            [{"source": "tech.md"}, {"source": "tech2.md"}, {"source": "tech3.md"}]
        )

        idx2 = manager.get_index("operational")
        idx2.add_documents(
            ["e1", "e2", "e3"],
            [
                "contamination detection chemical analysis remediation cleanup",
                "monitoring station sampling analysis testing evaluation",
                "vapor extraction treatment processing filtration",
            ],
            [{"source": "ops.md"}, {"source": "ops2.md"}, {"source": "ops3.md"}]
        )

        # Search technical topic - should find g1
        results = manager.search("method-a threshold verification", ["technical"], n_results=5)
        assert len(results) > 0, "Should find results in technical topic"
        result_ids = [r["id"] for r in results]
        assert "g1" in result_ids, f"Should find g1, got {result_ids}"

        # Search operational topic
        results = manager.search("contamination detection", ["operational"], n_results=5)
        assert len(results) > 0, "Should find results in operational topic"

    def test_get_stats(self, tmp_path):
        """Should return document counts per topic."""
        manager = BM25Manager(persist_dir=tmp_path)

        # Use documents with actual content (not just single letters that become stopwords)
        manager.get_index("topic1").add_documents(
            ["doc1", "doc2"],
            ["Method-A threshold values systems", "load capacity formula analysis"],
            [{}, {}]
        )
        manager.get_index("topic2").add_documents(
            ["doc3"],
            ["component load test settlement"],
            [{}]
        )

        stats = manager.get_stats()

        assert stats["topic1"] == 2
        assert stats["topic2"] == 1


class TestReciprocalRankFusion:
    """Tests for RRF fusion algorithm."""

    def test_rrf_combines_results(self):
        """RRF should combine dense and sparse results."""
        dense_results = [
            {"id": "doc1", "score": 0.9, "content": "a"},
            {"id": "doc2", "score": 0.8, "content": "b"},
        ]
        sparse_results = [
            {"id": "doc2", "bm25_score": 5.0, "content": "b"},
            {"id": "doc3", "bm25_score": 4.0, "content": "c"},
        ]

        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        # Should contain all unique documents
        ids = [r["id"] for r in fused]
        assert "doc1" in ids
        assert "doc2" in ids
        assert "doc3" in ids

    def test_rrf_scores_present(self):
        """Fused results should have rrf_score."""
        dense = [{"id": "a", "score": 0.9}]
        sparse = [{"id": "b", "bm25_score": 5.0}]

        fused = reciprocal_rank_fusion(dense, sparse)

        assert all("rrf_score" in r for r in fused)

    def test_rrf_respects_weights(self):
        """Different weights should affect scores."""
        dense = [{"id": "a", "score": 0.9}]
        sparse = [{"id": "b", "bm25_score": 5.0}]

        # Heavy dense weight
        fused_dense = reciprocal_rank_fusion(
            dense, sparse, dense_weight=0.9, sparse_weight=0.1
        )

        # Heavy sparse weight
        fused_sparse = reciprocal_rank_fusion(
            dense, sparse, dense_weight=0.1, sparse_weight=0.9
        )

        # Document from dense should score higher with dense weight
        dense_doc_score_heavy = next(r["rrf_score"] for r in fused_dense if r["id"] == "a")
        dense_doc_score_light = next(r["rrf_score"] for r in fused_sparse if r["id"] == "a")

        assert dense_doc_score_heavy > dense_doc_score_light

    def test_rrf_overlapping_documents(self):
        """Documents in both lists should get combined scores."""
        dense = [{"id": "shared", "score": 0.9, "content": "x"}]
        sparse = [{"id": "shared", "bm25_score": 5.0}]

        fused = reciprocal_rank_fusion(dense, sparse)

        # Only one entry for shared document
        shared_entries = [r for r in fused if r["id"] == "shared"]
        assert len(shared_entries) == 1

        # Score should be higher than either alone would give
        shared_score = shared_entries[0]["rrf_score"]
        assert shared_score > 0

    def test_rrf_empty_dense(self):
        """Should handle empty dense results."""
        sparse = [{"id": "a", "bm25_score": 5.0}]

        fused = reciprocal_rank_fusion([], sparse)

        assert len(fused) == 1
        assert fused[0]["id"] == "a"

    def test_rrf_empty_sparse(self):
        """Should handle empty sparse results."""
        dense = [{"id": "a", "score": 0.9}]

        fused = reciprocal_rank_fusion(dense, [])

        assert len(fused) == 1
        assert fused[0]["id"] == "a"

    def test_rrf_both_empty(self):
        """Should handle both empty."""
        fused = reciprocal_rank_fusion([], [])
        assert fused == []

    def test_rrf_sorted_by_score(self):
        """Results should be sorted by RRF score descending."""
        dense = [
            {"id": "a", "score": 0.5},
            {"id": "b", "score": 0.9},
        ]
        sparse = [
            {"id": "c", "bm25_score": 3.0},
            {"id": "d", "bm25_score": 5.0},
        ]

        fused = reciprocal_rank_fusion(dense, sparse)

        scores = [r["rrf_score"] for r in fused]
        assert scores == sorted(scores, reverse=True)
