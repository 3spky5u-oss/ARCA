"""
CRITICAL: Content-specific ranking quality tests.

These tests verify that the RAG pipeline correctly ranks specialized
content above generic LRFD/structural content.

The Qwen3 reranker was observed promoting LRFD content (score 0.813) over
specialized content (score 0.688) for domain queries. These tests catch
and prevent this bias.

Run with: pytest tests/test_cohesionn/test_ranking_quality.py -v -s
"""

import pytest
from typing import List, Dict, Any


class TestDomainSpecificRanking:
    """Tests that verify specialized content ranks above generic content."""

    @pytest.mark.parametrize("query_data", [
        pytest.param(
            {
                "query": "What are typical threshold values for high-performance systems?",
                "preferred": ["spt_n60"],
                "non_preferred": ["lrfd_factors", "lrfd_combinations"],
            },
            id="threshold_vs_lrfd"
        ),
        pytest.param(
            {
                "query": "load capacity safety factor for installations",
                "preferred": ["bearing_capacity"],
                "non_preferred": ["lrfd_factors", "structural_steel"],
            },
            id="capacity_vs_lrfd"
        ),
        pytest.param(
            {
                "query": "component testing procedures and settlement",
                "preferred": ["pile_load_test"],
                "non_preferred": ["lrfd_factors", "structural_steel"],
            },
            id="component_test_vs_lrfd"
        ),
        pytest.param(
            {
                "query": "environmental factor depth seasonal variation",
                "preferred": ["frost_depth"],
                "non_preferred": ["lrfd_factors", "lrfd_combinations"],
            },
            id="env_factor_vs_lrfd"
        ),
    ])
    def test_domain_content_ranks_above_generic(
        self,
        query_data: Dict[str, Any],
        real_reranker,
        all_test_content,
        check_ranking,
        score_margin_threshold,
    ):
        """
        CRITICAL TEST: Specialized content MUST rank above generic LRFD content.

        This test catches the reranker bias bug where Qwen3 incorrectly promotes
        LRFD content over specialized content.
        """
        query = query_data["query"]
        preferred = query_data["preferred"]
        non_preferred = query_data["non_preferred"]

        # Build results from test content
        results = []
        for key, content in all_test_content.items():
            results.append({
                "id": key,
                "content": content,
                "metadata": {"source": f"test_{key}.md"},
            })

        # Rerank
        reranked = real_reranker.rerank(query, results, top_k=len(results))

        # Check ranking preference
        ranking_check = check_ranking(reranked, preferred, non_preferred)

        # Assert with detailed failure message
        assert ranking_check["passed"], (
            f"RERANKER BIAS DETECTED for query: '{query}'\n"
            f"Preferred content ({preferred}) scored: {ranking_check['preferred_scores']}\n"
            f"Non-preferred content ({non_preferred}) scored: {ranking_check['non_preferred_scores']}\n"
            f"Margin: {ranking_check['margin']:.3f} (need > 0)\n"
            f"Min preferred: {ranking_check['min_preferred']:.3f}\n"
            f"Max non-preferred: {ranking_check['max_non_preferred']:.3f}"
        )

        # Also check margin is meaningful (not just barely passing)
        assert ranking_check["margin"] > score_margin_threshold, (
            f"Score margin too small for query: '{query}'\n"
            f"Margin: {ranking_check['margin']:.3f} (need > {score_margin_threshold})\n"
            f"This indicates weak domain preference - investigate reranker."
        )


class TestRerankerScoreDistribution:
    """Tests for score distribution and calibration."""

    def test_domain_query_produces_high_domain_scores(
        self,
        real_reranker,
        domain_content,
        generic_content,
    ):
        """Specialized queries should produce notably higher scores for specialized content."""
        query = "What are threshold values for high-performance systems in load testing?"

        # Combine domain and generic content
        all_content = {**domain_content, **generic_content}
        results = [
            {"id": k, "content": v, "metadata": {"source": f"test_{k}.md"}}
            for k, v in all_content.items()
        ]

        reranked = real_reranker.rerank(query, results, top_k=len(results))

        # Get scores
        domain_scores = [r["rerank_score"] for r in reranked if r["id"] in domain_content]
        generic_scores = [r["rerank_score"] for r in reranked if r["id"] in generic_content]

        avg_domain = sum(domain_scores) / len(domain_scores) if domain_scores else 0
        avg_generic = sum(generic_scores) / len(generic_scores) if generic_scores else 0

        # Domain average should be meaningfully higher
        assert avg_domain > avg_generic, (
            f"Domain content should score higher on average.\n"
            f"Avg domain: {avg_domain:.3f}, Avg generic: {avg_generic:.3f}"
        )

    def test_top_result_is_most_relevant(
        self,
        real_reranker,
        all_test_content,
    ):
        """The top result for a specific query should be the most relevant content."""
        test_cases = [
            ("threshold values high-performance systems", "spt_n60"),
            ("load capacity factor of safety", "bearing_capacity"),
            ("component testing settlement", "pile_load_test"),
            ("environmental factor depth seasonal variation", "frost_depth"),
        ]

        for query, expected_top_id in test_cases:
            results = [
                {"id": k, "content": v, "metadata": {"source": f"test_{k}.md"}}
                for k, v in all_test_content.items()
            ]

            reranked = real_reranker.rerank(query, results, top_k=len(results))
            top_result = reranked[0]

            assert top_result["id"] == expected_top_id, (
                f"Query: '{query}'\n"
                f"Expected top: {expected_top_id}\n"
                f"Got top: {top_result['id']} (score: {top_result['rerank_score']:.3f})\n"
                f"Top 3 results: {[(r['id'], round(r['rerank_score'], 3)) for r in reranked[:3]]}"
            )


class TestLRFDBiasInvestigation:
    """Diagnostic tests to investigate LRFD bias in reranker."""

    def test_isolate_cross_encoder_vs_embedding_bias(
        self,
        real_embedder,
        real_reranker,
        domain_content,
        generic_content,
    ):
        """
        Compare embedding cosine similarity vs cross-encoder scores.

        If embeddings rank correctly but cross-encoder doesn't,
        the bias is in the reranker model.
        """
        import numpy as np

        query = "What are threshold values for high-performance systems?"
        specialized_content = domain_content["spt_n60"]
        lrfd_content = generic_content["lrfd_factors"]

        # Get embedding similarities
        query_emb = np.array(real_embedder.embed_query(query))
        spec_emb = np.array(real_embedder.embed_document(specialized_content))
        lrfd_emb = np.array(real_embedder.embed_document(lrfd_content))

        # Cosine similarity (embeddings are normalized)
        spec_embedding_sim = float(np.dot(query_emb, spec_emb))
        lrfd_embedding_sim = float(np.dot(query_emb, lrfd_emb))

        # Get cross-encoder scores
        results = [
            {"id": "spt_n60", "content": specialized_content, "metadata": {}},
            {"id": "lrfd_factors", "content": lrfd_content, "metadata": {}},
        ]
        reranked = real_reranker.rerank(query, results, top_k=2)

        spec_ce_score = next(r["rerank_score"] for r in reranked if r["id"] == "spt_n60")
        lrfd_ce_score = next(r["rerank_score"] for r in reranked if r["id"] == "lrfd_factors")

        # Log diagnostic info
        print("\nDiagnostic: Embedding vs Cross-Encoder Comparison")
        print(f"Query: {query}")
        print(f"Specialized content: {specialized_content[:80]}...")
        print(f"LRFD content: {lrfd_content[:80]}...")
        print("\nEmbedding similarities:")
        print(f"  Specialized:  {spec_embedding_sim:.4f}")
        print(f"  LRFD: {lrfd_embedding_sim:.4f}")
        print(f"  Embedding prefers: {'Specialized' if spec_embedding_sim > lrfd_embedding_sim else 'LRFD'}")
        print("\nCross-encoder scores:")
        print(f"  Specialized:  {spec_ce_score:.4f}")
        print(f"  LRFD: {lrfd_ce_score:.4f}")
        print(f"  CE prefers: {'Specialized' if spec_ce_score > lrfd_ce_score else 'LRFD'}")

        # Check if bias is in cross-encoder
        embedding_correct = spec_embedding_sim > lrfd_embedding_sim
        ce_correct = spec_ce_score > lrfd_ce_score

        if embedding_correct and not ce_correct:
            pytest.fail(
                f"CROSS-ENCODER BIAS CONFIRMED:\n"
                f"Embeddings correctly prefer specialized ({spec_embedding_sim:.3f} > {lrfd_embedding_sim:.3f})\n"
                f"But cross-encoder incorrectly prefers LRFD ({lrfd_ce_score:.3f} > {spec_ce_score:.3f})\n"
                f"The reranker model is biased toward structural/LRFD content."
            )

        # Both should prefer specialized
        assert spec_ce_score > lrfd_ce_score, (
            f"Specialized content should score higher than LRFD for this query.\n"
            f"Specialized score: {spec_ce_score:.3f}, LRFD score: {lrfd_ce_score:.3f}"
        )

    def test_raw_score_comparison_specialized_vs_lrfd(
        self,
        real_reranker,
    ):
        """
        Direct comparison of specialized vs LRFD content with controlled test content.

        This is the core test for reranker bias.
        """
        query = "What is V60?"

        specialized_content = "V60 is the corrected threshold value for high-performance systems (30-50 units per cycle)."
        lrfd_content = "LRFD uses load factors 1.25 for dead load and 1.5 for live load."

        results = [
            {"id": "specialized", "content": specialized_content, "metadata": {}},
            {"id": "lrfd", "content": lrfd_content, "metadata": {}},
        ]

        reranked = real_reranker.rerank(query, results, top_k=2)

        spec_score = next(r["rerank_score"] for r in reranked if r["id"] == "specialized")
        lrfd_score = next(r["rerank_score"] for r in reranked if r["id"] == "lrfd")

        print("\nRaw score comparison:")
        print(f"Query: '{query}'")
        print(f"Specialized content: '{specialized_content}'")
        print(f"LRFD content: '{lrfd_content}'")
        print(f"Specialized score: {spec_score:.4f}")
        print(f"LRFD score: {lrfd_score:.4f}")
        print(f"Margin: {spec_score - lrfd_score:.4f}")

        assert spec_score > lrfd_score, (
            f"BUG: Specialized content should rank above LRFD for query '{query}'.\n"
            f"Specialized score: {spec_score:.3f}\n"
            f"LRFD score: {lrfd_score:.3f}\n"
            f"This indicates reranker bias toward LRFD/structural content."
        )

        # Margin should be meaningful
        margin = spec_score - lrfd_score
        assert margin > 0.1, (
            f"Score margin too small ({margin:.3f}). Need > 0.1 for confident ranking.\n"
            f"Weak margins can cause ranking flips with slightly different content."
        )


class TestKnownAnswerValidation:
    """Tests that verify expected content is retrieved for known queries."""

    @pytest.mark.parametrize("query_data", [
        {
            "query": "What are threshold values for high-performance systems?",
            "expected_phrases": ["30", "50", "threshold", "high-performance"],
        },
        {
            "query": "load capacity safety factor",
            "expected_phrases": ["3.0", "FS", "factor"],
        },
        {
            "query": "component testing procedure",
            "expected_phrases": ["25%", "settlement", "increment"],
        },
        {
            "query": "environmental factor depth seasonal",
            "expected_phrases": ["3.0", "4.0", "depth"],
        },
    ])
    def test_known_answer_phrases_found(
        self,
        query_data: Dict[str, Any],
        real_reranker,
        all_test_content,
        check_phrases,
    ):
        """Verify that expected phrases appear in top results."""
        query = query_data["query"]
        expected = query_data["expected_phrases"]

        results = [
            {"id": k, "content": v, "metadata": {"source": f"test_{k}.md"}}
            for k, v in all_test_content.items()
        ]

        reranked = real_reranker.rerank(query, results, top_k=3)

        # Check phrases in top result
        top_content = reranked[0]["content"]
        phrase_check = check_phrases(top_content, expected)

        assert phrase_check["passed"], (
            f"Query: '{query}'\n"
            f"Expected phrases not found in top result.\n"
            f"Found: {phrase_check['found']}\n"
            f"Missing: {phrase_check['missing']}\n"
            f"Match ratio: {phrase_check['match_ratio']:.1%}\n"
            f"Top result content: {top_content[:200]}..."
        )


class TestScoreStability:
    """Tests that scores are consistent across runs."""

    def test_same_query_same_scores(
        self,
        real_reranker,
        all_test_content,
    ):
        """Same query should produce identical scores on repeated runs."""
        query = "What are threshold values for high-performance systems?"

        results = [
            {"id": k, "content": v, "metadata": {"source": f"test_{k}.md"}}
            for k, v in all_test_content.items()
        ]

        # Run twice
        reranked1 = real_reranker.rerank(query, results.copy(), top_k=5)
        reranked2 = real_reranker.rerank(query, results.copy(), top_k=5)

        # Scores should be identical
        for r1, r2 in zip(reranked1, reranked2):
            assert r1["id"] == r2["id"], "Result order changed between runs"
            assert abs(r1["rerank_score"] - r2["rerank_score"]) < 0.001, (
                f"Score instability for {r1['id']}: "
                f"{r1['rerank_score']:.4f} vs {r2['rerank_score']:.4f}"
            )
