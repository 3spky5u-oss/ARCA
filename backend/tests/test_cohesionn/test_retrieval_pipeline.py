"""
Integration tests for the full RAG retrieval pipeline.

Tests the complete flow: Query -> Dense + Sparse -> RRF -> Rerank -> Diversity
without involving the LLM.

Run with: pytest tests/test_cohesionn/test_retrieval_pipeline.py -v -s
"""

import pytest
from typing import Dict, Any, List


class TestRetrievalPipelineIntegration:
    """Integration tests for full retrieval pipeline."""

    @pytest.fixture
    def pipeline_components(self, real_embedder, real_reranker, real_diversity_reranker, tmp_path):
        """Set up all pipeline components."""
        from tools.cohesionn.sparse_retrieval import BM25Index, reciprocal_rank_fusion

        # Create BM25 index
        bm25_index = BM25Index(topic="integration_test", persist_dir=tmp_path)

        return {
            "embedder": real_embedder,
            "reranker": real_reranker,
            "diversity_reranker": real_diversity_reranker,
            "bm25_index": bm25_index,
            "rrf": reciprocal_rank_fusion,
        }

    @pytest.fixture
    def indexed_content(self, pipeline_components, all_test_content):
        """Index all test content in BM25."""
        bm25 = pipeline_components["bm25_index"]

        doc_ids = list(all_test_content.keys())
        documents = list(all_test_content.values())
        metadatas = [{"source": f"test_{k}.md", "topic": "test"} for k in all_test_content.keys()]

        bm25.add_documents(doc_ids, documents, metadatas)

        return {
            "bm25": bm25,
            "content": all_test_content,
            "doc_ids": doc_ids,
        }

    def test_full_pipeline_produces_results(self, pipeline_components, indexed_content, all_test_content):
        """Full pipeline should produce ranked results."""
        import numpy as np

        embedder = pipeline_components["embedder"]
        reranker = pipeline_components["reranker"]
        diversity_reranker = pipeline_components["diversity_reranker"]
        rrf = pipeline_components["rrf"]
        bm25 = indexed_content["bm25"]

        query = "What are threshold values for high-performance systems?"

        # Step 1: Dense retrieval (simulated via embedding similarity)
        query_emb = np.array(embedder.embed_query(query))
        dense_results = []
        for doc_id, content in all_test_content.items():
            doc_emb = np.array(embedder.embed_document(content))
            score = float(np.dot(query_emb, doc_emb))
            dense_results.append({
                "id": doc_id,
                "content": content,
                "score": score,
                "metadata": {"source": f"test_{doc_id}.md"},
            })
        dense_results.sort(key=lambda x: x["score"], reverse=True)

        # Step 2: Sparse retrieval (BM25)
        sparse_results = bm25.search(query, n_results=10)
        # Add content to sparse results
        for r in sparse_results:
            r["content"] = all_test_content.get(r["id"], "")

        # Step 3: RRF fusion
        fused = rrf(dense_results, sparse_results, dense_weight=0.7, sparse_weight=0.3)

        # Step 4: Cross-encoder reranking
        reranked = reranker.rerank(query, fused, top_k=10)

        # Step 5: Diversity reranking
        final = diversity_reranker.rerank(reranked, top_k=5)

        # Verify results
        assert len(final) > 0
        assert len(final) <= 5
        assert all("rerank_score" in r for r in final)

        # Specialized content should be in top results
        top_ids = [r["id"] for r in final[:3]]
        assert "spt_n60" in top_ids, f"Specialized content should be in top 3, got: {top_ids}"

    def test_pipeline_handles_exact_term_queries(self, pipeline_components, indexed_content, all_test_content):
        """Pipeline should handle exact technical term queries."""
        import numpy as np

        embedder = pipeline_components["embedder"]
        reranker = pipeline_components["reranker"]
        rrf = pipeline_components["rrf"]
        bm25 = indexed_content["bm25"]

        # Exact term that BM25 should boost
        query = "V60"

        # Dense retrieval
        query_emb = np.array(embedder.embed_query(query))
        dense_results = []
        for doc_id, content in all_test_content.items():
            doc_emb = np.array(embedder.embed_document(content))
            score = float(np.dot(query_emb, doc_emb))
            dense_results.append({
                "id": doc_id,
                "content": content,
                "score": score,
                "metadata": {"source": f"test_{doc_id}.md"},
            })
        dense_results.sort(key=lambda x: x["score"], reverse=True)

        # Sparse retrieval
        sparse_results = bm25.search(query, n_results=10)
        for r in sparse_results:
            r["content"] = all_test_content.get(r["id"], "")

        # BM25 should have spt_n60 with high score
        sparse_ids = [r["id"] for r in sparse_results]
        if sparse_results:
            assert "spt_n60" in sparse_ids, "BM25 should find specialized content for V60 query"

        # Fusion
        fused = rrf(dense_results, sparse_results)

        # Rerank
        reranked = reranker.rerank(query, fused, top_k=5)

        # Specialized content should be top
        assert reranked[0]["id"] == "spt_n60" or "V60" in reranked[0]["content"]

    def test_pipeline_rrf_boosts_hybrid_matches(self, pipeline_components, indexed_content, all_test_content):
        """Documents matching both dense and sparse should score higher in RRF."""
        import numpy as np

        embedder = pipeline_components["embedder"]
        rrf = pipeline_components["rrf"]
        bm25 = indexed_content["bm25"]

        query = "load capacity factor of safety"

        # Dense
        query_emb = np.array(embedder.embed_query(query))
        dense_results = []
        for doc_id, content in all_test_content.items():
            doc_emb = np.array(embedder.embed_document(content))
            score = float(np.dot(query_emb, doc_emb))
            dense_results.append({"id": doc_id, "content": content, "score": score, "metadata": {}})
        dense_results.sort(key=lambda x: x["score"], reverse=True)

        # Sparse
        sparse_results = bm25.search(query, n_results=10)
        for r in sparse_results:
            r["content"] = all_test_content.get(r["id"], "")

        # Fusion
        fused = rrf(dense_results, sparse_results)

        # Check that bearing_capacity is boosted (appears in both)
        bc_result = next((r for r in fused if r["id"] == "bearing_capacity"), None)
        if bc_result:
            # It should have relatively high RRF score
            rrf_scores = [r["rrf_score"] for r in fused]
            bc_rank = rrf_scores.index(bc_result["rrf_score"]) + 1
            assert bc_rank <= 3, f"Load capacity should be top 3 in RRF, got rank {bc_rank}"


class TestPipelineQualityChecks:
    """Quality checks for the integrated pipeline."""

    @pytest.fixture
    def full_pipeline(self, real_embedder, real_reranker, real_diversity_reranker, tmp_path, all_test_content):
        """Create a complete pipeline function."""
        import numpy as np
        from tools.cohesionn.sparse_retrieval import BM25Index, reciprocal_rank_fusion

        bm25 = BM25Index(topic="quality_test", persist_dir=tmp_path)
        doc_ids = list(all_test_content.keys())
        documents = list(all_test_content.values())
        metadatas = [{"source": f"test_{k}.md"} for k in all_test_content.keys()]
        bm25.add_documents(doc_ids, documents, metadatas)

        def pipeline(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
            # Dense
            query_emb = np.array(real_embedder.embed_query(query))
            dense_results = []
            for doc_id, content in all_test_content.items():
                doc_emb = np.array(real_embedder.embed_document(content))
                score = float(np.dot(query_emb, doc_emb))
                dense_results.append({"id": doc_id, "content": content, "score": score, "metadata": {"source": f"test_{doc_id}.md"}})
            dense_results.sort(key=lambda x: x["score"], reverse=True)

            # Sparse
            sparse_results = bm25.search(query, n_results=20)
            for r in sparse_results:
                r["content"] = all_test_content.get(r["id"], "")

            # Fusion
            fused = reciprocal_rank_fusion(dense_results, sparse_results)

            # Rerank
            reranked = real_reranker.rerank(query, fused, top_k=top_k * 3)

            # Diversity
            final = real_diversity_reranker.rerank(reranked, top_k=top_k)

            return final

        return pipeline

    @pytest.mark.parametrize("query,expected_top", [
        ("What are threshold values for high-performance systems?", "spt_n60"),
        ("load capacity safety factor", "bearing_capacity"),
        ("component testing settlement", "pile_load_test"),
        ("environmental factor depth seasonal variation", "frost_depth"),
    ])
    def test_correct_content_ranked_first(self, full_pipeline, query, expected_top):
        """Correct domain content should be ranked first."""
        results = full_pipeline(query, top_k=5)

        assert len(results) > 0
        top_id = results[0]["id"]

        assert top_id == expected_top, (
            f"Query: '{query}'\n"
            f"Expected top: {expected_top}\n"
            f"Got: {top_id}\n"
            f"Top 3: {[(r['id'], round(r['rerank_score'], 3)) for r in results[:3]]}"
        )

    def test_lrfd_not_in_top_for_specialized_queries(self, full_pipeline):
        """LRFD content should not dominate specialized queries."""
        specialized_queries = [
            "threshold values high-performance systems",
            "capacity analysis equation methods",
            "component testing procedure",
        ]

        for query in specialized_queries:
            results = full_pipeline(query, top_k=3)

            top_ids = [r["id"] for r in results]

            # LRFD content should not be #1
            assert results[0]["id"] not in ["lrfd_factors", "lrfd_combinations", "structural_steel"], (
                f"Query: '{query}' incorrectly ranked LRFD content first.\n"
                f"Top 3: {top_ids}"
            )


class TestPipelinePerformance:
    """Performance-related tests for the pipeline."""

    def test_pipeline_completes_in_reasonable_time(self, real_embedder, real_reranker, all_test_content):
        """Pipeline should complete within reasonable time."""
        import time
        import numpy as np

        query = "load capacity factor of safety"

        start = time.perf_counter()

        # Dense retrieval
        query_emb = np.array(real_embedder.embed_query(query))
        for content in all_test_content.values():
            _ = np.array(real_embedder.embed_document(content))

        # Reranking
        results = [
            {"id": k, "content": v, "metadata": {}}
            for k, v in all_test_content.items()
        ]
        _ = real_reranker.rerank(query, results, top_k=5)

        elapsed = time.perf_counter() - start

        # Should complete in under 10 seconds for small test set
        assert elapsed < 10.0, f"Pipeline took {elapsed:.1f}s, expected < 10s"

    def test_consistent_results_across_runs(self, real_embedder, real_reranker, all_test_content):
        """Same query should produce same results."""
        query = "threshold values"

        results = [
            {"id": k, "content": v, "metadata": {}}
            for k, v in all_test_content.items()
        ]

        run1 = real_reranker.rerank(query, results.copy(), top_k=5)
        run2 = real_reranker.rerank(query, results.copy(), top_k=5)

        ids1 = [r["id"] for r in run1]
        ids2 = [r["id"] for r in run2]

        assert ids1 == ids2, "Results should be consistent across runs"
