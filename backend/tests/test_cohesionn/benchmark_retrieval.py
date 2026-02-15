"""
Benchmark suite for RAG retrieval pipeline performance.

Measures timing for each pipeline stage:
- Query embedding
- Dense retrieval
- BM25 sparse retrieval
- RRF fusion
- Cross-encoder reranking
- MMR diversity reranking

Run with: pytest tests/test_cohesionn/benchmark_retrieval.py -v -s
"""

import pytest
import time
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Result from timing a pipeline stage."""
    stage: str
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class BenchmarkProfiler:
    """Collects timing data for benchmark stages."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def time_stage(self, stage_name: str, **details):
        """Context manager to time a stage."""
        class Timer:
            def __init__(self, profiler, stage, details):
                self.profiler = profiler
                self.stage = stage
                self.details = details
                self.start = None

            def __enter__(self):
                self.start = time.perf_counter()
                return self

            def __exit__(self, *args):
                elapsed_ms = (time.perf_counter() - self.start) * 1000
                self.profiler.results.append(
                    BenchmarkResult(self.stage, elapsed_ms, self.details)
                )

        return Timer(self, stage_name, details)

    def get_summary(self) -> Dict[str, float]:
        """Get timing summary by stage."""
        return {r.stage: r.duration_ms for r in self.results}

    def total_ms(self) -> float:
        """Total time across all stages."""
        return sum(r.duration_ms for r in self.results)

    def print_report(self):
        """Print formatted benchmark report."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        total = self.total_ms()
        for r in self.results:
            pct = (r.duration_ms / total * 100) if total > 0 else 0
            details_str = ", ".join(f"{k}={v}" for k, v in r.details.items())
            print(f"{r.stage:<25} {r.duration_ms:>8.1f}ms ({pct:>5.1f}%)  {details_str}")

        print("-" * 60)
        print(f"{'TOTAL':<25} {total:>8.1f}ms")
        print("=" * 60)


class TestRetrievalBenchmarks:
    """Benchmark tests for retrieval pipeline."""

    @pytest.fixture
    def benchmark_content(self, all_test_content):
        """Expanded content for more realistic benchmarks."""
        # Duplicate content to simulate larger corpus
        expanded = {}
        for i in range(3):  # 3x content
            for k, v in all_test_content.items():
                expanded[f"{k}_{i}"] = v
        return expanded

    def test_benchmark_query_embedding(self, real_embedder):
        """Benchmark query embedding time."""
        profiler = BenchmarkProfiler()

        queries = [
            "What are threshold values for high-performance systems?",
            "load capacity safety factor analysis",
            "component testing settlement acceptance",
            "environmental factor depth seasonal variation",
            "capacity analysis equation methods",
        ]

        # Warm up
        _ = real_embedder.embed_query("warmup")

        for query in queries:
            with profiler.time_stage("embed_query", query=query[:30]):
                _ = real_embedder.embed_query(query)

        profiler.print_report()

        # Assert reasonable performance
        avg_ms = profiler.total_ms() / len(queries)
        assert avg_ms < 100, f"Average query embedding {avg_ms:.1f}ms too slow"

    def test_benchmark_document_embedding(self, real_embedder, benchmark_content):
        """Benchmark document embedding time."""
        profiler = BenchmarkProfiler()

        documents = list(benchmark_content.values())

        with profiler.time_stage("embed_documents", count=len(documents)):
            _ = real_embedder.embed_documents(documents, batch_size=32)

        profiler.print_report()

        # Assert reasonable throughput
        docs_per_sec = len(documents) / (profiler.total_ms() / 1000)
        print(f"Throughput: {docs_per_sec:.1f} docs/sec")
        assert docs_per_sec > 5, f"Document embedding too slow: {docs_per_sec:.1f} docs/sec"

    def test_benchmark_bm25_search(self, populated_bm25_index):
        """Benchmark BM25 search time."""
        profiler = BenchmarkProfiler()

        queries = [
            "threshold values systems",
            "load capacity",
            "component testing",
            "environmental factor depth",
            "LRFD factors",
        ]

        for query in queries:
            with profiler.time_stage("bm25_search", query=query):
                _ = populated_bm25_index.search(query, n_results=20)

        profiler.print_report()

        # BM25 should be very fast
        avg_ms = profiler.total_ms() / len(queries)
        assert avg_ms < 10, f"Average BM25 search {avg_ms:.1f}ms too slow"

    def test_benchmark_reranking(self, real_reranker, benchmark_content):
        """Benchmark cross-encoder reranking time."""
        profiler = BenchmarkProfiler()

        query = "What are threshold values for high-performance systems?"
        results = [
            {"id": k, "content": v, "metadata": {}}
            for k, v in benchmark_content.items()
        ]

        # Benchmark different candidate counts
        for n_candidates in [10, 20, 30]:
            subset = results[:n_candidates]
            with profiler.time_stage("rerank", candidates=n_candidates):
                _ = real_reranker.rerank(query, subset, top_k=5)

        profiler.print_report()

        # Check scaling is reasonable
        times = [r.duration_ms for r in profiler.results]
        # Should scale roughly linearly, not exponentially
        if len(times) >= 2:
            ratio = times[-1] / times[0]
            assert ratio < 5, f"Reranking scaling {ratio:.1f}x too steep"

    def test_benchmark_diversity_reranking(self, real_diversity_reranker, benchmark_content):
        """Benchmark MMR diversity reranking time."""
        profiler = BenchmarkProfiler()

        results = [
            {"id": k, "content": v, "rerank_score": 0.9 - i * 0.01, "metadata": {"source": f"src_{i % 5}.pdf"}}
            for i, (k, v) in enumerate(benchmark_content.items())
        ]

        for top_k in [5, 10, 15]:
            with profiler.time_stage("diversity_rerank", top_k=top_k, candidates=len(results)):
                _ = real_diversity_reranker.rerank(results, top_k=top_k)

        profiler.print_report()

        # MMR should be fast
        assert profiler.total_ms() < 500, "Diversity reranking too slow"

    def test_benchmark_full_pipeline(self, real_embedder, real_reranker, real_diversity_reranker, tmp_path, benchmark_content):
        """Benchmark complete retrieval pipeline."""
        import numpy as np
        from tools.cohesionn.sparse_retrieval import BM25Index, reciprocal_rank_fusion

        profiler = BenchmarkProfiler()

        # Setup
        bm25 = BM25Index(topic="benchmark", persist_dir=tmp_path)
        with profiler.time_stage("setup_bm25", docs=len(benchmark_content)):
            bm25.add_documents(
                list(benchmark_content.keys()),
                list(benchmark_content.values()),
                [{"source": f"test_{k}.md"} for k in benchmark_content.keys()],
            )

        query = "What are threshold values for high-performance systems?"

        # Pipeline stages
        with profiler.time_stage("query_embedding"):
            query_emb = np.array(real_embedder.embed_query(query))

        with profiler.time_stage("dense_retrieval", docs=len(benchmark_content)):
            dense_results = []
            for doc_id, content in benchmark_content.items():
                doc_emb = np.array(real_embedder.embed_document(content))
                score = float(np.dot(query_emb, doc_emb))
                dense_results.append({
                    "id": doc_id, "content": content, "score": score, "metadata": {}
                })
            dense_results.sort(key=lambda x: x["score"], reverse=True)

        with profiler.time_stage("bm25_retrieval"):
            sparse_results = bm25.search(query, n_results=20)
            for r in sparse_results:
                r["content"] = benchmark_content.get(r["id"], "")

        with profiler.time_stage("rrf_fusion"):
            fused = reciprocal_rank_fusion(dense_results[:20], sparse_results)

        with profiler.time_stage("cross_encoder_rerank", candidates=len(fused)):
            reranked = real_reranker.rerank(query, fused, top_k=15)

        with profiler.time_stage("diversity_rerank"):
            final = real_diversity_reranker.rerank(reranked, top_k=5)

        profiler.print_report()

        # Overall pipeline should complete reasonably
        # Note: Most time will be in dense_retrieval and reranking
        print(f"\nFinal results: {len(final)}")
        print(f"Top result: {final[0]['id']}")

        # Identify bottlenecks
        print("\nBOTTLENECKS (>200ms):")
        for r in profiler.results:
            if r.duration_ms > 200:
                print(f"  - {r.stage}: {r.duration_ms:.0f}ms")


class TestCacheEffectiveness:
    """Test cache effectiveness on repeated queries."""

    def test_embedding_cache_speedup(self, real_embedder):
        """Cache should speed up repeated query embedding."""
        real_embedder.clear_cache()

        query = "What are threshold values for high-performance systems?"

        # Cold
        start = time.perf_counter()
        _ = real_embedder.embed_query(query)
        cold_ms = (time.perf_counter() - start) * 1000

        # Warm (cached)
        start = time.perf_counter()
        _ = real_embedder.embed_query(query)
        warm_ms = (time.perf_counter() - start) * 1000

        print("\nCache effectiveness:")
        print(f"  Cold: {cold_ms:.1f}ms")
        print(f"  Warm: {warm_ms:.1f}ms")
        print(f"  Speedup: {cold_ms / warm_ms:.1f}x" if warm_ms > 0 else "  Speedup: inf")

        # Cache should provide significant speedup
        if cold_ms > 1:  # Only check if cold was measurable
            assert warm_ms < cold_ms, "Cache should provide speedup"


class TestScalabilityBenchmarks:
    """Benchmarks for scaling behavior."""

    def test_bm25_scales_linearly(self, tmp_path, all_test_content):
        """BM25 search should scale linearly with corpus size."""
        from tools.cohesionn.sparse_retrieval import BM25Index

        results = []

        for multiplier in [1, 2, 4]:
            # Create corpus of different sizes
            bm25 = BM25Index(topic=f"scale_{multiplier}", persist_dir=tmp_path)

            expanded = {}
            for i in range(multiplier):
                for k, v in all_test_content.items():
                    expanded[f"{k}_{i}"] = v

            bm25.add_documents(
                list(expanded.keys()),
                list(expanded.values()),
                [{} for _ in expanded],
            )

            # Benchmark search
            start = time.perf_counter()
            for _ in range(10):
                _ = bm25.search("threshold values systems", n_results=20)
            elapsed_ms = (time.perf_counter() - start) * 1000 / 10

            results.append((len(expanded), elapsed_ms))
            print(f"  {len(expanded)} docs: {elapsed_ms:.1f}ms avg")

        # Check scaling is roughly linear (not exponential)
        if len(results) >= 2:
            size_ratio = results[-1][0] / results[0][0]
            time_ratio = results[-1][1] / results[0][1]

            print(f"\nScaling: {size_ratio:.1f}x docs -> {time_ratio:.1f}x time")

            # Time should scale at most 2x worse than linear
            assert time_ratio < size_ratio * 2, "BM25 scaling worse than expected"
