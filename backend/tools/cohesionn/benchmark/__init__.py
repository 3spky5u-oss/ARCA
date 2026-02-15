"""
Cohesionn Benchmark Module

Provides tools for benchmarking RAG pipeline performance:

Ingestion benchmarks:
- Page sampling from PDFs
- Stage-by-stage timing
- Quality comparison between extractors
- Extrapolation to full-book ingestion times

Model shootout benchmarks:
- Embedding model comparison (separate Qdrant collections per model)
- Reranker comparison (zero-cost: cached retrieval + re-score)
- Cross-matrix: top embedders × top rerankers
- LLM generation quality comparison
- RAG parameter sweep via runtime_config

Usage:
    # Ingestion benchmark
    from tools.cohesionn.benchmark import PageSampler, BenchmarkReport

    # Model shootout
    from tools.cohesionn.benchmark.config import ShootoutConfig
    from tools.cohesionn.benchmark.queries import BENCHMARK_QUERIES
    from tools.cohesionn.benchmark.metrics import ScoringEngine
    from tools.cohesionn.benchmark.phases import PHASE_REGISTRY
"""

from .page_sampler import PageSampler, SampledPage
from .report_generator import BenchmarkReport, generate_console_report

__all__ = [
    "PageSampler",
    "SampledPage",
    "BenchmarkReport",
    "generate_console_report",
]
