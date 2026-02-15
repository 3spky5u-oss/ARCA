"""
Benchmark v2 Configuration
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

@dataclass
class ChunkingConfig:
    """One chunking configuration to test."""
    config_id: str
    chunk_size: int
    chunk_overlap: int
    context_prefix: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "context_prefix": self.context_prefix,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChunkingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def generate_chunking_matrix() -> List[ChunkingConfig]:
    """Generate ~90 valid chunking configurations.

    Prunes configs where overlap >= 50% of chunk_size.
    """
    sizes = [400, 600, 800, 1000, 1200, 1500, 1800, 2000, 2500]
    overlaps = [0, 50, 100, 150, 200, 300, 400]
    prefix_options = [True, False]

    configs = []
    for size in sizes:
        for overlap in overlaps:
            if overlap >= size * 0.5:
                continue  # Invalid: overlap too large
            for prefix in prefix_options:
                config_id = f"c{size}_o{overlap}_p{1 if prefix else 0}"
                configs.append(ChunkingConfig(
                    config_id=config_id,
                    chunk_size=size,
                    chunk_overlap=overlap,
                    context_prefix=prefix,
                ))
    return configs


# Named retrieval configurations for Layer 1
RETRIEVAL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "dense_only": {
        "rerank": False, "use_hybrid": False, "use_expansion": False,
        "use_hyde": False, "use_raptor": False, "use_graph": False,
        "use_global": False, "apply_diversity": False,
    },
    "dense_rerank": {
        "rerank": True, "use_hybrid": False, "use_expansion": False,
        "use_hyde": False, "use_raptor": False, "use_graph": False,
        "use_global": False, "apply_diversity": False,
    },
    "hybrid_basic": {
        "rerank": True, "use_hybrid": True, "use_expansion": False,
        "use_hyde": False, "use_raptor": False, "use_graph": False,
        "use_global": False, "apply_diversity": False,
    },
    "fast_profile": {
        "rerank": True, "use_hybrid": True, "use_expansion": True,
        "use_hyde": False, "use_raptor": False, "use_graph": False,
        "use_global": False, "apply_diversity": True,
    },
    "deep_profile": {
        "rerank": True, "use_hybrid": True, "use_expansion": True,
        "use_hyde": True, "use_raptor": True, "use_graph": True,
        "use_global": True, "apply_diversity": True,
    },
    "deep_no_hyde": {
        "rerank": True, "use_hybrid": True, "use_expansion": True,
        "use_hyde": False, "use_raptor": True, "use_graph": True,
        "use_global": True, "apply_diversity": True,
    },
    "deep_no_bm25": {
        "rerank": True, "use_hybrid": False, "use_expansion": True,
        "use_hyde": True, "use_raptor": True, "use_graph": True,
        "use_global": True, "apply_diversity": True,
    },
    "deep_no_raptor": {
        "rerank": True, "use_hybrid": True, "use_expansion": True,
        "use_hyde": True, "use_raptor": False, "use_graph": True,
        "use_global": True, "apply_diversity": True,
    },
    "deep_no_graph": {
        "rerank": True, "use_hybrid": True, "use_expansion": True,
        "use_hyde": True, "use_raptor": True, "use_graph": False,
        "use_global": True, "apply_diversity": True,
    },
    "deep_no_expansion": {
        "rerank": True, "use_hybrid": True, "use_expansion": False,
        "use_hyde": True, "use_raptor": True, "use_graph": True,
        "use_global": True, "apply_diversity": True,
    },
    "deep_no_diversity": {
        "rerank": True, "use_hybrid": True, "use_expansion": True,
        "use_hyde": True, "use_raptor": True, "use_graph": True,
        "use_global": True, "apply_diversity": False,
    },
    "deep_no_global": {
        "rerank": True, "use_hybrid": True, "use_expansion": True,
        "use_hyde": True, "use_raptor": True, "use_graph": True,
        "use_global": False, "apply_diversity": True,
    },
    "deep_no_domain_boost": {
        "rerank": True, "use_hybrid": True, "use_expansion": True,
        "use_hyde": True, "use_raptor": True, "use_graph": True,
        "use_global": True, "apply_diversity": True,
        # domain_boost handled via runtime_config
    },
    "hybrid_no_rerank": {
        "rerank": False, "use_hybrid": True, "use_expansion": True,
        "use_hyde": False, "use_raptor": False, "use_graph": False,
        "use_global": False, "apply_diversity": False,
    },
    "expansion_only": {
        "rerank": False, "use_hybrid": False, "use_expansion": True,
        "use_hyde": False, "use_raptor": False, "use_graph": False,
        "use_global": False, "apply_diversity": False,
    },
}

# Continuous parameter sweep ranges for Layer 2 (OAT)
PARAM_SWEEP_RANGES: Dict[str, List[Any]] = {
    "bm25_weight": [0.0, 0.2, 0.3, 0.5, 0.7],
    "rag_diversity_lambda": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "rag_top_k": [3, 5, 7, 10],
    "reranker_candidates": [5, 8, 10, 15, 20],
    "rag_min_score": [0.05, 0.10, 0.15, 0.20],
    "domain_boost_factor": [0.0, 0.5, 1.0, 1.5, 2.0],
}


@dataclass
class BenchmarkConfig:
    """Master configuration for a benchmark run."""

    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    corpus_dir: str = "/app/data/Synthetic Reports"
    output_base: str = "/app/data/benchmarks/v2"
    topic: str = "benchmark"
    top_k: int = 5

    # Layer 0
    max_configs: int = 0  # 0 = all valid configs
    chunking_configs: List[ChunkingConfig] = field(default_factory=list)

    # Layer 1
    retrieval_configs: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: dict(RETRIEVAL_CONFIGS)
    )

    # Layer 2
    param_sweep_ranges: Dict[str, List[Any]] = field(
        default_factory=lambda: dict(PARAM_SWEEP_RANGES)
    )

    # Layer 4
    gemini_model: str = field(
        default_factory=lambda: os.environ.get("GEMINI_BENCHMARK_MODEL", "gemini-3-flash-preview")
    )
    gemini_rate_limit: float = 5.0  # seconds between calls (free tier: 15 RPM)

    # Provider configuration (populated from benchmark_providers.json)
    judge_provider: str = "local"       # "local"|"gemini"|"anthropic"|"openai"
    judge_model: str = ""               # empty = use provider default
    ceiling_provider: str = "local"
    ceiling_model: str = ""

    # Control
    cleanup_after_score: bool = True  # Delete Qdrant points after scoring each config

    @property
    def output_dir(self) -> str:
        return f"{self.output_base}/{self.run_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "corpus_dir": self.corpus_dir,
            "output_base": self.output_base,
            "topic": self.topic,
            "top_k": self.top_k,
            "max_configs": self.max_configs,
            "gemini_model": self.gemini_model,
            "gemini_rate_limit": self.gemini_rate_limit,
            "cleanup_after_score": self.cleanup_after_score,
            "n_chunking_configs": len(self.chunking_configs),
            "n_retrieval_configs": len(self.retrieval_configs),
        }
