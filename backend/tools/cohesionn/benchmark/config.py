"""
Shootout Configuration
======================
Model catalogs, configuration dataclasses, and defaults for the benchmark harness.

Pure Python — no ARCA imports.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelSpec:
    """Specification for a candidate model."""

    name: str  # Display name
    hf_id: str  # HuggingFace model ID or GGUF filename
    short_name: str  # Slug for collection names / filenames
    model_type: str  # "embedding", "reranker", or "llm"
    dimension: Optional[int] = None  # Vector dimension (embedding only)
    notes: str = ""
    trust_remote_code: bool = False  # Models with custom HF code (nomic, stella)
    max_length: Optional[int] = None  # Max token length override (reranker)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "hf_id": self.hf_id,
            "short_name": self.short_name,
            "model_type": self.model_type,
            "notes": self.notes,
        }
        if self.dimension is not None:
            d["dimension"] = self.dimension
        if self.trust_remote_code:
            d["trust_remote_code"] = True
        if self.max_length is not None:
            d["max_length"] = self.max_length
        return d


# ── Model Catalogs ───────────────────────────────────────────────────────────

EMBEDDING_CANDIDATES: List[ModelSpec] = [
    ModelSpec(
        name="Qwen3-Embedding-0.6B",
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        short_name="qwen3-06b",
        model_type="embedding",
        dimension=1024,
        notes="Current default. MTEB top-tier for size.",
    ),
    ModelSpec(
        name="BGE-Large-EN-v1.5",
        hf_id="BAAI/bge-large-en-v1.5",
        short_name="bge-large",
        model_type="embedding",
        dimension=1024,
        notes="Proven dense retrieval model.",
    ),
    ModelSpec(
        name="Nomic-Embed-Text-v1.5",
        hf_id="nomic-ai/nomic-embed-text-v1.5",
        short_name="nomic-v15",
        model_type="embedding",
        dimension=768,
        notes="Matryoshka embeddings, good on technical text.",
        trust_remote_code=True,
    ),
    ModelSpec(
        name="Stella-EN-400M-v5",
        hf_id="dunzhang/stella_en_400M_v5",
        short_name="stella-400m",
        model_type="embedding",
        dimension=1024,
        notes="MTEB contender, instruction-tuned.",
        trust_remote_code=True,
    ),
]

RERANKER_CANDIDATES: List[ModelSpec] = [
    ModelSpec(
        name="BGE-Reranker-v2-M3",
        hf_id="BAAI/bge-reranker-v2-m3",
        short_name="bge-v2-m3",
        model_type="reranker",
        notes="Current default. Multilingual, reliable.",
    ),
    ModelSpec(
        name="BGE-Reranker-v2-Gemma",
        hf_id="BAAI/bge-reranker-v2-gemma",
        short_name="bge-v2-gemma",
        model_type="reranker",
        notes="LLM-based reranker, higher quality but slower.",
    ),
    ModelSpec(
        name="MixedBread-MxBAI-Rerank-Base-v2",
        hf_id="mixedbread-ai/mxbai-rerank-base-v2",
        short_name="mxbai-base-v2",
        model_type="reranker",
        notes="Strong zero-shot reranker.",
    ),
    ModelSpec(
        name="Jina-Reranker-v2-Turbo",
        hf_id="jinaai/jina-reranker-v2-base-multilingual",
        short_name="jina-v2-turbo",
        model_type="reranker",
        notes="Fast multilingual reranker.",
        trust_remote_code=True,
    ),
    ModelSpec(
        name="MS-MARCO-MiniLM-L12",
        hf_id="cross-encoder/ms-marco-MiniLM-L-12-v2",
        short_name="marco-minilm",
        model_type="reranker",
        notes="Classic baseline. Fast, well-tested.",
        max_length=512,
    ),
]

LLM_CANDIDATES: List[ModelSpec] = [
    ModelSpec(
        name="GLM-4.7-Flash",
        hf_id="GLM-4.7-Flash-Q4_K_M.gguf",
        short_name="glm47-flash",
        model_type="llm",
        notes="Current default. Fast, good tool routing.",
    ),
    ModelSpec(
        name="Qwen3-32B",
        hf_id="qwen3-32b-q4_k_m.gguf",
        short_name="qwen3-32b",
        model_type="llm",
        notes="Expert mode. Best quality, needs 22GB VRAM.",
    ),
    ModelSpec(
        name="Qwen3-30B-A3B",
        hf_id="Qwen3-30B-A3B-Q4_K_M.gguf",
        short_name="qwen3-30b-moe",
        model_type="llm",
        notes="MoE variant. Faster inference, lower VRAM.",
    ),
]


# ── Sweep Param Definitions ─────────────────────────────────────────────────

# ── Ingestion Sweep Definitions ────────────────────────────────────────────

DEFAULT_CHUNK_SIZES: List[int] = [600, 800, 1000, 1200, 1500]
DEFAULT_OVERLAPS: List[int] = [100, 150, 200, 300]
DEFAULT_EXTRACTORS: List[str] = ["pymupdf4llm", "marker", "hybrid"]
DEFAULT_CFEM_PATH: str = ""


# ── Profile Definitions ───────────────────────────────────────────────────

DEFAULT_BENCHMARK_PROFILES = ["fast", "deep"]


# ── Ablation Toggle Definitions ───────────────────────────────────────────

DEFAULT_ABLATION_TOGGLES: Dict[str, str] = {
    "bm25_enabled": "BM25 Hybrid Search",
    "raptor_enabled": "RAPTOR Hierarchical",
    "graph_rag_enabled": "GraphRAG Knowledge Graph",
    "domain_boost_enabled": "Domain Score Boost",
    "reranker_enabled": "Cross-Encoder Reranker",
    "query_expansion_enabled": "Query Expansion",
    "hyde_enabled": "HyDE Hypothetical Docs",
    "crag_enabled": "CRAG Corrective RAG",
}


# ── Sweep Param Definitions ─────────────────────────────────────────────────

DEFAULT_SWEEP_PARAMS: Dict[str, List[Any]] = {
    "bm25_weight": [0.0, 0.2, 0.3, 0.5, 0.7, 1.0],
    "rag_diversity_lambda": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "rag_top_k": [3, 5, 7, 10, 15],
    "reranker_candidates": [5, 8, 10, 15, 20, 30],
    "rag_min_score": [0.05, 0.10, 0.15, 0.20, 0.30],
    "domain_boost_factor": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
}


# ── Main Config ──────────────────────────────────────────────────────────────

@dataclass
class ShootoutConfig:
    """Configuration for the full benchmark shootout."""

    # Which phases to run
    phases: List[str] = field(default_factory=lambda: ["reranker"])

    # Model selections (can be overridden via CLI)
    embedding_models: List[ModelSpec] = field(default_factory=lambda: list(EMBEDDING_CANDIDATES))
    reranker_models: List[ModelSpec] = field(default_factory=lambda: list(RERANKER_CANDIDATES))
    llm_models: List[ModelSpec] = field(default_factory=lambda: list(LLM_CANDIDATES))

    # RAG retrieval params
    topic: str = "general"
    top_k: int = 5
    initial_k: int = 30

    # Param sweep
    sweep_params: Dict[str, List[Any]] = field(default_factory=lambda: dict(DEFAULT_SWEEP_PARAMS))

    # Cross-matrix
    cross_matrix_top_n: int = 3  # Top N embedders and rerankers to cross

    # Ingestion sweep
    cfem_path: str = DEFAULT_CFEM_PATH
    chunk_sizes: List[int] = field(default_factory=lambda: list(DEFAULT_CHUNK_SIZES))
    overlaps: List[int] = field(default_factory=lambda: list(DEFAULT_OVERLAPS))
    extractors: List[str] = field(default_factory=lambda: list(DEFAULT_EXTRACTORS))

    # Ablation
    ablation_toggles: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ABLATION_TOGGLES))

    # Profile comparison
    benchmark_profiles: List[str] = field(default_factory=lambda: list(DEFAULT_BENCHMARK_PROFILES))

    # Output
    output_dir: str = "/app/data/benchmarks/shootout"
    generate_charts: bool = True
    cleanup_collections: bool = True  # Delete shootout_ collections after

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phases": self.phases,
            "embedding_models": [m.to_dict() for m in self.embedding_models],
            "reranker_models": [m.to_dict() for m in self.reranker_models],
            "llm_models": [m.to_dict() for m in self.llm_models],
            "topic": self.topic,
            "top_k": self.top_k,
            "initial_k": self.initial_k,
            "sweep_params": self.sweep_params,
            "cross_matrix_top_n": self.cross_matrix_top_n,
            "cfem_path": self.cfem_path,
            "chunk_sizes": self.chunk_sizes,
            "overlaps": self.overlaps,
            "extractors": self.extractors,
            "ablation_toggles": self.ablation_toggles,
            "benchmark_profiles": self.benchmark_profiles,
            "output_dir": self.output_dir,
            "generate_charts": self.generate_charts,
            "cleanup_collections": self.cleanup_collections,
        }

    @classmethod
    def default(cls) -> "ShootoutConfig":
        """Sensible defaults for a full shootout."""
        return cls(
            phases=["reranker", "embedding", "cross_matrix", "param_sweep", "llm"],
        )

    def filter_models(
        self,
        model_type: str,
        short_names: Optional[List[str]] = None,
    ) -> List[ModelSpec]:
        """Filter model catalog by type and optional short_name list."""
        if model_type == "embedding":
            models = self.embedding_models
        elif model_type == "reranker":
            models = self.reranker_models
        elif model_type == "llm":
            models = self.llm_models
        else:
            return []

        if short_names:
            return [m for m in models if m.short_name in short_names]
        return models
