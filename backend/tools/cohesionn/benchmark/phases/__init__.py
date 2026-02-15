"""
Benchmark Phases
================
Each phase evaluates a different RAG pipeline component.

Phase 0: IngestionPhase   — Sweep extractor × chunk_size × overlap
Phase 1: RerankerPhase    — Swap rerankers, no re-embedding (fastest)
Phase 2: EmbeddingPhase   — Re-embed into separate Qdrant collections
Phase 3: CrossMatrixPhase — Top embedders × top rerankers
Phase 4: LLMPhase         — Swap expert LLM, test generation quality
Phase 5: ParamSweepPhase  — Sweep RAG params via runtime_config
Phase 6: AblationPhase    — Toggle components on/off, measure delta
Phase 7: ProfilePhase     — Compare retrieval profiles (fast vs deep)
"""

from .base import BasePhase, PhaseResult
from .reranker import RerankerPhase
from .embedding import EmbeddingPhase
from .cross_matrix import CrossMatrixPhase
from .llm import LLMPhase
from .param_sweep import ParamSweepPhase
from .ingestion import IngestionPhase
from .ablation import AblationPhase
from .profile import ProfilePhase

PHASE_REGISTRY = {
    "ingestion": IngestionPhase,
    "reranker": RerankerPhase,
    "embedding": EmbeddingPhase,
    "cross_matrix": CrossMatrixPhase,
    "llm": LLMPhase,
    "param_sweep": ParamSweepPhase,
    "ablation": AblationPhase,
    "profile_comparison": ProfilePhase,
}

__all__ = [
    "BasePhase",
    "PhaseResult",
    "IngestionPhase",
    "RerankerPhase",
    "EmbeddingPhase",
    "CrossMatrixPhase",
    "LLMPhase",
    "ParamSweepPhase",
    "AblationPhase",
    "ProfilePhase",
    "PHASE_REGISTRY",
]
