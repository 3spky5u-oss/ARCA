"""
Admin Benchmark Router - RAG Pipeline Benchmark API (v2)

Wraps the benchmark harness v2 (backend/benchmark/) in HTTP endpoints
for the admin panel. Runs benchmarks in background threads, reports progress,
and allows applying winners to RuntimeConfig.
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

router = APIRouter(prefix="/api/admin/benchmark", tags=["admin-benchmark"])

# ── In-memory job state ─────────────────────────────────────────────────────

_current_job: Optional[Dict[str, Any]] = None
_job_lock = threading.Lock()

# LLM chat server port (same as utils/llm.py uses for "chat" slot)
_LLM_CHAT_PORT = 8081

BENCHMARKS_DIR = Path("/app/data/benchmarks/v2")

ALL_LAYERS = [
    "layer0_chunking", "layer1_retrieval", "layer2_params",
    "layer_embed", "layer_rerank", "layer_cross", "layer_llm",
    "layer3_answers", "layer4_judge", "layer5_analysis", "layer6_failures",
    "layer_live", "layer_ceiling",
]
QUICK_LAYERS = ["layer0_chunking", "layer1_retrieval"]

# Map layer names to their module path and class name
LAYER_REGISTRY = {
    "layer0_chunking": ("benchmark.layers.layer0_chunking", "ChunkingSweepLayer"),
    "layer1_retrieval": ("benchmark.layers.layer1_retrieval", "RetrievalSweepLayer"),
    "layer2_params": ("benchmark.layers.layer2_params", "ParamSweepLayer"),
    "layer_embed": ("benchmark.layers.layer_embed", "EmbeddingShootoutLayer"),
    "layer_rerank": ("benchmark.layers.layer_rerank", "RerankerShootoutLayer"),
    "layer_cross": ("benchmark.layers.layer_cross", "CrossModelSweepLayer"),
    "layer_llm": ("benchmark.layers.layer_llm", "LLMComparisonLayer"),
    "layer3_answers": ("benchmark.layers.layer3_answers", "AnswerGenerationLayer"),
    "layer4_judge": ("benchmark.layers.layer4_judge", "JudgeLayer"),
    "layer5_analysis": ("benchmark.layers.layer5_analysis", "AnalysisLayer"),
    "layer6_failures": ("benchmark.layers.layer6_failures", "FailureLayer"),
    "layer_live": ("benchmark.layers.layer_live", "LivePipelineLayer"),
    "layer_ceiling": ("benchmark.layers.layer_ceiling", "CeilingComparisonLayer"),
}

# Import sub-modules to register routes on `router`
from . import endpoints  # noqa: E402, F401
