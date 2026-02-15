"""
Base Phase
==========
Abstract base class for all benchmark phases.
"""

import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import ModelSpec, ShootoutConfig
from ..metrics import AggregateMetrics, QueryMetrics, ScoringEngine
from ..queries import BenchmarkQuery


@dataclass
class PhaseResult:
    """Result from a single variant within a phase."""

    phase: str
    variant_name: str
    model_spec: Optional[Dict[str, Any]] = None
    aggregate: Optional[Dict[str, Any]] = None
    per_query: List[Dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "phase": self.phase,
            "variant_name": self.variant_name,
            "duration_s": round(self.duration_s, 1),
        }
        if self.model_spec:
            d["model_spec"] = self.model_spec
        if self.aggregate:
            d["aggregate"] = self.aggregate
        if self.per_query:
            d["per_query"] = self.per_query
        if self.error:
            d["error"] = self.error
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class BasePhase(ABC):
    """Abstract base class for benchmark phases."""

    phase_name: str = "base"

    def __init__(
        self,
        config: ShootoutConfig,
        queries: List[BenchmarkQuery],
        scorer: Optional[ScoringEngine] = None,
    ):
        self.config = config
        self.queries = queries
        self.scorer = scorer or ScoringEngine()
        self._start_time = 0.0

    @abstractmethod
    def run(self) -> List[PhaseResult]:
        """Execute the phase and return results for each variant."""
        ...

    def _gpu_cleanup(self):
        """Release GPU memory between model swaps."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def _get_vram_info(self) -> Dict[str, Any]:
        """Get current VRAM usage. Returns empty dict if no GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                used = total - free
                return {
                    "total_mb": total / 1024 / 1024,
                    "used_mb": used / 1024 / 1024,
                    "free_mb": free / 1024 / 1024,
                    "utilization_pct": used / total * 100,
                }
        except (ImportError, RuntimeError):
            pass
        return {}

    def _log_vram(self, label: str = ""):
        """Log current VRAM state."""
        info = self._get_vram_info()
        if info:
            tag = f" ({label})" if label else ""
            self._log(
                f"VRAM{tag}: {info['used_mb']:.0f}/{info['total_mb']:.0f} MB "
                f"({info['utilization_pct']:.0f}% used, {info['free_mb']:.0f} MB free)"
            )

    def _check_vram_budget(self, needed_mb: float) -> bool:
        """Check if enough VRAM is available. Returns True if safe or no GPU."""
        info = self._get_vram_info()
        if not info:
            return True
        if info["free_mb"] < needed_mb:
            self._log(
                f"VRAM BUDGET EXCEEDED: need {needed_mb:.0f} MB, "
                f"only {info['free_mb']:.0f} MB free. Aborting to prevent OOM."
            )
            return False
        return True

    def _snapshot_config(self) -> Dict[str, Any]:
        """Capture current runtime_config state for reproducibility."""
        try:
            from config import runtime_config
            return runtime_config.to_dict()
        except ImportError:
            return {}

    def _log(self, msg: str):
        """Print phase log message."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        print(f"  [{self.phase_name}] {elapsed:6.1f}s  {msg}")

    def _begin(self):
        """Mark phase start."""
        self._start_time = time.time()
        self._log(f"Starting with {len(self.queries)} queries")

    def _retrieve_all(
        self,
        rerank: bool = True,
        retriever=None,
    ) -> Dict[str, Any]:
        """
        Retrieve candidates for all queries.

        Returns:
            {query_id: {"chunks": [...], "latency_ms": float, "result": RetrievalResult}}
        """
        from tools.cohesionn.retriever import CohesionnRetriever

        if retriever is None:
            retriever = CohesionnRetriever()

        results = {}
        for q in self.queries:
            start = time.time()
            try:
                r = retriever.retrieve(
                    query=q.query,
                    topics=[self.config.topic],
                    top_k=self.config.top_k,
                    initial_k=self.config.initial_k,
                    rerank=rerank,
                )
                elapsed_ms = (time.time() - start) * 1000
                results[q.id] = {
                    "chunks": r.chunks if hasattr(r, "chunks") else [],
                    "latency_ms": elapsed_ms,
                    "result": r,
                }
            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000
                results[q.id] = {
                    "chunks": [],
                    "latency_ms": elapsed_ms,
                    "error": str(e),
                }
        return results

    def _score_all(
        self,
        variant_name: str,
        retrieval_results: Dict[str, Any],
    ) -> tuple:
        """
        Score all queries and return (AggregateMetrics, List[QueryMetrics]).
        """
        per_query: List[QueryMetrics] = []
        for q in self.queries:
            r = retrieval_results.get(q.id, {})
            if "error" in r:
                per_query.append(QueryMetrics(
                    query_id=q.id,
                    tier=q.tier,
                    latency_ms=r.get("latency_ms", 0),
                    error=r["error"],
                ))
                continue

            metrics = self.scorer.score_retrieval(
                query_id=q.id,
                tier=q.tier,
                chunks=r.get("chunks", []),
                latency_ms=r.get("latency_ms", 0),
                expect_keywords=q.expect_keywords,
                expect_entities=q.expect_entities,
            )
            per_query.append(metrics)

        aggregate = self.scorer.aggregate(variant_name, per_query)
        return aggregate, per_query

    def _make_result(
        self,
        variant_name: str,
        model_spec: Optional[ModelSpec],
        aggregate: AggregateMetrics,
        per_query: List[QueryMetrics],
        duration_s: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PhaseResult:
        """Build a PhaseResult from computed metrics."""
        return PhaseResult(
            phase=self.phase_name,
            variant_name=variant_name,
            model_spec=model_spec.to_dict() if model_spec else None,
            aggregate=aggregate.to_dict(),
            per_query=[q.to_dict() for q in per_query],
            duration_s=duration_s,
            metadata=metadata or {},
        )
