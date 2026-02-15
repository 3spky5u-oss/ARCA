"""
Profile Comparison Phase
========================
Benchmarks named retrieval profiles head-to-head.

Runs the same query set against each profile ("fast", "deep", etc.)
and reports composite score, keyword hit rate, and latency per profile.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ..config import ShootoutConfig
from ..metrics import ScoringEngine
from ..queries import BenchmarkQuery
from .base import BasePhase, PhaseResult

logger = logging.getLogger(__name__)


class ProfilePhase(BasePhase):
    """
    Phase 7: Compare retrieval profiles head-to-head.

    For each profile, runs all queries through the retriever with that
    profile active, then scores and reports differences.
    """

    phase_name = "profile_comparison"

    def __init__(
        self,
        config: ShootoutConfig,
        queries: List[BenchmarkQuery],
        scorer: Optional[ScoringEngine] = None,
        profiles: Optional[List[str]] = None,
    ):
        super().__init__(config, queries, scorer)
        self.profiles = profiles or getattr(config, "benchmark_profiles", ["fast", "deep"])

    def run(self) -> List[PhaseResult]:
        """Run all queries against each profile and collect metrics."""
        self._begin()
        results: List[PhaseResult] = []

        from config import runtime_config

        original_profile = getattr(runtime_config, "active_profile", None)

        for profile_name in self.profiles:
            t0 = time.time()
            self._log(
                f"Running {len(self.queries)} queries with profile '{profile_name}'"
            )

            try:
                # Activate the profile
                if hasattr(runtime_config, "apply_profile"):
                    runtime_config.apply_profile(profile_name)
                elif hasattr(runtime_config, "update"):
                    runtime_config.update(active_profile=profile_name)

                # Retrieve and score
                retrieval_results = self._retrieve_all(rerank=True)
                aggregate, per_query = self._score_all(profile_name, retrieval_results)

                duration = time.time() - t0

                # Compute keyword summary from per-query data
                keyword_scores = []
                for qm in per_query:
                    kh = getattr(qm, "keyword_hits", None)
                    if kh is not None:
                        keyword_scores.append(kh)

                avg_keyword = (
                    sum(keyword_scores) / len(keyword_scores)
                    if keyword_scores
                    else 0.0
                )

                self._log(
                    f"  composite={aggregate.avg_composite:.3f}  "
                    f"keywords={avg_keyword:.3f}  "
                    f"{duration:.1f}s"
                )

                results.append(self._make_result(
                    variant_name=profile_name,
                    model_spec=None,
                    aggregate=aggregate,
                    per_query=per_query,
                    duration_s=duration,
                    metadata={
                        "profile": profile_name,
                        "avg_keyword_score": round(avg_keyword, 4),
                        "avg_composite": round(aggregate.avg_composite, 4),
                        "avg_latency_ms": round(aggregate.avg_latency_ms, 1),
                    },
                ))

            except Exception as e:
                self._log(f"  ERROR on profile '{profile_name}': {e}")
                results.append(PhaseResult(
                    phase=self.phase_name,
                    variant_name=profile_name,
                    error=str(e),
                    duration_s=time.time() - t0,
                    metadata={"profile": profile_name},
                ))

        # Restore original profile
        try:
            if original_profile is not None:
                if hasattr(runtime_config, "apply_profile"):
                    runtime_config.apply_profile(original_profile)
                elif hasattr(runtime_config, "update"):
                    runtime_config.update(active_profile=original_profile)
        except Exception:
            pass

        # Log recommendation
        recommendation = self._generate_recommendation(results)
        if recommendation:
            self._log(f"Recommendation: {recommendation}")

        self._log(f"Profile comparison complete: {len(results)} profiles tested")
        return results

    def _generate_recommendation(self, results: List[PhaseResult]) -> str:
        """Generate a human-readable recommendation based on profile comparison."""
        valid = [r for r in results if not r.error and r.aggregate]
        if len(valid) < 2:
            return "Insufficient profiles for comparison."

        summaries = {}
        for r in valid:
            summaries[r.variant_name] = {
                "avg_composite": r.aggregate.get("avg_composite", 0),
                "avg_keyword_hits": r.aggregate.get("avg_keyword_hits", 0),
                "avg_latency_ms": r.aggregate.get("avg_latency_ms", 0),
            }

        best_quality = max(summaries.items(), key=lambda x: x[1]["avg_composite"])
        best_speed = min(summaries.items(), key=lambda x: x[1]["avg_latency_ms"])

        if best_quality[0] == best_speed[0]:
            return "'" + best_quality[0] + "' leads in both quality and speed."

        quality_name, quality_stats = best_quality
        speed_name, speed_stats = best_speed

        quality_pct = quality_stats["avg_composite"] * 100
        speed_pct = speed_stats["avg_composite"] * 100
        latency_ratio = quality_stats["avg_latency_ms"] / max(
            speed_stats["avg_latency_ms"], 1
        )

        return (
            "'" + speed_name + "' delivers " + str(round(speed_pct)) + "% composite at "
            + str(round(speed_stats["avg_latency_ms"])) + "ms avg latency. "
            + "'" + quality_name + "' reaches " + str(round(quality_pct)) + "% at "
            + str(round(latency_ratio, 1)) + "x the latency."
        )
