"""
Ablation Phase
==============
Toggles RAG pipeline components on/off and measures the delta.

Tests each toggle in both states (enabled vs disabled) against all 30 queries.
Reports the composite score delta for each component.
"""

import time
from typing import Dict, List, Optional

from ..config import ShootoutConfig
from ..metrics import ScoringEngine
from ..queries import BenchmarkQuery
from .base import BasePhase, PhaseResult


# Toggle keys mapped to runtime_config attribute names and display labels
ABLATION_TOGGLES: Dict[str, str] = {
    "bm25_enabled": "BM25 Hybrid Search",
    "raptor_enabled": "RAPTOR Hierarchical",
    "graph_rag_enabled": "GraphRAG Knowledge Graph",
    "domain_boost_enabled": "Domain Score Boost",
    "reranker_enabled": "Cross-Encoder Reranker",
    "query_expansion_enabled": "Query Expansion",
    "hyde_enabled": "HyDE Hypothetical Docs",
    "crag_enabled": "CRAG Corrective RAG",
}


class AblationPhase(BasePhase):
    """Phase 6: Toggle pipeline components on/off, measure delta."""

    phase_name = "ablation"

    def __init__(
        self,
        config: ShootoutConfig,
        queries: List[BenchmarkQuery],
        scorer: Optional[ScoringEngine] = None,
        toggles: Optional[Dict[str, str]] = None,
    ):
        super().__init__(config, queries, scorer)
        self.toggles = toggles or ABLATION_TOGGLES

    def run(self) -> List[PhaseResult]:
        """Toggle each component on/off and measure delta."""
        self._begin()
        results: List[PhaseResult] = []

        from config import runtime_config

        for toggle_key, label in self.toggles.items():
            t0 = time.time()
            original_value = None
            self._log(f"Testing toggle: {label} ({toggle_key})")

            try:
                original_value = getattr(runtime_config, toggle_key, None)
                if original_value is None:
                    self._log(f"  Skipping {toggle_key}: not found in runtime_config")
                    continue

                # Test with toggle ON
                runtime_config.update(**{toggle_key: True})
                on_results = self._retrieve_all(rerank=True)
                on_aggregate, on_per_query = self._score_all(f"{toggle_key}_on", on_results)

                # Test with toggle OFF
                runtime_config.update(**{toggle_key: False})
                off_results = self._retrieve_all(rerank=True)
                off_aggregate, off_per_query = self._score_all(f"{toggle_key}_off", off_results)

                # Restore original
                runtime_config.update(**{toggle_key: original_value})

                # Compute delta
                delta = on_aggregate.avg_composite - off_aggregate.avg_composite
                duration = time.time() - t0

                self._log(
                    f"  ON={on_aggregate.avg_composite:.3f}  "
                    f"OFF={off_aggregate.avg_composite:.3f}  "
                    f"Δ={delta:+.3f}  {duration:.1f}s"
                )

                # Per-tier deltas
                tier_deltas = {}
                for tier in on_aggregate.by_tier:
                    on_tier = on_aggregate.by_tier.get(tier, {}).get("avg_composite", 0)
                    off_tier = off_aggregate.by_tier.get(tier, {}).get("avg_composite", 0)
                    tier_deltas[tier] = round(on_tier - off_tier, 4)

                results.append(self._make_result(
                    variant_name=toggle_key,
                    model_spec=None,
                    aggregate=on_aggregate,
                    per_query=on_per_query,
                    duration_s=duration,
                    metadata={
                        "label": label,
                        "toggle_key": toggle_key,
                        "on_composite": round(on_aggregate.avg_composite, 4),
                        "off_composite": round(off_aggregate.avg_composite, 4),
                        "delta": round(delta, 4),
                        "tier_deltas": tier_deltas,
                        "off_aggregate": off_aggregate.to_dict(),
                        "off_per_query": [q.to_dict() for q in off_per_query],
                    },
                ))

            except Exception as e:
                self._log(f"  ERROR: {e}")
                # Restore original value on error
                try:
                    if original_value is not None:
                        runtime_config.update(**{toggle_key: original_value})
                except Exception:
                    pass

                results.append(PhaseResult(
                    phase=self.phase_name,
                    variant_name=toggle_key,
                    error=str(e),
                    duration_s=time.time() - t0,
                    metadata={"label": label, "toggle_key": toggle_key},
                ))

        self._log(f"Ablation complete: {len(results)} toggles tested")
        return results
