"""
Phase 1: Reranker Shootout
==========================
Swap rerankers without re-embedding. Zero-cost on Qdrant.

Flow:
  1. Retrieve candidates for all queries with rerank=False (cached)
  2. For each reranker model: instantiate, rerank cached candidates, score
  3. Release GPU between swaps
"""

import time
from typing import Any, Dict, List

from ..config import ModelSpec
from .base import BasePhase, PhaseResult


class RerankerPhase(BasePhase):
    """Evaluate reranker models by re-scoring cached retrieval candidates."""

    phase_name = "reranker"

    def run(self) -> List[PhaseResult]:
        self._begin()
        results: List[PhaseResult] = []

        # Step 1: Retrieve candidates ONCE without reranking
        self._log("Retrieving candidates (rerank=False)...")
        cached = self._retrieve_all(rerank=False)
        self._log(f"Cached {len(cached)} query results")

        # Step 2: Test each reranker
        rerankers = self.config.filter_models("reranker")
        for i, model in enumerate(rerankers):
            self._log(f"[{i + 1}/{len(rerankers)}] Testing {model.name}...")
            t0 = time.time()

            try:
                result = self._test_reranker(model, cached)
                results.append(result)
                self._log(
                    f"  → composite={result.aggregate.get('avg_composite', 0):.3f}  "
                    f"keywords={result.aggregate.get('avg_keyword_hits', 0):.1%}  "
                    f"{time.time() - t0:.1f}s"
                )
            except Exception as e:
                self._log(f"  → ERROR: {e}")
                results.append(PhaseResult(
                    phase=self.phase_name,
                    variant_name=model.short_name,
                    model_spec=model.to_dict(),
                    error=str(e),
                    duration_s=time.time() - t0,
                ))
            finally:
                self._gpu_cleanup()

        return results

    def _test_reranker(
        self,
        model: ModelSpec,
        cached: Dict[str, Any],
    ) -> PhaseResult:
        """Instantiate a reranker, score all cached queries, return result."""
        from tools.cohesionn.reranker import BGEReranker, RERANKER_MAX_LENGTH

        t0 = time.time()

        # Create fresh reranker instance
        reranker = BGEReranker(
            model_name=model.hf_id,
            max_length=model.max_length or RERANKER_MAX_LENGTH,
            trust_remote_code=model.trust_remote_code,
        )

        # Rerank each query's cached candidates
        reranked: Dict[str, Any] = {}
        for q in self.queries:
            qr = cached.get(q.id, {})
            if "error" in qr or not qr.get("chunks"):
                reranked[q.id] = qr
                continue

            start = time.time()
            try:
                ranked_chunks = reranker.rerank(
                    query=q.query,
                    results=qr["chunks"],
                    top_k=self.config.top_k,
                )
                elapsed_ms = (time.time() - start) * 1000
                reranked[q.id] = {
                    "chunks": ranked_chunks,
                    "latency_ms": elapsed_ms,
                }
            except Exception as e:
                reranked[q.id] = {
                    "chunks": [],
                    "latency_ms": (time.time() - start) * 1000,
                    "error": str(e),
                }

        # Score
        aggregate, per_query = self._score_all(model.short_name, reranked)

        # Cleanup — proper GPU memory release
        reranker.unload()
        duration = time.time() - t0

        return self._make_result(
            variant_name=model.short_name,
            model_spec=model,
            aggregate=aggregate,
            per_query=per_query,
            duration_s=duration,
        )
