"""
Phase 3: Cross-Matrix
=====================
Evaluate top N embedders × top M rerankers from Phase 1 + Phase 2 results.

Re-uses shootout collections from Phase 2 (cleanup deferred).
Output includes heatmap_data for visualization.
"""

import time
from typing import Any, Dict, List, Optional

from ..config import ShootoutConfig
from ..metrics import ScoringEngine
from ..queries import BenchmarkQuery
from .base import BasePhase, PhaseResult


class CrossMatrixPhase(BasePhase):
    """Test top embedder × reranker combinations."""

    phase_name = "cross_matrix"

    def __init__(
        self,
        config: ShootoutConfig,
        queries: List[BenchmarkQuery],
        scorer: Optional[ScoringEngine] = None,
        embedding_results: Optional[List[PhaseResult]] = None,
        reranker_results: Optional[List[PhaseResult]] = None,
    ):
        super().__init__(config, queries, scorer)
        self.embedding_results = embedding_results or []
        self.reranker_results = reranker_results or []

    def run(self) -> List[PhaseResult]:
        self._begin()
        results: List[PhaseResult] = []

        # Select top N embedders and rerankers by composite score
        top_embedders = self._select_top(
            self.embedding_results, self.config.cross_matrix_top_n
        )
        top_rerankers = self._select_top(
            self.reranker_results, self.config.cross_matrix_top_n
        )

        if not top_embedders or not top_rerankers:
            self._log("Need Phase 1 + Phase 2 results. Skipping cross-matrix.")
            return results

        self._log(
            f"Testing {len(top_embedders)} embedders × {len(top_rerankers)} rerankers "
            f"= {len(top_embedders) * len(top_rerankers)} combinations"
        )

        heatmap_data: Dict[str, Dict[str, float]] = {}

        # Group by embedder: load each embedder ONCE, iterate rerankers
        for emb in top_embedders:
            emb_name = emb.variant_name
            emb_spec = emb.model_spec or {}
            collection = emb.metadata.get("collection", f"shootout_{emb_name}")
            heatmap_data[emb_name] = {}

            # Load this embedder once for all its reranker combos
            from tools.cohesionn.embeddings import UniversalEmbedder
            embedder = UniversalEmbedder(
                model_name=emb_spec.get("hf_id", ""),
                trust_remote_code=emb_spec.get("trust_remote_code", False),
            )
            self._log(f"Loaded embedder: {emb_name}")
            self._log_vram("embedder loaded")

            for rer in top_rerankers:
                rer_name = rer.variant_name
                rer_spec = rer.model_spec or {}
                combo_name = f"{emb_name}+{rer_name}"
                self._log(f"Testing {combo_name}...")
                t0 = time.time()

                # Load reranker for this combo
                from tools.cohesionn.reranker import BGEReranker, RERANKER_MAX_LENGTH
                reranker = BGEReranker(
                    model_name=rer_spec.get("hf_id", ""),
                    max_length=rer_spec.get("max_length") or RERANKER_MAX_LENGTH,
                    trust_remote_code=rer_spec.get("trust_remote_code", False),
                )

                try:
                    result = self._test_combination(
                        combo_name, collection, emb, rer,
                        shared_embedder=embedder,
                        shared_reranker=reranker,
                    )
                    results.append(result)
                    score = result.aggregate.get("avg_composite", 0) if result.aggregate else 0
                    heatmap_data[emb_name][rer_name] = score
                    self._log(f"  → composite={score:.3f}")
                except Exception as e:
                    self._log(f"  → ERROR: {e}")
                    results.append(PhaseResult(
                        phase=self.phase_name,
                        variant_name=combo_name,
                        error=str(e),
                        duration_s=time.time() - t0,
                    ))
                    heatmap_data[emb_name][rer_name] = 0.0
                finally:
                    reranker.unload()
                    self._gpu_cleanup()

            # Done with this embedder, unload before loading next
            embedder.unload()
            self._gpu_cleanup()
            self._log_vram("embedder unloaded")

        # Attach heatmap data to last result for visualization
        if results:
            results[-1].metadata["heatmap_data"] = heatmap_data

        return results

    def _test_combination(
        self,
        combo_name: str,
        collection_name: str,
        emb_result: PhaseResult,
        rer_result: PhaseResult,
        shared_embedder=None,
        shared_reranker=None,
    ) -> PhaseResult:
        """Test a specific embedder+reranker combination.

        Accepts pre-loaded models to avoid redundant load/unload cycles.
        """
        from qdrant_client import QdrantClient
        import os

        t0 = time.time()

        emb_spec = emb_result.model_spec or {}
        rer_spec = rer_result.model_spec or {}

        embedder = shared_embedder
        reranker = shared_reranker

        # Fallback: load if not provided (shouldn't happen with fixed run())
        owns_embedder = False
        owns_reranker = False
        if embedder is None:
            from tools.cohesionn.embeddings import UniversalEmbedder
            embedder = UniversalEmbedder(
                model_name=emb_spec.get("hf_id", ""),
                trust_remote_code=emb_spec.get("trust_remote_code", False),
            )
            owns_embedder = True
        if reranker is None:
            from tools.cohesionn.reranker import BGEReranker, RERANKER_MAX_LENGTH
            reranker = BGEReranker(
                model_name=rer_spec.get("hf_id", ""),
                max_length=rer_spec.get("max_length") or RERANKER_MAX_LENGTH,
                trust_remote_code=rer_spec.get("trust_remote_code", False),
            )
            owns_reranker = True

        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url)

        retrieval_results: Dict[str, Any] = {}

        for q in self.queries:
            start = time.time()
            try:
                query_vector = embedder.embed_query(q.query)
                search_results = client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=self.config.initial_k,
                    with_payload=True,
                ).points

                search_chunks = []
                for sr in search_results:
                    payload = sr.payload or {}
                    search_chunks.append({
                        "content": payload.get("content", ""),
                        "score": sr.score,
                        "source": payload.get("source", "unknown"),
                        "page": payload.get("page"),
                        "topic": payload.get("topic", ""),
                        "is_raptor": payload.get("raptor_level", 0) > 0,
                        "is_graph_result": payload.get("is_graph_result", False),
                    })

                ranked = reranker.rerank(
                    query=q.query,
                    results=search_chunks,
                    top_k=self.config.top_k,
                )
                elapsed_ms = (time.time() - start) * 1000
                retrieval_results[q.id] = {
                    "chunks": ranked,
                    "latency_ms": elapsed_ms,
                }
            except Exception as e:
                retrieval_results[q.id] = {
                    "chunks": [],
                    "latency_ms": (time.time() - start) * 1000,
                    "error": str(e),
                }

        aggregate, per_query = self._score_all(combo_name, retrieval_results)

        # Only unload models we own
        if owns_embedder:
            embedder.unload()
        if owns_reranker:
            reranker.unload()

        duration = time.time() - t0

        return self._make_result(
            variant_name=combo_name,
            model_spec=None,
            aggregate=aggregate,
            per_query=per_query,
            duration_s=duration,
            metadata={
                "embedder": emb_spec.get("short_name", ""),
                "reranker": rer_spec.get("short_name", ""),
                "collection": collection_name,
            },
        )

    @staticmethod
    def _select_top(
        results: List[PhaseResult], n: int
    ) -> List[PhaseResult]:
        """Select top N results by composite score."""
        valid = [
            r for r in results
            if r.aggregate and r.error is None
        ]
        valid.sort(
            key=lambda r: r.aggregate.get("avg_composite", 0),
            reverse=True,
        )
        return valid[:n]
