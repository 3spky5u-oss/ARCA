"""
Ingestion Phase
===============
Tests extractor × chunk_size × chunk_overlap combinations.

Three-step one-at-a-time sweep:
  Step 1: Fix extractor=pymupdf4llm, overlap=150, sweep chunk_size
  Step 2: Fix best chunk_size, sweep overlap
  Step 3: Fix best (cs, ov), sweep extractor

Each variant: clear topic → re-ingest corpus → retrieve queries → score → record.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import ShootoutConfig
from ..metrics import ScoringEngine
from ..queries import BenchmarkQuery
from .base import BasePhase, PhaseResult


@dataclass
class IngestionVariant:
    """A single ingestion configuration to test."""

    name: str
    extractor: str
    chunk_size: int
    chunk_overlap: int
    step: str  # "chunk_size", "overlap", or "extractor"


# ── Default Sweep Parameters ────────────────────────────────────────────────

DEFAULT_CHUNK_SIZES = [600, 800, 1000, 1200, 1500]
DEFAULT_OVERLAPS = [100, 150, 200, 300]
DEFAULT_EXTRACTORS = ["pymupdf4llm", "marker", "hybrid"]

DEFAULT_CORPUS_PATH = ""


class IngestionPhase(BasePhase):
    """Phase 0: Test extractor × chunk_size × chunk_overlap combinations."""

    phase_name = "ingestion"

    def __init__(
        self,
        config: ShootoutConfig,
        queries: List[BenchmarkQuery],
        scorer: Optional[ScoringEngine] = None,
        corpus_path: Optional[str] = None,
        chunk_sizes: Optional[List[int]] = None,
        overlaps: Optional[List[int]] = None,
        extractors: Optional[List[str]] = None,
    ):
        super().__init__(config, queries, scorer)
        self.corpus_path = corpus_path or DEFAULT_CORPUS_PATH
        self.chunk_sizes = chunk_sizes or DEFAULT_CHUNK_SIZES
        self.overlaps = overlaps or DEFAULT_OVERLAPS
        self.extractors = extractors or DEFAULT_EXTRACTORS

    def run(self) -> List[PhaseResult]:
        """Execute three-step ingestion sweep."""
        self._begin()
        results: List[PhaseResult] = []

        # Step 1: Chunk size sweep
        self._log("Step 1: Chunk size sweep (fixed extractor=pymupdf4llm, overlap=150)")
        best_cs = 800
        best_cs_score = 0.0

        for cs in self.chunk_sizes:
            variant = IngestionVariant(
                name=f"cs_{cs}",
                extractor="pymupdf4llm",
                chunk_size=cs,
                chunk_overlap=150,
                step="chunk_size",
            )
            result = self._test_variant(variant)
            results.append(result)

            score = result.aggregate.get("avg_composite", 0) if result.aggregate else 0
            if score > best_cs_score:
                best_cs_score = score
                best_cs = cs

        self._log(f"Step 1 winner: chunk_size={best_cs} (composite={best_cs_score:.3f})")

        # Step 2: Overlap sweep with best chunk_size
        self._log(f"Step 2: Overlap sweep (fixed cs={best_cs}, extractor=pymupdf4llm)")
        best_ov = 150
        best_ov_score = 0.0

        for ov in self.overlaps:
            variant = IngestionVariant(
                name=f"ov_{ov}",
                extractor="pymupdf4llm",
                chunk_size=best_cs,
                chunk_overlap=ov,
                step="overlap",
            )
            result = self._test_variant(variant)
            results.append(result)

            score = result.aggregate.get("avg_composite", 0) if result.aggregate else 0
            if score > best_ov_score:
                best_ov_score = score
                best_ov = ov

        self._log(f"Step 2 winner: overlap={best_ov} (composite={best_ov_score:.3f})")

        # Step 3: Extractor sweep with best cs+ov
        self._log(f"Step 3: Extractor sweep (fixed cs={best_cs}, ov={best_ov})")

        for ext in self.extractors:
            variant = IngestionVariant(
                name=f"ext_{ext}",
                extractor=ext,
                chunk_size=best_cs,
                chunk_overlap=best_ov,
                step="extractor",
            )

            # For hybrid extractor, manage VRAM
            if ext == "hybrid":
                self._prepare_vision_server()

            result = self._test_variant(variant)
            results.append(result)

            if ext == "hybrid":
                self._restore_chat_server()

        self._log(f"Ingestion shootout complete: {len(results)} variants tested")
        return results

    def _test_variant(self, variant: IngestionVariant) -> PhaseResult:
        """Clear topic, re-ingest, retrieve, score."""
        t0 = time.time()
        self._log(f"  Testing {variant.name}: ext={variant.extractor}, "
                   f"cs={variant.chunk_size}, ov={variant.chunk_overlap}")

        try:
            # Step A: Clear the topic
            chunks_before = self._get_topic_count()
            self._clear_topic()
            self._log(f"    Cleared topic ({chunks_before} chunks removed)")

            # Step B: Re-ingest with this variant's settings
            ingest_result = self._ingest_corpus(
                extractor=variant.extractor,
                chunk_size=variant.chunk_size,
                chunk_overlap=variant.chunk_overlap,
            )
            self._log(f"    Ingested: {ingest_result.get('chunks_created', 0)} chunks "
                       f"via {ingest_result.get('extractor_used', 'unknown')} "
                       f"in {ingest_result.get('ingest_time_s', 0):.1f}s")

            if not ingest_result.get("success", False):
                raise RuntimeError(f"Ingestion failed: {ingest_result.get('error', 'unknown')}")

            # Step C: Retrieve all queries
            retrieval_results = self._retrieve_all(rerank=True)

            # Step D: Score
            aggregate, per_query = self._score_all(variant.name, retrieval_results)
            duration = time.time() - t0

            self._log(f"    → composite={aggregate.avg_composite:.3f}  "
                       f"keywords={aggregate.avg_keyword_hits:.1%}  {duration:.1f}s")

            return self._make_result(
                variant_name=variant.name,
                model_spec=None,
                aggregate=aggregate,
                per_query=per_query,
                duration_s=duration,
                metadata={
                    "extractor": variant.extractor,
                    "chunk_size": variant.chunk_size,
                    "chunk_overlap": variant.chunk_overlap,
                    "step": variant.step,
                    "chunks_created": ingest_result.get("chunks_created", 0),
                    "ingest_time_s": ingest_result.get("ingest_time_s", 0),
                    "pages_by_extractor": ingest_result.get("pages_by_extractor", {}),
                },
            )

        except Exception as e:
            self._log(f"    → ERROR: {e}")
            return PhaseResult(
                phase=self.phase_name,
                variant_name=variant.name,
                error=str(e),
                duration_s=time.time() - t0,
                metadata={
                    "extractor": variant.extractor,
                    "chunk_size": variant.chunk_size,
                    "chunk_overlap": variant.chunk_overlap,
                    "step": variant.step,
                },
            )

    def _clear_topic(self):
        """Clear all chunks for the benchmark topic."""
        from tools.cohesionn.vectorstore import get_knowledge_base

        kb = get_knowledge_base()
        kb.clear_topic(self.config.topic)

    def _get_topic_count(self) -> int:
        """Get current chunk count for the benchmark topic."""
        try:
            from tools.cohesionn.vectorstore import get_knowledge_base

            kb = get_knowledge_base()
            stats = kb.get_stats()
            return stats.get(self.config.topic, 0)
        except Exception:
            return 0

    def _ingest_corpus(
        self,
        extractor: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Dict[str, Any]:
        """Ingest corpus document with specific settings, return result dict."""
        from tools.cohesionn.ingest import DocumentIngester, IngestMode

        t0 = time.time()
        ingester = DocumentIngester(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_readd=True,
            force_extractor=extractor,
            mode=IngestMode.KNOWLEDGE_BASE,
        )

        result = ingester.ingest_file(
            Path(self.corpus_path),
            topic=self.config.topic,
        )

        ingest_time = time.time() - t0

        return {
            "success": result.success,
            "chunks_created": result.chunks_created,
            "extractor_used": result.extractor_used,
            "ingest_time_s": ingest_time,
            "warnings": result.warnings,
            "error": result.error,
            "pages_by_extractor": result.pages_by_extractor,
        }

    def _prepare_vision_server(self):
        """Unload chat LLM and start vision server for hybrid extraction."""
        self._log("    Preparing vision server (unloading chat LLM)...")
        try:
            from services.llm_server_manager import get_server_manager

            mgr = get_server_manager()
            mgr.stop_slot("chat")
            self._gpu_cleanup()
            self._log("    Chat LLM unloaded, GPU freed for vision")
        except Exception as e:
            self._log(f"    Warning: Could not unload chat LLM: {e}")

    def _restore_chat_server(self):
        """Restore chat LLM after hybrid extraction."""
        self._log("    Restoring chat LLM server...")
        try:
            from services.llm_server_manager import get_server_manager

            mgr = get_server_manager()
            mgr.stop_slot("vision")
            self._gpu_cleanup()
            mgr.ensure_slot("chat")
            self._log("    Chat LLM restored")
        except Exception as e:
            self._log(f"    Warning: Could not restore chat LLM: {e}")

    def get_best_settings(self, results: List[PhaseResult]) -> Dict[str, Any]:
        """Extract the best ingestion settings from results."""
        best_score = 0.0
        best_settings: Dict[str, Any] = {
            "extractor": "pymupdf4llm",
            "chunk_size": 800,
            "chunk_overlap": 150,
        }

        for r in results:
            if r.error or not r.aggregate:
                continue
            score = r.aggregate.get("avg_composite", 0)
            if score > best_score:
                best_score = score
                best_settings = {
                    "extractor": r.metadata.get("extractor", "pymupdf4llm"),
                    "chunk_size": r.metadata.get("chunk_size", 800),
                    "chunk_overlap": r.metadata.get("chunk_overlap", 150),
                }

        return best_settings
