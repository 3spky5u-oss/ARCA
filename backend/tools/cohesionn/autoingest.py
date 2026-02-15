"""
Auto-ingestion service for knowledge base.

Automatically detects and ingests new files on startup,
with manifest-based tracking to avoid re-ingesting unchanged files.

Supports two ingestion modes:
- run(): Legacy per-file extract->chunk->embed (session uploads)
- run_phased(): Three-phase batch ingest (text->vision->embed) to avoid
  CUDA conflicts between vision llama-server and ONNX embedder

Phase logic lives in autoingest_phases.py (IngestPhasesMixin).
Types and constants live in autoingest_types.py.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

from .manifest import IngestManifest
from .autoingest_types import PhasedFileState, SUPPORTED_EXTENSIONS, _RAM_PRESSURE_THRESHOLD
from .autoingest_phases import IngestPhasesMixin

logger = logging.getLogger(__name__)

# Re-export for backward compatibility — external code imports these from autoingest
__all__ = [
    "AutoIngestService",
    "run_auto_ingestion",
    "PhasedFileState",
    "SUPPORTED_EXTENSIONS",
]


class AutoIngestService(IngestPhasesMixin):
    """
    Service for automatic knowledge base ingestion on startup.

    Features:
    - Dynamic topic discovery from filesystem
    - Manifest-based file tracking (by content hash)
    - Detection of new, changed, and deleted files
    - Background-safe execution
    - Mode support for hybrid vision/text extraction
    """

    def __init__(self, knowledge_dir: Path, db_dir: Path, mode: str = "session",
                 skip_vision: bool = None):
        """
        Args:
            knowledge_dir: Directory containing topic subdirectories with PDFs
            db_dir: Vector store persistence directory
            mode: Ingestion mode ("knowledge_base" for hybrid vision, "session" for fast text)
            skip_vision: Skip Phase 2 vision extraction (text-only, faster)
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.db_dir = Path(db_dir)
        self.manifest = IngestManifest(db_dir / "manifest.json")
        self.mode = mode
        if skip_vision is None:
            from config import runtime_config
            self._skip_vision = not runtime_config.vision_ingest_enabled
        else:
            self._skip_vision = skip_vision
        self._ingester = None
        self._knowledge_base = None

    @property
    def knowledge_base(self):
        """Lazy-load knowledge base"""
        if self._knowledge_base is None:
            from .vectorstore import KnowledgeBase

            self._knowledge_base = KnowledgeBase(persist_dir=self.db_dir, knowledge_dir=self.knowledge_dir)
        return self._knowledge_base

    @property
    def ingester(self):
        """Lazy-load document ingester"""
        if self._ingester is None:
            from .ingest import DocumentIngester

            self._ingester = DocumentIngester(
                knowledge_base=self.knowledge_base,
                use_readd=True,  # Enable Readd pipeline for quality extraction
                mode=self.mode,  # Pass through mode for hybrid/session extraction
            )
        return self._ingester

    def discover_topics(self) -> List[str]:
        """
        Discover topics from filesystem.

        Any subdirectory under knowledge_dir becomes a topic.
        """
        if not self.knowledge_dir.exists():
            logger.warning(f"Knowledge directory not found: {self.knowledge_dir}")
            return []
        return [d.name for d in self.knowledge_dir.iterdir() if d.is_dir()]

    def scan_new_files(self) -> Dict[str, List[Path]]:
        """
        Scan for files that need ingestion.

        Returns:
            Dict mapping topic -> list of file paths to ingest
        """
        new_files = {}
        for topic in self.discover_topics():
            topic_dir = self.knowledge_dir / topic
            topic_new = []
            for ext in SUPPORTED_EXTENSIONS:
                for f in topic_dir.glob(f"*{ext}"):
                    if self.manifest.needs_reingestion(f):
                        topic_new.append(f)
            if topic_new:
                new_files[topic] = topic_new
        return new_files

    def cleanup_stale_entries(self) -> int:
        """
        Remove manifest entries for deleted files.

        Returns:
            Number of entries removed
        """
        stale = self.manifest.get_stale_hashes(self.knowledge_dir)
        if stale:
            self.manifest.remove_stale(stale)
            self.manifest.save()
            logger.info(f"Removed {len(stale)} stale manifest entries")
        return len(stale)

    def run_phased(self) -> Dict[str, Any]:
        """
        Three-phase batch ingest: text extraction -> vision extraction -> embedding.

        Separates GPU consumers so only one runs at a time:
        - Phase 1: Text extraction (CPU only — PyMuPDF4LLM + page classification)
        - Phase 2: Vision extraction (GPU — llama.cpp vision server only)
        - Phase 3: Chunk + Embed (GPU — ONNX embedder only)

        Returns:
            Dict with ingestion results including per-phase stats
        """
        logger.info("=== Three-Phase Batch Ingest ===")

        # 1. Cleanup stale entries
        stale_count = self.cleanup_stale_entries()

        # 2. Scan for new files
        new_files = self.scan_new_files()
        if not new_files:
            logger.info("Knowledge base up to date - no new files to ingest")
            return {
                "new_files": 0,
                "stale_removed": stale_count,
                "topics_discovered": self.discover_topics(),
                "results": {},
                "phases": {"text": 0, "vision": 0, "embed": 0},
            }

        total = sum(len(files) for files in new_files.values())
        logger.info(f"Found {total} new/changed files across {len(new_files)} topics")

        # Create temp dir for RAM spillover
        temp_dir = Path(tempfile.mkdtemp(prefix="arca_ingest_"))

        file_states = []
        try:
            # ===== PHASE 1: Text Extraction (CPU only) =====
            file_states = self._phase1_extract_all(new_files, temp_dir)
            logger.info(f"Phase 1 complete: {len(file_states)} files extracted")

            # ===== PHASE 2: Vision Extraction (GPU: vision model) =====
            # Vision is optional — text from Phase 1 is sufficient for RAG.
            # Vision adds richer chart/figure descriptions but is slow (~10s/page).
            if self._skip_vision:
                for s in file_states:
                    s.vision_page_specs = []
                vision_count = 0
                logger.info("Phase 2: Skipped (skip_vision=True)")
            else:
                vision_count = self._phase2_extract_vision(file_states)
                logger.info(f"Phase 2 complete: {vision_count} vision pages processed")

            # ===== PHASE 3: Chunk + Embed (GPU: ONNX embedder) =====
            results, successful_total, failed_total, total_chunks = self._phase3_chunk_and_embed(file_states)
            logger.info(
                f"Phase 3 complete: {total_chunks} chunks embedded, "
                f"{successful_total} files succeeded, {failed_total} failed"
            )

            # Save manifest
            self.manifest.save()

            # Reload chat model — RAPTOR and GraphRAG need the LLM
            try:
                from services.llm_server_manager import get_server_manager
                mgr = get_server_manager()
                mgr.start("chat")
                # Wait for chat model to become healthy (sync poll)
                import httpx, time as _time
                from services.llm_config import SLOTS
                chat_port = SLOTS["chat"].port
                health_url = f"http://localhost:{chat_port}/health"
                deadline = _time.time() + 300  # 5 min max
                healthy = False
                while _time.time() < deadline:
                    try:
                        r = httpx.get(health_url, timeout=5.0)
                        if r.status_code == 200:
                            healthy = True
                            break
                    except Exception:
                        pass
                    _time.sleep(2.0)
                if healthy:
                    logger.info("Chat model reloaded and healthy for Phases 4-5")
                else:
                    logger.error("Chat model failed to become healthy within 300s — skipping Phases 4-5")
            except Exception as e:
                logger.warning(f"Could not reload chat model: {e}")
                healthy = False

            if not healthy:
                logger.warning("Skipping RAPTOR and GraphRAG — chat model unavailable")
                raptor_nodes = 0
                graph_entities = 0
                graph_relationships = 0
            else:
                # ===== PHASE 4: RAPTOR Tree Building =====
                raptor_nodes = self._phase4_build_raptor(list(new_files.keys()))
                logger.info(f"Phase 4 complete: {raptor_nodes} RAPTOR nodes built")

                # ===== PHASE 5: GraphRAG Entity Extraction =====
                graph_entities, graph_relationships = self._phase5_build_graph(list(new_files.keys()))
                logger.info(
                    f"Phase 5 complete: {graph_entities} entities, "
                    f"{graph_relationships} relationships"
                )

            logger.info(
                f"=== Five-Phase Ingest Complete: "
                f"{successful_total}/{total} files, {total_chunks} chunks, "
                f"{raptor_nodes} RAPTOR nodes, {graph_entities} graph entities ==="
            )

            # Auto-enable ingested topics so they're searchable immediately
            try:
                from config import runtime_config
                enabled = runtime_config.get_enabled_topics()
                ingested_topics = list(results.keys())
                added = [t for t in ingested_topics if t not in enabled]
                if added:
                    runtime_config.set_enabled_topics(enabled + added)
                    runtime_config.save_overrides()
                    logger.info(f"Auto-enabled topics: {added}")
            except Exception as e:
                logger.debug(f"Topic auto-enable skipped: {e}")

            return {
                "new_files": total,
                "successful": successful_total,
                "failed": failed_total,
                "stale_removed": stale_count,
                "topics_discovered": self.discover_topics(),
                "results": results,
                "phases": {
                    "text": len(file_states),
                    "vision": vision_count,
                    "embed": total_chunks,
                    "raptor": raptor_nodes,
                    "graph_entities": graph_entities,
                    "graph_relationships": graph_relationships,
                },
            }

        finally:
            # Cleanup spillover files
            for state in file_states:
                state.cleanup()
            # Remove temp dir (best effort)
            try:
                temp_dir.rmdir()
            except OSError:
                pass

    def ingest_single_file(
        self,
        file_path: Path,
        topic: str,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> Dict[str, Any]:
        """
        Ingest a single file through the Docling pipeline (phases 1-3).

        Used by admin panel re-index/ingest buttons for individual files.
        Skips RAPTOR and GraphRAG (phases 4-5) — those run on full reprocess.

        Args:
            file_path: Path to the file to ingest
            topic: Target topic name
            chunk_size: Optional override for chunk size
            chunk_overlap: Optional override for chunk overlap

        Returns:
            Dict with ingestion results
        """
        import tempfile

        file_path = Path(file_path)
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        logger.info(f"Single file ingest: {file_path.name} -> {topic}")

        temp_dir = Path(tempfile.mkdtemp(prefix="arca_single_"))

        try:
            # Phase 1: Extract text (Docling for PDFs)
            new_files = {topic: [file_path]}
            file_states = self._phase1_extract_all(new_files, temp_dir)

            if not file_states:
                return {"success": False, "error": "Text extraction produced no results"}

            # Phase 2: Vision (if enabled)
            vision_count = 0
            if not self._skip_vision:
                vision_count = self._phase2_extract_vision(file_states)

            # Phase 3: Chunk + Embed
            # Apply chunk size overrides if provided
            if chunk_size is not None or chunk_overlap is not None:
                from config import runtime_config
                original_chunk_size = getattr(runtime_config, 'chunk_size', None)
                original_chunk_overlap = getattr(runtime_config, 'chunk_overlap', None)
                if chunk_size is not None and hasattr(runtime_config, 'chunk_size'):
                    runtime_config.chunk_size = chunk_size
                if chunk_overlap is not None and hasattr(runtime_config, 'chunk_overlap'):
                    runtime_config.chunk_overlap = chunk_overlap

            results, successful, failed, total_chunks = self._phase3_chunk_and_embed(file_states)

            # Restore chunk size if overridden
            if chunk_size is not None or chunk_overlap is not None:
                if chunk_size is not None and original_chunk_size is not None:
                    runtime_config.chunk_size = original_chunk_size
                if chunk_overlap is not None and original_chunk_overlap is not None:
                    runtime_config.chunk_overlap = original_chunk_overlap

            # Save manifest
            self.manifest.save()

            state = file_states[0]
            extractor_used = "docling" if file_path.suffix.lower() == ".pdf" else "text"

            # Auto-enable topic so it's searchable immediately
            if successful > 0:
                try:
                    from config import runtime_config
                    if not runtime_config.is_topic_enabled(topic):
                        enabled = runtime_config.get_enabled_topics()
                        runtime_config.set_enabled_topics(enabled + [topic])
                        runtime_config.save_overrides()
                        logger.info(f"Auto-enabled topic: {topic}")
                except Exception as e:
                    logger.debug(f"Topic auto-enable skipped: {e}")

            return {
                "success": successful > 0,
                "chunks_created": total_chunks,
                "chunks_failed": 0 if successful > 0 else 1,
                "extractor_used": extractor_used,
                "vision_pages": vision_count,
                "warnings": state.warnings if state else [],
                "error": None if successful > 0 else "Chunking/embedding failed",
            }

        except Exception as e:
            logger.error(f"Single file ingest failed: {e}", exc_info=True)
            return {
                "success": False,
                "chunks_created": 0,
                "chunks_failed": 1,
                "extractor_used": None,
                "warnings": [],
                "error": str(e),
            }
        finally:
            try:
                temp_dir.rmdir()
            except OSError:
                pass

    def run(self) -> Dict[str, Any]:
        """
        Run auto-ingestion.

        1. Cleanup stale manifest entries
        2. Scan for new/changed files
        3. Ingest new files
        4. Update manifest

        Returns:
            Dict with ingestion results
        """
        logger.info("Starting auto-ingestion scan...")

        # 1. Cleanup stale entries
        stale_count = self.cleanup_stale_entries()

        # 2. Scan for new files
        new_files = self.scan_new_files()
        if not new_files:
            logger.info("Knowledge base up to date - no new files to ingest")
            return {
                "new_files": 0,
                "stale_removed": stale_count,
                "topics_discovered": self.discover_topics(),
                "results": {},
            }

        total = sum(len(files) for files in new_files.values())
        logger.info(f"Found {total} new/changed files to ingest across {len(new_files)} topics")

        # 3. Ingest new files
        results = {}
        successful_total = 0
        failed_total = 0

        for topic, files in new_files.items():
            topic_results = []
            for file_path in files:
                logger.info(f"Ingesting: {file_path.name} -> {topic}")
                try:
                    result = self.ingester.ingest_file(file_path, topic)
                    if result.success:
                        self.manifest.record(file_path, topic, result.chunks_created)
                        successful_total += 1
                    else:
                        failed_total += 1
                        logger.warning(f"Failed to ingest {file_path.name}: {result.error}")

                    topic_results.append(
                        {
                            "file": file_path.name,
                            "success": result.success,
                            "chunks": result.chunks_created,
                            "error": result.error,
                        }
                    )
                except Exception as e:
                    failed_total += 1
                    logger.error(f"Exception ingesting {file_path.name}: {e}")
                    topic_results.append(
                        {
                            "file": file_path.name,
                            "success": False,
                            "chunks": 0,
                            "error": str(e),
                        }
                    )

            results[topic] = topic_results

        # 4. Save manifest
        self.manifest.save()

        logger.info(f"Auto-ingestion complete: {successful_total} succeeded, {failed_total} failed")

        return {
            "new_files": total,
            "successful": successful_total,
            "failed": failed_total,
            "stale_removed": stale_count,
            "topics_discovered": self.discover_topics(),
            "results": results,
        }


def run_auto_ingestion(knowledge_dir: Path, db_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Convenience function to run auto-ingestion.

    Args:
        knowledge_dir: Path to technical_knowledge directory
        db_dir: Path to cohesionn_db directory

    Returns:
        Ingestion results dict, or None on error
    """
    try:
        service = AutoIngestService(knowledge_dir, db_dir)
        return service.run()
    except Exception as e:
        logger.error(f"Auto-ingestion failed: {e}")
        return None
