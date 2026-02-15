"""
Ingest phase methods extracted from AutoIngestService.

IngestPhasesMixin provides the five ingest phases as a mixin class
that AutoIngestService inherits from. This keeps autoingest.py focused
on orchestration while the heavy phase logic lives here.

Phase overview:
  1. Text extraction (CPU — Docling/PyMuPDF4LLM + page classification)
  2. Vision extraction (GPU — llama.cpp vision server)
  3. Chunk + Embed (GPU — ONNX embedder)
  4. RAPTOR tree building (LLM — chat model)
  5. GraphRAG entity extraction (LLM — chat model)
"""

import logging
from pathlib import Path
from typing import Dict, List

from .autoingest_types import PhasedFileState, _RAM_PRESSURE_THRESHOLD

logger = logging.getLogger(__name__)


class IngestPhasesMixin:
    """Mixin providing the five ingest phase methods for AutoIngestService."""

    def _phase1_extract_all(
        self,
        new_files: Dict[str, List[Path]],
        temp_dir: Path,
    ) -> List[PhasedFileState]:
        """
        Phase 1: Extract text from all files, preferring Docling when available.

        Uses Docling as primary extractor for PDFs (better table/figure detection),
        falls back to HybridExtractor (pymupdf4llm + page classification) if Docling
        is not installed. For non-PDF files, delegates to the ingester's extract_text_only.

        Docling's figure_pages metadata triggers targeted vision extraction in Phase 2
        by creating synthetic PageClassification entries with has_charts=True.
        """
        total = sum(len(files) for files in new_files.values())
        logger.info(f"Phase 1: Text extraction ({total} files)")

        # Check if Docling is available
        docling_available = False
        try:
            from tools.readd.extractors import EXTRACTORS
            if "docling" in EXTRACTORS:
                docling_available = True
                logger.info("Phase 1: Using Docling as primary extractor")
        except ImportError:
            pass

        if not docling_available:
            logger.info("Phase 1: Docling not available, using legacy HybridExtractor path")
            return self._phase1_extract_text_legacy(new_files, temp_dir)

        from tools.readd.extractors import get_extractor
        from tools.readd.page_classifier import PageClassification, PageType

        file_states = []
        processed = 0

        for topic, files in new_files.items():
            for file_path in files:
                processed += 1
                logger.info(f"[Phase 1] [{processed}/{total}] {file_path.name} -> {topic}")

                try:
                    # Non-PDF files: use the existing text extraction path
                    if file_path.suffix.lower() != ".pdf":
                        result = self.ingester.extract_text_only(file_path, topic)
                        state = PhasedFileState(
                            file_path=file_path,
                            topic=topic,
                            text_pages=result.get("text_pages", []),
                            vision_page_specs=result.get("vision_page_specs", []),
                            page_classifications=result.get("page_classifications", []),
                            warnings=result.get("warnings", []),
                            full_text=result.get("full_text", ""),
                        )
                        file_states.append(state)
                        logger.info(
                            f"[Phase 1] {file_path.name}: "
                            f"{len(state.text_pages)} text pages (non-PDF)"
                        )
                        self._check_ram_pressure(file_states, temp_dir)
                        continue

                    # PDF files: use Docling extractor.
                    # GPU is safe now — the ONNX embedder runs in a subprocess with
                    # its own isolated CUDA context (embed_worker.py), so Docling's
                    # PyTorch allocations can't corrupt Phase 3 embedding.
                    from tools.readd.extractors import DoclingExtractor
                    extractor = DoclingExtractor(device="auto")
                    extraction = extractor.extract(file_path)

                    # Build text_pages from extraction result
                    text_pages = []
                    if extraction.pages:
                        for p in extraction.pages:
                            text_pages.append({
                                "page_num": p.page_num,
                                "text": p.text,
                                "content_type": p.metadata.get("content_type", "prose"),
                            })

                    # Build vision_page_specs from Docling's figure_pages metadata.
                    # Two-gate filter: only send pages to heavy vision model when
                    # the figure is large enough to be a chart/diagram worth capturing,
                    # OR when Docling failed to extract meaningful text (scanned/visual page).
                    vision_page_specs = []
                    figure_pages = extraction.metadata.get("figure_pages", [])
                    figure_details = extraction.metadata.get("figure_details", [])
                    page_dimensions = extraction.metadata.get("page_dimensions", {})

                    # Index: page_num -> largest figure area ratio on that page
                    page_max_area_ratio = {}
                    for fig in figure_details:
                        pg = fig.get("page")
                        bbox = fig.get("bbox")
                        if pg is None or bbox is None:
                            continue
                        dims = page_dimensions.get(pg)
                        if not dims:
                            continue
                        fig_area = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                        page_area = dims["width"] * dims["height"]
                        ratio = fig_area / page_area if page_area > 0 else 0
                        page_max_area_ratio[pg] = max(page_max_area_ratio.get(pg, 0), ratio)

                    # Index: page_num -> text char count from Docling extraction
                    page_text_len = {}
                    for p in extraction.pages:
                        page_text_len[p.page_num] = len(p.text.strip()) if p.text else 0

                    skipped_vision = 0
                    for pg in figure_pages:
                        area_ratio = page_max_area_ratio.get(pg, 0)
                        text_len = page_text_len.get(pg, 0)

                        # Gate 1: Large figure (>20% of page) — likely a chart worth capturing
                        # Gate 2: Sparse text (<300 chars) — Docling didn't get much, need vision
                        if area_ratio < 0.20 and text_len >= 300:
                            skipped_vision += 1
                            continue

                        fake_classification = PageClassification(
                            page_num=pg,
                            page_type=PageType.VISUAL,
                            confidence=0.9,
                            has_tables=False,
                            has_figures=True,
                            has_equations=False,
                            has_charts=True,
                            is_scanned=False,
                            text_density=0,
                            image_coverage=area_ratio,
                            detected_elements=["docling_figure"],
                        )
                        vision_page_specs.append((pg, fake_classification))

                    if skipped_vision:
                        logger.info(
                            f"[Phase 1] {file_path.name}: filtered {skipped_vision} "
                            f"small-figure pages (kept {len(vision_page_specs)} for vision)"
                        )

                    full_text = extraction.full_text or ""
                    warnings = list(extraction.warnings) if extraction.warnings else []

                    state = PhasedFileState(
                        file_path=file_path,
                        topic=topic,
                        text_pages=text_pages,
                        vision_page_specs=vision_page_specs,
                        warnings=warnings,
                        full_text=full_text,
                    )
                    file_states.append(state)

                    text_count = len(text_pages)
                    vision_count = len(vision_page_specs)
                    total_pages = text_count + vision_count
                    vision_pct = (vision_count / total_pages * 100) if total_pages else 0

                    logger.info(
                        f"[Phase 1] {file_path.name}: "
                        f"{text_count} text, {vision_count} vision "
                        f"({vision_pct:.0f}% vision) [docling]"
                    )

                except Exception as e:
                    logger.error(f"[Phase 1] Docling failed for {file_path.name}: {e}")
                    # Fall back to legacy extraction for this file
                    try:
                        result = self.ingester.extract_text_only(file_path, topic)
                        state = PhasedFileState(
                            file_path=file_path,
                            topic=topic,
                            text_pages=result.get("text_pages", []),
                            vision_page_specs=result.get("vision_page_specs", []),
                            page_classifications=result.get("page_classifications", []),
                            warnings=result.get("warnings", [])
                            + [f"Docling failed, used legacy extraction: {e}"],
                            full_text=result.get("full_text", ""),
                        )
                        file_states.append(state)
                    except Exception as e2:
                        logger.error(f"[Phase 1] Legacy fallback also failed for {file_path.name}: {e2}")
                        file_states.append(PhasedFileState(
                            file_path=file_path,
                            topic=topic,
                            text_pages=[],
                            warnings=[f"All extraction failed: docling={e}, legacy={e2}"],
                        ))

                # Check RAM pressure after each file
                self._check_ram_pressure(file_states, temp_dir)

        return file_states

    def _phase1_extract_text_legacy(
        self,
        new_files: Dict[str, List[Path]],
        temp_dir: Path,
    ) -> List[PhasedFileState]:
        """
        Phase 1 (legacy): Extract text from all files using HybridExtractor (CPU only).

        Classifies pages and extracts text for simple pages. Records
        which pages need vision for phase 2.

        Kept for backward compatibility. New code should use _phase1_extract_all().
        """
        total = sum(len(files) for files in new_files.values())
        logger.info(f"Phase 1: Text extraction ({total} files, CPU only)")

        file_states = []
        processed = 0

        for topic, files in new_files.items():
            for file_path in files:
                processed += 1
                logger.info(f"[Phase 1] [{processed}/{total}] {file_path.name} -> {topic}")

                try:
                    result = self.ingester.extract_text_only(file_path, topic)

                    state = PhasedFileState(
                        file_path=file_path,
                        topic=topic,
                        text_pages=result.get("text_pages", []),
                        vision_page_specs=result.get("vision_page_specs", []),
                        page_classifications=result.get("page_classifications", []),
                        warnings=result.get("warnings", []),
                        full_text=result.get("full_text", ""),
                    )
                    file_states.append(state)

                    text_count = len(state.text_pages)
                    vision_count = len(state.vision_page_specs)
                    total_pages = text_count + vision_count
                    vision_pct = (vision_count / total_pages * 100) if total_pages else 0

                    # Log density profile if available
                    density_profile = result.get("density_profile", {})
                    density_info = ""
                    if density_profile:
                        density_info = (
                            f", density P5={density_profile.get('p5', 'N/A')}"
                            f" P50={density_profile.get('p50', 'N/A')}"
                            f" scanned={density_profile.get('is_scanned', False)}"
                        )

                    logger.info(
                        f"[Phase 1] {file_path.name}: "
                        f"{text_count} text, {vision_count} vision "
                        f"({vision_pct:.0f}% vision){density_info}"
                    )

                except Exception as e:
                    logger.error(f"[Phase 1] Failed for {file_path.name}: {e}")
                    # Create empty state so we can still report the failure
                    file_states.append(PhasedFileState(
                        file_path=file_path,
                        topic=topic,
                        text_pages=[],
                        warnings=[f"Text extraction failed: {e}"],
                    ))

                # Check RAM pressure after each file
                self._check_ram_pressure(file_states, temp_dir)

        return file_states

    def _phase2_extract_vision(self, file_states: List[PhasedFileState]) -> int:
        """
        Phase 2: Vision extraction for pages that need it (GPU: vision model only).

        Starts the vision server, processes all vision pages across all files,
        then stops the vision server to free VRAM for embedding.

        Returns:
            Total number of vision pages processed
        """
        needs_vision = [s for s in file_states if s.vision_page_specs]
        if not needs_vision:
            logger.info("Phase 2: No vision pages needed, skipping")
            return 0

        total_vision_pages = sum(len(s.vision_page_specs) for s in needs_vision)
        total_all_pages = sum(
            len(s.text_pages) + len(s.vision_page_specs) for s in file_states
        )
        vision_pct = (total_vision_pages / total_all_pages * 100) if total_all_pages else 0
        logger.info(
            f"Phase 2: Vision extraction "
            f"({total_vision_pages} pages across {len(needs_vision)} files, "
            f"{vision_pct:.0f}% of {total_all_pages} total pages)"
        )

        # Unload ONNX embedder + reranker from GPU before starting vision server.
        # warm_models() loaded them at app startup — they must be evicted so the
        # vision llama-server (~16GB VRAM) gets exclusive GPU access.
        # Phase 3 will reload them fresh via get_embedder()/get_reranker().
        try:
            from tools.cohesionn import unload_onnx_models
            unload_onnx_models()
        except Exception as e:
            logger.warning(f"Failed to unload ONNX models: {e}")

        import asyncio

        # Start vision server
        try:
            from utils.llm import get_server_manager
            mgr = get_server_manager()
            mgr.start("vision")

            loop = asyncio.new_event_loop()
            try:
                healthy = loop.run_until_complete(
                    mgr._wait_for_healthy("vision", timeout=180.0)
                )
            finally:
                loop.close()

            if not healthy:
                logger.error("Vision server failed to start — falling back to text for all vision pages")
                self._fallback_vision_to_text(needs_vision)
                return 0

            logger.info("Vision server started and healthy")
        except Exception as e:
            logger.error(f"Vision server startup failed: {e}")
            self._fallback_vision_to_text(needs_vision)
            return 0

        # Process vision pages for each file, always stop vision server after
        vision_processed = 0
        try:
            try:
                from tools.readd import HybridExtractor, ExtractionMode

                extractor = HybridExtractor(mode=ExtractionMode.KNOWLEDGE_BASE)

                for state in needs_vision:
                    # Reload from disk if spilled
                    state.load_from_disk()

                    logger.info(
                        f"[Phase 2] {state.file_path.name}: "
                        f"{len(state.vision_page_specs)} vision pages"
                    )

                    try:
                        vision_results = extractor._extract_vision_pages(
                            state.file_path,
                            state.vision_page_specs,
                        )

                        # Merge vision results into text_pages
                        for page_num, extracted_page in vision_results.items():
                            content_type = extracted_page.metadata.get("content_type", "prose")

                            if content_type == "chart_data" and extracted_page.metadata.get("chart_json"):
                                # Dual-content strategy for charts
                                state.text_pages.append({
                                    "page_num": page_num,
                                    "text": extracted_page.metadata["chart_json"],
                                    "content_type": "chart_data",
                                })
                                state.text_pages.append({
                                    "page_num": page_num,
                                    "text": extracted_page.text,
                                    "content_type": "prose",
                                })
                            else:
                                state.text_pages.append({
                                    "page_num": page_num,
                                    "text": extracted_page.text,
                                    "content_type": content_type,
                                })
                            vision_processed += 1

                        # Sort pages by page_num for consistent ordering
                        state.text_pages.sort(key=lambda p: p["page_num"])

                        # Log any pages that failed
                        failed_pages = [
                            p for p in state.vision_page_specs
                            if p[0] not in vision_results
                        ]
                        if failed_pages:
                            page_nums = [p[0] for p in failed_pages]
                            state.warnings.append(
                                f"Could not capture visual content for pages {page_nums}"
                            )
                            logger.warning(
                                f"[Phase 2] {state.file_path.name}: "
                                f"failed pages: {page_nums}"
                            )

                    except Exception as e:
                        logger.error(f"[Phase 2] Vision failed for {state.file_path.name}: {e}")
                        state.warnings.append(f"Vision extraction failed: {e}")

            except ImportError as e:
                logger.error(f"HybridExtractor not available: {e}")
                self._fallback_vision_to_text(needs_vision)

        finally:
            # Always stop vision server to free VRAM for embedding (phase 3)
            try:
                from utils.llm import get_server_manager
                mgr = get_server_manager()
                mgr.stop("vision")
                logger.info("Vision server stopped (VRAM freed for embedding)")
                # Brief pause for GPU memory release
                import time
                time.sleep(2)

                # Full CUDA cleanup before embedding phase
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass

                from services.hardware import cuda_health_check
                if not cuda_health_check():
                    logger.warning(
                        "CUDA context unhealthy after vision phase — "
                        "subprocess embedder will use fresh context"
                    )
            except Exception as e:
                logger.warning(f"Failed to stop vision server: {e}")

        return vision_processed

    def _phase3_chunk_and_embed(
        self,
        file_states: List[PhasedFileState],
    ) -> tuple:
        """
        Phase 3: Chunk and embed all files (GPU: ONNX embedder only).

        Returns:
            (results_dict, successful_count, failed_count, total_chunks)
        """
        logger.info(f"Phase 3: Embedding ({len(file_states)} files, ONNX embedder)")

        results = {}
        successful_total = 0
        failed_total = 0
        total_chunks = 0

        for i, state in enumerate(file_states, 1):
            # Reload from disk if spilled
            state.load_from_disk()

            logger.info(
                f"[Phase 3] [{i}/{len(file_states)}] "
                f"{state.file_path.name} -> {state.topic}"
            )

            topic = state.topic
            if topic not in results:
                results[topic] = []

            try:
                result = self.ingester.chunk_and_embed(
                    text_pages=state.text_pages,
                    topic=topic,
                    file_path=state.file_path,
                    page_classifications=state.page_classifications,
                )

                if result.success:
                    self.manifest.record(state.file_path, topic, result.chunks_created)
                    successful_total += 1
                    total_chunks += result.chunks_created
                else:
                    failed_total += 1
                    logger.warning(f"[Phase 3] Failed: {state.file_path.name}: {result.error}")

                results[topic].append({
                    "file": state.file_path.name,
                    "success": result.success,
                    "chunks": result.chunks_created,
                    "error": result.error,
                    "warnings": state.warnings + result.warnings,
                })

            except Exception as e:
                failed_total += 1
                logger.error(f"[Phase 3] Exception for {state.file_path.name}: {e}")
                results[topic].append({
                    "file": state.file_path.name,
                    "success": False,
                    "chunks": 0,
                    "error": str(e),
                })

            # Cleanup state to free RAM
            state.text_pages = []
            state.full_text = ""
            state.cleanup()

        return results, successful_total, failed_total, total_chunks

    def _check_ram_pressure(
        self,
        file_states: List[PhasedFileState],
        temp_dir: Path,
    ) -> None:
        """Spill oldest non-spilled file to disk if RAM is under pressure."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_ratio = mem.available / mem.total
            if available_ratio < _RAM_PRESSURE_THRESHOLD:
                # Find oldest non-spilled state
                for state in file_states:
                    if state.text_pages and state._spilled_path is None:
                        logger.info(
                            f"RAM pressure ({available_ratio:.0%} available), "
                            f"spilling {state.file_path.name} to disk"
                        )
                        state.spill_to_disk(temp_dir)
                        break
        except ImportError:
            pass  # psutil not available, skip RAM monitoring

    def _phase4_build_raptor(self, topics: List[str]) -> int:
        """
        Phase 4: Build RAPTOR hierarchical summaries for ingested topics.

        Returns:
            Total number of RAPTOR nodes created
        """
        logger.info(f"Phase 4: RAPTOR tree building ({len(topics)} topics)")
        total_nodes = 0

        try:
            from tools.cohesionn.raptor import RaptorTreeBuilder

            builder = RaptorTreeBuilder(max_levels=3)

            for i, topic in enumerate(topics, 1):
                try:
                    result = builder.build_tree(topic=topic, rebuild=True)
                    total_nodes += result.total_nodes
                    logger.info(
                        f"[Phase 4] [{i}/{len(topics)}] {topic}: "
                        f"{result.total_nodes} nodes, {result.levels_built} levels"
                    )
                except Exception as e:
                    logger.error(f"[Phase 4] RAPTOR failed for {topic}: {e}")

        except ImportError as e:
            logger.warning(f"RAPTOR module not available, skipping: {e}")

        return total_nodes

    def _phase5_build_graph(self, topics: List[str]) -> tuple:
        """
        Phase 5: Build GraphRAG knowledge graph for ingested topics.

        Returns:
            (total_entities, total_relationships)
        """
        logger.info(f"Phase 5: GraphRAG building ({len(topics)} topics)")
        total_entities = 0
        total_relationships = 0

        try:
            from tools.cohesionn.graph_builder import GraphBuilder

            builder = GraphBuilder()

            for i, topic in enumerate(topics, 1):
                try:
                    result = builder.build_graph(topic=topic, incremental=False)
                    total_entities += result.entities_created
                    total_relationships += result.relationships_created
                    logger.info(
                        f"[Phase 5] [{i}/{len(topics)}] {topic}: "
                        f"{result.entities_created} entities, "
                        f"{result.relationships_created} relationships"
                    )
                except Exception as e:
                    logger.error(f"[Phase 5] GraphRAG failed for {topic}: {e}")

        except ImportError as e:
            logger.warning(f"GraphRAG module not available, skipping: {e}")

        return total_entities, total_relationships

    def _fallback_vision_to_text(self, states: List[PhasedFileState]) -> None:
        """When vision server fails, try text extraction for vision pages."""
        for state in states:
            for page_num, _classification in state.vision_page_specs:
                state.warnings.append(
                    f"Page {page_num}: vision unavailable, no text fallback in batch mode"
                )
            # Clear vision specs so phase 3 doesn't expect them
            state.vision_page_specs = []
