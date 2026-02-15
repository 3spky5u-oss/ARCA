"""
Hybrid Extractor - Page-level routing for optimal extraction

Routes individual pages to the most appropriate extractor based on
content classification, combining fast text extraction for simple pages
with vision extraction for complex content.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF

from .page_classifier import PageClassifier, PageClassification, PageType
from .extractors import (
    ExtractionResult,
    ExtractedPage,
    PyMuPDF4LLMExtractor,
    BaseExtractor,
)
from config import runtime_config
from utils.llm import get_llm_client, get_server_manager

logger = logging.getLogger(__name__)


class ExtractionMode:
    """Extraction mode constants."""

    KNOWLEDGE_BASE = "knowledge_base"  # Slow, high quality - aggressive vision routing
    SESSION = "session"  # Fast, good enough - minimal vision


@dataclass
class HybridExtractionResult(ExtractionResult):
    """Extended result with per-page extraction details."""

    page_classifications: List[Dict[str, Any]] = field(default_factory=list)
    pages_by_extractor: Dict[str, int] = field(default_factory=dict)


class HybridExtractor(BaseExtractor):
    """
    Routes pages to appropriate extractors based on content classification.

    In knowledge_base mode:
        - SIMPLE_TEXT → PyMuPDF4LLM (fast, layout-aware)
        - STRUCTURED  → PyMuPDF4LLM (tables handled well by layout-aware extraction)
        - VISUAL      → Observationn with "technical" prompt
        - SCANNED     → Observationn full extraction

    In session mode:
        - All pages → PyMuPDF4LLM first
        - Escalate to Observationn only on QA failure (existing behavior)
    """

    name = "hybrid"

    def __init__(
        self,
        mode: str = ExtractionMode.KNOWLEDGE_BASE,
        max_vision_workers: Optional[int] = None,  # From config if None
        vision_model: Optional[str] = None,
        vision_dpi: int = 150,
        vision_num_ctx: Optional[int] = None,  # From config if None
        vision_timeout: Optional[int] = None,  # From config if None
    ):
        """
        Args:
            mode: Extraction mode (knowledge_base or session)
            max_vision_workers: Max parallel vision model calls (default from config)
            vision_model: Override vision model (default from config)
            vision_dpi: DPI for vision page rendering (100-150)
            vision_num_ctx: Context window for vision model (default from config)
            vision_timeout: Max seconds per page (default from config)
        """
        self.mode = mode
        self.max_vision_workers = max_vision_workers or runtime_config.vision_max_workers
        self.vision_model = vision_model
        self.vision_dpi = vision_dpi
        self.vision_num_ctx = vision_num_ctx or runtime_config.vision_num_ctx
        self.vision_timeout = vision_timeout or runtime_config.vision_timeout

        self.classifier = PageClassifier()
        self.text_extractor = PyMuPDF4LLMExtractor()
        self._vision_extractor = None  # Lazy init
        self._graph_extractor = None  # Lazy init (KB mode only)
        self.failed_pages: List[Dict[str, Any]] = []  # Track failed extractions

        # KB mode: use 2 workers (chat model unloaded during ingest frees VRAM)
        if self.mode == ExtractionMode.KNOWLEDGE_BASE:
            self.max_vision_workers = max(self.max_vision_workers, 2)

    @property
    def vision_extractor(self):
        """Lazy-load vision extractor to avoid import overhead."""
        if self._vision_extractor is None:
            try:
                from tools.observationn import ObservationnExtractor

                kwargs = {
                    "dpi": self.vision_dpi,
                    "num_ctx": self.vision_num_ctx,
                    "timeout": self.vision_timeout,
                }
                if self.vision_model:
                    kwargs["model"] = self.vision_model
                self._vision_extractor = ObservationnExtractor(**kwargs)
            except ImportError:
                logger.warning("Observationn not available, falling back to text extraction")
                self._vision_extractor = None
        return self._vision_extractor

    @property
    def graph_extractor(self):
        """Lazy-load graph extractor for structured chart JSON extraction (KB mode only)."""
        if self._graph_extractor is None and self.mode == ExtractionMode.KNOWLEDGE_BASE:
            try:
                from .graph_extractor import GraphExtractor

                self._graph_extractor = GraphExtractor(
                    model=runtime_config.model_vision_structured,
                    dpi=200,
                    num_ctx=4096,
                )
            except ImportError:
                logger.warning("GraphExtractor not available")
                self._graph_extractor = None
        return self._graph_extractor

    def get_failed_pages(self) -> List[Dict[str, Any]]:
        """Return list of pages that failed extraction (for retry/review)."""
        return self.failed_pages

    def export_failed_pages_pdf(self, source_pdf: Path, output_path: Path) -> Optional[Path]:
        """
        Export failed pages to a new PDF for manual review or retry with different settings.

        Args:
            source_pdf: Original PDF file
            output_path: Where to save the failed pages PDF

        Returns:
            Path to the created PDF, or None if no failed pages
        """
        if not self.failed_pages:
            logger.info("No failed pages to export")
            return None

        failed_page_nums = [fp["page_num"] for fp in self.failed_pages]
        logger.info(f"Exporting {len(failed_page_nums)} failed pages to {output_path}")

        try:
            source_doc = fitz.open(str(source_pdf))
            failed_doc = fitz.open()  # New empty PDF

            for page_num in sorted(failed_page_nums):
                # PyMuPDF uses 0-indexed pages
                failed_doc.insert_pdf(source_doc, from_page=page_num-1, to_page=page_num-1)

            failed_doc.save(str(output_path))
            failed_doc.close()
            source_doc.close()

            logger.info(f"Saved failed pages PDF: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to export failed pages: {e}")
            return None

    def extract(self, file_path: Path) -> HybridExtractionResult:
        """
        Extract document using hybrid page-level routing.

        Args:
            file_path: Path to PDF file

        Returns:
            HybridExtractionResult with extracted content and metadata
        """
        file_path = Path(file_path)
        logger.info(f"Hybrid extraction ({self.mode}): {file_path.name}")

        # Session mode: use existing fast path
        if self.mode == ExtractionMode.SESSION:
            return self._extract_session_mode(file_path)

        # Knowledge base mode: per-page routing
        return self._extract_knowledge_base_mode(file_path)

    def _extract_session_mode(self, file_path: Path) -> HybridExtractionResult:
        """Session mode: fast extraction, current behavior."""
        # Just use PyMuPDF4LLM for everything
        result = self.text_extractor.extract(file_path)

        return HybridExtractionResult(
            file_path=result.file_path,
            extractor_used="hybrid_session",
            page_count=result.page_count,
            pages=result.pages,
            full_text=result.full_text,
            metadata={**result.metadata, "mode": "session"},
            warnings=result.warnings,
            page_classifications=[],
            pages_by_extractor={"pymupdf4llm": result.page_count},
        )

    def _extract_knowledge_base_mode(self, file_path: Path) -> HybridExtractionResult:
        """Knowledge base mode: per-page routing for maximum quality."""
        doc = fitz.open(str(file_path))

        # Phase 1: Classify all pages
        logger.info(f"Classifying {len(doc)} pages...")
        classifications, profile = self.classifier.classify_document(file_path)

        # Group pages by extraction method
        text_pages = []  # SIMPLE_TEXT, STRUCTURED → PyMuPDF4LLM
        vision_pages = []  # VISUAL, SCANNED → Observationn

        for classification in classifications:
            if classification.page_type in (PageType.SIMPLE_TEXT, PageType.STRUCTURED):
                text_pages.append(classification.page_num)
            else:
                # Only VISUAL and SCANNED go to vision
                vision_pages.append((classification.page_num, classification))

        logger.info(f"Routing: {len(text_pages)} text pages, {len(vision_pages)} vision pages")

        # Phase 2: Extract text pages in batch
        text_extracted = {}
        if text_pages:
            text_extracted = self._extract_text_pages(file_path, text_pages)

        # Phase 3: Extract vision pages (potentially in parallel)
        vision_extracted = {}
        if vision_pages and self.vision_extractor:
            vision_extracted = self._extract_vision_pages(file_path, vision_pages)

        doc.close()

        # Phase 4: Merge results in page order
        pages = []
        full_text_parts = []
        fallback_count = 0
        failed_count = 0

        for i in range(len(classifications)):
            page_num = i + 1

            if page_num in text_extracted:
                pages.append(text_extracted[page_num])
                full_text_parts.append(text_extracted[page_num].text)
            elif page_num in vision_extracted:
                ep = vision_extracted[page_num]
                extractor_name = ep.metadata.get("extractor", "")
                if "fallback" in extractor_name:
                    fallback_count += 1
                elif ep.metadata.get("error_type"):
                    failed_count += 1
                pages.append(ep)
                full_text_parts.append(ep.text)
            else:
                # Fallback: empty page
                pages.append(
                    ExtractedPage(
                        page_num=page_num,
                        text="[Extraction failed]",
                        metadata={"error": "No extraction result"},
                    )
                )
                full_text_parts.append("")

        # Count graph_extractor pages from metadata
        graph_count = sum(
            1 for ep in vision_extracted.values()
            if ep.metadata.get("extractor") == "graph_extractor"
        )
        vision_ok = len(vision_pages) - fallback_count - failed_count - graph_count
        pages_by_extractor = {
            "pymupdf4llm": len(text_pages),
            "observationn": vision_ok,
        }
        if graph_count:
            pages_by_extractor["graph_extractor"] = graph_count
        if fallback_count:
            pages_by_extractor["text_fallback"] = fallback_count
        if failed_count:
            pages_by_extractor["failed"] = failed_count

        if fallback_count:
            logger.info(f"Vision fallback: {fallback_count} pages recovered via text extraction")

        # Build metadata
        metadata = {
            "mode": "knowledge_base",
            "text_pages": len(text_pages),
            "vision_pages": len(vision_pages),
            "vision_fallback_pages": fallback_count,
            "vision_failed_pages": failed_count,
            "extractors_used": list(set(["pymupdf4llm"] * bool(text_pages) + ["observationn"] * bool(vision_pages))),
        }

        # Get PDF metadata
        doc = fitz.open(str(file_path))
        metadata["title"] = doc.metadata.get("title", "")
        metadata["author"] = doc.metadata.get("author", "")
        doc.close()

        return HybridExtractionResult(
            file_path=str(file_path),
            extractor_used="hybrid_knowledge_base",
            page_count=len(pages),
            pages=pages,
            full_text="\n\n".join(full_text_parts),
            metadata=metadata,
            warnings=[],
            page_classifications=[c.to_dict() for c in classifications],
            pages_by_extractor=pages_by_extractor,
        )

    def _extract_text_pages(
        self,
        file_path: Path,
        page_nums: List[int],
    ) -> Dict[int, ExtractedPage]:
        """Extract text pages using PyMuPDF4LLM.

        Only processes the requested pages (not the full document) and uses
        page_chunks=True to get per-page results directly, avoiding the
        full-document extraction + split overhead.
        """
        logger.info(f"Extracting {len(page_nums)} text pages with PyMuPDF4LLM")

        try:
            import pymupdf4llm

            # Convert 1-based page_nums to 0-based for pymupdf4llm
            pages_0based = [p - 1 for p in page_nums]

            # Extract only requested pages with per-page chunking
            chunks = pymupdf4llm.to_markdown(
                str(file_path),
                pages=pages_0based,
                page_chunks=True,
            )

            # Build result — chunks is a list of dicts with "text" key per page
            result = {}
            for i, page_num in enumerate(page_nums):
                if i < len(chunks):
                    text = chunks[i].get("text", "") if isinstance(chunks[i], dict) else str(chunks[i])
                else:
                    text = ""

                result[page_num] = ExtractedPage(
                    page_num=page_num,
                    text=text.strip(),
                    metadata={
                        "extractor": "pymupdf4llm",
                        "format": "markdown",
                    },
                )

            return result

        except ImportError:
            logger.warning("pymupdf4llm not available, using basic extraction")
            return self._extract_text_pages_basic(file_path, page_nums)

    def _extract_text_pages_basic(
        self,
        file_path: Path,
        page_nums: List[int],
    ) -> Dict[int, ExtractedPage]:
        """Fallback basic text extraction."""
        doc = fitz.open(str(file_path))
        result = {}

        for page_num in page_nums:
            page = doc[page_num - 1]
            text = page.get_text("text")

            result[page_num] = ExtractedPage(
                page_num=page_num,
                text=text.strip(),
                metadata={"extractor": "pymupdf_text"},
            )

        doc.close()
        return result

    def _fallback_extract_page(self, file_path: Path, page_num: int) -> Optional[ExtractedPage]:
        """
        Fallback text extraction for a single page when vision fails.
        Tries pymupdf4llm first, then basic fitz text.

        Returns ExtractedPage or None if even fallback fails.
        """
        # Try pymupdf4llm (layout-aware markdown) for the single page
        try:
            import pymupdf4llm
            md_text = pymupdf4llm.to_markdown(str(file_path), pages=[page_num - 1])
            text = self._clean_fallback_text(md_text)
            if text:
                logger.info(f"Fallback pymupdf4llm extracted {len(text)} chars for page {page_num}")
                return ExtractedPage(
                    page_num=page_num,
                    text=text,
                    metadata={
                        "extractor": "pymupdf4llm_fallback",
                        "fallback_reason": "vision_failed",
                        "format": "markdown",
                    },
                )
        except Exception as e:
            logger.debug(f"pymupdf4llm fallback failed for page {page_num}: {e}")

        # Last resort: basic fitz text extraction
        try:
            doc = fitz.open(str(file_path))
            page = doc[page_num - 1]
            text = page.get_text("text").strip()
            doc.close()
            if text:
                logger.info(f"Fallback fitz extracted {len(text)} chars for page {page_num}")
                return ExtractedPage(
                    page_num=page_num,
                    text=text,
                    metadata={
                        "extractor": "pymupdf_text_fallback",
                        "fallback_reason": "vision_failed",
                    },
                )
        except Exception as e:
            logger.debug(f"fitz fallback failed for page {page_num}: {e}")

        return None

    @staticmethod
    def _clean_fallback_text(text: str) -> str:
        """Clean up fallback-extracted text."""
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.replace("\x00", "")
        return text.strip()

    def _warmup_vision_model(self) -> bool:
        """Ensure vision server is running and responsive.

        Returns True if vision server is ready.
        """
        import asyncio
        logger.info("Checking vision server health before extraction...")
        try:
            mgr = get_server_manager()
            # Check if already healthy
            loop = asyncio.new_event_loop()
            try:
                health = loop.run_until_complete(mgr.health_check("vision"))
            finally:
                loop.close()
            if health.get("status") == "healthy":
                logger.info("Vision server is ready")
                return True
            # Not healthy — start it and wait
            logger.info("Starting vision server...")
            mgr.start("vision")
            loop = asyncio.new_event_loop()
            try:
                healthy = loop.run_until_complete(mgr._wait_for_healthy("vision", timeout=180.0))
            finally:
                loop.close()
            if healthy:
                logger.info("Vision server started and healthy")
                return True
            else:
                logger.error("Vision server failed health check after startup")
                return False
        except Exception as e:
            logger.error(f"Failed to start vision server: {e}")
            return False

    def _extract_vision_pages(
        self,
        file_path: Path,
        pages: List[tuple],  # List of (page_num, classification)
    ) -> Dict[int, ExtractedPage]:
        """Extract vision pages using Observationn."""
        logger.info(f"Extracting {len(pages)} vision pages with Observationn")

        # Warmup: ensure model is on GPU before starting (may have unloaded during text phase)
        self._warmup_vision_model()

        result = {}

        if self.max_vision_workers > 1 and len(pages) > 1:
            # Parallel extraction
            result = self._extract_vision_parallel(file_path, pages)
        else:
            # Sequential extraction
            result = self._extract_vision_sequential(file_path, pages)

        return result

    def _extract_vision_sequential(
        self,
        file_path: Path,
        pages: List[tuple],
    ) -> Dict[int, ExtractedPage]:
        """Extract vision pages sequentially with timeout handling."""
        import time
        result = {}

        for i, (page_num, classification) in enumerate(pages, 1):
            page_start = time.time()
            try:
                # Chart branch: use GraphExtractor for structured JSON extraction (KB mode only)
                if (
                    classification.has_charts
                    and self.mode == ExtractionMode.KNOWLEDGE_BASE
                    and self.graph_extractor
                ):
                    chart_result = self.graph_extractor.extract_page_charts(file_path, page_num)
                    elapsed = time.time() - page_start

                    if chart_result and chart_result.figures:
                        logger.info(
                            f"[{i}/{len(pages)}] Chart page {page_num}: "
                            f"{len(chart_result.figures)} figures extracted in {elapsed:.1f}s"
                        )
                        result[page_num] = ExtractedPage(
                            page_num=page_num,
                            text=chart_result.to_searchable_text(),
                            metadata={
                                "extractor": "graph_extractor",
                                "content_type": "chart_data",
                                "chart_json": chart_result.to_json_chunk(),
                                "figures_count": len(chart_result.figures),
                                "detected_elements": classification.detected_elements,
                                "extraction_time_s": elapsed,
                            },
                        )
                        continue
                    else:
                        logger.info(
                            f"[{i}/{len(pages)}] Chart extraction failed for page {page_num}, "
                            "falling through to Observationn"
                        )

                # Standard vision extraction via Observationn
                prompt_type = self._get_prompt_type(classification)
                page_content = self.vision_extractor.extract_page(file_path, page_num, prompt_type=prompt_type)
                elapsed = time.time() - page_start

                logger.info(f"[{i}/{len(pages)}] Vision page {page_num} completed in {elapsed:.1f}s")
                result[page_num] = ExtractedPage(
                    page_num=page_num,
                    text=page_content.text,
                    metadata={
                        "extractor": "observationn",
                        "content_type": classification.page_type.value,
                        "prompt_type": prompt_type,
                        "detected_elements": classification.detected_elements,
                        "extraction_time_s": elapsed,
                    },
                )
            except Exception as e:
                elapsed = time.time() - page_start
                error_type = "timeout" if "timeout" in str(e).lower() else "error"
                logger.warning(f"[{i}/{len(pages)}] Vision page {page_num} {error_type} after {elapsed:.1f}s: {e}")

                # Try text fallback before giving up
                fallback = self._fallback_extract_page(file_path, page_num)
                if fallback:
                    fallback.metadata["vision_error"] = str(e)
                    fallback.metadata["vision_error_type"] = error_type
                    result[page_num] = fallback
                else:
                    self.failed_pages.append({
                        "page_num": page_num,
                        "error": str(e),
                        "error_type": error_type,
                    })
                    result[page_num] = ExtractedPage(
                        page_num=page_num,
                        text=f"[Vision extraction {error_type}]",
                        metadata={"error": str(e), "error_type": error_type},
                    )

        return result

    def _extract_vision_parallel(
        self,
        file_path: Path,
        pages: List[tuple],
    ) -> Dict[int, ExtractedPage]:
        """Extract vision pages in parallel with timeout handling."""
        import time
        result = {}
        total = len(pages)
        completed = 0
        failed = 0
        start_time = time.time()

        def extract_single(page_info):
            page_num, classification = page_info
            page_start = time.time()
            try:
                # Chart branch: use GraphExtractor for structured JSON extraction (KB mode only)
                if (
                    classification.has_charts
                    and self.mode == ExtractionMode.KNOWLEDGE_BASE
                    and self.graph_extractor
                ):
                    chart_result = self.graph_extractor.extract_page_charts(file_path, page_num)
                    elapsed = time.time() - page_start

                    if chart_result and chart_result.figures:
                        logger.info(
                            f"Chart page {page_num}: {len(chart_result.figures)} figures in {elapsed:.1f}s"
                        )
                        return page_num, ExtractedPage(
                            page_num=page_num,
                            text=chart_result.to_searchable_text(),
                            metadata={
                                "extractor": "graph_extractor",
                                "content_type": "chart_data",
                                "chart_json": chart_result.to_json_chunk(),
                                "figures_count": len(chart_result.figures),
                                "detected_elements": classification.detected_elements,
                                "extraction_time_s": elapsed,
                            },
                        ), None
                    else:
                        logger.info(
                            f"Chart extraction failed for page {page_num}, "
                            "falling through to Observationn"
                        )

                # Standard vision extraction
                prompt_type = self._get_prompt_type(classification)
                page_content = self.vision_extractor.extract_page(file_path, page_num, prompt_type=prompt_type)
                elapsed = time.time() - page_start
                logger.info(f"Vision page {page_num} completed in {elapsed:.1f}s")
                return page_num, ExtractedPage(
                    page_num=page_num,
                    text=page_content.text,
                    metadata={
                        "extractor": "observationn",
                        "content_type": classification.page_type.value,
                        "prompt_type": prompt_type,
                        "detected_elements": classification.detected_elements,
                        "extraction_time_s": elapsed,
                    },
                ), None
            except Exception as e:
                elapsed = time.time() - page_start
                error_type = "timeout" if "timeout" in str(e).lower() else "error"
                logger.warning(f"Vision page {page_num} {error_type} after {elapsed:.1f}s: {e}")

                # Try text fallback before giving up
                fallback = self._fallback_extract_page(file_path, page_num)
                if fallback:
                    fallback.metadata["vision_error"] = str(e)
                    fallback.metadata["vision_error_type"] = error_type
                    return page_num, fallback, None
                return page_num, ExtractedPage(
                    page_num=page_num,
                    text=f"[Vision extraction {error_type}]",
                    metadata={"error": str(e), "error_type": error_type},
                ), {"page_num": page_num, "error": str(e), "error_type": error_type}

        logger.info(f"Starting parallel vision extraction: {total} pages, {self.max_vision_workers} workers")

        with ThreadPoolExecutor(max_workers=self.max_vision_workers) as executor:
            futures = {executor.submit(extract_single, p): p for p in pages}

            for future in as_completed(futures):
                page_num, extracted_page, failure = future.result()
                result[page_num] = extracted_page
                completed += 1
                if failure:
                    failed += 1
                    self.failed_pages.append(failure)

                # Progress logging
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total - completed) / rate if rate > 0 else 0
                logger.info(f"Progress: {completed}/{total} pages ({failed} failed), {rate:.1f} pages/min, ~{remaining:.0f}s remaining")

        total_time = time.time() - start_time
        logger.info(f"Vision extraction complete: {completed} pages in {total_time:.1f}s ({failed} failed)")
        return result

    def _get_prompt_type(self, classification: PageClassification) -> str:
        """Get the appropriate Observationn prompt type for a classification."""
        if classification.has_tables:
            return "table"
        elif classification.has_charts:
            return "technical"
        elif classification.has_equations:
            return "equations"
        elif classification.page_type == PageType.VISUAL:
            return "technical"
        else:
            return "general"

    def _split_markdown_by_pages(self, text: str, page_count: int) -> List[str]:
        """Split markdown text into approximate pages."""
        import re

        # Look for page markers
        page_pattern = re.compile(r"(?:^|\n)(?:---+|===+)\s*(?:Page\s*\d+)?\s*(?:---+|===+)?(?:\n|$)")

        parts = page_pattern.split(text)
        if len(parts) >= page_count * 0.8:  # Close enough to page count
            return parts[:page_count]

        # No good markers - split by approximate character count
        if not text:
            return [""] * page_count

        chars_per_page = len(text) // page_count
        pages = []

        for i in range(page_count):
            start = i * chars_per_page
            end = start + chars_per_page if i < page_count - 1 else len(text)
            pages.append(text[start:end])

        return pages


    def extract_text_only(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text pages only (no vision server, no GPU).

        Used by three-phase batch ingest to separate text extraction (CPU)
        from vision extraction (GPU). Classifies all pages and extracts
        text for SIMPLE_TEXT pages, returning vision page specs for later.

        Args:
            file_path: Path to PDF file

        Returns:
            Dict with keys:
                text_pages: List[Dict] — {page_num, text, content_type}
                vision_page_specs: List[tuple] — [(page_num, classification), ...]
                page_classifications: List[Dict]
                warnings: List[str]
                full_text: str — concatenated text from text pages
        """
        file_path = Path(file_path)
        logger.info(f"Text-only extraction: {file_path.name}")

        warnings = []
        text_pages = []
        vision_page_specs = []

        try:
            doc = fitz.open(str(file_path))
            page_count = len(doc)
            doc.close()
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            return {
                "text_pages": [],
                "vision_page_specs": [],
                "page_classifications": [],
                "warnings": [f"Failed to open PDF: {e}"],
                "full_text": "",
            }

        # Phase 1: Classify all pages (CPU, fast)
        try:
            classifications, profile = self.classifier.classify_document(file_path)
        except Exception as e:
            logger.error(f"Page classification failed: {e}")
            warnings.append(f"Classification failed: {e}")
            classifications = []
            profile = {}

        # Group pages by extraction method
        text_page_nums = []
        for classification in classifications:
            if classification.page_type in (PageType.SIMPLE_TEXT, PageType.STRUCTURED):
                text_page_nums.append(classification.page_num)
            elif classification.page_type in (PageType.VISUAL, PageType.SCANNED):
                vision_page_specs.append((classification.page_num, classification))

        logger.info(
            f"Classification: {len(text_page_nums)} text pages, "
            f"{len(vision_page_specs)} vision pages "
            f"(density P50={profile.get('p50', 'N/A')}, scanned={profile.get('is_scanned', False)})"
        )

        # Phase 2: Extract text pages with PyMuPDF4LLM (CPU only)
        if text_page_nums:
            extracted = self._extract_text_pages(file_path, text_page_nums)
            for page_num in sorted(extracted.keys()):
                ep = extracted[page_num]
                text_pages.append({
                    "page_num": ep.page_num,
                    "text": ep.text,
                    "content_type": "prose",
                })

        # For non-PDF files or if classification is empty, extract everything as text
        if not classifications:
            try:
                result = self.text_extractor.extract(file_path)
                if result.pages:
                    for p in result.pages:
                        text_pages.append({
                            "page_num": p.page_num,
                            "text": p.text,
                            "content_type": "prose",
                        })
            except Exception as e:
                warnings.append(f"Full text extraction failed: {e}")

        full_text = "\n\n".join(p["text"] for p in text_pages if p["text"])

        return {
            "text_pages": text_pages,
            "vision_page_specs": vision_page_specs,
            "page_classifications": [c.to_dict() for c in classifications],
            "warnings": warnings,
            "full_text": full_text,
            "density_profile": profile,
        }


def extract_hybrid(
    file_path: Path,
    mode: str = ExtractionMode.KNOWLEDGE_BASE,
) -> HybridExtractionResult:
    """Convenience function for hybrid extraction."""
    extractor = HybridExtractor(mode=mode)
    return extractor.extract(file_path)
