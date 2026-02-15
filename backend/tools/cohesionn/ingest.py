"""
Cohesionn Ingest - Document ingestion pipeline

Uses Readd for extraction, then chunks and indexes into knowledge base.

Supports two modes:
- knowledge_base: Slow, high-quality extraction with per-page vision routing
- session: Fast extraction for user-uploaded documents
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field

from .chunker import SemanticChunker
from .vectorstore import get_knowledge_base, KnowledgeBase

logger = logging.getLogger(__name__)


class IngestMode:
    """Ingestion mode constants."""

    KNOWLEDGE_BASE = "knowledge_base"  # Slow, high quality with vision routing
    SESSION = "session"  # Fast, text-only extraction


@dataclass
class IngestResult:
    """Result from ingesting a document"""

    file_path: str
    success: bool
    topic: str
    chunks_created: int
    extractor_used: Optional[str]
    warnings: List[str]
    error: Optional[str] = None
    chunks_failed: int = 0  # Track partial failures for resilience
    page_classifications: List[Dict[str, Any]] = field(default_factory=list)
    pages_by_extractor: Dict[str, int] = field(default_factory=dict)


class DocumentIngester:
    """
    Ingest documents into Cohesionn knowledge base.

    Uses Readd for intelligent extraction, then chunks and indexes.

    Modes:
    - knowledge_base: Uses HybridExtractor for per-page routing to vision
    - session: Uses fast text extraction (existing behavior)
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

    def __init__(
        self,
        knowledge_base: KnowledgeBase = None,
        chunk_size: int = 2500,
        chunk_overlap: int = 150,
        use_readd: bool = True,
        force_extractor: Optional[str] = None,
        mode: str = IngestMode.SESSION,  # Default to fast mode for backward compatibility
    ):
        """
        Args:
            knowledge_base: KnowledgeBase instance (uses singleton if None)
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            use_readd: Whether to use Readd pipeline (vs basic extraction)
            force_extractor: Force a specific extractor (pymupdf_text, pymupdf4llm, marker, vision_ocr, hybrid)
            mode: Ingestion mode (knowledge_base or session)
        """
        self.kb = knowledge_base or get_knowledge_base()
        self.chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.use_readd = use_readd
        self.force_extractor = force_extractor
        self.mode = mode

    def ingest_file(self, file_path: Path, topic: str) -> IngestResult:
        """
        Ingest a single document.

        Args:
            file_path: Path to document
            topic: Knowledge base topic name (e.g. general, your_topic_name)

        Returns:
            IngestResult with status and metadata
        """
        file_path = Path(file_path)
        warnings = []
        extractor_used = None

        # Validate
        if not file_path.exists():
            return IngestResult(
                file_path=str(file_path),
                success=False,
                topic=topic,
                chunks_created=0,
                extractor_used=None,
                warnings=[],
                error=f"File not found: {file_path}",
            )

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return IngestResult(
                file_path=str(file_path),
                success=False,
                topic=topic,
                chunks_created=0,
                extractor_used=None,
                warnings=[],
                error=f"Unsupported file type: {file_path.suffix}",
            )

        logger.info(f"Ingesting {file_path.name} into {topic} (mode={self.mode})")

        # Remove existing chunks for this source to prevent duplicates on re-ingest
        try:
            from .vectorstore import get_knowledge_base
            kb = get_knowledge_base()
            store = kb.get_store(topic)
            store.delete_by_source(str(file_path))
            logger.info(f"Cleared previous chunks for {file_path.name} in {topic}")
        except Exception as e:
            logger.debug(f"No previous chunks to clear for {file_path.name}: {e}")

        page_classifications = []
        pages_by_extractor = {}

        try:
            # Extract text
            if file_path.suffix.lower() == ".pdf" and self.use_readd:
                result = self._extract_with_readd(file_path)
                text, pages, extractor_used, extract_warnings = result[:4]
                # Handle extended return values from hybrid extraction
                if len(result) > 4:
                    page_classifications = result[4] or []
                    pages_by_extractor = result[5] or {}
                warnings.extend(extract_warnings)
            else:
                text, pages = self._extract_basic(file_path)
                extractor_used = "basic"

            if not text or len(text.strip()) < 100:
                return IngestResult(
                    file_path=str(file_path),
                    success=False,
                    topic=topic,
                    chunks_created=0,
                    extractor_used=extractor_used,
                    warnings=warnings,
                    error="No text extracted from document",
                )

            # Build metadata
            base_metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "title": file_path.stem,  # Will be overridden if detected
                "topic": topic,
            }

            # Chunk with content type awareness
            if pages:
                # Use page-aware chunking with content type metadata
                chunks = self.chunker.chunk_pages(pages, base_metadata)
            else:
                chunks = self.chunker.chunk_text(text, base_metadata)

            if not chunks:
                return IngestResult(
                    file_path=str(file_path),
                    success=False,
                    topic=topic,
                    chunks_created=0,
                    extractor_used=extractor_used,
                    warnings=warnings,
                    error="No chunks created",
                )

            # Index with resilience - continues on batch failures
            store = self.kb.get_store(topic)
            successful, failed = store.add_chunks(chunks)

            logger.info(f"Ingested {file_path.name}: {successful} chunks added, {failed} failed")

            # Run corpus profiling (non-blocking, fire-and-forget)
            if successful > 0:
                try:
                    from .corpus_profiler import profile_after_ingest
                    profile_after_ingest(chunks, document_count=1)
                except Exception as e:
                    logger.warning(f"Corpus profiling skipped: {e}")

            # Partial success is still considered success if any chunks were added
            return IngestResult(
                file_path=str(file_path),
                success=successful > 0,
                topic=topic,
                chunks_created=successful,
                chunks_failed=failed,
                extractor_used=extractor_used,
                warnings=warnings + ([f"{failed} chunks failed to index"] if failed > 0 else []),
                page_classifications=page_classifications,
                pages_by_extractor=pages_by_extractor,
            )

        except Exception as e:
            logger.error(f"Ingest failed for {file_path}: {e}")
            return IngestResult(
                file_path=str(file_path),
                success=False,
                topic=topic,
                chunks_created=0,
                extractor_used=extractor_used,
                warnings=warnings,
                error=str(e),
            )

    def extract_text_only(self, file_path: Path, topic: str) -> Dict[str, Any]:
        """
        Phase 1 of three-phase ingest: extract text only (CPU, no GPU).

        Classifies pages and extracts text for simple pages. Returns
        vision page specs for later processing. Does NOT call vision
        server or embedder.

        Args:
            file_path: Path to document
            topic: Knowledge base topic name

        Returns:
            Dict with text_pages, vision_page_specs, page_classifications,
            warnings, full_text
        """
        file_path = Path(file_path)
        warnings = []

        if not file_path.exists():
            return {
                "text_pages": [],
                "vision_page_specs": [],
                "page_classifications": [],
                "warnings": [f"File not found: {file_path}"],
                "full_text": "",
            }

        # Non-PDF files: extract all text, no vision needed
        if file_path.suffix.lower() != ".pdf":
            text, pages = self._extract_basic(file_path)
            text_pages = []
            if pages:
                for p in pages:
                    text_pages.append({
                        "page_num": p["page_num"],
                        "text": p["text"],
                        "content_type": "prose",
                    })
            elif text:
                # No page boundaries (docx, txt, md) — use page_num=0
                # to signal "pageless" so chunker omits misleading page metadata
                text_pages.append({
                    "page_num": 0,
                    "text": text,
                    "content_type": "prose",
                })
            return {
                "text_pages": text_pages,
                "vision_page_specs": [],
                "page_classifications": [],
                "warnings": warnings,
                "full_text": text or "",
            }

        # PDF files: use HybridExtractor.extract_text_only()
        try:
            from tools.readd import HybridExtractor, ExtractionMode

            extractor = HybridExtractor(mode=ExtractionMode.KNOWLEDGE_BASE)
            result = extractor.extract_text_only(file_path)
            return result

        except ImportError as e:
            logger.warning(f"HybridExtractor not available: {e}, using basic extraction")
            text, pages = self._extract_basic(file_path)
            text_pages = []
            if pages:
                for p in pages:
                    text_pages.append({
                        "page_num": p["page_num"],
                        "text": p["text"],
                        "content_type": "prose",
                    })
            return {
                "text_pages": text_pages,
                "vision_page_specs": [],
                "page_classifications": [],
                "warnings": [f"HybridExtractor unavailable: {e}"],
                "full_text": text or "",
            }

    def chunk_and_embed(
        self,
        text_pages: List[Dict[str, Any]],
        topic: str,
        file_path: Path,
        page_classifications: Optional[List[Dict]] = None,
    ) -> IngestResult:
        """
        Phase 3 of three-phase ingest: chunk pre-extracted text and embed.

        Takes merged text (from phase 1 text + phase 2 vision) and runs
        chunking + embedding. No extraction happens here.

        Args:
            text_pages: List of {page_num, text, content_type} dicts
            topic: Knowledge base topic name
            file_path: Original file path (for metadata)
            page_classifications: Optional classification data for metadata

        Returns:
            IngestResult with chunk counts and status
        """
        file_path = Path(file_path)
        warnings = []

        # Remove existing chunks for this source to prevent duplicates on re-ingest
        try:
            store = self.kb.get_store(topic)
            store.delete_by_source(str(file_path))
            logger.info(f"Cleared previous chunks for {file_path.name} in {topic}")
        except Exception as e:
            logger.debug(f"No previous chunks to clear for {file_path.name}: {e}")

        # Build full text for validation
        full_text = "\n\n".join(p["text"] for p in text_pages if p.get("text"))

        if not full_text or len(full_text.strip()) < 100:
            return IngestResult(
                file_path=str(file_path),
                success=False,
                topic=topic,
                chunks_created=0,
                extractor_used="phased",
                warnings=warnings,
                error="No text to embed (insufficient content after extraction)",
            )

        # Build metadata
        base_metadata = {
            "source": str(file_path),
            "file_name": file_path.name,
            "title": file_path.stem,
            "topic": topic,
        }

        # Chunk with content type awareness
        chunks = self.chunker.chunk_pages(text_pages, base_metadata)

        if not chunks:
            return IngestResult(
                file_path=str(file_path),
                success=False,
                topic=topic,
                chunks_created=0,
                extractor_used="phased",
                warnings=warnings,
                error="No chunks created from extracted text",
            )

        # Embed with auto-calibration (GPU: ONNX embedder only)
        try:
            store = self.kb.get_store(topic)
            successful, failed = store.add_chunks(chunks)
        except Exception as e:
            logger.error(f"Embedding failed for {file_path.name}: {e}")
            return IngestResult(
                file_path=str(file_path),
                success=False,
                topic=topic,
                chunks_created=0,
                extractor_used="phased",
                warnings=warnings,
                error=f"Embedding failed: {e}",
            )

        logger.info(f"Embedded {file_path.name}: {successful} chunks added, {failed} failed")

        # Corpus profiling (fire-and-forget)
        if successful > 0:
            try:
                from .corpus_profiler import profile_after_ingest
                profile_after_ingest(chunks, document_count=1)
            except Exception as e:
                logger.warning(f"Corpus profiling skipped: {e}")

        return IngestResult(
            file_path=str(file_path),
            success=successful > 0,
            topic=topic,
            chunks_created=successful,
            chunks_failed=failed,
            extractor_used="phased",
            warnings=warnings + ([f"{failed} chunks failed to index"] if failed > 0 else []),
            page_classifications=page_classifications or [],
        )

    def _extract_with_readd(self, file_path: Path):
        """Extract using Readd pipeline"""
        try:
            from tools.readd import process_document
            from tools.readd.extractors import get_extractor, EXTRACTORS

            # If force_extractor is specified, use that directly
            if self.force_extractor:
                # Special handling for hybrid extractor
                if self.force_extractor == "hybrid":
                    return self._extract_with_hybrid(file_path)

                if self.force_extractor not in EXTRACTORS:
                    logger.warning(f"Unknown extractor: {self.force_extractor}, using auto")
                else:
                    extractor = get_extractor(self.force_extractor)
                    extraction = extractor.extract(file_path)

                    pages = None
                    page_metadata = {}
                    if extraction.pages:
                        pages = [
                            {
                                "page_num": p.page_num,
                                "text": p.text,
                                "content_type": p.metadata.get("content_type", "prose"),
                            }
                            for p in extraction.pages
                        ]

                    return (
                        extraction.full_text or "",
                        pages,
                        self.force_extractor,
                        extraction.warnings,
                        [],  # page_classifications
                        {},  # pages_by_extractor
                    )

            # In knowledge_base mode, prefer Docling, fall back to hybrid
            if self.mode == IngestMode.KNOWLEDGE_BASE:
                try:
                    if "docling" in EXTRACTORS:
                        return self._extract_with_docling(file_path)
                except Exception as e:
                    logger.warning(f"Docling failed, falling back to hybrid: {e}")
                return self._extract_with_hybrid(file_path)

            # Use standard Readd pipeline (session mode)
            result = process_document(file_path)

            # Build pages list
            pages = None
            if result.extraction_result and result.extraction_result.pages:
                pages = [
                    {
                        "page_num": p.page_num,
                        "text": p.text,
                        "content_type": p.metadata.get("content_type", "prose"),
                    }
                    for p in result.extraction_result.pages
                ]

            return (
                result.text or "",
                pages,
                result.final_extractor,
                result.warnings,
                [],  # page_classifications
                {},  # pages_by_extractor
            )

        except ImportError:
            logger.warning("Readd not available, using basic extraction")
            text, pages = self._extract_basic(file_path)
            return text, pages, "basic_fallback", ["Readd not available"], [], {}

    def _extract_with_docling(self, file_path: Path):
        """Extract using Docling for knowledge_base mode."""
        from tools.readd.extractors import get_extractor

        extractor = get_extractor("docling")
        result = extractor.extract(file_path)

        pages = None
        if result.pages:
            pages = [
                {
                    "page_num": p.page_num,
                    "text": p.text,
                    "content_type": p.metadata.get("content_type", "prose"),
                }
                for p in result.pages
            ]

        figure_pages = result.metadata.get("figure_pages", [])
        pages_by_extractor = {"docling": result.page_count}
        if figure_pages:
            pages_by_extractor["figure_pages_detected"] = len(figure_pages)

        return (
            result.full_text or "",
            pages,
            "docling",
            result.warnings,
            [],  # page_classifications
            pages_by_extractor,
        )

    def _extract_with_hybrid(self, file_path: Path):
        """Extract using HybridExtractor for per-page vision routing."""
        try:
            from tools.readd import HybridExtractor, ExtractionMode

            # Map ingest mode to extraction mode
            extraction_mode = (
                ExtractionMode.KNOWLEDGE_BASE if self.mode == IngestMode.KNOWLEDGE_BASE else ExtractionMode.SESSION
            )

            extractor = HybridExtractor(mode=extraction_mode)
            result = extractor.extract(file_path)

            pages = []
            if result.pages:
                for p in result.pages:
                    content_type = p.metadata.get("content_type", "prose")

                    if content_type == "chart_data" and p.metadata.get("chart_json"):
                        # Dual-content strategy: chart pages produce TWO entries
                        # 1. JSON chunk for recreation
                        pages.append({
                            "page_num": p.page_num,
                            "text": p.metadata["chart_json"],
                            "content_type": "chart_data",
                        })
                        # 2. Searchable text summary for vector search
                        pages.append({
                            "page_num": p.page_num,
                            "text": p.text,
                            "content_type": "prose",
                        })
                    else:
                        # Non-chart pages: single entry as before
                        pages.append({
                            "page_num": p.page_num,
                            "text": p.text,
                            "content_type": content_type,
                        })

            return (
                result.full_text or "",
                pages or None,
                result.extractor_used,
                result.warnings,
                result.page_classifications,
                result.pages_by_extractor,
            )

        except ImportError as e:
            logger.warning(f"HybridExtractor not available: {e}, falling back to standard")
            from tools.readd import process_document

            result = process_document(file_path)

            pages = None
            if result.extraction_result and result.extraction_result.pages:
                pages = [{"page_num": p.page_num, "text": p.text} for p in result.extraction_result.pages]

            return (
                result.text or "",
                pages,
                result.final_extractor,
                result.warnings,
                [],
                {},
            )

    def _extract_basic(self, file_path: Path):
        """Basic extraction without Readd"""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            import fitz

            doc = fitz.open(str(file_path))
            pages = []
            for i, page in enumerate(doc, 1):
                text = page.get_text("text")
                pages.append({"page_num": i, "text": text})
            doc.close()

            full_text = "\n\n".join(p["text"] for p in pages)
            return full_text, pages

        elif suffix == ".docx":
            from docx import Document as DocxDocument

            doc = DocxDocument(str(file_path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)
            return text, None

        elif suffix in {".txt", ".md"}:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return text, None

        else:
            return "", None

    def ingest_directory(
        self,
        dir_path: Path,
        topic: str,
    ) -> Generator[IngestResult, None, None]:
        """
        Ingest all documents in a directory.

        Args:
            dir_path: Directory path
            topic: Target topic

        Yields:
            IngestResult for each file
        """
        dir_path = Path(dir_path)

        if not dir_path.exists():
            logger.error(f"Directory not found: {dir_path}")
            return

        files = list(dir_path.rglob("*"))
        pdf_files = [f for f in files if f.suffix.lower() in self.SUPPORTED_EXTENSIONS]

        logger.info(f"Found {len(pdf_files)} documents in {dir_path}")

        for file_path in pdf_files:
            yield self.ingest_file(file_path, topic)

    def ingest_all_topics(self, base_dir: Path) -> Dict[str, List[IngestResult]]:
        """
        Ingest all topics from base directory.

        Expects structure:
            base_dir/
                topic_a/
                topic_b/
                topic_c/

        Args:
            base_dir: Base directory containing topic subdirectories

        Returns:
            Dict mapping topic to list of results
        """
        base_dir = Path(base_dir)
        results = {}

        for topic in self.kb.TOPICS:
            topic_dir = base_dir / topic
            if topic_dir.exists():
                logger.info(f"Ingesting topic: {topic}")
                results[topic] = list(self.ingest_directory(topic_dir, topic))
            else:
                logger.warning(f"Topic directory not found: {topic_dir}")
                results[topic] = []

        return results


def ingest_file(file_path: Path, topic: str) -> IngestResult:
    """Convenience function to ingest a single file"""
    ingester = DocumentIngester()
    return ingester.ingest_file(file_path, topic)


def ingest_directory(dir_path: Path, topic: str) -> List[IngestResult]:
    """Convenience function to ingest a directory"""
    ingester = DocumentIngester()
    return list(ingester.ingest_directory(dir_path, topic))
