"""
Readd Extractors - Phase 2: Document Extraction Tools

Multiple extraction backends for different quality levels.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Generator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import fitz  # PyMuPDF

from config import runtime_config

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """A single extracted page"""

    page_num: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result from document extraction"""

    file_path: str
    extractor_used: str
    page_count: int
    pages: List[ExtractedPage]
    full_text: str
    metadata: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    def get_text(self) -> str:
        """Get full document text"""
        return self.full_text

    def iter_pages(self) -> Generator[ExtractedPage, None, None]:
        """Iterate over pages"""
        for page in self.pages:
            yield page


class BaseExtractor(ABC):
    """Base class for document extractors"""

    name: str = "base"

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractionResult:
        """Extract text from document"""
        pass

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Fix hyphenation at line breaks
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # Remove null characters
        text = text.replace("\x00", "")
        return text.strip()


class PyMuPDFTextExtractor(BaseExtractor):
    """
    Fast basic text extraction using PyMuPDF.
    Best for: Clean digital PDFs with simple layouts.
    """

    name = "pymupdf_text"

    def extract(self, file_path: Path) -> ExtractionResult:
        file_path = Path(file_path)
        doc = fitz.open(str(file_path))

        pages = []
        full_text_parts = []
        warnings = []

        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            text = self._clean_text(text)

            pages.append(ExtractedPage(page_num=page_num, text=text, metadata={"char_count": len(text)}))
            full_text_parts.append(text)

        # Check for potential issues
        total_chars = sum(len(p.text) for p in pages)
        if total_chars < 100 * len(pages):
            warnings.append("Low text yield - document may be scanned or image-based")

        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "page_count": len(pages),
            "total_chars": total_chars,
        }

        doc.close()

        return ExtractionResult(
            file_path=str(file_path),
            extractor_used=self.name,
            page_count=len(pages),
            pages=pages,
            full_text="\n\n".join(full_text_parts),
            metadata=metadata,
            warnings=warnings,
        )


class PyMuPDF4LLMExtractor(BaseExtractor):
    """
    Layout-aware extraction with markdown formatting.
    Best for: Documents with tables, columns, complex layouts.
    """

    name = "pymupdf4llm"

    def extract(self, file_path: Path) -> ExtractionResult:
        file_path = Path(file_path)

        try:
            import pymupdf4llm
        except ImportError:
            logger.warning("pymupdf4llm not installed, falling back to basic extraction")
            return PyMuPDFTextExtractor().extract(file_path)

        # Extract to markdown
        md_text = pymupdf4llm.to_markdown(str(file_path))

        # Also get page-by-page for metadata
        doc = fitz.open(str(file_path))
        pages = []

        # Split markdown by page markers if present, otherwise approximate
        page_texts = self._split_by_pages(md_text, len(doc))

        for page_num, text in enumerate(page_texts, 1):
            text = self._clean_text(text)
            pages.append(ExtractedPage(page_num=page_num, text=text, metadata={"format": "markdown"}))

        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "page_count": len(doc),
            "format": "markdown",
        }

        doc.close()

        return ExtractionResult(
            file_path=str(file_path),
            extractor_used=self.name,
            page_count=len(pages),
            pages=pages,
            full_text=md_text,
            metadata=metadata,
        )

    def _split_by_pages(self, text: str, page_count: int) -> List[str]:
        """Split markdown text into approximate pages"""
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


class MarkerExtractor(BaseExtractor):
    """
    ML-based extraction using Marker.
    Best for: Scanned PDFs, poor quality documents.
    Slower but more accurate for difficult documents.
    """

    name = "marker"

    def extract(self, file_path: Path) -> ExtractionResult:
        file_path = Path(file_path)

        try:
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models
        except ImportError:
            logger.warning("marker-pdf not installed, falling back to pymupdf4llm")
            return PyMuPDF4LLMExtractor().extract(file_path)

        # Load models (cached after first load)
        models = load_all_models()

        # Convert PDF
        full_text, images, metadata = convert_single_pdf(
            str(file_path),
            models,
        )

        # Get page count from original PDF
        doc = fitz.open(str(file_path))
        page_count = len(doc)
        doc.close()

        # Split into pages (marker doesn't preserve page boundaries well)
        page_texts = self._approximate_pages(full_text, page_count)

        pages = []
        for page_num, text in enumerate(page_texts, 1):
            pages.append(
                ExtractedPage(page_num=page_num, text=text, metadata={"format": "markdown", "ml_extracted": True})
            )

        return ExtractionResult(
            file_path=str(file_path),
            extractor_used=self.name,
            page_count=page_count,
            pages=pages,
            full_text=full_text,
            metadata={"format": "markdown", "ml_extracted": True},
        )

    def _approximate_pages(self, text: str, page_count: int) -> List[str]:
        """Split text into approximate pages"""
        if not text or page_count == 0:
            return [""]

        # Try to split at paragraph boundaries
        paragraphs = text.split("\n\n")
        paras_per_page = max(1, len(paragraphs) // page_count)

        pages = []
        for i in range(page_count):
            start = i * paras_per_page
            end = start + paras_per_page if i < page_count - 1 else len(paragraphs)
            page_text = "\n\n".join(paragraphs[start:end])
            pages.append(page_text)

        return pages


try:
    from docling.document_converter import DocumentConverter

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False


class DoclingExtractor(BaseExtractor):
    """
    IBM Docling-based extraction with layout analysis, OCR, and table extraction.
    Best for: Documents with complex tables, mixed layouts, scanned content.
    Replaces Marker for most use cases with better table and figure handling.
    """

    name = "docling"

    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: Accelerator device for Docling layout/table models.
                    "cpu" (default) — safe for batch ingest alongside ONNX embedder.
                    "auto" — let Docling pick (uses CUDA if available, risks VRAM conflicts).
        """
        self._device = device

    def extract(self, file_path: Path) -> ExtractionResult:
        file_path = Path(file_path)

        if not DOCLING_AVAILABLE:
            logger.warning("docling not installed, falling back to pymupdf4llm")
            return PyMuPDF4LLMExtractor().extract(file_path)

        # Lazy import — docling is heavy
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
        from docling.datamodel.base_models import InputFormat

        # Force CPU to avoid VRAM conflicts with ONNX embedder during batch ingest
        pipeline_opts = PdfPipelineOptions()
        device_map = {"cpu": AcceleratorDevice.CPU, "cuda": AcceleratorDevice.CUDA, "auto": AcceleratorDevice.AUTO}
        pipeline_opts.accelerator_options = AcceleratorOptions(device=device_map.get(self._device, AcceleratorDevice.CPU))

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
        )
        result = converter.convert(str(file_path))
        doc = result.document

        # Get full markdown text
        full_text = doc.export_to_markdown()

        # Get page count + dimensions from PyMuPDF for reliable count and area calculations
        pdf_doc = fitz.open(str(file_path))
        page_count = len(pdf_doc)
        page_dimensions = {}
        for i in range(page_count):
            rect = pdf_doc[i].rect
            page_dimensions[i + 1] = {"width": rect.width, "height": rect.height}
        pdf_doc.close()

        # Detect figure/chart pages
        figure_pages = []
        figure_details = []
        try:
            from docling.datamodel.document import PictureItem

            for item, _level in doc.iterate_items():
                if isinstance(item, PictureItem):
                    for prov in getattr(item, "prov", []):
                        page_no = getattr(prov, "page_no", None)
                        if page_no is not None:
                            figure_pages.append(page_no)
                            bbox = getattr(prov, "bbox", None)
                            figure_details.append({
                                "page": page_no,
                                "bbox": [bbox.l, bbox.t, bbox.r, bbox.b] if bbox else None,
                            })
        except Exception as e:
            logger.debug(f"Docling figure detection skipped: {e}")

        figure_pages = sorted(set(figure_pages))

        # Build per-page extraction
        pages = self._extract_pages(doc, full_text, page_count)

        warnings = []
        if figure_pages:
            warnings.append(f"Figures detected on pages: {figure_pages}")

        metadata = {
            "format": "markdown",
            "docling_extracted": True,
            "page_count": page_count,
            "page_dimensions": page_dimensions,
            "figure_pages": figure_pages,
            "figure_count": len(figure_details),
            "figure_details": figure_details,
        }

        return ExtractionResult(
            file_path=str(file_path),
            extractor_used=self.name,
            page_count=page_count,
            pages=pages,
            full_text=full_text,
            metadata=metadata,
            warnings=warnings,
        )

    def _extract_pages(
        self, doc: Any, full_text: str, page_count: int
    ) -> List[ExtractedPage]:
        """Extract per-page text from Docling document model."""
        pages = []

        try:
            # Try to build per-page text from iterate_items provenance
            page_texts: Dict[int, List[str]] = {i: [] for i in range(1, page_count + 1)}

            for item, _level in doc.iterate_items():
                text = getattr(item, "text", None) or ""
                if not text.strip():
                    continue
                for prov in getattr(item, "prov", []):
                    page_no = getattr(prov, "page_no", None)
                    if page_no is not None and page_no in page_texts:
                        page_texts[page_no].append(text)
                        break

            # Check if we got meaningful per-page content
            populated = sum(1 for texts in page_texts.values() if texts)
            if populated >= page_count * 0.5:
                for page_num in range(1, page_count + 1):
                    text = "\n\n".join(page_texts[page_num])
                    text = self._clean_text(text)
                    pages.append(ExtractedPage(
                        page_num=page_num,
                        text=text,
                        metadata={"format": "markdown", "docling_extracted": True},
                    ))
                return pages
        except Exception as e:
            logger.debug(f"Docling per-page extraction failed, using fallback: {e}")

        # Fallback: split full markdown by page count
        if not full_text or page_count == 0:
            return [ExtractedPage(page_num=1, text="", metadata={})]

        chars_per_page = len(full_text) // page_count
        for i in range(page_count):
            start = i * chars_per_page
            end = start + chars_per_page if i < page_count - 1 else len(full_text)
            text = self._clean_text(full_text[start:end])
            pages.append(ExtractedPage(
                page_num=i + 1,
                text=text,
                metadata={"format": "markdown", "docling_extracted": True, "split_fallback": True},
            ))

        return pages


class VisionOCRExtractor(BaseExtractor):
    """
    Vision LLM-based extraction for terrible quality documents.
    Uses vision model to read pages as images.
    Expensive but handles anything.
    """

    name = "vision_ocr"

    def __init__(self, model: str = None):
        self.model = model or runtime_config.model_vision

    def extract(self, file_path: Path) -> ExtractionResult:
        file_path = Path(file_path)

        import base64
        from utils.llm import get_llm_client

        client = get_llm_client("vision")
        doc = fitz.open(str(file_path))

        pages = []
        full_text_parts = []

        for page_num, page in enumerate(doc, 1):
            logger.info(f"Vision OCR: Processing page {page_num}/{len(doc)}")

            # Render page to image
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode()

            # Send to vision model
            try:
                response = client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": "Extract all text from this page. Preserve formatting, tables, and structure as much as possible. Only output the extracted text, no commentary.",
                            "images": [img_b64],
                        }
                    ],
                    options={"num_ctx": 4096},
                )

                text = response["message"]["content"]
                text = self._clean_text(text)

            except Exception as e:
                logger.error(f"Vision OCR failed on page {page_num}: {e}")
                text = f"[OCR FAILED: {e}]"

            pages.append(ExtractedPage(page_num=page_num, text=text, metadata={"ocr": True, "model": self.model}))
            full_text_parts.append(text)

        doc.close()

        return ExtractionResult(
            file_path=str(file_path),
            extractor_used=self.name,
            page_count=len(pages),
            pages=pages,
            full_text="\n\n".join(full_text_parts),
            metadata={"ocr": True, "model": self.model},
            warnings=["Vision OCR used - verify extraction quality"],
        )


# Import Observationn extractor
try:
    from tools.observationn import ObservationnExtractor

    OBSERVATIONN_AVAILABLE = True
except ImportError:
    OBSERVATIONN_AVAILABLE = False
    ObservationnExtractor = None


# Extractor registry
EXTRACTORS = {
    "pymupdf_text": PyMuPDFTextExtractor,
    "pymupdf4llm": PyMuPDF4LLMExtractor,
    "marker": MarkerExtractor,
    "vision_ocr": VisionOCRExtractor,
}

# Add Docling if available
if DOCLING_AVAILABLE:
    EXTRACTORS["docling"] = DoclingExtractor

# Add Observationn if available
if OBSERVATIONN_AVAILABLE:
    EXTRACTORS["observationn"] = ObservationnExtractor


def get_extractor(name: str) -> BaseExtractor:
    """Get extractor by name"""
    if name not in EXTRACTORS:
        raise ValueError(f"Unknown extractor: {name}. Available: {list(EXTRACTORS.keys())}")
    return EXTRACTORS[name]()


def extract_document(file_path: Path, extractor: str = "pymupdf_text") -> ExtractionResult:
    """Convenience function to extract a document"""
    ext = get_extractor(extractor)
    return ext.extract(file_path)
