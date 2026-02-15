"""
Observationn Extractor - Vision-Based Document Understanding

Uses vision model on a dedicated llama-server slot for intelligent
document understanding:
- Tables: Structured extraction with row/column detection
- Diagrams: Description and label extraction
- Equations: LaTeX transcription
- Mixed content: Intelligent content-type detection
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from tools.readd.extractors import ExtractionResult

import fitz  # PyMuPDF

from config import runtime_config
from domain_loader import get_pipeline_config
from utils.llm import get_llm_client

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Analyzed content from a single page."""

    page_num: int
    text: str
    content_types: List[str]  # e.g., ["text", "table", "diagram"]
    confidence: float
    metadata: Dict[str, Any]


class ObservationnExtractor:
    """
    Qwen2.5-VL based vision extraction for documents.

    Specializes in:
    - Multi-column layouts
    - Table structure preservation
    - Technical diagrams and figures
    - Mathematical equations
    """

    name = "observationn"

    # Content-specific prompts
    PROMPTS = {
        "general": """Extract all text from this document page. Preserve the exact formatting, including:
- Paragraph structure
- Headers and subheaders
- Lists (numbered and bulleted)
- Any table data (preserve column alignment)

Output ONLY the extracted text, no commentary.""",
        "table": """This page contains a table. Extract the table data in a clear format:
1. Preserve column headers
2. Align data in columns using | separators
3. Include all rows

Output the table in markdown format.""",
        "technical": """This is a technical/engineering document page. Extract:
1. All text content
2. Labels from any diagrams or figures
3. Values from tables
4. Any equations or formulas

Preserve technical terminology exactly as written.""",
        "equations": """This page contains mathematical equations. Extract:
1. All text content
2. Equations in LaTeX format where possible
3. Variable definitions

Format equations on separate lines.""",
    }

    def __init__(
        self,
        model: str = None,
        dpi: int = 150,
        batch_pages: int = 1,
        num_ctx: int = 8192,
        timeout: int = 60,
    ):
        """
        Initialize Observationn extractor.

        Args:
            model: Vision model to use (from config.model_vision)
            dpi: Resolution for page rendering (higher = better quality, slower)
            batch_pages: Pages to process per request (1 for best quality)
            num_ctx: Context window size (2048 for speed, 8192 for complex pages)
            timeout: Max seconds per page extraction (default 60)
        """
        self.model = model or runtime_config.model_vision
        self.dpi = dpi
        self.batch_pages = batch_pages
        self.num_ctx = num_ctx
        self.timeout = timeout

        logger.info(f"Observationn initialized: model={self.model}, dpi={dpi}, ctx={num_ctx}, timeout={timeout}s")

    def _ensure_vision_server(self):
        """Start vision server if not running. Blocks until healthy or fails."""
        import asyncio
        from utils.llm import get_server_manager

        mgr = get_server_manager()

        # Quick sync health check first
        loop = asyncio.new_event_loop()
        try:
            health = loop.run_until_complete(mgr.health_check("vision"))
            if health.get("status") == "healthy":
                return

            logger.info("Vision server not running — starting for Observationn extraction")
            mgr.start("vision")
            healthy = loop.run_until_complete(mgr._wait_for_healthy("vision", timeout=120))
            if not healthy:
                raise RuntimeError("Vision server failed to start within 120s")
            logger.info("Vision server ready")
        finally:
            loop.close()

    def extract(self, file_path: Path) -> "ExtractionResult":
        """
        Extract text from PDF using vision model.

        Args:
            file_path: Path to PDF file

        Returns:
            ExtractionResult with extracted content
        """
        from tools.readd.extractors import ExtractionResult, ExtractedPage

        file_path = Path(file_path)
        logger.info(f"Observationn: Processing {file_path.name}")

        # Ensure vision server is running before we try to use it
        self._ensure_vision_server()

        # Get vision client (runs on dedicated slot)
        client = get_llm_client("vision")
        doc = fitz.open(str(file_path))

        pages = []
        full_text_parts = []
        warnings = []

        total_chars = 0
        total_words = 0
        empty_pages = 0

        for page_num, page in enumerate(doc, 1):
            # Render page to image
            pix = page.get_pixmap(dpi=self.dpi)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode()

            # Detect content type based on page characteristics
            content_type = self._detect_content_type(page)
            prompt = self._get_prompt(content_type)

            try:
                response = client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
                    options={
                        "num_ctx": self.num_ctx,
                        "temperature": 0.1,  # Low temperature for accuracy
                    },
                )

                text = response["message"]["content"]
                text = self._clean_extracted_text(text)

            except Exception as e:
                logger.error(f"Observationn failed on page {page_num}: {e}")
                text = f"[EXTRACTION FAILED: {e}]"
                warnings.append(f"Page {page_num} extraction failed: {e}")

            # Track extraction stats
            char_count = len(text)
            word_count = len(text.split()) if text else 0
            total_chars += char_count
            total_words += word_count
            if char_count < 50:
                empty_pages += 1

            # Log progress with content stats so user knows if it's working
            status = "OK" if char_count >= 50 else "EMPTY"
            logger.info(
                f"Observationn: Page {page_num}/{len(doc)} - {status} " f"({word_count} words, {char_count} chars)"
            )

            pages.append(
                ExtractedPage(
                    page_num=page_num,
                    text=text,
                    metadata={
                        "extractor": "observationn",
                        "model": self.model,
                        "content_type": content_type,
                    },
                )
            )
            full_text_parts.append(text)

        doc.close()

        # Log summary
        logger.info(
            f"Observationn complete: {len(pages)} pages, {total_words} words, "
            f"{total_chars} chars ({empty_pages} empty pages)"
        )
        if empty_pages > len(pages) * 0.5:
            logger.warning(
                f"Observationn: {empty_pages}/{len(pages)} pages were empty - "
                "document may be scanned images that failed to OCR"
            )

        metadata = {
            "extractor": "observationn",
            "model": self.model,
            "dpi": self.dpi,
            "total_words": total_words,
            "total_chars": total_chars,
            "empty_pages": empty_pages,
        }

        return ExtractionResult(
            file_path=str(file_path),
            extractor_used=self.name,
            page_count=len(pages),
            pages=pages,
            full_text="\n\n".join(full_text_parts),
            metadata=metadata,
            warnings=warnings or ["Observationn extraction complete"],
        )

    def _detect_content_type(self, page: fitz.Page) -> str:
        """
        Detect the primary content type on a page.

        Args:
            page: PyMuPDF page object

        Returns:
            Content type string: "general", "table", "technical", "equations"
        """
        text = page.get_text("text")
        text_lower = text.lower() if text else ""

        # Check for tables (tab characters, aligned columns)
        has_tabs = "\t" in text
        has_table_pattern = bool(re.search(r"\d+\s+\d+\s+\d+", text))
        has_pipe_separators = "|" in text

        if has_tabs or has_table_pattern or has_pipe_separators:
            return "table"

        # Check for equations (math symbols, LaTeX-like content)
        math_chars = sum(1 for c in text if c in "∫∑∏√∞≈≠≤≥±×÷αβγδεθλμπσφω∂∇")
        equation_pattern = bool(re.search(r"[a-z]\s*=\s*[a-z0-9\+\-\*\/\(\)]+", text_lower))

        if math_chars > 5 or equation_pattern:
            return "equations"

        # Check for technical content (domain-configurable)
        tech_terms = get_pipeline_config().get(
            "observationn_technical_terms",
            ["analysis", "results", "data", "method", "test"],
        )
        if any(term in text_lower for term in tech_terms):
            return "technical"

        return "general"

    def _get_prompt(self, content_type: str) -> str:
        """Get the appropriate prompt for content type."""
        return self.PROMPTS.get(content_type, self.PROMPTS["general"])

    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean up extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove common LLM artifacts
        text = re.sub(r"^(Here is|I can see|The text reads?:?)\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(Let me know if|I hope this helps).*$", "", text, flags=re.IGNORECASE | re.DOTALL)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        return text.strip()

    def extract_page(self, file_path: Path, page_num: int, prompt_type: str = None) -> PageContent:
        """
        Extract content from a single page.

        Args:
            file_path: Path to PDF file
            page_num: Page number (1-indexed)
            prompt_type: Optional override for content type detection.
                         If provided, uses this instead of auto-detecting.
                         Valid values: "general", "table", "technical", "equations"

        Returns:
            PageContent with extracted text and metadata
        """
        client = get_llm_client("vision")
        doc = fitz.open(str(file_path))

        if page_num < 1 or page_num > len(doc):
            doc.close()
            raise ValueError(f"Page {page_num} out of range (1-{len(doc)})")

        page = doc[page_num - 1]

        # Render and detect content type
        pix = page.get_pixmap(dpi=self.dpi)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()

        content_type = prompt_type or self._detect_content_type(page)
        prompt = self._get_prompt(content_type)

        response = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
            options={"num_ctx": self.num_ctx, "temperature": 0.1},
        )

        text = self._clean_extracted_text(response["message"]["content"])

        doc.close()

        return PageContent(
            page_num=page_num,
            text=text,
            content_types=[content_type],
            confidence=0.85,  # Placeholder - could be derived from model confidence
            metadata={"model": self.model},
        )


def extract_with_observationn(file_path: Path, model: str = None) -> "ExtractionResult":
    """Convenience function for Observationn extraction."""
    extractor = ObservationnExtractor(model=model)
    return extractor.extract(file_path)
