"""
Page Classifier - Page-level content detection for hybrid extraction

Classifies individual PDF pages into content types to route them
to the most appropriate extractor (text vs vision).

Uses density-relative heuristics: compares each page's text density
against document-level percentiles rather than regex pattern matching.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Tuple

import fitz  # PyMuPDF

from domain_loader import get_pipeline_config

logger = logging.getLogger(__name__)


class PageType(Enum):
    """Classification of page content complexity."""

    SIMPLE_TEXT = "simple_text"  # Pure text, can use fast text extraction
    STRUCTURED = "structured"  # Tables, columns - needs layout-aware extraction
    VISUAL = "visual"  # Diagrams, figures, charts - needs vision extraction
    SCANNED = "scanned"  # Scanned/image-based - needs vision OCR


@dataclass
class PageClassification:
    """Classification result for a single page."""

    page_num: int
    page_type: PageType
    confidence: float  # 0.0 to 1.0

    # Detection details
    has_tables: bool
    has_figures: bool
    has_equations: bool
    has_charts: bool
    is_scanned: bool

    # Metrics
    text_density: float  # chars per page
    image_coverage: float  # percentage of page covered by images

    # Additional info
    detected_elements: List[str]  # e.g., ["table", "drawings"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_num": self.page_num,
            "page_type": self.page_type.value,
            "confidence": self.confidence,
            "has_tables": self.has_tables,
            "has_figures": self.has_figures,
            "has_equations": self.has_equations,
            "has_charts": self.has_charts,
            "is_scanned": self.is_scanned,
            "text_density": self.text_density,
            "image_coverage": self.image_coverage,
            "detected_elements": self.detected_elements,
        }


class PageClassifier:
    """
    Classify PDF pages for optimal extraction routing.

    Uses density-relative heuristics: profiles the entire document first,
    then classifies each page by comparing its text density against
    document-level percentiles. Vector drawing detection replaces regex
    pattern matching for charts/figures.
    """

    # Patterns for detecting tables
    TABLE_PATTERNS = [
        r"\|\s*\w+\s*\|",  # Pipe separators
        r"\t\w+\t",  # Tab separators
        r"\d+\.\d+\s+\d+\.\d+\s+\d+",  # Aligned numbers (common in tables)
    ]

    # Generic technical terms (domain packs add specialized terms via lexicon)
    _BASE_TECHNICAL_TERMS = {
        "analysis",
        "report",
        "method",
        "test",
        "data",
        "result",
        "standard",
        "specification",
        "modulus",
        "coefficient",
        "factor",
        "angle",
        "depth",
        "layer",
        "pressure",
        "stress",
        "strain",
    }

    def __init__(
        self,
        scanned_threshold: float = 500,  # chars/page below which considered scanned
        image_coverage_threshold: float = 0.3,  # 30% image coverage = visual page
        table_confidence_threshold: float = 0.8,
    ):
        """
        Args:
            scanned_threshold: Text density below which page is considered scanned
            image_coverage_threshold: Image coverage above which page is visual
            table_confidence_threshold: Confidence above which table detection triggers
        """
        self.scanned_threshold = scanned_threshold
        self.image_coverage_threshold = image_coverage_threshold
        self.table_confidence_threshold = table_confidence_threshold

        # Build technical terms from base set + domain pipeline config
        domain_terms = get_pipeline_config().get("readd_technical_terms", set())
        self.technical_terms = set(self._BASE_TECHNICAL_TERMS)
        self.technical_terms.update(t.lower() for t in domain_terms)

        # Compile patterns (tables only — figures/charts/equations removed)
        self.table_patterns = [re.compile(p, re.IGNORECASE) for p in self.TABLE_PATTERNS]

    def profile_document(self, file_path) -> Dict:
        """Calculate per-document text density statistics for adaptive classification."""
        from pathlib import Path
        file_path = Path(file_path)
        doc = fitz.open(str(file_path))
        densities = []
        for i in range(len(doc)):
            text = doc[i].get_text("text") or ""
            densities.append(len(text))
        doc.close()

        n = len(densities)
        if n == 0:
            return {"p5": 0, "p15": 0, "p50": 0, "mean": 0, "is_scanned": True, "densities": []}

        densities_sorted = sorted(densities)
        return {
            "p5": densities_sorted[max(0, int(n * 0.05))],
            "p15": densities_sorted[max(0, int(n * 0.15))],
            "p50": densities_sorted[n // 2],
            "mean": sum(densities) / n,
            "is_scanned": densities_sorted[n // 4] < 100,
            "densities": densities,  # per-page, 0-indexed
        }

    def _has_significant_drawings(self, page: fitz.Page) -> bool:
        """Check if page has vector drawings (charts, diagrams).

        Charts have many short line/curve segments (axes, grid lines, data curves).
        Simple decorative lines or borders are few.
        """
        try:
            drawings = page.get_drawings()
        except Exception:
            return False
        if not drawings:
            return False
        # Count vector elements: lines (l), curves (c), rectangles (re)
        line_count = sum(
            1 for d in drawings for item in d["items"]
            if item[0] in ("l", "c", "re")
        )
        return line_count > 20  # 20+ vector elements suggests a chart/diagram

    def classify_document(self, file_path) -> Tuple[List[PageClassification], Dict]:
        """
        Classify all pages in a document using density-aware heuristics.

        Returns:
            Tuple of (classifications list, density_profile dict)
        """
        from pathlib import Path
        file_path = Path(file_path)

        # Stage 1: Document-level density profiling
        profile = self.profile_document(file_path)

        doc = fitz.open(str(file_path))
        classifications = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            page_density = profile["densities"][page_idx] if page_idx < len(profile["densities"]) else 0
            classification = self._classify_page_with_profile(page, page_num, page_density, profile)
            classifications.append(classification)

        doc.close()

        # Log summary
        type_counts = {}
        for c in classifications:
            type_counts[c.page_type.value] = type_counts.get(c.page_type.value, 0) + 1

        logger.info(
            f"Classified {len(classifications)} pages "
            f"(density P5={profile['p5']}, P50={profile['p50']}): "
            f"{type_counts.get('simple_text', 0)} text, "
            f"{type_counts.get('structured', 0)} structured, "
            f"{type_counts.get('visual', 0)} visual, "
            f"{type_counts.get('scanned', 0)} scanned"
        )

        return classifications, profile

    def _classify_page_with_profile(
        self, page: fitz.Page, page_num: int, page_density: float, profile: Dict
    ) -> PageClassification:
        """Classify a single page using document-level density context."""
        text = page.get_text("text")
        image_coverage = self._calculate_image_coverage(page)

        # Table detection (kept, but with higher threshold)
        has_tables, table_confidence = self._detect_tables(page, text or "")

        # Check for vector drawings (replaces regex chart detection)
        has_drawings = self._has_significant_drawings(page)

        # Scanned document check (document-level)
        is_scanned = profile["is_scanned"]

        # Determine page type using density-relative thresholds
        if is_scanned and page_density < 100:
            page_type = PageType.SCANNED
            confidence = 0.9
        elif page_density >= profile["p50"]:
            # Above median density = text-dominant, PyMuPDF handles it
            page_type = PageType.SIMPLE_TEXT
            confidence = 0.9
        elif page_density < max(profile["p5"], 50) and page_density < 200:
            # Very sparse page (blank, copyright, divider) — nothing to extract visually
            page_type = PageType.SIMPLE_TEXT
            confidence = 0.8
        elif page_density < profile["p15"]:
            # Low density relative to document — potential visual content
            if has_drawings:
                page_type = PageType.VISUAL
                confidence = 0.85
            else:
                # Sparse text but no vector art (index, ToC, etc.)
                page_type = PageType.SIMPLE_TEXT
                confidence = 0.7
        elif has_tables and table_confidence > 0.8:
            # Mid-density with strong table signal
            page_type = PageType.STRUCTURED
            confidence = table_confidence
        elif has_drawings and image_coverage > 0.2:
            # Mid-density with both drawings and images — likely a figure page
            page_type = PageType.VISUAL
            confidence = 0.75
        else:
            # Default: text extraction handles it
            page_type = PageType.SIMPLE_TEXT
            confidence = 0.85

        # Build detected elements list
        detected_elements = []
        if has_tables:
            detected_elements.append("table")
        if has_drawings:
            detected_elements.append("drawings")

        return PageClassification(
            page_num=page_num,
            page_type=page_type,
            confidence=confidence,
            has_tables=has_tables,
            has_figures=has_drawings,  # repurpose: now means "has vector drawings"
            has_equations=False,  # no longer detected (PyMuPDF handles equation text fine)
            has_charts=has_drawings,  # repurpose: now means "has vector drawings"
            is_scanned=is_scanned and page_density < 100,
            text_density=page_density,
            image_coverage=image_coverage,
            detected_elements=detected_elements,
        )

    def classify_page(self, page: fitz.Page, page_num: int) -> PageClassification:
        """Classify a single page (without document-level context).

        For best results, use classify_document() which provides density profiling.
        """
        text = page.get_text("text")
        text_density = len(text) if text else 0
        image_coverage = self._calculate_image_coverage(page)
        has_tables, table_confidence = self._detect_tables(page, text or "")
        has_drawings = self._has_significant_drawings(page)
        is_scanned = self._is_scanned(page, text_density, image_coverage)

        if is_scanned:
            page_type, confidence = PageType.SCANNED, 0.9
        elif has_drawings and image_coverage > 0.2:
            page_type, confidence = PageType.VISUAL, 0.8
        elif has_tables and table_confidence > 0.8:
            page_type, confidence = PageType.STRUCTURED, table_confidence
        else:
            page_type, confidence = PageType.SIMPLE_TEXT, 0.9

        detected_elements = []
        if has_tables:
            detected_elements.append("table")
        if has_drawings:
            detected_elements.append("drawings")

        return PageClassification(
            page_num=page_num,
            page_type=page_type,
            confidence=confidence,
            has_tables=has_tables,
            has_figures=has_drawings,
            has_equations=False,
            has_charts=has_drawings,
            is_scanned=is_scanned,
            text_density=text_density,
            image_coverage=image_coverage,
            detected_elements=detected_elements,
        )

    def _calculate_image_coverage(self, page: fitz.Page) -> float:
        """Calculate what percentage of the page is covered by images."""
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        if page_area == 0:
            return 0.0

        image_area = 0
        for img in page.get_images():
            try:
                xref = img[0]
                # Get image bounding boxes
                img_rects = page.get_image_rects(xref)
                for rect in img_rects:
                    image_area += rect.width * rect.height
            except Exception:
                continue

        return min(1.0, image_area / page_area)

    def _detect_tables(self, page: fitz.Page, text: str) -> Tuple[bool, float]:
        """
        Detect if page contains tables.

        Returns:
            Tuple of (has_tables, confidence)
        """
        confidence = 0.0

        # Check for tab characters
        if "\t" in text:
            confidence += 0.3

        # Check for pipe separators (markdown tables)
        if "|" in text:
            pipe_lines = sum(1 for line in text.split("\n") if line.count("|") >= 2)
            if pipe_lines >= 2:
                confidence += 0.4

        # Check patterns
        for pattern in self.table_patterns:
            if pattern.search(text):
                confidence += 0.2

        # Check for aligned columns using PyMuPDF blocks
        try:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            # Look for multiple text blocks at similar x positions (column alignment)
            x_positions = []
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    x_positions.append(round(block["bbox"][0], -1))  # Round to 10px

            # If many blocks share x positions, likely a table
            from collections import Counter

            x_counts = Counter(x_positions)
            if x_counts and max(x_counts.values()) >= 3:
                confidence += 0.3
        except Exception:
            pass

        return confidence >= self.table_confidence_threshold, min(1.0, confidence)

    def _is_scanned(self, page: fitz.Page, text_density: float, image_coverage: float) -> bool:
        """Determine if page is scanned (image-based, not digital text)."""
        # Low text but high image coverage = scanned
        if text_density < self.scanned_threshold and image_coverage > 0.5:
            return True

        # Very low text density with any images = likely scanned
        if text_density < 100 and len(page.get_images()) > 0:
            return True

        return False


def classify_document(file_path) -> Tuple[List[PageClassification], Dict]:
    """Convenience function to classify all pages in a document."""
    classifier = PageClassifier()
    return classifier.classify_document(file_path)
