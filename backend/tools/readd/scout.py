"""
Readd Scout - Phase 1: Document Quality Assessment

Samples pages, assesses quality, recommends extraction approach.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class QualityTier(Enum):
    """Document quality tiers"""

    HIGH = "high"  # Clean digital PDF, score > 80
    MEDIUM = "medium"  # Mixed quality, score 50-80
    LOW = "low"  # Scanned/messy, score 20-50
    TERRIBLE = "terrible"  # Garbage, needs vision LLM, score < 20


class ExtractorType(Enum):
    """Available extraction tools"""

    PYMUPDF_TEXT = "pymupdf_text"  # Fast, basic text
    PYMUPDF4LLM = "pymupdf4llm"  # Layout-aware markdown
    DOCLING = "docling"  # IBM Docling layout + OCR + tables
    MARKER = "marker"  # ML-based extraction
    OBSERVATIONN = "observationn"  # Qwen3-VL vision extraction
    VISION_OCR = "vision_ocr"  # LLM page-by-page (llava fallback)


@dataclass
class ScoutReport:
    """Results from scouting a document"""

    file_path: str
    page_count: int
    quality_score: int  # 0-100
    quality_tier: QualityTier
    recommended_extractor: ExtractorType

    # Assessment details
    is_scanned: bool
    has_tables: bool
    has_figures: bool
    has_equations: bool
    text_density: float  # chars per page
    garble_ratio: float  # ratio of weird characters

    # Context extracted from scout
    detected_title: Optional[str]
    detected_structure: List[str]  # chapter/section headers found
    sample_text: str  # sample of extracted text

    # Warnings
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "page_count": self.page_count,
            "quality_score": self.quality_score,
            "quality_tier": self.quality_tier.value,
            "recommended_extractor": self.recommended_extractor.value,
            "is_scanned": self.is_scanned,
            "has_tables": self.has_tables,
            "has_figures": self.has_figures,
            "text_density": self.text_density,
            "garble_ratio": self.garble_ratio,
            "detected_title": self.detected_title,
            "detected_structure": self.detected_structure,
            "warnings": self.warnings,
        }


class DocumentScout:
    """
    Scout documents to assess quality before full extraction.

    Samples pages, runs quick extraction, analyzes results.
    """

    # Characters that indicate garbled text
    GARBLE_CHARS = set("�□■●◆▪▫◊○•★☆♦♣♠♥")
    CONTROL_CHARS = set(chr(i) for i in range(32) if chr(i) not in "\n\r\t")

    # Patterns for structure detection
    CHAPTER_PATTERN = re.compile(r"^(?:Chapter|CHAPTER)\s+\d+", re.MULTILINE)
    SECTION_PATTERN = re.compile(r"^\d+\.\d+(?:\.\d+)?\s+[A-Z]", re.MULTILINE)

    def __init__(self, sample_pages: int = 5):
        """
        Args:
            sample_pages: Number of pages to sample for assessment
        """
        self.sample_pages = sample_pages

    def scout(self, file_path: Path) -> ScoutReport:
        """
        Scout a document and return quality assessment.

        Args:
            file_path: Path to PDF file

        Returns:
            ScoutReport with quality assessment and recommendations
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF: {file_path}")

        doc = fitz.open(str(file_path))
        page_count = len(doc)

        # Select pages to sample
        sample_indices = self._select_sample_pages(page_count)

        # Extract sample text
        samples = []
        for idx in sample_indices:
            page = doc[idx]
            text = page.get_text("text")
            samples.append(
                {
                    "page": idx + 1,
                    "text": text,
                    "char_count": len(text),
                    "has_images": len(page.get_images()) > 0,
                }
            )

        # Analyze samples
        analysis = self._analyze_samples(samples)

        # Detect document structure
        all_text = "\n".join(s["text"] for s in samples)
        structure = self._detect_structure(all_text)

        # Extract title
        title = self._extract_title(doc, samples[0]["text"] if samples else "")

        # Calculate quality score
        quality_score = self._calculate_quality_score(analysis)
        quality_tier = self._score_to_tier(quality_score)
        recommended = self._recommend_extractor(quality_tier, analysis)

        # Build warnings
        warnings = self._build_warnings(analysis, quality_score)

        doc.close()

        return ScoutReport(
            file_path=str(file_path),
            page_count=page_count,
            quality_score=quality_score,
            quality_tier=quality_tier,
            recommended_extractor=recommended,
            is_scanned=analysis["is_scanned"],
            has_tables=analysis["has_tables"],
            has_figures=analysis["has_figures"],
            has_equations=analysis["has_equations"],
            text_density=analysis["text_density"],
            garble_ratio=analysis["garble_ratio"],
            detected_title=title,
            detected_structure=structure,
            sample_text=all_text[:2000],
            warnings=warnings,
        )

    def _select_sample_pages(self, total_pages: int) -> List[int]:
        """Select pages to sample - start, middle, end"""
        if total_pages <= self.sample_pages:
            return list(range(total_pages))

        indices = [0]  # First page always

        # Add evenly distributed pages
        step = total_pages // (self.sample_pages - 1)
        for i in range(1, self.sample_pages - 1):
            indices.append(min(i * step, total_pages - 1))

        indices.append(total_pages - 1)  # Last page

        return sorted(set(indices))

    def _analyze_samples(self, samples: List[Dict]) -> Dict[str, Any]:
        """Analyze sampled pages"""
        total_chars = sum(s["char_count"] for s in samples)
        total_pages = len(samples)

        # Calculate text density
        text_density = total_chars / total_pages if total_pages > 0 else 0

        # Check for scanned (low text, high images)
        has_images = any(s["has_images"] for s in samples)
        is_scanned = text_density < 500 and has_images

        # Calculate garble ratio
        all_text = "".join(s["text"] for s in samples)
        garble_count = sum(1 for c in all_text if c in self.GARBLE_CHARS or c in self.CONTROL_CHARS)
        garble_ratio = garble_count / len(all_text) if all_text else 0

        # Detect tables (look for tab characters, aligned columns)
        has_tables = "\t" in all_text or self._detect_table_patterns(all_text)

        # Detect figures (image references, figure captions)
        has_figures = bool(re.search(r"(?:Figure|Fig\.?)\s+\d+", all_text, re.IGNORECASE))

        # Detect equations (common math symbols density)
        math_chars = sum(1 for c in all_text if c in "∫∑∏√∞≈≠≤≥±×÷αβγδεθλμπσφω")
        has_equations = math_chars > 5 or bool(re.search(r"[a-z]\s*=\s*[a-z0-9]", all_text))

        # Check for encoding issues
        encoding_issues = "�" in all_text or "\x00" in all_text

        # Check for repeated headers/footers
        lines = all_text.split("\n")
        line_counts = {}
        for line in lines:
            line = line.strip()
            if len(line) > 10:
                line_counts[line] = line_counts.get(line, 0) + 1
        repeated_lines = sum(1 for count in line_counts.values() if count > 2)

        return {
            "text_density": text_density,
            "garble_ratio": garble_ratio,
            "is_scanned": is_scanned,
            "has_tables": has_tables,
            "has_figures": has_figures,
            "has_equations": has_equations,
            "has_images": has_images,
            "encoding_issues": encoding_issues,
            "repeated_lines": repeated_lines,
        }

    def _detect_table_patterns(self, text: str) -> bool:
        """Detect table-like patterns in text"""
        lines = text.split("\n")

        # Look for lines with multiple number columns
        number_pattern = re.compile(r"\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*")
        table_lines = sum(1 for line in lines if number_pattern.search(line))

        return table_lines > 3

    def _detect_structure(self, text: str) -> List[str]:
        """Detect document structure (chapters, sections)"""
        structure = []

        # Find chapter headings
        chapters = self.CHAPTER_PATTERN.findall(text)
        structure.extend(chapters[:5])

        # Find section headings
        sections = self.SECTION_PATTERN.findall(text)
        structure.extend(sections[:10])

        return structure

    def _extract_title(self, doc: fitz.Document, first_page_text: str) -> Optional[str]:
        """Try to extract document title"""
        # Try PDF metadata first
        metadata_title = doc.metadata.get("title", "").strip()
        if metadata_title and len(metadata_title) > 5:
            return metadata_title

        # Try first lines of first page
        lines = first_page_text.split("\n")
        for line in lines[:15]:
            line = line.strip()
            # Look for title-like lines (not too short, not too long, capitalized)
            if 10 < len(line) < 150:
                # Skip obvious non-titles
                if line.lower().startswith(("page", "chapter", "table", "figure", "copyright")):
                    continue
                if re.match(r"^\d+$", line):  # Just a number
                    continue
                return line

        return None

    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> int:
        """Calculate 0-100 quality score"""
        score = 100

        # Penalize low text density (likely scanned)
        if analysis["text_density"] < 500:
            score -= 40
        elif analysis["text_density"] < 1000:
            score -= 20
        elif analysis["text_density"] < 2000:
            score -= 10

        # Penalize garbled text
        if analysis["garble_ratio"] > 0.1:
            score -= 50
        elif analysis["garble_ratio"] > 0.05:
            score -= 30
        elif analysis["garble_ratio"] > 0.01:
            score -= 15

        # Penalize encoding issues
        if analysis["encoding_issues"]:
            score -= 20

        # Penalize scanned documents
        if analysis["is_scanned"]:
            score -= 25

        # Small penalty for complex content (tables, equations)
        if analysis["has_tables"]:
            score -= 5
        if analysis["has_equations"]:
            score -= 5

        # Penalize repeated lines (header/footer pollution)
        if analysis["repeated_lines"] > 10:
            score -= 15
        elif analysis["repeated_lines"] > 5:
            score -= 8

        return max(0, min(100, score))

    def _score_to_tier(self, score: int) -> QualityTier:
        """Convert score to quality tier"""
        if score > 80:
            return QualityTier.HIGH
        elif score > 50:
            return QualityTier.MEDIUM
        elif score > 20:
            return QualityTier.LOW
        else:
            return QualityTier.TERRIBLE

    def _recommend_extractor(self, tier: QualityTier, analysis: Dict[str, Any]) -> ExtractorType:
        """Recommend extraction tool based on tier and analysis"""
        if tier == QualityTier.HIGH:
            # Clean PDF - basic extraction is fine
            if analysis["has_tables"] or analysis["has_equations"]:
                return ExtractorType.PYMUPDF4LLM  # Better layout handling
            return ExtractorType.PYMUPDF_TEXT

        elif tier == QualityTier.MEDIUM:
            # Mixed quality - need layout awareness
            return ExtractorType.PYMUPDF4LLM

        elif tier == QualityTier.LOW:
            # Poor quality - need ML extraction
            return ExtractorType.MARKER

        else:  # TERRIBLE
            # Garbage - need vision LLM
            return ExtractorType.VISION_OCR

    def _build_warnings(self, analysis: Dict[str, Any], score: int) -> List[str]:
        """Build list of warnings based on analysis"""
        warnings = []

        if analysis["is_scanned"]:
            warnings.append("Document appears to be scanned - OCR may be needed")

        if analysis["garble_ratio"] > 0.05:
            warnings.append(f"High garble ratio ({analysis['garble_ratio']:.1%}) - text encoding issues")

        if analysis["encoding_issues"]:
            warnings.append("Encoding issues detected - may have missing characters")

        if analysis["text_density"] < 500:
            warnings.append("Low text density - document may be image-heavy or poorly digitized")

        if analysis["repeated_lines"] > 5:
            warnings.append("Repeated lines detected - may have header/footer pollution")

        if analysis["has_tables"]:
            warnings.append("Tables detected - verify table extraction quality")

        if analysis["has_equations"]:
            warnings.append("Equations detected - may need manual verification")

        if score < 50:
            warnings.append("Low quality score - extraction may require manual review")

        return warnings


def scout_document(file_path: Path) -> ScoutReport:
    """Convenience function to scout a document"""
    scout = DocumentScout()
    return scout.scout(file_path)
