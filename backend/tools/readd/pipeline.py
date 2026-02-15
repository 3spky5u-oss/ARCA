"""
Readd Pipeline - Intelligent Document Extraction Orchestrator

Coordinates Scout → Extract → QA phases with automatic escalation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .scout import DocumentScout, ScoutReport, ExtractorType
from .extractors import (
    ExtractionResult,
    get_extractor,
    PyMuPDFTextExtractor,
    PyMuPDF4LLMExtractor,
)
from .qa import ExtractionQA, QAReport

logger = logging.getLogger(__name__)


# Escalation path - from cheap/fast to expensive/thorough
ESCALATION_PATH = [
    ExtractorType.PYMUPDF_TEXT,
    ExtractorType.PYMUPDF4LLM,
    ExtractorType.DOCLING,       # Replaces MARKER + OBSERVATIONN for text/tables
    ExtractorType.OBSERVATIONN,  # Fallback for pure vision needs
    ExtractorType.VISION_OCR,    # Last resort
]


@dataclass
class PipelineResult:
    """Complete result from Readd pipeline"""

    file_path: str
    success: bool

    # Phase results
    scout_report: ScoutReport
    extraction_result: Optional[ExtractionResult]
    qa_report: Optional[QAReport]

    # Pipeline metadata
    extractors_tried: List[str]
    final_extractor: Optional[str]
    escalation_count: int
    processing_time_ms: int

    # Output
    text: Optional[str]
    page_count: int
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "success": self.success,
            "scout": self.scout_report.to_dict(),
            "qa": self.qa_report.to_dict() if self.qa_report else None,
            "extractors_tried": self.extractors_tried,
            "final_extractor": self.final_extractor,
            "escalation_count": self.escalation_count,
            "processing_time_ms": self.processing_time_ms,
            "page_count": self.page_count,
            "text_length": len(self.text) if self.text else 0,
            "warnings": self.warnings,
        }


class ReaddPipeline:
    """
    Intelligent document extraction pipeline.

    Workflow:
    1. Scout: Assess document quality, recommend extractor
    2. Extract: Run recommended extractor
    3. QA: Validate results
    4. Escalate: If QA fails, try next extractor tier
    5. Repeat until success or all extractors exhausted
    """

    MAX_ESCALATIONS = 3  # Don't try more than 3 extractors

    def __init__(
        self,
        auto_escalate: bool = True,
        max_escalations: int = None,
        qa_threshold: int = 55,
    ):
        """
        Args:
            auto_escalate: Whether to automatically try better extractors on failure
            max_escalations: Maximum number of escalation attempts
            qa_threshold: QA weirdness score threshold for escalation
        """
        self.auto_escalate = auto_escalate
        self.max_escalations = max_escalations or self.MAX_ESCALATIONS
        self.qa_threshold = qa_threshold

        self.scout = DocumentScout()
        self.qa = ExtractionQA(escalation_threshold=qa_threshold)

    def process(self, file_path: Path) -> PipelineResult:
        """
        Process a document through the full pipeline.

        Args:
            file_path: Path to PDF file

        Returns:
            PipelineResult with extraction results and metadata
        """
        import time

        start_time = time.time()

        file_path = Path(file_path)
        extractors_tried = []
        warnings = []

        # Phase 1: Scout
        logger.info(f"Readd: Scouting {file_path.name}")
        try:
            scout_report = self.scout.scout(file_path)
            logger.info(
                f"Scout result: quality={scout_report.quality_score}, "
                f"tier={scout_report.quality_tier.value}, "
                f"recommended={scout_report.recommended_extractor.value}"
            )
            warnings.extend(scout_report.warnings)
        except Exception as e:
            logger.error(f"Scout failed: {e}")
            return PipelineResult(
                file_path=str(file_path),
                success=False,
                scout_report=None,
                extraction_result=None,
                qa_report=None,
                extractors_tried=[],
                final_extractor=None,
                escalation_count=0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                text=None,
                page_count=0,
                warnings=[f"Scout failed: {e}"],
            )

        # Phase 2 & 3: Extract + QA (with escalation)
        current_extractor = scout_report.recommended_extractor
        extraction_result = None
        qa_report = None
        escalation_count = 0

        while escalation_count <= self.max_escalations:
            extractor_name = current_extractor.value

            # Skip if we've already tried this extractor
            if extractor_name in extractors_tried:
                current_extractor = self._get_next_extractor(current_extractor)
                if current_extractor is None:
                    break
                continue

            extractors_tried.append(extractor_name)

            # Phase 2: Extract
            logger.info(f"Readd: Extracting with {extractor_name}")
            try:
                extractor = get_extractor(extractor_name)
                extraction_result = extractor.extract(file_path)
                warnings.extend(extraction_result.warnings)
            except Exception as e:
                logger.error(f"Extraction failed with {extractor_name}: {e}")
                warnings.append(f"Extractor {extractor_name} failed: {e}")

                # Try next extractor
                if self.auto_escalate:
                    current_extractor = self._get_next_extractor(current_extractor)
                    if current_extractor:
                        escalation_count += 1
                        continue
                break

            # Phase 3: QA
            logger.info("Readd: Running QA")
            qa_report = self.qa.check(extraction_result)
            logger.info(
                f"QA result: weirdness={qa_report.weirdness_score}, "
                f"passed={qa_report.passed}, "
                f"issues={len(qa_report.issues)}"
            )

            # Check if we should escalate
            if qa_report.should_escalate and self.auto_escalate:
                logger.info(f"QA recommends escalation: {qa_report.escalation_reason}")
                warnings.append(f"Escalating from {extractor_name}: {qa_report.escalation_reason}")

                current_extractor = self._get_next_extractor(current_extractor)
                if current_extractor:
                    escalation_count += 1
                    continue
                else:
                    logger.warning("No more extractors to try")
                    break
            else:
                # QA passed or escalation disabled
                break

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Build final result
        success = extraction_result is not None and (qa_report is None or qa_report.passed)

        return PipelineResult(
            file_path=str(file_path),
            success=success,
            scout_report=scout_report,
            extraction_result=extraction_result,
            qa_report=qa_report,
            extractors_tried=extractors_tried,
            final_extractor=extractors_tried[-1] if extractors_tried else None,
            escalation_count=escalation_count,
            processing_time_ms=elapsed_ms,
            text=extraction_result.full_text if extraction_result else None,
            page_count=extraction_result.page_count if extraction_result else 0,
            warnings=warnings,
        )

    def _get_next_extractor(self, current: ExtractorType) -> Optional[ExtractorType]:
        """Get next extractor in escalation path"""
        try:
            idx = ESCALATION_PATH.index(current)
            if idx < len(ESCALATION_PATH) - 1:
                return ESCALATION_PATH[idx + 1]
        except ValueError:
            pass
        return None

    def process_batch(self, file_paths: List[Path]) -> List[PipelineResult]:
        """Process multiple documents"""
        results = []
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing {i}/{len(file_paths)}: {file_path.name}")
            result = self.process(file_path)
            results.append(result)
        return results


class ReaddQuickExtract:
    """
    Quick extraction without full pipeline.
    For when you know the document quality and want speed.
    """

    @staticmethod
    def extract(file_path: Path, extractor: str = "pymupdf_text") -> ExtractionResult:
        """Quick extraction with specified extractor"""
        ext = get_extractor(extractor)
        return ext.extract(file_path)

    @staticmethod
    def extract_text(file_path: Path) -> str:
        """Get just the text, fast"""
        ext = PyMuPDFTextExtractor()
        result = ext.extract(file_path)
        return result.full_text

    @staticmethod
    def extract_markdown(file_path: Path) -> str:
        """Get markdown-formatted text"""
        ext = PyMuPDF4LLMExtractor()
        result = ext.extract(file_path)
        return result.full_text


def process_document(file_path: Path, auto_escalate: bool = True) -> PipelineResult:
    """Convenience function to process a document"""
    pipeline = ReaddPipeline(auto_escalate=auto_escalate)
    return pipeline.process(file_path)


def quick_extract(file_path: Path) -> str:
    """Convenience function for quick text extraction"""
    return ReaddQuickExtract.extract_text(file_path)
