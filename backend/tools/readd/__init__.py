"""
Readd - Intelligent Document Extraction Pipeline

Phased extraction with automatic quality assessment and escalation.

Usage:
    from tools.readd import process_document, quick_extract

    # Full pipeline with auto-escalation
    result = process_document("/path/to/document.pdf")
    print(result.text)

    # Quick extraction (no QA, no escalation)
    text = quick_extract("/path/to/document.pdf")

    # Hybrid extraction (page-level routing)
    from tools.readd import HybridExtractor, ExtractionMode
    extractor = HybridExtractor(mode=ExtractionMode.KNOWLEDGE_BASE)
    result = extractor.extract("/path/to/document.pdf")
"""

from .pipeline import (
    ReaddPipeline,
    PipelineResult,
    process_document,
    quick_extract,
    ReaddQuickExtract,
)
from .scout import (
    DocumentScout,
    ScoutReport,
    QualityTier,
    ExtractorType,
    scout_document,
)
from .extractors import (
    ExtractionResult,
    ExtractedPage,
    get_extractor,
    extract_document,
    EXTRACTORS,
)
from .qa import (
    ExtractionQA,
    QAReport,
    QAIssue,
    validate_extraction,
)
from .page_classifier import (
    PageClassifier,
    PageClassification,
    PageType,
    classify_document,
)
from .hybrid_extractor import (
    HybridExtractor,
    HybridExtractionResult,
    ExtractionMode,
    extract_hybrid,
)
from .graph_extractor import (
    GraphExtractor,
    ChartData,
    TableExtractor,
    TableData,
    extract_chart,
    extract_table,
)

__all__ = [
    # Pipeline
    "ReaddPipeline",
    "PipelineResult",
    "process_document",
    "quick_extract",
    "ReaddQuickExtract",
    # Scout
    "DocumentScout",
    "ScoutReport",
    "QualityTier",
    "ExtractorType",
    "scout_document",
    # Extractors
    "ExtractionResult",
    "ExtractedPage",
    "get_extractor",
    "extract_document",
    "EXTRACTORS",
    # QA
    "ExtractionQA",
    "QAReport",
    "QAIssue",
    "validate_extraction",
    # Page Classifier
    "PageClassifier",
    "PageClassification",
    "PageType",
    "classify_document",
    # Hybrid Extractor
    "HybridExtractor",
    "HybridExtractionResult",
    "ExtractionMode",
    "extract_hybrid",
    # Graph/Table Extractor
    "GraphExtractor",
    "ChartData",
    "TableExtractor",
    "TableData",
    "extract_chart",
    "extract_table",
]
