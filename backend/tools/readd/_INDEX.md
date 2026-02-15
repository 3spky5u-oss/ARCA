# backend/tools/readd/

Document processing pipeline with a 4-phase architecture: Scout (quality assessment) -> Extract (tiered extraction) -> QA (quality checks) -> Escalate (try next extractor). Supports clean digital PDFs through terrible OCR scans with automatic quality-based extractor selection.

| File | Purpose | Key Exports |
|------|---------|-------------|
| __init__.py | Public API, convenience functions | `process_document()`, `quick_extract()`, `HybridExtractor` |
| __main__.py | `python -m tools.readd file.pdf` CLI entry | -- |
| scout.py | Document quality assessment (0-100 score, 4 tiers) | `DocumentScout`, `ScoutReport`, `QualityTier` |
| extractors.py | 5-tier extraction hierarchy (pymupdf_text, pymupdf4llm, docling, marker, vision_ocr). DoclingExtractor (IBM Docling v2.72.0, MIT) is the primary extractor for batch ingest — single-pass layout analysis + OCR + tables on CPU, also detects figure pages via PictureItem elements | `DoclingExtractor`, `PyMuPDFTextExtractor`, `PyMuPDF4LLMExtractor`, `MarkerExtractor`, `VisionOCRExtractor`, `ExtractionResult` |
| hybrid_extractor.py | Per-page routing: simple pages -> PyMuPDF, complex -> vision | `HybridExtractor`, `ExtractionMode` |
| page_classifier.py | Heuristic page type detection (text, structured, visual, scanned) | `PageClassifier`, `PageType` |
| graph_extractor.py | Vision-based chart/figure extraction to structured JSON | `GraphExtractor`, `FigureData`, `PageChartData`, `TableExtractor` |
| pipeline.py | Pipeline orchestrator with auto-escalation | `ReaddPipeline`, `ReaddQuickExtract`, `PipelineResult` |
| qa.py | 7 extraction quality checks with weirdness scoring | `ExtractionQA`, `QAIssue` |
| generate_test_docs.py | Synthetic test PDF generator (4 quality tiers) | `generate_test_documents()` |
