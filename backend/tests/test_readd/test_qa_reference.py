"""
QA Reference Tests - Validate extraction quality against ground truth.

These tests use synthetic PDFs with known content to verify that
extractors meet minimum accuracy thresholds.
"""

import pytest
from pathlib import Path


class TestQAReferenceDocuments:
    """Test extraction quality against QA reference documents."""

    def test_manifest_exists(self, qa_manifest):
        """Verify QA manifest is valid."""
        assert "documents" in qa_manifest
        assert len(qa_manifest["documents"]) > 0
        assert "version" in qa_manifest

    def test_all_reference_docs_exist(self, qa_manifest, qa_docs_dir, qa_expected_dir):
        """Verify all reference documents and ground truth files exist."""
        for doc in qa_manifest["documents"]:
            pdf_path = qa_docs_dir / doc["filename"]
            txt_path = qa_expected_dir / Path(doc["ground_truth"]).name

            if not pdf_path.exists():
                pytest.skip(f"Reference PDF not found: {pdf_path} - run generate_test_docs.py")

            assert txt_path.exists(), f"Ground truth missing: {txt_path}"


class TestCleanDigitalExtraction:
    """Tests for HIGH quality tier (clean digital PDFs)."""

    @pytest.fixture
    def clean_doc_config(self, qa_manifest):
        """Get config for clean digital document."""
        for doc in qa_manifest["documents"]:
            if doc["filename"] == "clean_digital.pdf":
                return doc
        pytest.skip("clean_digital.pdf not in manifest")

    def test_clean_digital_extraction_accuracy(self, clean_doc_config, qa_docs_dir, qa_expected_dir, levenshtein_ratio):
        """Test that clean digital PDF extraction meets accuracy threshold."""
        from tools.readd.extractors import PyMuPDFTextExtractor

        pdf_path = qa_docs_dir / clean_doc_config["filename"]
        if not pdf_path.exists():
            pytest.skip(f"PDF not found: {pdf_path}")

        expected_path = qa_expected_dir / Path(clean_doc_config["ground_truth"]).name
        expected_text = expected_path.read_text(encoding="utf-8")

        extractor = PyMuPDFTextExtractor()
        result = extractor.extract(pdf_path)

        accuracy = levenshtein_ratio(result.full_text, expected_text)
        min_accuracy = clean_doc_config["min_accuracy"]

        assert accuracy >= min_accuracy, f"Extraction accuracy {accuracy:.2%} below threshold {min_accuracy:.0%}"

    def test_clean_digital_scout_recommendation(self, clean_doc_config, qa_docs_dir):
        """Test that scout recommends appropriate extractor for clean PDF."""
        from tools.readd.scout import DocumentScout, QualityTier, ExtractorType

        pdf_path = qa_docs_dir / clean_doc_config["filename"]
        if not pdf_path.exists():
            pytest.skip(f"PDF not found: {pdf_path}")

        scout = DocumentScout()
        report = scout.scout(pdf_path)

        # Clean digital should be HIGH or MEDIUM tier (depends on text density)
        assert report.quality_tier in [
            QualityTier.HIGH,
            QualityTier.MEDIUM,
        ], f"Expected HIGH/MEDIUM tier, got {report.quality_tier.value}"

        # Should recommend a text-based extractor (not vision OCR)
        assert report.recommended_extractor in [
            ExtractorType.PYMUPDF_TEXT,
            ExtractorType.PYMUPDF4LLM,
        ], f"Unexpected extractor: {report.recommended_extractor.value}"


class TestMixedLayoutExtraction:
    """Tests for MEDIUM quality tier (tables, columns)."""

    @pytest.fixture
    def mixed_doc_config(self, qa_manifest):
        """Get config for mixed layout document."""
        for doc in qa_manifest["documents"]:
            if doc["filename"] == "mixed_layout.pdf":
                return doc
        pytest.skip("mixed_layout.pdf not in manifest")

    def test_mixed_layout_extraction_accuracy(self, mixed_doc_config, qa_docs_dir, qa_expected_dir, levenshtein_ratio):
        """Test that mixed layout PDF extraction meets accuracy threshold."""
        from tools.readd.extractors import PyMuPDF4LLMExtractor, PyMuPDFTextExtractor

        pdf_path = qa_docs_dir / mixed_doc_config["filename"]
        if not pdf_path.exists():
            pytest.skip(f"PDF not found: {pdf_path}")

        expected_path = qa_expected_dir / Path(mixed_doc_config["ground_truth"]).name
        expected_text = expected_path.read_text(encoding="utf-8")

        # Try PyMuPDF4LLM first, fall back to basic
        try:
            extractor = PyMuPDF4LLMExtractor()
        except Exception:
            extractor = PyMuPDFTextExtractor()

        result = extractor.extract(pdf_path)

        accuracy = levenshtein_ratio(result.full_text, expected_text)
        min_accuracy = mixed_doc_config["min_accuracy"]

        assert accuracy >= min_accuracy, f"Extraction accuracy {accuracy:.2%} below threshold {min_accuracy:.0%}"

    def test_mixed_layout_word_accuracy(self, mixed_doc_config, qa_docs_dir, qa_expected_dir, word_accuracy):
        """Test word-level accuracy for table content."""
        from tools.readd.extractors import PyMuPDFTextExtractor

        pdf_path = qa_docs_dir / mixed_doc_config["filename"]
        if not pdf_path.exists():
            pytest.skip(f"PDF not found: {pdf_path}")

        expected_path = qa_expected_dir / Path(mixed_doc_config["ground_truth"]).name
        expected_text = expected_path.read_text(encoding="utf-8")

        extractor = PyMuPDFTextExtractor()
        result = extractor.extract(pdf_path)

        accuracy = word_accuracy(result.full_text, expected_text)

        # Word accuracy should be at least 70% even with table formatting issues
        assert accuracy >= 0.70, f"Word accuracy {accuracy:.2%} too low"


class TestScannedDocumentExtraction:
    """Tests for LOW quality tier (scanned documents).

    Note: These tests use synthetic PDFs with text that simulates
    scanned documents. True scanned document detection would require
    actual image-based PDFs.
    """

    @pytest.fixture
    def scanned_doc_config(self, qa_manifest):
        """Get config for scanned document."""
        for doc in qa_manifest["documents"]:
            if doc["filename"] == "scanned_poor.pdf":
                return doc
        pytest.skip("scanned_poor.pdf not in manifest")

    def test_scanned_document_extraction(self, scanned_doc_config, qa_docs_dir, qa_expected_dir, levenshtein_ratio):
        """Test that scanned document extraction meets accuracy threshold."""
        from tools.readd.extractors import PyMuPDFTextExtractor

        pdf_path = qa_docs_dir / scanned_doc_config["filename"]
        if not pdf_path.exists():
            pytest.skip(f"PDF not found: {pdf_path}")

        expected_path = qa_expected_dir / Path(scanned_doc_config["ground_truth"]).name
        expected_text = expected_path.read_text(encoding="utf-8")

        extractor = PyMuPDFTextExtractor()
        result = extractor.extract(pdf_path)

        # For synthetic PDFs, text should extract well
        accuracy = levenshtein_ratio(result.full_text, expected_text)
        min_accuracy = scanned_doc_config["min_accuracy"]

        assert accuracy >= min_accuracy, f"Extraction accuracy {accuracy:.2%} below threshold {min_accuracy:.0%}"


class TestTerribleDocumentExtraction:
    """Tests for TERRIBLE quality tier (heavily degraded).

    Note: True terrible quality documents would be image-based with
    heavy degradation. These synthetic PDFs simulate degraded text
    but still have searchable text content.
    """

    @pytest.fixture
    def terrible_doc_config(self, qa_manifest):
        """Get config for terrible document."""
        for doc in qa_manifest["documents"]:
            if doc["filename"] == "terrible_ocr.pdf":
                return doc
        pytest.skip("terrible_ocr.pdf not in manifest")

    def test_terrible_document_low_density(self, terrible_doc_config, qa_docs_dir):
        """Test that scout detects low text density in terrible doc."""
        from tools.readd.scout import DocumentScout

        pdf_path = qa_docs_dir / terrible_doc_config["filename"]
        if not pdf_path.exists():
            pytest.skip(f"PDF not found: {pdf_path}")

        scout = DocumentScout()
        report = scout.scout(pdf_path)

        # The terrible doc has intentionally sparse content
        # Text density should be low (less than 500 chars/page)
        assert report.text_density < 500, f"Text density {report.text_density} too high for sparse content"

    def test_terrible_document_has_warnings(self, terrible_doc_config, qa_docs_dir):
        """Test that scout generates appropriate warnings."""
        from tools.readd.scout import DocumentScout

        pdf_path = qa_docs_dir / terrible_doc_config["filename"]
        if not pdf_path.exists():
            pytest.skip(f"PDF not found: {pdf_path}")

        scout = DocumentScout()
        report = scout.scout(pdf_path)

        # Should have quality warnings due to low text density
        assert len(report.warnings) > 0, "Expected quality warnings for terrible doc"

    def test_terrible_document_extraction(self, terrible_doc_config, qa_docs_dir, qa_expected_dir, levenshtein_ratio):
        """Test that terrible document content is still extractable."""
        from tools.readd.extractors import PyMuPDFTextExtractor

        pdf_path = qa_docs_dir / terrible_doc_config["filename"]
        if not pdf_path.exists():
            pytest.skip(f"PDF not found: {pdf_path}")

        expected_path = qa_expected_dir / Path(terrible_doc_config["ground_truth"]).name
        expected_text = expected_path.read_text(encoding="utf-8")

        extractor = PyMuPDFTextExtractor()
        result = extractor.extract(pdf_path)

        # Synthetic PDFs should still extract well despite visual noise
        accuracy = levenshtein_ratio(result.full_text, expected_text)
        min_accuracy = terrible_doc_config["min_accuracy"]

        assert accuracy >= min_accuracy, f"Extraction accuracy {accuracy:.2%} below threshold {min_accuracy:.0%}"


class TestPipelineEscalation:
    """Tests for extraction pipeline with automatic escalation."""

    def test_pipeline_processes_clean_document(self, qa_docs_dir):
        """Test that pipeline successfully processes clean document."""
        from tools.readd.pipeline import ReaddPipeline

        pdf_path = qa_docs_dir / "clean_digital.pdf"
        if not pdf_path.exists():
            pytest.skip(f"PDF not found: {pdf_path}")

        pipeline = ReaddPipeline(auto_escalate=True)
        result = pipeline.process(pdf_path)

        assert result.success, f"Pipeline failed: {result.warnings}"
        assert result.text is not None
        assert len(result.text) > 100
        # Note: escalation may occur if QA heuristics trigger it

    def test_pipeline_returns_metadata(self, qa_docs_dir):
        """Test that pipeline returns proper metadata."""
        from tools.readd.pipeline import ReaddPipeline

        pdf_path = qa_docs_dir / "clean_digital.pdf"
        if not pdf_path.exists():
            pytest.skip(f"PDF not found: {pdf_path}")

        pipeline = ReaddPipeline()
        result = pipeline.process(pdf_path)

        # Check all metadata fields
        assert result.file_path is not None
        assert result.scout_report is not None
        assert result.extractors_tried is not None
        assert result.processing_time_ms >= 0

    def test_pipeline_extracts_text(self, qa_docs_dir, qa_expected_dir, levenshtein_ratio):
        """Test that pipeline extracts correct content."""
        from tools.readd.pipeline import ReaddPipeline

        pdf_path = qa_docs_dir / "mixed_layout.pdf"
        expected_path = qa_expected_dir / "mixed_layout.txt"

        if not pdf_path.exists():
            pytest.skip(f"PDF not found: {pdf_path}")

        expected_text = expected_path.read_text(encoding="utf-8")

        pipeline = ReaddPipeline(auto_escalate=False)
        result = pipeline.process(pdf_path)

        assert result.success
        accuracy = levenshtein_ratio(result.text, expected_text)
        assert accuracy >= 0.80, f"Pipeline extraction accuracy {accuracy:.2%} too low"


class TestExtractionMetrics:
    """Tests for extraction quality metrics calculation."""

    def test_levenshtein_ratio_identical(self, levenshtein_ratio):
        """Test Levenshtein ratio for identical strings."""
        result = levenshtein_ratio("hello world", "hello world")
        assert result == 1.0

    def test_levenshtein_ratio_different(self, levenshtein_ratio):
        """Test Levenshtein ratio for different strings."""
        result = levenshtein_ratio("hello", "world")
        assert result < 0.5

    def test_levenshtein_ratio_similar(self, levenshtein_ratio):
        """Test Levenshtein ratio for similar strings."""
        result = levenshtein_ratio("hello world", "hello wurld")
        assert result > 0.8

    def test_word_accuracy_identical(self, word_accuracy):
        """Test word accuracy for identical strings."""
        result = word_accuracy("hello world test", "hello world test")
        assert result == 1.0

    def test_word_accuracy_partial(self, word_accuracy):
        """Test word accuracy for partial overlap."""
        result = word_accuracy("hello world", "hello there")
        assert 0.3 < result < 0.7  # ~0.33 (1 of 3 words overlap)

    def test_word_accuracy_no_overlap(self, word_accuracy):
        """Test word accuracy for no overlap."""
        result = word_accuracy("hello world", "foo bar")
        assert result == 0.0
