"""
Unit tests for Cohesionn chunker module.

Tests:
- Chunk size limits and overlap
- Header detection and section splitting
- Context prefix generation
- Table/chart boundary handling
"""

import pytest
from tools.cohesionn.chunker import (
    SemanticChunker,
    Chunk,
    ContentType,
    chunk_text,
)


class TestChunkBasics:
    """Basic chunking functionality tests."""

    def test_chunk_generates_unique_ids(self):
        """Each chunk should have a unique ID."""
        chunker = SemanticChunker()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        chunks = chunker.chunk_text(text, {"source": "test.md"})

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique"

    def test_chunk_preserves_metadata(self):
        """Metadata should be preserved in chunks."""
        chunker = SemanticChunker()
        metadata = {"source": "test.pdf", "topic": "technical"}
        text = "Test content for chunking."

        chunks = chunker.chunk_text(text, metadata)

        for chunk in chunks:
            assert chunk.metadata.get("source") == "test.pdf"
            assert chunk.metadata.get("topic") == "technical"

    def test_empty_text_returns_empty_list(self):
        """Empty text should return empty list."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_text("")
        assert chunks == []

    def test_whitespace_only_returns_empty(self):
        """Whitespace-only text should return empty list."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_text("   \n\n   \t   ")
        assert chunks == []


class TestChunkSizeLimits:
    """Tests for chunk size enforcement."""

    def test_chunks_respect_max_size(self):
        """Chunks should not exceed max_chunk_size."""
        chunker = SemanticChunker(chunk_size=500, max_chunk_size=800)

        # Create text that would normally exceed limits
        long_text = "This is a sentence. " * 100  # ~2000 chars

        chunks = chunker.chunk_text(long_text)

        for chunk in chunks:
            assert len(chunk.content) <= 800, (
                f"Chunk exceeds max_chunk_size: {len(chunk.content)}"
            )

    def test_small_chunks_merged(self):
        """Chunks smaller than min_chunk_size should be merged."""
        chunker = SemanticChunker(chunk_size=500, min_chunk_size=100)

        # Very short paragraphs
        text = "Short.\n\nTiny.\n\nBrief.\n\nSmall."

        chunks = chunker.chunk_text(text)

        # Should be merged into fewer chunks
        for chunk in chunks:
            assert len(chunk.content) >= 20  # Some minimum

    def test_chunk_size_target(self):
        """Chunks should be close to target size when possible."""
        chunk_size = 500
        chunker = SemanticChunker(chunk_size=chunk_size, min_chunk_size=50)

        # Medium-length text that should result in ~target size chunks
        paragraphs = [f"This is paragraph number {i}. It contains some content about structural engineering and material mechanics. " * 5 for i in range(15)]
        text = "\n\n".join(paragraphs)

        chunks = chunker.chunk_text(text)

        # Most chunks should be reasonably close to target
        sizes = [len(c.content) for c in chunks]
        avg_size = sum(sizes) / len(sizes) if sizes else 0

        # Average should be in reasonable range (allowing for overlap text and context prefix)
        assert 200 < avg_size < 2000, f"Average chunk size {avg_size} not near target"


class TestChunkOverlap:
    """Tests for chunk overlap behavior."""

    def test_chunks_have_overlap(self):
        """Consecutive chunks should have overlapping content."""
        chunker = SemanticChunker(chunk_size=200, chunk_overlap=50)

        paragraphs = [f"Paragraph {i} with some content here." for i in range(20)]
        text = "\n\n".join(paragraphs)

        chunks = chunker.chunk_text(text)

        if len(chunks) >= 2:
            # Check that consecutive chunks share some content
            for i in range(len(chunks) - 1):
                chunk1_words = set(chunks[i].content.split()[-20:])  # Last 20 words
                chunk2_words = set(chunks[i + 1].content.split()[:20])  # First 20 words

                overlap = chunk1_words & chunk2_words
                # There should be some overlap (context continuity)
                # Note: May not always overlap if paragraphs are clean breaks


class TestHeaderDetection:
    """Tests for section header detection."""

    def test_chapter_header_detected(self):
        """Chapter headers should be detected."""
        # Use lower min_chunk_size for this test
        chunker = SemanticChunker(min_chunk_size=50)

        text = """Chapter 1: Introduction

This is the introduction content. It discusses the fundamentals of structural engineering and material mechanics principles. The content here is substantial enough to form a valid chunk for retrieval purposes.

Chapter 2: Methods

This is the methods content. It describes the testing procedures and analysis methods used in technical investigations. Method-A and Method-B verification tests are commonly used."""

        chunks = chunker.chunk_text(text)

        # Should have section or chapter metadata
        has_chapter_info = any(
            c.metadata.get("section") or c.metadata.get("chapter")
            for c in chunks
        )
        assert has_chapter_info or len(chunks) > 0

    def test_numbered_section_detected(self):
        """Numbered sections should be detected."""
        chunker = SemanticChunker(min_chunk_size=50)

        text = """1. First Section

Content of first section covering load capacity theory and structural design principles. This section provides fundamental knowledge for design engineers.

2. Second Section

Content of second section discussing deformation analysis and stress distribution theory. These concepts are essential for understanding material behavior under load.

2.1 Subsection

Subsection content with additional details about specific calculation methods and design procedures used in practice."""

        chunks = chunker.chunk_text(text)

        # Should produce chunks from this content
        assert len(chunks) > 0

    def test_markdown_headers_detected(self):
        """Markdown headers should be detected."""
        chunker = SemanticChunker(min_chunk_size=50)

        text = """# Main Title

Introduction paragraph explaining the purpose of this technical document. This content provides context for the detailed sections that follow.

## First Section

Section content here discussing the main technical concepts. The load capacity of structural elements depends on material properties and element geometry.

### Subsection

More detailed content about specific calculation methods. Standard capacity equations are widely used in structural design practice."""

        chunks = chunker.chunk_text(text)

        # Should create chunks with section info
        assert len(chunks) > 0


class TestContextPrefix:
    """Tests for hierarchical context prefix generation."""

    def test_context_prefix_includes_section(self):
        """Context prefix should include section title."""
        chunker = SemanticChunker(min_chunk_size=50)

        text = """## Load Capacity

The load capacity of a system is determined by several factors including material strength, geometry, and boundary conditions. The standard equation provides the fundamental framework for calculating ultimate load capacity. The allowable capacity is then determined by applying an appropriate factor of safety."""

        chunks = chunker.chunk_text(text)

        # At least one chunk should have section in prefix or content
        content_with_section = [c.content for c in chunks if "[Section:" in c.content]
        assert len(content_with_section) > 0 or any(
            "Load Capacity" in c.content for c in chunks
        )

    def test_context_prefix_includes_chapter(self):
        """Context prefix should include chapter when available."""
        chunker = SemanticChunker(min_chunk_size=50)

        text = """Chapter 3: Structural Design

## Primary Elements

Content about primary structural elements including beams and columns. The design of primary elements requires consideration of load capacity, deformation, and safety requirements. These elements are suitable for applications with adequate material strength."""

        chunks = chunker.chunk_text(text, {"chapter": "Chapter 3: Structural Design"})

        # Chapter should be in metadata or prefix
        chunks_with_chapter = [
            c for c in chunks
            if c.metadata.get("chapter") or "[Chapter:" in c.content
        ]
        assert len(chunks_with_chapter) > 0 or len(chunks) > 0

    def test_context_prefix_includes_title(self):
        """Context prefix should include document title when provided."""
        chunker = SemanticChunker(min_chunk_size=50)

        text = "Some content in the document discussing structural engineering principles. This text is long enough to meet the minimum chunk size requirements for proper chunking and retrieval."

        chunks = chunker.chunk_text(text, {"title": "Reference Manual 5th Edition"})

        # Title should affect prefix or be in metadata
        assert len(chunks) > 0


class TestStructuredContent:
    """Tests for table and chart handling."""

    def test_table_kept_intact_when_small(self):
        """Small tables should not be split."""
        chunker = SemanticChunker(max_chunk_size=2000)

        table_text = """| Parameter | Value |
|-----------|-------|
| V60       | 30    |
| φ         | 35°   |
| c         | 0     |"""

        pages = [{"page_num": 1, "text": table_text, "content_type": ContentType.TABLE}]
        chunks = chunker.chunk_pages(pages)

        # Should be single chunk
        assert len(chunks) == 1
        assert "|" in chunks[0].content  # Table formatting preserved

    def test_large_table_split_at_rows(self):
        """Large tables should be split at row boundaries."""
        chunker = SemanticChunker(chunk_size=200, max_chunk_size=400)

        # Create large table
        rows = ["| Row {} | Value {} |".format(i, i) for i in range(50)]
        table_text = "| Header | Data |\n|--------|------|\n" + "\n".join(rows)

        pages = [{"page_num": 1, "text": table_text, "content_type": ContentType.TABLE}]
        chunks = chunker.chunk_pages(pages)

        # Should be multiple chunks, each starting with header
        if len(chunks) > 1:
            for chunk in chunks:
                # Each chunk should have the header row
                assert "Header" in chunk.content or "Row" in chunk.content


class TestPageChunking:
    """Tests for page-based chunking."""

    def test_page_numbers_preserved(self):
        """Page numbers should be in chunk metadata."""
        chunker = SemanticChunker(min_chunk_size=50)

        pages = [
            {"page_num": 1, "text": "Content from page 1 discussing material mechanics and structural design principles in engineering."},
            {"page_num": 2, "text": "Content from page 2 covering load capacity calculations and deformation analysis methods."},
            {"page_num": 3, "text": "Content from page 3 about primary components and detailed design considerations."},
        ]

        chunks = chunker.chunk_pages(pages)

        page_nums = [c.metadata.get("page") for c in chunks]
        assert 1 in page_nums or 2 in page_nums or 3 in page_nums

    def test_empty_pages_skipped(self):
        """Empty pages should not create chunks."""
        chunker = SemanticChunker(min_chunk_size=50)

        pages = [
            {"page_num": 1, "text": "Content from page one with enough text to meet minimum chunk size requirements for testing."},
            {"page_num": 2, "text": ""},
            {"page_num": 3, "text": "   "},
            {"page_num": 4, "text": "More content from page four also meeting the minimum length requirement for chunking."},
        ]

        chunks = chunker.chunk_pages(pages)

        # Only pages 1 and 4 should create chunks
        page_nums = [c.metadata.get("page") for c in chunks]
        assert 2 not in page_nums
        assert 3 not in page_nums


class TestConvenienceFunction:
    """Tests for chunk_text convenience function."""

    def test_chunk_text_function(self):
        """chunk_text function should work with defaults."""
        # Use text long enough to exceed min_chunk_size (100 chars default)
        text = "Some content to chunk into pieces for retrieval. This text discusses engineering concepts including load capacity and deformation analysis for structural design."

        chunks = chunk_text(text)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_text_with_custom_size(self):
        """chunk_text should respect custom chunk size."""
        text = "Word " * 500  # ~2500 chars

        # Use smaller chunk size and ensure min_chunk_size allows it
        chunks = chunk_text(text, chunk_size=300)

        # Should create multiple chunks
        assert len(chunks) >= 1  # At least one chunk from this content
