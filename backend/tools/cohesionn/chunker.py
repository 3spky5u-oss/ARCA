"""
Cohesionn Chunker - Smart text chunking for technical documents

Preserves document structure while creating retrieval-optimized chunks.
Includes table/chart boundary awareness to avoid splitting structured content.
"""

import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ContentType:
    """Content type constants for chunks."""

    PROSE = "prose"
    TABLE = "table"
    CHART_DATA = "chart_data"
    DIAGRAM_DESCRIPTION = "diagram_description"
    EQUATION = "equation"


@dataclass
class Chunk:
    """A chunk of text with metadata"""

    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.chunk_id:
            # Generate unique ID using all available metadata + full content hash
            # This prevents collisions when overlapped chunks share the same first N characters
            components = [
                self.metadata.get("source", ""),
                str(self.metadata.get("page", "")),
                str(self.metadata.get("chunk_index", "")),
                str(self.metadata.get("sub_chunk", "")),
                self.metadata.get("section", "")[:50] if self.metadata.get("section") else "",
                self.metadata.get("content_type", ""),
            ]
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            id_string = ":".join(str(c) for c in components if c) + f":{content_hash}"
            self.chunk_id = hashlib.md5(id_string.encode()).hexdigest()[:16]


class SemanticChunker:
    """
    Smart chunking that preserves document structure.

    Strategy:
    1. Split by major headers (chapters, sections)
    2. Within sections, split by paragraphs
    3. Merge small paragraphs, split large ones
    4. Add overlap for context continuity
    """

    # Header patterns for section detection
    HEADER_PATTERNS = [
        (r"^(?:Chapter|CHAPTER)\s+\d+[:\.\s]", 1),  # Chapter 1: Title
        (r"^(?:PART|Part)\s+[IVX\d]+[:\.\s]", 1),  # PART I: Title
        (r"^\d+\.\s+[A-Z][A-Za-z]", 2),  # 1. Section
        (r"^\d+\.\d+\s+[A-Z][A-Za-z]", 3),  # 1.1 Subsection
        (r"^\d+\.\d+\.\d+\s+[A-Z][A-Za-z]", 4),  # 1.1.1 Sub-subsection
        (r"^#{1,4}\s+", None),  # Markdown headers
    ]

    def __init__(
        self,
        chunk_size: int = 2500,
        chunk_overlap: int = 250,
        min_chunk_size: int = 100,
        max_chunk_size: int = 3000,
        context_prefix_enabled: bool = True,
    ):
        """
        Args:
            chunk_size: Target chunk size in characters (~375 words)
            chunk_overlap: Overlap between chunks for context
            min_chunk_size: Minimum chunk size (merge smaller)
            max_chunk_size: Maximum chunk size - allows atomic blocks (tables, equations)
                           to exceed target rather than splitting mid-structure
            context_prefix_enabled: Whether to prepend hierarchical context prefix
                                   (Document/Chapter/Section) to each chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.context_prefix_enabled = context_prefix_enabled

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Chunk text into retrieval-optimized pieces.

        Args:
            text: Full document text
            metadata: Base metadata to include in all chunks

        Returns:
            List of Chunk objects
        """
        metadata = metadata or {}
        chunks = []

        # Split into sections by headers with chapter tracking
        sections = self._split_by_headers(text, metadata)

        for section_title, section_text, level, chapter in sections:
            # Skip empty sections
            if not section_text.strip():
                continue

            # Build section metadata with chapter hierarchy
            section_meta = metadata.copy()
            if section_title:
                section_meta["section"] = section_title
                section_meta["section_level"] = level
            if chapter:
                section_meta["chapter"] = chapter

            # Chunk the section
            section_chunks = self._chunk_section(section_text, section_meta)
            chunks.extend(section_chunks)

        logger.info(f"Created {len(chunks)} chunks from {len(text)} chars")
        return chunks

    def chunk_pages(
        self,
        pages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Chunk from page-structured input (preserves page references).

        Args:
            pages: List of {"page_num": int, "text": str, "content_type": str (optional)}
            metadata: Base metadata

        Returns:
            List of chunks with page numbers and content type in metadata
        """
        metadata = metadata or {}
        chunks = []

        for page in pages:
            page_num = page.get("page_num", 0)
            text = page.get("text", "")
            content_type = page.get("content_type", ContentType.PROSE)

            if not text.strip():
                continue

            page_meta = metadata.copy()
            if page_num:  # 0 = pageless (docx, txt) — omit misleading page metadata
                page_meta["page"] = page_num
            page_meta["content_type"] = content_type

            # Special handling for structured content (tables, charts)
            if content_type in (ContentType.TABLE, ContentType.CHART_DATA):
                # Don't split structured content - keep as single chunk if reasonable
                page_chunks = self._chunk_structured_content(text, page_meta)
            else:
                # Regular chunking for prose/text
                page_chunks = self._chunk_section(text, page_meta)

            chunks.extend(page_chunks)

        return chunks

    def _chunk_structured_content(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Chunk structured content (tables, charts) with boundary awareness.

        Tries to keep tables/charts intact within chunks. If too large,
        splits at natural boundaries (empty lines, row separators).
        """
        content_type = metadata.get("content_type", ContentType.PROSE)

        # If small enough, keep as single chunk
        if len(text) <= self.max_chunk_size:
            return [
                Chunk(
                    chunk_id="",
                    content=text,
                    metadata=metadata,
                )
            ]

        # Need to split - find natural boundaries
        chunks = []
        chunk_index = 0

        if content_type == ContentType.TABLE:
            # Split tables at row boundaries (lines starting with |)
            chunks = self._split_table_at_boundaries(text, metadata)
        elif content_type == ContentType.CHART_DATA:
            # Chart data is JSON - try to split at data point boundaries
            chunks = self._split_chart_data(text, metadata)
        else:
            # Default structured splitting at double newlines
            parts = text.split("\n\n")
            current_chunk = []
            current_length = 0

            for part in parts:
                if current_length + len(part) > self.chunk_size and current_chunk:
                    chunk_meta = metadata.copy()
                    chunk_meta["chunk_index"] = chunk_index
                    chunks.append(
                        Chunk(
                            chunk_id="",
                            content="\n\n".join(current_chunk),
                            metadata=chunk_meta,
                        )
                    )
                    chunk_index += 1
                    current_chunk = [part]
                    current_length = len(part)
                else:
                    current_chunk.append(part)
                    current_length += len(part)

            if current_chunk:
                chunk_meta = metadata.copy()
                chunk_meta["chunk_index"] = chunk_index
                chunks.append(
                    Chunk(
                        chunk_id="",
                        content="\n\n".join(current_chunk),
                        metadata=chunk_meta,
                    )
                )

        return chunks

    def _split_table_at_boundaries(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """Split markdown table at row boundaries, preserving headers."""
        lines = text.split("\n")
        chunks = []
        chunk_index = 0

        # Find header lines (first two lines of a markdown table)
        header_lines = []
        data_lines = []

        for i, line in enumerate(lines):
            if i < 2 and ("|" in line or line.strip().startswith("|")):
                header_lines.append(line)
            elif line.strip():
                data_lines.append(line)

        header_text = "\n".join(header_lines) + "\n" if header_lines else ""
        header_len = len(header_text)

        # Group data lines into chunks
        current_rows = []
        current_length = header_len

        for row in data_lines:
            row_len = len(row) + 1  # +1 for newline

            if current_length + row_len > self.chunk_size and current_rows:
                # Save current chunk
                chunk_text = header_text + "\n".join(current_rows)
                chunk_meta = metadata.copy()
                chunk_meta["chunk_index"] = chunk_index
                chunks.append(
                    Chunk(
                        chunk_id="",
                        content=chunk_text,
                        metadata=chunk_meta,
                    )
                )
                chunk_index += 1
                current_rows = [row]
                current_length = header_len + row_len
            else:
                current_rows.append(row)
                current_length += row_len

        # Final chunk
        if current_rows:
            chunk_text = header_text + "\n".join(current_rows)
            chunk_meta = metadata.copy()
            chunk_meta["chunk_index"] = chunk_index
            chunks.append(
                Chunk(
                    chunk_id="",
                    content=chunk_text,
                    metadata=chunk_meta,
                )
            )

        return chunks if chunks else [Chunk(chunk_id="", content=text, metadata=metadata)]

    def _split_chart_data(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Split chart JSON data with schema-aware boundaries.

        Supports two schemas:
        - New: {"figures": [...]} — split per-figure, then by data_series if oversized
        - Legacy: {"data_points": [...]} — split at data point boundaries
        """
        import json

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: treat as regular text
            return [Chunk(chunk_id="", content=text, metadata=metadata)]

        if not isinstance(data, dict):
            return [Chunk(chunk_id="", content=text, metadata=metadata)]

        # New schema: figures[]
        if "figures" in data:
            return self._split_chart_data_figures(data, metadata)

        # Legacy schema: data_points[]
        if "data_points" in data:
            return self._split_chart_data_legacy(data, text, metadata)

        # Unknown structure — keep as single chunk
        return [Chunk(chunk_id="", content=text, metadata=metadata)]

    def _split_chart_data_figures(
        self,
        data: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Split new-schema chart data: one chunk per figure.

        Each figure is its own chunk to preserve recreation fidelity.
        If a single figure exceeds max_chunk_size, split by data_series.
        """
        import json

        figures = data.get("figures", [])
        if not figures:
            return [Chunk(chunk_id="", content=json.dumps(data, indent=2), metadata=metadata)]

        chunks = []
        chunk_index = 0

        for fig in figures:
            fig_json = json.dumps(
                {
                    "type": "chart_data",
                    "source_page": data.get("source_page"),
                    "figures": [fig],
                    "text_context": data.get("text_context", ""),
                },
                indent=2,
            )

            if len(fig_json) <= self.max_chunk_size:
                chunk_meta = metadata.copy()
                chunk_meta["chunk_index"] = chunk_index
                chunk_meta["figure_id"] = fig.get("figure_id", "")
                chunks.append(Chunk(chunk_id="", content=fig_json, metadata=chunk_meta))
                chunk_index += 1
            else:
                # Figure too large — split by data_series
                series_list = fig.get("data_series", [])
                if len(series_list) <= 1:
                    # Can't split further, keep as oversized chunk
                    chunk_meta = metadata.copy()
                    chunk_meta["chunk_index"] = chunk_index
                    chunk_meta["figure_id"] = fig.get("figure_id", "")
                    chunks.append(Chunk(chunk_id="", content=fig_json, metadata=chunk_meta))
                    chunk_index += 1
                else:
                    # One chunk per series
                    for series in series_list:
                        series_fig = fig.copy()
                        series_fig["data_series"] = [series]
                        series_json = json.dumps(
                            {
                                "type": "chart_data",
                                "source_page": data.get("source_page"),
                                "figures": [series_fig],
                            },
                            indent=2,
                        )
                        chunk_meta = metadata.copy()
                        chunk_meta["chunk_index"] = chunk_index
                        chunk_meta["figure_id"] = fig.get("figure_id", "")
                        chunk_meta["series_name"] = series.get("name", "")
                        chunks.append(Chunk(chunk_id="", content=series_json, metadata=chunk_meta))
                        chunk_index += 1

        return chunks if chunks else [Chunk(chunk_id="", content=json.dumps(data, indent=2), metadata=metadata)]

    def _split_chart_data_legacy(
        self,
        data: Dict[str, Any],
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """Split legacy chart JSON data at data_points boundaries."""
        import json

        points = data.get("data_points", [])

        if len(points) <= 20:
            return [Chunk(chunk_id="", content=text, metadata=metadata)]

        chunks = []
        chunk_index = 0
        points_per_chunk = 15

        for i in range(0, len(points), points_per_chunk):
            chunk_data = data.copy()
            chunk_data["data_points"] = points[i : i + points_per_chunk]
            chunk_data["_chunk_info"] = {
                "start_index": i,
                "end_index": min(i + points_per_chunk, len(points)),
                "total_points": len(points),
            }

            chunk_meta = metadata.copy()
            chunk_meta["chunk_index"] = chunk_index
            chunks.append(
                Chunk(
                    chunk_id="",
                    content=json.dumps(chunk_data, indent=2),
                    metadata=chunk_meta,
                )
            )
            chunk_index += 1

        return chunks

    def _split_by_headers(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str, int, Optional[str]]]:
        """
        Split text by section headers with chapter tracking.

        Returns:
            List of (header_title, section_text, header_level, chapter_title)
        """
        metadata = metadata or {}
        sections = []
        lines = text.split("\n")

        current_header = ""
        current_level = 0
        current_text = []
        current_chapter = metadata.get("chapter", "")  # Inherit from metadata

        for line in lines:
            header_match = self._match_header(line)

            if header_match:
                # Save previous section
                if current_text:
                    sections.append((current_header, "\n".join(current_text), current_level, current_chapter))

                current_header = line.strip()
                current_level = header_match

                # Track chapter hierarchy (level 1-2 are chapters, 3+ are sections)
                if current_level <= 2:
                    current_chapter = current_header

                current_text = []
            else:
                current_text.append(line)

        # Don't forget last section
        if current_text:
            sections.append((current_header, "\n".join(current_text), current_level, current_chapter))

        return sections if sections else [("", text, 0, current_chapter)]

    def _match_header(self, line: str) -> Optional[int]:
        """Check if line is a header, return level if so"""
        line = line.strip()
        if not line:
            return None

        for pattern, level in self.HEADER_PATTERNS:
            if re.match(pattern, line):
                if level is None:
                    # Markdown - count # chars
                    level = len(line) - len(line.lstrip("#"))
                return level

        return None

    def _chunk_section(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """Chunk a single section with enhanced context prefix and overlap.

        Prepends hierarchical context to each chunk for better retrieval.
        Format: "[Document: Title] [Chapter: Y] [Section: X]\n\nContent..."

        This contextual retrieval approach (per Anthropic research) reduces
        retrieval failures by ~35% by preserving document context.
        """
        chunks = []

        # Build enhanced context prefix from hierarchical metadata
        context_parts = [] if self.context_prefix_enabled else None

        # Document title (skip if prefix disabled)
        if context_parts is not None and metadata.get("title"):
            title = metadata["title"]
            # Clean and truncate long titles
            title = title[:60] + "..." if len(title) > 60 else title
            context_parts.append(f"[Document: {title}]")

        # Chapter (level 1-2 headers)
        if context_parts is not None and metadata.get("chapter"):
            chapter = metadata["chapter"]
            chapter = re.sub(r"^#+\s*", "", chapter)
            chapter = chapter.strip()[:50]
            if chapter:
                context_parts.append(f"[Chapter: {chapter}]")

        # Section (level 3+ headers)
        if context_parts is not None and metadata.get("section"):
            section = metadata["section"]
            # Clean section header (remove markdown #, extra whitespace)
            section = re.sub(r"^#+\s*", "", section)
            section = section.strip()
            if section:
                context_parts.append(f"[Section: {section}]")

        # Build final prefix
        context_prefix = ""
        if context_parts is not None and context_parts:
            context_prefix = " ".join(context_parts) + "\n\n"

        # Split into paragraphs
        paragraphs = self._split_paragraphs(text)

        current_chunk = []
        current_length = 0
        chunk_index = 0

        for para in paragraphs:
            para_len = len(para)

            # Check if adding this para exceeds target
            if current_length + para_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)

                if len(chunk_text) >= self.min_chunk_size:
                    chunk_meta = metadata.copy()
                    chunk_meta["chunk_index"] = chunk_index

                    # Prepend context prefix for better retrieval
                    final_content = context_prefix + chunk_text if context_prefix else chunk_text

                    chunks.append(
                        Chunk(
                            chunk_id="",  # Will be generated
                            content=final_content,
                            metadata=chunk_meta,
                        )
                    )
                    chunk_index += 1

                # Start new chunk with overlap
                overlap = self._get_overlap_paragraphs(current_chunk)
                current_chunk = overlap + [para]
                current_length = sum(len(p) for p in current_chunk)
            else:
                current_chunk.append(para)
                current_length += para_len

        # Handle last chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)

            if len(chunk_text) >= self.min_chunk_size:
                chunk_meta = metadata.copy()
                chunk_meta["chunk_index"] = chunk_index

                # Prepend context prefix for better retrieval
                final_content = context_prefix + chunk_text if context_prefix else chunk_text

                chunks.append(
                    Chunk(
                        chunk_id="",
                        content=final_content,
                        metadata=chunk_meta,
                    )
                )
            elif chunks:
                # Merge into previous chunk if too small
                last_chunk = chunks[-1]
                last_chunk.content += "\n\n" + chunk_text

        # Handle oversized chunks
        chunks = self._split_oversized(chunks)

        return chunks

    # Patterns indicating an equation or formula paragraph
    _EQUATION_PATTERNS = re.compile(
        r"(?:"
        r"^\s*[A-Za-z_]+\s*=\s*"           # Variable = expression
        r"|[Σ∑∫∂√±×÷∞≤≥≈∝∈∉∀∃π]"          # Math unicode symbols
        r"|\\(?:frac|sum|int|sqrt|alpha|beta|gamma|sigma|phi)\b"  # LaTeX commands
        r"|\$\$.+\$\$"                      # LaTeX display math
        r"|\$[^$]+\$"                       # LaTeX inline math
        r")",
        re.MULTILINE,
    )

    # Patterns indicating a "where:" definition block that follows an equation
    _WHERE_CLAUSE_PATTERN = re.compile(
        r"^\s*(?:where|in which|such that|with|and)\s*[:\-]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    def _merge_equation_blocks(self, paragraphs: List[str]) -> List[str]:
        """Merge equation paragraphs with their subsequent where-clause definitions.

        Prevents splitting an equation from its variable definitions, e.g.:
            'q_u = c × Nc + γ × D × Nq'  +  'Where: c = cohesion...'
        """
        if len(paragraphs) <= 1:
            return paragraphs

        merged = []
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i]

            # If this paragraph contains an equation, check if next is a where-clause
            if self._EQUATION_PATTERNS.search(para) and i + 1 < len(paragraphs):
                next_para = paragraphs[i + 1]
                # Merge if next paragraph starts with "where:" or is a definition list
                if self._WHERE_CLAUSE_PATTERN.match(next_para) or (
                    next_para.lstrip().lower().startswith("where") and "=" in next_para
                ):
                    combined = para + "\n\n" + next_para
                    # Only merge if combined size is reasonable
                    if len(combined) <= self.max_chunk_size:
                        merged.append(combined)
                        i += 2
                        continue

            merged.append(para)
            i += 1

        return merged

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs, preserving equation+definition blocks."""
        # Split by double newlines
        paragraphs = re.split(r"\n\s*\n", text)

        # Clean up
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Merge equation blocks with their where-clause definitions
        paragraphs = self._merge_equation_blocks(paragraphs)

        return paragraphs

    def _get_overlap_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Get paragraphs for overlap from end of chunk"""
        if not paragraphs:
            return []

        overlap = []
        length = 0

        for para in reversed(paragraphs):
            if length + len(para) > self.chunk_overlap:
                break
            overlap.insert(0, para)
            length += len(para)

        return overlap

    def _split_oversized(self, chunks: List[Chunk]) -> List[Chunk]:
        """Split any chunks that exceed max size"""
        result = []

        for chunk in chunks:
            if len(chunk.content) <= self.max_chunk_size:
                result.append(chunk)
            else:
                # Split by sentences
                sentences = re.split(r"(?<=[.!?])\s+", chunk.content)

                current = []
                current_len = 0
                sub_index = 0

                for sent in sentences:
                    if current_len + len(sent) > self.chunk_size and current:
                        # Save and start new
                        sub_meta = chunk.metadata.copy()
                        sub_meta["sub_chunk"] = sub_index

                        result.append(
                            Chunk(
                                chunk_id="",
                                content=" ".join(current),
                                metadata=sub_meta,
                            )
                        )
                        sub_index += 1
                        current = [sent]
                        current_len = len(sent)
                    else:
                        current.append(sent)
                        current_len += len(sent)

                if current:
                    sub_meta = chunk.metadata.copy()
                    sub_meta["sub_chunk"] = sub_index
                    result.append(
                        Chunk(
                            chunk_id="",
                            content=" ".join(current),
                            metadata=sub_meta,
                        )
                    )

        return result


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 250,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """Convenience function to chunk text"""
    chunker = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunker.chunk_text(text, metadata)
