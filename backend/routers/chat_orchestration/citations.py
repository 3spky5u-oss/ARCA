"""
ARCA Citation Collector - Citation accumulation from tools

Collects and manages citations from different tool types (RAG, web search).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


# Patterns to strip from citation titles (piracy sites, download artifacts, etc.)
CITATION_CLEANUP_PATTERNS = [
    r"\s*-\s*libgen\.l[ic]\.?",  # libgen.li, libgen.lc
    r"\s*\(\s*z-lib\.org\s*\)",  # (z-lib.org)
    r"\s*\[\s*z-lib\.org\s*\]",  # [z-lib.org]
    r"\s*-\s*z-lib\.org",  # - z-lib.org
    r"\s*\(\s*PDFDrive\.com\s*\)",  # (PDFDrive.com)
    r"\s*\[\s*PDFDrive\s*\]",  # [PDFDrive]
    r"\s*-\s*PDFDrive",  # - PDFDrive
    r"\s*\(\s*libgen\s*\)",  # (libgen)
    r"\s*_+\s*$",  # Trailing underscores
    r"\s*\(\s*\d+\s*\)\s*$",  # Trailing (year) if orphaned
]


def clean_citation_title(title: str) -> str:
    """Clean citation title by removing piracy site references and artifacts.

    Args:
        title: Raw citation title (often from PDF filename)

    Returns:
        Cleaned title suitable for display
    """
    if not title:
        return title

    cleaned = title
    for pattern in CITATION_CLEANUP_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # Clean up multiple spaces and trim
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Remove trailing punctuation artifacts
    cleaned = re.sub(r"[,\-_]+$", "", cleaned).strip()

    return cleaned if cleaned else title


@dataclass
class CitationCollector:
    """Accumulates citations from different tool executions.

    Collects citations from:
    - RAG tools (search_knowledge, search_session)
    - Web search tools
    - Auto-search results

    Tracks overall confidence for RAG results.
    """

    citations: List[Dict[str, Any]] = field(default_factory=list)
    _confidence_values: List[float] = field(default_factory=list)

    def add_from_rag(self, result: Dict[str, Any]) -> None:
        """Add citations from a RAG tool result.

        Args:
            result: Tool result dict with 'citations' and optionally 'avg_confidence'
        """
        if result.get("citations"):
            for citation in result["citations"]:
                # Clean up citation title (remove libgen, z-lib, etc.)
                if citation.get("title"):
                    citation["title"] = clean_citation_title(citation["title"])
                self.citations.append(citation)
        if result.get("avg_confidence") is not None:
            self._confidence_values.append(result["avg_confidence"])

    def add_from_web_search(self, result: Dict[str, Any]) -> None:
        """Add citations from web search result.

        Args:
            result: Web search result with 'results' list
        """
        if not result.get("success") or not result.get("results"):
            return

        is_deep = result.get("deep_search", False)
        topic = "Deep Search" if is_deep else "Web Search"

        for r in result.get("results", []):
            citation = {
                "source": r.get("url", ""),
                "title": r.get("title", "Web Result"),
                "topic": topic,
                "score": 1.0,
            }
            self.citations.append(citation)

    def add_citation(self, source: str, title: str, topic: str, score: float = 1.0) -> None:
        """Add a single citation manually.

        Args:
            source: URL or file path
            title: Human-readable title
            topic: Category (e.g., "engineering", "Web Search")
            score: Relevance score (0-1)
        """
        self.citations.append(
            {
                "source": source,
                "title": title,
                "topic": topic,
                "score": score,
            }
        )

    def get_all(self) -> Optional[List[Dict[str, Any]]]:
        """Get all collected citations.

        Returns:
            List of citation dicts, or None if empty
        """
        return self.citations if self.citations else None

    def get_confidence(self) -> Optional[float]:
        """Get average confidence across all RAG results.

        Returns:
            Average confidence score, or None if no RAG results
        """
        if not self._confidence_values:
            return None
        return sum(self._confidence_values) / len(self._confidence_values)

    def clear(self) -> None:
        """Clear all collected citations and confidence values."""
        self.citations = []
        self._confidence_values = []

    def __len__(self) -> int:
        """Return number of citations collected."""
        return len(self.citations)
