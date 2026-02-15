"""
Page Sampler for Benchmark

Extracts random pages from PDFs for benchmarking ingestion performance.
Uses PyMuPDF (fitz) to extract single-page PDFs.
"""

import random
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class SampledPage:
    """A sampled page from a PDF for benchmarking."""

    book_path: Path
    book_name: str
    page_num: int  # 1-indexed
    total_pages: int
    temp_pdf_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

    def cleanup(self):
        """Remove temporary PDF file."""
        if self.temp_pdf_path and self.temp_pdf_path.exists():
            try:
                self.temp_pdf_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


@dataclass
class BookInfo:
    """Information about a PDF book."""

    path: Path
    name: str
    total_pages: int
    file_size_mb: float


class PageSampler:
    """
    Sample random pages from PDF books for benchmarking.

    Avoids first/last N pages (TOC, index, appendices) for better
    representative sampling of actual content.
    """

    def __init__(
        self,
        books_dir: Path,
        seed: Optional[int] = None,
        skip_first: int = 5,
        skip_last: int = 5,
        min_pages: int = 20,
        temp_dir: Optional[Path] = None,
    ):
        """
        Args:
            books_dir: Directory containing PDF files
            seed: Random seed for reproducibility
            skip_first: Pages to skip at start (TOC, etc.)
            skip_last: Pages to skip at end (index, etc.)
            min_pages: Minimum pages required for a book to be sampled
            temp_dir: Directory for temporary single-page PDFs
        """
        self.books_dir = Path(books_dir)
        self.skip_first = skip_first
        self.skip_last = skip_last
        self.min_pages = min_pages
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "arca_benchmark"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            random.seed(seed)

        self._books: List[BookInfo] = []
        self._scanned = False

    def scan_books(self) -> List[BookInfo]:
        """Scan directory for valid PDF books."""
        if self._scanned:
            return self._books

        self._books = []
        pdf_files = list(self.books_dir.rglob("*.pdf"))

        for pdf_path in pdf_files:
            try:
                doc = fitz.open(str(pdf_path))
                page_count = len(doc)
                file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
                doc.close()

                # Check if book has enough pages for sampling
                usable_pages = page_count - self.skip_first - self.skip_last
                if usable_pages >= self.min_pages:
                    self._books.append(
                        BookInfo(
                            path=pdf_path,
                            name=pdf_path.stem,
                            total_pages=page_count,
                            file_size_mb=file_size_mb,
                        )
                    )
                else:
                    logger.debug(f"Skipping {pdf_path.name}: only {usable_pages} usable pages")

            except Exception as e:
                logger.warning(f"Failed to scan {pdf_path.name}: {e}")

        self._scanned = True
        logger.info(f"Found {len(self._books)} books with {self.min_pages}+ usable pages")
        return self._books

    def select_books(self, count: int) -> List[BookInfo]:
        """
        Select books for sampling with variety.

        Tries to select a mix of book sizes:
        - Large books (500+ pages)
        - Medium books (200-500 pages)
        - Small books (<200 pages)
        """
        if not self._scanned:
            self.scan_books()

        if count >= len(self._books):
            return self._books.copy()

        # Categorize by size
        large = [b for b in self._books if b.total_pages >= 500]
        medium = [b for b in self._books if 200 <= b.total_pages < 500]
        small = [b for b in self._books if b.total_pages < 200]

        selected = []

        # Try to get balanced mix
        target_per_category = max(1, count // 3)

        for category in [large, medium, small]:
            if category and len(selected) < count:
                pick_count = min(target_per_category, len(category), count - len(selected))
                selected.extend(random.sample(category, pick_count))

        # Fill remaining with random
        remaining = [b for b in self._books if b not in selected]
        if remaining and len(selected) < count:
            pick_count = count - len(selected)
            selected.extend(random.sample(remaining, min(pick_count, len(remaining))))

        return selected

    def sample_pages(self, book: BookInfo, count: int) -> List[SampledPage]:
        """
        Sample random pages from a book.

        Args:
            book: Book to sample from
            count: Number of pages to sample

        Returns:
            List of SampledPage objects with temp PDF paths
        """
        # Calculate valid page range (1-indexed)
        first_valid = self.skip_first + 1
        last_valid = book.total_pages - self.skip_last
        valid_range = range(first_valid, last_valid + 1)

        if len(valid_range) < count:
            # Sample all valid pages if not enough
            page_nums = list(valid_range)
        else:
            page_nums = sorted(random.sample(list(valid_range), count))

        samples = []
        doc = fitz.open(str(book.path))

        try:
            for page_num in page_nums:
                # Create single-page PDF
                temp_path = self.temp_dir / f"{book.name}_p{page_num}.pdf"

                single_page = fitz.open()
                # fitz uses 0-indexed pages
                single_page.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
                single_page.save(str(temp_path))
                single_page.close()

                samples.append(
                    SampledPage(
                        book_path=book.path,
                        book_name=book.name,
                        page_num=page_num,
                        total_pages=book.total_pages,
                        temp_pdf_path=temp_path,
                        metadata={
                            "file_size_mb": book.file_size_mb,
                            "relative_position": page_num / book.total_pages,
                        },
                    )
                )

        finally:
            doc.close()

        return samples

    def sample_all(self, num_books: int, pages_per_book: int) -> List[SampledPage]:
        """
        Sample pages from multiple books.

        Args:
            num_books: Number of books to sample
            pages_per_book: Pages to sample per book

        Returns:
            List of all sampled pages
        """
        books = self.select_books(num_books)
        all_samples = []

        for book in books:
            logger.info(f"Sampling {pages_per_book} pages from {book.name} ({book.total_pages} pages)")
            samples = self.sample_pages(book, pages_per_book)
            all_samples.extend(samples)

        logger.info(f"Total samples: {len(all_samples)} pages from {len(books)} books")
        return all_samples

    def cleanup_all(self, samples: List[SampledPage]):
        """Clean up all temporary files."""
        for sample in samples:
            sample.cleanup()

    def get_book_stats(self) -> Dict[str, Any]:
        """Get statistics about scanned books."""
        if not self._scanned:
            self.scan_books()

        if not self._books:
            return {"books": 0, "total_pages": 0, "avg_pages": 0}

        total_pages = sum(b.total_pages for b in self._books)
        total_size = sum(b.file_size_mb for b in self._books)

        return {
            "books": len(self._books),
            "total_pages": total_pages,
            "avg_pages": total_pages / len(self._books),
            "total_size_mb": total_size,
            "largest": max(self._books, key=lambda b: b.total_pages).name,
            "smallest": min(self._books, key=lambda b: b.total_pages).name,
        }
