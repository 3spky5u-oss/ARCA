"""
Benchmark Report Generator

Creates JSON and console reports from benchmark results.
"""

import json
import statistics
from math import sqrt
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class PageResult:
    """Result from benchmarking a single page."""

    book_name: str
    page_num: int
    total_ms: float
    stages: Dict[str, float]  # stage_name -> ms
    extractor_used: str
    page_type: Optional[str] = None
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BookExtrapolation:
    """Projected ingestion time for a full book."""

    book_name: str
    total_pages: int
    sampled_pages: int
    avg_ms_per_page: float
    stddev_ms: float
    projected_minutes: float
    confidence_95_low: float
    confidence_95_high: float


@dataclass
class StageStats:
    """Statistics for a single ingestion stage."""

    name: str
    avg_ms: float
    stddev_ms: float
    min_ms: float
    max_ms: float
    pct_of_total: float
    count: int


@dataclass
class ExtractorStats:
    """Statistics for a specific extractor."""

    name: str
    pages: int
    avg_ms: float
    stddev_ms: float


class BenchmarkReport:
    """
    Collects benchmark results and generates reports.
    """

    def __init__(
        self,
        num_books: int,
        pages_per_book: int,
        seed: Optional[int] = None,
        mode: str = "knowledge_base",
    ):
        self.metadata = {
            "books": num_books,
            "pages_per_book": pages_per_book,
            "seed": seed,
            "mode": mode,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
        }
        self.page_results: List[PageResult] = []
        self.book_page_counts: Dict[str, int] = {}  # book_name -> total_pages

    def add_page_result(self, result: PageResult, total_pages: Optional[int] = None):
        """Add a page benchmark result."""
        self.page_results.append(result)
        if total_pages and result.book_name not in self.book_page_counts:
            self.book_page_counts[result.book_name] = total_pages

    def finish(self):
        """Mark benchmark as complete."""
        self.metadata["completed_at"] = datetime.now().isoformat()

    def _get_stage_stats(self) -> List[StageStats]:
        """Calculate statistics for each stage."""
        stage_times: Dict[str, List[float]] = {}

        for result in self.page_results:
            if result.error:
                continue
            for stage, ms in result.stages.items():
                if stage not in stage_times:
                    stage_times[stage] = []
                stage_times[stage].append(ms)

        total_avg = sum(self._safe_mean(times) for times in stage_times.values())

        stats = []
        for name, times in stage_times.items():
            avg = self._safe_mean(times)
            stats.append(
                StageStats(
                    name=name,
                    avg_ms=avg,
                    stddev_ms=self._safe_stdev(times),
                    min_ms=min(times) if times else 0,
                    max_ms=max(times) if times else 0,
                    pct_of_total=(avg / total_avg * 100) if total_avg > 0 else 0,
                    count=len(times),
                )
            )

        # Sort by percentage of total (descending)
        stats.sort(key=lambda s: s.pct_of_total, reverse=True)
        return stats

    def _get_extractor_stats(self) -> List[ExtractorStats]:
        """Calculate statistics per extractor."""
        extractor_times: Dict[str, List[float]] = {}

        for result in self.page_results:
            if result.error:
                continue
            ext = result.extractor_used
            if ext not in extractor_times:
                extractor_times[ext] = []
            extractor_times[ext].append(result.total_ms)

        stats = []
        for name, times in extractor_times.items():
            stats.append(
                ExtractorStats(
                    name=name,
                    pages=len(times),
                    avg_ms=self._safe_mean(times),
                    stddev_ms=self._safe_stdev(times),
                )
            )

        return stats

    def _extrapolate_book(self, book_name: str) -> Optional[BookExtrapolation]:
        """Extrapolate full book ingestion time."""
        book_results = [r for r in self.page_results if r.book_name == book_name and not r.error]

        if not book_results:
            return None

        total_pages = self.book_page_counts.get(book_name, 0)
        if total_pages == 0:
            return None

        times = [r.total_ms for r in book_results]
        avg_ms = self._safe_mean(times)
        stddev = self._safe_stdev(times)
        n = len(times)

        # Project to full book
        projected_ms = avg_ms * total_pages
        projected_minutes = projected_ms / 60000

        # 95% confidence interval using t-distribution approximation
        # For n>30, t ≈ 1.96; for smaller n, use conservative 2.0
        t_value = 2.0 if n < 30 else 1.96
        margin_ms = t_value * stddev / sqrt(n) * total_pages if n > 1 else stddev * total_pages
        margin_minutes = margin_ms / 60000

        return BookExtrapolation(
            book_name=book_name,
            total_pages=total_pages,
            sampled_pages=n,
            avg_ms_per_page=avg_ms,
            stddev_ms=stddev,
            projected_minutes=projected_minutes,
            confidence_95_low=max(0, projected_minutes - margin_minutes),
            confidence_95_high=projected_minutes + margin_minutes,
        )

    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks and recommendations."""
        bottlenecks = []
        stage_stats = self._get_stage_stats()
        extractor_stats = self._get_extractor_stats()

        # Check for dominant stage
        if stage_stats and stage_stats[0].pct_of_total > 70:
            top = stage_stats[0]
            bottlenecks.append(f"{top.name.capitalize()} dominates ({top.pct_of_total:.1f}%) - primary optimization target")

        # Check vision vs text extractor ratio
        vision_stats = next((e for e in extractor_stats if "vision" in e.name.lower()), None)
        text_stats = next((e for e in extractor_stats if "pymupdf" in e.name.lower()), None)

        if vision_stats and text_stats and text_stats.avg_ms > 0:
            ratio = vision_stats.avg_ms / text_stats.avg_ms
            if ratio > 5:
                bottlenecks.append(
                    f"Vision pages are {ratio:.0f}x slower than text pages - "
                    "review classifier thresholds to reduce false positives"
                )

        # Check for high variance
        total_times = [r.total_ms for r in self.page_results if not r.error]
        if total_times:
            cv = self._safe_stdev(total_times) / self._safe_mean(total_times) if self._safe_mean(total_times) > 0 else 0
            if cv > 0.5:
                bottlenecks.append(
                    f"High timing variance (CV={cv:.2f}) - page complexity varies significantly"
                )

        return bottlenecks

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        stage_stats = self._get_stage_stats()
        extractor_stats = self._get_extractor_stats()

        # Extraction recommendations
        extraction_stat = next((s for s in stage_stats if s.name == "extraction"), None)
        if extraction_stat and extraction_stat.pct_of_total > 60:
            vision_count = sum(1 for r in self.page_results if "vision" in r.extractor_used.lower())
            total_count = len([r for r in self.page_results if not r.error])
            if total_count > 0:
                vision_pct = vision_count / total_count * 100
                if vision_pct > 30:
                    recommendations.append(
                        f"Vision extraction used for {vision_pct:.0f}% of pages - "
                        "tune classifier thresholds to reduce unnecessary vision calls"
                    )

        # Embedding recommendations
        embedding_stat = next((s for s in stage_stats if s.name == "embedding"), None)
        if embedding_stat and embedding_stat.avg_ms > 500:
            recommendations.append("Consider increasing embedding batch_size (64 -> 128) for better GPU utilization")

        # Indexing recommendations
        indexing_stat = next((s for s in stage_stats if s.name == "indexing"), None)
        if indexing_stat and indexing_stat.avg_ms > 200:
            recommendations.append("Indexing latency is high - consider batching Qdrant upserts")

        # Classification recommendations
        classification_stat = next((s for s in stage_stats if s.name == "classification"), None)
        if classification_stat and classification_stat.avg_ms > 50:
            recommendations.append("Page classification is slow - consider caching or simplifying detection logic")

        if not recommendations:
            recommendations.append("No obvious optimization targets - pipeline is well-balanced")

        return recommendations

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        successful = [r for r in self.page_results if not r.error]
        total_ms = sum(r.total_ms for r in successful)

        stage_stats = self._get_stage_stats()
        extractor_stats = self._get_extractor_stats()
        extrapolations = [
            self._extrapolate_book(name)
            for name in self.book_page_counts.keys()
        ]
        extrapolations = [e for e in extrapolations if e]

        return {
            "metadata": self.metadata,
            "summary": {
                "total_pages": len(self.page_results),
                "successful_pages": len(successful),
                "failed_pages": len(self.page_results) - len(successful),
                "total_time_s": total_ms / 1000,
                "avg_page_ms": self._safe_mean([r.total_ms for r in successful]),
                "stages": {
                    s.name: {
                        "avg_ms": round(s.avg_ms, 1),
                        "pct": round(s.pct_of_total, 1),
                        "stddev_ms": round(s.stddev_ms, 1),
                    }
                    for s in stage_stats
                },
            },
            "by_extractor": {
                e.name: {
                    "pages": e.pages,
                    "avg_ms": round(e.avg_ms, 1),
                    "stddev_ms": round(e.stddev_ms, 1),
                }
                for e in extractor_stats
            },
            "extrapolations": [
                {
                    "book": e.book_name,
                    "pages": e.total_pages,
                    "sampled": e.sampled_pages,
                    "projected_min": round(e.projected_minutes, 1),
                    "ci_95": [round(e.confidence_95_low, 1), round(e.confidence_95_high, 1)],
                }
                for e in extrapolations
            ],
            "bottlenecks": self._identify_bottlenecks(),
            "recommendations": self._generate_recommendations(),
            "page_results": [
                {
                    "book": r.book_name,
                    "page": r.page_num,
                    "total_ms": round(r.total_ms, 1),
                    "extractor": r.extractor_used,
                    "stages": {k: round(v, 1) for k, v in r.stages.items()},
                    "error": r.error,
                }
                for r in self.page_results
            ],
        }

    def save_json(self, path: Path):
        """Save report to JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        return statistics.mean(values) if values else 0.0

    @staticmethod
    def _safe_stdev(values: List[float]) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0


def generate_console_report(report: BenchmarkReport) -> str:
    """Generate human-readable console report."""
    data = report.to_dict()
    lines = []

    # Header
    lines.append("")
    lines.append("INGESTION BENCHMARK REPORT")
    lines.append("=" * 60)
    meta = data["metadata"]
    lines.append(f"Books: {meta['books']} | Pages: {data['summary']['total_pages']} | Mode: {meta['mode']}")
    if meta.get("seed"):
        lines.append(f"Seed: {meta['seed']}")
    lines.append("")

    # Summary
    summary = data["summary"]
    lines.append(f"Total time: {summary['total_time_s']:.1f}s | Avg per page: {summary['avg_page_ms']:.0f}ms")
    if summary["failed_pages"] > 0:
        lines.append(f"Failed: {summary['failed_pages']} pages")
    lines.append("")

    # Stage breakdown
    lines.append("STAGE BREAKDOWN")
    lines.append("-" * 60)
    lines.append(f"{'Stage':<20} {'Avg (ms)':>12} {'% Total':>10} {'StdDev':>10}")
    lines.append("-" * 60)

    for name, stats in summary["stages"].items():
        lines.append(f"{name.capitalize():<20} {stats['avg_ms']:>12,.0f} {stats['pct']:>9.1f}% {stats['stddev_ms']:>10,.0f}")

    lines.append("")

    # Extractor comparison
    if data["by_extractor"]:
        lines.append("EXTRACTOR COMPARISON")
        lines.append("-" * 60)
        lines.append(f"{'Extractor':<25} {'Pages':>8} {'Avg (ms)':>12} {'StdDev':>10}")
        lines.append("-" * 60)

        for name, stats in data["by_extractor"].items():
            lines.append(f"{name:<25} {stats['pages']:>8} {stats['avg_ms']:>12,.0f} {stats['stddev_ms']:>10,.0f}")

        lines.append("")

    # Extrapolations
    if data["extrapolations"]:
        lines.append("EXTRAPOLATED BOOK TIMES")
        lines.append("-" * 60)

        for ext in data["extrapolations"]:
            ci_low, ci_high = ext["ci_95"]
            lines.append(
                f"{ext['book'][:40]:<40} ({ext['pages']:>4} pages)"
            )
            lines.append(
                f"  ~{ext['projected_min']:.0f} min (95% CI: {ci_low:.0f}-{ci_high:.0f})"
            )

        lines.append("")

    # Bottlenecks
    if data["bottlenecks"]:
        lines.append("BOTTLENECKS")
        lines.append("-" * 60)
        for b in data["bottlenecks"]:
            lines.append(f"! {b}")
        lines.append("")

    # Recommendations
    if data["recommendations"]:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 60)
        for r in data["recommendations"]:
            lines.append(f"-> {r}")
        lines.append("")

    return "\n".join(lines)
