"""
Readd QA - Phase 3: Extraction Quality Assurance

Validates extraction results, detects problems, triggers escalation.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter

from .extractors import ExtractionResult, ExtractedPage

logger = logging.getLogger(__name__)


@dataclass
class QAIssue:
    """A quality issue detected during QA"""

    severity: str  # "low", "medium", "high", "critical"
    issue_type: str  # Type of issue
    description: str  # Human-readable description
    page_nums: List[int]  # Affected pages
    sample: Optional[str]  # Sample of problematic content


@dataclass
class QAReport:
    """Quality assurance report for extraction"""

    passed: bool
    weirdness_score: int  # 0-100, higher = more problems
    issues: List[QAIssue]
    should_escalate: bool
    escalation_reason: Optional[str]

    # Stats
    total_chars: int
    empty_pages: int
    avg_chars_per_page: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "weirdness_score": self.weirdness_score,
            "should_escalate": self.should_escalate,
            "escalation_reason": self.escalation_reason,
            "issue_count": len(self.issues),
            "issues": [
                {
                    "severity": i.severity,
                    "type": i.issue_type,
                    "description": i.description,
                    "pages": i.page_nums,
                }
                for i in self.issues
            ],
            "stats": {
                "total_chars": self.total_chars,
                "empty_pages": self.empty_pages,
                "avg_chars_per_page": self.avg_chars_per_page,
            },
        }


class ExtractionQA:
    """
    Quality assurance for document extraction.

    Checks for common extraction problems and decides whether to escalate
    to a higher-quality (more expensive) extraction method.
    """

    # Thresholds
    WEIRDNESS_THRESHOLD = 55  # Above this, consider escalation (40 was too aggressive for large technical docs)
    CRITICAL_THRESHOLD = 75  # Above this, definitely escalate
    MIN_CHARS_PER_PAGE = 50  # Expect at least this many chars (technical books have figure-only pages)
    MAX_REPEATED_LINE_RATIO = 0.1  # Max ratio of repeated lines
    MAX_GARBLE_RATIO = 0.02  # Max ratio of garbled characters

    # Problem patterns
    GARBLE_CHARS = set("�□■●◆▪▫◊○")
    ENCODING_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

    def __init__(self, escalation_threshold: int = None):
        self.escalation_threshold = escalation_threshold or self.WEIRDNESS_THRESHOLD

    def check(self, result: ExtractionResult) -> QAReport:
        """
        Run quality checks on extraction result.

        Args:
            result: ExtractionResult to validate

        Returns:
            QAReport with issues and escalation recommendation
        """
        issues = []

        # Basic stats
        total_chars = sum(len(p.text) for p in result.pages)
        empty_pages = sum(1 for p in result.pages if len(p.text.strip()) < 50)
        avg_chars = total_chars / len(result.pages) if result.pages else 0

        # Run checks
        issues.extend(self._check_empty_pages(result.pages))
        issues.extend(self._check_encoding_issues(result.pages))
        issues.extend(self._check_garbled_text(result.pages))
        issues.extend(self._check_repeated_content(result.pages))
        issues.extend(self._check_missing_pages(result.pages, result.page_count))
        issues.extend(self._check_truncation(result.pages))
        issues.extend(self._check_table_artifacts(result.pages))

        # Calculate weirdness score
        weirdness_score = self._calculate_weirdness(issues, result)

        # Determine pass/fail and escalation
        critical_issues = [i for i in issues if i.severity == "critical"]
        high_issues = [i for i in issues if i.severity == "high"]

        passed = weirdness_score < self.escalation_threshold and not critical_issues

        should_escalate = (
            weirdness_score >= self.escalation_threshold or len(critical_issues) > 0 or len(high_issues) >= 3
        )

        escalation_reason = None
        if should_escalate:
            if critical_issues:
                escalation_reason = f"Critical issues: {critical_issues[0].description}"
            elif weirdness_score >= self.CRITICAL_THRESHOLD:
                escalation_reason = f"High weirdness score: {weirdness_score}"
            else:
                escalation_reason = f"{len(high_issues)} high-severity issue(s)"

        return QAReport(
            passed=passed,
            weirdness_score=weirdness_score,
            issues=issues,
            should_escalate=should_escalate,
            escalation_reason=escalation_reason,
            total_chars=total_chars,
            empty_pages=empty_pages,
            avg_chars_per_page=avg_chars,
        )

    def _check_empty_pages(self, pages: List[ExtractedPage]) -> List[QAIssue]:
        """Check for empty or nearly-empty pages"""
        issues = []
        empty_pages = []

        for page in pages:
            if len(page.text.strip()) < 50:
                empty_pages.append(page.page_num)

        if empty_pages:
            ratio = len(empty_pages) / len(pages)
            severity = "critical" if ratio > 0.5 else "high" if ratio > 0.2 else "medium"

            issues.append(
                QAIssue(
                    severity=severity,
                    issue_type="empty_pages",
                    description=f"{len(empty_pages)} of {len(pages)} pages are empty or nearly empty",
                    page_nums=empty_pages[:10],  # Limit to first 10
                    sample=None,
                )
            )

        return issues

    def _check_encoding_issues(self, pages: List[ExtractedPage]) -> List[QAIssue]:
        """Check for encoding/control character issues"""
        issues = []
        affected_pages = []
        sample = None

        for page in pages:
            matches = self.ENCODING_PATTERN.findall(page.text)
            if matches:
                affected_pages.append(page.page_num)
                if not sample:
                    # Get context around the issue
                    match = self.ENCODING_PATTERN.search(page.text)
                    if match:
                        start = max(0, match.start() - 20)
                        end = min(len(page.text), match.end() + 20)
                        sample = repr(page.text[start:end])

        if affected_pages:
            issues.append(
                QAIssue(
                    severity="high",
                    issue_type="encoding_issues",
                    description=f"Control/encoding characters found in {len(affected_pages)} pages",
                    page_nums=affected_pages[:10],
                    sample=sample,
                )
            )

        return issues

    def _check_garbled_text(self, pages: List[ExtractedPage]) -> List[QAIssue]:
        """Check for garbled/corrupted text"""
        issues = []

        all_text = "".join(p.text for p in pages)
        garble_count = sum(1 for c in all_text if c in self.GARBLE_CHARS)
        garble_ratio = garble_count / len(all_text) if all_text else 0

        if garble_ratio > self.MAX_GARBLE_RATIO:
            severity = "critical" if garble_ratio > 0.1 else "high"

            # Find affected pages
            affected_pages = []
            for page in pages:
                page_garble = sum(1 for c in page.text if c in self.GARBLE_CHARS)
                if page_garble > 0:
                    affected_pages.append(page.page_num)

            # Get sample
            sample = None
            for page in pages:
                for char in self.GARBLE_CHARS:
                    if char in page.text:
                        idx = page.text.index(char)
                        start = max(0, idx - 20)
                        end = min(len(page.text), idx + 20)
                        sample = page.text[start:end]
                        break
                if sample:
                    break

            issues.append(
                QAIssue(
                    severity=severity,
                    issue_type="garbled_text",
                    description=f"Garbled characters: {garble_ratio:.1%} of text",
                    page_nums=affected_pages[:10],
                    sample=sample,
                )
            )

        return issues

    def _check_repeated_content(self, pages: List[ExtractedPage]) -> List[QAIssue]:
        """Check for repeated headers/footers polluting text"""
        issues = []

        # Collect all lines across pages
        line_counts = Counter()
        for page in pages:
            lines = page.text.split("\n")
            for line in lines:
                line = line.strip()
                if len(line) > 10:  # Ignore short lines
                    line_counts[line] += 1

        # Find lines that appear on many pages (likely headers/footers)
        repeated_lines = [(line, count) for line, count in line_counts.items() if count > max(3, len(pages) * 0.3)]

        if repeated_lines:
            total_repeated = sum(count for _, count in repeated_lines)
            total_lines = sum(len(p.text.split("\n")) for p in pages)
            ratio = total_repeated / total_lines if total_lines else 0

            if ratio > self.MAX_REPEATED_LINE_RATIO:
                severity = "medium" if ratio < 0.2 else "high"

                issues.append(
                    QAIssue(
                        severity=severity,
                        issue_type="repeated_content",
                        description=f"{len(repeated_lines)} lines repeated across pages (header/footer pollution)",
                        page_nums=[],  # Affects all pages
                        sample=repeated_lines[0][0][:100] if repeated_lines else None,
                    )
                )

        return issues

    def _check_missing_pages(self, pages: List[ExtractedPage], expected_count: int) -> List[QAIssue]:
        """Check for missing pages"""
        issues = []

        if len(pages) < expected_count:
            missing = expected_count - len(pages)
            severity = "critical" if missing > expected_count * 0.1 else "high"

            issues.append(
                QAIssue(
                    severity=severity,
                    issue_type="missing_pages",
                    description=f"Missing {missing} pages (got {len(pages)}, expected {expected_count})",
                    page_nums=[],
                    sample=None,
                )
            )

        return issues

    def _check_truncation(self, pages: List[ExtractedPage]) -> List[QAIssue]:
        """Check for truncated text (sentences cut off)"""
        issues = []
        truncated_pages = []

        for page in pages:
            text = page.text.strip()
            if not text:
                continue

            # Check if page ends mid-sentence
            last_char = text[-1]
            if last_char not in ".!?\"')]}":
                # Could be truncated - check if it's mid-word
                last_word = text.split()[-1] if text.split() else ""
                if last_word and not last_word[-1].isalnum():
                    continue  # Ends with punctuation
                if len(last_word) < 3:
                    continue  # Probably just a short word
                truncated_pages.append(page.page_num)

        if len(truncated_pages) > len(pages) * 0.3:
            issues.append(
                QAIssue(
                    severity="medium",
                    issue_type="truncation",
                    description=f"{len(truncated_pages)} pages may have truncated text",
                    page_nums=truncated_pages[:10],
                    sample=None,
                )
            )

        return issues

    def _check_table_artifacts(self, pages: List[ExtractedPage]) -> List[QAIssue]:
        """Check for poorly extracted tables"""
        issues = []
        affected_pages = []

        # Patterns that suggest table extraction issues
        table_patterns = [
            re.compile(r"\|\s*\|\s*\|"),  # Empty table cells
            re.compile(r"(\d+\s+){5,}"),  # Lots of numbers with no context
            re.compile(r"[-+|]{10,}"),  # Table borders
        ]

        for page in pages:
            for pattern in table_patterns:
                if pattern.search(page.text):
                    affected_pages.append(page.page_num)
                    break

        if affected_pages:
            issues.append(
                QAIssue(
                    severity="low",
                    issue_type="table_artifacts",
                    description=f"Potential table extraction artifacts in {len(affected_pages)} pages",
                    page_nums=affected_pages[:10],
                    sample=None,
                )
            )

        return issues

    def _calculate_weirdness(self, issues: List[QAIssue], result: ExtractionResult) -> int:
        """Calculate overall weirdness score"""
        score = 0

        # Score by severity
        severity_weights = {
            "critical": 30,
            "high": 15,
            "medium": 8,
            "low": 3,
        }

        for issue in issues:
            score += severity_weights.get(issue.severity, 5)

        # Additional factors
        avg_chars = result.metadata.get("total_chars", 0) / result.page_count if result.page_count else 0
        if avg_chars < self.MIN_CHARS_PER_PAGE:
            score += 20

        return min(100, score)


def validate_extraction(result: ExtractionResult) -> QAReport:
    """Convenience function to validate an extraction"""
    qa = ExtractionQA()
    return qa.check(result)
