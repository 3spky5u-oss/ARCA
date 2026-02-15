"""Pytest fixtures for Readd tests."""

import json
import pytest
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
QA_DIR = BASE_DIR / "data" / "qa_reference"


@pytest.fixture(scope="session")
def qa_manifest():
    """Load the QA reference manifest."""
    manifest_path = QA_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("QA manifest not found - run generate_test_docs.py first")
    return json.loads(manifest_path.read_text())


@pytest.fixture(scope="session")
def qa_docs_dir():
    """Path to QA test documents."""
    return QA_DIR / "docs"


@pytest.fixture(scope="session")
def qa_expected_dir():
    """Path to expected ground truth files."""
    return QA_DIR / "expected"


@pytest.fixture
def levenshtein_ratio():
    """Helper function to compute Levenshtein ratio between strings."""

    def compute(s1: str, s2: str) -> float:
        """Compute Levenshtein ratio (0-1, higher = more similar)."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Normalize whitespace
        s1 = " ".join(s1.split())
        s2 = " ".join(s2.split())

        len1, len2 = len(s1), len(s2)

        # Dynamic programming approach
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1

        # Use only two rows for space efficiency
        prev_row = list(range(len2 + 1))
        curr_row = [0] * (len2 + 1)

        for i, c1 in enumerate(s1, 1):
            curr_row[0] = i
            for j, c2 in enumerate(s2, 1):
                cost = 0 if c1 == c2 else 1
                curr_row[j] = min(
                    curr_row[j - 1] + 1, prev_row[j] + 1, prev_row[j - 1] + cost  # Insert  # Delete  # Replace
                )
            prev_row, curr_row = curr_row, prev_row

        distance = prev_row[len2]
        max_len = max(len1, len2)

        return 1 - (distance / max_len)

    return compute


@pytest.fixture
def word_accuracy():
    """Helper function to compute word-level accuracy (Jaccard similarity)."""

    def compute(s1: str, s2: str) -> float:
        """Compute word overlap as Jaccard similarity."""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    return compute
