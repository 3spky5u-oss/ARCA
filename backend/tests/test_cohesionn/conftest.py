"""
Shared pytest fixtures for Cohesionn RAG pipeline tests.

Provides:
- Mock embedders/rerankers for unit tests (fast, no GPU)
- Real component fixtures for integration tests
- Test content fixtures with specialized and generic content
- Quality validation helpers
"""

import pytest
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path


# =============================================================================
# Test Content Fixtures
# =============================================================================

# Specialized content that should rank HIGH for domain-specific queries
DOMAIN_CONTENT = {
    "spt_n60": """Load Testing Methodology: Standard procedures for evaluating structural capacity.
Method-A threshold values are recorded as performance indices normalized to reference conditions.
For high-performance systems, typical threshold values range from 30 to 50 units per cycle.
Systems with threshold > 30 indicate adequate load capacity and low deformation risk.
The correction formula is: V60 = Vm × (Em/60) where Vm is the measured value.""",

    "bearing_capacity": """Capacity Analysis: Analytical methods for determining maximum allowable loads.
The capacity equation calculates ultimate load: Qu = c×Nc + γ×Df×Nq + 0.5×γ×B×Nγ
The factor of safety for load capacity is typically FS = 3.0 for normal conditions.
For temporary installations or with verification tests, FS = 2.5 may be acceptable.
Capacity factors Nc, Nq, Nγ depend on the material friction angle φ.""",

    "pile_load_test": """Component Testing: Field verification procedures for installed elements.
Test procedure applies load in increments of 25% of design load.
Each load increment is held for minimum 1 hour to observe settlement.
Testing continues to 200% of design load or until failure criteria is met.
Settlement at design load should not exceed 25mm for acceptance.""",

    "frost_depth": """Environmental Factor Analysis: Seasonal variation effects on material properties.
In northern regions, maximum affected depths range from 3.0 to 4.0 meters.
Factors affecting depth include: Temperature (thermal index),
Insulation Cover (protective effect), Material Type (thermal conductivity), and Moisture Content.
Installations must extend below maximum affected depth to prevent damage.""",
}

# Generic LRFD content that should rank LOWER for domain-specific queries
GENERIC_CONTENT = {
    "lrfd_factors": """Load and Resistance Factor Design (LRFD) uses factored loads and resistances.
Typical load factors are: 1.25 for dead load, 1.5 for live load, 1.6 for wind.
Resistance factors range from 0.5 to 0.8 depending on material uncertainty.
LRFD provides a more rational approach to structural safety than allowable stress design.""",

    "lrfd_combinations": """LRFD load combinations include various scenarios:
1.4D (dead load only)
1.2D + 1.6L (dead plus live)
1.2D + 1.0L + 1.0W (dead, live, wind)
Each combination represents different loading conditions for structural design.""",

    "structural_steel": """Structural steel design follows AISC specifications.
Steel grades include A36 (Fy=36 ksi) and A992 (Fy=50 ksi).
Connection design uses bolts or welds with appropriate resistance factors.
Member design considers flexure, shear, and axial load interactions.""",
}

# Validation queries with expected rankings
RANKING_VALIDATION_QUERIES = [
    {
        "query": "What are typical threshold values for high-performance systems?",
        "should_prefer": ["spt_n60"],
        "should_not_prefer": ["lrfd_factors", "lrfd_combinations", "structural_steel"],
        "expected_phrases": ["30", "50", "threshold", "high-performance"],
    },
    {
        "query": "load capacity safety factor analysis",
        "should_prefer": ["bearing_capacity"],
        "should_not_prefer": ["lrfd_factors", "structural_steel"],
        "expected_phrases": ["3.0", "FS", "factor of safety"],
    },
    {
        "query": "component testing procedures settlement",
        "should_prefer": ["pile_load_test"],
        "should_not_prefer": ["lrfd_factors", "structural_steel"],
        "expected_phrases": ["25%", "settlement", "increments"],
    },
    {
        "query": "environmental factor depth seasonal variation",
        "should_prefer": ["frost_depth"],
        "should_not_prefer": ["lrfd_factors", "structural_steel"],
        "expected_phrases": ["3.0", "4.0", "depth"],
    },
]


@pytest.fixture
def domain_content() -> Dict[str, str]:
    """Specialized domain content."""
    return DOMAIN_CONTENT.copy()


@pytest.fixture
def generic_content() -> Dict[str, str]:
    """Generic LRFD/structural content that should rank lower."""
    return GENERIC_CONTENT.copy()


@pytest.fixture
def all_test_content() -> Dict[str, str]:
    """All test content combined."""
    return {**DOMAIN_CONTENT, **GENERIC_CONTENT}


@pytest.fixture
def ranking_validation_queries() -> List[Dict[str, Any]]:
    """Validation queries with expected rankings."""
    return RANKING_VALIDATION_QUERIES.copy()


# =============================================================================
# Mock Embedder (for fast unit tests)
# =============================================================================

class MockEmbedder:
    """Mock embedder that returns deterministic vectors based on content hash."""

    def __init__(self, dimension: int = 1024):
        self._dimension = dimension
        self._cache = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_family(self) -> str:
        return "mock"

    @property
    def cache_stats(self) -> dict:
        return {"hits": 0, "misses": len(self._cache), "size": len(self._cache), "maxsize": 10000, "hit_rate": "0.0%"}

    def _text_to_vector(self, text: str) -> List[float]:
        """Generate deterministic vector from text hash."""
        if text in self._cache:
            return self._cache[text]

        # Use hash to seed random for reproducibility
        seed = hash(text) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dimension).astype(np.float32)
        # Normalize
        vec = vec / np.linalg.norm(vec)
        self._cache[text] = vec.tolist()
        return self._cache[text]

    def embed_query(self, query: str) -> List[float]:
        return self._text_to_vector(f"query:{query}")

    def embed_document(self, document: str) -> List[float]:
        return self._text_to_vector(f"doc:{document}")

    def embed_documents(self, documents: List[str], batch_size: int = 64) -> List[List[float]]:
        return [self.embed_document(doc) for doc in documents]

    def embed_queries(self, queries: List[str], batch_size: int = 64) -> List[List[float]]:
        return [self.embed_query(q) for q in queries]

    def clear_cache(self):
        self._cache.clear()


@pytest.fixture
def mock_embedder():
    """Fast mock embedder for unit tests."""
    return MockEmbedder()


# =============================================================================
# Mock Reranker (for unit tests)
# =============================================================================

class MockReranker:
    """Mock reranker that uses simple keyword matching for scoring."""

    def __init__(self):
        self.model_name = "mock-reranker"

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Rerank using simple keyword overlap scoring."""
        if not results:
            return []

        query_words = set(query.lower().split())

        for result in results:
            content = result.get("content", "").lower()
            content_words = set(content.split())

            # Simple overlap scoring
            overlap = len(query_words & content_words)
            # Normalize by query length
            score = overlap / max(len(query_words), 1)
            result["rerank_score"] = score

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]


@pytest.fixture
def mock_reranker():
    """Fast mock reranker for unit tests."""
    return MockReranker()


# =============================================================================
# Real Component Fixtures (for integration tests - require GPU/models)
# =============================================================================

@pytest.fixture(scope="session")
def real_embedder():
    """Real embedder - requires models to be available."""
    try:
        from tools.cohesionn.embeddings import get_embedder
        embedder = get_embedder()
        # Verify it works
        _ = embedder.dimension
        return embedder
    except Exception as e:
        pytest.skip(f"Real embedder not available: {e}")


@pytest.fixture(scope="session")
def real_reranker():
    """Real reranker - requires models to be available."""
    try:
        from tools.cohesionn.reranker import get_reranker
        reranker = get_reranker()
        # Verify it works by checking model loads
        if reranker.model == "fallback":
            pytest.skip("Reranker in fallback mode")
        return reranker
    except Exception as e:
        pytest.skip(f"Real reranker not available: {e}")


@pytest.fixture(scope="session")
def real_diversity_reranker():
    """Real diversity reranker."""
    try:
        from tools.cohesionn.reranker import DiversityReranker
        return DiversityReranker(lambda_param=0.6, max_per_source=2)
    except Exception as e:
        pytest.skip(f"Diversity reranker not available: {e}")


# =============================================================================
# Qdrant Test Fixtures
# =============================================================================

@pytest.fixture
def qdrant_client():
    """In-memory Qdrant client for testing."""
    from qdrant_client import QdrantClient
    return QdrantClient(":memory:")


@pytest.fixture
def qdrant_collection(qdrant_client):
    """Qdrant collection for testing with proper configuration."""
    from qdrant_client.models import Distance, VectorParams

    collection_name = "test_cohesionn"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,  # Match Qwen3-Embedding-0.6B dimension
            distance=Distance.COSINE,
        ),
    )
    return collection_name


@pytest.fixture
def populated_qdrant(qdrant_client, qdrant_collection, all_test_content, mock_embedder):
    """Qdrant collection populated with test content."""
    from qdrant_client.models import PointStruct
    import hashlib

    points = []
    for idx, (key, content) in enumerate(all_test_content.items()):
        embedding = mock_embedder.embed_document(content)
        # Generate integer ID from key hash (Qdrant requires int or UUID)
        point_id = int(hashlib.md5(key.encode()).hexdigest()[:16], 16)
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": content,
                    "topic": "test",
                    "source": f"test_{key}.md",
                    "chunk_id": key,
                },
            )
        )

    qdrant_client.upsert(
        collection_name=qdrant_collection,
        points=points,
        wait=True,
    )

    return qdrant_client, qdrant_collection


# =============================================================================
# BM25 Test Fixtures
# =============================================================================

@pytest.fixture
def bm25_index(tmp_path):
    """Fresh BM25 index for testing."""
    from tools.cohesionn.sparse_retrieval import BM25Index
    return BM25Index(topic="test", persist_dir=tmp_path)


@pytest.fixture
def populated_bm25_index(bm25_index, all_test_content):
    """BM25 index populated with test content."""
    doc_ids = list(all_test_content.keys())
    documents = list(all_test_content.values())
    metadatas = [{"source": f"test_{k}.md", "topic": "test"} for k in all_test_content.keys()]

    bm25_index.add_documents(doc_ids, documents, metadatas)
    return bm25_index


# =============================================================================
# Test Result Builders
# =============================================================================

def build_test_results(content_dict: Dict[str, str], scores: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Build test results from content dictionary."""
    scores = scores or {}
    results = []

    for key, content in content_dict.items():
        results.append({
            "id": key,
            "content": content,
            "score": scores.get(key, 0.5),
            "metadata": {
                "source": f"test_{key}.md",
                "topic": "test",
            }
        })

    return results


@pytest.fixture
def test_results(all_test_content):
    """Pre-built test results from all content."""
    return build_test_results(all_test_content)


# =============================================================================
# Quality Validation Helpers
# =============================================================================

def check_ranking_preference(
    results: List[Dict[str, Any]],
    preferred_ids: List[str],
    non_preferred_ids: List[str],
    score_key: str = "rerank_score",
) -> Dict[str, Any]:
    """
    Check if results correctly rank preferred content above non-preferred.

    Returns dict with:
    - passed: bool
    - preferred_scores: dict of id -> score
    - non_preferred_scores: dict of id -> score
    - min_preferred: float
    - max_non_preferred: float
    - margin: float (min_preferred - max_non_preferred)
    """
    id_to_score = {r["id"]: r.get(score_key, r.get("score", 0)) for r in results}

    preferred_scores = {pid: id_to_score.get(pid, 0) for pid in preferred_ids if pid in id_to_score}
    non_preferred_scores = {nid: id_to_score.get(nid, 0) for nid in non_preferred_ids if nid in id_to_score}

    min_preferred = min(preferred_scores.values()) if preferred_scores else 0
    max_non_preferred = max(non_preferred_scores.values()) if non_preferred_scores else 0

    margin = min_preferred - max_non_preferred
    passed = margin > 0  # Preferred should score higher

    return {
        "passed": passed,
        "preferred_scores": preferred_scores,
        "non_preferred_scores": non_preferred_scores,
        "min_preferred": min_preferred,
        "max_non_preferred": max_non_preferred,
        "margin": margin,
    }


def check_expected_phrases(content: str, expected_phrases: List[str]) -> Dict[str, Any]:
    """
    Check if content contains expected phrases.

    Returns dict with:
    - passed: bool (50%+ phrases found)
    - found: list of found phrases
    - missing: list of missing phrases
    - match_ratio: float
    """
    content_lower = content.lower()
    found = []
    missing = []

    for phrase in expected_phrases:
        if phrase.lower() in content_lower:
            found.append(phrase)
        else:
            missing.append(phrase)

    match_ratio = len(found) / len(expected_phrases) if expected_phrases else 1.0

    return {
        "passed": match_ratio >= 0.5,
        "found": found,
        "missing": missing,
        "match_ratio": match_ratio,
    }


@pytest.fixture
def check_ranking():
    """Helper function to check ranking preference."""
    return check_ranking_preference


@pytest.fixture
def check_phrases():
    """Helper function to check expected phrases."""
    return check_expected_phrases


# =============================================================================
# Tolerance Settings
# =============================================================================

# Score margin thresholds for quality tests
SCORE_MARGIN_THRESHOLD = 0.05  # Minimum score difference for "correct" ranking
QUALITY_PASS_THRESHOLD = 0.5   # Minimum phrase match ratio to pass


@pytest.fixture
def score_margin_threshold():
    """Minimum score margin for ranking tests."""
    return SCORE_MARGIN_THRESHOLD


@pytest.fixture
def quality_pass_threshold():
    """Minimum pass threshold for quality tests."""
    return QUALITY_PASS_THRESHOLD
