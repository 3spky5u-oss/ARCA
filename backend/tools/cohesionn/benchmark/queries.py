"""
Benchmark Query Battery
=======================
Generic retrieval benchmark queries across 4 difficulty tiers.
Designed to stress-test RAG pipeline components without domain specificity.

Domain packs can inject additional queries via lexicon pipeline config
key "benchmark_queries" (list of dicts with query/tier/expect_keywords/
expect_entities/difficulty).

Pure Python — no ARCA imports. Safe to extract to standalone package.

Tiers:
  factual (3)       — Direct facts, should hit text chunks
  conceptual (3)    — Broad topics, benefits from RAPTOR summaries
  multi_hop (2)     — Requires synthesizing multiple chunks
  negation (2)      — "not", "except", "without" — tests reranker nuance
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BenchmarkQuery:
    """Single benchmark query with expected retrieval targets."""

    id: str
    query: str
    tier: str
    expect_keywords: List[str] = field(default_factory=list)
    expect_entities: List[str] = field(default_factory=list)
    difficulty: int = 1  # 1=easy, 2=medium, 3=hard

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "query": self.query,
            "tier": self.tier,
            "expect_keywords": self.expect_keywords,
            "expect_entities": self.expect_entities,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BenchmarkQuery":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# -- Tier Definitions ---------------------------------------------------------

TIERS = {
    "factual": "Direct factual recall from text chunks",
    "conceptual": "Broad concepts requiring summary-level understanding",
    "multi_hop": "Requires synthesizing information across multiple chunks",
    "negation": "Queries with negation -- tests reranker discrimination",
}


# -- Generic Query Battery (domain-agnostic) ----------------------------------

_DEFAULT_QUERIES: List[BenchmarkQuery] = [
    # -- Tier 1: Factual (3) --------------------------------------------------
    BenchmarkQuery(
        id="gen_fact_01",
        query="What methodology was used in this analysis?",
        tier="factual",
        expect_keywords=["methodology", "method", "approach", "procedure"],
        expect_entities=[],
        difficulty=1,
    ),
    BenchmarkQuery(
        id="gen_fact_02",
        query="What standards or regulations were referenced?",
        tier="factual",
        expect_keywords=["standard", "regulation", "code", "reference", "compliance"],
        expect_entities=[],
        difficulty=1,
    ),
    BenchmarkQuery(
        id="gen_fact_03",
        query="List the equipment or instruments mentioned in the documents",
        tier="factual",
        expect_keywords=["equipment", "instrument", "device", "tool", "apparatus"],
        expect_entities=[],
        difficulty=1,
    ),
    # -- Tier 2: Conceptual (3) ------------------------------------------------
    BenchmarkQuery(
        id="gen_concept_01",
        query="Summarize the key findings from the report",
        tier="conceptual",
        expect_keywords=["finding", "result", "conclusion", "summary", "key"],
        expect_entities=[],
        difficulty=2,
    ),
    BenchmarkQuery(
        id="gen_concept_02",
        query="Describe the testing procedures used",
        tier="conceptual",
        expect_keywords=["test", "procedure", "protocol", "step", "process"],
        expect_entities=[],
        difficulty=2,
    ),
    BenchmarkQuery(
        id="gen_concept_03",
        query="Explain the data collection process",
        tier="conceptual",
        expect_keywords=["data", "collection", "sampling", "measurement", "record"],
        expect_entities=[],
        difficulty=2,
    ),
    # -- Tier 3: Multi-Hop (2) -------------------------------------------------
    BenchmarkQuery(
        id="gen_hop_01",
        query="Compare the results across different test conditions",
        tier="multi_hop",
        expect_keywords=["compare", "result", "condition", "difference", "test"],
        expect_entities=[],
        difficulty=3,
    ),
    BenchmarkQuery(
        id="gen_hop_02",
        query="What safety factors were applied and how were they justified?",
        tier="multi_hop",
        expect_keywords=["safety", "factor", "justification", "margin", "risk"],
        expect_entities=[],
        difficulty=3,
    ),
    # -- Tier 4: Negation (2) --------------------------------------------------
    BenchmarkQuery(
        id="gen_neg_01",
        query="What limitations were identified in the study?",
        tier="negation",
        expect_keywords=["limitation", "constraint", "restriction", "scope", "caveat"],
        expect_entities=[],
        difficulty=2,
    ),
    BenchmarkQuery(
        id="gen_neg_02",
        query="Which methods were NOT recommended for this application?",
        tier="negation",
        expect_keywords=["not", "unsuitable", "inappropriate", "avoid", "limitation"],
        expect_entities=[],
        difficulty=3,
    ),
]


def _load_domain_queries() -> List[BenchmarkQuery]:
    """Load domain-specific benchmark queries from lexicon pipeline config."""
    try:
        from domain_loader import get_pipeline_config
        raw = get_pipeline_config().get("benchmark_queries", [])
        return [BenchmarkQuery.from_dict(q) for q in raw]
    except Exception:
        return []


def get_benchmark_queries() -> List[BenchmarkQuery]:
    """Get all benchmark queries (generic defaults + domain-specific)."""
    domain_queries = _load_domain_queries()
    return _DEFAULT_QUERIES + domain_queries


# For backward compatibility, expose as module-level list
# (lazy-loaded on first access to avoid import-time domain_loader call)
_all_queries = None


def _get_all():
    global _all_queries
    if _all_queries is None:
        _all_queries = get_benchmark_queries()
    return _all_queries


# Keep BENCHMARK_QUERIES as the public API but make it a property-like accessor
class _QueryList:
    """Lazy list wrapper that loads domain queries on first access."""

    def __iter__(self):
        return iter(_get_all())

    def __len__(self):
        return len(_get_all())

    def __getitem__(self, idx):
        return _get_all()[idx]

    def __repr__(self):
        return repr(_get_all())


BENCHMARK_QUERIES = _QueryList()


def get_queries_by_tier(tier: str) -> List[BenchmarkQuery]:
    """Return queries for a specific tier."""
    return [q for q in _get_all() if q.tier == tier]


def get_all_tiers() -> List[str]:
    """Return all tier names in order."""
    return list(TIERS.keys())


def load_custom_queries(path: str) -> List[BenchmarkQuery]:
    """Load queries from a JSON file."""
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text())
    return [BenchmarkQuery.from_dict(q) for q in data]
