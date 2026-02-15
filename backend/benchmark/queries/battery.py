"""
Extended Benchmark Query — adds ground truth answer and source project tracking.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkQuery:
    """Benchmark query with ground truth for LLM-as-judge evaluation.

    Extends the base BenchmarkQuery pattern with answer evaluation fields.
    """

    id: str
    query: str
    tier: str  # factual, neighborhood, cross_ref, multi_hop, negation
    difficulty: int = 1  # 1=easy, 2=medium, 3=hard
    expect_keywords: List[str] = field(default_factory=list)
    expect_entities: List[str] = field(default_factory=list)
    ground_truth_answer: str = ""
    source_projects: List[str] = field(default_factory=list)  # e.g. ["BG11-005", "BG24-030"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "tier": self.tier,
            "difficulty": self.difficulty,
            "expect_keywords": self.expect_keywords,
            "expect_entities": self.expect_entities,
            "ground_truth_answer": self.ground_truth_answer,
            "source_projects": self.source_projects,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkQuery":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
