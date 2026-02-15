"""
Scoring Engine
==============
Computes retrieval quality metrics for benchmark queries.

Pure Python + math only — no ARCA imports.

Metrics computed per query:
  - keyword_hit_rate: fraction of expected keywords found in retrieved text
  - entity_hit_rate: fraction of expected entities found
  - mrr: Mean Reciprocal Rank (position of first relevant chunk)
  - ndcg_at_k: normalized Discounted Cumulative Gain
  - source_diversity: unique sources / total chunks
  - latency_ms: retrieval time
  - raptor_contribution: fraction of chunks from RAPTOR
  - graph_contribution: fraction of chunks from GraphRAG

Composite score weights:
  35% keyword | 15% entity | 20% MRR | 15% nDCG | 10% diversity | 5% latency
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryMetrics:
    """Metrics for a single query evaluation."""

    query_id: str
    tier: str
    keyword_hit_rate: float = 0.0
    entity_hit_rate: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    source_diversity: float = 0.0
    latency_ms: float = 0.0
    n_chunks: int = 0
    raptor_contribution: float = 0.0
    graph_contribution: float = 0.0
    composite_score: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "query_id": self.query_id,
            "tier": self.tier,
            "keyword_hit_rate": round(self.keyword_hit_rate, 4),
            "entity_hit_rate": round(self.entity_hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "source_diversity": round(self.source_diversity, 4),
            "latency_ms": round(self.latency_ms, 1),
            "n_chunks": self.n_chunks,
            "raptor_contribution": round(self.raptor_contribution, 4),
            "graph_contribution": round(self.graph_contribution, 4),
            "composite_score": round(self.composite_score, 4),
        }
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple queries."""

    variant_name: str
    n_queries: int = 0
    avg_keyword_hits: float = 0.0
    avg_entity_hits: float = 0.0
    avg_mrr: float = 0.0
    avg_ndcg: float = 0.0
    avg_diversity: float = 0.0
    avg_latency_ms: float = 0.0
    avg_composite: float = 0.0
    std_keyword_hits: float = 0.0
    std_composite: float = 0.0
    by_tier: Dict[str, Dict[str, float]] = field(default_factory=dict)
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_name": self.variant_name,
            "n_queries": self.n_queries,
            "avg_keyword_hits": round(self.avg_keyword_hits, 4),
            "avg_entity_hits": round(self.avg_entity_hits, 4),
            "avg_mrr": round(self.avg_mrr, 4),
            "avg_ndcg": round(self.avg_ndcg, 4),
            "avg_diversity": round(self.avg_diversity, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_composite": round(self.avg_composite, 4),
            "std_keyword_hits": round(self.std_keyword_hits, 4),
            "std_composite": round(self.std_composite, 4),
            "by_tier": self.by_tier,
            "errors": self.errors,
        }


# ── Composite Score Weights ──────────────────────────────────────────────────

COMPOSITE_WEIGHTS = {
    "keyword": 0.35,
    "entity": 0.15,
    "mrr": 0.20,
    "ndcg": 0.15,
    "diversity": 0.10,
    "latency": 0.05,
}

# Latency normalization: 0ms → 1.0, ≥2000ms → 0.0
LATENCY_BEST_MS = 0.0
LATENCY_WORST_MS = 2000.0


class ScoringEngine:
    """Computes retrieval quality metrics."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or COMPOSITE_WEIGHTS

    def score_retrieval(
        self,
        query_id: str,
        tier: str,
        chunks: List[Dict[str, Any]],
        latency_ms: float,
        expect_keywords: List[str],
        expect_entities: List[str],
    ) -> QueryMetrics:
        """Score a single retrieval result."""
        if not chunks:
            return QueryMetrics(
                query_id=query_id,
                tier=tier,
                latency_ms=latency_ms,
                error="no_chunks_retrieved",
            )

        # Concatenate all chunk text for keyword matching
        all_text = " ".join(
            c.get("content", "") for c in chunks
        )

        kw_rate = self._keyword_hit_rate(all_text, expect_keywords)
        ent_rate = self._keyword_hit_rate(all_text, expect_entities)
        mrr = self._mrr(chunks, expect_keywords)
        ndcg = self._ndcg_at_k(chunks, expect_keywords, k=len(chunks))
        diversity = self._source_diversity(chunks)
        raptor = self._contribution(chunks, "is_raptor")
        graph = self._contribution(chunks, "is_graph_result")

        latency_score = self._normalize_latency(latency_ms)

        composite = (
            self.weights["keyword"] * kw_rate
            + self.weights["entity"] * ent_rate
            + self.weights["mrr"] * mrr
            + self.weights["ndcg"] * ndcg
            + self.weights["diversity"] * diversity
            + self.weights["latency"] * latency_score
        )

        return QueryMetrics(
            query_id=query_id,
            tier=tier,
            keyword_hit_rate=kw_rate,
            entity_hit_rate=ent_rate,
            mrr=mrr,
            ndcg_at_k=ndcg,
            source_diversity=diversity,
            latency_ms=latency_ms,
            n_chunks=len(chunks),
            raptor_contribution=raptor,
            graph_contribution=graph,
            composite_score=composite,
        )

    def aggregate(
        self, variant_name: str, per_query: List[QueryMetrics]
    ) -> AggregateMetrics:
        """
        Aggregate per-query metrics into variant-level summary.
        Uses tier-weighted means — each tier contributes equally.
        """
        valid = [q for q in per_query if q.error is None]
        errors = len(per_query) - len(valid)

        if not valid:
            return AggregateMetrics(
                variant_name=variant_name,
                n_queries=len(per_query),
                errors=errors,
            )

        # Group by tier
        by_tier: Dict[str, List[QueryMetrics]] = {}
        for q in valid:
            by_tier.setdefault(q.tier, []).append(q)

        # Compute per-tier averages
        tier_avgs: Dict[str, Dict[str, float]] = {}
        for tier, queries in by_tier.items():
            tier_avgs[tier] = {
                "avg_keyword_hits": _mean([q.keyword_hit_rate for q in queries]),
                "avg_entity_hits": _mean([q.entity_hit_rate for q in queries]),
                "avg_mrr": _mean([q.mrr for q in queries]),
                "avg_ndcg": _mean([q.ndcg_at_k for q in queries]),
                "avg_diversity": _mean([q.source_diversity for q in queries]),
                "avg_latency_ms": _mean([q.latency_ms for q in queries]),
                "avg_composite": _mean([q.composite_score for q in queries]),
                "n_queries": len(queries),
            }

        # Tier-weighted global means (each tier contributes equally)
        # Compute stddev across individual queries for ±spread reporting
        kw_values = [q.keyword_hit_rate for q in valid]
        comp_values = [q.composite_score for q in valid]

        return AggregateMetrics(
            variant_name=variant_name,
            n_queries=len(per_query),
            avg_keyword_hits=_mean([t["avg_keyword_hits"] for t in tier_avgs.values()]),
            avg_entity_hits=_mean([t["avg_entity_hits"] for t in tier_avgs.values()]),
            avg_mrr=_mean([t["avg_mrr"] for t in tier_avgs.values()]),
            avg_ndcg=_mean([t["avg_ndcg"] for t in tier_avgs.values()]),
            avg_diversity=_mean([t["avg_diversity"] for t in tier_avgs.values()]),
            avg_latency_ms=_mean([t["avg_latency_ms"] for t in tier_avgs.values()]),
            avg_composite=_mean([t["avg_composite"] for t in tier_avgs.values()]),
            std_keyword_hits=_stdev(kw_values),
            std_composite=_stdev(comp_values),
            by_tier=tier_avgs,
            errors=errors,
        )

    # ── Internal Metrics ─────────────────────────────────────────────────────

    @staticmethod
    def _keyword_hit_rate(text: str, keywords: List[str]) -> float:
        """Fraction of keywords found in text."""
        if not keywords:
            return 1.0
        text_lower = text.lower()
        return sum(1 for kw in keywords if kw.lower() in text_lower) / len(keywords)

    @staticmethod
    def _mrr(chunks: List[Dict], expect_keywords: List[str]) -> float:
        """
        Mean Reciprocal Rank — position of first chunk containing
        a majority of expected keywords.
        """
        if not expect_keywords:
            return 1.0
        threshold = max(1, len(expect_keywords) // 2)
        for i, chunk in enumerate(chunks):
            text = chunk.get("content", "").lower()
            hits = sum(1 for kw in expect_keywords if kw.lower() in text)
            if hits >= threshold:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def _ndcg_at_k(
        chunks: List[Dict], expect_keywords: List[str], k: int = 5
    ) -> float:
        """
        Normalized Discounted Cumulative Gain.
        Relevance per chunk = fraction of expected keywords found.
        """
        if not expect_keywords or not chunks:
            return 0.0 if expect_keywords else 1.0

        k = min(k, len(chunks))

        # Compute relevance scores
        relevances = []
        for chunk in chunks[:k]:
            text = chunk.get("content", "").lower()
            hits = sum(1 for kw in expect_keywords if kw.lower() in text)
            relevances.append(hits / len(expect_keywords))

        # DCG
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

        # Ideal DCG (sorted relevances descending)
        ideal = sorted(relevances, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _source_diversity(chunks: List[Dict]) -> float:
        """Fraction of unique sources across retrieved chunks."""
        if not chunks:
            return 0.0
        sources = set()
        for c in chunks:
            src = c.get("source", c.get("metadata", {}).get("source", "unknown"))
            sources.add(src)
        return len(sources) / len(chunks)

    @staticmethod
    def _contribution(chunks: List[Dict], flag_key: str) -> float:
        """Fraction of chunks with a given boolean flag."""
        if not chunks:
            return 0.0
        return sum(1 for c in chunks if c.get(flag_key, False)) / len(chunks)

    @staticmethod
    def _normalize_latency(latency_ms: float) -> float:
        """Convert latency to 0-1 score (lower is better)."""
        if latency_ms <= LATENCY_BEST_MS:
            return 1.0
        if latency_ms >= LATENCY_WORST_MS:
            return 0.0
        return 1.0 - (latency_ms - LATENCY_BEST_MS) / (LATENCY_WORST_MS - LATENCY_BEST_MS)


# ── Utility ──────────────────────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    """Safe mean that returns 0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def _stdev(values: List[float]) -> float:
    """Sample standard deviation. Returns 0 for lists with <2 values."""
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)
