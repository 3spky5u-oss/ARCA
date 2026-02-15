"""
Query Battery Loader — central orchestrator for benchmark queries.

Resolution order:
  1. Domain battery file — domains/{domain}/benchmark_battery.json
  2. Auto-generated — LLM reads corpus chunks, generates Q+A pairs
  3. Generic fallback — minimal set that works on any corpus
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from benchmark.queries.battery import BenchmarkQuery

logger = logging.getLogger(__name__)


class QueryBatteryLoader:
    """Load benchmark queries from the best available source."""

    def __init__(self):
        self._source: str = ""

    @property
    def source(self) -> str:
        """Which source provided queries: domain_battery | auto_generated | generic_fallback"""
        return self._source

    def load(self, corpus_dir: str, config) -> List[BenchmarkQuery]:
        """Load benchmark queries, trying sources in priority order."""
        # 1. Domain battery file
        queries = self._load_domain_battery()
        if queries:
            self._source = "domain_battery"
            logger.info(
                "Loaded %d queries from domain battery (%s)",
                len(queries), self._tier_summary(queries),
            )
            return queries

        # 2. Auto-generated from corpus
        queries = self._auto_generate(corpus_dir)
        if queries:
            self._source = "auto_generated"
            logger.info(
                "Auto-generated %d queries from corpus (%s)",
                len(queries), self._tier_summary(queries),
            )
            return queries

        # 3. Generic fallback
        queries = self._generic_fallback()
        self._source = "generic_fallback"
        logger.info("Using %d generic fallback queries", len(queries))
        return queries

    def load_adversarial(self) -> List[Dict[str, Any]]:
        """Load domain-specific adversarial queries from battery file.

        Returns list of adversarial query dicts (same shape as layer_live's
        ADVERSARIAL_QUERIES items).
        """
        try:
            from domain_loader import get_domain_config

            domain = get_domain_config()
            battery_path = domain.domain_dir / "benchmark_battery.json"
            if battery_path.exists():
                data = json.loads(battery_path.read_text(encoding="utf-8"))
                adversarial = data.get("adversarial", [])
                if adversarial:
                    logger.info(
                        "Loaded %d domain adversarial queries from battery",
                        len(adversarial),
                    )
                return adversarial
        except Exception as e:
            logger.debug("No domain adversarial queries: %s", e)
        return []

    # ------------------------------------------------------------------
    # Source 1: domain battery file
    # ------------------------------------------------------------------

    def _load_domain_battery(self) -> List[BenchmarkQuery]:
        """Try loading from domains/{domain}/benchmark_battery.json."""
        try:
            from domain_loader import get_domain_config

            domain = get_domain_config()
            battery_path = domain.domain_dir / "benchmark_battery.json"
            if not battery_path.exists():
                logger.debug("No battery file at %s", battery_path)
                return []

            data = json.loads(battery_path.read_text(encoding="utf-8"))
            raw_queries = data.get("queries", [])
            return [BenchmarkQuery.from_dict(q) for q in raw_queries]
        except Exception as e:
            logger.warning("Failed to load domain battery: %s", e)
            return []

    # ------------------------------------------------------------------
    # Source 2: auto-generated from corpus
    # ------------------------------------------------------------------

    def _auto_generate(self, corpus_dir: str) -> List[BenchmarkQuery]:
        """Generate queries from corpus content using LLM."""
        try:
            from benchmark.queries.auto_generator import QueryAutoGenerator

            generator = QueryAutoGenerator()
            return generator.generate(corpus_dir)
        except Exception as e:
            logger.warning("Auto-generation failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Source 3: generic fallback
    # ------------------------------------------------------------------

    def _generic_fallback(self) -> List[BenchmarkQuery]:
        """Minimal generic queries + lexicon benchmark_queries if present."""
        queries: List[BenchmarkQuery] = []

        # Pull any domain-provided benchmark_queries from lexicon
        try:
            from domain_loader import get_lexicon

            lexicon = get_lexicon()
            for i, dq in enumerate(lexicon.get("benchmark_queries", [])):
                if isinstance(dq, str):
                    queries.append(BenchmarkQuery(
                        id="lexicon_%02d" % (i + 1),
                        query=dq,
                        tier="factual",
                        difficulty=1,
                    ))
                elif isinstance(dq, dict):
                    queries.append(BenchmarkQuery.from_dict({
                        "id": dq.get("id", "lexicon_%02d" % (i + 1)),
                        "query": dq.get("query", ""),
                        "tier": dq.get("tier", "factual"),
                        "difficulty": dq.get("difficulty", 1),
                        "expect_keywords": dq.get("expect_keywords", []),
                    }))
        except Exception:
            pass

        # Generic queries that work on any corpus
        generic = [
            ("gen_01", "What are the key topics covered in this document collection?", "factual", 1),
            ("gen_02", "Summarize the main findings or conclusions", "factual", 1),
            ("gen_03", "What methodology or approach is described?", "conceptual", 2),
            ("gen_04", "What are the limitations or challenges mentioned?", "conceptual", 2),
            ("gen_05", "What recommendations are provided?", "factual", 1),
            ("gen_06", "How do the different documents relate to each other?", "cross_ref", 2),
            ("gen_07", "What technical terms or specialized vocabulary is used?", "factual", 1),
            ("gen_08", "What data or measurements are reported?", "factual", 1),
            ("gen_09", "Compare and contrast the approaches in different sections", "cross_ref", 2),
            ("gen_10", "What are the most important numerical values or specifications?", "factual", 1),
        ]

        for qid, query, tier, diff in generic:
            queries.append(BenchmarkQuery(
                id=qid,
                query=query,
                tier=tier,
                difficulty=diff,
            ))

        return queries

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tier_summary(queries: List[BenchmarkQuery]) -> str:
        """Compact tier breakdown string: '15 factual, 8 conceptual, ...'"""
        counts: Dict[str, int] = {}
        for q in queries:
            counts[q.tier] = counts.get(q.tier, 0) + 1
        return ", ".join(
            "%d %s" % (v, k) for k, v in sorted(counts.items(), key=lambda x: -x[1])
        )
