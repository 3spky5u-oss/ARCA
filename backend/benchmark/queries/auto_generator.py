"""
Query Auto-Generator — generates benchmark queries from corpus content via LLM.

When no domain battery file exists, this reads corpus text and asks an LLM
to produce tiered Q+A pairs.  Uses the existing LLMProvider abstraction
(same as judge / ceiling) so any configured provider works.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from benchmark.queries.battery import BenchmarkQuery
from benchmark.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# Target count per tier
TIER_DISTRIBUTION: Dict[str, int] = {
    "factual": 15,
    "conceptual": 8,
    "cross_ref": 5,
    "multi_hop": 6,
    "negation": 3,
}

# Per-tier instructions for the LLM
_TIER_PROMPTS: Dict[str, str] = {
    "factual": (
        "Generate {n} factual questions that can be answered by looking up "
        "specific information in the provided text. Questions should ask for "
        "values, names, dates, measurements, or definitions. "
        "Difficulty: mix of easy (1) and medium (2)."
    ),
    "conceptual": (
        "Generate {n} conceptual questions that test understanding of topics, "
        "processes, or relationships described in the text. These require "
        "synthesizing information, not just looking up a fact. "
        "Difficulty: medium (2)."
    ),
    "cross_ref": (
        "Generate {n} comparison or cross-reference questions that require "
        "finding and comparing information from different parts of the "
        "documents. Difficulty: medium (2)."
    ),
    "multi_hop": (
        "Generate {n} multi-hop reasoning questions that require connecting "
        "information from multiple documents or sections to arrive at an "
        "answer. Difficulty: hard (3)."
    ),
    "negation": (
        "Generate {n} negation questions that ask about what is NOT present, "
        "NOT recommended, or does NOT apply based on the documents. "
        "Difficulty: hard (3)."
    ),
}

_SYSTEM_PROMPT = (
    "You are generating benchmark test queries for a RAG (retrieval-augmented "
    "generation) system. Based on the provided document excerpts, create "
    "realistic test questions.\n\n"
    "Return ONLY valid JSON — an array of objects with these fields:\n"
    '  {"query": "...", "difficulty": 1|2|3, '
    '"expect_keywords": ["kw1", "kw2", ...]}\n\n'
    "Rules:\n"
    "- Questions must be answerable from the provided text\n"
    "- expect_keywords: 2-5 key terms that should appear in a good answer\n"
    "- Do not include meta-references like 'in the document'\n"
    "- Questions should be specific enough to have a definitive answer\n"
)


class QueryAutoGenerator:
    """Generate benchmark queries from corpus content using an LLM provider."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self._provider = provider

    # ------------------------------------------------------------------
    # Provider resolution
    # ------------------------------------------------------------------

    def _resolve_provider(self) -> Optional[LLMProvider]:
        """Resolve provider: explicit arg > benchmark_providers.json judge > local."""
        if self._provider:
            return self._provider

        # Try benchmark_providers.json "judge" section
        try:
            providers_path = Path("/app/data/config/benchmark_providers.json")
            if providers_path.exists():
                data = json.loads(providers_path.read_text(encoding="utf-8"))
                judge_cfg = data.get("judge", {})
                provider_type = judge_cfg.get("provider", "local")

                from benchmark.providers import get_provider
                return get_provider(provider_type, judge_cfg)
        except Exception as e:
            logger.debug("Could not load judge provider config: %s", e)

        # Fall back to local
        try:
            from benchmark.providers.local import LocalProvider
            return LocalProvider()
        except Exception as e:
            logger.warning("Could not create local provider: %s", e)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, corpus_dir: str) -> List[BenchmarkQuery]:
        """Generate tiered benchmark queries from corpus markdown files.

        Samples ~8 000 chars across corpus files, then makes one
        provider.generate() call per tier.  Returns empty list on failure.
        """
        provider = self._resolve_provider()
        if not provider:
            logger.warning("No LLM provider available for query generation")
            return []

        sample_text = self._sample_corpus(corpus_dir)
        if not sample_text:
            logger.warning("No text found in corpus directory: %s", corpus_dir)
            return []

        logger.info(
            "Auto-generating queries from %d chars of corpus text via %s",
            len(sample_text), provider.provider_name,
        )

        all_queries: List[BenchmarkQuery] = []

        for tier, count in TIER_DISTRIBUTION.items():
            tier_prompt = _TIER_PROMPTS[tier].replace("{n}", str(count))

            user_msg = (
                tier_prompt + "\n\n"
                "--- DOCUMENT EXCERPTS ---\n" + sample_text +
                "\n--- END EXCERPTS ---\n\n"
                "Return a JSON array of " + str(count) + " query objects."
            )

            try:
                response = provider.generate(
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.7,
                )
                parsed = self._parse_response(response, tier, len(all_queries))
                all_queries.extend(parsed)
                logger.info("  %s: generated %d queries", tier, len(parsed))

            except Exception as e:
                logger.warning("Query generation failed for tier '%s': %s", tier, e)

        logger.info(
            "Auto-generated %d total queries across %d tiers",
            len(all_queries), len(TIER_DISTRIBUTION),
        )
        return all_queries

    # ------------------------------------------------------------------
    # Corpus sampling
    # ------------------------------------------------------------------

    def _sample_corpus(self, corpus_dir: str, max_chars: int = 8000) -> str:
        """Sample text from corpus markdown / text files."""
        corpus_path = Path(corpus_dir)

        md_files = sorted(corpus_path.rglob("*.md"))
        if not md_files:
            md_files = sorted(corpus_path.rglob("*.txt"))
        if not md_files:
            return ""

        chars_per_file = max_chars // max(len(md_files), 1)
        samples: List[str] = []

        for f in md_files[:10]:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                if len(text) < 100:
                    continue
                # Sample from middle — richer content than header/footer
                mid = len(text) // 2
                start = max(0, mid - chars_per_file // 2)
                samples.append(text[start : start + chars_per_file])
            except Exception:
                continue

        return "\n\n---\n\n".join(samples)[:max_chars]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, response: str, tier: str, offset: int,
    ) -> List[BenchmarkQuery]:
        """Parse LLM JSON response into BenchmarkQuery objects."""
        # Strip think tags
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        # Extract JSON array
        match = re.search(r"\[[\s\S]*\]", response)
        if not match:
            logger.warning("No JSON array found in response for tier '%s'", tier)
            return []

        try:
            items = json.loads(match.group())
        except json.JSONDecodeError as e:
            logger.warning("JSON parse failed for tier '%s': %s", tier, e)
            return []

        queries: List[BenchmarkQuery] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict) or "query" not in item:
                continue

            qid = "auto_%s_%02d" % (tier, offset + i + 1)
            queries.append(BenchmarkQuery(
                id=qid,
                query=item["query"],
                tier=tier,
                difficulty=item.get("difficulty", 2),
                expect_keywords=item.get("expect_keywords", []),
                expect_entities=item.get("expect_entities", []),
                ground_truth_answer=item.get("ground_truth_answer", ""),
                source_projects=item.get("source_projects", []),
            ))

        return queries
