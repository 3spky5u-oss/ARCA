"""
Layer Ceiling: Frontier LLM Ceiling Comparison
=====================================================
Compares local LLM answers (from Layer 3) against a frontier/cloud
LLM's answers on the same retrieved context. Measures the "model quality
delta" — how much upgrading from a local GGUF model to a frontier API
model improves RAG answer quality.

The provider is configurable — any LLM provider can serve as the ceiling
model (Anthropic, Gemini, OpenAI, or even a different local model).

Flow:
  1. Load L3 answers from {output_dir}/layer3_answers/answers.json
  2. Load ground truth from the corpus bible
  3. If ceiling_answers.json already exists, load it (checkpoint)
  4. If a ceiling provider is configured, generate missing answers via API
  5. Score both local and ceiling answers on keyword/entity hits, text overlap,
     answer length, and abstention correctness
  6. Produce per-query and per-tier comparison tables
  7. Save ceiling_answers.json, comparison.json, summary.json
"""
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from benchmark.layers.base import BaseLayer, LayerResult

logger = logging.getLogger(__name__)

# System prompt for frontier model
CEILING_SYSTEM_PROMPT = (
    "You are a domain expert evaluating a RAG system. "
    "Given the retrieved context, provide the most accurate and complete "
    "answer possible. If the context doesn't contain the answer, say so "
    "explicitly."
)

# Phrases that indicate the model abstained from answering
ABSTENTION_PHRASES = [
    "not available",
    "not mentioned",
    "not provided",
    "does not contain",
    "doesn't contain",
    "no information",
    "not found",
    "cannot determine",
    "cannot be determined",
    "not specified",
    "insufficient context",
    "context does not",
    "context doesn't",
    "not enough information",
    "i don't have",
    "i do not have",
    "unable to determine",
    "unable to find",
    "no relevant",
]


def _text_overlap_score(answer: str, ground_truth: str) -> float:
    """Compute a simple word-level overlap score between answer and ground truth."""
    if not ground_truth or not answer:
        return 0.0

    gt_words = set(re.findall(r"\b\w+\b", ground_truth.lower()))
    ans_words = set(re.findall(r"\b\w+\b", answer.lower()))

    if not gt_words:
        return 0.0

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
        "both", "either", "neither", "each", "every", "all", "any", "few",
        "more", "most", "other", "some", "such", "no", "only", "own", "same",
        "than", "too", "very", "just", "because", "if", "when", "while",
        "this", "that", "these", "those", "it", "its", "they", "them",
        "their", "we", "our", "you", "your", "he", "she", "his", "her",
    }

    gt_content = gt_words - stopwords
    ans_content = ans_words - stopwords

    if not gt_content:
        return 1.0 if not ans_content else 0.5

    overlap = gt_content & ans_content
    return len(overlap) / len(gt_content)


def _keyword_hit_rate(answer: str, keywords: List[str]) -> float:
    """Fraction of expected keywords that appear in the answer."""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def _entity_hit_rate(answer: str, entities: List[str]) -> float:
    """Fraction of expected entities that appear in the answer."""
    if not entities:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for ent in entities if ent.lower() in answer_lower)
    return hits / len(entities)


def _detected_abstention(answer: str) -> bool:
    """Check if the answer contains abstention language."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in ABSTENTION_PHRASES)


def _score_answer(
    answer: str,
    ground_truth: str,
    expect_keywords: List[str],
    expect_entities: List[str],
) -> Dict[str, Any]:
    """Score a single answer on multiple dimensions."""
    kw_rate = _keyword_hit_rate(answer, expect_keywords)
    ent_rate = _entity_hit_rate(answer, expect_entities)
    overlap = _text_overlap_score(answer, ground_truth) if ground_truth else 0.0
    abstained = _detected_abstention(answer)
    answer_len = len(answer)

    weights = []
    scores = []

    if expect_keywords:
        weights.append(0.3)
        scores.append(kw_rate)
    if expect_entities:
        weights.append(0.3)
        scores.append(ent_rate)
    if ground_truth:
        weights.append(0.4)
        scores.append(overlap)

    if weights:
        composite = sum(w * s for w, s in zip(weights, scores)) / sum(weights)
    else:
        composite = 0.0

    return {
        "keyword_hit_rate": round(kw_rate, 4),
        "entity_hit_rate": round(ent_rate, 4),
        "text_overlap": round(overlap, 4),
        "answer_length": answer_len,
        "abstained": abstained,
        "composite": round(composite, 4),
    }


class CeilingComparisonLayer(BaseLayer):
    """Compare frontier/cloud LLM answers vs local LLM on same context."""

    LAYER_NAME = "layer_ceiling"

    def _load_ceiling_provider(self):
        """Load the ceiling LLM provider from config."""
        from benchmark.providers import get_provider

        providers_path = Path("/app/data/config/benchmark_providers.json")
        if providers_path.exists():
            data = json.loads(providers_path.read_text(encoding="utf-8"))
            cfg = data.get("ceiling", {})
            provider_type = cfg.get("provider", self.config.ceiling_provider)

            # Resolve API key: env var name first, then raw value
            api_key = ""
            key_env = cfg.get("api_key_env", "")
            if key_env:
                api_key = os.environ.get(key_env, "")
            if not api_key:
                api_key = cfg.get("api_key", "")

            return provider_type, get_provider(provider_type, {
                "model": cfg.get("model", self.config.ceiling_model),
                "api_key": api_key,
                "rate_limit": cfg.get("rate_limit", 1.0),
                "base_url": cfg.get("base_url", ""),
            })

        return self.config.ceiling_provider, get_provider(
            self.config.ceiling_provider,
            {"model": self.config.ceiling_model},
        )

    def execute(self, result: LayerResult) -> LayerResult:
        # 1. Load L3 answers
        answers_path = Path(self.config.output_dir) / "layer3_answers" / "answers.json"
        if not answers_path.exists():
            result.errors.append("No answers found from Layer 3. Run layer3_answers first.")
            return result

        l3_answers: List[Dict[str, Any]] = json.loads(
            answers_path.read_text(encoding="utf-8")
        )
        logger.info("Loaded %d answers from Layer 3", len(l3_answers))

        if not l3_answers:
            result.errors.append("Layer 3 answers file is empty.")
            return result

        result.configs_total = len(l3_answers)

        # 2. Load ground truth from corpus bible
        bible_path = os.environ.get(
            "BENCHMARK_GROUND_TRUTH",
            "/app/data/Synthetic Reports/Ground Truth/Ground Truth.txt",
        )
        bible_text = ""
        if Path(bible_path).exists():
            bible_text = Path(bible_path).read_text(encoding="utf-8", errors="ignore")
            logger.info("Loaded ground truth bible (%d chars)", len(bible_text))
        else:
            logger.warning("Ground truth file not found: %s", bible_path)

        # 3. Load query battery for expect_keywords/expect_entities
        queries_path = Path(self.config.output_dir) / "layer0_chunking" / "queries.json"
        query_lookup: Dict[str, Dict[str, Any]] = {}
        if queries_path.exists():
            from benchmark.queries.battery import BenchmarkQuery

            raw_queries = json.loads(queries_path.read_text(encoding="utf-8"))
            for q_raw in raw_queries:
                q = BenchmarkQuery.from_dict(q_raw)
                query_lookup[q.id] = {
                    "expect_keywords": q.expect_keywords,
                    "expect_entities": q.expect_entities,
                    "ground_truth_answer": q.ground_truth_answer,
                }
            logger.info("Loaded %d query definitions from L0", len(query_lookup))

        # 4. Load or generate ceiling answers
        ceiling_answers_path = self.output_dir / "ceiling_answers.json"
        ceiling_answers: Dict[str, str] = {}

        if ceiling_answers_path.exists():
            raw = json.loads(ceiling_answers_path.read_text(encoding="utf-8"))
            for entry in raw:
                ceiling_answers[entry["query_id"]] = entry["ceiling_answer"]
            logger.info("Loaded %d existing ceiling answers from cache", len(ceiling_answers))

        # Try to load provider
        provider_type = "none"
        provider = None
        try:
            provider_type, provider = self._load_ceiling_provider()
        except Exception as e:
            logger.warning("Could not load ceiling provider: %s", e)

        need_generation = [
            a for a in l3_answers if a["query_id"] not in ceiling_answers
        ]

        if need_generation and provider:
            logger.info(
                "Generating %d ceiling answers via %s",
                len(need_generation), provider.provider_name,
            )
            ceiling_answers = self._generate_ceiling_answers(
                l3_answers=l3_answers,
                existing_answers=ceiling_answers,
                provider=provider,
                ceiling_answers_path=ceiling_answers_path,
                result=result,
            )
        elif need_generation:
            logger.warning(
                "No ceiling provider configured. %d queries will not have "
                "ceiling answers. Configure a provider in the admin panel "
                "or set ceiling_provider in BenchmarkConfig.",
                len(need_generation),
            )

        has_ceiling = len(ceiling_answers) > 0
        logger.info(
            "Ceiling answers available for %d/%d queries",
            len(ceiling_answers), len(l3_answers),
        )

        # 5. Score both local and ceiling answers
        comparisons: List[Dict[str, Any]] = []

        for entry in l3_answers:
            qid = entry["query_id"]
            query = entry["query"]
            tier = entry.get("tier", "unknown")
            local_answer = entry.get("answer", "")
            ground_truth = entry.get("ground_truth", "")

            q_info = query_lookup.get(qid, {})
            expect_kw = q_info.get("expect_keywords", [])
            expect_ent = q_info.get("expect_entities", [])

            if not ground_truth:
                ground_truth = q_info.get("ground_truth_answer", "")

            local_scores = _score_answer(local_answer, ground_truth, expect_kw, expect_ent)

            ceiling_answer = ceiling_answers.get(qid, "")
            ceiling_scores = (
                _score_answer(ceiling_answer, ground_truth, expect_kw, expect_ent)
                if ceiling_answer
                else None
            )

            winner = "n/a"
            delta = 0.0
            if ceiling_scores is not None:
                local_c = local_scores["composite"]
                ceiling_c = ceiling_scores["composite"]
                delta = ceiling_c - local_c
                if delta > 0.01:
                    winner = "ceiling"
                elif delta < -0.01:
                    winner = "local"
                else:
                    winner = "tie"

            comparison = {
                "query_id": qid,
                "query": query,
                "tier": tier,
                "local_answer_length": len(local_answer),
                "local_scores": local_scores,
                "ceiling_answer_length": len(ceiling_answer) if ceiling_answer else 0,
                "ceiling_scores": ceiling_scores,
                "winner": winner,
                "composite_delta": round(delta, 4),
                "has_ground_truth": bool(ground_truth),
                "has_ceiling_answer": bool(ceiling_answer),
            }
            comparisons.append(comparison)
            result.configs_completed += 1

        # 6. Aggregate
        summary = self._build_summary(comparisons, has_ceiling, provider_type)

        # 7. Save
        comparison_path = self.output_dir / "comparison.json"
        comparison_path.write_text(
            json.dumps(comparisons, indent=2, default=str), encoding="utf-8"
        )

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

        logger.info("Saved comparison (%d entries) and summary", len(comparisons))

        result.summary = summary
        result.best_score = summary.get("local_avg_composite", 0.0)

        return result

    def _generate_ceiling_answers(
        self,
        l3_answers: List[Dict[str, Any]],
        existing_answers: Dict[str, str],
        provider,
        ceiling_answers_path: Path,
        result: LayerResult,
    ) -> Dict[str, str]:
        """Generate ceiling answers via the configured provider with per-query checkpointing."""
        answers = dict(existing_answers)

        for i, entry in enumerate(l3_answers):
            qid = entry["query_id"]

            if qid in answers:
                continue

            checkpoint_key = "ceiling_" + qid
            if self.checkpoint.is_completed(self.LAYER_NAME, checkpoint_key):
                saved = self.checkpoint.get_result(self.LAYER_NAME, checkpoint_key)
                if saved and "ceiling_answer" in saved:
                    answers[qid] = saved["ceiling_answer"]
                    continue

            context = entry.get("context", "")
            query = entry["query"]

            user_prompt = (
                "Retrieved context:\n" + context + "\n\n"
                "Question: " + query + "\n\n"
                "Provide a precise, factual answer based on the context above."
            )

            try:
                ceiling_answer = provider.generate(
                    [
                        {"role": "system", "content": CEILING_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )

                answers[qid] = ceiling_answer

                self.checkpoint.mark_completed(
                    self.LAYER_NAME,
                    checkpoint_key,
                    {"query_id": qid, "ceiling_answer": ceiling_answer},
                )

                logger.info(
                    "  [%d/%d] %s: %d chars from %s",
                    i + 1, len(l3_answers), qid,
                    len(ceiling_answer), provider.provider_name,
                )

                self._save_ceiling_answers(answers, ceiling_answers_path)

            except Exception as e:
                msg = "Ceiling provider error for " + qid + ": " + str(e)
                logger.warning(msg)
                result.errors.append(msg)
                continue

        self._save_ceiling_answers(answers, ceiling_answers_path)
        logger.info("Generated %d total ceiling answers", len(answers))

        return answers

    def _save_ceiling_answers(self, answers: Dict[str, str], path: Path):
        """Save ceiling answers dict to JSON file."""
        output = [
            {"query_id": qid, "ceiling_answer": ans}
            for qid, ans in sorted(answers.items())
        ]
        path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    def _build_summary(
        self,
        comparisons: List[Dict[str, Any]],
        has_ceiling: bool,
        provider_type: str = "",
    ) -> Dict[str, Any]:
        """Build aggregate summary from per-query comparisons."""
        n_total = len(comparisons)
        if n_total == 0:
            return {"error": "no comparisons to summarize"}

        local_composites = [c["local_scores"]["composite"] for c in comparisons]
        local_kw_rates = [c["local_scores"]["keyword_hit_rate"] for c in comparisons]
        local_ent_rates = [c["local_scores"]["entity_hit_rate"] for c in comparisons]
        local_overlaps = [
            c["local_scores"]["text_overlap"]
            for c in comparisons
            if c["has_ground_truth"]
        ]
        local_lengths = [c["local_answer_length"] for c in comparisons]
        local_abstained = sum(1 for c in comparisons if c["local_scores"]["abstained"])

        summary: Dict[str, Any] = {
            "total_queries": n_total,
            "has_ceiling_answers": has_ceiling,
            "ceiling_provider": provider_type,
            "local_avg_composite": round(_safe_mean(local_composites), 4),
            "local_avg_keyword_hit_rate": round(_safe_mean(local_kw_rates), 4),
            "local_avg_entity_hit_rate": round(_safe_mean(local_ent_rates), 4),
            "local_avg_text_overlap": round(_safe_mean(local_overlaps), 4),
            "local_avg_answer_length": round(_safe_mean(local_lengths), 1),
            "local_abstention_count": local_abstained,
        }

        ceiling_entries = [c for c in comparisons if c["has_ceiling_answer"]]
        if ceiling_entries:
            ceiling_composites = [c["ceiling_scores"]["composite"] for c in ceiling_entries]
            ceiling_kw_rates = [c["ceiling_scores"]["keyword_hit_rate"] for c in ceiling_entries]
            ceiling_ent_rates = [c["ceiling_scores"]["entity_hit_rate"] for c in ceiling_entries]
            ceiling_overlaps = [
                c["ceiling_scores"]["text_overlap"]
                for c in ceiling_entries
                if c["has_ground_truth"]
            ]
            ceiling_lengths = [c["ceiling_answer_length"] for c in ceiling_entries]
            ceiling_abstained = sum(
                1 for c in ceiling_entries if c["ceiling_scores"]["abstained"]
            )

            ceiling_wins = sum(1 for c in ceiling_entries if c["winner"] == "ceiling")
            local_wins = sum(1 for c in ceiling_entries if c["winner"] == "local")
            ties = sum(1 for c in ceiling_entries if c["winner"] == "tie")
            deltas = [c["composite_delta"] for c in ceiling_entries]

            summary.update({
                "ceiling_queries_scored": len(ceiling_entries),
                "ceiling_avg_composite": round(_safe_mean(ceiling_composites), 4),
                "ceiling_avg_keyword_hit_rate": round(_safe_mean(ceiling_kw_rates), 4),
                "ceiling_avg_entity_hit_rate": round(_safe_mean(ceiling_ent_rates), 4),
                "ceiling_avg_text_overlap": round(_safe_mean(ceiling_overlaps), 4),
                "ceiling_avg_answer_length": round(_safe_mean(ceiling_lengths), 1),
                "ceiling_abstention_count": ceiling_abstained,
                "ceiling_wins": ceiling_wins,
                "local_wins": local_wins,
                "ties": ties,
                "avg_composite_delta": round(_safe_mean(deltas), 4),
                "max_composite_delta": round(max(deltas), 4) if deltas else 0.0,
                "min_composite_delta": round(min(deltas), 4) if deltas else 0.0,
                "model_quality_delta": round(
                    _safe_mean(ceiling_composites) - _safe_mean(
                        [c["local_scores"]["composite"] for c in ceiling_entries]
                    ),
                    4,
                ),
                "ceiling_model": self.config.ceiling_model,
            })

        # Per-tier breakdown
        by_tier: Dict[str, List[Dict[str, Any]]] = {}
        for c in comparisons:
            tier = c.get("tier", "unknown")
            if tier not in by_tier:
                by_tier[tier] = []
            by_tier[tier].append(c)

        tier_summary: Dict[str, Dict[str, Any]] = {}
        for tier, entries in sorted(by_tier.items()):
            tier_local = [e["local_scores"]["composite"] for e in entries]
            tier_data: Dict[str, Any] = {
                "count": len(entries),
                "local_avg_composite": round(_safe_mean(tier_local), 4),
            }

            tier_ceiling = [
                e for e in entries if e["has_ceiling_answer"] and e["ceiling_scores"]
            ]
            if tier_ceiling:
                tier_ceiling_composites = [
                    e["ceiling_scores"]["composite"] for e in tier_ceiling
                ]
                tier_deltas = [e["composite_delta"] for e in tier_ceiling]
                tier_data.update({
                    "ceiling_avg_composite": round(_safe_mean(tier_ceiling_composites), 4),
                    "avg_delta": round(_safe_mean(tier_deltas), 4),
                    "ceiling_wins": sum(1 for e in tier_ceiling if e["winner"] == "ceiling"),
                    "local_wins": sum(1 for e in tier_ceiling if e["winner"] == "local"),
                    "ties": sum(1 for e in tier_ceiling if e["winner"] == "tie"),
                })

            tier_summary[tier] = tier_data

        summary["by_tier"] = tier_summary

        return summary


def _safe_mean(values: List[float]) -> float:
    """Compute mean, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)
