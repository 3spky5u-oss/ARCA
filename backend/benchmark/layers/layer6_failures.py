"""
Layer 6: Failure Categorization

For queries with accuracy < 3 or relevance < 3 in Layer 4,
categorizes the failure mode using the configured judge provider.
"""
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from benchmark.layers.base import BaseLayer, LayerResult

logger = logging.getLogger(__name__)


class FailureLayer(BaseLayer):
    """Layer 6: Failure analysis and categorization."""

    LAYER_NAME = "layer6_failures"

    FAILURE_THRESHOLD = 3

    def _create_judge(self):
        """Create LLMJudge from config + benchmark_providers.json."""
        from benchmark.judge import LLMJudge
        from benchmark.providers import get_provider

        providers_path = Path("/app/data/config/benchmark_providers.json")
        if providers_path.exists():
            data = json.loads(providers_path.read_text(encoding="utf-8"))
            cfg = data.get("judge", {})
            provider_type = cfg.get("provider", self.config.judge_provider)

            api_key = ""
            key_env = cfg.get("api_key_env", "")
            if key_env:
                api_key = os.environ.get(key_env, "")
            if not api_key:
                api_key = cfg.get("api_key", "")

            provider = get_provider(provider_type, {
                "model": cfg.get("model", self.config.judge_model),
                "api_key": api_key,
                "rate_limit": cfg.get("rate_limit", 1.0),
                "base_url": cfg.get("base_url", ""),
            })
        else:
            provider_type = self.config.judge_provider
            provider = get_provider(provider_type, {
                "model": self.config.judge_model,
            })

        model = self.config.judge_model
        return LLMJudge(provider=provider, model=model)

    def execute(self, result: LayerResult) -> LayerResult:
        # Load judgments from Layer 4
        judgments_path = Path(self.config.output_dir) / "layer4_judge" / "judgments.json"
        if not judgments_path.exists():
            result.errors.append("No judgments found from Layer 4")
            return result

        judgments = json.loads(judgments_path.read_text(encoding="utf-8"))

        # Load answers from Layer 3
        answers_path = Path(self.config.output_dir) / "layer3_answers" / "answers.json"
        answers_by_id = {}
        if answers_path.exists():
            answers = json.loads(answers_path.read_text(encoding="utf-8"))
            answers_by_id = {a["query_id"]: a for a in answers}

        # Filter for failures
        failures = [
            j for j in judgments
            if not j.get("error") and (
                j.get("relevance", 5) < self.FAILURE_THRESHOLD or
                j.get("accuracy", 5) < self.FAILURE_THRESHOLD
            )
        ]

        result.configs_total = len(failures)

        if not failures:
            logger.info("No failures to categorize")
            result.summary = {"total_failures": 0, "message": "No queries below threshold"}
            return result

        logger.info("Categorizing %d failed queries", len(failures))

        judge = self._create_judge()

        categorizations = []

        for j in failures:
            query_id = j["query_id"]

            if self.checkpoint.is_completed(self.LAYER_NAME, query_id):
                result.configs_skipped += 1
                saved = self.checkpoint.get_result(self.LAYER_NAME, query_id)
                if saved:
                    categorizations.append(saved)
                continue

            answer_data = answers_by_id.get(query_id, {})

            cat = judge.categorize_failure(
                query_id=query_id,
                question=j.get("query", answer_data.get("query", "")),
                answer=answer_data.get("answer", ""),
                ground_truth=answer_data.get("ground_truth", ""),
                context=answer_data.get("context", ""),
            )

            cat_data = {
                **cat.to_dict(),
                "relevance": j.get("relevance", 0),
                "accuracy": j.get("accuracy", 0),
                "tier": j.get("tier", ""),
                "query": j.get("query", ""),
            }

            categorizations.append(cat_data)
            self.checkpoint.mark_completed(self.LAYER_NAME, query_id, cat_data)
            result.configs_completed += 1

            logger.info("  %s: %s (%s)", query_id, cat.category, cat.severity)

        # Summarize
        category_counts = Counter(c.get("category", "unknown") for c in categorizations if not c.get("error"))
        severity_counts = Counter(c.get("severity", "unknown") for c in categorizations if not c.get("error"))

        summary = {
            "total_failures": len(failures),
            "categorized": len([c for c in categorizations if not c.get("error")]),
            "by_category": dict(category_counts.most_common()),
            "by_severity": dict(severity_counts.most_common()),
            "examples": categorizations[:5],
        }

        # Save
        cats_path = self.output_dir / "categorizations.json"
        cats_path.write_text(json.dumps(categorizations, indent=2, default=str), encoding="utf-8")

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        result.summary = summary

        return result
