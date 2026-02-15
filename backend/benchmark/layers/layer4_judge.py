"""
Layer 4: LLM-as-Judge Scoring

Evaluates each answer from Layer 3 on relevance, accuracy, and completeness.
Uses the configured judge provider (local, Gemini, Anthropic, OpenAI).
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from benchmark.layers.base import BaseLayer, LayerResult

logger = logging.getLogger(__name__)

# Warning shown when local LLM is used as judge
LOCAL_JUDGE_WARNING = (
    "Using local LLM as judge. Results may be less reliable than a frontier "
    "model. For best results, use a cloud provider or vary your judge model."
)


class JudgeLayer(BaseLayer):
    """Layer 4: LLM-as-judge evaluation using configurable provider."""

    LAYER_NAME = "layer4_judge"

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

        if provider_type == "local":
            logger.warning(LOCAL_JUDGE_WARNING)

        model = self.config.judge_model
        return LLMJudge(provider=provider, model=model)

    def execute(self, result: LayerResult) -> LayerResult:
        # Load answers from Layer 3
        answers_path = Path(self.config.output_dir) / "layer3_answers" / "answers.json"
        if not answers_path.exists():
            result.errors.append("No answers found from Layer 3")
            return result

        answers = json.loads(answers_path.read_text(encoding="utf-8"))
        result.configs_total = len(answers)

        judge = self._create_judge()

        judgments = []

        for answer_data in answers:
            query_id = answer_data["query_id"]

            if self.checkpoint.is_completed(self.LAYER_NAME, query_id):
                result.configs_skipped += 1
                saved = self.checkpoint.get_result(self.LAYER_NAME, query_id)
                if saved:
                    judgments.append(saved)
                continue

            judgment = judge.judge_answer(
                query_id=query_id,
                question=answer_data["query"],
                answer=answer_data.get("answer", ""),
                ground_truth=answer_data.get("ground_truth", ""),
                context=answer_data.get("context", ""),
            )

            judgment_data = {
                **judgment.to_dict(),
                "tier": answer_data.get("tier", ""),
                "query": answer_data["query"],
            }

            judgments.append(judgment_data)
            self.checkpoint.mark_completed(self.LAYER_NAME, query_id, judgment_data)
            result.configs_completed += 1

            logger.info(
                "  %s: R=%d A=%d C=%d avg=%.1f",
                query_id, judgment.relevance, judgment.accuracy,
                judgment.completeness, judgment.avg_score,
            )

        # Compute aggregates
        valid = [j for j in judgments if not j.get("error")]
        if valid:
            avg_relevance = sum(j["relevance"] for j in valid) / len(valid)
            avg_accuracy = sum(j["accuracy"] for j in valid) / len(valid)
            avg_completeness = sum(j["completeness"] for j in valid) / len(valid)
            avg_overall = sum(j["avg_score"] for j in valid) / len(valid)

            by_tier = {}
            for j in valid:
                tier = j.get("tier", "unknown")
                if tier not in by_tier:
                    by_tier[tier] = []
                by_tier[tier].append(j)

            tier_summary = {}
            for tier, tier_judgments in by_tier.items():
                tier_summary[tier] = {
                    "count": len(tier_judgments),
                    "avg_relevance": round(sum(j["relevance"] for j in tier_judgments) / len(tier_judgments), 2),
                    "avg_accuracy": round(sum(j["accuracy"] for j in tier_judgments) / len(tier_judgments), 2),
                    "avg_completeness": round(sum(j["completeness"] for j in tier_judgments) / len(tier_judgments), 2),
                    "avg_overall": round(sum(j["avg_score"] for j in tier_judgments) / len(tier_judgments), 2),
                }
        else:
            avg_relevance = avg_accuracy = avg_completeness = avg_overall = 0
            tier_summary = {}

        summary = {
            "total_judged": len(valid),
            "errors": len(judgments) - len(valid),
            "avg_relevance": round(avg_relevance, 2),
            "avg_accuracy": round(avg_accuracy, 2),
            "avg_completeness": round(avg_completeness, 2),
            "avg_overall": round(avg_overall, 2),
            "by_tier": tier_summary,
        }

        # Save
        judgments_path = self.output_dir / "judgments.json"
        judgments_path.write_text(json.dumps(judgments, indent=2, default=str), encoding="utf-8")

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        result.summary = summary
        result.best_score = avg_overall

        return result
