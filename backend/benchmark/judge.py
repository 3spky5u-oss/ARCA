"""
LLM Judge — evaluates answer quality using any LLM provider.

Scores each answer on 3 dimensions (1-5):
  - Relevance: Does it address the question?
  - Accuracy: Factually correct vs ground truth?
  - Completeness: All key info included?

Also supports failure categorization for Layer 6.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from benchmark.providers.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class JudgmentResult:
    """Result from LLM-as-judge evaluation."""
    query_id: str
    relevance: int = 0
    accuracy: int = 0
    completeness: int = 0
    reasoning: str = ""
    error: Optional[str] = None

    @property
    def avg_score(self) -> float:
        if self.error:
            return 0.0
        return (self.relevance + self.accuracy + self.completeness) / 3.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "relevance": self.relevance,
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "avg_score": round(self.avg_score, 2),
            "reasoning": self.reasoning,
            "error": self.error,
        }


@dataclass
class FailureCategorization:
    """Result from failure categorization."""
    query_id: str
    category: str = ""
    explanation: str = ""
    severity: str = "medium"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "category": self.category,
            "explanation": self.explanation,
            "severity": self.severity,
            "error": self.error,
        }


JUDGE_PROMPT = """You are evaluating a RAG (Retrieval-Augmented Generation) system's answer quality.

**Question:** {question}

**Ground Truth Answer:** {ground_truth}

**System's Answer:** {answer}

**Retrieved Context:** {context}

Rate the system's answer on these 3 dimensions (1-5 scale):

1. **Relevance** (1-5): Does the answer directly address the question asked?
   - 5: Perfectly on-topic, addresses every aspect
   - 3: Partially relevant, misses some aspects
   - 1: Completely off-topic or refuses to answer

2. **Accuracy** (1-5): Is the answer factually correct compared to the ground truth?
   - 5: All facts match ground truth perfectly
   - 3: Some facts correct, some wrong or uncertain
   - 1: Mostly or entirely incorrect

3. **Completeness** (1-5): Does the answer include all key information from the ground truth?
   - 5: All important details included
   - 3: Some key details missing
   - 1: Only trivial or no relevant details included

Respond in this exact JSON format:
{{"relevance": <1-5>, "accuracy": <1-5>, "completeness": <1-5>, "reasoning": "<brief explanation>"}}"""


FAILURE_PROMPT = """You are categorizing why a RAG system failed on a query.

**Question:** {question}
**Ground Truth Answer:** {ground_truth}
**System's Answer:** {answer}
**Retrieved Context:** {context}

The system scored poorly (relevance or accuracy below 3). Categorize the failure:

Categories:
- **chunking_boundary**: Relevant info was split across chunk boundaries
- **missing_context**: The needed information wasn't retrieved at all
- **wrong_document**: Retrieved chunks from wrong documents/topics
- **hallucination**: System generated plausible but incorrect information
- **incomplete_answer**: System found some info but missed critical details
- **negation_failure**: System failed to handle negation/exclusion in the query

Respond in this exact JSON format:
{{"category": "<one of the categories above>", "explanation": "<brief explanation>", "severity": "<low|medium|high>"}}"""


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code fences."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


class LLMJudge:
    """LLM-as-judge that works with any LLMProvider."""

    def __init__(self, provider: LLMProvider, model: str = ""):
        self.provider = provider
        self.model = model

    def judge_answer(
        self,
        query_id: str,
        question: str,
        answer: str,
        ground_truth: str,
        context: str = "",
    ) -> JudgmentResult:
        """Evaluate a single answer using the configured provider."""
        try:
            prompt = JUDGE_PROMPT.replace("{question}", question)
            prompt = prompt.replace("{ground_truth}", ground_truth or "Not available")
            prompt = prompt.replace("{answer}", answer or "No answer generated")
            prompt = prompt.replace("{context}", context[:2000] if context else "No context")

            text = self.provider.generate(
                [{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
            )

            data = _parse_json_response(text)

            return JudgmentResult(
                query_id=query_id,
                relevance=int(data.get("relevance", 0)),
                accuracy=int(data.get("accuracy", 0)),
                completeness=int(data.get("completeness", 0)),
                reasoning=data.get("reasoning", ""),
            )

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse judge response for %s: %s", query_id, e)
            return JudgmentResult(query_id=query_id, error="JSON parse error: " + str(e))
        except Exception as e:
            logger.error("Judge failed for %s: %s", query_id, e)
            return JudgmentResult(query_id=query_id, error=str(e))

    def categorize_failure(
        self,
        query_id: str,
        question: str,
        answer: str,
        ground_truth: str,
        context: str = "",
    ) -> FailureCategorization:
        """Categorize why a query failed."""
        try:
            prompt = FAILURE_PROMPT.replace("{question}", question)
            prompt = prompt.replace("{ground_truth}", ground_truth or "Not available")
            prompt = prompt.replace("{answer}", answer or "No answer generated")
            prompt = prompt.replace("{context}", context[:2000] if context else "No context")

            text = self.provider.generate(
                [{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
            )

            data = _parse_json_response(text)

            return FailureCategorization(
                query_id=query_id,
                category=data.get("category", "unknown"),
                explanation=data.get("explanation", ""),
                severity=data.get("severity", "medium"),
            )

        except Exception as e:
            logger.error("Failure categorization failed for %s: %s", query_id, e)
            return FailureCategorization(query_id=query_id, error=str(e))
