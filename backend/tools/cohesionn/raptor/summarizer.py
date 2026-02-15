"""
RAPTOR Summarizer - LLM-based summarization for hierarchical nodes

Generates summaries at each level of the RAPTOR tree, preserving
engineering terminology and technical accuracy.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    """Result from summarization"""

    summary: str
    source_count: int  # Number of source items summarized
    level: int  # RAPTOR level (1=cluster, 2=section, 3=topic)
    model_used: str
    tokens_used: Optional[int] = None


class RaptorSummarizer:
    """
    LLM-based summarizer for RAPTOR tree nodes.

    Features:
    - Uses fast small model (qwen2.5:1.5b) for batch processing
    - Domain-specific prompts preserving engineering terminology
    - Level-aware prompts (cluster -> section -> topic)
    - Configurable summary length by level
    """

    # Summary length targets by level (characters)
    LEVEL_LENGTHS = {
        1: 800,   # Cluster summaries - detailed
        2: 1200,  # Section summaries - comprehensive
        3: 2000,  # Topic summaries - thorough overview
    }

    # Level-specific prompt templates
    LEVEL_PROMPTS = {
        # preserving domain-specific terminology
        1: """Summarize the following technical content chunks into a coherent summary.
Preserve all technical terms, test methods, equations, and specific values.
Focus on the key concepts and their relationships.

Content chunks:
{content}

Write a {length}-character technical summary:""",

        2: """Synthesize the following cluster summaries into a comprehensive section summary.
This should capture the main themes and relationships between clusters.
Preserve technical terminology and key specifications.

Cluster summaries:
{content}

Write a {length}-character section summary:""",

        3: """Create a topic overview from the following section summaries.
This should provide a high-level understanding of the entire topic area.
Highlight key concepts, methodologies, and their interconnections.

Section summaries:
{content}

Write a {length}-character topic overview:""",
    }

    def __init__(
        self,
        model: str = None,
        timeout: float = 60.0,
    ):
        """
        Args:
            model: Model name for summarization (default from env)
            timeout: Request timeout in seconds
        """
        self.model = model or os.environ.get("RAPTOR_SUMMARY_MODEL", "qwen2.5:1.5b")
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        """Lazy-load LLM client."""
        if self._client is None:
            from utils.llm import get_llm_client
            self._client = get_llm_client("chat")
        return self._client

    @staticmethod
    def _get_pipeline_config() -> dict:
        """Get pipeline config from lexicon with fallbacks."""
        try:
            from domain_loader import get_pipeline_config
            return get_pipeline_config()
        except Exception:
            return {
                "raptor_context": "technical documentation",
                "raptor_summary_intro": "Summarize the following technical content.",
                "raptor_preserve": [
                    "Key terminology and definitions",
                    "Equations and formulas",
                    "Numerical values and specifications",
                    "Standards and regulatory references",
                ],
            }

    def _get_raptor_context(self) -> str:
        """Get RAPTOR specialization context from lexicon."""
        return self._get_pipeline_config().get("raptor_context", "technical documentation")

    def summarize(
        self,
        contents: List[str],
        level: int = 1,
        custom_prompt: Optional[str] = None,
    ) -> SummaryResult:
        """
        Generate summary from content items.

        Args:
            contents: List of text content to summarize
            level: RAPTOR level (1=cluster, 2=section, 3=topic)
            custom_prompt: Optional custom prompt template

        Returns:
            SummaryResult with summary and metadata
        """
        if not contents:
            return SummaryResult(
                summary="",
                source_count=0,
                level=level,
                model_used=self.model,
            )

        # Combine contents with separators
        combined = "\n\n---\n\n".join(contents)

        # Get target length
        target_length = self.LEVEL_LENGTHS.get(level, 1000)

        # Build prompt
        if custom_prompt:
            prompt = custom_prompt.format(content=combined, length=target_length)
        else:
            template = self.LEVEL_PROMPTS.get(level, self.LEVEL_PROMPTS[1])
            prompt = template.format(content=combined, length=target_length)

        # Generate summary
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a technical documentation assistant specializing in {self._get_raptor_context()}. Your summaries preserve exact technical terminology, test method names, equations, and numerical specifications.",
                    },
                    {"role": "user", "content": prompt},
                ],
                options={
                    "num_ctx": 8192,
                    "temperature": 0.3,  # Lower for factual accuracy
                    "num_predict": target_length // 3,  # ~3 chars per token
                },
            )

            summary = response["message"]["content"].strip()

            # Clean up any thinking tags if present
            if "<think>" in summary and "</think>" in summary:
                # Extract content after thinking
                parts = summary.split("</think>")
                if len(parts) > 1:
                    summary = parts[-1].strip()

            tokens_used = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)

            return SummaryResult(
                summary=summary,
                source_count=len(contents),
                level=level,
                model_used=self.model,
                tokens_used=tokens_used,
            )

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: concatenate with truncation
            fallback = self._fallback_summary(contents, target_length)
            return SummaryResult(
                summary=fallback,
                source_count=len(contents),
                level=level,
                model_used="fallback",
            )

    def summarize_batch(
        self,
        content_groups: List[List[str]],
        level: int = 1,
    ) -> List[SummaryResult]:
        """
        Summarize multiple groups efficiently.

        Args:
            content_groups: List of content lists (one per cluster)
            level: RAPTOR level

        Returns:
            List of SummaryResults
        """
        results = []
        total = len(content_groups)

        for i, contents in enumerate(content_groups):
            logger.debug(f"Summarizing group {i+1}/{total} ({len(contents)} items)")
            result = self.summarize(contents, level=level)
            results.append(result)

            # Log progress for large batches
            if total > 10 and (i + 1) % 10 == 0:
                logger.info(f"Summarization progress: {i+1}/{total}")

        return results

    def _fallback_summary(self, contents: List[str], max_length: int) -> str:
        """Create fallback summary by extracting key sentences."""
        # Simple extractive fallback
        combined = " ".join(contents)

        # Take first N characters with word boundary
        if len(combined) <= max_length:
            return combined

        truncated = combined[:max_length]
        # Find last complete sentence
        for end in [". ", "! ", "? "]:
            last_idx = truncated.rfind(end)
            if last_idx > max_length // 2:
                return truncated[:last_idx + 1]

        # Fall back to word boundary
        last_space = truncated.rfind(" ")
        if last_space > max_length // 2:
            return truncated[:last_space] + "..."

        return truncated + "..."

    def create_engineering_summary(
        self,
        contents: List[str],
        focus_areas: Optional[List[str]] = None,
    ) -> SummaryResult:
        """
        Create summary with engineering-specific focus.

        Args:
            contents: Content to summarize
            focus_areas: Optional list of focus areas (e.g., ["bearing capacity", "settlement"])

        Returns:
            SummaryResult
        """
        focus_str = ""
        if focus_areas:
            focus_str = f"\n\nFocus particularly on: {', '.join(focus_areas)}"

        pipeline = self._get_pipeline_config()
        summary_intro = pipeline["raptor_summary_intro"]
        preserve_items = "\n".join(f"- {item}" for item in pipeline["raptor_preserve"])
        custom_prompt = f"""{summary_intro}
Extract and preserve:
{preserve_items}
{focus_str}

Content:
{{content}}

Write a technical summary ({{length}} characters):"""

        return self.summarize(contents, level=1, custom_prompt=custom_prompt)
