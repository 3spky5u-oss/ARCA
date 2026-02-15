"""
Cohesionn CRAG - Corrective Retrieval Augmented Generation

When retrieval confidence is low, triggers web search fallback via SearXNG.
Based on CRAG paper: https://arxiv.org/abs/2401.15884

Flow:
1. Evaluate retrieval confidence from reranker scores
2. If below threshold, query SearXNG for supplementary context
3. Merge web results with knowledge base results
"""

import logging
import os
import httpx
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# SearXNG instance URL (Docker internal or localhost) — base URL, /search appended at call sites
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")
SEARXNG_TIMEOUT = float(os.environ.get("SEARXNG_TIMEOUT", "5.0"))


class CRAGEvaluator:
    """
    Evaluates retrieval confidence and triggers corrective actions.

    Confidence classification:
    - HIGH (>0.35): Use KB results directly
    - MEDIUM (0.25-0.35): Use KB results but flag uncertainty
    - LOW (<0.25): Trigger web search fallback
    """

    def __init__(
        self,
        min_confidence: float = 0.25,
        web_search_on_low: bool = True,
        searxng_url: str = None,
    ):
        """
        Args:
            min_confidence: Score threshold below which triggers web search
            web_search_on_low: Whether to actually query web on low confidence
            searxng_url: SearXNG instance URL
        """
        self.min_confidence = min_confidence
        self.web_search_on_low = web_search_on_low
        self.searxng_url = searxng_url or SEARXNG_URL

    def evaluate_confidence(
        self,
        results: List[Dict[str, Any]],
        score_key: str = "rerank_score",
    ) -> Tuple[str, float]:
        """
        Evaluate retrieval confidence from result scores.

        Args:
            results: Retrieval results with scores
            score_key: Key to read score from (rerank_score or score)

        Returns:
            Tuple of (confidence_level, max_score)
            confidence_level: "high", "medium", or "low"
        """
        if not results:
            return ("low", 0.0)

        # Get maximum score from results
        scores = [r.get(score_key, r.get("score", 0)) for r in results]
        max_score = max(scores) if scores else 0.0

        # Classify confidence
        if max_score >= 0.35:
            return ("high", max_score)
        elif max_score >= self.min_confidence:
            return ("medium", max_score)
        else:
            return ("low", max_score)

    def search_web(
        self,
        query: str,
        num_results: int = 5,
        categories: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search web via SearXNG for supplementary context.

        Args:
            query: Search query
            num_results: Number of results to return
            categories: SearXNG categories (default: general, science)

        Returns:
            List of web results with title, url, content
        """
        if not self.web_search_on_low:
            logger.debug("Web search disabled, skipping")
            return []

        categories = categories or ["general", "science"]

        try:
            params = {
                "q": query,
                "format": "json",
                "categories": ",".join(categories),
            }

            searxng_search_url = self.searxng_url.rstrip("/") + "/search"
            with httpx.Client(timeout=SEARXNG_TIMEOUT, follow_redirects=True) as client:
                response = client.get(searxng_search_url, params=params)
                response.raise_for_status()
                data = response.json()

            results = []
            for item in data.get("results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "source": "web_search",
                    "engine": item.get("engine", "unknown"),
                })

            logger.info(f"Web search returned {len(results)} results for: {query[:50]}")
            return results

        except httpx.TimeoutException:
            logger.warning(f"Web search timed out for: {query[:50]}")
            return []
        except httpx.HTTPStatusError as e:
            logger.warning(f"Web search HTTP error: {e}")
            return []
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []

    def evaluate_and_correct(
        self,
        query: str,
        kb_results: List[Dict[str, Any]],
        score_key: str = "rerank_score",
    ) -> Dict[str, Any]:
        """
        Evaluate KB results and trigger web search if needed.

        Args:
            query: Original query
            kb_results: Results from knowledge base
            score_key: Key to read score from

        Returns:
            Dict with:
            - results: Final results (KB + web if triggered)
            - confidence: Confidence level
            - max_score: Maximum KB score
            - web_triggered: Whether web search was triggered
            - web_results: Web search results (if any)
        """
        confidence, max_score = self.evaluate_confidence(kb_results, score_key)

        result = {
            "results": kb_results,
            "confidence": confidence,
            "max_score": max_score,
            "web_triggered": False,
            "web_results": [],
        }

        # Trigger web search on low confidence
        if confidence == "low" and self.web_search_on_low:
            logger.info(f"Low confidence ({max_score:.3f}), triggering web search")
            web_results = self.search_web(query)

            if web_results:
                result["web_triggered"] = True
                result["web_results"] = web_results

                # Format web results for context
                formatted_web = []
                for wr in web_results:
                    formatted_web.append({
                        "content": f"[Web: {wr['title']}]\n{wr['content']}",
                        "metadata": {
                            "source": wr["url"],
                            "title": wr["title"],
                            "source_type": "web",
                        },
                        "score": 0.1,  # Lower score than KB results
                        "topic": "web_search",
                    })

                # Append web results after KB results
                result["results"] = kb_results + formatted_web

        return result

    @classmethod
    def from_config(cls) -> "CRAGEvaluator":
        """Create evaluator from runtime config"""
        from config import runtime_config
        return cls(
            min_confidence=runtime_config.crag_min_confidence,
            web_search_on_low=runtime_config.crag_web_search_on_low,
        )


# Singleton
_crag_evaluator: Optional[CRAGEvaluator] = None


def get_crag_evaluator() -> CRAGEvaluator:
    """Get singleton CRAG evaluator"""
    global _crag_evaluator
    if _crag_evaluator is None:
        _crag_evaluator = CRAGEvaluator.from_config()
    return _crag_evaluator


def evaluate_and_correct(
    query: str,
    kb_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Convenience function to evaluate and correct retrieval"""
    evaluator = get_crag_evaluator()
    return evaluator.evaluate_and_correct(query, kb_results)
