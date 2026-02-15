"""
ARCA Chat Executors - Web Search (Core)

Web search via SearXNG. Domain-specific search (guideline lookup)
moved to domains/{domain}/executors/.
"""

import logging
import re
from html import unescape
from typing import Dict, Any, List

import httpx

from config import runtime_config
from errors import (
    handle_tool_errors,
    ExternalServiceError,
)

logger = logging.getLogger(__name__)

# Load domain-specific trusted sources from lexicon pipeline config
try:
    from domain_loader import get_pipeline_config
    _domain_sources = get_pipeline_config().get("trusted_search_domains", [])
except Exception:
    _domain_sources = []

# Academic/engineering source domains (higher priority for web search)
ACADEMIC_DOMAINS = [
    # Universities
    ".edu",
    ".ac.uk",
    ".edu.au",
    ".edu.ca",
    # Government/Research
    ".gov",
    "nrcan.gc.ca",
    "usgs.gov",
    # Professional Organizations
    "asce.org",
    "csce.ca",
    "ieee.org",
    # Journals/Publishers
    "sciencedirect.com",
    "springer.com",
    "tandfonline.com",
    "jstor.org",
    "wiley.com",
    "elsevier.com",
    # Preprints/Research
    "arxiv.org",
    "researchgate.net",
    "scholar.google",
    # Standards
    "astm.org",
    "csa.ca",
    "iso.org",
    # Domain-specific (from lexicon)
    *_domain_sources,
]

_RESULT_ARTICLE_RE = re.compile(
    r"<article[^>]*class=[\"'][^\"']*result[^\"']*[\"'][^>]*>(.*?)</article>",
    re.IGNORECASE | re.DOTALL,
)
_LINK_RE = re.compile(
    r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_CONTENT_RE = re.compile(
    r'<p[^>]*class=["\'][^"\']*content[^"\']*["\'][^>]*>(.*?)</p>',
    re.IGNORECASE | re.DOTALL,
)


def _score_source_quality(url: str) -> int:
    """Score URL by source quality (higher = better)."""
    url_lower = url.lower()
    for domain in ACADEMIC_DOMAINS:
        if domain in url_lower:
            return 2  # Academic/engineering source
    if ".ca" in url_lower or ".gov" in url_lower:
        return 1  # Canadian/government source
    return 0  # General source


def _get_llm_client():
    """Get LLM client for query expansion."""
    from utils.llm import get_llm_client
    return get_llm_client("chat")


def _generate_related_queries(query: str, count: int = 2) -> List[str]:
    """
    Generate related search queries for deep search.

    Uses LLM to expand the query with synonyms and related terms.
    """
    try:
        client = _get_llm_client()

        prompt = f"""Generate {count} alternative search queries for: "{query}"

Requirements:
- Each query should search for the same topic but with different terminology
- Include synonyms, technical terms, or related concepts
- Keep queries concise (under 10 words each)

Output ONLY the queries, one per line, no numbering or explanation."""

        response = client.chat(
            model=runtime_config.model_chat,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": 2048, "temperature": 0.7, "num_predict": 100},
        )

        content = response["message"]["content"]
        related = [q.strip() for q in content.strip().split("\n") if q.strip()]
        return related[:count]

    except Exception as e:
        logger.warning(f"Query expansion failed: {e}", exc_info=True)
        return []


def _deduplicate_results(results: List[Dict]) -> List[Dict]:
    """Remove duplicate results by URL/domain."""
    seen_urls = set()
    unique = []

    for result in results:
        url = result.get("url", "")
        normalized = url.lower().split("?")[0].rstrip("/")
        if normalized not in seen_urls:
            seen_urls.add(normalized)
            unique.append(result)

    return unique


def _normalize_categories(categories: str) -> str:
    parts = [p.strip() for p in (categories or "").split(",") if p.strip()]
    return ",".join(parts) if parts else "general"


def _strip_html(value: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", value or "")
    cleaned = unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _parse_json_results(data: Dict[str, Any], limit: int) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for item in data.get("results", [])[:limit]:
        results.append(
            {
                "title": (item.get("title") or "").strip(),
                "url": (item.get("url") or "").strip(),
                "content": (item.get("content") or "").strip()[:300],
            }
        )
    return results


def _parse_html_results(html_text: str, limit: int) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for block in _RESULT_ARTICLE_RE.findall(html_text or ""):
        link_match = _LINK_RE.search(block)
        if not link_match:
            continue

        url = unescape(link_match.group(1)).strip()
        if not url or url.startswith("/"):
            continue

        title = _strip_html(link_match.group(2))
        content_match = _CONTENT_RE.search(block)
        content = _strip_html(content_match.group(1)) if content_match else ""

        results.append(
            {
                "title": title,
                "url": url,
                "content": content[:300],
            }
        )
        if len(results) >= limit:
            break

    return results


def _fetch_searx_results(query: str, limit: int) -> List[Dict[str, str]]:
    """Fetch search results from SearXNG with JSON->HTML fallback."""
    if not runtime_config.searxng_enabled:
        raise ExternalServiceError(
            "Web search is disabled",
            details="Enable SearXNG in Admin > Configuration > Connections.",
            service="searxng",
            status_code=503,
        )

    searxng_url = (runtime_config.searxng_url or "http://searxng:8080").rstrip("/")
    timeout_s = float(runtime_config.searxng_timeout_s or 10.0)
    categories = _normalize_categories(runtime_config.searxng_categories)
    language = (runtime_config.searxng_language or "").strip()
    preferred_format = (runtime_config.searxng_request_format or "json").lower()
    if preferred_format not in {"json", "html"}:
        preferred_format = "json"

    base_params = {"q": query, "categories": categories}
    if language:
        base_params["language"] = language

    formats = [preferred_format]
    if preferred_format != "html":
        formats.append("html")

    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
            for response_format in formats:
                params = dict(base_params)
                if response_format == "json":
                    params["format"] = "json"

                try:
                    response = client.get(f"{searxng_url}/search", params=params)
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    if response_format == "json" and status_code in {400, 401, 403, 404, 406, 415}:
                        logger.warning(
                            "SearXNG JSON format unavailable (HTTP %s); retrying with HTML",
                            status_code,
                        )
                        continue
                    raise ExternalServiceError(
                        "Search service error",
                        details=f"SearXNG returned status {status_code}",
                        service="searxng",
                        status_code=status_code,
                    )

                if response_format == "json":
                    try:
                        return _parse_json_results(response.json(), limit)
                    except Exception as exc:
                        logger.warning("SearXNG JSON parse failed (%s); retrying with HTML", exc)
                        continue

                return _parse_html_results(response.text, limit)
    except httpx.TimeoutException:
        raise ExternalServiceError(
            "Search service timed out",
            details="The search request took too long. Try again.",
            service="searxng",
        )
    except httpx.RequestError:
        raise ExternalServiceError(
            "Search service unavailable",
            details="Could not connect to the search service",
            service="searxng",
        )

    return []


@handle_tool_errors("web_search")
def execute_web_search(query: str, deep: bool = False) -> Dict[str, Any]:
    """
    Execute web search via SearXNG.

    Args:
        query: Search query
        deep: If True, performs multi-query search with query expansion

    Returns:
        Search results with optional related queries for deep mode
    """
    if deep:
        return execute_deep_web_search(query)

    max_results = max(1, int(runtime_config.searxng_max_results or 5))
    raw_results = _fetch_searx_results(query, limit=max(max_results * 3, 15))

    scored_results = []
    for item in raw_results:
        url = item.get("url", "")
        scored_results.append(
            {
                "title": item.get("title", ""),
                "url": url,
                "content": item.get("content", "")[:300],
                "_quality": _score_source_quality(url),
            }
        )

    scored_results.sort(key=lambda x: -x["_quality"])
    results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in scored_results[:max_results]]

    return {"success": True, "query": query, "results": results, "result_count": len(results)}


@handle_tool_errors("web_search")
def execute_deep_web_search(query: str) -> Dict[str, Any]:
    """
    Extended multi-query web search with LLM-based query expansion.

    Performs:
    1. Query expansion to generate related queries
    2. Parallel searches for all queries
    3. Result deduplication and scoring
    4. Returns top results with source attribution
    """
    logger.info(f"Deep search: {query}")

    related_queries = _generate_related_queries(query, count=2)
    all_queries = [query] + related_queries
    logger.info(f"Deep search queries: {all_queries}")

    all_results = []
    search_errors = []
    per_query_limit = max(10, int(runtime_config.searxng_max_results or 5) * 2)

    for q in all_queries:
        try:
            results = _fetch_searx_results(q, limit=per_query_limit)
            for item in results:
                url = item.get("url", "")
                all_results.append(
                    {
                        "title": item.get("title", ""),
                        "url": url,
                        "content": item.get("content", "")[:300],
                        "_quality": _score_source_quality(url),
                        "_query": q,
                    }
                )
        except Exception as e:
            logger.warning(f"Search failed for '{q}': {e}", exc_info=True)
            search_errors.append(q)
            continue

    if not all_results and search_errors:
        raise ExternalServiceError(
            "All search queries failed",
            details="Could not retrieve results from the search service",
            service="searxng",
        )

    unique_results = _deduplicate_results(all_results)

    url_counts: Dict[str, int] = {}
    for result in all_results:
        url = result.get("url", "")
        url_counts[url] = url_counts.get(url, 0) + 1

    for result in unique_results:
        url = result.get("url", "")
        result["_score"] = result["_quality"] + (url_counts.get(url, 1) - 1) * 0.5

    unique_results.sort(key=lambda x: -x.get("_score", 0))

    max_results = max(8, int(runtime_config.searxng_max_results or 5))
    top_results = []
    for result in unique_results[:max_results]:
        top_results.append(
            {
                "title": result["title"],
                "url": result["url"],
                "content": result["content"],
            }
        )

    return {
        "success": True,
        "query": query,
        "related_queries": related_queries,
        "results": top_results,
        "result_count": len(top_results),
        "deep_search": True,
    }
