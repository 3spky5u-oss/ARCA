"""
Cohesionn HyDE - Hypothetical Document Embeddings

Generates a hypothetical answer to the query, then embeds that instead
of the raw query. Improves retrieval for complex/indirect questions.

Based on: https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde

Flow:
1. Generate hypothetical document/answer using fast LLM
2. Embed the hypothetical document
3. Use that embedding for retrieval

This bridges the semantic gap between questions and document content.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# HyDE timeout — shorter than the 180s chat default but enough for GPU
# contention during benchmarks (embedding + LLM + reranker share VRAM).
HYDE_TIMEOUT = 60.0

# Default HyDE prompts for different query types
HYDE_PROMPTS = {
    "default": """You are a technical document. Write a short paragraph (3-4 sentences) that would directly answer this question. Write as if you are the source document, not as an assistant. Do not include any preamble.

Question: {query}

Document excerpt:""",

    "engineering": """You are {reference_type}. Write a concise technical paragraph (3-4 sentences) that would answer this question. Include specific values, formulas, or standards where appropriate. Write as if you are the source document. Do not include any preamble.

Question: {query}

Textbook excerpt:""",

    "standards": """You are a technical standard or code document (e.g., ASTM, AASHTO, CSA). Write a brief excerpt (3-4 sentences) that addresses this topic. Include typical requirements, limits, or procedures. Write as if you are the standard document. Do not include any preamble.

Topic: {query}

Standard excerpt:""",
}


class HyDEGenerator:
    """
    Generates hypothetical documents for improved retrieval.

    Uses a fast, small model to generate plausible document content
    that would answer the query, then embeds that for retrieval.
    """

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        enabled: bool = True,
        max_tokens: int = 150,
        temperature: float = 0.7,
    ):
        self.model = model
        self.enabled = enabled
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._cache: dict = {}
        self._client = None

    def _detect_query_type(self, query: str) -> str:
        """Detect query type for prompt selection"""
        query_lower = query.lower()
        try:
            from domain_loader import get_pipeline_config
            pipeline = get_pipeline_config()
            hyde_keywords = pipeline.get("hyde_detection_keywords", {})
        except Exception:
            hyde_keywords = {}

        # Standards detection (keep generic defaults — standards are cross-domain)
        standards_kw = hyde_keywords.get("standards", ["standard", "code", "specification", "requirement"])
        if any(kw in query_lower for kw in standards_kw):
            return "standards"

        # Domain-specific detection
        engineering_kw = hyde_keywords.get("engineering", [])
        if engineering_kw and any(kw in query_lower for kw in engineering_kw):
            return "engineering"

        return "default"

    def generate_hypothetical(
        self,
        query: str,
        prompt_type: str = None,
    ) -> str:
        """Generate a hypothetical document that would answer the query.

        Returns generated text, or the original query on failure.
        """
        if not self.enabled:
            return query

        # Cache check — same query always produces same hypothetical
        if query in self._cache:
            return self._cache[query]

        prompt_type = prompt_type or self._detect_query_type(query)
        prompt_template = HYDE_PROMPTS.get(prompt_type, HYDE_PROMPTS["default"])
        # Inject domain reference type from lexicon pipeline config
        try:
            from domain_loader import get_pipeline_config
            reference_type = get_pipeline_config()["reference_type"]
        except Exception:
            reference_type = "a technical reference document"
        prompt = prompt_template.format(query=query, reference_type=reference_type)

        try:
            # Dedicated client with short timeout — don't reuse the shared
            # 180s chat client, HyDE is latency-sensitive pre-retrieval
            if self._client is None:
                from services.llm_client import LLMClient
                from services.llm_config import SLOTS
                port = SLOTS["chat"].port
                self._client = LLMClient(base_url=f"http://localhost:{port}", timeout=HYDE_TIMEOUT)

            client = self._client

            # Gate: skip if chat LLM is not running
            if not client.is_healthy(timeout=2.0):
                logger.debug("HyDE skipped: chat LLM not available")
                return query

            response = client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_predict": self.max_tokens,
                    "temperature": self.temperature,
                    # Use triple newline — double fires immediately on some models
                    "stop": ["\n\n\n", "Question:", "---"],
                },
            )

            hypothetical = response.get("response", "").strip()

            if hypothetical:
                logger.debug(f"HyDE generated ({len(hypothetical)} chars) for: {query[:50]}")
                self._cache[query] = hypothetical
                return hypothetical
            else:
                logger.warning("HyDE generated empty response, using original query")
                return query

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}, using original query")
            return query

    def expand_query(self, query: str) -> str:
        """Expand query using HyDE.

        The hypothetical document is prepended to the original query
        for hybrid matching (both direct and hypothetical).
        """
        if not self.enabled:
            return query

        hypothetical = self.generate_hypothetical(query)

        if hypothetical == query:
            return query

        # Combine: hypothetical provides semantic context, original preserves keywords
        return f"{hypothetical}\n\n{query}"

    def clear_cache(self):
        """Clear the HyDE response cache."""
        self._cache.clear()

    @classmethod
    def from_config(cls) -> "HyDEGenerator":
        """Create generator from runtime config"""
        from config import runtime_config
        return cls(
            model=runtime_config.hyde_model,
            enabled=runtime_config.hyde_enabled,
        )


# Singleton
_hyde_generator: Optional[HyDEGenerator] = None


def get_hyde_generator() -> HyDEGenerator:
    """Get singleton HyDE generator"""
    global _hyde_generator
    if _hyde_generator is None:
        _hyde_generator = HyDEGenerator.from_config()
    return _hyde_generator


def generate_hypothetical(query: str) -> str:
    """Convenience function to generate hypothetical document"""
    generator = get_hyde_generator()
    return generator.generate_hypothetical(query)


def expand_query_hyde(query: str) -> str:
    """Convenience function to expand query with HyDE"""
    generator = get_hyde_generator()
    return generator.expand_query(query)
