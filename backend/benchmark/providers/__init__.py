"""Benchmark LLM Providers — factory for provider instances."""
from typing import Any, Dict

from benchmark.providers.base import LLMProvider


def get_provider(provider_type: str, config: Dict[str, Any]) -> LLMProvider:
    """Create an LLM provider instance.

    Args:
        provider_type: "local" | "gemini" | "anthropic" | "openai"
        config: Dict with keys: model, api_key, rate_limit, base_url
    """
    if provider_type == "local":
        from benchmark.providers.local import LocalProvider
        return LocalProvider(
            base_url=config.get("base_url", ""),
            model=config.get("model", ""),
        )
    elif provider_type == "gemini":
        from benchmark.providers.gemini import GeminiProvider
        return GeminiProvider(
            api_key=config.get("api_key", ""),
            model=config.get("model", ""),
            rate_limit=config.get("rate_limit", 5.0),
        )
    elif provider_type == "anthropic":
        from benchmark.providers.anthropic import AnthropicProvider
        return AnthropicProvider(
            api_key=config.get("api_key", ""),
            model=config.get("model", ""),
            rate_limit=config.get("rate_limit", 1.0),
        )
    elif provider_type == "openai":
        from benchmark.providers.openai_compat import OpenAIProvider
        return OpenAIProvider(
            api_key=config.get("api_key", ""),
            model=config.get("model", ""),
            base_url=config.get("base_url", ""),
            rate_limit=config.get("rate_limit", 1.0),
        )
    else:
        raise ValueError("Unknown provider type: " + provider_type)
