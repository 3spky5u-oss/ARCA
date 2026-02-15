"""
OpenAI-Compatible Provider — wraps the OpenAI SDK with configurable base_url.

Covers OpenAI, Azure OpenAI, and any OpenAI-compatible endpoint.
"""
import logging
import os
import time
from typing import Dict, List, Tuple

from benchmark.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        rate_limit: float = 1.0,
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model_name = model or "gpt-4o"
        self._base_url = base_url or None  # None = default OpenAI endpoint
        self._rate_limit = rate_limit
        self._last_call = 0.0

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    def _rate_limit_wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_call = time.time()

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = "",
        temperature: float = 0.0,
    ) -> str:
        try:
            import openai
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Install with: pip install openai"
            )

        if not self._api_key:
            raise ValueError("OpenAI API key not configured")

        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url

        client = openai.OpenAI(**kwargs)

        self._rate_limit_wait()

        response = client.chat.completions.create(
            model=model or self._model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""

    def test_connection(self) -> Tuple[bool, str]:
        try:
            result = self.generate(
                [{"role": "user", "content": "Say OK"}],
                temperature=0.0,
            )
            if result:
                return True, "OpenAI API responding"
            return False, "Empty response from OpenAI"
        except Exception as e:
            return False, str(e)
