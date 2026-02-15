"""
Anthropic Provider — wraps the Anthropic SDK.
"""
import logging
import os
import time
from typing import Dict, List, Tuple

from benchmark.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Provider that calls the Anthropic API."""

    def __init__(self, api_key: str = "", model: str = "", rate_limit: float = 1.0):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model_name = model or "claude-sonnet-4-5-20250929"
        self._rate_limit = rate_limit
        self._last_call = 0.0

    @property
    def provider_name(self) -> str:
        return "Anthropic"

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
            import anthropic
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        if not self._api_key:
            raise ValueError("Anthropic API key not configured")

        client = anthropic.Anthropic(api_key=self._api_key)

        # Extract system message
        system_content = ""
        user_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                user_messages.append(msg)

        self._rate_limit_wait()

        response = client.messages.create(
            model=model or self._model_name,
            max_tokens=1024,
            system=system_content,
            messages=user_messages,
            temperature=temperature,
        )

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        return text

    def test_connection(self) -> Tuple[bool, str]:
        try:
            result = self.generate(
                [{"role": "user", "content": "Say OK"}],
                temperature=0.0,
            )
            if result:
                return True, "Anthropic API responding"
            return False, "Empty response from Anthropic"
        except Exception as e:
            return False, str(e)
