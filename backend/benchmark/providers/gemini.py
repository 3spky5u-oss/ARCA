"""
Gemini Provider — wraps Google's Generative AI SDK.
"""
import logging
import os
import time
from typing import Dict, List, Tuple

from benchmark.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Provider that calls Google Gemini API."""

    def __init__(self, api_key: str = "", model: str = "", rate_limit: float = 5.0):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model_name = model or "gemini-3-flash-preview"
        self._rate_limit = rate_limit
        self._max_retries = 3
        self._client = None
        self._last_call = 0.0

    @property
    def provider_name(self) -> str:
        return "Gemini"

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai

            if not self._api_key:
                raise ValueError("Gemini API key not configured")

            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self._model_name)
            logger.info("Gemini provider initialized: %s", self._model_name)
        return self._client

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
        client = self._get_client()

        # Combine system + user messages into a single prompt for Gemini
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append("Instructions: " + content)
            else:
                parts.append(content)
        prompt = "\n\n".join(parts)

        for attempt in range(self._max_retries + 1):
            self._rate_limit_wait()
            try:
                response = client.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                    if attempt < self._max_retries:
                        wait = min(60, (2 ** attempt) * 10)
                        logger.warning(
                            "Gemini rate limited (attempt %d/%d), waiting %ds",
                            attempt + 1, self._max_retries + 1, wait,
                        )
                        time.sleep(wait)
                        continue
                raise

        return ""

    def test_connection(self) -> Tuple[bool, str]:
        try:
            result = self.generate(
                [{"role": "user", "content": "Say OK"}],
                temperature=0.0,
            )
            if result:
                return True, "Gemini responding"
            return False, "Empty response from Gemini"
        except Exception as e:
            return False, str(e)
