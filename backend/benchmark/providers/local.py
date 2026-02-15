"""
Local LLM Provider — calls llama.cpp's OpenAI-compatible endpoint.
"""
import logging
import os
import time
from typing import Dict, List, Tuple

import httpx

from benchmark.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_DEFAULT_LLM_URL = "http://localhost:8081"


class LocalProvider(LLMProvider):
    """Provider that calls a local llama.cpp OpenAI-compatible API."""

    def __init__(self, base_url: str = "", model: str = ""):
        self._base_url = (
            base_url or os.environ.get("LLM_URL", _DEFAULT_LLM_URL)
        ).rstrip("/")
        self._model = model or "default"
        self._max_retries = 3
        self._retry_delay = 3.0

    @property
    def provider_name(self) -> str:
        return "Local LLM"

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = "",
        temperature: float = 0.0,
    ) -> str:
        url = self._base_url + "/v1/chat/completions"
        payload = {
            "model": model or self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
        }

        for attempt in range(self._max_retries):
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(url, json=payload)

                if resp.status_code == 503:
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            "Local LLM returned 503 (loading), retry %d/%d in %.0fs",
                            attempt + 1, self._max_retries, self._retry_delay,
                        )
                        time.sleep(self._retry_delay)
                        continue
                    resp.raise_for_status()

                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    logger.warning(
                        "Local LLM timeout, retry %d/%d",
                        attempt + 1, self._max_retries,
                    )
                    time.sleep(self._retry_delay)
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
                return True, "Local LLM responding"
            return False, "Empty response from local LLM"
        except Exception as e:
            return False, str(e)
