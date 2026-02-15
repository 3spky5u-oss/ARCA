"""
LLM Provider — abstract base class for all benchmark LLM providers.

Providers wrap different LLM APIs behind a uniform interface so the
benchmark judge, ceiling layer, and auto-tune can use any model.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class LLMProvider(ABC):
    """Abstract LLM provider interface for benchmark evaluation."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name (e.g. 'Local LLM', 'Gemini')."""
        ...

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = "",
        temperature: float = 0.0,
    ) -> str:
        """Generate a text response.

        Args:
            messages: [{"role": "system"|"user", "content": "..."}]
            model: Model identifier. Empty = provider default.
            temperature: Sampling temperature.

        Returns:
            The text response.
        """
        ...

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """Test that the provider is reachable.

        Returns:
            (success, message) tuple.
        """
        ...
