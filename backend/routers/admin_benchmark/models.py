"""Pydantic request models for admin benchmark endpoints."""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class BenchmarkStartRequest(BaseModel):
    """Request to start a benchmark run."""
    phases: str = "quick"  # "quick", "full", or comma-separated layer names
    corpus_path: Optional[str] = None
    topic: Optional[str] = None


class ApplyWinnersRequest(BaseModel):
    """Request to apply benchmark winners to RuntimeConfig."""
    winners: Dict[str, Any]


class ProviderConfigRequest(BaseModel):
    """Request to update provider config."""
    judge: Optional[Dict[str, Any]] = None
    ceiling: Optional[Dict[str, Any]] = None


class ProviderTestRequest(BaseModel):
    """Request to test a provider connection."""
    role: str  # "judge" or "ceiling"
    provider: str  # "local", "gemini", "anthropic", "openai"
    model: str = ""
    api_key: str = ""
    base_url: str = ""


class AutoTuneRequest(BaseModel):
    """Request to run auto-tune (L0+L1+L2)."""
    topic: str = ""
    include_judge: bool = False
