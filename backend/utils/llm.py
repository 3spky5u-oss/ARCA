"""LLM client and server manager utilities."""

import asyncio
import logging
from typing import Optional

from services.llm_client import LLMClient
from services.llm_config import SLOTS, get_slot, get_model_path

logger = logging.getLogger(__name__)

# Singleton clients per slot
_clients = {}


def get_llm_client(slot: str = "chat") -> LLMClient:
    """Get LLM client for a slot.

    Args:
        slot: Slot name ("chat", "vision"). Defaults to "chat".

    Returns:
        LLMClient instance pointing at the slot's port.
    """
    if slot not in _clients:
        slot_config = get_slot(slot)
        if not slot_config:
            # Fallback to chat slot
            slot_config = SLOTS["chat"]
        _clients[slot] = LLMClient(base_url=f"http://localhost:{slot_config.port}")
    return _clients[slot]


def get_server_manager():
    """Get the singleton server manager."""
    from services.llm_server_manager import get_server_manager as _get_mgr
    return _get_mgr()


async def ensure_vision_server(timeout: float = 120.0) -> None:
    """Start vision server if not running and wait for health.

    The vision model GGUF should already be in OS page cache
    (pre-cached at startup), so loading from RAM → VRAM is fast (~5-10s).
    """
    mgr = get_server_manager()
    health = await mgr.health_check("vision")
    if health.get("status") == "healthy":
        return
    logger.info("Starting vision server on demand...")
    mgr.start("vision")
    healthy = await mgr._wait_for_healthy("vision", timeout=timeout)
    if not healthy:
        raise RuntimeError("Vision server failed to start within timeout")
    logger.info("Vision server ready")


def stop_vision_server() -> None:
    """Stop vision server to free VRAM after use."""
    mgr = get_server_manager()
    mgr.stop("vision")
    logger.info("Vision server stopped (VRAM freed)")


def precache_model_file(gguf_filename: str) -> None:
    """Read GGUF file into OS page cache for fast subsequent loads.

    Reads the file in 1MB chunks to populate the kernel page cache
    without holding data in Python's heap. Subsequent llama-server
    starts load from RAM instead of disk.
    """
    path = get_model_path(gguf_filename)
    if not path.exists():
        logger.warning(f"Cannot precache {gguf_filename}: file not found")
        return
    size_gb = path.stat().st_size / (1024**3)
    logger.info(f"Pre-caching {gguf_filename} ({size_gb:.1f} GB) into page cache...")
    with open(path, "rb") as f:
        while f.read(1024 * 1024):  # 1MB chunks
            pass
    logger.info(f"Pre-cached {gguf_filename} into page cache")
