import asyncio
import logging

from typing import List

logger = logging.getLogger(__name__)


async def unload_llm_models() -> List[str]:
    """
    Stop chat LLM server to free VRAM for parallel vision ingestion.

    Unloading chat (~6-8GB) gives enough headroom to run the vision
    server with --parallel 2, doubling OCR throughput.

    Returns list of slot names that were stopped.
    """
    stopped = []
    try:
        from utils.llm import get_server_manager
        mgr = get_server_manager()
        health = await mgr.health_check("chat")
        if health.get("status") == "healthy":
            logger.info("Unloading chat model to free VRAM for parallel vision ingestion")
            mgr.stop("chat")
            stopped.append("chat")
            # Brief pause for GPU memory release
            await asyncio.sleep(2)
        else:
            logger.info("Chat server not running, no unloading needed")
    except Exception as e:
        logger.error(f"Failed to unload LLM servers: {e}")

    return stopped


async def warmup_vision_model(model_name: str = None, timeout: float = 120.0) -> bool:
    """
    Ensure the vision server is running before ingestion starts.

    Args:
        model_name: Unused (vision model set in llm_config)
        timeout: Max seconds to wait for health check

    Returns True if vision server is healthy.
    """
    logger.info(f"Checking vision server health (timeout={timeout}s)")

    try:
        from utils.llm import get_server_manager
        mgr = get_server_manager()
        health = await mgr.health_check("vision")
        if health.get("status") == "healthy":
            logger.info("Vision server is healthy and ready")
            return True
        else:
            logger.warning(f"Vision server not healthy: {health}")
            # Try starting it
            mgr.start("vision")
            healthy = await mgr._wait_for_healthy("vision", timeout=timeout)
            return healthy
    except Exception as e:
        logger.error(f"Vision server warmup failed: {e}")

    return False


async def preload_chat_model(model_name: str = None) -> bool:
    """
    Ensure chat server is running after ingestion completes.

    Args:
        model_name: Unused (chat model set in llm_config)

    Returns True if chat server is healthy.
    """
    try:
        from utils.llm import get_server_manager
        mgr = get_server_manager()
        health = await mgr.health_check("chat")
        if health.get("status") == "healthy":
            logger.info("Chat server already running")
            return True
        else:
            mgr.start("chat")
            healthy = await mgr._wait_for_healthy("chat")
            return healthy
    except Exception as e:
        logger.error(f"Failed to ensure chat server: {e}")

    return False
