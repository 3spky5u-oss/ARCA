import logging
from typing import Dict, Any

from fastapi import Depends, Query

from services.admin_auth import verify_admin
from . import router, KNOWLEDGE_DIR, COHESIONN_DB_DIR

logger = logging.getLogger(__name__)


@router.get("/topics")
async def list_topics(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """
    List all available topics with their enabled status.

    Returns topics from both filesystem and Qdrant, along with
    whether each is enabled for search.
    """


    from config import runtime_config

    topics = set()

    # From filesystem
    if KNOWLEDGE_DIR.exists():
        for d in KNOWLEDGE_DIR.iterdir():
            if d.is_dir():
                topics.add(d.name)

    # From Qdrant
    try:
        from tools.cohesionn import get_knowledge_base

        kb = get_knowledge_base(COHESIONN_DB_DIR)
        for topic in kb.TOPICS:
            topics.add(topic)
    except Exception:
        pass

    # Get enabled status for each topic
    enabled_topics = runtime_config.get_enabled_topics()
    topics_list = []
    for topic in sorted(topics):
        topics_list.append(
            {
                "name": topic,
                "enabled": topic in enabled_topics,
            }
        )

    return {
        "topics": topics_list,
        "enabled_topics": enabled_topics,
    }


@router.put("/topics/toggle")
async def toggle_topic(
    topic: str = Query(..., description="Topic name to toggle"),
    enabled: bool = Query(..., description="Whether to enable or disable"),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Enable or disable a topic for knowledge search.

    Args:
        topic: Topic name
        enabled: True to enable, False to disable

    Returns:
        Updated list of enabled topics
    """


    from config import runtime_config

    current_enabled = runtime_config.get_enabled_topics()

    if enabled and topic not in current_enabled:
        current_enabled.append(topic)
    elif not enabled and topic in current_enabled:
        current_enabled.remove(topic)

    runtime_config.set_enabled_topics(current_enabled)
    runtime_config.save_overrides()

    return {
        "success": True,
        "topic": topic,
        "enabled": enabled,
        "enabled_topics": current_enabled,
    }
