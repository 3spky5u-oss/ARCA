"""
Domain management and model deletion endpoints.
"""

import logging
import urllib.parse
from typing import Dict, Any

from fastapi import Depends, HTTPException

from . import router
from .models import DomainActivateRequest
from services.admin_auth import verify_admin

logger = logging.getLogger(__name__)


@router.get("/domains")
async def list_domains(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """List available domain packs and current active domain."""
    from domain_loader import get_domain_config, list_available_domains

    current = get_domain_config()
    domains = list_available_domains()

    return {
        "current": current.name,
        "admin_visible": current.admin_visible,
        "domains": domains,
    }


@router.post("/domains/activate")
async def activate_domain(
    request: DomainActivateRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Switch the active domain pack.

    Reloads tools and lexicon immediately. Route changes require a restart.
    """
    from domain_loader import set_active_domain, reload_domain, get_domain_config
    from tools.registry import ToolRegistry

    old_domain = get_domain_config().name

    if request.domain == old_domain:
        return {
            "success": True,
            "message": f"Already on domain '{request.domain}'",
            "domain": old_domain,
            "tools": len(ToolRegistry._tools),
            "restart_needed": False,
        }

    try:
        # Persist the choice
        set_active_domain(request.domain)

        # Reload domain config + lexicon
        new_config = reload_domain()

        # Re-register tools for new domain
        tool_count = ToolRegistry.reinitialize()

        # Clear prompt caches so system prompt picks up new lexicon pipeline config
        from routers.chat_prompts import clear_prompt_caches
        clear_prompt_caches()

        # Check if routes changed (requires restart)
        old_config_routes = set()  # We can't easily get old routes after reload
        restart_needed = new_config.routes != []  # Routes changed from/to domain

        logger.info(
            f"Domain switched: {old_domain} -> {new_config.name} "
            f"({tool_count} tools)"
        )

        return {
            "success": True,
            "message": f"Switched to '{new_config.display_name}'",
            "domain": new_config.name,
            "display_name": new_config.display_name,
            "tools": tool_count,
            "restart_needed": restart_needed,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Domain switch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Domain switch failed: {e}")


@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Delete a GGUF model file from disk.

    Warning: Cannot delete models currently assigned to config slots.
    """

    # URL decode the model name
    model_name = urllib.parse.unquote(model_name)

    try:
        from config import runtime_config

        # Check if model is in use
        in_use = []
        if model_name == runtime_config.model_chat:
            in_use.append("chat")
        if model_name == runtime_config.model_code:
            in_use.append("code")
        if model_name == runtime_config.model_expert:
            in_use.append("expert")
        if model_name == runtime_config.model_vision:
            in_use.append("vision")
        if model_name == runtime_config.model_vision_structured:
            in_use.append("vision_structured")
        if in_use:
            raise HTTPException(
                status_code=400, detail=f"Cannot delete model in use by: {', '.join(in_use)}. Reassign first."
            )

        # Delete GGUF file from models directory
        from services.llm_config import get_model_path
        model_path = get_model_path(model_name)
        if model_path.exists():
            model_path.unlink()
            logger.info(f"Deleted model file: {model_name}")
            return {"success": True, "message": f"Deleted {model_name}"}
        else:
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_name}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
