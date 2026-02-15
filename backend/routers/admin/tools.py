"""
Admin endpoints for tool inventory and custom tool scaffolding.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field

from . import router
from services.admin_auth import verify_admin
from services.custom_tool_scaffold import (
    custom_manifest_path,
    custom_tools_dir,
    list_custom_tools,
    scaffold_custom_tool,
    update_custom_tool,
)

logger = logging.getLogger(__name__)


class ToolScaffoldRequest(BaseModel):
    name: str = Field(..., description="snake_case tool name")
    friendly_name: Optional[str] = None
    brief: Optional[str] = None
    description: Optional[str] = None
    category: str = "analysis"
    parameters: Optional[Dict[str, Any]] = None
    required_params: Optional[List[str]] = None
    enabled: bool = True
    provides_citations: bool = False
    updates_session: bool = False
    triggers_analysis_result: bool = False
    extracts_nested_result: bool = False


class CustomToolUpdateRequest(BaseModel):
    friendly_name: Optional[str] = None
    brief: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    required_params: Optional[List[str]] = None
    enabled: Optional[bool] = None
    provides_citations: Optional[bool] = None
    updates_session: Optional[bool] = None
    triggers_analysis_result: Optional[bool] = None
    extracts_nested_result: Optional[bool] = None


def _tool_source(executor_module: str, active_domain: str) -> str:
    if executor_module.startswith("routers.chat_executors"):
        return "core"
    if executor_module.startswith(f"domains.{active_domain}.custom_tools"):
        return "custom"
    if executor_module.startswith(f"domains.{active_domain}."):
        return "domain"
    if executor_module.startswith("domains."):
        return "other_domain"
    return "other"


def _reload_tools() -> int:
    from tools.registry import ToolRegistry
    from routers.chat_prompts import clear_prompt_caches

    tool_count = ToolRegistry.reinitialize()
    clear_prompt_caches()
    return tool_count


@router.get("/tools")
async def list_tools(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    from domain_loader import get_domain_config
    from tools.registry import ToolRegistry, register_all_tools

    register_all_tools()

    domain = get_domain_config()
    all_tools = []
    for tool in ToolRegistry.get_all_tools().values():
        executor_module = getattr(tool.executor, "__module__", "")
        all_tools.append(
            {
                "name": tool.name,
                "friendly_name": tool.friendly_name or tool.name.replace("_", " ").title(),
                "brief": tool.brief or "",
                "description": tool.description,
                "category": tool.category.value,
                "required_params": tool.required_params,
                "parameters": tool.parameters,
                "source": _tool_source(executor_module, domain.name),
                "executor_module": executor_module,
                "requires_config": tool.requires_config,
            }
        )

    all_tools.sort(key=lambda t: (t["source"], t["name"]))

    mcp_enabled = bool(os.environ.get("MCP_API_KEY"))
    return {
        "success": True,
        "domain": domain.name,
        "tool_count": len(all_tools),
        "tools": all_tools,
        "custom_tools": list_custom_tools(domain.name),
        "custom_tools_path": str(custom_tools_dir(domain.name)),
        "custom_manifest_path": str(custom_manifest_path(domain.name)),
        "mcp": {
            "enabled": mcp_enabled,
            "tools_endpoint": "/api/mcp/tools",
            "execute_endpoint": "/api/mcp/execute",
            "api_key_configured": mcp_enabled,
        },
    }


@router.post("/tools/reload")
async def reload_tools(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    try:
        count = _reload_tools()
        return {"success": True, "tool_count": count}
    except Exception as exc:
        logger.error(f"Tool reload failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tool reload failed: {exc}")


@router.post("/tools/scaffold")
async def scaffold_tool(
    request: ToolScaffoldRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    from domain_loader import get_domain_config

    domain = get_domain_config()
    try:
        entry = scaffold_custom_tool(domain.name, request.model_dump())
        count = _reload_tools()
        return {
            "success": True,
            "domain": domain.name,
            "tool": entry,
            "tool_count": count,
            "message": f"Scaffolded custom tool '{entry['name']}' in domain '{domain.name}'",
        }
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Tool scaffold failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tool scaffold failed: {exc}")


@router.put("/tools/custom/{tool_name}")
async def update_tool(
    tool_name: str,
    request: CustomToolUpdateRequest,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    from domain_loader import get_domain_config

    domain = get_domain_config()
    updates = {
        key: value for key, value in request.model_dump().items() if value is not None
    }

    try:
        updated = update_custom_tool(domain.name, tool_name, updates)
        count = _reload_tools()
        return {
            "success": True,
            "domain": domain.name,
            "tool": updated,
            "tool_count": count,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Tool update failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tool update failed: {exc}")

