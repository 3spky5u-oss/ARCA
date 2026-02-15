"""
Auto-loader for admin-generated custom tools.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

from tools.registry import ToolCategory, ToolDefinition, ToolRegistry

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path(__file__).with_name("manifest.json")


def _load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"version": 1, "tools": []}

    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Custom tool manifest read failed: {exc}")
        return {"version": 1, "tools": []}

    if not isinstance(data, dict):
        return {"version": 1, "tools": []}
    if not isinstance(data.get("tools"), list):
        data["tools"] = []
    return data


def _to_category(raw: str | None) -> ToolCategory:
    if raw:
        try:
            return ToolCategory(raw)
        except Exception:
            pass
    return ToolCategory.ANALYSIS


def register_custom_tools() -> int:
    manifest = _load_manifest()
    specs = manifest.get("tools", [])
    if not specs:
        return 0

    registered = 0
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        if not spec.get("enabled", True):
            continue

        name = str(spec.get("name") or "").strip()
        module_name = str(spec.get("module") or name).strip()
        executor_name = str(spec.get("executor") or f"execute_{name}").strip()
        if not name or not module_name or not executor_name:
            continue

        try:
            module = importlib.import_module(f"{__package__}.{module_name}")
            executor = getattr(module, executor_name)
        except Exception as exc:
            logger.warning(f"Custom tool load failed for {name}: {exc}")
            continue

        ToolRegistry.register(
            ToolDefinition(
                name=name,
                friendly_name=str(spec.get("friendly_name") or name.replace("_", " ").title()),
                brief=str(spec.get("brief") or "Custom tool"),
                description=str(spec.get("description") or f"Custom tool: {name}"),
                parameters=spec.get("parameters") or {},
                required_params=spec.get("required_params") or [],
                executor=executor,
                category=_to_category(spec.get("category")),
                provides_citations=bool(spec.get("provides_citations", False)),
                updates_session=bool(spec.get("updates_session", False)),
                triggers_analysis_result=bool(spec.get("triggers_analysis_result", False)),
                extracts_nested_result=bool(spec.get("extracts_nested_result", False)),
            )
        )
        registered += 1

    if registered:
        logger.info(f"Registered {registered} custom tools from {MANIFEST_PATH}")
    return registered
