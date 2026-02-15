"""
Helpers for domain-agnostic custom tool scaffolding.

This service creates and maintains `domains/<active_domain>/custom_tools/*`
without hard-coding any private domain pack details.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


VALID_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


LOADER_TEMPLATE = '''"""
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
'''


INIT_TEMPLATE = '''"""
Custom tools package.

This file is generated/maintained by ARCA admin tooling.
"""

from .loader import register_custom_tools
'''


def _repo_root() -> Path:
    # /app/services/custom_tool_scaffold.py -> repo root is /app
    return Path(__file__).resolve().parents[1]


def custom_tools_dir(domain_name: str) -> Path:
    return _repo_root() / "domains" / domain_name / "custom_tools"


def custom_manifest_path(domain_name: str) -> Path:
    return custom_tools_dir(domain_name) / "manifest.json"


def ensure_custom_tools_package(domain_name: str) -> None:
    package_dir = custom_tools_dir(domain_name)
    package_dir.mkdir(parents=True, exist_ok=True)

    init_path = package_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text(INIT_TEMPLATE, encoding="utf-8")

    loader_path = package_dir / "loader.py"
    if not loader_path.exists():
        loader_path.write_text(LOADER_TEMPLATE, encoding="utf-8")

    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps({"version": 1, "tools": []}, indent=2) + "\n",
            encoding="utf-8",
        )


def load_custom_manifest(domain_name: str) -> Dict[str, Any]:
    try:
        ensure_custom_tools_package(domain_name)
    except PermissionError:
        # Read-only domains mount: expose empty custom tools instead of failing admin tools panel.
        return {"version": 1, "tools": []}
    except OSError:
        return {"version": 1, "tools": []}
    manifest_path = custom_manifest_path(domain_name)
    if not manifest_path.exists():
        return {"version": 1, "tools": []}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        data = {"version": 1, "tools": []}
    if "tools" not in data or not isinstance(data["tools"], list):
        data["tools"] = []
    if "version" not in data:
        data["version"] = 1
    return data


def save_custom_manifest(domain_name: str, manifest: Dict[str, Any]) -> None:
    manifest_path = custom_manifest_path(domain_name)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def validate_tool_name(name: str) -> str:
    normalized = name.strip().lower()
    if not VALID_TOOL_NAME.match(normalized):
        raise ValueError("Tool name must be snake_case, 3-64 chars, starting with a letter.")
    return normalized


def list_custom_tools(domain_name: str) -> List[Dict[str, Any]]:
    manifest = load_custom_manifest(domain_name)
    return [t for t in manifest.get("tools", []) if isinstance(t, dict)]


def scaffold_custom_tool(domain_name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    name = validate_tool_name(str(spec.get("name") or ""))
    try:
        ensure_custom_tools_package(domain_name)
    except PermissionError as exc:
        raise PermissionError(
            "Custom tools directory is not writable. Ensure /app/domains is mounted read-write."
        ) from exc

    package_dir = custom_tools_dir(domain_name)
    module_path = package_dir / f"{name}.py"
    if module_path.exists():
        raise FileExistsError(f"Tool module already exists: {module_path.name}")

    executor_name = f"execute_{name}"
    module_content = f'''"""
Custom tool scaffold: {name}
"""

from __future__ import annotations

from typing import Any, Dict


def {executor_name}(**kwargs: Any) -> Dict[str, Any]:
    """
    TODO: Implement this tool.
    """
    return {{
        "success": False,
        "error": "Tool '{name}' is scaffolded but not implemented yet.",
        "received": kwargs,
    }}
'''
    module_path.write_text(module_content, encoding="utf-8")

    parameters = spec.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {
            "input": {
                "type": "string",
                "description": "Primary input for this tool",
            }
        }

    required_params = spec.get("required_params")
    if not isinstance(required_params, list):
        required_params = []

    manifest = load_custom_manifest(domain_name)
    entry = {
        "name": name,
        "module": name,
        "executor": executor_name,
        "friendly_name": spec.get("friendly_name") or name.replace("_", " ").title(),
        "brief": spec.get("brief") or "Custom tool",
        "description": spec.get("description") or f"Custom tool: {name}",
        "category": spec.get("category") or "analysis",
        "parameters": parameters,
        "required_params": required_params,
        "enabled": bool(spec.get("enabled", True)),
        "provides_citations": bool(spec.get("provides_citations", False)),
        "updates_session": bool(spec.get("updates_session", False)),
        "triggers_analysis_result": bool(spec.get("triggers_analysis_result", False)),
        "extracts_nested_result": bool(spec.get("extracts_nested_result", False)),
    }
    manifest["tools"] = [t for t in manifest["tools"] if t.get("name") != name] + [entry]
    save_custom_manifest(domain_name, manifest)
    return entry


def update_custom_tool(domain_name: str, tool_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    name = validate_tool_name(tool_name)
    manifest = load_custom_manifest(domain_name)
    tools = manifest.get("tools", [])
    match = None
    for tool in tools:
        if tool.get("name") == name:
            match = tool
            break
    if match is None:
        raise KeyError(f"Custom tool not found: {name}")

    allowed = {
        "friendly_name",
        "brief",
        "description",
        "category",
        "parameters",
        "required_params",
        "enabled",
        "provides_citations",
        "updates_session",
        "triggers_analysis_result",
        "extracts_nested_result",
    }
    for key, value in updates.items():
        if key in allowed and value is not None:
            match[key] = value

    save_custom_manifest(domain_name, manifest)
    return match
