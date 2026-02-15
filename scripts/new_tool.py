#!/usr/bin/env python3
"""
Tool Scaffold Generator - Create boilerplate for new tool modules.

Usage:
    python scripts/new_tool.py toolname "Description of what it does"

Creates:
    backend/tools/toolname/
        __init__.py     (public API exports)
        core.py         (main implementation stub)
    docs/tools/TOOLNAME.md  (documentation from template)

Example:
    python scripts/new_tool.py weatherr "Weather data integration for site conditions"
"""

import sys
import os
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TOOLS_DIR = PROJECT_ROOT / "backend" / "tools"
DOCS_DIR = PROJECT_ROOT / "docs" / "tools"
TEMPLATE_PATH = DOCS_DIR / "TEMPLATE.md"


def create_init_py(tool_name: str, description: str) -> str:
    """Generate __init__.py content."""
    class_name = tool_name.title().replace("_", "")
    return f'''"""
{tool_name.title()} - {description}

Public API:
    from tools.{tool_name} import execute_{tool_name}, {class_name}
"""

from .core import execute_{tool_name}, {class_name}

__all__ = [
    "execute_{tool_name}",
    "{class_name}",
]


def warm_models():
    """Pre-load any models (called on app startup)."""
    pass


def get_stats() -> dict:
    """Return statistics for admin panel."""
    return {{
        "status": "ready",
    }}
'''


def create_core_py(tool_name: str, description: str) -> str:
    """Generate core.py content."""
    class_name = tool_name.title().replace("_", "")
    return f'''"""
{tool_name.title()} - Core Implementation

{description}
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class {class_name}Result:
    """Result from {tool_name} operation."""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None


class {class_name}:
    """
    {description}

    Usage:
        tool = {class_name}()
        result = tool.process(input_data)
    """

    def __init__(self):
        """Initialize {class_name}."""
        pass

    def process(self, data: Any) -> {class_name}Result:
        """
        Main processing method.

        Args:
            data: Input data to process

        Returns:
            {class_name}Result with success status and output data
        """
        try:
            # TODO: Implement processing logic
            result_data = {{"processed": True}}
            return {class_name}Result(success=True, data=result_data)
        except Exception as e:
            logger.error(f"{class_name} error: {{e}}", exc_info=True)
            return {class_name}Result(success=False, data={{}}, error=str(e))


def execute_{tool_name}(**kwargs) -> Dict[str, Any]:
    """
    Executor function for chat router integration.

    This function is registered in tools/registry.py and called
    when the LLM invokes the {tool_name} tool.

    Args:
        **kwargs: Parameters from LLM tool call

    Returns:
        Dict with success status and results
    """
    try:
        tool = {class_name}()
        result = tool.process(kwargs)

        if result.success:
            return {{
                "success": True,
                **result.data,
            }}
        else:
            return {{
                "success": False,
                "error": result.error,
            }}
    except Exception as e:
        logger.error(f"execute_{tool_name} failed: {{e}}", exc_info=True)
        return {{
            "success": False,
            "error": str(e),
        }}
'''


def create_doc(tool_name: str, description: str) -> str:
    """Generate documentation from template."""
    if TEMPLATE_PATH.exists():
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
    else:
        template = "# [TOOL_NAME]\n\n[Description]"

    class_name = tool_name.title().replace("_", "")

    # Replace placeholders
    doc = template
    doc = doc.replace("[TOOL_NAME]", tool_name.title())
    doc = doc.replace("[Brief Description]", description)
    doc = doc.replace("[toolname]", tool_name)
    doc = doc.replace("[tool_name]", tool_name)
    doc = doc.replace("MainClass", class_name)
    doc = doc.replace("main_function", f"execute_{tool_name}")
    doc = doc.replace("[SIMPLE | RAG | ANALYSIS | EXTERNAL | DOCUMENT]", "SIMPLE")
    doc = doc.replace("[1-2 sentences describing what this tool does and when it's used]", description)

    return doc


def create_registry_snippet(tool_name: str, description: str) -> str:
    """Generate registry.py registration snippet."""
    return f'''
# Add to backend/tools/registry.py:

from tools.{tool_name} import execute_{tool_name}

ToolRegistry.register(ToolDefinition(
    name="{tool_name}",
    description="{description}",
    parameters={{
        "param1": {{"type": "string", "description": "Description of param1"}},
    }},
    required_params=["param1"],
    executor=execute_{tool_name},
    category=ToolCategory.SIMPLE,
    provides_citations=False,
    triggers_analysis_result=False,
))
'''


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    tool_name = sys.argv[1].lower().replace("-", "_")
    description = sys.argv[2]

    # Validate tool name
    if not tool_name.isidentifier():
        print(f"Error: '{tool_name}' is not a valid Python identifier")
        sys.exit(1)

    # Check if tool already exists
    tool_dir = TOOLS_DIR / tool_name
    if tool_dir.exists():
        print(f"Error: Tool directory already exists: {tool_dir}")
        sys.exit(1)

    print(f"Creating tool: {tool_name}")
    print(f"Description: {description}")
    print()

    # Create tool directory
    tool_dir.mkdir(parents=True)
    print(f"Created: {tool_dir}/")

    # Create __init__.py
    init_path = tool_dir / "__init__.py"
    init_path.write_text(create_init_py(tool_name, description), encoding="utf-8")
    print(f"Created: {init_path}")

    # Create core.py
    core_path = tool_dir / "core.py"
    core_path.write_text(create_core_py(tool_name, description), encoding="utf-8")
    print(f"Created: {core_path}")

    # Create documentation
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    doc_path = DOCS_DIR / f"{tool_name.upper()}.md"
    doc_path.write_text(create_doc(tool_name, description), encoding="utf-8")
    print(f"Created: {doc_path}")

    # Print registry snippet
    print()
    print("=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print(create_registry_snippet(tool_name, description))
    print()
    print("1. Implement the tool logic in core.py")
    print("2. Add registration to backend/tools/registry.py (snippet above)")
    print("3. Update the documentation in docs/tools/")
    print("4. Add tests in backend/tests/test_{}/".format(tool_name))


if __name__ == "__main__":
    main()
