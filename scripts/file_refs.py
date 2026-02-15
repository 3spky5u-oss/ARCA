#!/usr/bin/env python3
"""
File Reference Tracker - Show what imports a file and what it imports.

Usage:
    python scripts/file_refs.py backend/config.py
    python scripts/file_refs.py backend/routers/chat.py
    python scripts/file_refs.py frontend/app/page.tsx

Helps Claude understand impact of changes to a file.
"""

import sys
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def get_python_imports(file_path: Path) -> list:
    """Extract imports from a Python file."""
    if not file_path.exists():
        return []

    content = file_path.read_text(encoding="utf-8")
    imports = []

    # from X import Y
    for match in re.finditer(r"^from ([\w.]+) import", content, re.MULTILINE):
        imports.append(match.group(1))

    # import X
    for match in re.finditer(r"^import ([\w.]+)", content, re.MULTILINE):
        imports.append(match.group(1))

    return imports


def get_ts_imports(file_path: Path) -> list:
    """Extract imports from a TypeScript file."""
    if not file_path.exists():
        return []

    content = file_path.read_text(encoding="utf-8")
    imports = []

    # import X from "Y"
    for match in re.finditer(r'import .+ from ["\']([^"\']+)["\']', content):
        imports.append(match.group(1))

    return imports


def find_python_references(target_module: str) -> list:
    """Find Python files that import this module."""
    refs = []
    backend_dir = PROJECT_ROOT / "backend"

    for py_file in backend_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        content = py_file.read_text(encoding="utf-8")

        # Check for imports of the target
        if re.search(rf"from {target_module}", content) or \
           re.search(rf"import {target_module}", content):
            # Find what's imported
            imports_what = []
            for match in re.finditer(rf"from {target_module} import ([\w, ]+)", content):
                imports_what.extend([x.strip() for x in match.group(1).split(",")])

            rel_path = py_file.relative_to(PROJECT_ROOT)
            refs.append((str(rel_path), imports_what))

    return refs


def find_ts_references(target_path: str) -> list:
    """Find TypeScript files that import this file."""
    refs = []
    frontend_dir = PROJECT_ROOT / "frontend"

    # Normalize target for matching
    target_name = Path(target_path).stem

    for ts_file in frontend_dir.rglob("*.tsx"):
        if "node_modules" in str(ts_file):
            continue

        content = ts_file.read_text(encoding="utf-8")

        # Check for imports
        if target_name in content:
            rel_path = ts_file.relative_to(PROJECT_ROOT)
            refs.append((str(rel_path), []))

    return refs


def analyze_file(file_path_str: str):
    """Analyze a file's imports and references."""
    # Normalize path
    if file_path_str.startswith("./"):
        file_path_str = file_path_str[2:]

    file_path = PROJECT_ROOT / file_path_str

    if not file_path.exists():
        print(f"File not found: {file_path_str}")
        return

    print(f"\n=== {file_path_str} ===\n")

    # Determine file type
    is_python = file_path.suffix == ".py"
    is_typescript = file_path.suffix in (".ts", ".tsx")

    # Get imports
    if is_python:
        imports = get_python_imports(file_path)
        local_imports = [i for i in imports if i.startswith(("tools", "routers", "config"))]
        external_imports = [i for i in imports if not i.startswith(("tools", "routers", "config"))]
    elif is_typescript:
        imports = get_ts_imports(file_path)
        local_imports = [i for i in imports if i.startswith(("@/", "./", "../"))]
        external_imports = [i for i in imports if not i.startswith(("@/", "./", "../"))]
    else:
        print(f"Unsupported file type: {file_path.suffix}")
        return

    # Show imports
    print("This file imports:")
    if local_imports:
        print("  Local:")
        for imp in sorted(set(local_imports)):
            print(f"    - {imp}")
    if external_imports:
        print("  External:")
        for imp in sorted(set(external_imports))[:10]:  # Limit external
            print(f"    - {imp}")
        if len(external_imports) > 10:
            print(f"    ... and {len(external_imports) - 10} more")
    if not imports:
        print("  (none)")

    print()

    # Find references
    print("Imported by:")
    if is_python:
        # Convert file path to module name
        rel_path = file_path.relative_to(PROJECT_ROOT / "backend")
        module_name = str(rel_path.with_suffix("")).replace("/", ".").replace("\\", ".")

        refs = find_python_references(module_name)
        if refs:
            for ref_path, imports_what in refs:
                if imports_what:
                    print(f"  - {ref_path} ({', '.join(imports_what)})")
                else:
                    print(f"  - {ref_path}")
        else:
            print("  (no references found)")
    elif is_typescript:
        refs = find_ts_references(file_path_str)
        if refs:
            for ref_path, _ in refs:
                print(f"  - {ref_path}")
        else:
            print("  (no references found)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]

    if arg in ("--help", "-h", "help"):
        print(__doc__)
    else:
        analyze_file(arg)


if __name__ == "__main__":
    main()
