#!/usr/bin/env python3
"""
Pattern Finder - Find examples of common patterns in the codebase.

Usage:
    python scripts/find_pattern.py "tool registration"
    python scripts/find_pattern.py "router endpoint"
    python scripts/find_pattern.py "frontend component"
    python scripts/find_pattern.py --list

This helps Claude find existing examples to follow when adding new code.
"""

import sys
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Pattern definitions: keyword -> (files to search, regex pattern, description)
PATTERNS = {
    "tool": {
        "description": "Tool registration in registry.py",
        "files": ["backend/tools/registry.py"],
        "pattern": r"ToolRegistry\.register\(ToolDefinition\([^)]+\)\)",
        "context_lines": 15,
    },
    "tool registration": {
        "description": "How tools are registered with the dispatcher",
        "files": ["backend/tools/registry.py"],
        "pattern": r"ToolRegistry\.register\(ToolDefinition\(",
        "context_lines": 20,
    },
    "router": {
        "description": "FastAPI router endpoint definitions",
        "files": ["backend/routers/admin.py", "backend/routers/upload.py"],
        "pattern": r"@router\.(get|post|put|delete)\(",
        "context_lines": 10,
    },
    "router endpoint": {
        "description": "How API endpoints are defined",
        "files": ["backend/routers/admin.py"],
        "pattern": r"@router\.(get|post|put|delete)\([^)]+\)\nasync def",
        "context_lines": 15,
    },
    "websocket": {
        "description": "WebSocket message handling",
        "files": ["backend/routers/chat.py"],
        "pattern": r"await websocket\.(send|receive)",
        "context_lines": 5,
    },
    "executor": {
        "description": "Tool executor function pattern",
        "files": ["backend/routers/chat.py"],
        "pattern": r"def execute_\w+\([^)]*\) -> Dict",
        "context_lines": 20,
    },
    "dataclass": {
        "description": "Dataclass definitions for type safety",
        "files": ["backend/config.py", "backend/tools/exceedee/checker.py"],
        "pattern": r"@dataclass\nclass \w+:",
        "context_lines": 15,
    },
    "component": {
        "description": "React functional component pattern",
        "files": ["frontend/components/upload-zone.tsx", "frontend/components/exceedance-table.tsx"],
        "pattern": r"(const|function) \w+.*: (React\.)?FC",
        "context_lines": 20,
    },
    "frontend component": {
        "description": "React component with TypeScript",
        "files": ["frontend/components/upload-zone.tsx"],
        "pattern": r"interface \w+Props",
        "context_lines": 30,
    },
    "hook": {
        "description": "React hooks usage patterns",
        "files": ["frontend/app/page.tsx"],
        "pattern": r"const \[\w+, set\w+\] = useState",
        "context_lines": 3,
    },
    "useeffect": {
        "description": "useEffect patterns",
        "files": ["frontend/app/page.tsx"],
        "pattern": r"useEffect\(\(\) =>",
        "context_lines": 10,
    },
    "test": {
        "description": "Pytest test patterns",
        "files": ["backend/tests/test_solverr/test_bearing_capacity.py"],
        "pattern": r"def test_\w+\(",
        "context_lines": 10,
    },
    "fixture": {
        "description": "Pytest fixture definitions",
        "files": ["backend/tests/conftest.py"],
        "pattern": r"@pytest\.fixture",
        "context_lines": 10,
    },
    "admin tab": {
        "description": "Admin panel tab structure",
        "files": ["frontend/app/admin/page.tsx"],
        "pattern": r"<TabsContent value=",
        "context_lines": 15,
    },
    "api fetch": {
        "description": "Frontend API fetch patterns",
        "files": ["frontend/app/admin/page.tsx", "frontend/app/page.tsx"],
        "pattern": r"await fetch\(",
        "context_lines": 8,
    },
}


def list_patterns():
    """List available patterns."""
    print("Available patterns:\n")
    print(f"{'Pattern':<20} Description")
    print("-" * 60)
    for name, info in sorted(PATTERNS.items()):
        print(f"{name:<20} {info['description']}")


def find_pattern(pattern_name: str):
    """Find and display pattern examples."""
    # Find matching pattern
    pattern_name = pattern_name.lower()

    if pattern_name not in PATTERNS:
        # Try partial match
        matches = [k for k in PATTERNS if pattern_name in k]
        if len(matches) == 1:
            pattern_name = matches[0]
        elif len(matches) > 1:
            print(f"Multiple matches: {', '.join(matches)}")
            print("Please be more specific.")
            return
        else:
            print(f"Unknown pattern: {pattern_name}")
            print(f"Available: {', '.join(sorted(PATTERNS.keys()))}")
            return

    info = PATTERNS[pattern_name]
    print(f"\n=== Pattern: {pattern_name} ===")
    print(f"Description: {info['description']}")
    print()

    regex = re.compile(info["pattern"], re.MULTILINE | re.DOTALL)
    context_lines = info.get("context_lines", 10)

    for file_path in info["files"]:
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            print(f"[File not found: {file_path}]")
            continue

        content = full_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        matches = list(regex.finditer(content))
        if not matches:
            continue

        print(f"--- {file_path} ({len(matches)} examples) ---\n")

        for i, match in enumerate(matches[:3]):  # Show first 3 matches
            # Find line number
            line_num = content[:match.start()].count("\n") + 1

            # Get context
            start_line = max(0, line_num - 2)
            end_line = min(len(lines), line_num + context_lines)

            print(f"Example {i+1} (line {line_num}):")
            print("-" * 40)
            for j in range(start_line, end_line):
                prefix = ">>>" if j == line_num - 1 else "   "
                print(f"{prefix} {j+1:4}: {lines[j]}")
            print()

        if len(matches) > 3:
            print(f"... and {len(matches) - 3} more examples\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUse --list to see available patterns")
        sys.exit(1)

    arg = sys.argv[1].lower()

    if arg in ("--list", "-l", "list"):
        list_patterns()
    elif arg in ("--help", "-h", "help"):
        print(__doc__)
    else:
        # Join all args as the pattern name
        pattern_name = " ".join(sys.argv[1:]).lower()
        find_pattern(pattern_name)


if __name__ == "__main__":
    main()
