#!/usr/bin/env python3
"""
TODO/FIXME Scanner - Find incomplete work in the codebase.

Usage:
    python scripts/todo_scan.py              # All TODOs and FIXMEs
    python scripts/todo_scan.py --fixme      # Only FIXMEs
    python scripts/todo_scan.py --backend    # Only backend
    python scripts/todo_scan.py --frontend   # Only frontend

Helps Claude know what's unfinished before making changes.
"""

import sys
import re
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Patterns to search for
PATTERNS = [
    (r"#\s*TODO:?\s*(.+)$", "TODO"),
    (r"#\s*FIXME:?\s*(.+)$", "FIXME"),
    (r"#\s*HACK:?\s*(.+)$", "HACK"),
    (r"#\s*XXX:?\s*(.+)$", "XXX"),
    (r"//\s*TODO:?\s*(.+)$", "TODO"),
    (r"//\s*FIXME:?\s*(.+)$", "FIXME"),
    (r"/\*\s*TODO:?\s*(.+?)\*/", "TODO"),
    (r"{/\*\s*TODO:?\s*(.+?)\*/}", "TODO"),
]


def scan_file(file_path: Path) -> list:
    """Scan a file for TODO/FIXME comments."""
    results = []

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return results

    lines = content.split("\n")

    for line_num, line in enumerate(lines, 1):
        for pattern, tag in PATTERNS:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                comment = match.group(1).strip()
                results.append({
                    "file": file_path,
                    "line": line_num,
                    "tag": tag,
                    "comment": comment[:80],  # Truncate long comments
                })
                break  # Only match first pattern per line

    return results


def scan_directory(directory: Path, extensions: list) -> list:
    """Scan a directory for TODO/FIXME comments."""
    results = []

    for ext in extensions:
        for file_path in directory.rglob(f"*{ext}"):
            # Skip common exclusions
            path_str = str(file_path)
            if any(x in path_str for x in ["node_modules", "__pycache__", ".git", "venv", ".venv", ".next"]):
                continue

            results.extend(scan_file(file_path))

    return results


def main():
    # Parse args
    fixme_only = "--fixme" in sys.argv
    backend_only = "--backend" in sys.argv
    frontend_only = "--frontend" in sys.argv

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return

    results = []

    # Scan backend
    if not frontend_only:
        backend_dir = PROJECT_ROOT / "backend"
        if backend_dir.exists():
            results.extend(scan_directory(backend_dir, [".py"]))

    # Scan frontend
    if not backend_only:
        frontend_dir = PROJECT_ROOT / "frontend"
        if frontend_dir.exists():
            results.extend(scan_directory(frontend_dir, [".ts", ".tsx", ".js", ".jsx"]))

    # Scan scripts
    if not frontend_only and not backend_only:
        scripts_dir = PROJECT_ROOT / "scripts"
        if scripts_dir.exists():
            results.extend(scan_directory(scripts_dir, [".py", ".sh"]))

    # Filter if needed
    if fixme_only:
        results = [r for r in results if r["tag"] == "FIXME"]

    if not results:
        print("No TODO/FIXME comments found.")
        return

    # Group by file
    by_file = defaultdict(list)
    for r in results:
        rel_path = r["file"].relative_to(PROJECT_ROOT)
        by_file[str(rel_path)].append(r)

    # Print summary
    print(f"\n=== Found {len(results)} items ===\n")

    # Count by tag
    tag_counts = defaultdict(int)
    for r in results:
        tag_counts[r["tag"]] += 1

    print("Summary:")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")
    print()

    # Print details
    print("Details:")
    for file_path, items in sorted(by_file.items()):
        print(f"\n{file_path}:")
        for item in items:
            print(f"  {item['line']:4}: [{item['tag']}] {item['comment']}")


if __name__ == "__main__":
    main()
