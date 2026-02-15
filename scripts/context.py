#!/usr/bin/env python3
"""
Context Loader - Output focused documentation for specific modules.

Usage:
    python scripts/context.py frontend     # Frontend architecture
    python scripts/context.py backend      # Backend architecture
    python scripts/context.py cohesionn    # RAG system
    python scripts/context.py exceedee     # Compliance checking
    python scripts/context.py tools        # All tools overview
    python scripts/context.py all          # Full CLAUDE.md
    python scripts/context.py --list       # Show available contexts
"""

import sys
from pathlib import Path

# Base paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DOCS_DIR = PROJECT_ROOT / "docs"
TOOLS_DIR = DOCS_DIR / "tools"

# Context mappings: module -> list of doc files
CONTEXTS = {
    "frontend": [
        DOCS_DIR / "FRONTEND.md",
    ],
    "backend": [
        DOCS_DIR / "BACKEND.md",
    ],
    "tools": [
        DOCS_DIR / "TOOLS_OVERVIEW.md",
    ],
    "testing": [
        DOCS_DIR / "TESTING.md",
    ],
    "admin": [
        DOCS_DIR / "admin" / "ADMIN_PANEL.md",
    ],
    "cohesionn": [
        TOOLS_DIR / "COHESIONN.md",
    ],
    "exceedee": [
        TOOLS_DIR / "EXCEEDEE.md",
    ],
    "redactrr": [
        TOOLS_DIR / "REDACTRR.md",
    ],
    "readd": [
        TOOLS_DIR / "READD.md",
    ],
    "solverr": [
        TOOLS_DIR / "SOLVERR.md",
    ],
    "template": [
        TOOLS_DIR / "TEMPLATE.md",
    ],
    "all": [
        PROJECT_ROOT / "CLAUDE.md",
    ],
    # Composite contexts
    "chat": [
        DOCS_DIR / "BACKEND.md",
        DOCS_DIR / "TOOLS_OVERVIEW.md",
    ],
    "rag": [
        TOOLS_DIR / "COHESIONN.md",
        TOOLS_DIR / "READD.md",
    ],
    "full-tools": [
        DOCS_DIR / "TOOLS_OVERVIEW.md",
        TOOLS_DIR / "COHESIONN.md",
        TOOLS_DIR / "EXCEEDEE.md",
        TOOLS_DIR / "REDACTRR.md",
        TOOLS_DIR / "READD.md",
        TOOLS_DIR / "SOLVERR.md",
    ],
}


def get_file_size(path: Path) -> str:
    """Get human-readable file size."""
    if not path.exists():
        return "N/A"
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"


def count_lines(path: Path) -> int:
    """Count lines in a file."""
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def list_contexts():
    """List available contexts with sizes."""
    print("Available contexts:\n")
    print(f"{'Context':<15} {'Files':<5} {'Est. Lines':<12} Description")
    print("-" * 60)

    descriptions = {
        "frontend": "React/Next.js frontend architecture",
        "backend": "FastAPI backend, routers, config",
        "tools": "Tool overview and registry",
        "testing": "pytest patterns and fixtures",
        "admin": "Admin panel tabs and endpoints",
        "cohesionn": "RAG pipeline deep dive",
        "exceedee": "Compliance checking system",
        "redactrr": "PII redaction pipeline",
        "readd": "Document extraction pipeline",
        "solverr": "Engineering calculations",
        "template": "Template for new tool docs",
        "all": "Full CLAUDE.md (main reference)",
        "chat": "Backend + tools (for chat work)",
        "rag": "Cohesionn + Readd (for RAG work)",
        "full-tools": "All tool documentation",
    }

    for name, files in CONTEXTS.items():
        total_lines = sum(count_lines(f) for f in files if f.exists())
        file_count = len([f for f in files if f.exists()])
        desc = descriptions.get(name, "")
        print(f"{name:<15} {file_count:<5} ~{total_lines:<10} {desc}")


def load_context(name: str) -> str:
    """Load and concatenate documentation for a context."""
    if name not in CONTEXTS:
        available = ", ".join(sorted(CONTEXTS.keys()))
        return f"Unknown context: {name}\n\nAvailable: {available}"

    files = CONTEXTS[name]
    output = []

    for file_path in files:
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            output.append(f"# === {file_path.name} ===\n\n{content}")
        else:
            output.append(f"# === {file_path.name} === (NOT FOUND)\n")

    return "\n\n".join(output)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUse --list to see available contexts")
        sys.exit(1)

    arg = sys.argv[1].lower()

    if arg in ("--list", "-l", "list"):
        list_contexts()
    elif arg in ("--help", "-h", "help"):
        print(__doc__)
    else:
        content = load_context(arg)
        print(content)


if __name__ == "__main__":
    main()
