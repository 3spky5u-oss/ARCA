"""
ARCA Chat Executors - Common Utilities

Shared utilities used by multiple executor modules.
"""

import re
from pathlib import Path


def clean_citation_source(source: str) -> str:
    """Clean up file paths to readable citation titles."""
    # Get filename without path and extension
    name = Path(source).stem

    # If stem is empty or still looks like a path, try .name
    if not name or "\\" in name or "/" in name:
        name = Path(source).name

    # Remove common prefixes/suffixes
    name = re.sub(r"^\d{10,}-", "", name)  # ISBN prefixes
    name = re.sub(r"^\[\d+\]", "", name)  # [1234] prefixes
    name = re.sub(r"\(\d{4}\)$", "", name)  # Year suffixes like (2014)

    # Clean separators
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()

    # Title case if all lowercase
    if name == name.lower():
        name = name.title()

    return name if name else "Unknown Source"
