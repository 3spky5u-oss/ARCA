#!/usr/bin/env python3
"""
Dependency Viewer - Show dependencies for a module.

Usage:
    python scripts/deps.py cohesionn    # Show Cohesionn dependencies
    python scripts/deps.py exceedee     # Show Exceedee dependencies
    python scripts/deps.py chat         # Show chat router dependencies
    python scripts/deps.py --all        # Show all modules
    python scripts/deps.py --list       # List available modules

Example output:
    cohesionn depends on:
      External:
        - qdrant-client (vector storage)
        - sentence-transformers (embeddings, reranking)
      Internal:
        - tools.readd (PDF extraction)
        - config (RuntimeConfig)

    Used by:
        - routers/chat.py (execute_search_knowledge)
        - routers/admin_knowledge.py (ingestion endpoints)
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Dependency mappings (manually maintained for accuracy)
DEPENDENCIES = {
    "cohesionn": {
        "description": "Technical Knowledge RAG System",
        "external": [
            ("qdrant-client", "Vector database storage"),
            ("sentence-transformers", "Embeddings and reranking (ONNX backend)"),
            ("PyMuPDF (fitz)", "Basic PDF text access"),
        ],
        "internal": [
            ("tools.readd", "PDF extraction pipeline"),
            ("config", "RuntimeConfig for RAG settings"),
        ],
        "used_by": [
            ("routers/chat.py", "execute_search_knowledge()"),
            ("routers/admin_knowledge.py", "Ingestion, stats, topic management"),
            ("routers/upload.py", "Session document indexing"),
        ],
    },
    "exceedee": {
        "description": "Environmental Compliance Checking",
        "external": [
            ("openpyxl", "Excel parsing"),
            ("xlsxwriter", "Excel report generation"),
            ("python-docx", "Word report generation"),
            ("pandas", "Data manipulation (optional)"),
        ],
        "internal": [
            ("config", "RuntimeConfig"),
        ],
        "used_by": [
            ("routers/chat.py", "analyze_files, lookup_guideline, generate_report"),
            ("routers/admin_exceedee.py", "Guideline browsing"),
            ("routers/upload.py", "Lab data parsing"),
        ],
    },
    "redactrr": {
        "description": "Intelligent PII Redaction",
        "external": [
            ("llama-server", "LLM for context-aware detection"),
            ("python-docx", "Word document handling"),
            ("PyMuPDF (fitz)", "PDF redaction"),
        ],
        "internal": [
            ("config", "Model selection (REDACTRR_MODEL)"),
        ],
        "used_by": [
            ("routers/chat.py", "execute_redact_document()"),
        ],
    },
    "readd": {
        "description": "Intelligent Document Extraction",
        "external": [
            ("PyMuPDF (fitz)", "Basic text extraction"),
            ("pymupdf4llm", "Markdown-aware extraction"),
            ("marker", "ML-based layout extraction (optional)"),
        ],
        "internal": [
            ("tools.observationn", "Vision OCR fallback"),
            ("config", "Extractor settings"),
        ],
        "used_by": [
            ("tools.cohesionn/ingest.py", "PDF ingestion"),
            ("routers/admin.py", "Extraction tester"),
        ],
    },
    "solverr": {
        "description": "Engineering Calculations",
        "external": [
            ("math (stdlib)", "Trigonometric functions"),
        ],
        "internal": [],
        "used_by": [
            ("routers/chat.py", "execute_solve_engineering()"),
        ],
    },
    "observationn": {
        "description": "Vision-Based OCR",
        "external": [
            ("llama-server", "Qwen3-VL vision model"),
        ],
        "internal": [
            ("config", "Model selection (OBSERVATIONN_MODEL)"),
        ],
        "used_by": [
            ("tools.readd", "Fallback extractor for scanned docs"),
        ],
    },
    "chat": {
        "description": "Main Chat Router",
        "external": [
            ("fastapi", "WebSocket handling"),
            ("llama-server", "LLM inference"),
            ("httpx", "Web search (SearXNG)"),
        ],
        "internal": [
            ("tools.registry", "Tool dispatch"),
            ("tools.cohesionn", "Knowledge search"),
            ("tools.exceedee", "Compliance analysis"),
            ("tools.redactrr", "Document redaction"),
            ("tools.solverr", "Engineering calculations"),
            ("routers.upload", "File access (files_db)"),
            ("config", "Runtime parameters"),
        ],
        "used_by": [
            ("main.py", "Router registration"),
            ("frontend/app/page.tsx", "WebSocket connection"),
        ],
    },
    "registry": {
        "description": "Tool Registration & Dispatch",
        "external": [],
        "internal": [
            ("routers.chat", "Executor functions"),
        ],
        "used_by": [
            ("routers/chat.py", "Tool schema, execution"),
        ],
    },
    "config": {
        "description": "Runtime Configuration",
        "external": [
            ("threading", "Lock for thread safety"),
        ],
        "internal": [],
        "used_by": [
            ("routers/chat.py", "Context sizing, model selection"),
            ("routers/admin.py", "Config endpoints"),
            ("tools/*", "Settings access"),
        ],
    },
}


def list_modules():
    """List available modules."""
    print("Available modules:\n")
    print(f"{'Module':<15} Description")
    print("-" * 50)
    for name, info in sorted(DEPENDENCIES.items()):
        print(f"{name:<15} {info['description']}")


def show_deps(module: str):
    """Show dependencies for a module."""
    if module not in DEPENDENCIES:
        print(f"Unknown module: {module}")
        print(f"Available: {', '.join(sorted(DEPENDENCIES.keys()))}")
        sys.exit(1)

    info = DEPENDENCIES[module]

    print(f"\n{module.upper()}: {info['description']}")
    print("=" * 50)

    # External dependencies
    if info["external"]:
        print("\nExternal Dependencies:")
        for pkg, desc in info["external"]:
            print(f"  - {pkg}: {desc}")
    else:
        print("\nExternal Dependencies: None")

    # Internal dependencies
    if info["internal"]:
        print("\nInternal Dependencies:")
        for pkg, desc in info["internal"]:
            print(f"  - {pkg}: {desc}")
    else:
        print("\nInternal Dependencies: None")

    # Used by
    if info["used_by"]:
        print("\nUsed By:")
        for location, desc in info["used_by"]:
            print(f"  - {location}: {desc}")
    else:
        print("\nUsed By: (not tracked)")


def show_all():
    """Show all module dependencies."""
    for module in sorted(DEPENDENCIES.keys()):
        show_deps(module)
        print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1].lower()

    if arg in ("--list", "-l", "list"):
        list_modules()
    elif arg in ("--all", "-a", "all"):
        show_all()
    elif arg in ("--help", "-h", "help"):
        print(__doc__)
    else:
        show_deps(arg)


if __name__ == "__main__":
    main()
