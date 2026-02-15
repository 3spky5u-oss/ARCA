#!/usr/bin/env python3
"""
Redactrr CLI - Document redaction tool

Usage:
    python -m tools.redactrr redact /path/to/document.docx
    python -m tools.redactrr preview /path/to/document.pdf
    python -m tools.redactrr redact report.docx --output clean_report.docx
    python -m tools.redactrr redact report.docx --add-terms "Project ABC" "Client XYZ"
"""

import argparse
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def cmd_redact(args):
    """Redact a document"""
    from .redactor import redact_document

    input_path = Path(args.file)
    output_path = Path(args.output) if args.output else None
    additional = args.add_terms if args.add_terms else None

    print(f"Redacting: {input_path.name}")
    if not args.use_llm:
        print("  (LLM detection disabled, using regex only)")

    result = redact_document(
        input_path,
        output_path,
        use_llm=args.use_llm,
        additional_terms=additional,
    )

    if result.success:
        print("\nOK: Success!")
        print(f"  Output: {result.redacted_path}")
        print(f"  Entities found: {result.entities_found}")
        print(f"  Redactions made: {result.entities_redacted}")

        if result.entity_types:
            print("\n  By type:")
            for pii_type, count in sorted(result.entity_types.items()):
                print(f"    {pii_type}: {count}")

        if result.warnings:
            print("\n  Warnings:")
            for w in result.warnings:
                print(f"    - {w}")
    else:
        print(f"\nFAILED: {result.error}")


def cmd_preview(args):
    """Preview what would be redacted"""
    from .redactor import preview_redaction

    input_path = Path(args.file)

    print(f"Analyzing: {input_path.name}")

    result = preview_redaction(input_path, use_llm=args.use_llm)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\nFound {result['entities_found']} PII entities:\n")

    # Group by type
    by_type = {}
    for entity in result["entities"]:
        pii_type = entity["type"]
        if pii_type not in by_type:
            by_type[pii_type] = []
        by_type[pii_type].append(entity)

    for pii_type, entities in sorted(by_type.items()):
        print(f"  {pii_type.upper()} ({len(entities)}):")
        for e in entities[:10]:  # Limit display
            text = e["text"][:50] + "..." if len(e["text"]) > 50 else e["text"]
            print(f"    - {text} -> {e['replacement']}")
        if len(entities) > 10:
            print(f"    ... and {len(entities) - 10} more")
        print()


def cmd_batch(args):
    """Redact multiple documents"""
    from .redactor import Redactrr

    input_dir = Path(args.directory)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "redacted"
    output_dir.mkdir(exist_ok=True)

    redactor = Redactrr(use_llm=args.use_llm)

    # Find all supported files
    files = []
    for ext in [".docx", ".pdf"]:
        files.extend(input_dir.glob(f"*{ext}"))

    print(f"Found {len(files)} documents in {input_dir}")
    print(f"Output directory: {output_dir}\n")

    success = 0
    failed = 0

    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file_path.name}...", end=" ")

        output_path = output_dir / f"{file_path.stem}_REDACTED{file_path.suffix}"
        result = redactor.redact(file_path, output_path)

        if result.success:
            print(f"OK ({result.entities_redacted} redactions)")
            success += 1
        else:
            print(f"FAILED ({result.error})")
            failed += 1

    print(f"\nComplete: {success} succeeded, {failed} failed")


def main():
    parser = argparse.ArgumentParser(description="Redactrr - Intelligent document redaction")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Redact command
    redact_parser = subparsers.add_parser("redact", help="Redact a document")
    redact_parser.add_argument("file", help="Document to redact (.docx or .pdf)")
    redact_parser.add_argument("--output", "-o", help="Output path")
    redact_parser.add_argument(
        "--no-llm", dest="use_llm", action="store_false", help="Disable LLM detection (regex only)"
    )
    redact_parser.add_argument("--add-terms", nargs="+", help="Additional terms to redact")

    # Preview command
    preview_parser = subparsers.add_parser("preview", help="Preview redactions")
    preview_parser.add_argument("file", help="Document to analyze")
    preview_parser.add_argument("--no-llm", dest="use_llm", action="store_false")
    preview_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Redact multiple documents")
    batch_parser.add_argument("directory", help="Directory containing documents")
    batch_parser.add_argument("--output-dir", "-o", help="Output directory")
    batch_parser.add_argument("--no-llm", dest="use_llm", action="store_false")

    args = parser.parse_args()

    if args.command == "redact":
        cmd_redact(args)
    elif args.command == "preview":
        cmd_preview(args)
    elif args.command == "batch":
        cmd_batch(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
