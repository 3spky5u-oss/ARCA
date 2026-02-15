"""Allow running as: python -m tools.readd"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Readd - Intelligent Document Extraction")
    parser.add_argument("file", help="PDF file to extract")
    parser.add_argument("--no-escalate", action="store_true", help="Disable auto-escalation")
    parser.add_argument(
        "--extractor", choices=["pymupdf_text", "pymupdf4llm", "marker", "vision_ocr"], help="Force specific extractor"
    )
    parser.add_argument("--output", "-o", help="Output file for extracted text")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    from .pipeline import ReaddPipeline, ReaddQuickExtract

    file_path = Path(args.file)

    if args.extractor:
        # Direct extraction with specified tool
        print(f"Extracting {file_path.name} with {args.extractor}...")
        result = ReaddQuickExtract.extract(file_path, args.extractor)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result.full_text)
            print(f"Saved to {args.output}")
        else:
            print(result.full_text[:2000])
            if len(result.full_text) > 2000:
                print(f"\n... ({len(result.full_text)} chars total)")
    else:
        # Full pipeline
        pipeline = ReaddPipeline(auto_escalate=not args.no_escalate)
        result = pipeline.process(file_path)

        if args.json:
            import json

            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"\nResults for {file_path.name}:")
            print(f"  Success: {result.success}")
            print(f"  Quality score: {result.scout_report.quality_score}")
            print(f"  Quality tier: {result.scout_report.quality_tier.value}")
            print(f"  Extractors tried: {result.extractors_tried}")
            print(f"  Final extractor: {result.final_extractor}")
            print(f"  Pages: {result.page_count}")
            print(f"  Text length: {len(result.text) if result.text else 0}")

            if result.warnings:
                print("\nWarnings:")
                for w in result.warnings:
                    print(f"  - {w}")

            if args.output and result.text:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(result.text)
                print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
