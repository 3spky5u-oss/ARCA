#!/usr/bin/env python3
"""
Cohesionn CLI - Knowledge base management

Usage:
    python -m tools.cohesionn.cli ingest --topic my_topic --source /path/to/docs
    python -m tools.cohesionn.cli ingest --all --source /app/technical_knowledge
    python -m tools.cohesionn.cli stats
    python -m tools.cohesionn.cli search "what methodology was used"
    python -m tools.cohesionn.cli clear --topic my_topic
"""

import argparse
import logging
from pathlib import Path
import sys
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def cmd_ingest(args):
    """Ingest documents into knowledge base"""
    from .ingest import DocumentIngester
    from .vectorstore import get_knowledge_base

    source = Path(args.source)
    kb = get_knowledge_base(Path(args.db_path))
    ingester = DocumentIngester(
        knowledge_base=kb,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_readd=not args.no_readd,
    )

    if args.all:
        # Ingest all topics
        print(f"Ingesting all topics from {source}")
        results = ingester.ingest_all_topics(source)

        for topic, topic_results in results.items():
            success = sum(1 for r in topic_results if r.success)
            failed = len(topic_results) - success
            chunks = sum(r.chunks_created for r in topic_results)
            print(f"  {topic}: {success} files, {chunks} chunks ({failed} failed)")

    else:
        # Ingest specific topic
        if not args.topic:
            print("Error: --topic required when not using --all")
            sys.exit(1)

        if source.is_file():
            print(f"Ingesting {source.name} into {args.topic}")
            result = ingester.ingest_file(source, args.topic)

            if result.success:
                print(f"  Success: {result.chunks_created} chunks")
            else:
                print(f"  Failed: {result.error}")
        else:
            print(f"Ingesting {args.topic} from {source}")
            results = list(ingester.ingest_directory(source, args.topic))

            success = sum(1 for r in results if r.success)
            failed = len(results) - success
            chunks = sum(r.chunks_created for r in results)

            print(f"  Processed: {len(results)} files")
            print(f"  Success: {success}, Failed: {failed}")
            print(f"  Total chunks: {chunks}")

    # Print final stats
    print("\nKnowledge Base Stats:")
    for topic, count in kb.get_stats().items():
        print(f"  {topic}: {count:,} chunks")


def cmd_stats(args):
    """Show knowledge base statistics"""
    from .vectorstore import get_knowledge_base

    kb = get_knowledge_base(Path(args.db_path))
    stats = kb.get_stats()

    print("Cohesionn Knowledge Base Statistics")
    print("=" * 40)

    total = 0
    for topic, count in stats.items():
        print(f"  {topic:20s}: {count:,} chunks")
        total += count

    print("-" * 40)
    print(f"  {'TOTAL':20s}: {total:,} chunks")


def cmd_search(args):
    """Test search functionality"""
    from .retriever import search_knowledge

    result = search_knowledge(
        query=args.query,
        topics=args.topics.split(",") if args.topics else None,
        top_k=args.top_k,
        db_path=Path(args.db_path),
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"\nQuery: {result['query']}")
    print(f"Topics: {result.get('topics_searched', [])}")
    print(f"Results: {result.get('num_results', 0)}")
    print("\n" + "=" * 60)

    if result.get("success"):
        for i, chunk in enumerate(result.get("chunks", []), 1):
            score = chunk.get("score", 0)
            source = chunk.get("source", "Unknown")
            page = chunk.get("page", "?")

            print(f"\n[{i}] {source} (p.{page}) - Score: {score:.3f}")
            print("-" * 40)
            print(chunk.get("content", "")[:400])

        refs = result.get("references", "")
        if refs:
            print("\n" + "=" * 60)
            print(refs)
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


def cmd_clear(args):
    """Clear knowledge base"""
    from .vectorstore import get_knowledge_base

    kb = get_knowledge_base(Path(args.db_path))

    if args.all:
        confirm = input("Clear ALL topics? This cannot be undone. (yes/no): ")
        if confirm.lower() == "yes":
            kb.clear_all()
            print("All topics cleared")
    else:
        if not args.topic:
            print("Error: --topic required when not using --all")
            sys.exit(1)

        confirm = input(f"Clear {args.topic}? This cannot be undone. (yes/no): ")
        if confirm.lower() == "yes":
            kb.clear_topic(args.topic)
            print(f"Cleared {args.topic}")


def cmd_test_readd(args):
    """Test Readd extraction on a file"""
    from tools.readd import process_document

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    print(f"Processing {file_path.name} with Readd...")
    result = process_document(file_path)

    print("\nResults:")
    print(f"  Success: {result.success}")
    print(f"  Quality score: {result.scout_report.quality_score}")
    print(f"  Quality tier: {result.scout_report.quality_tier.value}")
    print(f"  Extractors tried: {result.extractors_tried}")
    print(f"  Final extractor: {result.final_extractor}")
    print(f"  Escalations: {result.escalation_count}")
    print(f"  Pages: {result.page_count}")
    print(f"  Text length: {len(result.text) if result.text else 0}")

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    if result.qa_report:
        print("\nQA Report:")
        print(f"  Weirdness score: {result.qa_report.weirdness_score}")
        print(f"  Issues: {len(result.qa_report.issues)}")

    if args.preview and result.text:
        print("\nText preview:")
        print("-" * 40)
        print(result.text[:1000])


def main():
    parser = argparse.ArgumentParser(description="Cohesionn - Technical Knowledge RAG Pipeline")
    parser.add_argument("--db-path", default="/app/cohesionn_db", help="Path to knowledge base storage")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--source", required=True, help="Source file or directory")
    ingest_parser.add_argument("--topic", help="Topic name (e.g. general, my_topic)")
    ingest_parser.add_argument("--all", action="store_true", help="Ingest all topics")
    ingest_parser.add_argument("--chunk-size", type=int, default=800)
    ingest_parser.add_argument("--chunk-overlap", type=int, default=150)
    ingest_parser.add_argument("--no-readd", action="store_true", help="Skip Readd, use basic extraction")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")

    # Search command
    search_parser = subparsers.add_parser("search", help="Test search")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--topics", help="Comma-separated topics")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear knowledge base")
    clear_parser.add_argument("--topic", help="Topic to clear")
    clear_parser.add_argument("--all", action="store_true", help="Clear all topics")

    # Test Readd command
    readd_parser = subparsers.add_parser("test-readd", help="Test Readd extraction")
    readd_parser.add_argument("file", help="File to test")
    readd_parser.add_argument("--preview", action="store_true", help="Show text preview")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "clear":
        cmd_clear(args)
    elif args.command == "test-readd":
        cmd_test_readd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
