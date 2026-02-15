"""
Community CLI - Command-line interface for community detection and summarization

Usage:
    python -m tools.cohesionn.community_cli detect --level medium
    python -m tools.cohesionn.community_cli summarize
    python -m tools.cohesionn.community_cli stats
"""

import argparse
import logging
import sys

from .community_detection import CommunityDetector
from .community_summarizer import CommunitySummarizer
from .global_retriever import get_community_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def detect_communities(level: str = "medium", topic: str = None) -> None:
    """Detect communities in the knowledge graph."""
    logger.info(f"Detecting communities at level '{level}'")

    detector = CommunityDetector()
    result = detector.detect(topic=topic, level=level)

    print("\nCommunity Detection Results:")
    print(f"  Level: {result.level}")
    print(f"  Total nodes: {result.total_nodes}")
    print(f"  Communities found: {len(result.communities)}")

    if result.communities:
        print("\n  Top 10 communities by size:")
        sorted_communities = sorted(result.communities, key=lambda c: c.node_count, reverse=True)
        for i, community in enumerate(sorted_communities[:10], 1):
            entities_preview = ", ".join(community.entity_ids[:5])
            if len(community.entity_ids) > 5:
                entities_preview += f"... (+{len(community.entity_ids) - 5} more)"
            print(f"    {i}. {community.community_id}: {community.node_count} chunks, {community.entity_count} entities")
            print(f"       Entities: {entities_preview}")


def summarize_communities(level: str = "medium", topic: str = None) -> None:
    """Detect communities and generate summaries."""
    logger.info(f"Building community summaries at level '{level}'")

    # Detect
    detector = CommunityDetector()
    detection_result = detector.detect(topic=topic, level=level)

    if not detection_result.communities:
        print("No communities found to summarize.")
        return

    print(f"Found {len(detection_result.communities)} communities")

    # Summarize
    summarizer = CommunitySummarizer()
    summaries = summarizer.summarize_batch(detection_result.communities)

    print(f"Generated {len(summaries)} summaries")

    # Store
    store = get_community_store()
    count = store.store_summaries(summaries, topic=topic)

    print(f"Stored {count} community summaries")

    # Preview
    print("\nSample summaries:")
    for summary in summaries[:3]:
        print(f"\n  Community: {summary.community_id}")
        print(f"  Themes: {', '.join(summary.themes[:3])}")
        print(f"  Summary: {summary.summary[:200]}...")


def show_stats() -> None:
    """Show community summary statistics."""
    store = get_community_store()
    stats = store.get_stats()

    print("\nCommunity Summary Statistics:")
    print(f"  Collection: {stats['collection']}")
    print(f"  Total summaries: {stats['points_count']}")


def test_classifier() -> None:
    """Test the query classifier with sample queries."""
    from .query_classifier import get_query_classifier

    classifier = get_query_classifier()

    test_queries = [
        "What are the main types of analysis methods?",
        "Overview of testing procedures",
        "What is the recommended safety factor?",
        "Calculate the expected output for this scenario",
        "Compare method A and method B",
        "ASTM standard requirements for testing",
        "Factors affecting project outcomes",
        "How much variation is acceptable?",
    ]

    print("\nQuery Classification Test:")
    print("-" * 60)

    for query in test_queries:
        strategy = classifier.get_search_strategy(query)
        print(f"\nQuery: {query}")
        print(f"  Type: {strategy['query_type']}")
        print(f"  Confidence: {strategy['confidence']:.2f}")
        print(f"  Use community: {strategy['use_community_search']}")
        print(f"  Use chunks: {strategy['use_chunk_search']}")


def main():
    parser = argparse.ArgumentParser(
        description="Community detection and global search CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.cohesionn.community_cli detect --level medium
  python -m tools.cohesionn.community_cli summarize --level medium
  python -m tools.cohesionn.community_cli stats
  python -m tools.cohesionn.community_cli test-classifier
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect communities")
    detect_parser.add_argument(
        "--level",
        type=str,
        default="medium",
        choices=["coarse", "medium", "fine"],
        help="Resolution level",
    )
    detect_parser.add_argument("--topic", type=str, help="Optional topic filter")

    # Summarize command
    summarize_parser = subparsers.add_parser("summarize", help="Detect and summarize communities")
    summarize_parser.add_argument(
        "--level",
        type=str,
        default="medium",
        choices=["coarse", "medium", "fine"],
        help="Resolution level",
    )
    summarize_parser.add_argument("--topic", type=str, help="Optional topic filter")

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    # Test classifier command
    subparsers.add_parser("test-classifier", help="Test query classifier")

    args = parser.parse_args()

    if args.command == "detect":
        detect_communities(level=args.level, topic=args.topic)

    elif args.command == "summarize":
        summarize_communities(level=args.level, topic=args.topic)

    elif args.command == "stats":
        show_stats()

    elif args.command == "test-classifier":
        test_classifier()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
