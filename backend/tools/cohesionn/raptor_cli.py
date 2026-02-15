"""
RAPTOR CLI - Command-line interface for RAPTOR tree building

Usage:
    python -m tools.cohesionn.raptor_cli build --topic my_topic
    python -m tools.cohesionn.raptor_cli build --all
    python -m tools.cohesionn.raptor_cli stats
    python -m tools.cohesionn.raptor_cli delete --topic my_topic
"""

import argparse
import logging
import sys
from typing import List

from .raptor import RaptorTreeBuilder
from .vectorstore import get_knowledge_base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_tree(topic: str, rebuild: bool = False, max_levels: int = 3) -> None:
    """Build RAPTOR tree for a topic."""
    logger.info(f"Building RAPTOR tree for topic: {topic}")

    builder = RaptorTreeBuilder(max_levels=max_levels)
    result = builder.build_tree(topic=topic, rebuild=rebuild)

    print(f"\nRAPTOR Tree Build Results for '{topic}':")
    print(f"  Levels built: {result.levels_built}")
    print(f"  Nodes per level: {result.nodes_per_level}")
    print(f"  Total nodes: {result.total_nodes}")
    print(f"  Build time: {result.build_time_seconds:.1f}s")

    if result.errors:
        print(f"  Errors: {len(result.errors)}")
        for error in result.errors[:5]:
            print(f"    - {error}")


def build_all(rebuild: bool = False, max_levels: int = 3) -> None:
    """Build RAPTOR trees for all topics."""
    kb = get_knowledge_base()
    topics = kb.discover_topics()

    logger.info(f"Building RAPTOR trees for {len(topics)} topics: {topics}")

    for topic in topics:
        build_tree(topic, rebuild=rebuild, max_levels=max_levels)
        print()


def show_stats(topics: List[str] = None) -> None:
    """Show RAPTOR tree statistics."""
    kb = get_knowledge_base()

    if topics is None:
        topics = kb.discover_topics()

    builder = RaptorTreeBuilder()

    print("\nRAPTOR Tree Statistics:")
    print("-" * 50)

    for topic in topics:
        stats = builder.get_tree_stats(topic)
        print(f"\nTopic: {topic}")
        print(f"  Total RAPTOR nodes: {stats['total_nodes']}")
        for level, count in sorted(stats["levels"].items()):
            level_name = {1: "Cluster", 2: "Section", 3: "Topic"}.get(level, f"L{level}")
            print(f"  Level {level} ({level_name}): {count}")


def delete_tree(topic: str) -> None:
    """Delete RAPTOR tree for a topic."""
    logger.info(f"Deleting RAPTOR tree for topic: {topic}")

    builder = RaptorTreeBuilder()
    count = builder.delete_tree(topic)

    print(f"Deleted {count} RAPTOR nodes for '{topic}'")


def main():
    parser = argparse.ArgumentParser(
        description="RAPTOR tree management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.cohesionn.raptor_cli build --topic my_topic
  python -m tools.cohesionn.raptor_cli build --all --rebuild
  python -m tools.cohesionn.raptor_cli stats
  python -m tools.cohesionn.raptor_cli delete --topic my_topic
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build RAPTOR tree")
    build_parser.add_argument(
        "--topic",
        type=str,
        help="Topic to build tree for",
    )
    build_parser.add_argument(
        "--all",
        action="store_true",
        help="Build trees for all topics",
    )
    build_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild even if tree exists",
    )
    build_parser.add_argument(
        "--max-levels",
        type=int,
        default=3,
        help="Maximum tree depth (default: 3)",
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show RAPTOR tree statistics")
    stats_parser.add_argument(
        "--topic",
        type=str,
        nargs="*",
        help="Topics to show stats for (default: all)",
    )

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete RAPTOR tree")
    delete_parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Topic to delete tree for",
    )

    args = parser.parse_args()

    if args.command == "build":
        if args.all:
            build_all(rebuild=args.rebuild, max_levels=args.max_levels)
        elif args.topic:
            build_tree(args.topic, rebuild=args.rebuild, max_levels=args.max_levels)
        else:
            print("Error: Specify --topic or --all")
            sys.exit(1)

    elif args.command == "stats":
        show_stats(topics=args.topic)

    elif args.command == "delete":
        delete_tree(args.topic)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
