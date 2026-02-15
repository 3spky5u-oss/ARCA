#!/usr/bin/env python3
"""
Quick Ingest Script - Run from backend directory

Usage:
    python ingest_knowledge.py --setup            # Create directories first
    python ingest_knowledge.py                    # Ingest all topics (auto-detects new files)
    python ingest_knowledge.py --all              # Force re-ingest all files
    python ingest_knowledge.py my_topic             # Ingest one topic
    python ingest_knowledge.py --file book.pdf my_topic

    # Extraction modes:
    python ingest_knowledge.py --mode knowledge_base  # Slow, high-quality with vision (default for --all)
    python ingest_knowledge.py --mode session         # Fast text-only extraction
    python ingest_knowledge.py --fast                 # Alias for --mode session
"""

import sys
from pathlib import Path

# Setup paths
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

KNOWLEDGE_DIR = BACKEND_DIR / "data" / "technical_knowledge"
DB_DIR = BACKEND_DIR / "data" / "cohesionn_db"
def _get_default_topics():
    """Get default topics from domain lexicon."""
    try:
        from domain_loader import get_lexicon
        lexicon = get_lexicon()
        return lexicon.get("topics", ["general"])
    except Exception:
        return ["general"]

DEFAULT_TOPICS = _get_default_topics()


def discover_topics():
    """Discover topics from filesystem"""
    if not KNOWLEDGE_DIR.exists():
        return DEFAULT_TOPICS
    topics = [d.name for d in KNOWLEDGE_DIR.iterdir() if d.is_dir()]
    return topics if topics else DEFAULT_TOPICS


def setup():
    """Create directory structure"""
    print("Creating directories...\n")

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    for topic in DEFAULT_TOPICS:
        d = KNOWLEDGE_DIR / topic
        d.mkdir(exist_ok=True)
        print(f"  {d}")

    print("\nDone! Now add PDFs to the topic folders and run:")
    print("  python ingest_knowledge.py")


def _prepare_vram_for_vision():
    """Ensure the vision model slot is running for ingestion."""
    try:
        from utils.llm import get_server_manager
        manager = get_server_manager()

        # Check if vision slot is already running
        running = manager.list_running()
        if "vision" in running and running["vision"].get("alive"):
            print(f"  Vision slot already running: {running['vision'].get('model')}")
            return

        # Start vision slot
        print("  Starting vision model slot...")
        started = manager.start("vision")
        if started:
            print("  Vision model slot started")
        else:
            print("  Warning: Failed to start vision slot")
    except Exception as e:
        print(f"  Warning: Vision prep failed: {e}")


def main():
    args = sys.argv[1:]

    # Setup command
    if "--setup" in args:
        setup()
        return

    # Check dirs exist
    if not KNOWLEDGE_DIR.exists():
        print(f"Directory not found: {KNOWLEDGE_DIR}\n")
        print("Run setup first:")
        print("  python ingest_knowledge.py --setup")
        return

    # Parse args
    single_file = None
    topic = None
    force_all = "--all" in args
    use_auto = not force_all and "--file" not in args and len([a for a in args if not a.startswith("-")]) == 0

    # Parse extraction mode
    ingest_mode = "session"  # Default to fast mode
    if "--mode" in args:
        idx = args.index("--mode")
        if idx + 1 < len(args):
            ingest_mode = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]
    elif "--fast" in args:
        ingest_mode = "session"
        args = [a for a in args if a != "--fast"]
    elif force_all:
        # Default to knowledge_base mode for --all (re-ingestion)
        ingest_mode = "knowledge_base"

    if "--file" in args:
        idx = args.index("--file")
        single_file = args[idx + 1]
        args = args[:idx] + args[idx + 2 :]

    # Discover topics from filesystem
    topics = discover_topics()

    # Check if topic argument matches discovered topics
    remaining_args = [a for a in args if not a.startswith("-")]
    if remaining_args and remaining_args[0] in topics:
        topic = remaining_args[0]

    # Import
    try:
        from tools.cohesionn.ingest import DocumentIngester, IngestMode
        from tools.cohesionn.vectorstore import KnowledgeBase
        from tools.cohesionn.autoingest import AutoIngestService
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're in the backend directory and tools are installed.")
        return

    # Validate mode
    if ingest_mode not in (IngestMode.KNOWLEDGE_BASE, IngestMode.SESSION):
        print(f"Invalid mode: {ingest_mode}")
        print(f"  Valid modes: {IngestMode.KNOWLEDGE_BASE}, {IngestMode.SESSION}")
        return

    print(f"Knowledge: {KNOWLEDGE_DIR}")
    print(f"Database:  {DB_DIR}")
    print(f"Topics:    {', '.join(topics)}")
    print(f"Mode:      {ingest_mode}")
    if ingest_mode == IngestMode.KNOWLEDGE_BASE:
        print("           (Hybrid extraction with per-page vision routing)")
    print()

    DB_DIR.mkdir(parents=True, exist_ok=True)

    # For knowledge_base mode: unload competing models and warm up vision model
    if ingest_mode == IngestMode.KNOWLEDGE_BASE:
        _prepare_vram_for_vision()

    # Use auto-ingestion for default case (smart detection of new files)
    if use_auto:
        print("Running auto-ingestion (detects new/changed files)...\n")
        service = AutoIngestService(KNOWLEDGE_DIR, DB_DIR, mode=ingest_mode)
        result = service.run()

        if result["new_files"] == 0:
            print("Knowledge base up to date - no new files to ingest")
        else:
            print(f"Processed {result['new_files']} files:")
            print(f"  Successful: {result.get('successful', 0)}")
            print(f"  Failed: {result.get('failed', 0)}")

        print("\nManifest stats:")
        stats = service.manifest.get_stats()
        print(f"  Total files tracked: {stats['total_files']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        for t, s in stats.get("by_topic", {}).items():
            print(f"    {t}: {s['files']} files, {s['chunks']} chunks")
        return

    # Manual ingestion modes below
    kb = KnowledgeBase(DB_DIR, KNOWLEDGE_DIR)
    ingester = DocumentIngester(knowledge_base=kb, use_readd=True, mode=ingest_mode)

    if single_file:
        if not topic:
            print("Error: --file requires a topic")
            print("  python ingest_knowledge.py --file book.pdf my_topic")
            return

        fpath = Path(single_file)
        if not fpath.is_absolute():
            fpath = KNOWLEDGE_DIR / topic / fpath

        if not fpath.exists():
            print(f"File not found: {fpath}")
            return

        print(f"Ingesting: {fpath.name} -> {topic}")
        r = ingester.ingest_file(fpath, topic)
        print(f"  {'OK' if r.success else 'FAILED'}: {r.chunks_created} chunks")
        if r.extractor_used:
            print(f"  Extractor: {r.extractor_used}")
        if r.pages_by_extractor:
            print(f"  Pages by extractor: {r.pages_by_extractor}")
        if r.error:
            print(f"  Error: {r.error}")

    elif topic:
        topic_dir = KNOWLEDGE_DIR / topic
        if not topic_dir.exists():
            print(f"Not found: {topic_dir}")
            return

        files = list(topic_dir.glob("*.pdf")) + list(topic_dir.glob("*.txt"))
        if not files:
            print(f"No files in {topic_dir}")
            return

        print(f"Ingesting {topic}: {len(files)} files")
        total_text_pages = 0
        total_vision_pages = 0
        for r in ingester.ingest_directory(topic_dir, topic):
            status = "OK" if r.success else "FAIL"
            extractor_info = ""
            if r.pages_by_extractor:
                text_p = r.pages_by_extractor.get("pymupdf4llm", 0)
                vision_p = r.pages_by_extractor.get("observationn", 0)
                total_text_pages += text_p
                total_vision_pages += vision_p
                if vision_p > 0:
                    extractor_info = f" (text:{text_p}, vision:{vision_p})"
            print(f"  [{status}] {Path(r.file_path).name}: {r.chunks_created} chunks{extractor_info}")

        if ingest_mode == IngestMode.KNOWLEDGE_BASE:
            print(f"\n  Total: {total_text_pages} text pages, {total_vision_pages} vision pages")

    else:
        # Force re-ingest all (--all flag)
        print("Force re-ingesting all topics...\n")

        total_text_pages = 0
        total_vision_pages = 0

        for t in topics:
            tdir = KNOWLEDGE_DIR / t
            if not tdir.exists():
                tdir.mkdir(exist_ok=True)
                print(f"  {t}: created (empty)")
                continue

            files = list(tdir.glob("*.pdf")) + list(tdir.glob("*.txt"))
            if not files:
                print(f"  {t}: no files")
                continue

            print(f"  {t}: {len(files)} files")
            results = list(ingester.ingest_directory(tdir, t))
            ok = sum(1 for r in results if r.success)
            chunks = sum(r.chunks_created for r in results)

            # Aggregate extraction stats
            topic_text = sum(r.pages_by_extractor.get("pymupdf4llm", 0) for r in results)
            topic_vision = sum(r.pages_by_extractor.get("observationn", 0) for r in results)
            total_text_pages += topic_text
            total_vision_pages += topic_vision

            extractor_stats = ""
            if ingest_mode == IngestMode.KNOWLEDGE_BASE and (topic_text or topic_vision):
                extractor_stats = f", text:{topic_text} vision:{topic_vision} pages"

            print(f"    -> {ok}/{len(results)} succeeded, {chunks} chunks{extractor_stats}")

    if ingest_mode == IngestMode.KNOWLEDGE_BASE:
        print("\nExtraction summary:")
        print(f"  Text pages (PyMuPDF4LLM): {total_text_pages}")
        print(f"  Vision pages (Observationn): {total_vision_pages}")

    print("\nStats:")
    for t, n in kb.get_stats().items():
        print(f"  {t}: {n:,} chunks")


if __name__ == "__main__":
    main()
