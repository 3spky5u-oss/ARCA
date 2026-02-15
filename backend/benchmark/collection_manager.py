"""
Benchmark Collection Manager — Qdrant topic-per-config lifecycle.

Uses topic filtering (not separate collections) for near-free isolation.
Each chunking config gets topic "bench_{config_id}".
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class BenchmarkCollectionManager:
    """Manages benchmark topics in the shared Qdrant collection."""

    TOPIC_PREFIX = "bench_"

    def __init__(self):
        from tools.cohesionn.vectorstore import get_knowledge_base
        self.kb = get_knowledge_base()

    def get_topic(self, config_id: str) -> str:
        """Get the topic name for a config."""
        return f"{self.TOPIC_PREFIX}{config_id}"

    def ingest_corpus(
        self,
        config_id: str,
        md_files: List[str],
        chunk_size: int,
        chunk_overlap: int,
        context_prefix: bool,
    ) -> int:
        """Ingest markdown corpus files into a benchmark topic.

        Returns number of chunks created.
        """
        from pathlib import Path
        from tools.cohesionn.chunker import SemanticChunker

        topic = self.get_topic(config_id)
        store = self.kb.get_store(topic)

        chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            context_prefix_enabled=context_prefix,
        )

        total_chunks = 0
        for md_path in md_files:
            path = Path(md_path)
            if not path.exists():
                logger.warning(f"File not found: {md_path}")
                continue

            text = path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue

            metadata = {
                "source": str(path),
                "file_name": path.name,
                "title": path.stem,
                "topic": topic,
            }

            chunks = chunker.chunk_text(text, metadata)
            if chunks:
                successful, failed = store.add_chunks(chunks)
                total_chunks += successful
                if failed:
                    logger.warning(f"{path.name}: {failed} chunks failed to embed")

        logger.info(f"Config {config_id}: ingested {total_chunks} chunks into topic {topic}")
        return total_chunks

    def cleanup_topic(self, config_id: str):
        """Delete all points for a benchmark config topic."""
        topic = self.get_topic(config_id)
        try:
            store = self.kb.get_store(topic)
            store.clear()
            logger.info(f"Cleaned up topic: {topic}")
        except Exception as e:
            logger.warning(f"Cleanup failed for {topic}: {e}")

    def cleanup_all_benchmark_topics(self):
        """Delete all benchmark topics (bench_* prefix)."""
        for topic_name in list(self.kb.stores.keys()):
            if topic_name.startswith(self.TOPIC_PREFIX):
                try:
                    self.kb.stores[topic_name].clear()
                    logger.info(f"Cleaned up benchmark topic: {topic_name}")
                except Exception as e:
                    logger.warning(f"Cleanup failed for {topic_name}: {e}")

    def get_chunk_count(self, config_id: str) -> int:
        """Get chunk count for a benchmark config."""
        topic = self.get_topic(config_id)
        if topic in self.kb.stores:
            return self.kb.stores[topic].count
        return 0
