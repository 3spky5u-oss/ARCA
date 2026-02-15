"""
RAPTOR Tree Builder - Orchestrates hierarchical summary construction

Builds the RAPTOR tree structure:
- Level 0: Existing leaf chunks
- Level 1: Cluster summaries
- Level 2: Section summaries
- Level 3: Topic summaries
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time

from .clusterer import RaptorClusterer
from .summarizer import RaptorSummarizer

logger = logging.getLogger(__name__)


@dataclass
class RaptorNode:
    """A node in the RAPTOR tree"""

    node_id: str  # Unique identifier
    content: str  # Summary text
    level: int  # 0=leaf, 1=cluster, 2=section, 3=topic
    embedding: Optional[List[float]] = None
    children: List[str] = field(default_factory=list)  # Child node IDs
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeBuildResult:
    """Result from building RAPTOR tree"""

    topic: str
    levels_built: int
    nodes_per_level: Dict[int, int]
    total_nodes: int
    build_time_seconds: float
    errors: List[str] = field(default_factory=list)


class RaptorTreeBuilder:
    """
    Orchestrates RAPTOR tree construction.

    Process:
    1. Fetch leaf chunks from Qdrant (level 0)
    2. Cluster chunks using UMAP + GMM
    3. Generate cluster summaries (level 1)
    4. Recursively cluster summaries to build higher levels
    5. Store summary nodes back to Qdrant with raptor_level metadata
    """

    def __init__(
        self,
        clusterer: Optional[RaptorClusterer] = None,
        summarizer: Optional[RaptorSummarizer] = None,
        max_levels: int = 3,
        cluster_size: int = 10,
        min_items_for_level: int = 5,
    ):
        """
        Args:
            clusterer: RaptorClusterer instance
            summarizer: RaptorSummarizer instance
            max_levels: Maximum tree depth (3 recommended)
            cluster_size: Target items per cluster
            min_items_for_level: Minimum items to create next level
        """
        self.clusterer = clusterer or RaptorClusterer(target_cluster_size=cluster_size)
        self.summarizer = summarizer or RaptorSummarizer()
        self.max_levels = max_levels
        self.cluster_size = cluster_size
        self.min_items_for_level = min_items_for_level

    def build_tree(
        self,
        topic: str,
        rebuild: bool = False,
    ) -> TreeBuildResult:
        """
        Build RAPTOR tree for a topic.

        Args:
            topic: Knowledge base topic (e.g., "general")
            rebuild: If True, rebuild even if tree exists

        Returns:
            TreeBuildResult with statistics
        """
        start_time = time.time()
        errors = []
        nodes_per_level = {0: 0}

        logger.info(f"Building RAPTOR tree for topic: {topic}")

        try:
            # Get vectorstore and embedder
            from ..vectorstore import get_knowledge_base
            from ..embeddings import get_embedder

            kb = get_knowledge_base()
            embedder = get_embedder()

            # Check if tree exists and skip if not rebuilding
            if not rebuild:
                existing_count = self._count_raptor_nodes(kb, topic)
                if existing_count > 0:
                    logger.info(f"RAPTOR tree exists for {topic} ({existing_count} nodes), skipping")
                    return TreeBuildResult(
                        topic=topic,
                        levels_built=0,
                        nodes_per_level={},
                        total_nodes=existing_count,
                        build_time_seconds=0,
                    )

            # Step 1: Fetch leaf chunks (level 0)
            leaf_chunks = self._fetch_leaf_chunks(kb, topic)
            nodes_per_level[0] = len(leaf_chunks)

            if len(leaf_chunks) < self.min_items_for_level:
                logger.warning(f"Not enough leaf chunks ({len(leaf_chunks)}) to build RAPTOR tree")
                return TreeBuildResult(
                    topic=topic,
                    levels_built=0,
                    nodes_per_level=nodes_per_level,
                    total_nodes=len(leaf_chunks),
                    build_time_seconds=time.time() - start_time,
                    errors=["Insufficient leaf chunks"],
                )

            logger.info(f"Found {len(leaf_chunks)} leaf chunks")

            # Step 2: Build hierarchy level by level
            current_items = leaf_chunks
            current_level = 0

            while current_level < self.max_levels and len(current_items) >= self.min_items_for_level:
                current_level += 1
                logger.info(f"Building level {current_level} from {len(current_items)} items")

                # Cluster items
                embeddings = [item["embedding"] for item in current_items]
                cluster_result = self.clusterer.cluster(embeddings)

                if cluster_result.n_clusters == 0:
                    logger.warning(f"No clusters formed at level {current_level}")
                    break

                # Group items by cluster
                clusters = self._group_by_cluster(current_items, cluster_result.cluster_assignments)

                # Generate summaries for each cluster
                summary_nodes = []
                for cluster_id, cluster_items in clusters.items():
                    try:
                        node = self._create_summary_node(
                            cluster_items=cluster_items,
                            level=current_level,
                            topic=topic,
                            cluster_id=cluster_id,
                            embedder=embedder,
                        )
                        summary_nodes.append(node)
                    except Exception as e:
                        errors.append(f"Level {current_level} cluster {cluster_id}: {e}")
                        logger.error(f"Failed to create summary for cluster {cluster_id}: {e}")

                nodes_per_level[current_level] = len(summary_nodes)
                logger.info(f"Level {current_level}: created {len(summary_nodes)} summary nodes")

                # Store summary nodes in Qdrant
                self._store_raptor_nodes(kb, summary_nodes, topic)

                # Prepare for next level
                current_items = [
                    {
                        "content": node.content,
                        "embedding": node.embedding,
                        "node_id": node.node_id,
                    }
                    for node in summary_nodes
                ]

            build_time = time.time() - start_time
            total_nodes = sum(nodes_per_level.values())

            logger.info(
                f"RAPTOR tree built: {current_level} levels, {total_nodes} total nodes, "
                f"{build_time:.1f}s"
            )

            return TreeBuildResult(
                topic=topic,
                levels_built=current_level,
                nodes_per_level=nodes_per_level,
                total_nodes=total_nodes,
                build_time_seconds=build_time,
                errors=errors,
            )

        except Exception as e:
            logger.error(f"RAPTOR tree build failed: {e}")
            return TreeBuildResult(
                topic=topic,
                levels_built=0,
                nodes_per_level=nodes_per_level,
                total_nodes=0,
                build_time_seconds=time.time() - start_time,
                errors=[str(e)],
            )

    def _fetch_leaf_chunks(self, kb: Any, topic: str) -> List[Dict[str, Any]]:
        """Fetch all leaf chunks (level 0) for a topic from Qdrant."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        store = kb.get_store(topic)
        client = store.client
        collection_name = "cohesionn"

        # Scroll through all chunks for this topic that are level 0 (or no level)
        all_chunks = []
        offset = None

        while True:
            response = client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="topic", match=MatchValue(value=topic)),
                    ],
                    must_not=[
                        # Exclude existing RAPTOR nodes (level > 0)
                        FieldCondition(key="raptor_level", match=MatchValue(value=1)),
                        FieldCondition(key="raptor_level", match=MatchValue(value=2)),
                        FieldCondition(key="raptor_level", match=MatchValue(value=3)),
                    ],
                ),
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            points, next_offset = response

            for point in points:
                all_chunks.append({
                    "id": point.payload.get("chunk_id", str(point.id)),
                    "content": point.payload.get("content", ""),
                    "embedding": point.vector if isinstance(point.vector, list) else list(point.vector),
                    "metadata": {k: v for k, v in point.payload.items() if k not in ("content", "chunk_id")},
                })

            if next_offset is None:
                break
            offset = next_offset

        return all_chunks

    def _count_raptor_nodes(self, kb: Any, topic: str) -> int:
        """Count existing RAPTOR summary nodes for a topic."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        store = kb.get_store(topic)
        client = store.client

        try:
            result = client.count(
                collection_name="cohesionn",
                count_filter=Filter(
                    must=[
                        FieldCondition(key="topic", match=MatchValue(value=topic)),
                        FieldCondition(key="raptor_level", match=MatchAny(any=[1, 2, 3])),
                    ],
                ),
            )
            return result.count
        except Exception:
            return 0

    def _group_by_cluster(
        self,
        items: List[Dict[str, Any]],
        assignments: List[int],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Group items by cluster assignment."""
        clusters: Dict[int, List[Dict[str, Any]]] = {}

        for item, cluster_id in zip(items, assignments):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(item)

        return clusters

    def _create_summary_node(
        self,
        cluster_items: List[Dict[str, Any]],
        level: int,
        topic: str,
        cluster_id: int,
        embedder: Any,
    ) -> RaptorNode:
        """Create a summary node for a cluster."""
        # Extract content from items
        contents = [item["content"] for item in cluster_items]

        # Generate summary
        summary_result = self.summarizer.summarize(contents, level=level)

        # Generate embedding for summary
        embedding = embedder.embed_document(summary_result.summary)

        # Create unique node ID
        content_hash = hashlib.md5(summary_result.summary.encode()).hexdigest()[:12]
        node_id = f"raptor_{topic}_L{level}_{cluster_id}_{content_hash}"

        # Collect child IDs
        child_ids = [item.get("node_id") or item.get("id", "") for item in cluster_items]

        return RaptorNode(
            node_id=node_id,
            content=summary_result.summary,
            level=level,
            embedding=embedding,
            children=child_ids,
            metadata={
                "topic": topic,
                "raptor_level": level,
                "source_count": len(cluster_items),
                "model_used": summary_result.model_used,
            },
        )

    def _store_raptor_nodes(self, kb: Any, nodes: List[RaptorNode], topic: str) -> int:
        """Store RAPTOR nodes in Qdrant."""
        from qdrant_client.models import PointStruct

        if not nodes:
            return 0

        store = kb.get_store(topic)
        client = store.client

        # Convert nodes to Qdrant points
        points = []
        for node in nodes:
            # Generate integer ID from node_id hash
            point_id = int(hashlib.md5(node.node_id.encode()).hexdigest()[:16], 16)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=node.embedding,
                    payload={
                        "content": node.content,
                        "topic": topic,
                        "chunk_id": node.node_id,
                        "raptor_level": node.level,
                        "raptor_children": node.children,
                        **node.metadata,
                    },
                )
            )

        # Upsert to Qdrant
        client.upsert(
            collection_name="cohesionn",
            points=points,
            wait=True,
        )

        logger.debug(f"Stored {len(points)} RAPTOR nodes for {topic}")
        return len(points)

    def delete_tree(self, topic: str) -> int:
        """Delete all RAPTOR nodes for a topic."""
        from ..vectorstore import get_knowledge_base
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        kb = get_knowledge_base()
        store = kb.get_store(topic)
        client = store.client

        # Count before deletion
        count = self._count_raptor_nodes(kb, topic)

        if count == 0:
            return 0

        # Delete RAPTOR nodes (level > 0)
        client.delete(
            collection_name="cohesionn",
            points_selector=Filter(
                must=[
                    FieldCondition(key="topic", match=MatchValue(value=topic)),
                    FieldCondition(key="raptor_level", match=MatchAny(any=[1, 2, 3])),
                ],
            ),
        )

        logger.info(f"Deleted {count} RAPTOR nodes for {topic}")
        return count

    def get_tree_stats(self, topic: str) -> Dict[str, Any]:
        """Get statistics about RAPTOR tree for a topic."""
        from ..vectorstore import get_knowledge_base
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        kb = get_knowledge_base()
        store = kb.get_store(topic)
        client = store.client

        stats = {
            "topic": topic,
            "levels": {},
            "total_nodes": 0,
        }

        for level in range(4):  # 0, 1, 2, 3
            try:
                if level == 0:
                    # Level 0: chunks without raptor_level
                    # This is approximated by total minus RAPTOR nodes
                    pass
                else:
                    result = client.count(
                        collection_name="cohesionn",
                        count_filter=Filter(
                            must=[
                                FieldCondition(key="topic", match=MatchValue(value=topic)),
                                FieldCondition(key="raptor_level", match=MatchValue(value=level)),
                            ],
                        ),
                    )
                    stats["levels"][level] = result.count
                    stats["total_nodes"] += result.count
            except Exception:
                stats["levels"][level] = 0

        return stats
