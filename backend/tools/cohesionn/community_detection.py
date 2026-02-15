"""
Community Detection - Leiden algorithm for graph community detection

Detects communities in the knowledge graph for global search.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """A detected community in the graph."""

    community_id: str
    level: str  # "coarse", "medium", "fine"
    node_ids: List[str]  # Chunk IDs in this community
    entity_ids: List[str]  # Entity names in this community
    node_count: int
    entity_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Result from community detection."""

    communities: List[Community]
    level: str
    total_nodes: int
    modularity: float = 0.0


class CommunityDetector:
    """
    Detect communities in the knowledge graph using Leiden algorithm.

    Supports multiple resolution levels:
    - Coarse (5-15 communities): Broad themes
    - Medium (20-50): Standard topics (default for global search)
    - Fine (50-200): Specific subtopics

    Uses NetworkX for graph representation and leidenalg for clustering.
    """

    # Resolution parameters for different levels
    RESOLUTION_PARAMS = {
        "coarse": 0.2,   # Lower = fewer, larger communities
        "medium": 0.5,   # Balanced
        "fine": 1.0,     # Higher = more, smaller communities
    }

    def __init__(
        self,
        default_level: str = "medium",
        min_community_size: int = 3,
    ):
        """
        Args:
            default_level: Default resolution level
            min_community_size: Minimum nodes to form a community
        """
        self.default_level = default_level
        self.min_community_size = min_community_size

    def detect(
        self,
        topic: str = None,
        level: str = None,
    ) -> DetectionResult:
        """
        Detect communities in the knowledge graph.

        Args:
            topic: Optional topic filter (None = entire graph)
            level: Resolution level ("coarse", "medium", "fine")

        Returns:
            DetectionResult with communities
        """
        level = level or self.default_level
        resolution = self.RESOLUTION_PARAMS.get(level, 0.5)

        logger.info(f"Detecting communities at level '{level}' (resolution={resolution})")

        try:
            # Load graph from Neo4j
            graph = self._load_graph_from_neo4j(topic)

            if graph.number_of_nodes() == 0:
                logger.warning("Empty graph, no communities to detect")
                return DetectionResult(
                    communities=[],
                    level=level,
                    total_nodes=0,
                )

            # Run Leiden algorithm
            communities = self._run_leiden(graph, resolution)

            # Filter small communities
            communities = [c for c in communities if c.node_count >= self.min_community_size]

            logger.info(f"Detected {len(communities)} communities from {graph.number_of_nodes()} nodes")

            return DetectionResult(
                communities=communities,
                level=level,
                total_nodes=graph.number_of_nodes(),
            )

        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return DetectionResult(
                communities=[],
                level=level,
                total_nodes=0,
            )

    def _load_graph_from_neo4j(self, topic: str = None) -> Any:
        """Load knowledge graph from Neo4j into NetworkX."""
        try:
            import networkx as nx
            from services.neo4j_client import get_neo4j_client

            neo4j = get_neo4j_client()

            # Build query with optional topic filter
            topic_filter = "WHERE c.topic = $topic" if topic else ""

            # Query all nodes and relationships
            query = f"""
                MATCH (c:Chunk){topic_filter}
                OPTIONAL MATCH (c)-[:CONTAINS]->(e)
                OPTIONAL MATCH (e)-[r]-(e2)
                WHERE NOT e2:Chunk
                RETURN c.chunk_id AS chunk_id,
                       c.topic AS topic,
                       collect(DISTINCT e.name) AS entities,
                       collect(DISTINCT {{
                           source: e.name,
                           target: e2.name,
                           type: type(r)
                       }}) AS relationships
            """

            params = {"topic": topic} if topic else {}
            results = neo4j.run_query(query, params)

            # Build NetworkX graph
            G = nx.Graph()

            for r in results:
                chunk_id = r["chunk_id"]
                if not chunk_id:
                    continue

                # Add chunk node
                G.add_node(chunk_id, node_type="chunk", topic=r.get("topic", ""))

                # Add entity nodes and edges
                for entity in r.get("entities", []) or []:
                    if entity:
                        G.add_node(entity, node_type="entity")
                        G.add_edge(chunk_id, entity, rel_type="CONTAINS")

                # Add entity-entity relationships
                for rel in r.get("relationships", []) or []:
                    if rel.get("source") and rel.get("target"):
                        G.add_edge(rel["source"], rel["target"], rel_type=rel.get("type", "RELATED"))

            logger.debug(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G

        except ImportError:
            logger.error("networkx not installed")
            import networkx as nx
            return nx.Graph()

    def _run_leiden(self, graph: Any, resolution: float) -> List[Community]:
        """Run Leiden community detection algorithm."""
        try:
            import igraph as ig
            import leidenalg

            # Convert NetworkX to igraph
            nx_nodes = list(graph.nodes())
            node_to_idx = {n: i for i, n in enumerate(nx_nodes)}

            ig_graph = ig.Graph()
            ig_graph.add_vertices(len(nx_nodes))

            edges = [
                (node_to_idx[u], node_to_idx[v])
                for u, v in graph.edges()
                if u in node_to_idx and v in node_to_idx
            ]
            ig_graph.add_edges(edges)

            # Run Leiden
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
            )

            # Convert partition to Community objects
            communities = []
            for i, members in enumerate(partition):
                node_names = [nx_nodes[idx] for idx in members]

                # Separate chunks from entities
                chunk_ids = [n for n in node_names if graph.nodes[n].get("node_type") == "chunk"]
                entity_ids = [n for n in node_names if graph.nodes[n].get("node_type") == "entity"]

                communities.append(
                    Community(
                        community_id=f"community_{i}",
                        level=self.default_level,
                        node_ids=chunk_ids,
                        entity_ids=entity_ids,
                        node_count=len(chunk_ids),
                        entity_count=len(entity_ids),
                    )
                )

            return communities

        except ImportError as e:
            logger.warning(f"Leiden algorithm not available: {e}, using fallback")
            return self._fallback_clustering(graph)

    def _fallback_clustering(self, graph: Any) -> List[Community]:
        """Fallback: use connected components as communities."""
        try:
            import networkx as nx

            components = list(nx.connected_components(graph))

            communities = []
            for i, component in enumerate(components):
                node_names = list(component)
                chunk_ids = [n for n in node_names if graph.nodes[n].get("node_type") == "chunk"]
                entity_ids = [n for n in node_names if graph.nodes[n].get("node_type") == "entity"]

                communities.append(
                    Community(
                        community_id=f"component_{i}",
                        level="fallback",
                        node_ids=chunk_ids,
                        entity_ids=entity_ids,
                        node_count=len(chunk_ids),
                        entity_count=len(entity_ids),
                    )
                )

            return communities

        except Exception as e:
            logger.error(f"Fallback clustering failed: {e}")
            return []

    def detect_at_all_levels(self, topic: str = None) -> Dict[str, DetectionResult]:
        """Detect communities at all resolution levels."""
        results = {}
        for level in ["coarse", "medium", "fine"]:
            results[level] = self.detect(topic=topic, level=level)
        return results
