"""
GraphRAG Graph Retriever - Entity-based retrieval via graph traversal

Extracts entities from query, finds them in graph, traverses to linked chunks.
"""

import logging
import warnings
from typing import List, Dict, Any, Optional

from .graph_extraction import EntityExtractor

logger = logging.getLogger(__name__)

# Suppress verbose Neo4j GqlStatusObject warnings from Cypher queries
warnings.filterwarnings("ignore", message=".*GqlStatusObject.*")
logging.getLogger("neo4j").setLevel(logging.ERROR)


class GraphRetriever:
    """
    Graph-based retrieval using Neo4j knowledge graph.

    Process:
    1. Extract entities from query
    2. Find matching entities in Neo4j (full-text search)
    3. Traverse graph to find connected chunks (max 2 hops)
    4. Return chunk IDs with graph-based relevance scores

    Scores are based on:
    - Direct entity match: 1.0
    - 1-hop connected: 0.7
    - 2-hop connected: 0.4
    """

    # Hop decay factors for scoring
    HOP_SCORES = {
        0: 1.0,   # Direct match
        1: 0.7,   # 1 hop away
        2: 0.4,   # 2 hops away
    }

    def __init__(
        self,
        extractor: Optional[EntityExtractor] = None,
        max_hops: int = 2,
    ):
        """
        Args:
            extractor: EntityExtractor for query analysis
            max_hops: Maximum graph traversal depth
        """
        self.extractor = extractor or EntityExtractor(use_spacy=False)
        self.max_hops = max_hops

    def retrieve(
        self,
        query: str,
        topics: List[str],
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks via graph traversal.

        Args:
            query: Search query
            topics: Topics to filter results
            n_results: Maximum results to return

        Returns:
            List of results with chunk_id, graph_score, and path info
        """
        try:
            from services.neo4j_client import get_neo4j_client

            neo4j = get_neo4j_client()

            # Step 1: Extract entities from query
            extraction = self.extractor.extract(query)

            if not extraction.entities:
                logger.debug("No entities extracted from query, skipping graph retrieval")
                return []

            entity_names = [e.name for e in extraction.entities]
            logger.debug(f"Query entities: {entity_names}")

            # Step 2: Find entities in graph and traverse to chunks
            results = self._traverse_to_chunks(neo4j, entity_names, topics)

            # Step 3: Sort by score and limit
            results.sort(key=lambda x: x["graph_score"], reverse=True)
            results = results[:n_results]

            logger.debug(f"Graph retrieval: {len(results)} results from {len(entity_names)} entities")
            return results

        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return []

    def _traverse_to_chunks(
        self,
        neo4j: Any,
        entity_names: List[str],
        topics: List[str],
    ) -> List[Dict[str, Any]]:
        """Traverse from entities to linked chunks."""
        # Build topic filter
        topic_filter = "c.topic IN $topics" if topics else "true"

        # Query for direct matches (0 hops) and traversals (1-2 hops)
        query = f"""
            // Find entities matching query terms
            UNWIND $entity_names AS name
            CALL db.index.fulltext.queryNodes('entity_search', name + '~')
            YIELD node AS entity, score AS match_score

            // Traverse to chunks (0-2 hops)
            MATCH path = (c:Chunk)-[:CONTAINS*1..{self.max_hops + 1}]-(entity)
            WHERE {topic_filter}

            // Calculate score based on path length
            WITH c, entity, match_score,
                 length(path) - 1 AS hops
            ORDER BY hops ASC

            // Return unique chunks with best score
            RETURN DISTINCT c.chunk_id AS chunk_id,
                   c.topic AS topic,
                   c.source AS source,
                   collect(DISTINCT entity.name)[0..3] AS matched_entities,
                   min(hops) AS min_hops,
                   max(match_score) AS entity_match_score
        """

        # Fallback query if full-text index doesn't exist
        fallback_query = f"""
            UNWIND $entity_names AS name
            MATCH (entity)
            WHERE (entity:Standard OR entity:TestMethod OR entity:Parameter
                   OR entity:SoilType OR entity:Equipment)
              AND toLower(entity.name) CONTAINS toLower(name)

            MATCH path = (c:Chunk)-[:CONTAINS*1..{self.max_hops + 1}]-(entity)
            WHERE {topic_filter}

            WITH c, entity,
                 length(path) - 1 AS hops
            ORDER BY hops ASC

            RETURN DISTINCT c.chunk_id AS chunk_id,
                   c.topic AS topic,
                   c.source AS source,
                   collect(DISTINCT entity.name)[0..3] AS matched_entities,
                   min(hops) AS min_hops,
                   1.0 AS entity_match_score
        """

        params = {
            "entity_names": entity_names,
            "topics": topics,
        }

        try:
            results = neo4j.run_query(query, params)
        except Exception:
            # Fallback if full-text index not available
            try:
                results = neo4j.run_query(fallback_query, params)
            except Exception as e:
                logger.warning(f"Graph traversal query failed: {e}")
                return []

        # Calculate graph scores
        processed = []
        for r in results:
            hops = r.get("min_hops", 1)
            hop_score = self.HOP_SCORES.get(hops, 0.2)
            entity_score = r.get("entity_match_score", 1.0)

            # Combined score
            graph_score = hop_score * min(entity_score, 1.0)

            processed.append({
                "id": r["chunk_id"],
                "chunk_id": r["chunk_id"],
                "topic": r.get("topic", ""),
                "source": r.get("source", ""),
                "graph_score": graph_score,
                "graph_hops": hops,
                "matched_entities": r.get("matched_entities", []),
                "is_graph_result": True,
            })

        return processed

    def find_related_chunks(
        self,
        chunk_id: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find chunks related to a given chunk via shared entities.

        Useful for "related content" recommendations.
        """
        try:
            from services.neo4j_client import get_neo4j_client

            neo4j = get_neo4j_client()

            query = """
                MATCH (c1:Chunk {chunk_id: $chunk_id})-[:CONTAINS]->(e)<-[:CONTAINS]-(c2:Chunk)
                WHERE c1 <> c2
                WITH c2, count(DISTINCT e) AS shared_entities
                ORDER BY shared_entities DESC
                LIMIT $limit
                RETURN c2.chunk_id AS chunk_id,
                       c2.topic AS topic,
                       c2.source AS source,
                       shared_entities
            """

            results = neo4j.run_query(query, {
                "chunk_id": chunk_id,
                "limit": max_results,
            })

            return [
                {
                    "chunk_id": r["chunk_id"],
                    "topic": r.get("topic", ""),
                    "source": r.get("source", ""),
                    "shared_entities": r["shared_entities"],
                    "relevance_score": min(1.0, r["shared_entities"] / 5.0),
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Related chunk lookup failed: {e}")
            return []

    def get_entity_context(
        self,
        entity_name: str,
    ) -> Dict[str, Any]:
        """
        Get rich context for an entity from the graph.

        Returns connected entities, relationships, and source chunks.
        """
        try:
            from services.neo4j_client import get_neo4j_client

            neo4j = get_neo4j_client()

            query = """
                MATCH (e {name: $name})
                OPTIONAL MATCH (e)-[r]-(related)
                WHERE NOT related:Chunk
                WITH e, collect(DISTINCT {
                    name: related.name,
                    type: labels(related)[0],
                    relationship: type(r)
                }) AS connections
                OPTIONAL MATCH (c:Chunk)-[:CONTAINS]->(e)
                WITH e, connections, collect(DISTINCT c.chunk_id)[0..5] AS source_chunks
                RETURN e.name AS name,
                       labels(e)[0] AS type,
                       properties(e) AS properties,
                       connections,
                       source_chunks
            """

            results = neo4j.run_query(query, {"name": entity_name})

            if not results:
                return {}

            r = results[0]
            return {
                "name": r["name"],
                "type": r["type"],
                "properties": r["properties"],
                "connections": r["connections"],
                "source_chunks": r["source_chunks"],
            }

        except Exception as e:
            logger.error(f"Entity context lookup failed: {e}")
            return {}


# Singleton
_graph_retriever: Optional[GraphRetriever] = None


def get_graph_retriever() -> GraphRetriever:
    """Get or create singleton graph retriever."""
    global _graph_retriever

    if _graph_retriever is None:
        from config import runtime_config

        _graph_retriever = GraphRetriever(max_hops=runtime_config.graph_max_hops)

    return _graph_retriever
