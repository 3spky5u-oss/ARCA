"""
GraphRAG Graph Builder - Build knowledge graph from extracted entities

Batch processes chunks to build Neo4j graph with entities and relationships.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

from .graph_extraction import EntityExtractor, Entity, Relationship, ExtractionResult

logger = logging.getLogger(__name__)

# Auto-registered entity types (avoid repeated constraint creation calls)
_registered_entity_types: set = {"Standard", "Chunk"}

# SpaCy NER types that are too noisy/generic for graph nodes
_SPACY_NOISE_TYPES = {"DATE", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "TIME", "QUANTITY"}


@dataclass
class GraphBuildResult:
    """Result from graph building operation."""

    topic: str
    chunks_processed: int
    entities_created: int
    relationships_created: int
    build_time_seconds: float
    errors: List[str]


class GraphBuilder:
    """
    Build knowledge graph from document chunks.

    Process:
    1. Fetch chunks from Qdrant (by topic)
    2. Extract entities and relationships from each chunk
    3. MERGE nodes and relationships into Neo4j
    4. Create Chunk nodes linked to entities for retrieval

    Supports incremental updates - only processes new chunks.
    """

    # Cypher templates for node creation
    NODE_MERGE_TEMPLATES = {
        "Standard": """
            MERGE (n:Standard {code: $code})
            ON CREATE SET n.name = $name, n.org = $org, n.created_at = datetime()
            ON MATCH SET n.updated_at = datetime()
            RETURN n
        """,
        "TestMethod": """
            MERGE (n:TestMethod {name: $name})
            ON CREATE SET n.abbreviation = $abbreviation, n.full_name = $full_name, n.created_at = datetime()
            ON MATCH SET n.updated_at = datetime()
            RETURN n
        """,
        "SoilType": """
            MERGE (n:SoilType {name: $name})
            ON CREATE SET n.classification = $classification, n.description = $description,
                          n.system = $system, n.created_at = datetime()
            ON MATCH SET n.updated_at = datetime()
            RETURN n
        """,
        "Parameter": """
            MERGE (n:Parameter {name: $name})
            ON CREATE SET n.symbol = $symbol, n.description = $description,
                          n.unit = $unit, n.created_at = datetime()
            ON MATCH SET n.updated_at = datetime()
            RETURN n
        """,
        "Equipment": """
            MERGE (n:Equipment {name: $name})
            ON CREATE SET n.created_at = datetime()
            ON MATCH SET n.updated_at = datetime()
            RETURN n
        """,
        "Chunk": """
            MERGE (n:Chunk {chunk_id: $chunk_id})
            ON CREATE SET n.topic = $topic, n.source = $source, n.created_at = datetime()
            ON MATCH SET n.updated_at = datetime()
            RETURN n
        """,
    }

    # Relationship creation template
    RELATIONSHIP_TEMPLATE = """
        MATCH (source:{source_label} {{name: $source_name}})
        MATCH (target:{target_label} {{name: $target_name}})
        MERGE (source)-[r:{rel_type}]->(target)
        ON CREATE SET r.created_at = datetime()
        RETURN r
    """

    # Chunk-entity link template
    CHUNK_LINK_TEMPLATE = """
        MATCH (chunk:Chunk {{chunk_id: $chunk_id}})
        MATCH (entity:{entity_label} {{name: $entity_name}})
        MERGE (chunk)-[r:CONTAINS]->(entity)
        RETURN r
    """

    def __init__(
        self,
        extractor: Optional[EntityExtractor] = None,
        batch_size: int = 100,
    ):
        """
        Args:
            extractor: EntityExtractor instance
            batch_size: Number of chunks to process in each batch
        """
        self.extractor = extractor or EntityExtractor()
        self.batch_size = batch_size

    def _ensure_type_registered(self, entity_type: str) -> None:
        """Auto-register constraints for newly discovered entity types."""
        if entity_type in _registered_entity_types:
            return
        if entity_type in _SPACY_NOISE_TYPES:
            return

        try:
            from services.neo4j_client import get_neo4j_client
            neo4j = get_neo4j_client()
            neo4j.ensure_entity_constraint(entity_type)
            _registered_entity_types.add(entity_type)

            # Update fulltext index to include new type
            neo4j.update_fulltext_index(list(_registered_entity_types - {"Chunk"}))
        except Exception as e:
            logger.warning(f"Failed to auto-register entity type {entity_type}: {e}")

    def build_graph(
        self,
        topic: str,
        incremental: bool = True,
    ) -> GraphBuildResult:
        """
        Build knowledge graph for a topic.

        Args:
            topic: Knowledge base topic
            incremental: If True, only process new chunks

        Returns:
            GraphBuildResult with statistics
        """
        start_time = time.time()
        errors = []
        entities_created = 0
        relationships_created = 0
        chunks_processed = 0

        logger.info(f"Building graph for topic: {topic}")

        try:
            from services.neo4j_client import get_neo4j_client
            from .vectorstore import get_knowledge_base

            neo4j = get_neo4j_client()
            kb = get_knowledge_base()

            # Fetch chunks from Qdrant
            chunks = self._fetch_chunks(kb, topic)
            total_chunks = len(chunks)

            if total_chunks == 0:
                logger.warning(f"No chunks found for topic: {topic}")
                return GraphBuildResult(
                    topic=topic,
                    chunks_processed=0,
                    entities_created=0,
                    relationships_created=0,
                    build_time_seconds=0,
                    errors=["No chunks found"],
                )

            logger.info(f"Processing {total_chunks} chunks")

            # Filter to unprocessed chunks if incremental
            if incremental:
                processed_ids = self._get_processed_chunk_ids(neo4j, topic)
                chunks = [c for c in chunks if c["chunk_id"] not in processed_ids]
                logger.info(f"Incremental: {len(chunks)} new chunks to process")

            # Process in batches
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]

                # Extract entities and relationships
                extractions = self.extractor.extract_batch(batch)

                # Auto-register new entity types with sufficient instances
                entity_type_counts: dict = {}
                for extraction in extractions:
                    for entity in extraction.entities:
                        t = entity.entity_type
                        entity_type_counts[t] = entity_type_counts.get(t, 0) + 1

                for etype, count in entity_type_counts.items():
                    if count >= 2:
                        self._ensure_type_registered(etype)

                # Build graph from extractions
                for chunk, extraction in zip(batch, extractions):
                    try:
                        # Create Chunk node
                        self._create_chunk_node(neo4j, chunk, topic)
                        chunks_processed += 1

                        # Create entity nodes and link to chunk
                        for entity in extraction.entities:
                            self._create_entity_node(neo4j, entity)
                            self._link_chunk_to_entity(neo4j, chunk["chunk_id"], entity)
                            entities_created += 1

                        # Create relationships
                        for rel in extraction.relationships:
                            self._create_relationship(neo4j, rel, extraction.entities)
                            relationships_created += 1

                    except Exception as e:
                        errors.append(f"Chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                        logger.error(f"Failed to process chunk: {e}")

                # Log progress
                if len(chunks) > 100 and (i + self.batch_size) % 500 == 0:
                    logger.info(f"Progress: {min(i + self.batch_size, len(chunks))}/{len(chunks)}")

        except Exception as e:
            logger.error(f"Graph build failed: {e}")
            errors.append(str(e))

        build_time = time.time() - start_time

        logger.info(
            f"Graph build complete: {chunks_processed} chunks, "
            f"{entities_created} entities, {relationships_created} relationships, "
            f"{build_time:.1f}s"
        )

        return GraphBuildResult(
            topic=topic,
            chunks_processed=chunks_processed,
            entities_created=entities_created,
            relationships_created=relationships_created,
            build_time_seconds=build_time,
            errors=errors,
        )

    def _fetch_chunks(self, kb: Any, topic: str) -> List[Dict[str, Any]]:
        """Fetch chunks from Qdrant for a topic."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        store = kb.get_store(topic)
        client = store.client

        all_chunks = []
        offset = None

        while True:
            response = client.scroll(
                collection_name="cohesionn",
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="topic", match=MatchValue(value=topic)),
                    ],
                    must_not=[
                        # Exclude RAPTOR summary nodes
                        FieldCondition(key="raptor_level", match=MatchValue(value=1)),
                        FieldCondition(key="raptor_level", match=MatchValue(value=2)),
                        FieldCondition(key="raptor_level", match=MatchValue(value=3)),
                    ],
                ),
                limit=1000,
                offset=offset,
                with_payload=True,
            )

            points, next_offset = response

            for point in points:
                payload = point.payload or {}
                all_chunks.append({
                    "chunk_id": payload.get("chunk_id", str(point.id)),
                    "content": payload.get("content", ""),
                    "source": payload.get("source", ""),
                    "topic": topic,
                })

            if next_offset is None:
                break
            offset = next_offset

        return all_chunks

    def _get_processed_chunk_ids(self, neo4j: Any, topic: str) -> set:
        """Get set of chunk IDs already in the graph."""
        query = """
            MATCH (c:Chunk {topic: $topic})
            RETURN c.chunk_id as chunk_id
        """
        results = neo4j.run_query(query, {"topic": topic})
        return {r["chunk_id"] for r in results}

    def _create_chunk_node(self, neo4j: Any, chunk: Dict[str, Any], topic: str):
        """Create a Chunk node in Neo4j."""
        query = self.NODE_MERGE_TEMPLATES["Chunk"]
        neo4j.run_write_query(query, {
            "chunk_id": chunk["chunk_id"],
            "topic": topic,
            "source": chunk.get("source", ""),
        })

    def _create_entity_node(self, neo4j: Any, entity: Entity):
        """Create an entity node in Neo4j."""
        template = self.NODE_MERGE_TEMPLATES.get(entity.entity_type)

        if not template:
            # Generic fallback for auto-discovered entity types
            if entity.entity_type not in _SPACY_NOISE_TYPES:
                template = f"""
                    MERGE (n:{entity.entity_type} {{name: $name}})
                    ON CREATE SET n.created_at = datetime()
                    ON MATCH SET n.updated_at = datetime()
                    RETURN n
                """
            else:
                return  # Skip noisy types entirely

        # Build parameters from entity
        params = {"name": entity.name}
        params.update(entity.properties)

        # Handle Standard nodes (use code as key, ensure org is set)
        if entity.entity_type == "Standard":
            params["code"] = entity.properties.get("code", entity.name)
            params.setdefault("org", "")

        neo4j.run_write_query(template, params)

    def _link_chunk_to_entity(self, neo4j: Any, chunk_id: str, entity: Entity):
        """Create CONTAINS relationship from Chunk to Entity."""
        query = self.CHUNK_LINK_TEMPLATE.format(entity_label=entity.entity_type)
        neo4j.run_write_query(query, {
            "chunk_id": chunk_id,
            "entity_name": entity.name,
        })

    def _create_relationship(
        self,
        neo4j: Any,
        rel: Relationship,
        entities: List[Entity],
    ):
        """Create a relationship between entities."""
        # Find entity types for source and target
        source_entity = next((e for e in entities if e.name == rel.source), None)
        target_entity = next((e for e in entities if e.name == rel.target), None)

        if not source_entity or not target_entity:
            return

        query = self.RELATIONSHIP_TEMPLATE.format(
            source_label=source_entity.entity_type,
            target_label=target_entity.entity_type,
            rel_type=rel.rel_type,
        )

        neo4j.run_write_query(query, {
            "source_name": rel.source,
            "target_name": rel.target,
        })

    def delete_topic_graph(self, topic: str) -> int:
        """Delete all graph nodes and relationships for a topic."""
        from services.neo4j_client import get_neo4j_client

        neo4j = get_neo4j_client()

        # Delete Chunk nodes and their relationships for this topic
        query = """
            MATCH (c:Chunk {topic: $topic})
            DETACH DELETE c
        """
        result = neo4j.run_write_query(query, {"topic": topic})
        deleted = result.get("nodes_deleted", 0)

        logger.info(f"Deleted {deleted} Chunk nodes for topic: {topic}")
        return deleted

    def get_graph_stats(self, topic: str = None) -> Dict[str, Any]:
        """Get graph statistics."""
        from services.neo4j_client import get_neo4j_client

        neo4j = get_neo4j_client()

        stats = {
            "node_counts": neo4j.get_node_counts(),
            "relationship_counts": neo4j.get_relationship_counts(),
        }

        if topic:
            # Topic-specific stats
            query = """
                MATCH (c:Chunk {topic: $topic})
                OPTIONAL MATCH (c)-[:CONTAINS]->(e)
                RETURN count(DISTINCT c) as chunks, count(DISTINCT e) as entities
            """
            result = neo4j.run_query(query, {"topic": topic})
            if result:
                stats["topic_chunks"] = result[0]["chunks"]
                stats["topic_entities"] = result[0]["entities"]

        return stats
