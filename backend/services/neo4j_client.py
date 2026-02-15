"""
Neo4j Client Service - Connection pooling and health checks

Provides a connection pool to Neo4j for GraphRAG operations.
"""

import logging
import os
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j client with connection pooling and health checks.

    Features:
    - Connection pooling for concurrent queries
    - Automatic reconnection on failure
    - Health check endpoint
    - Schema initialization
    """

    def __init__(
        self,
        url: str = None,
        user: str = None,
        password: str = None,
        max_connection_pool_size: int = 50,
    ):
        """
        Args:
            url: Neo4j bolt URL
            user: Neo4j username
            password: Neo4j password
            max_connection_pool_size: Max connections in pool
        """
        self.url = url or os.environ.get("NEO4J_URL", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "")
        self.max_connection_pool_size = max_connection_pool_size

        self._driver = None
        self._initialized = False

    @property
    def driver(self):
        """Lazy-load Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase

                self._driver = GraphDatabase.driver(
                    self.url,
                    auth=(self.user, self.password),
                    max_connection_pool_size=self.max_connection_pool_size,
                    connection_timeout=5,  # 5 second timeout
                    max_transaction_retry_time=5,  # Don't retry forever
                )
                logger.info(f"Neo4j driver connected to {self.url}")
            except ImportError:
                logger.error("neo4j package not installed")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise

        return self._driver

    def close(self):
        """Close the driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    def verify_connectivity(self) -> bool:
        """Verify connection to Neo4j."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Neo4j connectivity check failed: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS health")
                _ = result.single()

            return {
                "status": "healthy",
                "url": self.url,
                "connected": True,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "url": self.url,
                "connected": False,
                "error": str(e),
            }

    @contextmanager
    def session(self, database: str = None):
        """
        Get a session context manager.

        Usage:
            with client.session() as session:
                result = session.run("MATCH (n) RETURN n LIMIT 10")
        """
        session = self.driver.session(database=database)
        try:
            yield session
        finally:
            session.close()

    def run_query(
        self,
        query: str,
        parameters: Dict[str, Any] = None,
        database: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a Cypher query and return results.

        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Database name (optional)

        Returns:
            List of result records as dicts
        """
        with self.session(database=database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def run_write_query(
        self,
        query: str,
        parameters: Dict[str, Any] = None,
        database: str = None,
    ) -> Dict[str, Any]:
        """
        Run a write query in a transaction.

        Returns summary with counters.
        """
        with self.session(database=database) as session:

            def _write_tx(tx):
                result = tx.run(query, parameters or {})
                summary = result.consume()
                return {
                    "nodes_created": summary.counters.nodes_created,
                    "nodes_deleted": summary.counters.nodes_deleted,
                    "relationships_created": summary.counters.relationships_created,
                    "relationships_deleted": summary.counters.relationships_deleted,
                    "properties_set": summary.counters.properties_set,
                }

            return session.execute_write(_write_tx)

    def initialize_schema(self):
        """
        Initialize Neo4j schema with constraints and indexes.

        Creates:
        - Core node uniqueness constraints (always)
        - Domain-specific constraints (if patterns configured)
        - Full-text indexes for entity search
        """
        if self._initialized:
            return

        # Load domain pipeline config to determine which entity types are active
        try:
            from domain_loader import get_pipeline_config
            pipeline = get_pipeline_config()
        except Exception:
            pipeline = {}

        # Core constraints (always present — generic RAG types)
        schema_queries = [
            "CREATE CONSTRAINT standard_code IF NOT EXISTS FOR (s:Standard) REQUIRE s.code IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            # Indexes for common queries
            "CREATE INDEX standard_org IF NOT EXISTS FOR (s:Standard) ON (s.org)",
            "CREATE INDEX chunk_topic IF NOT EXISTS FOR (c:Chunk) ON (c.topic)",
        ]

        # Domain-specific constraints (only if patterns configured in lexicon)
        if pipeline.get("graph_equipment"):
            schema_queries.append(
                "CREATE CONSTRAINT equipment_name IF NOT EXISTS FOR (e:Equipment) REQUIRE e.name IS UNIQUE"
            )
        if pipeline.get("graph_test_methods"):
            schema_queries.append(
                "CREATE CONSTRAINT test_method_name IF NOT EXISTS FOR (t:TestMethod) REQUIRE t.name IS UNIQUE"
            )
        if pipeline.get("graph_soil_types"):
            schema_queries.append(
                "CREATE CONSTRAINT soil_type_name IF NOT EXISTS FOR (s:SoilType) REQUIRE s.name IS UNIQUE"
            )
        if pipeline.get("graph_parameters"):
            schema_queries.append(
                "CREATE CONSTRAINT parameter_name IF NOT EXISTS FOR (p:Parameter) REQUIRE p.name IS UNIQUE"
            )

        # Full-text index — include only configured entity types
        entity_labels = ["Standard"]
        if pipeline.get("graph_equipment"):
            entity_labels.append("Equipment")
        if pipeline.get("graph_test_methods"):
            entity_labels.append("TestMethod")
        if pipeline.get("graph_soil_types"):
            entity_labels.append("SoilType")
        if pipeline.get("graph_parameters"):
            entity_labels.append("Parameter")

        labels_str = "|".join(entity_labels)
        schema_queries.append(
            f"CREATE FULLTEXT INDEX entity_search IF NOT EXISTS "
            f"FOR (n:{labels_str}) "
            f"ON EACH [n.name, n.code, n.abbreviation]"
        )

        for query in schema_queries:
            try:
                self.run_write_query(query.strip())
                logger.debug(f"Schema query executed: {query[:50]}...")
            except Exception as e:
                err_str = str(e).lower()
                # Constraint may already exist - that's fine
                if "already exists" in err_str:
                    continue
                # Auth failures won't resolve by retrying more queries
                if "unauthorized" in err_str or "authentication" in err_str:
                    logger.warning(f"Neo4j auth failed during schema init: {e}")
                    logger.warning("Skipping remaining schema queries - fix credentials and restart")
                    break
                logger.warning(f"Schema query failed: {e}")

        self._initialized = True
        logger.info("Neo4j schema initialized")

    def get_node_counts(self) -> Dict[str, int]:
        """Get count of nodes by label."""
        query = """
        CALL db.labels() YIELD label
        CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) as count', {}) YIELD value
        RETURN label, value.count as count
        """

        # Fallback if APOC not available
        fallback_query = """
        MATCH (n)
        WITH labels(n) as labels, count(*) as cnt
        UNWIND labels as label
        RETURN label, sum(cnt) as count
        """

        try:
            results = self.run_query(query)
        except Exception:
            try:
                results = self.run_query(fallback_query)
            except Exception as e:
                logger.warning(f"Could not get node counts: {e}")
                return {}

        return {r["label"]: r["count"] for r in results}

    def get_relationship_counts(self) -> Dict[str, int]:
        """Get count of relationships by type."""
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(r) as count
        """

        try:
            results = self.run_query(query)
            return {r["type"]: r["count"] for r in results}
        except Exception as e:
            logger.warning(f"Could not get relationship counts: {e}")
            return {}

    def ensure_entity_constraint(self, entity_type: str) -> None:
        """Dynamically create uniqueness constraint for a new entity type.

        Called during ingestion when new entity types are discovered.
        Safe to call repeatedly — uses IF NOT EXISTS.
        """
        safe_name = entity_type.lower().replace(" ", "_")
        query = f"CREATE CONSTRAINT {safe_name}_name IF NOT EXISTS FOR (n:{entity_type}) REQUIRE n.name IS UNIQUE"
        try:
            self.run_write_query(query)
            logger.info(f"Auto-created constraint for entity type: {entity_type}")
        except Exception as e:
            logger.warning(f"Could not create constraint for {entity_type}: {e}")

    def update_fulltext_index(self, entity_types: list) -> None:
        """Recreate fulltext index to cover all active entity types."""
        labels_str = "|".join(entity_types)
        try:
            self.run_write_query("DROP INDEX entity_search IF EXISTS")
            self.run_write_query(
                f"CREATE FULLTEXT INDEX entity_search IF NOT EXISTS "
                f"FOR (n:{labels_str}) ON EACH [n.name, n.code, n.abbreviation]"
            )
            logger.info(f"Updated fulltext index for types: {entity_types}")
        except Exception as e:
            logger.warning(f"Fulltext index update warning: {e}")


# Singleton
_neo4j_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """Get or create singleton Neo4j client."""
    global _neo4j_client

    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
        _neo4j_client.initialize_schema()

    return _neo4j_client


def close_neo4j_client():
    """Close the singleton client."""
    global _neo4j_client

    if _neo4j_client is not None:
        _neo4j_client.close()
        _neo4j_client = None
