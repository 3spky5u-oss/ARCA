"""
Admin Graph Router - Neo4j Knowledge Graph Management

Provides endpoints for exploring and managing the GraphRAG knowledge graph:
- Statistics dashboard (node/relationship counts)
- Entity browser with search and pagination
- Interactive graph visualization
- Relationship explorer
- Cypher query console (power user feature)

Protected by ADMIN_KEY header authentication.
"""

import logging
import re
import time
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from config import runtime_config
from services.admin_auth import verify_admin
from services.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)


def _to_dict(obj: Any) -> dict:
    """Safely convert neo4j Node/Relationship/dict to a plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "items"):
        return dict(obj.items())
    return {}


def _sanitize_neo4j(obj: Any) -> Any:
    """Convert neo4j types to JSON-serializable Python types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_neo4j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_neo4j(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, timedelta):
        return str(obj)
    # neo4j Node/Relationship — convert to dict first
    if hasattr(obj, "items"):
        return {k: _sanitize_neo4j(v) for k, v in obj.items()}
    # neo4j.time types have .iso_format() or can be str()-ed
    if hasattr(obj, "iso_format"):
        return obj.iso_format()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)

router = APIRouter(prefix="/api/admin/graph", tags=["admin-graph"])

# =============================================================================
# REQUEST MODELS
# =============================================================================


class CypherQueryRequest(BaseModel):
    """Request for custom Cypher query."""

    query: str
    parameters: Optional[Dict[str, Any]] = None


# =============================================================================
# CYPHER SECURITY
# =============================================================================

# Patterns that indicate write operations - blocked by default
WRITE_PATTERNS = [
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bDELETE\b",
    r"\bDETACH\b",
    r"\bSET\b",
    r"\bREMOVE\b",
    r"\bDROP\b",
    r"\bCALL\s+db\.",
    r"\bCALL\s+dbms\.",
    r"\bCALL\s+apoc\.create",
    r"\bCALL\s+apoc\.refactor",
]


def is_read_only_query(query: str) -> bool:
    """Check if a Cypher query is read-only."""
    query_upper = query.upper()
    for pattern in WRITE_PATTERNS:
        if re.search(pattern, query_upper):
            return False
    return True


# =============================================================================
# STATS ENDPOINT
# =============================================================================


@router.get("/stats")
async def get_graph_stats(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """
    Get comprehensive graph statistics.

    Returns:
        - Node counts by label
        - Relationship counts by type
        - Total nodes and relationships
        - Graph connectivity metrics
    """


    try:
        neo4j = get_neo4j_client()

        # Get node counts by label
        node_counts = neo4j.get_node_counts()

        # Get relationship counts by type
        rel_counts = neo4j.get_relationship_counts()

        # Get total counts
        total_nodes = sum(node_counts.values())
        total_relationships = sum(rel_counts.values())

        # Get connectivity metrics
        avg_degree = 0.0
        if total_nodes > 0:
            avg_degree = (total_relationships * 2) / total_nodes  # Approximate

        return {
            "total_nodes": total_nodes,
            "total_relationships": total_relationships,
            "node_counts": node_counts,
            "relationship_counts": rel_counts,
            "average_degree": round(avg_degree, 2),
            "health": neo4j.health_check(),
        }

    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        return {
            "total_nodes": 0,
            "total_relationships": 0,
            "node_counts": {},
            "relationship_counts": {},
            "average_degree": 0,
            "health": {"status": "error", "error": str(e)},
        }


# =============================================================================
# ENTITY ENDPOINTS
# =============================================================================


@router.get("/entities")
async def get_entities(
    entity_type: Optional[str] = Query(None, description="Filter by node label (e.g., Standard, TestMethod)"),
    search: Optional[str] = Query(None, description="Search term for name/code"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Get paginated list of entities with optional filtering.

    Returns entities with their labels, properties, and relationship counts.
    """


    try:
        neo4j = get_neo4j_client()

        # Build query based on filters
        label_filter = f":{entity_type}" if entity_type else ""

        if search:
            # Use full-text search if available, fallback to CONTAINS
            query = f"""
            MATCH (n{label_filter})
            WHERE n.name CONTAINS $search
               OR n.code CONTAINS $search
               OR n.abbreviation CONTAINS $search
            WITH n
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) as rel_count
            RETURN n, labels(n) as labels, rel_count
            ORDER BY rel_count DESC, n.name
            SKIP $offset LIMIT $limit
            """
            params = {"search": search, "offset": offset, "limit": limit}
        else:
            query = f"""
            MATCH (n{label_filter})
            WITH n
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) as rel_count
            RETURN n, labels(n) as labels, rel_count
            ORDER BY rel_count DESC, n.name
            SKIP $offset LIMIT $limit
            """
            params = {"offset": offset, "limit": limit}

        results = neo4j.run_query(query, params)

        # Get total count for pagination
        count_query = f"""
        MATCH (n{label_filter})
        {"WHERE n.name CONTAINS $search OR n.code CONTAINS $search OR n.abbreviation CONTAINS $search" if search else ""}
        RETURN count(n) as total
        """
        count_params = {"search": search} if search else {}
        count_result = neo4j.run_query(count_query, count_params)
        total = count_result[0]["total"] if count_result else 0

        entities = []
        for record in results:
            node = record["n"]
            entities.append({
                "id": node.get("name") or node.get("code") or node.get("chunk_id"),
                "name": node.get("name") or node.get("code", "Unknown"),
                "labels": record["labels"],
                "properties": _sanitize_neo4j(_to_dict(node)),
                "relationship_count": record["rel_count"],
            })

        return {
            "entities": entities,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
        }

    except Exception as e:
        logger.error(f"Failed to get entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entity/{name}")
async def get_entity_details(
    name: str,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Get detailed information about a single entity.

    Returns the entity's properties and all its relationships.
    """


    try:
        neo4j = get_neo4j_client()

        # Get entity and its relationships
        query = """
        MATCH (n)
        WHERE n.name = $name OR n.code = $name OR n.chunk_id = $name
        OPTIONAL MATCH (n)-[r]-(connected)
        RETURN n, labels(n) as labels,
               collect(DISTINCT {
                   type: type(r),
                   direction: CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END,
                   target_name: connected.name,
                   target_code: connected.code,
                   target_labels: labels(connected),
                   properties: properties(r)
               }) as relationships
        """

        results = neo4j.run_query(query, {"name": name})

        if not results:
            raise HTTPException(status_code=404, detail=f"Entity not found: {name}")

        record = results[0]
        node = record["n"]

        # Filter out null relationships
        relationships = [r for r in record["relationships"] if r["type"] is not None]

        return {
            "name": node.get("name") or node.get("code") or name,
            "labels": record["labels"],
            "properties": _sanitize_neo4j(_to_dict(node)),
            "relationships": _sanitize_neo4j(relationships),
            "relationship_count": len(relationships),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get entity details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# VISUALIZATION ENDPOINT
# =============================================================================


@router.get("/visualization")
async def get_visualization_data(
    entity_type: Optional[str] = Query(None, description="Filter by node label"),
    limit: int = Query(100, ge=1, le=500, description="Max nodes to return"),
    include_chunks: bool = Query(False, description="Include Chunk nodes (can be many)"),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Get graph data formatted for visualization (nodes and edges).

    Returns data suitable for force-directed graph rendering.
    """


    try:
        neo4j = get_neo4j_client()

        # Build query to get nodes and relationships
        label_filter = f":{entity_type}" if entity_type else ""

        if include_chunks:
            query = f"""
            MATCH (n{label_filter})
            WITH n LIMIT $limit
            OPTIONAL MATCH (n)-[r]-(m)
            WHERE id(m) < id(n) OR NOT (m)--(n)
            RETURN DISTINCT n, labels(n) as n_labels, r, m, labels(m) as m_labels
            """
        else:
            query = f"""
            MATCH (n{label_filter})
            WHERE NOT 'Chunk' IN labels(n)
            WITH n LIMIT $limit
            OPTIONAL MATCH (n)-[r]-(m)
            WHERE NOT 'Chunk' IN labels(m)
            RETURN DISTINCT n, labels(n) as n_labels, r, m, labels(m) as m_labels
            """

        results = neo4j.run_query(query, {"limit": limit})

        # Build nodes and edges
        nodes_map = {}
        edges = []

        for record in results:
            n = record["n"]
            n_labels = record["n_labels"]
            n_id = n.get("name") or n.get("code") or n.get("chunk_id") or str(id(n))

            if n_id not in nodes_map:
                nodes_map[n_id] = {
                    "id": n_id,
                    "label": n.get("name") or n.get("code") or n_id,
                    "type": n_labels[0] if n_labels else "Unknown",
                    "properties": _sanitize_neo4j(_to_dict(n)),
                }

            if record["m"] is not None and record["r"] is not None:
                m = record["m"]
                m_labels = record["m_labels"]
                m_id = m.get("name") or m.get("code") or m.get("chunk_id") or str(id(m))

                if m_id not in nodes_map:
                    nodes_map[m_id] = {
                        "id": m_id,
                        "label": m.get("name") or m.get("code") or m_id,
                        "type": m_labels[0] if m_labels else "Unknown",
                        "properties": _sanitize_neo4j(_to_dict(m)),
                    }

                r = record["r"]
                # Neo4j Relationship has .type for the rel type string
                rel_type = getattr(r, "type", None) or type(r).__name__
                edge_id = f"{n_id}-{rel_type}-{m_id}"
                edges.append({
                    "id": edge_id,
                    "source": n_id,
                    "target": m_id,
                    "type": rel_type,
                    "properties": _sanitize_neo4j(_to_dict(r)) if r else {},
                })

        return {
            "nodes": list(nodes_map.values()),
            "edges": edges,
            "node_count": len(nodes_map),
            "edge_count": len(edges),
        }

    except Exception as e:
        logger.error(f"Failed to get visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/neighbors/{name}")
async def get_entity_neighbors(
    name: str,
    depth: int = Query(1, ge=1, le=3, description="Traversal depth"),
    limit: int = Query(50, ge=1, le=200),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Get neighbors of an entity for interactive graph expansion.

    Useful for "click to expand" functionality in visualizations.
    """


    try:
        neo4j = get_neo4j_client()

        # Variable length path up to specified depth
        query = f"""
        MATCH (start)
        WHERE start.name = $name OR start.code = $name OR start.chunk_id = $name
        MATCH path = (start)-[*1..{depth}]-(neighbor)
        WHERE NOT 'Chunk' IN labels(neighbor) OR neighbor = start
        WITH DISTINCT neighbor, labels(neighbor) as labels
        LIMIT $limit
        OPTIONAL MATCH (neighbor)-[r]-()
        RETURN neighbor, labels, count(r) as rel_count
        """

        results = neo4j.run_query(query, {"name": name, "limit": limit})

        neighbors = []
        for record in results:
            node = record["neighbor"]
            neighbors.append({
                "id": node.get("name") or node.get("code") or node.get("chunk_id"),
                "name": node.get("name") or node.get("code", "Unknown"),
                "labels": record["labels"],
                "properties": _sanitize_neo4j(_to_dict(node)),
                "relationship_count": record["rel_count"],
            })

        return {
            "source": name,
            "depth": depth,
            "neighbors": neighbors,
            "count": len(neighbors),
        }

    except Exception as e:
        logger.error(f"Failed to get neighbors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RELATIONSHIPS ENDPOINT
# =============================================================================


@router.get("/relationships")
async def get_relationships(
    rel_type: Optional[str] = Query(None, description="Filter by relationship type"),
    source_type: Optional[str] = Query(None, description="Filter by source node label"),
    target_type: Optional[str] = Query(None, description="Filter by target node label"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Browse relationships with optional filtering.

    Returns paginated list of relationships with source and target info.
    """


    try:
        neo4j = get_neo4j_client()

        # Build filters
        source_filter = f":{source_type}" if source_type else ""
        target_filter = f":{target_type}" if target_type else ""
        rel_filter = f":{rel_type}" if rel_type else ""

        query = f"""
        MATCH (source{source_filter})-[r{rel_filter}]->(target{target_filter})
        RETURN source.name as source_name, source.code as source_code,
               labels(source) as source_labels,
               type(r) as rel_type, properties(r) as rel_props,
               target.name as target_name, target.code as target_code,
               labels(target) as target_labels
        SKIP $offset LIMIT $limit
        """

        results = neo4j.run_query(query, {"offset": offset, "limit": limit})

        # Get total count
        count_query = f"""
        MATCH (source{source_filter})-[r{rel_filter}]->(target{target_filter})
        RETURN count(r) as total
        """
        count_result = neo4j.run_query(count_query)
        total = count_result[0]["total"] if count_result else 0

        relationships = []
        for record in results:
            relationships.append({
                "source": record["source_name"] or record["source_code"] or "Unknown",
                "source_labels": record["source_labels"],
                "relationship": record["rel_type"],
                "properties": _sanitize_neo4j(record["rel_props"]) or {},
                "target": record["target_name"] or record["target_code"] or "Unknown",
                "target_labels": record["target_labels"],
            })

        return {
            "relationships": relationships,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
        }

    except Exception as e:
        logger.error(f"Failed to get relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CYPHER QUERY ENDPOINT
# =============================================================================


@router.post("/query")
async def run_cypher_query(
    request: CypherQueryRequest,
    allow_writes: bool = Query(False, description="Allow write operations (dangerous!)"),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Execute a custom Cypher query.

    Security restrictions:
    - Read-only by default (block CREATE, MERGE, DELETE, SET, DROP)
    - 5 second timeout
    - Max 1000 result rows
    - All executions are logged
    """


    query = request.query.strip()
    parameters = request.parameters or {}

    # Security check
    if not allow_writes and not is_read_only_query(query):
        logger.warning(f"Blocked write query attempt: {query[:100]}")
        raise HTTPException(
            status_code=403,
            detail="Write operations are not allowed. Use allow_writes=true to enable (dangerous!)",
        )

    # Log query execution
    logger.info(f"Admin Cypher query: {query[:200]}... (params: {list(parameters.keys())})")

    try:
        neo4j = get_neo4j_client()
        start_time = time.time()

        # Execute query with row limit
        limited_query = f"{query}\nLIMIT 1000" if "LIMIT" not in query.upper() else query

        if allow_writes:
            result = neo4j.run_write_query(limited_query, parameters)
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "success": True,
                "type": "write",
                "summary": result,
                "elapsed_ms": elapsed_ms,
            }
        else:
            results = neo4j.run_query(limited_query, parameters)
            elapsed_ms = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "type": "read",
                "rows": _sanitize_neo4j(results),
                "row_count": len(results),
                "elapsed_ms": elapsed_ms,
            }

    except Exception as e:
        logger.error(f"Cypher query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RELATIONSHIP TYPES ENDPOINT
# =============================================================================


@router.get("/relationship-types")
async def get_relationship_types(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Get all relationship types in the graph."""


    try:
        neo4j = get_neo4j_client()
        rel_counts = neo4j.get_relationship_counts()

        types = [{"type": t, "count": c} for t, c in sorted(rel_counts.items(), key=lambda x: -x[1])]

        return {
            "types": types,
            "total_types": len(types),
        }

    except Exception as e:
        logger.error(f"Failed to get relationship types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ENTITY LABELS ENDPOINT
# =============================================================================


@router.get("/labels")
async def get_entity_labels(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """Get all node labels in the graph."""


    try:
        neo4j = get_neo4j_client()
        node_counts = neo4j.get_node_counts()

        labels = [{"label": l, "count": c} for l, c in sorted(node_counts.items(), key=lambda x: -x[1]) if c > 0]

        return {
            "labels": labels,
            "total_labels": len(labels),
        }

    except Exception as e:
        logger.error(f"Failed to get labels: {e}")
        raise HTTPException(status_code=500, detail=str(e))
