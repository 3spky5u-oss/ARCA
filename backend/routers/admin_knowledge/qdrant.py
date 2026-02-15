import logging
from typing import Dict, Any

import httpx
from fastapi import Depends, HTTPException, Query

from services.admin_auth import verify_admin
from . import router, QDRANT_URL

logger = logging.getLogger(__name__)


@router.get("/collections")
async def list_qdrant_collections(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """
    List all Qdrant collections with vector counts and dimensions.

    Returns collection name, vector count, dimensions, and disk size.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{QDRANT_URL}/collections")
            resp.raise_for_status()
            data = resp.json()

        collections = []
        for col in data.get("result", {}).get("collections", []):
            name = col.get("name", "unknown")
            # Fetch details for each collection
            try:
                detail_resp = await httpx.AsyncClient(timeout=10.0).__aenter__()
                async with httpx.AsyncClient(timeout=10.0) as detail_client:
                    detail = await detail_client.get(f"{QDRANT_URL}/collections/{name}")
                    detail.raise_for_status()
                    detail_data = detail.json().get("result", {})

                vectors_count = detail_data.get("vectors_count", 0)
                points_count = detail_data.get("points_count", 0)

                # Get dimension from config
                dimension = None
                config = detail_data.get("config", {}).get("params", {}).get("vectors", {})
                if isinstance(config, dict):
                    # Could be named vectors or single vector
                    if "size" in config:
                        dimension = config["size"]
                    else:
                        # Named vectors — get first one
                        for vec_name, vec_config in config.items():
                            if isinstance(vec_config, dict) and "size" in vec_config:
                                dimension = vec_config["size"]
                                break

                collections.append({
                    "name": name,
                    "vectors_count": vectors_count,
                    "points_count": points_count,
                    "dimension": dimension,
                })
            except Exception as e:
                collections.append({
                    "name": name,
                    "vectors_count": 0,
                    "points_count": 0,
                    "dimension": None,
                    "error": str(e),
                })

        return {
            "success": True,
            "collections": collections,
            "total_collections": len(collections),
        }

    except Exception as e:
        logger.error(f"Failed to list Qdrant collections: {e}")
        raise HTTPException(status_code=500, detail=f"Qdrant error: {e}")


@router.get("/collections/{name}")
async def get_collection_details(
    name: str,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Get detailed info about a specific Qdrant collection."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{QDRANT_URL}/collections/{name}")
            resp.raise_for_status()
            data = resp.json().get("result", {})

        return {
            "success": True,
            "name": name,
            "vectors_count": data.get("vectors_count", 0),
            "points_count": data.get("points_count", 0),
            "status": data.get("status", "unknown"),
            "config": data.get("config", {}),
            "segments_count": data.get("segments_count", 0),
        }

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Collection not found: {name}")
        raise HTTPException(status_code=500, detail=f"Qdrant error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant error: {e}")


@router.delete("/collections/{name}")
async def delete_collection(
    name: str,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Delete a Qdrant collection. This action cannot be undone."""
    # Protect main cohesionn collection from deletion (contains all knowledge + arca_core)
    if name == "cohesionn":
        raise HTTPException(
            status_code=403,
            detail="Cannot delete the main knowledge collection. Use topic-level deletion instead.",
        )
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(f"{QDRANT_URL}/collections/{name}")
            resp.raise_for_status()

        logger.info(f"Deleted Qdrant collection: {name}")
        return {"success": True, "deleted": name}

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Collection not found: {name}")
        raise HTTPException(status_code=500, detail=f"Qdrant error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant error: {e}")


@router.post("/collections/{name}/search")
async def search_collection(
    name: str,
    query: str = Query(..., description="Search query text"),
    limit: int = Query(5, description="Number of results"),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Test search against a specific Qdrant collection.

    Embeds the query and finds nearest neighbors.
    """
    try:
        from tools.cohesionn.embeddings import get_embedder

        embedder = get_embedder()
        query_vector = embedder.embed([query])[0]

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{QDRANT_URL}/collections/{name}/points/search",
                json={
                    "vector": query_vector,
                    "limit": limit,
                    "with_payload": True,
                },
            )
            resp.raise_for_status()
            data = resp.json().get("result", [])

        results = []
        for point in data:
            payload = point.get("payload", {})
            results.append({
                "id": point.get("id"),
                "score": round(point.get("score", 0), 4),
                "content": (payload.get("content", ""))[:300],
                "source": payload.get("source", ""),
                "topic": payload.get("topic", ""),
            })

        return {
            "success": True,
            "query": query,
            "collection": name,
            "results": results,
            "count": len(results),
        }

    except Exception as e:
        logger.error(f"Collection search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {e}")
