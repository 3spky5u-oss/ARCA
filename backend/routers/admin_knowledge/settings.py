import logging
from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException, Query

from services.admin_auth import verify_admin
from . import router

logger = logging.getLogger(__name__)


@router.get("/profiles")
async def get_profiles(_admin=Depends(verify_admin)):
    """List all retrieval profiles with active status and manual overrides."""
    from profile_loader import get_profile_manager
    pm = get_profile_manager()
    return {
        "profiles": pm.list_profiles(),
        "active_profile": pm.active_profile,
        "manual_overrides": pm.manual_overrides,
        "has_manual_overrides": pm.has_manual_overrides,
    }


@router.put("/profiles/active")
async def set_active_profile(
    profile: str = Query(..., description="Profile name to activate"),
    _admin=Depends(verify_admin),
):
    """Switch active retrieval profile. Clears manual overrides and applies toggle values."""
    from config import runtime_config
    try:
        result = runtime_config.apply_profile(profile)
        return {"success": True, **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/reranker")
async def get_reranker_settings(_: bool = Depends(verify_admin)) -> Dict[str, Any]:
    """
    Get current reranker, diversity, and retrieval settings.
    """


    from config import runtime_config

    # Profile state
    from profile_loader import get_profile_manager
    pm = get_profile_manager()

    return {
        # Core retrieval
        "reranker_enabled": runtime_config.reranker_enabled,
        "reranker_candidates": runtime_config.reranker_candidates,
        "reranker_batch_size": runtime_config.reranker_batch_size,
        "router_use_semantic": runtime_config.router_use_semantic,
        "rag_top_k": runtime_config.rag_top_k,
        "rag_min_score": runtime_config.rag_min_score,
        "rag_min_final_score": runtime_config.rag_min_final_score,
        # BM25 & Diversity
        "bm25_enabled": runtime_config.bm25_enabled,
        "bm25_weight": runtime_config.bm25_weight,
        "rag_diversity_enabled": runtime_config.rag_diversity_enabled,
        "rag_diversity_lambda": runtime_config.rag_diversity_lambda,
        "rag_max_per_source": runtime_config.rag_max_per_source,
        "domain_boost_enabled": runtime_config.domain_boost_enabled,
        "domain_boost_factor": runtime_config.domain_boost_factor,
        # Advanced retrieval
        "query_expansion_enabled": runtime_config.query_expansion_enabled,
        "hyde_enabled": runtime_config.hyde_enabled,
        "crag_enabled": runtime_config.crag_enabled,
        "crag_min_confidence": runtime_config.crag_min_confidence,
        "crag_web_search_on_low": runtime_config.crag_web_search_on_low,
        # RAPTOR
        "raptor_enabled": runtime_config.raptor_enabled,
        "raptor_max_levels": runtime_config.raptor_max_levels,
        "raptor_cluster_size": runtime_config.raptor_cluster_size,
        "raptor_summary_model": runtime_config.raptor_summary_model,
        "raptor_retrieval_strategy": runtime_config.raptor_retrieval_strategy,
        # GraphRAG
        "graph_rag_enabled": runtime_config.graph_rag_enabled,
        "graph_rag_auto": runtime_config.graph_rag_auto,
        "graph_rag_weight": runtime_config.graph_rag_weight,
        "graph_max_hops": runtime_config.graph_max_hops,
        # Global Search / Community
        "community_detection_enabled": runtime_config.community_detection_enabled,
        "global_search_enabled": runtime_config.global_search_enabled,
        "global_search_top_k": runtime_config.global_search_top_k,
        "community_level_default": runtime_config.community_level_default,
        # Profile state
        "retrieval_profile": runtime_config.retrieval_profile,
        "vision_ingest_enabled": runtime_config.vision_ingest_enabled,
        "manual_overrides": pm.manual_overrides,
        "has_manual_overrides": pm.has_manual_overrides,
    }


@router.put("/reranker")
async def update_reranker_settings(
    # Core retrieval
    reranker_enabled: Optional[bool] = Query(None),
    reranker_candidates: Optional[int] = Query(None),
    reranker_batch_size: Optional[int] = Query(None),
    router_use_semantic: Optional[bool] = Query(None),
    rag_top_k: Optional[int] = Query(None),
    rag_min_score: Optional[float] = Query(None),
    rag_min_final_score: Optional[float] = Query(None),
    # BM25 & Diversity
    bm25_enabled: Optional[bool] = Query(None),
    bm25_weight: Optional[float] = Query(None),
    rag_diversity_enabled: Optional[bool] = Query(None),
    rag_diversity_lambda: Optional[float] = Query(None),
    rag_max_per_source: Optional[int] = Query(None),
    domain_boost_enabled: Optional[bool] = Query(None),
    domain_boost_factor: Optional[float] = Query(None),
    # Advanced retrieval
    query_expansion_enabled: Optional[bool] = Query(None),
    hyde_enabled: Optional[bool] = Query(None),
    crag_enabled: Optional[bool] = Query(None),
    crag_min_confidence: Optional[float] = Query(None),
    crag_web_search_on_low: Optional[bool] = Query(None),
    # RAPTOR
    raptor_enabled: Optional[bool] = Query(None),
    raptor_max_levels: Optional[int] = Query(None),
    raptor_cluster_size: Optional[int] = Query(None),
    raptor_summary_model: Optional[str] = Query(None),
    raptor_retrieval_strategy: Optional[str] = Query(None),
    # GraphRAG
    graph_rag_enabled: Optional[bool] = Query(None),
    graph_rag_auto: Optional[bool] = Query(None),
    graph_rag_weight: Optional[float] = Query(None),
    graph_max_hops: Optional[int] = Query(None),
    # Global Search / Community
    community_detection_enabled: Optional[bool] = Query(None),
    global_search_enabled: Optional[bool] = Query(None),
    global_search_top_k: Optional[int] = Query(None),
    community_level_default: Optional[str] = Query(None),
    # Profile & Vision
    retrieval_profile: Optional[str] = Query(None),
    vision_ingest_enabled: Optional[bool] = Query(None),
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """
    Update reranker, diversity, and retrieval settings.
    """

    from config import runtime_config

    # Apply profile if specified (clears manual overrides first)
    if retrieval_profile is not None:
        try:
            runtime_config.apply_profile(retrieval_profile)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Collect all provided params into a dict for generic update
    params = {
        "reranker_enabled": reranker_enabled,
        "reranker_candidates": reranker_candidates,
        "reranker_batch_size": reranker_batch_size,
        "router_use_semantic": router_use_semantic,
        "rag_top_k": rag_top_k,
        "rag_min_score": rag_min_score,
        "rag_min_final_score": rag_min_final_score,
        "bm25_enabled": bm25_enabled,
        "bm25_weight": bm25_weight,
        "rag_diversity_enabled": rag_diversity_enabled,
        "rag_diversity_lambda": rag_diversity_lambda,
        "rag_max_per_source": rag_max_per_source,
        "domain_boost_enabled": domain_boost_enabled,
        "domain_boost_factor": domain_boost_factor,
        "query_expansion_enabled": query_expansion_enabled,
        "hyde_enabled": hyde_enabled,
        "crag_enabled": crag_enabled,
        "crag_min_confidence": crag_min_confidence,
        "crag_web_search_on_low": crag_web_search_on_low,
        "raptor_enabled": raptor_enabled,
        "raptor_max_levels": raptor_max_levels,
        "raptor_cluster_size": raptor_cluster_size,
        "raptor_summary_model": raptor_summary_model,
        "raptor_retrieval_strategy": raptor_retrieval_strategy,
        "graph_rag_enabled": graph_rag_enabled,
        "graph_rag_auto": graph_rag_auto,
        "graph_rag_weight": graph_rag_weight,
        "graph_max_hops": graph_max_hops,
        "vision_ingest_enabled": vision_ingest_enabled,
        "community_detection_enabled": community_detection_enabled,
        "global_search_enabled": global_search_enabled,
        "global_search_top_k": global_search_top_k,
        "community_level_default": community_level_default,
    }

    updated = []
    for key, value in params.items():
        if value is not None and hasattr(runtime_config, key):
            setattr(runtime_config, key, value)
            updated.append(key)

    # Persist so settings survive container restarts
    if updated:
        runtime_config.save_overrides()

    return {
        "success": True,
        "updated": updated,
        "settings": {key: getattr(runtime_config, key) for key in params if hasattr(runtime_config, key)},
    }
