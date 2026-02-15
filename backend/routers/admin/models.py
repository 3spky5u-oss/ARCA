"""
Pydantic models for admin API requests.
"""

from typing import Optional

from pydantic import BaseModel


class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str
    module: str


class ConfigUpdate(BaseModel):
    """Configuration update request."""

    # Context window sizes
    ctx_small: Optional[int] = None
    ctx_medium: Optional[int] = None
    ctx_large: Optional[int] = None
    ctx_xlarge: Optional[int] = None
    # Model parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    kv_cache_type_k: Optional[str] = None
    kv_cache_type_v: Optional[str] = None
    # Model names (hot-swappable)
    model_chat: Optional[str] = None
    model_code: Optional[str] = None
    model_expert: Optional[str] = None
    model_vision: Optional[str] = None
    model_vision_heavy: Optional[str] = None
    model_vision_structured: Optional[str] = None
    model_chat_finetuned: Optional[str] = None
    # Vision settings
    vision_num_ctx: Optional[int] = None
    vision_timeout: Optional[int] = None
    vision_max_workers: Optional[int] = None
    # RAG settings
    rag_top_k: Optional[int] = None
    rag_min_score: Optional[float] = None
    rag_min_final_score: Optional[float] = None
    # RAG diversity
    rag_diversity_enabled: Optional[bool] = None
    rag_diversity_lambda: Optional[float] = None
    rag_max_per_source: Optional[int] = None
    # Phase 1 - BM25 + Dense
    bm25_enabled: Optional[bool] = None
    bm25_weight: Optional[float] = None
    query_expansion_enabled: Optional[bool] = None
    # Phase 2 - HyDE, CRAG
    hyde_enabled: Optional[bool] = None
    hyde_model: Optional[str] = None
    crag_enabled: Optional[bool] = None
    crag_min_confidence: Optional[float] = None
    crag_web_search_on_low: Optional[bool] = None
    searxng_enabled: Optional[bool] = None
    searxng_url: Optional[str] = None
    searxng_categories: Optional[str] = None
    searxng_language: Optional[str] = None
    searxng_timeout_s: Optional[float] = None
    searxng_max_results: Optional[int] = None
    searxng_request_format: Optional[str] = None
    # Streaming
    stream_token_delay_ms: Optional[int] = None
    # Topic and reranker settings
    enabled_topics: Optional[str] = None
    reranker_enabled: Optional[bool] = None
    reranker_candidates: Optional[int] = None
    reranker_batch_size: Optional[int] = None
    reranker_device: Optional[str] = None
    router_use_semantic: Optional[bool] = None
    # Domain boost
    domain_boost_enabled: Optional[bool] = None
    domain_boost_factor: Optional[float] = None
    # Tool router settings
    tool_router_enabled: Optional[bool] = None
    tool_router_timeout: Optional[float] = None
    tool_router_min_confidence: Optional[float] = None
    # RAPTOR hierarchical search
    raptor_enabled: Optional[bool] = None
    raptor_max_levels: Optional[int] = None
    raptor_cluster_size: Optional[int] = None
    raptor_summary_model: Optional[str] = None
    raptor_retrieval_strategy: Optional[str] = None
    # Vision Ingest
    vision_ingest_enabled: Optional[bool] = None
    # GraphRAG
    graph_rag_enabled: Optional[bool] = None
    graph_rag_auto: Optional[bool] = None
    graph_rag_weight: Optional[float] = None
    graph_max_hops: Optional[int] = None
    # Community/Global Search
    community_detection_enabled: Optional[bool] = None
    global_search_enabled: Optional[bool] = None
    global_search_top_k: Optional[int] = None
    community_level_default: Optional[str] = None
    community_summary_model: Optional[str] = None
    # LLM timeouts
    llm_timeout: Optional[int] = None
    llm_timeout_think: Optional[int] = None
    max_output_tokens: Optional[int] = None
    # Phii behavior
    phii_energy_matching: Optional[bool] = None
    phii_specialty_detection: Optional[bool] = None
    phii_implicit_feedback: Optional[bool] = None
    # Rate limiting
    rate_limit_ws_conn: Optional[int] = None
    rate_limit_ws_msg: Optional[int] = None
    rate_limit_upload: Optional[int] = None
    # Ingest lock
    ingest_lock_enabled: Optional[bool] = None
    # MCP mode
    mcp_mode: Optional[bool] = None


class ModelPullRequest(BaseModel):
    """Request to pull a model."""

    name: str  # Model name like "qwen3:32b" or "llama3:8b"


class ExtractionTestRequest(BaseModel):
    """Request for extraction test."""

    extractor: Optional[str] = None  # None = auto-select via scout


class LoginRequest(BaseModel):
    """Login request."""
    username: str
    password: str


class RegisterRequest(BaseModel):
    """User registration request."""
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    """Password change request."""
    old_password: str
    new_password: str


class CreateUserRequest(BaseModel):
    """Admin create-user request."""
    username: str
    password: str


class UserSettingsUpdate(BaseModel):
    """User settings update request."""
    theme: Optional[str] = None
    display_name: Optional[str] = None
    phii_level_override: Optional[str] = None


class DomainActivateRequest(BaseModel):
    """Request to switch active domain pack."""
    domain: str


class ModelAssignRequest(BaseModel):
    """Request to assign a model to a slot."""
    slot: str  # e.g. "chat", "vision", "code"
    model: str  # GGUF filename
