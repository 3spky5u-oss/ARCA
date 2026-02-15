"""
Runtime Configuration for ARCA.

Provides a singleton RuntimeConfig class that allows dynamic adjustment of
model parameters at runtime, without requiring service restart.

Usage:
    from config import runtime_config
    ctx_size = runtime_config.ctx_medium
    runtime_config.update(ctx_medium=12288, temperature=0.8)
"""

import json
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
from threading import Lock
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


def _get_default_topics() -> str:
    """Get default topics from domain lexicon."""
    try:
        from domain_loader import get_lexicon
        lexicon = get_lexicon()
        topics = lexicon.get("topics", ["general"])
        return ",".join(topics)
    except Exception:
        return "general"


# Hardware-aware context window defaults per VRAM tier
CONTEXT_DEFAULTS = {
    "cpu":    {"small": 512,  "medium": 2048,  "large": 2048,  "xlarge": 4096},
    "small":  {"small": 1024, "medium": 4096,  "large": 4096,  "xlarge": 8192},
    "medium": {"small": 2048, "medium": 8192,  "large": 8192,  "xlarge": 16384},
    "large":  {"small": 2048, "medium": 8192,  "large": 16384, "xlarge": 24576},
}


def _hw_ctx_default(env_key: str, tier_key: str, fallback: int) -> int:
    """Get context default: env var > hardware profile > hardcoded fallback."""
    env_val = os.environ.get(env_key)
    if env_val:
        return int(env_val)
    try:
        from services.hardware import get_hardware_info
        hw = get_hardware_info()
        return CONTEXT_DEFAULTS.get(hw.profile, {}).get(tier_key, fallback)
    except Exception:
        return fallback


def _first_env(*keys: str, default: str) -> str:
    """Return the first non-empty environment value from keys, else default."""
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return default


def _build_database_url_default() -> str:
    """
    Build a PostgreSQL URL from env vars when DATABASE_URL is not explicitly set.

    Password is URL-encoded to avoid auth breakage with special characters.
    """
    explicit = os.environ.get("DATABASE_URL", "").strip()
    if explicit:
        return explicit

    user = os.environ.get("POSTGRES_USER", "arca").strip() or "arca"
    password = os.environ.get("POSTGRES_PASSWORD", "arca-local-dev")
    host = os.environ.get("POSTGRES_HOST", "localhost").strip() or "localhost"
    port = os.environ.get("POSTGRES_PORT", "5432").strip() or "5432"
    db = os.environ.get("POSTGRES_DB", "arca").strip() or "arca"

    safe_password = quote_plus(password)
    return f"postgresql://{user}:{safe_password}@{host}:{port}/{db}"


@dataclass
class RuntimeConfig:
    """
    Singleton configuration for runtime-adjustable parameters.

    All values have defaults from environment variables, but can be
    changed at runtime via the update() method.
    """

    # Context window sizes (tokens)
    ctx_small: int = field(default_factory=lambda: _hw_ctx_default("LLM_CTX_SMALL", "small", 2048))
    ctx_medium: int = field(default_factory=lambda: _hw_ctx_default("LLM_CTX_MEDIUM", "medium", 8192))
    ctx_large: int = field(default_factory=lambda: _hw_ctx_default("LLM_CTX_LARGE", "large", 16384))
    ctx_xlarge: int = field(default_factory=lambda: _hw_ctx_default("LLM_CTX_XLARGE", "xlarge", 24576))

    # KV cache quantization
    kv_cache_type_k: str = field(default_factory=lambda: os.environ.get("LLM_CACHE_TYPE_K", "q8_0"))
    kv_cache_type_v: str = field(default_factory=lambda: os.environ.get("LLM_CACHE_TYPE_V", "q8_0"))

    # Model parameters
    temperature: float = field(default_factory=lambda: float(os.environ.get("LLM_TEMPERATURE", "0.7")))
    top_p: float = field(default_factory=lambda: float(os.environ.get("LLM_TOP_P", "0.9")))
    top_k: int = field(default_factory=lambda: int(os.environ.get("LLM_TOP_K", "40")))

    # Model names (can be hot-swapped)
    model_chat: str = field(
        default_factory=lambda: _first_env(
            "LLM_CHAT_MODEL",
            default="Qwen3-30B-A3B-Q4_K_M.gguf",
        )
    )
    model_chat_finetuned: str = field(
        default_factory=lambda: os.environ.get("LLM_FINETUNED_MODEL", "")
    )
    model_code: str = field(
        default_factory=lambda: _first_env(
            "LLM_CODE_MODEL",
            "LLM_CHAT_MODEL",
            "LLM_EXPERT_MODEL",
            default="Qwen3-30B-A3B-Q4_K_M.gguf",
        )
    )
    model_expert: str = field(
        default_factory=lambda: _first_env(
            "LLM_EXPERT_MODEL",
            "LLM_CHAT_MODEL",
            default="Qwen3-30B-A3B-Q4_K_M.gguf",
        )
    )
    model_vision: str = field(
        default_factory=lambda: _first_env(
            "LLM_VISION_MODEL",
            "VISION_OCR_MODEL",  # Backward compatibility
            default="Qwen3VL-8B-Instruct-Q8_0.gguf",
        )
    )  # OCR (Logg, Observationn) - 8B fits in VRAM
    model_vision_heavy: str = field(
        default_factory=lambda: _first_env(
            "LLM_VISION_HEAVY_MODEL",
            "VISION_HEAVY_MODEL",  # Backward compatibility
            "LLM_VISION_MODEL",
            default="Qwen3VL-8B-Instruct-Q8_0.gguf",
        )
    )  # Heavy vision for critical accuracy (needs chat model unloaded)

    # Vision extraction settings
    vision_num_ctx: int = field(
        default_factory=lambda: int(os.environ.get("VISION_NUM_CTX", "2048"))
    )  # Context window for vision (2048 for speed, 8192 for complex pages)
    vision_timeout: int = field(
        default_factory=lambda: int(os.environ.get("VISION_TIMEOUT", "60"))
    )  # Max seconds per page extraction
    vision_max_workers: int = field(
        default_factory=lambda: int(os.environ.get("VISION_MAX_WORKERS", "2"))
    )  # Parallel vision extractions for BATCH INGESTION only
    # NOTE: Chat agent remains single-request - this only affects HybridExtractor
    model_vision_structured: str = field(
        default_factory=lambda: _first_env(
            "LLM_VISION_STRUCTURED_MODEL",
            "VISION_STRUCTURED_MODEL",  # Backward compatibility
            "LLM_VISION_MODEL",
            default="Qwen3VL-8B-Instruct-Q8_0.gguf",
        )
    )  # Structured extraction (LoggView, Mapperr)

    # RAG settings
    rag_top_k: int = field(default_factory=lambda: int(os.environ.get("RAG_TOP_K", "5")))
    rag_min_score: float = field(
        default_factory=lambda: float(os.environ.get("RAG_MIN_SCORE", "0.15"))
    )  # Stricter pre-filter
    rag_min_final_score: float = field(
        default_factory=lambda: float(os.environ.get("RAG_MIN_FINAL_SCORE", "0.20"))
    )  # Post-rerank quality gate

    # RAG diversity settings
    rag_diversity_enabled: bool = field(
        default_factory=lambda: os.environ.get("RAG_DIVERSITY_ENABLED", "true").lower() == "true"
    )
    rag_diversity_lambda: float = field(
        default_factory=lambda: float(os.environ.get("RAG_DIVERSITY_LAMBDA", "0.4"))
    )  # 0=diversity, 1=relevance
    rag_max_per_source: int = field(default_factory=lambda: int(os.environ.get("RAG_MAX_PER_SOURCE", "2")))

    # Phase 1 - Hybrid Retrieval (BM25 + Dense)
    bm25_enabled: bool = field(
        default_factory=lambda: os.environ.get("BM25_ENABLED", "false").lower() == "true"
    )  # Benchmark v2 validated: BM25 harmful in v1+v2, dense-only wins
    bm25_weight: float = field(
        default_factory=lambda: float(os.environ.get("BM25_WEIGHT", "0.5"))
    )  # 50% BM25, 50% dense in RRF
    query_expansion_enabled: bool = field(
        default_factory=lambda: os.environ.get("QUERY_EXPANSION_ENABLED", "true").lower() == "true"
    )  # Benchmark-validated: biggest ablation delta (+0.030)

    # Phase 2 - Intelligence Layer (HyDE, CRAG)
    hyde_enabled: bool = field(
        default_factory=lambda: os.environ.get("HYDE_ENABLED", "true").lower() == "true"
    )  # Quality-first: always on
    hyde_model: str = field(
        default_factory=lambda: os.environ.get("HYDE_MODEL", "qwen2.5:1.5b")
    )  # Fast model for hypothetical doc generation
    crag_enabled: bool = field(
        default_factory=lambda: os.environ.get("CRAG_ENABLED", "true").lower() == "true"
    )
    crag_min_confidence: float = field(
        default_factory=lambda: float(os.environ.get("CRAG_MIN_CONFIDENCE", "0.25"))
    )  # Below this triggers web search
    crag_web_search_on_low: bool = field(
        default_factory=lambda: os.environ.get("CRAG_WEB_SEARCH_ON_LOW", "true").lower() == "true"
    )
    searxng_enabled: bool = field(
        default_factory=lambda: os.environ.get("SEARXNG_ENABLED", "true").lower() == "true"
    )
    searxng_url: str = field(
        default_factory=lambda: os.environ.get("SEARXNG_URL", "http://searxng:8080").strip() or "http://searxng:8080"
    )
    searxng_categories: str = field(
        default_factory=lambda: os.environ.get("SEARXNG_CATEGORIES", "general").strip() or "general"
    )
    searxng_language: str = field(
        default_factory=lambda: os.environ.get("SEARXNG_LANGUAGE", "").strip()
    )
    searxng_timeout_s: float = field(
        default_factory=lambda: float(os.environ.get("SEARXNG_TIMEOUT_S", "10"))
    )
    searxng_max_results: int = field(
        default_factory=lambda: int(os.environ.get("SEARXNG_MAX_RESULTS", "5"))
    )
    searxng_request_format: str = field(
        default_factory=lambda: os.environ.get("SEARXNG_REQUEST_FORMAT", "json").strip().lower() or "json"
    )

    # Phase 3a - RAPTOR Hierarchical Summarization
    raptor_enabled: bool = field(
        default_factory=lambda: os.environ.get("RAPTOR_ENABLED", "true").lower() == "true"
    )
    raptor_max_levels: int = field(
        default_factory=lambda: int(os.environ.get("RAPTOR_MAX_LEVELS", "3"))
    )  # Tree depth (1=cluster, 2=section, 3=topic)
    raptor_cluster_size: int = field(
        default_factory=lambda: int(os.environ.get("RAPTOR_CLUSTER_SIZE", "10"))
    )  # Target items per cluster
    raptor_summary_model: str = field(
        default_factory=lambda: os.environ.get("RAPTOR_SUMMARY_MODEL", "qwen2.5:1.5b")
    )  # Fast model for batch summarization
    raptor_retrieval_strategy: str = field(
        default_factory=lambda: os.environ.get("RAPTOR_RETRIEVAL_STRATEGY", "collapsed")
    )  # "collapsed" or "tree_traversal"

    # Phase 3b - GraphRAG with Neo4j
    graph_rag_enabled: bool = field(
        default_factory=lambda: os.environ.get("GRAPH_RAG_ENABLED", "true").lower() == "true"
    )
    neo4j_url: str = field(
        default_factory=lambda: os.environ.get("NEO4J_URL", "bolt://localhost:7687")
    )
    neo4j_user: str = field(
        default_factory=lambda: os.environ.get("NEO4J_USER", "neo4j")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.environ.get("NEO4J_PASSWORD", "")
    )
    graph_rag_auto: bool = field(
        default_factory=lambda: os.environ.get("GRAPH_RAG_AUTO", "true").lower() == "true"
    )  # Auto-activate on cross-reference queries even when graph_rag_enabled=false
    graph_rag_weight: float = field(
        default_factory=lambda: float(os.environ.get("GRAPH_RAG_WEIGHT", "0.2"))
    )  # Weight in RRF fusion
    graph_max_hops: int = field(
        default_factory=lambda: int(os.environ.get("GRAPH_MAX_HOPS", "2"))
    )  # Max traversal depth

    # Phase 3c - Community Summaries for Global Search
    community_detection_enabled: bool = field(
        default_factory=lambda: os.environ.get("COMMUNITY_DETECTION_ENABLED", "true").lower() == "true"
    )
    global_search_enabled: bool = field(
        default_factory=lambda: os.environ.get("GLOBAL_SEARCH_ENABLED", "true").lower() == "true"
    )
    global_search_top_k: int = field(
        default_factory=lambda: int(os.environ.get("GLOBAL_SEARCH_TOP_K", "3"))
    )
    community_level_default: str = field(
        default_factory=lambda: os.environ.get("COMMUNITY_LEVEL_DEFAULT", "medium")
    )  # "coarse", "medium", "fine"
    community_summary_model: str = field(
        default_factory=lambda: os.environ.get("COMMUNITY_SUMMARY_MODEL", "qwen2.5:1.5b")
    )

    # Topic toggles (which knowledge bases are enabled for search)
    # Default: reads from domain lexicon, falls back to "general"
    enabled_topics: str = field(
        default_factory=lambda: os.environ.get("ENABLED_TOPICS", _get_default_topics())
    )

    # Reranker settings
    reranker_enabled: bool = field(default_factory=lambda: os.environ.get("RERANKER_ENABLED", "true").lower() == "true")
    reranker_candidates: int = field(
        default_factory=lambda: int(os.environ.get("RERANKER_CANDIDATES", "15"))
    )  # Benchmark v2 validated: 15 candidates optimal (+0.0055 composite)
    reranker_batch_size: int = field(default_factory=lambda: int(os.environ.get("RERANKER_BATCH_SIZE", "32")))
    reranker_device: str = field(default_factory=lambda: os.environ.get("COHESIONN_RERANK_DEVICE", "cuda"))

    # Domain boost settings (counters reranker LRFD bias)
    # Benchmark-validated defaults (shootout_20260208_211323)
    domain_boost_enabled: bool = field(
        default_factory=lambda: os.environ.get("DOMAIN_BOOST_ENABLED", "true").lower() == "true"
    )
    domain_boost_factor: float = field(
        default_factory=lambda: float(os.environ.get("DOMAIN_BOOST_FACTOR", "0.5"))
    )  # Benchmark v2 validated: 0.5 optimal (+0.0035 composite)

    # Retrieval profile (controls which pipeline stages are active)
    retrieval_profile: str = field(
        default_factory=lambda: os.environ.get("RETRIEVAL_PROFILE", "fast")
    )

    # Vision ingest toggle (Phase 2 vision extraction during ingest)
    vision_ingest_enabled: bool = field(
        default_factory=lambda: os.environ.get("VISION_INGEST_ENABLED", "true").lower() == "true"
    )

    # Router settings
    router_use_semantic: bool = field(
        default_factory=lambda: os.environ.get("ROUTER_USE_SEMANTIC", "true").lower() == "true"
    )

    # Streaming settings
    stream_token_delay_ms: int = field(
        default_factory=lambda: int(os.environ.get("STREAM_DELAY_MS", "50"))
    )  # 50ms = ~20 words/sec (readable pace)

    # Phii behavior module settings
    phii_energy_matching: bool = field(
        default_factory=lambda: os.environ.get("PHII_ENERGY_MATCHING", "true").lower() == "true"
    )
    phii_specialty_detection: bool = field(
        default_factory=lambda: os.environ.get("PHII_SPECIALTY_DETECTION", "true").lower() == "true"
    )
    phii_implicit_feedback: bool = field(
        default_factory=lambda: os.environ.get("PHII_IMPLICIT_FEEDBACK", "true").lower() == "true"
    )

    # LLM timeout settings (seconds)
    llm_timeout: int = field(default_factory=lambda: int(os.environ.get("LLM_TIMEOUT", "180")))  # 3 min default
    llm_timeout_think: int = field(
        default_factory=lambda: int(os.environ.get("LLM_TIMEOUT_THINK", "300"))
    )  # 5 min for think mode
    max_output_tokens: int = field(
        default_factory=lambda: int(os.environ.get("LLM_MAX_OUTPUT", "4096"))
    )  # Cap output length

    # Tool Router (deprecated - GLM-4.7 handles tool calls natively via --tool-call-parser)
    tool_router_enabled: bool = field(default_factory=lambda: False)

    # Admin API key (deprecated - use admin panel setup instead)
    # Kept temporarily for migration: if set, auto-hashes as initial password on first run
    admin_key: str = field(default_factory=lambda: os.environ.get("ADMIN_KEY", ""))

    # Startup cleanup (wipes sessions, uploads, reports on boot)
    # Default false — set true for production multi-user deployments
    cleanup_on_startup: bool = field(
        default_factory=lambda: os.environ.get("CLEANUP_ON_STARTUP", "false").lower() == "true"
    )

    # Admin password reset flag (set to true + restart to force re-setup)
    admin_reset: bool = field(
        default_factory=lambda: os.environ.get("ADMIN_RESET", "false").lower() == "true"
    )

    # Data directories (Docker mounts vs local dev)
    # In Docker: ./data/maps is mounted to /data/maps
    # In local dev: relative path from backend/ to data/maps
    maps_dir: str = field(default_factory=lambda: os.environ.get("MAPS_DIR", "/data/maps"))

    # Redis session cache settings
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    redis_enabled: bool = field(
        default_factory=lambda: os.environ.get("REDIS_ENABLED", "true").lower() == "true"
    )
    redis_session_ttl: int = field(
        default_factory=lambda: int(os.environ.get("REDIS_SESSION_TTL", "86400"))
    )  # 24 hours in seconds

    # PostgreSQL database settings
    database_url: str = field(
        default_factory=_build_database_url_default
    )
    database_enabled: bool = field(
        default_factory=lambda: os.environ.get("DATABASE_ENABLED", "true").lower() == "true"
    )
    database_pool_size: int = field(
        default_factory=lambda: int(os.environ.get("DATABASE_POOL_SIZE", "20"))
    )

    # Rate limiting settings (per minute)
    rate_limit_ws_conn: int = field(
        default_factory=lambda: int(os.environ.get("RATE_LIMIT_WS_CONN", "30"))
    )  # WebSocket connections per IP
    rate_limit_ws_msg: int = field(
        default_factory=lambda: int(os.environ.get("RATE_LIMIT_WS_MSG", "30"))
    )  # Messages per session
    rate_limit_upload: int = field(
        default_factory=lambda: int(os.environ.get("RATE_LIMIT_UPLOAD", "10"))
    )  # File uploads per IP
    rate_limit_admin_login: int = field(
        default_factory=lambda: int(os.environ.get("RATE_LIMIT_ADMIN_LOGIN", "5"))
    )  # Admin login attempts per IP per 15 min window

    # VRAM budget enforcement - if true, warn but don't block when over budget
    allow_vram_spillover: bool = field(
        default_factory=lambda: os.environ.get("ALLOW_VRAM_SPILLOVER", "false").lower() == "true"
    )

    # Ingest lock - blocks chat/tool models from loading during ingestion
    ingest_lock_enabled: bool = field(
        default_factory=lambda: os.environ.get("INGEST_LOCK_ENABLED", "true").lower() == "true"
    )

    # Environment mode — controls security behavior (development, staging, production)
    arca_env: str = field(
        default_factory=lambda: os.environ.get("ARCA_ENV", "development")
    )

    # MCP Mode — disables local chat, ARCA becomes tool backend for cloud AI models
    mcp_mode: bool = field(
        default_factory=lambda: os.environ.get("MCP_MODE", "false").lower() == "true"
    )

    # Core Knowledge — built-in ARCA self-knowledge corpus
    core_knowledge_enabled: bool = field(
        default_factory=lambda: os.environ.get("CORE_KNOWLEDGE_ENABLED", "true").lower() == "true"
    )
    core_knowledge_dir: str = field(
        default_factory=lambda: os.environ.get("CORE_KNOWLEDGE_DIR", "/app/data/core_knowledge")
    )
    core_knowledge_collection: str = field(
        default_factory=lambda: os.environ.get("CORE_KNOWLEDGE_COLLECTION", "arca_core")
    )

    # Corpus Profiling — term extraction after ingest for Phii context
    corpus_profiling_enabled: bool = field(
        default_factory=lambda: os.environ.get("CORPUS_PROFILING_ENABLED", "true").lower() == "true"
    )
    corpus_profile_path: str = field(
        default_factory=lambda: os.environ.get("CORPUS_PROFILE_PATH", "data/config/corpus_profile.json")
    )

    # Internal state
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)
    _update_count: int = field(default=0, repr=False)
    _ingest_active: bool = field(default=False, repr=False, compare=False)

    @property
    def ingest_active(self) -> bool:
        """Check if ingestion is currently running (blocks chat model loading)."""
        return self._ingest_active and self.ingest_lock_enabled

    def set_ingest_active(self, active: bool) -> None:
        """Set the ingest-active flag. When active, chat/tool models are blocked from loading."""
        with self._lock:
            self._ingest_active = active
            if active:
                logger.info("INGEST LOCK ON - chat and tool model loading blocked")
            else:
                logger.info("INGEST LOCK OFF - chat and tool model loading restored")

    # Validation ranges for numeric config values
    _VALIDATION_RANGES: Dict[str, tuple] = field(default_factory=lambda: {
        "temperature": (0.0, 2.0),
        "top_p": (0.0, 1.0),
        "top_k": (1, 200),
        "ctx_small": (512, 131072),
        "ctx_medium": (512, 131072),
        "ctx_large": (512, 131072),
        "ctx_xlarge": (512, 131072),
        "max_output_tokens": (64, 32768),
        "rate_limit_ws_conn": (1, 1000),
        "rate_limit_ws_msg": (1, 1000),
        "rate_limit_upload": (1, 100),
        "rate_limit_admin_login": (1, 50),
        "searxng_timeout_s": (1.0, 60.0),
        "searxng_max_results": (1, 25),
    })

    def update(self, **kwargs) -> Dict[str, Any]:
        """
        Update configuration values at runtime.

        Args:
            **kwargs: Key-value pairs to update (e.g., ctx_medium=12288)

        Returns:
            Dict with 'updated' (changed keys) and 'ignored' (unknown keys)
        """
        updated = []
        ignored = []

        with self._lock:
            for key, value in kwargs.items():
                if key.startswith("_"):
                    ignored.append(key)
                    continue

                if hasattr(self, key):
                    if key == "searxng_url" and isinstance(value, str):
                        cleaned = value.strip()
                        if not cleaned.startswith(("http://", "https://")):
                            ignored.append(key)
                            logger.warning(f"Config rejected invalid URL: {key}={value!r}")
                            continue
                        value = cleaned.rstrip("/")

                    if key in {"searxng_categories", "searxng_language"} and isinstance(value, str):
                        value = value.strip()
                        if key == "searxng_categories":
                            parts = [p.strip() for p in value.split(",") if p.strip()]
                            value = ",".join(parts) if parts else "general"

                    if key == "searxng_request_format" and isinstance(value, str):
                        value = value.strip().lower()
                        if value not in {"json", "html"}:
                            ignored.append(key)
                            logger.warning(
                                f"Config rejected invalid format: {key}={value!r} (must be 'json' or 'html')"
                            )
                            continue

                    # Validate model names (alphanumeric, colons, dots, dashes only)
                    if key.startswith("model_") and isinstance(value, str) and value:
                        import re as _re
                        if not _re.match(r'^[a-zA-Z0-9._:-]+$', value) or len(value) > 100:
                            ignored.append(key)
                            logger.warning(f"Config rejected invalid model name: {key}={value!r}")
                            continue

                    # Validate numeric ranges
                    if key in self._VALIDATION_RANGES:
                        lo, hi = self._VALIDATION_RANGES[key]
                        if not (lo <= value <= hi):
                            ignored.append(key)
                            logger.warning(f"Config rejected {key}={value} (must be {lo}-{hi})")
                            continue

                    old_value = getattr(self, key)
                    setattr(self, key, value)
                    updated.append(key)
                    logger.info(f"Config updated: {key} = {value} (was {old_value})")
                else:
                    ignored.append(key)
                    logger.warning(f"Config ignored unknown key: {key}")

            self._update_count += 1

        return {"updated": updated, "ignored": ignored, "update_count": self._update_count}

    def get_enabled_topics(self) -> List[str]:
        """Get list of enabled topic names."""
        topics = [t.strip() for t in self.enabled_topics.split(",") if t.strip()] if self.enabled_topics else []

        # Built-in docs should always be searchable when core knowledge is enabled.
        if self.core_knowledge_enabled:
            core_topic = (self.core_knowledge_collection or "arca_core").strip() or "arca_core"
            lowered = {t.lower() for t in topics}
            if core_topic.lower() not in lowered:
                topics.append(core_topic)

        return topics

    def set_enabled_topics(self, topics: List[str]) -> None:
        """Set enabled topics from a list."""
        self.enabled_topics = ",".join(topics)

    def is_topic_enabled(self, topic: str) -> bool:
        """Check if a specific topic is enabled."""
        return topic in self.get_enabled_topics()

    def get_context_size(
        self,
        think_mode: bool = False,
        code_mode: bool = False,
        has_rag_context: bool = False,
        simple_tools_only: bool = False,
    ) -> int:
        """
        Determine optimal context size based on task complexity.

        Args:
            think_mode: Extended reasoning active
            code_mode: Code generation detected
            has_rag_context: RAG results will be included
            simple_tools_only: Only simple tools like unit_convert

        Returns:
            Optimal context window size in tokens
        """
        if think_mode or code_mode:
            return self.ctx_xlarge
        if has_rag_context:
            return self.ctx_large
        if simple_tools_only:
            return self.ctx_small
        return self.ctx_medium

    def apply_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Apply a named retrieval profile. Loads toggles from ProfileManager,
        applies them to config fields via setattr, saves overrides.

        Returns dict of changes made.
        """
        from profile_loader import get_profile_manager
        pm = get_profile_manager()
        result = pm.set_active(profile_name)
        self.retrieval_profile = profile_name
        self.save_overrides()
        return result

    def get_llm_params(self) -> Dict[str, Any]:
        """Get LLM parameters for OpenAI API calls."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_output_tokens,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export current config as dict (excludes internal fields)."""
        from dataclasses import fields as dataclass_fields

        result = {}
        for field_info in dataclass_fields(self):
            if not field_info.name.startswith("_"):
                result[field_info.name] = getattr(self, field_info.name)
        return result

    # Config persistence
    _overrides_path: Path = field(
        default_factory=lambda: Path(os.environ.get(
            "CONFIG_OVERRIDES_PATH", "/app/data/config/config_overrides.json"
        )),
        repr=False, compare=False,
    )

    def save_overrides(self) -> None:
        """Save non-default values to persistent storage, including profile state."""
        defaults = RuntimeConfig()
        overrides = {}

        # Skip credential/connection/environment fields - those stay env-only
        skip_fields = {
            "admin_key", "admin_reset", "redis_url",
            "database_url", "neo4j_url", "neo4j_user", "neo4j_password",
            "maps_dir", "arca_env",
        }

        current = self.to_dict()
        default_dict = defaults.to_dict()

        for key, value in current.items():
            if key in skip_fields:
                continue
            if value != default_dict.get(key):
                overrides[key] = value

        # Persist profile manager state (active profile + manual overrides)
        try:
            from profile_loader import get_profile_manager
            pm = get_profile_manager()
            profile_state = pm.get_state_for_persistence()
            overrides["_profile_state"] = profile_state
        except Exception as e:
            logger.debug(f"Could not persist profile state: {e}")

        try:
            self._overrides_path.parent.mkdir(parents=True, exist_ok=True)
            self._overrides_path.write_text(
                json.dumps(overrides, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info(f"Config overrides saved: {len(overrides)} values to {self._overrides_path}")
        except Exception as e:
            logger.error(f"Failed to save config overrides: {e}")

    def load_overrides(self) -> Dict[str, Any]:
        """Load overrides from persistent storage. Env vars take precedence."""
        if not self._overrides_path.exists():
            return {}

        try:
            overrides = json.loads(self._overrides_path.read_text(encoding="utf-8"))
            if not isinstance(overrides, dict):
                return {}

            # Extract profile state before applying field overrides
            profile_state = overrides.pop("_profile_state", None)

            # Only apply overrides for fields that still have their default value
            # (env vars would have already changed them from default)
            defaults = RuntimeConfig()
            applied = []

            with self._lock:
                for key, value in overrides.items():
                    if key.startswith("_") or not hasattr(self, key):
                        continue
                    current = getattr(self, key)
                    default = getattr(defaults, key)
                    # Only apply if current == default (env var didn't override)
                    if current == default and value != default:
                        # Cast to correct type
                        field_type = type(default)
                        try:
                            typed_value = field_type(value)
                            setattr(self, key, typed_value)
                            applied.append(key)
                        except (ValueError, TypeError):
                            logger.warning(f"Config override type mismatch: {key}={value}")

            # Restore profile manager state
            if profile_state and isinstance(profile_state, dict):
                try:
                    from profile_loader import get_profile_manager, PROFILE_TOGGLE_KEYS
                    pm = get_profile_manager()
                    saved_profile = profile_state.get("retrieval_profile", "fast")
                    saved_overrides = profile_state.get("manual_overrides", {})
                    pm.load_state(saved_profile, saved_overrides)
                    self.retrieval_profile = saved_profile
                    applied.append("_profile_state")
                    logger.info(f"Restored profile state: {saved_profile}")
                except Exception as e:
                    logger.debug(f"Could not restore profile state: {e}")
            else:
                # Migration: no profile state saved — check if existing toggles
                # overlap with profile toggle keys to preserve custom configs
                from profile_loader import PROFILE_TOGGLE_KEYS
                has_toggle_overrides = any(
                    k in PROFILE_TOGGLE_KEYS for k in overrides if not k.startswith("_")
                )
                if has_toggle_overrides:
                    self.retrieval_profile = "custom"
                    logger.info("Migration: existing toggle overrides found, set profile='custom'")

            if applied:
                logger.info(f"Config overrides loaded: {', '.join(applied)}")
            return {"applied": applied, "total": len(overrides)}

        except Exception as e:
            logger.error(f"Failed to load config overrides: {e}")
            return {}

    def reset_to_defaults(self) -> Dict[str, Any]:
        """Reset all values to environment defaults and clear overrides."""
        defaults = RuntimeConfig()
        changes = {}

        with self._lock:
            for key in self.to_dict().keys():
                old_value = getattr(self, key)
                new_value = getattr(defaults, key)
                if old_value != new_value:
                    setattr(self, key, new_value)
                    changes[key] = {"old": old_value, "new": new_value}
                    logger.info(f"Config reset: {key} = {new_value}")

            self._update_count += 1

        # Clear persisted overrides
        try:
            if self._overrides_path.exists():
                self._overrides_path.write_text("{}", encoding="utf-8")
                logger.info("Config overrides file cleared")
        except Exception as e:
            logger.error(f"Failed to clear overrides file: {e}")

        return {"reset": True, "changes": changes, "update_count": self._update_count}


# Singleton instance
runtime_config = RuntimeConfig()

# Load persisted overrides on startup
runtime_config.load_overrides()


def get_config() -> RuntimeConfig:
    """Get the singleton config instance."""
    return runtime_config
