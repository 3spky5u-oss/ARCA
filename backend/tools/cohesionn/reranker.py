"""
Cohesionn Reranker - Cross-encoder for precise relevance scoring

Default: Jina-Reranker-v2-Turbo via COHESIONN_RERANK_MODEL env var

Backend selection:
- ONNX (default): Eliminates PyTorch CUDA context (~17 GB host RAM savings)
- PyTorch: Legacy fallback, set COHESIONN_RERANK_BACKEND=torch

GPU-accelerated by default when CUDA is available.
Includes DiversityReranker for Maximal Marginal Relevance (MMR).
"""

import logging
import warnings
from typing import List, Dict, Any, Optional
from pathlib import Path
import os

logger = logging.getLogger(__name__)

# Suppress noisy flash_attn warnings from Jina reranker models
# (flash_attn is optional — PyTorch native attention works fine)
warnings.filterwarnings("ignore", message="flash_attn is not installed")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class _FlashAttnDenyFilter(logging.Filter):
    """Drop repetitive flash_attn warnings emitted by remote model code."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "flash_attn is not installed" not in record.getMessage()


logging.getLogger().addFilter(_FlashAttnDenyFilter())
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers_modules").setLevel(logging.ERROR)

# Benchmark-validated defaults (shootout_20260208_211323)
# jina-v2-turbo scored 0.752 composite in cross-matrix (paired with qwen3-06b)
# Beat bge-v2-m3 (0.732), mxbai-base-v2 (0.749), bge-v2-gemma (0.747)
RERANKER_MODEL = os.environ.get("COHESIONN_RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
RERANKER_MAX_LENGTH = int(os.environ.get("COHESIONN_RERANK_MAX_LENGTH", "1024"))


def _suppress_ort_warnings():
    """Suppress verbose ONNX Runtime C++ warnings (memcpy nodes, EP assignments)."""
    try:
        import onnxruntime
        onnxruntime.set_default_logger_severity(3)
    except (ImportError, AttributeError):
        pass


_suppress_ort_warnings()


def _detect_backend() -> str:
    """Detect best available backend: ONNX (preferred) or PyTorch."""
    env = os.environ.get("COHESIONN_RERANK_BACKEND", "auto")
    if env != "auto":
        return env
    try:
        import onnxruntime
        return "onnx"
    except ImportError:
        return "torch"


RERANKER_BACKEND = _detect_backend()


def _onnx_cache_dir(model_name: str) -> str:
    """Get persistent cache path for ONNX-exported models."""
    slug = model_name.replace("/", "--")
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(hf_home, "onnx_exports", slug)


def _onnx_cuda_provider_available() -> bool:
    """Return True when ONNX Runtime advertises CUDAExecutionProvider."""
    try:
        import onnxruntime
        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


def _cuda_runtime_available(log_failure: bool = False) -> bool:
    """Return True when CUDA is actually usable from PyTorch."""
    try:
        import torch
    except ImportError:
        return False

    try:
        if not torch.cuda.is_available():
            return False
        if torch.cuda.device_count() < 1:
            return False
        _ = torch.cuda.get_device_name(0)
        return True
    except Exception as e:
        if log_failure:
            logger.warning(f"CUDA runtime unavailable for reranker, using CPU: {e}")
        return False


def _auto_detect_device() -> str:
    """Auto-detect best available device (GPU or CPU)."""
    if RERANKER_BACKEND == "onnx":
        if _onnx_cuda_provider_available() and _cuda_runtime_available():
            logger.debug("ONNX CUDA available for reranker")
            return "cuda"
        return "cpu"

    if _cuda_runtime_available():
        try:
            import torch
            device_name = torch.cuda.get_device_name(0)
            logger.debug(f"CUDA available: {device_name}")
        except Exception:
            pass
        return "cuda"
    return "cpu"


def _resolve_device(preference: str = "auto") -> str:
    """Resolve device preference to actual device string.

    Args:
        preference: "auto", "cuda", or "cpu"
    """
    if preference in ("cuda", "cpu"):
        if preference == "cuda" and not _cuda_runtime_available(log_failure=True):
            logger.warning("COHESIONN_RERANK_DEVICE=cuda requested but CUDA is unavailable. Falling back to CPU.")
            return "cpu"
        return preference
    return _auto_detect_device()


def _detect_device() -> str:
    """Get initial device from env var or auto-detect."""
    env_device = os.environ.get("COHESIONN_RERANK_DEVICE", "cuda")
    return _resolve_device(env_device)


RERANKER_DEVICE = _detect_device()


class BGEReranker:
    """
    Cross-encoder reranker (supports Qwen3 and BGE models).

    Cross-encoders are more accurate than bi-encoders for ranking
    but slower, so we use them to re-rank top candidates.

    Default: BGE-Reranker-v2-M3 (CrossEncoder compatible)
    """

    def __init__(self, model_name: str = None, device: str = None, max_length: int = None, backend: str = None, trust_remote_code: bool = None, device_id: int = None):
        self.model_name = model_name or RERANKER_MODEL
        self.device = device or RERANKER_DEVICE
        self.max_length = max_length or RERANKER_MAX_LENGTH
        self.backend = backend or RERANKER_BACKEND
        # Jina rerankers require trust_remote_code=True for custom architecture
        if trust_remote_code is None:
            self.trust_remote_code = "jinaai" in self.model_name.lower()
        else:
            self.trust_remote_code = trust_remote_code
        # GPU device index from device map (multi-GPU support)
        if device_id is not None:
            self._device_id = device_id
        else:
            try:
                from services.hardware import get_gpu_for_component
                self._device_id = get_gpu_for_component("reranker")
            except Exception:
                self._device_id = 0
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                if self.backend == "onnx":
                    if self.device == "cuda" and not _cuda_runtime_available(log_failure=True):
                        logger.warning("ONNX CUDA requested for reranker but CUDA is unavailable. Switching to CPU.")
                        self.device = "cpu"
                    provider = "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
                    # Multi-GPU: route to assigned device
                    model_kwargs: dict = {"provider": provider}
                    if provider == "CUDAExecutionProvider" and self._device_id != 0:
                        model_kwargs["provider_options"] = {"device_id": self._device_id}
                        logger.info(f"ONNX reranker targeting GPU {self._device_id}")
                    cache_dir = _onnx_cache_dir(self.model_name)
                    try:
                        # Check for cached ONNX export first (avoids PyTorch re-export)
                        if os.path.isdir(cache_dir) and os.path.isfile(os.path.join(cache_dir, "config.json")):
                            logger.info(f"Loading reranker from ONNX cache: {cache_dir}")
                            self._model = CrossEncoder(
                                cache_dir,
                                backend="onnx",
                                model_kwargs=model_kwargs,
                                max_length=self.max_length,
                                trust_remote_code=self.trust_remote_code,
                            )
                        else:
                            logger.info(f"Loading reranker: {self.model_name} (ONNX, provider={provider}, max_length={self.max_length})")
                            self._model = CrossEncoder(
                                self.model_name,
                                backend="onnx",
                                model_kwargs=model_kwargs,
                                max_length=self.max_length,
                                trust_remote_code=self.trust_remote_code,
                            )
                            # Cache the export for next startup
                            try:
                                self._model.save_pretrained(cache_dir)
                                logger.info(f"Cached ONNX export to: {cache_dir}")
                            except Exception as save_err:
                                logger.warning(f"Failed to cache ONNX export: {save_err}")

                        self._model.predict([("test query", "test document")], show_progress_bar=False)
                    except Exception as e:
                        logger.info(f"ONNX unavailable for {self.model_name}; using PyTorch backend.")
                        logger.debug(f"ONNX fallback reason for {self.model_name}: {e}")
                        self._model = None
                        self.backend = "torch"

                if self.backend != "onnx":
                    if self._model is None:
                        device = self.device
                        if device == "cuda" and not _cuda_runtime_available(log_failure=True):
                            logger.warning("PyTorch CUDA requested for reranker but CUDA is unavailable. Switching to CPU.")
                            device = "cpu"
                            self.device = "cpu"
                        logger.info(f"Loading reranker: {self.model_name} (PyTorch, device={device}, max_length={self.max_length})")
                        self._model = CrossEncoder(
                            self.model_name,
                            device=device,
                            max_length=self.max_length,
                            trust_remote_code=self.trust_remote_code,
                        )

                logger.info(f"Reranker loaded (backend={self.backend})")

            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
                self._model = "fallback"

        return self._model

    def unload(self):
        """Explicitly free model from GPU/RAM."""
        if self._model is not None and self._model != "fallback":
            del self._model
            self._model = None
        import gc
        gc.collect()
        if self.backend != "onnx":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
        logger.info(f"Reranker {self.model_name} unloaded")

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
        batch_size: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results using cross-encoder with GPU-optimized batching.

        Args:
            query: Original search query
            results: List of search results with "content" field
            top_k: Number of top results to return
            batch_size: Batch size for GPU scoring (default from config)

        Returns:
            Re-ranked results with added "rerank_score"
        """
        if not results:
            return []

        if self.model == "fallback":
            # Just return top-k by original score
            logger.warning("Reranker not available, using original scores")
            return results[:top_k]

        # Use config batch size if not explicitly provided
        if batch_size is None:
            try:
                from config import runtime_config
                batch_size = runtime_config.reranker_batch_size
            except Exception:
                batch_size = 32

        # Filter results without content (e.g., sparse-only BM25 results)
        results = [r for r in results if r.get("content")]
        if not results:
            return []

        # Prepare query-document pairs
        pairs = [(query, r["content"]) for r in results]

        # Get cross-encoder scores with batching for GPU efficiency
        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
            batch_size=batch_size,
        )

        # Add scores to results
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        # Sort by rerank score
        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return results[:top_k]


class DiversityReranker:
    """
    Maximal Marginal Relevance (MMR) reranker for source diversity.

    MMR balances relevance with diversity by penalizing results that are
    too similar to already-selected results. Also enforces a maximum
    number of chunks per source document.

    Formula: score = λ * relevance - (1-λ) * max_similarity_to_selected

    - λ = 1.0: Pure relevance (no diversity)
    - λ = 0.5: Balanced relevance and diversity
    - λ = 0.0: Maximum diversity (ignore relevance after first pick)
    """

    def __init__(
        self,
        lambda_param: float = 0.6,
        max_per_source: int = 2,
    ):
        """
        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0).
                         Default 0.6 favors relevance while encouraging diversity.
            max_per_source: Maximum chunks allowed from a single source document.
        """
        self.lambda_param = lambda_param
        self.max_per_source = max_per_source

    def _get_source_key(self, result: Dict[str, Any]) -> str:
        """Extract source identifier from result metadata."""
        meta = result.get("metadata", {})
        source = meta.get("source", "")
        if source:
            # Use just the filename for source grouping
            return Path(source).stem.lower()
        return f"unknown_{id(result)}"

    def _compute_similarity(self, result_a: Dict[str, Any], result_b: Dict[str, Any]) -> float:
        """
        Compute content similarity between two results.

        Uses Jaccard similarity on word sets for efficiency.
        """
        content_a = result_a.get("content", "").lower()
        content_b = result_b.get("content", "").lower()

        # Tokenize into word sets
        words_a = set(content_a.split())
        words_b = set(content_b.split())

        if not words_a or not words_b:
            return 0.0

        # Jaccard similarity
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0

    def _max_similarity_to_selected(
        self,
        candidate: Dict[str, Any],
        selected: List[Dict[str, Any]],
    ) -> float:
        """Compute maximum similarity between candidate and any selected result."""
        if not selected:
            return 0.0

        return max(self._compute_similarity(candidate, s) for s in selected)

    def rerank(
        self,
        results: List[Dict[str, Any]],
        top_k: int = 5,
        score_key: str = "rerank_score",
    ) -> List[Dict[str, Any]]:
        """
        Apply MMR diversity reranking with source limits.

        Args:
            results: Results with relevance scores (from cross-encoder)
            top_k: Number of results to return
            score_key: Key to read relevance score from (rerank_score or score)

        Returns:
            Reranked results balancing relevance with diversity
        """
        if not results:
            return []

        if len(results) <= 1:
            return results[:top_k]

        # Normalize scores to 0-1 range for MMR calculation
        scores = [r.get(score_key, r.get("score", 0)) for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        for r, raw_score in zip(results, scores):
            r["_mmr_relevance"] = (raw_score - min_score) / score_range

        selected = []
        source_counts = {}
        remaining = list(results)

        while len(selected) < top_k and remaining:
            best_score = float("-inf")
            best_idx = 0

            for i, candidate in enumerate(remaining):
                source_key = self._get_source_key(candidate)

                # Skip if source has reached max
                if source_counts.get(source_key, 0) >= self.max_per_source:
                    continue

                # MMR score
                relevance = candidate["_mmr_relevance"]
                similarity = self._max_similarity_to_selected(candidate, selected)
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            # Check if we found a valid candidate
            if best_score == float("-inf"):
                # All remaining candidates are from maxed-out sources
                logger.debug("MMR: All remaining candidates exceed source limits")
                break

            # Select the best candidate
            chosen = remaining.pop(best_idx)
            source_key = self._get_source_key(chosen)
            source_counts[source_key] = source_counts.get(source_key, 0) + 1

            # Store MMR score for debugging
            chosen["_mmr_score"] = best_score
            selected.append(chosen)

        # Clean up temporary fields
        for r in results:
            r.pop("_mmr_relevance", None)

        unique_sources = len(set(self._get_source_key(r) for r in selected))
        logger.info(f"MMR selected {len(selected)} chunks from {unique_sources} unique sources")

        return selected


def apply_domain_boost(
    results: List[Dict[str, Any]],
    query: str,
    boost_factor: float = 0.15,
    score_key: str = "rerank_score",
) -> List[Dict[str, Any]]:
    """
    Boost scores for results containing domain-specific keywords.

    Counters reranker bias (e.g., LRFD bias in Qwen models) by boosting
    results that match domain-specific terms found in the query.

    Args:
        results: List of reranked results
        query: Original user query
        boost_factor: Amount to boost matching results (0.15 = +15% score)
        score_key: Key containing the score to boost

    Returns:
        Results with boosted scores, re-sorted by boosted score
    """
    if not results or boost_factor <= 0:
        return results

    query_lower = query.lower()

    # Find domain keywords present in query
    query_keywords = set()
    config = _get_topic_config()
    for topic, keywords in config["keywords"].items():
        for kw in keywords:
            if kw in query_lower:
                query_keywords.add(kw)

    if not query_keywords:
        logger.debug("Domain boost: no domain keywords in query")
        return results

    logger.debug(f"Domain boost: query contains keywords {query_keywords}")

    # Boost results containing those keywords
    boosted_count = 0
    for result in results:
        content_lower = result.get("content", "").lower()
        original_score = result.get(score_key, result.get("score", 0))

        # Count matching keywords in content
        matches = sum(1 for kw in query_keywords if kw in content_lower)

        if matches > 0:
            # Apply boost proportional to keyword matches (capped at 3)
            boost = boost_factor * min(matches, 3) / 3
            boosted_score = original_score + boost
            result["_domain_boost"] = boost
            result[score_key] = boosted_score
            boosted_count += 1

    if boosted_count > 0:
        # Re-sort by boosted scores
        results.sort(key=lambda x: x.get(score_key, 0), reverse=True)
        logger.info(f"Domain boost applied to {boosted_count}/{len(results)} results")

    return results


_topic_cache = None


def _get_topic_config():
    """Load topic descriptions and keywords from lexicon pipeline config."""
    global _topic_cache
    if _topic_cache is not None:
        return _topic_cache
    try:
        from domain_loader import get_pipeline_config
        from config import runtime_config
        pipeline = get_pipeline_config()
        descriptions = pipeline.get("rag_topic_descriptions", {"general": "General knowledge and technical documentation."})
        keywords = pipeline.get("rag_topic_keywords", {"general": ["document", "report", "technical", "standard", "reference"]})

        # Always add arca_core topic if core knowledge is enabled
        if getattr(runtime_config, 'core_knowledge_enabled', True):
            descriptions["arca_core"] = "ARCA platform documentation, configuration guides, RAG pipeline architecture, domain pack system, tool descriptions, troubleshooting, and system administration."
            keywords["arca_core"] = [
                "arca", "configure", "configuration", "setup", "install",
                "pipeline", "rag", "retrieval", "embedding", "reranker",
                "domain pack", "lexicon", "tool", "benchmark", "ingest",
                "admin", "settings", "docker", "qdrant", "neo4j",
                "troubleshoot", "error", "how to", "what is", "help",
            ]

        _topic_cache = {"descriptions": descriptions, "keywords": keywords}
    except Exception:
        _topic_cache = {
            "descriptions": {"general": "General knowledge and technical documentation."},
            "keywords": {"general": ["document", "report", "technical", "standard", "reference"]},
        }
    return _topic_cache


def clear_topic_cache():
    """Clear cached topic config (called on domain switch)."""
    global _topic_cache
    _topic_cache = None


class TopicRouter:
    """
    Route queries to appropriate topic knowledge bases.

    Uses hybrid scoring: 30% keyword matching + 70% semantic similarity.
    Pre-computes topic embeddings on first use for fast routing.

    Note: The RAPTOR_WEIGHT field is reserved for Phase 3a integration.
    When RAPTOR tree retrieval is enabled, results from the hierarchical
    summary tree are fused with dense+sparse results using this weight
    in reciprocal rank fusion. See raptor.py and retriever.py for usage.
    """

    # Path for persisting topic embeddings alongside BM25 indices
    _EMBEDDINGS_CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "cohesionn_db" / "topic_embeddings.pkl"

    def __init__(self):
        """Initialize router with lazy-loaded embeddings"""
        config = _get_topic_config()
        self.TOPIC_DESCRIPTIONS = config["descriptions"]
        self.TOPIC_KEYWORDS = config["keywords"]
        self._topic_embeddings: Optional[Dict[str, List[float]]] = None
        self._embedder = None

    def _get_embedder(self):
        """Lazy load embedder"""
        if self._embedder is None:
            from .embeddings import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def _get_topic_embeddings(self) -> Dict[str, List[float]]:
        """Pre-compute topic description embeddings (cached on disk)."""
        if self._topic_embeddings is None:
            # Try loading from disk first
            if self._EMBEDDINGS_CACHE_PATH.exists():
                try:
                    import pickle
                    with open(self._EMBEDDINGS_CACHE_PATH, "rb") as f:
                        cached = pickle.load(f)
                    # Validate cached topics match current definitions
                    if set(cached.keys()) == set(self.TOPIC_DESCRIPTIONS.keys()):
                        self._topic_embeddings = cached
                        logger.info(f"Loaded topic embeddings from cache ({len(cached)} topics)")
                        return self._topic_embeddings
                    else:
                        logger.info("Topic definitions changed, recomputing embeddings")
                except Exception as e:
                    logger.warning(f"Failed to load cached topic embeddings: {e}")

            # Compute fresh embeddings
            embedder = self._get_embedder()
            self._topic_embeddings = {}
            for topic, description in self.TOPIC_DESCRIPTIONS.items():
                self._topic_embeddings[topic] = embedder.embed_query(description)
            logger.info(f"Pre-computed embeddings for {len(self._topic_embeddings)} topics")

            # Persist to disk
            try:
                import pickle
                self._EMBEDDINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(self._EMBEDDINGS_CACHE_PATH, "wb") as f:
                    pickle.dump(self._topic_embeddings, f)
                logger.info(f"Saved topic embeddings to {self._EMBEDDINGS_CACHE_PATH}")
            except Exception as e:
                logger.warning(f"Failed to persist topic embeddings: {e}")

        return self._topic_embeddings

    def _keyword_score(self, query: str, topic: str) -> float:
        """Calculate keyword-based relevance score (0-1 normalized)"""
        query_lower = query.lower()
        keywords = self.TOPIC_KEYWORDS.get(topic, [])
        if not keywords:
            return 0.0
        matches = sum(1 for kw in keywords if kw in query_lower)
        # Normalize to 0-1 range (cap at 5 matches = 1.0)
        return min(matches / 5.0, 1.0)

    def _semantic_score(self, query: str, topic: str) -> float:
        """Calculate semantic similarity score using embeddings"""
        try:
            embedder = self._get_embedder()
            topic_embeddings = self._get_topic_embeddings()

            query_emb = embedder.embed_query(query)
            topic_emb = topic_embeddings.get(topic)

            if topic_emb is None:
                return 0.0

            # Cosine similarity (embeddings are normalized)
            import numpy as np

            similarity = np.dot(query_emb, topic_emb)
            # Convert from [-1, 1] to [0, 1]
            return (similarity + 1) / 2
        except Exception as e:
            logger.warning(f"Semantic scoring failed: {e}")
            return 0.0

    def route(self, query: str, use_semantic: bool = True) -> List[str]:
        """
        Determine which topics are relevant to a query using hybrid scoring.

        Args:
            query: User query
            use_semantic: Whether to use semantic matching (30% keyword + 70% semantic)

        Returns:
            List of relevant topics, most relevant first
        """
        scores = {}

        for topic in self.TOPIC_KEYWORDS.keys():
            keyword_score = self._keyword_score(query, topic)

            if use_semantic:
                semantic_score = self._semantic_score(query, topic)
                # Hybrid: 30% keyword + 70% semantic
                scores[topic] = 0.3 * keyword_score + 0.7 * semantic_score
            else:
                scores[topic] = keyword_score

        # Return all topics sorted by score (even if score is 0)
        sorted_topics = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        return sorted_topics

    def get_primary_topic(self, query: str) -> Optional[str]:
        """Get single most relevant topic"""
        topics = self.route(query)
        return topics[0] if topics else None

    def warm_embeddings(self):
        """Pre-warm topic embeddings on startup (call from lifespan)"""
        _ = self._get_topic_embeddings()


# Singletons
_reranker: Optional[BGEReranker] = None
_diversity_reranker: Optional[DiversityReranker] = None
_router: Optional[TopicRouter] = None


def get_reranker() -> BGEReranker:
    global _reranker

    # Check if admin changed device preference — reload if needed
    try:
        from config import runtime_config
        desired = _resolve_device(runtime_config.reranker_device)
        if _reranker is not None and _reranker.device != desired:
            logger.info(f"Reranker device changed: {_reranker.device} → {desired}, reloading")
            _reranker.unload()
            _reranker = None
    except Exception:
        pass

    if _reranker is None:
        # Resolve device from runtime config (falls back to env/auto)
        try:
            from config import runtime_config
            device = _resolve_device(runtime_config.reranker_device)
        except Exception:
            device = RERANKER_DEVICE

        # VRAM budget check before loading (on assigned GPU)
        try:
            from services.hardware import check_vram_budget, VRAM_ESTIMATES, get_gpu_for_component
            from config import runtime_config as rc
            gpu_id = get_gpu_for_component("reranker")
            estimated = VRAM_ESTIMATES.get(RERANKER_MODEL, 600)
            if not check_vram_budget(
                f"reranker ({RERANKER_MODEL})",
                estimated,
                allow_spillover=rc.allow_vram_spillover,
                device_id=gpu_id,
            ):
                logger.warning(
                    "Reranker VRAM check failed — will fall back to original scores"
                )
        except Exception as e:
            logger.debug(f"VRAM check skipped: {e}")

        _reranker = BGEReranker(device=device)
    return _reranker


def get_diversity_reranker(
    lambda_param: float = None,
    max_per_source: int = None,
) -> DiversityReranker:
    """
    Get or create diversity reranker with current config.

    Args:
        lambda_param: Override config lambda (0.0-1.0)
        max_per_source: Override config max per source

    Returns:
        DiversityReranker instance
    """
    global _diversity_reranker

    # Import config here to avoid circular imports
    from config import runtime_config

    # Use provided values or fall back to config
    final_lambda = lambda_param if lambda_param is not None else runtime_config.rag_diversity_lambda
    final_max = max_per_source if max_per_source is not None else runtime_config.rag_max_per_source

    # Create new instance if params changed or not initialized
    if _diversity_reranker is None or (
        _diversity_reranker.lambda_param != final_lambda or _diversity_reranker.max_per_source != final_max
    ):
        _diversity_reranker = DiversityReranker(
            lambda_param=final_lambda,
            max_per_source=final_max,
        )
        logger.info(f"DiversityReranker configured: lambda={final_lambda}, max_per_source={final_max}")

    return _diversity_reranker


def get_router() -> TopicRouter:
    global _router
    if _router is None:
        _router = TopicRouter()
    return _router


def warm_router():
    """Pre-warm router embeddings on startup"""
    router = get_router()
    router.warm_embeddings()
