"""
Cohesionn Embeddings - sentence-transformers embedding engine

Supports Qwen3-Embedding-0.6B (default), Nomic-Embed-Text-v1.5, and BGE-M3
with automatic instruction prefix handling based on model family.

Backend selection:
- ONNX (default): Eliminates PyTorch CUDA context (~17 GB host RAM savings)
- PyTorch: Legacy fallback, set COHESIONN_EMBED_BACKEND=torch

Features:
- 1024 dimensions (Qwen3, BGE) / Matryoshka embeddings (Nomic)
- 32K token context (Qwen3) / 8K (Nomic, BGE)
- Instruction-aware query embedding
- LRU caching for query embeddings
"""

import logging
import multiprocessing
import os
import time
from typing import List, Optional, Tuple, Union
from functools import lru_cache
from uuid import uuid4

logger = logging.getLogger(__name__)

# Benchmark-validated defaults (shootout_20260208_211323)
# Cross-matrix champion pairing: qwen3-06b + jina-v2-turbo (0.752 composite)
EMBEDDING_MODEL = os.environ.get("COHESIONN_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# Cache config - 10,000 entries ~40MB memory for typical 1024-dim embeddings
QUERY_CACHE_SIZE = int(os.environ.get("COHESIONN_QUERY_CACHE_SIZE", "10000"))


def _suppress_ort_warnings():
    """Suppress verbose ONNX Runtime C++ warnings (memcpy nodes, EP assignments)."""
    try:
        import onnxruntime
        # Severity levels: 0=verbose, 1=info, 2=warning, 3=error, 4=fatal
        onnxruntime.set_default_logger_severity(3)
    except (ImportError, AttributeError):
        pass


_suppress_ort_warnings()


def _detect_backend() -> str:
    """Detect best available backend: ONNX (preferred) or PyTorch."""
    env = os.environ.get("COHESIONN_EMBED_BACKEND", "auto")
    if env != "auto":
        return env
    try:
        import onnxruntime
        logger.info(f"ONNX Runtime available (providers: {onnxruntime.get_available_providers()})")
        return "onnx"
    except ImportError:
        return "torch"


EMBEDDING_BACKEND = _detect_backend()


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
            logger.warning(f"CUDA runtime unavailable for embeddings, using CPU: {e}")
        return False


def _detect_device() -> str:
    """Auto-detect best available device based on backend."""
    env_device = os.environ.get("COHESIONN_EMBED_DEVICE", "cuda")
    if env_device and env_device.lower() != "auto":
        if env_device.lower() == "cuda" and not _cuda_runtime_available(log_failure=True):
            logger.warning("COHESIONN_EMBED_DEVICE=cuda requested but CUDA is unavailable. Falling back to CPU.")
            return "cpu"
        return env_device.lower()

    if EMBEDDING_BACKEND == "onnx":
        if _onnx_cuda_provider_available() and _cuda_runtime_available():
            logger.info("ONNX CUDA available for embeddings")
            return "cuda"
        return "cpu"

    if _cuda_runtime_available():
        try:
            import torch
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available for embeddings: {device_name}")
        except Exception:
            pass
        return "cuda"
    return "cpu"


EMBEDDING_DEVICE = _detect_device()


def _detect_model_family(model_name: str) -> str:
    """Detect model family for instruction handling."""
    model_lower = model_name.lower()
    if "qwen3" in model_lower or "qwen/qwen3" in model_lower:
        return "qwen3"
    elif "bge" in model_lower:
        return "bge"
    elif "nomic" in model_lower:
        return "nomic"
    return "generic"


class UniversalEmbedder:
    """
    Universal embedding model supporting Qwen3-Embedding and BGE-M3.

    Automatically detects model family and applies correct instruction format:
    - Qwen3: Uses prompt_name="query" (built-in instruction)
    - BGE: Uses string prefix for queries
    - Nomic: Uses search_query/search_document prefixes

    Features:
    - 1024 dimensions (default for both Qwen3 and BGE)
    - 32K token context (Qwen3) / 8K (BGE)
    - LRU cache for query embeddings (10,000 entries ~40MB)
    """

    # Legacy prefix for BGE models
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = None, device: str = None, backend: str = None, trust_remote_code: bool = False, device_id: int = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.device = device or EMBEDDING_DEVICE
        self.backend = backend or EMBEDDING_BACKEND
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._dimension = None
        self._model_family = _detect_model_family(self.model_name)
        # GPU device index from device map (multi-GPU support)
        if device_id is not None:
            self._device_id = device_id
        else:
            try:
                from services.hardware import get_gpu_for_component
                self._device_id = get_gpu_for_component("embedder")
            except Exception:
                self._device_id = 0

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                if self.backend == "onnx":
                    if self.device == "cuda" and not _cuda_runtime_available(log_failure=True):
                        logger.warning("ONNX CUDA requested for embeddings but CUDA is unavailable. Switching to CPU.")
                        self.device = "cpu"
                    provider = "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
                    # Multi-GPU: route to assigned device
                    model_kwargs: dict = {"provider": provider}
                    if provider == "CUDAExecutionProvider" and self._device_id != 0:
                        model_kwargs["provider_options"] = {"device_id": self._device_id}
                        logger.info(f"ONNX embedder targeting GPU {self._device_id}")
                    cache_dir = _onnx_cache_dir(self.model_name)
                    try:
                        # Check for cached ONNX export first (avoids PyTorch re-export)
                        if os.path.isfile(os.path.join(cache_dir, "modules.json")):
                            logger.info(f"Loading embedder from ONNX cache: {cache_dir}")
                            self._model = SentenceTransformer(
                                cache_dir,
                                backend="onnx",
                                model_kwargs=model_kwargs,
                                trust_remote_code=self.trust_remote_code,
                            )
                        else:
                            logger.info(f"Loading embedding model: {self.model_name} (ONNX, provider={provider})")
                            self._model = SentenceTransformer(
                                self.model_name,
                                backend="onnx",
                                model_kwargs=model_kwargs,
                                trust_remote_code=self.trust_remote_code,
                            )
                            # Cache the export for next startup
                            try:
                                self._model.save(cache_dir)
                                logger.info(f"Cached ONNX export to: {cache_dir}")
                            except Exception as save_err:
                                logger.warning(f"Failed to cache ONNX export: {save_err}")

                        self._patch_position_ids(self._model)
                        test_emb = self._model.encode("test", show_progress_bar=False)
                        self._dimension = len(test_emb)
                    except Exception as e:
                        logger.info(f"ONNX unavailable for {self.model_name}; using PyTorch backend.")
                        logger.debug(f"ONNX fallback reason for {self.model_name}: {e}")
                        self._model = None
                        self.backend = "torch"

                if self.backend != "onnx":
                    if self._model is None:
                        device = self.device
                        if device == "cuda" and not _cuda_runtime_available(log_failure=True):
                            logger.warning("PyTorch CUDA requested for embeddings but CUDA is unavailable. Switching to CPU.")
                            device = "cpu"
                            self.device = "cpu"
                        logger.info(f"Loading embedding model: {self.model_name} (PyTorch, device={device})")
                        self._model = SentenceTransformer(
                            self.model_name,
                            device=device,
                            trust_remote_code=self.trust_remote_code,
                        )
                    if self._dimension is None:
                        test_emb = self._model.encode("test", show_progress_bar=False)
                        self._dimension = len(test_emb)

                logger.info(f"Embedder loaded: {self._dimension} dimensions, backend={self.backend}")

            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            _ = self.model  # Trigger load
        return self._dimension

    @property
    def model_family(self) -> str:
        """Get detected model family (qwen3, bge, nomic, generic)"""
        return self._model_family

    @property
    def cache_stats(self) -> dict:
        """Get cache hit/miss statistics from lru_cache"""
        info = self._embed_query_cached.cache_info()
        total = info.hits + info.misses
        hit_rate = info.hits / total if total > 0 else 0
        return {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "maxsize": info.maxsize,
            "hit_rate": f"{hit_rate:.1%}",
        }

    @staticmethod
    def _patch_position_ids(model):
        """Inject position_ids into tokenizer output for decoder-only ONNX models.

        Decoder-only models (Qwen3, GPT, etc.) require position_ids but the HuggingFace
        tokenizer doesn't generate them. This patches the Transformer module's tokenize
        method to add position_ids = [0, 1, 2, ..., seq_len-1] automatically.
        """
        import torch

        transformer = model[0]
        onnx_inputs = set()
        if hasattr(transformer, "auto_model") and hasattr(transformer.auto_model, "inputs_names"):
            onnx_inputs = set(transformer.auto_model.inputs_names)
        elif hasattr(transformer, "auto_model") and hasattr(transformer.auto_model, "model"):
            try:
                sess = transformer.auto_model.model
                onnx_inputs = {inp.name for inp in sess.get_inputs()}
            except Exception:
                pass

        if "position_ids" not in onnx_inputs:
            return

        original_tokenize = transformer.tokenize

        def tokenize_with_position_ids(texts, **kwargs):
            features = original_tokenize(texts, **kwargs)
            if "input_ids" in features:
                batch_size, seq_len = features["input_ids"].shape
                features["position_ids"] = (
                    torch.arange(seq_len, dtype=features["input_ids"].dtype)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                    .contiguous()
                )
            return features

        transformer.tokenize = tokenize_with_position_ids
        logger.info(f"Patched tokenizer with position_ids for decoder-only ONNX model")

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query with caching.
        Uses model-appropriate instruction format.
        """
        result = self._embed_query_cached(query)
        return list(result)  # Return new list to prevent mutation

    @lru_cache(maxsize=QUERY_CACHE_SIZE)
    def _embed_query_cached(self, query: str) -> Tuple[float, ...]:
        """
        Internal cached embedding method.
        Returns tuple for hashability in cache.
        """
        if self._model_family == "qwen3":
            # Qwen3 uses built-in prompt_name for query instruction
            embedding = self.model.encode(query, prompt_name="query", normalize_embeddings=True)
        elif self._model_family == "bge":
            # BGE uses string prefix
            prefixed = f"{self.BGE_QUERY_PREFIX}{query}"
            embedding = self.model.encode(prefixed, normalize_embeddings=True)
        else:
            # Generic/other - no prefix
            embedding = self.model.encode(query, normalize_embeddings=True)

        return tuple(embedding.tolist())

    def clear_cache(self):
        """Clear the query embedding cache"""
        self._embed_query_cached.cache_clear()
        logger.info("Query embedding cache cleared")

    def unload(self):
        """Explicitly free model from GPU/RAM. Call between benchmark model swaps."""
        if self._model is not None:
            del self._model
            self._model = None
        self._embed_query_cached.cache_clear()
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
        logger.info(f"Embedder {self.model_name} unloaded")

    def embed_document(self, document: str) -> List[float]:
        """
        Embed a document chunk.
        No prefix/prompt needed for documents.
        """
        embedding = self.model.encode(document, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, documents: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Embed multiple documents efficiently.
        """
        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_queries(self, queries: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Embed multiple queries with model-appropriate instruction format.
        """
        if self._model_family == "qwen3":
            # Qwen3 uses prompt_name for queries
            embeddings = self.model.encode(
                queries,
                prompt_name="query",
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        elif self._model_family == "bge":
            # BGE uses string prefix
            prefixed = [f"{self.BGE_QUERY_PREFIX}{q}" for q in queries]
            embeddings = self.model.encode(
                prefixed,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        else:
            embeddings = self.model.encode(
                queries,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        return embeddings.tolist()


# Alias for backward compatibility
BGEEmbedder = UniversalEmbedder


class EmbeddingProxy:
    """Delegates embedding to a subprocess with isolated CUDA context.

    Drop-in replacement for UniversalEmbedder. Same public API.
    If the worker crashes (CUDA std::terminate), the parent survives
    and respawns it. After 3 crashes in 120s, falls back to CPU.
    """

    def __init__(self, force_cpu=False, device_id=None):
        self._force_cpu = force_cpu
        self._device_id = device_id
        self._process = None
        self._req_queue = None
        self._resp_queue = None
        self._dimension = None
        self._crash_times: list = []
        self._cpu_fallback = False
        self._start_worker()

    def _start_worker(self):
        ctx = multiprocessing.get_context("spawn")
        self._req_queue = ctx.Queue()
        self._resp_queue = ctx.Queue()

        from .embed_worker import worker_main

        self._process = ctx.Process(
            target=worker_main,
            args=(
                self._req_queue,
                self._resp_queue,
                self._force_cpu or self._cpu_fallback,
                self._device_id,
                logging.getLogger().level,
            ),
            daemon=True,
        )
        self._process.start()
        logger.info(f"Embedding worker spawned (PID {self._process.pid})")
        # Fetch dimension to verify worker is alive
        try:
            resp = self._call("dimension")
            self._dimension = resp["dimension"]
        except RuntimeError as e:
            if not (self._force_cpu or self._cpu_fallback) and self._looks_like_cuda_error(str(e)):
                logger.warning("Embedding worker CUDA init failed; retrying with CPU fallback.")
                self._cpu_fallback = True
                if self._process is not None and self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=2)
                self._start_worker()
                return
            raise

    @staticmethod
    def _looks_like_cuda_error(message: str) -> bool:
        msg = message.lower()
        return any(
            token in msg
            for token in (
                "cuda",
                "nvidia",
                "driver",
                "cudnn",
                "cublas",
                "torch._c._cuda_init",
            )
        )

    def _call(self, msg_type, **kwargs):
        if self._process is None or not self._process.is_alive():
            self._handle_crash()
        req_id = uuid4().hex
        self._req_queue.put({"id": req_id, "type": msg_type, **kwargs})
        try:
            resp = self._resp_queue.get(timeout=300)
        except Exception:
            # Worker probably died mid-request
            self._handle_crash()
            # Retry once after respawn
            req_id = uuid4().hex
            self._req_queue.put({"id": req_id, "type": msg_type, **kwargs})
            resp = self._resp_queue.get(timeout=300)
        if resp.get("error"):
            raise RuntimeError(f"Embedding worker error: {resp['error']}")
        return resp

    def _handle_crash(self):
        now = time.time()
        self._crash_times = [t for t in self._crash_times if now - t < 120]
        self._crash_times.append(now)
        if len(self._crash_times) >= 3:
            logger.error(
                "Embedding worker crashed 3x in 120s — falling back to CPU"
            )
            self._cpu_fallback = True
        # Kill zombie if still around
        if self._process is not None and self._process.is_alive():
            self._process.kill()
        logger.warning("Embedding worker died, respawning...")
        self._start_worker()

    def embed_query(self, text: str) -> List[float]:
        return self._call("embed_query", text=text)["vector"]

    def embed_document(self, document: str) -> List[float]:
        return self._call("embed_documents", texts=[document])["vectors"][0]

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        return self._call("embed_documents", texts=texts, batch_size=batch_size)["vectors"]

    def embed_queries(self, queries: List[str], batch_size: int = 64) -> List[List[float]]:
        return self._call("embed_queries", texts=queries, batch_size=batch_size)["vectors"]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_family(self) -> str:
        return _detect_model_family(EMBEDDING_MODEL)

    @property
    def cache_stats(self) -> dict:
        # Subprocess handles its own caching; return placeholder
        return {"hits": 0, "misses": 0, "size": 0, "maxsize": 0, "hit_rate": "N/A (subprocess)"}

    def clear_cache(self):
        pass  # Cache is in the subprocess; cleared on respawn

    def health_check(self) -> bool:
        """Check if the worker subprocess is alive and responsive."""
        try:
            resp = self._call("health")
            return resp.get("healthy", False)
        except Exception:
            return False

    def unload(self):
        """Shut down the worker subprocess."""
        if self._process and self._process.is_alive():
            try:
                self._req_queue.put({"id": "shutdown", "type": "shutdown"})
                self._process.join(timeout=10)
            except Exception:
                pass
            if self._process.is_alive():
                self._process.kill()
        self._process = None
        logger.info("Embedding worker stopped")


# Singleton embedder
_embedder: Optional[Union[UniversalEmbedder, EmbeddingProxy]] = None


def get_embedder() -> Union[UniversalEmbedder, EmbeddingProxy]:
    """Get or create singleton embedder.

    By default, uses EmbeddingProxy (subprocess isolation) to survive CUDA
    crashes. Set ARCA_EMBED_SUBPROCESS=false for in-process mode (debugging).
    """
    global _embedder

    if _embedder is None:
        use_subprocess = os.environ.get(
            "ARCA_EMBED_SUBPROCESS", "true"
        ).lower() == "true"

        if use_subprocess:
            try:
                from services.hardware import get_gpu_for_component
                device_id = get_gpu_for_component("embedder")
            except Exception:
                device_id = 0
            logger.info("Creating subprocess-isolated embedder (EmbeddingProxy)")
            _embedder = EmbeddingProxy(device_id=device_id)
        else:
            # In-process mode — original behavior
            logger.info("Creating in-process embedder (ARCA_EMBED_SUBPROCESS=false)")
            try:
                from services.hardware import check_vram_budget, VRAM_ESTIMATES, get_gpu_for_component
                from config import runtime_config
                gpu_id = get_gpu_for_component("embedder")
                estimated = VRAM_ESTIMATES.get(EMBEDDING_MODEL, 800)
                if not check_vram_budget(
                    f"embedder ({EMBEDDING_MODEL})",
                    estimated,
                    allow_spillover=runtime_config.allow_vram_spillover,
                    device_id=gpu_id,
                ):
                    logger.warning(
                        "Embedder VRAM check failed — loading anyway on CPU fallback"
                    )
            except Exception as e:
                logger.debug(f"VRAM check skipped: {e}")
            _embedder = UniversalEmbedder()

        # Verify it works
        _ = _embedder.dimension

    return _embedder
