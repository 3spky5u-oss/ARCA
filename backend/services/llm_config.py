"""
LLM Server Configuration - Model slots, ports, and GGUF mappings.

Defines the available model slots and their configuration for llama-server instances.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelSlot:
    """Configuration for a single llama-server slot."""
    name: str
    port: int
    gguf_filename: str  # e.g. "Qwen3-30B-A3B-Q4_K_M.gguf"
    ctx_size: int = 24576
    n_gpu_layers: int = -1  # -1 = offload all to GPU
    n_cpu_moe: Optional[int] = None  # llama.cpp MoE experts kept on CPU RAM (e.g., --n-cpu-moe 16)
    always_running: bool = True
    split_mode: str = "none"  # none|layer|row (llama.cpp multi-GPU split strategy)
    tensor_split: Optional[str] = None  # e.g. "1,1" (weights per GPU ratio)
    cache_type_k: str = "q8_0"  # KV cache quantization (q8_0 = half memory vs f16, minimal quality loss)
    cache_type_v: str = "q8_0"  # Don't go below q8_0 for engineering precision
    mlock: bool = False  # Lock model weights in RAM (prevents paging, requires sufficient RAM)
    extra_args: list = field(default_factory=list)  # e.g. ["--jinja", "--tool-call-parser", "glm47"]


def _parse_split_mode(raw: Optional[str], default: str = "none") -> str:
    """Parse llama.cpp split mode with safe fallback."""
    value = (raw or default).strip().lower()
    allowed = {"none", "layer", "row"}
    if value not in allowed:
        logger.warning(
            f"Invalid split mode '{raw}'. Valid values: {sorted(allowed)}. Using '{default}'."
        )
        return default
    return value


def _parse_tensor_split(raw: Optional[str]) -> Optional[str]:
    """Parse and normalize llama.cpp tensor split ratio string."""
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None

    normalized: list[str] = []
    for part in parts:
        try:
            value = float(part)
        except ValueError:
            logger.warning(
                f"Invalid tensor split value '{part}' in '{raw}'. "
                "Expected comma-separated positive numbers (e.g., 1,1 or 3,1)."
            )
            return None
        if value <= 0:
            logger.warning(
                f"Invalid tensor split value '{part}' in '{raw}'. "
                "Values must be > 0."
            )
            return None
        normalized.append(str(int(value)) if value.is_integer() else str(value))

    return ",".join(normalized)


def _parse_non_negative_int(raw: Optional[str], *, env_name: str) -> Optional[int]:
    """Parse optional non-negative integer env var with safe fallback."""
    if raw is None:
        return None
    value = raw.strip()
    if value == "":
        return None
    try:
        parsed = int(value)
    except ValueError:
        logger.warning(f"Invalid {env_name}='{raw}'. Expected non-negative integer; ignoring.")
        return None
    if parsed < 0:
        logger.warning(f"Invalid {env_name}='{raw}'. Value must be >= 0; ignoring.")
        return None
    return parsed


# Default slot definitions
SLOTS: Dict[str, ModelSlot] = {
    "chat": ModelSlot(
        name="chat",
        port=int(os.environ.get("LLM_CHAT_PORT", "8081")),
        gguf_filename=os.environ.get("LLM_CHAT_MODEL", "Qwen3-30B-A3B-Q4_K_M.gguf"),
        ctx_size=int(os.environ.get("LLM_CHAT_CTX", "8192")),
        n_gpu_layers=int(os.environ.get("LLM_CHAT_GPU_LAYERS", "-1")),
        n_cpu_moe=_parse_non_negative_int(
            os.environ.get("LLM_CHAT_N_CPU_MOE", os.environ.get("LLM_N_CPU_MOE")),
            env_name="LLM_CHAT_N_CPU_MOE",
        ),
        always_running=os.environ.get("LLM_CHAT_ALWAYS_RUNNING", "true").lower() == "true",
        split_mode=_parse_split_mode(
            os.environ.get("LLM_CHAT_SPLIT_MODE", os.environ.get("LLM_SPLIT_MODE", "none"))
        ),
        tensor_split=_parse_tensor_split(
            os.environ.get("LLM_CHAT_TENSOR_SPLIT", os.environ.get("LLM_TENSOR_SPLIT"))
        ),
        cache_type_k=os.environ.get("LLM_CACHE_TYPE_K", "q8_0"),
        cache_type_v=os.environ.get("LLM_CACHE_TYPE_V", "q8_0"),
        mlock=os.environ.get("LLM_MLOCK", "false").lower() == "true",
        extra_args=[],  # jinja enabled by default in latest llama.cpp
    ),
    "vision": ModelSlot(
        name="vision",
        port=int(os.environ.get("LLM_VISION_PORT", "8082")),
        gguf_filename=os.environ.get("LLM_VISION_MODEL", "Qwen3VL-8B-Instruct-Q8_0.gguf"),
        ctx_size=int(os.environ.get("LLM_VISION_CTX", "4096")),
        n_gpu_layers=int(os.environ.get("LLM_VISION_GPU_LAYERS", "-1")),
        n_cpu_moe=_parse_non_negative_int(
            os.environ.get("LLM_VISION_N_CPU_MOE", os.environ.get("LLM_N_CPU_MOE")),
            env_name="LLM_VISION_N_CPU_MOE",
        ),
        always_running=os.environ.get("LLM_VISION_ALWAYS_RUNNING", "false").lower() == "true",
        split_mode=_parse_split_mode(
            os.environ.get("LLM_VISION_SPLIT_MODE", os.environ.get("LLM_SPLIT_MODE", "none"))
        ),
        tensor_split=_parse_tensor_split(
            os.environ.get("LLM_VISION_TENSOR_SPLIT", os.environ.get("LLM_TENSOR_SPLIT"))
        ),
        mlock=os.environ.get("LLM_MLOCK", "false").lower() == "true",
        cache_type_k=os.environ.get("LLM_VISION_CACHE_TYPE_K", "q8_0"),
        cache_type_v=os.environ.get("LLM_VISION_CACHE_TYPE_V", "q8_0"),
        extra_args=[
            "--mmproj", os.environ.get(
                "LLM_VISION_MMPROJ",
                str(Path(os.environ.get("LLM_MODELS_DIR", "/models")) / "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf")
            ),
            "--parallel", os.environ.get("LLM_VISION_PARALLEL", "1"),
        ],
    ),
}

# Expert model (swaps chat slot on-demand)
EXPERT_GGUF = os.environ.get("LLM_EXPERT_MODEL", "qwen3-32b-q4_k_m.gguf")

# Models directory (bind-mounted from host)
MODELS_DIR = Path(os.environ.get("LLM_MODELS_DIR", "/models"))

# llama-server binary path
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "/usr/local/bin/llama-server")


def get_slot(name: str) -> Optional[ModelSlot]:
    """Get a slot configuration by name."""
    return SLOTS.get(name)


def get_model_path(gguf_filename: str) -> Path:
    """Get full path to a GGUF model file."""
    return MODELS_DIR / gguf_filename


def list_available_models() -> list:
    """List all GGUF files in the models directory."""
    if not MODELS_DIR.exists():
        return []
    return sorted([f.name for f in MODELS_DIR.glob("*.gguf")])
