"""
Hardware Intelligence — detection, profiling, VRAM budget, self-calibrating timeout.

Detects GPU/CPU/RAM at startup, assigns a hardware profile, and provides
VRAM budget checks before loading models. Also manages a self-calibrating
LLM startup timeout based on observed load times.

Usage:
    from services.hardware import get_hardware_info, check_vram_budget
    hw = get_hardware_info()
    check_vram_budget("embedder", estimated_vram_mb=600)
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from services.system_metrics import _get_static_specs, _get_gpu_metrics, _get_all_gpu_specs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware Simulation (set ARCA_SIMULATE_VRAM to test degraded hardware)
# ---------------------------------------------------------------------------

def _parse_simulate_vram() -> Optional[int]:
    """
    Parse ARCA_SIMULATE_VRAM env var.

    Accepts:
        "8192"   → 8192 MB
        "8g"     → 8192 MB
        "8gb"    → 8192 MB
        "0"      → 0 MB (CPU-only simulation)
        unset    → None (no simulation)
    """
    val = os.environ.get("ARCA_SIMULATE_VRAM")
    if not val:
        return None
    val = val.strip().lower()
    if val.endswith("gb"):
        return int(float(val[:-2].strip()) * 1024)
    if val.endswith("g"):
        return int(float(val[:-1].strip()) * 1024)
    return int(val)


def is_simulated() -> bool:
    """Check if hardware simulation is active."""
    return os.environ.get("ARCA_SIMULATE_VRAM") is not None


# VRAM estimates for known model components (MB)
# These are approximate runtime VRAM usage, not file sizes
VRAM_ESTIMATES: Dict[str, int] = {
    # Embedders (ONNX uses less, but budget for worst case)
    "Qwen/Qwen3-Embedding-0.6B": 800,
    # Rerankers
    "BAAI/bge-reranker-v2-m3": 600,
    # LLM models (by quantization — rough per-billion-param estimates)
    # Actual usage depends on ctx_size and batch, these are base model only
    "llm_q4_7b": 5000,
    "llm_q4_14b": 9000,
    "llm_q4_32b": 20000,
    "llm_q8_7b": 8000,
    "llm_q8_14b": 16000,
}

# Hardware profiles — VRAM tiers and their tuning defaults
PROFILES = {
    "cpu": {
        "description": "CPU-only / <4 GB VRAM",
        "ctx_size": 4096,
        "gpu_layers": 0,
        "batch_size": 8,
    },
    "small": {
        "description": "4-8 GB VRAM",
        "ctx_size": 4096,
        "gpu_layers": -1,
        "batch_size": 16,
    },
    "medium": {
        "description": "8-16 GB VRAM",
        "ctx_size": 8192,
        "gpu_layers": -1,
        "batch_size": 32,
    },
    "large": {
        "description": ">16 GB VRAM",
        "ctx_size": 16384,
        "gpu_layers": -1,
        "batch_size": 64,
    },
}

# ---------------------------------------------------------------------------
# Multi-GPU Device Map
# ---------------------------------------------------------------------------

# Default: all components on device 0 (single-GPU, identical to pre-multi-GPU)
_DEFAULT_DEVICE_MAP = {
    "chat": 0,
    "vision": 0,
    "expert": 0,
    "embedder": 0,
    "reranker": 0,
}


def _parse_device_map() -> Dict[str, int]:
    """Parse ARCA_DEVICE_MAP env var.

    Format: comma-separated component:device pairs.
    Example: ARCA_DEVICE_MAP=chat:0,vision:1,embedder:1,reranker:1
    Unspecified components default to device 0.
    """
    device_map = dict(_DEFAULT_DEVICE_MAP)
    raw = os.environ.get("ARCA_DEVICE_MAP", "")
    if not raw:
        return device_map
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            component, device_str = pair.split(":", 1)
            component = component.strip().lower()
            try:
                device_id = int(device_str.strip())
                if component in device_map:
                    device_map[component] = device_id
                else:
                    logger.warning(
                        f"Unknown component '{component}' in ARCA_DEVICE_MAP. "
                        f"Valid: {list(device_map.keys())}"
                    )
            except ValueError:
                logger.warning(f"Invalid device ID in ARCA_DEVICE_MAP: {pair}")
    return device_map


def get_gpu_for_component(component: str) -> int:
    """Get GPU device ID for a component.

    Components: chat, vision, expert, embedder, reranker.
    Returns 0 for unknown components (safe default).
    """
    return _parse_device_map().get(component, 0)


def get_gpu_count() -> int:
    """Get number of available NVIDIA GPUs."""
    from services.system_metrics import _get_all_gpu_specs
    return len(_get_all_gpu_specs())


# Model recommendations per VRAM tier
MODEL_RECOMMENDATIONS: Dict[str, Dict[str, str]] = {
    "cpu": {
        "chat": "7B Q4_K_S",
        "vision": "N/A",
        "expert": "N/A",
    },
    "small": {
        "chat": "7B Q4_K_M",
        "vision": "8B Q4_K_M (on-demand)",
        "expert": "Same as chat",
    },
    "medium": {
        "chat": "7B Q8_0 or 14B Q4_K_M",
        "vision": "8B Q8_0 (on-demand)",
        "expert": "32B Q4_K_M (swaps with chat)",
    },
    "large": {
        "chat": "14B Q8_0 or 7B Q8_0",
        "vision": "8B Q8_0",
        "expert": "32B Q4_K_M",
    },
}

# Directory for per-slot timeout baselines (one file per slot: chat, vision, etc.)
_TIMEOUT_BASELINE_DIR = Path(
    os.environ.get("HW_TIMEOUT_BASELINE_DIR", "/app/data/config")
)

# Hard cap for startup timeout regardless of baseline
TIMEOUT_HARD_CAP = 300.0


@dataclass
class HardwareInfo:
    """Detected hardware specifications and derived profile."""

    gpu_name: str = "Unknown"
    gpu_vram_total_mb: int = 0
    gpu_vram_available_mb: int = 0
    cpu_model: str = "Unknown"
    cpu_cores: int = 0
    ram_total_gb: float = 0
    profile: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        profile_info = PROFILES.get(self.profile, {})
        return {
            "gpu": {
                "name": self.gpu_name,
                "vram_total_mb": self.gpu_vram_total_mb,
                "vram_total_gb": round(self.gpu_vram_total_mb / 1024, 1),
                "vram_available_mb": self.gpu_vram_available_mb,
                "vram_available_gb": round(self.gpu_vram_available_mb / 1024, 1),
            },
            "cpu": {
                "model": self.cpu_model,
                "cores": self.cpu_cores,
            },
            "ram_total_gb": self.ram_total_gb,
            "profile": self.profile,
            "profile_description": profile_info.get("description", ""),
            "profile_defaults": {
                "ctx_size": profile_info.get("ctx_size"),
                "gpu_layers": profile_info.get("gpu_layers"),
                "batch_size": profile_info.get("batch_size"),
            },
            "profile_notes": "",
            "model_recommendations": MODEL_RECOMMENDATIONS.get(self.profile, {}),
        }


def _assign_profile(vram_mb: int) -> str:
    """Assign hardware profile based on total VRAM."""
    if vram_mb < 4096:
        return "cpu"
    elif vram_mb < 8192:
        return "small"
    elif vram_mb < 16384:
        return "medium"
    else:
        return "large"


def _detect_hardware() -> HardwareInfo:
    """Detect hardware specs using system_metrics helpers.

    If ARCA_SIMULATE_VRAM is set, returns synthetic hardware info instead of
    querying the real GPU. This lets you test how ARCA behaves on lesser hardware
    without actually having that hardware.
    """
    sim_vram = _parse_simulate_vram()
    if sim_vram is not None:
        sim_gpu = os.environ.get("ARCA_SIMULATE_GPU", "Simulated GPU")
        sim_ram = float(os.environ.get("ARCA_SIMULATE_RAM", "16"))
        sim_cpu = os.environ.get("ARCA_SIMULATE_CPU", "Simulated CPU")
        sim_cores = int(os.environ.get("ARCA_SIMULATE_CORES", "4"))

        info = HardwareInfo(
            gpu_name=sim_gpu,
            gpu_vram_total_mb=sim_vram,
            gpu_vram_available_mb=sim_vram,
            cpu_model=sim_cpu,
            cpu_cores=sim_cores,
            ram_total_gb=sim_ram,
            profile=_assign_profile(sim_vram),
        )

        logger.warning(
            f"HARDWARE SIMULATION ACTIVE: {sim_gpu} "
            f"({round(sim_vram / 1024, 1)} GB VRAM), "
            f"{sim_cores}-core, {sim_ram} GB RAM "
            f"-> profile: {info.profile}"
        )
        return info

    specs = _get_static_specs()
    gpu_metrics = _get_gpu_metrics()

    vram_total = specs.get("gpu_vram_total_mb", 0)
    vram_used = gpu_metrics.get("gpu_vram_used_mb") or 0
    vram_available = max(vram_total - vram_used, 0) if vram_total > 0 else 0

    info = HardwareInfo(
        gpu_name=specs.get("gpu_name", "Unknown"),
        gpu_vram_total_mb=vram_total,
        gpu_vram_available_mb=vram_available,
        cpu_model=specs.get("cpu_model", "Unknown"),
        cpu_cores=specs.get("cpu_cores", 0),
        ram_total_gb=specs.get("ram_total_gb", 0),
        profile=_assign_profile(vram_total),
    )

    logger.info(
        f"Hardware: {info.gpu_name} ({round(vram_total / 1024, 1)} GB VRAM), "
        f"{info.cpu_cores}-core, {info.ram_total_gb} GB RAM "
        f"-> profile: {info.profile}"
    )

    # Log multi-GPU inventory and device map
    all_gpus = _get_all_gpu_specs()
    if len(all_gpus) > 1:
        logger.info(f"Multi-GPU: {len(all_gpus)} GPUs detected")
        for gpu in all_gpus:
            logger.info(
                f"  GPU {gpu['index']}: {gpu['name']} "
                f"({round(gpu['vram_total_mb'] / 1024, 1)} GB)"
            )
        device_map = _parse_device_map()
        assignments = ", ".join(
            f"{comp}->GPU{dev}" for comp, dev in device_map.items()
        )
        logger.info(f"  Device map: {assignments}")

    return info


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_hardware_info: Optional[HardwareInfo] = None


def get_hardware_info() -> HardwareInfo:
    """Get or create the singleton HardwareInfo (detects on first call)."""
    global _hardware_info
    if _hardware_info is None:
        _hardware_info = _detect_hardware()
    return _hardware_info


def refresh_hardware_info() -> HardwareInfo:
    """Force re-detection (e.g., after loading/unloading a model)."""
    global _hardware_info
    _hardware_info = _detect_hardware()
    return _hardware_info


# ---------------------------------------------------------------------------
# VRAM Budget Enforcement (2B)
# ---------------------------------------------------------------------------


def get_vram_available_mb(device_id: int = 0) -> int:
    """Get current available VRAM in MB for a specific GPU.

    Args:
        device_id: GPU device index (default 0 for backward compat).

    In simulation mode, returns the simulated total (assumes nothing loaded yet).
    """
    sim_vram = _parse_simulate_vram()
    if sim_vram is not None:
        return sim_vram

    if device_id == 0:
        # Fast path: use existing single-GPU helpers (cached specs)
        gpu_metrics = _get_gpu_metrics()
        specs = _get_static_specs()
        vram_total = specs.get("gpu_vram_total_mb", 0)
        vram_used = gpu_metrics.get("gpu_vram_used_mb") or 0
        return max(vram_total - vram_used, 0)

    # Multi-GPU: query the specific device
    from services.system_metrics import _get_gpu_metrics_for_device
    gpu_metrics = _get_gpu_metrics_for_device(device_id)
    vram_used = gpu_metrics.get("gpu_vram_used_mb") or 0

    vram_total = 0
    for spec in _get_all_gpu_specs():
        if spec["index"] == device_id:
            vram_total = spec["vram_total_mb"]
            break

    return max(vram_total - vram_used, 0)


def check_vram_budget(
    component: str,
    estimated_vram_mb: int,
    allow_spillover: bool = False,
    device_id: int = 0,
) -> bool:
    """
    Check whether there's enough VRAM to load a component.

    Args:
        component: Human-readable name (e.g., "embedder", "reranker", "LLM chat slot")
        estimated_vram_mb: Estimated VRAM needed in MB
        allow_spillover: If True, warn but don't block when over budget
        device_id: GPU device index to check (default 0)

    Returns:
        True if safe to proceed, False if blocked

    Raises nothing — logs errors with actionable guidance.
    """
    hw = get_hardware_info()

    # CPU-only: no VRAM to check
    if hw.gpu_vram_total_mb == 0:
        logger.debug(f"VRAM check skipped for {component} (no GPU detected)")
        return True

    available = get_vram_available_mb(device_id)

    if available >= estimated_vram_mb:
        logger.debug(
            f"VRAM OK for {component}: needs ~{estimated_vram_mb} MB, "
            f"{available} MB available"
        )
        return True

    # Over budget
    shortfall = estimated_vram_mb - available
    msg = (
        f"VRAM budget exceeded for {component}: "
        f"needs ~{estimated_vram_mb} MB but only {available} MB available "
        f"(short by {shortfall} MB). "
        f"Try: use a smaller/more quantized model, unload unused models, "
        f"or reduce context size."
    )

    if allow_spillover:
        logger.warning(f"{msg} Proceeding anyway (allow_vram_spillover=true).")
        return True
    else:
        logger.error(
            f"{msg} Loading blocked. Set ALLOW_VRAM_SPILLOVER=true in .env "
            f"to force loading (model will spill to system RAM, slower)."
        )
        return False


def cuda_health_check() -> bool:
    """Quick CUDA sanity — allocate, compute, free. Returns False if poisoned."""
    try:
        import torch
        if not torch.cuda.is_available():
            return True  # No CUDA = nothing to check
        t = torch.zeros(16, device="cuda")
        t = t + 1
        assert t.sum().item() == 16
        del t
        torch.cuda.empty_cache()
        return True
    except Exception:
        logger.warning("CUDA health check FAILED — GPU context may be poisoned")
        return False


def estimate_gguf_vram_mb(gguf_path: Path, ctx_size: int = 8192) -> int:
    """
    Estimate VRAM usage for a GGUF model based on file size.

    Rough heuristic: VRAM ~ file_size * 1.1 + KV_cache_overhead.
    KV cache overhead is ~2 bytes per token per layer for Q8_0 cache.
    This is intentionally conservative (over-estimates by ~10-20%).
    """
    if not gguf_path.exists():
        return 0

    file_size_mb = gguf_path.stat().st_size / (1024 * 1024)

    # Model weights in VRAM (slightly more than file size due to metadata/alignment)
    weights_mb = file_size_mb * 1.1

    # KV cache estimate: conservative 256 MB for 8k ctx, scales linearly
    kv_cache_mb = (ctx_size / 8192) * 256

    return int(weights_mb + kv_cache_mb)


# ---------------------------------------------------------------------------
# Self-Calibrating LLM Timeout (2C)
# ---------------------------------------------------------------------------


def _baseline_path(slot_name: str) -> Path:
    """Get per-slot timeout baseline file path."""
    return _TIMEOUT_BASELINE_DIR / f"llm_timeout_{slot_name}.json"


def _load_timeout_baseline(slot_name: str) -> Optional[Dict[str, Any]]:
    """Load the timeout baseline for a specific slot from disk."""
    path = _baseline_path(slot_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "baseline_seconds" in data:
            return data
    except Exception as e:
        logger.warning(f"Failed to load timeout baseline for {slot_name}: {e}")
    return None


def _save_timeout_baseline(slot_name: str, baseline_seconds: float, model_name: str) -> None:
    """Save a new timeout baseline for a specific slot to disk."""
    try:
        path = _baseline_path(slot_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "baseline_seconds": round(baseline_seconds, 1),
            "model": model_name,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        logger.info(
            f"Timeout baseline saved: {baseline_seconds:.1f}s for {slot_name}/{model_name}"
        )
    except Exception as e:
        logger.warning(f"Failed to save timeout baseline for {slot_name}: {e}")


def get_calibrated_timeout(slot_name: str, model_name: str) -> float:
    """
    Get the startup timeout for the LLM, calibrated from previous loads.

    Each slot (chat, vision, expert) has its own baseline file so they
    don't clobber each other on startup.

    First boot: uses TIMEOUT_HARD_CAP (300s).
    Subsequent boots: baseline x4 (capped at TIMEOUT_HARD_CAP).
    Model change within slot: resets to TIMEOUT_HARD_CAP until new baseline recorded.
    """
    baseline = _load_timeout_baseline(slot_name)

    if baseline is None:
        logger.info(
            f"No timeout baseline for {slot_name}. Using hard cap: {TIMEOUT_HARD_CAP}s. "
            f"First successful load will calibrate future timeouts."
        )
        return TIMEOUT_HARD_CAP

    if baseline.get("model") != model_name:
        logger.info(
            f"Model changed for {slot_name} ({baseline.get('model')} -> {model_name}). "
            f"Resetting timeout to hard cap: {TIMEOUT_HARD_CAP}s."
        )
        return TIMEOUT_HARD_CAP

    base = baseline["baseline_seconds"]
    timeout = min(base * 4, TIMEOUT_HARD_CAP)

    logger.info(
        f"Calibrated timeout for {slot_name}: {timeout:.0f}s "
        f"(baseline {base:.1f}s x4, cap {TIMEOUT_HARD_CAP}s)"
    )

    return timeout


def record_load_time(slot_name: str, model_name: str, elapsed_seconds: float) -> None:
    """
    Record a successful LLM load time to calibrate future timeouts.

    Also logs warnings if load time is unusually slow compared to baseline.
    """
    baseline = _load_timeout_baseline(slot_name)

    if baseline and baseline.get("model") == model_name:
        old_base = baseline["baseline_seconds"]
        if elapsed_seconds > old_base * 4:
            logger.error(
                f"LLM {slot_name} load took {elapsed_seconds:.1f}s — 4x slower than "
                f"baseline ({old_base:.1f}s). Check for: thermal throttling, "
                f"background processes consuming VRAM, or disk I/O issues."
            )
        elif elapsed_seconds > old_base * 2:
            logger.warning(
                f"LLM {slot_name} load took {elapsed_seconds:.1f}s — 2x slower than "
                f"baseline ({old_base:.1f}s). This may indicate system load."
            )

    _save_timeout_baseline(slot_name, elapsed_seconds, model_name)


# ---------------------------------------------------------------------------
# Model Filename Validation (2D)
# ---------------------------------------------------------------------------


def validate_model_files() -> Dict[str, Any]:
    """
    Validate that configured GGUF model files exist in the models directory.

    Returns a dict with validation results per slot. Logs clear errors
    with what's missing and how to fix it.
    """
    from services.llm_config import SLOTS, MODELS_DIR, get_model_path

    results: Dict[str, Any] = {"valid": True, "slots": {}, "available_files": []}

    if not MODELS_DIR.exists():
        logger.error(
            f"Models directory not found: {MODELS_DIR}. "
            f"Create it and download GGUF model files. "
            f"See README.md for model download links."
        )
        results["valid"] = False
        results["error"] = f"Models directory not found: {MODELS_DIR}"
        return results

    available = sorted(f.name for f in MODELS_DIR.glob("*.gguf"))
    results["available_files"] = available

    if not available:
        logger.error(
            f"No GGUF files found in {MODELS_DIR}. "
            f"Download model files to ./models/ before starting. "
            f"See README.md for recommended models per hardware profile."
        )
        results["valid"] = False
        return results

    for slot_name, slot in SLOTS.items():
        model_path = get_model_path(slot.gguf_filename)
        slot_result = {
            "filename": slot.gguf_filename,
            "found": model_path.exists(),
            "always_running": slot.always_running,
        }

        if model_path.exists():
            size_gb = round(model_path.stat().st_size / (1024 ** 3), 1)
            slot_result["size_gb"] = size_gb
            logger.info(f"Model OK: {slot.gguf_filename} ({slot_name}, {size_gb} GB)")
        else:
            slot_result["size_gb"] = 0
            if slot.always_running:
                results["valid"] = False
                logger.error(
                    f"Model not found: '{slot.gguf_filename}' (slot: {slot_name}). "
                    f"Files in {MODELS_DIR}: {available}. "
                    f"Fix: update LLM_{slot_name.upper()}_MODEL in .env to match "
                    f"an available file, or download the missing model."
                )
            else:
                logger.warning(
                    f"On-demand model not found: '{slot.gguf_filename}' "
                    f"(slot: {slot_name}). It will fail when requested. "
                    f"Files in {MODELS_DIR}: {available}."
                )

        results["slots"][slot_name] = slot_result

    return results
