"""
Model bootstrap helpers for startup auto-download.

When enabled, ARCA can download missing GGUF model files for configured slots
at backend startup.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.llm_config import MODELS_DIR, SLOTS

logger = logging.getLogger(__name__)


# Filename -> Hugging Face repo fallback map.
# Prefer explicit env vars in production (LLM_<SLOT>_MODEL_REPO).
KNOWN_MODEL_REPOS: Dict[str, str] = {
    "Qwen3-8B-Q4_K_M.gguf": "unsloth/Qwen3-8B-GGUF",
    "Qwen3-14B-Q4_K_M.gguf": "unsloth/Qwen3-14B-GGUF",
    "Qwen3-30B-A3B-Q4_K_M.gguf": "unsloth/Qwen3-30B-A3B-GGUF",
    "GLM-4.7-Flash-Q4_K_M.gguf": "unsloth/GLM-4.7-Flash-GGUF",
    "Qwen2.5-7B-Instruct-Q4_K_M.gguf": "bartowski/Qwen2.5-7B-Instruct-GGUF",
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf": "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "Qwen3VL-8B-Instruct-Q4_K_M.gguf": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
    "Qwen3VL-8B-Instruct-Q8_0.gguf": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
    "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
}


def _is_truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_repo_for_slot(slot_name: str, filename: str) -> Optional[str]:
    """Resolve Hugging Face repo for a slot/model from env or known map."""
    slot_key = slot_name.upper()

    # Preferred explicit per-slot repo setting
    repo = os.environ.get(f"LLM_{slot_key}_MODEL_REPO")
    if repo:
        return repo

    # Backward-compatible chat defaults
    if slot_name == "chat":
        repo = os.environ.get("ARCA_DEFAULT_CHAT_REPO")
        if repo:
            return repo
    elif slot_name in {"code", "expert"}:
        # Common case: code/expert share the chat GGUF/repo.
        repo = os.environ.get("LLM_CHAT_MODEL_REPO")
        if repo:
            return repo
    elif slot_name == "vision_structured":
        repo = os.environ.get("LLM_VISION_MODEL_REPO")
        if repo:
            return repo

    # Generic fallback (ARCA_DEFAULT_<SLOT>_REPO)
    repo = os.environ.get(f"ARCA_DEFAULT_{slot_key}_REPO")
    if repo:
        return repo

    # Last resort: filename lookup table
    return KNOWN_MODEL_REPOS.get(filename)


def _download_gguf(
    models_dir: Path,
    repo_id: str,
    filename: str,
    token: str | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    models_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(models_dir),
        token=token,
    )
    return Path(downloaded)


def _looks_like_gguf(filename: str | None) -> bool:
    return bool(filename and filename.strip().lower().endswith(".gguf"))


def _resolve_mmproj_filename(slot_name: str) -> Optional[str]:
    """
    Resolve mmproj filename for vision slots.

    Uses LLM_VISION_MMPROJ when set, otherwise the default file expected by
    services.llm_config.
    """
    if slot_name not in {"vision", "vision_structured"}:
        return None

    env_mmproj = os.environ.get("LLM_VISION_MMPROJ", "").strip()
    if env_mmproj:
        return Path(env_mmproj).name

    return "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"


def _configured_models(auto_optional: bool) -> List[Tuple[str, str]]:
    """
    Build the model target list.

    Always includes always-running llama slots. When auto_optional=True, also
    includes configured on-demand runtime models (code/expert/vision_structured).
    """
    targets: List[Tuple[str, str]] = []

    for slot_name, slot in SLOTS.items():
        if slot.always_running or auto_optional:
            targets.append((slot_name, slot.gguf_filename))

    if not auto_optional:
        return targets

    # Include runtime-configured models that can be selected from admin.
    extra: Dict[str, str] = {}
    try:
        from config import runtime_config

        extra = {
            "code": runtime_config.model_code,
            "expert": runtime_config.model_expert,
            "vision_structured": runtime_config.model_vision_structured,
        }
    except Exception:
        # Early startup fallback: read env directly.
        extra = {
            "code": os.environ.get("LLM_CODE_MODEL", ""),
            "expert": os.environ.get("LLM_EXPERT_MODEL", ""),
            "vision_structured": (
                os.environ.get("LLM_VISION_STRUCTURED_MODEL")
                or os.environ.get("VISION_STRUCTURED_MODEL", "")
            ),
        }

    for slot_name, filename in extra.items():
        if _looks_like_gguf(filename):
            targets.append((slot_name, filename.strip()))

    return targets


def _ensure_one_model_file(
    result: Dict[str, Any],
    slot_name: str,
    filename: str,
    hf_token: str | None,
) -> Optional[str]:
    """Ensure one GGUF file exists; download it when missing."""
    model_path = MODELS_DIR / filename
    repo = _resolve_repo_for_slot(slot_name, filename)

    if model_path.exists():
        result["already_present"].append({"slot": slot_name, "model": filename})
        return repo

    if not repo:
        result["missing_repo"].append({"slot": slot_name, "model": filename})
        logger.warning(
            f"Auto-download skipped for {slot_name}: no repo mapping for {filename}. "
            f"Set LLM_{slot_name.upper()}_MODEL_REPO in .env."
        )
        return None

    try:
        logger.info(f"Auto-downloading model for {slot_name}: {repo}/{filename}")
        downloaded_path = _download_gguf(MODELS_DIR, repo, filename, token=hf_token)
        size_gb = downloaded_path.stat().st_size / (1024 ** 3)
        logger.info(f"Downloaded {downloaded_path.name} ({size_gb:.2f} GB)")
        result["downloaded"].append(
            {"slot": slot_name, "model": filename, "repo": repo, "size_gb": round(size_gb, 2)}
        )
        return repo
    except Exception as e:
        fallback_repo = KNOWN_MODEL_REPOS.get(filename)
        if fallback_repo and fallback_repo != repo:
            logger.warning(
                f"Auto-download failed for {slot_name} ({filename}) using {repo}: {e}. "
                f"Retrying with fallback repo {fallback_repo}."
            )
            try:
                downloaded_path = _download_gguf(MODELS_DIR, fallback_repo, filename, token=hf_token)
                size_gb = downloaded_path.stat().st_size / (1024 ** 3)
                logger.info(f"Downloaded {downloaded_path.name} ({size_gb:.2f} GB) from fallback repo")
                result["downloaded"].append(
                    {
                        "slot": slot_name,
                        "model": filename,
                        "repo": fallback_repo,
                        "size_gb": round(size_gb, 2),
                        "fallback_from": repo,
                    }
                )
                return fallback_repo
            except Exception as fallback_error:
                logger.warning(
                    f"Fallback auto-download also failed for {slot_name} ({filename}) "
                    f"using {fallback_repo}: {fallback_error}"
                )

        logger.warning(f"Auto-download failed for {slot_name} ({filename}): {e}")
        result["failed"].append({"slot": slot_name, "model": filename, "repo": repo, "error": str(e)})
        return None


def ensure_required_models(auto_optional: bool = False) -> Dict[str, Any]:
    """
    Ensure required GGUF model files exist.

    Downloads missing files for always-running slots. If auto_optional=True,
    also downloads configured on-demand models.
    """
    result: Dict[str, Any] = {
        "downloaded": [],
        "already_present": [],
        "missing_repo": [],
        "failed": [],
        "checked_slots": [],
    }

    if not MODELS_DIR.exists():
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Cannot create models directory {MODELS_DIR}: {e}")
            result["failed"].append({"slot": "global", "model": "", "error": str(e)})
            return result

    if not os.access(MODELS_DIR, os.W_OK):
        msg = (
            f"Models directory is not writable: {MODELS_DIR}. "
            "Auto-download disabled for this startup."
        )
        logger.warning(msg)
        result["failed"].append({"slot": "global", "model": "", "error": msg})
        return result

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

    slots_to_check = _configured_models(auto_optional=auto_optional)

    result["checked_slots"] = [name for name, _ in slots_to_check]

    seen_filenames: set[str] = set()
    for slot_name, filename in slots_to_check:
        if not _looks_like_gguf(filename):
            continue
        if filename in seen_filenames:
            continue
        seen_filenames.add(filename)

        repo = _ensure_one_model_file(
            result=result,
            slot_name=slot_name,
            filename=filename,
            hf_token=hf_token,
        )

        # Vision slots also need the matching mmproj GGUF.
        mmproj_filename = _resolve_mmproj_filename(slot_name)
        if mmproj_filename:
            if mmproj_filename in seen_filenames:
                continue
            seen_filenames.add(mmproj_filename)
            _ensure_one_model_file(
                result=result,
                slot_name=slot_name,
                filename=mmproj_filename,
                hf_token=hf_token,
            )

    return result


def bootstrap_models_from_env() -> Dict[str, Any]:
    """
    Entrypoint used by startup.

    Env:
      - ARCA_AUTO_DOWNLOAD_MODELS=true|false (default: true)
      - ARCA_AUTO_DOWNLOAD_OPTIONAL_MODELS=true|false (default: true)
    """
    enabled = _is_truthy(os.environ.get("ARCA_AUTO_DOWNLOAD_MODELS"), default=True)
    optional = _is_truthy(os.environ.get("ARCA_AUTO_DOWNLOAD_OPTIONAL_MODELS"), default=True)

    if not enabled:
        return {
            "enabled": False,
            "downloaded": [],
            "already_present": [],
            "missing_repo": [],
            "failed": [],
            "checked_slots": [],
        }

    out = ensure_required_models(auto_optional=optional)
    out["enabled"] = True
    out["auto_optional"] = optional
    return out
