#!/usr/bin/env python3
"""
Bootstrap GGUF model files for ARCA.

By default this script ensures all configured model slots exist locally:
- chat
- code
- expert
- vision
- vision_structured

It is safe to run repeatedly; existing files are not re-downloaded.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

KNOWN_MODEL_REPOS: dict[str, str] = {
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

DEFAULT_SLOT_MODELS: dict[str, str] = {
    "chat": "Qwen3-30B-A3B-Q4_K_M.gguf",
    "code": "Qwen3-30B-A3B-Q4_K_M.gguf",
    "expert": "Qwen3-30B-A3B-Q4_K_M.gguf",
    "vision": "Qwen3VL-8B-Instruct-Q8_0.gguf",
    "vision_structured": "Qwen3VL-8B-Instruct-Q8_0.gguf",
}


def _read_env_map(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _list_gguf(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("*.gguf"))


def _confirm_download(auto_yes: bool) -> bool:
    if auto_yes:
        return True
    answer = input("Download missing configured models now? [Y/n]: ").strip().lower()
    return answer in {"", "y", "yes"}


def _ensure_hf_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
        return
    except ImportError:
        print("huggingface_hub not installed; installing it now...")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "huggingface-hub"],
        check=True,
    )


def _download_model(models_dir: Path, repo_id: str, filename: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    models_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(models_dir),
        token=token,
    )
    return Path(downloaded)


def _resolve_slot_file(
    slot: str,
    env_values: Dict[str, str],
    chat_override: str | None,
) -> str:
    if slot == "chat" and chat_override:
        return chat_override

    env_key = f"LLM_{slot.upper()}_MODEL"
    value = env_values.get(env_key, "").strip()
    if value:
        return value

    if slot in {"code", "expert"}:
        chat_model = env_values.get("LLM_CHAT_MODEL", "").strip()
        if chat_model:
            return chat_model

    if slot == "vision_structured":
        vision_model = env_values.get("LLM_VISION_MODEL", "").strip()
        if vision_model:
            return vision_model

    return DEFAULT_SLOT_MODELS[slot]


def _resolve_slot_repo(
    slot: str,
    filename: str,
    env_values: Dict[str, str],
    chat_repo_override: str | None,
) -> str | None:
    if slot == "chat" and chat_repo_override:
        return chat_repo_override

    env_key = f"LLM_{slot.upper()}_MODEL_REPO"
    value = env_values.get(env_key, "").strip()
    if value:
        return value

    if slot in {"code", "expert"}:
        value = env_values.get("LLM_CHAT_MODEL_REPO", "").strip()
        if value:
            return value

    if slot == "vision_structured":
        value = env_values.get("LLM_VISION_MODEL_REPO", "").strip()
        if value:
            return value

    value = env_values.get("ARCA_DEFAULT_CHAT_REPO", "").strip()
    if slot == "chat" and value:
        return value

    return KNOWN_MODEL_REPOS.get(filename)


def _build_targets(
    env_values: Dict[str, str],
    chat_override: str | None,
    chat_repo_override: str | None,
) -> List[Tuple[str, str, str | None]]:
    slots = ["chat", "code", "expert", "vision", "vision_structured"]
    targets: List[Tuple[str, str, str | None]] = []
    seen_models: set[str] = set()

    for slot in slots:
        filename = _resolve_slot_file(slot, env_values, chat_override)
        if not filename.lower().endswith(".gguf"):
            continue
        if filename in seen_models:
            continue
        seen_models.add(filename)
        repo = _resolve_slot_repo(slot, filename, env_values, chat_repo_override)
        targets.append((slot, filename, repo))

        if slot in {"vision", "vision_structured"}:
            mmproj = Path(
                env_values.get("LLM_VISION_MMPROJ", "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf")
            ).name
            mmproj_repo = KNOWN_MODEL_REPOS.get(mmproj) or repo
            if mmproj.lower().endswith(".gguf") and mmproj not in seen_models:
                seen_models.add(mmproj)
                targets.append((slot, mmproj, mmproj_repo))

    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap GGUF models for ARCA")
    parser.add_argument("--models-dir", default="models", help="Directory containing GGUF files")
    parser.add_argument("--yes", action="store_true", help="Skip prompt and download automatically")
    parser.add_argument("--check-only", action="store_true", help="Only check for required GGUF files")
    parser.add_argument("--chat-repo", help="Override Hugging Face repo for chat model")
    parser.add_argument("--chat-file", help="Override GGUF filename for chat model")
    parser.add_argument("--env-file", default=".env", help="Environment file used for model defaults")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token for gated/private models",
    )
    args = parser.parse_args()

    env_path = Path(args.env_file).resolve()
    env_values = _read_env_map(env_path)
    targets = _build_targets(
        env_values=env_values,
        chat_override=args.chat_file,
        chat_repo_override=args.chat_repo,
    )

    models_dir = Path(args.models_dir).resolve()
    existing = _list_gguf(models_dir)
    if existing:
        print(f"Found {len(existing)} GGUF model(s) in {models_dir}")
        for model in existing:
            size_gb = model.stat().st_size / (1024 ** 3)
            print(f"- {model.name} ({size_gb:.2f} GB)")

    missing: List[Tuple[str, str, str | None]] = []
    for slot, filename, repo in targets:
        if not (models_dir / filename).exists():
            missing.append((slot, filename, repo))

    if not missing:
        print("All configured model files are already present.")
        return 0

    print("Missing configured model files:")
    for slot, filename, repo in missing:
        repo_note = repo if repo else "<repo not configured>"
        print(f"- {slot}: {filename} ({repo_note})")

    if args.check_only:
        return 1

    if not _confirm_download(args.yes):
        print("Model download skipped.")
        return 1

    _ensure_hf_hub()

    failures: List[str] = []
    for slot, filename, repo in missing:
        selected_repo = repo
        fallback_repo = KNOWN_MODEL_REPOS.get(filename)

        if not selected_repo:
            failures.append(
                f"{slot}/{filename}: repo missing (set LLM_{slot.upper()}_MODEL_REPO in .env)"
            )
            continue

        try:
            path = _download_model(
                models_dir=models_dir,
                repo_id=selected_repo,
                filename=filename,
                token=args.hf_token,
            )
        except Exception as exc:
            if fallback_repo and fallback_repo != selected_repo:
                print(
                    f"Download failed for {slot}/{filename} from {selected_repo}: {exc}\n"
                    f"Retrying with fallback repo: {fallback_repo}"
                )
                try:
                    path = _download_model(
                        models_dir=models_dir,
                        repo_id=fallback_repo,
                        filename=filename,
                        token=args.hf_token,
                    )
                    selected_repo = fallback_repo
                except Exception as fallback_exc:
                    failures.append(f"{slot}/{filename}: {fallback_exc}")
                    continue
            else:
                failures.append(f"{slot}/{filename}: {exc}")
                continue

        if not path.exists():
            failures.append(f"{slot}/{filename}: download returned no file")
            continue

        size_gb = path.stat().st_size / (1024 ** 3)
        print(f"Downloaded {path.name} ({size_gb:.2f} GB) from {selected_repo}")

    if failures:
        print("Some model downloads failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
