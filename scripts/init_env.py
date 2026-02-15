#!/usr/bin/env python3
"""
Initialize .env for first-time setup.

- Copies .env.example to .env if missing.
- Replaces placeholder secrets with generated values.
- Optionally sets MCP_API_KEY for MCP integrations.
"""

from __future__ import annotations

import argparse
import secrets
import subprocess
import shutil
from pathlib import Path


PLACEHOLDERS = {
    "",
    "change-me",
    "change-me-in-production",
    "changeme",
    "replace-me",
    "your-password-here",
    "password",
}

MODEL_TIERS = [
    {
        "tier": "0-8GB",
        "min_mb": 0,
        "max_mb": 8 * 1024,
        "chat_model": "Qwen3-8B-Q4_K_M.gguf",
        "chat_repo": "unsloth/Qwen3-8B-GGUF",
        "vision_model": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "vision_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "note": "Entry tier: Qwen3 8B",
    },
    {
        "tier": "8-12GB",
        "min_mb": 8 * 1024,
        "max_mb": 12 * 1024,
        "chat_model": "Qwen3-8B-Q4_K_M.gguf",
        "chat_repo": "unsloth/Qwen3-8B-GGUF",
        "vision_model": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "vision_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "note": "Conservative tier: Qwen3 8B",
    },
    {
        "tier": "12-16GB",
        "min_mb": 12 * 1024,
        "max_mb": 16 * 1024,
        "chat_model": "Qwen3-14B-Q4_K_M.gguf",
        "chat_repo": "unsloth/Qwen3-14B-GGUF",
        "vision_model": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "vision_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "note": "Mid tier: Qwen3 14B",
    },
    {
        "tier": "16-24GB",
        "min_mb": 16 * 1024,
        "max_mb": 24 * 1024,
        "chat_model": "Qwen3-14B-Q4_K_M.gguf",
        "chat_repo": "unsloth/Qwen3-14B-GGUF",
        "vision_model": "Qwen3VL-8B-Instruct-Q8_0.gguf",
        "vision_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "note": "Balanced tier: Qwen3 14B + stronger vision quant",
    },
    {
        "tier": "24-32GB",
        "min_mb": 24 * 1024,
        "max_mb": 32 * 1024,
        "chat_model": "Qwen3-30B-A3B-Q4_K_M.gguf",
        "chat_repo": "unsloth/Qwen3-30B-A3B-GGUF",
        "vision_model": "Qwen3VL-8B-Instruct-Q8_0.gguf",
        "vision_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "note": "High tier: Qwen3 30B A3B",
    },
    {
        "tier": "32-48GB",
        "min_mb": 32 * 1024,
        "max_mb": 48 * 1024,
        "chat_model": "Qwen3-30B-A3B-Q4_K_M.gguf",
        "chat_repo": "unsloth/Qwen3-30B-A3B-GGUF",
        "vision_model": "Qwen3VL-8B-Instruct-Q8_0.gguf",
        "vision_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "note": "High tier: Qwen3 30B A3B",
    },
    {
        "tier": "48-96GB",
        "min_mb": 48 * 1024,
        "max_mb": 96 * 1024,
        "chat_model": "Qwen3-30B-A3B-Q4_K_M.gguf",
        "chat_repo": "unsloth/Qwen3-30B-A3B-GGUF",
        "vision_model": "Qwen3VL-8B-Instruct-Q8_0.gguf",
        "vision_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "note": "Enthusiast tier default; override for 72B/large MoE as desired",
    },
    {
        "tier": "96GB+",
        "min_mb": 96 * 1024,
        "max_mb": 10**9,
        "chat_model": "Qwen3-30B-A3B-Q4_K_M.gguf",
        "chat_repo": "unsloth/Qwen3-30B-A3B-GGUF",
        "vision_model": "Qwen3VL-8B-Instruct-Q8_0.gguf",
        "vision_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "note": "Safe default for very high VRAM; override to larger models if desired",
    },
]


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _find_key(lines: list[str], key: str) -> int | None:
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        lhs = line.split("=", 1)[0].strip()
        if lhs == key:
            return idx
    return None


def _get_value(lines: list[str], key: str) -> str | None:
    idx = _find_key(lines, key)
    if idx is None:
        return None
    return lines[idx].split("=", 1)[1].strip()


def _set_value(lines: list[str], key: str, value: str) -> None:
    idx = _find_key(lines, key)
    entry = f"{key}={value}"
    if idx is None:
        lines.append(entry)
    else:
        lines[idx] = entry


def _needs_rotation(value: str | None, force: bool) -> bool:
    if force:
        return True
    if value is None:
        return True
    return value.strip().lower() in PLACEHOLDERS


def _new_secret() -> str:
    # URL-safe secret with enough entropy for local service credentials.
    return secrets.token_urlsafe(24)


def _detect_gpu_inventory() -> dict[str, int]:
    """Best-effort NVIDIA inventory for first-run model tiering."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode != 0:
            return {"gpu_count": 0, "max_vram_mb": 0, "total_vram_mb": 0}

        vram_values: list[int] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                vram_values.append(int(line))
            except ValueError:
                continue

        if not vram_values:
            return {"gpu_count": 0, "max_vram_mb": 0, "total_vram_mb": 0}

        return {
            "gpu_count": len(vram_values),
            "max_vram_mb": max(vram_values),
            "total_vram_mb": sum(vram_values),
        }
    except Exception:
        return {"gpu_count": 0, "max_vram_mb": 0, "total_vram_mb": 0}


def _select_model_tier(gpu_inventory: dict[str, int]) -> dict[str, str]:
    """
    Tier by total VRAM (supports multi-GPU).

    Ranges:
    - 0-8 GB
    - 8-12 GB
    - 12-16 GB
    - 16-24 GB
    - 24-32 GB
    - 32-48 GB
    - 48-96 GB
    - 96+ GB
    """
    total_vram_mb = gpu_inventory.get("total_vram_mb", 0)
    for tier in MODEL_TIERS:
        if tier["min_mb"] <= total_vram_mb < tier["max_mb"]:
            return tier
    return MODEL_TIERS[0]


def initialize_env(
    env_path: Path,
    example_path: Path,
    force: bool,
    set_mcp_key: bool,
    auto_model_tier: bool,
) -> dict[str, str]:
    changed: dict[str, str] = {}
    created_env = False

    if not env_path.exists():
        if not example_path.exists():
            raise FileNotFoundError(f"Missing template: {example_path}")
        shutil.copy2(example_path, env_path)
        changed[".env"] = "created from .env.example"
        created_env = True

    lines = _read_lines(env_path)

    for key in ("POSTGRES_PASSWORD", "NEO4J_PASSWORD"):
        current = _get_value(lines, key)
        if _needs_rotation(current, force):
            _set_value(lines, key, _new_secret())
            changed[key] = "generated"

    if set_mcp_key:
        current = _get_value(lines, "MCP_API_KEY")
        if _needs_rotation(current, force):
            _set_value(lines, "MCP_API_KEY", _new_secret())
            changed["MCP_API_KEY"] = "generated"

    if auto_model_tier:
        chat_model = _get_value(lines, "LLM_CHAT_MODEL")
        chat_repo = _get_value(lines, "LLM_CHAT_MODEL_REPO")
        should_apply_tier = force or created_env or not chat_model or not chat_repo

        if should_apply_tier:
            inventory = _detect_gpu_inventory()
            tier = _select_model_tier(inventory)
            _set_value(lines, "LLM_CHAT_MODEL", tier["chat_model"])
            _set_value(lines, "LLM_CHAT_MODEL_REPO", tier["chat_repo"])
            _set_value(lines, "LLM_CODE_MODEL", tier["chat_model"])
            _set_value(lines, "LLM_CODE_MODEL_REPO", tier["chat_repo"])
            _set_value(lines, "LLM_EXPERT_MODEL", tier["chat_model"])
            _set_value(lines, "LLM_EXPERT_MODEL_REPO", tier["chat_repo"])
            _set_value(lines, "ARCA_DEFAULT_CHAT_FILE", tier["chat_model"])
            _set_value(lines, "ARCA_DEFAULT_CHAT_REPO", tier["chat_repo"])
            _set_value(lines, "LLM_VISION_MODEL", tier["vision_model"])
            _set_value(lines, "LLM_VISION_MODEL_REPO", tier["vision_repo"])
            _set_value(lines, "LLM_VISION_STRUCTURED_MODEL", tier["vision_model"])
            _set_value(lines, "LLM_VISION_STRUCTURED_MODEL_REPO", tier["vision_repo"])

            gpu_count = inventory.get("gpu_count", 0)
            total_vram_mb = inventory.get("total_vram_mb", 0)
            total_vram_gb = round(total_vram_mb / 1024, 1) if total_vram_mb else 0
            changed["MODEL_TIER"] = (
                f"{tier['tier']} ({tier['chat_model']}, {gpu_count} GPU(s), total VRAM ~{total_vram_gb} GB)"
            )

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize ARCA .env with safe defaults")
    parser.add_argument("--env-file", default=".env", help="Path to target .env file")
    parser.add_argument("--example-file", default=".env.example", help="Path to .env template")
    parser.add_argument("--force", action="store_true", help="Regenerate secrets even if set")
    parser.add_argument(
        "--set-mcp-key",
        action="store_true",
        help="Generate MCP_API_KEY when missing or placeholder",
    )
    parser.add_argument(
        "--no-auto-model-tier",
        action="store_true",
        help="Disable first-run GPU-based chat model tier selection",
    )
    args = parser.parse_args()

    env_path = Path(args.env_file).resolve()
    example_path = Path(args.example_file).resolve()

    changes = initialize_env(
        env_path=env_path,
        example_path=example_path,
        force=args.force,
        set_mcp_key=args.set_mcp_key,
        auto_model_tier=not args.no_auto_model_tier,
    )

    if not changes:
        print(".env already configured")
    else:
        for key, action in changes.items():
            print(f"{key}: {action}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
