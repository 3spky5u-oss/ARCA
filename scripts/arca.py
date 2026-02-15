#!/usr/bin/env python3
"""
ARCA local operator CLI.

Provides one-command helpers for bootstrap, update, and health checks.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

RUNTIME_DIRS = [
    "models",
    "data/auth",
    "data/benchmarks",
    "data/cohesionn_db",
    "data/config",
    "data/core_knowledge",
    "data/guidelines",
    "data/hf_cache",
    "data/maps",
    "data/neo4j",
    "data/neo4j_logs",
    "data/phii",
    "data/postgres",
    "data/qdrant_storage",
    "data/redis",
    "data/searxng",
    "data/Synthetic Reports",
    "data/technical_knowledge",
    "data/test_knowledge",
    "backend/training",
]


def _run(command: list[str], check: bool = True) -> int:
    print("$", " ".join(command))
    result = subprocess.run(command, cwd=REPO_ROOT)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")
    return result.returncode


def _compose(args: list[str]) -> list[str]:
    # Docker Compose v5 may ignore GPU reservations unless compatibility mode is enabled.
    return ["docker", "compose", "--compatibility", *args]


def _python_script(script_name: str, extra_args: list[str] | None = None) -> list[str]:
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _ensure_runtime_dirs() -> None:
    for relative in RUNTIME_DIRS:
        path = REPO_ROOT / relative
        path.mkdir(parents=True, exist_ok=True)


def _enable_gpu_overlay() -> None:
    src = REPO_ROOT / "docker-compose.gpu.yml"
    dst = REPO_ROOT / "docker-compose.override.yml"

    if not src.exists():
        raise RuntimeError("docker-compose.gpu.yml not found")

    if dst.exists():
        print("GPU override already present: docker-compose.override.yml")
        return

    shutil.copy2(src, dst)
    print("Created docker-compose.override.yml from GPU overlay")


def _read_dotenv_flag(key: str, default: bool = False) -> bool:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return default
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() != key:
                continue
            normalized = value.strip().strip('"').strip("'").lower()
            return normalized in ("1", "true", "yes", "on")
    except Exception:
        return default
    return default


def cmd_bootstrap(args: argparse.Namespace) -> int:
    init_args = []
    if args.with_mcp_key:
        init_args.append("--set-mcp-key")
    _run(_python_script("init_env.py", init_args))

    if args.gpu and args.cpu:
        raise RuntimeError("Use either --gpu or --cpu, not both.")

    if args.cpu:
        print(
            "CPU mode requested (--cpu): ARCA compose defaults are GPU-first.\n"
            "Set COHESIONN_EMBED_DEVICE=cpu, COHESIONN_RERANK_DEVICE=cpu,\n"
            "LLM_CHAT_GPU_LAYERS=0, and LLM_VISION_GPU_LAYERS=0 in .env to force CPU execution."
        )
    elif args.gpu:
        print("GPU mode is default; --gpu is a no-op.")

    mcp_mode_enabled = _read_dotenv_flag("MCP_MODE", default=False)
    if mcp_mode_enabled and not args.skip_model_download:
        print("MCP_MODE=true in .env: skipping model bootstrap download.")
    elif not args.skip_model_download:
        model_args = ["--yes"]
        if args.chat_repo:
            model_args.extend(["--chat-repo", args.chat_repo])
        if args.chat_file:
            model_args.extend(["--chat-file", args.chat_file])
        _run(_python_script("model_bootstrap.py", model_args))

    if not args.skip_preflight:
        _run(_python_script("preflight.py"))

    if not args.no_pull:
        _run(_compose(["pull"]), check=False)

    _ensure_runtime_dirs()
    up_cmd = _compose(["up", "-d"])
    if args.build:
        up_cmd.append("--build")
    _run(up_cmd)

    print("ARCA is up:")
    print("- Frontend: http://localhost:3000")
    print("- Backend:  http://localhost:8000")
    if mcp_mode_enabled:
        print("- MCP mode: local chat disabled; use admin panel + MCP endpoints")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    if not args.no_git:
        _run(["git", "pull", "--ff-only"])

    if args.download_missing_models:
        _run(_python_script("model_bootstrap.py", ["--yes"]), check=False)

    if not args.no_pull:
        _run(_compose(["pull"]), check=False)

    _ensure_runtime_dirs()
    up_cmd = _compose(["up", "-d"])
    if args.build:
        up_cmd.append("--build")
    _run(up_cmd)
    return 0


def cmd_up(args: argparse.Namespace) -> int:
    if args.pull:
        _run(_compose(["pull"]), check=False)

    _ensure_runtime_dirs()
    up_cmd = _compose(["up", "-d"])
    if args.build:
        up_cmd.append("--build")
    _run(up_cmd)
    return 0


def cmd_down(_: argparse.Namespace) -> int:
    _run(_compose(["down"]))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    preflight_args: list[str] = []
    if args.json_output:
        preflight_args.append("--json")
    if args.json_out:
        preflight_args.extend(["--json-out", args.json_out])
    if args.strict_warnings:
        preflight_args.append("--strict-warnings")
    if args.deep_gpu_check:
        preflight_args.append("--deep-gpu-check")
    if args.no_color:
        preflight_args.append("--no-color")
    _run(_python_script("preflight.py", preflight_args))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    model_args: list[str] = []
    if args.yes:
        model_args.append("--yes")
    if args.check_only:
        model_args.append("--check-only")
    _run(_python_script("model_bootstrap.py", model_args), check=False)
    return 0


def cmd_init_env(args: argparse.Namespace) -> int:
    init_args: list[str] = []
    if args.with_mcp_key:
        init_args.append("--set-mcp-key")
    if args.force:
        init_args.append("--force")
    _run(_python_script("init_env.py", init_args))
    return 0


def cmd_export_public(args: argparse.Namespace) -> int:
    export_args: list[str] = []
    if args.dest:
        export_args.extend(["--dest", args.dest])
    if args.clean:
        export_args.append("--clean")
    if args.no_preserve_git:
        export_args.append("--no-preserve-git")
    _run(_python_script("export_public.py", export_args))
    return 0


def cmd_public_init(args: argparse.Namespace) -> int:
    init_args: list[str] = []
    if args.dest:
        init_args.extend(["--dest", args.dest])
    if args.branch:
        init_args.extend(["--branch", args.branch])
    if args.remote:
        init_args.extend(["--remote", args.remote])
    if args.remote_url:
        init_args.extend(["--remote-url", args.remote_url])
    _run(_python_script("public_repo_init.py", init_args))
    return 0


def cmd_publish_public(args: argparse.Namespace) -> int:
    publish_args: list[str] = []
    if args.dest:
        publish_args.extend(["--dest", args.dest])
    if args.branch:
        publish_args.extend(["--branch", args.branch])
    if args.remote:
        publish_args.extend(["--remote", args.remote])
    if args.remote_url:
        publish_args.extend(["--remote-url", args.remote_url])
    if args.commit_message:
        publish_args.extend(["--commit-message", args.commit_message])
    if args.skip_push:
        publish_args.append("--skip-push")
    if args.no_clean:
        publish_args.append("--no-clean")
    if args.no_preserve_git:
        publish_args.append("--no-preserve-git")
    _run(_python_script("publish_public.py", publish_args))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARCA operations helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_bootstrap = subparsers.add_parser("bootstrap", help="First-time setup and launch")
    p_bootstrap.add_argument("--build", action="store_true", help="Build backend/frontend images locally")
    p_bootstrap.add_argument("--gpu", action="store_true", help="Legacy flag (GPU override is now default)")
    p_bootstrap.add_argument("--cpu", action="store_true", help="Force CPU mode (skip GPU override auto-detect)")
    p_bootstrap.add_argument("--skip-model-download", action="store_true", help="Do not auto-download GGUF model")
    p_bootstrap.add_argument("--skip-preflight", action="store_true", help="Skip preflight checks")
    p_bootstrap.add_argument("--no-pull", action="store_true", help="Skip docker compose pull")
    p_bootstrap.add_argument("--with-mcp-key", action="store_true", help="Generate MCP_API_KEY in .env")
    p_bootstrap.add_argument("--chat-repo", help="Override Hugging Face repo for initial chat model")
    p_bootstrap.add_argument("--chat-file", help="Override GGUF filename for initial chat model")
    p_bootstrap.set_defaults(func=cmd_bootstrap)

    p_update = subparsers.add_parser("update", help="Pull latest code/images and restart")
    p_update.add_argument("--no-git", action="store_true", help="Skip git pull")
    p_update.add_argument("--no-pull", action="store_true", help="Skip docker compose pull")
    p_update.add_argument("--build", action="store_true", help="Rebuild local images")
    p_update.add_argument(
        "--download-missing-models",
        action="store_true",
        help="Download missing configured slot models",
    )
    p_update.set_defaults(func=cmd_update)

    p_up = subparsers.add_parser("up", help="Start ARCA services")
    p_up.add_argument("--build", action="store_true", help="Build local images")
    p_up.add_argument("--pull", action="store_true", help="Pull images before starting")
    p_up.set_defaults(func=cmd_up)

    p_down = subparsers.add_parser("down", help="Stop ARCA services")
    p_down.set_defaults(func=cmd_down)

    p_doctor = subparsers.add_parser("doctor", help="Run environment preflight checks")
    p_doctor.add_argument("--json", dest="json_output", action="store_true", help="Emit machine-readable JSON")
    p_doctor.add_argument("--json-out", help="Write JSON report to file")
    p_doctor.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat warnings as failures (automation mode)",
    )
    p_doctor.add_argument(
        "--deep-gpu-check",
        action="store_true",
        help="Run optional CUDA container smoke test when local image is available",
    )
    p_doctor.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    p_doctor.set_defaults(func=cmd_doctor)

    p_models = subparsers.add_parser("models", help="Check/download GGUF models")
    p_models.add_argument("--yes", action="store_true", help="Skip prompt and download automatically")
    p_models.add_argument("--check-only", action="store_true", help="Exit non-zero when no GGUF exists")
    p_models.set_defaults(func=cmd_models)

    p_init = subparsers.add_parser("init-env", help="Create/update .env secrets")
    p_init.add_argument("--with-mcp-key", action="store_true", help="Generate MCP_API_KEY if missing")
    p_init.add_argument("--force", action="store_true", help="Force rotate generated secrets")
    p_init.set_defaults(func=cmd_init_env)

    p_export = subparsers.add_parser("export-public", help="Build sanitized public export tree")
    p_export.add_argument("--dest", help="Destination directory (default: ../ARCA-public-export)")
    p_export.add_argument("--clean", action="store_true", help="Clean destination before export")
    p_export.add_argument("--no-preserve-git", action="store_true", help="Allow cleaning to remove .git metadata")
    p_export.set_defaults(func=cmd_export_public)

    p_public_init = subparsers.add_parser("public-init", help="Initialize public export repository")
    p_public_init.add_argument("--dest", help="Public export repo directory")
    p_public_init.add_argument("--branch", default="main", help="Target branch")
    p_public_init.add_argument("--remote", default="origin", help="Remote name")
    p_public_init.add_argument("--remote-url", help="Public repo remote URL")
    p_public_init.set_defaults(func=cmd_public_init)

    p_publish = subparsers.add_parser("publish-public", help="Export + commit + push public repository")
    p_publish.add_argument("--dest", help="Public export repo directory")
    p_publish.add_argument("--branch", default="main", help="Target branch")
    p_publish.add_argument("--remote", default="origin", help="Remote name")
    p_publish.add_argument("--remote-url", help="Public repo remote URL")
    p_publish.add_argument("--commit-message", default="chore(public): sync from private source")
    p_publish.add_argument("--skip-push", action="store_true", help="Export + commit only")
    p_publish.add_argument("--no-clean", action="store_true", help="Do not clean destination before export")
    p_publish.add_argument("--no-preserve-git", action="store_true", help="Allow cleaning to remove .git metadata")
    p_publish.set_defaults(func=cmd_publish_public)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
