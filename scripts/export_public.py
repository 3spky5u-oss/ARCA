#!/usr/bin/env python3
"""
Create a sanitized export tree for public GitHub publishing.

Workflow:
1. Keep full source history in private Gitea.
2. Run this script to materialize only allowlisted files.
3. Push export directory to the public GitHub repo.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT.parent / "ARCA-public-export"
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
PUBLIC_GITIGNORE_TEMPLATE = REPO_ROOT / "packaging" / "public" / ".gitignore"
PUBLIC_GITIGNORE_DEST = ".gitignore"


def _read_patterns(path: Path) -> list[str]:
    if not path.exists():
        return []
    patterns: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    return patterns


def _git_tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return files


def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_public_gitignore(dest: Path) -> bool:
    if not PUBLIC_GITIGNORE_TEMPLATE.exists():
        return False
    _copy_file(PUBLIC_GITIGNORE_TEMPLATE, dest / PUBLIC_GITIGNORE_DEST)
    return True


def _clean_destination(dest: Path, preserve_git: bool) -> None:
    if not dest.exists():
        return

    if preserve_git and (dest / ".git").exists():
        for child in dest.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        return

    shutil.rmtree(dest)


def _ensure_runtime_dirs(dest: Path) -> None:
    for rel in RUNTIME_DIRS:
        (dest / rel).mkdir(parents=True, exist_ok=True)


def _running_container_mount_sources() -> list[tuple[str, str]]:
    """
    Return list of (container_name, mount_source_path) for running containers.
    Best-effort only; returns [] when Docker is unavailable.
    """
    try:
        ps = subprocess.run(
            ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    out: list[tuple[str, str]] = []
    for line in ps.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            container_id, container_name = line.split("\t", 1)
        except ValueError:
            continue
        try:
            inspect = subprocess.run(
                ["docker", "inspect", container_id, "--format", "{{json .Mounts}}"],
                check=True,
                capture_output=True,
                text=True,
            )
            mounts = json.loads(inspect.stdout.strip() or "[]")
            for mount in mounts:
                src = mount.get("Source")
                if isinstance(src, str) and src:
                    out.append((container_name, src))
        except Exception:
            continue
    return out


def _containers_using_path(path: Path) -> list[str]:
    target = str(path.resolve()).replace("/", "\\").rstrip("\\").lower()
    matches: set[str] = set()

    for name, src in _running_container_mount_sources():
        normalized = src.replace("/", "\\").rstrip("\\").lower()
        if normalized == target or normalized.startswith(target + "\\"):
            matches.add(name)

    return sorted(matches)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export allowlisted files for public release")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Destination directory for export")
    parser.add_argument(
        "--include-file",
        default=str(REPO_ROOT / ".public-export-include"),
        help="Allowlist patterns file",
    )
    parser.add_argument(
        "--exclude-file",
        default=str(REPO_ROOT / ".public-export-exclude"),
        help="Blocklist patterns file",
    )
    parser.add_argument("--clean", action="store_true", help="Delete destination before export")
    parser.add_argument(
        "--preserve-git",
        dest="preserve_git",
        action="store_true",
        default=True,
        help="When cleaning, keep destination .git metadata if present (default: true)",
    )
    parser.add_argument(
        "--no-preserve-git",
        dest="preserve_git",
        action="store_false",
        help="When cleaning, remove destination completely including .git",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write PUBLIC_EXPORT_REPORT.md into destination (default: false)",
    )
    args = parser.parse_args()

    dest = Path(args.dest).resolve()
    include_patterns = _read_patterns(Path(args.include_file).resolve())
    exclude_patterns = _read_patterns(Path(args.exclude_file).resolve())
    tracked = _git_tracked_files()

    included = [path for path in tracked if _matches(path, include_patterns)]
    selected = [path for path in included if not _matches(path, exclude_patterns)]
    blocked = sorted(path for path in tracked if _matches(path, exclude_patterns))
    skipped = sorted(path for path in tracked if path not in selected)

    if dest.exists():
        if not args.clean:
            print(f"Destination exists: {dest} (use --clean to replace)")
            return 1
        in_use = _containers_using_path(dest)
        if in_use:
            print(f"Refusing to clean export destination while mounted by running containers: {dest}")
            print("Containers:")
            for name in in_use:
                print(f"- {name}")
            print("Stop the stack first (docker compose down) and retry export.")
            return 1
        _clean_destination(dest, preserve_git=args.preserve_git)

    dest.mkdir(parents=True, exist_ok=True)

    for relative in selected:
        src = REPO_ROOT / relative
        if not src.exists():
            continue
        _copy_file(src, dest / relative)

    gitignore_written = _write_public_gitignore(dest)
    _ensure_runtime_dirs(dest)

    report_path: Path | None = None
    if args.write_report:
        report_lines = [
            "# Public Export Report",
            "",
            f"Destination: `{dest.name}`",
            f"Tracked files: {len(tracked)}",
            f"Included by allowlist: {len(included)}",
            f"Exported: {len(selected)}",
            f"Blocked by blocklist: {len(blocked)}",
            f"Skipped: {len(skipped)}",
            f"Public .gitignore template written: {'yes' if gitignore_written else 'no'}",
            f"Preserved .git metadata: {'yes' if args.preserve_git and (dest / '.git').exists() else 'no'}",
            "",
            "## Blocked Files (matched exclude rules)",
        ]
        if blocked:
            report_lines.extend([f"- `{path}`" for path in blocked])
        else:
            report_lines.append("- none")

        report_lines.extend(["", "## Exported Files"])
        report_lines.extend([f"- `{path}`" for path in selected])
        report_path = dest / "PUBLIC_EXPORT_REPORT.md"
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Public export complete: {len(selected)} files -> {dest}")
    if report_path is not None:
        print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
