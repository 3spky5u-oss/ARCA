#!/usr/bin/env python3
"""
Initialize/normalize the public export repository directory.

This is intended for a two-folder workflow:
- private/dev source repo (this repo)
- public export repo (default: ../ARCA-public-export)
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT.parent / "ARCA-public-export"


def _run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return result


def _ensure_repo(dest: Path, branch: str) -> None:
    if (dest / ".git").exists():
        return

    dest.mkdir(parents=True, exist_ok=True)

    init_with_branch = _run(["git", "init", "-b", branch], cwd=dest, check=False)
    if init_with_branch.returncode != 0:
        _run(["git", "init"], cwd=dest)
        _run(["git", "checkout", "-b", branch], cwd=dest, check=False)


def _ensure_branch(dest: Path, branch: str) -> None:
    current = _run(["git", "branch", "--show-current"], cwd=dest).stdout.strip()
    if current == branch:
        return

    has_branch = _run(["git", "show-ref", "--verify", f"refs/heads/{branch}"], cwd=dest, check=False)
    if has_branch.returncode == 0:
        _run(["git", "checkout", branch], cwd=dest)
    else:
        _run(["git", "checkout", "-b", branch], cwd=dest)


def _set_remote(dest: Path, remote: str, remote_url: str) -> None:
    has_remote = _run(["git", "remote", "get-url", remote], cwd=dest, check=False)
    if has_remote.returncode == 0:
        _run(["git", "remote", "set-url", remote, remote_url], cwd=dest)
    else:
        _run(["git", "remote", "add", remote, remote_url], cwd=dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize public export git repository")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Public export repo directory")
    parser.add_argument("--branch", default="main", help="Target branch")
    parser.add_argument("--remote", default="origin", help="Remote name")
    parser.add_argument(
        "--remote-url",
        default=os.environ.get("ARCA_PUBLIC_REPO_URL", ""),
        help="Public repo remote URL (or set ARCA_PUBLIC_REPO_URL)",
    )
    args = parser.parse_args()

    dest = Path(args.dest).resolve()
    _ensure_repo(dest, branch=args.branch)
    _ensure_branch(dest, branch=args.branch)

    if args.remote_url:
        _set_remote(dest, remote=args.remote, remote_url=args.remote_url)
        print(f"Remote '{args.remote}' set to: {args.remote_url}")
    else:
        print("No remote URL provided. Set ARCA_PUBLIC_REPO_URL or pass --remote-url.")

    print(f"Public repo ready: {dest}")
    print(f"Active branch: {args.branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
