#!/usr/bin/env python3
"""
Export allowlisted files to public repo, commit, and optionally push.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
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


def _ensure_public_repo(dest: Path, branch: str, remote: str, remote_url: str | None) -> None:
    if (dest / ".git").exists():
        return

    init_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "public_repo_init.py"),
        "--dest",
        str(dest),
        "--branch",
        branch,
        "--remote",
        remote,
    ]
    if remote_url:
        init_cmd.extend(["--remote-url", remote_url])
    _run(init_cmd, cwd=REPO_ROOT)


def _checkout_branch(dest: Path, branch: str) -> None:
    current = _run(["git", "branch", "--show-current"], cwd=dest).stdout.strip()
    if current == branch:
        return

    has_branch = _run(["git", "show-ref", "--verify", f"refs/heads/{branch}"], cwd=dest, check=False)
    if has_branch.returncode == 0:
        _run(["git", "checkout", branch], cwd=dest)
    else:
        _run(["git", "checkout", "-b", branch], cwd=dest)


def _has_staged_changes(dest: Path) -> bool:
    diff = _run(["git", "diff", "--cached", "--quiet"], cwd=dest, check=False)
    return diff.returncode != 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export + commit + push public repo")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Public export repo directory")
    parser.add_argument("--branch", default="main", help="Target branch")
    parser.add_argument("--remote", default="origin", help="Remote name")
    parser.add_argument("--remote-url", help="Remote URL (used if repo/remote is missing)")
    parser.add_argument("--commit-message", default="chore(public): sync from private source")
    parser.add_argument("--skip-push", action="store_true", help="Export + commit only (no push)")
    parser.add_argument("--no-clean", action="store_true", help="Do not clean destination before export")
    parser.add_argument("--no-preserve-git", action="store_true", help="Allow export clean to remove .git metadata")
    args = parser.parse_args()

    dest = Path(args.dest).resolve()
    _ensure_public_repo(dest, branch=args.branch, remote=args.remote, remote_url=args.remote_url)
    _checkout_branch(dest, branch=args.branch)

    export_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "export_public.py"),
        "--dest",
        str(dest),
    ]
    if not args.no_clean:
        export_cmd.append("--clean")
    if args.no_preserve_git:
        export_cmd.append("--no-preserve-git")
    _run(export_cmd, cwd=REPO_ROOT)

    _run(["git", "add", "-A"], cwd=dest)
    if not _has_staged_changes(dest):
        print("No public export changes to commit.")
        return 0

    _run(["git", "commit", "-m", args.commit_message], cwd=dest)

    if args.skip_push:
        print("Committed public export changes (push skipped).")
        return 0

    _run(["git", "push", args.remote, args.branch], cwd=dest)
    print(f"Pushed public export to {args.remote}/{args.branch}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
