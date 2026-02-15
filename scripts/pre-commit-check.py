#!/usr/bin/env python3
"""
Pre-commit Check Script - Catches common issues before commits.

Usage:
    python scripts/pre-commit-check.py              # Run all checks
    python scripts/pre-commit-check.py --quick      # Skip slow checks (TypeScript)
    python scripts/pre-commit-check.py --verbose    # Show detailed output

Setup as Git Hook:
    Option 1 - Direct:
        cp scripts/pre-commit-check.py .git/hooks/pre-commit
        chmod +x .git/hooks/pre-commit

    Option 2 - Wrapper script:
        echo '#!/bin/sh\npython scripts/pre-commit-check.py' > .git/hooks/pre-commit
        chmod +x .git/hooks/pre-commit

    Option 3 - pre-commit framework (pyproject.toml):
        [tool.pre-commit]
        repos = [
            { repo = "local", hooks = [
                { id = "arca-check", name = "ARCA pre-commit", entry = "python scripts/pre-commit-check.py", language = "system", pass_filenames = false }
            ]}
        ]

Exit codes:
    0 = All checks passed
    1 = One or more checks failed
"""

import ast
import importlib.util
import os
import re
import subprocess
import sys
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import List, Tuple, Optional

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Minimum expected tool count for a public/default domain build.
# Domain packs can register additional tools above this baseline.
MIN_TOOL_COUNT = 5

# Secret patterns to check for
SECRET_PATTERNS = [
    (r'(?i)api[_-]?key\s*[=:]\s*["\'][^"\']{10,}["\']', "API key"),
    (r'(?i)password\s*[=:]\s*["\'][^"\']{5,}["\']', "Password"),
    (r'(?i)secret\s*[=:]\s*["\'][^"\']{10,}["\']', "Secret"),
    (r'(?i)token\s*[=:]\s*["\'][^"\']{10,}["\']', "Token"),
    (r'(?i)private[_-]?key\s*[=:]\s*["\']', "Private key"),
    (r'sk-[a-zA-Z0-9]{20,}', "OpenAI API key"),
    (r'ghp_[a-zA-Z0-9]{36}', "GitHub token"),
    (r'AKIA[0-9A-Z]{16}', "AWS access key"),
]

# Files/directories to exclude from secret scanning
SECRET_SCAN_EXCLUDES = [
    ".git",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "*.pyc",
    "*.pyo",
    ".env.example",
    "CLAUDE.md",  # Documentation may contain example patterns
    "Dev/**",
    "data/**",
    "backend/profiling/**",
    "backend/reports/**",
    "backend/training/**",
]


class CheckResult:
    """Result of a single check."""

    def __init__(self, name: str, passed: bool, message: str = "", skipped: bool = False):
        self.name = name
        self.passed = passed
        self.message = message
        self.skipped = skipped

    def __str__(self):
        if self.skipped:
            status = "[SKIP]"
        elif self.passed:
            status = "[PASS]"
        else:
            status = "[FAIL]"

        result = f"{status} {self.name}"
        if self.message:
            result += f": {self.message}"
        return result


def get_staged_files() -> List[Path]:
    """Get list of staged files for commit."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            return [PROJECT_ROOT / f for f in result.stdout.strip().split("\n") if f]
    except Exception:
        pass
    return []


def get_changed_files() -> List[Path]:
    """Get list of changed (staged + unstaged) files."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            files = []
            for line in result.stdout.strip().split("\n"):
                if line and len(line) > 3:
                    # Format: "XY filename" or "XY original -> renamed"
                    filename = line[3:].split(" -> ")[-1]
                    files.append(PROJECT_ROOT / filename)
            return files
    except Exception:
        pass
    return []


def get_tracked_files() -> set[str]:
    """Return all tracked paths (repo-relative POSIX style)."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return {line.strip() for line in result.stdout.splitlines() if line.strip()}
    except Exception:
        pass
    return set()


def _is_secret_scan_excluded(rel_path: str) -> bool:
    rel_path = rel_path.replace("\\", "/")
    for pattern in SECRET_SCAN_EXCLUDES:
        if "/" in pattern or "*" in pattern:
            if fnmatch(rel_path, pattern):
                return True
        if pattern in rel_path:
            return True
    return False


def check_python_syntax(verbose: bool = False) -> CheckResult:
    """Check that all Python files parse without syntax errors."""
    errors = []
    checked = 0

    for py_file in BACKEND_DIR.rglob("*.py"):
        # Skip pycache and test fixtures
        if "__pycache__" in str(py_file) or "fixtures" in str(py_file):
            continue

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source)
            checked += 1
        except SyntaxError as e:
            rel_path = py_file.relative_to(PROJECT_ROOT)
            errors.append(f"{rel_path}:{e.lineno}: {e.msg}")

    # Also check scripts directory
    for py_file in SCRIPT_DIR.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source)
            checked += 1
        except SyntaxError as e:
            rel_path = py_file.relative_to(PROJECT_ROOT)
            errors.append(f"{rel_path}:{e.lineno}: {e.msg}")

    if errors:
        msg = f"{len(errors)} syntax error(s)"
        if verbose:
            msg += "\n  " + "\n  ".join(errors[:5])
            if len(errors) > 5:
                msg += f"\n  ... and {len(errors) - 5} more"
        return CheckResult("Python syntax check", False, msg)

    return CheckResult("Python syntax check", True, f"{checked} files OK")


def check_imports(verbose: bool = False) -> CheckResult:
    """Check that key modules import without crashing."""
    runtime_deps = ["fastapi", "qdrant_client"]
    missing = [dep for dep in runtime_deps if importlib.util.find_spec(dep) is None]
    if missing:
        return CheckResult(
            "Import validation",
            True,
            skipped=True,
            message=f"missing runtime deps ({', '.join(missing)})",
        )

    # Add backend to path for imports
    original_path = sys.path.copy()
    sys.path.insert(0, str(BACKEND_DIR))

    errors = []
    checked = 0

    # Key modules that should always import
    modules_to_check = [
        "config",
        "errors",
        "errors.codes",
        "errors.exceptions",
        "errors.handlers",
        "tools.registry",
        "tools.cohesionn",
        "tools.readd",
        "tools.phii",
        "tools.redactrr",
    ]

    for module in modules_to_check:
        try:
            __import__(module)
            checked += 1
        except Exception as e:
            errors.append(f"{module}: {type(e).__name__}: {e}")

    # Restore path
    sys.path = original_path

    if errors:
        msg = f"{len(errors)} import error(s)"
        if verbose:
            msg += "\n  " + "\n  ".join(errors[:3])
        return CheckResult("Import validation", False, msg)

    return CheckResult("Import validation", True, f"{checked} modules OK")


def check_tool_registry(verbose: bool = False) -> CheckResult:
    """Check that tool registry loads without errors and has expected tools."""
    runtime_deps = ["fastapi", "qdrant_client"]
    missing = [dep for dep in runtime_deps if importlib.util.find_spec(dep) is None]
    if missing:
        return CheckResult(
            "Tool registry",
            True,
            skipped=True,
            message=f"missing runtime deps ({', '.join(missing)})",
        )

    # Add backend to path for imports
    original_path = sys.path.copy()
    sys.path.insert(0, str(BACKEND_DIR))

    try:
        # Clear any cached registry state
        from tools.registry import ToolRegistry, register_all_tools
        ToolRegistry.clear()

        # Register all tools
        register_all_tools()

        # Check tool count
        tools = ToolRegistry.get_all_tools()
        tool_count = len(tools)

        if tool_count < MIN_TOOL_COUNT:
            return CheckResult(
                "Tool registry",
                False,
                f"Expected {MIN_TOOL_COUNT}+ tools, got {tool_count}"
            )

        # Verify each tool has required fields
        for name, tool in tools.items():
            if not tool.name or not tool.executor:
                return CheckResult(
                    "Tool registry",
                    False,
                    f"Tool '{name}' missing required fields"
                )

        msg = f"{tool_count} tools registered"
        if verbose:
            msg += f" ({', '.join(sorted(tools.keys()))})"

        return CheckResult("Tool registry", True, msg)

    except Exception as e:
        return CheckResult("Tool registry", False, f"{type(e).__name__}: {e}")
    finally:
        sys.path = original_path


def check_typescript(verbose: bool = False, quick: bool = False) -> CheckResult:
    """Check TypeScript compilation if frontend files changed."""
    # Check if any frontend files changed
    changed = get_changed_files()
    frontend_changed = any(
        FRONTEND_DIR in f.parents or f.parent == FRONTEND_DIR
        for f in changed
        if f.suffix in (".ts", ".tsx", ".js", ".jsx")
    )

    if not frontend_changed and not quick:
        # Check if we should skip entirely
        staged = get_staged_files()
        frontend_staged = any(
            FRONTEND_DIR in f.parents or f.parent == FRONTEND_DIR
            for f in staged
            if f.suffix in (".ts", ".tsx", ".js", ".jsx")
        )
        if not frontend_staged:
            return CheckResult("TypeScript check", True, skipped=True, message="no frontend changes")

    if quick:
        return CheckResult("TypeScript check", True, skipped=True, message="quick mode")

    # Check if tsc is available
    tsc_path = FRONTEND_DIR / "node_modules" / ".bin" / "tsc"
    if sys.platform == "win32":
        tsc_path = FRONTEND_DIR / "node_modules" / ".bin" / "tsc.cmd"

    if not tsc_path.exists():
        # Try npx
        try:
            result = subprocess.run(
                ["npx", "--yes", "tsc", "--noEmit"],
                capture_output=True,
                text=True,
                cwd=FRONTEND_DIR,
                timeout=60
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return CheckResult("TypeScript check", True, skipped=True, message="tsc not available")
    else:
        try:
            result = subprocess.run(
                [str(tsc_path), "--noEmit"],
                capture_output=True,
                text=True,
                cwd=FRONTEND_DIR,
                timeout=60
            )
        except subprocess.TimeoutExpired:
            return CheckResult("TypeScript check", False, "tsc timed out")

    if result.returncode != 0:
        error_count = result.stdout.count("error TS")
        msg = f"{error_count} TypeScript error(s)"
        if verbose and result.stdout:
            lines = [l for l in result.stdout.split("\n") if "error TS" in l][:3]
            if lines:
                msg += "\n  " + "\n  ".join(lines)
        return CheckResult("TypeScript check", False, msg)

    return CheckResult("TypeScript check", True)


def check_secrets(verbose: bool = False) -> CheckResult:
    """Scan for potential secrets in code."""
    findings = []

    tracked = get_tracked_files()
    if not tracked:
        return CheckResult("Secret scan", True, skipped=True, message="could not list tracked files")

    text_exts = {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yaml", ".yml", ".env", ".md", ".sh", ".ps1"}

    # Prefer staged files in pre-commit context, then changed files.
    candidates = get_staged_files()
    if not candidates:
        candidates = get_changed_files()

    files_to_check = []
    for file_path in candidates:
        if not file_path.exists() or file_path.suffix.lower() not in text_exts:
            continue
        rel_path = str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if rel_path not in tracked:
            continue
        files_to_check.append(file_path)

    if not files_to_check:
        return CheckResult("Secret scan", True, skipped=True, message="no staged text files")

    for file_path in files_to_check:
        # Skip excluded paths
        rel_path = str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if _is_secret_scan_excluded(rel_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            for pattern, secret_type in SECRET_PATTERNS:
                matches = re.finditer(pattern, content)
                for match in matches:
                    # Find line number
                    line_num = content[:match.start()].count("\n") + 1
                    findings.append((rel_path, line_num, secret_type))

        except Exception:
            continue

    if findings:
        msg = f"{len(findings)} potential secret(s) found"
        if verbose:
            for path, line, secret_type in findings[:5]:
                msg += f"\n  {path}:{line}: {secret_type}"
            if len(findings) > 5:
                msg += f"\n  ... and {len(findings) - 5} more"
        return CheckResult("Secret scan", False, msg)

    return CheckResult("Secret scan", True)


def run_all_checks(verbose: bool = False, quick: bool = False) -> List[CheckResult]:
    """Run all pre-commit checks."""
    results = []

    # Python syntax (fast)
    start = time.time()
    results.append(check_python_syntax(verbose))
    if verbose:
        print(f"  Syntax check: {time.time() - start:.2f}s")

    # Import validation (fast)
    start = time.time()
    results.append(check_imports(verbose))
    if verbose:
        print(f"  Import check: {time.time() - start:.2f}s")

    # Tool registry (fast)
    start = time.time()
    results.append(check_tool_registry(verbose))
    if verbose:
        print(f"  Registry check: {time.time() - start:.2f}s")

    # TypeScript (can be slow)
    start = time.time()
    results.append(check_typescript(verbose, quick))
    if verbose:
        print(f"  TypeScript check: {time.time() - start:.2f}s")

    # Secret scan (fast)
    start = time.time()
    results.append(check_secrets(verbose))
    if verbose:
        print(f"  Secret scan: {time.time() - start:.2f}s")

    return results


def main():
    """Main entry point."""
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    quick = "--quick" in sys.argv or "-q" in sys.argv

    print("Running pre-commit checks...\n")
    start_time = time.time()

    results = run_all_checks(verbose=verbose, quick=quick)

    # Print results
    print()
    for result in results:
        print(result)

    # Summary
    elapsed = time.time() - start_time
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped ({elapsed:.2f}s)")

    # Exit code
    if failed > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
