#!/usr/bin/env python3
"""
Module Test Runner - Run tests for specific modules.

Usage:
    python scripts/test_module.py solverr      # Run Solverr tests
    python scripts/test_module.py readd        # Run Readd tests
    python scripts/test_module.py all          # Run all tests
    python scripts/test_module.py --cov        # All tests with coverage
    python scripts/test_module.py solverr --cov # Solverr with coverage
    python scripts/test_module.py --list       # List available test modules

Example:
    python scripts/test_module.py solverr -v   # Verbose Solverr tests
"""

import sys
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
TESTS_DIR = BACKEND_DIR / "tests"

# Test module mappings
TEST_MODULES = {
    "solverr": {
        "test_dir": "test_solverr",
        "coverage": "tools/solverr",
        "description": "Engineering calculations (bearing, settlement, frost)",
    },
    "readd": {
        "test_dir": "test_readd",
        "coverage": "tools/readd",
        "description": "Document extraction and QA",
    },
    "cohesionn": {
        "test_dir": "test_cohesionn",
        "coverage": "tools/cohesionn",
        "description": "RAG pipeline and ingestion",
    },
    "exceedee": {
        "test_dir": "test_exceedee",
        "coverage": "tools/exceedee",
        "description": "Compliance checking",
    },
    "redactrr": {
        "test_dir": "test_redactrr",
        "coverage": "tools/redactrr",
        "description": "PII redaction",
    },
    "all": {
        "test_dir": "",
        "coverage": "tools",
        "description": "All tests",
    },
}


def list_modules():
    """List available test modules."""
    print("Available test modules:\n")
    print(f"{'Module':<12} {'Test Dir':<20} Description")
    print("-" * 60)

    for name, info in TEST_MODULES.items():
        test_path = TESTS_DIR / info["test_dir"] if info["test_dir"] else TESTS_DIR
        exists = "Y" if test_path.exists() else "N"
        print(f"{name:<12} {info['test_dir'] or '(all)':<20} {info['description']} [{exists}]")


def run_tests(module: str, extra_args: list):
    """Run pytest for a specific module."""
    if module not in TEST_MODULES:
        print(f"Unknown module: {module}")
        print(f"Available: {', '.join(TEST_MODULES.keys())}")
        sys.exit(1)

    info = TEST_MODULES[module]
    test_dir = info["test_dir"]
    coverage_target = info["coverage"]

    # Build pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test directory
    if test_dir:
        cmd.append(f"tests/{test_dir}/")
    else:
        cmd.append("tests/")

    # Check for coverage flag
    if "--cov" in extra_args:
        extra_args.remove("--cov")
        cmd.extend(["--cov=" + coverage_target, "--cov-report=term-missing"])

    # Add remaining args (like -v)
    cmd.extend(extra_args)

    # Change to backend directory and run
    print(f"Running: {' '.join(cmd)}")
    print(f"Directory: {BACKEND_DIR}")
    print("-" * 60)

    result = subprocess.run(cmd, cwd=BACKEND_DIR)
    sys.exit(result.returncode)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1].lower()
    extra_args = sys.argv[2:]

    if arg in ("--list", "-l", "list"):
        list_modules()
    elif arg in ("--help", "-h", "help"):
        print(__doc__)
    elif arg == "--cov":
        # --cov without module means all with coverage
        run_tests("all", ["--cov"] + extra_args)
    else:
        run_tests(arg, extra_args)


if __name__ == "__main__":
    main()
