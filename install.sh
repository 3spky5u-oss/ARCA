#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${ARCA_REPO_URL:-https://github.com/3spky5u-oss/ARCA.git}"
TARGET_DIR="${1:-ARCA}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/scripts/bootstrap.sh" ]]; then
  cd "$SCRIPT_DIR"
  ./scripts/bootstrap.sh
  exit 0
fi

if [[ -d "$TARGET_DIR/.git" ]]; then
  echo "Using existing repo: $TARGET_DIR"
else
  git clone "$REPO_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"
./scripts/bootstrap.sh
