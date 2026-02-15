#!/bin/bash
# Ensure volume-mounted directories are writable by appuser.
# Docker volumes are created as root; this fixes ownership at runtime.
set -e

dirs=(
    /models
    /app/data/sessions
    /app/data/technical_knowledge
    /app/data/cohesionn_db
    /app/data/auth
    /app/data/config
    /app/data/phii
    /app/data/benchmarks
    /app/data/bm25_index
    /app/data/exceedee_db
    /app/uploads
    /app/reports
    /app/guidelines
    /app/.cache/huggingface
)

# Ensure appuser can create new subdirectories under /app/data
chown appuser:appuser /app/data 2>/dev/null || true

for d in "${dirs[@]}"; do
    mkdir -p "$d" 2>/dev/null || true
    chown -R appuser:appuser "$d" 2>/dev/null || true
done

exec gosu appuser "$@"
