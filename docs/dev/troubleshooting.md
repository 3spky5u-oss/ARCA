# Troubleshooting

## Docker

```bash
docker compose --compatibility up -d              # Start all
docker compose logs -f backend    # View backend logs
docker compose --compatibility up -d --build      # Rebuild after code changes
```

## LLM Server

```bash
docker compose logs -f backend    # llama-server logs appear in backend output
```

VRAM management: Qwen3-30B-A3B runs on port 8081. Vision (qwen3-vl-8b) starts on-demand on port 8082, stopped after use. Expert model (qwen3-32b) swaps with chat slot on demand. During KB ingestion, chat is unloaded for vision with `--parallel 2`.

## Frontend

```bash
cd frontend && npm ci  # Clean install (deletes node_modules)
```

## Knowledge Base

```bash
python scripts/model_bootstrap.py            # Ensure configured slot GGUF files exist
python backend/ingest_knowledge.py --setup  # Recreate directories
python backend/ingest_knowledge.py          # Re-ingest all
python backend/ingest_knowledge.py topic    # Ingest specific topic
```

Startup also supports auto-download for configured models when:
- `ARCA_AUTO_DOWNLOAD_MODELS=true`
- `ARCA_AUTO_DOWNLOAD_OPTIONAL_MODELS=true` (default)
- `LLM_<SLOT>_MODEL_REPO=<huggingface-repo>` (or known fallback mapping)
- `./models` is writable from the backend container

## Dependencies

```bash
python scripts/deps.py chat       # Chat router dependencies
python scripts/deps.py cohesionn  # RAG dependencies
python scripts/deps.py --list     # All available modules
```

## Quick Fixes

| Symptom | Fix |
|---------|-----|
| New data files not visible | `docker compose up -d --build` (data dirs need docker-compose volume mounts) |
| Model not responding | Check backend logs for llama-server errors |
| Frontend changes not showing | `npm run dev` needs restart |
| Tool not being called | Check registry.py registration, verify LLM sees tool in TOOLS_SECTION |
| Curly brace error in prompts | Use `.replace()` not `.format()` for system prompts with dynamic content |
| JSON regex not matching | Double-escape backslashes in JSON (`\\b` not `\b`) |
| Jina reranker fails to load | Ensure `trust_remote_code=True` is set (auto-detected for Jina models) |
| SemanticChunker metadata error | Use `metadata={"source": name}` not `source=name` |
| First query takes 35-50s | Normal cold start -- `warm_models()` pre-loads embedder, reranker, BM25 |
| Pipeline config keys missing | `get_pipeline_config()` passes through ALL lexicon keys, not just defaults |
| Domain tool import fails | Domain imports must use `try/except ImportError` pattern |
| Auth lost after restart | Check `data/auth/admin_auth.json` persistence and volume mounts. JWT secret is persisted when auth storage is healthy. |
| Rate limiting not working | Fallback mode (Redis down) disables all rate limits |
| Admin account locked out | Set `ADMIN_RESET=true` in `.env`, restart, re-register |
| Docker credential issue in WSL | Clear `~/.docker/config.json` to `{}` (remove `credsStore: desktop.exe`) |
| numpy/numba version conflict | Pin numpy<=2.3.0 with numba>=0.59.0 + llvmlite>=0.42.0 |
| ONNX Runtime GPU not found | Force-reinstall in Dockerfile overrides CPU-only version from optimum |
| `Found no NVIDIA driver on your system` in backend logs | Use `python scripts/arca.py bootstrap` (GPU mode default) and verify Docker GPU passthrough (`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`) |

## Circuit Breaker

The LLM orchestrator has a circuit breaker (threshold=5 consecutive failures, 30s recovery timeout). If the LLM server is down, the circuit opens and subsequent requests fail fast instead of timing out. Check `/api/admin/status` for LLM health.

## Service Fallbacks

All services degrade gracefully:

| Service | Fallback |
|---------|----------|
| Redis | In-memory OrderedDict LRU (1000 entries, TTL support) |
| PostgreSQL | SQLite file (`data/arca.db`) |
| Neo4j | Graph features disabled (returns empty) |
| Qdrant | RAG features disabled |
| SearXNG | Web search unavailable |
| LLM | Error message to user |
