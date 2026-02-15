# Configuration Guide

ARCA configuration comes from three places:

1. `.env` at startup
2. Admin Panel runtime updates (`/admin`)
3. `backend/config.py` defaults (fallbacks)

Use this file as a practical map, not an exhaustive field dump.

## Recommended First-Run Edits

Set these before first launch:

```env
POSTGRES_PASSWORD=change-me
NEO4J_PASSWORD=change-me
LLM_CHAT_MODEL=<your-chat-model.gguf>
```

Optional PostgreSQL startup race tuning (if postgres starts slowly on your host):

```env
POSTGRES_CONNECT_RETRIES=15
POSTGRES_CONNECT_RETRY_DELAY_S=2.0
```

If you use MCP clients:

```env
MCP_API_KEY=<random-secret>
```

Container image channel (optional):

```env
ARCA_IMAGE_TAG=latest
ARCA_BACKEND_IMAGE=ghcr.io/3spky5u-oss/arca-backend
ARCA_FRONTEND_IMAGE=ghcr.io/3spky5u-oss/arca-frontend
```

If you use multiple GPUs:

```env
ARCA_DEVICE_MAP=chat:0,vision:1,embedder:1,reranker:1
# Optional per-slot tensor split for one model across multiple GPUs:
# LLM_CHAT_SPLIT_MODE=layer
# LLM_CHAT_TENSOR_SPLIT=1,1
```

First-run auto-tiering picks defaults by detected total GPU VRAM:

1. `0-8` / `8-12` GB: Qwen3 8B Q4
2. `12-16` / `16-24` GB: Qwen3 14B Q4
3. `24-32` / `32-48` / `48-96` / `96+` GB: Qwen3 30B-A3B Q4 (safe default)

You can override any slot manually in `.env` or Admin > Models.

## High-Impact Knobs

### Models

- `LLM_CHAT_MODEL`: primary chat model (required)
- `LLM_CHAT_MODEL_REPO`: Hugging Face repo for chat model auto-download
- `LLM_CODE_MODEL`: code slot model (defaults to chat model)
- `LLM_EXPERT_MODEL`: expert/think slot model (defaults to chat model)
- `LLM_VISION_MODEL`: vision slot model for OCR/charts
- `LLM_VISION_STRUCTURED_MODEL`: structured vision slot model (defaults to vision model)
- `LLM_<SLOT>_MODEL_REPO`: explicit repo mapping for each slot
- `LLM_N_CPU_MOE`: global llama.cpp `--n-cpu-moe` override for MoE models
- `LLM_CHAT_N_CPU_MOE`, `LLM_VISION_N_CPU_MOE`: slot-specific `--n-cpu-moe` overrides
- `ARCA_AUTO_DOWNLOAD_MODELS`: auto-pull missing configured GGUF files at startup
- `ARCA_AUTO_DOWNLOAD_OPTIONAL_MODELS`: include on-demand slots (default: true)

### Context and generation

- `LLM_CTX_SMALL`, `LLM_CTX_MEDIUM`, `LLM_CTX_LARGE`, `LLM_CTX_XLARGE`
- `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_TOP_K`, `LLM_MAX_OUTPUT`

### Retrieval quality/latency

- `RAG_TOP_K`, `RAG_MIN_SCORE`, `RAG_MIN_FINAL_SCORE`
- `RERANKER_ENABLED`, `RERANKER_CANDIDATES`
- `BM25_ENABLED`, `HYDE_ENABLED`, `CRAG_ENABLED`
- `RAPTOR_ENABLED`, `GRAPH_RAG_ENABLED`

Do not assume enabling every retrieval feature is best.
Use benchmark runs to find a winning subset, then hand-tune only what your corpus needs.
Practical loop: feed a sample corpus, run benchmark to completion, apply winners, then tweak.

### Web search (SearXNG)

- `SEARXNG_ENABLED`: enable/disable web search tool execution
- `SEARXNG_URL`: backend URL for your SearXNG instance
- `SEARXNG_CATEGORIES`: default categories (comma-separated)
- `SEARXNG_LANGUAGE`: optional language code (blank = auto)
- `SEARXNG_TIMEOUT_S`: request timeout per query
- `SEARXNG_MAX_RESULTS`: final web results returned by `web_search`
- `SEARXNG_REQUEST_FORMAT`: `json` (preferred) or `html`

These are also editable at runtime in Admin > Configuration > Connections.

### Safety and operations

- `CORE_KNOWLEDGE_ENABLED`: ingest built-in ARCA docs at startup
- `INGEST_LOCK_ENABLED`: block chat during ingest jobs
- `ADMIN_RESET`: reset admin auth on restart (dev-only)

## GPU vs CPU

- Base compose is GPU-first by default.
- `python scripts/arca.py bootstrap` starts with GPU settings automatically.
- `--gpu` is retained as a compatibility no-op.
- CPU fallback still works, but is opt-in: set `COHESIONN_EMBED_DEVICE=cpu`, `COHESIONN_RERANK_DEVICE=cpu`, `LLM_CHAT_GPU_LAYERS=0`, and `LLM_VISION_GPU_LAYERS=0` in `.env`.

## Where To Change What

- Frequent tuning: Admin Panel (`/admin`) -> Configuration
- Infrastructure secrets and startup defaults: `.env`
- Deep defaults and field names: `backend/config.py`

## Validation Commands

```bash
python scripts/arca.py doctor
python scripts/preflight.py
docker compose config
docker compose up -d
```

## Related Docs

- `docs/dev/troubleshooting.md`
- `docs/API.md`
- `README.md`
