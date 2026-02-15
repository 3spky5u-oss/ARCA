# ARCA Platform Overview

ARCA is a local-first LLM + RAG platform for technical workflows. It is designed to run on your own hardware and expose every major retrieval and model control as an operator-visible setting.

## Product Intent

ARCA is built for users who want:
- Local control over models, data, and retrieval behavior.
- A practical document-to-answer pipeline (not just a chat wrapper).
- Runtime tuning through an admin panel without rebuild loops.
- Domain extension through configuration and optional tool modules.

It is intentionally transparent: the system surfaces what it did, where evidence came from, and which components are healthy.

## High-Level Capabilities

ARCA provides:
- Local chat inference with GGUF models via llama.cpp servers.
- Hybrid retrieval over ingested documents (dense + optional sparse/hierarchical/graph stages).
- Document ingestion with text extraction, optional vision extraction, chunking, and indexing.
- Tool use for retrieval, web search, unit conversion, and document redaction.
- Admin control plane for configuration, diagnostics, model assignment, and benchmark runs.
- MCP endpoints so external AI clients can call ARCA tools.

For a complete feature inventory, see `features_reference.md` in this same core corpus.

## Runtime Architecture

Default Compose stack:
1. `frontend` (Next.js) on `3000`: chat UI and admin UI.
2. `backend` (FastAPI) on `8000`: orchestration, tools, API, and startup control.
3. `qdrant` on `6333`: vector storage for chunk embeddings.
4. `redis` on `6379`: cache and session/perf support.
5. `postgres` on `5432`: auth and relational persistence.
6. `neo4j` on `7474/7687`: graph storage for entity relationship features.
7. `searxng` on `8080`: web retrieval backend used by web-search tools.

## Core Backend Responsibilities

The backend startup and control plane are primarily in:
- `backend/main.py`
- `backend/config.py`
- `backend/routers/chat.py`
- `backend/tools/registry.py`

Responsibilities include:
- Boot sequencing and health checks.
- Optional model bootstrap when configured models are missing.
- Runtime config management and persisted overrides.
- WebSocket chat orchestration and tool dispatch.
- Knowledge ingestion orchestration and core knowledge ingestion.

## Model Runtime Model

ARCA uses named model slots and llama.cpp server processes:
- Chat slot: always-on conversational model.
- Code slot: optional specialized coding model.
- Expert slot: higher-cost reasoning model.
- Vision slot: multimodal OCR/extraction model.
- Vision structured slot: structured multimodal extraction path.

Slots can be reassigned at runtime from the admin UI when compatible files are present.

## Data Flow Overview

### Ingestion flow (offline/batch)
1. Documents are uploaded or discovered.
2. Text extraction runs (with vision escalation for complex pages when enabled).
3. Content is chunked with metadata.
4. Embeddings are generated and stored in Qdrant.
5. Optional sparse, hierarchy, and graph indexes are refreshed.
6. Topic metadata is available to retrieval and admin views.

### Query flow (online)
1. User message enters WebSocket chat endpoint.
2. Prompt context and session state are assembled.
3. Tools can be invoked (knowledge search, web search, conversion, etc.).
4. Retrieval results and tool outputs are passed to the model.
5. Streamed answer and citations return to UI.

## Retrieval Philosophy

ARCA supports many retrieval stages because corpora and workloads differ. The point is configurability, not "always run everything".

Practical guidance:
- Start with profile defaults.
- Benchmark against your own corpus.
- Apply winning toggles.
- Hand tune only where real query behavior still misses.

## Hardware and Deployment Philosophy

ARCA is GPU-first, with CPU fallback retained for compatibility. Startup includes hardware profiling and runtime defaults that adapt context sizes and related settings. Multi-GPU routing can be defined per component with `ARCA_DEVICE_MAP`.

Model files are externalized to `./models` so container images stay smaller and model choice remains user-controlled.

## Domain Pack Model

ARCA core is generic. Domain packs can inject:
- Lexicon and terminology behavior.
- Topic definitions and retrieval hints.
- Optional tool modules and executors.
- Optional route modules and UI branding metadata.

Domain packs are additive and intended to avoid core forks.

## MCP and External AI Clients

ARCA can run as an MCP-capable backend so external AI clients can call ARCA tools over API endpoints. This allows cloud reasoning models to leverage local retrieval/tooling while ARCA still manages your corpus and retrieval stack.

## What ARCA Is Not

ARCA is not:
- A hosted SaaS service.
- A zero-hardware local app.
- A one-click "best model for everyone" preset.

It is an operator-facing system with substantial knobs and observability.

## Canonical Companion Docs in Core Corpus

- `features_reference.md`: complete feature map.
- `retrieval_pipeline.md`: stage-by-stage retrieval details.
- `configuration.md`: config layers, priority, and common knobs.
- `admin_guide.md`: admin workflows.
- `benchmarks.md`: tuning workflow and benchmark layers.
- `troubleshooting.md`: common failure modes and fixes.
