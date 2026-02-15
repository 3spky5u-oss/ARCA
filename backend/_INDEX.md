# backend/

FastAPI backend for ARCA. Contains the API server, tool modules, services infrastructure, chat orchestration, and domain loader. Runs on port 8000 inside Docker with CUDA-enabled llama.cpp for local LLM inference.

| File | Purpose | Key Exports |
|------|---------|-------------|
| main.py | FastAPI app, lifespan, middleware stack, route mounting | `app`, `lifespan()` |
| config.py | RuntimeConfig singleton (~60+ fields), hardware-aware defaults, JSON persistence | `runtime_config`, `RuntimeConfig`, `CONTEXT_DEFAULTS` |
| validation.py | 7 startup validation checks, abort on critical issues | `run_validations()` |
| logging_config.py | ColorFormatter, structured logging helpers | `setup_logging()`, `ColorFormatter` |
| domain_loader.py | Domain pack loader, lexicon cache, pipeline config pass-through | `get_domain_config()`, `get_lexicon()`, `get_pipeline_config()`, `DomainConfig` |
| ingest_knowledge.py | CLI for knowledge base ingestion (auto-ingest, single file, full re-ingest) | CLI script (no class exports) |
| requirements.txt | Python dependencies (~50 packages) | -- |
| Dockerfile | Multi-stage build: CUDA llama.cpp builder + Python runtime | -- |
| ruff.toml | Linter configuration | -- |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| routers/ | API routes (chat WebSocket, admin, upload, sessions, knowledge, benchmarks) |
| services/ | Infrastructure singletons (database, redis, LLM, Neo4j, auth, hardware) |
| tools/ | Tool modules (cohesionn RAG, readd PDF, phii personality, redactrr PII, observationn vision) |
| errors/ | Exception hierarchy, error codes, handler decorators, response formatting |
| middleware/ | Rate limiting (Redis INCR + fallback) |
| migrations/ | PostgreSQL schema (init.sql) |
| profiling/ | Performance benchmarks and baseline JSONs |
| utils/ | LLM client factory, vision server lifecycle |
| tests/ | pytest test suites |
| scripts/ | Backend-specific benchmarks and test scripts |
| training/ | Fine-tuning pipeline (data, evaluation, guardrails, deploy) |
| data/ | Runtime data (QA reference docs) |
