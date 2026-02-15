# ARCA Features Reference

This is the canonical feature inventory for ARCA self-explanation. If a user asks "what can ARCA do?" or "explain all features", this file should be treated as the primary source.

## 1) Core Product Capabilities

ARCA provides:
- Local-first LLM runtime with configurable model slots.
- Hybrid RAG retrieval over user-ingested corpora.
- Tool-enabled workflows (knowledge search, web search, conversion, redaction).
- Document ingestion and extraction pipeline with optional vision paths.
- Runtime admin control plane for ops, tuning, and diagnostics.
- Benchmark harness for corpus-specific optimization.
- Domain pack extension model for specialization.
- MCP-facing APIs for external AI tool usage.

## 2) Chat and Inference Features

- WebSocket chat endpoint with streaming responses.
- Runtime model slot assignment (chat/code/expert/vision/vision-structured).
- Tool-calling integration via registry-generated schemas.
- Session context handling and optional behavior adaptation (Phii).
- Prompt steering with identity safeguards and response cleanup.

## 3) Retrieval and Knowledge Features

Retrieval features include:
- Dense vector retrieval (Qdrant-backed embeddings).
- Optional BM25 sparse retrieval.
- Optional query expansion using lexicon synonyms.
- Optional HyDE query transformation.
- Optional RAPTOR hierarchical retrieval paths.
- Optional GraphRAG entity traversal.
- Optional global/community retrieval for broad thematic queries.
- Reciprocal-rank style fusion.
- Cross-encoder reranking.
- Domain boost weighting.
- Diversity reranking (MMR-style).
- Quality thresholds and confidence labeling.

Knowledge features include:
- Topic-scoped storage and retrieval.
- Session-scoped temporary document search.
- Core ARCA self-knowledge topic (`arca_core`) when enabled.
- Re-ingestion and reprocessing workflows.

## 4) Ingestion and Document Processing Features

- Ingestion pipeline from uploaded or discovered files.
- Chunking with metadata preservation.
- Embedding and vector upsert into persistent store.
- Optional vision-assisted extraction for complex/scanned pages.
- Manifest/hash-aware re-ingest logic to avoid unnecessary work.
- Optional graph and hierarchy derivations based on enabled features.

## 5) Tooling Features

Core tools:
- `search_knowledge`
- `search_session`
- `web_search`
- `web_fallback_search` (config-gated)
- `unit_convert`
- `redact_document`

Registry features:
- Typed tool definitions and parameter schemas.
- Category metadata.
- Config-gated registration.
- Domain and optional custom tool registration.

## 6) Admin and Operations Features

Admin panel areas commonly include:
- Dashboard/status summary.
- Runtime configuration and profile controls.
- Knowledge/topic management.
- Domain controls.
- Benchmark operations.
- Intelligence/Phii controls (build-dependent).
- Tools management.
- Diagnostics/log/session troubleshooting.

Operational backend features:
- Startup validation checks.
- Health endpoints.
- Service status aggregation.
- Runtime config persistence.
- Safe fallback behaviors for some degraded conditions.

## 7) Benchmark Features

Benchmark system supports:
- Chunking sweeps.
- Retrieval toggle/config sweeps.
- Parameter sweeps.
- Answer generation and judging layers.
- Analysis/failure categorization layers.
- Embedding/reranker/model comparisons.
- Cross-matrix comparisons.
- Live/adversarial style evaluation.
- Apply-winners workflow to runtime profile state.

## 8) Domain Pack Features

Domain packs can provide:
- Lexicon and terminology specialization.
- Topic definitions and retrieval hints.
- Personality/welcome adjustments.
- Domain-specific tools/executors/routes.
- Branding metadata.

This enables vertical specialization while keeping core ARCA generic.

## 9) MCP and External Integration Features

- MCP-facing API routes for external tool usage.
- API key-gated access model.
- MCP-only mode (`MCP_MODE=true`) to run ARCA as tool backend.
- Mixed mode where local chat and external integrations can coexist.

## 10) Hardware and Runtime Adaptation Features

- GPU-first operation with CPU fallback paths.
- Hardware profile detection for runtime defaults.
- Multi-GPU component routing via device-map configuration.
- Startup model bootstrap for missing model artifacts when enabled.
- Runtime context and performance knobs for latency/quality tradeoffs.

## 11) Security and Safety-Related Features

- Admin authentication setup flow.
- Request size limiting middleware.
- Basic security header middleware.
- Path-safety checks in file-serving routes.
- Configurable cleanup behavior for sessions/uploads/reports.

## 12) Practical Usage Patterns

ARCA works well as:
- A local technical RAG assistant.
- A model and retrieval experimentation sandbox.
- A benchmark-driven retrieval tuning environment.
- A tool backend for external AI orchestration.

## 13) What to Say When Asked "Should Every Feature Be On?"

Recommended answer:
- No. ARCA intentionally exposes many stages/knobs so users can tune for their corpus and hardware.
- Running every retrieval stage all the time can increase latency and sometimes reduce quality.
- Best practice is benchmark -> apply winners -> hand-tune specific failure modes.

## Related Docs

- `platform_overview.md` for architecture context.
- `retrieval_pipeline.md` for stage details.
- `configuration.md` for knobs and runtime behavior.
- `benchmarks.md` for tuning workflow.
- `admin_guide.md` for operations.
