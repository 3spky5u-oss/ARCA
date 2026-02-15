# backend/routers/

API route handlers for ARCA. The main chat interface uses WebSocket; all admin and file operations use REST. Chat processing is split across executors (tool implementations), orchestration (LLM coordination), and handlers (query classification).

| File | Purpose | Key Exports |
|------|---------|-------------|
| chat.py | WebSocket `/ws/chat` -- message processing, handler classification, tool dispatch, streaming | WebSocket endpoint |
| upload.py | RAM-based file upload (50MB max), lab data parsing, RAG indexing | `/api/upload` endpoint |
| admin/ | Subpackage: auth, users, status, testing, sessions_logs, models_routes, domains, log_capture, models | `/api/admin/*` endpoints |
| admin_benchmark/ | Subpackage: endpoints, helpers, models — background benchmark execution, progress, apply winners | `/api/admin/benchmark/*` endpoints |
| admin_knowledge/ | Subpackage: stats, ingest, deletion, search, topics, settings, reprocess, core_knowledge, qdrant, llm_management, models | `/api/admin/knowledge/*` endpoints |
| admin_lexicon.py | Domain lexicon CRUD with cache refresh on write | `/api/admin/lexicon/*` endpoints |
| admin_graph.py | Neo4j management, Cypher security (write patterns blocked), visualization | `/api/admin/graph/*` endpoints |
| admin_phii.py | Phii behavior: stats, flags, corrections, patterns, debug, metrics | `/api/admin/phii/*` endpoints |
| admin_training.py | Fine-tuning pipeline: parse, generate, filter, format, evaluate, deploy | `/api/admin/training/*` endpoints |
| sessions.py | JSON file-based session persistence, path traversal protection | `/api/sessions/*` endpoints |
| visualize.py | 3D visualization API, in-memory store, 1-hour TTL cleanup | `/api/viz/*` endpoints |
| mcp_api.py | MCP (Model Context Protocol) endpoints for external AI tool integration -- API key auth, proxies to existing executors | `/api/mcp/*` endpoints |
| chat_prompts.py | System prompts (composable sections), technical detection, response cleanup | `get_instructions_section()`, `get_rules_section()`, `cleanup_response_text()` |
| chat_streaming.py | Word-by-word streaming, final response metadata assembly | `stream_response()`, `build_final_response()` |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| chat_executors/ | Tool execution implementations (RAG search, web search, calculations, documents) |
| chat_orchestration/ | LLM coordination (two-call pattern, circuit breaker, session state, citations, auto-search) |
| chat_orchestration/handlers/ | Query classification and specialized handlers (geology, calculate, think, technical) |
