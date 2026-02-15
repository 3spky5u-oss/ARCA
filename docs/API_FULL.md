# ARCA API Documentation

**Version**: 0.1.0
**Base URL**: `http://localhost:8000`

---

## Table of Contents

1. [Authentication](#authentication)
2. [Chat & WebSocket](#chat--websocket)
3. [File Upload](#file-upload)
4. [Admin Endpoints](#admin-endpoints)
5. [Graph Endpoints](#graph-endpoints)
6. [Benchmark Endpoints](#benchmark-endpoints)
7. [Lexicon Endpoints](#lexicon-endpoints)
8. [Knowledge Endpoints](#knowledge-endpoints)
9. [Phii Endpoints](#phii-endpoints)
10. [Training Endpoints](#training-endpoints)
11. [Domain Endpoints](#domain-endpoints)
12. [MCP Endpoints](#mcp-endpoints)
13. [Error Responses](#error-responses)

---

## Authentication

### Admin Authentication

Admin endpoints require JWT authentication via Bearer token.

**Authentication Flow**:
1. Check setup: `GET /api/admin/auth/status` (returns `{has_users, setup_required}`)
2. First boot: `POST /api/admin/auth/register` (first user becomes admin)
3. Login: `POST /api/admin/auth/login` with `{username, password}` -> Returns JWT token
4. Use token: `Authorization: Bearer <token>`
5. Token expiry: 8 hours (HS256)

#### GET /api/admin/auth/status

Check if any users exist (first-boot detection).

**Response**:
```json
{
  "has_users": true,
  "setup_required": false
}
```

---

#### POST /api/admin/auth/register

Register a new user. First registered user automatically gets admin role.

**Request**:
```json
{
  "username": "admin",
  "password": "your-secure-password"
}
```

**Response**:
```json
{
  "token": "eyJhbGc...",
  "username": "admin",
  "role": "admin"
}
```

**Errors**:
- `400`: Username already exists
- `400`: Password too short (min 8 characters)

---

#### POST /api/admin/auth/login

Authenticate with username and password.

**Request**:
```json
{
  "username": "admin",
  "password": "your-secure-password"
}
```

**Response**:
```json
{
  "token": "eyJhbGc...",
  "username": "admin",
  "role": "admin"
}
```

**Errors**:
- `401`: Invalid credentials
- `429`: Rate limited (5 attempts per 15 minutes)

---

#### GET /api/admin/auth/verify

Verify JWT token validity.

**Headers**:
```
Authorization: Bearer <token>
```

**Response**:
```json
{
  "valid": true,
  "username": "admin",
  "role": "admin"
}
```

**Errors**:
- `401`: Invalid or expired token

---

#### POST /api/admin/auth/change-password

Change user password (requires current password).

**Headers**:
```
Authorization: Bearer <token>
```

**Request**:
```json
{
  "old_password": "current-password",
  "new_password": "new-secure-password"
}
```

**Response**:
```json
{
  "success": true,
  "token": "new_jwt_token..."
}
```

**Side Effects**:
- Rotates JWT secret -> invalidates all existing tokens
- Returns new token for immediate use

---

## Chat & WebSocket

### WebSocket Connection

**Endpoint**: `ws://localhost:8000/ws/chat`

**Protocol**: JSON messages over WebSocket

**Connection Flow**:
```
1. Client connects → WebSocket established
2. Client sends message → Server processes with LLM + tools
3. Server streams response → Incremental chunks
4. Server sends done → Complete with metadata
```

---

### Client → Server Messages

#### User Message

```json
{
  "message": "What are the recommended safety factors?",
  "search_mode": false,
  "deep_search": false,
  "think_mode": false,
  "phii_enabled": true,
  "session_id": "optional-session-id"
}
```

**Fields**:
- `message` (required): User query string (max 10,000 chars)
- `search_mode` (optional): Enable web search via SearXNG
- `deep_search` (optional): Extended multi-query search (requires search_mode=true)
- `think_mode` (optional): Use enhanced reasoning prompts for extended reasoning
- `calculate_mode` (optional): Use dedicated calculation model for engineering calculations
- `phii_enabled` (optional): Enable adaptive response behavior (Phii personality module)
- `session_id` (optional): Associate message with session

---

### Server → Client Messages

#### Stream Token

Incremental response chunks during LLM generation.

```json
{
  "type": "stream",
  "content": "Based on the documentation, ",
  "done": false
}
```

---

#### Stream Complete

Final message with metadata.

```json
{
  "type": "stream",
  "content": "the recommended approach is to follow the guidelines in section 4.2.",
  "done": true,
  "tools_used": ["search_knowledge"],
  "citations": [
    {
      "url": "",
      "title": "Technical Reference Guide",
      "topic": "general",
      "score": 0.87
    }
  ]
}
```

**Fields**:
- `tools_used` (optional): List of tool names executed
- `citations` (optional): Sources used in response (RAG/web search)
- `analysis_result` (optional): Tool output data (compliance checks, calculations, etc.)

---

#### Tool Start

Notification that a tool is executing.

```json
{
  "type": "tool_start",
  "tool_name": "search_knowledge",
  "args": {
    "query": "recommended safety factors"
  }
}
```

---

#### Tool End

Tool execution complete with result.

```json
{
  "type": "tool_end",
  "tool_name": "search_knowledge",
  "result": {
    "success": true,
    "results": [
      {
        "content": "The recommended safety factors are...",
        "score": 0.92,
        "source": "Technical Reference Guide"
      }
    ]
  }
}
```

---

#### Error Message

```json
{
  "type": "error",
  "content": "Message too long (max 10,000 characters)",
  "error_code": "MESSAGE_TOO_LONG"
}
```

---

## File Upload

### POST /api/upload

Upload files for analysis (Excel, CSV, PDF, Word).

**Content-Type**: `multipart/form-data`

**Request**:
```
POST /api/upload
Content-Type: multipart/form-data

file: <binary data>
session_id: "optional-session-id"
```

**Supported Formats**:
- Excel: `.xlsx`, `.xls` (compliance checking)
- CSV: `.csv` (compliance checking)
- PDF: `.pdf` (redaction, session search)
- Word: `.docx`, `.doc` (redaction, session search)

**Response**:
```json
{
  "file_id": "abc123...",
  "filename": "lab_data.xlsx",
  "size": 45678,
  "type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "status": "ready"
}
```

**Errors**:
- `400`: Invalid file type
- `413`: File too large (max 50MB)
- `500`: Processing error

**Note**: Files stored in RAM only, cleared on session disconnect.

---

### GET /api/files

List uploaded files for current session.

**Query Parameters**:
- `session_id` (optional): Filter by session ID

**Response**:
```json
{
  "files": [
    {
      "file_id": "abc123",
      "filename": "lab_data.xlsx",
      "size": 45678,
      "type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "uploaded_at": "2026-02-06T10:30:00Z"
    }
  ]
}
```

---

## Admin Endpoints

All admin endpoints require authentication via JWT Bearer token.

**Headers**:
```
Authorization: Bearer <token>
```

---

### GET /api/admin/status

System status and health metrics.

**Response**:
```json
{
  "llm": {
    "status": "connected",
    "models": ["glm-4.7-flash", "qwen3-32b", "qwen3-vl-8b"]
  },
  "storage": {
    "reports_path": "/app/data/reports",
    "total_reports": 5,
    "oldest_report_age_hours": 12.5
  },
  "rag": {
    "collection_name": "technical_knowledge",
    "total_docs": 1247,
    "total_chunks": 15832,
    "topics": ["general", "technical", "reference"]
  },
  "sessions": {
    "active_sessions": 3,
    "total_files": 7
  }
}
```

---

### GET /api/admin/config

Get current runtime configuration.

**Response**:
```json
{
  "ctx_small": 4096,
  "ctx_medium": 8192,
  "ctx_large": 16384,
  "ctx_xlarge": 24576,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "model_chat": "Qwen3-30B-A3B",
  "model_expert": "qwen3-32b",
  "model_vision": "qwen3-vl-8b",
  "rag_top_k": 5,
  "rag_min_score": 0.15,
  "reranker_enabled": true
}
```

---

### PUT /api/admin/config

Update runtime configuration (hot-swap, no restart required).

**Request**:
```json
{
  "temperature": 0.8,
  "model_chat": "Qwen3-30B-A3B",
  "rag_top_k": 10
}
```

**Response**:
```json
{
  "success": true,
  "updated": ["temperature", "model_chat", "rag_top_k"],
  "ignored": [],
  "config": {
    "temperature": 0.8,
    "model_chat": "Qwen3-30B-A3B",
    "rag_top_k": 10
  }
}
```

**Notes**:
- Only provided fields are updated
- Model names validated against available GGUF models
- Invalid fields logged and ignored

---

### POST /api/admin/config/reset

Reset configuration to defaults from environment variables.

**Response**:
```json
{
  "success": true,
  "message": "Configuration reset to defaults"
}
```

---

### GET /api/admin/logs

Recent application logs (last 500 entries).

**Query Parameters**:
- `level` (optional): Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

**Response**:
```json
{
  "logs": [
    {
      "timestamp": "2026-02-06T10:30:15.234Z",
      "level": "INFO",
      "module": "cohesionn.retriever",
      "message": "Retrieved 5 results for query 'safety factors'"
    }
  ],
  "count": 245
}
```

**Note**: Log buffer is limited to 500 entries. Logs are cleared on restart.

---

### GET /api/admin/sessions

List active WebSocket sessions and uploaded files.

**Response**:
```json
{
  "sessions": [
    {
      "session_id": "ws_abc123",
      "files": [
        {
          "file_id": "file_xyz",
          "filename": "lab_data.xlsx",
          "size": 45678
        }
      ],
      "connected_at": "2026-02-06T10:00:00Z"
    }
  ]
}
```

---

### POST /api/admin/test-extraction

Test document extraction with different extractors.

**Request**:
```
POST /api/admin/test-extraction
Content-Type: multipart/form-data

file: <PDF binary>
extractor: "auto" | "pymupdf" | "pymupdf4llm" | "marker" | "vision"
```

**Response**:
```json
{
  "extractor_used": "pymupdf",
  "extraction_time_ms": 1245,
  "page_count": 15,
  "text_length": 34567,
  "qa_score": 0.87,
  "issues": [
    {
      "type": "missing_tables",
      "severity": "medium",
      "affected_pages": [5, 8]
    }
  ],
  "sample_text": "First 500 characters of extracted text..."
}
```

---

### GET /api/admin/test-suite

List available QA test cases.

**Response**:
```json
{
  "test_suite": "qa_reference",
  "total_tests": 12,
  "documents": [
    {
      "doc_id": "synthetic_001",
      "name": "Technical Report - Complex Tables",
      "pages": 8,
      "accuracy_threshold": 0.85
    }
  ]
}
```

---

### POST /api/admin/test-suite/run

Run QA test suite against reference documents.

**Request**:
```json
{
  "doc_ids": ["synthetic_001", "synthetic_002"],
  "extractor": "auto"
}
```

**Response**:
```json
{
  "total_tests": 2,
  "passed": 1,
  "failed": 1,
  "results": [
    {
      "doc_id": "synthetic_001",
      "status": "passed",
      "accuracy": 0.92,
      "threshold": 0.85,
      "extraction_time_ms": 1234
    },
    {
      "doc_id": "synthetic_002",
      "status": "failed",
      "accuracy": 0.67,
      "threshold": 0.85,
      "issues": ["Table structure lost", "Missing equations"]
    }
  ]
}
```

---

### POST /api/admin/recalibrate

Trigger RAG pipeline recalibration (recompute embeddings, rebuild indices).

**Response**:
```json
{
  "success": true,
  "message": "Recalibration started",
  "estimated_time_minutes": 15
}
```

---

## Graph Endpoints

Neo4j knowledge graph administration.

**Base Path**: `/api/admin/graph`

---

### GET /api/admin/graph/stats

Graph database statistics.

**Response**:
```json
{
  "node_count": 1247,
  "relationship_count": 3456,
  "labels": ["Concept", "Document", "Person", "Location"],
  "relationship_types": ["MENTIONS", "RELATED_TO", "AUTHORED_BY"],
  "health": "healthy",
  "last_updated": "2026-02-06T09:00:00Z"
}
```

---

### GET /api/admin/graph/entities

List entities with pagination and filtering.

**Query Parameters**:
- `label` (optional): Filter by node label
- `search` (optional): Search by name
- `page` (optional): Page number (default: 1)
- `page_size` (optional): Items per page (default: 50, max: 100)

**Response**:
```json
{
  "entities": [
    {
      "id": "concept_12345",
      "name": "Safety Factor",
      "label": "Concept",
      "properties": {
        "definition": "Ratio of capacity to demand...",
        "importance": 0.92
      },
      "relationship_count": 15
    }
  ],
  "total": 1247,
  "page": 1,
  "page_size": 50
}
```

---

### GET /api/admin/graph/entity/{name}

Get detailed entity information.

**Response**:
```json
{
  "id": "concept_12345",
  "name": "Safety Factor",
  "label": "Concept",
  "properties": {
    "definition": "Ratio of capacity to demand used in engineering design",
    "importance": 0.92,
    "first_seen": "Technical_Reference.pdf"
  },
  "relationships": [
    {
      "type": "RELATED_TO",
      "target": "Design Criteria",
      "target_label": "Concept",
      "properties": {"weight": 0.85}
    }
  ]
}
```

---

### GET /api/admin/graph/visualization

Get graph data for force-directed visualization.

**Query Parameters**:
- `node_limit` (optional): Max nodes to return (default: 100, max: 500)
- `depth` (optional): Relationship depth (default: 2)

**Response**:
```json
{
  "nodes": [
    {
      "id": "concept_12345",
      "name": "Safety Factor",
      "label": "Concept",
      "size": 15
    }
  ],
  "edges": [
    {
      "source": "concept_12345",
      "target": "concept_67890",
      "type": "RELATED_TO",
      "weight": 0.85
    }
  ]
}
```

---

### POST /api/admin/graph/query

Execute Cypher query against Neo4j.

**Request**:
```json
{
  "query": "MATCH (c:Concept)-[:RELATED_TO]->(t:Concept) WHERE c.name = 'Safety Factor' RETURN t",
  "limit": 10
}
```

**Response**:
```json
{
  "results": [
    {
      "t": {
        "id": "concept_67890",
        "name": "Design Criteria",
        "label": "Concept"
      }
    }
  ],
  "count": 5,
  "execution_time_ms": 23
}
```

**Errors**:
- `400`: Invalid Cypher syntax
- `500`: Query execution error

---

## Benchmark Endpoints

Admin endpoints for running and analyzing RAG pipeline benchmarks.

**Base Path**: `/api/admin/benchmark`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/admin/benchmark/status` | Current benchmark status (running, idle) |
| POST | `/api/admin/benchmark/run` | Start a benchmark run (background thread) |
| GET | `/api/admin/benchmark/results` | List completed benchmark results |
| GET | `/api/admin/benchmark/results/{run_id}` | Get specific benchmark results |
| GET | `/api/admin/benchmark/chart/{name}` | Serve benchmark chart image |
| POST | `/api/admin/benchmark/discuss` | Conversational LLM analysis of benchmark results |
| POST | `/api/admin/benchmark/corpus/upload` | Upload corpus file for benchmarking |
| GET | `/api/admin/benchmark/corpus` | List uploaded corpus files |
| DELETE | `/api/admin/benchmark/corpus/{filename}` | Delete a corpus file |
| POST | `/api/admin/benchmark/corpus/generate-queries` | Auto-generate test queries via LLM |

---

## Lexicon Endpoints

Domain lexicon CRUD with cache refresh on write.

**Base Path**: `/api/admin/lexicon`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/admin/lexicon` | Get current domain lexicon |
| PUT | `/api/admin/lexicon` | Update domain lexicon (triggers cache refresh) |

---

## Knowledge Endpoints

Knowledge base management, ingestion, and retrieval testing.

**Base Path**: `/api/admin/knowledge`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/admin/knowledge/stats` | Knowledge base statistics by topic |
| GET | `/api/admin/knowledge/topics` | List available topics |
| POST | `/api/admin/knowledge/ingest` | Trigger ingestion (file or directory) |
| POST | `/api/admin/knowledge/upload` | Upload file for knowledge base |
| DELETE | `/api/admin/knowledge/file` | Delete a knowledge file |
| POST | `/api/admin/knowledge/reprocess` | Reprocess file with different extractor |
| GET | `/api/admin/knowledge/collections` | List Qdrant collections |
| DELETE | `/api/admin/knowledge/collection/{name}` | Purge a collection |
| POST | `/api/admin/knowledge/search` | Test retrieval query |

---

## Phii Endpoints

Phii behavior module management.

**Base Path**: `/api/admin/phii`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/admin/phii/stats` | Phii system statistics |
| POST | `/api/admin/phii/flag` | Flag a message for feedback |
| GET | `/api/admin/phii/corrections` | List learned corrections |
| DELETE | `/api/admin/phii/corrections/{id}` | Delete a correction |
| GET | `/api/admin/phii/patterns` | Get pattern statistics |
| GET | `/api/admin/phii/debug/{session_id}` | Debug session Phii state |
| GET | `/api/admin/phii/metrics` | Phii engagement metrics |

---

## Domain Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/domain` | Get current domain config (name, branding, tools, routes) |
| GET | `/api/domain/logo` | Get domain logo image (with fallback) |

---

## MCP Endpoints

Model Context Protocol (MCP) API for external AI tool integration. Any MCP-compatible client (Claude Desktop, GPT desktop apps, or custom agents) can use ARCA as a tool provider. These endpoints are read-only proxies to existing executor functions, gated by a static API key.

**Authentication**: `X-MCP-Key` header matching `MCP_API_KEY` env var. If `MCP_API_KEY` is unset, all MCP endpoints return `503 Service Unavailable`.

**Base Path**: `/api/mcp`

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/mcp/search` | Search knowledge base via RAG pipeline |
| POST | `/api/mcp/web-search` | Web search via SearXNG |
| POST | `/api/mcp/unit-convert` | Unit conversion |
| GET | `/api/mcp/topics` | List knowledge topics with enabled status |
| GET | `/api/mcp/stats` | Knowledge base statistics |
| GET | `/api/mcp/health` | Component health checks |

---

### POST /api/mcp/search

Search the ingested knowledge base using the hybrid RAG pipeline.

**Request**:
```json
{
  "query": "recommended safety factors for structural design",
  "topics": ["general"]
}
```

**Response**: Same as internal `execute_search_knowledge` -- includes `chunks`, `citations`, `avg_confidence`, `topics_searched`.

---

### POST /api/mcp/web-search

**Request**:
```json
{
  "query": "latest ISO 9001 quality management updates"
}
```

**Response**: Same as internal `execute_web_search` -- includes `results` array with `title`, `url`, `content`.

---

### POST /api/mcp/unit-convert

**Request**:
```json
{
  "value": 100,
  "from_unit": "kg",
  "to_unit": "lb"
}
```

**Response**:
```json
{
  "success": true,
  "result": 220.462,
  "expression": "100 kg = 220.462 lb"
}
```

---

### GET /api/mcp/topics

**Response**:
```json
{
  "topics": [
    {"name": "general", "enabled": true},
    {"name": "technical", "enabled": false}
  ],
  "enabled_count": 1,
  "total_count": 2
}
```

---

### GET /api/mcp/stats

**Response**:
```json
{
  "collections": [
    {"name": "general", "chunks": 15832}
  ],
  "total_chunks": 15832,
  "total_files": 12
}
```

---

### GET /api/mcp/health

**Response**:
```json
{
  "status": "healthy",
  "checks": {
    "llm": "ok",
    "redis": "ok",
    "qdrant": "ok",
    "postgres": "ok"
  }
}
```

**Errors**:
- `401`: Invalid API key
- `503`: MCP disabled (MCP_API_KEY not set)

---

## Error Responses

### Standard Error Format

```json
{
  "error": "Error message",
  "error_code": "ERROR_CODE",
  "details": {
    "additional": "context"
  }
}
```

---

### Common HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 400 | Bad Request | Invalid parameters, malformed JSON |
| 401 | Unauthorized | Missing or invalid JWT token |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Endpoint or resource doesn't exist |
| 413 | Payload Too Large | File exceeds 50MB limit |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side exception |
| 503 | Service Unavailable | LLM server down, database unavailable |

---

### Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `MESSAGE_TOO_LONG` | Message exceeds 10k char limit | Shorten message |
| `FILE_TOO_LARGE` | File exceeds 50MB | Reduce file size |
| `INVALID_FILE_TYPE` | Unsupported file format | Use .xlsx/.csv/.pdf/.docx |
| `LLM_UNAVAILABLE` | LLM service down | Check backend logs for llama-server errors |
| `MODEL_NOT_FOUND` | Requested model not pulled | Pull model via admin panel |
| `INVALID_CREDENTIALS` | Wrong admin password | Check password |
| `TOKEN_EXPIRED` | JWT expired (>8h) | Re-authenticate |
| `RATE_LIMITED` | Too many requests | Wait and retry |

---

## Rate Limits

- File upload: 10 requests per minute per IP
- Admin login: 5 attempts per 15 minutes
- Admin token verification: 20 requests per minute
- WebSocket frame size: 1 MB maximum

---

## Security Notes

ARCA is designed for local-first deployments. All LLM inference runs on your hardware — no data leaves your network unless web search is enabled.

**Before exposing to a network**:
- Change all default passwords in `.env`
- Restrict CORS origins to your specific domain (see `backend/main.py`)
- Use a reverse proxy with HTTPS termination (see [SECURITY.md](SECURITY.md))

---

**Last Updated**: 2026-02-10
