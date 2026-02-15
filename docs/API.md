# API Quick Reference

Use this as the fast path. Full endpoint-level detail is in `docs/API_FULL.md`.

## Base URLs

- Backend root: `http://localhost:8000`
- API prefix: `/api`

## Health Checks

- `GET /health`
- `GET /api/admin/status` (admin auth required)

## Chat

- `WS /ws/chat`

Typical flow:
1. Upload/ingest knowledge in the Admin Panel.
2. Send chat requests via WebSocket.
3. Use retrieval profiles (`fast`/`deep`) for latency vs depth.

## Admin (Most Used)

- `GET /api/admin/config` and `PUT /api/admin/config`
- `GET /api/admin/knowledge/topics`
- `POST /api/admin/knowledge/ingest`
- `DELETE /api/admin/knowledge/topic/{topic}`
- `GET /api/admin/benchmark/*`

## Knowledge / Retrieval

- Topic management and ingestion are exposed under `/api/admin/knowledge/*`.
- Runtime retrieval behavior is controlled via config + active profile.

## MCP Endpoints

When `MCP_API_KEY` is set, ARCA exposes MCP-safe tool endpoints for external clients.

- `POST /api/mcp/search`
- `POST /api/mcp/upload`
- `POST /api/mcp/web-search`
- `POST /api/mcp/unit-convert`
- `GET /api/mcp/topics`
- `GET /api/mcp/stats`
- `GET /api/mcp/health`
- `GET /api/mcp/tools`
- `POST /api/mcp/execute`

Auth: set `ARCA_MCP_KEY` in your MCP client to the same value as backend `MCP_API_KEY`.

## Errors

- Standard HTTP status codes + JSON error payloads.
- Check backend logs for model/runtime failures:

```bash
docker compose logs -f backend
```

## Full Reference

- Detailed contracts and examples: `docs/API_FULL.md`
