# Security Guide

ARCA runs as a local-first application. All services run on your machine — no data leaves your network unless you explicitly enable web search.

## First Steps After Install

1. **Change default passwords** in `.env`:
   - `POSTGRES_PASSWORD` — PostgreSQL database
   - `NEO4J_PASSWORD` — Neo4j graph database

2. **Create your admin account** at first launch — the setup wizard prompts you automatically.

3. **Review port exposure** -- default compose exposes frontend/backend publicly and binds data stores to localhost (`127.0.0.1`). For production deployments, place frontend/backend behind a reverse proxy and keep internal services localhost-only:
   | Service     | Port | Purpose              | Recommendation |
   |-------------|------|----------------------|----------------|
   | Frontend    | 3000 | Web UI               | Public (via reverse proxy) |
   | Backend API | 8000 | REST + WebSocket     | Public (via reverse proxy) |
   | PostgreSQL  | 5432 | User data, feedback  | Bind to `127.0.0.1` |
   | Qdrant      | 6333 | Vector store         | Bind to `127.0.0.1` |
   | Redis       | 6379 | Session cache        | Bind to `127.0.0.1` |
   | Neo4j       | 7474 | Knowledge graph      | Bind to `127.0.0.1` |
   | SearXNG     | 8080 | Web search proxy     | Bind to `127.0.0.1` |

   To restrict, ensure port mappings use `127.0.0.1:host:container` for non-public services.

## Reverse Proxy Setup

If exposing ARCA beyond localhost, use a reverse proxy (nginx, Caddy, Traefik).

Key requirements:
- **WebSocket support** — the chat interface uses WebSockets on `/ws`
- **HTTPS termination** — handle TLS at the proxy, not in ARCA
- **CORS** — ARCA allows origins matching `*:3000` by default. Update the regex in `backend/main.py` if your domain differs.
- **Request size** — ARCA limits uploads to 50 MB and API calls to 1 MB

Example nginx location block:
```nginx
location / {
    proxy_pass http://localhost:3000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}

location /api/ {
    proxy_pass http://localhost:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

## Authentication

- **bcrypt + JWT**: Passwords hashed with bcrypt, sessions use HS256 JWT tokens
- **First-boot setup**: First user to register becomes admin (no default credentials in production)
- **Admin rate limiting**: 5 login attempts per 15 minutes per IP
- **Password change**: Rotates JWT secret (invalidates all active sessions)
- **Recovery**: Set `ADMIN_RESET=true` in `.env`, restart, re-enter setup flow

## Data Privacy

- All document processing happens locally (embedding, chunking, graph extraction)
- LLM inference runs on your GPU via llama.cpp — no API calls to cloud providers
- Web search (SearXNG) is the only feature that contacts external servers, and it's optional
- Upload filenames are sanitized before processing to prevent prompt injection

## Known Considerations

- `.env` contains database passwords -- ensure it stays in `.gitignore` (it is by default)
- The CORS regex matches localhost, private network IPs (100.x, 192.168.x), and hostnames on port 3000. For production deployments, restrict to your specific domain in `backend/main.py`.
- Neo4j Cypher console blocks write queries by default (regex-based pattern matching blocks CREATE, MERGE, DELETE, SET, REMOVE, DROP). An `allow_writes` parameter exists for admin operations.
- WebSocket frame size is limited to 1 MB to prevent memory exhaustion (uvicorn `ws_max_size`)
- Rate limiting is Redis-backed. If Redis is unavailable, rate limits are bypassed (fallback mode). Monitor Redis health.

## Container Security

- **Backend process runs as non-root (`appuser`)** after entrypoint setup. Entrypoint starts as root briefly to fix mounted volume ownership, then drops privileges.
- **No TLS between containers** -- all inter-service communication is unencrypted HTTP. On shared networks, use Docker network isolation.
- **SearXNG uses `latest` tag** -- pin to a specific version for reproducible builds.
- **Pickle files** (`data/guidelines/*.pkl`, BM25 indices) are a deserialization risk if files are tampered with. They are mounted read-only from the host.
- **16 GB tmpfs for uploads** could exhaust container memory under abuse. Monitor upload rates.

## AGPL/GPL Dependencies

PyMuPDF (AGPL-3.0) and marker-pdf (GPL-3.0) are used for document extraction. If deploying commercially, review license compatibility or acquire commercial licenses from Artifex (PyMuPDF).

## Reporting Security Issues

If you discover a security vulnerability, please report it privately rather than opening a public issue. Contact the maintainer directly.
