# ARCA MCP Integration

ARCA can act as a tool backend for external AI clients through MCP-style endpoints and adapters.

## Core Idea

In MCP-style usage, an external model handles conversation while ARCA provides:
- Local document retrieval.
- Web retrieval through your configured search backend.
- Utility tools (unit conversion, status checks, corpus stats).

This model is useful when you want strong external reasoning with locally operated retrieval infrastructure.

## ARCA MCP Surfaces

ARCA exposes API routes under:
- `backend/routers/mcp_api.py`
- Route prefix: `/api/mcp/*`

The host-side MCP adapter can broker between desktop clients and ARCA HTTP endpoints.

## Typical Available MCP Tools

Tool set can vary by runtime/domain, but commonly includes:
- `search_knowledge`
- `web_search`
- `unit_convert`
- `list_topics`
- `corpus_stats`
- `system_status`

These map to the same underlying executors used by in-app chat workflows.

## MCP Mode vs Mixed Mode

### MCP Mode (`MCP_MODE=true`)
Behavior:
- Local chat generation path is disabled.
- Local llama-server startup/validation may be skipped.
- ARCA primarily serves tool/API behavior.

Use when:
- You want ARCA as a retrieval/tool backend for external clients only.

### Mixed Mode (`MCP_MODE=false`)
Behavior:
- Local ARCA chat remains active.
- MCP endpoints can still be available for external clients.

Use when:
- You want both local chat UI and external tool access.

## Authentication and Security

Use a dedicated API key value for MCP access (for example `MCP_API_KEY`) and keep it out of public source control.

Operational guidance:
- Treat MCP endpoints as privileged internal APIs.
- Restrict network exposure to trusted hosts/users.
- Rotate keys if there is any access concern.

## Data and Privacy Model

ARCA keeps corpus data local by default. External clients receive only tool outputs returned by endpoint calls.

Practical note:
- If an external cloud model is the orchestrator, tool outputs sent to it are part of that external model session context.
- Keep this in mind for sensitive workflows.

## Setup Outline

1. Configure ARCA with required API key and mode.
2. Start ARCA stack and verify backend health.
3. Configure host-side MCP adapter/client integration.
4. Validate tool calls with a simple endpoint or client test.

## Troubleshooting Checklist

- Verify backend reachable on expected host/port.
- Verify key values match between adapter and ARCA config.
- Confirm `MCP_MODE` aligns with desired behavior.
- Check backend logs for auth or route errors.

## Related Docs

- `configuration.md`: env vars and mode flags.
- `tools.md`: tool semantics exposed through MCP.
- `troubleshooting.md`: startup and connectivity debugging.
