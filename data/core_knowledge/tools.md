# ARCA Core Tools

This document describes the always-available ARCA core tools and how they participate in runtime flows.

Tool registration is centralized in `backend/tools/registry.py`.

## Tool Registry Model

ARCA uses a registry pattern:
- Each tool has a name, schema, category, and executor.
- The registry emits OpenAI-style tool schemas to the model runtime.
- Tool execution returns normalized result objects with optional citations.

Core tools are always registered. Domain tools are optional and loaded from the active domain pack.

## Core Tool Set

### `search_knowledge` (Cohesionn)
Purpose:
- Retrieve evidence from the ingested local corpus.

Execution path:
- `execute_search_knowledge` -> Cohesionn retriever pipeline.

Typical use:
- Technical Q&A where corpus evidence is required.
- Product/system self-explanation when core knowledge is enabled.

Output characteristics:
- Ranked chunks with metadata and citations.
- Confidence estimate and retrieved context.

### `search_session`
Purpose:
- Search temporary files uploaded in the current chat/session scope.

Execution path:
- Session-scoped retrieval path, separate from persistent corpus collections.

Typical use:
- Ad hoc analysis of a one-off file without full corpus ingestion.

### `web_search`
Purpose:
- Retrieve current/external information via SearXNG.

Execution path:
- Web-search executor -> SearXNG backend.

Typical use:
- Questions about recent events or data not present in local corpus.

Output characteristics:
- Result snippets and URL metadata.
- UI-level citation display handles source presentation.

### `web_fallback_search`
Purpose:
- Conditional web retrieval path used when enabled by runtime config.

Execution path:
- Uses the same underlying web search executor with config gating.

Typical use:
- Corrective fallback patterns in workflows that permit external retrieval.

### `unit_convert`
Purpose:
- Deterministic unit conversion for engineering/scientific workflows.

Execution path:
- Direct calculator-like conversion executor.

Typical use:
- Pressure, length, area, flow, thermal, mass, force, and related conversions.

Output characteristics:
- Structured numeric result with unit labels.

### `redact_document` (Redactrr)
Purpose:
- Detect and redact sensitive information in documents.

Execution path:
- Redaction pipeline combining pattern, heuristic, and optional model-based logic.

Typical use:
- Preparing documents for external sharing.

Output characteristics:
- Redaction actions and output artifacts that can be downloaded.

## Tool Categories

Tool categories in registry metadata:
- `SIMPLE`: deterministic calculations/lookups.
- `RAG`: corpus and session retrieval.
- `EXTERNAL`: web/API retrieval.
- `DOCUMENT`: document processing pipelines.
- `ANALYSIS`: analysis workflows (used by some domain packs).

Categories are useful for UI grouping, orchestration logic, and maintainability.

## How Tool Invocation Works at Runtime

1. Chat orchestration builds prompt and tool schema list.
2. Model can emit tool calls using registered function signatures.
3. Executor runs with validated arguments.
4. Results are normalized to a `ToolResult` shape.
5. Citations/metadata are attached for UI rendering.
6. Model receives tool output and produces final user-facing response.

## Tool Reliability Practices

ARCA tool execution includes:
- Argument filtering to match executor signatures.
- Error trapping with structured failure payloads.
- Config-gated tool visibility (`requires_config`).
- Domain tool import isolation so missing optional packs do not break core behavior.

## Domain Tool Extension

Domain packs can contribute tools by providing:
- `domains/<pack>/register_tools.py`
- Optional `tools/`, `executors/`, and routes.

At runtime:
- Core tools register first.
- Active domain tools register second.
- Optional admin-generated custom tools can register afterward.

This allows extensibility without changing core registry logic.

## Practical Guidance

- Use `search_knowledge` for evidence-backed answers.
- Use `web_search` only for external/current knowledge gaps.
- Keep `unit_convert` and similar deterministic operations tool-driven.
- Use redaction tools before exporting sensitive docs.

## Related Docs

- `retrieval_pipeline.md`: internals of Cohesionn retrieval.
- `domain_packs.md`: how domain-specific tools are added.
- `mcp_integration.md`: exposing tool calls to external AI clients.
