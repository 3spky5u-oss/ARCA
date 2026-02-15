# Adding Tools to ARCA

## Quick Start

```bash
python scripts/new_tool.py mytool "Description here"
# Creates: backend/tools/mytool/, docs/tools/MYTOOL.md
```

## After Scaffolding

1. Register in `backend/tools/registry.py` (inside `_register_core_tools()` for core tools, or in `domains/{name}/register_tools.py` for domain tools)
2. Implement in `backend/tools/mytool/core.py`
3. Create an executor in `backend/routers/chat_executors/` that the tool dispatches to
4. Add to dispatch whitelist if tool returns `analysis_result`

## Tool Categories

| Category | Use For |
|----------|---------|
| SIMPLE | Quick calculations, conversions |
| RAG | Semantic search, knowledge retrieval |
| ANALYSIS | Data processing, compliance checks |
| EXTERNAL | Web search, external APIs |
| DOCUMENT | File processing, generation |

## Registry Fields

Required:
- `name` - Tool identifier used in LLM function calling
- `description` - Full description (the LLM uses this to decide when to call the tool)
- `parameters` - OpenAI-compatible JSON schema for parameters
- `required_params` - List of required parameter names
- `executor` - Callable that implements the tool logic
- `category` - One of the `ToolCategory` enum values

Optional:
- `friendly_name` - Display name in UI
- `brief` - One-line description (used in auto-generated TOOLS_SECTION for system prompt)
- `provides_citations` - Set true if tool returns citation sources
- `updates_session` - Set true if tool modifies session state
- `triggers_analysis_result` - Set true if returning structured analysis (enables UI download buttons)
- `extracts_nested_result` - Set true if result has a nested `analysis_result` to extract

## Registration Flow

1. `main.py` startup calls `register_all_tools()`
2. Guard: `if ToolRegistry._initialized: return`
3. `_register_core_tools()` registers 5 built-in tools: `unit_convert`, `search_knowledge`, `search_session`, `web_search`, `redact_document`
4. Domain tools loaded via `importlib.import_module(f"domains.{domain.name}.register_tools")`
5. Domain's `register_domain_tools()` calls `ToolRegistry.register()` for each tool
6. `ToolRegistry.generate_tools_section()` builds the TOOLS_SECTION for the system prompt (single source of truth)

## Error Handling

Use the `@handle_tool_errors` decorator and raise error subclasses:

```python
from errors.handlers import handle_tool_errors
from errors.exceptions import ParseError, ValidationError

@handle_tool_errors("mytool")
async def execute(params):
    if invalid:
        raise ValidationError("Clear message", details={"field": "value"})
```

Exception hierarchy: `ARCAError` -> `ParseError`, `ValidationError`, `NotFoundError`, `LLMError`, `ExternalServiceError`, `DependencyError`.

See `backend/errors/` for the full exception hierarchy and handler implementations.
