# Creating a Domain Pack

Domain packs specialize ARCA for a specific industry or use case. They add tools, vocabulary, personality, and branding without modifying core code.

## Structure

```
domains/mypack/
├── __init__.py          # Empty or package init
├── manifest.json        # Pack declaration (required)
├── lexicon.json         # Domain vocabulary and personality (required)
├── register_tools.py    # Tool registration function
├── tools/               # Tool modules
│   └── mytool/
│       ├── __init__.py
│       └── engine.py
├── executors/           # Tool execution logic
│   └── myexecutor.py
└── routes/              # Additional API endpoints
    └── myadmin.py
```

## manifest.json

Declares the pack's metadata, tools, and routes:

```json
{
  "name": "mypack",
  "display_name": "My Domain Pack",
  "version": "0.1.0",
  "arca_min_version": "0.1.0",
  "description": "Domain pack for [your industry].",
  "tools": ["mytool"],
  "handlers": [],
  "routes": ["myadmin"],
  "lexicon": "lexicon.json",
  "branding": {
    "app_name": "MyApp",
    "tagline": "Your custom tagline",
    "primary_color": "#6366f1",
    "description": "What this domain pack does"
  }
}
```

**Fields:**
- `name` — Directory name under `domains/`, used as `ARCA_DOMAIN` value
- `tools` — List of tool directory names in `tools/`
- `routes` — List of route module names in `routes/`
- `branding.app_name` — Displayed in the UI header, sidebar, welcome screen

## lexicon.json

Defines domain-specific vocabulary, personality, and detection patterns:

```json
{
  "identity": {
    "personality": "You are MyApp, a helpful assistant for [domain]...",
    "welcome_message": "Hello! I'm MyApp...\n\nWhat I can help with:\n- ..."
  },
  "topics": ["topic1", "topic2"],
  "thinking_messages": [
    "Processing",
    "Analyzing",
    "Working on it"
  ],
  "pipeline": {
    "specialty": "your domain description here",
    "reference_type": "a [your field] reference document",
    "raptor_context": "your field",
    "raptor_summary_intro": "Summarize the following [domain] content.",
    "raptor_preserve": ["Key items to preserve in summaries"],
    "confidence_example": "domain-specific confidence example",
    "equation_example": "$$your = domain + equations$$",
    "default_topic": "general"
  },
  "advanced_triggers": [],
  "skip_patterns": [],
  "technical_patterns": [
    "pattern1.*regex",
    "pattern2.*regex"
  ],
  "specialties": {
    "category1": {
      "keywords": ["keyword1", "keyword2"],
      "description": "Category description"
    }
  },
  "terminology_variants": {}
}
```

**Key sections:**
- `identity.personality` — System prompt personality injected into every LLM call
- `identity.welcome_message` — Shown on the welcome screen and streamed as intro
- `topics` — Used for RAG topic routing and embedding
- `thinking_messages` — Randomly displayed while the LLM is generating
- `technical_patterns` — Regex patterns that trigger domain-specific handling
- `specialties` — Keyword categories for Phii specialty detection
- `pipeline` — Domain expertise injected into the entire RAG pipeline (see below)

### pipeline section

The `pipeline` section controls how ARCA's RAG pipeline, system prompt, and summarizers behave for your domain. Without it, ARCA uses neutral defaults suitable for any field.

```json
"pipeline": {
  "specialty": "technical analysis and data interpretation",
  "reference_type": "a technical reference manual",
  "raptor_context": "technical analysis and applied sciences",
  "raptor_summary_intro": "Summarize the following technical content.",
  "raptor_preserve": [
    "Test methods and procedures (Method-A, Method-B, etc.)",
    "Equations and formulas",
    "Design parameters and typical values",
    "Safety factors and limits",
    "Standards references (ISO, ANSI, IEEE, etc.)"
  ],
  "confidence_example": "Method-A correlation, only 2 samples in dataset",
  "equation_example": "$$R = k_1 \\cdot F_a + k_2 \\cdot F_b + k_3 \\cdot F_c$$",
  "default_topic": "technical",
  "graph_test_methods": {"METHOD_A": ["method-a test"]},
  "graph_entity_types": {"MATERIAL": ["material", "composite"]},
  "graph_parameters": {"K_FACTOR": ["correction factor"]},
  "phii_technical_patterns": ["\\bMethod-A\\b", "\\bMethod-B\\b"],
  "phii_seed_corrections": [{"wrong": "...", "right": "..."}],
  "phii_compliance_suggestion": "Your domain compliance text",
  "rag_topic_descriptions": {"technical": "Technical analysis and data interpretation."},
  "rag_topic_keywords": {"technical": ["analysis", "measurement", "parameter"]},
  "trusted_search_domains": ["example.org"],
  "calculate_validation_examples": {},
  "hyde_detection_keywords": {"standards": ["ISO", "ANSI"]},
  "rag_synonyms": [["Method-A", "primary test method"]],
  "rag_preserve_terms": ["Method-A", "Method-B", "K_FACTOR"]
}
```

**Core fields (always recognized):**
- `specialty` — Injected into tool descriptions and system prompt. Tells the LLM what domain it serves.
- `reference_type` — HyDE (Hypothetical Document Embeddings) uses this as its persona when generating retrieval documents.
- `raptor_context` — RAPTOR hierarchical summarizer specialization. Controls how document summaries are framed.
- `raptor_summary_intro` — Opening line of RAPTOR summary prompts.
- `raptor_preserve` — List of content types RAPTOR should prioritize when summarizing.
- `confidence_example` — Example text shown in the system prompt for confidence level formatting.
- `equation_example` — KaTeX equation example in the system prompt.
- `default_topic` — Default Qdrant topic for benchmarks and ingestion.

**RAG domain-gating keys (added in v0.1.0 Wave 5):**
- `rag_topic_descriptions` / `rag_topic_keywords` — Feed the reranker's TopicRouter and domain boost.
- `rag_synonyms` — Query expansion synonym groups.
- `rag_preserve_terms` — BM25 tokenizer preserves these terms from splitting.
- `hyde_detection_keywords` — HyDE query type detection keywords.
- `graph_test_methods` / `graph_entity_types` / `graph_parameters` — GraphRAG entity extraction patterns.
- `phii_technical_patterns` — Phii energy detector technical patterns.
- `phii_seed_corrections` — Phii reinforcement seed corrections.
- `phii_compliance_suggestion` — Phii compliance suggestion text.
- `trusted_search_domains` — Trusted domains for web search scoring.
- `calculate_validation_examples` — Calculation validation examples.

**Pass-through behavior:** `get_pipeline_config()` passes through ALL keys from the lexicon `pipeline` section, not just the defaults. This means domain packs can inject arbitrary pipeline configuration that any component can read.

**Defaults (when pipeline section is omitted):** ARCA uses generic values -- "scientific and engineering disciplines", neutral equations, no domain-specific summarization. This is suitable for general-purpose use.

## register_tools.py

Registers domain tools with the core tool registry:

```python
from tools.registry import ToolRegistry, ToolDefinition, ToolCategory

def register_domain_tools():
    """Called by ARCA core at startup when this domain pack is loaded."""
    _register_mytool()

def _register_mytool():
    from domains.mypack.executors.myexecutor import execute_my_action

    ToolRegistry.register(ToolDefinition(
        name="my_action",
        description="Does something domain-specific",
        category=ToolCategory.SIMPLE,
        executor=execute_my_action,
        parameters={
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The input to process"
                }
            },
            "required": ["input"]
        }
    ))
```

The LLM sees the tool name, description, and parameters. It decides when to call the tool based on the user's query.

## Executors

Executors contain the actual logic for tool execution:

```python
# domains/mypack/executors/myexecutor.py

async def execute_my_action(params: dict) -> dict:
    """Execute the domain-specific action."""
    input_text = params.get("input", "")

    # Your domain logic here
    result = process_something(input_text)

    return {
        "result": result,
        "summary": f"Processed: {input_text[:50]}"
    }
```

Executors can import from:
- `domains.mypack.tools.*` — Your domain tool modules
- `services.*` — Core ARCA services (database, redis, etc.)
- `errors` — Core error types

## Routes

Additional API endpoints mounted by the domain pack:

```python
# domains/mypack/routes/myadmin.py
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/api/admin/mypack")

@router.get("/status")
async def get_status():
    return {"status": "ok", "domain": "mypack"}
```

Routes declared in `manifest.json` are automatically mounted at startup.

## Activation

Set `ARCA_DOMAIN=mypack` in `.env` and restart:

```bash
# Edit .env
ARCA_DOMAIN=mypack

# Restart
docker compose up -d --build
```

ARCA loads the pack's manifest, registers tools, mounts routes, and applies branding. The frontend fetches `/api/domain` and adapts the UI automatically.

## Testing

```bash
# Verify tools registered
docker compose exec backend python -c "
from tools.registry import register_all_tools, ToolRegistry
register_all_tools()
print(f'Tools: {len(ToolRegistry._tools)}')
for name in sorted(ToolRegistry._tools.keys()):
    print(f'  {name}')
"

# Check domain config
curl http://localhost:8000/api/domain | python -m json.tool

# Health check
curl http://localhost:8000/health
```

## Tips

- Start with `domains/example/` as a template
- Keep tool descriptions clear and concise — the LLM uses them for routing
- Test with `ARCA_DOMAIN=example` to verify core still works without your pack
- Domain tools are added to `sys.path` automatically — internal imports resolve without path hacking
