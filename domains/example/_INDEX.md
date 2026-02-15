# domains/example/

The vanilla ARCA domain pack. Provides the default (domain-agnostic) configuration with no specialized tools, handlers, or routes. Serves as the template for creating custom domain packs and as the fallback when `ARCA_DOMAIN` is unset.

| File | Purpose | Key Exports |
|------|---------|-------------|
| __init__.py | Package marker | -- |
| manifest.json | Pack metadata: name, version, branding (app_name="ARCA", primary_color="#6366f1"), empty tools/handlers/routes | Loaded by `domain_loader.get_domain_config()` |
| lexicon.json | Domain knowledge injection: identity, topics, pipeline config (neutral defaults), empty specialties/terminology | Loaded by `domain_loader.get_lexicon()` |

## Lexicon Structure

The lexicon.json contains these top-level sections:

| Section | Purpose |
|---------|---------|
| `identity` | Personality prompt text and welcome message |
| `topics` | Knowledge base topic list (default: `["general"]`) |
| `thinking_messages` | Rotating UI messages during LLM processing |
| `pipeline` | RAG pipeline config: specialty, reference_type, RAPTOR settings, graph patterns, Phii patterns, topic descriptions/keywords, synonyms, preserve terms, HyDE keywords |
| `advanced_triggers` | Patterns triggering advanced mode |
| `skip_patterns` | Patterns to skip in processing |
| `technical_patterns` | Technical term detection patterns |
| `specialties` | Specialty keyword sets (empty = detection disabled) |
| `terminology_variants` | Term variant mappings for Phii mirroring |

## Creating a New Domain Pack

1. Copy `domains/example/` to `domains/yourpack/`
2. Edit `manifest.json` (name, branding, tools/handlers/routes)
3. Edit `lexicon.json` (pipeline config, specialties, terminology)
4. Optionally add `register_tools.py` for custom tool registration
5. Set `ARCA_DOMAIN=yourpack` in environment

See [docs/dev/domain_packs.md](../../docs/dev/domain_packs.md) for the full domain pack development guide.
