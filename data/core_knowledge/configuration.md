# ARCA Configuration

ARCA configuration is layered so operators can combine stable startup defaults with runtime tuning.

## Configuration Layers and Priority

ARCA effectively uses these layers:
1. Environment variables (`.env`) at startup.
2. Runtime overrides from admin/API updates.
3. Profile-controlled toggle sets for retrieval behavior.
4. Domain pack lexicon settings for specialization.

For retrieval toggles, profile resolution logic can override raw env defaults.

## Main Configuration Surfaces

### Environment File (`.env`)
Use for:
- Model filenames and model repository hints.
- Service URLs and credentials.
- Feature toggles that should persist across restarts.
- Startup behavior (auto download, cleanup, mcp mode, etc.).

### Runtime Config (`RuntimeConfig`)
Key implementation:
- `backend/config.py`

Provides:
- Dataclass-backed config singleton.
- Validation and persistence of overrides.
- Runtime updates used by admin APIs.

### Retrieval Profile Manager
Key implementation:
- `backend/profile_loader.py`

Provides:
- Named profiles (`fast`, `auto`, `deep`).
- Manual toggle overrides on top of active profile.
- Persistence of active profile + overrides.

## Critical Environment Variables

### Model and bootstrap
- `LLM_CHAT_MODEL`, `LLM_CODE_MODEL`, `LLM_EXPERT_MODEL`, `LLM_VISION_MODEL`, `LLM_VISION_STRUCTURED_MODEL`
- `LLM_<SLOT>_MODEL_REPO` for repository resolution.
- `ARCA_AUTO_DOWNLOAD_MODELS`, `ARCA_AUTO_DOWNLOAD_OPTIONAL_MODELS` for startup bootstrap behavior.

### Retrieval core
- `RAG_TOP_K`, `RAG_MIN_SCORE`, `RAG_MIN_FINAL_SCORE`
- `BM25_ENABLED`, `HYDE_ENABLED`, `RERANKER_ENABLED`
- `DOMAIN_BOOST_ENABLED`, `DOMAIN_BOOST_FACTOR`
- `RAG_DIVERSITY_ENABLED`, `RAG_DIVERSITY_LAMBDA`

### Profile and topic controls
- `RETRIEVAL_PROFILE` (fast/auto/deep/custom behavior)
- `ENABLED_TOPICS`
- `CORE_KNOWLEDGE_ENABLED`, `CORE_KNOWLEDGE_COLLECTION`

### SearXNG / web retrieval
- `SEARXNG_ENABLED`, `SEARXNG_URL`
- `SEARXNG_CATEGORIES`, `SEARXNG_LANGUAGE`
- `SEARXNG_TIMEOUT_S`, `SEARXNG_MAX_RESULTS`

### Hardware and runtime placement
- `ARCA_DEVICE_MAP` for multi-GPU routing.
- Context window vars: `LLM_CTX_SMALL|MEDIUM|LARGE|XLARGE`.
- Optional GPU layer and cache settings.

### Runtime mode controls
- `MCP_MODE` to disable local chat generation and run as tool backend mode.
- `CLEANUP_ON_STARTUP` and related cleanup intervals.

## Runtime Override Persistence

Runtime updates from admin/API can be written to persistent overrides so settings survive restarts. Sensitive credentials remain env-centric.

## Topic Enablement and Core Knowledge

Topic enablement is used for retrieval scope. When core knowledge is enabled, ARCA should include the core topic (`arca_core` by default) so system self-documentation remains searchable.

If capability questions produce shallow answers, verify:
- Core knowledge ingestion completed.
- `CORE_KNOWLEDGE_ENABLED=true`.
- Topic list includes `arca_core`.

## Model Assignment Behavior

ARCA supports model assignment via:
- Env defaults at boot.
- Admin runtime assignment where compatible files exist.

Model files are expected in mounted `./models` storage, not baked into app images.

## Recommended Config Workflow for New Deployments

1. Set `.env` with model slots and service credentials.
2. Bring stack up and confirm startup health.
3. Let model bootstrap pull missing files if enabled.
4. In admin UI, verify model assignments and retrieval profile.
5. Ingest representative docs.
6. Run benchmark and apply winner profile.
7. Persist final runtime overrides.

## Security and Secrets Guidance

- Keep admin/API secrets out of source control.
- Rotate default passwords before exposed deployments.
- Use least-privileged networking around admin APIs.

## Related Docs

- `platform_overview.md`: system architecture context.
- `retrieval_pipeline.md`: retrieval toggle effects.
- `benchmarks.md`: benchmark-driven tuning flow.
- `troubleshooting.md`: config and startup failure diagnosis.
