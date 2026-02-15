# ARCA Troubleshooting

This guide covers common operational failures in ARCA and practical recovery steps.

## Fast Triage Sequence

When users report problems, check in this order:
1. `docker compose ps` for container health.
2. Backend logs for startup warnings/errors.
3. Admin status panel for service/model state.
4. Model directory contents and assigned filenames.
5. Retrieval topic state and ingestion counts.

## Startup and Model Issues

### Symptom: Missing model errors at startup
Typical log clues:
- "No GGUF files found"
- "Model file not found"

Fix:
- Verify files exist in mounted `./models` directory.
- Ensure configured model filenames match exact GGUF names.
- Confirm auto-download flags are enabled if relying on bootstrap.

### Symptom: Model redownload confusion
If logs show downloads unexpectedly:
- Verify model volume persistence and mount path.
- Confirm you are not rebuilding into an empty host path.
- Check whether optional slots are configured to different files.

### Symptom: Slow first startup
Large GGUF + mmproj downloads and first model load can be slow.
- This is expected on first run.
- Subsequent runs should be faster if model storage is persistent.

## GPU and Acceleration Issues

### Symptom: CUDA/NVIDIA not detected in container
Typical causes:
- Docker runtime not configured for GPU passthrough.
- Host driver/runtime mismatch.
- Incorrect compose GPU reservation settings.

Checks:
- Run `nvidia-smi` on host.
- Verify Docker GPU access in container context.
- Validate compose GPU config and toolkit installation.

### Symptom: ONNX falls back to torch
You may see warnings that ONNX export/load failed and torch fallback was used.

Interpretation:
- Not always fatal.
- It can still run, but with different memory/performance characteristics.

Action:
- Check backend dependency versions.
- Validate GPU availability and provider selection.

### Symptom: Unexpected CPU execution
If embedding/rerank/model paths run on CPU:
- Check `COHESIONN_*_DEVICE` values.
- Check runtime logs for provider selection and fallback reasons.
- Confirm GPU is visible from the backend container.

## Database and Service Issues

### Symptom: PostgreSQL auth failures / SQLite fallback
Typical cause:
- Credentials mismatch between env and initialized volume.

Fix options:
- Correct `POSTGRES_*` env vars to match existing volume.
- Or reinitialize postgres data if you intentionally changed credentials.

### Symptom: Neo4j critical error state
Action:
- Inspect Neo4j logs directly.
- Restart Neo4j container.
- Verify disk space and memory limits.
- Disable graph-dependent retrieval temporarily if needed.

### Symptom: SearXNG web search failures
Check:
- `SEARXNG_URL` reachable from backend container.
- SearXNG container health.
- Request format/timeouts in runtime config.

## Knowledge and Retrieval Issues

### Symptom: Weak or generic answers
Check:
- Ingestion status and chunk counts.
- Topic enablement state.
- Whether core knowledge topic is enabled (`arca_core`).
- Active retrieval profile and toggle state.

### Symptom: "Cannot explain features" quality issues
Check:
- Core knowledge docs present in `data/core_knowledge`.
- Core knowledge ingestion completed at startup or re-ingested.
- Retrieval profile not over-pruned.

### Symptom: Empty result sets
Possible causes:
- Thresholds too aggressive.
- Topic routing mismatch.
- Missing ingest artifacts.

Fix:
- Lower thresholds temporarily for diagnosis.
- Run direct search tester from admin.
- Reindex affected topic.

## Prompt and Output Issues

### Symptom: Response includes "Sources:" text in message body
Expected behavior in current UX:
- UI should handle citations separately.

Fix:
- Ensure prompt rules and cleanup logic are current.
- Verify no legacy prompt templates remain in active domain pack overrides.

### Symptom: Messy markdown list layout
Action:
- Verify markdown renderer CSS and list component mappings.
- Confirm model prompt instructs bullet-list structure for feature inventories.

## Ingestion Performance Issues

### Symptom: Ingestion is very slow
Possible drivers:
- Vision extraction activated on many pages.
- Large corpora with expensive reranking/graph pipelines.
- GPU resource contention with chat inference.

Fix:
- Reduce concurrent heavy paths during ingest.
- Disable optional heavy stages for initial indexing.
- Re-enable selectively after baseline ingest.

## Recovery Patterns

### Safe reset pattern
1. Export/backup needed data.
2. Stop stack.
3. Clear only corrupted service volumes.
4. Restart and validate each service.
5. Re-ingest known-good corpus sample before full run.

### Configuration rollback pattern
1. Capture current runtime overrides.
2. Reset to known baseline profile.
3. Reapply only validated changes incrementally.

## Log Inspection Tips

Focus on backend logs for:
- Startup phase transitions.
- Model bootstrap decisions.
- Retrieval profile and topic routing decisions.
- Service fallback warnings.
- Tool execution failures.

## Related Docs

- `configuration.md`: config semantics and override behavior.
- `admin_guide.md`: where to operate fixes from UI.
- `retrieval_pipeline.md`: retrieval-specific tuning and failure causes.
