# ARCA Admin Guide

The admin panel is the operational control plane for ARCA. It is used to configure runtime behavior, manage knowledge, inspect health, and run benchmarks.

## Access and Authentication

Access paths:
- `/admin` route in the frontend.
- Shortcut path if configured in UI.

Authentication model:
- First run requires admin account setup.
- Subsequent access requires authenticated admin session.

If login state appears broken, check backend auth logs and database health.

## Operational Areas

### Dashboard / Status
Provides a high-level health summary:
- Service health cards (backend dependencies and model status).
- Session counts and high-level runtime indicators.
- Quick visual signal for degraded components.

Use this first when triaging reports from users.

### Configuration
Main runtime tuning surface:
- Model assignment by slot.
- Generation controls (temperature, top_p, top_k, context windows).
- Retrieval profile and stage toggles.
- Web search and related external retrieval settings.

Changes here are designed to apply without full stack restart in most cases.

### Knowledge
Knowledge management features include:
- Topic/file ingestion operations.
- Reindex/reprocess controls where available.
- Search testing against indexed corpus.
- Topic enable/disable controls for retrieval scope.

Core ARCA docs are represented in core topic data when enabled.

### Domain
Domain controls include:
- Active domain pack selection where allowed.
- Domain metadata visibility.
- Domain-specific behavior alignment with lexicon/tooling.

### Benchmark
Benchmark area supports:
- Layered benchmark runs.
- Provider/judge configuration.
- Progress and result review.
- Apply-winners workflow for retrieval profile updates.

### Intelligence (Phii and related controls)
Depending on build state, this area can expose:
- Personality or behavior controls.
- Reinforcement/correction inspection.
- Specialty/expertise behavior insights.

If fine-tuning surfaces are unavailable in this release build, they may be intentionally hidden until pipeline dependencies are present.

### Tools
Tools area is for runtime tool visibility and management:
- Active tool listing.
- Custom tool scaffolding/integration paths where enabled.
- Domain-provided tool visibility.

### Diagnostics
Diagnostics area is used for deep troubleshooting:
- Log streaming and error visibility.
- Health/status endpoint inspection.
- Request/response testing helpers.
- Session troubleshooting helpers.

## Typical Admin Workflows

### First deployment hardening
1. Complete admin setup.
2. Verify service health cards.
3. Confirm model assignments.
4. Ingest starter corpus.
5. Run benchmark quick pass.
6. Apply winners and validate chat quality.

### Model swap workflow
1. Place model file in mounted model directory.
2. Refresh model list in admin.
3. Assign model to target slot.
4. Run smoke queries and monitor backend logs.

### Retrieval tuning workflow
1. Start from fast profile.
2. Reproduce failure query in search tester.
3. Toggle targeted retrieval stages.
4. Re-run query and compare citations/confidence.
5. Persist changes if stable.

## Admin Safety Guidance

- Change one high-impact setting at a time during live debugging.
- Keep notes of changed settings before/after benchmark apply.
- Treat exposed admin endpoints as privileged interfaces.

## Related Docs

- `configuration.md`: setting-level guidance.
- `benchmarks.md`: benchmark strategy and apply-winners flow.
- `troubleshooting.md`: failure triage and recovery.
