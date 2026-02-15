# Launch Playbook (v0.1.0)

This is a maintainer runbook for launch week.

## 1. Release Checklist (Execution)

1. Confirm `main` is green and public export is synced.
2. Push tag `v0.1.0`.
3. Publish GitHub release with notes from `docs/RELEASE_NOTES_v0.1.0.md`.
4. Pin roadmap/limitations issue.
5. Post launch announcements.

## 2. Suggested GitHub Issue To Pin

Title:

`Known Limitations + Near-Term Roadmap (v0.1.0)`

Body:

```md
This issue is the single place for:
- known limitations in v0.1.0
- near-term roadmap priorities
- high-signal follow-up items from real users

If you're filing a new issue:
- use bug/setup templates for reproducible defects
- use feature requests for scoped proposals

Current priorities:
1. setup reliability (doctor + bootstrap UX)
2. runtime stability/perf regressions
3. benchmark usability and result quality

Lower priority for now:
- broad architecture rewrites
- niche integrations without reproducible demand
```

## 3. Announcement Drafts

### Reddit / Discord (long form)

```md
I just open-sourced ARCA (v0.1.0), a local-first LLM + RAG platform for technical document workflows.

What it does:
- local chat + retrieval + ingestion + admin controls in one stack
- benchmark harness so you can tune retrieval against your own corpus
- model playground behavior (swap GGUF models/settings at runtime)
- domain-pack customization (tools/vocab/personality) without forking core code

Quick start is one command:
- Windows: `./scripts/bootstrap.ps1`
- Unix/macOS: `./scripts/bootstrap.sh`

Important first-run note:
- if models are missing, ARCA auto-downloads a tier-selected set
- this can be large/slow (roughly ~10-30 GB depending on detected VRAM tier)

Repo: https://github.com/3spky5u-oss/ARCA

If you try it and hit issues, please file reproducible setup bugs with logs/environment details.
```

### X / Short Post

```md
Open-sourced ARCA v0.1.0:
Local-first LLM + RAG platform + benchmark harness + model playground.

One-command bootstrap:
Windows `./scripts/bootstrap.ps1`
Unix/macOS `./scripts/bootstrap.sh`

First run can auto-download ~10-30GB of models (VRAM-tiered).

Repo: https://github.com/3spky5u-oss/ARCA
```

### Hacker News (Show HN)

Title:

`Show HN: ARCA – local-first RAG platform + benchmark harness for technical corpora`

Post:

```md
I built and open-sourced ARCA: a local-first LLM + RAG platform focused on technical document workflows.

Core idea: give people a complete local stack (chat/retrieval/ingestion/admin) but keep tuning evidence-based with a built-in benchmark harness.

Key points:
- one-command bootstrap
- VRAM-tiered model defaults + auto-download if missing
- admin panel for runtime tuning (no SSH required)
- domain-pack approach for specialization without forking core

Repo: https://github.com/3spky5u-oss/ARCA
Whitepaper: https://github.com/3spky5u-oss/ARCA/blob/main/docs/WHITEPAPER.md
Benchmark report: https://github.com/3spky5u-oss/ARCA/blob/main/docs/BENCHMARK_V2_PUBLIC.md

Known caveat: first run may pull large model files, so setup can be bandwidth/time heavy depending on hardware tier.
```

## 4. First-Week Triage Policy

For launch week, prioritize:

1. Reproducible setup failures (doctor/bootstrap/build/startup).
2. Crashes, dead services, hard regressions.
3. Incorrect/corrupt outputs in core retrieval/benchmark paths.

De-prioritize:

1. broad refactors without clear failure reports
2. feature requests without concrete use cases
3. edge-case performance tuning without reproducible baseline

Response standard:

- ask for command + logs + environment upfront
- reproduce before promising fix
- close low-signal/non-reproducible issues politely with request for more detail
