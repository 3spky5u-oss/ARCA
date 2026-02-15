# ARCA v0.1.0 Release Notes

Initial public OSS release.

## Highlights

- One-command bootstrap for Windows/Linux/macOS wrappers.
- GPU-first defaults with VRAM-tiered model auto-selection.
- Automatic first-run model download for configured slots.
- Admin panel improvements for models, knowledge, diagnostics, and tools.
- Benchmark harness + public benchmark report for evidence-driven tuning.
- Expanded public documentation (README, config/troubleshooting guides, whitepaper).

## Setup and Reliability

- New preflight doctor with actionable fix commands:
  - Docker and Compose checks
  - GPU host/container checks
  - RAM/disk/ports checks
  - `.env`/models/compose config checks
- Optional machine-readable doctor output:
  - `python scripts/arca.py doctor --json --json-out data/preflight.json`
- Startup reliability hardening:
  - PostgreSQL startup retry window before fallback
  - clearer diagnostics for model/startup failures
  - compose override conflict fixes for GPU paths

## LLM Runtime and Hardware

- Multi-GPU tensor split support (env-driven).
- MoE CPU offload knob (env-driven):
  - `LLM_N_CPU_MOE`
  - `LLM_CHAT_N_CPU_MOE`
  - `LLM_VISION_N_CPU_MOE`

## Operator Notes

- First run may download large model files (tier-dependent; see README table).
- CPU-only mode still exists but is slower and not the default path.
- ARCA is local-first and resource-sensitive; reproducible setup reports are prioritized.

## Documentation Added/Updated

- `README.md`
- `docs/dev/config.md`
- `docs/dev/troubleshooting.md`
- `docs/dev/domain_packs.md`
- `docs/dev/adding_tools.md`
- `docs/WHITEPAPER.md`
- `docs/BENCHMARK_V2_PUBLIC.md`
- `docs/RELEASE_CHECKLIST.md`
- `docs/ROADMAP.md`

## Known Limitations

- Large model startup and warmup can still be slow on first boot.
- Optimal retrieval configuration is corpus-dependent; benchmark first.
- Some advanced serving/performance tracks remain roadmap items.
