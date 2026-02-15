# ARCA Domain Packs

Domain packs are ARCA's extension mechanism for specialization without hard-forking core logic.

A domain pack can control:
- Vocabulary and synonyms used in retrieval behavior.
- Topic definitions and retrieval routing hints.
- Personality and messaging defaults.
- Optional domain-specific tools, executors, and routes.
- UI branding metadata.

## Why Domain Packs Exist

ARCA core is public and generic. Real workloads differ by vertical and team conventions. Domain packs let you specialize behavior while keeping a stable core codebase.

## Pack Discovery and Activation

Domain packs live under `domains/<pack_name>/` and are selected by:
- `ARCA_DOMAIN` in environment configuration.
- Admin domain switching flows where supported.

The active pack is loaded through `backend/domain_loader.py`.

## Typical Domain Pack Structure

Required/typical files:
- `manifest.json` (required)
- `lexicon.json` (strongly recommended)
- `register_tools.py` (optional if adding tools)
- `tools/` (optional)
- `executors/` (optional)
- `routes/` (optional)

## `manifest.json` Responsibilities

Manifest typically defines:
- Name, version, description.
- Declared tools and optional route modules.
- Branding metadata (app name/tagline/color or related fields).

Manifest keeps activation and package metadata explicit.

## `lexicon.json` Responsibilities

Lexicon is the most important behavior file. It can define:
- Identity/personality blocks used in prompts.
- Topic list and per-topic routing hints.
- Retrieval pipeline vocabulary:
  - Synonyms
  - Preserve terms
  - Domain keywords and patterns
- Technical/detection patterns used by orchestration.
- Optional Phii seed terms/corrections.

In practice, lexicon changes are the fastest way to specialize ARCA behavior without touching Python code.

## Domain Tools

When a pack provides `register_tools.py`, it can register additional tools through the common registry. Core tools remain available.

Load sequence:
1. Core tools register first.
2. Domain tools register second.
3. Optional custom/admin-generated tools can register afterward.

## Domain Routes

Domain route modules can add admin or API surfaces when declared and available. Route imports are guarded so missing optional modules do not crash core startup.

## Safety and Isolation Model

Domain components are optional by design:
- Missing pack modules should degrade gracefully.
- Core platform behavior should continue without domain modules.
- Import guards isolate optional dependencies.

## Domain Pack Development Workflow

1. Copy a baseline pack (for example from `domains/example`).
2. Update `manifest.json` metadata.
3. Build or revise `lexicon.json` for terminology and retrieval behavior.
4. Add tools/executors if needed.
5. Validate startup and tool registration logs.
6. Run retrieval benchmarks against representative corpus data.

## Domain Design Guidance

- Keep lexicon terminology focused and maintainable.
- Prefer configuration changes before adding custom code.
- Add tools only when existing tools cannot express required workflows.
- Benchmark before and after major lexicon/pipeline changes.

## Public vs Private Pack Strategy

Common pattern:
- Public repo: generic `example` or broadly useful packs.
- Private repo: proprietary domain packs and custom workflows.

This keeps OSS core clean while preserving private specialization.

## Related Docs

- `configuration.md`: activation and runtime configuration context.
- `tools.md`: core registry and tool model.
- `retrieval_pipeline.md`: where lexicon settings influence retrieval behavior.
