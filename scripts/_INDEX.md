# scripts/

Root-level utility scripts for development, deployment, and maintenance. Includes audit tools, smoke testing, and hardware simulation.

| File | Purpose | Key Exports |
|------|---------|-------------|
| arca.py | Unified local operator CLI (`bootstrap`, `update`, `up`, `down`, `doctor`, `public-init`, `export-public`, `publish-public`) for GPU-first local operations | CLI script |
| bootstrap.sh | Unix wrapper for one-command bootstrap | -- |
| bootstrap.ps1 | PowerShell wrapper for one-command bootstrap | -- |
| update.sh | Unix wrapper for one-command updates | -- |
| update.ps1 | PowerShell wrapper for one-command updates | -- |
| init_env.py | Creates `.env` from template and generates secrets | -- |
| model_bootstrap.py | Ensures configured slot GGUF files exist (`chat/code/expert/vision/vision_structured` + vision mmproj), optional auto-download via Hugging Face | -- |
| export_public.py | Creates allowlisted public export tree for GitHub release | -- |
| public_repo_init.py | Initializes persistent public export git repo | -- |
| publish_public.py | One-command export + commit + push to public repo | -- |
| publish_public.sh | Unix wrapper for publish-public flow | -- |
| publish_public.ps1 | PowerShell wrapper for publish-public flow | -- |
| start.sh | Docker compose startup + health wait loop | -- |
| context.py | Load focused doc contexts for Claude sessions | -- |
| deps.py | Show module dependency graph (manually maintained) | -- |
| file_refs.py | Track file imports/references for impact analysis | -- |
| find_pattern.py | Find existing codebase patterns by name | -- |
| preflight.py | Host preflight doctor (Docker/GPU/container checks, RAM/disk/ports, .env/models/compose) with optional JSON output | -- |
| smoke_test.py | Hit all major endpoints for post-deploy validation | -- |
| test_hardware_sim.py | Hardware profile simulation runner across 5 tiers | -- |
| test_module.py | Run pytest for specific modules (wrapper) | -- |
| test_phii_learning.py | Validate Phii system end-to-end | -- |
| pre-commit-check.py | Syntax, import, registry, secret checks | -- |
| post_ingest_test_suite.py | 11-module validation suite (1420 lines) | -- |
| new_tool.py | Scaffold new tool module from template | -- |
| todo_scan.py | Find TODO/FIXME comments in codebase | -- |

## Subdirectory

| Directory | Purpose |
|-----------|---------|
| audit/ | Codebase analysis: inventory, dependency graph, dead code, security scan, doc generation |

### audit/

| File | Purpose | Key Exports |
|------|---------|-------------|
| __init__.py | Package marker | -- |
| run_full_audit.py | Orchestrates all audit scripts (~500 lines) | `run_audit()` |
| inventory.py | Codebase metrics, file analysis (~750 lines) | `InventoryAudit` |
| dependency_graph.py | Import graph, circular dependency detection (~630 lines) | `DependencyGraphAudit` |
| dead_code.py | Unused imports/functions/classes detection (~700 lines) | `DeadCodeAudit` |
| security_scan.py | Security pattern detection (~700 lines) | `SecurityScanAudit` |
| generate_docs.py | Auto-generate module documentation (~740 lines) | `DocsGenerator` |
