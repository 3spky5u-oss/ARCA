# backend/benchmark/ — Benchmark Harness v2

Comprehensive chunking + retrieval pipeline optimization.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Module docstring + version |
| `__main__.py` | Entry point (`python -m benchmark`) |
| `config.py` | `BenchmarkConfig`, `ChunkingConfig`, `RETRIEVAL_CONFIGS`, `PARAM_SWEEP_RANGES` |
| `corpus.py` | `DocxConverter` — .docx to markdown via python-docx |
| `checkpoint.py` | `CheckpointManager` — per-config JSON crash recovery |
| `collection_manager.py` | `BenchmarkCollectionManager` — topic-per-config Qdrant isolation |
| `judge.py` | `LLMJudge` — provider-agnostic LLM-as-judge evaluation |
| `cli.py` | argparse CLI: `layer0`..`layer6`, `full`, `status` |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `providers/` | LLM provider abstraction: base ABC, local, gemini, anthropic, openai |
| `queries/` | Query battery loading, auto-generation, ground truth parsing, BenchmarkQuery dataclass |
| `layers/` | Layer implementations (L0-L6 + embed, rerank, cross, llm, ceiling, live) + BaseLayer ABC |

### queries/ Files

| File | Purpose |
|------|---------|
| `battery.py` | `BenchmarkQuery` dataclass |
| `loader.py` | `QueryBatteryLoader` — 3-tier resolution: domain battery → auto-gen → generic fallback |
| `auto_generator.py` | `QueryAutoGenerator` — LLM-based query generation from corpus text |
| `ground_truth.py` | `GroundTruthParser` — structured bible file parser (legacy) |
| `synthesizer.py` | `QuerySynthesizer` — hardcoded domain queries (legacy, replaced by loader) |

## Layer Architecture

```
L0 → Chunking sweep (~90 configs, dense-only retrieval)
L1 → Retrieval toggle sweep (~15 named configs)
L2 → Continuous parameter sweep (OAT, ~30 evaluations)
L3 → Answer generation via local LLM
L4 → LLM-as-judge (configurable provider: local, Gemini, Anthropic, OpenAI)
L5 → Statistical analysis + matplotlib visualizations
L6 → Failure categorization (configurable provider)
LC → Frontier LLM ceiling comparison (configurable provider)
OL → Live pipeline test via MCP API
```

## Output

`data/benchmarks/v2/{run_id}/` with per-layer subdirectories containing JSON results, charts, and markdown reports.
