# backend/tools/cohesionn/benchmark/

Model shootout and parameter optimization harness for the RAG pipeline. Supports 7 benchmark phases comparing embedding models, reranker models, LLMs, chunking strategies, and pipeline parameter sweeps. Generates visualizations, reports, and composite scores.

| File | Purpose | Key Exports |
|------|---------|-------------|
| __init__.py | Package exports | `ShootoutConfig`, `run_shootout()` |
| config.py | Shootout configuration, model catalogs (4 embedders, 5 rerankers, 3 LLMs), sweep params | `ShootoutConfig`, `ModelSpec`, `EMBEDDING_MODELS`, `RERANKER_MODELS`, `LLM_MODELS` |
| metrics.py | Composite scoring (35% keyword + 15% entity + 20% MRR + 15% nDCG + 10% diversity + 5% latency) | `compute_composite_score()`, `evaluate_retrieval()` |
| queries.py | Test query sets for benchmarking | `get_test_queries()`, `TestQuery` |
| page_sampler.py | Sample pages from corpus for benchmark evaluation | `PageSampler` |
| report.py | Text-based benchmark report generation | `generate_report()` |
| report_generator.py | Structured report with per-phase summaries | `ReportGenerator` |
| publication_report.py | Publication-quality report formatting | `PublicationReport` |
| visualizations.py | Matplotlib/seaborn charts for benchmark results | `generate_visualizations()` |

## Subdirectory

| Directory | Purpose |
|-----------|---------|
| phases/ | Phase implementations (ingestion, reranker, embedding, cross_matrix, LLM, param_sweep, ablation) |

### phases/

| File | Purpose | Key Exports |
|------|---------|-------------|
| __init__.py | Phase registry | -- |
| base.py | Base phase class with shared utilities | `BasePhase` |
| ingestion.py | Chunk size/overlap/extractor sweep | `IngestionPhase` |
| reranker.py | Compare 5 reranker models | `RerankerPhase` |
| embedding.py | Compare 4 embedding models | `EmbeddingPhase` |
| cross_matrix.py | Top-N embedders x Top-N rerankers matrix | `CrossMatrixPhase` |
| llm.py | LLM answer quality evaluation | `LLMPhase` |
| param_sweep.py | Sweep 6 pipeline parameters | `ParamSweepPhase` |
| ablation.py | Toggle 8 features on/off | `AblationPhase` |
