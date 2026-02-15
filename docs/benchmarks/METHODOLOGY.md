# ARCA Benchmark Methodology

## Purpose

ARCA benchmarking is designed to tune retrieval and generation behavior against the user's actual corpus, not generic leaderboard data.

The harness is meant to answer:
- Which retrieval stages help this corpus?
- Which model combinations are best together?
- Which settings should become runtime defaults?

## Published Public Benchmark (v2)

For a full worked example with real numbers, see:
- [Benchmark V2 Public Report](../BENCHMARK_V2_PUBLIC.md)

That run includes 14 layers, 168+ configurations, 40 ground-truth queries, and 20 adversarial live-pipeline queries.

Key v2 signals (corpus-specific but operationally useful):
- Dense-first retrieval was near-ceiling on that structured corpus.
- Full deep pipeline was slower and worse in aggregate on that corpus.
- Graph-heavy traversal helped cross-reference queries.
- Cross-model sweep validated default embedder/reranker pairing.

## Corpus Design Guidance

Use a corpus slice that matches production reality:
- document types
- formatting quality
- table/chart density
- vocabulary style

A benchmark built on toy data will produce toy defaults.

## Query Battery Design

Use a balanced query set across difficulty classes, such as:
- factual lookup
- numerical/structured
- conceptual synthesis
- cross-reference
- multi-hop
- negation/exclusion

Include adversarial queries to test robustness:
- prompt injection
- ambiguous requests
- out-of-scope requests
- malformed/garbage input

## Scoring Components

Typical ARCA composite retrieval score combines:
- keyword hit rate
- entity hit rate
- MRR
- nDCG@k
- source diversity
- latency bonus/penalty

Answer quality can be evaluated with LLM-as-judge (relevance, accuracy, completeness), ideally with ground-truth-aware judging for factual domains.

## Layered Evaluation Model

ARCA supports layered evaluation so each component can be isolated:

- L0: chunking sweep
- L1: retrieval stage ablation
- L2: continuous parameter sweep
- LE: embedding model shootout
- LR: reranker model shootout
- LX: embedder x reranker cross-matrix
- L3: answer generation with selected retrieval config
- LLLM: local model comparison
- L4/L4b: answer judging (single or dual-judge)
- L5: statistical analysis and charting
- L6: failure categorization
- LC: model ceiling comparison
- OL: live/adversarial production-style testing

## How to Interpret Results

1. Do not over-optimize one metric.
2. Check per-tier behavior, not just global averages.
3. Prefer stable improvements over fragile one-off wins.
4. Distinguish component value by query type.
5. Validate winners with manual smoke queries before production rollout.

## Practical Decision Rules

- If dense-first is already strong, keep deep stages selective.
- If cross-reference queries are important, enable graph-heavy paths conditionally.
- If reranker choice differences are tiny, prioritize latency and stability.
- If LLM-as-judge disagrees strongly, use dual-judge or ground-truth-aware evaluation.

## Reproducibility

Example benchmark commands:

```bash
# Full benchmark
docker exec -t arca-backend python /app/scripts/benchmark_full_pipeline.py --phases all

# Selected phases
docker exec -t arca-backend python /app/scripts/benchmark_full_pipeline.py --phases reranker embedding

# Skip ingestion and reuse current data
docker exec -t arca-backend python /app/scripts/benchmark_full_pipeline.py --skip-ingestion --phases all
```

## Output Artifacts

A benchmark run typically produces:
- structured results JSON
- markdown report
- phase-level charts
- summary dashboard assets

Store these artifacts with timestamped run IDs so parameter changes remain auditable.
