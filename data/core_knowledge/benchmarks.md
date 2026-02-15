# ARCA Benchmark Harness

ARCA includes a benchmark system so retrieval and model settings can be tuned against your own corpus instead of generic defaults.

## Benchmark Philosophy

The benchmark is intended to answer:
- Which retrieval stages help this corpus?
- Which chunking strategy performs best here?
- Which embedding/reranker pairing wins on this data?
- Which runtime profile should be used in production?

The point is operational optimization, not leaderboard chasing.

## Layered Benchmark Design

ARCA benchmarking is organized as layers/phases. Depending on run mode and configuration, you can execute targeted sweeps or full chains.

Common layers include:
- Layer 0: chunking sweep.
- Layer 1: retrieval toggle/config sweep.
- Layer 2: parameter sensitivity sweep.
- Layer 3: answer generation.
- Layer 4: answer judging.
- Layer 5: analysis/reporting.
- Layer 6: failure analysis.
- Embedding layer: embedding model comparison.
- Reranker layer: reranker model comparison.
- Cross-matrix layer: embedder/reranker pair matrix.
- LLM layer: generation model comparison.
- Ceiling layer: external reference ceiling comparison (optional).
- Live layer: production-style/adversarial testing.

Exact availability can depend on installed providers and run flags.

## Query and Ground-Truth Inputs

Benchmarks can use:
- Domain battery query sets.
- Auto-generated query sets from corpus sampling.
- Ground-truth files where available.
- Generic fallback queries for broad sanity testing.

Quality of benchmark output depends heavily on query representativeness.

## Judging Model Options

Judging can be local or provider-backed depending on configured credentials and endpoints. Judge selection affects cost, latency, and score consistency.

Practical guidance:
- Keep one stable judge for comparable historical runs.
- Change judges only when you intentionally re-baseline.

## Typical Benchmark Workflow

1. Ingest representative documents first.
2. Prepare benchmark query/ground-truth inputs.
3. Run quick mode for initial tuning.
4. Run deeper sweeps when quick mode leaves clear failure modes.
5. Review output reports and failure clusters.
6. Apply winning toggles/profile.
7. Re-test with adversarial/live queries.

## Apply-Winners Flow

Benchmark winners can be applied into runtime profile state (commonly `auto`) so production behavior reflects measured outcomes.

Always verify with a manual spot-check before locking into long-term defaults.

## What Good Results Look Like

A good benchmark outcome is not just high average score. It should also show:
- Reasonable latency for your workload.
- Stable behavior across query types.
- Fewer catastrophic failures.
- Better source diversity and citation quality.

## Common Benchmark Mistakes

- Testing only one document type.
- Using toy queries unlike real user behavior.
- Optimizing for one metric while latency explodes.
- Enabling every retrieval stage and assuming higher complexity equals quality.
- Ignoring failure categories and only reading mean scores.

## Recommended Cadence

Run benchmarks:
- After major corpus changes.
- After major model changes.
- After major retrieval/pipeline logic changes.
- Before public releases.

## Practical Interpretation Rules

- If a stage helps only a narrow query class, gate it by profile or classifier.
- If a stage adds latency with no robust score gain, disable by default.
- If quality gains are small but failure reductions are large, that may still be worth keeping.

## Related Docs

- `retrieval_pipeline.md`: stage internals.
- `configuration.md`: profile and toggle controls.
- `troubleshooting.md`: debugging slow or unstable runs.
