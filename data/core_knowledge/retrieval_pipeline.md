# ARCA Retrieval Pipeline

This document describes the retrieval system ARCA uses for knowledge search. The implementation is centered in `backend/tools/cohesionn/retriever.py` and related modules.

## Why ARCA Uses a Multi-Stage Pipeline

Different retrieval strategies fail differently:
- Dense retrieval is strong at semantic similarity but can miss exact identifiers.
- Sparse retrieval is strong at lexical precision but can miss paraphrases.
- Graph and hierarchy retrieval can recover context spread across documents.

ARCA combines these methods with filters and reranking so you can optimize for your own corpus.

## Retrieval Entry Point

Primary flow:
- `search_knowledge(...)` calls `CohesionnRetriever.retrieve(...)`.
- Retrieval toggles resolve through profile hierarchy (`fast`, `auto`, `deep`, plus manual overrides).
- Topic routing, candidate retrieval, fusion, reranking, diversity, and quality gates run in sequence.

## Stage-by-Stage Flow

### Stage 0a: Query Expansion
Purpose:
- Expand query terms with lexicon synonyms.

Behavior:
- Enabled by `query_expansion_enabled`.
- Skipped for very short queries.
- Uses domain lexicon term maps when available.

Tradeoff:
- Improves recall on domain aliases; can add latency and occasional noise.

### Stage 0b: HyDE
Purpose:
- Generate a hypothetical answer/document for semantic retrieval probing.

Behavior:
- Enabled by `hyde_enabled`.
- Usually skipped for short queries.

Tradeoff:
- Can improve semantic matching on complex questions; adds model-call overhead.

### Stage 1: Topic Routing + Dense Retrieval
Purpose:
- Pick likely topics and retrieve semantic candidates from Qdrant.

Behavior:
- Topic router predicts topic scopes.
- Dense retrieval queries vector store per topics + candidate budget.

Tradeoff:
- Strong baseline recall with reasonable speed.

### Stage 2: Sparse Retrieval (BM25)
Purpose:
- Add lexical candidates for exact tokens and identifiers.

Behavior:
- Enabled by `bm25_enabled`.
- Managed by BM25 manager/index path.

Tradeoff:
- Helps exact matches; can degrade results on some corpora if over-weighted.

### Stage 2.5: RAPTOR Hierarchical Retrieval
Purpose:
- Retrieve from hierarchical summaries for broad conceptual queries.

Behavior:
- Enabled by `raptor_enabled`.
- Triggered conditionally via RAPTOR mixin logic.

Tradeoff:
- Better global/contextual coverage, higher latency.

### Stage 2.6: Graph Retrieval
Purpose:
- Retrieve chunks through entity/relationship graph traversal.

Behavior:
- Enabled by `graph_rag_enabled`.
- Can auto-enable for cross-reference query classes when `graph_rag_auto` is true.

Tradeoff:
- Useful for relationship/comparison queries; depends on graph quality.

### Stage 2.7: Global Community Retrieval
Purpose:
- Retrieve higher-level thematic context from community summaries.

Behavior:
- Enabled by `global_search_enabled`.
- Triggered when classifier sees broad theme-style queries.

Tradeoff:
- Good for broad queries, less useful for narrow factual questions.

### Stage 3: Reciprocal Rank Fusion
Purpose:
- Fuse dense, sparse, and optional graph candidate sets.

Behavior:
- Uses weighted reciprocal-rank style fusion.
- Weights depend on which components are active.

Tradeoff:
- Better robustness than any single retriever; still needs good input candidates.

### Stage 3.5: RAPTOR Merge
Purpose:
- Merge hierarchical candidates into fused set.

Behavior:
- Applies an explicit RAPTOR merge weight.

Tradeoff:
- Can improve broad-question coverage; may add summary noise if overused.

### Stage 4: Pre-Filter by Raw Score
Purpose:
- Drop obvious low-quality candidates before expensive reranking.

Behavior:
- Controlled by `rag_min_score`.
- Keeps top fallback candidate if everything is below threshold.

### Stage 5: Cross-Encoder Reranking
Purpose:
- Re-score candidates with query+document joint model.

Behavior:
- Controlled by `reranker_enabled` and `reranker_candidates`.

Tradeoff:
- Large precision lift in many corpora; adds latency/compute.

### Stage 5.5: Domain Boost
Purpose:
- Counteract generic-content bias and favor domain-relevant matches.

Behavior:
- Controlled by `domain_boost_enabled` and `domain_boost_factor`.

### Stage 6: Diversity Reranking (MMR)
Purpose:
- Prevent top-k collapse into near-duplicates from one source.

Behavior:
- Controlled by `rag_diversity_enabled` and related settings.

### Stage 7: Final Quality Gate
Purpose:
- Remove low-confidence results before generation context assembly.

Behavior:
- Controlled by `rag_min_final_score`.
- If all are filtered, keeps limited fallback results.

### Stage 8: Confidence Labeling + Citations
Purpose:
- Attach confidence and structured citations for UI rendering.

Behavior:
- Labels final result confidence from score bands.
- Citation metadata includes source/topic/page fields when available.

## Profiles and Toggle Resolution

Profile manager resolves toggles in this order (highest priority first):
1. Per-query override.
2. Manual runtime override.
3. Active profile (`fast`, `auto`, `deep`).
4. Runtime config defaults.

Default profile intent:
- `fast`: lower-latency path for interactive use.
- `deep`: broader coverage at higher latency.
- `auto`: populated by benchmark winner settings.

## Important Operational Notes

### More stages is not automatically better
A longer stage chain can reduce quality if it introduces noisy candidates or over-complex fusion.

### Benchmark first
Use representative corpus queries and compare stage toggles empirically. Do not assume defaults are globally optimal.

### Tune thresholds carefully
- Too low: noisy context, hallucination pressure.
- Too high: empty context and brittle failures.

### Use topic routing plus explicit topic controls
For high-value workflows, topic constraints can reduce noise and speed up retrieval.

## Common Retrieval Failure Patterns

- Overly small chunks with weak local context.
- BM25 over-weighted for semantic queries.
- Graph retrieval active without high-quality graph extraction.
- Too many weak candidates passed into reranker.
- Inconsistent topic enablement.

## Practical Tuning Sequence

1. Verify ingestion quality and chunking first.
2. Start from `fast` profile.
3. Run benchmark layers for retrieval and parameter sweeps.
4. Apply winner settings to `auto` profile.
5. Hand-tune only specific failure classes.

## Related Docs

- `benchmarks.md`: how to measure and apply winners.
- `configuration.md`: retrieval and reranker settings.
- `troubleshooting.md`: failure mode diagnosis.
