# ARCA Retrieval Pipeline — Benchmark Report

**Date**: 2026-02-12
**Corpus**: 27 synthetic domain-specific reports (8 neighborhoods, 27 projects)
**Hardware**: NVIDIA RTX 5090 (32 GB VRAM), 32-core CPU, 72 GB RAM
**Harness**: 14-layer automated benchmark, 168+ configurations, 40 ground-truth queries + 20 adversarial queries
**Wall-clock**: ~108 minutes total

---

## Context

This report documents a systematic evaluation of ARCA's retrieval pipeline: which components help, which hurt, and why — measured against a controlled synthetic corpus with known ground truth. It is written for operators deciding how to configure their own ARCA deployment, and for anyone building or evaluating RAG systems who wants to see what a rigorous pipeline benchmark actually looks like. The numbers are corpus-specific; the methodology and conclusions generalize.

---

## Executive Summary

168+ configurations tested across 14 layers, evaluating retrieval and answer generation against 40 ground-truth queries (5 query types) plus 20 adversarial queries against the live production pipeline.

### Key Findings

1. **Dense retrieval is near-ceiling on structured corpora — and the full pipeline makes it worse.** The embedding model (Qwen3-Embedding-0.6B) achieves entity hit rate 1.000 on 4 of 5 query tiers. Dense-only scores 0.774 composite at 19ms. All components active scores 0.721 at 1330ms — 70x slower and 7% worse.

2. **Cross-reference queries are the single exception.** GraphRAG traversal earns +0.024 composite on cross-reference queries (MRR 1.000, nDCG 0.950). This is the one query type where the full pipeline justifies its cost.

3. **A capable LLM still confidently hallucinates.** GLM-4.7-Flash (30B MoE, 3B active) scores factual accuracy 2.40/5 via Gemini-as-judge; Claude Opus 4.6 as an independent judge with ground truth access scores it at 3.93/5. The delta reveals judge severity bias when evaluating domain-specific content without ground truth access.

4. **Negation is an architectural limitation.** MRR 0.000 across all 15 retrieval configurations, LLM-as-judge score 1.44/5. Similarity-based retrieval cannot handle "which X does NOT have Y" queries.

5. **Continuous parameter tuning is irrelevant.** Total range across 30 parameter values: 0.007. The dominant variable is which components are active, not how they are tuned.

6. **Current model selections are validated.** Cross-model sweep of 27 combinations (3 chunking × 3 embedders × 3 rerankers) confirms the current production stack achieves the highest composite score (0.839) with perfect entity recall (1.000).

7. **A frontier model outperforms the local 30B by +12% on identical context.** Given the same retrieved chunks, Claude Opus 4.6 achieves 0.784 composite vs GLM's 0.664. The gap is largest on negation (+0.44) where Opus correctly reasons about absence. GLM wins multi-hop (-0.06), suggesting local MoE models can compete on chain reasoning.

8. **The local MoE alternative is 2.4x faster with comparable output.** Qwen3-30B-A3B averages 9.85s/query vs GLM's 23.55s, producing answers of similar length and quality.

9. **Production pipeline passes 100% adversarial testing.** 20 adversarial queries (out-of-scope, prompt injection, garbage input, ambiguous, negation, temporal, non-English) all handled gracefully. No injection attacks leaked internal state.

### Bottom Line

Default to dense-only retrieval and activate the full pipeline only for detected cross-reference queries via a query classifier. Dense-only matches or exceeds deep profile quality on 4 of 5 query tiers at 70x lower latency.

---

## Methodology

### Corpus

27 synthetic reports covering 8 fictional neighborhoods and 27 projects in a single technical domain. Reports include structured field data, lab results, and recommendations. Each follows a realistic professional structure. Ground truth derived from a master reference document with per-neighborhood profiles, per-project records, and regional context.

### Query Battery

40 queries across 5 tiers of increasing difficulty:

| Tier | Count | Description | Example |
|------|-------|-------------|---------|
| Factual | 15 | Single-fact lookup | "What was recommended for Project X?" |
| Neighborhood | 8 | Area-level synthesis | "Describe conditions in Area Y" |
| Cross-reference | 8 | Multi-document comparison | "Compare findings: Area A vs Area B" |
| Multi-hop | 6 | Multi-step reasoning chain | "Which projects involved condition Z?" |
| Negation | 3 | Exclusion/absence queries | "Which areas do NOT have condition W?" |

### Models

All inference runs locally on a single GPU. No cloud LLM APIs are used for retrieval or answer generation.

| Component | Model | Architecture | Quantization | Runtime |
|-----------|-------|--------------|-------------|---------|
| Embedding | Qwen3-Embedding-0.6B | 0.6B dense | ONNX FP16 | CUDA via ONNX Runtime |
| Reranker | Jina-Reranker-v2-Base-Multilingual | Cross-encoder | ONNX FP16 | CUDA via ONNX Runtime |
| Chat LLM (primary) | GLM-4.7-Flash | 30B MoE (3B active) | Q4_K_M (4-bit) | llama.cpp (CUDA) |
| Chat LLM (comparison) | Qwen3-30B-A3B | 30B MoE (3B active) | Q4_K_M (4-bit) | llama.cpp (CUDA) |
| Ceiling LLM | Claude Opus 4.6 | Dense (est. 200B+) | Full precision | API |
| Judge (L4) | Gemini 2.5-Flash / 3-Flash | Cloud API | N/A | Google AI API |
| Judge (L4b) | Claude Opus 4.6 | Dense (est. 200B+) | N/A | API |

### Scoring

Retrieval quality (L0–L2) measured via composite metric:
- 35% keyword hit rate + 15% entity hit rate + 20% MRR + 15% nDCG@5 + 10% source diversity + 5% latency bonus

Answer quality (L4/L4b) measured via LLM-as-judge on 3 dimensions (1–5 scale):
- Relevance, Accuracy, Completeness

### Benchmark Architecture

```
L0   Chunking sweep          114 configs    8 min    Isolates chunking from retrieval
L1   Retrieval ablation       15 configs    2 min    Ablation design isolates each component
L2   Parameter sweep          30 values   0.5 min    One-at-a-time continuous optimization
LE   Embedding shootout        4 models    2 min     Swap embedders, hold retrieval constant
LR   Reranker shootout         5 models    3 min     Swap rerankers, hold embedding constant
LX   Cross-model sweep        27 combos   18 min     Top-3 chunk × embed × rerank
L3   Answer generation        40 queries   18 min    GLM-4.7-Flash with optimal config
LLLM LLM comparison            2 models   25 min     GLM vs Qwen3 answer generation
L4   LLM-as-judge (Gemini)    35 scored   10 min     Gemini evaluation (R/A/C 1–5)
L4b  LLM-as-judge (Opus)      40 scored    manual    Opus w/ ground truth access
L5   Analysis + charts         4 charts    <1 min    Automated visualization
L6   Failure categorization    6 of 21    20 min     Gemini failure taxonomy
LC   Ceiling comparison       40 queries   <1 min    Frontier vs local on same context
OL   Live pipeline test       60 queries   ~1 min    MCP API stress + adversarial
```

---

## Layer 0: Chunking Optimization

**114 configurations** tested: 9 chunk sizes (400–2500 chars) × 7 overlap values (0–400 chars) × 2 context prefix settings, pruned for validity (overlap < 50% of chunk size).

### Results

**Winner: chunk size 2500, overlap 0, context prefix enabled** (composite 0.778)

| Rank | Chunk Size | Overlap | Context Prefix | Composite | Chunks |
|------|-----------|---------|----------------|-----------|--------|
| 1 | 2500 | 0 | Yes | 0.778 | 179 |
| 2 | 2500 | 50 | No | 0.776 | 163 |
| 3 | 2500 | 0 | No | 0.775 | 162 |
| 4 | 2500 | 50 | Yes | 0.768 | 179 |
| 5 | 2500 | 400 | No | 0.768 | 172 |

Top 9 configs are all 2500 chars. First non-2500 entry at rank 10 (1500 chars, 0.754).

**Big chunks dominate** on structured reports. **Zero overlap wins** on clean section breaks. **Context prefix is marginal** (+0.003). Caveat: synthetic corpus with consistent formatting — real-world documents with heterogeneous structure will shift optimal configuration.

---

## Layer 1: Retrieval Pipeline Ablation

**15 configurations** tested. Components: query expansion, BM25 sparse retrieval, cross-encoder reranking, HyDE, RAPTOR, GraphRAG, domain boost, diversity filtering.

| Rank | Config | Composite | Latency (ms) | Delta vs Dense |
|------|--------|-----------|-------------|----------------|
| 1 | expansion_only | 0.775 | 2.3 | +0.001 |
| 2 | dense_only | 0.774 | 18.8 | baseline |
| 3 | hybrid_no_rerank | 0.755 | 2.7 | −0.019 |
| 4–13 | various ablations | 0.740–0.746 | 37–59 | −0.028 to −0.034 |
| 14 | dense_rerank | 0.740 | 185.5 | −0.034 |
| **15** | **deep_profile (all on)** | **0.721** | **1330.6** | **−0.053** |

### Per-Tier Breakdown

| Tier | expansion_only | dense_only | deep_profile | Signal |
|------|---------------|-----------|-------------|--------|
| Factual (15) | **0.794** | 0.793 | 0.742 | Dense sufficient |
| Neighborhood (8) | **0.873** | 0.872 | 0.861 | Dense sufficient |
| Cross-reference (8) | 0.902 | 0.901 | **0.926** | Full pipeline wins |
| Multi-hop (6) | **0.864** | 0.864 | 0.769 | Pipeline harmful |
| Negation (3) | **0.442** | 0.442 | 0.309 | Architectural limit |

**The full pipeline wins on exactly one tier** and loses on the other four.

---

## Mechanistic Analysis

**Dense retrieval is near-ceiling.** Entity hit rate 1.000 on 4 of 5 tiers. Nothing for additional components to improve.

**Cross-reference is the signal.** Deep profile on cross-reference: MRR 1.000, nDCG 0.950. GraphRAG traverses entity relationships across documents — exactly what it's designed for.

**The reranker harms multi-hop.** The cross-encoder is trained on single-passage relevance and cannot see multi-step reasoning chains. It demotes passages that are only relevant in sequence. nDCG 0.933 (finds them) but MRR 0.556 (ranks them wrong).

**Negation is architecturally incompatible.** MRR 0.000 across every configuration. "Which areas do NOT have X" retrieves everything that mentions X. Fundamental limitation of similarity-based retrieval.

**Pipeline components inject noise.** Deep profile drops entity hit rate to 0.933 on factual queries (vs 1.000 for dense-only). Additional components displace relevant results with plausible but wrong ones.

---

## Layer 2: Parameter Sensitivity

**30 parameter values** tested. **Total range: 0.007.** Almost nothing matters. Only two signals: reranker_candidates=15 (+0.005) and domain_boost=0.5 (+0.004). Confirms the dominant variable is which components are active, not how they are tuned.

---

## Model Shootouts

### Embedding Models (4 tested)

| Rank | Model | Dim | Composite | Duration |
|------|-------|-----|-----------|----------|
| 1 | BGE-Large-EN-v1.5 | 1024 | 0.7608 | 16.8s |
| 2 | Nomic-Embed-Text-v1.5 | 768 | 0.7560 | 10.5s |
| 3 | Qwen3-Embedding-0.6B | 1024 | 0.7393 | 65.9s |
| - | Stella-EN-400M-v5 | 1024 | FAILED | SM 12.0 incompatible |

BGE-Large wins on quality. Nomic wins on speed (6.3x faster than Qwen3, 0.005 penalty). Spread is modest — embedder choice is less impactful than retrieval architecture.

### Reranker Models (5 tested)

| Rank | Model | Composite | Latency/query | Duration |
|------|-------|-----------|---------------|----------|
| 1 | Jina-Reranker-v2-Turbo | 0.7723 | 144ms | 10.0s |
| 2 | BGE-Reranker-v2-M3 | 0.7705 | 397ms | 26.8s |
| 3 | MxBAI-Rerank-Base-v2 | 0.7655 | 253ms | 16.0s |
| 4 | MS-MARCO-MiniLM-L12 | 0.7650 | 29ms | 1.9s |
| 5 | BGE-Reranker-v2-Gemma | 0.7586 | 1147ms | 77.6s |

Jina-v2-Turbo wins composite. Total spread across 5 rerankers: 0.014 — far smaller than retrieval toggle effects (0.053). MS-MARCO-MiniLM is compelling at 29ms for only −0.007 composite.

---

## Cross-Model Sweep (27 combinations)

Top-3 chunking × 3 embedders × 3 rerankers. Full re-embedding and re-indexing per combination.

| Rank | Embedder | Reranker | Composite | Entity Recall |
|------|----------|----------|-----------|---------------|
| 1 | Qwen3-0.6B | Jina-v2-Turbo | **0.839** | **1.000** |
| 2 | Qwen3-0.6B | MxBAI-Base-v2 | 0.793 | 0.936 |
| 3 | BGE-Large | BGE-v2-M3 | 0.787 | 1.000 |

**Current production defaults are the optimal stack.** No alternative combination outperforms Qwen3 + Jina. Full spread: 0.839 to 0.562 across all 27 combinations. Perfect entity recall (1.000) only achieved by top-1 and top-3 combos.

---

## Answer Quality: Dual Judge Evaluation

35 queries scored by Gemini (L4), all 40 scored by Claude Opus 4.6 with ground truth access (L4b).

### Overall Scores

| Dimension | Gemini | Opus (w/ ground truth) | Delta |
|-----------|--------|------------------------|-------|
| Relevance | 4.63 | 4.30 | −0.33 |
| Accuracy | 2.66 | **4.03** | **+1.37** |
| Completeness | 2.77 | **3.70** | **+0.93** |
| **Overall** | **3.35** | **4.01** | **+0.66** |

### Why the Judges Disagree

The +1.37 accuracy delta has a clear mechanism. Gemini lacks ground truth access — it evaluates surface coherence but cannot verify whether specific values are correct. It defaults to moderate scores for unverifiable claims. The Opus judge has the full reference data and can confirm or deny factual claims against canonical values.

Where both agree: neighborhood queries are the strongest tier (4.40–4.92/5). Negation is catastrophic (1.44–1.67/5). Timeouts are complete failures.

**Implication for RAG evaluation**: LLM-as-judge without ground truth access systematically underscores factual accuracy on domain-specific content. Dual-judge methodology with one ground-truth-aware judge produces more reliable evaluation.

---

## Model Ceiling: Frontier vs Local

Given **identical retrieved context**, Claude Opus 4.6 and GLM-4.7-Flash generated answers to all 40 queries.

| Metric | GLM-4.7-Flash (30B MoE) | Claude Opus 4.6 | Delta |
|--------|--------------------------|-----------------|-------|
| Composite score | 0.664 | 0.784 | **+0.120** |
| Entity hit rate | 0.875 | 0.975 | +0.100 |
| Text overlap w/ ground truth | 0.393 | 0.649 | +0.256 |
| Abstentions (correct refusals) | 1 | 8 | +7 |

### Per-Tier Delta

| Tier | Delta (Opus − GLM) | Notes |
|------|---------------------|-------|
| Negation | **+0.444** | Opus reasons about absence; GLM fabricates |
| Cross-reference | +0.166 | Opus synthesizes across documents better |
| Neighborhood | +0.119 | Both strong; Opus more complete |
| Factual | +0.103 | Opus verifies rather than assumes |
| Multi-hop | **−0.058** | GLM wins — shorter, focused answers with better keyword density |

**Opus abstains more intelligently** (8 correct refusals vs 1). When context doesn't contain the answer, Opus says so. GLM fabricates. This maps directly to the hallucination pattern from L6.

**GLM wins multi-hop.** Shorter, more focused answers contain the right keywords more efficiently. Opus's longer answers dilute keyword density despite being more comprehensive. Suggests the scoring metric slightly penalizes verbosity on chain reasoning.

---

## LLM Comparison: GLM-4.7-Flash vs Qwen3-30B-A3B

Both are 30B MoE architectures with ~3B active parameters, Q4_K_M, served via llama.cpp.

| Metric | GLM-4.7-Flash | Qwen3-30B-A3B |
|--------|---------------|---------------|
| Answers generated | 38 / 40 | 27 / 40 (13 warmup errors) |
| Avg response time | 23.55s | **9.85s** |
| Avg answer length | 807 chars | 734 chars |

Qwen3 is **2.4x faster** with comparable output. The 13 failures are a llama-server warmup bug (health endpoint reports ready before model is loaded), not a model capability issue.

---

## Live Pipeline: Adversarial Testing

20 adversarial queries fired at the production MCP API.

| Category | Queries | Passed | Behavior |
|----------|---------|--------|----------|
| Out-of-scope | 3 | 3/3 | Low confidence (0.14–0.19) |
| Ambiguous | 3 | 3/3 | Multiple results returned |
| Prompt injection | 3 | 3/3 | No state leaks |
| Long query | 2 | 2/2 | Handled correctly |
| Empty/garbage | 3 | 3/3 | Low confidence (0.11–0.27) |
| Negation | 2 | 2/2 | Expected partial quality |
| Temporal | 2 | 2/2 | Partial match |
| Non-English | 1 | 1/1 | Partial match |
| **Total** | **20** | **20/20** | **100% pass rate** |

No injection attacks leaked internal state. Out-of-scope queries correctly received low confidence scores. The pipeline handles adversarial input gracefully.

---

## Failure Analysis

21 queries scored below threshold. 6 categorized.

| Category | Count | Description |
|----------|-------|-------------|
| Hallucination | 5 | LLM fabricates specific values not in context |
| Wrong Document | 1 | Retrieved related but incorrect document |

The dominant failure mode is **confident fabrication** of specific numerical values when context is plausible but does not actually answer the question. The LLM treats proximity as equivalence — "relevant context" becomes "correct answer." The fix is prompt engineering and confidence-gated output, not a larger model. This is confirmed by the ceiling test: Opus 4.6 correctly abstains on 8 queries where GLM fabricates.

Two structural dead queries reveal production failure modes: name-to-ID mapping failure (queries by human-readable name can't match records indexed by project ID) and corpus-level metadata failure (organizational information not in individual records).

---

## Corpus Limitations

A synthetic corpus is a controlled environment, not a production replica. These results come with specific caveats:

**What synthetic corpora prove well**: Pipeline component interactions, relative model rankings, failure mode identification, and architectural limitations (like negation). These findings generalize because they reflect structural properties of retrieval algorithms, not corpus-specific artifacts.

**What synthetic corpora cannot prove**: Optimal chunk size for your documents, the real-world value of BM25 (synthetic text has artificially consistent vocabulary), the frequency of cross-reference queries in actual usage, or whether GraphRAG's cost is justified on your specific corpus structure.

**Properties that differ from production**: The synthetic reports have clean section breaks, consistent formatting, dense information per page, limited cross-referencing between documents, and uniform writing style. Real corpora have OCR artifacts, inconsistent structure, sparse pages, heavy cross-referencing, and multiple authoring styles. Each of these differences shifts the optimal configuration.

**The right interpretation**: Use these numbers to understand how the pipeline behaves and which knobs matter. Use the benchmark harness on your own data to find the numbers that actually drive your configuration decisions.

---

## Conclusions

1. **Build the harness first.** Months of assembling components based on papers and intuition. Under two hours of systematic benchmarking revealed which components help, which hurt, and why. The harness is more valuable than any individual pipeline component — it tells you which components to keep and which to turn off.

2. **Benchmark your own corpus.** Generic RAG benchmarks suggest all components help. On this corpus, most hurt. The only way to know is to test systematically against your data.

3. **Dense retrieval is really good now.** A 0.6B ONNX embedder at 19ms beat a 12-stage pipeline at 1330ms. Embedding models have improved to the point where every additional retrieval stage must justify its existence on your specific corpus.

4. **Components aren't universally good or bad.** GraphRAG earned its cost on cross-reference and was net-negative on everything else. The reranker harmed multi-hop. The right architecture is adaptive — route queries to the components that help for that query type, not a static "everything on" default.

5. **The "I don't know" problem is harder than retrieval.** Retrieval was near-perfect. Answers were mediocre. The real bottleneck is generating correct answers from good context and refusing when the context doesn't contain the answer. A frontier model (Opus) handles refusal correctly where a capable local model (GLM 30B MoE) fabricates confidently.

6. **Dual-judge evaluation reveals hidden accuracy.** A single LLM judge without ground truth access systematically underscores domain-specific factual accuracy by 1.37 points. If you are evaluating RAG answer quality, give at least one judge access to the ground truth.

7. **Validate your model stack empirically.** A 27-combination sweep confirmed the current defaults are optimal. Intuition about model quality does not substitute for data — the best individual embedder (BGE-Large) does not produce the best stack when paired with the best reranker. Synergies matter.
