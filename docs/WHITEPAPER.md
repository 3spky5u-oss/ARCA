# ARCA Technical Whitepaper

Version: 0.1.0
Date: 2026-02-14
Status: Public release technical document

---

## 0. Preface

This is a technical document for builders — people evaluating local RAG platforms, building their own, or contributing to this one.

It covers:

- how ARCA works end to end
- what tradeoffs drive the design
- where the design is still rough
- what the benchmarks actually show

If you are building your own local RAG stack, this is meant to save you time. If you are evaluating ARCA, this is meant to give you enough depth to decide whether it fits your problem.

---

## 1. Problem Landscape

### 1.1 The Gap Between RAG Research and Production RAG

Most RAG literature evaluates on curated datasets where vocabulary overlap is relatively friendly and documents are structurally consistent. Real technical corpora are rarely like that.

Consider a practical example: an engineer asks "what were the soil conditions at the downtown site?" The answer lives in a report that discusses "subsurface stratigraphy" in "the central business district investigation area." Dense embeddings might bridge this gap — or might not, depending on the model and the corpus. Compound that with tables of SPT blow counts, cross-references to national standards, and a figure showing the soil profile, and you have the kind of document that breaks naive retrieval.

In production you typically get:

- Vocabulary mismatch between user phrasing and document terminology.
- Mixed structure (narrative text, tables, figures, equations, appendices).
- Cross-document dependency chains that look more like graphs than isolated passages.
- Strong corpus-specific behavior where "best" retrieval settings vary by dataset.

That gap is where many otherwise-strong LLM deployments fail. ARCA was built to close it with operational tooling, not just prompt tricks.

### 1.2 Why Local-First Matters

Many teams cannot move source documents to cloud services, regardless of convenience. Engineering firms, legal practices, healthcare providers, defense contractors — their documents cannot leave their network, and no contractual assurance changes the architectural reality of cloud-hosted RAG.

ARCA is designed for those constraints:

- Data stays local by default.
- Model runtime is local by default.
- Retrieval/indexing infrastructure is self-hosted.

Local-first is not automatically easier. It shifts responsibility to deployment, hardware, and observability. ARCA explicitly exposes those operational realities instead of hiding them.

### 1.3 Why Existing Options Often Miss the Middle

Toolkits like LangChain, LlamaIndex, and Haystack provide composable primitives for building retrieval pipelines but leave critical production concerns to the implementer: user authentication, session management, VRAM budgeting, model lifecycle management, document processing with quality assurance, and end-user interfaces. Building a production-ready RAG application on these frameworks requires months of integration work.

At the other end, turnkey solutions like PrivateGPT and LocalGPT offer easy deployment but limited retrieval sophistication — typically single-mode dense retrieval without reranking, BM25 hybrid search, or hierarchical summarization.

ARCA targets the middle: a deployable platform with deeper retrieval and ops controls, while remaining configurable without forking core logic.

---

## 2. What ARCA Is

ARCA is a local-first LLM + retrieval platform for technical document workflows. It combines:

- Local model serving (`llama.cpp` / `llama-server`).
- Multi-stage hybrid retrieval (dense + rerank + optional sparse/hierarchy/graph/community paths).
- Document ingestion and extraction pipelines.
- Runtime operator controls (admin panel, diagnostics, benchmark harness).
- Domain pack extensibility without core-code forks.
- MCP-compatible backend interfaces for external clients.

ARCA is not a cloud chatbot wrapper and not a no-tuning "one click magic" stack.
It is an operator-facing system: inspectable, tunable, and explicit about tradeoffs.

---

## 3. Design Principles

### 3.1 Hybrid Retrieval Over Single-Mode Search

No single retrieval mode is best for all query types. ARCA supports a dense-first baseline and optional retrieval modes that can be enabled selectively by profile and query class.

### 3.2 Domain Packs Over Hardcoded Vertical Logic

Domain specialization is injected through pack configuration (manifest + lexicon + optional tools/routes), not hardcoded in core runtime. This keeps public core maintainable while allowing private specialization.

### 3.3 Adaptive Behavior Over Static Prompting

ARCA includes behavior adaptation (Phii) so response style and depth can track user interaction patterns while staying grounded in shared system rules.

### 3.4 Benchmark-Driven Tuning Over Assumption-Driven Defaults

RAG behavior is corpus-specific. ARCA treats benchmarking as a first-class workflow so defaults can be evidence-based for each deployment.

### 3.5 Hardware-Aware Operation Over Fixed Spec Assumptions

ARCA is GPU-first but supports fallback paths. Runtime behavior should scale to available resources instead of assuming one hardware class.

### 3.6 Goals

1. Local-first operation with explicit infrastructure ownership.
2. Retrieval quality controls exposed to operators.
3. Runtime configurability without rebuild loops.
4. Domain specialization without core forks.
5. Reproducible tuning via benchmark workflows.

### 3.7 Non-goals

1. Zero-friction setup on all hardware.
2. Hidden automation that obscures tradeoffs.
3. A universal static model/retrieval recipe.

---

## 4. RAG Fundamentals (Applied View)

This section covers the conceptual ground that the rest of the document builds on. If you already work with retrieval systems, skip to Section 5.

### 4.1 RAG Has Two Pipelines

1. Indexing pipeline (offline/batch):
   - parse and normalize documents
   - chunk and annotate metadata
   - embed and index
   - optionally build sparse, graph, and hierarchical indexes

2. Query pipeline (online):
   - classify/normalize query
   - retrieve candidates
   - fuse and rerank
   - apply quality gates
   - synthesize grounded answer with citations

Most production failures come from indexing and retrieval quality, not generation fluency.

### 4.2 Core Retrieval Primitives

- Dense retrieval: semantic nearest-neighbor search over embeddings.
- Sparse retrieval: lexical precision for identifiers and exact tokens.
- Fusion: rank aggregation across retrieval modes.
- Reranking: query-document joint scoring for precision.
- Diversity: reduce duplicate-source collapse in final context.

### 4.3 Why Hybrid Works in Practice

Dense and sparse modes fail differently. Dense misses exact strings; sparse misses paraphrases. Hybrid strategies are often more robust than either mode alone when combined with proper filtering and reranking.

### 4.4 Quality Gates Are Mandatory

Without explicit thresholds and confidence handling, RAG systems either over-answer with weak evidence or over-filter and become unhelpful. ARCA keeps these controls visible and tunable.

### 4.5 Common Failure Modes

1. Chunk-size mismatch (context dilution vs context loss).
2. Metadata drift (bad source labeling/page fidelity).
3. Over-broad candidate pools (latency and noise).
4. Over-aggressive filtering (valid evidence dropped).
5. Query-type mismatch (same pipeline path for all query intents).

ARCA's architecture exists to reduce these failure classes while keeping behavior observable.

---

## 5. ARCA System Architecture

### 5.1 Service Topology

ARCA deploys as Docker Compose with the following services:

```
+-----------------------------------------------------------+
|  Frontend (Next.js)                                        |
|  Chat UI / Admin Panel / Knowledge Graph / 3D Viz          |
+-----------------------------+-----------------------------+
                              | WebSocket + REST
+-----------------------------+-----------------------------+
|  Backend (FastAPI + llama-server)                           |
|  +----------+  +-----------+  +---------+  +-------------+ |
|  | Chat     |  | RAG       |  | Tools   |  | Admin       | |
|  | Engine   |  | Pipeline  |  | Registry|  | API         | |
|  +----+-----+  +-----+-----+  +----+----+  +-------------+ |
|       |              |              |                        |
|  +----+-----+  +-----+--------+    +--- Core tools (5)      |
|  | llama    |  | Embedder     |    +--- Domain tools (N)     |
|  | server   |  | + Reranker   |                              |
|  | (GGUF)   |  | + BM25       |                              |
|  +----------+  +--------------+                              |
+-----------------------------+-----------------------------+
                              |
  +--------+ +--------+ +----------+ +-------+ +--------+
  | Qdrant | |Postgres| |  Neo4j   | | Redis | |SearXNG |
  |Vectors | |  Data  | |  Graph   | | Cache | | Search |
  +--------+ +--------+ +----------+ +-------+ +--------+
```

| Service | Role | Port |
|---------|------|------|
| **Frontend** | Next.js 16 web interface, admin panel, graph visualization | 3000 |
| **Backend** | FastAPI application server, hosts llama-server processes | 8000 |
| **Qdrant** | Vector database for dense retrieval (HNSW + INT8 quantization) | 6333 |
| **PostgreSQL** | Structured data: user accounts, Phii learning, session metadata | 5432 |
| **Neo4j** | Knowledge graph for entity-based retrieval (GraphRAG) | 7474 |
| **Redis** | Session cache, rate limiting, Phii profile caching | 6379 |
| **SearXNG** | Privacy-focused web search engine (CRAG fallback) | 8080 |

LLM serving:

- chat slot: `llama-server` on `8081`
- vision slot: `llama-server` on `8082` (on-demand by default)

### 5.2 Backend Control Plane

Core files:

- [Backend startup and orchestration (`main.py`)](../backend/main.py)
- [Runtime configuration (`config.py`)](../backend/config.py)
- [Startup validation (`validation.py`)](../backend/validation.py)
- [Domain loading (`domain_loader.py`)](../backend/domain_loader.py)

`main.py` handles startup sequence and service orchestration.
`RuntimeConfig` in `config.py` provides runtime-tunable behavior with persisted overrides.

### 5.3 Why This Service Split Exists

ARCA uses service separation because local LLM systems fail in different ways across storage, indexing, and inference layers. The architecture is not minimal, but it is diagnosable.

Practical reasons:

- A vector DB issue should not take down auth/session management.
- Graph operations can fail independently without blocking base dense retrieval.
- Search backend (SearXNG) can be disabled/degraded while core local workflows still function.
- Operators can inspect bottlenecks per service instead of debugging one monolith.

### 5.4 Data Flow: Query to Response

A typical technical question follows this path:

1. **Connection**: Client establishes WebSocket connection. Backend authenticates user, restores session from Redis, loads Phii personality profile.
2. **Classification**: QueryClassifier examines the message and selects a handler based on priority. Handlers can force specific tools, inject context, or override model selection.
3. **Pre-retrieval**: If the query is technical, AutoSearchManager triggers a preemptive RAG search. Results are cached to avoid duplicate retrieval during tool dispatch.
4. **System prompt assembly**: PhiiContextBuilder composes the system prompt from personality template, energy/specialty/expertise guidance, active corrections, corpus context, tools section, and shared rules. All sections are functions, not static strings, enabling runtime adaptation.
5. **Initial LLM call**: The orchestrator sends the composed prompt and conversation history to llama-server. The model may return tool calls (native function calling) or a direct response.
6. **Tool dispatch**: Tool calls are validated against the ToolRegistry and executed. The N+1 optimization skips duplicate RAG calls when auto-search already retrieved results.
7. **Final LLM call**: Tool results are injected into the conversation and the model generates a grounded response.
8. **Streaming**: Response tokens are streamed word-by-word to the client with metadata (citations, confidence scores, thinking content, tool indicators).
9. **Post-processing**: Phii observes the exchange for implicit feedback and correction detection. Session state is saved to Redis with debounced persistence.

From an operator perspective, this is why one user message can involve multiple subsystems even when the UI appears simple.

### 5.5 Graceful Degradation

Every external service has a fallback path:

| Service | Fallback | Impact |
|---------|----------|--------|
| Redis | In-memory LRU cache (1000 entries, TTL support) | Session data not persistent across restarts |
| PostgreSQL | SQLite file database | Same schema, local file storage |
| Neo4j | GraphRAG disabled | Entity traversal unavailable; dense+sparse still work |
| Qdrant | RAG features disabled | Knowledge search unavailable |
| SearXNG | CRAG web fallback disabled | No web search supplement |
| LLM server | Error message to user | Chat unavailable but admin panel functional |

This means ARCA can run in a minimal configuration (backend + frontend + LLM only) with reduced features, scaling up as services are added.

### 5.6 Document Processing Pipeline

ARCA's document processing engine (Readd) implements a three-phase sequential architecture designed to prevent CUDA memory conflicts by ensuring only one GPU consumer runs at a time.

**Phase 1: Text Extraction (CPU).** PyMuPDF extracts text from every page. The PageClassifier categorizes each page using density-relative thresholds — P5, P15, and P50 percentiles computed per document, not absolute character counts — combined with vector path analysis for chart detection. This produces three categories: TEXT (~85% of pages in a typical technical document), VISUAL (~10%, charts and figures), and SCANNED (~5%, image-only pages). Pages with adequate table text route to PyMuPDF rather than the vision model.

**Phase 2: Vision Extraction (GPU).** Only VISUAL and SCANNED pages (~3-5% of a typical corpus) are sent to the vision LLM. This phase holds exclusive GPU access. Chart-detected pages receive specialized extraction that preserves data series, axis labels, and classification zones as structured data.

**Phase 3: Embedding (GPU).** After all text is assembled, the ONNX embedder takes exclusive GPU access. A token-aware auto-calibrating batch size uses binary search across the 4-128 range to find the largest batch that fits in available VRAM without OOM.

Key implementation details:

- **Memory management**: RAM-first buffering, spilling to NVMe when available system RAM drops below 25%.
- **Quality assurance**: Seven automated checks (empty pages, encoding issues, garbled text, repeated headers, missing pages, truncation, table artifacts) score weirdness points per page. Scores above 40 trigger re-extraction with a higher-quality extractor, up to 3 escalation attempts.
- **Session uploads** (text-only paste, no PDF processing) bypass all three phases and enter the chunking pipeline directly.

### 5.7 Personality Engine (Phii)

Phii is ARCA's adaptive personality system. It analyzes user communication patterns across six dimensions and adjusts response style in real time:

| Dimension | Detection Method | Adaptation |
|-----------|-----------------|------------|
| Brevity | Running word count average (last 10 messages) | Terse users get concise responses |
| Formality | Regex patterns (casual vs. formal vocabulary) | Match the user's register |
| Technical depth | Pattern counting (units, Greek letters, equations) | Adjust explanation depth |
| Specialty | Keyword analysis against lexicon-defined specialties | Focus on relevant domain areas |
| Expertise level | Signal accumulation (junior/intermediate/senior/management) | Calibrate response complexity |
| Preferences | Explicit signals ("use bullets", "metric units") | Remember and apply preferences |

The reinforcement learning subsystem maintains a PostgreSQL-backed store of corrections, action patterns, and user profiles. When a user says "Don't use X, use Y instead," the CorrectionDetector parses the correction with 0.95 confidence, stores it with context keywords and a semantic embedding, and applies it to future interactions. A verification loop monitors subsequent feedback: positive reactions boost correction confidence by 0.05; negative reactions decrease it by 0.10. Corrections below 0.3 confidence are automatically deactivated.

All personality features are domain-gated: technical patterns, seed corrections, specialty definitions, and terminology variants come from the lexicon, not from hardcoded values.

---

## 6. LLM Runtime and Model Lifecycle

### 6.1 Slot Definitions

[LLM slot configuration (`llm_config.py`)](../backend/services/llm_config.py) defines slot properties:

- model file
- context length
- GPU layer settings
- cache type and startup flags

### 6.2 Server Lifecycle Management

[LLM server lifecycle manager (`llm_server_manager.py`)](../backend/services/llm_server_manager.py) provides:

- process start/stop
- health checks
- slot model swapping
- startup timeout calibration
- warmup inference pass

Each slot maintains a self-calibrating startup timeout that learns from previous startup durations, adjusting the wait time based on observed performance on the specific hardware.

### 6.3 API Client Abstraction

[LLM API client (`llm_client.py`)](../backend/services/llm_client.py) wraps OpenAI-compatible calls:

- chat calls
- streaming handling
- tool-call translation
- image message translation

Important implementation detail: the `model` field in payload is API-shape compatible, but actual model comes from loaded slot state.

---

## 7. Model Bootstrap and First-Run Strategy

Large local models are operationally painful if onboarding is manual.
ARCA defaults to auto-bootstrap behavior:

- detect missing configured model files
- download from configured or known repos
- include vision mmproj where needed

### 7.1 Startup Bootstrap Path

[Backend model bootstrap (`model_bootstrap.py`)](../backend/services/model_bootstrap.py):

- controlled by:
  - `ARCA_AUTO_DOWNLOAD_MODELS`
  - `ARCA_AUTO_DOWNLOAD_OPTIONAL_MODELS`
- targets configured slots:
  - chat
  - code
  - expert
  - vision
  - vision_structured
- resolves repo from:
  - `LLM_<SLOT>_MODEL_REPO`
  - fallback maps

### 7.2 Host-Side Bootstrap Path

[Host-side model bootstrap script](../scripts/model_bootstrap.py) mirrors this logic pre-start so users can pull files before launching stack services.

This avoids shipping giant container images and keeps model upgrades independent from app image updates.

---

## 8. End-to-End Query Lifecycle in ARCA

### 8.1 Request Ingress

- frontend sends chat via WebSocket to `/ws/chat`
- backend validates session and operational state
- orchestration path selected

### 8.2 Handler and Orchestration Flow

Chat processing uses modular orchestration under [chat orchestration modules](../backend/routers/chat_orchestration/):

- query classification handlers
- tool dispatch routing
- model selection logic (chat/code/expert/vision intents)
- circuit breaker behavior for failure containment

### 8.3 Tool Execution

Execution modules live in [chat executor modules](../backend/routers/chat_executors/).
The central tool registry ([tool registry implementation](../backend/tools/registry.py)) controls available tools and schemas.

### 8.4 Response Synthesis

- tool outputs and retrieved context are assembled
- LLM response is streamed
- citations and metadata are attached in final payload

Section 5.4 covers the full data flow in more detail.

---

## 9. ARCA Retrieval Pipeline (Cohesionn)

[Cohesionn retrieval subsystem](../backend/tools/cohesionn/) is the retrieval subsystem.
It is modular by design so each stage can be tuned or disabled.

### 9.1 Major Modules

- [Retriever core (`retriever.py`)](../backend/tools/cohesionn/retriever.py): main orchestration and fusion
- [Embedding stack (`embeddings.py`)](../backend/tools/cohesionn/embeddings.py): ONNX-first embedding path
- [Reranker (`reranker.py`)](../backend/tools/cohesionn/reranker.py): cross-encoder reranking + diversity
- [Vector store adapter (`vectorstore.py`)](../backend/tools/cohesionn/vectorstore.py): Qdrant storage/search
- [Sparse retrieval (`sparse_retrieval.py`)](../backend/tools/cohesionn/sparse_retrieval.py): BM25 path
- [Query expansion (`query_expansion.py`)](../backend/tools/cohesionn/query_expansion.py): synonym expansion
- [HyDE generation (`hyde.py`)](../backend/tools/cohesionn/hyde.py): hypothetical document generation
- [Graph retrieval modules](../backend/tools/cohesionn/): extraction/build/traversal paths
- [RAPTOR hierarchy modules](../backend/tools/cohesionn/raptor/): summary-tree retrieval path

### 9.2 Pipeline Shape

High-level flow in [retriever core](../backend/tools/cohesionn/retriever.py):

1. query normalization/expansion
2. optional HyDE augmentation
3. topic routing
4. dense candidate retrieval
5. optional sparse retrieval
6. optional hierarchy/graph retrieval
7. rank fusion
8. reranking
9. diversity re-selection
10. quality thresholding
11. optional corrective fallback
12. citation formatting

### 9.3 Fusion and Re-ranking

ARCA uses rank-oriented blending and post-fusion reranking.
The exact coefficients are runtime-configurable and benchmark-driven.

Useful mental model:

- retrieval stages maximize recall
- reranking and thresholds maximize precision
- diversity controls avoid citation collapse into one source

### 9.4 Important: More Stages Is Not Automatically Better

A 12-stage capable pipeline does not imply a 12-stage recommended runtime profile.

In real corpora, over-stacking retrieval stages can:

- increase latency with marginal or negative quality gain
- amplify noisy candidates through unnecessary transforms
- make failure analysis harder

ARCA intentionally provides many switches so operators can benchmark and tune, not to run all switches "on" by default. The v2 benchmark demonstrated this directly: the full deep pipeline scored 7% worse than dense-only at 70x the latency on a structured corpus. The toolbox exists for cases where specific stages earn their cost — cross-reference queries, vocabulary mismatch, hierarchical reasoning — not for blanket activation.

### 9.5 Profile Strategy by Workload

A practical way to choose profiles:

- Interactive Q&A / operator chat:
  - Start with `fast`.
  - Enable additional stages only when specific query classes fail.
- Cross-reference and synthesis-heavy workflows:
  - Use `deep` selectively or route by query classifier.
- Production baseline after tuning:
  - Use `auto` once benchmark winners are validated on representative queries.

This keeps latency predictable while preserving escalation paths.

### 9.6 Retrieval Profiles

The platform supports named retrieval profiles that control pipeline stage activation. Profiles resolve through a four-layer override hierarchy: per-query parameters take precedence over manual toggle overrides, which override the active profile, which overrides environment defaults.

This allows operators to set a system-wide default while permitting per-query escalation to full pipeline coverage when needed.

---

## 10. Document Ingestion and Extraction

ARCA separates ingestion concerns from chat flow. Section 5.6 covers the three-phase document processing pipeline in detail.

### 10.1 Extraction Subsystem

Key tool modules:

- [Readd extraction pipeline](../backend/tools/readd/)
- [Observationn vision extraction](../backend/tools/observationn/)
- [Redactrr redaction pipeline](../backend/tools/redactrr/)

This stack handles:

- text-rich digital documents
- mixed layout pages
- vision-required pages
- fallback and escalation behavior

### 10.2 Why Ingestion Is Hard

Real corpora include:

- low-quality OCR
- scanned pages
- tables and charts
- inconsistent structure

The ingestion pipeline must optimize for throughput while preserving extraction quality. If extraction and chunking metadata are bad, no reranker can fully recover. This is one of the biggest engineering effort areas in ARCA, and it is why the pipeline treats ingestion as a first-class engineering system rather than a preprocessing script.

---

## 11. Domain Pack Extension Model

Domain packs let ARCA specialize without hardcoding a single industry into core code. Out of the box, ARCA is a general-purpose RAG platform with neutral defaults. Specialization for any industry — engineering, legal, medical, financial — is achieved through domain packs: self-contained configuration directories that customize behavior without modifying core code.

This separation has two benefits:

1. **No forking**: Domain-specific knowledge does not pollute the core codebase. Upstream updates apply without merge conflicts.
2. **Multiple domains**: Different instances can serve different domains by changing one environment variable (`ARCA_DOMAIN`).

### 11.1 Pack Structure

```
domains/{pack_name}/
  manifest.json       # Pack metadata, tool declarations, branding
  lexicon.json        # Vocabulary, detection patterns, pipeline config
  tools/              # Custom tool modules (optional)
  executors/          # Tool execution logic (optional)
  routes/             # Additional API endpoints (optional)
  docs/               # Domain-specific documentation (optional)
```

### 11.2 Manifest

The manifest declares pack metadata, tool registrations, handler registrations, API routes, and branding:

```json
{
  "name": "example",
  "display_name": "ARCA Assistant",
  "version": "0.1.0",
  "arca_min_version": "0.1.0",
  "description": "A general-purpose document intelligence assistant.",
  "tools": [],
  "handlers": [],
  "routes": [],
  "lexicon": "lexicon.json",
  "branding": {
    "app_name": "ARCA",
    "tagline": "From Documents to Domain Expert",
    "primary_color": "#6366f1"
  }
}
```

### 11.3 Lexicon-Driven Pipeline Injection

The lexicon is the primary mechanism for domain customization. Its `pipeline` section flows through the entire retrieval and personality system via `get_pipeline_config()`, which merges lexicon values with neutral defaults and passes through all keys — including domain-specific keys not present in the defaults.

| Pipeline Key | Consumer | Effect |
|-------------|----------|--------|
| `rag_topic_descriptions` | TopicRouter, Reranker | Semantic routing and boost |
| `rag_topic_keywords` | TopicRouter, Domain Boost | Keyword-based routing and scoring |
| `rag_synonyms` | Query Expansion | Synonym injection for queries |
| `rag_preserve_terms` | BM25 Tokenizer | Terms preserved during tokenization |
| `hyde_detection_keywords` | HyDE Generator | Query type detection for prompt selection |
| `graph_test_methods` | Entity Extractor | Pattern matching for graph construction |
| `graph_parameters` | Entity Extractor | Domain parameter extraction |
| `phii_technical_patterns` | Energy Detector | Technical depth analysis patterns |
| `phii_seed_corrections` | Reinforcement Store | Baseline corrections |
| `phii_compliance_suggestion` | Context Builder | Compliance behavior injection |
| `raptor_context` | RAPTOR Summarizer | Domain-appropriate summarization prompts |
| `raptor_preserve` | RAPTOR Summarizer | Terms to preserve in summaries |
| `trusted_search_domains` | Web Search | Domain-specific trusted sources |

### 11.4 Domain-Gated Imports

All domain-specific code imports in ARCA's backend use the `try/except ImportError` pattern:

```python
try:
    from tools.mapperr import process_location
    MAPPERR_AVAILABLE = True
except ImportError:
    MAPPERR_AVAILABLE = False
```

This is applied consistently across the backend, ensuring ARCA starts cleanly regardless of which domain pack is active. Missing domain tools report as unavailable rather than crashing.

### 11.5 Cache Invalidation

When the domain is switched or the lexicon is edited, `clear_prompt_caches()` invalidates all pipeline caches: graph extraction patterns, reranker topic config, synonym groups, BM25 preserve terms, technical patterns, and HyDE keywords. The next query triggers fresh loading from the updated lexicon.

Core loader/registry path:

- [Domain loader (`domain_loader.py`)](../backend/domain_loader.py)
- [Tool registry (`registry.py`)](../backend/tools/registry.py)
- [Domain registration entrypoints (`domains/<name>/register_tools.py`)](../domains/)

---

## 12. Admin Control Plane and Operability

Admin APIs are grouped under [admin router modules](../backend/routers/admin/) and related packages.
They cover:

- model inspection and assignment
- health/status visibility
- log/session operations
- knowledge ingestion and reprocessing
- benchmark execution and result application
- graph and tool management

Frontend admin implementation:

- [Admin UI app directory](../frontend/app/admin/)
- centralized state in [useAdminState hook](../frontend/app/admin/hooks/useAdminState.ts)

The goal is to reduce "SSH-only operations" and make tuning/recovery accessible from UI. In practice, most day-to-day ARCA operations — model swaps, retrieval profile changes, knowledge ingestion, benchmark runs — can be done entirely from the admin panel.

---

## 13. Benchmark Philosophy and Workflow

ARCA includes benchmark tooling because retrieval configuration should be measured, not guessed. This is not leaderboard chasing — it is operator-facing model/retrieval tuning against the operator's own data.

### 13.1 Why Benchmarking Matters

RAG pipeline performance depends on the interaction between corpus characteristics, model selection, and parameter tuning. A configuration that scores well on general-purpose benchmarks may underperform on a specific technical library because the vocabulary, document structure, and query patterns differ substantially.

The benchmark harness addresses this by evaluating pipeline components against the user's actual data, accessible from the admin panel UI.

Primary areas:

- [Benchmark engine](../backend/benchmark/)
- [Admin benchmark endpoints](../backend/routers/admin_benchmark/)

### 13.2 Benchmark Methodology

The benchmark harness has evolved through two generations of evaluation methodology.

**v1 methodology** uses a composite scoring approach combining keyword hit rate (35%), entity hit rate (15%), MRR (20%), nDCG@k (15%), source diversity (10%), and latency (5%). Queries span seven difficulty tiers (factual, numerical, conceptual, multi-entity, domain-specific, negation, multi-hop). This approach effectively measures retrieval recall — whether the retriever found chunks containing the right terms — but has a known limitation: it cannot measure whether retrieved context enables a correct and complete answer. Components like RAPTOR (synthesized summaries), GraphRAG (relationship language), and HyDE (vocabulary bridging) can contribute real value that keyword overlap cannot detect.

**v2 methodology** addresses this with a layered evaluation pipeline. The published v2 benchmark (`docs/BENCHMARK_V2_PUBLIC.md`) implements 14 layers including: chunking optimization sweeps (114 configurations), retrieval toggle ablation (15 configurations), continuous parameter sweeps, embedding and reranker model shootouts, cross-model combination sweeps (27 combinations), answer generation evaluation, dual LLM-as-judge scoring (Gemini + Claude Opus with ground truth access), statistical analysis, failure categorization, model ceiling comparison, and live adversarial pipeline testing. The dual-judge approach revealed that a single LLM judge without ground truth access systematically underscores domain-specific factual accuracy by +1.37 points — a finding that has implications beyond ARCA.

### 13.3 Published Benchmark V2 Snapshot

ARCA's public v2 benchmark report covers 168+ tested configurations with 40 ground-truth queries and 20 adversarial queries.

High-value findings:

1. Dense-first baseline was near-ceiling on the tested structured corpus.
   - `dense_only`: composite ~0.774 at ~19ms
   - `expansion_only`: composite ~0.775 at ~2.3ms
2. Full deep pipeline underperformed on aggregate.
   - `deep_profile`: composite ~0.721 at ~1330ms
3. Cross-reference queries were the key exception.
   - Graph-heavy traversal improved cross-reference performance (~+0.024 composite, MRR 1.000).
4. Model stack validated by cross-matrix testing.
   - Best combination: Qwen3-Embedding-0.6B + Jina-Reranker-v2-Turbo (composite ~0.839, entity recall 1.000).
5. Frontier model ceiling gap is real.
   - Given identical context, Claude Opus 4.6 scored +0.120 composite over the local 30B MoE model. The gap was largest on negation queries (+0.444) where abstention behavior matters most.

These values are corpus-specific, not universal constants. The important general lesson is to keep a dense-first default and gate deep features by query type and benchmark evidence.

### 13.4 How Benchmark Results Map to Defaults

Operationally, benchmark outcomes drive:

- Retrieval profile defaults (`fast` for broad interactive use).
- Which optional stages remain off by default (BM25/HyDE/RAPTOR/Graph/CRAG as selective tools, not always-on).
- Reranker and embedder model choices.
- Threshold and candidate-count settings where signal is measurable.

### 13.5 Benchmark Runbook (Practical)

For a first serious tuning pass:

1. Build a corpus slice that matches real production documents.
2. Prepare queries that represent actual user behavior (not toy prompts).
3. Run chunking and retrieval sweeps first.
4. Lock retrieval toggles before chasing generation-model differences.
5. Inspect failure categories, not just average score.
6. Apply winners to `auto` profile and run a manual smoke set.

Most quality gains come from retrieval and chunk strategy, not from swapping the generation model.

### 13.6 Planned Methodology Extensions

Future benchmark work targets three areas:

- **User archetype testing**: Expressing each benchmark query in multiple interaction styles (power user, mid-career, terse veteran, verbose junior) to measure which pipeline components serve which user populations.
- **Synthetic corpus blind evaluation**: Generating internally consistent fictional corpora so the benchmark operator cannot unconsciously optimize queries toward known content.
- **Confidence-based adaptive routing**: Using dense retrieval confidence scores to dynamically select pipeline depth per query, routing high-confidence queries to a fast path and escalating low-confidence queries to the full pipeline.

---

## 14. External Integration (MCP)

ARCA exposes its retrieval pipeline as a set of tools via the Model Context Protocol (MCP), enabling external AI assistants — Claude Desktop, GPT desktop apps, or custom agents — to use ARCA as a domain-aware backend.

### 14.1 Architecture

The MCP server is a thin adapter layer:

```
MCP Client (Claude Desktop, GPT, custom agents)
    | spawns subprocess, STDIO transport
MCP Server (Python, host machine, ~200 lines)
    | HTTP calls to localhost:8000
ARCA Backend (Docker container)
    | existing executor functions
Qdrant, Neo4j, Redis, PostgreSQL, llama.cpp, SearXNG
```

The server defines six tools:

| Tool | Backend Executor | Purpose |
|------|-----------------|---------|
| `search_knowledge` | RAG pipeline (Cohesionn) | Hybrid retrieval over ingested corpus |
| `web_search` | SearXNG proxy | Privacy-focused web search |
| `unit_convert` | Unit conversion engine | Engineering/scientific unit conversion |
| `list_topics` | Topic discovery | List knowledge topics with enabled status |
| `corpus_stats` | Qdrant + manifest | Chunk counts, file counts, collection sizes |
| `system_status` | Health checks | Component status (LLM, Redis, Qdrant, Postgres) |

Three MCP resources provide read-only state: `arca://topics` (topic list), `arca://domain` (active domain config), and `arca://status` (system health).

### 14.2 Authentication

MCP endpoints use a static API key (`MCP_API_KEY` environment variable) checked via an `X-MCP-Key` header. This avoids the JWT login flow — the MCP server is a long-running subprocess that needs stateless authentication. When `MCP_API_KEY` is unset, all MCP endpoints return 503, disabling the feature by default.

### 14.3 Design Constraints

The MCP server is deliberately read-only. It cannot trigger ingestion, modify configuration, create sessions, or upload files. This limits the blast radius of a compromised API key to information disclosure from the existing corpus — no writes, no deletes, no state changes.

The server runs on the host machine (not in Docker) because MCP clients spawn it as a local subprocess via STDIO transport. It communicates with the ARCA backend over localhost HTTP, keeping all traffic on the loopback interface.

---

## 15. Comparison with Alternatives

The following comparison is based on publicly documented capabilities as of February 2026. Each project has strengths that ARCA does not match, and those are noted explicitly.

| Capability | ARCA | LangChain | LlamaIndex | Haystack | PrivateGPT | LocalGPT | RAGFlow |
|-----------|------|-----------|------------|----------|-----------|---------|---------|
| **Deployment** | Complete platform (Docker Compose) | Toolkit (assemble yourself) | Toolkit (assemble yourself) | Toolkit (assemble yourself) | Turnkey app | Turnkey app | Turnkey app |
| **Retrieval modes** | 7 (dense + BM25 + RAPTOR + GraphRAG + global + HyDE + CRAG) | User-assembled | User-assembled | Configurable pipeline | Dense + optional sparse | Dense only | Dense + sparse |
| **Cross-encoder reranking** | Built-in (5 models benchmarked) | Via integration | Via integration | Built-in | Via integration | No | Built-in |
| **Domain specialization** | Lexicon-driven packs (no code changes) | Custom code per domain | Custom code per domain | Custom code per domain | No | No | Template system |
| **Benchmark harness** | 14-layer, corpus-specific | No | No | Eval framework | No | No | No |
| **Personality learning** | Phii (energy, specialty, expertise, reinforcement) | No | No | No | No | No | No |
| **Hardware auto-scaling** | VRAM-aware context + model selection | No | No | No | Manual | Manual | Manual |
| **Vision extraction** | Per-page routing (text/vision/chart) | No | Document loaders | No | No | No | Document processing |
| **Admin panel** | Full (config, knowledge, graph, benchmark, diagnostics) | No | No | Deepset Studio | Minimal | No | Document management |
| **LLM backend** | llama.cpp (GGUF, local) | Any (API-first) | Any (API-first) | Any | llama.cpp / GPT4All | llama.cpp | Ollama / API |
| **Knowledge graph** | Neo4j (auto-built from entities) | Via integration | KnowledgeGraph index | No | No | No | No |
| **MCP integration** | Built-in (6 tools, STDIO transport) | No | No | No | No | No | No |
| **License** | MIT | MIT | MIT | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |

### Where Alternatives Are Stronger

- **LangChain** and **LlamaIndex** offer far broader integration ecosystems (hundreds of connectors, vector stores, and LLM providers). If you need to connect to many data sources or switch between cloud LLM APIs, these frameworks provide that flexibility. ARCA is opinionated about its stack.

- **Haystack** provides a mature evaluation framework and pipeline-as-graph abstraction. Its Deepset Studio offers a visual pipeline editor.

- **PrivateGPT** and **LocalGPT** are simpler to set up for basic use cases. If you need a local chatbot over your documents in 10 minutes with no configuration, these projects have lower barriers to entry.

- **RAGFlow** offers an intuitive document processing pipeline with template-based chunk editing and a polished document management interface.

### Where ARCA Is Stronger

- **Retrieval depth**: 7-mode hybrid retrieval with RRF fusion, cross-encoder reranking, and MMR diversity is not available as a pre-assembled pipeline in any alternative.

- **Corpus-specific optimization**: No alternative includes an integrated benchmark harness that evaluates embedder x reranker x parameter combinations against the user's data.

- **Domain specialization without forking**: The lexicon-driven domain pack system allows full behavioral customization through configuration files.

- **Adaptive personality**: Phii's real-time expertise detection, energy matching, and reinforcement learning have no equivalent in the listed alternatives.

- **Hardware-aware operation**: Automatic VRAM budgeting and context scaling across CPU-only to multi-GPU configurations.

---

## 16. Security and Privacy Model

ARCA is local-first but not "secure by magic."

### 16.1 Security Posture

The security model is built on local deployment as the primary control:

- All services run on the operator's infrastructure by default.
- No telemetry, no external API calls, no data exfiltration paths in default configuration.
- Service separation limits blast radius — a vector DB compromise does not expose auth data.
- Auth/session controls gate sensitive operations.
- Admin/API gating restricts configuration changes and knowledge management.
- Configurable secrets and policy documents.

### 16.2 Operational Reality

Local-first reduces cloud data exposure, but:

- Host and network security remain the operator's responsibility.
- Docker container isolation is not a hard security boundary.
- WebSocket authentication has a known gap (see Section 19).
- MCP mode uses a static API key — appropriate for localhost, not for network exposure.

See [Security Policy](SECURITY.md) for policy-level details and responsible disclosure process.

---

## 17. Performance and Hardware Realities

ARCA is built for GPU-first local inference. Different hardware tiers need different defaults, and ARCA is designed to detect and adapt rather than assume one hardware class.

### 17.1 VRAM Budget System

At startup, ARCA detects available hardware using nvidia-smi (for GPUs), `/proc/meminfo`, and `/proc/cpuinfo`. The detected VRAM budget determines defaults across the entire system:

| Tier | VRAM | Context Windows | GPU Layers | Capabilities |
|------|------|----------------|------------|-------------|
| CPU-only | 0 | 2K / 4K / 8K | 0 | 7B Q4 on CPU, embedder on CPU |
| Small | < 8 GB | 2K / 4K / 8K | Partial | 7B Q4 on GPU, embedder on GPU |
| Medium | 8-16 GB | 4K / 8K / 16K | Full (7B) | 14B Q4 + full RAG pipeline on GPU |
| Large | 16+ GB | 4K / 8K / 24K | Full | 32B Q4 + concurrent vision model |

Context windows auto-scale by task complexity: simple tool calls use the smallest context, standard chat uses medium, RAG-heavy queries use large, and think/expert mode uses the largest available.

### 17.2 Model Lifecycle Management

ARCA manages two llama-server processes internally:

- **Chat slot**: Always loaded with the configured chat model (default: Qwen3-30B-A3B at ~18.6 GB Q4_K_M).
- **Vision slot**: Loaded on-demand when document processing or chart extraction is needed.

The LLM server manager checks VRAM budget before starting each process. If insufficient VRAM is available, it reports the constraint rather than starting a process that would fail silently.

### 17.3 ONNX Optimization

Both the embedder and reranker prefer ONNX Runtime over PyTorch, saving approximately 17 GB of host RAM by avoiding CUDA context initialization. Models are exported to ONNX on first load and cached to `~/.cache/huggingface/onnx_exports/` for subsequent startups.

### 17.4 Multi-GPU Device Map

On systems with multiple GPUs, ARCA supports explicit device mapping via `ARCA_DEVICE_MAP`:

```
ARCA_DEVICE_MAP=chat:0,vision:1,embedder:2,reranker:2
```

This routes each component to a specific GPU index, enabling concurrent operation. On single-GPU systems, all components route to GPU 0 by default.

### 17.5 Sizing Guidance

- **8-12 GB VRAM**: Use smaller quantized chat models and conservative context windows. Keep optional heavy retrieval stages selective.
- **16-24 GB VRAM**: Comfortable baseline for a strong chat model + full retrieval stack. Good balance for most local technical workloads.
- **24+ GB VRAM or multi-GPU**: Better headroom for larger models, longer contexts, and richer concurrent workloads. Still requires careful profile tuning.

### 17.6 Hardware Simulation

For testing and development, ARCA supports hardware simulation via environment variables (`ARCA_SIMULATE_VRAM`, `ARCA_SIMULATE_GPU`, etc.). This allows validating behavior across all VRAM tiers without physical hardware.

---

## 18. Failure Modes and Mitigations

No system this complex is immune to failure. ARCA's approach is to make failures diagnosable and recoverable rather than invisible.

### 18.1 Representative Failure Classes

| Failure | Mitigation |
|---------|-----------|
| Missing model artifacts | Startup bootstrap, explicit slot/repo env, mmproj handling |
| GPU startup mismatch | Compose compatibility mode and diagnostics |
| Retrieval quality collapse on noisy corpora | Configurable reranker/threshold/fusion controls |
| Long-tail extraction errors | Fallback extractors and staged ingestion logic with quality scoring |
| Configuration drift | Runtime config persistence + admin visibility |
| Service dependency failure | Graceful degradation paths per service (Section 5.5) |

ARCA does not claim these are fully solved, only that they are explicitly handled and visible to operators.

### 18.2 Degraded-Mode Runbook

When startup lands in degraded mode:

1. Confirm dependency health (PostgreSQL, Redis, Qdrant, Neo4j).
2. Check model availability and exact filename matches.
3. Validate GPU runtime visibility from inside containers.
4. Verify retrieval topics and core knowledge ingest state.
5. Re-run with focused logs before changing multiple settings at once.

This disciplined loop prevents configuration thrash and makes root cause easier to isolate.

---

## 19. Known Limitations (v0.1)

These are the limitations that matter most in the current release:

1. **Single-user concurrency model**: ARCA manages one chat LLM process and one vision LLM process. Multiple concurrent users share these processes sequentially. True multi-user scaling requires a queuing system or multiple backend replicas.

2. **WebSocket authentication gap**: The main chat WebSocket connection does not currently transmit authentication tokens. File uploads similarly lack authentication headers. Acceptable on a trusted local network; must be addressed for any network-exposed deployment.

3. **Monolithic frontend components**: The main chat interface and admin panel are large React components. Functional but would benefit from decomposition.

4. **AGPL dependency chain**: PyMuPDF (AGPL-3.0) and marker-pdf (GPL-3.0) are used for PDF processing. Users must either acquire commercial licenses or accept the copyleft implications.

5. **Limited document formats**: Currently supports PDF, DOCX, TXT, and MD. Spreadsheet and HTML ingestion are not yet implemented.

6. **Partial multi-GPU support**: Device mapping and optional llama.cpp tensor split are supported for slot-level inference, but advanced autoscheduling and deep multi-GPU orchestration are still limited.

7. **Hardware and driver variability**: Edge cases exist across GPU vendors, driver versions, and Docker configurations. Setup remains medium complexity for non-technical users.

8. **Evaluation on one corpus type**: Published benchmark results are from structured technical corpora. Results on other corpus types (legal, medical, conversational) will vary. The benchmark harness exists precisely so users can generate their own numbers.

This release is practical and usable, but not "done forever." Planned improvements are tracked in repository issues and future release notes.

---

## 20. How To Read ARCA Code Quickly

If you want full system understanding fast, read in this order:

1. [Backend startup and lifecycle (`main.py`)](../backend/main.py)
2. [Runtime configuration (`config.py`)](../backend/config.py)
3. [Chat entrypoint (`chat.py`)](../backend/routers/chat.py)
4. [Chat orchestration modules](../backend/routers/chat_orchestration/)
5. [Tool registry (`registry.py`)](../backend/tools/registry.py)
6. [Retrieval core (`cohesionn/retriever.py`)](../backend/tools/cohesionn/retriever.py)
7. [Document extraction (readd)](../backend/tools/readd/) + [vision extraction (observationn)](../backend/tools/observationn/)
8. [LLM server manager (`llm_server_manager.py`)](../backend/services/llm_server_manager.py)
9. [Admin frontend (`frontend/app/admin/`)](../frontend/app/admin/) state + model tabs

Then use:

- [Backend index](../backend/_INDEX.md)
- [Router index](../backend/routers/_INDEX.md)
- [Tool index](../backend/tools/_INDEX.md)
- [Cohesionn index](../backend/tools/cohesionn/_INDEX.md)

---

## 21. Practical Guidance for Builders

If you are building your own complex RAG system, the core lessons from ARCA:

1. Treat ingestion as a first-class engineering system, not a preprocessing script
2. Build retrieval as composable stages with toggles
3. Keep thresholds explicit and testable
4. Add benchmark paths early — you will not regret having a systematic way to answer "does this change actually help?"
5. Build operational visibility from day one
6. Avoid hiding hardware realities from users
7. Keep extension points clean (domain/tool packs)

The most valuable thing ARCA produced was not any single pipeline component — it was the benchmark harness that told us which components actually mattered on our corpus. Months of building followed by two hours of systematic testing revealed that the baseline was doing most of the work. Build the harness first.

---

## 22. Closing

ARCA exists because building local, domain-aware AI systems is harder than it looks — and because most of the available options either leave too much assembly work to the operator or hide too much behind a simple interface that falls apart on real corpora.

This project is the accumulated engineering work from that problem. It is one person's attempt to build a RAG platform that takes retrieval seriously, exposes its tradeoffs honestly, and gives operators the tools to tune for their own data instead of trusting generic defaults. It is intentionally released in a state where others can inspect, use, and improve it — not just consume it.

The code, the benchmark harness, the domain pack system, the hardware-aware scaling — all of it is MIT-licensed and open. If you find issues and report them clearly, that makes it better for everyone. If you build something useful on top of it, even better.

If this saves you time, that is the point.

---

## 23. Technical Appendix: How Complex RAG Works (Practical View)

This appendix is the deeper "under the hood" explanation for readers who want implementation-level intuition.

### 23.1 Retrieval Is a Ranking Problem, Not a Generation Problem

Most RAG outcomes are determined before the model writes a single token.
Generation quality can only be as good as retrieved evidence.

A robust stack should therefore optimize:

1. recall at candidate stage
2. precision at rerank/filter stage
3. citation integrity in final assembly

ARCA maps this to:

- candidate stages in `retriever.py` and vector/sparse/graph modules
- precision stages in reranker + score thresholds
- citation formatting in `RetrievalResult` and chat assembly paths

### 23.2 Dense Retrieval (Semantic Layer)

Dense retrieval uses embedding vectors for query and chunk text.
Similarity is usually cosine:

`sim(q, d) = (q . d) / (||q|| * ||d||)`

Dense retrieval excels at semantic paraphrase matching but can miss literal identifiers or rare tokens.
ARCA uses dense retrieval as core retrieval substrate through Qdrant-backed stores and embedder modules.

### 23.3 Sparse Retrieval (Lexical Layer)

Sparse retrieval (BM25-like) scores lexical overlap with term-frequency and inverse-document-frequency weighting.
A common form:

`score(q, d) = sum( IDF(t) * ((f(t,d) * (k1 + 1)) / (f(t,d) + k1 * (1 - b + b * |d|/avgdl))) )`

Sparse retrieval catches exact domain terms and identifiers dense models can blur.
ARCA exposes sparse retrieval as optional/fused stage rather than mandatory path.

### 23.4 Fusion (Multi-Retriever Blending)

ARCA uses rank-based blending approaches (including RRF-style fusion paths) to combine dense/sparse/other rank lists.

Reciprocal Rank Fusion form:

`RRF(d) = sum_i 1 / (k + rank_i(d))`

Why rank fusion is practical:

- avoids score-scale mismatch between retrievers
- tends to be robust under heterogeneous retrieval distributions
- easy to reason about in ops settings

### 23.5 Reranking (Precision Pass)

Rerankers (cross-encoders) score query+chunk jointly and usually improve top-k relevance.
They are expensive, so they are best used on reduced candidate sets.

ARCA pattern:

1. retrieve broader candidate pool
2. rerank top N
3. apply score threshold
4. diversity pass

This gives better precision while constraining latency.

### 23.6 Diversity Re-selection (MMR Intuition)

Maximum Marginal Relevance style selection balances relevance with novelty.
Typical objective:

`argmax_d [ lambda * Rel(d, q) - (1-lambda) * max_{s in S} Sim(d, s) ]`

This prevents top-k from collapsing into near-duplicate chunks from one source.
ARCA includes diversity-aware reranking to improve evidence spread.

### 23.7 Quality Gates and Confidence Bands

Without hard quality gates, RAG systems tend to:

- include weak evidence and hallucinate with confidence
- or block too aggressively and return no help

ARCA uses thresholding and confidence classification to choose fallback behavior (including optional corrective paths).

### 23.8 HyDE, Expansion, Hierarchy, Graph: Why These Exist

These are recall and robustness tools, not mandatory magic:

- query expansion: alias/synonym coverage
- HyDE: better semantic probe text for difficult queries
- hierarchy (RAPTOR-style): global/contextual retrieval for broad questions
- graph retrieval: entity-relation traversal for cross-reference logic

In practice, each can improve some corpora and hurt others.
That is why ARCA keeps them configurable and benchmarked, not hard-wired.

### 23.9 Ingestion Quality Is First-Order

If extraction/chunking metadata is bad, no reranker can fully recover.
High-performing RAG systems treat ingestion as an engineering product:

- extraction fallback chains
- page-type-aware handling
- metadata normalization
- manifesting and reprocess controls

### 23.10 Operational Formula for Real Systems

A practical mental model:

`RAG quality = f(extraction_quality, chunk_strategy, recall, rerank_precision, thresholding, prompt_assembly, model_fit)`

Most teams over-focus on `model_fit`.
ARCA was built around raising the floor on the other terms.

---

## 24. References and Companion Docs

Primary docs:

- [Project README](../README.md)
- [API Quick Reference](API.md)
- [API Full Reference](API_FULL.md)
- [Configuration Guide](dev/config.md)
- [Troubleshooting](dev/troubleshooting.md)
- [Domain Pack Guide](dev/domain_packs.md)
- [Benchmark V2 Public Report](BENCHMARK_V2_PUBLIC.md)
- [Security Policy](SECURITY.md)

Code map documents:

- [Backend index](../backend/_INDEX.md)
- [Router index](../backend/routers/_INDEX.md)
- [Tool index](../backend/tools/_INDEX.md)
- [Cohesionn index](../backend/tools/cohesionn/_INDEX.md)

---

## 25. Citations and Source Map

### 25.1 ARCA Code/Document Citations

- [A1] [Backend index](../backend/_INDEX.md)
- [A2] [Router index](../backend/routers/_INDEX.md)
- [A3] [Tool index](../backend/tools/_INDEX.md)
- [A4] [Cohesionn index](../backend/tools/cohesionn/_INDEX.md)
- [A5] [Retriever core](../backend/tools/cohesionn/retriever.py)
- [A6] [Backend model bootstrap](../backend/services/model_bootstrap.py)
- [A7] [Host model bootstrap script](../scripts/model_bootstrap.py)
- [A8] [LLM server manager](../backend/services/llm_server_manager.py)
- [A9] [LLM API client](../backend/services/llm_client.py)
- [A10] [Runtime config](../backend/config.py)
- [A11] [Domain loader](../backend/domain_loader.py)
- [A12] [Project README](../README.md)

### 25.2 External References

- [E1] Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- [E2] Robertson and Zaragoza (2009), The Probabilistic Relevance Framework: BM25 and Beyond.
- [E3] Cormack, Clarke, and Buettcher (2009), Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.
- [E4] Carbonell and Goldstein (1998), The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.
- [E5] Gao et al. (2022), Precise Zero-Shot Dense Retrieval Without Relevance Labels.
- [E6] Nogueira and Cho (2019), Passage Re-Ranking with BERT.
- [E7] Sarthi et al. (2024), RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.
- [E8] Yan et al. (2024), Corrective Retrieval Augmented Generation.

---

## 26. Citation

```bibtex
@software{arca2026,
  title     = {ARCA: Local-First RAG Platform for Technical Document Workflows},
  author    = {ARCA Contributors},
  year      = {2026},
  version   = {0.1.0},
  url       = {https://github.com/3spky5u-oss/ARCA},
  note      = {A local-first RAG platform with 7-mode hybrid retrieval,
               domain pack specialization, adaptive personality,
               and integrated benchmark harness},
  license   = {MIT}
}
```

---

*ARCA v0.1.0 — February 2026*
*License: MIT*
