# backend/tools/

ARCA's tool modules. Each tool is a self-contained package with its own models, logic, and CLI. Tools register via the central ToolRegistry singleton. Domain-specific tools load conditionally via `try/except ImportError`.

| File | Purpose | Key Exports |
|------|---------|-------------|
| __init__.py | Package marker | -- |
| registry.py | Singleton tool registry: register, lookup, execute, schema generation | `ToolRegistry`, `ToolDefinition`, `ToolResult`, `ToolCategory`, `register_all_tools()` |

## Tool Modules

| Directory | Purpose | Key Classes |
|-----------|---------|-------------|
| cohesionn/ | RAG pipeline: hybrid retrieval, embeddings, reranking, chunking, ingestion, benchmarks | `CohesionnRetriever`, `UniversalEmbedder`, `BGEReranker`, `SemanticChunker` |
| readd/ | PDF/document processing: scout, extract, QA, escalation pipeline | `ReaddPipeline`, `DocumentScout`, `HybridExtractor`, `GraphExtractor` |
| phii/ | Personality/behavior system: energy, expertise, specialty detection, reinforcement learning | `PhiiContextBuilder`, `EnergyDetector`, `ReinforcementStore` |
| redactrr/ | PII detection and redaction for PDF/DOCX (regex + heuristic + LLM) | `Redactrr`, `PIIDetector`, `PDFRedactor`, `WordRedactor` |
| observationn/ | Vision-based extraction using Qwen3-VL for scanned/complex pages | `ObservationnExtractor` |

## Registration Flow

1. `main.py` startup calls `register_all_tools()`
2. Core tools registered unconditionally (unit_convert, search_knowledge, search_session, web_search, redact_document)
3. Domain tools loaded via `importlib.import_module(f"domains.{name}.register_tools")`
4. `ToolRegistry.reinitialize()` supports runtime domain switching
