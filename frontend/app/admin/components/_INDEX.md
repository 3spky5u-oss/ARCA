# frontend/app/admin/components/

Admin panel tab components. Only StatusTab is eagerly loaded; all others use `next/dynamic` for code splitting. Each tab receives `apiCall` as a prop for authenticated API requests. Organized into 8 main tabs with sub-tabs.

| File | Purpose | Key Exports |
|------|---------|-------------|
| index.tsx | Tab type definitions, tab configuration, shared type exports | `MainTab`, `TabConfig`, component type interfaces |
| shared.tsx | Shared admin UI primitives (SettingTip tooltips, SelectField) | `SettingTip`, `SelectField` |
| AdminStatusBar.tsx | Persistent top bar: model name, VRAM, sessions, benchmark/ingest indicators (10s poll) | `AdminStatusBar` |
| SubTabLayout.tsx | Horizontal sub-tab navigation layout | `SubTabLayout` |
| StatusTab.tsx | System health dashboard: 6 service statuses, hardware profile | `StatusTab` |
| ConfigTab.tsx | RuntimeConfig editor with hardware-aware defaults, KV cache, model params | `ConfigTab` |
| ModelsTab.tsx | Ollama model management: list, pull, assign roles | `ModelsTab` |
| LogsTab.tsx | Live log viewer with level filtering | `LogsTab` |
| SessionsTab.tsx | Active session browser with details | `SessionsTab` |
| KnowledgeTab.tsx | Parent wrapper for knowledge sub-tabs | `KnowledgeTab` |
| KnowledgeDocumentsTab.tsx | Knowledge file management: upload, ingest, reprocess, untracked detection | `KnowledgeDocumentsTab` |
| KnowledgeRetrievalTab.tsx | Search test interface, reranker settings editor (all RAG params) | `KnowledgeRetrievalTab` |
| KnowledgeCollectionsTab.tsx | Qdrant collection management: list, purge, rebuild | `KnowledgeCollectionsTab` |
| GraphTab.tsx | Neo4j graph exploration (entities, viewer, relationships, Cypher console) | `GraphTab` |
| ComplianceTab.tsx | Compliance guideline management and exceedance records | `ComplianceTab` |
| ExceedeeDBTab.tsx | Exceedance database browser (domain-specific) | `ExceedeeDBTab` |
| PersonalityTab.tsx | Phii behavior management: stats, corrections, patterns, debug | `PersonalityTab` |
| TesterTab.tsx | Extraction tester: upload PDF, view results | `TesterTab` |
| TrainingTab.tsx | Fine-tuning pipeline UI: parse, generate, filter, format, evaluate, deploy | `TrainingTab` |
| BenchmarkTab.tsx | Benchmark execution, results, charts, corpus upload, discuss chat | `BenchmarkTab` |
| LexiconTab.tsx | Domain lexicon JSON editor with save/reload | `LexiconTab` |

## Subdirectory

| Directory | Purpose |
|-----------|---------|
| graph/ | Neo4j visualization sub-components (GraphStats, EntityBrowser, GraphViewer, RelationshipExplorer, CypherConsole) |
