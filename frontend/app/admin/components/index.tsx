// Admin panel tab components
export { StatusTab } from './StatusTab';
export { ConfigTab } from './ConfigTab';
export { ModelsTab } from './ModelsTab';
export { LogsTab } from './LogsTab';
export { SessionsTab } from './SessionsTab';
export { KnowledgeTab } from './KnowledgeTab';
export { KnowledgeDocumentsTab } from './KnowledgeDocumentsTab';
export { KnowledgeRetrievalTab } from './KnowledgeRetrievalTab';
export { GraphTab } from './GraphTab';
export { KnowledgeCollectionsTab } from './KnowledgeCollectionsTab';
export { ComplianceTab } from './ComplianceTab';
export { ExceedeeDBTab } from './ExceedeeDBTab';
export { PersonalityTab } from './PersonalityTab';
export { TesterTab } from './TesterTab';
export { TrainingTab } from './TrainingTab';
export { BenchmarkTab } from './BenchmarkTab';
export { LexiconTab } from './LexiconTab';
export { ToolsTab } from './ToolsTab';
export type { AdminToolsData, AdminToolInfo, CustomToolInfo, ToolScaffoldPayload } from './ToolsTab';
export { SubTabLayout } from './SubTabLayout';
export { AdminStatusBar } from './AdminStatusBar';
export { SettingTip } from './shared';

// Types used by multiple components
export interface ServiceHealth {
  status: string;
  latency_ms?: number;
  error?: string;
  mode?: string;
  connected?: boolean;
}

export interface SystemMetrics {
  cpu_percent: number | null;
  cpu_temperature_c: number | null;
  ram_used_gb: number | null;
  ram_percent: number | null;
  disk_used_gb: number | null;
  disk_total_gb: number | null;
  disk_percent: number | null;
  drive_temperature_c: number | null;
  gpu_utilization_pct: number | null;
  gpu_vram_used_mb: number | null;
  gpu_temperature_c: number | null;
  gpu_power_w: number | null;
  gpu_power_limit_w: number | null;
  specs: {
    cpu_model: string;
    cpu_cores: number;
    ram_total_gb: number;
    gpu_name: string;
    gpu_vram_total_mb: number;
  };
}

export interface SystemStatus {
  timestamp: string;
  llm: {
    status: string;
    models?: string[];
    error?: string;
  };
  storage: {
    uploads_count: number;
    uploads_mb: number;
    reports_count: number;
    reports_mb: number;
    total_mb: number;
    note?: string;
  };
  sessions: {
    active_files?: number;
  };
  config: Record<string, unknown>;
  rag: {
    knowledge_chunks: number;
    topics?: string[];
  };
  services?: Record<string, ServiceHealth>;
  system?: SystemMetrics;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ConfigValues = Record<string, any>;

export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  module: string;
}

export interface Session {
  file_id: string;
  filename: string;
  type: string;
  rag_chunks: number;
}

export interface KnowledgeStats {
  total_chunks: number;
  total_files: number;
  topics: Record<string, { files: number; chunks: number }>;
  manifest: {
    tracked_files: number;
    stale_entries: number;
    last_updated: string | null;
  };
  error?: string;
}

export interface TopicFile {
  filename: string;
  path: string;
  chunks: number;
  size_mb: number;
  ingested_at: string;
  file_hash: string;
  exists: boolean;
  status: 'ingested' | 'untracked';
}

export interface UntrackedFile {
  filename: string;
  path: string;
  size_mb: number;
  status: 'untracked';
}

export interface TopicWithStatus {
  name: string;
  enabled: boolean;
}

export interface RerankerSettings {
  reranker_enabled: boolean;
  reranker_candidates: number;
  reranker_batch_size: number;
  router_use_semantic: boolean;
  rag_top_k: number;
  rag_min_score: number;
  rag_min_final_score: number;
  rag_diversity_enabled: boolean;
  rag_diversity_lambda: number;
  rag_max_per_source: number;
  // BM25
  bm25_enabled: boolean;
  bm25_weight: number;
  // Query expansion
  query_expansion_enabled: boolean;
  // Domain boost
  domain_boost_enabled: boolean;
  domain_boost_factor: number;
  // HyDE
  hyde_enabled: boolean;
  hyde_model: string;
  // CRAG
  crag_enabled: boolean;
  crag_min_confidence: number;
  crag_web_search_on_low: boolean;
  // RAPTOR hierarchical search
  raptor_enabled: boolean;
  raptor_max_levels: number;
  raptor_cluster_size: number;
  raptor_summary_model: string;
  raptor_retrieval_strategy: string;
  // Vision Ingest
  vision_ingest_enabled: boolean;
  // GraphRAG
  graph_rag_enabled: boolean;
  graph_rag_auto: boolean;
  graph_rag_weight: number;
  graph_max_hops: number;
  // Community/Global Search
  community_detection_enabled: boolean;
  global_search_enabled: boolean;
  global_search_top_k: number;
  community_level_default: string;
  community_summary_model: string;
  // Retrieval profiles
  retrieval_profile?: string;
  manual_overrides?: Record<string, boolean>;
  has_manual_overrides?: boolean;
}

export interface ModelInfo {
  name: string;
  size_gb: number;
  modified?: string;
  family?: string;
  assigned_to: string[];
}

export interface ModelsData {
  models: ModelInfo[];
  config_assignments: Record<string, string>;
  total_size_gb: number;
  models_dir?: {
    path: string;
    exists: boolean;
    readable: boolean;
    writable: boolean;
    error?: string | null;
  };
}

export interface PullStatus {
  status: string;
  progress_percent?: number;
  completed_gb?: number;
  total_gb?: number;
  done?: boolean;
  error?: string;
}

export interface SearchResult {
  content: string;
  source: string;
  page?: number;
  topic: string;
  normalized_score: number;
  raw_score?: number;
  rerank_score?: number;
}

export interface SearchTestResult {
  success: boolean;
  query: string;
  topics_searched: string[];
  num_results: number;
  results: SearchResult[];
  routing?: {
    query: string;
    routed_to: string[];
  };
  processing_ms: number;
}

export interface GuidelineFile {
  name: string;
  filename: string;
  path: string;
  size_kb: number;
  entries: number;
  parameters: number;
  soil_types: string[];
  land_uses: string[];
  error?: string;
}

export interface GuidelineStats {
  guidelines_dir: string;
  dir_exists: boolean;
  files: GuidelineFile[];
  total_entries: number;
}

export interface GuidelineEntry {
  parameter: string;
  soil_type: string;
  land_use: string;
  value: number;
  table?: string;
}

export interface GuidelineSearchResult {
  query: string;
  total_matches: number;
  unique_parameters: number;
  results: GuidelineEntry[];
  grouped: Record<string, Array<{
    table: string;
    soil_type: string;
    land_use: string;
    value: number;
  }>>;
}

export interface GuidelineComparisonResult {
  success: boolean;
  comparison?: {
    changed: Array<{
      parameter: string;
      soil_type: string;
      land_use: string;
      old_value: number;
      new_value: number;
      pct_change: number;
    }>;
    added: Array<{
      parameter: string;
      soil_type: string;
      land_use: string;
      limit_value: number;
    }>;
    removed: Array<{
      parameter: string;
      soil_type: string;
      land_use: string;
      limit_value: number;
    }>;
    unchanged_count: number;
    summary: string;
  };
  extracted_count?: number;
  summary?: string;
  error?: string;
}

export interface ExceedeeDBStats {
  total_records: number;
  pending_count: number;
  approved_count: number;
  rejected_count: number;
  last_updated: string | null;
  last_backup: string | null;
  backup_count: number;
  projects: string[];
}

export interface ExceedeeRecord {
  record_id: string;
  status: 'pending' | 'approved' | 'rejected';
  created_at: string;
  approved_at: string | null;
  rejected_at: string | null;
  rejection_reason: string | null;
  project_number: string;
  project_name: string | null;
  client_name: string | null;
  location_name: string;
  sample_id: string;
  sample_date: string | null;
  depth: string | null;
  lab_report_filename: string;
  exceedance_count: number;
  soil_type: string | null;
  land_use: string | null;
}

export interface ReprocessJobStatus {
  job_id: string;
  status: 'starting' | 'running' | 'completed' | 'failed' | 'cancelling' | 'cancelled';
  mode: string;
  progress: number;
  total_files: number;
  processed_files: number;
  successful: number;
  failed: number;
  current_file: string | null;
  current_topic: string | null;
  topics: string[];
  elapsed_seconds: number;
  remaining_seconds: number;
  extraction_stats: {
    text_pages: number;
    vision_pages: number;
  };
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  recent_results: Array<{
    file: string;
    topic: string;
    success: boolean;
    chunks: number;
    extractor?: string;
    pages_by_extractor?: Record<string, number>;
    error?: string;
  }>;
  // Phase tracking for full pipeline rebuild
  phase?: 'purge' | 'ingest' | 'raptor' | 'graph' | 'complete';
  phase_progress?: number;
  phases?: {
    purge: { status: string; deleted_vectors: number; deleted_bm25: number; deleted_neo4j: number };
    ingest: { status: string; files: number; chunks: number };
    raptor: { status: string; nodes: number; levels: Record<string, number> };
    graph: { status: string; entities: number; relationships: number };
  };
}

export interface ReprocessOptions {
  purge_qdrant: boolean;
  clear_bm25: boolean;
  clear_neo4j: boolean;
  build_raptor: boolean;
  build_graph: boolean;
  skip_vision: boolean;
}

export interface AvailableDomain {
  name: string;
  display_name: string;
  description: string;
  version: string;
  tools_count: number;
  routes_count: number;
  admin_visible: boolean;
}
