// Benchmark tab types, constants, and utility functions

export interface BenchmarkTabProps {
  apiCall: (endpoint: string, options?: RequestInit) => Promise<Record<string, unknown>>;
  setMessage: (msg: { type: 'success' | 'error'; text: string } | null) => void;
}

export interface ChartInfo {
  name: string;
  filename: string;
  url: string;
}

export interface PhaseRanking {
  rank: number;
  variant: string;
  composite: number;
  delta?: number;
}

export interface PhaseData {
  duration_s: number;
  n_variants: number;
  winner: string | null;
  ranking: PhaseRanking[];
}

export interface BenchmarkResults {
  timestamp?: string;
  phases: Record<string, PhaseData>;
  overall_winner?: {
    phase: string;
    variant: string;
    composite: number;
  };
  overall_ranking?: Array<{
    phase: string;
    variant: string;
    composite: number;
  }>;
}

export interface HistoryRun {
  id: string;
  timestamp: string;
  timestamp_str?: string;
  has_results: boolean;
  has_report: boolean;
  phases?: string[];
  overall_winner?: {
    phase: string;
    variant: string;
    composite: number;
  };
  chart_count?: number;
}

export interface JobStatus {
  job_id?: string;
  status: string;
  current_phase?: string | null;
  phases?: string[];
  phases_completed?: number;
  phases_total?: number;
  progress_pct?: number;
  estimated_remaining_s?: number | null;
  phase_durations?: Record<string, number>;
  results_so_far?: Record<string, PhaseData>;
  winners?: Record<string, string>;
  error?: string | null;
  auto_applied?: { applied: boolean; updated?: string[]; ignored?: string[]; auto_profile_updated?: Record<string, unknown> };
  started_at?: string;
  completed_at?: string;
  message?: string;
}

export function formatTimeRemaining(seconds: number): string {
  if (seconds < 60) return `~${Math.round(seconds)}s remaining`;
  const mins = Math.round(seconds / 60);
  return `~${mins} min remaining`;
}

export const PHASE_LABELS: Record<string, string> = {
  layer0_chunking: 'Chunking Sweep',
  layer1_retrieval: 'Retrieval Config',
  layer2_params: 'Parameter Tuning',
  layer_embed: 'Embedding Shootout',
  layer_rerank: 'Reranker Shootout',
  layer_cross: 'Cross-Model Sweep',
  layer_llm: 'Chat LLM Comparison',
  layer3_answers: 'Answer Generation',
  layer4_judge: 'LLM-as-Judge',
  layer5_analysis: 'Analysis & Charts',
  layer6_failures: 'Failure Analysis',
  layer_live: 'Live Pipeline Test',
  layer_ceiling: 'Frontier vs Local LLM Ceiling',
};

// All layers in run order
export const ALL_LAYER_KEYS = [
  'layer0_chunking', 'layer1_retrieval', 'layer2_params',
  'layer_embed', 'layer_rerank', 'layer_cross', 'layer_llm',
  'layer3_answers', 'layer4_judge', 'layer5_analysis', 'layer6_failures',
  'layer_live', 'layer_ceiling',
];

// Human-readable labels for RuntimeConfig winner keys
export const WINNER_LABELS: Record<string, string> = {
  chunk_size: 'Chunk Size',
  chunk_overlap: 'Chunk Overlap',
  bm25_weight: 'BM25 Weight',
  rag_diversity_lambda: 'Diversity Lambda',
  rag_top_k: 'Top K',
  reranker_candidates: 'Reranker Candidates',
  rag_min_score: 'Min Score',
  domain_boost_factor: 'Domain Boost',
  reranker_enabled: 'Reranker',
  bm25_enabled: 'BM25',
  query_expansion_enabled: 'Query Expansion',
  hyde_enabled: 'HyDE',
  raptor_enabled: 'RAPTOR',
  graph_rag_enabled: 'GraphRAG',
  global_search_enabled: 'Global Search',
  rag_diversity_enabled: 'Diversity',
  crag_enabled: 'CRAG',
  vision_ingest_enabled: 'Vision Ingest',
};

export function formatWinnerValue(value: unknown): string {
  if (typeof value === 'boolean') return value ? 'On' : 'Off';
  if (typeof value === 'number') return String(value);
  if (value === '' || value === null || value === undefined) return '-';
  return String(value);
}
