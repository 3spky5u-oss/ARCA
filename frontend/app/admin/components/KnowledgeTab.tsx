'use client';

import React from 'react';
import { SettingTip } from './shared';

interface KnowledgeStats {
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

interface TopicFile {
  filename: string;
  path: string;
  chunks: number;
  size_mb: number;
  ingested_at: string;
  file_hash: string;
  exists: boolean;
  status: 'ingested' | 'untracked';
}

interface UntrackedFile {
  filename: string;
  path: string;
  size_mb: number;
  status: 'untracked';
}

interface TopicWithStatus {
  name: string;
  enabled: boolean;
}

interface RerankerSettings {
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
}

interface SearchResult {
  content: string;
  source: string;
  page?: number;
  topic: string;
  normalized_score: number;
  raw_score?: number;
  rerank_score?: number;
}

interface SearchTestResult {
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

interface ReprocessJobStatus {
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

interface ReprocessOptions {
  purge_qdrant: boolean;
  clear_bm25: boolean;
  clear_neo4j: boolean;
  build_raptor: boolean;
  build_graph: boolean;
}

interface KnowledgeTabProps {
  knowledgeStats: KnowledgeStats | null;
  expandedTopics: Set<string>;
  topicFiles: Record<string, TopicFile[]>;
  topicUntracked: Record<string, UntrackedFile[]>;
  availableTopics: string[];
  topicsWithStatus: TopicWithStatus[];
  rerankerSettings: RerankerSettings | null;
  searchTestResult: SearchTestResult | null;
  fileExtractorSelection: Record<string, string>;
  processingFiles: Record<string, string>;
  // Ingest state
  ingestFile: File | null;
  ingestTopic: string;
  newTopicName: string;
  showAdvancedIngest: boolean;
  showRetrievalSettings: boolean;
  ingestChunkSize: number;
  ingestChunkOverlap: number;
  // Search state
  searchQuery: string;
  searchTopics: string[];
  searchTopK: number;
  // Reprocess state
  reprocessStatus: ReprocessJobStatus | null;
  showReprocessConfirm: boolean;
  reprocessMode: 'knowledge_base' | 'session';
  reprocessOptions: ReprocessOptions;
  // Callbacks
  onToggleTopic: (topic: string) => void;
  onAutoIngest: () => void;
  onCleanupStale: () => void;
  onIngestFile: () => void;
  onCreateTopic: () => void;
  onReindexTopic: (topic: string) => void;
  onDeleteFile: (path: string, topic: string) => void;
  onIngestUntracked: (path: string, topic: string) => void;
  onReindexFile: (path: string, topic: string) => void;
  onToggleTopicEnabled: (topic: string, enabled: boolean) => void;
  onUpdateRerankerSettings: (updates: Partial<RerankerSettings>) => void;
  onSearchTest: () => void;
  onStartReprocess: () => void;
  onCancelReprocess: () => void;
  // State setters
  onIngestFileChange: (file: File | null) => void;
  onIngestTopicChange: (topic: string) => void;
  onNewTopicNameChange: (name: string) => void;
  onShowAdvancedIngestChange: (show: boolean) => void;
  onShowRetrievalSettingsChange: (show: boolean) => void;
  onIngestChunkSizeChange: (size: number) => void;
  onIngestChunkOverlapChange: (overlap: number) => void;
  onSearchQueryChange: (query: string) => void;
  onSearchTopicsChange: (topics: string[]) => void;
  onSearchTopKChange: (topK: number) => void;
  onFileExtractorSelectionChange: (selection: Record<string, string>) => void;
  onShowReprocessConfirmChange: (show: boolean) => void;
  onReprocessModeChange: (mode: 'knowledge_base' | 'session') => void;
  onReprocessOptionsChange: (options: ReprocessOptions) => void;
  loading: boolean;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (mins < 60) return `${mins}m ${secs}s`;
  const hours = Math.floor(mins / 60);
  const remMins = mins % 60;
  return `${hours}h ${remMins}m`;
}

export function KnowledgeTab({
  knowledgeStats,
  expandedTopics,
  topicFiles,
  topicUntracked,
  availableTopics,
  topicsWithStatus,
  rerankerSettings,
  searchTestResult,
  fileExtractorSelection,
  processingFiles,
  ingestFile,
  ingestTopic,
  newTopicName,
  showAdvancedIngest,
  showRetrievalSettings,
  ingestChunkSize,
  ingestChunkOverlap,
  searchQuery,
  searchTopics,
  searchTopK,
  reprocessStatus,
  showReprocessConfirm,
  reprocessMode,
  reprocessOptions,
  onToggleTopic,
  onAutoIngest,
  onCleanupStale,
  onIngestFile,
  onCreateTopic,
  onReindexTopic,
  onDeleteFile,
  onIngestUntracked,
  onReindexFile,
  onToggleTopicEnabled,
  onUpdateRerankerSettings,
  onSearchTest,
  onStartReprocess,
  onCancelReprocess,
  onIngestFileChange,
  onIngestTopicChange,
  onNewTopicNameChange,
  onShowAdvancedIngestChange,
  onShowRetrievalSettingsChange,
  onIngestChunkSizeChange,
  onIngestChunkOverlapChange,
  onSearchQueryChange,
  onSearchTopicsChange,
  onSearchTopKChange,
  onFileExtractorSelectionChange,
  onShowReprocessConfirmChange,
  onReprocessModeChange,
  onReprocessOptionsChange,
  loading,
}: KnowledgeTabProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">Knowledge Base Management</h2>
        <div className="flex gap-2">
          {knowledgeStats?.manifest.stale_entries ? (
            <button
              onClick={onCleanupStale}
              className="px-4 py-2 bg-amber-600 hover:bg-amber-700 rounded-xl text-sm transition-colors"
            >
              Cleanup Stale ({knowledgeStats.manifest.stale_entries})
            </button>
          ) : null}
          <button
            onClick={onAutoIngest}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm transition-colors"
          >
            Auto-Ingest
          </button>
        </div>
      </div>

      {/* Overview Stats */}
      {knowledgeStats && (
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-blue-400">{knowledgeStats.total_chunks.toLocaleString()}</p>
            <p className="text-xs text-gray-400 mt-1">Chunks</p>
          </div>
          <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-green-400">{knowledgeStats.total_files}</p>
            <p className="text-xs text-gray-400 mt-1">Files</p>
          </div>
          <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-purple-400">{Object.keys(knowledgeStats.topics).length}</p>
            <p className="text-xs text-gray-400 mt-1">Topics</p>
          </div>
          <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
            <p className={`text-2xl font-bold ${knowledgeStats.manifest.stale_entries > 0 ? 'text-amber-400' : 'text-gray-400'}`}>
              {knowledgeStats.manifest.stale_entries}
            </p>
            <p className="text-xs text-gray-400 mt-1">Stale</p>
          </div>
        </div>
      )}

      {/* Reprocess Section */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-medium flex items-center gap-2">
              <svg className="w-4 h-4 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Re-process Knowledge Base
            </h3>
            <p className="text-xs text-gray-500 mt-1">
              Re-extract all documents using hybrid vision/text pipeline for maximum quality
            </p>
          </div>
          {!reprocessStatus || reprocessStatus.status === 'completed' || reprocessStatus.status === 'failed' ? (
            <button
              onClick={() => onShowReprocessConfirmChange(true)}
              className="px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded-xl text-sm transition-colors"
              disabled={loading}
            >
              Re-process All
            </button>
          ) : (
            <button
              onClick={onCancelReprocess}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-xl text-sm transition-colors"
            >
              Cancel
            </button>
          )}
        </div>

        {/* Reprocess Progress */}
        {reprocessStatus && (reprocessStatus.status === 'running' || reprocessStatus.status === 'starting') && (
          <div className="space-y-3">
            {/* Phase Indicators */}
            <div className="flex items-center justify-between gap-2 mb-2">
              {['purge', 'ingest', 'raptor', 'graph'].map((phaseName, idx) => {
                const phase = reprocessStatus.phases?.[phaseName as keyof typeof reprocessStatus.phases];
                const isCurrent = reprocessStatus.phase === phaseName;
                const isComplete = phase?.status === 'completed';
                const isSkipped = phase?.status === 'skipped';

                return (
                  <div key={phaseName} className="flex-1">
                    <div className={`h-1.5 rounded-full transition-all ${
                      isComplete ? 'bg-green-500' :
                      isCurrent ? 'bg-orange-500 animate-pulse' :
                      isSkipped ? 'bg-gray-600' :
                      'bg-[#1a1a1a]'
                    }`} />
                    <span className={`text-[10px] mt-1 block text-center capitalize ${
                      isCurrent ? 'text-orange-400 font-medium' :
                      isComplete ? 'text-green-400' :
                      isSkipped ? 'text-gray-500' :
                      'text-gray-500'
                    }`}>
                      {phaseName}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Overall Progress Bar */}
            <div className="relative">
              <div className="h-3 bg-[#1a1a1a] rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-orange-500 to-orange-400 transition-all duration-500"
                  style={{ width: `${reprocessStatus.phase_progress || reprocessStatus.progress}%` }}
                />
              </div>
              <span className="absolute right-0 top-4 text-xs text-gray-400">
                {reprocessStatus.phase_progress || reprocessStatus.progress}%
              </span>
            </div>

            {/* Phase-specific details */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-400">Phase: <span className="text-orange-400 capitalize">{reprocessStatus.phase || 'starting'}</span></p>
                <p className="text-white truncate">
                  {reprocessStatus.current_file || 'Starting...'}
                </p>
                {reprocessStatus.current_topic && (
                  <p className="text-xs text-gray-500">Topic: {reprocessStatus.current_topic}</p>
                )}
              </div>
              <div>
                {reprocessStatus.phase === 'ingest' && (
                  <>
                    <p className="text-gray-400">Files:</p>
                    <p className="text-white">
                      {reprocessStatus.processed_files} / {reprocessStatus.total_files}
                    </p>
                    <p className="text-xs text-gray-500">
                      Vision: {reprocessStatus.extraction_stats.vision_pages} pages
                    </p>
                  </>
                )}
                {reprocessStatus.phase === 'raptor' && reprocessStatus.phases?.raptor && (
                  <>
                    <p className="text-gray-400">RAPTOR Nodes:</p>
                    <p className="text-white">{reprocessStatus.phases.raptor.nodes || 0}</p>
                    <p className="text-xs text-gray-500">
                      Levels: {Object.keys(reprocessStatus.phases.raptor.levels || {}).length}
                    </p>
                  </>
                )}
                {reprocessStatus.phase === 'graph' && reprocessStatus.phases?.graph && (
                  <>
                    <p className="text-gray-400">Graph:</p>
                    <p className="text-white">{reprocessStatus.phases.graph.entities || 0} entities</p>
                    <p className="text-xs text-gray-500">
                      {reprocessStatus.phases.graph.relationships || 0} relationships
                    </p>
                  </>
                )}
                {reprocessStatus.phase === 'purge' && (
                  <>
                    <p className="text-gray-400">Clearing:</p>
                    <p className="text-white">Old data...</p>
                  </>
                )}
              </div>
            </div>

            <div className="flex justify-between text-xs text-gray-400">
              <span>Elapsed: {formatDuration(reprocessStatus.elapsed_seconds)}</span>
              {reprocessStatus.remaining_seconds > 0 && (
                <span>Remaining: ~{formatDuration(reprocessStatus.remaining_seconds)}</span>
              )}
            </div>
          </div>
        )}

        {/* Completed/Failed Status */}
        {reprocessStatus && reprocessStatus.status === 'completed' && (
          <div className="bg-green-900/30 border border-green-700 rounded-lg p-3 mt-3">
            <p className="text-sm text-green-400 font-medium">Full Pipeline Rebuild Complete</p>
            <div className="grid grid-cols-4 gap-2 mt-2 text-xs">
              <div>
                <span className="text-gray-400">Files:</span>
                <span className="text-white ml-1">{reprocessStatus.successful}</span>
              </div>
              <div>
                <span className="text-gray-400">Chunks:</span>
                <span className="text-white ml-1">{reprocessStatus.phases?.ingest?.chunks || 0}</span>
              </div>
              <div>
                <span className="text-gray-400">RAPTOR:</span>
                <span className="text-white ml-1">{reprocessStatus.phases?.raptor?.nodes || 0} nodes</span>
              </div>
              <div>
                <span className="text-gray-400">Graph:</span>
                <span className="text-white ml-1">{reprocessStatus.phases?.graph?.entities || 0} entities</span>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Text: {reprocessStatus.extraction_stats.text_pages} pages |
              Vision: {reprocessStatus.extraction_stats.vision_pages} pages
            </p>
          </div>
        )}

        {reprocessStatus && reprocessStatus.status === 'failed' && (
          <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 mt-3">
            <p className="text-sm text-red-400 font-medium">Reprocessing Failed</p>
            <p className="text-xs text-gray-400 mt-1">{reprocessStatus.error}</p>
          </div>
        )}
      </div>

      {/* Reprocess Confirmation Modal */}
      {showReprocessConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-[#2a2a2a] rounded-lg p-6 max-w-lg w-full mx-4 shadow-xl">
            <h3 className="text-lg font-medium mb-4">Full RAG Pipeline Rebuild</h3>
            <div className="space-y-4 text-sm text-gray-300">
              <p>This will rebuild the complete RAG pipeline for {knowledgeStats?.total_files || 0} documents.</p>

              {/* Purge Options */}
              <div className="bg-[#1a1a1a] rounded-lg p-3">
                <h4 className="text-xs font-medium text-red-400 mb-2 flex items-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  Purge Old Data
                </h4>
                <div className="space-y-2">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={reprocessOptions.purge_qdrant}
                      onChange={(e) => onReprocessOptionsChange({ ...reprocessOptions, purge_qdrant: e.target.checked })}
                      className="w-4 h-4 rounded bg-[#2a2a2a] border-[#3a3a3a]"
                    />
                    <span className="text-xs">Purge Qdrant vectors</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={reprocessOptions.clear_bm25}
                      onChange={(e) => onReprocessOptionsChange({ ...reprocessOptions, clear_bm25: e.target.checked })}
                      className="w-4 h-4 rounded bg-[#2a2a2a] border-[#3a3a3a]"
                    />
                    <span className="text-xs">Clear BM25 indices</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={reprocessOptions.clear_neo4j}
                      onChange={(e) => onReprocessOptionsChange({ ...reprocessOptions, clear_neo4j: e.target.checked })}
                      className="w-4 h-4 rounded bg-[#2a2a2a] border-[#3a3a3a]"
                    />
                    <span className="text-xs">Clear Neo4j GraphRAG</span>
                  </label>
                </div>
              </div>

              {/* Build Options */}
              <div className="bg-[#1a1a1a] rounded-lg p-3">
                <h4 className="text-xs font-medium text-green-400 mb-2 flex items-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                  Build Pipeline
                </h4>
                <div className="space-y-2">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={reprocessOptions.build_raptor}
                      onChange={(e) => onReprocessOptionsChange({ ...reprocessOptions, build_raptor: e.target.checked })}
                      className="w-4 h-4 rounded bg-[#2a2a2a] border-[#3a3a3a]"
                    />
                    <span className="text-xs">Build RAPTOR trees (hierarchical summaries)</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={reprocessOptions.build_graph}
                      onChange={(e) => onReprocessOptionsChange({ ...reprocessOptions, build_graph: e.target.checked })}
                      className="w-4 h-4 rounded bg-[#2a2a2a] border-[#3a3a3a]"
                    />
                    <span className="text-xs">Build GraphRAG entities (knowledge graph)</span>
                  </label>
                </div>
              </div>

              {/* Mode Selection */}
              <div>
                <label className="block text-xs text-gray-400 mb-2">Extraction Mode:</label>
                <div className="flex gap-2">
                  <button
                    onClick={() => onReprocessModeChange('knowledge_base')}
                    className={`flex-1 px-3 py-2 rounded-xl text-xs transition-colors ${
                      reprocessMode === 'knowledge_base'
                        ? 'bg-orange-600 text-white'
                        : 'bg-[#1a1a1a] text-gray-400 hover:bg-[#333]'
                    }`}
                  >
                    Knowledge Base
                    <br />
                    <span className="text-[10px] opacity-70">(Vision + Text)</span>
                  </button>
                  <button
                    onClick={() => onReprocessModeChange('session')}
                    className={`flex-1 px-3 py-2 rounded-xl text-xs transition-colors ${
                      reprocessMode === 'session'
                        ? 'bg-blue-600 text-white'
                        : 'bg-[#1a1a1a] text-gray-400 hover:bg-[#333]'
                    }`}
                  >
                    Session
                    <br />
                    <span className="text-[10px] opacity-70">(Fast Text Only)</span>
                  </button>
                </div>
              </div>

              <div className="bg-amber-900/30 border border-amber-700 rounded-lg p-3">
                <p className="text-amber-400 text-xs">
                  This process may take several hours for large knowledge bases.
                  All selected phases will run sequentially.
                </p>
              </div>
            </div>
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => onShowReprocessConfirmChange(false)}
                className="px-4 py-2 bg-[#1a1a1a] hover:bg-[#333] rounded-xl text-sm transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={onStartReprocess}
                className="px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded-xl text-sm transition-colors"
              >
                Start Full Rebuild
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Topic Toggles & Reranker Settings */}
      <div className="grid grid-cols-2 gap-4">
        {/* Topic Toggles */}
        <div className="bg-[#2a2a2a] rounded-lg p-4">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
            Active Knowledge Bases
          </h3>
          <p className="text-xs text-gray-500 mb-3">Toggle topics to enable/disable them for chat search</p>
          <div className="space-y-2">
            {topicsWithStatus.map((topic) => (
              <label key={topic.name} className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-lg hover:bg-[#222] cursor-pointer">
                <span className="text-sm capitalize">{topic.name}</span>
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={topic.enabled}
                    onChange={(e) => onToggleTopicEnabled(topic.name, e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-green-500"></div>
                </div>
              </label>
            ))}
            {topicsWithStatus.length === 0 && (
              <p className="text-xs text-gray-500 py-2">No topics available</p>
            )}
          </div>
        </div>

        {/* Reranker Settings */}
        <div className="bg-[#2a2a2a] rounded-lg p-4">
          <button
            onClick={() => onShowRetrievalSettingsChange(!showRetrievalSettings)}
            className="w-full text-left"
          >
            <h3 className="text-sm font-medium flex items-center gap-2">
              <span className="text-gray-400">{showRetrievalSettings ? '▼' : '▶'}</span>
              <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              Retrieval Settings
              <span className="text-xs text-gray-500 ml-2">(reranking, diversity, thresholds)</span>
            </h3>
          </button>
          {showRetrievalSettings && rerankerSettings && (
            <div className="space-y-3 mt-4">
              {/* Reranker Toggle */}
              <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-lg hover:bg-[#222] cursor-pointer">
                <div>
                  <span className="text-sm">
                    Cross-Encoder Reranking
                    <SettingTip tip="Cross-encoder reranking improves relevance but adds ~200-500ms latency" />
                  </span>
                  <p className="text-xs text-gray-500">More accurate but slower</p>
                </div>
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={rerankerSettings.reranker_enabled}
                    onChange={(e) => onUpdateRerankerSettings({ reranker_enabled: e.target.checked })}
                    className="sr-only peer"
                  />
                  <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-500"></div>
                </div>
              </label>

              {/* Semantic Router Toggle */}
              <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-lg hover:bg-[#222] cursor-pointer">
                <div>
                  <span className="text-sm">Semantic Topic Routing</span>
                  <p className="text-xs text-gray-500">Use embeddings for topic selection</p>
                </div>
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={rerankerSettings.router_use_semantic}
                    onChange={(e) => onUpdateRerankerSettings({ router_use_semantic: e.target.checked })}
                    className="sr-only peer"
                  />
                  <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-500"></div>
                </div>
              </label>

              {/* Numeric Settings */}
              <div className="grid grid-cols-2 gap-2 pt-2">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">
                    Top K Results
                    <SettingTip tip="Number of results returned to the LLM" />
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={rerankerSettings.rag_top_k}
                    onChange={(e) => onUpdateRerankerSettings({ rag_top_k: parseInt(e.target.value) || 5 })}
                    className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">
                    Candidates
                    <SettingTip tip="How many results to score before selecting Top K. Higher = better quality, more latency" />
                  </label>
                  <input
                    type="number"
                    min="5"
                    max="50"
                    value={rerankerSettings.reranker_candidates}
                    onChange={(e) => onUpdateRerankerSettings({ reranker_candidates: parseInt(e.target.value) || 20 })}
                    className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
                  />
                </div>
              </div>

              {/* Diversity Settings */}
              <div className="border-t border-[#3a3a3a] pt-3 mt-3">
                <h4 className="text-xs font-medium text-gray-400 mb-2 flex items-center gap-1">
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                  </svg>
                  Source Diversity
                </h4>

                <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-lg hover:bg-[#222] cursor-pointer mb-2">
                  <div>
                    <span className="text-sm">
                      MMR Diversity
                      <SettingTip tip="Maximal Marginal Relevance ensures results come from different sources" />
                    </span>
                    <p className="text-xs text-gray-500">Balance relevance with source variety</p>
                  </div>
                  <div className="relative">
                    <input
                      type="checkbox"
                      checked={rerankerSettings.rag_diversity_enabled}
                      onChange={(e) => onUpdateRerankerSettings({ rag_diversity_enabled: e.target.checked })}
                      className="sr-only peer"
                    />
                    <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-emerald-500"></div>
                  </div>
                </label>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Diversity λ ({rerankerSettings.rag_diversity_lambda.toFixed(1)})
                      <SettingTip tip="Balance: 0.3 = max diversity, 0.9 = max relevance" />
                    </label>
                    <input
                      type="range"
                      min="0.3"
                      max="0.9"
                      step="0.1"
                      value={rerankerSettings.rag_diversity_lambda}
                      onChange={(e) => onUpdateRerankerSettings({ rag_diversity_lambda: parseFloat(e.target.value) })}
                      className="w-full h-2 bg-[#1a1a1a] rounded-lg appearance-none cursor-pointer accent-emerald-500"
                      disabled={!rerankerSettings.rag_diversity_enabled}
                    />
                    <div className="flex justify-between text-[10px] text-gray-500 mt-0.5">
                      <span>Diverse</span>
                      <span>Relevant</span>
                    </div>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Max/Source
                      <SettingTip tip="Limit results per document to ensure variety" />
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="5"
                      value={rerankerSettings.rag_max_per_source}
                      onChange={(e) => onUpdateRerankerSettings({ rag_max_per_source: parseInt(e.target.value) || 2 })}
                      className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
                      disabled={!rerankerSettings.rag_diversity_enabled}
                    />
                  </div>
                </div>
              </div>

              {/* Quality Thresholds */}
              <div className="border-t border-[#3a3a3a] pt-3 mt-3">
                <h4 className="text-xs font-medium text-gray-400 mb-2">Quality Thresholds</h4>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Min Pre-filter
                      <SettingTip tip="Minimum similarity score (0-1) before reranking" />
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="0.5"
                      step="0.05"
                      value={rerankerSettings.rag_min_score}
                      onChange={(e) => onUpdateRerankerSettings({ rag_min_score: parseFloat(e.target.value) || 0.15 })}
                      className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Min Final
                      <SettingTip tip="Minimum score after reranking to include in results" />
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="0.5"
                      step="0.05"
                      value={rerankerSettings.rag_min_final_score}
                      onChange={(e) => onUpdateRerankerSettings({ rag_min_final_score: parseFloat(e.target.value) || 0.20 })}
                      className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Advanced RAG Settings */}
      {rerankerSettings && (
        <div className="grid grid-cols-3 gap-4">
          {/* RAPTOR Hierarchical Search */}
          <div className="bg-[#2a2a2a] rounded-lg p-4 border-l-4 border-green-500">
            <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
              <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
              </svg>
              RAPTOR Hierarchical Search
            </h3>
            <p className="text-xs text-gray-500 mb-3">Recursive abstractive processing for tree-organized retrieval</p>

            <div className="space-y-3">
              {/* Enable Toggle */}
              <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-lg hover:bg-[#222] cursor-pointer">
                <div>
                  <span className="text-sm">Enable RAPTOR</span>
                  <p className="text-xs text-gray-500">Hierarchical summarization</p>
                </div>
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={rerankerSettings.raptor_enabled ?? false}
                    onChange={(e) => onUpdateRerankerSettings({ raptor_enabled: e.target.checked })}
                    className="sr-only peer"
                  />
                  <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-green-500"></div>
                </div>
              </label>

              {/* Retrieval Strategy */}
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Retrieval Strategy
                  <SettingTip tip="Collapsed: search all levels at once. Tree traversal: start from top summaries" />
                </label>
                <select
                  value={rerankerSettings.raptor_retrieval_strategy ?? 'collapsed'}
                  onChange={(e) => onUpdateRerankerSettings({ raptor_retrieval_strategy: e.target.value })}
                  disabled={!rerankerSettings.raptor_enabled}
                  className="w-full px-2 py-1.5 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm disabled:opacity-50"
                >
                  <option value="collapsed">Collapsed (faster)</option>
                  <option value="tree_traversal">Tree Traversal (deeper)</option>
                </select>
              </div>

              {/* Max Levels & Cluster Size */}
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Max Levels</label>
                  <input
                    type="number"
                    min="1"
                    max="5"
                    value={rerankerSettings.raptor_max_levels ?? 3}
                    onChange={(e) => onUpdateRerankerSettings({ raptor_max_levels: parseInt(e.target.value) || 3 })}
                    disabled={!rerankerSettings.raptor_enabled}
                    className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm disabled:opacity-50"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Cluster Size</label>
                  <input
                    type="number"
                    min="5"
                    max="20"
                    value={rerankerSettings.raptor_cluster_size ?? 10}
                    onChange={(e) => onUpdateRerankerSettings({ raptor_cluster_size: parseInt(e.target.value) || 10 })}
                    disabled={!rerankerSettings.raptor_enabled}
                    className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm disabled:opacity-50"
                  />
                </div>
              </div>

              {/* Summary Model */}
              <div>
                <label className="block text-xs text-gray-400 mb-1">Summary Model</label>
                <input
                  type="text"
                  value={rerankerSettings.raptor_summary_model ?? 'qwen2.5:1.5b'}
                  onChange={(e) => onUpdateRerankerSettings({ raptor_summary_model: e.target.value })}
                  disabled={!rerankerSettings.raptor_enabled}
                  className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm disabled:opacity-50"
                  placeholder="qwen2.5:1.5b"
                />
              </div>
            </div>
          </div>

          {/* Knowledge Graph (GraphRAG) */}
          <div className="bg-[#2a2a2a] rounded-lg p-4 border-l-4 border-cyan-500">
            <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
              <svg className="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
              Knowledge Graph
            </h3>
            <p className="text-xs text-gray-500 mb-3">Neo4j-powered entity relationships and reasoning</p>

            <div className="space-y-3">
              {/* Enable Toggle */}
              <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-lg hover:bg-[#222] cursor-pointer">
                <div>
                  <span className="text-sm">Enable GraphRAG</span>
                  <p className="text-xs text-gray-500">Entity-based retrieval</p>
                </div>
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={rerankerSettings.graph_rag_enabled ?? false}
                    onChange={(e) => onUpdateRerankerSettings({ graph_rag_enabled: e.target.checked })}
                    className="sr-only peer"
                  />
                  <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-cyan-500"></div>
                </div>
              </label>

              {/* Graph Weight Slider */}
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Graph Weight in RRF ({(rerankerSettings.graph_rag_weight ?? 0.2).toFixed(2)})
                  <SettingTip tip="How much weight to give graph results in fusion ranking (0-0.5)" />
                </label>
                <input
                  type="range"
                  min="0"
                  max="0.5"
                  step="0.05"
                  value={rerankerSettings.graph_rag_weight ?? 0.2}
                  onChange={(e) => onUpdateRerankerSettings({ graph_rag_weight: parseFloat(e.target.value) })}
                  disabled={!rerankerSettings.graph_rag_enabled}
                  className="w-full h-2 bg-[#1a1a1a] rounded-lg appearance-none cursor-pointer accent-cyan-500 disabled:opacity-50"
                />
                <div className="flex justify-between text-[10px] text-gray-500 mt-0.5">
                  <span>Vector only</span>
                  <span>Balanced</span>
                </div>
              </div>

              {/* Max Hops */}
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Max Hops
                  <SettingTip tip="Maximum relationship traversal depth (1-3)" />
                </label>
                <input
                  type="number"
                  min="1"
                  max="3"
                  value={rerankerSettings.graph_max_hops ?? 2}
                  onChange={(e) => onUpdateRerankerSettings({ graph_max_hops: parseInt(e.target.value) || 2 })}
                  disabled={!rerankerSettings.graph_rag_enabled}
                  className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm disabled:opacity-50"
                />
              </div>

              {/* Neo4j Status - placeholder for future implementation */}
              <div className="flex items-center gap-2 p-2 bg-[#1a1a1a] rounded-lg">
                <span className={`w-2 h-2 rounded-full ${rerankerSettings.graph_rag_enabled ? 'bg-green-400' : 'bg-gray-500'}`}></span>
                <span className="text-xs text-gray-400">
                  {rerankerSettings.graph_rag_enabled ? 'Neo4j Connected' : 'Neo4j Standby'}
                </span>
              </div>
            </div>
          </div>

          {/* Global Search (Community Detection) */}
          <div className="bg-[#2a2a2a] rounded-lg p-4 border-l-4 border-amber-500">
            <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
              <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Global Search
            </h3>
            <p className="text-xs text-gray-500 mb-3">Community-based summaries for broad questions</p>

            <div className="space-y-3">
              {/* Community Detection Toggle */}
              <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-lg hover:bg-[#222] cursor-pointer">
                <div>
                  <span className="text-sm">Community Detection</span>
                  <p className="text-xs text-gray-500">Cluster related documents</p>
                </div>
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={rerankerSettings.community_detection_enabled ?? false}
                    onChange={(e) => onUpdateRerankerSettings({ community_detection_enabled: e.target.checked })}
                    className="sr-only peer"
                  />
                  <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-amber-500"></div>
                </div>
              </label>

              {/* Global Search Toggle */}
              <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-lg hover:bg-[#222] cursor-pointer">
                <div>
                  <span className="text-sm">Enable Global Search</span>
                  <p className="text-xs text-gray-500">Search community summaries</p>
                </div>
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={rerankerSettings.global_search_enabled ?? false}
                    onChange={(e) => onUpdateRerankerSettings({ global_search_enabled: e.target.checked })}
                    disabled={!rerankerSettings.community_detection_enabled}
                    className="sr-only peer"
                  />
                  <div className={`w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-amber-500 ${!rerankerSettings.community_detection_enabled ? 'opacity-50' : ''}`}></div>
                </div>
              </label>

              {/* Top K Summaries */}
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Top K Summaries
                  <SettingTip tip="Number of community summaries to include (1-5)" />
                </label>
                <input
                  type="number"
                  min="1"
                  max="5"
                  value={rerankerSettings.global_search_top_k ?? 3}
                  onChange={(e) => onUpdateRerankerSettings({ global_search_top_k: parseInt(e.target.value) || 3 })}
                  disabled={!rerankerSettings.global_search_enabled}
                  className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm disabled:opacity-50"
                />
              </div>

              {/* Community Level */}
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Community Level
                  <SettingTip tip="Granularity of community clusters" />
                </label>
                <select
                  value={rerankerSettings.community_level_default ?? 'medium'}
                  onChange={(e) => onUpdateRerankerSettings({ community_level_default: e.target.value })}
                  disabled={!rerankerSettings.community_detection_enabled}
                  className="w-full px-2 py-1.5 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm disabled:opacity-50"
                >
                  <option value="coarse">Coarse (fewer, broader)</option>
                  <option value="medium">Medium (balanced)</option>
                  <option value="fine">Fine (more, specific)</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Topics List */}
      <div className="bg-[#2a2a2a] rounded-lg overflow-hidden">
        <div className="px-4 py-3 bg-[#1a1a1a] border-b border-[#3a3a3a]">
          <h3 className="text-sm font-medium">Topics</h3>
        </div>
        <div className="divide-y divide-[#3a3a3a]">
          {knowledgeStats && Object.entries(knowledgeStats.topics).map(([topic, data]) => (
            <div key={topic}>
              <div
                className="px-4 py-3 flex items-center justify-between hover:bg-[#333] cursor-pointer"
                onClick={() => onToggleTopic(topic)}
              >
                <div className="flex items-center gap-3">
                  <span className="text-gray-400">
                    {expandedTopics.has(topic) ? '▼' : '▶'}
                  </span>
                  <span className="font-medium">{topic}</span>
                  <span className="text-xs text-gray-500">
                    ({data.files} files, {data.chunks.toLocaleString()} chunks)
                  </span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onReindexTopic(topic);
                  }}
                  className="px-3 py-1 bg-[#1a1a1a] hover:bg-blue-600 rounded-xl text-xs transition-colors"
                >
                  Reindex
                </button>
              </div>

              {/* Expanded file list */}
              {expandedTopics.has(topic) && (
                <div className="bg-[#1a1a1a] px-4 py-2">
                  {/* Untracked files */}
                  {topicUntracked[topic] && topicUntracked[topic].length > 0 && (
                    <div className="mb-4">
                      <h4 className="text-xs font-medium text-amber-400 mb-2 flex items-center gap-2">
                        <span className="w-2 h-2 bg-amber-400 rounded-full"></span>
                        Untracked Files ({topicUntracked[topic].length})
                      </h4>
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-left text-gray-500 text-xs">
                            <th className="py-2">Filename</th>
                            <th className="py-2">Size</th>
                            <th className="py-2">Extractor</th>
                            <th className="py-2"></th>
                          </tr>
                        </thead>
                        <tbody>
                          {topicUntracked[topic].map((file) => {
                            const isProcessing = !!processingFiles[file.path];
                            const statusText = processingFiles[file.path];
                            return (
                              <tr key={file.path} className={`border-t border-amber-900/30 ${isProcessing ? 'bg-blue-900/20' : 'bg-amber-900/10'}`}>
                                <td className="py-2 text-amber-200">
                                  <div className="flex items-center gap-2">
                                    {isProcessing && (
                                      <svg className="animate-spin h-4 w-4 text-blue-400" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                      </svg>
                                    )}
                                    <span>{file.filename}</span>
                                  </div>
                                  {isProcessing ? (
                                    <span className="text-xs text-blue-400 ml-6">{statusText}</span>
                                  ) : (
                                    <span className="text-xs text-amber-400 ml-2">(not ingested)</span>
                                  )}
                                </td>
                                <td className="py-2 text-gray-400">{file.size_mb} MB</td>
                                <td className="py-2">
                                  <select
                                    value={fileExtractorSelection[file.path] || ''}
                                    onChange={(e) => onFileExtractorSelectionChange({
                                      ...fileExtractorSelection,
                                      [file.path]: e.target.value
                                    })}
                                    disabled={isProcessing}
                                    className="px-2 py-1 bg-[#2a2a2a] border border-[#3a3a3a] rounded-xl text-xs w-32 disabled:opacity-50"
                                  >
                                    <option value="">Auto</option>
                                    <option value="pymupdf_text">PyMuPDF Text</option>
                                    <option value="pymupdf4llm">PyMuPDF4LLM</option>
                                    <option value="marker">Marker</option>
                                    <option value="vision_ocr">Vision OCR</option>
                                  </select>
                                </td>
                                <td className="py-2 text-right">
                                  <button
                                    onClick={() => onIngestUntracked(file.path, topic)}
                                    disabled={isProcessing}
                                    className="px-2 py-1 bg-amber-600 hover:bg-amber-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-xs transition-colors"
                                  >
                                    {isProcessing ? 'Processing...' : 'Ingest'}
                                  </button>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {/* Ingested files */}
                  {topicFiles[topic] && topicFiles[topic].length === 0 && (!topicUntracked[topic] || topicUntracked[topic].length === 0) ? (
                    <p className="text-sm text-gray-500 py-2">No files in this topic</p>
                  ) : topicFiles[topic] && topicFiles[topic].length > 0 && (
                    <>
                      <h4 className="text-xs font-medium text-green-400 mb-2 flex items-center gap-2">
                        <span className="w-2 h-2 bg-green-400 rounded-full"></span>
                        Ingested Files ({topicFiles[topic].length})
                      </h4>
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-left text-gray-500 text-xs">
                            <th className="py-2">Filename</th>
                            <th className="py-2">Chunks</th>
                            <th className="py-2">Size</th>
                            <th className="py-2">Ingested</th>
                            <th className="py-2">Re-index</th>
                            <th className="py-2"></th>
                          </tr>
                        </thead>
                        <tbody>
                          {topicFiles[topic].map((file) => {
                            const isProcessing = !!processingFiles[file.path];
                            const statusText = processingFiles[file.path];
                            return (
                              <tr key={file.file_hash} className={`border-t border-[#333] ${isProcessing ? 'bg-blue-900/20' : ''}`}>
                                <td className="py-2">
                                  <div className="flex items-center gap-2">
                                    {isProcessing && (
                                      <svg className="animate-spin h-4 w-4 text-blue-400" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                      </svg>
                                    )}
                                    <span className={file.exists ? '' : 'text-red-400 line-through'}>
                                      {file.filename}
                                    </span>
                                  </div>
                                  {!file.exists && <span className="text-xs text-red-400 ml-6">(missing)</span>}
                                  {isProcessing && <span className="text-xs text-blue-400 ml-6">{statusText}</span>}
                                </td>
                                <td className="py-2 text-gray-400">{file.chunks.toLocaleString()}</td>
                                <td className="py-2 text-gray-400">{file.size_mb} MB</td>
                                <td className="py-2 text-gray-400 text-xs">
                                  {new Date(file.ingested_at).toLocaleDateString()}
                                </td>
                                <td className="py-2">
                                  <div className="flex items-center gap-1">
                                    <select
                                      value={fileExtractorSelection[file.path] || ''}
                                      onChange={(e) => onFileExtractorSelectionChange({
                                        ...fileExtractorSelection,
                                        [file.path]: e.target.value
                                      })}
                                      disabled={isProcessing}
                                      className="px-1 py-0.5 bg-[#2a2a2a] border border-[#3a3a3a] rounded-xl text-xs w-24 disabled:opacity-50"
                                    >
                                      <option value="">Auto</option>
                                      <option value="pymupdf_text">PyMuPDF</option>
                                      <option value="pymupdf4llm">4LLM</option>
                                      <option value="marker">Marker</option>
                                      <option value="vision_ocr">Vision</option>
                                    </select>
                                    <button
                                      onClick={() => onReindexFile(file.path, topic)}
                                      disabled={isProcessing}
                                      className="px-2 py-0.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-xs transition-colors whitespace-nowrap"
                                    >
                                      {isProcessing ? 'Processing...' : 'Re-index'}
                                    </button>
                                  </div>
                                </td>
                                <td className="py-2 text-right">
                                  <button
                                    onClick={() => onDeleteFile(file.path, topic)}
                                    disabled={isProcessing}
                                    className="text-red-400 hover:text-red-300 disabled:text-gray-500 disabled:cursor-not-allowed text-xs"
                                  >
                                    Delete
                                  </button>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </>
                  )}
                </div>
              )}
            </div>
          ))}

          {knowledgeStats && Object.keys(knowledgeStats.topics).length === 0 && (
            <div className="px-4 py-8 text-center text-gray-500">
              No topics found. Create a topic and ingest documents.
            </div>
          )}
        </div>
      </div>

      {/* Add Knowledge Section */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-sm font-medium mb-4">Add Knowledge</h3>

        <div className="flex gap-4 mb-4">
          <div className="flex-1">
            <label className="block text-xs text-gray-400 mb-2">Document File</label>
            <input
              type="file"
              accept=".pdf,.txt,.md,.docx"
              onChange={(e) => onIngestFileChange(e.target.files?.[0] || null)}
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm file:mr-4 file:py-1 file:px-3 file:rounded-xl file:border-0 file:bg-blue-600 file:text-white"
            />
          </div>
          <div className="w-48">
            <label className="block text-xs text-gray-400 mb-2">Topic</label>
            <select
              value={ingestTopic}
              onChange={(e) => onIngestTopicChange(e.target.value)}
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
            >
              {availableTopics.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
          <div className="flex items-end">
            <button
              onClick={onIngestFile}
              disabled={!ingestFile || !ingestTopic}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors"
            >
              Ingest Document
            </button>
          </div>
        </div>

        {/* New Topic */}
        <div className="flex gap-4 mb-4">
          <div className="flex-1">
            <label className="block text-xs text-gray-400 mb-2">Create New Topic</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={newTopicName}
                onChange={(e) => onNewTopicNameChange(e.target.value)}
                placeholder="topic_name"
                className="flex-1 px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
              />
              <button
                onClick={onCreateTopic}
                disabled={!newTopicName.trim()}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors"
              >
                + Create
              </button>
            </div>
          </div>
        </div>

        {/* Advanced Options */}
        <button
          onClick={() => onShowAdvancedIngestChange(!showAdvancedIngest)}
          className="text-xs text-gray-400 hover:text-white"
        >
          {showAdvancedIngest ? '▼' : '▶'} Advanced Options
        </button>

        {showAdvancedIngest && (
          <div className="mt-3 grid grid-cols-4 gap-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Chunk Size</label>
              <input
                type="number"
                value={ingestChunkSize}
                onChange={(e) => onIngestChunkSizeChange(parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Chunk Overlap</label>
              <input
                type="number"
                value={ingestChunkOverlap}
                onChange={(e) => onIngestChunkOverlapChange(parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
              />
            </div>
          </div>
        )}
      </div>

      {/* Search Testing */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-sm font-medium mb-4">Search Testing</h3>

        <div className="flex gap-4 mb-4 overflow-hidden">
          <div className="flex-1 min-w-0">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => onSearchQueryChange(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && onSearchTest()}
              placeholder="Enter search query..."
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
            />
          </div>
          <div className="w-32">
            <select
              value={searchTopK}
              onChange={(e) => onSearchTopKChange(parseInt(e.target.value))}
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
            >
              <option value={3}>Top 3</option>
              <option value={5}>Top 5</option>
              <option value={10}>Top 10</option>
            </select>
          </div>
          <button
            onClick={onSearchTest}
            disabled={!searchQuery.trim()}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors"
          >
            Search
          </button>
        </div>

        {/* Topic filter chips */}
        <div className="flex flex-wrap gap-2 mb-4">
          {availableTopics.map((t) => (
            <button
              key={t}
              onClick={() => {
                if (searchTopics.includes(t)) {
                  onSearchTopicsChange(searchTopics.filter(x => x !== t));
                } else {
                  onSearchTopicsChange([...searchTopics, t]);
                }
              }}
              className={`px-3 py-1 rounded-full text-xs transition-colors ${
                searchTopics.includes(t)
                  ? 'bg-blue-600 text-white'
                  : 'bg-[#1a1a1a] text-gray-400 hover:text-white'
              }`}
            >
              {t}
            </button>
          ))}
          {searchTopics.length > 0 && (
            <button
              onClick={() => onSearchTopicsChange([])}
              className="px-3 py-1 rounded-full text-xs bg-gray-600 text-white hover:bg-gray-500"
            >
              Clear
            </button>
          )}
        </div>

        {/* Search Results */}
        {searchTestResult && (
          <div className="space-y-3">
            <div className="flex items-center gap-4 text-xs text-gray-400">
              <span>Query: &quot;{searchTestResult.query}&quot;</span>
              <span>Topics: {searchTestResult.topics_searched.join(', ')}</span>
              <span>{searchTestResult.processing_ms}ms</span>
            </div>

            {searchTestResult.routing && (
              <div className="text-xs bg-[#1a1a1a] rounded-lg p-2">
                <span className="text-gray-400">Router decision: </span>
                <span className="text-blue-400">{searchTestResult.routing.routed_to.join(', ')}</span>
              </div>
            )}

            {searchTestResult.results.map((result, i) => (
              <div key={i} className="bg-[#1a1a1a] rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs bg-blue-600 px-2 py-0.5 rounded-full">#{i + 1}</span>
                    <span className="text-sm font-medium">{result.source}</span>
                    {result.page && <span className="text-xs text-gray-400">p.{result.page}</span>}
                  </div>
                  <div className="flex items-center gap-3 text-xs">
                    <span className={`font-medium ${
                      result.normalized_score >= 0.75 ? 'text-green-400' :
                      result.normalized_score >= 0.5 ? 'text-yellow-400' : 'text-gray-400'
                    }`}>
                      {(result.normalized_score * 100).toFixed(0)}%
                    </span>
                    {result.raw_score !== undefined && (
                      <span className="text-gray-500">raw: {result.raw_score.toFixed(3)}</span>
                    )}
                    {result.rerank_score !== undefined && result.rerank_score !== null && (
                      <span className="text-gray-500">rerank: {result.rerank_score.toFixed(3)}</span>
                    )}
                  </div>
                </div>
                <p className="text-xs text-gray-300 whitespace-pre-wrap">{result.content}</p>
                <p className="text-xs text-gray-500 mt-2">Topic: {result.topic}</p>
              </div>
            ))}

            {searchTestResult.num_results === 0 && (
              <div className="text-center text-gray-500 py-4">No results found</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
