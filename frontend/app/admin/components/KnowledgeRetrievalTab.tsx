'use client';

import React from 'react';
import { SettingTip } from './shared';

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
  bm25_enabled: boolean;
  bm25_weight: number;
  query_expansion_enabled: boolean;
  domain_boost_enabled: boolean;
  domain_boost_factor: number;
  hyde_enabled: boolean;
  hyde_model: string;
  crag_enabled: boolean;
  crag_min_confidence: number;
  crag_web_search_on_low: boolean;
  // RAPTOR
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
}

interface RetrievalProfile {
  name: string;
  display_name: string;
  description: string;
  is_active: boolean;
  is_default: boolean;
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

export interface KnowledgeRetrievalTabProps {
  topicsWithStatus: TopicWithStatus[];
  availableTopics: string[];
  rerankerSettings: RerankerSettings | null;
  searchTestResult: SearchTestResult | null;
  // Search state
  searchQuery: string;
  searchTopics: string[];
  searchTopK: number;
  // Callbacks
  onToggleTopicEnabled: (topic: string, enabled: boolean) => void;
  onUpdateRerankerSettings: (updates: Partial<RerankerSettings>) => void;
  onSearchTest: () => void;
  // State setters
  onSearchQueryChange: (query: string) => void;
  onSearchTopicsChange: (topics: string[]) => void;
  onSearchTopKChange: (topK: number) => void;
  loading: boolean;
  // Retrieval profiles
  profiles?: RetrievalProfile[];
  activeProfile?: string;
  hasManualOverrides?: boolean;
  onSelectProfile?: (name: string) => void;
  onClearOverrides?: () => void;
}

export function KnowledgeRetrievalTab({
  topicsWithStatus,
  availableTopics,
  rerankerSettings,
  searchTestResult,
  searchQuery,
  searchTopics,
  searchTopK,
  onToggleTopicEnabled,
  onUpdateRerankerSettings,
  onSearchTest,
  onSearchQueryChange,
  onSearchTopicsChange,
  onSearchTopKChange,
  loading,
  profiles,
  activeProfile,
  hasManualOverrides,
  onSelectProfile,
  onClearOverrides,
}: KnowledgeRetrievalTabProps) {
  return (
    <div className="space-y-6">
      <h2 className="text-lg font-medium">Retrieval Settings</h2>

      {/* Retrieval Profile */}
      {profiles && profiles.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-zinc-300">Retrieval Profile</h3>
            {hasManualOverrides && (
              <button
                onClick={onClearOverrides}
                className="text-xs text-amber-400 hover:text-amber-300 transition-colors"
              >
                Reset to profile defaults
              </button>
            )}
          </div>
          <div className="grid grid-cols-3 gap-3">
            {profiles.map((p) => (
              <button
                key={p.name}
                onClick={() => onSelectProfile?.(p.name)}
                className={`p-3 rounded-lg border text-left transition-all ${
                  p.is_active
                    ? 'border-blue-500/50 bg-blue-500/10'
                    : 'border-zinc-700 bg-zinc-800/50 hover:border-zinc-600'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-medium ${p.is_active ? 'text-blue-400' : 'text-zinc-300'}`}>
                    {p.display_name}
                  </span>
                  {p.is_active && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400">
                      Active
                    </span>
                  )}
                  {p.is_default && !p.is_active && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-700 text-zinc-400">
                      Default
                    </span>
                  )}
                </div>
                <p className="text-xs text-zinc-500 mt-1">{p.description}</p>
              </button>
            ))}
          </div>
          {hasManualOverrides && (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
              <span className="text-xs text-amber-400">
                Custom overrides active — individual toggles differ from profile defaults
              </span>
            </div>
          )}
        </div>
      )}

      {/* Active Knowledge Bases */}
      <div className="bg-[#2a2a2a] rounded-xl p-4">
        <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
          <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
          </svg>
          Active Knowledge Bases
        </h3>
        <p className="text-xs text-gray-500 mb-3">Toggle topics to enable/disable them for chat search</p>
        <div className="space-y-2">
          {topicsWithStatus.map((topic) => (
            <label key={topic.name} className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-xl hover:bg-[#222] cursor-pointer">
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

      {/* Core Retrieval Settings — always visible */}
      {rerankerSettings && (
        <div className="grid grid-cols-2 gap-6">
          {/* Core Settings */}
          <div className="bg-[#2a2a2a] rounded-xl p-4">
            <h3 className="text-sm font-medium mb-4 flex items-center gap-2">
              <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              Core Settings
            </h3>
            <div className="space-y-3">
              {/* Reranker */}
              <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-xl hover:bg-[#222] cursor-pointer">
                <div>
                  <span className="text-sm">
                    Cross-Encoder Reranking
                    <SettingTip tip="Cross-encoder reranking improves relevance but adds ~200-500ms latency" />
                  </span>
                  <p className="text-xs text-gray-500">More accurate but slower</p>
                </div>
                <div className="relative">
                  <input type="checkbox" checked={rerankerSettings.reranker_enabled}
                    onChange={(e) => onUpdateRerankerSettings({ reranker_enabled: e.target.checked })}
                    className="sr-only peer" />
                  <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-500"></div>
                </div>
              </label>

              {/* Semantic Routing */}
              <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-xl hover:bg-[#222] cursor-pointer">
                <div>
                  <span className="text-sm">Semantic Topic Routing</span>
                  <p className="text-xs text-gray-500">Use embeddings for topic selection</p>
                </div>
                <div className="relative">
                  <input type="checkbox" checked={rerankerSettings.router_use_semantic}
                    onChange={(e) => onUpdateRerankerSettings({ router_use_semantic: e.target.checked })}
                    className="sr-only peer" />
                  <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-500"></div>
                </div>
              </label>

              {/* Numeric: Top K + Candidates */}
              <div className="grid grid-cols-2 gap-2 pt-2">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">
                    Top K Results <SettingTip tip="Number of results returned to the LLM" />
                  </label>
                  <input type="number" min="1" max="20" value={rerankerSettings.rag_top_k}
                    onChange={(e) => onUpdateRerankerSettings({ rag_top_k: parseInt(e.target.value) || 5 })}
                    className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm" />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">
                    Candidates <SettingTip tip="How many results to score before selecting Top K" />
                  </label>
                  <input type="number" min="5" max="50" value={rerankerSettings.reranker_candidates}
                    onChange={(e) => onUpdateRerankerSettings({ reranker_candidates: parseInt(e.target.value) || 20 })}
                    className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm" />
                </div>
              </div>

              {/* Quality Thresholds */}
              <div className="border-t border-[#3a3a3a] pt-3 mt-3">
                <h4 className="text-xs font-medium text-gray-400 mb-2">Quality Thresholds</h4>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Min Pre-filter <SettingTip tip="Minimum similarity score (0-1) before reranking" />
                    </label>
                    <input type="number" min="0" max="0.5" step="0.05" value={rerankerSettings.rag_min_score}
                      onChange={(e) => onUpdateRerankerSettings({ rag_min_score: parseFloat(e.target.value) || 0.15 })}
                      className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm" />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Min Final <SettingTip tip="Minimum score after reranking to include in results" />
                    </label>
                    <input type="number" min="0" max="0.5" step="0.05" value={rerankerSettings.rag_min_final_score}
                      onChange={(e) => onUpdateRerankerSettings({ rag_min_final_score: parseFloat(e.target.value) || 0.20 })}
                      className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm" />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* BM25 + Diversity */}
          <div className="bg-[#2a2a2a] rounded-xl p-4">
            <h3 className="text-sm font-medium mb-4">BM25 &amp; Diversity</h3>
            <div className="space-y-3">
              {/* BM25 */}
              <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-xl hover:bg-[#222] cursor-pointer">
                <div>
                  <span className="text-sm">
                    BM25 Hybrid <SettingTip tip="Sparse retrieval combined with dense via RRF" />
                  </span>
                  <p className="text-xs text-gray-500">Keyword matching + semantic</p>
                </div>
                <div className="relative">
                  <input type="checkbox" checked={rerankerSettings.bm25_enabled ?? false}
                    onChange={(e) => onUpdateRerankerSettings({ bm25_enabled: e.target.checked })}
                    className="sr-only peer" />
                  <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-500"></div>
                </div>
              </label>
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  BM25 Weight ({(rerankerSettings.bm25_weight ?? 0.3).toFixed(1)})
                  <SettingTip tip="0.5 = equal BM25/dense in RRF fusion" />
                </label>
                <input type="range" min="0" max="1" step="0.1" value={rerankerSettings.bm25_weight ?? 0.3}
                  onChange={(e) => onUpdateRerankerSettings({ bm25_weight: parseFloat(e.target.value) })}
                  disabled={!rerankerSettings.bm25_enabled}
                  className="w-full h-2 bg-[#1a1a1a] rounded-lg appearance-none cursor-pointer accent-blue-500 disabled:opacity-50" />
                <div className="flex justify-between text-[10px] text-gray-500 mt-0.5">
                  <span>Dense only</span><span>BM25 heavy</span>
                </div>
              </div>

              {/* Diversity */}
              <div className="border-t border-[#3a3a3a] pt-3 mt-3">
                <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-xl hover:bg-[#222] cursor-pointer mb-2">
                  <div>
                    <span className="text-sm">
                      MMR Diversity <SettingTip tip="Maximal Marginal Relevance ensures results come from different sources" />
                    </span>
                    <p className="text-xs text-gray-500">Balance relevance with source variety</p>
                  </div>
                  <div className="relative">
                    <input type="checkbox" checked={rerankerSettings.rag_diversity_enabled}
                      onChange={(e) => onUpdateRerankerSettings({ rag_diversity_enabled: e.target.checked })}
                      className="sr-only peer" />
                    <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-emerald-500"></div>
                  </div>
                </label>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Diversity λ ({rerankerSettings.rag_diversity_lambda.toFixed(1)})
                    </label>
                    <input type="range" min="0.3" max="0.9" step="0.1" value={rerankerSettings.rag_diversity_lambda}
                      onChange={(e) => onUpdateRerankerSettings({ rag_diversity_lambda: parseFloat(e.target.value) })}
                      disabled={!rerankerSettings.rag_diversity_enabled}
                      className="w-full h-2 bg-[#1a1a1a] rounded-lg appearance-none cursor-pointer accent-emerald-500" />
                    <div className="flex justify-between text-[10px] text-gray-500 mt-0.5">
                      <span>Diverse</span><span>Relevant</span>
                    </div>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Max/Source <SettingTip tip="Limit results per document" />
                    </label>
                    <input type="number" min="1" max="5" value={rerankerSettings.rag_max_per_source}
                      onChange={(e) => onUpdateRerankerSettings({ rag_max_per_source: parseInt(e.target.value) || 2 })}
                      disabled={!rerankerSettings.rag_diversity_enabled}
                      className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm" />
                  </div>
                </div>
              </div>

              {/* Domain Boost */}
              <div className="border-t border-[#3a3a3a] pt-3 mt-3">
                <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-xl hover:bg-[#222] cursor-pointer mb-2">
                  <div>
                    <span className="text-sm">
                      Domain Boost <SettingTip tip="Counters reranker bias toward generic results by boosting domain-relevant matches" />
                    </span>
                    <p className="text-xs text-gray-500">Boost domain-relevant results</p>
                  </div>
                  <div className="relative">
                    <input type="checkbox" checked={rerankerSettings.domain_boost_enabled ?? false}
                      onChange={(e) => onUpdateRerankerSettings({ domain_boost_enabled: e.target.checked })}
                      className="sr-only peer" />
                    <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-amber-500"></div>
                  </div>
                </label>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">
                    Boost Factor ({(rerankerSettings.domain_boost_factor ?? 0.15).toFixed(2)})
                  </label>
                  <input type="range" min="0" max="0.5" step="0.05" value={rerankerSettings.domain_boost_factor ?? 0.15}
                    onChange={(e) => onUpdateRerankerSettings({ domain_boost_factor: parseFloat(e.target.value) })}
                    disabled={!rerankerSettings.domain_boost_enabled}
                    className="w-full h-2 bg-[#1a1a1a] rounded-lg appearance-none cursor-pointer accent-amber-500 disabled:opacity-50" />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Vision Extraction (Ingestion) */}
      {rerankerSettings && (
        <div className="bg-[#2a2a2a] rounded-xl p-4 border-l-4 border-orange-500">
          <label className="flex items-center justify-between cursor-pointer">
            <div className="flex items-center gap-3">
              <svg className="w-5 h-5 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
              <div>
                <span className="text-sm font-medium">Vision Extraction</span>
                <p className="text-xs text-gray-500">
                  Use vision model for charts, figures, and diagrams during ingestion.
                  Slower but captures visual content. Disable for text-only speed.
                </p>
              </div>
            </div>
            <div className="relative ml-4 shrink-0">
              <input type="checkbox" checked={rerankerSettings.vision_ingest_enabled ?? true}
                onChange={(e) => onUpdateRerankerSettings({ vision_ingest_enabled: e.target.checked })}
                className="sr-only peer" />
              <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-orange-500"></div>
            </div>
          </label>
        </div>
      )}

      {/* Advanced Retrieval */}
      {rerankerSettings && (
        <div className="grid grid-cols-3 gap-4">
          {/* HyDE */}
          <div className="bg-[#2a2a2a] rounded-xl p-4">
            <h3 className="text-sm font-medium mb-3">HyDE</h3>
            <label className="flex items-center justify-between">
              <span className="text-sm">
                Enabled <SettingTip tip="Hypothetical Document Embeddings — generates a hypothetical answer to improve retrieval" />
              </span>
              <div className="relative">
                <input type="checkbox" checked={rerankerSettings.hyde_enabled ?? false}
                  onChange={(e) => onUpdateRerankerSettings({ hyde_enabled: e.target.checked })}
                  className="sr-only peer" />
                <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-500"></div>
              </div>
            </label>
          </div>

          {/* CRAG */}
          <div className="bg-[#2a2a2a] rounded-xl p-4">
            <h3 className="text-sm font-medium mb-3">CRAG</h3>
            <label className="flex items-center justify-between mb-2">
              <span className="text-sm">
                Enabled <SettingTip tip="Corrective RAG — scores confidence and falls back to web search" />
              </span>
              <div className="relative">
                <input type="checkbox" checked={rerankerSettings.crag_enabled ?? false}
                  onChange={(e) => onUpdateRerankerSettings({ crag_enabled: e.target.checked })}
                  className="sr-only peer" />
                <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-500"></div>
              </div>
            </label>
            <div className="space-y-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Min Confidence</label>
                <input type="number" min="0" max="1" step="0.05" value={rerankerSettings.crag_min_confidence ?? 0.3}
                  onChange={(e) => onUpdateRerankerSettings({ crag_min_confidence: parseFloat(e.target.value) })}
                  disabled={!rerankerSettings.crag_enabled}
                  className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-xs disabled:opacity-50" />
              </div>
              <label className="flex items-center gap-2 text-xs">
                <input type="checkbox" checked={rerankerSettings.crag_web_search_on_low ?? false}
                  onChange={(e) => onUpdateRerankerSettings({ crag_web_search_on_low: e.target.checked })}
                  disabled={!rerankerSettings.crag_enabled}
                  className="w-3.5 h-3.5 disabled:opacity-50" />
                Web search on low
              </label>
            </div>
          </div>

          {/* Query Expansion */}
          <div className="bg-[#2a2a2a] rounded-xl p-4">
            <h3 className="text-sm font-medium mb-3">Query Expansion</h3>
            <label className="flex items-center justify-between">
              <span className="text-sm">
                Enabled <SettingTip tip="Expand queries with synonyms and related terms for better recall" />
              </span>
              <div className="relative">
                <input type="checkbox" checked={rerankerSettings.query_expansion_enabled ?? false}
                  onChange={(e) => onUpdateRerankerSettings({ query_expansion_enabled: e.target.checked })}
                  className="sr-only peer" />
                <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-500"></div>
              </div>
            </label>
          </div>
        </div>
      )}

      {/* RAPTOR / GraphRAG / Global Search — 3-column cards */}
      {rerankerSettings && (
        <div className="grid grid-cols-3 gap-4">
          {/* RAPTOR */}
          <div className="bg-[#2a2a2a] rounded-xl p-4 border-l-4 border-green-500">
            <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
              <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
              </svg>
              RAPTOR Hierarchical Search
            </h3>
            <p className="text-xs text-gray-500 mb-3">Recursive abstractive processing for tree-organized retrieval</p>
            <div className="space-y-3">
              <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-xl hover:bg-[#222] cursor-pointer">
                <div>
                  <span className="text-sm">Enable RAPTOR</span>
                  <p className="text-xs text-gray-500">Hierarchical summarization</p>
                </div>
                <div className="relative">
                  <input type="checkbox" checked={rerankerSettings.raptor_enabled ?? false}
                    onChange={(e) => onUpdateRerankerSettings({ raptor_enabled: e.target.checked })}
                    className="sr-only peer" />
                  <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-green-500"></div>
                </div>
              </label>
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Retrieval Strategy <SettingTip tip="Collapsed: search all levels at once. Tree traversal: start from top" />
                </label>
                <select value={rerankerSettings.raptor_retrieval_strategy ?? 'collapsed'}
                  onChange={(e) => onUpdateRerankerSettings({ raptor_retrieval_strategy: e.target.value })}
                  disabled={!rerankerSettings.raptor_enabled}
                  className="w-full px-2 py-1.5 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm disabled:opacity-50">
                  <option value="collapsed">Collapsed (faster)</option>
                  <option value="tree_traversal">Tree Traversal (deeper)</option>
                </select>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Max Levels</label>
                  <input type="number" min="1" max="5" value={rerankerSettings.raptor_max_levels ?? 3}
                    onChange={(e) => onUpdateRerankerSettings({ raptor_max_levels: parseInt(e.target.value) || 3 })}
                    disabled={!rerankerSettings.raptor_enabled}
                    className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm disabled:opacity-50" />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Cluster Size</label>
                  <input type="number" min="5" max="20" value={rerankerSettings.raptor_cluster_size ?? 10}
                    onChange={(e) => onUpdateRerankerSettings({ raptor_cluster_size: parseInt(e.target.value) || 10 })}
                    disabled={!rerankerSettings.raptor_enabled}
                    className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm disabled:opacity-50" />
                </div>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Summary Model</label>
                <input type="text" value={rerankerSettings.raptor_summary_model ?? 'qwen2.5:1.5b'}
                  onChange={(e) => onUpdateRerankerSettings({ raptor_summary_model: e.target.value })}
                  disabled={!rerankerSettings.raptor_enabled}
                  className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm disabled:opacity-50" />
              </div>
            </div>
          </div>

          {/* GraphRAG */}
          <div className="bg-[#2a2a2a] rounded-xl p-4 border-l-4 border-cyan-500">
            <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
              <svg className="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
              Knowledge Graph
            </h3>
            <p className="text-xs text-gray-500 mb-3">Neo4j-powered entity relationships and reasoning</p>
            <div className="space-y-3">
              <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-xl hover:bg-[#222] cursor-pointer">
                <div>
                  <span className="text-sm">Enable GraphRAG</span>
                  <p className="text-xs text-gray-500">Entity-based retrieval</p>
                </div>
                <div className="relative">
                  <input type="checkbox" checked={rerankerSettings.graph_rag_enabled ?? false}
                    onChange={(e) => onUpdateRerankerSettings({ graph_rag_enabled: e.target.checked })}
                    className="sr-only peer" />
                  <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-cyan-500"></div>
                </div>
              </label>
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Graph Weight in RRF ({(rerankerSettings.graph_rag_weight ?? 0.2).toFixed(2)})
                  <SettingTip tip="How much weight to give graph results in fusion ranking" />
                </label>
                <input type="range" min="0" max="0.5" step="0.05" value={rerankerSettings.graph_rag_weight ?? 0.2}
                  onChange={(e) => onUpdateRerankerSettings({ graph_rag_weight: parseFloat(e.target.value) })}
                  disabled={!rerankerSettings.graph_rag_enabled}
                  className="w-full h-2 bg-[#1a1a1a] rounded-lg appearance-none cursor-pointer accent-cyan-500 disabled:opacity-50" />
                <div className="flex justify-between text-[10px] text-gray-500 mt-0.5">
                  <span>Vector only</span><span>Balanced</span>
                </div>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Max Hops <SettingTip tip="Maximum relationship traversal depth (1-3)" />
                </label>
                <input type="number" min="1" max="3" value={rerankerSettings.graph_max_hops ?? 2}
                  onChange={(e) => onUpdateRerankerSettings({ graph_max_hops: parseInt(e.target.value) || 2 })}
                  disabled={!rerankerSettings.graph_rag_enabled}
                  className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm disabled:opacity-50" />
              </div>
              <div className="flex items-center gap-2 p-2 bg-[#1a1a1a] rounded-xl">
                <span className={`w-2 h-2 rounded-full ${rerankerSettings.graph_rag_enabled ? 'bg-green-400' : 'bg-gray-500'}`}></span>
                <span className="text-xs text-gray-400">
                  {rerankerSettings.graph_rag_enabled ? 'Neo4j Connected' : 'Neo4j Standby'}
                </span>
              </div>
              {!rerankerSettings.graph_rag_enabled && (
                <label className="flex items-center justify-between p-2 bg-cyan-500/5 border border-cyan-500/15 rounded-xl cursor-pointer">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-400 font-medium shrink-0">AUTO</span>
                    <span className="text-xs text-cyan-400/70">
                      Auto-activate on cross-reference queries
                    </span>
                  </div>
                  <div className="relative">
                    <input type="checkbox" checked={rerankerSettings.graph_rag_auto ?? true}
                      onChange={(e) => onUpdateRerankerSettings({ graph_rag_auto: e.target.checked })}
                      className="sr-only peer" />
                    <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-cyan-500"></div>
                  </div>
                </label>
              )}
            </div>
          </div>

          {/* Global Search */}
          <div className="bg-[#2a2a2a] rounded-xl p-4 border-l-4 border-amber-500">
            <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
              <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Global Search
            </h3>
            <p className="text-xs text-gray-500 mb-3">Community-based summaries for broad questions</p>
            <div className="space-y-3">
              <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-xl hover:bg-[#222] cursor-pointer">
                <div>
                  <span className="text-sm">Community Detection</span>
                  <p className="text-xs text-gray-500">Cluster related documents</p>
                </div>
                <div className="relative">
                  <input type="checkbox" checked={rerankerSettings.community_detection_enabled ?? false}
                    onChange={(e) => onUpdateRerankerSettings({ community_detection_enabled: e.target.checked })}
                    className="sr-only peer" />
                  <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-amber-500"></div>
                </div>
              </label>
              <label className="flex items-center justify-between p-2 bg-[#1a1a1a] rounded-xl hover:bg-[#222] cursor-pointer">
                <div>
                  <span className="text-sm">Enable Global Search</span>
                  <p className="text-xs text-gray-500">Search community summaries</p>
                </div>
                <div className="relative">
                  <input type="checkbox" checked={rerankerSettings.global_search_enabled ?? false}
                    onChange={(e) => onUpdateRerankerSettings({ global_search_enabled: e.target.checked })}
                    disabled={!rerankerSettings.community_detection_enabled}
                    className="sr-only peer" />
                  <div className={`w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-amber-500 ${!rerankerSettings.community_detection_enabled ? 'opacity-50' : ''}`}></div>
                </div>
              </label>
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Top K Summaries <SettingTip tip="Number of community summaries to include" />
                </label>
                <input type="number" min="1" max="5" value={rerankerSettings.global_search_top_k ?? 3}
                  onChange={(e) => onUpdateRerankerSettings({ global_search_top_k: parseInt(e.target.value) || 3 })}
                  disabled={!rerankerSettings.global_search_enabled}
                  className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm disabled:opacity-50" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Community Level <SettingTip tip="Granularity of community clusters" />
                </label>
                <select value={rerankerSettings.community_level_default ?? 'medium'}
                  onChange={(e) => onUpdateRerankerSettings({ community_level_default: e.target.value })}
                  disabled={!rerankerSettings.community_detection_enabled}
                  className="w-full px-2 py-1.5 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm disabled:opacity-50">
                  <option value="coarse">Coarse (fewer, broader)</option>
                  <option value="medium">Medium (balanced)</option>
                  <option value="fine">Fine (more, specific)</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Search Testing */}
      <div className="bg-[#2a2a2a] rounded-xl p-4">
        <h3 className="text-sm font-medium mb-4">Search Testing</h3>

        <div className="flex gap-4 mb-4 min-w-0">
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
          <div className="w-32 shrink-0">
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
            disabled={!searchQuery.trim() || loading}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors shrink-0"
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

        {/* Empty state */}
        {!searchTestResult && !loading && (
          <div className="text-center py-8">
            <p className="text-sm text-gray-600">Run a search to test retrieval results</p>
          </div>
        )}

        {/* Loading state */}
        {loading && (
          <div className="flex items-center justify-center py-8 gap-2">
            <svg className="animate-spin h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <span className="text-sm text-gray-400">Searching...</span>
          </div>
        )}

        {/* Search Results */}
        {searchTestResult && (
          <div className="space-y-3">
            <div className="flex items-center gap-4 text-xs text-gray-400">
              <span>Query: &quot;{searchTestResult.query}&quot;</span>
              <span>Topics: {searchTestResult.topics_searched.join(', ')}</span>
              <span>{Math.round(searchTestResult.processing_ms)}ms</span>
            </div>

            {searchTestResult.routing && (
              <div className="text-xs bg-[#1a1a1a] rounded-xl p-2">
                <span className="text-gray-400">Router decision: </span>
                <span className="text-blue-400">{searchTestResult.routing.routed_to.join(', ')}</span>
              </div>
            )}

            {searchTestResult.results.map((result, i) => (
              <div key={i} className="bg-[#1a1a1a] rounded-xl p-3">
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
