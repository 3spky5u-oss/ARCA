'use client';

import React, { useState, useEffect, useCallback } from 'react';

interface QdrantCollection {
  name: string;
  vectors_count: number;
  points_count: number;
  dimension: number | null;
  error?: string;
}

interface CollectionDetail {
  name: string;
  vectors_count: number;
  points_count: number;
  status: string;
  config: Record<string, unknown>;
  segments_count: number;
}

interface SearchResult {
  id: string | number;
  score: number;
  content: string;
  source: string;
  topic: string;
}

interface KnowledgeCollectionsTabProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  apiCall: (endpoint: string, options?: RequestInit) => Promise<any>;
  setMessage: (msg: { type: 'success' | 'error'; text: string } | null) => void;
}

const CORE_KNOWLEDGE_COLLECTION = 'arca_core';

export function KnowledgeCollectionsTab({ apiCall, setMessage }: KnowledgeCollectionsTabProps) {
  const [collections, setCollections] = useState<QdrantCollection[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null);
  const [collectionDetail, setCollectionDetail] = useState<CollectionDetail | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [coreKnowledgeEnabled, setCoreKnowledgeEnabled] = useState(true);

  const fetchCollections = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiCall('/api/admin/knowledge/collections');
      setCollections(data.collections || []);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch collections: ${e}` });
    } finally {
      setLoading(false);
    }
    // Fetch core knowledge status
    try {
      const ckStatus = await apiCall('/api/admin/knowledge/core-knowledge/status');
      setCoreKnowledgeEnabled(ckStatus.enabled ?? true);
    } catch {
      // Non-critical — ignore
    }
  }, [apiCall, setMessage]);

  useEffect(() => {
    fetchCollections();
  }, [fetchCollections]);

  const handleSelectCollection = async (name: string) => {
    if (selectedCollection === name) {
      setSelectedCollection(null);
      setCollectionDetail(null);
      setSearchResults([]);
      return;
    }
    setSelectedCollection(name);
    setSearchResults([]);
    try {
      const data = await apiCall(`/api/admin/knowledge/collections/${encodeURIComponent(name)}`);
      setCollectionDetail(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch collection details: ${e}` });
    }
  };

  const handleToggleCoreKnowledge = async () => {
    const newState = !coreKnowledgeEnabled;
    try {
      await apiCall(`/api/admin/knowledge/core-knowledge/toggle?enabled=${newState}`, { method: 'POST' });
      setCoreKnowledgeEnabled(newState);
      setMessage({ type: 'success', text: `Core knowledge ${newState ? 'enabled' : 'disabled'}` });
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to toggle core knowledge: ${e}` });
    }
  };

  const handleDeleteCollection = async (name: string) => {
    if (!confirm(`Delete collection "${name}"? This will permanently remove all vectors. This cannot be undone.`)) {
      return;
    }
    try {
      await apiCall(`/api/admin/knowledge/collections/${encodeURIComponent(name)}`, { method: 'DELETE' });
      setMessage({ type: 'success', text: `Collection "${name}" deleted` });
      setSelectedCollection(null);
      setCollectionDetail(null);
      await fetchCollections();
    } catch (e) {
      setMessage({ type: 'error', text: `Delete failed: ${e}` });
    }
  };

  const handleSearch = async () => {
    if (!selectedCollection || !searchQuery.trim()) return;
    setSearching(true);
    setSearchResults([]);
    try {
      const data = await apiCall(
        `/api/admin/knowledge/collections/${encodeURIComponent(selectedCollection)}/search?query=${encodeURIComponent(searchQuery)}&limit=5`,
        { method: 'POST' },
      );
      setSearchResults(data.results || []);
    } catch (e) {
      setMessage({ type: 'error', text: `Search failed: ${e}` });
    } finally {
      setSearching(false);
    }
  };

  const totalVectors = collections.reduce((sum, c) => sum + (c.vectors_count || 0), 0);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium text-gray-300">Qdrant Collections</h3>
          <p className="text-xs text-gray-500 mt-1">
            {collections.length} collection{collections.length !== 1 ? 's' : ''}, {totalVectors.toLocaleString()} total vectors
          </p>
        </div>
        <button
          onClick={fetchCollections}
          disabled={loading}
          className="px-3 py-1.5 bg-[#333] hover:bg-[#3a3a3a] rounded-lg text-xs transition-colors disabled:opacity-50"
        >
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      {/* Collections Grid */}
      {collections.length === 0 && !loading && (
        <div className="bg-[#2a2a2a] rounded-lg p-8 text-center">
          <p className="text-gray-500 text-sm">No collections found in Qdrant</p>
          <p className="text-gray-600 text-xs mt-2">Ingest documents to create collections</p>
        </div>
      )}

      <div className="grid grid-cols-3 gap-3">
        {collections.map((col) => (
          <button
            key={col.name}
            onClick={() => handleSelectCollection(col.name)}
            className={`text-left bg-[#2a2a2a] rounded-lg p-4 transition-colors border ${
              selectedCollection === col.name
                ? 'border-blue-500'
                : 'border-transparent hover:border-[#3a3a3a]'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-300 truncate">{col.name}</span>
              <div className="flex items-center gap-1 shrink-0">
                {col.name === CORE_KNOWLEDGE_COLLECTION && (
                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-blue-900 text-blue-300">System</span>
                )}
                {col.error && (
                  <span className="w-2 h-2 rounded-full bg-red-500" />
                )}
              </div>
            </div>
            <div className="text-xs text-gray-500 space-y-0.5">
              <div>{(col.vectors_count || 0).toLocaleString()} vectors</div>
              {col.dimension && <div>{col.dimension}d</div>}
              {col.points_count > 0 && col.points_count !== col.vectors_count && (
                <div>{col.points_count.toLocaleString()} points</div>
              )}
            </div>
          </button>
        ))}
      </div>

      {/* Collection Detail Panel */}
      {selectedCollection && collectionDetail && (
        <div className="bg-[#2a2a2a] rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-medium text-gray-300">{selectedCollection}</h4>
              {selectedCollection === CORE_KNOWLEDGE_COLLECTION && (
                <span className="px-1.5 py-0.5 rounded text-[10px] bg-blue-900 text-blue-300">System Knowledge</span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <span className={`px-2 py-0.5 rounded text-xs ${
                collectionDetail.status === 'green' ? 'bg-green-900 text-green-300' : 'bg-gray-700 text-gray-300'
              }`}>
                {collectionDetail.status === 'green' ? 'Healthy' : collectionDetail.status === 'yellow' ? 'Optimizing' : collectionDetail.status === 'red' ? 'Error' : collectionDetail.status}
              </span>
              {selectedCollection === CORE_KNOWLEDGE_COLLECTION ? (
                <button
                  onClick={handleToggleCoreKnowledge}
                  className={`px-3 py-1 rounded text-xs transition-colors ${
                    coreKnowledgeEnabled
                      ? 'bg-green-700 hover:bg-green-800 text-green-100'
                      : 'bg-gray-600 hover:bg-gray-700 text-gray-300'
                  }`}
                >
                  {coreKnowledgeEnabled ? 'Enabled' : 'Disabled'}
                </button>
              ) : (
                <button
                  onClick={() => handleDeleteCollection(selectedCollection)}
                  className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-xs transition-colors"
                >
                  Delete
                </button>
              )}
            </div>
          </div>
          {selectedCollection === CORE_KNOWLEDGE_COLLECTION && (
            <p className="text-xs text-gray-500">Built-in knowledge base — helps ARCA answer questions about its own features, pipeline, and configuration.</p>
          )}

          <div className="grid grid-cols-4 gap-4 text-xs">
            <div>
              <span className="text-gray-500 block">Vectors</span>
              <span className="text-gray-300">{collectionDetail.vectors_count.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-gray-500 block">Points</span>
              <span className="text-gray-300">{collectionDetail.points_count.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-gray-500 block">Segments</span>
              <span className="text-gray-300">{collectionDetail.segments_count}</span>
            </div>
            <div>
              <span className="text-gray-500 block">Status</span>
              <span className="text-gray-300">{collectionDetail.status}</span>
            </div>
          </div>

          {/* Search Test */}
          <div className="border-t border-[#3a3a3a] pt-3">
            <span className="text-xs text-gray-500 block mb-2">Search Test</span>
            <div className="flex gap-2">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Enter a search query..."
                className="flex-1 px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-lg text-sm"
              />
              <button
                onClick={handleSearch}
                disabled={searching || !searchQuery.trim()}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-sm transition-colors"
              >
                {searching ? 'Searching...' : 'Search'}
              </button>
            </div>

            {searchResults.length > 0 && (
              <div className="mt-3 space-y-2">
                {searchResults.map((r, i) => (
                  <div key={i} className="bg-[#1a1a1a] rounded-lg p-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-blue-400">Score: {r.score}</span>
                      {r.topic && <span className="text-xs text-gray-500">{r.topic}</span>}
                    </div>
                    <p className="text-xs text-gray-400 line-clamp-3">{r.content}</p>
                    {r.source && (
                      <p className="text-xs text-gray-600 mt-1 truncate">{r.source}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
