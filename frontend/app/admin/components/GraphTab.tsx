'use client';

import React, { useState, useCallback, useEffect } from 'react';
import type {
  GraphStatsData,
  GraphEntity,
  EntityDetails,
  VisualizationData,
  GraphRelationship,
  LabelInfo,
  RelationshipTypeInfo,
  CypherQueryResult,
  GraphSubTab,
} from './graph';
import {
  GraphStats,
  NodeTypeBreakdown,
  RelationshipTypeBreakdown,
  EntityBrowser,
  GraphViewer,
  RelationshipExplorer,
  CypherConsole,
} from './graph';

// Use shared API base (respects NEXT_PUBLIC_API_URL for reverse proxy)
import { getApiBase } from '@/lib/api';

interface GraphTabProps {
  apiCall: (endpoint: string, options?: RequestInit) => Promise<unknown>;
  setMessage: (message: { type: 'success' | 'error'; text: string } | null) => void;
}

export function GraphTab({ apiCall, setMessage }: GraphTabProps) {
  // Sub-tab state
  const [activeSubTab, setActiveSubTab] = useState<GraphSubTab>('entities');

  // Stats state
  const [stats, setStats] = useState<GraphStatsData | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);

  // Labels and relationship types
  const [labels, setLabels] = useState<LabelInfo[]>([]);
  const [relationshipTypes, setRelationshipTypes] = useState<RelationshipTypeInfo[]>([]);

  // Entity browser state
  const [entities, setEntities] = useState<GraphEntity[]>([]);
  const [selectedEntity, setSelectedEntity] = useState<EntityDetails | null>(null);
  const [entitySearchTerm, setEntitySearchTerm] = useState('');
  const [entityTypeFilter, setEntityTypeFilter] = useState('');
  const [entityOffset, setEntityOffset] = useState(0);
  const [totalEntities, setTotalEntities] = useState(0);
  const [entitiesLoading, setEntitiesLoading] = useState(false);
  const PAGE_SIZE = 50;

  // Visualization state
  const [vizData, setVizData] = useState<VisualizationData | null>(null);
  const [vizLoading, setVizLoading] = useState(false);
  const [vizTypeFilter, setVizTypeFilter] = useState('');
  const [includeChunks, setIncludeChunks] = useState(false);
  const [nodeLimit, setNodeLimit] = useState(100);

  // Relationships state
  const [relationships, setRelationships] = useState<GraphRelationship[]>([]);
  const [relTypeFilter, setRelTypeFilter] = useState('');
  const [relSourceTypeFilter, setRelSourceTypeFilter] = useState('');
  const [relTargetTypeFilter, setRelTargetTypeFilter] = useState('');
  const [relOffset, setRelOffset] = useState(0);
  const [totalRelationships, setTotalRelationships] = useState(0);
  const [relsLoading, setRelsLoading] = useState(false);

  // =============================================================================
  // DATA FETCHING
  // =============================================================================

  const fetchStats = useCallback(async () => {
    setStatsLoading(true);
    try {
      const data = await apiCall('/api/admin/graph/stats') as GraphStatsData;
      setStats(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch graph stats: ${e}` });
    } finally {
      setStatsLoading(false);
    }
  }, [apiCall, setMessage]);

  const fetchLabels = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/graph/labels') as { labels: LabelInfo[] };
      setLabels(data.labels || []);
    } catch (e) {
      console.error('Failed to fetch labels:', e);
    }
  }, [apiCall]);

  const fetchRelationshipTypes = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/graph/relationship-types') as { types: RelationshipTypeInfo[] };
      setRelationshipTypes(data.types || []);
    } catch (e) {
      console.error('Failed to fetch relationship types:', e);
    }
  }, [apiCall]);

  const fetchEntities = useCallback(async () => {
    setEntitiesLoading(true);
    try {
      const params = new URLSearchParams({
        limit: String(PAGE_SIZE),
        offset: String(entityOffset),
      });
      if (entityTypeFilter) params.set('entity_type', entityTypeFilter);
      if (entitySearchTerm) params.set('search', entitySearchTerm);

      const data = await apiCall(`/api/admin/graph/entities?${params}`) as {
        entities: GraphEntity[];
        total: number;
      };
      setEntities(data.entities || []);
      setTotalEntities(data.total || 0);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch entities: ${e}` });
    } finally {
      setEntitiesLoading(false);
    }
  }, [apiCall, setMessage, entityOffset, entityTypeFilter, entitySearchTerm]);

  const fetchEntityDetails = useCallback(async (name: string) => {
    try {
      const data = await apiCall(`/api/admin/graph/entity/${encodeURIComponent(name)}`) as EntityDetails;
      setSelectedEntity(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch entity details: ${e}` });
    }
  }, [apiCall, setMessage]);

  const fetchVisualization = useCallback(async () => {
    setVizLoading(true);
    try {
      const params = new URLSearchParams({
        limit: String(nodeLimit),
        include_chunks: String(includeChunks),
      });
      if (vizTypeFilter) params.set('entity_type', vizTypeFilter);

      const data = await apiCall(`/api/admin/graph/visualization?${params}`) as VisualizationData;
      setVizData(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch visualization: ${e}` });
    } finally {
      setVizLoading(false);
    }
  }, [apiCall, setMessage, nodeLimit, includeChunks, vizTypeFilter]);

  const fetchRelationships = useCallback(async () => {
    setRelsLoading(true);
    try {
      const params = new URLSearchParams({
        limit: String(PAGE_SIZE),
        offset: String(relOffset),
      });
      if (relTypeFilter) params.set('rel_type', relTypeFilter);
      if (relSourceTypeFilter) params.set('source_type', relSourceTypeFilter);
      if (relTargetTypeFilter) params.set('target_type', relTargetTypeFilter);

      const data = await apiCall(`/api/admin/graph/relationships?${params}`) as {
        relationships: GraphRelationship[];
        total: number;
      };
      setRelationships(data.relationships || []);
      setTotalRelationships(data.total || 0);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch relationships: ${e}` });
    } finally {
      setRelsLoading(false);
    }
  }, [apiCall, setMessage, relOffset, relTypeFilter, relSourceTypeFilter, relTargetTypeFilter]);

  const executeCypherQuery = useCallback(async (query: string, allowWrites: boolean): Promise<CypherQueryResult> => {
    const params = new URLSearchParams();
    if (allowWrites) params.set('allow_writes', 'true');

    const result = await apiCall(`/api/admin/graph/query?${params}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    }) as CypherQueryResult;

    return result;
  }, [apiCall]);

  // =============================================================================
  // EFFECTS
  // =============================================================================

  // Load stats and labels on mount
  useEffect(() => {
    fetchStats();
    fetchLabels();
    fetchRelationshipTypes();
  }, [fetchStats, fetchLabels, fetchRelationshipTypes]);

  // Load data when sub-tab changes
  useEffect(() => {
    if (activeSubTab === 'entities') {
      fetchEntities();
    } else if (activeSubTab === 'graph') {
      if (!vizData) fetchVisualization();
    } else if (activeSubTab === 'relationships') {
      fetchRelationships();
    }
  }, [activeSubTab, fetchEntities, fetchVisualization, fetchRelationships, vizData]);

  // Refetch entities when filters change
  useEffect(() => {
    if (activeSubTab === 'entities') {
      fetchEntities();
    }
  }, [entityOffset, entityTypeFilter, entitySearchTerm, fetchEntities, activeSubTab]);

  // Refetch relationships when filters change
  useEffect(() => {
    if (activeSubTab === 'relationships') {
      fetchRelationships();
    }
  }, [relOffset, relTypeFilter, relSourceTypeFilter, relTargetTypeFilter, fetchRelationships, activeSubTab]);

  // =============================================================================
  // HANDLERS
  // =============================================================================

  const handleEntitySearch = (term: string) => {
    setEntitySearchTerm(term);
    setEntityOffset(0);
  };

  const handleEntityTypeFilter = (type: string) => {
    setEntityTypeFilter(type);
    setEntityOffset(0);
  };

  const handleSelectEntity = (name: string) => {
    fetchEntityDetails(name);
  };

  const handleEntityPageChange = (offset: number) => {
    setEntityOffset(offset);
  };

  const handleVizRefresh = () => {
    fetchVisualization();
  };

  const handleVizTypeFilter = (type: string) => {
    setVizTypeFilter(type);
    // Don't auto-refresh - let user click refresh
  };

  const handleNodeClick = (name: string) => {
    setActiveSubTab('entities');
    setEntitySearchTerm(name);
    fetchEntityDetails(name);
  };

  const handleRelTypeFilter = (type: string) => {
    setRelTypeFilter(type);
    setRelOffset(0);
  };

  const handleRelSourceTypeFilter = (type: string) => {
    setRelSourceTypeFilter(type);
    setRelOffset(0);
  };

  const handleRelTargetTypeFilter = (type: string) => {
    setRelTargetTypeFilter(type);
    setRelOffset(0);
  };

  const handleRelPageChange = (offset: number) => {
    setRelOffset(offset);
  };

  const handleRelEntityClick = (name: string) => {
    setActiveSubTab('entities');
    setEntitySearchTerm(name);
    fetchEntityDetails(name);
  };

  // =============================================================================
  // RENDER
  // =============================================================================

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium flex items-center gap-2">
          <svg className="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
          </svg>
          Knowledge Graph (Neo4j)
        </h2>
        <button
          onClick={fetchStats}
          disabled={statsLoading}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-xl text-sm transition-colors"
        >
          Refresh Stats
        </button>
      </div>

      {/* Stats Cards */}
      <GraphStats stats={stats} loading={statsLoading} />

      {/* Node & Relationship Breakdowns */}
      {stats && (
        <div className="grid grid-cols-2 gap-4">
          <NodeTypeBreakdown nodeCounts={stats.node_counts} />
          <RelationshipTypeBreakdown relationshipCounts={stats.relationship_counts} />
        </div>
      )}

      {/* Sub-tab Navigation */}
      <div className="flex gap-2 border-b border-[#3a3a3a] pb-2">
        {(['entities', 'graph', 'relationships', 'query'] as GraphSubTab[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveSubTab(tab)}
            className={`px-4 py-2 rounded-t-xl text-sm transition-colors ${
              activeSubTab === tab
                ? 'bg-[#2a2a2a] text-white'
                : 'text-gray-400 hover:text-white hover:bg-[#2a2a2a]/50'
            }`}
          >
            {tab === 'entities' && 'Entities'}
            {tab === 'graph' && 'Graph View'}
            {tab === 'relationships' && 'Relationships'}
            {tab === 'query' && 'Cypher Console'}
          </button>
        ))}
      </div>

      {/* Sub-tab Content */}
      {activeSubTab === 'entities' && (
        <EntityBrowser
          entities={entities}
          labels={labels}
          selectedEntity={selectedEntity}
          totalEntities={totalEntities}
          currentOffset={entityOffset}
          pageSize={PAGE_SIZE}
          searchTerm={entitySearchTerm}
          selectedType={entityTypeFilter}
          loading={entitiesLoading}
          onSearch={handleEntitySearch}
          onTypeFilter={handleEntityTypeFilter}
          onSelectEntity={handleSelectEntity}
          onPageChange={handleEntityPageChange}
        />
      )}

      {activeSubTab === 'graph' && (
        <GraphViewer
          data={vizData}
          labels={labels}
          loading={vizLoading}
          selectedType={vizTypeFilter}
          includeChunks={includeChunks}
          nodeLimit={nodeLimit}
          onRefresh={handleVizRefresh}
          onTypeFilter={handleVizTypeFilter}
          onIncludeChunksChange={setIncludeChunks}
          onNodeLimitChange={setNodeLimit}
          onNodeClick={handleNodeClick}
        />
      )}

      {activeSubTab === 'relationships' && (
        <RelationshipExplorer
          relationships={relationships}
          relationshipTypes={relationshipTypes}
          labels={labels}
          totalRelationships={totalRelationships}
          currentOffset={relOffset}
          pageSize={PAGE_SIZE}
          selectedRelType={relTypeFilter}
          selectedSourceType={relSourceTypeFilter}
          selectedTargetType={relTargetTypeFilter}
          loading={relsLoading}
          onRelTypeFilter={handleRelTypeFilter}
          onSourceTypeFilter={handleRelSourceTypeFilter}
          onTargetTypeFilter={handleRelTargetTypeFilter}
          onPageChange={handleRelPageChange}
          onEntityClick={handleRelEntityClick}
        />
      )}

      {activeSubTab === 'query' && (
        <CypherConsole onExecuteQuery={executeCypherQuery} />
      )}
    </div>
  );
}
