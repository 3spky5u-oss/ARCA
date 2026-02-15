'use client';

import React from 'react';
import { GraphStatsData } from './types';

interface GraphStatsProps {
  stats: GraphStatsData | null;
  loading: boolean;
}

export function GraphStats({ stats, loading }: GraphStatsProps) {
  if (loading && !stats) {
    return (
      <div className="grid grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-[#2a2a2a] rounded-lg p-4 text-center animate-pulse">
            <div className="h-8 bg-[#3a3a3a] rounded mb-2" />
            <div className="h-4 bg-[#3a3a3a] rounded w-16 mx-auto" />
          </div>
        ))}
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="bg-[#2a2a2a] rounded-lg p-4 text-center text-gray-500">
        Unable to load graph statistics
      </div>
    );
  }

  const isConnected = stats.health?.status === 'healthy' || stats.health?.connected;

  return (
    <div className="grid grid-cols-4 gap-4">
      {/* Total Nodes */}
      <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
        <p className="text-2xl font-bold text-blue-400">
          {stats.total_nodes.toLocaleString()}
        </p>
        <p className="text-xs text-gray-400 mt-1">Nodes</p>
      </div>

      {/* Total Relationships */}
      <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
        <p className="text-2xl font-bold text-purple-400">
          {stats.total_relationships.toLocaleString()}
        </p>
        <p className="text-xs text-gray-400 mt-1">Relationships</p>
      </div>

      {/* Average Degree */}
      <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
        <p className="text-2xl font-bold text-green-400">
          {stats.average_degree.toFixed(1)}
        </p>
        <p className="text-xs text-gray-400 mt-1">Avg Connections</p>
      </div>

      {/* Neo4j Status */}
      <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
        <p className={`text-2xl font-bold ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
          {isConnected ? 'Online' : 'Offline'}
        </p>
        <p className="text-xs text-gray-400 mt-1">Neo4j Status</p>
        {stats.health?.error && (
          <p className="text-xs text-red-400 mt-1 truncate" title={stats.health.error}>
            {stats.health.error}
          </p>
        )}
      </div>
    </div>
  );
}

interface NodeTypeBreakdownProps {
  nodeCounts: Record<string, number>;
}

export function NodeTypeBreakdown({ nodeCounts }: NodeTypeBreakdownProps) {
  const sortedTypes = Object.entries(nodeCounts)
    .filter(([, count]) => count > 0)
    .sort(([, a], [, b]) => b - a);

  if (sortedTypes.length === 0) {
    return null;
  }

  // Color mapping for node types
  const typeColors: Record<string, string> = {
    Standard: 'bg-blue-400',
    TestMethod: 'bg-purple-400',
    Concept: 'bg-green-400',
    Parameter: 'bg-amber-400',
    Equipment: 'bg-red-400',
    Chunk: 'bg-gray-400',
  };

  const total = Object.values(nodeCounts).reduce((a, b) => a + b, 0);

  return (
    <div className="bg-[#2a2a2a] rounded-lg p-4">
      <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
        <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
        </svg>
        Node Types
      </h3>
      <div className="space-y-2">
        {sortedTypes.map(([type, count]) => {
          const percentage = total > 0 ? (count / total) * 100 : 0;
          const colorClass = typeColors[type] || 'bg-gray-400';

          return (
            <div key={type} className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${colorClass}`} />
              <span className="text-sm flex-1">{type}</span>
              <span className="text-xs text-gray-400">{count.toLocaleString()}</span>
              <div className="w-20 h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
                <div
                  className={`h-full ${colorClass} transition-all`}
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

interface RelationshipTypeBreakdownProps {
  relationshipCounts: Record<string, number>;
}

export function RelationshipTypeBreakdown({ relationshipCounts }: RelationshipTypeBreakdownProps) {
  const sortedTypes = Object.entries(relationshipCounts)
    .sort(([, a], [, b]) => b - a);

  if (sortedTypes.length === 0) {
    return null;
  }

  const total = Object.values(relationshipCounts).reduce((a, b) => a + b, 0);

  return (
    <div className="bg-[#2a2a2a] rounded-lg p-4">
      <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
        <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
        </svg>
        Relationship Types
      </h3>
      <div className="space-y-2">
        {sortedTypes.map(([type, count]) => {
          const percentage = total > 0 ? (count / total) * 100 : 0;

          return (
            <div key={type} className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-purple-400" />
              <span className="text-sm flex-1 font-mono text-xs">{type}</span>
              <span className="text-xs text-gray-400">{count.toLocaleString()}</span>
              <div className="w-20 h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-400 transition-all"
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
