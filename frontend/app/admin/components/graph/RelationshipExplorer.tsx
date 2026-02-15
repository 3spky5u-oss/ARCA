'use client';

import React, { useState } from 'react';
import { GraphRelationship, RelationshipTypeInfo, LabelInfo } from './types';

interface RelationshipExplorerProps {
  relationships: GraphRelationship[];
  relationshipTypes: RelationshipTypeInfo[];
  labels: LabelInfo[];
  totalRelationships: number;
  currentOffset: number;
  pageSize: number;
  selectedRelType: string;
  selectedSourceType: string;
  selectedTargetType: string;
  loading: boolean;
  onRelTypeFilter: (type: string) => void;
  onSourceTypeFilter: (type: string) => void;
  onTargetTypeFilter: (type: string) => void;
  onPageChange: (offset: number) => void;
  onEntityClick: (name: string) => void;
}

export function RelationshipExplorer({
  relationships,
  relationshipTypes,
  labels,
  totalRelationships,
  currentOffset,
  pageSize,
  selectedRelType,
  selectedSourceType,
  selectedTargetType,
  loading,
  onRelTypeFilter,
  onSourceTypeFilter,
  onTargetTypeFilter,
  onPageChange,
  onEntityClick,
}: RelationshipExplorerProps) {
  const totalPages = Math.ceil(totalRelationships / pageSize);
  const currentPage = Math.floor(currentOffset / pageSize) + 1;

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-sm font-medium mb-3">Filters</h3>

        <div className="grid grid-cols-3 gap-4">
          {/* Relationship Type Filter */}
          <div>
            <label className="block text-xs text-gray-400 mb-2">Relationship Type</label>
            <select
              value={selectedRelType}
              onChange={(e) => onRelTypeFilter(e.target.value)}
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Types</option>
              {relationshipTypes.map((rt) => (
                <option key={rt.type} value={rt.type}>
                  {rt.type} ({rt.count})
                </option>
              ))}
            </select>
          </div>

          {/* Source Type Filter */}
          <div>
            <label className="block text-xs text-gray-400 mb-2">Source Node Type</label>
            <select
              value={selectedSourceType}
              onChange={(e) => onSourceTypeFilter(e.target.value)}
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Types</option>
              {labels.map((label) => (
                <option key={label.label} value={label.label}>
                  {label.label} ({label.count})
                </option>
              ))}
            </select>
          </div>

          {/* Target Type Filter */}
          <div>
            <label className="block text-xs text-gray-400 mb-2">Target Node Type</label>
            <select
              value={selectedTargetType}
              onChange={(e) => onTargetTypeFilter(e.target.value)}
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Types</option>
              {labels.map((label) => (
                <option key={label.label} value={label.label}>
                  {label.label} ({label.count})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Relationships Table */}
      <div className="bg-[#2a2a2a] rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-[#1a1a1a]">
            <tr className="text-left text-gray-400 text-xs">
              <th className="p-3">Source</th>
              <th className="p-3">Relationship</th>
              <th className="p-3">Target</th>
              <th className="p-3">Properties</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#3a3a3a]">
            {loading ? (
              <tr>
                <td colSpan={4} className="p-8 text-center text-gray-500">
                  <svg className="animate-spin h-5 w-5 mx-auto mb-2" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Loading...
                </td>
              </tr>
            ) : relationships.length === 0 ? (
              <tr>
                <td colSpan={4} className="p-8 text-center text-gray-500">
                  No relationships found
                </td>
              </tr>
            ) : (
              relationships.map((rel, i) => (
                <tr key={i} className="hover:bg-[#333]">
                  <td className="p-3">
                    <button
                      onClick={() => onEntityClick(rel.source)}
                      className="text-blue-400 hover:text-blue-300 hover:underline"
                    >
                      {rel.source}
                    </button>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {rel.source_labels.map((label) => (
                        <span
                          key={label}
                          className={`px-1 py-0.5 rounded text-[10px] ${getLabelColor(label)}`}
                        >
                          {label}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="p-3">
                    <span className="px-2 py-1 bg-purple-500/30 text-purple-300 rounded font-mono text-xs">
                      {rel.relationship}
                    </span>
                  </td>
                  <td className="p-3">
                    <button
                      onClick={() => onEntityClick(rel.target)}
                      className="text-blue-400 hover:text-blue-300 hover:underline"
                    >
                      {rel.target}
                    </button>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {rel.target_labels.map((label) => (
                        <span
                          key={label}
                          className={`px-1 py-0.5 rounded text-[10px] ${getLabelColor(label)}`}
                        >
                          {label}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="p-3">
                    {Object.keys(rel.properties).length > 0 ? (
                      <div className="text-xs text-gray-400">
                        {Object.entries(rel.properties).slice(0, 2).map(([k, v]) => (
                          <div key={k}>{k}: {String(v)}</div>
                        ))}
                        {Object.keys(rel.properties).length > 2 && (
                          <div className="text-gray-500">+{Object.keys(rel.properties).length - 2} more</div>
                        )}
                      </div>
                    ) : (
                      <span className="text-gray-500">-</span>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="p-3 bg-[#1a1a1a] border-t border-[#3a3a3a] flex items-center justify-between">
            <button
              onClick={() => onPageChange(Math.max(0, currentOffset - pageSize))}
              disabled={currentOffset === 0}
              className="px-3 py-1 bg-[#2a2a2a] hover:bg-[#333] disabled:opacity-50 disabled:cursor-not-allowed rounded text-xs transition-colors"
            >
              Previous
            </button>
            <span className="text-xs text-gray-400">
              Page {currentPage} of {totalPages} ({totalRelationships} total)
            </span>
            <button
              onClick={() => onPageChange(currentOffset + pageSize)}
              disabled={currentOffset + pageSize >= totalRelationships}
              className="px-3 py-1 bg-[#2a2a2a] hover:bg-[#333] disabled:opacity-50 disabled:cursor-not-allowed rounded text-xs transition-colors"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function getLabelColor(label: string): string {
  const colors: Record<string, string> = {
    Standard: 'bg-blue-500/30 text-blue-300',
    TestMethod: 'bg-purple-500/30 text-purple-300',
    Concept: 'bg-green-500/30 text-green-300',
    Parameter: 'bg-amber-500/30 text-amber-300',
    Equipment: 'bg-red-500/30 text-red-300',
    Chunk: 'bg-gray-500/30 text-gray-300',
  };
  return colors[label] || 'bg-gray-500/30 text-gray-300';
}
