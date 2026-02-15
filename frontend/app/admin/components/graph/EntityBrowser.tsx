'use client';

import React, { useState } from 'react';
import { GraphEntity, EntityDetails, LabelInfo } from './types';

interface EntityBrowserProps {
  entities: GraphEntity[];
  labels: LabelInfo[];
  selectedEntity: EntityDetails | null;
  totalEntities: number;
  currentOffset: number;
  pageSize: number;
  searchTerm: string;
  selectedType: string;
  loading: boolean;
  onSearch: (term: string) => void;
  onTypeFilter: (type: string) => void;
  onSelectEntity: (name: string) => void;
  onPageChange: (offset: number) => void;
}

export function EntityBrowser({
  entities,
  labels,
  selectedEntity,
  totalEntities,
  currentOffset,
  pageSize,
  searchTerm,
  selectedType,
  loading,
  onSearch,
  onTypeFilter,
  onSelectEntity,
  onPageChange,
}: EntityBrowserProps) {
  const [localSearch, setLocalSearch] = useState(searchTerm);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(localSearch);
  };

  const totalPages = Math.ceil(totalEntities / pageSize);
  const currentPage = Math.floor(currentOffset / pageSize) + 1;

  return (
    <div className="grid grid-cols-2 gap-4">
      {/* Left: Entity List */}
      <div className="bg-[#2a2a2a] rounded-lg overflow-hidden">
        {/* Search and Filter */}
        <div className="p-4 bg-[#1a1a1a] border-b border-[#3a3a3a]">
          <form onSubmit={handleSearchSubmit} className="flex gap-2 mb-3">
            <input
              type="text"
              value={localSearch}
              onChange={(e) => setLocalSearch(e.target.value)}
              placeholder="Search entities..."
              className="flex-1 px-3 py-2 bg-[#2a2a2a] border border-[#3a3a3a] rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm transition-colors"
            >
              Search
            </button>
          </form>

          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => onTypeFilter('')}
              className={`px-3 py-1 rounded-full text-xs transition-colors ${
                !selectedType
                  ? 'bg-blue-600 text-white'
                  : 'bg-[#2a2a2a] text-gray-400 hover:text-white'
              }`}
            >
              All
            </button>
            {labels.slice(0, 6).map((label) => (
              <button
                key={label.label}
                onClick={() => onTypeFilter(label.label)}
                className={`px-3 py-1 rounded-full text-xs transition-colors ${
                  selectedType === label.label
                    ? 'bg-blue-600 text-white'
                    : 'bg-[#2a2a2a] text-gray-400 hover:text-white'
                }`}
              >
                {label.label} ({label.count})
              </button>
            ))}
          </div>
        </div>

        {/* Entity List */}
        <div className="max-h-96 overflow-y-auto">
          {loading ? (
            <div className="p-4 text-center text-gray-500">
              <svg className="animate-spin h-5 w-5 mx-auto mb-2" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Loading...
            </div>
          ) : entities.length === 0 ? (
            <div className="p-4 text-center text-gray-500">
              No entities found
            </div>
          ) : (
            <div className="divide-y divide-[#3a3a3a]">
              {entities.map((entity) => (
                <button
                  key={entity.id}
                  onClick={() => onSelectEntity(entity.name)}
                  className={`w-full p-3 text-left hover:bg-[#333] transition-colors ${
                    selectedEntity?.name === entity.name ? 'bg-[#333]' : ''
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{entity.name}</span>
                    <span className="text-xs text-gray-400">
                      {entity.relationship_count} rels
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {entity.labels.map((label) => (
                      <span
                        key={label}
                        className={`px-1.5 py-0.5 rounded text-[10px] ${getLabelColor(label)}`}
                      >
                        {label}
                      </span>
                    ))}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

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
              Page {currentPage} of {totalPages} ({totalEntities} total)
            </span>
            <button
              onClick={() => onPageChange(currentOffset + pageSize)}
              disabled={currentOffset + pageSize >= totalEntities}
              className="px-3 py-1 bg-[#2a2a2a] hover:bg-[#333] disabled:opacity-50 disabled:cursor-not-allowed rounded text-xs transition-colors"
            >
              Next
            </button>
          </div>
        )}
      </div>

      {/* Right: Entity Detail Panel */}
      <div className="bg-[#2a2a2a] rounded-lg overflow-hidden">
        {selectedEntity ? (
          <EntityDetailPanel entity={selectedEntity} />
        ) : (
          <div className="h-full flex items-center justify-center text-gray-500 p-8">
            <div className="text-center">
              <svg className="w-12 h-12 mx-auto mb-2 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Select an entity to view details
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function EntityDetailPanel({ entity }: { entity: EntityDetails }) {
  const [showAllRelationships, setShowAllRelationships] = useState(false);

  const displayedRelationships = showAllRelationships
    ? entity.relationships
    : entity.relationships.slice(0, 10);

  return (
    <div className="h-full overflow-y-auto">
      {/* Header */}
      <div className="p-4 bg-[#1a1a1a] border-b border-[#3a3a3a]">
        <h3 className="text-lg font-medium">{entity.name}</h3>
        <div className="flex flex-wrap gap-1 mt-2">
          {entity.labels.map((label) => (
            <span
              key={label}
              className={`px-2 py-0.5 rounded text-xs ${getLabelColor(label)}`}
            >
              {label}
            </span>
          ))}
        </div>
      </div>

      {/* Properties */}
      <div className="p-4 border-b border-[#3a3a3a]">
        <h4 className="text-sm font-medium text-gray-400 mb-2">Properties</h4>
        <div className="space-y-1">
          {Object.entries(entity.properties).map(([key, value]) => (
            <div key={key} className="flex gap-2">
              <span className="text-xs text-gray-500 min-w-[100px]">{key}:</span>
              <span className="text-xs text-gray-300 break-all">
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Relationships */}
      <div className="p-4">
        <h4 className="text-sm font-medium text-gray-400 mb-2">
          Relationships ({entity.relationships.length})
        </h4>
        <div className="space-y-2">
          {displayedRelationships.map((rel, i) => (
            <div key={i} className="bg-[#1a1a1a] rounded p-2 text-xs">
              <div className="flex items-center gap-2">
                <span className={rel.direction === 'outgoing' ? 'text-green-400' : 'text-blue-400'}>
                  {rel.direction === 'outgoing' ? '→' : '←'}
                </span>
                <span className="font-mono text-purple-400">{rel.type}</span>
                <span className={rel.direction === 'outgoing' ? 'text-green-400' : 'text-blue-400'}>
                  {rel.direction === 'outgoing' ? '→' : '←'}
                </span>
                <span className="text-gray-300">
                  {rel.target_name || rel.target_code || 'Unknown'}
                </span>
              </div>
              {rel.target_labels && rel.target_labels.length > 0 && (
                <div className="ml-4 mt-1">
                  {rel.target_labels.map((label) => (
                    <span
                      key={label}
                      className={`px-1 py-0.5 rounded text-[10px] mr-1 ${getLabelColor(label)}`}
                    >
                      {label}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
          {entity.relationships.length > 10 && !showAllRelationships && (
            <button
              onClick={() => setShowAllRelationships(true)}
              className="w-full py-2 text-xs text-blue-400 hover:text-blue-300"
            >
              Show all {entity.relationships.length} relationships
            </button>
          )}
        </div>
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
