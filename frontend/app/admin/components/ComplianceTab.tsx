'use client';

import React from 'react';

interface GuidelineFile {
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

interface GuidelineStats {
  guidelines_dir: string;
  dir_exists: boolean;
  files: GuidelineFile[];
  total_entries: number;
}

interface GuidelineSearchResult {
  query: string;
  total_matches: number;
  unique_parameters: number;
  results: Array<{
    parameter: string;
    soil_type: string;
    land_use: string;
    value: number;
    table?: string;
  }>;
  grouped: Record<string, Array<{
    table: string;
    soil_type: string;
    land_use: string;
    value: number;
  }>>;
}

interface GuidelineComparisonResult {
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

interface ComplianceTabProps {
  stats: GuidelineStats | null;
  expandedTables: Set<string>;
  onToggleTable: (tableName: string) => void;
  guidelineSearch: string;
  onGuidelineSearchChange: (search: string) => void;
  guidelineSearchResult: GuidelineSearchResult | null;
  onGuidelineSearch: () => void;
  comparisonFile: File | null;
  onComparisonFileChange: (file: File | null) => void;
  comparisonResult: GuidelineComparisonResult | null;
  isComparing: boolean;
  onGuidelineComparison: () => void;
  onRefresh: () => void;
}

export function ComplianceTab({
  stats,
  expandedTables,
  onToggleTable,
  guidelineSearch,
  onGuidelineSearchChange,
  guidelineSearchResult,
  onGuidelineSearch,
  comparisonFile,
  onComparisonFileChange,
  comparisonResult,
  isComparing,
  onGuidelineComparison,
  onRefresh,
}: ComplianceTabProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">Compliance Guidelines (Exceedee)</h2>
        <button
          onClick={onRefresh}
          className="px-4 py-2 bg-[#2a2a2a] hover:bg-[#333] rounded-xl text-sm transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Overview Stats */}
      {stats && (
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-blue-400">{stats.total_entries.toLocaleString()}</p>
            <p className="text-xs text-gray-400 mt-1">Total Entries</p>
          </div>
          <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-green-400">{stats.files.length}</p>
            <p className="text-xs text-gray-400 mt-1">Tables Loaded</p>
          </div>
          <div className="bg-[#2a2a2a] rounded-lg p-4 text-center">
            <p className={`text-2xl font-bold ${stats.dir_exists ? 'text-green-400' : 'text-red-400'}`}>
              {stats.dir_exists ? 'OK' : 'Missing'}
            </p>
            <p className="text-xs text-gray-400 mt-1">Guidelines Dir</p>
          </div>
        </div>
      )}

      {/* Guidelines Tables */}
      {stats && stats.files.length > 0 && (
        <div className="bg-[#2a2a2a] rounded-lg overflow-hidden">
          <div className="px-4 py-3 bg-[#1a1a1a] border-b border-[#3a3a3a]">
            <h3 className="text-sm font-medium">Loaded Guidelines</h3>
          </div>
          <div className="divide-y divide-[#3a3a3a]">
            {stats.files.map((file) => (
              <div key={file.name}>
                <div
                  className="px-4 py-3 flex items-center justify-between hover:bg-[#333] cursor-pointer"
                  onClick={() => onToggleTable(file.name)}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-gray-400">
                      {expandedTables.has(file.name) ? '\u25BC' : '\u25B6'}
                    </span>
                    <span className="font-medium">{file.filename}</span>
                    {file.error ? (
                      <span className="text-xs text-red-400">Error: {file.error}</span>
                    ) : (
                      <span className="text-xs text-gray-500">
                        ({file.entries.toLocaleString()} entries, {file.parameters} parameters)
                      </span>
                    )}
                  </div>
                  <span className="text-xs text-gray-500">{file.size_kb} KB</span>
                </div>

                {/* Expanded table info */}
                {expandedTables.has(file.name) && !file.error && (
                  <div className="bg-[#1a1a1a] px-4 py-3">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Categories</p>
                        <div className="flex flex-wrap gap-1">
                          {file.soil_types.map((st) => (
                            <span key={st} className="px-2 py-0.5 bg-[#2a2a2a] rounded-full text-xs">
                              {st}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Land Uses</p>
                        <div className="flex flex-wrap gap-1">
                          {file.land_uses.map((lu) => (
                            <span key={lu} className="px-2 py-0.5 bg-[#2a2a2a] rounded-full text-xs">
                              {lu}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {stats && stats.files.length === 0 && (
        <div className="bg-[#2a2a2a] rounded-lg p-8 text-center text-gray-400">
          No guideline files found in {stats.guidelines_dir}
        </div>
      )}

      {/* Guideline Search */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-sm font-medium mb-4">Search Guidelines</h3>

        <div className="flex gap-4 items-end mb-4">
          <div className="flex-1">
            <label className="block text-xs text-gray-400 mb-2">Parameter Name</label>
            <input
              type="text"
              value={guidelineSearch}
              onChange={(e) => onGuidelineSearchChange(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && onGuidelineSearch()}
              placeholder="e.g., benzene, arsenic, lead..."
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm h-[42px]"
            />
          </div>
          <button
            onClick={onGuidelineSearch}
            disabled={!guidelineSearch.trim()}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors h-[42px]"
          >
            Search
          </button>
        </div>

        {/* Search Results */}
        {guidelineSearchResult && (
          <div className="space-y-3">
            <div className="flex items-center gap-4 text-xs text-gray-400">
              <span>Query: &quot;{guidelineSearchResult.query}&quot;</span>
              <span>{guidelineSearchResult.total_matches} matches</span>
              <span>{guidelineSearchResult.unique_parameters} unique parameters</span>
            </div>

            {Object.entries(guidelineSearchResult.grouped).map(([param, entries]) => (
              <div key={param} className="bg-[#1a1a1a] rounded-lg p-3">
                <h4 className="text-sm font-medium mb-2 text-blue-400">{param}</h4>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {entries.map((entry, i) => (
                    <div key={i} className="flex justify-between py-1 border-b border-[#333]">
                      <span className="text-gray-400">
                        {entry.soil_type} / {entry.land_use}
                      </span>
                      <span className="font-mono">
                        {typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value} mg/kg
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {guidelineSearchResult.total_matches === 0 && (
              <div className="text-center text-gray-500 py-4">No guidelines found for &quot;{guidelineSearchResult.query}&quot;</div>
            )}
          </div>
        )}
      </div>

      {/* Guideline Comparison */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-sm font-medium mb-4">Guideline Comparison</h3>
        <p className="text-xs text-gray-400 mb-4">
          Upload a new guideline PDF to compare against current guidelines and identify changes.
        </p>

        <div className="flex gap-4 items-end mb-4">
          <div className="flex-1">
            <label className="block text-xs text-gray-400 mb-2">Guideline PDF</label>
            <input
              type="file"
              accept=".pdf"
              onChange={(e) => onComparisonFileChange(e.target.files?.[0] || null)}
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm file:mr-4 file:py-1 file:px-3 file:rounded-xl file:border-0 file:bg-blue-600 file:text-white h-[42px]"
            />
          </div>
          <button
            onClick={onGuidelineComparison}
            disabled={!comparisonFile || isComparing}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors h-[42px]"
          >
            {isComparing ? 'Comparing...' : 'Compare'}
          </button>
        </div>

        {/* Comparison Results */}
        {comparisonResult && (
          <div className="space-y-4">
            {comparisonResult.error && (
              <div className="bg-red-900/20 border border-red-600/30 rounded-lg p-3 text-red-400 text-sm">
                {comparisonResult.error}
              </div>
            )}

            {comparisonResult.success && comparisonResult.comparison && (
              <>
                {/* Summary Stats */}
                <div className="grid grid-cols-4 gap-4">
                  <div className="bg-[#1a1a1a] rounded-lg p-3 text-center">
                    <p className="text-xl font-bold text-amber-400">{comparisonResult.comparison.changed.length}</p>
                    <p className="text-xs text-gray-400">Changed</p>
                  </div>
                  <div className="bg-[#1a1a1a] rounded-lg p-3 text-center">
                    <p className="text-xl font-bold text-green-400">{comparisonResult.comparison.added.length}</p>
                    <p className="text-xs text-gray-400">New</p>
                  </div>
                  <div className="bg-[#1a1a1a] rounded-lg p-3 text-center">
                    <p className="text-xl font-bold text-red-400">{comparisonResult.comparison.removed.length}</p>
                    <p className="text-xs text-gray-400">Removed</p>
                  </div>
                  <div className="bg-[#1a1a1a] rounded-lg p-3 text-center">
                    <p className="text-xl font-bold text-gray-400">{comparisonResult.comparison.unchanged_count}</p>
                    <p className="text-xs text-gray-400">Unchanged</p>
                  </div>
                </div>

                {/* Changed Limits */}
                {comparisonResult.comparison.changed.length > 0 && (
                  <div className="bg-[#1a1a1a] rounded-lg p-3">
                    <h4 className="text-sm font-medium mb-3 text-amber-400">Changed Limits</h4>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {comparisonResult.comparison.changed.map((c, i) => (
                        <div key={i} className="flex items-center justify-between py-1 border-b border-[#333] text-xs">
                          <div>
                            <span className="font-medium">{c.parameter}</span>
                            <span className="text-gray-500 ml-2">({c.soil_type}/{c.land_use})</span>
                          </div>
                          <div className="flex items-center gap-4">
                            <span className="text-gray-400">{c.old_value}</span>
                            <span className="text-gray-500">→</span>
                            <span className="font-medium">{c.new_value}</span>
                            <span className={`px-2 py-0.5 rounded-full ${c.pct_change > 0 ? 'bg-red-600/20 text-red-400' : 'bg-green-600/20 text-green-400'}`}>
                              {c.pct_change > 0 ? '+' : ''}{c.pct_change.toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* New Parameters */}
                {comparisonResult.comparison.added.length > 0 && (
                  <div className="bg-[#1a1a1a] rounded-lg p-3">
                    <h4 className="text-sm font-medium mb-3 text-green-400">New Parameters</h4>
                    <div className="grid grid-cols-2 gap-2 text-xs max-h-40 overflow-y-auto">
                      {comparisonResult.comparison.added.map((a, i) => (
                        <div key={i} className="flex justify-between py-1 border-b border-[#333]">
                          <span>{a.parameter} ({a.soil_type}/{a.land_use})</span>
                          <span className="font-mono">{a.limit_value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Removed Parameters */}
                {comparisonResult.comparison.removed.length > 0 && (
                  <div className="bg-[#1a1a1a] rounded-lg p-3">
                    <h4 className="text-sm font-medium mb-3 text-red-400">Removed Parameters</h4>
                    <div className="grid grid-cols-2 gap-2 text-xs max-h-40 overflow-y-auto">
                      {comparisonResult.comparison.removed.map((r, i) => (
                        <div key={i} className="flex justify-between py-1 border-b border-[#333]">
                          <span>{r.parameter} ({r.soil_type}/{r.land_use})</span>
                          <span className="font-mono text-gray-500 line-through">{r.limit_value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {comparisonResult.success && !comparisonResult.comparison && (
              <div className="text-center text-gray-400 py-4">
                Extracted {comparisonResult.extracted_count} entries. No existing guidelines to compare against.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
