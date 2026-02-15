'use client';

import React, { useState } from 'react';
import { CypherQueryResult } from './types';

interface CypherConsoleProps {
  onExecuteQuery: (query: string, allowWrites: boolean) => Promise<CypherQueryResult>;
}

// Example queries for quick access
const EXAMPLE_QUERIES = [
  {
    name: 'Count all nodes',
    query: 'MATCH (n) RETURN count(n) as total',
  },
  {
    name: 'List Standards',
    query: 'MATCH (s:Standard) RETURN s.code, s.name LIMIT 10',
  },
  {
    name: 'Standards with Test Methods',
    query: 'MATCH (s:Standard)-[:REQUIRES]->(t:TestMethod) RETURN s.code, collect(t.name) as methods LIMIT 10',
  },
  {
    name: 'Most connected entities',
    query: 'MATCH (n)-[r]-() RETURN n.name, labels(n)[0] as type, count(r) as connections ORDER BY connections DESC LIMIT 10',
  },
  {
    name: 'Relationship types',
    query: 'MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC',
  },
  {
    name: 'Find entity by name',
    query: "MATCH (n) WHERE n.name CONTAINS 'ASTM' RETURN n.name, labels(n) LIMIT 20",
  },
];

export function CypherConsole({ onExecuteQuery }: CypherConsoleProps) {
  const [query, setQuery] = useState('');
  const [allowWrites, setAllowWrites] = useState(false);
  const [result, setResult] = useState<CypherQueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [queryHistory, setQueryHistory] = useState<string[]>([]);

  const handleExecute = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const res = await onExecuteQuery(query, allowWrites);
      setResult(res);

      // Add to history if not already there
      if (!queryHistory.includes(query)) {
        setQueryHistory((prev) => [query, ...prev.slice(0, 9)]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Ctrl/Cmd + Enter to execute
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleExecute();
    }
  };

  return (
    <div className="space-y-4">
      {/* Query Input */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
            Cypher Query Console
          </h3>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={allowWrites}
                onChange={(e) => setAllowWrites(e.target.checked)}
                className="w-4 h-4 rounded"
              />
              <span className={allowWrites ? 'text-red-400' : 'text-gray-400'}>
                Allow Writes (Dangerous!)
              </span>
            </label>
            <button
              onClick={handleExecute}
              disabled={loading || !query.trim()}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors"
            >
              {loading ? 'Executing...' : 'Execute'} (Ctrl+Enter)
            </button>
          </div>
        </div>

        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter Cypher query..."
          className="w-full h-32 px-4 py-3 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
        />

        {/* Example Queries */}
        <div className="mt-3">
          <p className="text-xs text-gray-400 mb-2">Example queries:</p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_QUERIES.map((example) => (
              <button
                key={example.name}
                onClick={() => setQuery(example.query)}
                className="px-2 py-1 bg-[#1a1a1a] hover:bg-[#333] rounded text-xs transition-colors"
              >
                {example.name}
              </button>
            ))}
          </div>
        </div>

        {/* Query History */}
        {queryHistory.length > 0 && (
          <div className="mt-3">
            <p className="text-xs text-gray-400 mb-2">Recent queries:</p>
            <div className="flex flex-wrap gap-2">
              {queryHistory.map((q, i) => (
                <button
                  key={i}
                  onClick={() => setQuery(q)}
                  className="px-2 py-1 bg-[#1a1a1a] hover:bg-[#333] rounded text-xs transition-colors truncate max-w-[200px]"
                  title={q}
                >
                  {q.length > 30 ? q.slice(0, 30) + '...' : q}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Security Warning */}
      {allowWrites && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-3">
          <div className="flex items-center gap-2 text-red-400">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="text-sm font-medium">Write Mode Enabled</span>
          </div>
          <p className="text-xs text-gray-400 mt-1">
            CREATE, MERGE, DELETE, SET, and other write operations are now allowed.
            Use with extreme caution - changes are permanent!
          </p>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-4">
          <p className="text-sm text-red-400 font-medium">Query Error</p>
          <p className="text-xs text-gray-300 mt-1 font-mono break-all">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="bg-[#2a2a2a] rounded-lg overflow-hidden">
          <div className="p-3 bg-[#1a1a1a] border-b border-[#3a3a3a] flex items-center justify-between">
            <div className="flex items-center gap-4">
              <span className="text-sm font-medium text-green-400">
                {result.success ? 'Success' : 'Failed'}
              </span>
              <span className="text-xs text-gray-400">
                {result.elapsed_ms}ms
              </span>
            </div>
            {result.type === 'read' && result.row_count !== undefined && (
              <span className="text-xs text-gray-400">
                {result.row_count} row{result.row_count !== 1 ? 's' : ''} returned
              </span>
            )}
          </div>

          {/* Read Result */}
          {result.type === 'read' && result.rows && result.rows.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-[#1a1a1a]">
                  <tr className="text-left text-gray-400 text-xs">
                    {Object.keys(result.rows[0]).map((col) => (
                      <th key={col} className="p-3 font-mono">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#3a3a3a]">
                  {result.rows.map((row, i) => (
                    <tr key={i} className="hover:bg-[#333]">
                      {Object.values(row).map((val, j) => (
                        <td key={j} className="p-3 text-xs font-mono">
                          {formatValue(val)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Empty Result */}
          {result.type === 'read' && (!result.rows || result.rows.length === 0) && (
            <div className="p-8 text-center text-gray-500">
              Query returned no results
            </div>
          )}

          {/* Write Result */}
          {result.type === 'write' && result.summary && (
            <div className="p-4">
              <h4 className="text-sm font-medium text-gray-400 mb-2">Write Summary</h4>
              <div className="grid grid-cols-4 gap-4">
                {Object.entries(result.summary).map(([key, value]) => (
                  <div key={key} className="bg-[#1a1a1a] rounded p-3 text-center">
                    <p className="text-lg font-bold text-blue-400">{value}</p>
                    <p className="text-xs text-gray-400">{key.replace(/_/g, ' ')}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function formatValue(val: unknown): string {
  if (val === null || val === undefined) {
    return 'null';
  }
  if (typeof val === 'object') {
    return JSON.stringify(val, null, 2);
  }
  return String(val);
}
