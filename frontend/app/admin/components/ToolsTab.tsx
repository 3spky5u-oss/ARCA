'use client';

import React, { useMemo, useState } from 'react';

export interface AdminToolInfo {
  name: string;
  friendly_name: string;
  brief: string;
  description: string;
  category: string;
  required_params: string[];
  parameters: Record<string, unknown>;
  source: string;
  executor_module: string;
  requires_config?: string | null;
}

export interface CustomToolInfo {
  name: string;
  module: string;
  executor: string;
  friendly_name: string;
  brief: string;
  description: string;
  category: string;
  parameters: Record<string, unknown>;
  required_params: string[];
  enabled: boolean;
}

export interface AdminToolsData {
  domain: string;
  tool_count: number;
  tools: AdminToolInfo[];
  custom_tools: CustomToolInfo[];
  custom_tools_path: string;
  custom_manifest_path: string;
  mcp: {
    enabled: boolean;
    tools_endpoint: string;
    execute_endpoint: string;
    api_key_configured: boolean;
  };
}

export interface ToolScaffoldPayload {
  name: string;
  friendly_name?: string;
  brief?: string;
  description?: string;
  category: string;
}

export interface ToolsTabProps {
  toolsData: AdminToolsData | null;
  loading: boolean;
  onRefresh: () => void;
  onReloadTools: () => void;
  onScaffoldTool: (payload: ToolScaffoldPayload) => Promise<void>;
  onUpdateCustomTool: (toolName: string, updates: Record<string, unknown>) => Promise<void>;
}

export function ToolsTab({
  toolsData,
  loading,
  onRefresh,
  onReloadTools,
  onScaffoldTool,
  onUpdateCustomTool,
}: ToolsTabProps) {
  const [name, setName] = useState('');
  const [friendlyName, setFriendlyName] = useState('');
  const [brief, setBrief] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('analysis');
  const [scaffolding, setScaffolding] = useState(false);
  const [sourceFilter, setSourceFilter] = useState<string>('all');

  const filteredTools = useMemo(() => {
    if (!toolsData) return [];
    if (sourceFilter === 'all') return toolsData.tools;
    return toolsData.tools.filter((tool) => tool.source === sourceFilter);
  }, [toolsData, sourceFilter]);

  const handleScaffold = async () => {
    if (!name.trim()) return;
    setScaffolding(true);
    try {
      await onScaffoldTool({
        name: name.trim(),
        friendly_name: friendlyName.trim() || undefined,
        brief: brief.trim() || undefined,
        description: description.trim() || undefined,
        category,
      });
      setName('');
      setFriendlyName('');
      setBrief('');
      setDescription('');
      setCategory('analysis');
    } finally {
      setScaffolding(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-medium">Tools</h2>
          <p className="text-xs text-gray-500 mt-1">
            Manage runtime tools, MCP exposure, and scaffold new custom tools for the active domain.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={onReloadTools}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-xl text-sm transition-colors"
          >
            Reload Registry
          </button>
          <button
            onClick={onRefresh}
            disabled={loading}
            className="px-4 py-2 bg-[#1a1a1a] hover:bg-[#333] disabled:bg-gray-700 rounded-xl text-sm transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {!toolsData ? (
        <div className="bg-[#2a2a2a] rounded-xl p-6 text-sm text-gray-400">
          Loading tool metadata...
        </div>
      ) : (
        <>
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-[#2a2a2a] rounded-xl p-4 text-center">
              <p className="text-2xl font-bold text-blue-400">{toolsData.tool_count}</p>
              <p className="text-xs text-gray-400 mt-1">Registered Tools</p>
            </div>
            <div className="bg-[#2a2a2a] rounded-xl p-4 text-center">
              <p className="text-2xl font-bold text-green-400">{toolsData.custom_tools.length}</p>
              <p className="text-xs text-gray-400 mt-1">Custom Tools</p>
            </div>
            <div className="bg-[#2a2a2a] rounded-xl p-4 text-center">
              <p className={`text-2xl font-bold ${toolsData.mcp.enabled ? 'text-green-400' : 'text-amber-400'}`}>
                {toolsData.mcp.enabled ? 'On' : 'Off'}
              </p>
              <p className="text-xs text-gray-400 mt-1">MCP API</p>
            </div>
            <div className="bg-[#2a2a2a] rounded-xl p-4 text-center">
              <p className="text-2xl font-bold text-purple-400">{toolsData.domain}</p>
              <p className="text-xs text-gray-400 mt-1">Active Domain</p>
            </div>
          </div>

          <div className="bg-[#2a2a2a] rounded-xl p-4">
            <h3 className="text-sm font-medium mb-3">MCP Exposure</h3>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-[#1a1a1a] rounded-lg p-3">
                <p className="text-gray-400 mb-1">Tools Endpoint</p>
                <p className="font-mono text-gray-200">{toolsData.mcp.tools_endpoint}</p>
              </div>
              <div className="bg-[#1a1a1a] rounded-lg p-3">
                <p className="text-gray-400 mb-1">Execute Endpoint</p>
                <p className="font-mono text-gray-200">{toolsData.mcp.execute_endpoint}</p>
              </div>
            </div>
            {!toolsData.mcp.api_key_configured && (
              <p className="text-xs text-amber-400 mt-3">
                MCP API key is not configured. Set `MCP_API_KEY` to expose tools externally.
              </p>
            )}
          </div>

          <div className="bg-[#2a2a2a] rounded-xl p-4 space-y-4">
            <div>
              <h3 className="text-sm font-medium">Create Custom Tool</h3>
              <p className="text-xs text-gray-500 mt-1">
                Scaffolds a new tool module and registers it in the active domain&apos;s custom manifest.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="tool_name (snake_case)"
                className="px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
              />
              <input
                value={friendlyName}
                onChange={(e) => setFriendlyName(e.target.value)}
                placeholder="Friendly name (optional)"
                className="px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
              />
              <input
                value={brief}
                onChange={(e) => setBrief(e.target.value)}
                placeholder="One-line brief (optional)"
                className="px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
              />
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
              >
                <option value="analysis">analysis</option>
                <option value="simple">simple</option>
                <option value="rag">rag</option>
                <option value="external">external</option>
                <option value="document">document</option>
              </select>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Tool description (optional)"
                rows={3}
                className="px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm col-span-2 resize-none"
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="text-[11px] text-gray-500 font-mono">
                {toolsData.custom_tools_path}
              </div>
              <button
                onClick={handleScaffold}
                disabled={!name.trim() || scaffolding}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-xl text-sm transition-colors"
              >
                {scaffolding ? 'Scaffolding...' : 'Scaffold Tool'}
              </button>
            </div>
          </div>

          <div className="bg-[#2a2a2a] rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium">Custom Tool Config</h3>
              <span className="text-xs text-gray-500 font-mono">{toolsData.custom_manifest_path}</span>
            </div>
            {toolsData.custom_tools.length === 0 ? (
              <p className="text-sm text-gray-500">No custom tools scaffolded yet.</p>
            ) : (
              <div className="space-y-2">
                {toolsData.custom_tools.map((tool) => (
                  <div key={tool.name} className="bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl p-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium">{tool.friendly_name || tool.name}</p>
                        <p className="text-xs text-gray-500">{tool.name} ({tool.category})</p>
                        {tool.brief && <p className="text-xs text-gray-400 mt-1">{tool.brief}</p>}
                      </div>
                      <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={tool.enabled}
                          onChange={(e) => onUpdateCustomTool(tool.name, { enabled: e.target.checked })}
                          className="w-4 h-4 rounded"
                        />
                        Enabled
                      </label>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="bg-[#2a2a2a] rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium">Tool Inventory</h3>
              <select
                value={sourceFilter}
                onChange={(e) => setSourceFilter(e.target.value)}
                className="px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded text-xs"
              >
                <option value="all">All Sources</option>
                <option value="core">Core</option>
                <option value="domain">Domain</option>
                <option value="custom">Custom</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
              {filteredTools.map((tool) => (
                <div key={tool.name} className="bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-sm font-medium">
                        {tool.friendly_name}
                        <span className="ml-2 text-xs text-gray-500 font-mono">{tool.name}</span>
                      </p>
                      {tool.brief && <p className="text-xs text-gray-400 mt-1">{tool.brief}</p>}
                    </div>
                    <div className="flex gap-1">
                      <span className="px-2 py-0.5 rounded text-[10px] bg-[#333] text-gray-300">{tool.source}</span>
                      <span className="px-2 py-0.5 rounded text-[10px] bg-blue-900/40 text-blue-300">{tool.category}</span>
                    </div>
                  </div>
                  <p className="text-xs text-gray-500 mt-2 line-clamp-2">{tool.description}</p>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
