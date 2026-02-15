'use client';

import React, { useRef, useCallback, useEffect, useState } from 'react';
import { VisualizationData, getNodeColor, LabelInfo } from './types';

interface ForceGraph2DProps {
  graphData: { nodes: NodeObject[]; links: LinkObject[] };
  nodeLabel?: (node: NodeObject) => string;
  nodeColor?: (node: NodeObject) => string;
  nodeVal?: (node: NodeObject) => number;
  linkLabel?: (link: LinkObject) => string;
  linkColor?: () => string;
  linkWidth?: () => number;
  onNodeClick?: (node: NodeObject) => void;
  width?: number;
  height?: number;
  backgroundColor?: string;
}

interface NodeObject {
  id: string;
  label: string;
  type: string;
  properties?: Record<string, unknown>;
  x?: number;
  y?: number;
}

interface LinkObject {
  source: string | NodeObject;
  target: string | NodeObject;
  type: string;
}

interface GraphViewerProps {
  data: VisualizationData | null;
  labels: LabelInfo[];
  loading: boolean;
  selectedType: string;
  includeChunks: boolean;
  nodeLimit: number;
  onRefresh: () => void;
  onTypeFilter: (type: string) => void;
  onIncludeChunksChange: (include: boolean) => void;
  onNodeLimitChange: (limit: number) => void;
  onNodeClick: (name: string) => void;
}

export function GraphViewer({
  data,
  labels,
  loading,
  selectedType,
  includeChunks,
  nodeLimit,
  onRefresh,
  onTypeFilter,
  onIncludeChunksChange,
  onNodeLimitChange,
  onNodeClick,
}: GraphViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });
  const [ForceGraphComponent, setForceGraphComponent] = useState<React.ComponentType<ForceGraph2DProps> | null>(null);

  // Load ForceGraph2D dynamically
  useEffect(() => {
    import('react-force-graph-2d').then((mod) => {
      setForceGraphComponent(() => mod.default as unknown as React.ComponentType<ForceGraph2DProps>);
    }).catch((err) => {
      console.error('Failed to load react-force-graph-2d:', err);
    });
  }, []);

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: rect.width,
          height: Math.max(500, rect.height),
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  const handleNodeClick = useCallback((node: NodeObject) => {
    if (node.id) {
      onNodeClick(node.id);
    }
  }, [onNodeClick]);

  // Transform data for force graph
  const graphData = React.useMemo(() => {
    if (!data) return { nodes: [], links: [] };

    const nodes: NodeObject[] = data.nodes.map((n) => ({
      id: n.id,
      label: n.label,
      type: n.type,
      properties: n.properties,
    }));

    const nodeIds = new Set(nodes.map((n) => n.id));

    const links: LinkObject[] = data.edges
      .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target))
      .map((e) => ({
        source: e.source,
        target: e.target,
        type: e.type,
      }));

    return { nodes, links };
  }, [data]);

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <svg className="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
            </svg>
            Graph Visualization
          </h3>
          <button
            onClick={onRefresh}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-xl text-sm transition-colors"
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          {/* Type Filter */}
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => onTypeFilter('')}
              className={`px-3 py-1 rounded-full text-xs transition-colors ${
                !selectedType
                  ? 'bg-blue-600 text-white'
                  : 'bg-[#1a1a1a] text-gray-400 hover:text-white'
              }`}
            >
              All
            </button>
            {labels.filter((l) => l.label !== 'Chunk').slice(0, 5).map((label) => (
              <button
                key={label.label}
                onClick={() => onTypeFilter(label.label)}
                className={`px-3 py-1 rounded-full text-xs transition-colors ${
                  selectedType === label.label
                    ? 'bg-blue-600 text-white'
                    : 'bg-[#1a1a1a] text-gray-400 hover:text-white'
                }`}
              >
                {label.label}
              </button>
            ))}
          </div>

          {/* Include Chunks Toggle */}
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={includeChunks}
              onChange={(e) => onIncludeChunksChange(e.target.checked)}
              className="w-4 h-4 rounded"
            />
            Include Chunks
          </label>

          {/* Node Limit */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400">Max Nodes:</span>
            <select
              value={nodeLimit}
              onChange={(e) => onNodeLimitChange(Number(e.target.value))}
              className="px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded text-xs"
            >
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
              <option value={500}>500</option>
            </select>
          </div>
        </div>
      </div>

      {/* Graph Canvas */}
      <div
        ref={containerRef}
        className="bg-[#1a1a1a] rounded-lg overflow-hidden"
        style={{ minHeight: 500 }}
      >
        {loading ? (
          <div className="h-[500px] flex items-center justify-center">
            <svg className="animate-spin h-8 w-8 text-blue-500" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          </div>
        ) : !data || graphData.nodes.length === 0 ? (
          <div className="h-[500px] flex items-center justify-center text-gray-500">
            <div className="text-center">
              <svg className="w-12 h-12 mx-auto mb-2 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
              No graph data available. Click Refresh to load.
            </div>
          </div>
        ) : ForceGraphComponent ? (
          <ForceGraphComponent
            graphData={graphData}
            nodeLabel={(node: NodeObject) => `${node.label} (${node.type})`}
            nodeColor={(node: NodeObject) => getNodeColor(node.type)}
            nodeVal={(node: NodeObject) => node.type === 'Chunk' ? 1 : 3}
            linkLabel={(link: LinkObject) => link.type}
            linkColor={() => '#666'}
            linkWidth={() => 1}
            onNodeClick={handleNodeClick}
            width={dimensions.width}
            height={dimensions.height}
            backgroundColor="#1a1a1a"
          />
        ) : (
          <div className="h-[500px] flex items-center justify-center text-gray-500">
            Loading graph library...
          </div>
        )}
      </div>

      {/* Legend */}
      {data && graphData.nodes.length > 0 && (
        <div className="bg-[#2a2a2a] rounded-lg p-4">
          <div className="flex items-center justify-between text-sm text-gray-400">
            <span>
              Showing {graphData.nodes.length} nodes and {graphData.links.length} edges
            </span>
            <div className="flex items-center gap-4">
              {labels.filter((l) => l.label !== 'Chunk').slice(0, 5).map((label) => (
                <span key={label.label} className="flex items-center gap-1">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: getNodeColor(label.label) }}
                  />
                  {label.label}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
