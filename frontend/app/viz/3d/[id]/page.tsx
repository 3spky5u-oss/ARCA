'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import dynamic from 'next/dynamic';
import type { Data, Layout, Config } from 'plotly.js';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full">
      <div className="text-gray-400">Loading visualization...</div>
    </div>
  )
});

// =============================================================================
// TYPES
// =============================================================================

interface BoreholePoint {
  x: number;
  y: number;
  z: number;
  borehole_id: string;
  depth: number;
  soil_type?: string;
  color?: string;
}

interface VizData {
  viz_id: string;
  plotly_data: {
    data: Data[];
    layout?: Partial<Layout>;
  };
}

interface VizMetadata {
  viz_id: string;
  created_at: string;
  has_preview: boolean;
  metadata: {
    borehole_count?: number;
    title?: string;
    total_depth_range?: string;
  };
}

// =============================================================================
// API
// =============================================================================

// Use shared API base (respects NEXT_PUBLIC_API_URL for reverse proxy)
import { getApiBase } from '@/lib/api';

async function fetchVizData(id: string): Promise<VizData> {
  const response = await fetch(`${getApiBase()}/api/viz/data/${id}`);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to fetch visualization data' }));
    throw new Error(error.detail || 'Failed to fetch visualization data');
  }
  return response.json();
}

async function fetchVizMetadata(id: string): Promise<VizMetadata> {
  const response = await fetch(`${getApiBase()}/api/viz/metadata/${id}`);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to fetch visualization metadata' }));
    throw new Error(error.detail || 'Failed to fetch visualization metadata');
  }
  return response.json();
}

// =============================================================================
// ICONS
// =============================================================================

const CloseIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
  </svg>
);

const DownloadIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
  </svg>
);

const ResetIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
  </svg>
);

const LayersIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
  </svg>
);

const ChevronIcon = ({ expanded }: { expanded: boolean }) => (
  <svg
    className={`w-4 h-4 transition-transform ${expanded ? 'rotate-180' : ''}`}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
  </svg>
);

const ImageIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
  </svg>
);

const GroundIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 21h18M5 21V7l7-4 7 4v14" />
  </svg>
);

const WaterIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
  </svg>
);

const ProjectionIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
  </svg>
);

// =============================================================================
// TOGGLE CONTROL COMPONENT
// =============================================================================

interface ToggleControlProps {
  label: string;
  enabled: boolean;
  onToggle: () => void;
  icon?: React.ReactNode;
  accentColor?: 'default' | 'blue';
}

function ToggleControl({ label, enabled, onToggle, icon, accentColor = 'default' }: ToggleControlProps) {
  const bgColor = enabled
    ? accentColor === 'blue' ? 'bg-blue-600' : 'bg-green-600'
    : 'bg-gray-600';

  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-sm text-gray-300">{label}</span>
      </div>
      <button
        onClick={onToggle}
        className={`w-10 h-5 rounded-full transition-colors relative ${bgColor}`}
      >
        <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
          enabled ? 'left-5' : 'left-0.5'
        }`} />
      </button>
    </div>
  );
}

// =============================================================================
// COMPONENT
// =============================================================================

export default function Viz3DPage() {
  const params = useParams();
  const router = useRouter();
  const [plotElement, setPlotElement] = useState<HTMLElement | null>(null);

  const id = params.id as string;

  // State
  const [vizData, setVizData] = useState<VizData | null>(null);
  const [metadata, setMetadata] = useState<VizMetadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [controlsExpanded, setControlsExpanded] = useState(true);
  const [slicePlaneEnabled, setSlicePlaneEnabled] = useState(false);

  // Layer visibility toggles
  const [groundSurfaceVisible, setGroundSurfaceVisible] = useState(false);  // Off by default
  const [gwtVisible, setGwtVisible] = useState(true);
  const [orthoProjection, setOrthoProjection] = useState(false);  // Perspective by default

  // Fetch data on mount
  useEffect(() => {
    if (!id) return;

    const loadData = async () => {
      setLoading(true);
      setError(null);

      try {
        const [data, meta] = await Promise.all([
          fetchVizData(id),
          fetchVizMetadata(id)
        ]);

        // Hide Ground Surface by default
        if (data?.plotly_data?.data) {
          data.plotly_data.data = data.plotly_data.data.map((trace: any) => {
            if (trace.name === 'Ground Surface') {
              return { ...trace, visible: false };
            }
            return trace;
          });
        }

        setVizData(data);
        setMetadata(meta);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load visualization');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [id]);

  // Handle close/back
  const handleClose = useCallback(() => {
    router.back();
  }, [router]);

  // Export as PNG
  const handleExportPNG = useCallback(async () => {
    if (!plotElement) return;

    try {
      const Plotly = await import('plotly.js-dist-min');
      await Plotly.default.downloadImage(plotElement, {
        format: 'png',
        width: 1920,
        height: 1080,
        filename: `viz-3d-${id}`
      });
    } catch (err) {
      console.error('Failed to export PNG:', err);
    }
  }, [id, plotElement]);

  // Export current view data (JSON)
  const handleExportView = useCallback(() => {
    if (!vizData) return;

    const viewData = {
      metadata,
      data: vizData,
      exportedAt: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(viewData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `viz-3d-${id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [vizData, metadata, id]);

  // Reset view to default
  const handleResetView = useCallback(async () => {
    if (!plotElement) return;

    try {
      const Plotly = await import('plotly.js-dist-min');
      // Use type assertion for Plotly.relayout which accepts string keys
      await Plotly.default.relayout(plotElement, {
        'scene.camera.eye.x': 1.5,
        'scene.camera.eye.y': 1.5,
        'scene.camera.eye.z': 1.2,
        'scene.camera.center.x': 0,
        'scene.camera.center.y': 0,
        'scene.camera.center.z': 0,
        'scene.camera.up.x': 0,
        'scene.camera.up.y': 0,
        'scene.camera.up.z': 1,
      } as Record<string, number>);
    } catch (err) {
      console.error('Failed to reset view:', err);
    }
  }, [plotElement]);

  // Toggle ground surface visibility
  const handleToggleGroundSurface = useCallback(async () => {
    if (!plotElement || !vizData?.plotly_data?.data) return;

    const newVisible = !groundSurfaceVisible;
    setGroundSurfaceVisible(newVisible);

    try {
      const Plotly = await import('plotly.js-dist-min');
      // Find ground surface trace index
      const groundIdx = vizData.plotly_data.data.findIndex(
        (trace: any) => trace.name === 'Ground Surface'
      );
      if (groundIdx >= 0) {
        await Plotly.default.restyle(plotElement, { visible: newVisible }, [groundIdx]);
      }
    } catch (err) {
      console.error('Failed to toggle ground surface:', err);
    }
  }, [plotElement, vizData, groundSurfaceVisible]);

  // Toggle GWT visibility
  const handleToggleGWT = useCallback(async () => {
    if (!plotElement || !vizData?.plotly_data?.data) return;

    const newVisible = !gwtVisible;
    setGwtVisible(newVisible);

    try {
      const Plotly = await import('plotly.js-dist-min');
      // Find water level trace index (trace name set by backend domain tool)
      const gwtIdx = vizData.plotly_data.data.findIndex(
        (trace: any) => trace.name === 'Groundwater Table' || trace.name === 'Water Level'
      );
      if (gwtIdx >= 0) {
        await Plotly.default.restyle(plotElement, { visible: newVisible }, [gwtIdx]);
      }
    } catch (err) {
      console.error('Failed to toggle GWT:', err);
    }
  }, [plotElement, vizData, gwtVisible]);

  // Toggle projection mode
  const handleToggleProjection = useCallback(async () => {
    if (!plotElement) return;

    const newOrtho = !orthoProjection;
    setOrthoProjection(newOrtho);

    try {
      const Plotly = await import('plotly.js-dist-min');
      await Plotly.default.relayout(plotElement, {
        'scene.camera.projection.type': newOrtho ? 'orthographic' : 'perspective',
      } as Record<string, string>);
    } catch (err) {
      console.error('Failed to toggle projection:', err);
    }
  }, [plotElement, orthoProjection]);

  // Default Plotly layout for 3D
  const defaultLayout: Partial<Layout> = {
    paper_bgcolor: '#212121',
    plot_bgcolor: '#212121',
    font: {
      color: '#e5e7eb',
      family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    },
    scene: {
      bgcolor: '#1a1a1a',
      xaxis: {
        title: { text: 'Easting (m)', font: { color: '#e5e7eb' } },
        gridcolor: '#333',
        zerolinecolor: '#444',
        tickfont: { color: '#9ca3af' },
      },
      yaxis: {
        title: { text: 'Northing (m)', font: { color: '#e5e7eb' } },
        gridcolor: '#333',
        zerolinecolor: '#444',
        tickfont: { color: '#9ca3af' },
      },
      zaxis: {
        title: { text: 'Elevation (m)', font: { color: '#e5e7eb' } },
        gridcolor: '#333',
        zerolinecolor: '#444',
        tickfont: { color: '#9ca3af' },
        autorange: 'reversed' // Depth increases downward
      },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.2 },
        center: { x: 0, y: 0, z: 0 },
        up: { x: 0, y: 0, z: 1 }
      },
      aspectmode: 'data'
    },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    showlegend: true,
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(42, 42, 42, 0.8)',
      bordercolor: '#3a3a3a',
      borderwidth: 1,
      font: { color: '#e5e7eb' }
    }
  };

  // Merge with any layout from the API
  const plotLayout = {
    ...defaultLayout,
    ...vizData?.plotly_data?.layout
  };

  // Plotly config
  const plotConfig: Partial<Config> = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
    responsive: true,
    scrollZoom: true
  };

  // Loading state
  if (loading) {
    return (
      <div className="h-full flex flex-col">
        <Header metadata={null} onClose={handleClose} />
        <div className="flex-1 flex items-center justify-center">
          <div className="flex flex-col items-center gap-4">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-gray-400">Loading visualization...</span>
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="h-full flex flex-col">
        <Header metadata={null} onClose={handleClose} />
        <div className="flex-1 flex items-center justify-center">
          <div className="flex flex-col items-center gap-4 max-w-md text-center">
            <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center">
              <svg className="w-6 h-6 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h2 className="text-lg font-medium text-gray-100">Failed to Load Visualization</h2>
            <p className="text-gray-400">{error}</p>
            <button
              onClick={handleClose}
              className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm font-medium transition-colors"
            >
              Go Back
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <Header metadata={metadata} onClose={handleClose} />

      {/* Main content area */}
      <div className="flex-1 flex relative overflow-hidden">
        {/* 3D Visualization */}
        <div className="flex-1 relative">
          {vizData?.plotly_data?.data && (
            <Plot
              data={vizData.plotly_data.data}
              layout={plotLayout}
              config={plotConfig}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler={true}
              onInitialized={(figure, graphDiv) => setPlotElement(graphDiv as HTMLElement)}
              onUpdate={(figure, graphDiv) => setPlotElement(graphDiv as HTMLElement)}
            />
          )}
        </div>

        {/* Control Panel - Bottom bar */}
        <div className="absolute bottom-4 left-4 right-4 md:left-auto md:right-4 md:w-80">
          <div className="bg-[#2a2a2a] border border-[#3a3a3a] rounded-xl overflow-hidden">
            {/* Panel header (clickable to expand/collapse) */}
            <button
              onClick={() => setControlsExpanded(!controlsExpanded)}
              className="w-full flex items-center justify-between px-4 py-3 hover:bg-[#333] transition-colors"
            >
              <span className="text-sm font-medium text-gray-200">Controls</span>
              <ChevronIcon expanded={controlsExpanded} />
            </button>

            {/* Panel content */}
            {controlsExpanded && (
              <div className="px-4 pb-4 space-y-3 animate-fadeIn">
                {/* Ground Surface Toggle */}
                <ToggleControl
                  label="Ground Surface"
                  enabled={groundSurfaceVisible}
                  onToggle={handleToggleGroundSurface}
                  icon={<GroundIcon />}
                />

                {/* Water Level Toggle (trace name from backend domain tool) */}
                <ToggleControl
                  label="Water Level"
                  enabled={gwtVisible}
                  onToggle={handleToggleGWT}
                  icon={<WaterIcon />}
                  accentColor="blue"
                />

                {/* Projection Toggle */}
                <ToggleControl
                  label={orthoProjection ? 'Orthographic' : 'Perspective'}
                  enabled={orthoProjection}
                  onToggle={handleToggleProjection}
                  icon={<ProjectionIcon />}
                />

                {/* Divider */}
                <div className="border-t border-[#3a3a3a]" />

                {/* Slice Plane (Coming Soon) */}
                <div className="flex items-center justify-between opacity-50">
                  <div className="flex items-center gap-2">
                    <LayersIcon />
                    <span className="text-sm text-gray-300">Slice Plane</span>
                    <span className="text-xs px-1.5 py-0.5 bg-amber-500/20 text-amber-400 rounded">Soon</span>
                  </div>
                  <button
                    disabled
                    className="w-10 h-5 rounded-full bg-gray-600 relative cursor-not-allowed"
                  >
                    <span className="absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full" />
                  </button>
                </div>

                {/* Divider */}
                <div className="border-t border-[#3a3a3a]" />

                {/* Export buttons */}
                <div className="space-y-2">
                  <button
                    onClick={handleExportPNG}
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm font-medium transition-colors"
                  >
                    <ImageIcon />
                    Export PNG
                  </button>

                  <button
                    onClick={handleExportView}
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-[#333] hover:bg-[#3a3a3a] border border-[#444] rounded-xl text-sm font-medium transition-colors"
                  >
                    <DownloadIcon />
                    Export View Data
                  </button>

                  <button
                    onClick={handleResetView}
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-[#333] hover:bg-[#3a3a3a] border border-[#444] rounded-xl text-sm font-medium transition-colors"
                  >
                    <ResetIcon />
                    Reset View
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// HEADER COMPONENT
// =============================================================================

interface HeaderProps {
  metadata: VizMetadata | null;
  onClose: () => void;
}

function Header({ metadata, onClose }: HeaderProps) {
  const meta = metadata?.metadata;
  return (
    <header className="flex items-center justify-between px-4 py-3 bg-[#1a1a1a] border-b border-[#2a2a2a]">
      {/* Title and metadata */}
      <div className="flex items-center gap-4">
        <h1 className="text-lg font-semibold text-gray-100">
          {meta?.title || '3D Visualization'}
        </h1>

        {meta && (
          <div className="hidden md:flex items-center gap-3">
            {meta.borehole_count && (
              <MetadataBadge
                label="Boreholes"
                value={meta.borehole_count.toString()}
              />
            )}
            {meta.total_depth_range && (
              <MetadataBadge
                label="Depth"
                value={meta.total_depth_range}
              />
            )}
          </div>
        )}
      </div>

      {/* Close button */}
      <button
        onClick={onClose}
        aria-label="Close visualization"
        className="w-9 h-9 flex items-center justify-center rounded-xl text-gray-400 hover:text-white hover:bg-[#333] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
      >
        <CloseIcon />
      </button>
    </header>
  );
}

// =============================================================================
// METADATA BADGE COMPONENT
// =============================================================================

interface MetadataBadgeProps {
  label: string;
  value: string;
}

function MetadataBadge({ label, value }: MetadataBadgeProps) {
  return (
    <div className="flex items-center gap-1.5 px-2.5 py-1 bg-[#2a2a2a] border border-[#3a3a3a] rounded-full">
      <span className="text-xs text-gray-500">{label}:</span>
      <span className="text-xs font-medium text-gray-300">{value}</span>
    </div>
  );
}
