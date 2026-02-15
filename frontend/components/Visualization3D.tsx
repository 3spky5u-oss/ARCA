'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';
import type { Data, Layout, Config } from 'plotly.js';

// Dynamic import for Plotly to avoid SSR issues
// NOTE: Requires react-plotly.js and plotly.js packages:
//   npm install react-plotly.js plotly.js
//   npm install -D @types/react-plotly.js
const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
  loading: () => null
});

/**
 * Props for Visualization3D component
 */
export interface Visualization3DProps {
  vizId: string;
  previewUrl: string;
  dataUrl: string;
  fullPageUrl: string;
}

/**
 * Plotly data trace (simplified type for 3D scatter/surface)
 */
interface PlotlyTrace {
  type?: string;
  mode?: string;
  x?: number[];
  y?: number[];
  z?: number[];
  marker?: {
    size?: number;
    color?: string | number[];
    colorscale?: string;
    opacity?: number;
  };
  line?: {
    color?: string;
    width?: number;
  };
  text?: string[];
  hoverinfo?: string;
  name?: string;
  [key: string]: unknown;
}

/**
 * Plotly layout configuration (simplified type)
 */
interface PlotlyLayout {
  title?: string | { text: string };
  paper_bgcolor?: string;
  plot_bgcolor?: string;
  font?: {
    color?: string;
    family?: string;
    size?: number;
  };
  margin?: {
    l?: number;
    r?: number;
    t?: number;
    b?: number;
  };
  scene?: {
    xaxis?: {
      title?: string;
      gridcolor?: string;
      zerolinecolor?: string;
      [key: string]: unknown;
    };
    yaxis?: {
      title?: string;
      gridcolor?: string;
      zerolinecolor?: string;
      [key: string]: unknown;
    };
    zaxis?: {
      title?: string;
      gridcolor?: string;
      zerolinecolor?: string;
      [key: string]: unknown;
    };
    bgcolor?: string;
    [key: string]: unknown;
  };
  autosize?: boolean;
  [key: string]: unknown;
}

/**
 * Plotly config (simplified type)
 */
interface PlotlyConfig {
  responsive?: boolean;
  displayModeBar?: boolean;
  displaylogo?: boolean;
  modeBarButtonsToRemove?: string[];
  [key: string]: unknown;
}

/**
 * Plotly data structure for 3D visualizations
 */
interface PlotlyData {
  data: PlotlyTrace[];
  layout: PlotlyLayout;
  config?: PlotlyConfig;
}

/**
 * Visualization3D component for displaying interactive 3D data visualizations.
 *
 * Features:
 * - Initially shows static preview image with "Click to interact" overlay
 * - Uses IntersectionObserver to detect when in viewport
 * - On click OR after 2s visible: fetches data and renders 3D
 * - When scrolled out of view: cleans up Plotly to free memory
 * - When scrolled back: reloads from cached data
 * - Expand button to open full-page visualization
 */
export function Visualization3D({
  vizId,
  previewUrl,
  dataUrl,
  fullPageUrl
}: Visualization3DProps) {
  // State
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [plotData, setPlotData] = useState<PlotlyData | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Refs
  const containerRef = useRef<HTMLDivElement>(null);
  const dataCache = useRef<PlotlyData | null>(null);
  const visibleTimerRef = useRef<NodeJS.Timeout | null>(null);
  const hasAutoLoaded = useRef(false);

  /**
   * Fetch visualization data from the backend
   */
  const fetchData = useCallback(async () => {
    // Check cache first
    if (dataCache.current) {
      setPlotData(dataCache.current);
      setIsLoaded(true);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(dataUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch visualization data: ${response.statusText}`);
      }

      const data: PlotlyData = await response.json();

      // Cache the data
      dataCache.current = data;
      setPlotData(data);
      setIsLoaded(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load visualization';
      setError(message);
      console.error('Visualization3D fetch error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [dataUrl]);

  /**
   * Handle click to load the visualization
   */
  const handleClick = useCallback(() => {
    if (!isLoaded && !isLoading) {
      fetchData();
    }
  }, [isLoaded, isLoading, fetchData]);

  /**
   * Handle expand button click
   */
  const handleExpand = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    window.open(fullPageUrl, '_blank', 'noopener,noreferrer');
  }, [fullPageUrl]);

  /**
   * Cleanup Plotly when component unmounts or goes out of view
   */
  const cleanup = useCallback(() => {
    setIsLoaded(false);
    setPlotData(null);
    // Note: dataCache is preserved so we can reload from cache
  }, []);

  // IntersectionObserver for viewport detection
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        const nowVisible = entry.isIntersecting;

        if (nowVisible) {
          // Start timer for auto-load after 2 seconds visible
          if (!hasAutoLoaded.current && !isLoaded && !isLoading) {
            visibleTimerRef.current = setTimeout(() => {
              if (!isLoaded && !isLoading) {
                hasAutoLoaded.current = true;
                fetchData();
              }
            }, 2000);
          }
          // Reload from cache if we have data
          else if (dataCache.current && !isLoaded && !isLoading) {
            setPlotData(dataCache.current);
            setIsLoaded(true);
          }
        } else {
          // Clear auto-load timer
          if (visibleTimerRef.current) {
            clearTimeout(visibleTimerRef.current);
            visibleTimerRef.current = null;
          }
          // Cleanup when scrolled out of view (but keep cache)
          if (isLoaded) {
            cleanup();
          }
        }
      },
      {
        threshold: 0.1,
        rootMargin: '50px'
      }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => {
      observer.disconnect();
      if (visibleTimerRef.current) {
        clearTimeout(visibleTimerRef.current);
      }
    };
  }, [isLoaded, isLoading, fetchData, cleanup]);

  // Plotly layout configuration for dark theme
  const getPlotlyLayout = useCallback((): PlotlyLayout => {
    const baseLayout = plotData?.layout || {};
    return {
      ...baseLayout,
      paper_bgcolor: '#2a2a2a',
      plot_bgcolor: '#2a2a2a',
      font: {
        color: '#e5e7eb',
        family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
      },
      margin: { l: 40, r: 40, t: 40, b: 40 },
      scene: {
        ...(baseLayout.scene || {}),
        xaxis: {
          gridcolor: '#3a3a3a',
          zerolinecolor: '#3a3a3a',
          ...(baseLayout.scene?.xaxis || {})
        },
        yaxis: {
          gridcolor: '#3a3a3a',
          zerolinecolor: '#3a3a3a',
          ...(baseLayout.scene?.yaxis || {})
        },
        zaxis: {
          gridcolor: '#3a3a3a',
          zerolinecolor: '#3a3a3a',
          ...(baseLayout.scene?.zaxis || {})
        },
        bgcolor: '#2a2a2a'
      },
      autosize: true
    };
  }, [plotData]);

  // Plotly config
  const plotlyConfig: PlotlyConfig = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    ...(plotData?.config || {})
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full rounded-xl border border-[#3a3a3a] bg-[#2a2a2a] overflow-hidden"
      style={{ height: '400px' }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={handleClick}
      role="button"
      tabIndex={0}
      aria-label={`3D visualization ${vizId}. ${isLoaded ? 'Interactive' : 'Click to interact'}`}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleClick();
        }
      }}
    >
      {/* Preview image with overlay (shown when not loaded and no error) */}
      {!isLoaded && !isLoading && !error && (
        <div className="absolute inset-0 cursor-pointer">
          {/* Preview image */}
          <img
            src={previewUrl}
            alt={`Preview of 3D visualization ${vizId}`}
            className="w-full h-full object-cover"
          />

          {/* Semi-transparent overlay */}
          <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
            <div className="flex flex-col items-center gap-2">
              {/* 3D cube icon */}
              <svg
                className="w-12 h-12 text-blue-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"
                />
                <polyline
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  points="3.27 6.96 12 12.01 20.73 6.96"
                />
                <line
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  x1="12"
                  y1="22.08"
                  x2="12"
                  y2="12"
                />
              </svg>
              <span className="text-white text-sm font-medium">Click to interact</span>
            </div>
          </div>
        </div>
      )}

      {/* Loading state */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#2a2a2a]">
          <div className="flex flex-col items-center gap-3">
            {/* Spinner */}
            <svg
              className="w-8 h-8 text-blue-400 animate-spin"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            <span className="text-gray-400 text-sm">Loading visualization</span>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && !isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#2a2a2a]">
          <div className="flex flex-col items-center gap-2 text-center px-4">
            {/* Error icon */}
            <svg
              className="w-8 h-8 text-red-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span className="text-red-400 text-sm">{error}</span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                dataCache.current = null;
                fetchData();
              }}
              className="mt-2 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-sm rounded-xl transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Plotly visualization */}
      {isLoaded && plotData && (
        <Plot
          data={plotData.data as Data[]}
          layout={getPlotlyLayout() as Partial<Layout>}
          config={plotlyConfig as Partial<Config>}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler={true}
        />
      )}

      {/* Expand button (shown on hover) */}
      <button
        onClick={handleExpand}
        className={`absolute top-3 right-3 w-9 h-9 flex items-center justify-center
                    rounded-xl bg-[#212121]/80 hover:bg-[#333]
                    text-gray-400 hover:text-white
                    transition-all duration-200
                    focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500
                    ${isHovered ? 'opacity-100' : 'opacity-0'}`}
        aria-label="Open in full page"
        title="Open in full page"
      >
        {/* Expand icon */}
        <svg
          className="w-4 h-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
          />
        </svg>
      </button>
    </div>
  );
}

export default Visualization3D;
