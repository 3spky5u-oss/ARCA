'use client';

import React, { useState, useEffect, useRef } from 'react';
import { formatModelName } from './shared';

interface AdminStatusBarProps {
  apiCall: (endpoint: string, options?: RequestInit) => Promise<Record<string, unknown>>;
}

interface StatusData {
  modelName: string | null;
  vramUsedGb: number | null;
  vramTotalGb: number | null;
  activeSessions: number | null;
  isBenchmarking: boolean;
  isIngesting: boolean;
}

export function AdminStatusBar({ apiCall }: AdminStatusBarProps) {
  const [data, setData] = useState<StatusData>({
    modelName: null,
    vramUsedGb: null,
    vramTotalGb: null,
    activeSessions: null,
    isBenchmarking: false,
    isIngesting: false,
  });
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statusRes, benchRes] = await Promise.allSettled([
          apiCall('/api/admin/status'),
          apiCall('/api/admin/benchmark/status'),
        ]);

        const next: StatusData = {
          modelName: null,
          vramUsedGb: null,
          vramTotalGb: null,
          activeSessions: null,
          isBenchmarking: false,
          isIngesting: false,
        };

        if (statusRes.status === 'fulfilled' && statusRes.value) {
          const s = statusRes.value;

          // Model name from actually-loaded models (not configured list)
          const llm = s.llm as Record<string, unknown> | undefined;
          if (llm) {
            const loaded = llm.loaded_models as Array<Record<string, unknown>> | undefined;
            if (loaded && loaded.length > 0) {
              next.modelName = (loaded[0].name as string) || null;
            } else {
              // Fallback to configured models if nothing loaded
              const models = llm.models as string[] | undefined;
              if (models && models.length > 0) {
                next.modelName = models[0];
              }
            }
          }

          // VRAM from system metrics
          const system = s.system as Record<string, unknown> | undefined;
          if (system) {
            const vramUsedMb = system.gpu_vram_used_mb as number | null;
            if (vramUsedMb != null) {
              next.vramUsedGb = vramUsedMb / 1024;
            }
            const specs = system.specs as Record<string, unknown> | undefined;
            if (specs) {
              const vramTotalMb = specs.gpu_vram_total_mb as number | null;
              if (vramTotalMb != null) {
                next.vramTotalGb = vramTotalMb / 1024;
              }
            }
          }

          // Active sessions
          const sessions = s.sessions as Record<string, unknown> | undefined;
          if (sessions) {
            const activeFiles = sessions.active_files as number | undefined;
            if (activeFiles != null) {
              next.activeSessions = activeFiles;
            }
          }

          // Ingest active
          const config = s.config as Record<string, unknown> | undefined;
          if (config && config._ingest_active === true) {
            next.isIngesting = true;
          }
        }

        if (benchRes.status === 'fulfilled' && benchRes.value) {
          const b = benchRes.value;
          if (b.status === 'running') {
            next.isBenchmarking = true;
          }
        }

        setData(next);
      } catch {
        // Silently ignore — bar will show fallback state
      }
    };

    fetchData();
    intervalRef.current = setInterval(fetchData, 10000);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [apiCall]);

  const formatVram = () => {
    if (data.vramUsedGb == null || data.vramTotalGb == null) return null;
    return `${data.vramUsedGb.toFixed(1)}/${data.vramTotalGb.toFixed(1)} GB`;
  };

  const vram = formatVram();
  const hasAnyData = data.modelName || vram || data.activeSessions != null || data.isBenchmarking || data.isIngesting;

  if (!hasAnyData) return null;

  return (
    <div className="bg-[#1a1a1a] border-b border-[#2a2a2a] px-4 py-1.5 flex items-center justify-between text-xs">
      {/* Left: model + VRAM */}
      <div className="flex items-center gap-2 text-gray-400 min-w-0">
        {data.modelName && (() => {
          const f = formatModelName(data.modelName);
          return (
            <span className="text-gray-300 truncate">
              {f.display}
              {f.quant && <span className="text-gray-500 ml-1 text-xs">{f.quant}</span>}
            </span>
          );
        })()}
        {data.modelName && vram && (
          <span className="text-gray-600">&mdash;</span>
        )}
        {vram && (
          <span className="text-gray-500 whitespace-nowrap">{vram} VRAM</span>
        )}
      </div>

      {/* Right: sessions + process indicators */}
      <div className="flex items-center gap-3 text-gray-400 flex-shrink-0">
        {data.activeSessions != null && (
          <span className="whitespace-nowrap">
            {data.activeSessions} session{data.activeSessions !== 1 ? 's' : ''}
          </span>
        )}
        {data.isBenchmarking && (
          <span className="inline-flex items-center gap-1.5 bg-blue-900/40 text-blue-300 px-2 py-0.5 rounded-full">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
            Benchmarking
          </span>
        )}
        {data.isIngesting && (
          <span className="inline-flex items-center gap-1.5 bg-amber-900/40 text-amber-300 px-2 py-0.5 rounded-full">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
            Ingesting
          </span>
        )}
      </div>
    </div>
  );
}
