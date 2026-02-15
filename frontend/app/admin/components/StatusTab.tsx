'use client';

import React from 'react';
import { formatModelName } from './shared';

interface ServiceHealth {
  status: string;
  latency_ms?: number;
  error?: string;
  mode?: string;
  connected?: boolean;
}

interface SystemMetrics {
  cpu_percent: number | null;
  cpu_temperature_c: number | null;
  ram_used_gb: number | null;
  ram_percent: number | null;
  disk_used_gb: number | null;
  disk_total_gb: number | null;
  disk_percent: number | null;
  drive_temperature_c: number | null;
  gpu_utilization_pct: number | null;
  gpu_vram_used_mb: number | null;
  gpu_temperature_c: number | null;
  gpu_power_w: number | null;
  gpu_power_limit_w: number | null;
  specs: {
    cpu_model: string;
    cpu_cores: number;
    ram_total_gb: number;
    gpu_name: string;
    gpu_vram_total_mb: number;
  };
}

interface LoadedModel {
  name: string;
  size_gb: number;
  vram_gb: number;
  cpu_gb: number;
  gpu_pct: number;
  quantization: string;
  parameter_size: string;
  slot?: string;
  parallel?: number;
}

interface SystemStatus {
  timestamp: string;
  llm: {
    status: string;
    models?: string[];
    loaded_models?: LoadedModel[];
    error?: string;
  };
  storage: {
    uploads_count: number;
    uploads_mb: number;
    reports_count: number;
    reports_mb: number;
    total_mb: number;
    note?: string;
  };
  sessions: {
    active_files?: number;
  };
  config: Record<string, unknown>;
  rag: {
    knowledge_chunks: number;
    topics?: string[];
  };
  services?: Record<string, ServiceHealth>;
  system?: SystemMetrics;
}

interface HardwareProfile {
  gpu?: { name: string; vram_total_mb: number; vram_total_gb: number; vram_available_mb: number; vram_available_gb: number };
  cpu?: { model: string; cores: number };
  ram_total_gb?: number;
  profile?: string;
  profile_description?: string;
  profile_notes?: string;
  model_recommendations?: Record<string, string>;
}

interface StatusTabProps {
  status: SystemStatus;
  onRecalibrate: () => void;
  onRefresh?: () => void;
  hardwareProfile?: HardwareProfile | null;
}

// Color thresholds for progress bars
function getBarColor(pct: number): string {
  if (pct >= 90) return 'bg-red-500';
  if (pct >= 70) return 'bg-amber-500';
  return 'bg-green-500';
}

function getBarBgColor(pct: number): string {
  if (pct >= 90) return 'text-red-400';
  if (pct >= 70) return 'text-amber-400';
  return 'text-green-400';
}

// Temperature color thresholds (°C)
function getTempBarColor(temp: number): string {
  if (temp >= 85) return 'bg-red-500';
  if (temp >= 70) return 'bg-amber-500';
  return 'bg-green-500';
}

function getTempTextColor(temp: number): string {
  if (temp >= 85) return 'text-red-400';
  if (temp >= 70) return 'text-amber-400';
  return 'text-green-400';
}

// Service health status dot
function ServiceDot({ health }: { health?: ServiceHealth }) {
  if (!health) return <span className="inline-block w-2.5 h-2.5 rounded-full bg-gray-600" />;

  const isOk = health.status === 'connected' || health.status === 'healthy' || health.connected === true;
  const isFallback = health.status === 'fallback';

  let color = 'bg-red-500';
  if (isOk) color = 'bg-green-500';
  else if (isFallback) color = 'bg-amber-500';

  return (
    <span className="relative flex items-center">
      <span className={`inline-block w-2.5 h-2.5 rounded-full ${color}`} />
      {isOk && (
        <span className={`absolute inline-block w-2.5 h-2.5 rounded-full ${color} animate-ping opacity-40`} />
      )}
    </span>
  );
}

function ResourceBar({ label, value, max, unit, percent }: {
  label: string;
  value: string;
  max?: string;
  unit: string;
  percent: number | null;
}) {
  const pct = percent ?? 0;
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-sm text-gray-400">{label}</span>
        <span className={`text-sm font-medium ${getBarBgColor(pct)}`}>
          {value}{max ? ` / ${max}` : ''} {unit}
        </span>
      </div>
      <div className="w-full bg-[#1a1a1a] rounded-full h-2.5">
        <div
          className={`h-2.5 rounded-full transition-all duration-500 ${getBarColor(pct)}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
    </div>
  );
}

function ProfileBadge({ profile, vramGb }: { profile: string; vramGb?: number }) {
  const colors: Record<string, string> = {
    cpu: 'bg-gray-600 text-gray-200',
    small: 'bg-amber-600 text-amber-100',
    medium: 'bg-blue-600 text-blue-100',
    large: 'bg-green-600 text-green-100',
  };
  const label = profile === 'cpu' ? 'CPU Only' : vramGb ? `${vramGb} GB VRAM` : profile;
  return (
    <span className={`px-2.5 py-1 rounded-full text-xs font-medium tracking-wide ${colors[profile] || 'bg-gray-600 text-gray-200'}`}>
      {label}
    </span>
  );
}

function shortStatusText(text: string | undefined, maxLen = 140): string {
  if (!text) return 'Unknown';
  if (text.length <= maxLen) return text;
  return `${text.slice(0, maxLen - 3)}...`;
}

export function StatusTab({ status, onRecalibrate, onRefresh, hardwareProfile }: StatusTabProps) {
  const services = status.services || {};
  const system = status.system;
  const specs = system?.specs;

  // Compute GPU VRAM percent
  const gpuVramPct = (specs?.gpu_vram_total_mb && system?.gpu_vram_used_mb != null)
    ? Math.round((system.gpu_vram_used_mb / specs.gpu_vram_total_mb) * 100)
    : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">System Status</h2>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500">
            Auto-refresh 30s
          </span>
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="px-4 py-2 bg-[#333] hover:bg-[#3a3a3a] rounded-xl text-sm transition-colors"
            >
              Refresh
            </button>
          )}
          <button
            onClick={onRecalibrate}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm transition-colors"
          >
            Recalibrate
          </button>
        </div>
      </div>

      {/* Service Health */}
      <div>
        <h3 className="text-sm text-gray-400 mb-3 font-medium">Service Health</h3>
        <div className="grid grid-cols-6 gap-3">
          {/* llama.cpp */}
          <div className="bg-[#2a2a2a] rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <ServiceDot health={{ status: status.llm.status, connected: status.llm.status === 'connected' }} />
              <span className="text-sm font-medium">llama.cpp</span>
            </div>
            <p className="text-xs text-gray-500">
              {status.llm.status === 'connected'
                ? `${status.llm.loaded_models?.length ?? status.llm.models?.length ?? 0} models`
                : shortStatusText(status.llm.error || 'Disconnected')}
            </p>
          </div>

          {/* Redis */}
          <div className="bg-[#2a2a2a] rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <ServiceDot health={services.redis} />
              <span className="text-sm font-medium">Redis</span>
            </div>
            <p className="text-xs text-gray-500">
              {services.redis?.status === 'connected'
                ? `${services.redis.latency_ms ? Math.round(services.redis.latency_ms) + 'ms' : 'OK'}`
                : services.redis?.status === 'fallback'
                  ? 'In-memory fallback'
                  : shortStatusText(services.redis?.error)}
            </p>
          </div>

          {/* PostgreSQL */}
          <div className="bg-[#2a2a2a] rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <ServiceDot health={services.postgres} />
              <span className="text-sm font-medium">PostgreSQL</span>
            </div>
            <p className="text-xs text-gray-500">
              {services.postgres?.status === 'connected'
                ? `${services.postgres.latency_ms ? Math.round(services.postgres.latency_ms) + 'ms' : 'OK'}`
                : services.postgres?.status === 'fallback'
                  ? 'SQLite fallback'
                  : shortStatusText(services.postgres?.error)}
            </p>
          </div>

          {/* Qdrant */}
          <div className="bg-[#2a2a2a] rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <ServiceDot health={services.qdrant} />
              <span className="text-sm font-medium">Qdrant</span>
            </div>
            <p className="text-xs text-gray-500">
              {services.qdrant?.status === 'connected'
                ? `${services.qdrant.latency_ms ? Math.round(services.qdrant.latency_ms) + 'ms' : 'OK'}`
                : shortStatusText(services.qdrant?.error)}
            </p>
          </div>

          {/* Neo4j */}
          <div className="bg-[#2a2a2a] rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <ServiceDot health={services.neo4j} />
              <span className="text-sm font-medium">Neo4j</span>
            </div>
            <p className="text-xs text-gray-500">
              {services.neo4j?.status === 'healthy' || services.neo4j?.connected
                ? 'Connected'
                : services.neo4j?.status === 'disabled'
                  ? 'Disabled'
                  : shortStatusText(services.neo4j?.error)}
            </p>
          </div>

          {/* SearXNG */}
          <div className="bg-[#2a2a2a] rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <ServiceDot health={services.searxng} />
              <span className="text-sm font-medium">SearXNG</span>
            </div>
            <p className="text-xs text-gray-500">
              {services.searxng?.status === 'connected'
                ? `${services.searxng.latency_ms ? Math.round(services.searxng.latency_ms) + 'ms' : 'OK'}`
                : shortStatusText(services.searxng?.error || 'Disconnected')}
            </p>
          </div>
        </div>
      </div>

      {/* System Resources */}
      {system && (
        <div>
          <h3 className="text-sm text-gray-400 mb-3 font-medium">System Resources</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-[#2a2a2a] rounded-lg p-4 space-y-4">
              <ResourceBar
                label="CPU"
                value={system.cpu_percent != null ? `${system.cpu_percent}%` : '--'}
                unit=""
                percent={system.cpu_percent}
              />
              <ResourceBar
                label="RAM"
                value={system.ram_used_gb != null ? `${system.ram_used_gb}` : '--'}
                max={specs ? `${specs.ram_total_gb}` : undefined}
                unit="GB"
                percent={system.ram_percent}
              />
              <ResourceBar
                label="Disk"
                value={system.disk_used_gb != null ? `${system.disk_used_gb}` : '--'}
                max={system.disk_total_gb != null ? `${system.disk_total_gb}` : undefined}
                unit="GB"
                percent={system.disk_percent}
              />
            </div>
            <div className="bg-[#2a2a2a] rounded-lg p-4 space-y-4">
              <ResourceBar
                label="GPU Load"
                value={system.gpu_utilization_pct != null ? `${system.gpu_utilization_pct}%` : '--'}
                unit=""
                percent={system.gpu_utilization_pct}
              />
              {/* GPU VRAM — always green (high usage is expected, not alarming) */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-sm text-gray-400">GPU VRAM</span>
                  <span className="text-sm font-medium text-green-400">
                    {system.gpu_vram_used_mb != null ? `${(system.gpu_vram_used_mb / 1024).toFixed(1)}` : '--'}
                    {specs?.gpu_vram_total_mb ? ` / ${(specs.gpu_vram_total_mb / 1024).toFixed(1)}` : ''} GB
                  </span>
                </div>
                <div className="w-full bg-[#1a1a1a] rounded-full h-2.5">
                  <div
                    className="h-2.5 rounded-full transition-all duration-500 bg-green-500"
                    style={{ width: `${Math.min(gpuVramPct ?? 0, 100)}%` }}
                  />
                </div>
              </div>
              <ResourceBar
                label="GPU Power"
                value={system.gpu_power_w != null ? `${system.gpu_power_w}` : '--'}
                max={system.gpu_power_limit_w != null ? `${system.gpu_power_limit_w}` : undefined}
                unit="W"
                percent={system.gpu_power_w != null && system.gpu_power_limit_w
                  ? Math.round((system.gpu_power_w / system.gpu_power_limit_w) * 100)
                  : null}
              />
            </div>
            <div className="bg-[#2a2a2a] rounded-lg p-4 space-y-4">
              {/* CPU Temperature */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-sm text-gray-400">CPU Temperature</span>
                  <span className={`text-sm font-medium ${system.cpu_temperature_c != null ? getTempTextColor(system.cpu_temperature_c) : 'text-gray-500'}`}>
                    {system.cpu_temperature_c != null ? `${system.cpu_temperature_c}°C` : (
                      <span className="text-gray-600 text-xs" title="CPU sensors unavailable in containerized/WSL environments">N/A</span>
                    )}
                  </span>
                </div>
                <div className="w-full bg-[#1a1a1a] rounded-full h-2.5">
                  <div
                    className={`h-2.5 rounded-full transition-all duration-500 ${system.cpu_temperature_c != null ? getTempBarColor(system.cpu_temperature_c) : 'bg-gray-600'}`}
                    style={{ width: `${system.cpu_temperature_c != null ? Math.min(system.cpu_temperature_c, 100) : 0}%` }}
                  />
                </div>
              </div>
              {/* GPU Temperature */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-sm text-gray-400">GPU Temperature</span>
                  <span className={`text-sm font-medium ${system.gpu_temperature_c != null ? getTempTextColor(system.gpu_temperature_c) : 'text-gray-500'}`}>
                    {system.gpu_temperature_c != null ? `${system.gpu_temperature_c}°C` : (
                      <span className="text-gray-600 text-xs" title="GPU sensors unavailable — nvidia-smi may not expose temperature in this environment">N/A</span>
                    )}
                  </span>
                </div>
                <div className="w-full bg-[#1a1a1a] rounded-full h-2.5">
                  <div
                    className={`h-2.5 rounded-full transition-all duration-500 ${system.gpu_temperature_c != null ? getTempBarColor(system.gpu_temperature_c) : 'bg-gray-600'}`}
                    style={{ width: `${system.gpu_temperature_c != null ? Math.min(system.gpu_temperature_c, 100) : 0}%` }}
                  />
                </div>
              </div>
              {/* Drive Temperature */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-sm text-gray-400">Drive Temperature</span>
                  <span className={`text-sm font-medium ${system.drive_temperature_c != null ? getTempTextColor(system.drive_temperature_c) : 'text-gray-500'}`}>
                    {system.drive_temperature_c != null ? `${system.drive_temperature_c}°C` : (
                      <span className="text-gray-600 text-xs" title="Drive sensors unavailable in containerized/WSL environments">N/A</span>
                    )}
                  </span>
                </div>
                <div className="w-full bg-[#1a1a1a] rounded-full h-2.5">
                  <div
                    className={`h-2.5 rounded-full transition-all duration-500 ${system.drive_temperature_c != null ? getTempBarColor(system.drive_temperature_c) : 'bg-gray-600'}`}
                    style={{ width: `${system.drive_temperature_c != null ? Math.min(system.drive_temperature_c, 100) : 0}%` }}
                  />
                </div>
              </div>
              {/* Note when sensors unavailable */}
              {system.cpu_temperature_c == null && system.drive_temperature_c == null && (
                <p className="text-[10px] text-gray-600 mt-1">Sensor data unavailable — common in Docker/WSL environments</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Model Resource Allocation — full-width GPU/CPU split bars */}
      {status.llm.loaded_models && status.llm.loaded_models.length > 0 && (
        <div>
          <h3 className="text-sm text-gray-400 mb-3 font-medium">Model Resource Allocation</h3>
          <div className="bg-[#2a2a2a] rounded-lg p-4 space-y-3">
            {status.llm.loaded_models.map((m) => {
              const parallel = m.parallel ?? 1;
              const slotLabel = m.slot === 'vision' ? 'Vision' : m.slot === 'chat' ? 'Chat' : '';
              return (
                <div key={m.name + (m.slot || '')}>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-sm text-gray-400">
                      {slotLabel && <span className="text-gray-500 mr-1.5">[{slotLabel}]</span>}
                      {(() => { const f = formatModelName(m.name); return <>{f.display}{(f.quant || m.quantization) && <span className="text-gray-600 ml-1.5">{f.quant || m.quantization}</span>}</>; })()}
                      {parallel > 1 && <span className="text-blue-400 ml-1.5">x{parallel}</span>}
                    </span>
                    <span className={`text-sm font-medium ${m.cpu_gb > 0 ? 'text-amber-400' : 'text-green-400'}`}>
                      {m.cpu_gb > 0
                        ? `${m.gpu_pct}% GPU / ${100 - m.gpu_pct}% CPU`
                        : '100% GPU'}
                    </span>
                  </div>
                  {parallel > 1 ? (
                    <div className="space-y-1.5">
                      {Array.from({ length: parallel }, (_, i) => (
                        <div key={i}>
                          <div className="flex items-center justify-between mb-0.5">
                            <span className="text-[10px] text-gray-600">Worker {i + 1}</span>
                            <span className={`text-[10px] font-medium ${m.cpu_gb > 0 ? 'text-amber-400' : 'text-green-400'}`}>
                              {m.cpu_gb > 0
                                ? `${m.gpu_pct}% GPU / ${100 - m.gpu_pct}% CPU`
                                : '100% GPU'}
                            </span>
                          </div>
                          <div className="w-full bg-[#1a1a1a] rounded-full h-2 flex overflow-hidden">
                            <div
                              className="h-2 bg-green-500 transition-all duration-500"
                              style={{ width: `${m.gpu_pct}%` }}
                              title={`Worker ${i + 1} — GPU: ${m.vram_gb} GB`}
                            />
                            {m.cpu_gb > 0 && (
                              <div
                                className="h-2 bg-red-500 transition-all duration-500"
                                style={{ width: `${100 - m.gpu_pct}%` }}
                                title={`Worker ${i + 1} — CPU: ${m.cpu_gb} GB`}
                              />
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="w-full bg-[#1a1a1a] rounded-full h-2.5 flex overflow-hidden">
                      <div
                        className="h-2.5 bg-green-500 transition-all duration-500"
                        style={{ width: `${m.gpu_pct}%` }}
                        title={`GPU: ${m.vram_gb} GB`}
                      />
                      {m.cpu_gb > 0 && (
                        <div
                          className="h-2.5 bg-red-500 transition-all duration-500"
                          style={{ width: `${100 - m.gpu_pct}%` }}
                          title={`CPU: ${m.cpu_gb} GB`}
                        />
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Hardware Card */}
      {specs && (
        <div>
          <div className="flex items-center gap-3 mb-3">
            <h3 className="text-sm text-gray-400 font-medium">Hardware</h3>
            {hardwareProfile?.profile && (
              <ProfileBadge profile={hardwareProfile.profile} vramGb={hardwareProfile?.gpu?.vram_total_gb} />
            )}
            {hardwareProfile?.profile_description && (
              <span className="text-xs text-gray-500">{hardwareProfile.profile_description}</span>
            )}
          </div>
          <div className="bg-[#2a2a2a] rounded-lg p-4 space-y-4">
            <div className="grid grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-500 block text-xs mb-1">CPU</span>
                <span className="text-gray-300">{specs.cpu_model || 'Unknown'}</span>
                <span className="text-gray-500 text-xs ml-1">({specs.cpu_cores} cores)</span>
              </div>
              <div>
                <span className="text-gray-500 block text-xs mb-1">RAM</span>
                <span className="text-gray-300">{specs.ram_total_gb} GB</span>
              </div>
              <div>
                <span className="text-gray-500 block text-xs mb-1">GPU</span>
                <span className="text-gray-300">{specs.gpu_name || 'None'}</span>
              </div>
              <div>
                <span className="text-gray-500 block text-xs mb-1">GPU VRAM</span>
                <span className="text-gray-300">
                  {specs.gpu_vram_total_mb ? `${(specs.gpu_vram_total_mb / 1024).toFixed(1)} GB` : 'N/A'}
                  {hardwareProfile?.gpu?.vram_available_gb != null && specs.gpu_vram_total_mb > 0 && (
                    <span className="text-gray-500 text-xs ml-1">
                      ({hardwareProfile.gpu.vram_available_gb} GB free)
                    </span>
                  )}
                </span>
              </div>
            </div>

            {/* Model Recommendations */}
            {hardwareProfile?.model_recommendations && Object.keys(hardwareProfile.model_recommendations).length > 0 && (
              <div className="border-t border-[#3a3a3a] pt-3">
                <span className="text-xs text-gray-500 block mb-2">Recommended Models</span>
                <div className="grid grid-cols-3 gap-3">
                  {Object.entries(hardwareProfile.model_recommendations).map(([slot, rec]) => (
                    <div key={slot} className="text-xs">
                      <span className="text-gray-500 uppercase tracking-wide">{slot}</span>
                      <p className="text-gray-400 mt-0.5">{rec as string}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        </div>
      )}

      {/* Storage, Knowledge Base, Available Models — existing cards */}
      <div className="grid grid-cols-2 gap-4">
        {/* Storage */}
        <div className="bg-[#2a2a2a] rounded-lg p-4">
          <h3 className="text-sm text-gray-400 mb-2">Storage</h3>
          <p className="text-lg font-medium">{Number(status.storage.total_mb).toFixed(1)} MB</p>
          <p className="text-xs text-gray-500 mt-1">
            {status.storage.uploads_count} uploads, {status.storage.reports_count} reports
          </p>
          {status.storage.note && (
            <p className="text-xs text-amber-400 mt-1">{status.storage.note}</p>
          )}
        </div>

        {/* RAG */}
        <div className="bg-[#2a2a2a] rounded-lg p-4">
          <h3 className="text-sm text-gray-400 mb-2">Knowledge Base</h3>
          <p className="text-lg font-medium">{status.rag.knowledge_chunks.toLocaleString()} chunks</p>
          {status.rag.topics && Array.isArray(status.rag.topics) && status.rag.topics.length > 0 && (
            <p className="text-xs text-gray-500 mt-1">
              Topics: {status.rag.topics.join(', ')}
            </p>
          )}
        </div>

        {/* LLM error detail (only shown when something is wrong) */}
        {status.llm.status !== 'connected' && (
          <div className="bg-[#2a2a2a] rounded-lg p-4">
            <h3 className="text-sm text-gray-400 mb-2">LLM Server</h3>
            <p className="text-lg font-medium text-red-400">{status.llm.status}</p>
            {status.llm.error && (
              <p className="text-xs text-red-400 mt-2 break-words">{status.llm.error}</p>
            )}
          </div>
        )}
      </div>

      {/* Models List */}
      {status.llm.models && status.llm.models.length > 0 && (
        <div className="bg-[#2a2a2a] rounded-lg p-4">
          <h3 className="text-sm text-gray-400 mb-3">Available Models</h3>
          <div className="flex flex-wrap gap-2">
            {status.llm.models.map((model) => {
              const f = formatModelName(model);
              return (
                <span key={model} className="px-2 py-1 bg-[#1a1a1a] rounded-xl text-sm">
                  {f.display}{f.quant && <span className="text-gray-500 ml-1">{f.quant}</span>}
                </span>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
