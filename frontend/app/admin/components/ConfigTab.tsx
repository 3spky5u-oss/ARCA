'use client';

import React, { useState } from 'react';
import { SettingTip } from './shared';
import { SubTabLayout } from './SubTabLayout';
import { ModelsTab } from './ModelsTab';

// Config is a dynamic object from the API — use Record<string, any> for flexibility
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ConfigValues = Record<string, any>;

interface ServiceStatus {
  llm: { status: string; error?: string };
  redis?: { status: string };
  postgres?: { status: string };
  neo4j?: { status: string };
  qdrant?: { status: string };
}

interface ModelInfo {
  name: string;
  size_gb: number;
  modified?: string;
  family?: string;
  assigned_to: string[];
}

interface ModelsData {
  models: ModelInfo[];
  config_assignments: Record<string, string>;
  total_size_gb: number;
  models_dir?: {
    path: string;
    exists: boolean;
    readable: boolean;
    writable: boolean;
    error?: string | null;
  };
}

interface PullStatus {
  status: string;
  progress_percent?: number;
  completed_gb?: number;
  total_gb?: number;
  done?: boolean;
  error?: string;
}

interface SearxngStatus {
  status: 'connected' | 'disabled' | 'error' | 'unknown';
  healthy: boolean;
  url: string;
  error?: string;
}

interface ConfigTabProps {
  config: ConfigValues;
  editedConfig: ConfigValues;
  serviceStatus: ServiceStatus | null;
  onEditConfig: (updates: ConfigValues) => void;
  onSave: () => void;
  onReset: () => void;
  // Model management props
  modelsData: ModelsData | null;
  pullModelName: string;
  onPullModelNameChange: (name: string) => void;
  pullStatus: PullStatus | null;
  pullingModel: string | null;
  onRefreshModels: () => void;
  onPullModel: () => void;
  onAssignModel: (slot: string, model: string) => void;
  onDeleteModel: (name: string) => void;
  // SearXNG status
  searxngStatus?: SearxngStatus | null;
  // Hardware info for context-aware warnings
  vramTotalGb?: number | null;
  // Optional controlled sub-tab state from parent
  subTab?: ConfigSubTab;
  onSubTabChange?: (tab: ConfigSubTab) => void;
}

// Reusable field components
function NumberField({ label, tip, value, onChange, step, min, max }: {
  label: string; tip?: string; value: number;
  onChange: (v: number) => void; step?: number; min?: number; max?: number;
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="w-40 text-sm shrink-0">
        {label}
        {tip && <SettingTip tip={tip} />}
      </label>
      <input
        type="number" step={step} min={min} max={max} value={value}
        onChange={(e) => onChange(step && step < 1 ? parseFloat(e.target.value) : parseInt(e.target.value))}
        className="flex-1 px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
      />
    </div>
  );
}

function TextField({ label, tip, value, onChange }: {
  label: string; tip?: string; value: string; onChange: (v: string) => void;
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="w-40 text-sm shrink-0">
        {label}
        {tip && <SettingTip tip={tip} />}
      </label>
      <input
        type="text" value={value}
        onChange={(e) => onChange(e.target.value)}
        className="flex-1 px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
      />
    </div>
  );
}

function ToggleField({ label, tip, value, onChange }: {
  label: string; tip?: string; value: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="w-40 text-sm shrink-0">
        {label}
        {tip && <SettingTip tip={tip} />}
      </label>
      <button
        onClick={() => onChange(!value)}
        className={`relative w-11 h-6 rounded-full transition-colors ${value ? 'bg-blue-600' : 'bg-gray-600'}`}
      >
        <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform ${value ? 'translate-x-5' : ''}`} />
      </button>
    </div>
  );
}

function SelectField({ label, tip, value, onChange, options }: {
  label: string; tip?: string; value: string;
  onChange: (v: string) => void;
  options: Array<{ value: string; label: string }>;
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="w-40 text-sm shrink-0">
        {label}
        {tip && <SettingTip tip={tip} />}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="flex-1 px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </div>
  );
}

function DisplayField({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-3">
      <label className="w-40 text-sm shrink-0 text-gray-400">{label}</label>
      <span className="flex-1 min-w-0 px-3 py-2 bg-[#1a1a1a] border border-[#333] rounded-xl text-sm text-gray-500 font-mono truncate">
        {value}
      </span>
    </div>
  );
}

function StatusDot({ status }: { status: 'connected' | 'enabled' | 'disabled' | 'error' | 'unknown' }) {
  const colors = {
    connected: 'bg-green-400',
    enabled: 'bg-green-400',
    disabled: 'bg-gray-500',
    error: 'bg-red-400',
    unknown: 'bg-yellow-400',
  };
  const labels = {
    connected: 'Connected',
    enabled: 'Enabled',
    disabled: 'Disabled',
    error: 'Error',
    unknown: 'Unknown',
  };
  return (
    <span className="relative inline-flex ml-2 group">
      <span className={`inline-block w-2 h-2 rounded-full ${colors[status]}`} />
      <span className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 px-2 py-1
        text-xs text-gray-200 bg-[#1a1a1a] border border-[#3a3a3a] rounded-lg
        opacity-0 pointer-events-none group-hover:opacity-100
        transition-opacity z-30 whitespace-nowrap">
        {labels[status]}
      </span>
    </span>
  );
}

function SectionCard({ title, status, children }: { title: string; status?: 'connected' | 'enabled' | 'disabled' | 'error' | 'unknown'; children: React.ReactNode }) {
  return (
    <div className="bg-[#2a2a2a] rounded-lg p-4 h-full overflow-visible min-w-0">
      <h3 className="text-sm text-gray-400 mb-4 font-medium">
        {title}
        {status && <StatusDot status={status} />}
      </h3>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

// Helper to get value with edit override
function val<T>(config: ConfigValues, edited: ConfigValues, key: string): T {
  return (edited[key] ?? config[key]) as T;
}

function set(edited: ConfigValues, onEdit: (u: ConfigValues) => void, key: string, value: unknown) {
  onEdit({ ...edited, [key]: value });
}

type ConfigSubTab = 'general' | 'models' | 'connections';

const SUB_TABS = [
  { id: 'general', label: 'General' },
  { id: 'models', label: 'Models' },
  { id: 'connections', label: 'Connections' },
];

const SEARXNG_CATEGORY_OPTIONS = [
  { value: 'general', label: 'General' },
  { value: 'science', label: 'Science' },
  { value: 'it', label: 'Tech/IT' },
  { value: 'news', label: 'News' },
  { value: 'files', label: 'Files/PDF' },
  { value: 'social media', label: 'Social' },
  { value: 'images', label: 'Images' },
  { value: 'videos', label: 'Videos' },
  { value: 'map', label: 'Maps' },
  { value: 'music', label: 'Music' },
];

function parseCsvList(value: string): string[] {
  return value
    .split(',')
    .map((v) => v.trim())
    .filter(Boolean);
}

export function ConfigTab({ config, editedConfig, serviceStatus, onEditConfig, onSave, onReset, modelsData, pullModelName, onPullModelNameChange, pullStatus, pullingModel, onRefreshModels, onPullModel, onAssignModel, onDeleteModel, searxngStatus, vramTotalGb, subTab, onSubTabChange }: ConfigTabProps) {
  const [internalSubTab, setInternalSubTab] = useState<ConfigSubTab>('general');
  const activeSubTab = subTab ?? internalSubTab;
  const setActiveSubTab = onSubTabChange ?? setInternalSubTab;
  const hasChanges = Object.keys(editedConfig).length > 0;

  const v = <T,>(key: string): T => val<T>(config, editedConfig, key);
  const s = (key: string, value: unknown) => set(editedConfig, onEditConfig, key, value);
  const searxCategories = parseCsvList(v<string>('searxng_categories') || 'general');
  const searxCategorySet = new Set(searxCategories);

  const toggleSearxCategory = (category: string) => {
    const next = new Set(searxCategorySet);
    if (next.has(category)) {
      next.delete(category);
    } else {
      next.add(category);
    }
    if (next.size === 0) {
      next.add('general');
    }
    s('searxng_categories', Array.from(next).join(','));
  };

  const setSearxPreset = (categories: string[]) => {
    s('searxng_categories', categories.join(','));
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">Configuration</h2>
        <div className="flex items-center gap-3">
          {hasChanges && (
            <span className="text-xs text-amber-400">Unsaved changes</span>
          )}
          <button onClick={onReset}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-xl text-sm transition-colors">
            Reset to Defaults
          </button>
          <button onClick={onSave} disabled={!hasChanges}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors">
            Save Changes
          </button>
        </div>
      </div>

      <SubTabLayout
        tabs={SUB_TABS}
        activeTab={activeSubTab}
        onTabChange={(tab) => setActiveSubTab(tab as ConfigSubTab)}
      >
        {/* GENERAL SUB-TAB */}
        {activeSubTab === 'general' && (
          <div className="grid grid-cols-2 gap-6">
            <SectionCard title="Context Window Sizes">
              <NumberField label="Small" tip="Simple tools (unit convert)" value={v<number>('ctx_small')} onChange={(n) => s('ctx_small', n)} min={512} />
              <NumberField label="Medium" tip="Default chat context" value={v<number>('ctx_medium')} onChange={(n) => s('ctx_medium', n)} min={1024} />
              <NumberField label="Large" tip="RAG-augmented queries" value={v<number>('ctx_large')} onChange={(n) => s('ctx_large', n)} min={4096} />
              <NumberField label="XLarge" tip="Think mode / code generation" value={v<number>('ctx_xlarge')} onChange={(n) => s('ctx_xlarge', n)} min={8192} />
              {v<number>('ctx_xlarge') > 16384 && (() => {
                const ctx = v<number>('ctx_xlarge');
                if (vramTotalGb != null && vramTotalGb >= 24 && ctx <= 32768) return null;
                const needed = ctx > 32768 ? '24+' : '16+';
                const msg = vramTotalGb != null
                  ? `Context ${ctx} may exceed your ${vramTotalGb.toFixed(0)} GB VRAM (needs ${needed} GB)`
                  : `Context sizes above 16384 may require ${needed} GB VRAM`;
                return <p className="text-xs text-amber-400 mt-1">{msg}</p>;
              })()}
            </SectionCard>

            <SectionCard title="KV Cache">
              <SelectField
                label="Cache Type (K)"
                tip="KV cache quantization for keys. Lower = less VRAM, slight quality tradeoff"
                value={v<string>('kv_cache_type_k') || 'q8_0'}
                onChange={(val) => s('kv_cache_type_k', val)}
                options={[
                  { value: 'f16', label: 'f16 -- Full precision (most VRAM)' },
                  { value: 'q8_0', label: 'q8_0 -- 8-bit (recommended)' },
                  { value: 'q4_0', label: 'q4_0 -- 4-bit (least VRAM)' },
                ]}
              />
              <SelectField
                label="Cache Type (V)"
                tip="KV cache quantization for values. Match K type unless experimenting"
                value={v<string>('kv_cache_type_v') || 'q8_0'}
                onChange={(val) => s('kv_cache_type_v', val)}
                options={[
                  { value: 'f16', label: 'f16 -- Full precision' },
                  { value: 'q8_0', label: 'q8_0 -- 8-bit (recommended)' },
                  { value: 'q4_0', label: 'q4_0 -- 4-bit' },
                ]}
              />
            </SectionCard>

            <SectionCard title="Model Parameters">
              <NumberField label="Temperature" tip="0.1-0.3 = focused, 0.7-1.0 = creative" value={v<number>('temperature')} onChange={(n) => s('temperature', n)} step={0.1} min={0} max={2} />
              <NumberField label="Top P" tip="Nucleus sampling. 0.9 = top 90% probability mass" value={v<number>('top_p')} onChange={(n) => s('top_p', n)} step={0.05} min={0} max={1} />
              <NumberField label="Top K" tip="Token choice limit. Lower = more focused" value={v<number>('top_k')} onChange={(n) => s('top_k', n)} min={1} max={200} />
              <NumberField label="Max Output Tokens" tip="Cap output length per response" value={v<number>('max_output_tokens')} onChange={(n) => s('max_output_tokens', n)} min={256} />
            </SectionCard>

            <SectionCard title="Streaming">
              <NumberField label="Token Delay (ms)" tip="50ms = ~20 words/sec readable pace" value={v<number>('stream_token_delay_ms')} onChange={(n) => s('stream_token_delay_ms', n)} min={0} max={200} />
            </SectionCard>

            <SectionCard title="Timeouts">
              <NumberField label="LLM Timeout (s)" tip="General LLM request timeout" value={v<number>('llm_timeout')} onChange={(n) => s('llm_timeout', n)} min={30} max={600} />
              <NumberField label="Think Timeout (s)" tip="Extended reasoning timeout" value={v<number>('llm_timeout_think')} onChange={(n) => s('llm_timeout_think', n)} min={60} max={900} />
            </SectionCard>

            <SectionCard title="Startup">
              <ToggleField
                label="Cleanup on Boot"
                tip="Wipe sessions, uploads, and reports on restart. Enable for shared/production deployments."
                value={v<boolean>('cleanup_on_startup')}
                onChange={(b) => s('cleanup_on_startup', b)}
              />
              <p className="text-xs text-gray-500 mt-1">Takes effect on next restart.</p>
            </SectionCard>

            <SectionCard title="Phii Behavior">
              <ToggleField label="Energy Matching" tip="Match user's communication energy/style" value={v<boolean>('phii_energy_matching')} onChange={(b) => s('phii_energy_matching', b)} />
              <ToggleField label="Specialty Detection" tip="Detect user's engineering specialty" value={v<boolean>('phii_specialty_detection')} onChange={(b) => s('phii_specialty_detection', b)} />
              <ToggleField label="Implicit Feedback" tip="Learn from user corrections" value={v<boolean>('phii_implicit_feedback')} onChange={(b) => s('phii_implicit_feedback', b)} />
            </SectionCard>

            <SectionCard title="Topics">
              <TextField label="Enabled Topics" tip="Comma-separated knowledge bases to search" value={v<string>('enabled_topics')} onChange={(t) => s('enabled_topics', t)} />
            </SectionCard>

            <SectionCard title="RAG & Reranker">
              <ToggleField label="Reranker Enabled" tip="Cross-encoder reranking for more accurate results" value={v<boolean>('reranker_enabled')} onChange={(b) => s('reranker_enabled', b)} />
              <SelectField
                label="Reranker Device"
                tip="Auto detects GPU. CPU uses less VRAM but is slower. Takes effect on next query."
                value={v<string>('reranker_device') || 'auto'}
                onChange={(val) => s('reranker_device', val)}
                options={[
                  { value: 'auto', label: 'Auto (detect GPU)' },
                  { value: 'cuda', label: 'GPU (CUDA)' },
                  { value: 'cpu', label: 'CPU' },
                ]}
              />
              <NumberField label="Candidates" tip="Chunks sent to reranker. More = better quality, slower" value={v<number>('reranker_candidates')} onChange={(n) => s('reranker_candidates', n)} min={3} max={50} />
              <NumberField label="Batch Size" tip="GPU batch size for scoring. Higher = faster if VRAM allows" value={v<number>('reranker_batch_size')} onChange={(n) => s('reranker_batch_size', n)} min={4} max={128} />
            </SectionCard>

            <SectionCard title="Vision Settings">
              <NumberField label="Context Window" tip="2048 for speed, 4096 recommended, 8192 for complex" value={v<number>('vision_num_ctx')} onChange={(n) => s('vision_num_ctx', n)} min={1024} max={16384} />
              <NumberField label="Timeout (s)" tip="Max seconds per page extraction" value={v<number>('vision_timeout')} onChange={(n) => s('vision_timeout', n)} min={10} max={300} />
              <NumberField label="Max Workers" tip="Parallel vision extractions (batch ingest only)" value={v<number>('vision_max_workers')} onChange={(n) => s('vision_max_workers', n)} min={1} max={4} />
            </SectionCard>

            <SectionCard title="Tool Routing">
              <ToggleField label="Separate Router" tip="If disabled, main LLM handles tool routing natively (recommended)" value={v<boolean>('tool_router_enabled')} onChange={(b) => s('tool_router_enabled', b)} />
            </SectionCard>
          </div>
        )}

        {/* MODELS SUB-TAB — full model management */}
        {activeSubTab === 'models' && modelsData && (
          <ModelsTab
            modelsData={modelsData}
            pullModelName={pullModelName}
            onPullModelNameChange={onPullModelNameChange}
            pullStatus={pullStatus}
            pullingModel={pullingModel}
            onRefresh={onRefreshModels}
            onPullModel={onPullModel}
            onAssignModel={onAssignModel}
            onDeleteModel={onDeleteModel}
          />
        )}

        {/* CONNECTIONS SUB-TAB */}
        {activeSubTab === 'connections' && (
          <div className="grid grid-cols-2 gap-6">
            <SectionCard title="llama.cpp" status={serviceStatus?.llm?.status === 'connected' ? 'connected' : serviceStatus?.llm?.error ? 'error' : 'unknown'}>
              <DisplayField label="Status" value={serviceStatus?.llm?.status || 'unknown'} />
              <ToggleField
                label="MCP Mode"
                tip="Disables chat UI — ARCA serves as a tool backend for Claude Desktop, GPT, etc."
                value={v<boolean>('mcp_mode')}
                onChange={(b) => s('mcp_mode', b)}
              />
              {v<boolean>('mcp_mode') && (
                <p className="text-xs text-amber-400 mt-1">
                  Chat UI disabled. Serving as tool backend for external AI clients.
                </p>
              )}
            </SectionCard>

            <SectionCard title="Redis" status={v<boolean>('redis_enabled') ? (serviceStatus?.redis?.status === 'connected' ? 'connected' : 'enabled') : 'disabled'}>
              <DisplayField label="URL" value={config.redis_url || 'redis://localhost:6379/0'} />
              <ToggleField label="Enabled" value={v<boolean>('redis_enabled')} onChange={(b) => s('redis_enabled', b)} />
              <NumberField label="Session TTL (s)" tip="Session cache expiry (86400 = 24h)" value={v<number>('redis_session_ttl')} onChange={(n) => s('redis_session_ttl', n)} min={300} />
            </SectionCard>

            <SectionCard title="PostgreSQL" status={v<boolean>('database_enabled') ? (serviceStatus?.postgres?.status === 'connected' ? 'connected' : 'enabled') : 'disabled'}>
              <DisplayField label="URL" value={(config.database_url || '').replace(/:[^:@]+@/, ':***@')} />
              <ToggleField label="Enabled" value={v<boolean>('database_enabled')} onChange={(b) => s('database_enabled', b)} />
              <NumberField label="Pool Size" value={v<number>('database_pool_size')} onChange={(n) => s('database_pool_size', n)} min={1} max={100} />
            </SectionCard>

            <SectionCard title="Neo4j" status={serviceStatus?.neo4j?.status === 'connected' ? 'connected' : 'enabled'}>
              <DisplayField label="URL" value={config.neo4j_url || 'bolt://localhost:7687'} />
              <DisplayField label="User" value={config.neo4j_user || 'neo4j'} />
            </SectionCard>

            <SectionCard title="Rate Limiting">
              <NumberField label="WS Connections" tip="WebSocket connections per IP per minute" value={v<number>('rate_limit_ws_conn')} onChange={(n) => s('rate_limit_ws_conn', n)} min={1} max={100} />
              <NumberField label="WS Messages" tip="Messages per session per minute" value={v<number>('rate_limit_ws_msg')} onChange={(n) => s('rate_limit_ws_msg', n)} min={1} max={100} />
              <NumberField label="File Uploads" tip="Uploads per IP per minute" value={v<number>('rate_limit_upload')} onChange={(n) => s('rate_limit_upload', n)} min={1} max={100} />
            </SectionCard>

            <SectionCard title="Ingest Lock">
              <ToggleField label="Ingest Lock" tip="Block chat during ingestion to protect VRAM" value={v<boolean>('ingest_lock_enabled')} onChange={(b) => s('ingest_lock_enabled', b)} />
            </SectionCard>

            <SectionCard
              title="SearXNG"
              status={
                searxngStatus?.status === 'connected' ? 'connected'
                  : searxngStatus?.status === 'disabled' ? 'disabled'
                    : searxngStatus ? 'error'
                      : 'unknown'
              }
            >
              <ToggleField
                label="Enabled"
                tip="Master switch for web search tool execution."
                value={v<boolean>('searxng_enabled') ?? true}
                onChange={(b) => s('searxng_enabled', b)}
              />
              <TextField
                label="URL"
                tip="Internal SearXNG base URL used by backend requests."
                value={v<string>('searxng_url') || searxngStatus?.url || 'http://searxng:8080'}
                onChange={(value) => s('searxng_url', value)}
              />
              <div className="flex items-start gap-3">
                <label className="w-40 text-sm shrink-0">
                  Category Preset
                  <SettingTip tip="Quick presets for common web-search profiles." />
                </label>
                <div className="flex-1 flex flex-wrap gap-2">
                  <button
                    onClick={() => setSearxPreset(['general'])}
                    className="px-2.5 py-1 text-xs rounded-lg bg-[#1a1a1a] border border-[#3a3a3a] hover:border-blue-500"
                  >
                    General
                  </button>
                  <button
                    onClick={() => setSearxPreset(['general', 'science', 'it', 'files'])}
                    className="px-2.5 py-1 text-xs rounded-lg bg-[#1a1a1a] border border-[#3a3a3a] hover:border-blue-500"
                  >
                    Research
                  </button>
                  <button
                    onClick={() => setSearxPreset(['general', 'news'])}
                    className="px-2.5 py-1 text-xs rounded-lg bg-[#1a1a1a] border border-[#3a3a3a] hover:border-blue-500"
                  >
                    News
                  </button>
                  <button
                    onClick={() => setSearxPreset(['general', 'it', 'science', 'news', 'files'])}
                    className="px-2.5 py-1 text-xs rounded-lg bg-[#1a1a1a] border border-[#3a3a3a] hover:border-blue-500"
                  >
                    Balanced
                  </button>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <label className="w-40 text-sm shrink-0">
                  Categories
                  <SettingTip tip="Select one or more categories. These are converted into the backend comma-separated value." />
                </label>
                <div className="flex-1 grid grid-cols-2 gap-2">
                  {SEARXNG_CATEGORY_OPTIONS.map((opt) => {
                    const active = searxCategorySet.has(opt.value);
                    return (
                      <button
                        key={opt.value}
                        onClick={() => toggleSearxCategory(opt.value)}
                        className={`text-left px-2.5 py-1.5 text-xs rounded-lg border transition-colors ${
                          active
                            ? 'bg-blue-600/25 border-blue-500 text-blue-300'
                            : 'bg-[#1a1a1a] border-[#3a3a3a] text-gray-300 hover:border-blue-500'
                        }`}
                      >
                        {opt.label}
                      </button>
                    );
                  })}
                </div>
              </div>
              <TextField
                label="Categories (Raw)"
                tip="Advanced override as a comma-separated list used directly by backend requests."
                value={v<string>('searxng_categories') || 'general'}
                onChange={(value) => s('searxng_categories', value)}
              />
              <TextField
                label="Language"
                tip="Optional SearXNG language code (for example: en-US). Leave blank for auto."
                value={v<string>('searxng_language') || ''}
                onChange={(value) => s('searxng_language', value)}
              />
              <SelectField
                label="Response Format"
                tip="JSON is preferred. HTML mode is for restricted SearXNG instances."
                value={v<string>('searxng_request_format') || 'json'}
                onChange={(value) => s('searxng_request_format', value)}
                options={[
                  { value: 'json', label: 'json' },
                  { value: 'html', label: 'html' },
                ]}
              />
              <NumberField
                label="Timeout (s)"
                tip="HTTP timeout per search request."
                value={v<number>('searxng_timeout_s') ?? 10}
                onChange={(value) => s('searxng_timeout_s', value)}
                step={0.5}
                min={1}
                max={60}
              />
              <NumberField
                label="Max Results"
                tip="Final number of web results returned by web_search."
                value={v<number>('searxng_max_results') ?? 5}
                onChange={(value) => s('searxng_max_results', value)}
                min={1}
                max={25}
              />
              <DisplayField
                label="Status"
                value={
                  searxngStatus?.status === 'connected' ? 'Healthy'
                    : searxngStatus?.status === 'disabled' ? 'Disabled'
                      : searxngStatus?.error || 'Unknown'
                }
              />
              <ToggleField label="CRAG Web Search" tip="Enable web fallback when KB confidence is low" value={v<boolean>('crag_web_search_on_low')} onChange={(b) => s('crag_web_search_on_low', b)} />
            </SectionCard>
          </div>
        )}
      </SubTabLayout>
    </div>
  );
}
