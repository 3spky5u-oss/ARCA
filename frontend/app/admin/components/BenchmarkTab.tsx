'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { getApiBase } from '@/lib/api';
import { AuthImage } from './benchmark/AuthImage';
import { PhaseResultCard } from './benchmark/BenchmarkPhaseCard';
import {
  PHASE_LABELS,
  WINNER_LABELS,
  formatWinnerValue,
  formatTimeRemaining,
} from './benchmark/constants';
import type {
  BenchmarkTabProps,
  BenchmarkResults,
  ChartInfo,
  HistoryRun,
  JobStatus,
  PhaseData,
} from './benchmark/constants';

type RunMode = 'quick' | 'full' | string; // string = individual layer name

export function BenchmarkTab({ apiCall, setMessage }: BenchmarkTabProps) {
  const [mode, setMode] = useState<RunMode>('quick');
  const [reportMarkdown, setReportMarkdown] = useState<string | null>(null);
  const [showReport, setShowReport] = useState(false);
  const [topic, setTopic] = useState('');
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [results, setResults] = useState<{
    results: BenchmarkResults;
    charts: ChartInfo[];
    winners: Record<string, string>;
    llm_analysis?: string | null;
  } | null>(null);
  const [history, setHistory] = useState<HistoryRun[]>([]);
  const [viewingHistory, setViewingHistory] = useState<string | null>(null);
  const [historyResults, setHistoryResults] = useState<{
    results: BenchmarkResults;
    charts: ChartInfo[];
    winners: Record<string, string>;
    llm_analysis?: string | null;
  } | null>(null);
  const [applying, setApplying] = useState(false);
  const [expandedChart, setExpandedChart] = useState<string | null>(null);
  const [corpusFiles, setCorpusFiles] = useState<Array<{name: string; size_mb: number; path: string}>>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  const [generatingQueries, setGeneratingQueries] = useState(false);
  const [generatedQueries, setGeneratedQueries] = useState<string[] | null>(null);
  const [queryTierBreakdown, setQueryTierBreakdown] = useState<Record<string, number> | null>(null);
  const [discussInput, setDiscussInput] = useState('');
  const [discussHistory, setDiscussHistory] = useState<Array<{role: string; content: string}>>([]);
  const [discussing, setDiscussing] = useState(false);

  const [providers, setProviders] = useState<{
    judge: { provider: string; model: string; api_key: string; api_key_env: string; rate_limit: number; base_url: string };
    ceiling: { provider: string; model: string; api_key: string; api_key_env: string; rate_limit: number; base_url: string };
  }>({
    judge: { provider: 'local', model: '', api_key: '', api_key_env: '', rate_limit: 1.0, base_url: '' },
    ceiling: { provider: 'local', model: '', api_key: '', api_key_env: '', rate_limit: 1.0, base_url: '' },
  });
  const [providersDirty, setProvidersDirty] = useState(false);
  const [testingProvider, setTestingProvider] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<Record<string, { success: boolean; message: string }>>({});
  const [showProviders, setShowProviders] = useState(false);
  const [autoTuning, setAutoTuning] = useState(false);
  const [autoTuneIncludeJudge, setAutoTuneIncludeJudge] = useState(false);

  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const fetchHistory = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/benchmark/history') as { runs: HistoryRun[] };
      setHistory(data.runs || []);
    } catch {
      // Silent
    }
  }, [apiCall]);

  const fetchCorpus = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/benchmark/corpus') as { files: Array<{name: string; size_mb: number; path: string}> };
      setCorpusFiles(data.files || []);
    } catch {
      // Silent
    }
  }, [apiCall]);

  const fetchProviders = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/benchmark/providers') as Record<string, any>;
      if (data.judge && data.ceiling) {
        setProviders({
          judge: { provider: 'local', model: '', api_key: '', api_key_env: '', rate_limit: 1.0, base_url: '', ...data.judge },
          ceiling: { provider: 'local', model: '', api_key: '', api_key_env: '', rate_limit: 1.0, base_url: '', ...data.ceiling },
        });
      }
    } catch {
      // Silent
    }
  }, [apiCall]);

  const fetchStatus = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/benchmark/status') as unknown as JobStatus;
      setJobStatus(data);
      return data;
    } catch {
      return null;
    }
  }, [apiCall]);

  const fetchResults = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/benchmark/results') as {
        results: BenchmarkResults;
        charts: ChartInfo[];
        winners: Record<string, string>;
        llm_analysis?: string | null;
        error?: string;
      };
      if (!data.error) {
        setResults(data);
      }
    } catch {
      // Silent
    }
  }, [apiCall]);

  // Initial load
  useEffect(() => {
    fetchHistory();
    fetchCorpus();
    fetchProviders();
    fetchStatus().then((status) => {
      if (status && status.status === 'completed') {
        fetchResults();
      } else if (status && (status.status === 'running' || status.status === 'starting')) {
        startPolling();
      }
    });
    return () => stopPolling();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startPolling = useCallback(() => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      const status = await fetchStatus();
      if (status && status.status === 'completed') {
        stopPolling();
        fetchResults();
        fetchHistory();
        const autoApplied = status.auto_applied?.applied;
        const updatedCount = status.auto_applied?.updated?.length || 0;
        if (autoApplied && updatedCount > 0) {
          setMessage({ type: 'success', text: 'Benchmark completed — ' + updatedCount + ' settings auto-applied' });
        } else {
          setMessage({ type: 'success', text: 'Benchmark completed' });
        }
      } else if (status && status.status === 'failed') {
        stopPolling();
        setMessage({ type: 'error', text: `Benchmark failed: ${status.error || 'Unknown error'}` });
      }
    }, 3000);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchStatus, fetchResults, fetchHistory, setMessage]);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const handleStart = async () => {
    try {
      const data = await apiCall('/api/admin/benchmark/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          phases: mode,  // "quick", "full", or individual layer name
          topic,
          corpus_path: corpusFiles.length > 0 ? corpusFiles[0].path : undefined,
        }),
      }) as { job_id?: string; status?: string; error?: string };

      if (data.error) {
        setMessage({ type: 'error', text: data.error });
        return;
      }

      setResults(null);
      setViewingHistory(null);
      setHistoryResults(null);
      setMessage({ type: 'success', text: `Benchmark started (${mode})` });
      await fetchStatus();
      startPolling();
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to start benchmark: ${e}` });
    }
  };

  const handleApplyWinners = async (winners: Record<string, string>) => {
    setApplying(true);
    try {
      const data = await apiCall('/api/admin/benchmark/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ winners }),
      }) as { success: boolean; updated: string[]; ignored: string[] };

      if (data.success) {
        const updated = (data.updated || []).join(', ');
        setMessage({ type: 'success', text: `Applied: ${updated || 'No changes needed'}` });
      }
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to apply winners: ${e}` });
    } finally {
      setApplying(false);
    }
  };

  const handleViewHistory = async (runId: string) => {
    setViewingHistory(runId);
    try {
      const data = await apiCall(`/api/admin/benchmark/history/${runId}`) as {
        results: BenchmarkResults;
        charts: ChartInfo[];
        winners: Record<string, string>;
        llm_analysis?: string | null;
        error?: string;
      };
      if (data.error) {
        setMessage({ type: 'error', text: data.error });
        setViewingHistory(null);
      } else {
        setHistoryResults(data);
      }
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to load run: ${e}` });
      setViewingHistory(null);
    }
  };

  const handleUploadCorpus = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(`Uploading ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)...`);

    try {
      const formData = new FormData();
      formData.append('file', file);

      await apiCall('/api/admin/benchmark/upload-corpus', {
        method: 'POST',
        body: formData,
      });

      setMessage({ type: 'success', text: `Uploaded ${file.name}` });
      await fetchCorpus();
    } catch (err) {
      setMessage({ type: 'error', text: `Upload failed: ${err}` });
    } finally {
      setUploading(false);
      setUploadProgress(null);
    }
  };

  const handleDeleteCorpus = async (filename: string) => {
    try {
      await apiCall(`/api/admin/benchmark/corpus/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      await fetchCorpus();
    } catch (err) {
      setMessage({ type: 'error', text: `Delete failed: ${err}` });
    }
  };

  const handleGenerateQueries = async () => {
    setGeneratingQueries(true);
    try {
      const data = await apiCall('/api/admin/benchmark/generate-queries', { method: 'POST' }) as { queries?: string[]; error?: string; count?: number; tier_breakdown?: Record<string, number> };
      if (data.error) {
        setMessage({ type: 'error', text: data.error });
      } else {
        setGeneratedQueries(data.queries || []);
        setQueryTierBreakdown(data.tier_breakdown || null);
        const tierSummary = data.tier_breakdown
          ? Object.entries(data.tier_breakdown).map(([k, v]) => `${v} ${k}`).join(', ')
          : '';
        setMessage({ type: 'success', text: `Generated ${data.count} queries${tierSummary ? ': ' + tierSummary : ''}` });
      }
    } catch (err) {
      setMessage({ type: 'error', text: `Query generation failed: ${err}` });
    } finally {
      setGeneratingQueries(false);
    }
  };

  const handleDiscuss = async () => {
    if (!discussInput.trim() || discussing) return;
    const question = discussInput.trim();
    setDiscussHistory(prev => [...prev, { role: 'user', content: question }]);
    setDiscussInput('');
    setDiscussing(true);
    try {
      const resp = await apiCall('/api/admin/benchmark/discuss', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      setDiscussHistory(prev => [...prev, { role: 'assistant', content: (resp as Record<string, string>).answer || 'No response' }]);
    } catch {
      setDiscussHistory(prev => [...prev, { role: 'assistant', content: 'Failed to get response.' }]);
    }
    setDiscussing(false);
  };

  const handleSaveProviders = async () => {
    try {
      await apiCall('/api/admin/benchmark/providers', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(providers),
      });
      setProvidersDirty(false);
      setMessage({ type: 'success', text: 'Provider config saved' });
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to save: ${e}` });
    }
  };

  const handleTestProvider = async (role: 'judge' | 'ceiling') => {
    const cfg = providers[role];
    setTestingProvider(role);
    try {
      const data = await apiCall('/api/admin/benchmark/providers/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          role,
          provider: cfg.provider,
          model: cfg.model,
          api_key: cfg.api_key,
          base_url: cfg.base_url,
        }),
      }) as { success: boolean; message: string };
      setTestResult(prev => ({ ...prev, [role]: data }));
    } catch (e) {
      setTestResult(prev => ({ ...prev, [role]: { success: false, message: String(e) } }));
    } finally {
      setTestingProvider(null);
    }
  };

  const handleAutoTune = async () => {
    setAutoTuning(true);
    try {
      const data = await apiCall('/api/admin/benchmark/auto-tune', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          include_judge: autoTuneIncludeJudge,
        }),
      }) as { job_id?: string; error?: string };

      if (data.error) {
        setMessage({ type: 'error', text: data.error });
        setAutoTuning(false);
        return;
      }

      setResults(null);
      setMessage({ type: 'success', text: 'Auto-tune started' });
      await fetchStatus();
      startPolling();
    } catch (e) {
      setMessage({ type: 'error', text: `Auto-tune failed: ${e}` });
    } finally {
      setAutoTuning(false);
    }
  };

  const updateProvider = (role: 'judge' | 'ceiling', field: string, value: string | number) => {
    setProviders(prev => ({
      ...prev,
      [role]: { ...prev[role], [field]: value },
    }));
    setProvidersDirty(true);
  };

  const isRunning = jobStatus && (jobStatus.status === 'running' || jobStatus.status === 'starting');

  // Decide which results to display
  const displayResults = viewingHistory ? historyResults : results;
  const displayWinners = displayResults?.winners || {};
  const displayAnalysis = displayResults?.llm_analysis;

  const apiBase = getApiBase();

  return (
    <div className="space-y-6">
      {/* Guidance Panel */}
      <div className="bg-blue-900/20 border border-blue-700/30 rounded-xl p-5">
        <h3 className="text-blue-300 font-medium text-sm mb-2">RAG Pipeline Benchmark</h3>
        <p className="text-gray-300 text-sm leading-relaxed">
          Optimize your RAG pipeline for <strong>your corpus</strong>. ARCA will sweep chunking
          configurations, retrieval toggles, and continuous parameters to find the best settings.
        </p>
        <div className="mt-3 flex gap-4 text-xs text-gray-400">
          <span>Quick: Chunking + retrieval optimization (L0+L1)</span>
          <span>Full: All 7 layers including LLM-as-judge evaluation</span>
        </div>
      </div>

      {/* Benchmark Corpus */}
      <div className="bg-[#2a2a2a] rounded-xl p-5">
        <h3 className="text-sm font-medium mb-3">Benchmark Corpus</h3>

        {/* Upload Area */}
        <label className={`block border-2 border-dashed border-[#444] rounded-xl p-6 text-center cursor-pointer hover:border-blue-500/50 transition-colors ${uploading ? 'opacity-50 pointer-events-none' : ''}`}>
          <input
            type="file"
            accept=".pdf,.docx,.txt,.md"
            onChange={handleUploadCorpus}
            className="hidden"
            disabled={uploading}
          />
          {uploading ? (
            <p className="text-sm text-gray-400">{uploadProgress}</p>
          ) : (
            <>
              <p className="text-sm text-gray-300">Drop a PDF or DOCX here, or click to upload</p>
              <p className="text-xs text-gray-500 mt-1">Large files (&gt;100 MB) may take several minutes to process</p>
            </>
          )}
        </label>

        {/* Corpus File List */}
        {corpusFiles.length > 0 && (
          <div className="mt-3 space-y-2">
            {corpusFiles.map((file) => (
              <div key={file.name} className="flex items-center justify-between bg-[#1a1a1a] rounded-xl px-3 py-2">
                <div className="flex items-center gap-2 min-w-0">
                  <svg className="w-4 h-4 text-green-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-sm truncate">{file.name}</span>
                  <span className="text-xs text-gray-500 shrink-0">({file.size_mb} MB)</span>
                </div>
                <button
                  onClick={() => handleDeleteCorpus(file.name)}
                  className="text-xs text-red-400 hover:text-red-300 ml-2 shrink-0"
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Auto-generate queries */}
        {corpusFiles.length > 0 && (
          <div className="mt-3 flex items-center gap-3">
            <button
              onClick={handleGenerateQueries}
              disabled={generatingQueries}
              className="px-3 py-1.5 bg-[#333] hover:bg-[#444] rounded-lg text-xs text-gray-300 transition-colors disabled:opacity-50"
            >
              {generatingQueries ? 'Generating...' : 'Auto-generate test queries'}
            </button>
            {generatedQueries && (
              <span className="text-xs text-gray-500">
                {generatedQueries.length} queries ready
                {queryTierBreakdown && (
                  <span className="ml-1 text-gray-600">
                    ({Object.entries(queryTierBreakdown).map(([tier, count]) => `${count} ${tier}`).join(', ')})
                  </span>
                )}
              </span>
            )}
          </div>
        )}
      </div>

      {/* LLM Providers */}
      <div className="bg-[#2a2a2a] rounded-xl p-5">
        <button
          onClick={() => setShowProviders(!showProviders)}
          className="w-full flex items-center justify-between"
        >
          <h3 className="text-sm font-medium">LLM Providers</h3>
          <svg
            className={`w-4 h-4 text-gray-400 transition-transform ${showProviders ? 'rotate-180' : ''}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {showProviders && (
          <div className="mt-4 space-y-4">
            {(['judge', 'ceiling'] as const).map((role) => (
              <div key={role} className="bg-[#1a1a1a] rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-xs text-gray-400 uppercase tracking-wide">
                    {role === 'judge' ? 'Judge Provider' : 'Ceiling Provider'}
                  </h4>
                  <div className="flex items-center gap-2">
                    {testResult[role] && (
                      <span className={`text-xs ${testResult[role].success ? 'text-green-400' : 'text-red-400'}`}>
                        {testResult[role].message}
                      </span>
                    )}
                    <button
                      onClick={() => handleTestProvider(role)}
                      disabled={testingProvider === role}
                      className="px-2 py-1 bg-[#333] hover:bg-[#444] rounded text-xs text-gray-300 transition-colors disabled:opacity-50"
                    >
                      {testingProvider === role ? 'Testing...' : 'Test'}
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-gray-500 block mb-1">Provider</label>
                    <select
                      value={providers[role].provider}
                      onChange={(e) => updateProvider(role, 'provider', e.target.value)}
                      className="w-full px-2 py-1.5 bg-[#2a2a2a] border border-[#3a3a3a] rounded text-sm text-white"
                    >
                      <option value="local">Local LLM</option>
                      <option value="gemini">Gemini</option>
                      <option value="anthropic">Claude</option>
                      <option value="openai">OpenAI / Custom</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-gray-500 block mb-1">Model</label>
                    <input
                      type="text"
                      value={providers[role].model}
                      onChange={(e) => updateProvider(role, 'model', e.target.value)}
                      placeholder={
                        providers[role].provider === 'local' ? 'default'
                        : providers[role].provider === 'gemini' ? 'gemini-3-flash-preview'
                        : providers[role].provider === 'anthropic' ? 'claude-sonnet-4-5-20250929'
                        : 'gpt-4o'
                      }
                      className="w-full px-2 py-1.5 bg-[#2a2a2a] border border-[#3a3a3a] rounded text-sm text-white placeholder-gray-600"
                    />
                  </div>
                </div>

                {providers[role].provider !== 'local' && (
                  <div className="grid grid-cols-2 gap-3 mt-3">
                    <div>
                      <label className="text-xs text-gray-500 block mb-1">API Key</label>
                      <input
                        type="password"
                        value={providers[role].api_key}
                        onChange={(e) => updateProvider(role, 'api_key', e.target.value)}
                        placeholder="From env var or paste key"
                        className="w-full px-2 py-1.5 bg-[#2a2a2a] border border-[#3a3a3a] rounded text-sm text-white placeholder-gray-600"
                      />
                    </div>
                    {providers[role].provider === 'openai' && (
                      <div>
                        <label className="text-xs text-gray-500 block mb-1">Base URL</label>
                        <input
                          type="text"
                          value={providers[role].base_url}
                          onChange={(e) => updateProvider(role, 'base_url', e.target.value)}
                          placeholder="https://api.openai.com/v1"
                          className="w-full px-2 py-1.5 bg-[#2a2a2a] border border-[#3a3a3a] rounded text-sm text-white placeholder-gray-600"
                        />
                      </div>
                    )}
                  </div>
                )}

                {providers[role].provider === 'local' && (
                  <p className="text-xs text-gray-500 mt-2">
                    Uses your loaded llama.cpp model. Results may vary — consider a cloud
                    provider for more reliable evaluation.
                  </p>
                )}
              </div>
            ))}

            {providersDirty && (
              <button
                onClick={handleSaveProviders}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
              >
                Save Provider Config
              </button>
            )}
          </div>
        )}
      </div>

      {/* Start Controls */}
      <div className="bg-[#2a2a2a] rounded-xl p-5">
        <div className="flex items-center gap-4 flex-wrap">
          {/* Phase Selection Dropdown */}
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Phase</label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              disabled={!!isRunning}
              className="px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-lg text-sm text-white disabled:opacity-50 min-w-[220px] appearance-none cursor-pointer"
            >
              <optgroup label="Presets">
                <option value="quick">Quick (Chunking + Retrieval)</option>
                <option value="full">Full (All 9 Layers)</option>
              </optgroup>
              <optgroup label="Retrieval Optimization">
                <option value="layer0_chunking">L0: Chunking Sweep</option>
                <option value="layer1_retrieval">L1: Retrieval Config</option>
                <option value="layer2_params">L2: Parameter Tuning</option>
              </optgroup>
              <optgroup label="Model Shootout">
                <option value="layer_embed">Embedding Shootout</option>
                <option value="layer_rerank">Reranker Shootout</option>
                <option value="layer_cross">Cross-Model Sweep</option>
                <option value="layer_llm">Chat LLM Comparison</option>
              </optgroup>
              <optgroup label="Answer Evaluation">
                <option value="layer3_answers">L3: Answer Generation</option>
                <option value="layer4_judge">L4: LLM-as-Judge</option>
                <option value="layer5_analysis">L5: Analysis &amp; Charts</option>
                <option value="layer6_failures">L6: Failure Analysis</option>
              </optgroup>
              <optgroup label="Cloud/Frontier Comparison">
                <option value="layer_live">Live Pipeline Test</option>
                <option value="layer_ceiling">Frontier vs Local LLM Ceiling</option>
              </optgroup>
            </select>
          </div>

          {/* Topic */}
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Topic</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              disabled={!!isRunning}
              placeholder="Auto from domain"
              className="px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-lg text-sm text-white w-40 disabled:opacity-50"
            />
          </div>

          {/* Start Button */}
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500 invisible">Action</label>
            <button
              onClick={handleStart}
              disabled={!!isRunning}
              className="px-6 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium transition-colors"
            >
              {isRunning ? 'Running...' : 'Start Benchmark'}
            </button>
          </div>

          {/* Auto-Tune Button */}
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500 invisible">Action</label>
            <button
              onClick={handleAutoTune}
              disabled={!!isRunning || autoTuning}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium transition-colors"
            >
              {autoTuning ? 'Tuning...' : 'Optimize My Pipeline'}
            </button>
          </div>
        </div>

        {/* Auto-tune options */}
        <div className="mt-2 flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-gray-400">
            <input
              type="checkbox"
              checked={autoTuneIncludeJudge}
              onChange={(e) => setAutoTuneIncludeJudge(e.target.checked)}
              disabled={providers.judge.provider === 'local' && !providers.judge.model}
              className="rounded border-gray-600 bg-[#1a1a1a]"
            />
            Include answer quality evaluation (requires judge provider)
          </label>
        </div>

        {/* Phase description hint */}
        <div className="mt-2 text-xs text-gray-500">
          {mode === 'quick' && 'Chunking sweep + retrieval toggle optimization (~15 min)'}
          {mode === 'full' && 'All layers including model shootouts and LLM-as-judge evaluation'}
          {mode === 'layer0_chunking' && 'Sweep chunk size, overlap, and context prefix configurations'}
          {mode === 'layer1_retrieval' && 'Toggle retrieval features (BM25, HyDE, RAPTOR, GraphRAG)'}
          {mode === 'layer2_params' && 'Continuous parameter sweep (weights, thresholds, top_k)'}
          {mode === 'layer_embed' && 'Compare embedding models — requires L0 results'}
          {mode === 'layer_rerank' && 'Compare reranker models — requires L0 results'}
          {mode === 'layer_cross' && 'Top-N chunking x embedding x reranker combinations — requires L0, embed, rerank results'}
          {mode === 'layer_llm' && 'Compare chat LLM models by generating answers — requires L0 results'}
          {mode === 'layer3_answers' && 'Generate answers with optimal retrieval config'}
          {mode === 'layer4_judge' && 'Score answers with configured judge provider — requires L3 results'}
          {mode === 'layer5_analysis' && 'Statistical analysis, heatmaps, and markdown report'}
          {mode === 'layer6_failures' && 'Categorize retrieval and answer failures'}
          {mode === 'layer_live' && 'End-to-end pipeline test via MCP API'}
          {mode === 'layer_ceiling' && 'Compare frontier/cloud LLM answer quality against local LLM'}
        </div>
      </div>

      {/* Progress Display */}
      {isRunning && jobStatus && (
        <div className="bg-[#2a2a2a] rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-white">
              Running: {jobStatus.current_phase ? PHASE_LABELS[jobStatus.current_phase] || jobStatus.current_phase : 'Initializing...'}
            </h3>
            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-400">
                Phase {(jobStatus.phases_completed || 0) + 1} of {jobStatus.phases_total || 0}
              </span>
              <span className="text-sm text-blue-400">
                {jobStatus.estimated_remaining_s != null && (jobStatus.phases_completed || 0) > 0
                  ? (jobStatus.estimated_remaining_s <= 0
                    ? 'Completing...'
                    : formatTimeRemaining(jobStatus.estimated_remaining_s))
                  : 'Calibrating...'}
              </span>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="w-full bg-[#1a1a1a] rounded-full h-3 mb-4">
            <div
              className="bg-blue-500 h-3 rounded-full transition-all duration-500"
              style={{ width: `${jobStatus.progress_pct || 0}%` }}
            />
          </div>

          {/* Phase Indicators */}
          <div className="flex gap-2 flex-wrap">
            {(jobStatus.phases || []).map((phase, idx) => {
              const completed = idx < (jobStatus.phases_completed || 0);
              const active = phase === jobStatus.current_phase;
              return (
                <span
                  key={phase}
                  className={`px-3 py-1 rounded-full text-xs font-medium ${
                    completed
                      ? 'bg-green-900/40 text-green-400 border border-green-700/30'
                      : active
                        ? 'bg-blue-900/40 text-blue-400 border border-blue-700/30 animate-pulse'
                        : 'bg-[#1a1a1a] text-gray-500 border border-[#3a3a3a]'
                  }`}
                >
                  {PHASE_LABELS[phase] || phase}
                </span>
              );
            })}
          </div>

          {/* Partial Results */}
          {jobStatus.results_so_far && Object.keys(jobStatus.results_so_far).length > 0 && (
            <div className="mt-4 pt-4 border-t border-[#3a3a3a]">
              <h4 className="text-xs text-gray-400 mb-2">Completed Phases</h4>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(jobStatus.results_so_far).map(([phase, data]) => (
                  <div key={phase} className="bg-[#1a1a1a] rounded-lg px-3 py-2">
                    <span className="text-xs text-gray-400">{PHASE_LABELS[phase] || phase}</span>
                    <div className="text-sm text-green-400 font-medium">
                      {(data as PhaseData).winner || 'N/A'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Results Display */}
      {displayResults && (
        <div className="space-y-4">
          {viewingHistory && (
            <div className="flex items-center gap-3">
              <button
                onClick={() => { setViewingHistory(null); setHistoryResults(null); }}
                className="text-blue-400 hover:text-blue-300 text-sm"
              >
                &larr; Back to current
              </button>
              <span className="text-gray-400 text-sm">Viewing: {viewingHistory}</span>
            </div>
          )}

          {/* Winners Summary */}
          <div className="bg-[#2a2a2a] rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-white">Winners</h3>
              <div className="flex items-center gap-2">
                {/* View Report Button */}
                <button
                  onClick={async () => {
                    const runId = viewingHistory || (history.length > 0 ? history[0].id : '');
                    if (!runId) return;
                    try {
                      const data = await apiCall(`/api/admin/benchmark/report/${runId}`) as { report?: string; error?: string };
                      if (data.report) {
                        setReportMarkdown(data.report);
                        setShowReport(true);
                      } else {
                        setMessage({ type: 'error', text: 'No report available. Run Layer 5 first.' });
                      }
                    } catch {
                      setMessage({ type: 'error', text: 'Report not available. Run Layer 5 (Analysis) first.' });
                    }
                  }}
                  className="px-3 py-1.5 bg-[#333] hover:bg-[#444] text-gray-300 rounded-lg text-xs font-medium transition-colors"
                >
                  View Report
                </button>
                {Object.keys(displayWinners).length > 0 && (
                <button
                  onClick={() => handleApplyWinners(displayWinners)}
                  disabled={applying}
                  className="px-4 py-1.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg text-xs font-medium transition-colors"
                >
                  {applying ? 'Applying...' : 'Apply Winners to Config'}
                </button>
              )}
              </div>
            </div>

            {displayResults.results.overall_winner && (
              <div className="bg-green-900/20 border border-green-700/30 rounded-lg px-4 py-3 mb-4">
                <div className="text-xs text-green-300 mb-1">Overall Champion</div>
                <div className="text-white font-medium">
                  {displayResults.results.overall_winner.variant}
                  <span className="text-gray-400 text-sm ml-2">
                    ({PHASE_LABELS[displayResults.results.overall_winner.phase] || displayResults.results.overall_winner.phase}) — {displayResults.results.overall_winner.composite.toFixed(3)}
                  </span>
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
              {Object.entries(displayWinners).map(([key, value]) => {
                const label = WINNER_LABELS[key] || PHASE_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                const phaseData = displayResults.results.phases[key];
                const topScore = phaseData?.ranking?.[0]?.composite;
                const isBool = typeof value === 'boolean';
                return (
                  <div key={key} className="bg-[#1a1a1a] rounded-lg px-4 py-3">
                    <div className="text-xs text-gray-400 mb-1">{label}</div>
                    <div className={`text-sm font-medium truncate ${isBool ? (value ? 'text-green-400' : 'text-gray-500') : 'text-white'}`}>
                      {formatWinnerValue(value)}
                    </div>
                    {topScore !== undefined && (
                      <div className="text-xs text-blue-400 mt-1">{topScore.toFixed(3)}</div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* AI Analysis */}
          {displayAnalysis && (
            <div className="bg-[#2a2a2a] rounded-xl p-5">
              <h3 className="text-sm font-medium text-white mb-3">AI Analysis</h3>
              <div className="bg-[#1a1a1a] rounded-lg px-4 py-3 text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
                {displayAnalysis}
              </div>
            </div>
          )}

          {/* Discuss Results */}
          <div className="bg-[#2a2a2a] rounded-xl p-5">
            <h3 className="text-sm font-medium text-white mb-3">Discuss Results</h3>

            {/* Chat history */}
            {discussHistory.length > 0 && (
              <div className="space-y-3 mb-4 max-h-64 overflow-y-auto">
                {discussHistory.map((msg, i) => (
                  <div key={i} className={`text-sm ${msg.role === 'user' ? 'text-blue-400' : 'text-gray-300'}`}>
                    <span className="text-gray-500 text-xs">{msg.role === 'user' ? 'You' : 'ARCA'}:</span>
                    <p className="whitespace-pre-wrap mt-0.5">{msg.content}</p>
                  </div>
                ))}
                {discussing && (
                  <div className="text-sm text-gray-500">Analyzing...</div>
                )}
              </div>
            )}

            {/* Starter chips */}
            {discussHistory.length === 0 && (
              <div className="flex flex-wrap gap-2 mb-3">
                {['What should I tune next?', 'Why did the winner beat the others?', 'Are any phases bottlenecking?'].map(q => (
                  <button
                    key={q}
                    onClick={() => { setDiscussInput(q); }}
                    className="px-3 py-1 text-xs bg-[#333] hover:bg-[#3a3a3a] text-gray-400 rounded-full transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}

            {/* Input */}
            <div className="flex gap-2">
              <input
                type="text"
                value={discussInput}
                onChange={(e) => setDiscussInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleDiscuss()}
                placeholder="Ask about your benchmark results..."
                className="flex-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-blue-500/50"
              />
              <button
                onClick={handleDiscuss}
                disabled={discussing || !discussInput.trim()}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm rounded-lg transition-colors"
              >
                Ask
              </button>
            </div>
          </div>

          {/* Phase Details */}
          {Object.entries(displayResults.results.phases).map(([phase, data]) => (
            <PhaseResultCard key={phase} phase={phase} data={data} />
          ))}

          {/* Charts */}
          {displayResults.charts.length > 0 && (
            <div className="bg-[#2a2a2a] rounded-xl p-5">
              <h3 className="text-sm font-medium text-white mb-4">Charts</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {displayResults.charts.map((chart) => (
                  <div
                    key={chart.filename}
                    className="cursor-pointer group"
                    onClick={() => setExpandedChart(
                      expandedChart === chart.url ? null : chart.url
                    )}
                  >
                    <div className="bg-[#1a1a1a] rounded-lg p-2 border border-[#3a3a3a] group-hover:border-blue-500/50 transition-colors">
                      <AuthImage
                        src={`${apiBase}${chart.url}`}
                        alt={chart.name}
                        className="w-full rounded"
                        loading="lazy"
                      />
                      <div className="text-xs text-gray-400 mt-2 text-center">{chart.name}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Expanded Chart Modal */}
          {expandedChart && (
            <div
              className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-8"
              onClick={() => setExpandedChart(null)}
            >
              <div className="max-w-5xl max-h-full" onClick={(e) => e.stopPropagation()}>
                <AuthImage
                  src={`${apiBase}${expandedChart}`}
                  alt="Expanded chart"
                  className="max-w-full max-h-[85vh] rounded-lg"
                />
                <button
                  onClick={() => setExpandedChart(null)}
                  className="absolute top-4 right-4 text-white/60 hover:text-white text-2xl"
                >
                  &times;
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Report Modal */}
      {showReport && reportMarkdown && (
        <div
          className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-8"
          onClick={() => setShowReport(false)}
        >
          <div
            className="bg-[#1a1a1a] rounded-xl max-w-4xl w-full max-h-[85vh] overflow-y-auto p-8"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4 sticky top-0 bg-[#1a1a1a] pb-3 border-b border-[#333]">
              <h3 className="text-lg font-medium text-white">Benchmark Report</h3>
              <button
                onClick={() => setShowReport(false)}
                className="text-gray-400 hover:text-white text-xl"
              >
                &times;
              </button>
            </div>
            <div className="prose prose-invert prose-sm max-w-none text-gray-300 whitespace-pre-wrap font-mono text-xs leading-relaxed">
              {reportMarkdown}
            </div>
          </div>
        </div>
      )}

      {/* History */}
      <div className="bg-[#2a2a2a] rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-white">History</h3>
          <button
            onClick={fetchHistory}
            className="text-xs text-gray-400 hover:text-white"
          >
            Refresh
          </button>
        </div>

        {history.length === 0 ? (
          <p className="text-gray-500 text-sm">No previous benchmark runs found.</p>
        ) : (
          <div className="space-y-2">
            {history.map((run) => (
              <div
                key={run.id}
                className={`flex items-center justify-between px-4 py-3 rounded-lg cursor-pointer transition-colors ${
                  viewingHistory === run.id
                    ? 'bg-blue-900/30 border border-blue-700/30'
                    : 'bg-[#1a1a1a] hover:bg-[#333]'
                }`}
                onClick={() => handleViewHistory(run.id)}
              >
                <div>
                  <div className="text-sm text-white">
                    {run.timestamp_str || run.timestamp.replace('_', ' ')}
                  </div>
                  <div className="text-xs text-gray-400 mt-0.5">
                    {(run.phases || []).map(p => PHASE_LABELS[p] || p).join(', ')}
                    {run.chart_count ? ` — ${run.chart_count} charts` : ''}
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {run.has_report && (
                    <button
                      onClick={async (e) => {
                        e.stopPropagation();
                        try {
                          const data = await apiCall(`/api/admin/benchmark/report/${run.id}`) as { report?: string };
                          if (data.report) {
                            setReportMarkdown(data.report);
                            setShowReport(true);
                          }
                        } catch {
                          setMessage({ type: 'error', text: 'Could not load report' });
                        }
                      }}
                      className="text-xs text-blue-400 hover:text-blue-300"
                    >
                      Report
                    </button>
                  )}
                  {run.overall_winner && (
                    <div className="text-right">
                      <div className="text-xs text-green-400">{run.overall_winner.variant}</div>
                      <div className="text-xs text-gray-500">{run.overall_winner.composite.toFixed(3)}</div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
