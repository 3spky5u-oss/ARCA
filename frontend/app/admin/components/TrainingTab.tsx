'use client';

import React, { useState, useEffect, useCallback } from 'react';

interface TrainingStatus {
  pipeline_available: boolean;
  source_pdfs: number;
  parsed_files: number;
  generated_files: number;
  curated_files: number;
  dataset_ready: boolean;
  active_jobs: Array<{
    id: string;
    type: string;
    status: string;
    phase: string;
    progress: { current: number; total: number; percent: number };
  }>;
  finetuned_model: string;
}

interface TrainingJob {
  id: string;
  type: string;
  status: string;
  phase: string;
  progress: { current: number; total: number; percent: number };
  started_at: string;
  completed_at: string | null;
  error: string | null;
  result?: Record<string, unknown>;
}

interface DatasetStats {
  train_count: number;
  val_count: number;
  total_pairs: number;
  ready: boolean;
}

interface BufferStatus {
  total: number;
  by_status: Record<string, number>;
  by_source: Record<string, number>;
  error?: string;
}

interface GoldenSetStats {
  exists: boolean;
  count: number;
}

interface EvaluationResult {
  has_results: boolean;
  latest?: Record<string, unknown>;
  file?: string;
}

interface TrainingTabProps {
  apiCall: (endpoint: string, options?: RequestInit) => Promise<Record<string, unknown>>;
  setMessage: (msg: { type: 'success' | 'error'; text: string } | null) => void;
}

export function TrainingTab({ apiCall, setMessage }: TrainingTabProps) {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [dataset, setDataset] = useState<DatasetStats | null>(null);
  const [buffer, setBuffer] = useState<BufferStatus | null>(null);
  const [goldenSet, setGoldenSet] = useState<GoldenSetStats | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [approveCount, setApproveCount] = useState(10);

  const fetchAll = useCallback(async () => {
    try {
      const [statusData, jobsData, datasetData, bufferData, goldenData, evalData] = await Promise.all([
        apiCall('/api/admin/training/status'),
        apiCall('/api/admin/training/jobs'),
        apiCall('/api/admin/training/dataset'),
        apiCall('/api/admin/training/buffer'),
        apiCall('/api/admin/training/golden-set'),
        apiCall('/api/admin/training/evaluation'),
      ]);
      setStatus(statusData as unknown as TrainingStatus);
      setJobs((jobsData as { jobs: TrainingJob[] }).jobs || []);
      setDataset(datasetData as unknown as DatasetStats);
      setBuffer(bufferData as unknown as BufferStatus);
      setGoldenSet(goldenData as unknown as GoldenSetStats);
      setEvaluation(evalData as unknown as EvaluationResult);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch training data: ${e}` });
    }
  }, [apiCall, setMessage]);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  // Poll active jobs
  useEffect(() => {
    if (!status?.active_jobs?.length) return;
    const interval = setInterval(fetchAll, 3000);
    return () => clearInterval(interval);
  }, [status?.active_jobs?.length, fetchAll]);

  const startJob = async (stage: string) => {
    setLoading(true);
    try {
      const data = await apiCall(`/api/admin/training/${stage}`, { method: 'POST' });
      setMessage({ type: 'success', text: `${stage} job started: ${(data as { job_id: string }).job_id}` });
      await fetchAll();
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to start ${stage}: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  const approveBuffer = async () => {
    setLoading(true);
    try {
      const data = await apiCall('/api/admin/training/buffer/approve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: approveCount, min_score: 7.5 }),
      });
      const approved = (data as { approved: number }).approved;
      setMessage({ type: 'success', text: `Approved ${approved} pairs` });
      await fetchAll();
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to approve: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  const deployModel = async () => {
    setLoading(true);
    try {
      const data = await apiCall('/api/admin/training/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: 'arca-finetuned' }),
      });
      setMessage({ type: 'success', text: `Deploy started: ${(data as { job_id: string }).job_id}` });
      await fetchAll();
    } catch (e) {
      setMessage({ type: 'error', text: `Deploy failed: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  const rollbackModel = async () => {
    setLoading(true);
    try {
      await apiCall('/api/admin/training/rollback', { method: 'POST' });
      setMessage({ type: 'success', text: 'Rolled back to base model' });
      await fetchAll();
    } catch (e) {
      setMessage({ type: 'error', text: `Rollback failed: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  const startEvaluation = async () => {
    setLoading(true);
    try {
      const data = await apiCall('/api/admin/training/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      setMessage({ type: 'success', text: `Evaluation started: ${(data as { job_id: string }).job_id}` });
      await fetchAll();
    } catch (e) {
      setMessage({ type: 'error', text: `Evaluation failed: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  const hasActiveJobs = (status?.active_jobs?.length ?? 0) > 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">Fine-Tuning Pipeline</h2>
        <button
          onClick={fetchAll}
          className="px-4 py-2 bg-[#333] hover:bg-[#444] rounded-xl text-sm transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Pipeline not available */}
      {status && !status.pipeline_available && (
        <div className="bg-yellow-900/20 border border-yellow-700 rounded-xl p-4 text-yellow-300 text-sm">
          Training pipeline not found. Expected at training/ directory.
        </div>
      )}

      {/* Pipeline Status Overview */}
      {status && status.pipeline_available && (
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
          <div className="bg-[#2a2a2a] rounded-xl p-4">
            <div className="text-2xl font-bold">{status.source_pdfs}</div>
            <div className="text-xs text-gray-400">Source PDFs</div>
          </div>
          <div className="bg-[#2a2a2a] rounded-xl p-4">
            <div className="text-2xl font-bold">{status.parsed_files}</div>
            <div className="text-xs text-gray-400">Parsed Files</div>
          </div>
          <div className="bg-[#2a2a2a] rounded-xl p-4">
            <div className="text-2xl font-bold">{status.generated_files}</div>
            <div className="text-xs text-gray-400">Generated QA</div>
          </div>
          <div className="bg-[#2a2a2a] rounded-xl p-4">
            <div className="text-2xl font-bold">{status.curated_files}</div>
            <div className="text-xs text-gray-400">Curated QA</div>
          </div>
          <div className="bg-[#2a2a2a] rounded-xl p-4">
            <div className={`text-2xl font-bold ${status.dataset_ready ? 'text-green-400' : 'text-gray-500'}`}>
              {status.dataset_ready ? 'Ready' : 'Not Ready'}
            </div>
            <div className="text-xs text-gray-400">Dataset</div>
          </div>
        </div>
      )}

      {/* Active Model */}
      {status?.finetuned_model && (
        <div className="bg-green-900/20 border border-green-700 rounded-xl p-4 flex items-center justify-between">
          <div>
            <div className="text-sm text-green-300">Active Fine-Tuned Model</div>
            <div className="text-white font-mono">{status.finetuned_model}</div>
          </div>
          <button
            onClick={rollbackModel}
            disabled={loading}
            className="px-3 py-1.5 bg-red-600/80 hover:bg-red-600 rounded-xl text-sm transition-colors"
          >
            Rollback
          </button>
        </div>
      )}

      {/* Pipeline Actions */}
      {status?.pipeline_available && (
        <div className="bg-[#2a2a2a] rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Pipeline Stages</h3>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => startJob('parse')}
              disabled={loading || hasActiveJobs}
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors"
            >
              1. Parse PDFs
            </button>
            <button
              onClick={() => startJob('generate')}
              disabled={loading || hasActiveJobs}
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors"
            >
              2. Generate QA
            </button>
            <button
              onClick={() => startJob('filter')}
              disabled={loading || hasActiveJobs}
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors"
            >
              3. Filter/Curate
            </button>
            <button
              onClick={() => startJob('format')}
              disabled={loading || hasActiveJobs}
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors"
            >
              4. Format Dataset
            </button>
            <button
              onClick={startEvaluation}
              disabled={loading || hasActiveJobs}
              className="px-3 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors"
            >
              Evaluate
            </button>
            <button
              onClick={deployModel}
              disabled={loading || hasActiveJobs}
              className="px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors"
            >
              Deploy GGUF
            </button>
          </div>
        </div>
      )}

      {/* Active Jobs */}
      {hasActiveJobs && (
        <div className="bg-[#2a2a2a] rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Running Jobs</h3>
          {status!.active_jobs.map(job => (
            <div key={job.id} className="flex items-center gap-3 py-2">
              <svg className="animate-spin h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <span className="text-sm font-mono text-gray-300">{job.type}</span>
              <span className="text-xs text-gray-500">{job.id}</span>
              {job.phase && <span className="text-xs text-blue-400">{job.phase}</span>}
              {job.progress.percent > 0 && (
                <div className="flex-1 max-w-[200px]">
                  <div className="bg-[#1a1a1a] rounded-full h-2">
                    <div
                      className="bg-blue-500 rounded-full h-2 transition-all"
                      style={{ width: `${job.progress.percent}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Dataset Stats */}
      {dataset && (
        <div className="bg-[#2a2a2a] rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Dataset</h3>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Train:</span>{' '}
              <span className="text-white">{dataset.train_count}</span>
            </div>
            <div>
              <span className="text-gray-400">Val:</span>{' '}
              <span className="text-white">{dataset.val_count}</span>
            </div>
            <div>
              <span className="text-gray-400">Total:</span>{' '}
              <span className="text-white">{dataset.total_pairs}</span>
            </div>
          </div>
        </div>
      )}

      {/* Candidate Buffer */}
      {buffer && !buffer.error && (
        <div className="bg-[#2a2a2a] rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Candidate Buffer</h3>
          <div className="grid grid-cols-4 gap-4 text-sm mb-3">
            <div>
              <span className="text-gray-400">Total:</span>{' '}
              <span className="text-white">{buffer.total}</span>
            </div>
            <div>
              <span className="text-gray-400">Pending:</span>{' '}
              <span className="text-yellow-400">{buffer.by_status?.pending || 0}</span>
            </div>
            <div>
              <span className="text-gray-400">Approved:</span>{' '}
              <span className="text-green-400">{buffer.by_status?.approved || 0}</span>
            </div>
            <div>
              <span className="text-gray-400">Rejected:</span>{' '}
              <span className="text-red-400">{buffer.by_status?.rejected || 0}</span>
            </div>
          </div>
          {(buffer.by_status?.pending || 0) > 0 && (
            <div className="flex items-center gap-2">
              <input
                type="number"
                value={approveCount}
                onChange={e => setApproveCount(parseInt(e.target.value) || 10)}
                className="w-20 px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-lg text-sm"
                min={1}
              />
              <button
                onClick={approveBuffer}
                disabled={loading}
                className="px-3 py-1.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-xl text-sm transition-colors"
              >
                Approve Top {approveCount}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Golden Set */}
      {goldenSet && (
        <div className="bg-[#2a2a2a] rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Golden Validation Set</h3>
          <div className="text-sm">
            {goldenSet.exists ? (
              <span className="text-green-400">{goldenSet.count} pairs (target: 150-200)</span>
            ) : (
              <span className="text-gray-500">Not found</span>
            )}
          </div>
        </div>
      )}

      {/* Evaluation Results */}
      {evaluation?.has_results && evaluation.latest && (
        <div className="bg-[#2a2a2a] rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Latest Evaluation</h3>
          <div className="text-xs font-mono text-gray-400 space-y-1">
            {Object.entries(evaluation.latest).map(([key, value]) => (
              <div key={key} className="flex gap-2">
                <span className="text-gray-500 w-40">{key}:</span>
                <span className="text-gray-300">
                  {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Jobs */}
      {jobs.length > 0 && (
        <div className="bg-[#2a2a2a] rounded-xl p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Job History</h3>
          <div className="space-y-2">
            {jobs.slice(0, 10).map(job => (
              <div key={job.id} className="flex items-center gap-3 text-sm py-1 border-b border-[#333] last:border-0">
                <span className={`w-2 h-2 rounded-full ${
                  job.status === 'completed' ? 'bg-green-500' :
                  job.status === 'failed' ? 'bg-red-500' :
                  job.status === 'running' ? 'bg-blue-500 animate-pulse' :
                  'bg-gray-500'
                }`} />
                <span className="font-mono text-gray-300 w-20">{job.type}</span>
                <span className="text-xs text-gray-500">{job.id}</span>
                <span className="text-xs text-gray-500 flex-1">
                  {job.started_at ? new Date(job.started_at).toLocaleString() : ''}
                </span>
                {job.error && (
                  <span className="text-xs text-red-400 truncate max-w-[200px]">{job.error}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
