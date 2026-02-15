'use client';

import React from 'react';
import { formatModelName } from './shared';

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

interface ModelsTabProps {
  modelsData: ModelsData;
  pullModelName: string;
  onPullModelNameChange: (name: string) => void;
  pullStatus: PullStatus | null;
  pullingModel: string | null;
  onRefresh: () => void;
  onPullModel: () => void;
  onAssignModel: (slot: string, model: string) => void;
  onDeleteModel: (name: string) => void;
}

export function ModelsTab({
  modelsData,
  pullModelName,
  onPullModelNameChange,
  pullStatus,
  pullingModel,
  onRefresh,
  onPullModel,
  onAssignModel,
  onDeleteModel,
}: ModelsTabProps) {
  const installedModelNames = new Set(modelsData.models.map((model) => model.name));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">Model Management</h2>
        <button
          onClick={onRefresh}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Model Assignments */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-4">Model Assignments</h3>
        <p className="text-xs text-gray-500 mb-4">
          Select which model to use for each role. Changes take effect immediately.
          If a configured model is missing from <code className="px-1 py-0.5 bg-[#1a1a1a] rounded">./models</code>, it is shown as missing until reassigned.
        </p>
        <div className="grid grid-cols-2 gap-4">
          {Object.entries(modelsData.config_assignments).map(([slot, currentModel]) => (
            <div key={slot} className="flex items-center gap-2">
              <label className="text-sm text-gray-400 w-32 capitalize">{slot.replace('_', ' ')}:</label>
              <select
                value={currentModel}
                onChange={(e) => onAssignModel(slot, e.target.value)}
                className="flex-1 px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
              >
                {!installedModelNames.has(currentModel) && currentModel && (
                  <option value={currentModel}>
                    [Missing] {currentModel}
                  </option>
                )}
                {modelsData.models.length === 0 && (
                  <option value="" disabled>
                    No models installed
                  </option>
                )}
                {modelsData.models.map((m) => {
                  const f = formatModelName(m.name);
                  return <option key={m.name} value={m.name}>{f.display}{f.quant ? ` ${f.quant}` : ''}</option>;
                })}
              </select>
            </div>
          ))}
        </div>
      </div>

      {/* Add Models */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-2">Add Models</h3>
        <p className="text-sm text-gray-400">
          Download GGUF files from{' '}
          <a href="https://huggingface.co/models?sort=trending&search=gguf" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
            HuggingFace
          </a>
          {' '}and place them in the <code className="px-1.5 py-0.5 bg-[#1a1a1a] rounded text-xs">./models/</code> directory, then refresh.
        </p>
        {modelsData.models_dir?.error && (
          <p className="mt-3 text-xs text-amber-400">
            Models directory issue: <code className="px-1 py-0.5 bg-[#1a1a1a] rounded">{modelsData.models_dir.path}</code>{' '}
            ({modelsData.models_dir.error})
          </p>
        )}
      </div>

      {/* Installed Models */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-gray-300">Installed Models</h3>
          <span className="text-xs text-gray-500">Total: {Number(modelsData.total_size_gb).toFixed(1)} GB</span>
        </div>
        <div className="space-y-2">
          {modelsData.models.map((model) => (
            <div
              key={model.name}
              className="flex items-center justify-between p-3 bg-[#1a1a1a] rounded-lg"
            >
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  {(() => {
                    const f = formatModelName(model.name);
                    return (
                      <>
                        <span className="text-sm font-medium">{f.display}</span>
                        {f.quant && <span className="text-xs text-gray-500">{f.quant}</span>}
                      </>
                    );
                  })()}
                  {model.assigned_to.length > 0 && (
                    <span className="text-xs px-2 py-0.5 bg-blue-600/30 text-blue-400 rounded-full">
                      {model.assigned_to.join(', ')}
                    </span>
                  )}
                </div>
                <span className="text-xs text-gray-500">{Number(model.size_gb).toFixed(1)} GB</span>
              </div>
              <button
                onClick={() => onDeleteModel(model.name)}
                disabled={model.assigned_to.length > 0}
                className="px-3 py-1 text-sm text-red-400 hover:bg-red-600/20 disabled:opacity-30 disabled:cursor-not-allowed rounded-lg transition-colors"
                title={model.assigned_to.length > 0 ? 'Cannot delete - model in use' : 'Delete model'}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
