'use client';

import React, { useState } from 'react';
import { PHASE_LABELS } from './constants';
import type { PhaseData } from './constants';

export function PhaseResultCard({ phase, data }: { phase: string; data: PhaseData }) {
  const [expanded, setExpanded] = useState(false);

  if (!data.ranking || data.ranking.length === 0) return null;

  return (
    <div className="bg-[#2a2a2a] rounded-xl overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-5 py-3 flex items-center justify-between hover:bg-[#333] transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-white">{PHASE_LABELS[phase] || phase}</span>
          <span className="text-xs text-gray-400">
            {data.n_variants} variants, {data.duration_s.toFixed(0)}s
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-green-400 font-medium">{data.winner}</span>
          <svg
            className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {expanded && (
        <div className="px-5 pb-4">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 text-xs border-b border-[#3a3a3a]">
                <th className="text-left py-2 pr-4">Rank</th>
                <th className="text-left py-2 pr-4">Variant</th>
                <th className="text-right py-2 pr-4">Composite</th>
                {phase === 'ablation' && <th className="text-right py-2">Delta</th>}
              </tr>
            </thead>
            <tbody>
              {data.ranking.slice(0, 10).map((r) => (
                <tr
                  key={r.variant}
                  className={`border-b border-[#333] ${r.rank === 1 ? 'text-green-400' : r.rank <= 3 ? 'text-yellow-400' : 'text-gray-300'}`}
                >
                  <td className="py-1.5 pr-4 text-xs">#{r.rank}</td>
                  <td className="py-1.5 pr-4">{r.variant}</td>
                  <td className="py-1.5 pr-4 text-right font-mono">{r.composite.toFixed(3)}</td>
                  {phase === 'ablation' && (
                    <td className={`py-1.5 text-right font-mono ${
                      (r.delta || 0) > 0 ? 'text-green-400' : (r.delta || 0) < 0 ? 'text-red-400' : 'text-gray-500'
                    }`}>
                      {(r.delta || 0) > 0 ? '+' : ''}{(r.delta || 0).toFixed(3)}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
