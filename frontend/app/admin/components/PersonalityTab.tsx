'use client';

import React from 'react';

interface PhiiStats {
  feedback: {
    counts: Record<string, number>;
    unresolved_flags: number;
    recent_24h: number;
    total: number;
  };
  settings: {
    phii_energy_matching: boolean;
    phii_specialty_detection: boolean;
    phii_implicit_feedback: boolean;
  };
}

interface PhiiFlag {
  id: number;
  timestamp: string;
  session_id: string;
  message_id: string;
  feedback_type: string;
  user_message: string;
  assistant_response: string;
  tools_used: string[];
  resolved: boolean;
  admin_notes: string;
}

interface Correction {
  id: number;
  timestamp: string;
  wrong_behavior: string;
  right_behavior: string;
  context_keywords: string[];
  confidence: number;
  times_applied: number;
}

interface CorrectionsStats {
  active_count: number;
  total_applied: number;
  recent_7d: number;
}

interface PatternStats {
  pattern_count: number;
  action_count: number;
  top_patterns: Array<{ from: string; to: string; count: number }>;
}

interface PhiiDebugResult {
  corrections_applied: Array<{ wrong_behavior: string; right_behavior: string }>;
  expertise_level: string;
  energy_analysis: { brevity: string; formality: string; urgency: string };
  specialty: string;
}

interface PersonalityTabProps {
  phiiStats: PhiiStats | null;
  phiiFlags: PhiiFlag[];
  corrections: Correction[];
  correctionsStats: CorrectionsStats;
  patternStats: PatternStats;
  phiiDebugMessage: string;
  onPhiiDebugMessageChange: (message: string) => void;
  phiiDebugResult: PhiiDebugResult | null;
  phiiDebugLoading: boolean;
  onPhiiSettingChange: (key: string, value: boolean) => void;
  onResolveFlag: (flagId: number) => void;
  onDeleteCorrection: (correctionId: number) => void;
  onRefreshFlags: () => void;
  onRefreshCorrections: () => void;
  onRefreshPatterns: () => void;
  onPhiiDebug: () => void;
}

export function PersonalityTab({
  phiiStats,
  phiiFlags,
  corrections,
  correctionsStats,
  patternStats,
  phiiDebugMessage,
  onPhiiDebugMessageChange,
  phiiDebugResult,
  phiiDebugLoading,
  onPhiiSettingChange,
  onResolveFlag,
  onDeleteCorrection,
  onRefreshFlags,
  onRefreshCorrections,
  onRefreshPatterns,
  onPhiiDebug,
}: PersonalityTabProps) {
  return (
    <div className="space-y-6">
      <h2 className="text-lg font-medium">Phii Personality Settings</h2>

      {/* Behavior Toggles */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-sm font-medium mb-4">Behavior Toggles</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-3 bg-[#1a1a1a] rounded-lg">
            <div>
              <div className="font-medium text-sm">Energy Matching</div>
              <div className="text-xs text-gray-500">Adapt response style to user&apos;s brevity, formality, and urgency</div>
            </div>
            <button
              onClick={() => onPhiiSettingChange('phii_energy_matching', !(phiiStats?.settings.phii_energy_matching ?? true))}
              className={`w-9 h-9 flex items-center justify-center rounded-xl transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 ${
                (phiiStats?.settings.phii_energy_matching ?? true) ? 'text-amber-400' : 'text-gray-500 hover:text-white'
              }`}
            >
              {(phiiStats?.settings.phii_energy_matching ?? true) ? (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              )}
            </button>
          </div>
          <div className="flex items-center justify-between p-3 bg-[#1a1a1a] rounded-lg">
            <div>
              <div className="font-medium text-sm">Specialty Detection</div>
              <div className="text-xs text-gray-500">Detect domain-specific focus from user messages</div>
            </div>
            <button
              onClick={() => onPhiiSettingChange('phii_specialty_detection', !(phiiStats?.settings.phii_specialty_detection ?? true))}
              className={`w-9 h-9 flex items-center justify-center rounded-xl transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 ${
                (phiiStats?.settings.phii_specialty_detection ?? true) ? 'text-amber-400' : 'text-gray-500 hover:text-white'
              }`}
            >
              {(phiiStats?.settings.phii_specialty_detection ?? true) ? (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              )}
            </button>
          </div>
          <div className="flex items-center justify-between p-3 bg-[#1a1a1a] rounded-lg">
            <div>
              <div className="font-medium text-sm">Implicit Feedback</div>
              <div className="text-xs text-gray-500">Track positive/negative cues from user messages (thanks, wrong, etc.)</div>
            </div>
            <button
              onClick={() => onPhiiSettingChange('phii_implicit_feedback', !(phiiStats?.settings.phii_implicit_feedback ?? true))}
              className={`w-9 h-9 flex items-center justify-center rounded-xl transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 ${
                (phiiStats?.settings.phii_implicit_feedback ?? true) ? 'text-amber-400' : 'text-gray-500 hover:text-white'
              }`}
            >
              {(phiiStats?.settings.phii_implicit_feedback ?? true) ? (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Reinforcement Stats */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-sm font-medium mb-4">Reinforcement Stats</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-[#1a1a1a] rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-amber-400">{phiiStats?.feedback.unresolved_flags ?? 0}</div>
            <div className="text-xs text-gray-500">Unresolved Flags</div>
          </div>
          <div className="bg-[#1a1a1a] rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-green-400">{phiiStats?.feedback.counts?.positive ?? 0}</div>
            <div className="text-xs text-gray-500">Positive Cues</div>
          </div>
          <div className="bg-[#1a1a1a] rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-red-400">{phiiStats?.feedback.counts?.negative ?? 0}</div>
            <div className="text-xs text-gray-500">Negative Cues</div>
          </div>
          <div className="bg-[#1a1a1a] rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-blue-400">{phiiStats?.feedback.recent_24h ?? 0}</div>
            <div className="text-xs text-gray-500">Last 24h</div>
          </div>
        </div>
      </div>

      {/* Flagged Responses */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium">Flagged Responses</h3>
          <button
            onClick={onRefreshFlags}
            className="px-3 py-1 text-xs bg-[#333] hover:bg-[#444] rounded-xl transition-colors"
          >
            Refresh
          </button>
        </div>
        {phiiFlags.length > 0 ? (
          <div className="space-y-3">
            {phiiFlags.map((flag) => (
              <div key={flag.id} className="bg-[#1a1a1a] rounded-lg p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="px-2 py-0.5 text-xs bg-amber-500/20 text-amber-400 rounded-full">
                        #{flag.id}
                      </span>
                      <span className="text-xs text-gray-500">
                        {new Date(flag.timestamp).toLocaleString()}
                      </span>
                      <span className="text-xs text-gray-600">
                        Session: {flag.session_id.slice(0, 8)}...
                      </span>
                    </div>
                    <div className="mb-2">
                      <div className="text-xs text-gray-500 mb-1">User:</div>
                      <div className="text-sm text-gray-300 bg-[#222] p-2 rounded-lg max-h-20 overflow-auto">
                        {flag.user_message || <span className="text-gray-600 italic">No message</span>}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500 mb-1">Assistant:</div>
                      <div className="text-sm text-gray-300 bg-[#222] p-2 rounded-lg max-h-32 overflow-auto">
                        {flag.assistant_response?.slice(0, 500) || <span className="text-gray-600 italic">No response</span>}
                        {(flag.assistant_response?.length ?? 0) > 500 && '...'}
                      </div>
                    </div>
                    {flag.tools_used && flag.tools_used.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {flag.tools_used.map((tool, i) => (
                          <span key={i} className="px-2 py-0.5 text-xs bg-[#333] text-gray-400 rounded-full">
                            {tool}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => onResolveFlag(flag.id)}
                    className="px-3 py-1.5 text-xs bg-green-600 hover:bg-green-700 rounded-xl transition-colors whitespace-nowrap"
                  >
                    Resolve
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <svg className="w-12 h-12 mx-auto mb-3 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm">No flagged responses to review</p>
          </div>
        )}
      </div>

      {/* Learned Corrections */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-medium">Learned Corrections</h3>
            <p className="text-xs text-gray-500 mt-1">
              {correctionsStats.active_count} active · {correctionsStats.total_applied} times applied
            </p>
          </div>
          <button
            onClick={onRefreshCorrections}
            className="px-3 py-1 text-xs bg-[#333] hover:bg-[#444] rounded-xl transition-colors"
          >
            Refresh
          </button>
        </div>
        {corrections.length > 0 ? (
          <div className="space-y-3">
            {corrections.map((c) => (
              <div key={c.id} className="bg-[#1a1a1a] rounded-lg p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="px-2 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded-full">
                        #{c.id}
                      </span>
                      <span className="text-xs text-gray-500">
                        {new Date(c.timestamp).toLocaleString()}
                      </span>
                      <span className="text-xs text-gray-600">
                        Applied {c.times_applied}x
                      </span>
                    </div>
                    <div className="space-y-1.5">
                      <div className="flex items-start gap-2">
                        <span className="text-red-400 text-xs shrink-0">Avoid:</span>
                        <span className="text-sm text-gray-300">{c.wrong_behavior || '—'}</span>
                      </div>
                      <div className="flex items-start gap-2">
                        <span className="text-green-400 text-xs shrink-0">Instead:</span>
                        <span className="text-sm text-gray-300">{c.right_behavior || '—'}</span>
                      </div>
                    </div>
                    {c.context_keywords && c.context_keywords.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {c.context_keywords.map((kw, i) => (
                          <span key={i} className="px-2 py-0.5 text-xs bg-[#333] text-gray-400 rounded-full">
                            {kw}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => onDeleteCorrection(c.id)}
                    className="px-3 py-1.5 text-xs bg-red-600/20 text-red-400 hover:bg-red-600/30 rounded-xl transition-colors whitespace-nowrap"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <svg className="w-12 h-12 mx-auto mb-3 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            <p className="text-sm">No corrections learned yet</p>
            <p className="text-xs text-gray-600 mt-1">Corrections are learned when you correct the assistant&apos;s behavior</p>
          </div>
        )}
      </div>

      {/* Action Patterns */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-medium">Action Patterns</h3>
            <p className="text-xs text-gray-500 mt-1">
              {patternStats.pattern_count} patterns · {patternStats.action_count} actions tracked
            </p>
          </div>
          <button
            onClick={onRefreshPatterns}
            className="px-3 py-1 text-xs bg-[#333] hover:bg-[#444] rounded-xl transition-colors"
          >
            Refresh
          </button>
        </div>
        {patternStats.top_patterns.length > 0 ? (
          <div className="space-y-2">
            {patternStats.top_patterns.map((p, i) => (
              <div key={i} className="flex items-center gap-3 p-3 bg-[#1a1a1a] rounded-lg">
                <div className="flex-1 flex items-center gap-2 text-sm">
                  <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded-full text-xs">
                    {p.from.replace(/_/g, ' ')}
                  </span>
                  <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                  <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-400 rounded-full text-xs">
                    {p.to.replace(/_/g, ' ')}
                  </span>
                </div>
                <span className="text-sm font-medium text-gray-400">{p.count}x</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <svg className="w-12 h-12 mx-auto mb-3 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
            <p className="text-sm">No patterns learned yet</p>
            <p className="text-xs text-gray-600 mt-1">Patterns emerge as you use the assistant&apos;s tools</p>
          </div>
        )}
      </div>

      {/* Phii Debug Panel */}
      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-sm font-medium mb-4">Test Phii Injection</h3>
        <p className="text-xs text-gray-500 mb-4">
          Test what Phii would inject for a given user message (corrections, expertise detection, energy analysis).
        </p>
        <div className="flex gap-4 mb-4">
          <input
            type="text"
            value={phiiDebugMessage}
            onChange={(e) => onPhiiDebugMessageChange(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && onPhiiDebug()}
            placeholder="Enter test message..."
            className="flex-1 px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm"
          />
          <button
            onClick={onPhiiDebug}
            disabled={!phiiDebugMessage.trim() || phiiDebugLoading}
            className="px-4 py-2 bg-amber-600 hover:bg-amber-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors"
          >
            {phiiDebugLoading ? 'Testing...' : 'Test'}
          </button>
        </div>

        {phiiDebugResult && (
          <div className="space-y-3">
            {/* Energy Analysis */}
            {phiiDebugResult.energy_analysis && (
              <div className="bg-[#1a1a1a] rounded-lg p-3">
                <h4 className="text-xs font-medium text-amber-400 mb-2">Energy Analysis</h4>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div>
                    <span className="text-gray-500">Brevity:</span>
                    <span className="ml-1 text-white">{phiiDebugResult.energy_analysis.brevity}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Formality:</span>
                    <span className="ml-1 text-white">{phiiDebugResult.energy_analysis.formality}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Urgency:</span>
                    <span className="ml-1 text-white">{phiiDebugResult.energy_analysis.urgency}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Expertise & Specialty */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-[#1a1a1a] rounded-lg p-3">
                <h4 className="text-xs font-medium text-blue-400 mb-1">Expertise Level</h4>
                <p className="text-sm text-white">{phiiDebugResult.expertise_level || 'Not detected'}</p>
              </div>
              <div className="bg-[#1a1a1a] rounded-lg p-3">
                <h4 className="text-xs font-medium text-purple-400 mb-1">Specialty</h4>
                <p className="text-sm text-white">{phiiDebugResult.specialty || 'Not detected'}</p>
              </div>
            </div>

            {/* Corrections */}
            {phiiDebugResult.corrections_applied && phiiDebugResult.corrections_applied.length > 0 ? (
              <div className="bg-[#1a1a1a] rounded-lg p-3">
                <h4 className="text-xs font-medium text-green-400 mb-2">
                  Corrections Applied ({phiiDebugResult.corrections_applied.length})
                </h4>
                <div className="space-y-2">
                  {phiiDebugResult.corrections_applied.map((c, i) => (
                    <div key={i} className="text-xs border-l-2 border-green-500/30 pl-2">
                      <div className="flex items-start gap-2">
                        <span className="text-red-400 shrink-0">Avoid:</span>
                        <span className="text-gray-300">{c.wrong_behavior}</span>
                      </div>
                      <div className="flex items-start gap-2 mt-1">
                        <span className="text-green-400 shrink-0">Instead:</span>
                        <span className="text-gray-300">{c.right_behavior}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="bg-[#1a1a1a] rounded-lg p-3 text-center text-gray-500 text-xs">
                No corrections matched for this message
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
