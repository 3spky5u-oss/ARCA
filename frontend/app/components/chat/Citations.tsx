'use client';

import React, { useState } from 'react';
import type { Citation } from '@/app/lib/chat-types';

export interface CitationsProps {
  citations: Citation[];
  confidence?: number;
}

export const Citations = ({ citations, confidence }: CitationsProps) => {
  const [isOpen, setIsOpen] = useState(false);

  if (!citations || citations.length === 0) return null;

  const getConfidenceClass = (score?: number) => {
    if (!score) return 'confidence-medium';
    if (score >= 0.8) return 'confidence-high';
    if (score >= 0.5) return 'confidence-medium';
    return 'confidence-low';
  };

  const getConfidenceLabel = (score?: number) => {
    if (!score) return 'Based on knowledge base';
    if (score >= 0.8) return 'High confidence';
    if (score >= 0.5) return 'Medium confidence';
    return 'Low confidence';
  };

  // Relevance color helpers
  const getRelevanceColor = (score?: number) => {
    if (!score) return 'text-gray-400';
    const pct = score * 100;
    if (pct >= 90) return 'text-green-400';
    if (pct >= 70) return 'text-yellow-400';
    if (pct >= 50) return 'text-orange-400';
    return 'text-red-400';
  };

  const getRelevanceBg = (score?: number) => {
    if (!score) return 'bg-gray-500/20';
    const pct = score * 100;
    if (pct >= 90) return 'bg-green-500/20';
    if (pct >= 70) return 'bg-yellow-500/20';
    if (pct >= 50) return 'bg-orange-500/20';
    return 'bg-red-500/20';
  };

  return (
    <div className="mt-3 pt-3 border-t border-gray-600/30">
      {/* Confidence Badge */}
      {confidence !== undefined && (
        <div className="mb-2">
          <span className={`confidence-badge ${getConfidenceClass(confidence)}`}>
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {getConfidenceLabel(confidence)}
          </span>
        </div>
      )}

      {/* Citations Toggle */}
      <button
        className="citations-toggle"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-label={`${isOpen ? 'Hide' : 'Show'} sources`}
      >
        <svg
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-90' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        <span>Sources ({citations.length})</span>
      </button>

      {/* Citations List */}
      {isOpen && (
        <div className="ml-4 mt-2 text-sm space-y-2">
          {citations.map((citation, i) => (
            <div key={i} className="py-1">
              {/* Title with page - clickable if URL */}
              <div className="text-gray-200">
                [{i + 1}]{' '}
                {citation.source?.startsWith('http') ? (
                  <a
                    href={citation.source}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-400 hover:text-blue-300 hover:underline"
                  >
                    {citation.title || citation.source}
                  </a>
                ) : (
                  <span>{citation.title || citation.source}</span>
                )}
                {citation.page && <span className="text-gray-500"> (p. {citation.page})</span>}
              </div>
              {/* Topic + Relevance badges */}
              <div className="flex items-center gap-2 mt-1">
                {citation.topic && (
                  <span className="px-2 py-0.5 bg-blue-500/20 text-blue-300 text-xs rounded-full">
                    {citation.topic}
                  </span>
                )}
                {citation.score !== undefined && (
                  <span className={`px-2 py-0.5 ${getRelevanceBg(citation.score)} ${getRelevanceColor(citation.score)} text-xs rounded-full font-medium`}>
                    {(citation.score * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
