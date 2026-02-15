'use client';

import React from 'react';
import { TOOL_DISPLAY_NAMES } from '@/app/lib/chat-utils';

export interface ToolIndicatorProps {
  tools: Set<string>;
  removingTools: Set<string>;
  streamingThinking?: string;
  showStreamingThinking?: boolean;
  onToggleThinking?: () => void;
}

export const ToolIndicator = ({
  tools,
  removingTools,
  streamingThinking = '',
  showStreamingThinking = false,
  onToggleThinking
}: ToolIndicatorProps) => {
  // Show both active tools and tools that are animating out
  const allTools = new Set([...tools, ...removingTools]);
  if (allTools.size === 0) return null;

  const toolNames = Array.from(allTools);
  const hasThinking = tools.has('thinking');
  const hasStreamingContent = streamingThinking.length > 0;

  return (
    <div className="mb-3">
      <div className="flex flex-wrap gap-2">
        {toolNames.map(tool => {
          const isRemoving = removingTools.has(tool);
          const isThinking = tool === 'thinking';
          const isClickable = isThinking && hasStreamingContent && onToggleThinking;

          return (
            <div
              key={tool}
              onClick={isClickable ? onToggleThinking : undefined}
              className={`tool-indicator flex items-center gap-2 px-3 py-1.5 bg-blue-600/20 border border-blue-600/30 rounded-full text-sm text-blue-400 ${
                isRemoving ? 'animate-fadeOut' : 'animate-fadeIn'
              } ${isClickable ? 'cursor-pointer hover:bg-blue-600/30' : ''}`}
            >
              {isClickable && (
                <svg
                  className={`w-3 h-3 transition-transform ${showStreamingThinking ? 'rotate-90' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              )}
              <span>{TOOL_DISPLAY_NAMES[tool] || `Using ${tool}`}</span>
              <span className="thinking-ellipsis">
                <span className="dot">.</span>
                <span className="dot">.</span>
                <span className="dot">.</span>
              </span>
              {isClickable && (
                <span className="text-blue-500/60 text-xs">({streamingThinking.length.toLocaleString()})</span>
              )}
            </div>
          );
        })}
      </div>
      {/* Live streaming thinking content */}
      {hasThinking && showStreamingThinking && hasStreamingContent && (
        <div className="mt-2 p-3 bg-gray-800/50 border-l-2 border-blue-600/30 rounded text-xs text-gray-400 font-mono max-h-[300px] overflow-y-auto whitespace-pre-wrap animate-fadeIn">
          {streamingThinking}
        </div>
      )}
    </div>
  );
};
