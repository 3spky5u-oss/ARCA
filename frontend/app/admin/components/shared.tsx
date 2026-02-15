'use client';

import React, { useState, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';

/**
 * Format raw GGUF filename into clean display name + quantization.
 * "GLM-4.7-Flash-Q4_K_M.gguf" → { display: "GLM-4.7-Flash", quant: "Q4_K_M" }
 * "Qwen3VL-8B-Instruct-Q4_K_M.gguf" → { display: "Qwen3VL-8B-Instruct", quant: "Q4_K_M" }
 */
export function formatModelName(filename: string): { display: string; quant: string } {
  let stem = filename.replace(/\.gguf$/i, '');

  // Extract quantization suffix (Q4_K_M, IQ4_XS, etc.)
  const quantMatch = stem.match(/[-_](I?Q\d+[-_]?\w*)$/i);
  let quant = '';
  if (quantMatch) {
    quant = quantMatch[1].replace(/-/g, '_').toUpperCase();
    stem = stem.slice(0, quantMatch.index);
  }

  // Clean trailing hyphens/underscores
  stem = stem.replace(/[-_]+$/, '');

  return { display: stem, quant };
}

/**
 * Tooltip icon for settings - renders tooltip in a portal with fixed positioning
 * to escape overflow:auto/hidden containers and z-index stacking contexts.
 */
export function SettingTip({ tip }: { tip: string }) {
  const [show, setShow] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const iconRef = useRef<HTMLSpanElement>(null);

  const onEnter = useCallback(() => {
    if (iconRef.current) {
      const rect = iconRef.current.getBoundingClientRect();
      setPos({ top: rect.top - 8, left: rect.left + rect.width / 2 });
    }
    setShow(true);
  }, []);

  return (
    <span ref={iconRef} className="inline-flex ml-1.5" onMouseEnter={onEnter} onMouseLeave={() => setShow(false)}>
      <svg className="w-3.5 h-3.5 text-gray-500 cursor-help" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      {show && typeof document !== 'undefined' && createPortal(
        <span
          className="fixed px-3 py-1.5 text-xs text-gray-200 bg-[#1a1a1a] border border-[#3a3a3a] rounded-lg shadow-xl max-w-xs whitespace-normal w-max pointer-events-none"
          style={{ top: pos.top, left: pos.left, transform: 'translate(-50%, -100%)', zIndex: 99999 }}
        >
          {tip}
        </span>,
        document.body,
      )}
    </span>
  );
}
