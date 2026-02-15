'use client';

import React from 'react';

interface TesterTabProps {
  testFile: File | null;
  onTestFileChange: (file: File | null) => void;
  testExtractor: string;
  onTestExtractorChange: (extractor: string) => void;
  extractionResult: Record<string, unknown> | null;
  onTestExtraction: () => void;
}

export function TesterTab({
  testFile,
  onTestFileChange,
  testExtractor,
  onTestExtractorChange,
  extractionResult,
  onTestExtraction,
}: TesterTabProps) {
  return (
    <div className="space-y-6">
      <h2 className="text-lg font-medium">Extraction Tester</h2>

      <div className="bg-[#2a2a2a] rounded-lg p-4">
        <div className="flex gap-4 items-end mb-4">
          <div className="flex-1">
            <label className="block text-sm text-gray-400 mb-2">PDF File</label>
            <input
              type="file"
              accept=".pdf"
              onChange={(e) => onTestFileChange(e.target.files?.[0] || null)}
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm file:mr-4 file:py-1 file:px-3 file:rounded-xl file:border-0 file:bg-blue-600 file:text-white h-[42px]"
            />
          </div>
          <div className="w-48">
            <label className="block text-sm text-gray-400 mb-2">Extractor (optional)</label>
            <select
              value={testExtractor}
              onChange={(e) => onTestExtractorChange(e.target.value)}
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm h-[42px]"
            >
              <option value="">Auto (pipeline)</option>
              <option value="pymupdf_text">PyMuPDF Text</option>
              <option value="pymupdf4llm">PyMuPDF4LLM</option>
              <option value="marker">Marker</option>
              <option value="observationn">Observationn</option>
              <option value="vision_ocr">Vision OCR</option>
            </select>
          </div>
          <button
            onClick={onTestExtraction}
            disabled={!testFile}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl text-sm transition-colors h-[42px]"
          >
            Test Extraction
          </button>
        </div>

        {extractionResult && (
          <div className="mt-4 space-y-4">
            {/* Scout Report */}
            {extractionResult.scout ? (
              <div className="bg-[#1a1a1a] rounded-lg p-3">
                <h4 className="text-sm font-medium mb-2">Scout Report</h4>
                <div className="text-xs text-gray-400 space-y-1">
                  <p>Quality Tier: <span className="text-white">{String((extractionResult.scout as Record<string, unknown>).quality_tier)}</span></p>
                  <p>Recommended: <span className="text-white">{String((extractionResult.scout as Record<string, unknown>).recommended_extractor)}</span></p>
                  <p>Text Density: <span className="text-white">{String((extractionResult.scout as Record<string, unknown>).text_density)} chars/page</span></p>
                </div>
              </div>
            ) : null}

            {/* Extraction Result */}
            {extractionResult.extraction ? (
              <div className="bg-[#1a1a1a] rounded-lg p-3">
                <h4 className="text-sm font-medium mb-2">Extraction Result</h4>
                <div className="text-xs text-gray-400 space-y-1">
                  <p>Extractor: <span className="text-white">{String((extractionResult.extraction as Record<string, unknown>).extractor_used)}</span></p>
                  <p>Length: <span className="text-white">{Number((extractionResult.extraction as Record<string, unknown>).text_length).toLocaleString()} chars</span></p>
                  <p>Time: <span className="text-white">{String((extractionResult.extraction as Record<string, unknown>).processing_ms)}ms</span></p>
                </div>
                {(extractionResult.extraction as Record<string, unknown>).text_preview ? (
                  <div className="mt-3">
                    <h5 className="text-xs text-gray-500 mb-1">Preview:</h5>
                    <pre className="text-xs bg-[#222] p-2 rounded-lg overflow-auto max-h-48 whitespace-pre-wrap">
                      {String((extractionResult.extraction as Record<string, unknown>).text_preview)}
                    </pre>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
