"use client";

import { useMemo, useRef } from "react";

interface FlaggedField {
  page_number: number;
  borehole_id: string;
  field_type: string;
  field_name: string;
  extracted_value: unknown;
  confidence: number;
  context?: string;
}

interface FieldLogPreviewProps {
  field: FlaggedField;
  pageImage?: string; // Base64 encoded
  currentValue?: Record<string, unknown>;
  onCorrection: (value: unknown) => void;
}

export function FieldLogPreview({
  field,
  pageImage,
  currentValue,
  onCorrection,
}: FieldLogPreviewProps) {
  const editorRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null);

  const sourceValue = useMemo(
    () => currentValue?.corrected_value ?? field.extracted_value,
    [currentValue?.corrected_value, field.extracted_value]
  );
  const initialEditValue = useMemo(() => serializeEditorValue(sourceValue), [sourceValue]);
  const editorKey = useMemo(
    () => `${field.page_number}:${field.field_name}:${initialEditValue}`,
    [field.page_number, field.field_name, initialEditValue]
  );

  const handleSave = () => {
    const valueToSave = editorRef.current?.value ?? initialEditValue;

    try {
      // Try to parse as JSON if it looks like JSON
      if (valueToSave.startsWith("{") || valueToSave.startsWith("[")) {
        onCorrection(JSON.parse(valueToSave));
      } else {
        onCorrection(valueToSave);
      }
    } catch {
      // If JSON parse fails, use as string
      onCorrection(valueToSave);
    }
  };

  const confidenceColor = field.confidence >= 0.6
    ? "text-amber-400"
    : "text-red-400";

  const confidenceBg = field.confidence >= 0.6
    ? "bg-amber-500/20 border-amber-500/30"
    : "bg-red-500/20 border-red-500/30";

  return (
    <div className="p-6">
      {/* Field info header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
              {field.field_type}
            </span>
            <span className="text-gray-600">|</span>
            <span className="text-sm text-gray-400">{field.borehole_id}</span>
          </div>
          <h3 className="text-lg font-semibold text-gray-100">
            {formatFieldName(field.field_name)}
          </h3>
        </div>

        {/* Confidence badge */}
        <span className={`px-2 py-1 ${confidenceBg} border rounded-full text-xs ${confidenceColor}`}>
          {Math.round(field.confidence * 100)}% confidence
        </span>
      </div>

      {/* Two-column layout: Image + Editor */}
      <div className="grid grid-cols-2 gap-6">
        {/* Left: Page image */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-400">Source Image</h4>
          <div className="bg-[#171717] border border-[#2a2a2a] rounded-lg overflow-hidden">
            {pageImage ? (
              <img
                src={`data:image/png;base64,${pageImage}`}
                alt={`Page ${field.page_number}`}
                className="w-full h-auto"
              />
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-600">
                <p className="text-sm">Image not available</p>
              </div>
            )}
          </div>
          <p className="text-xs text-gray-500">Page {field.page_number}</p>
        </div>

        {/* Right: Editor */}
        <div className="space-y-4">
          {/* Extracted value */}
          <div>
          <h4 className="text-sm font-medium text-gray-400 mb-2">Extracted Value</h4>
          <div className="p-3 bg-[#171717] border border-[#2a2a2a] rounded-lg">
            <pre className="text-sm text-gray-300 whitespace-pre-wrap break-words font-mono">
              {serializeEditorValue(field.extracted_value)}
            </pre>
          </div>
        </div>

          {/* Edit field */}
          <div>
            <h4 className="text-sm font-medium text-gray-400 mb-2">Corrected Value</h4>
            {typeof field.extracted_value === "object" ? (
              <textarea
                key={editorKey}
                ref={(element) => {
                  editorRef.current = element;
                }}
                defaultValue={initialEditValue}
                rows={6}
                className="w-full px-3 py-2 bg-[#171717] border border-[#3a3a3a] rounded-lg text-sm text-gray-200 font-mono resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter corrected value (JSON)..."
              />
            ) : (
              <input
                key={editorKey}
                ref={(element) => {
                  editorRef.current = element;
                }}
                type="text"
                defaultValue={initialEditValue}
                className="w-full px-3 py-2 bg-[#171717] border border-[#3a3a3a] rounded-xl text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter corrected value..."
              />
            )}
          </div>

          {/* Context if available */}
          {field.context && (
            <div>
              <h4 className="text-sm font-medium text-gray-400 mb-2">Context</h4>
              <p className="text-sm text-gray-500 italic">{field.context}</p>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-3 pt-2">
            <button
              onClick={handleSave}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm font-medium transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              Save Correction
            </button>

            <button
              onClick={() => onCorrection(field.extracted_value)}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors text-sm"
            >
              Accept Original
            </button>
          </div>

          {/* Status indicator */}
          {currentValue && (
            <div className="flex items-center gap-2 text-green-400 text-sm">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>Correction saved</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function serializeEditorValue(value: unknown): string {
  if (typeof value === "object" && value !== null) {
    return JSON.stringify(value, null, 2);
  }
  return String(value ?? "");
}

function formatFieldName(name: string): string {
  // Convert snake_case to Title Case
  return name
    .split("_")
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
