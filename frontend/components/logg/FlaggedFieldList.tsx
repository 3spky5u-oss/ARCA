"use client";

import React from 'react';

interface FlaggedField {
  page_number: number;
  borehole_id: string;
  field_type: string;
  field_name: string;
  extracted_value: unknown;
  confidence: number;
  context?: string;
}

interface FlaggedFieldListProps {
  flaggedByType: Record<string, FlaggedField[]>;
  corrections: Record<string, unknown>;
  selectedField: FlaggedField | null;
  onSelectField: (field: FlaggedField) => void;
}

export function FlaggedFieldList({
  flaggedByType,
  corrections,
  selectedField,
  onSelectField,
}: FlaggedFieldListProps) {
  const typeLabels: Record<string, string> = {
    header: "Header Fields",
    sample: "Sample Data",
    layer: "Layers",
    footer: "Footer Fields",
  };

  const typeIcons: Record<string, React.ReactElement> = {
    header: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    sample: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
      </svg>
    ),
    layer: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
      </svg>
    ),
    footer: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
  };

  const isCorrected = (field: FlaggedField): boolean => {
    const key = `${field.page_number}-${field.field_type}-${field.field_name}`;
    return key in corrections;
  };

  const isSelected = (field: FlaggedField): boolean => {
    if (!selectedField) return false;
    return (
      selectedField.page_number === field.page_number &&
      selectedField.field_type === field.field_type &&
      selectedField.field_name === field.field_name
    );
  };

  const typeOrder = ["header", "sample", "layer", "footer"];
  const sortedTypes = Object.keys(flaggedByType).sort(
    (a, b) => typeOrder.indexOf(a) - typeOrder.indexOf(b)
  );

  return (
    <div className="divide-y divide-[#2a2a2a]">
      {sortedTypes.map(type => (
        <div key={type} className="py-3 px-4">
          {/* Type header */}
          <div className="flex items-center gap-2 mb-2 text-gray-400">
            {typeIcons[type] || typeIcons.header}
            <span className="text-sm font-medium">
              {typeLabels[type] || type}
            </span>
            <span className="ml-auto text-xs bg-[#2a2a2a] px-2 py-0.5 rounded-full">
              {flaggedByType[type].length}
            </span>
          </div>

          {/* Fields in this type */}
          <div className="space-y-1">
            {flaggedByType[type].map((field, idx) => {
              const corrected = isCorrected(field);
              const selected = isSelected(field);

              return (
                <button
                  key={`${field.page_number}-${field.field_name}-${idx}`}
                  onClick={() => onSelectField(field)}
                  className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
                    selected
                      ? "bg-blue-600/20 border border-blue-600/30"
                      : "bg-[#171717] hover:bg-[#2a2a2a]"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-200 truncate">
                          {formatFieldName(field.field_name)}
                        </span>
                        {corrected && (
                          <svg className="w-3.5 h-3.5 text-green-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                        )}
                      </div>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-xs text-gray-500">
                          {field.borehole_id}
                        </span>
                        <span className="text-xs text-gray-600">•</span>
                        <span className="text-xs text-gray-500">
                          Page {field.page_number}
                        </span>
                      </div>
                    </div>

                    {/* Confidence indicator */}
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium ${
                      field.confidence >= 0.6
                        ? "bg-amber-500/20 text-amber-400"
                        : "bg-red-500/20 text-red-400"
                    }`}>
                      {Math.round(field.confidence * 100)}
                    </div>
                  </div>

                  {/* Preview of extracted value */}
                  <div className="mt-1.5 text-xs text-gray-500 truncate">
                    {formatPreview(field.extracted_value)}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      ))}

      {Object.keys(flaggedByType).length === 0 && (
        <div className="p-8 text-center text-gray-500">
          <svg className="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-sm">No fields flagged for review</p>
          <p className="text-xs mt-1">All extractions passed confidence threshold</p>
        </div>
      )}
    </div>
  );
}

function formatFieldName(name: string): string {
  // Convert snake_case to Title Case
  return name
    .split("_")
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function formatPreview(value: unknown): string {
  if (value === null || value === undefined) {
    return "(empty)";
  }
  if (typeof value === "object") {
    // For objects, show key-value summary
    const obj = value as Record<string, unknown>;
    const entries = Object.entries(obj);
    if (entries.length === 0) return "{}";
    if (entries.length <= 2) {
      return entries.map(([k, v]) => `${k}: ${v}`).join(", ");
    }
    return `${entries.slice(0, 2).map(([k, v]) => `${k}: ${v}`).join(", ")}...`;
  }
  const str = String(value);
  return str.length > 50 ? str.slice(0, 50) + "..." : str;
}
