'use client';

import React from 'react';

/**
 * Standardized error object from backend
 */
export interface ErrorData {
  code: string;
  message: string;
  details?: string;
  tool?: string;
  recoverable?: boolean;
  context?: Record<string, unknown>;
}

/**
 * Props for ErrorDisplay component
 */
interface ErrorDisplayProps {
  error: ErrorData;
  onRetry?: () => void;
  className?: string;
}

/**
 * Map error codes to user-friendly titles
 */
const ERROR_TITLES: Record<string, string> = {
  // Parse errors
  PARSE_EXCEL_FAILED: 'File Processing Error',
  PARSE_PDF_FAILED: 'PDF Processing Error',
  PARSE_WORD_FAILED: 'Document Processing Error',
  PARSE_FORMAT_UNKNOWN: 'Unknown File Format',
  PARSE_DATA_INVALID: 'Invalid Data',

  // Validation errors
  VALIDATION_MISSING_PARAM: 'Missing Information',
  VALIDATION_INVALID_TYPE: 'Invalid Input',
  VALIDATION_OUT_OF_RANGE: 'Value Out of Range',
  VALIDATION_INVALID_FORMAT: 'Invalid Format',

  // Not found errors
  NOT_FOUND_FILE: 'File Not Found',
  NOT_FOUND_GUIDELINE: 'Guideline Not Found',
  NOT_FOUND_CALCULATION: 'Calculation Not Found',
  NOT_FOUND_TOPIC: 'Topic Not Found',
  NOT_FOUND_TEMPLATE: 'Template Not Found',

  // LLM errors
  LLM_UNAVAILABLE: 'AI Service Unavailable',
  LLM_TIMEOUT: 'AI Response Timeout',
  LLM_PARSE_FAILED: 'AI Response Error',
  LLM_RESPONSE_INVALID: 'Invalid AI Response',

  // External service errors
  EXTERNAL_SEARXNG_FAILED: 'Search Service Error',
  EXTERNAL_LLM_FAILED: 'AI Service Error',
  EXTERNAL_NETWORK_ERROR: 'Network Error',

  // Dependency errors
  DEPENDENCY_MISSING: 'Missing Component',
  DEPENDENCY_INIT_FAILED: 'Initialization Error',

  // Internal errors
  INTERNAL_UNEXPECTED: 'Unexpected Error',
  INTERNAL_CONFIG_ERROR: 'Configuration Error',
};

/**
 * Get user-friendly title for an error code
 */
export function getErrorTitle(code: string): string {
  return ERROR_TITLES[code] || 'Error';
}

/**
 * ErrorDisplay component for showing structured errors in the UI.
 *
 * Displays errors with:
 * - Distinct red styling to differentiate from normal messages
 * - Error title based on error code
 * - Main error message
 * - Optional details
 * - Retry button for recoverable errors
 *
 * @example
 * <ErrorDisplay
 *   error={{
 *     code: "NOT_FOUND_FILE",
 *     message: "No files uploaded",
 *     details: "Upload lab data files to analyze",
 *     recoverable: true
 *   }}
 *   onRetry={() => refetchFiles()}
 * />
 */
export function ErrorDisplay({ error, onRetry, className = '' }: ErrorDisplayProps) {
  const title = getErrorTitle(error.code);

  return (
    <div className={`bg-red-900/20 border border-red-500/30 rounded-lg p-4 ${className}`}>
      <div className="flex items-start gap-3">
        {/* Error icon */}
        <svg
          className="h-5 w-5 text-red-400 mt-0.5 flex-shrink-0"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>

        <div className="flex-1 min-w-0">
          {/* Title */}
          <p className="font-medium text-red-300">{title}</p>

          {/* Message */}
          <p className="text-sm text-red-200/80 mt-1">{error.message}</p>

          {/* Details */}
          {error.details && (
            <p className="text-sm text-red-200/60 mt-1">{error.details}</p>
          )}

          {/* Tool indicator */}
          {error.tool && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 mt-2 bg-red-500/20 rounded-full text-xs text-red-300">
              {error.tool}
            </span>
          )}

          {/* Retry button */}
          {error.recoverable && onRetry && (
            <button
              onClick={onRetry}
              className="mt-3 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
            >
              Try Again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Inline error display for smaller contexts (e.g., within file chips)
 */
interface InlineErrorProps {
  message: string;
  className?: string;
}

export function InlineError({ message, className = '' }: InlineErrorProps) {
  return (
    <span className={`inline-flex items-center gap-1 text-red-400 text-xs ${className}`}>
      <svg
        className="h-3 w-3"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
      {message}
    </span>
  );
}

export default ErrorDisplay;
