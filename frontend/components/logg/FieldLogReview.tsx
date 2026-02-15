"use client";

import { useState, useEffect, useCallback } from "react";
import { FieldLogPreview } from "./FieldLogPreview";
import { FlaggedFieldList } from "./FlaggedFieldList";

interface FlaggedField {
  page_number: number;
  borehole_id: string;
  field_type: string;
  field_name: string;
  extracted_value: unknown;
  confidence: number;
  context?: string;
}

interface ReviewData {
  job_id: string;
  status: string;
  flagged_by_type: Record<string, FlaggedField[]>;
  page_images: Record<number, string>;
  total_flagged: number;
}

interface FieldLogReviewProps {
  jobId: string;
  onComplete?: (result: unknown) => void;
  onCancel?: () => void;
}

export function FieldLogReview({ jobId, onComplete, onCancel }: FieldLogReviewProps) {
  const [reviewData, setReviewData] = useState<ReviewData | null>(null);
  const [corrections, setCorrections] = useState<Record<string, unknown>>({});
  const [selectedField, setSelectedField] = useState<FlaggedField | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch review data
  useEffect(() => {
    const fetchReviewData = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(`/api/logg/job/${jobId}/review`);
        if (!response.ok) {
          throw new Error(`Failed to fetch review data: ${response.statusText}`);
        }
        const data = await response.json();
        setReviewData(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load review data");
      } finally {
        setIsLoading(false);
      }
    };

    fetchReviewData();
  }, [jobId]);

  // Handle field correction
  const handleCorrection = useCallback((field: FlaggedField, value: unknown) => {
    const key = `${field.page_number}-${field.field_type}-${field.field_name}`;
    setCorrections(prev => ({
      ...prev,
      [key]: {
        page_number: field.page_number,
        field_type: field.field_type,
        field_name: field.field_name,
        corrected_value: value,
      },
    }));
  }, []);

  // Submit corrections
  const handleSubmit = useCallback(async () => {
    if (!reviewData) return;

    try {
      setIsSubmitting(true);
      const correctionsList = Object.values(corrections);

      const response = await fetch(`/api/logg/job/${jobId}/correct`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corrections: correctionsList }),
      });

      if (!response.ok) {
        throw new Error(`Failed to submit corrections: ${response.statusText}`);
      }

      // Finalize the job
      const finalizeResponse = await fetch(`/api/logg/job/${jobId}/finalize`, {
        method: "POST",
      });

      if (!finalizeResponse.ok) {
        throw new Error(`Failed to finalize job: ${finalizeResponse.statusText}`);
      }

      const result = await finalizeResponse.json();
      onComplete?.(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit corrections");
    } finally {
      setIsSubmitting(false);
    }
  }, [jobId, corrections, reviewData, onComplete]);

  // Accept all and finalize
  const handleAcceptAll = useCallback(async () => {
    try {
      setIsSubmitting(true);
      const response = await fetch(`/api/logg/job/${jobId}/finalize`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error(`Failed to finalize job: ${response.statusText}`);
      }

      const result = await response.json();
      onComplete?.(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to finalize");
    } finally {
      setIsSubmitting(false);
    }
  }, [jobId, onComplete]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        <div className="flex items-center gap-2">
          <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <span>Loading review data...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-600/20 border border-red-600/30 rounded-lg text-red-400">
        <p className="font-medium">Error loading review</p>
        <p className="text-sm mt-1">{error}</p>
        <button
          onClick={onCancel}
          className="mt-3 px-3 py-1.5 bg-[#2a2a2a] hover:bg-[#3a3a3a] rounded-xl text-sm transition-colors"
        >
          Close
        </button>
      </div>
    );
  }

  if (!reviewData) return null;

  const correctedCount = Object.keys(corrections).length;
  const remainingCount = reviewData.total_flagged - correctedCount;

  return (
    <div className="flex flex-col h-full bg-[#212121]">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-[#2a2a2a]">
        <div>
          <h2 className="text-lg font-semibold text-gray-100">Review Field Log Extraction</h2>
          <p className="text-sm text-gray-400 mt-1">
            {reviewData.total_flagged} fields flagged for review
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Progress indicator */}
          <span className="px-3 py-1 bg-amber-500/20 border border-amber-500/30 rounded-full text-xs text-amber-400">
            {correctedCount} corrected / {remainingCount} remaining
          </span>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel: Flagged field list */}
        <div className="w-80 border-r border-[#2a2a2a] overflow-y-auto">
          <FlaggedFieldList
            flaggedByType={reviewData.flagged_by_type}
            corrections={corrections}
            selectedField={selectedField}
            onSelectField={setSelectedField}
          />
        </div>

        {/* Right panel: Preview and editor */}
        <div className="flex-1 overflow-y-auto">
          {selectedField ? (
            <FieldLogPreview
              field={selectedField}
              pageImage={reviewData.page_images[selectedField.page_number]}
              currentValue={
                corrections[`${selectedField.page_number}-${selectedField.field_type}-${selectedField.field_name}`] as Record<string, unknown> | undefined
              }
              onCorrection={(value) => handleCorrection(selectedField, value)}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              <p>Select a field to review</p>
            </div>
          )}
        </div>
      </div>

      {/* Footer actions */}
      <div className="flex items-center justify-between p-4 border-t border-[#2a2a2a] bg-[#171717]">
        <button
          onClick={onCancel}
          className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
        >
          Cancel
        </button>

        <div className="flex items-center gap-3">
          <button
            onClick={handleAcceptAll}
            disabled={isSubmitting}
            className="px-4 py-2 bg-[#2a2a2a] hover:bg-[#3a3a3a] rounded-xl text-sm font-medium transition-colors disabled:opacity-50"
          >
            Accept All & Finish
          </button>
          <button
            onClick={handleSubmit}
            disabled={isSubmitting || correctedCount === 0}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm font-medium transition-colors disabled:opacity-50"
          >
            {isSubmitting ? (
              <>
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span>Submitting...</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span>Submit Corrections</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
