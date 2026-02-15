"use client";

import { useCallback, useState } from "react";
import { Upload, FileSpreadsheet, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import type { UploadResponse } from "@/lib/api";

interface UploadZoneProps {
  onUpload: (response: UploadResponse) => void;
  onError: (error: string) => void;
  disabled?: boolean;
}

export function UploadZone({ onUpload, onError, disabled }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<UploadResponse | null>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  }, []);

  const handleDragOut = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const processFile = async (file: File) => {
    // Validate file type
    const validTypes = [
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "application/vnd.ms-excel",
      "text/csv",
    ];
    const validExtensions = [".xlsx", ".xls", ".csv"];
    const hasValidExtension = validExtensions.some((ext) =>
      file.name.toLowerCase().endsWith(ext)
    );

    if (!validTypes.includes(file.type) && !hasValidExtension) {
      onError("Please upload an Excel (.xlsx, .xls) or CSV file");
      return;
    }

    setIsUploading(true);

    try {
      const { uploadFile } = await import("@/lib/api");
      const response = await uploadFile(file);
      setUploadedFile(response);
      onUpload(response);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Upload failed";
      onError(message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      if (disabled || isUploading) return;

      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        processFile(e.dataTransfer.files[0]);
      }
    },
    [disabled, isUploading]
  );

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const clearFile = () => {
    setUploadedFile(null);
  };

  // Show uploaded file state
  if (uploadedFile) {
    return (
      <div className="border-2 border-green-500 bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-green-100 dark:bg-green-800 rounded-lg">
              <FileSpreadsheet className="h-8 w-8 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="font-medium text-green-900 dark:text-green-100">
                {uploadedFile.filename}
              </p>
              <p className="text-sm text-green-700 dark:text-green-300">
                {uploadedFile.samples} samples • {uploadedFile.parameters} parameters
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={clearFile}
            className="text-green-700 hover:text-green-900 hover:bg-green-100"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "border-2 border-dashed rounded-lg p-12 text-center transition-all cursor-pointer",
        isDragging && "upload-zone-active border-primary bg-primary/5",
        isUploading && "opacity-50 cursor-wait",
        !isDragging && !isUploading && "border-muted-foreground/25 hover:border-muted-foreground/50"
      )}
      onDragEnter={handleDragIn}
      onDragLeave={handleDragOut}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => {
        if (!disabled && !isUploading) {
          document.getElementById("file-input")?.click();
        }
      }}
    >
      <input
        id="file-input"
        type="file"
        accept=".xlsx,.xls,.csv"
        onChange={handleFileSelect}
        className="hidden"
        disabled={disabled || isUploading}
      />

      <div className="flex flex-col items-center gap-4">
        {isUploading ? (
          <>
            <div className="p-4 bg-primary/10 rounded-full">
              <Upload className="h-10 w-10 text-primary animate-pulse" />
            </div>
            <div>
              <p className="text-lg font-medium">Uploading...</p>
              <p className="text-sm text-muted-foreground">Parsing lab data</p>
            </div>
          </>
        ) : (
          <>
            <div className="p-4 bg-muted rounded-full">
              <Upload className="h-10 w-10 text-muted-foreground" />
            </div>
            <div>
              <p className="text-lg font-medium">
                {isDragging ? "Drop file here" : "Drop Excel file here"}
              </p>
              <p className="text-sm text-muted-foreground">
                or click to browse • BV ESDAT format (.xlsx, .csv)
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
