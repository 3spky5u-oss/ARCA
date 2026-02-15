// =============================================================================
// TYPES — Extracted from page.tsx
// =============================================================================

export interface Citation {
  source: string;
  title?: string;
  page?: number;
  section?: string;
  topic?: string;
  score?: number;
}

export interface PhiiMetadata {
  corrections_applied?: number;
  expertise_level?: string;
  personalized?: boolean;
  command?: boolean;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  analysisResult?: AnalysisResult;
  citations?: Citation[];
  confidence?: number;
  toolsUsed?: string[];
  attachedFiles?: { name: string; type: string }[];
  thinkMode?: boolean;  // Extended reasoning mode (qwen3 + prompts)
  autoThink?: boolean;  // Think mode was auto-triggered by keywords
  calculateMode?: boolean;  // Math mode (qwq for calculations)
  autoCalculate?: boolean;  // Calculate mode was auto-triggered
  error?: import('@/components/error-display').ErrorData;  // Structured error for distinct display
  phii?: PhiiMetadata;  // Phii learning metadata
  thinkingContent?: string;  // LLM thinking tokens for expandable display
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export interface AnalysisResult {
  summary?: {
    total_samples: number;
    total_parameters?: number;
    total_comparisons: number;
    exceedance_count: number;
    compliant_count: number;
    non_detect_count?: number;
  };
  exceedances?: Exceedance[];
  excel_report?: string;
  word_report?: string;
  redacted_file?: string;
  // Groundidd template/generation results
  template_file?: string;
  openground_zip?: string;
  soil_analysis?: string;
  generated_files?: Array<{
    name: string;
    type: string;
    description: string;
    url: string;
  }>;
  // LoggView borehole extraction
  boreholes?: Array<{
    borehole_id: string;
    total_depth: number;
    soil_layers?: Array<unknown>;
    spt_records?: Array<unknown>;
    water_level?: number;
    warnings?: string[];
  }>;
  borehole_summary?: string;
  borehole_count?: number;
  filename?: string;
  cross_section_available?: boolean;
  // LoggView visualization outputs
  cross_section_file?: string;
  visualization_3d?: {
    viz_id: string;
    preview_url?: string;
    data_url: string;
    fullpage_url: string;
  };
  // Mapperr geology results (flat structure when type === 'geology')
  type?: string;  // "geology" for Mapperr results
  location?: string;
  unit_code?: string;
  unit_name?: string;
  map_segment?: string;  // URL to map snippet image
  citation?: string;
  soil_types?: string[];
  // AGS legend columns
  stratigraphic_unit?: string;
  lithology?: string;
  lithogenesis?: string;
  morphology?: string;
  comment?: string;
  confidence?: number;
}

export interface Exceedance {
  sample_id: string;
  parameter: string;
  value: number;
  value_str: string;
  unit: string;
  guideline: number;
  exceedance_factor: number;
}

export interface UploadedFile {
  id: string;
  name: string;
  samples: number;
  parameters: number;
  status: 'uploading' | 'parsing' | 'ready';
  rag_chunks?: number;
}

// File manifest for recovery after disconnect/refresh
export interface FileManifest {
  file_id: string;
  filename: string;
  size: number;
  type: 'excel' | 'pdf' | 'word';
  uploadedAt: string;
}

export interface StoredChat {
  id: string;
  title: string;
  messages: {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
    analysisResult?: AnalysisResult;
    citations?: Citation[];
    confidence?: number;
    toolsUsed?: string[];
    attachedFiles?: { name: string; type: string }[];
  }[];
  createdAt: string;
  updatedAt: string;
}
