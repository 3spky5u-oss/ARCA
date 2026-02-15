/**
 * API Client for ARCA Backend
 */

// API base: NEXT_PUBLIC_API_URL (build-time) > dynamic hostname detection
// For reverse proxy / HTTPS: rebuild frontend with NEXT_PUBLIC_API_URL=https://your-domain.com/api
export const getApiBase = () => {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window === 'undefined') return 'http://localhost:8000';
  return `http://${window.location.hostname}:8000`;
};

// WebSocket base derived from same source — http→ws, https→wss
export const getWsBase = () => {
  const base = getApiBase();
  return base.replace(/^http/, 'ws');
};

const API_BASE = getApiBase();

// Default fetch timeout (30 seconds)
const FETCH_TIMEOUT_MS = 30_000;

function fetchWithTimeout(url: string, options?: RequestInit, timeoutMs = FETCH_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(timer));
}

// === Types ===

export interface UploadResponse {
  file_id: string;
  filename: string;
  samples: number;
  parameters: number;
  uploaded_at: string;
}

export interface Exceedance {
  sample_id: string;
  parameter: string;
  value: number;
  value_str: string;
  unit: string;
  guideline: number;
  guideline_unit: string;
  exceedance_factor: number;
  exceedance_percent: number;
  notes: string;
}

export interface AnalysisSummary {
  total_samples: number;
  total_parameters: number;
  total_comparisons: number;
  exceedance_count: number;
  compliant_count: number;
  non_detect_count: number;
  no_guideline_count: number;
  exceedance_rate: number;
}

export interface AnalysisResponse {
  success: boolean;
  file_id: string;
  soil_type: string;
  land_use: string;
  summary: AnalysisSummary;
  exceedances: Exceedance[];
  excel_report: string;
  word_report?: string;
  analyzed_at: string;
  analysis_time_ms: number;
}

export interface SystemStatus {
  status: "healthy" | "degraded";
  model?: string;
  components: {
    llm: string;
    guidelines: string;
  };
  storage: {
    uploads: number;
    reports: number;
  };
}

// === API Functions ===

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetchWithTimeout(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(error.detail || "Upload failed");
  }

  return response.json();
}

export async function uploadFiles(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("files", file);
  });

  const response = await fetchWithTimeout(`${API_BASE}/api/upload/multi`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(error.detail || "Upload failed");
  }

  return response.json();
}

export async function runAnalysis(
  fileId: string,
  soilType: "fine" | "coarse",
  landUse: string
): Promise<AnalysisResponse> {
  const response = await fetchWithTimeout(`${API_BASE}/api/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      file_id: fileId,
      soil_type: soilType,
      land_use: landUse,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Analysis failed" }));
    throw new Error(error.detail || "Analysis failed");
  }

  return response.json();
}

export async function getSystemStatus(): Promise<SystemStatus> {
  const response = await fetchWithTimeout(`${API_BASE}/api/status`);
  if (!response.ok) {
    throw new Error("Failed to get system status");
  }
  return response.json();
}

export function getDownloadUrl(path: string): string {
  if (path.startsWith("/")) {
    return `${API_BASE}${path}`;
  }
  return `${API_BASE}/api/download/${path}`;
}

// === WebSocket Chat ===

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatCallback = (
  token: string,
  done: boolean,
  fullResponse?: string,
  analysisResult?: AnalysisResponse
) => void;

export class ChatClient {
  private ws: WebSocket | null = null;
  private messageQueue: Array<{ message: string; fileId?: string }> = [];
  private currentCallback: ChatCallback | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private accumulatedResponse: string = "";

  connect(onConnect?: () => void, onError?: (error: string) => void) {
    const wsUrl = API_BASE.replace("http", "ws") + "/api/chat";

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log("Chat connected");
      this.reconnectAttempts = 0;
      onConnect?.();

      while (this.messageQueue.length > 0) {
        const msg = this.messageQueue.shift();
        if (msg) this.send(msg.message, msg.fileId);
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "error") {
          this.currentCallback?.("", true, `Error: ${data.content}`);
          this.accumulatedResponse = "";
          return;
        }

        if (this.currentCallback) {
          const token = data.content || "";
          const isDone = data.done === true;

          if (!isDone && token) {
            this.accumulatedResponse += token;
          }

          const analysisResult = data.analysis_result 
            ? this.transformAnalysisResult(data.analysis_result) 
            : undefined;

          this.currentCallback(
            token,
            isDone,
            isDone ? this.accumulatedResponse : undefined,
            analysisResult
          );

          if (isDone) {
            this.accumulatedResponse = "";
          }
        }
      } catch (e) {
        console.error("Failed to parse chat message", e);
      }
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket error", error);
      onError?.("Connection error");
    };

    this.ws.onclose = () => {
      console.log("Chat disconnected");

      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        setTimeout(() => this.connect(onConnect, onError), 1000 * this.reconnectAttempts);
      }
    };
  }

  private transformAnalysisResult(result: any): AnalysisResponse {
    return {
      success: result.success,
      file_id: result.file_id || "",
      soil_type: result.soil_type,
      land_use: result.land_use,
      summary: {
        total_samples: result.summary?.total_samples || 0,
        total_parameters: result.summary?.total_parameters || 0,
        total_comparisons: result.summary?.total_comparisons || 0,
        exceedance_count: result.summary?.exceedance_count || 0,
        compliant_count: result.summary?.compliant_count || 0,
        non_detect_count: result.summary?.non_detect_count || 0,
        no_guideline_count: result.summary?.no_guideline_count || 0,
        exceedance_rate: result.summary?.exceedance_count && result.summary?.total_comparisons
          ? (result.summary.exceedance_count / result.summary.total_comparisons) * 100
          : 0,
      },
      exceedances: (result.exceedances || []).map((exc: any) => ({
        sample_id: exc.sample_id,
        parameter: exc.parameter,
        value: exc.value || 0,
        value_str: exc.value_str || String(exc.value),
        unit: exc.unit || "mg/kg",
        guideline: exc.guideline,
        guideline_unit: exc.unit || "mg/kg",
        exceedance_factor: exc.exceedance_factor,
        exceedance_percent: ((exc.exceedance_factor || 1) - 1) * 100,
        notes: exc.notes || "",
      })),
      excel_report: result.excel_report || "",
      word_report: result.word_report || undefined,
      analyzed_at: new Date().toISOString(),
      analysis_time_ms: result.analysis_time_ms || 0,
    };
  }

  send(message: string, fileId?: string, callback?: ChatCallback) {
    if (callback) {
      this.currentCallback = callback;
    }

    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.messageQueue.push({ message, fileId });
      return;
    }

    this.ws.send(JSON.stringify({ message, file_id: fileId }));
  }

  disconnect() {
    this.ws?.close();
    this.ws = null;
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
