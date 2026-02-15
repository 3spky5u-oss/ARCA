// =============================================================================
// UTILITIES — Extracted from page.tsx
// =============================================================================

import { getWsBase } from '@/lib/api';

export const generateId = () => Math.random().toString(36).substring(2, 15);

// Debounce utility for performance optimization
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function debounce<T extends (...args: any[]) => void>(fn: T, delay: number): T {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  return ((...args: Parameters<T>) => {
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  }) as T;
}

export const generateSessionId = () => {
  const stored = localStorage.getItem('arca_session_id');
  if (stored) return stored;
  const newId = `session_${Date.now()}_${generateId()}`;
  localStorage.setItem('arca_session_id', newId);
  return newId;
};

export const getWsUrl = () => {
  return `${getWsBase()}/ws/chat`;
};

export const formatTime = (date: Date): string => {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));

  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days} days ago`;
  return date.toLocaleDateString();
};

export const TOOL_DISPLAY_NAMES: Record<string, string> = {
  'search_knowledge': 'Searching knowledge base',
  'search_session': 'Searching uploaded documents',
  'web_search': 'Searching the web',
  'analyze_files': 'Analyzing lab data',
  'generate_report': 'Generating report',
  'unit_convert': 'Converting units',
  'redact_document': 'Redacting document',
  'lookup_guideline': 'Looking up guideline',
  'thinking': 'Thinking',
};

// Fun rotating messages for "thinking" state (no trailing ... - ellipsis is animated)
export const THINKING_MESSAGES = [
  'Digging through the data',
  'Searching the archives',
  'Scanning the corpus',
  'Sampling the knowledge base',
  'Consolidating information',
  'Connecting the threads',
  'Sifting through results',
  'Piecing it together',
  'Cross-referencing sources',
  'Following the trail',
  'Parsing the details',
  'Weighing the evidence',
  // Generic fun ones
  'Pondering',
  'Cogitating',
  'Ruminating',
  'Mulling it over',
  'Connecting the dots',
  'Brewing up a response',
  'Assembling the pieces',
  'Working on it',
  'Crunching the numbers',
  'Putting pen to paper',
];

// Fun rotating messages for generic processing state (no trailing ... - ellipsis is animated)
export const PROCESSING_MESSAGES = [
  'On it',
  'Working',
  'Crunching',
  'Spinning up',
  'Getting to it',
  'Firing up',
  'Revving up',
  'Warming up',
  'Loading',
  'Engaging',
];

// Get a random thinking message (accepts domain messages or falls back to defaults)
export const getRandomThinkingMessage = (messages?: string[]): string => {
  const pool = messages && messages.length > 0 ? messages : THINKING_MESSAGES;
  return pool[Math.floor(Math.random() * pool.length)];
};

// Get a random processing message
export const getRandomProcessingMessage = (): string => {
  return PROCESSING_MESSAGES[Math.floor(Math.random() * PROCESSING_MESSAGES.length)];
};

// Get the most relevant tool label for inline display (no trailing ... - ellipsis is animated)
export const getActiveToolLabel = (tools: Set<string>, thinkingMessage: string, processingMessage: string): string => {
  // Priority order: specific tools first, thinking last
  const toolsArray = Array.from(tools);

  // If there's a non-thinking tool, show that
  const specificTool = toolsArray.find(t => t !== 'thinking');
  if (specificTool) {
    return TOOL_DISPLAY_NAMES[specificTool] || `Using ${specificTool}`;
  }

  // For thinking, use the fun rotating message
  if (tools.has('thinking')) {
    return thinkingMessage;
  }

  return processingMessage || 'Working';
};
