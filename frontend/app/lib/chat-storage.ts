// =============================================================================
// LOCAL STORAGE HELPERS — Extracted from page.tsx
// =============================================================================

import type { Chat, StoredChat, FileManifest } from './chat-types';

export const STORAGE_KEY = 'arca_chats';

export const saveChatsToStorage = (chats: Chat[]) => {
  try {
    const data: StoredChat[] = chats.map(chat => ({
      ...chat,
      createdAt: chat.createdAt.toISOString(),
      updatedAt: chat.updatedAt.toISOString(),
      messages: chat.messages.map(msg => ({
        ...msg,
        timestamp: msg.timestamp.toISOString(),
      })),
    }));
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch (e) {
    console.error('Failed to save chats to localStorage:', e);
  }
};

export const loadChatsFromStorage = (): Chat[] => {
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    if (!data) return [];

    const stored: StoredChat[] = JSON.parse(data);
    return stored.map(chat => ({
      ...chat,
      createdAt: new Date(chat.createdAt),
      updatedAt: new Date(chat.updatedAt),
      messages: chat.messages.map(msg => ({
        ...msg,
        timestamp: new Date(msg.timestamp),
      })),
    }));
  } catch (e) {
    console.error('Failed to load chats from localStorage:', e);
    return [];
  }
};

export const saveCurrentChatId = (id: string | null) => {
  if (id) {
    localStorage.setItem('arca_current_chat', id);
  } else {
    localStorage.removeItem('arca_current_chat');
  }
};

export const loadCurrentChatId = (): string | null => {
  return localStorage.getItem('arca_current_chat');
};

// =============================================================================
// FILE MANIFEST HELPERS (for recovery after disconnect)
// =============================================================================

export const FILE_MANIFEST_KEY = 'arca_file_manifest';

export const saveFileManifest = (files: FileManifest[]) => {
  try {
    localStorage.setItem(FILE_MANIFEST_KEY, JSON.stringify(files));
  } catch (e) {
    console.error('Failed to save file manifest:', e);
  }
};

export const loadFileManifest = (): FileManifest[] => {
  try {
    const data = localStorage.getItem(FILE_MANIFEST_KEY);
    return data ? JSON.parse(data) : [];
  } catch (e) {
    console.error('Failed to load file manifest:', e);
    return [];
  }
};

export const clearFileManifest = () => {
  localStorage.removeItem(FILE_MANIFEST_KEY);
};

export const getFileType = (filename: string): 'excel' | 'pdf' | 'word' => {
  const ext = filename.toLowerCase().split('.').pop();
  if (ext === 'pdf') return 'pdf';
  if (ext === 'docx' || ext === 'doc') return 'word';
  return 'excel';
};
