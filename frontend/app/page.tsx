'use client';

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import 'katex/dist/katex.min.css';
import { useToast } from '@/components/ui/use-toast';
import { ErrorDisplay, getErrorTitle, type ErrorData } from '@/components/error-display';
import { getApiBase } from '@/lib/api';
import { useDomain } from '@/lib/domain-context';

import type { Citation, Message, Chat, UploadedFile, FileManifest } from '@/app/lib/chat-types';
import {
  generateId, debounce, generateSessionId, getWsUrl, formatTime,
  getRandomThinkingMessage, getRandomProcessingMessage, getActiveToolLabel,
} from '@/app/lib/chat-utils';
import {
  saveChatsToStorage, loadChatsFromStorage, saveCurrentChatId, loadCurrentChatId,
  saveFileManifest, loadFileManifest, clearFileManifest, getFileType,
} from '@/app/lib/chat-storage';
import { MessageContent, Citations } from '@/app/components/chat';

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function Home() {
  // Domain config for branding/theming
  const domain = useDomain();

  // Toast hook for notifications
  const { toast } = useToast();

  // Chat state
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [sessionId] = useState(() => typeof window !== 'undefined' ? generateSessionId() : '');

  // Connection state
  const [isConnected, setIsConnected] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [mcpMode, setMcpMode] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeTools, setActiveTools] = useState<Set<string>>(new Set());
  const [removingTools, setRemovingTools] = useState<Set<string>>(new Set());

  // Mode toggles
  const [searchMode, setSearchMode] = useState(false);
  const [deepSearchMode, setDeepSearchMode] = useState(false);
  const [thinkMode, setThinkMode] = useState(false);
  const [calculateMode, setCalculateMode] = useState(false);  // qwq for math
  const [phiiEnabled, setPhiiEnabled] = useState(true);  // Phii enhancement enabled by default
  const [selectedModel, setSelectedModel] = useState<'qwen3' | 'coder' | 'deepseek'>('qwen3');

  // UI state
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [recoveryFiles, setRecoveryFiles] = useState<FileManifest[]>([]);
  const [showRecoveryPrompt, setShowRecoveryPrompt] = useState(false);
  const [flaggedMessages, setFlaggedMessages] = useState<Set<string>>(new Set());
  const [expandedThinking, setExpandedThinking] = useState<Set<string>>(new Set());  // Track expanded thinking per message
  const [showStreamingThinking, setShowStreamingThinking] = useState(false);  // Toggle live thinking stream visibility
  const [streamingThinking, setStreamingThinking] = useState('');  // Accumulated thinking tokens during streaming
  const [thinkingMessage, setThinkingMessage] = useState(() => getRandomThinkingMessage());
  const [processingMessage, setProcessingMessage] = useState(() => getRandomProcessingMessage());

  // Streaming text state (magic typewriter effect)
  const [streamingText, setStreamingText] = useState('');  // Accumulated streaming text
  const [introText, setIntroText] = useState('');  // Accumulated intro text
  const [newestToken, setNewestToken] = useState('');  // Current token being highlighted
  const [tokenKey, setTokenKey] = useState(0);  // Key to re-trigger animation

  // Auth state
  const [authChecked, setAuthChecked] = useState(false);
  const [authHasUsers, setAuthHasUsers] = useState(true);
  const [authToken, setAuthToken] = useState('');
  const [authUsername, setAuthUsername] = useState('');
  const [authRole, setAuthRole] = useState('');
  const [authFormUser, setAuthFormUser] = useState('');
  const [authFormPass, setAuthFormPass] = useState('');
  const [authFormConfirm, setAuthFormConfirm] = useState('');
  const [authError, setAuthError] = useState('');
  const [authShowRegister, setAuthShowRegister] = useState(false);
  // User menu / settings
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showChangePassword, setShowChangePassword] = useState(false);
  const [settingsDisplayName, setSettingsDisplayName] = useState('');
  const [settingsTheme, setSettingsTheme] = useState('dark');
  const [settingsPhiiLevel, setSettingsPhiiLevel] = useState('');
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [pwOld, setPwOld] = useState('');
  const [pwNew, setPwNew] = useState('');
  const [pwConfirm, setPwConfirm] = useState('');
  const [pwError, setPwError] = useState('');
  const [pwSaving, setPwSaving] = useState(false);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamingContentRef = useRef('');
  const currentChatIdRef = useRef<string | null>(null);
  const pendingCitationsRef = useRef<Citation[]>([]);
  const pendingConfidenceRef = useRef<number | undefined>(undefined);

  // Keep ref in sync
  useEffect(() => {
    currentChatIdRef.current = currentChatId;
    saveCurrentChatId(currentChatId);
  }, [currentChatId]);

  // Load chats from localStorage on mount
  useEffect(() => {
    const savedChats = loadChatsFromStorage();
    if (savedChats.length > 0) {
      setChats(savedChats);
      const savedCurrentId = loadCurrentChatId();
      if (savedCurrentId && savedChats.some(c => c.id === savedCurrentId)) {
        setCurrentChatId(savedCurrentId);
      }
    }
  }, []);

  // Auth check on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        // Check if any users exist
        const statusRes = await fetch(`${getApiBase()}/api/admin/auth/status`);
        const statusData = await statusRes.json();
        setAuthHasUsers(statusData.has_users);
      } catch {
        // Backend might not have the endpoint yet — assume users exist
        setAuthHasUsers(true);
      }

      // Check for stored token
      const stored = localStorage.getItem('arca_auth_token');
      const storedUser = localStorage.getItem('arca_auth_username');
      const storedRole = localStorage.getItem('arca_auth_role');
      if (stored) {
        try {
          const verifyRes = await fetch(`${getApiBase()}/api/admin/auth/verify`, {
            headers: { 'Authorization': `Bearer ${stored}` },
          });
          const verifyData = await verifyRes.json();
          if (verifyData.valid) {
            setAuthToken(stored);
            setAuthUsername(verifyData.username || storedUser || '');
            setAuthRole(verifyData.role || storedRole || 'user');
          } else {
            localStorage.removeItem('arca_auth_token');
            localStorage.removeItem('arca_auth_username');
            localStorage.removeItem('arca_auth_role');
          }
        } catch {
          localStorage.removeItem('arca_auth_token');
          localStorage.removeItem('arca_auth_username');
          localStorage.removeItem('arca_auth_role');
        }
      }
      setAuthChecked(true);
    };
    checkAuth();
  }, []);

  // Debounced save function - only syncs current chat, not all chats
  const debouncedSave = useMemo(
    () => debounce((chatsToSave: Chat[]) => {
      saveChatsToStorage(chatsToSave);
    }, 500),
    []
  );

  // Save chats to localStorage whenever they change (debounced)
  useEffect(() => {
    if (chats.length > 0) {
      debouncedSave(chats);
    }
  }, [chats, debouncedSave]);

  // Sync to backend periodically
  useEffect(() => {
    if (!sessionId || chats.length === 0) return;

    const syncToBackend = async () => {
      try {
        await fetch(`${getApiBase()}/api/sessions/${sessionId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            chats: chats.map(chat => ({
              ...chat,
              createdAt: chat.createdAt.toISOString(),
              updatedAt: chat.updatedAt.toISOString(),
              messages: chat.messages.map(msg => ({
                ...msg,
                timestamp: msg.timestamp.toISOString(),
              })),
            })),
          }),
        });
      } catch (e) {
        // Backend sync is optional - localStorage is primary
      }
    };

    const interval = setInterval(syncToBackend, 30000);
    return () => clearInterval(interval);
  }, [sessionId, chats]);

  // Current chat helper
  const currentChat = chats.find(c => c.id === currentChatId);
  const messages = currentChat?.messages || [];

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-scroll during streaming as tokens appear
  useEffect(() => {
    if (streamingText) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [streamingText]);

  // Auto-scroll during intro as text appears
  useEffect(() => {
    if (introText) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [introText]);

  // Admin panel shortcut (Ctrl+Shift+D)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'D') {
        e.preventDefault();
        window.open('/admin', '_blank');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Cycle through thinking messages while streaming
  useEffect(() => {
    if (!isStreaming || activeTools.size === 0) return;

    const interval = setInterval(() => {
      if (activeTools.has('thinking')) {
        setThinkingMessage(getRandomThinkingMessage(domain.thinking_messages));
      } else {
        setProcessingMessage(getRandomProcessingMessage());
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [isStreaming, activeTools]);

  // =============================================================================
  // MCP MODE POLLING — auto-recover when MCP mode is disabled
  // =============================================================================

  useEffect(() => {
    if (!mcpMode) return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${getApiBase()}/health`);
        if (res.ok) {
          const data = await res.json();
          if (!data.mcp_mode) { setMcpMode(false); }
        }
      } catch { /* ignore */ }
    }, 10000);
    return () => clearInterval(interval);
  }, [mcpMode]);

  // =============================================================================
  // WEBSOCKET CONNECTION
  // =============================================================================

  useEffect(() => {
    let reconnectTimeout: NodeJS.Timeout;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;

    const connect = async () => {
      // Check if backend was restarted
      try {
        const res = await fetch(`${getApiBase()}/api/instance`);
        const data = await res.json();
        const storedInstanceId = localStorage.getItem('arca_instance_id');

        if (storedInstanceId && storedInstanceId !== data.instance_id) {
          // Backend restarted - clear all local data
          console.log('Backend restarted - clearing local data');
          localStorage.removeItem('arca_chats');
          localStorage.removeItem('arca_current_chat');
          localStorage.removeItem('arca_session_id');
          setChats([]);
          setCurrentChatId(null);
          setUploadedFiles([]);
          setIntroText('');
        }

        // Store current instance ID
        localStorage.setItem('arca_instance_id', data.instance_id);

        // Check if files need to be recovered (backend restarted but we have manifest)
        const manifest = loadFileManifest();
        if (manifest.length > 0) {
          try {
            const filesRes = await fetch(`${getApiBase()}/api/files`);
            const filesData = await filesRes.json();
            const backendFiles = filesData.files || [];

            // Find files in manifest that are missing from backend
            const missing = manifest.filter(
              m => !backendFiles.some((b: { file_id: string }) => b.file_id === m.file_id)
            );

            if (missing.length > 0) {
              setRecoveryFiles(missing);
              setShowRecoveryPrompt(true);
            }
          } catch (e) {
            // If we can't check, show recovery for all manifest files
            setRecoveryFiles(manifest);
            setShowRecoveryPrompt(true);
          }
        }
      } catch (e) {
        console.warn('Could not check instance ID:', e);

        // Still check for file recovery on connection issues
        const manifest = loadFileManifest();
        if (manifest.length > 0) {
          setRecoveryFiles(manifest);
          setShowRecoveryPrompt(true);
        }
      }

      // Clear any stale files from previous session only if no recovery needed
      if (!showRecoveryPrompt) {
        try {
          await fetch(`${getApiBase()}/api/files`, { method: 'DELETE' });
        } catch (e) {}
      }

      // Wait for backend health before opening WS — avoids hammering during restart
      setReconnecting(true);
      let backendReady = false;
      for (let i = 0; i < 60; i++) {
        try {
          const res = await fetch(`${getApiBase()}/health`);
          if (res.ok) {
            const healthData = await res.json();
            if (healthData.mcp_mode) {
              setMcpMode(true);
              setReconnecting(false);
              return; // Don't open WebSocket
            }
            backendReady = true;
            break;
          }
        } catch {}
        await new Promise(r => setTimeout(r, 3000));
      }
      if (!backendReady) {
        console.log('Backend not reachable after 3 minutes');
        setReconnecting(false);
        return;
      }

      setMcpMode(false);
      const ws = new WebSocket(getWsUrl());

      ws.onopen = async () => {
        console.log('WebSocket connected');
        reconnectAttempts = 0;
        setIsConnected(true);
        setReconnecting(false);

        // Wait for LLM to be ready before sending intro
        const waitForLLM = async () => {
          for (let i = 0; i < 60; i++) {  // up to 5 minutes (60 * 5s)
            try {
              const res = await fetch(`${getApiBase()}/health`);
              const data = await res.json();
              if (data.checks?.llm === 'ok') return true;
            } catch {}
            await new Promise(r => setTimeout(r, 5000));
          }
          return false;
        };

        const llmReady = await waitForLLM();
        if (llmReady && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            message: '[INTRO] Say hi briefly',
            search_mode: false,
            think_mode: false
          }));
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        setActiveTools(new Set());
        setRemovingTools(new Set());
        // Reconnect with backoff, up to max attempts
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++;
          setReconnecting(true);
          const delay = Math.min(5000 * reconnectAttempts, 30000);
          console.log(`Reconnecting (${reconnectAttempts}/${maxReconnectAttempts}) in ${delay}ms...`);
          reconnectTimeout = setTimeout(connect, delay);
        } else {
          console.log('Max reconnect attempts reached');
          setReconnecting(false);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Handle tool events
          if (data.type === 'tool_start') {
            setActiveTools(prev => new Set([...prev, data.tool]));
            // Pick a new random message when thinking starts
            if (data.tool === 'thinking') {
              setThinkingMessage(getRandomThinkingMessage(domain.thinking_messages));
            } else {
              // Pick a processing message for other tools
              setProcessingMessage(getRandomProcessingMessage());
            }
            return;
          }

          if (data.type === 'tool_end') {
            // Mark as removing (triggers fadeOut animation)
            setRemovingTools(prev => new Set([...prev, data.tool]));

            // Actually remove after animation completes (200ms matches CSS duration)
            setTimeout(() => {
              setActiveTools(prev => {
                const next = new Set(prev);
                next.delete(data.tool);
                return next;
              });
              setRemovingTools(prev => {
                const next = new Set(prev);
                next.delete(data.tool);
                return next;
              });
            }, 200);

            // Clear streaming thinking when thinking ends
            if (data.tool === 'thinking') {
              // Keep the content for final message, just hide the live view
              setShowStreamingThinking(false);
            }
            return;
          }

          // Handle live thinking stream (from qwq calculate mode)
          if (data.type === 'thinking_stream') {
            setStreamingThinking(prev => prev + data.content);
            return;
          }

          if (data.type === 'stream') {
            const chatId = currentChatIdRef.current;

            // If no active chat, this is the intro message
            if (!chatId) {
              if (!data.done) {
                // Magic typewriter for intro - highlight newest token
                setIntroText(prev => prev + data.content);
                setNewestToken(data.content);
                setTokenKey(k => k + 1);
              }
              return;
            }

            if (!data.done) {
              // Accumulate streaming content in ref (for final state)
              streamingContentRef.current += data.content;
              // Magic typewriter - track newest token for highlight
              setStreamingText(prev => prev + data.content);
              setNewestToken(data.content);
              setTokenKey(k => k + 1);
            } else {
              // Stream complete - clear streaming state
              setStreamingText('');
              setNewestToken('');
              setIsStreaming(false);
              setActiveTools(new Set());
              setRemovingTools(new Set());

              // Use content from done message if available, otherwise use accumulated content
              const finalContent = data.content || streamingContentRef.current || '';
              streamingContentRef.current = '';

              // Debug log for final response
              console.log('Final response:', {
                tools_used: data.tools_used,
                citations: data.citations,
                content_length: data.content?.length
              });

              // Add content, analysis result, citations, and tools used
              setChats(prev => prev.map(chat => {
                if (chat.id !== chatId) return chat;
                const msgs = [...chat.messages];
                const lastMsg = msgs[msgs.length - 1];
                if (lastMsg && lastMsg.role === 'assistant') {
                  msgs[msgs.length - 1] = {
                    ...lastMsg,
                    content: finalContent || lastMsg.content,
                    analysisResult: data.analysis_result || lastMsg.analysisResult,
                    // Fix: Check for non-empty arrays explicitly
                    citations: (Array.isArray(data.citations) && data.citations.length > 0)
                      ? data.citations
                      : (pendingCitationsRef.current.length > 0 ? pendingCitationsRef.current : lastMsg.citations),
                    confidence: data.confidence ?? pendingConfidenceRef.current ?? lastMsg.confidence,
                    // Fix: Check for non-empty array explicitly
                    toolsUsed: (Array.isArray(data.tools_used) && data.tools_used.length > 0)
                      ? data.tools_used
                      : lastMsg.toolsUsed,
                    thinkMode: data.think_mode || lastMsg.thinkMode,
                    autoThink: data.auto_think || lastMsg.autoThink,
                    calculateMode: data.calculate_mode || lastMsg.calculateMode,
                    autoCalculate: data.auto_calculate || lastMsg.autoCalculate,
                    phii: data.phii || lastMsg.phii,
                    // Use backend thinking_content, or accumulated streaming thinking, or existing
                    thinkingContent: data.thinking_content || streamingThinking || lastMsg.thinkingContent,
                  };
                }
                return { ...chat, messages: msgs };
              }));

              // Clear streaming thinking after saving to message
              setStreamingThinking('');

              // Clear pending citations
              pendingCitationsRef.current = [];
              pendingConfidenceRef.current = undefined;
            }
          } else if (data.type === 'citations') {
            // Store citations for when stream completes
            pendingCitationsRef.current = data.citations || [];
            pendingConfidenceRef.current = data.confidence;
          } else if (data.type === 'error') {
            setIsStreaming(false);
            setActiveTools(new Set());
            setRemovingTools(new Set());

            // Parse structured error or create from content
            const errorData: ErrorData = typeof data.error === 'object' && data.error
              ? data.error
              : {
                  code: 'INTERNAL_UNEXPECTED',
                  message: data.content || 'An unexpected error occurred',
                  recoverable: false,
                };

            // Show toast for recoverable errors
            if (errorData.recoverable) {
              toast({
                title: getErrorTitle(errorData.code),
                description: errorData.message,
                variant: 'destructive',
              });
            }

            const chatId = currentChatIdRef.current;
            if (!chatId) return;
            setChats(prev => prev.map(chat => {
              if (chat.id !== chatId) return chat;
              const msgs = [...chat.messages];
              const lastMsg = msgs[msgs.length - 1];
              if (lastMsg && lastMsg.role === 'assistant') {
                msgs[msgs.length - 1] = {
                  ...lastMsg,
                  content: '',  // Clear content for error display
                  error: errorData,  // Store structured error
                };
              }
              return { ...chat, messages: msgs };
            }));
          }
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      };

      wsRef.current = ws;
    };

    connect();

    return () => {
      clearTimeout(reconnectTimeout);
      wsRef.current?.close();
    };
  }, []);

  // =============================================================================
  // AUTH ACTIONS
  // =============================================================================

  const isAuthenticated = !!authToken;

  const handleAuthSubmit = async (isRegister: boolean) => {
    setAuthError('');
    if (isRegister && authFormPass !== authFormConfirm) {
      setAuthError('Passwords do not match');
      return;
    }
    try {
      const endpoint = isRegister ? '/api/admin/auth/register' : '/api/admin/auth/login';
      const res = await fetch(`${getApiBase()}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: authFormUser, password: authFormPass }),
      });
      const data = await res.json();
      if (!res.ok) {
        setAuthError(data.detail || 'Authentication failed');
        return;
      }
      setAuthToken(data.token);
      setAuthUsername(data.username || authFormUser);
      setAuthRole(data.role || 'user');
      localStorage.setItem('arca_auth_token', data.token);
      localStorage.setItem('arca_auth_username', data.username || authFormUser);
      localStorage.setItem('arca_auth_role', data.role || 'user');
      // Also store for admin panel compatibility
      sessionStorage.setItem('arca_admin_token', data.token);
      sessionStorage.setItem('arca_admin_username', data.username || authFormUser);
      setAuthHasUsers(true);
      setAuthFormUser('');
      setAuthFormPass('');
      setAuthFormConfirm('');
      setAuthShowRegister(false);
    } catch {
      setAuthError('Could not connect to server');
    }
  };

  const handleLogout = () => {
    setAuthToken('');
    setAuthUsername('');
    setAuthRole('');
    localStorage.removeItem('arca_auth_token');
    localStorage.removeItem('arca_auth_username');
    localStorage.removeItem('arca_auth_role');
    sessionStorage.removeItem('arca_admin_token');
    sessionStorage.removeItem('arca_admin_username');
    setUserMenuOpen(false);
  };

  const handleLoadSettings = async () => {
    try {
      const res = await fetch(`${getApiBase()}/api/admin/user/settings`, {
        headers: { 'Authorization': `Bearer ${authToken}` },
      });
      if (res.ok) {
        const data = await res.json();
        const s = data.settings || {};
        setSettingsDisplayName(s.display_name || '');
        setSettingsTheme(s.theme || 'dark');
        setSettingsPhiiLevel(s.phii_level_override || '');
      }
    } catch {
      // Defaults are fine
    }
  };

  const handleSaveSettings = async () => {
    setSettingsSaving(true);
    try {
      const res = await fetch(`${getApiBase()}/api/admin/user/settings`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          display_name: settingsDisplayName,
          theme: settingsTheme,
          phii_level_override: settingsPhiiLevel,
        }),
      });
      if (res.ok) {
        setShowSettings(false);
        toast({ title: 'Settings saved', description: 'Your preferences have been updated.' });
      } else {
        const data = await res.json();
        toast({ title: 'Error', description: data.detail || 'Failed to save settings', variant: 'destructive' });
      }
    } catch {
      toast({ title: 'Error', description: 'Could not connect to server', variant: 'destructive' });
    } finally {
      setSettingsSaving(false);
    }
  };

  const handleChangePassword = async () => {
    setPwError('');
    if (pwNew !== pwConfirm) {
      setPwError('Passwords do not match');
      return;
    }
    if (pwNew.length < 8) {
      setPwError('New password must be at least 8 characters');
      return;
    }
    setPwSaving(true);
    try {
      const res = await fetch(`${getApiBase()}/api/admin/auth/change-password`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ old_password: pwOld, new_password: pwNew }),
      });
      const data = await res.json();
      if (!res.ok) {
        setPwError(data.detail || 'Failed to change password');
        return;
      }
      // Update stored token
      setAuthToken(data.token);
      localStorage.setItem('arca_auth_token', data.token);
      sessionStorage.setItem('arca_admin_token', data.token);
      setShowChangePassword(false);
      setPwOld('');
      setPwNew('');
      setPwConfirm('');
      toast({ title: 'Password changed', description: 'All other sessions have been invalidated.' });
    } catch {
      setPwError('Could not connect to server');
    } finally {
      setPwSaving(false);
    }
  };

  // Close user menu when clicking outside
  useEffect(() => {
    if (!userMenuOpen) return;
    const handleClick = () => setUserMenuOpen(false);
    document.addEventListener('click', handleClick);
    return () => document.removeEventListener('click', handleClick);
  }, [userMenuOpen]);

  // =============================================================================
  // CHAT ACTIONS
  // =============================================================================

  const createNewChat = useCallback(async () => {
    // Clear backend files
    try {
      await fetch(`${getApiBase()}/api/files`, { method: 'DELETE' });
    } catch (e) {
      console.error('Failed to clear files:', e);
    }

    const newChat: Chat = {
      id: generateId(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    };
    setChats(prev => [newChat, ...prev]);
    setCurrentChatId(newChat.id);
    setUploadedFiles([]);
    setIntroText('');
    // Clear file manifest for new session
    clearFileManifest();
    setShowRecoveryPrompt(false);
    setRecoveryFiles([]);
  }, []);

  const deleteChat = useCallback((id: string) => {
    setChats(prev => prev.filter(c => c.id !== id));
    if (currentChatId === id) {
      setCurrentChatId(null);
    }
  }, [currentChatId]);

  // Flag a message for Phii review
  const flagMessage = useCallback(async (messageId: string, userMessage: string, assistantResponse: string) => {
    if (flaggedMessages.has(messageId)) return;

    try {
      const response = await fetch(`${getApiBase()}/api/admin/phii/flag?session_id=${sessionId}&message_id=${messageId}&user_message=${encodeURIComponent(userMessage)}&assistant_response=${encodeURIComponent(assistantResponse.slice(0, 1000))}`, {
        method: 'POST',
      });
      const data = await response.json();
      if (data.success) {
        setFlaggedMessages(prev => new Set([...prev, messageId]));
      }
    } catch (e) {
      console.error('Failed to flag message:', e);
    }
  }, [sessionId, flaggedMessages]);

  // Toggle thinking content expansion for a message
  const toggleThinking = useCallback((messageId: string) => {
    setExpandedThinking(prev => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  }, []);

  const sendMessage = useCallback(() => {
    if (!input.trim() || !wsRef.current || isStreaming || !isConnected) return;

    const userMessage = input.trim();
    setInput('');

    // Create chat if needed
    let chatId = currentChatId;
    if (!chatId) {
      const newChat: Chat = {
        id: generateId(),
        title: userMessage.slice(0, 40) + (userMessage.length > 40 ? '...' : ''),
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date()
      };
      setChats(prev => [newChat, ...prev]);
      chatId = newChat.id;
      setCurrentChatId(chatId);
    }

    // Capture attached files before clearing
    const attachedFiles = uploadedFiles
      .filter(f => f.status === 'ready')
      .map(f => ({
        name: f.name,
        type: f.name.split('.').pop()?.toLowerCase() || 'unknown'
      }));

    // Add messages
    const userMsg: Message = {
      id: generateId(),
      role: 'user',
      content: userMessage,
      timestamp: new Date(),
      attachedFiles: attachedFiles.length > 0 ? attachedFiles : undefined
    };

    const assistantMsg: Message = {
      id: generateId(),
      role: 'assistant',
      content: '',
      timestamp: new Date()
    };

    setChats(prev => prev.map(chat => {
      if (chat.id !== chatId) return chat;
      const title = chat.messages.length === 0
        ? userMessage.slice(0, 40) + (userMessage.length > 40 ? '...' : '')
        : chat.title;
      return {
        ...chat,
        title,
        messages: [...chat.messages, userMsg, assistantMsg],
        updatedAt: new Date()
      };
    }));

    setIsStreaming(true);
    streamingContentRef.current = '';
    setStreamingThinking('');  // Clear previous thinking
    setShowStreamingThinking(false);  // Reset visibility
    // Reset streaming state for new message
    setStreamingText('');
    setNewestToken('');

    // Send to backend
    wsRef.current.send(JSON.stringify({
      message: userMessage,
      search_mode: searchMode,
      deep_search: deepSearchMode,
      think_mode: thinkMode,
      calculate_mode: calculateMode,
      phii_enabled: phiiEnabled
    }));

    // Clear uploaded files from UI after sending
    setUploadedFiles([]);
  }, [input, currentChatId, isStreaming, isConnected, searchMode, deepSearchMode, thinkMode, calculateMode, phiiEnabled, uploadedFiles]);

  // =============================================================================
  // FILE UPLOAD
  // =============================================================================

  // Track pending uploads to prevent duplicate temp entries
  const pendingUploadsRef = useRef<Set<string>>(new Set());

  const handleFileUpload = useCallback(async (file: File) => {
    // Generate unique ID with timestamp + random string + filename hash
    const tempId = `temp_${Date.now()}_${Math.random().toString(36).substring(2, 9)}_${file.name.length}`;

    // Prevent duplicate uploads for the same file
    const fileKey = `${file.name}_${file.size}_${file.lastModified}`;
    if (pendingUploadsRef.current.has(fileKey)) {
      console.log('Upload already in progress for:', file.name);
      return;
    }
    pendingUploadsRef.current.add(fileKey);

    // 1. Show file immediately with uploading status
    const pendingFile: UploadedFile = {
      id: tempId,
      name: file.name,
      samples: 0,
      parameters: 0,
      status: 'uploading'
    };
    setUploadedFiles(prev => {
      // Extra safety: don't add if temp entry for this file already exists
      const existingTemp = prev.find(f => f.id.startsWith('temp_') && f.name === file.name);
      if (existingTemp) return prev;
      return [...prev, pendingFile];
    });

    const formData = new FormData();
    formData.append('file', file);

    try {
      // 2. Update to parsing status
      setUploadedFiles(prev => prev.map(f =>
        f.id === tempId ? { ...f, status: 'parsing' as const } : f
      ));

      const response = await fetch(`${getApiBase()}/api/upload`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        // 3. Replace temp file with real data - filter ALL temp entries for this filename
        setUploadedFiles(prev => [
          ...prev.filter(f => !(f.id === tempId || (f.id.startsWith('temp_') && f.name === file.name))),
          {
            id: data.file_id,
            name: file.name,
            samples: data.samples,
            parameters: data.parameters,
            status: 'ready' as const,
            rag_chunks: data.rag_chunks
          }
        ]);

        // 4. Save to manifest for recovery
        const manifest = loadFileManifest();
        const newEntry: FileManifest = {
          file_id: data.file_id,
          filename: file.name,
          size: file.size,
          type: getFileType(file.name),
          uploadedAt: new Date().toISOString(),
        };
        saveFileManifest([...manifest.filter(m => m.file_id !== data.file_id), newEntry]);
      } else {
        // Parse error response
        let errorMessage = 'Upload failed';
        try {
          const errorData = await response.json();
          if (errorData.error?.message) {
            errorMessage = errorData.error.message;
          } else if (errorData.detail) {
            errorMessage = errorData.detail;
          }
        } catch {
          // Use default message
        }

        // Show toast notification
        toast({
          title: 'Upload Failed',
          description: errorMessage,
          variant: 'destructive',
        });

        // Remove the failed file
        setUploadedFiles(prev => prev.filter(f => !(f.id === tempId || (f.id.startsWith('temp_') && f.name === file.name))));
      }
    } catch (error) {
      console.error('Upload failed:', error);

      // Show toast for network errors
      toast({
        title: 'Upload Failed',
        description: 'Could not connect to the server. Please try again.',
        variant: 'destructive',
      });

      setUploadedFiles(prev => prev.filter(f => !(f.id === tempId || (f.id.startsWith('temp_') && f.name === file.name))));
    } finally {
      // Always clear the pending flag
      pendingUploadsRef.current.delete(fileKey);
    }
  }, [toast]);

  // Drag & drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files) {
      Array.from(files).forEach(file => handleFileUpload(file));
    }
  };

  // Keyboard handler
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // =============================================================================
  // RENDER
  // =============================================================================

  // Helper to get file type color classes
  const getFileTypeStyles = (filename: string) => {
    const ext = filename.toLowerCase().split('.').pop();
    if (ext === 'pdf') {
      return { bg: 'bg-red-600/20', border: 'border-red-600/30', text: 'text-red-400' };
    }
    if (ext === 'docx' || ext === 'doc') {
      return { bg: 'bg-blue-600/20', border: 'border-blue-600/30', text: 'text-blue-400' };
    }
    // Excel/CSV default
    return { bg: 'bg-green-600/20', border: 'border-green-600/30', text: 'text-green-400' };
  };

  // MCP Mode landing page — full-screen status when chat is disabled
  if (mcpMode) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-[#1a1a1a] text-white">
        <div className="text-center space-y-4 max-w-md">
          <h1 className="text-xl font-semibold">MCP Mode Active</h1>
          <p className="text-gray-400 text-sm">
            ARCA is running as a tool backend for cloud AI models.
            The local chat interface is disabled.
          </p>
          <p className="text-gray-500 text-xs">
            Your MCP client (Claude Desktop, GPT, etc.) needs to be configured to connect to ARCA.
            See the MCP Integration section in the README for setup steps.
          </p>
          <div className="flex gap-3 mt-4">
            <a href="/admin" className="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm transition-colors">
              Open Admin Panel
            </a>
            <a href="https://github.com/3spky5u-oss/ARCA#mcp-integration-external-ai-clients" target="_blank" rel="noopener noreferrer"
              className="inline-block px-4 py-2 bg-[#333] hover:bg-[#444] rounded-xl text-sm transition-colors text-gray-300">
              MCP Setup Guide
            </a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-[#212121] text-gray-100">
      {/* Mobile Overlay - shown when sidebar is open on mobile */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 md:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside className={`
        ${sidebarOpen ? 'w-full md:w-64' : 'w-0'}
        fixed md:relative inset-y-0 left-0 z-40
        bg-[#171717] border-r border-[#2a2a2a]
        flex flex-col transition-all duration-200 overflow-hidden
      `}>
        {/* New Chat Button */}
        <div className="p-3">
          <button
            onClick={createNewChat}
            aria-label="Start new chat"
            className="w-full flex items-center gap-3 px-4 py-3 bg-[#2a2a2a] hover:bg-[#333]
                       rounded-lg transition-colors text-sm font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-[#171717]"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New Chat
          </button>
        </div>

        {/* Chat List */}
        <div className="flex-1 overflow-y-auto px-2">
          {chats.length === 0 ? (
            <p className="text-center text-gray-500 text-sm py-8">No conversations yet</p>
          ) : (
            <>
              {['Today', 'Yesterday', 'Previous'].map(timeGroup => {
                const groupChats = chats.filter(chat => {
                  const label = formatTime(chat.updatedAt);
                  if (timeGroup === 'Previous') {
                    return label !== 'Today' && label !== 'Yesterday';
                  }
                  return label === timeGroup;
                });

                if (groupChats.length === 0) return null;

                return (
                  <div key={timeGroup} className="mb-4">
                    <div className="px-3 py-2 text-xs font-medium text-gray-500 uppercase">
                      {timeGroup}
                    </div>
                    {groupChats.map(chat => (
                      <div
                        key={chat.id}
                        onClick={() => setCurrentChatId(chat.id)}
                        className={`
                          group flex items-center justify-between px-3 py-2.5 rounded-lg cursor-pointer mb-1
                          ${currentChatId === chat.id
                            ? 'bg-[#2a2a2a] text-white'
                            : 'text-gray-400 hover:bg-[#2a2a2a]/50 hover:text-gray-200'}
                        `}
                      >
                        <span className="truncate text-sm">{chat.title}</span>
                        <button
                          onClick={(e) => { e.stopPropagation(); deleteChat(chat.id); }}
                          aria-label="Delete chat"
                          className="opacity-0 group-hover:opacity-100 p-1 hover:bg-[#333] rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:opacity-100"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                );
              })}
            </>
          )}
        </div>

        {/* Sidebar Footer */}
        <div className="p-3 border-t border-[#2a2a2a]">
          <div className="flex items-center gap-2 px-2">
            <img src="/logo.png" alt="" className="w-12 h-12 object-contain" />
            <span className="text-lg font-brand font-light tracking-wide opacity-80">{domain.app_name}</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between px-4 py-3 border-b border-[#2a2a2a] bg-[#212121]">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label="Toggle sidebar"
              aria-expanded={sidebarOpen}
              className="p-2 hover:bg-[#2a2a2a] rounded-lg transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-[#212121]"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>

            {/* Logo */}
            <div className="flex items-center gap-2">
              <img src="/logo.png" alt="" className="w-20 h-20 object-contain" />
              <span className="text-xl font-brand font-light tracking-wide opacity-90">{domain.app_name}</span>
            </div>
          </div>

          {/* Right side: User menu (admin gear moved inside dropdown) */}
          <div className="flex items-center gap-2">
            {/* User Menu */}
            {isAuthenticated && (
              <div className="relative">
                <button
                  onClick={(e) => { e.stopPropagation(); setUserMenuOpen(!userMenuOpen); }}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-400 hover:text-gray-200 hover:bg-[#2a2a2a] rounded-lg transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                  <span>{authUsername}</span>
                  <svg className={`w-3 h-3 transition-transform ${userMenuOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>

                {userMenuOpen && (
                  <div className="absolute right-0 top-full mt-1 w-48 bg-[#2a2a2a] border border-[#3a3a3a] rounded-lg shadow-xl z-50 py-1">
                    <button
                      onClick={() => { setUserMenuOpen(false); handleLoadSettings(); setShowSettings(true); }}
                      className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-[#333] hover:text-white transition-colors"
                    >
                      Settings
                    </button>
                    <button
                      onClick={() => { setUserMenuOpen(false); setShowChangePassword(true); }}
                      className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-[#333] hover:text-white transition-colors"
                    >
                      Change Password
                    </button>
                    {authRole === 'admin' && (
                      <>
                        <div className="border-t border-[#3a3a3a] my-1" />
                        <button
                          onClick={() => { setUserMenuOpen(false); window.location.href = '/admin'; }}
                          className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-[#333] hover:text-white transition-colors"
                        >
                          Admin Panel
                        </button>
                      </>
                    )}
                    <div className="border-t border-[#3a3a3a] my-1" />
                    <button
                      onClick={handleLogout}
                      className="w-full text-left px-4 py-2 text-sm text-red-400 hover:bg-[#333] hover:text-red-300 transition-colors"
                    >
                      Logout
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </header>

        {/* Auth Modal - First-boot registration or Login */}
        {authChecked && !isAuthenticated && (
          <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
            <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded-2xl w-full max-w-md p-8">
              <div className="flex flex-col items-center mb-6">
                <img src="/logo.png" alt="" className="w-32 h-32 object-contain mb-1" />
                <h2 className="text-2xl font-brand font-light text-gray-100">
                  {!authHasUsers ? 'Welcome to ARCA' : (authShowRegister ? 'Create Account' : 'Sign In')}
                </h2>
                <p className="text-sm text-gray-500 mt-1">
                  {!authHasUsers
                    ? 'Create your admin account to get started'
                    : (authShowRegister ? 'Create a new user account' : 'Enter your credentials to continue')}
                </p>
              </div>

              {authError && (
                <div className="mb-4 px-3 py-2 bg-red-900/30 border border-red-600/30 rounded-lg text-sm text-red-400">
                  {authError}
                </div>
              )}

              <div className="space-y-3">
                <input
                  type="text"
                  placeholder="Username"
                  value={authFormUser}
                  onChange={e => setAuthFormUser(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.nextElementSibling?.querySelector('input')?.focus(); }}
                  autoFocus
                  className="w-full px-4 py-3 bg-[#2a2a2a] border border-[#3a3a3a] rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
                />
                <input
                  type="password"
                  placeholder="Password"
                  value={authFormPass}
                  onChange={e => setAuthFormPass(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter') {
                      if (!authHasUsers || authShowRegister) {
                        // Focus confirm field
                      } else {
                        handleAuthSubmit(false);
                      }
                    }
                  }}
                  className="w-full px-4 py-3 bg-[#2a2a2a] border border-[#3a3a3a] rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
                />
                {(!authHasUsers || authShowRegister) && (
                  <input
                    type="password"
                    placeholder="Confirm Password"
                    value={authFormConfirm}
                    onChange={e => setAuthFormConfirm(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') handleAuthSubmit(true); }}
                    className="w-full px-4 py-3 bg-[#2a2a2a] border border-[#3a3a3a] rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
                  />
                )}
              </div>

              <button
                onClick={() => handleAuthSubmit(!authHasUsers || authShowRegister)}
                className="w-full mt-5 px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
              >
                {!authHasUsers ? 'Create Admin Account' : (authShowRegister ? 'Create Account' : 'Sign In')}
              </button>

              {authHasUsers && (
                <p className="text-center text-sm text-gray-500 mt-4">
                  {authShowRegister ? (
                    <button onClick={() => { setAuthShowRegister(false); setAuthError(''); }} className="text-blue-400 hover:text-blue-300">
                      Back to sign in
                    </button>
                  ) : (
                    <button onClick={() => { setAuthShowRegister(true); setAuthError(''); }} className="text-blue-400 hover:text-blue-300">
                      Create account
                    </button>
                  )}
                </p>
              )}
            </div>
          </div>
        )}

        {/* Settings Modal */}
        {showSettings && (
          <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={() => setShowSettings(false)}>
            <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded-2xl w-full max-w-md p-6" onClick={e => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-5">
                <h3 className="text-lg font-medium text-gray-100">Settings</h3>
                <button onClick={() => setShowSettings(false)} className="p-1 text-gray-500 hover:text-gray-300">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Display Name</label>
                  <input
                    type="text"
                    placeholder={authUsername}
                    value={settingsDisplayName}
                    onChange={e => setSettingsDisplayName(e.target.value)}
                    className="w-full px-3 py-2 bg-[#2a2a2a] border border-[#3a3a3a] rounded-lg text-white placeholder-gray-600 focus:outline-none focus:border-blue-500 text-sm"
                  />
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-1">Theme</label>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setSettingsTheme('dark')}
                      className={`flex-1 px-3 py-2 rounded-lg text-sm transition-colors ${
                        settingsTheme === 'dark'
                          ? 'bg-blue-600/20 border border-blue-600/50 text-blue-400'
                          : 'bg-[#2a2a2a] border border-[#3a3a3a] text-gray-400 hover:text-gray-200'
                      }`}
                    >
                      Dark
                    </button>
                    <button
                      onClick={() => setSettingsTheme('light')}
                      className={`flex-1 px-3 py-2 rounded-lg text-sm transition-colors ${
                        settingsTheme === 'light'
                          ? 'bg-blue-600/20 border border-blue-600/50 text-blue-400'
                          : 'bg-[#2a2a2a] border border-[#3a3a3a] text-gray-400 hover:text-gray-200'
                      }`}
                    >
                      Light
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-1">Expertise Level</label>
                  <select
                    value={settingsPhiiLevel}
                    onChange={e => setSettingsPhiiLevel(e.target.value)}
                    className="w-full px-3 py-2 bg-[#2a2a2a] border border-[#3a3a3a] rounded-lg text-white focus:outline-none focus:border-blue-500 text-sm"
                  >
                    <option value="">Let ARCA detect</option>
                    <option value="beginner">Beginner</option>
                    <option value="experienced">Experienced</option>
                    <option value="expert">Expert</option>
                  </select>
                  <p className="text-xs text-gray-600 mt-1">Controls how technical ARCA&apos;s responses are</p>
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => setShowSettings(false)}
                  className="flex-1 px-4 py-2 bg-[#2a2a2a] hover:bg-[#333] text-gray-300 rounded-lg transition-colors text-sm"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveSettings}
                  disabled={settingsSaving}
                  className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm disabled:opacity-50"
                >
                  {settingsSaving ? 'Saving...' : 'Save'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Change Password Modal */}
        {showChangePassword && (
          <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={() => setShowChangePassword(false)}>
            <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded-2xl w-full max-w-md p-6" onClick={e => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-5">
                <h3 className="text-lg font-medium text-gray-100">Change Password</h3>
                <button onClick={() => { setShowChangePassword(false); setPwError(''); }} className="p-1 text-gray-500 hover:text-gray-300">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {pwError && (
                <div className="mb-4 px-3 py-2 bg-red-900/30 border border-red-600/30 rounded-lg text-sm text-red-400">
                  {pwError}
                </div>
              )}

              <div className="space-y-3">
                <input
                  type="password"
                  placeholder="Current password"
                  value={pwOld}
                  onChange={e => setPwOld(e.target.value)}
                  autoFocus
                  className="w-full px-3 py-2 bg-[#2a2a2a] border border-[#3a3a3a] rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 text-sm"
                />
                <input
                  type="password"
                  placeholder="New password"
                  value={pwNew}
                  onChange={e => setPwNew(e.target.value)}
                  className="w-full px-3 py-2 bg-[#2a2a2a] border border-[#3a3a3a] rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 text-sm"
                />
                <input
                  type="password"
                  placeholder="Confirm new password"
                  value={pwConfirm}
                  onChange={e => setPwConfirm(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') handleChangePassword(); }}
                  className="w-full px-3 py-2 bg-[#2a2a2a] border border-[#3a3a3a] rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 text-sm"
                />
              </div>

              <div className="flex gap-3 mt-5">
                <button
                  onClick={() => { setShowChangePassword(false); setPwError(''); }}
                  className="flex-1 px-4 py-2 bg-[#2a2a2a] hover:bg-[#333] text-gray-300 rounded-lg transition-colors text-sm"
                >
                  Cancel
                </button>
                <button
                  onClick={handleChangePassword}
                  disabled={pwSaving}
                  className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm disabled:opacity-50"
                >
                  {pwSaving ? 'Changing...' : 'Change Password'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* File Recovery Prompt */}
        {showRecoveryPrompt && recoveryFiles.length > 0 && (
          <div className="bg-amber-900/30 border-b border-amber-600/30 px-4 py-3">
            <div className="max-w-3xl mx-auto flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <svg className="w-5 h-5 text-amber-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <div>
                  <p className="text-amber-200 text-sm font-medium">
                    Session files need to be re-uploaded
                  </p>
                  <p className="text-amber-400/70 text-xs">
                    {recoveryFiles.map(f => f.filename).join(', ')}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    fileInputRef.current?.click();
                  }}
                  className="px-3 py-1.5 bg-amber-600 hover:bg-amber-700 text-white text-sm rounded-lg transition-colors"
                >
                  Re-upload
                </button>
                <button
                  onClick={() => {
                    setShowRecoveryPrompt(false);
                    setRecoveryFiles([]);
                    clearFileManifest();
                  }}
                  className="px-3 py-1.5 text-amber-400 hover:text-amber-300 text-sm transition-colors"
                >
                  Dismiss
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Chat Area */}
        <main
          className="flex-1 overflow-y-auto"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {/* Drag Overlay */}
          {isDragging && (
            <div className="fixed inset-0 bg-blue-500/10 border-2 border-dashed border-blue-500 z-50 flex items-center justify-center">
              <div className="text-center">
                <svg className="w-12 h-12 mx-auto text-blue-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="text-blue-400 font-medium">Drop file here</p>
                <p className="text-gray-500 text-sm">Excel, PDF, or Word</p>
              </div>
            </div>
          )}

          {messages.length === 0 ? (
            /* Welcome Screen - Clean */
            <div className="flex flex-col items-center justify-center h-full px-4">
              <img src="/logo.png" alt="" className={`w-44 h-44 mb-0 object-contain transition-all duration-300 ${
                isConnected ? 'opacity-100' : 'opacity-40'
              }`} />
              <h1 className="text-3xl font-brand font-light tracking-wide opacity-90 mb-3">{domain.app_name}</h1>

              {/* LLM Intro Message - magic typewriter effect */}
              {introText ? (
                <p className="text-gray-400 max-w-md text-center leading-relaxed">
                  {/* Text without newest token */}
                  <span>{introText.slice(0, -newestToken.length)}</span>
                  {/* Newest token with highlight animation */}
                  <span key={tokenKey} className="streaming-token-highlight">{newestToken}</span>
                </p>
              ) : !isConnected && (
                <div className="flex flex-col items-center gap-2">
                  <p className="text-gray-500 text-sm">
                    {reconnecting ? 'Reconnecting...' : 'Warming up...'}
                  </p>
                  {reconnecting && (
                    <button
                      onClick={() => window.location.reload()}
                      className="text-blue-400 text-xs hover:underline"
                    >
                      Reload page
                    </button>
                  )}
                </div>
              )}

            </div>
          ) : (
            /* Messages */
            <div className="max-w-4xl mx-auto py-6 px-4">
              {messages.map((msg, msgIndex) => {
                // Check if this is the currently streaming message
                const isCurrentlyStreaming = isStreaming && msg.role === 'assistant' && msgIndex === messages.length - 1;

                // Get previous user message for flagging context
                const prevUserMsg = msgIndex > 0 ? messages[msgIndex - 1] : null;
                const prevUserContent = prevUserMsg?.role === 'user' ? prevUserMsg.content : '';

                return (
                <div key={msg.id} className={`group flex mb-6 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  {/* Logo icon for assistant messages */}
                  {msg.role === 'assistant' && (
                    <img src="/logo.png" alt="" className="w-10 h-10 object-contain mr-3 select-none flex-shrink-0 mt-1" />
                  )}
                  <div className={`
                    max-w-[85%] py-2
                    ${msg.role === 'user'
                      ? 'text-gray-300'
                      : 'text-gray-100'}
                  `}>
                    <div className="text-[15px] leading-relaxed">
                      {/* Error display takes precedence */}
                      {msg.error ? (
                        <ErrorDisplay error={msg.error} />
                      ) : isCurrentlyStreaming && streamingText ? (
                        // Streaming: magic typewriter (newest word highlights then fades)
                        <div className="whitespace-pre-wrap">
                          {/* Text without newest token */}
                          <span>{streamingText.slice(0, -newestToken.length)}</span>
                          {/* Newest token with highlight animation */}
                          <span key={tokenKey} className="streaming-token-highlight">{newestToken}</span>
                          <span className="opacity-30 animate-pulse">|</span>
                        </div>
                      ) : msg.content ? (
                        // Done streaming: full markdown
                        <MessageContent content={msg.content} isUser={msg.role === 'user'} />
                      ) : (isStreaming && msg.role === 'assistant' ? (
                        <div>
                          <span
                            onClick={activeTools.has('thinking') && streamingThinking.length > 0 ? () => setShowStreamingThinking(!showStreamingThinking) : undefined}
                            className={`inline-flex items-center text-gray-400 ${
                              activeTools.has('thinking') && streamingThinking.length > 0
                                ? 'cursor-pointer hover:text-gray-300'
                                : ''
                            }`}
                          >
                            {activeTools.has('thinking') && streamingThinking.length > 0 && (
                              <svg
                                className={`w-3 h-3 mr-1.5 transition-transform ${showStreamingThinking ? 'rotate-90' : ''}`}
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                              >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                              </svg>
                            )}
                            <span className="text-sm">{getActiveToolLabel(activeTools, thinkingMessage, processingMessage)}</span>
                            <span className="thinking-ellipsis text-sm">
                              <span className="dot">.</span>
                              <span className="dot">.</span>
                              <span className="dot">.</span>
                            </span>
                            {activeTools.has('thinking') && streamingThinking.length > 0 && (
                              <span className="ml-2 text-xs text-gray-500">({streamingThinking.length.toLocaleString()})</span>
                            )}
                          </span>
                          {/* Live streaming thinking content */}
                          {showStreamingThinking && streamingThinking.length > 0 && (
                            <div className="mt-2 p-3 bg-gray-800/50 border-l-2 border-gray-600 rounded text-xs text-gray-400 font-mono max-h-[300px] overflow-y-auto whitespace-pre-wrap animate-fadeIn">
                              {streamingThinking}
                            </div>
                          )}
                        </div>
                      ) : '')}
                    </div>

                    {/* Attached Files Indicator */}
                    {msg.role === 'user' && msg.attachedFiles && msg.attachedFiles.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 mt-2">
                        {msg.attachedFiles.map((file, idx) => {
                          const ext = file.type;
                          const isExcel = ext === 'xlsx' || ext === 'xls' || ext === 'csv';
                          const isPdf = ext === 'pdf';
                          const isWord = ext === 'docx' || ext === 'doc';
                          const bgColor = isExcel ? 'bg-green-500/20' : isPdf ? 'bg-red-500/20' : isWord ? 'bg-blue-500/20' : 'bg-gray-500/20';
                          const textColor = isExcel ? 'text-green-300' : isPdf ? 'text-red-300' : isWord ? 'text-blue-300' : 'text-gray-300';
                          return (
                            <span key={idx} className={`inline-flex items-center gap-1 px-2 py-0.5 ${bgColor} rounded-full text-xs ${textColor}`}>
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                              </svg>
                              {file.name.length > 20 ? file.name.slice(0, 17) + '...' : file.name}
                            </span>
                          );
                        })}
                      </div>
                    )}

                    {/* Response indicators (subtle) */}
                    {msg.role === 'assistant' && (msg.phii?.personalized || msg.autoThink || msg.autoCalculate) && !msg.phii?.command && (
                      <div className="flex items-center gap-2 text-[11px] text-gray-500 mt-2">
                        {msg.phii?.personalized && <span>Personalized</span>}
                        {msg.autoThink && (
                          <span className="text-purple-400/70">
                            Think auto-enabled
                          </span>
                        )}
                        {msg.autoCalculate && (
                          <span className="text-green-400/70">
                            Calculate auto-enabled
                          </span>
                        )}
                      </div>
                    )}

                    {/* Thinking Content (Expandable) */}
                    {msg.role === 'assistant' && msg.thinkingContent && (
                      <div className="mt-3">
                        <button
                          onClick={() => toggleThinking(msg.id)}
                          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors"
                        >
                          <svg
                            className={`w-3 h-3 transition-transform ${expandedThinking.has(msg.id) ? 'rotate-90' : ''}`}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                          {expandedThinking.has(msg.id) ? 'Hide thinking' : 'View thinking'}
                          <span className="text-gray-600">({msg.thinkingContent.length.toLocaleString()} chars)</span>
                        </button>
                        {expandedThinking.has(msg.id) && (
                          <div className="mt-2 p-3 bg-gray-800/50 border-l-2 border-gray-600 rounded text-xs text-gray-400 font-mono max-h-[400px] overflow-y-auto whitespace-pre-wrap animate-fadeIn">
                            {msg.thinkingContent}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Citations */}
                    {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
                      <Citations citations={msg.citations} confidence={msg.confidence} />
                    )}

                    {/* Analysis Results (Lab Compliance - only if summary is an object with total_samples) */}
                    {msg.analysisResult?.summary && typeof msg.analysisResult.summary === 'object' && 'total_samples' in msg.analysisResult.summary && (
                      <div className="mt-4 pt-4 border-t border-gray-600/30">
                        {/* Stats Grid - 2 cols mobile, 4 cols desktop */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
                          <div className="bg-[#333] rounded-lg p-2 text-center">
                            <div className="text-lg font-bold">{msg.analysisResult.summary.total_samples}</div>
                            <div className="text-xs text-gray-400">Samples</div>
                          </div>
                          <div className="bg-[#333] rounded-lg p-2 text-center">
                            <div className="text-lg font-bold">{msg.analysisResult.summary.total_comparisons}</div>
                            <div className="text-xs text-gray-400">Compared</div>
                          </div>
                          <div className="bg-[#333] rounded-lg p-2 text-center">
                            <div className="text-lg font-bold text-red-400">{msg.analysisResult.summary.exceedance_count}</div>
                            <div className="text-xs text-gray-400">Exceedances</div>
                          </div>
                          <div className="bg-[#333] rounded-lg p-2 text-center">
                            <div className="text-lg font-bold text-green-400">{msg.analysisResult.summary.compliant_count}</div>
                            <div className="text-xs text-gray-400">Compliant</div>
                          </div>
                        </div>

                        {/* Exceedances Table */}
                        {msg.analysisResult.exceedances && msg.analysisResult.exceedances.length > 0 && (
                          <div className="overflow-x-auto mb-4">
                            <table className="w-full text-sm">
                              <thead>
                                <tr className="text-gray-400 border-b border-gray-600/30">
                                  <th className="text-left py-1.5 pr-2">Sample</th>
                                  <th className="text-left py-1.5 pr-2">Parameter</th>
                                  <th className="text-right py-1.5 pr-2">Value</th>
                                  <th className="text-right py-1.5 pr-2">Guideline</th>
                                  <th className="text-right py-1.5">Factor</th>
                                </tr>
                              </thead>
                              <tbody>
                                {msg.analysisResult.exceedances.slice(0, 8).map((exc, i) => (
                                  <tr key={i} className="border-b border-gray-600/20">
                                    <td className="py-1.5 pr-2 font-mono text-xs">{exc.sample_id}</td>
                                    <td className="py-1.5 pr-2">{exc.parameter}</td>
                                    <td className="py-1.5 pr-2 text-right text-red-400 font-mono">{exc.value_str}</td>
                                    <td className="py-1.5 pr-2 text-right text-gray-400 font-mono">{exc.guideline}</td>
                                    <td className="py-1.5 text-right">
                                      <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${
                                        exc.exceedance_factor >= 5 ? 'bg-red-600' :
                                        exc.exceedance_factor >= 2 ? 'bg-red-500/30 text-red-400' :
                                        'bg-orange-500/30 text-orange-400'
                                      }`}>
                                        {exc.exceedance_factor.toFixed(1)}x
                                      </span>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                            {msg.analysisResult.exceedances.length > 8 && (
                              <p className="text-center text-gray-500 text-xs mt-2">
                                +{msg.analysisResult.exceedances.length - 8} more
                              </p>
                            )}
                          </div>
                        )}

                        {/* Download Buttons */}
                        <div className="flex gap-2">
                          {msg.analysisResult.excel_report && (
                            <a
                              href={`${getApiBase()}${msg.analysisResult.excel_report}`}
                              download
                              className="flex items-center gap-2 px-3 py-1.5 bg-green-600 hover:bg-green-700 rounded-xl text-sm font-medium transition-colors"
                            >
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                              </svg>
                              Excel
                            </a>
                          )}
                          {msg.analysisResult.word_report && (
                            <a
                              href={`${getApiBase()}${msg.analysisResult.word_report}`}
                              download
                              className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm font-medium transition-colors"
                            >
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                              </svg>
                              Word
                            </a>
                          )}
                          {msg.analysisResult.redacted_file && (
                            <a
                              href={`${getApiBase()}${msg.analysisResult.redacted_file}`}
                              download
                              className="flex items-center gap-2 px-3 py-1.5 bg-purple-600 hover:bg-purple-700 rounded-xl text-sm font-medium transition-colors"
                            >
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                              </svg>
                              Redacted
                            </a>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Redacted File Download (when no summary - just redaction) */}
                    {msg.analysisResult?.redacted_file && !msg.analysisResult?.summary && (
                      <div className="mt-4 pt-4 border-t border-gray-600/30">
                        <div className="flex gap-2">
                          <a
                            href={`${getApiBase()}${msg.analysisResult.redacted_file}`}
                            download
                            className="flex items-center gap-2 px-3 py-1.5 bg-purple-600 hover:bg-purple-700 rounded-xl text-sm font-medium transition-colors"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                            </svg>
                            Redacted
                          </a>
                        </div>
                      </div>
                    )}

                    {/* Borehole Extraction Results (LoggView) */}
                    {msg.analysisResult?.boreholes && msg.analysisResult.boreholes.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-gray-600/30">
                        <div className="flex items-center gap-2 mb-3">
                          <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                          </svg>
                          <span className="text-sm font-medium text-gray-300">
                            Extracted {msg.analysisResult.borehole_count || msg.analysisResult.boreholes.length} Borehole{(msg.analysisResult.borehole_count || msg.analysisResult.boreholes.length) !== 1 ? 's' : ''}
                          </span>
                        </div>
                        {/* Borehole cards grid */}
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                          {msg.analysisResult.boreholes.slice(0, 4).map((bh, i) => (
                            <div key={i} className="bg-[#333] rounded-lg p-2.5">
                              <div className="font-medium text-blue-400 text-sm">{bh.borehole_id}</div>
                              <div className="text-xs text-gray-400 mt-1 space-y-0.5">
                                <div>Depth: {bh.total_depth}m</div>
                                {((bh.soil_layers?.length ?? 0) > 0 || (bh.spt_records?.length ?? 0) > 0) && (
                                  <div>
                                    {(bh.soil_layers?.length ?? 0) > 0 && `Layers: ${bh.soil_layers!.length}`}
                                    {(bh.soil_layers?.length ?? 0) > 0 && (bh.spt_records?.length ?? 0) > 0 && ' | '}
                                    {(bh.spt_records?.length ?? 0) > 0 && `SPT: ${bh.spt_records!.length}`}
                                  </div>
                                )}
                                {bh.water_level && <div>GWL: {bh.water_level}m</div>}
                              </div>
                            </div>
                          ))}
                        </div>
                        {msg.analysisResult.boreholes.length > 4 && (
                          <p className="text-center text-gray-500 text-xs mt-2">
                            +{msg.analysisResult.boreholes.length - 4} more boreholes
                          </p>
                        )}
                        {/* Action buttons */}
                        {msg.analysisResult.cross_section_available && (
                          <div className="flex gap-2 mt-3">
                            <button
                              onClick={() => {
                                // Open window first (sync) to avoid Safari popup blocker
                                const newWindow = window.open('about:blank', '_blank');
                                fetch(`${getApiBase()}/api/viz/generate/cross-section`, {
                                  method: 'POST',
                                  headers: { 'Content-Type': 'application/json' },
                                })
                                  .then(res => res.json())
                                  .then(data => {
                                    if (data.success && data.cross_section_file) {
                                      if (newWindow) {
                                        newWindow.location.href = `${getApiBase()}${data.cross_section_file}`;
                                      }
                                    } else {
                                      if (newWindow) newWindow.close();
                                      toast({ title: 'Error', description: data.detail || 'Failed to generate cross-section', variant: 'destructive' });
                                    }
                                  })
                                  .catch(() => {
                                    if (newWindow) newWindow.close();
                                    toast({ title: 'Error', description: 'Failed to generate cross-section', variant: 'destructive' });
                                  });
                              }}
                              className="px-3 py-1.5 bg-[#333] hover:bg-[#444] text-gray-300 text-xs rounded-xl transition-colors"
                            >
                              Generate Cross-Section
                            </button>
                            <button
                              onClick={() => {
                                // Open window first (sync) to avoid Safari popup blocker
                                const newWindow = window.open('about:blank', '_blank');
                                fetch(`${getApiBase()}/api/viz/generate/3d`, {
                                  method: 'POST',
                                  headers: { 'Content-Type': 'application/json' },
                                })
                                  .then(res => res.json())
                                  .then(data => {
                                    if (data.success && data.fullpage_url) {
                                      if (newWindow) {
                                        newWindow.location.href = data.fullpage_url;
                                      }
                                    } else {
                                      if (newWindow) newWindow.close();
                                      toast({ title: 'Error', description: data.detail || 'Failed to generate 3D visualization', variant: 'destructive' });
                                    }
                                  })
                                  .catch(() => {
                                    if (newWindow) newWindow.close();
                                    toast({ title: 'Error', description: 'Failed to generate 3D visualization', variant: 'destructive' });
                                  });
                              }}
                              className="px-3 py-1.5 bg-[#333] hover:bg-[#444] text-gray-300 text-xs rounded-xl transition-colors"
                            >
                              Generate 3D View
                            </button>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Template File Download */}
                    {msg.analysisResult?.template_file && (
                      <div className="mt-4 pt-4 border-t border-gray-600/30">
                        <div className="flex gap-2">
                          <a
                            href={`${getApiBase()}${msg.analysisResult.template_file}`}
                            download
                            className="flex items-center gap-2 px-3 py-1.5 bg-green-600 hover:bg-green-700 rounded-xl text-sm font-medium transition-colors"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                            </svg>
                            Lab Template
                          </a>
                        </div>
                      </div>
                    )}

                    {/* Generated Files Download (domain tool outputs) */}
                    {(msg.analysisResult?.openground_zip || msg.analysisResult?.soil_analysis) && (
                      <div className="mt-4 pt-4 border-t border-gray-600/30">
                        <div className="flex flex-wrap gap-2">
                          {msg.analysisResult.openground_zip && (
                            <a
                              href={`${getApiBase()}${msg.analysisResult.openground_zip}`}
                              download
                              className="flex items-center gap-2 px-3 py-1.5 bg-amber-600 hover:bg-amber-700 rounded-xl text-sm font-medium transition-colors"
                            >
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                              </svg>
                              OpenGround ZIP
                            </a>
                          )}
                          {msg.analysisResult.soil_analysis && (
                            <a
                              href={`${getApiBase()}${msg.analysisResult.soil_analysis}`}
                              download
                              className="flex items-center gap-2 px-3 py-1.5 bg-green-600 hover:bg-green-700 rounded-xl text-sm font-medium transition-colors"
                            >
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                              </svg>
                              Analysis Report
                            </a>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Cross-Section Download (LoggView) */}
                    {msg.analysisResult?.cross_section_file && (
                      <div className="mt-4 pt-4 border-t border-gray-600/30">
                        <div className="flex gap-2">
                          <a
                            href={`${getApiBase()}${msg.analysisResult.cross_section_file}`}
                            download
                            className="flex items-center gap-2 px-3 py-1.5 bg-cyan-600 hover:bg-cyan-700 rounded-xl text-sm font-medium transition-colors"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                            </svg>
                            Cross-Section
                          </a>
                        </div>
                      </div>
                    )}

                    {/* 3D Visualization (LoggView) */}
                    {msg.analysisResult?.visualization_3d && (
                      <div className="mt-4 pt-4 border-t border-gray-600/30">
                        <div className="flex flex-col gap-3">
                          {/* Preview image if available */}
                          {msg.analysisResult.visualization_3d.preview_url && (
                            <div className="relative rounded-xl overflow-hidden border border-[#3a3a3a] bg-[#2a2a2a]">
                              <img
                                src={`${getApiBase()}${msg.analysisResult.visualization_3d.preview_url}`}
                                alt="3D Preview"
                                className="w-full h-48 object-contain"
                              />
                              <div className="absolute inset-0 flex items-center justify-center bg-black/30 opacity-0 hover:opacity-100 transition-opacity">
                                <a
                                  href={msg.analysisResult.visualization_3d.fullpage_url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm font-medium transition-colors"
                                >
                                  Open Interactive View
                                </a>
                              </div>
                            </div>
                          )}
                          {/* Action buttons */}
                          <div className="flex gap-2">
                            <a
                              href={msg.analysisResult.visualization_3d.fullpage_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm font-medium transition-colors"
                            >
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                              </svg>
                              3D Viewer
                            </a>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Map Result (domain tool, rendered when type === 'geology') */}
                    {msg.analysisResult?.type === 'geology' && (
                      <div className="mt-4 pt-4 border-t border-gray-600/30">
                        <div className="flex items-center gap-2 mb-3">
                          <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                          </svg>
                          <span className="text-sm font-medium text-gray-300">
                            Map Result - {msg.analysisResult.location}
                          </span>
                        </div>
                        {/* Unit info */}
                        <div className="mb-3 p-3 bg-[#2a2a2a] rounded-lg border border-[#3a3a3a]">
                          <div className="flex items-center gap-2">
                            <span className="px-2 py-0.5 bg-amber-600/20 text-amber-300 text-xs rounded font-mono">
                              {msg.analysisResult.unit_code}
                            </span>
                            {msg.analysisResult.unit_name && (
                              <span className="text-sm text-gray-300">{msg.analysisResult.unit_name}</span>
                            )}
                          </div>
                          {msg.analysisResult.soil_types && msg.analysisResult.soil_types.length > 0 && (
                            <div className="flex gap-1 mt-2 flex-wrap">
                              {msg.analysisResult.soil_types.map((soil: string, i: number) => (
                                <span key={i} className="px-2 py-0.5 bg-gray-700/50 text-gray-300 text-xs rounded">
                                  {soil}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                        {/* AGS Legend Table - horizontal layout matching AGS map format */}
                        {(msg.analysisResult.lithology || msg.analysisResult.lithogenesis ||
                          msg.analysisResult.morphology || msg.analysisResult.comment ||
                          msg.analysisResult.stratigraphic_unit) && (
                          <div className="mb-3 overflow-x-auto">
                            <table className="w-full text-xs border-collapse">
                              <thead>
                                <tr className="bg-[#2a2a2a]">
                                  {msg.analysisResult.stratigraphic_unit && (
                                    <th className="text-left py-2 px-2 text-amber-300/80 font-medium border-b border-[#3a3a3a] whitespace-nowrap">Stratigraphic Unit</th>
                                  )}
                                  {msg.analysisResult.lithology && (
                                    <th className="text-left py-2 px-2 text-amber-300/80 font-medium border-b border-[#3a3a3a] whitespace-nowrap">Lithology</th>
                                  )}
                                  {msg.analysisResult.lithogenesis && (
                                    <th className="text-left py-2 px-2 text-amber-300/80 font-medium border-b border-[#3a3a3a] whitespace-nowrap">Lithogenesis</th>
                                  )}
                                  {msg.analysisResult.morphology && (
                                    <th className="text-left py-2 px-2 text-amber-300/80 font-medium border-b border-[#3a3a3a] whitespace-nowrap">Morphology</th>
                                  )}
                                  {msg.analysisResult.comment && (
                                    <th className="text-left py-2 px-2 text-amber-300/80 font-medium border-b border-[#3a3a3a] whitespace-nowrap">Comment</th>
                                  )}
                                </tr>
                              </thead>
                              <tbody className="text-gray-300">
                                <tr>
                                  {msg.analysisResult.stratigraphic_unit && (
                                    <td className="py-2 px-2 align-top">{msg.analysisResult.stratigraphic_unit}</td>
                                  )}
                                  {msg.analysisResult.lithology && (
                                    <td className="py-2 px-2 align-top">{msg.analysisResult.lithology}</td>
                                  )}
                                  {msg.analysisResult.lithogenesis && (
                                    <td className="py-2 px-2 align-top">{msg.analysisResult.lithogenesis}</td>
                                  )}
                                  {msg.analysisResult.morphology && (
                                    <td className="py-2 px-2 align-top">{msg.analysisResult.morphology}</td>
                                  )}
                                  {msg.analysisResult.comment && (
                                    <td className="py-2 px-2 align-top">{msg.analysisResult.comment}</td>
                                  )}
                                </tr>
                              </tbody>
                            </table>
                          </div>
                        )}
                        {/* Map snippet image (only if available) */}
                        {msg.analysisResult.map_segment && (
                          <div className="relative rounded-xl overflow-hidden border border-[#3a3a3a] bg-[#2a2a2a]">
                            <img
                              src={`${getApiBase()}${msg.analysisResult.map_segment}`}
                              alt="Map Snippet"
                              className="w-full max-h-80 object-contain"
                            />
                          </div>
                        )}
                        {/* Citation */}
                        {msg.analysisResult.citation && (
                          <div className="mt-2 p-2 bg-[#2a2a2a] rounded border border-[#3a3a3a]">
                            <p className="text-xs text-gray-400 font-mono select-all cursor-text">
                              {msg.analysisResult.citation}
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Flag button for assistant messages (appears on hover, at end of message) */}
                    {msg.role === 'assistant' && !isCurrentlyStreaming && msg.content && (
                      <button
                        onClick={() => flagMessage(msg.id, prevUserContent, msg.content)}
                        title={flaggedMessages.has(msg.id) ? "Flagged for review" : "Flag this response for human review"}
                        className={`inline-flex items-center gap-1 mt-2 text-xs opacity-0 group-hover:opacity-100 transition-opacity focus:opacity-100 ${
                          flaggedMessages.has(msg.id)
                            ? 'text-amber-400'
                            : 'text-gray-500 hover:text-amber-400'
                        }`}
                      >
                        <svg className="w-3 h-3" fill={flaggedMessages.has(msg.id) ? "currentColor" : "none"} stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 21v-4m0 0V5a2 2 0 012-2h6.5l1 1H21l-3 6 3 6h-8.5l-1-1H5a2 2 0 00-2 2zm9-13.5V9" />
                        </svg>
                        {flaggedMessages.has(msg.id) ? 'Flagged' : 'Flag'}
                      </button>
                    )}
                  </div>
                </div>
              );
              })}
              <div ref={messagesEndRef} />
            </div>
          )}
        </main>

        {/* Input Bar */}
        <footer className="border-t border-[#2a2a2a] p-4 bg-[#212121]">
          <div className="max-w-4xl mx-auto">
            {/* Uploaded Files Indicator */}
            {uploadedFiles.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-3">
                {uploadedFiles.map((file, idx) => {
                  const styles = getFileTypeStyles(file.name);
                  const isParsing = file.status === 'uploading' || file.status === 'parsing';
                  return (
                    <div key={file.id} className={`flex items-center gap-2 px-3 py-1.5 ${styles.bg} border ${styles.border} rounded-full text-sm`}>
                      {isParsing ? (
                        <svg className={`w-4 h-4 ${styles.text} animate-spin flex-shrink-0`} fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                      ) : (
                        <svg className={`w-4 h-4 ${styles.text} flex-shrink-0`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                      )}
                      <span className={`${styles.text} truncate max-w-[150px]`}>{file.name}</span>
                      {isParsing && <span className="text-gray-500 text-xs">Parsing...</span>}
                      {!isParsing && file.samples > 0 && <span className={`${styles.text} opacity-60 text-xs`}>({file.samples})</span>}
                      <button
                        onClick={() => setUploadedFiles(prev => prev.filter((_, i) => i !== idx))}
                        className={`p-0.5 hover:${styles.bg} rounded-full`}
                      >
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  );
                })}
                {uploadedFiles.length > 1 && (
                  <button
                    onClick={async () => {
                      try {
                        await fetch(`${getApiBase()}/api/files`, { method: 'DELETE' });
                      } catch (e) {}
                      setUploadedFiles([]);
                      clearFileManifest();
                    }}
                    className="px-2 py-1.5 text-xs text-gray-400 hover:text-white hover:bg-[#333] rounded-full"
                  >
                    Clear all
                  </button>
                )}
              </div>
            )}

            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept="*/*"
              multiple
              onChange={(e) => {
                const files = e.target.files;
                if (files) {
                  Array.from(files).forEach(file => handleFileUpload(file));
                }
              }}
              className="hidden"
            />

            {/* Text Input Box with all controls inside */}
            <div className="bg-[#2a2a2a] border border-[#3a3a3a] rounded-xl overflow-hidden transition-colors">
              {/* Text area */}
              <textarea
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  // Auto-resize with viewport-aware max height
                  const maxHeight = Math.min(200, window.innerHeight * 0.25);
                  e.target.style.height = 'auto';
                  e.target.style.height = Math.max(72, Math.min(e.target.scrollHeight, maxHeight)) + 'px';
                }}
                onKeyDown={handleKeyDown}
                onPaste={(e) => {
                  // Handle image paste from clipboard
                  const items = e.clipboardData?.items;
                  if (items) {
                    for (const item of Array.from(items)) {
                      if (item.type.startsWith('image/')) {
                        e.preventDefault();
                        const file = item.getAsFile();
                        if (file) {
                          // Create a proper filename for the pasted image
                          const ext = item.type.split('/')[1] || 'png';
                          const pastedFile = new File([file], `Screenshot ${new Date().toISOString().slice(0,19).replace(/[T:]/g, '-')}.${ext}`, { type: item.type });
                          handleFileUpload(pastedFile);
                        }
                      }
                    }
                  }
                }}
                placeholder=""
                aria-label="Message input"
                rows={3}
                className="w-full min-h-[72px] max-h-[25vh] md:max-h-[200px] bg-transparent px-4 py-3
                           text-white placeholder-gray-500 resize-none overflow-y-auto
                           focus:outline-none"
                disabled={isStreaming}
              />

              {/* Bottom toolbar inside the box */}
              <div className="flex items-center justify-between px-3 py-2 border-t border-[#3a3a3a]">
                {/* Left side - Mode indicators */}
                <div className="flex items-center gap-2 text-xs">
                  {searchMode && (
                    <span className={`flex items-center gap-1.5 px-3 py-1 rounded-full ${deepSearchMode ? 'bg-cyan-600/20 text-cyan-400' : 'bg-blue-600/20 text-blue-400'}`}>
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        {deepSearchMode ? (
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                        ) : (
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                        )}
                      </svg>
                      {deepSearchMode ? 'Deep web search' : 'Web search (click again for deep web search)'}
                    </span>
                  )}
                  {thinkMode && (
                    <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-purple-600/20 text-purple-400">
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                      Think mode
                    </span>
                  )}
                  {calculateMode && (
                    <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-green-600/20 text-green-400">
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                      </svg>
                      Calculate mode
                    </span>
                  )}
                </div>

                {/* Right side - Toggle buttons, attach, and send */}
                <div className="flex items-center gap-1">
                  {/* Web/Deep Search Toggle */}
                  <button
                    onClick={() => {
                      if (!searchMode) {
                        setSearchMode(true);
                        setDeepSearchMode(false);
                      } else if (!deepSearchMode) {
                        setDeepSearchMode(true);
                      } else {
                        setSearchMode(false);
                        setDeepSearchMode(false);
                      }
                    }}
                    aria-label={!searchMode ? "Enable web search" : (!deepSearchMode ? "Enable deep search" : "Disable search")}
                    aria-pressed={searchMode}
                    title={!searchMode ? "Enable web search" : (!deepSearchMode ? "Click for deep search" : "Disable search")}
                    className={`w-9 h-9 flex items-center justify-center rounded-xl transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                      searchMode ? (deepSearchMode ? 'text-cyan-400' : 'text-blue-400') : 'text-gray-500 hover:text-white'
                    }`}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      {deepSearchMode ? (
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                      ) : (
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                      )}
                    </svg>
                  </button>

                  {/* Think Mode Toggle */}
                  <button
                    onClick={() => { setThinkMode(!thinkMode); if (!thinkMode) setCalculateMode(false); }}
                    aria-label="Toggle extended reasoning"
                    aria-pressed={thinkMode}
                    title="Think mode: Detailed analysis and reasoning"
                    className={`w-9 h-9 flex items-center justify-center rounded-xl transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-purple-500 ${
                      thinkMode ? 'text-purple-400' : 'text-gray-500 hover:text-white'
                    }`}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </button>

                  {/* Calculate Mode Toggle */}
                  <button
                    onClick={() => { setCalculateMode(!calculateMode); if (!calculateMode) setThinkMode(false); }}
                    aria-label="Toggle calculation mode"
                    aria-pressed={calculateMode}
                    title="Calculate mode: Step-by-step math derivations (qwq)"
                    className={`w-9 h-9 flex items-center justify-center rounded-xl transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-green-500 ${
                      calculateMode ? 'text-green-400' : 'text-gray-500 hover:text-white'
                    }`}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                    </svg>
                  </button>

                  {/* Phii Enhancement Toggle */}
                  <button
                    onClick={() => setPhiiEnabled(!phiiEnabled)}
                    aria-label={phiiEnabled ? "Phii active: Adaptive responses. Click to disable." : "Phii disabled: Standard mode. Click to enable."}
                    aria-pressed={phiiEnabled}
                    title={phiiEnabled
                      ? "Phii active: Adaptive responses matching your style. Click to disable."
                      : "Phii disabled: Standard mode. Click to enable."
                    }
                    className={`w-9 h-9 flex items-center justify-center rounded-xl transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 ${
                      phiiEnabled ? 'text-amber-400' : 'text-gray-500 hover:text-white'
                    }`}
                  >
                    <span className="text-base font-normal leading-none -translate-y-[3px]">φ</span>
                  </button>

                  <div className="w-px h-6 bg-[#3a3a3a] mx-1" />

                  {/* Attach Button */}
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    aria-label="Attach file"
                    className="w-9 h-9 flex items-center justify-center text-gray-500 hover:text-white rounded-xl transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                    </svg>
                  </button>

                  {/* Send Button */}
                  <button
                    onClick={sendMessage}
                    disabled={!input.trim() || isStreaming || !isConnected}
                    aria-label="Send message"
                    className={`w-9 h-9 flex items-center justify-center rounded-xl transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                                 input.trim() && !isStreaming && isConnected
                                   ? 'text-blue-400 hover:text-blue-300'
                                   : 'text-gray-600'
                               }`}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
