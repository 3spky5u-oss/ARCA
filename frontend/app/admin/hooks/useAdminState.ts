'use client';

import { useState, useEffect, useCallback } from 'react';
import { getApiBase } from '@/lib/api';
import { useDomain, useDomainRefresh } from '@/lib/domain-context';
import type {
  SystemStatus,
  ConfigValues,
  LogEntry,
  Session,
  KnowledgeStats,
  TopicFile,
  UntrackedFile,
  TopicWithStatus,
  RerankerSettings,
  ModelsData,
  PullStatus,
  SearchTestResult,
  GuidelineStats,
  GuidelineSearchResult,
  GuidelineComparisonResult,
  ExceedeeDBStats,
  ExceedeeRecord,
  ReprocessJobStatus,
  ReprocessOptions,
  AvailableDomain,
  AdminToolsData,
  ToolScaffoldPayload,
} from '../components';

export type MainTab = 'dashboard' | 'config' | 'domain' | 'knowledge' | 'benchmark' | 'compliance' | 'intelligence' | 'tools' | 'diagnostics';

export function useAdminState() {
  const domain = useDomain();
  const refreshDomain = useDomainRefresh();

  // Auth state
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [token, setToken] = useState('');
  const [currentUsername, setCurrentUsername] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authError, setAuthError] = useState('');
  const [setupRequired, setSetupRequired] = useState(false);
  const [authChecked, setAuthChecked] = useState(false);

  // User management state
  const [usersList, setUsersList] = useState<Array<{ id: number; username: string; role: string; created_at: string }>>([]);
  const [newUserUsername, setNewUserUsername] = useState('');
  const [newUserPassword, setNewUserPassword] = useState('');
  const [userMgmtMessage, setUserMgmtMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Data state
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [config, setConfig] = useState<ConfigValues | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);

  // Hardware profile state
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [hardwareProfile, setHardwareProfile] = useState<Record<string, any> | null>(null);

  // Knowledge state
  const [knowledgeStats, setKnowledgeStats] = useState<KnowledgeStats | null>(null);
  const [expandedTopics, setExpandedTopics] = useState<Set<string>>(new Set());
  const [topicFiles, setTopicFiles] = useState<Record<string, TopicFile[]>>({});
  const [topicUntracked, setTopicUntracked] = useState<Record<string, UntrackedFile[]>>({});
  const [availableTopics, setAvailableTopics] = useState<string[]>([]);
  const [searchTestResult, setSearchTestResult] = useState<SearchTestResult | null>(null);
  const [fileExtractorSelection, setFileExtractorSelection] = useState<Record<string, string>>({});
  const [topicsWithStatus, setTopicsWithStatus] = useState<TopicWithStatus[]>([]);
  const [rerankerSettings, setRerankerSettings] = useState<RerankerSettings | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [profiles, setProfiles] = useState<any[]>([]);
  const [processingFiles, setProcessingFiles] = useState<Record<string, string>>({}); // path -> status message

  // UI state
  const [activeTab, setActiveTab] = useState<MainTab>('dashboard');
  const [configSubTab, setConfigSubTab] = useState<'general' | 'models' | 'connections'>('general');
  const [knowledgeSubTab, setKnowledgeSubTab] = useState<'documents' | 'retrieval' | 'collections' | 'graph'>('documents');
  const [complianceSubTab, setComplianceSubTab] = useState<'guidelines' | 'records'>('guidelines');
  const [intelligenceSubTab, setIntelligenceSubTab] = useState<'personality'>('personality');
  const [diagnosticsSubTab, setDiagnosticsSubTab] = useState<'extraction' | 'logs' | 'sessions'>('extraction');
  const [loading, setLoading] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [firstBootDismissed, setFirstBootDismissed] = useState(() => {
    if (typeof window !== 'undefined') {
      return sessionStorage.getItem('arca_first_boot_dismissed') === 'true';
    }
    return false;
  });

  // Auto-dismiss toast after 3s
  useEffect(() => {
    if (!message) return;
    const timer = setTimeout(() => setMessage(null), 3000);
    return () => clearTimeout(timer);
  }, [message]);

  // Auto-dismiss user mgmt message after 3s
  useEffect(() => {
    if (!userMgmtMessage) return;
    const timer = setTimeout(() => setUserMgmtMessage(null), 3000);
    return () => clearTimeout(timer);
  }, [userMgmtMessage]);

  // Models state
  const [modelsData, setModelsData] = useState<ModelsData | null>(null);
  const [pullModelName, setPullModelName] = useState('');
  const [pullStatus, setPullStatus] = useState<PullStatus | null>(null);
  const [pullingModel, setPullingModel] = useState<string | null>(null);

  // Tools state
  const [toolsData, setToolsData] = useState<AdminToolsData | null>(null);

  // Phii/Personality state
  const [phiiStats, setPhiiStats] = useState<{
    feedback: {
      counts: Record<string, number>;
      unresolved_flags: number;
      recent_24h: number;
      total: number;
    };
    settings: {
      phii_energy_matching: boolean;
      phii_specialty_detection: boolean;
      phii_implicit_feedback: boolean;
    };
  } | null>(null);
  const [phiiFlags, setPhiiFlags] = useState<Array<{
    id: number;
    timestamp: string;
    session_id: string;
    message_id: string;
    feedback_type: string;
    user_message: string;
    assistant_response: string;
    tools_used: string[];
    resolved: boolean;
    admin_notes: string;
  }>>([]);

  // Corrections state
  const [corrections, setCorrections] = useState<Array<{
    id: number;
    timestamp: string;
    wrong_behavior: string;
    right_behavior: string;
    context_keywords: string[];
    confidence: number;
    times_applied: number;
  }>>([]);
  const [correctionsStats, setCorrectionsStats] = useState<{
    active_count: number;
    total_applied: number;
    recent_7d: number;
  }>({ active_count: 0, total_applied: 0, recent_7d: 0 });

  // Pattern stats state
  const [patternStats, setPatternStats] = useState<{
    pattern_count: number;
    action_count: number;
    top_patterns: Array<{ from: string; to: string; count: number }>;
  }>({ pattern_count: 0, action_count: 0, top_patterns: [] });

  // Extraction tester state
  const [testFile, setTestFile] = useState<File | null>(null);
  const [testExtractor, setTestExtractor] = useState('');
  const [extractionResult, setExtractionResult] = useState<Record<string, unknown> | null>(null);

  // Phii debug state
  const [phiiDebugMessage, setPhiiDebugMessage] = useState('');
  const [phiiDebugResult, setPhiiDebugResult] = useState<{
    corrections_applied: Array<{ wrong_behavior: string; right_behavior: string }>;
    expertise_level: string;
    energy_analysis: { brevity: string; formality: string; urgency: string };
    specialty: string;
  } | null>(null);
  const [phiiDebugLoading, setPhiiDebugLoading] = useState(false);

  // Domain pack state
  const [availableDomains, setAvailableDomains] = useState<AvailableDomain[]>([]);
  const [domainSwitching, setDomainSwitching] = useState(false);

  // Config edit state
  const [editedConfig, setEditedConfig] = useState<Partial<ConfigValues>>({});

  // Knowledge ingest state
  const [ingestFile, setIngestFile] = useState<File | null>(null);
  const [ingestTopic, setIngestTopic] = useState('');
  const [newTopicName, setNewTopicName] = useState('');
  const [showAdvancedIngest, setShowAdvancedIngest] = useState(false);
  const [ingestChunkSize, setIngestChunkSize] = useState(800);
  const [ingestChunkOverlap, setIngestChunkOverlap] = useState(150);

  // Search test state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchTopics, setSearchTopics] = useState<string[]>([]);
  const [searchTopK, setSearchTopK] = useState(5);

  // Compliance (Exceedee) state
  const [guidelineStats, setGuidelineStats] = useState<GuidelineStats | null>(null);
  const [expandedTables, setExpandedTables] = useState<Set<string>>(new Set());
  const [guidelineSearch, setGuidelineSearch] = useState('');
  const [guidelineSearchResult, setGuidelineSearchResult] = useState<GuidelineSearchResult | null>(null);
  const [comparisonFile, setComparisonFile] = useState<File | null>(null);
  const [comparisonResult, setComparisonResult] = useState<GuidelineComparisonResult | null>(null);
  const [isComparing, setIsComparing] = useState(false);

  // ExceedeeDB state
  const [exceedeeDBStats, setExceedeeDBStats] = useState<ExceedeeDBStats | null>(null);
  const [pendingRecords, setPendingRecords] = useState<ExceedeeRecord[]>([]);
  const [browseRecords, setBrowseRecords] = useState<ExceedeeRecord[]>([]);
  const [dbProjectFilter, setDbProjectFilter] = useState('');
  const [dbStatusFilter, setDbStatusFilter] = useState<string>('');

  // Reprocess state
  const [reprocessJobId, setReprocessJobId] = useState<string | null>(null);
  const [reprocessStatus, setReprocessStatus] = useState<ReprocessJobStatus | null>(null);
  const [showReprocessConfirm, setShowReprocessConfirm] = useState(false);
  const [reprocessMode, setReprocessMode] = useState<'knowledge_base' | 'session'>('knowledge_base');
  const [reprocessOptions, setReprocessOptions] = useState<ReprocessOptions>({
    purge_qdrant: true,
    clear_bm25: true,
    clear_neo4j: true,
    build_raptor: true,
    build_graph: true,
    skip_vision: false,
  });

  // =========================================================================
  // AUTH CHECK
  // =========================================================================

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const res = await fetch(`${getApiBase()}/api/admin/auth-status`);
        const data = await res.json();
        setSetupRequired(data.setup_required);
      } catch {
        // Backend unreachable - will show login by default
      }

      const stored = sessionStorage.getItem('arca_admin_token') || localStorage.getItem('arca_auth_token');
      const storedUsername = sessionStorage.getItem('arca_admin_username') || localStorage.getItem('arca_auth_username');
      if (stored) {
        try {
          const verifyRes = await fetch(`${getApiBase()}/api/admin/auth/verify`, {
            headers: { 'Authorization': `Bearer ${stored}` },
          });
          const verifyData = await verifyRes.json();
          if (verifyData.valid) {
            setToken(stored);
            setCurrentUsername(verifyData.username || storedUsername || 'admin');
            setIsAuthenticated(true);
          } else {
            sessionStorage.removeItem('arca_admin_token');
            sessionStorage.removeItem('arca_admin_username');
          }
        } catch {
          sessionStorage.removeItem('arca_admin_token');
          sessionStorage.removeItem('arca_admin_username');
        }
      }
      setAuthChecked(true);
    };
    checkAuth();
  }, []);

  // Keyboard shortcut handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'D') {
        e.preventDefault();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // =========================================================================
  // API HELPERS
  // =========================================================================

  const apiCall = useCallback(async (endpoint: string, options: RequestInit = {}) => {
    const headers: Record<string, string> = {
      'Authorization': `Bearer ${token}`,
      ...options.headers as Record<string, string>,
    };

    const response = await fetch(`${getApiBase()}${endpoint}`, {
      ...options,
      headers,
    });

    if (response.status === 401) {
      setIsAuthenticated(false);
      setToken('');
      setCurrentUsername('');
      sessionStorage.removeItem('arca_admin_token');
      sessionStorage.removeItem('arca_admin_username');
      throw new Error('Session expired');
    }

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || 'API request failed');
    }

    return response.json();
  }, [token]);

  // =========================================================================
  // DATA FETCHING
  // =========================================================================

  const fetchStatus = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/status');
      setStatus(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch status: ${e}` });
    }
  }, [apiCall]);

  const fetchHardware = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/hardware');
      setHardwareProfile(data);
    } catch {
      // Non-critical
    }
  }, [apiCall]);

  const fetchConfig = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/config');
      setConfig(data.config);
      setEditedConfig({});
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch config: ${e}` });
    }
  }, [apiCall]);

  const fetchLogs = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/logs?limit=200');
      setLogs(data.logs);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch logs: ${e}` });
    }
  }, [apiCall]);

  const fetchSessions = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/sessions');
      setSessions(data.sessions);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch sessions: ${e}` });
    }
  }, [apiCall]);

  const fetchModels = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/models');
      setModelsData(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch models: ${e}` });
    }
  }, [apiCall]);

  const fetchTools = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/tools');
      setToolsData(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch tools: ${e}` });
    }
  }, [apiCall]);

  const handlePullModel = useCallback(async () => {
    if (!pullModelName.trim()) {
      setMessage({ type: 'error', text: 'Enter a model name' });
      return;
    }
    try {
      setPullingModel(pullModelName);
      setPullStatus({ status: 'starting' });
      await apiCall('/api/admin/models/pull', {
        method: 'POST',
        body: JSON.stringify({ name: pullModelName }),
      });
      const pollStatus = async () => {
        try {
          const status = await apiCall(`/api/admin/models/pull/${encodeURIComponent(pullModelName)}/status`);
          setPullStatus(status);
          if (status.done) {
            setPullingModel(null);
            setPullModelName('');
            setMessage({ type: 'success', text: `Successfully pulled ${pullModelName}` });
            fetchModels();
          } else if (status.error) {
            setPullingModel(null);
            setMessage({ type: 'error', text: status.error });
          } else {
            setTimeout(pollStatus, 1000);
          }
        } catch {
          setTimeout(pollStatus, 1000);
        }
      };
      setTimeout(pollStatus, 1000);
    } catch (e) {
      setPullingModel(null);
      setPullStatus(null);
      setMessage({ type: 'error', text: `Failed to pull model: ${e}` });
    }
  }, [apiCall, pullModelName, fetchModels]);

  const handleDeleteModel = useCallback(async (modelName: string) => {
    if (!confirm(`Delete model ${modelName}? This cannot be undone.`)) return;
    try {
      await apiCall(`/api/admin/models/${encodeURIComponent(modelName)}`, { method: 'DELETE' });
      setMessage({ type: 'success', text: `Deleted ${modelName}` });
      fetchModels();
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to delete model: ${e}` });
    }
  }, [apiCall, fetchModels]);

  const handleAssignModel = useCallback(async (slot: string, modelName: string) => {
    try {
      setMessage({ type: 'success', text: `Swapping ${slot} model...` });
      const result = await apiCall('/api/admin/models/assign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slot, model: modelName }),
      });
      const swapNote = result.server_swapped ? ' (server restarted)' : ' (config updated, restart to apply)';
      setMessage({ type: 'success', text: `${slot}: ${modelName}${swapNote}` });
      fetchModels();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setMessage({ type: 'error', text: `Failed to assign model: ${msg}` });
    }
  }, [apiCall, fetchModels]);

  const handleReloadTools = useCallback(async () => {
    try {
      await apiCall('/api/admin/tools/reload', { method: 'POST' });
      await fetchTools();
      setMessage({ type: 'success', text: 'Tool registry reloaded' });
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to reload tools: ${e}` });
    }
  }, [apiCall, fetchTools]);

  const handleScaffoldTool = useCallback(async (payload: ToolScaffoldPayload) => {
    await apiCall('/api/admin/tools/scaffold', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    await fetchTools();
    setMessage({ type: 'success', text: `Scaffolded tool ${String(payload.name || '')}` });
  }, [apiCall, fetchTools]);

  const handleUpdateCustomTool = useCallback(async (toolName: string, updates: Record<string, unknown>) => {
    await apiCall(`/api/admin/tools/custom/${encodeURIComponent(toolName)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    await fetchTools();
    setMessage({ type: 'success', text: `Updated custom tool ${toolName}` });
  }, [apiCall, fetchTools]);

  const fetchKnowledgeStats = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/knowledge/stats');
      setKnowledgeStats(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch knowledge stats: ${e}` });
    }
  }, [apiCall]);

  const fetchTopicFiles = useCallback(async (topic: string) => {
    try {
      const data = await apiCall(`/api/admin/knowledge/topic/${topic}`);
      setTopicFiles(prev => ({ ...prev, [topic]: data.files }));
      setTopicUntracked(prev => ({ ...prev, [topic]: data.untracked || [] }));
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch topic files: ${e}` });
    }
  }, [apiCall]);

  const fetchAvailableTopics = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/knowledge/topics');
      if (data.topics && data.topics.length > 0) {
        if (typeof data.topics[0] === 'object') {
          setTopicsWithStatus(data.topics);
          const topicNames = data.topics.map((t: TopicWithStatus) => t.name);
          setAvailableTopics(topicNames);
          if (topicNames.length > 0 && !ingestTopic) {
            setIngestTopic(topicNames[0]);
          }
        } else {
          setAvailableTopics(data.topics);
          if (data.topics.length > 0 && !ingestTopic) {
            setIngestTopic(data.topics[0]);
          }
        }
      }
    } catch {
      // Silent fail
    }
  }, [apiCall, ingestTopic]);

  const fetchRerankerSettings = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/knowledge/reranker');
      setRerankerSettings(data);
    } catch {
      // Silent fail
    }
    try {
      const profileData = await apiCall('/api/admin/knowledge/profiles');
      setProfiles(profileData.profiles || []);
    } catch (err) {
      console.error('Failed to fetch profiles:', err);
    }
  }, [apiCall]);

  const fetchGuidelineStats = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/exceedee/stats');
      setGuidelineStats(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch guideline stats: ${e}` });
    }
  }, [apiCall]);

  const fetchPhiiStats = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/phii/stats');
      setPhiiStats(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch Phii stats: ${e}` });
    }
  }, [apiCall]);

  const fetchPhiiFlags = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/phii/flags');
      setPhiiFlags(data.flags || []);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch Phii flags: ${e}` });
    }
  }, [apiCall]);

  const fetchCorrections = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/phii/corrections');
      if (data.success) {
        setCorrections(data.corrections || []);
        setCorrectionsStats(data.stats || { active_count: 0, total_applied: 0, recent_7d: 0 });
      }
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch corrections: ${e}` });
    }
  }, [apiCall]);

  const fetchPatterns = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/phii/patterns');
      if (data.success) {
        setPatternStats(data.patterns || { pattern_count: 0, action_count: 0, top_patterns: [] });
      }
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch patterns: ${e}` });
    }
  }, [apiCall]);

  const fetchExceedeeDBStats = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/exceedee/db/stats');
      setExceedeeDBStats(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch ExceedeeDB stats: ${e}` });
    }
  }, [apiCall]);

  const fetchPendingRecords = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/exceedee/db/pending');
      setPendingRecords(data.records || []);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to fetch pending records: ${e}` });
    }
  }, [apiCall]);

  const fetchBrowseRecords = useCallback(async () => {
    try {
      let url = '/api/admin/exceedee/db/browse?limit=50';
      if (dbProjectFilter) url += `&project=${encodeURIComponent(dbProjectFilter)}`;
      if (dbStatusFilter) url += `&status=${encodeURIComponent(dbStatusFilter)}`;
      const data = await apiCall(url);
      setBrowseRecords(data.records || []);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to browse records: ${e}` });
    }
  }, [apiCall, dbProjectFilter, dbStatusFilter]);

  const fetchDomains = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/domains');
      setAvailableDomains(data.domains || []);
    } catch {
      // Silent fail
    }
  }, [apiCall]);

  const handleDomainSwitch = useCallback(async (name: string) => {
    if (name === domain.name || domainSwitching) return;
    setDomainSwitching(true);
    try {
      const data = await apiCall('/api/admin/domains/activate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ domain: name }),
      });
      refreshDomain();
      setMessage({
        type: 'success',
        text: data.restart_needed
          ? `Switched to ${data.display_name} (${data.tools} tools). Restart needed for route changes.`
          : `Switched to ${data.display_name} (${data.tools} tools)`,
      });
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to switch domain: ${e}` });
    } finally {
      setDomainSwitching(false);
    }
  }, [apiCall, domain.name, domainSwitching, refreshDomain]);

  const fetchUsers = useCallback(async () => {
    try {
      const data = await apiCall('/api/admin/users');
      setUsersList(data.users || []);
    } catch {
      // Silently fail
    }
  }, [apiCall]);

  // Fetch domains when authenticated
  useEffect(() => {
    if (isAuthenticated) fetchDomains();
  }, [isAuthenticated, fetchDomains]);

  // Load data when authenticated and tab changes
  useEffect(() => {
    if (!isAuthenticated) return;

    const loadData = async () => {
      setLoading(true);
      try {
        if (activeTab === 'dashboard') { await fetchStatus(); await fetchHardware(); }
        else if (activeTab === 'config') { await fetchConfig(); await fetchModels(); await fetchUsers(); }
        else if (activeTab === 'knowledge') {
          await fetchKnowledgeStats();
          await fetchAvailableTopics();
          await fetchRerankerSettings();
        } else if (activeTab === 'compliance') {
          await fetchGuidelineStats();
          await fetchExceedeeDBStats();
          await fetchPendingRecords();
          await fetchBrowseRecords();
        } else if (activeTab === 'intelligence') {
          await fetchPhiiStats();
          await fetchPhiiFlags();
          await fetchCorrections();
          await fetchPatterns();
        } else if (activeTab === 'tools') {
          await fetchTools();
        } else if (activeTab === 'diagnostics') {
          if (diagnosticsSubTab === 'logs') await fetchLogs();
          else if (diagnosticsSubTab === 'sessions') await fetchSessions();
        }
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [isAuthenticated, activeTab, diagnosticsSubTab, fetchStatus, fetchHardware, fetchConfig, fetchModels, fetchUsers, fetchLogs, fetchSessions, fetchKnowledgeStats, fetchAvailableTopics, fetchRerankerSettings, fetchGuidelineStats, fetchExceedeeDBStats, fetchPendingRecords, fetchBrowseRecords, fetchPhiiStats, fetchPhiiFlags, fetchCorrections, fetchPatterns, fetchTools]);

  // Auto-refresh dashboard every 30 seconds
  useEffect(() => {
    if (!isAuthenticated || activeTab !== 'dashboard') return;

    const interval = setInterval(() => {
      fetchStatus();
    }, 30000);

    return () => clearInterval(interval);
  }, [isAuthenticated, activeTab, fetchStatus]);

  // =========================================================================
  // ACTIONS
  // =========================================================================

  const handleAuth = async () => {
    setAuthError('');
    try {
      const endpoint = setupRequired ? '/api/admin/auth/register' : '/api/admin/auth/login';
      const res = await fetch(`${getApiBase()}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });

      const data = await res.json();

      if (!res.ok) {
        setAuthError(data.detail || 'Authentication failed');
        return;
      }

      setToken(data.token);
      setCurrentUsername(data.username || username);
      sessionStorage.setItem('arca_admin_token', data.token);
      sessionStorage.setItem('arca_admin_username', data.username || username);
      localStorage.setItem('arca_auth_token', data.token);
      localStorage.setItem('arca_auth_username', data.username || username);
      localStorage.setItem('arca_auth_role', data.role || 'admin');
      setIsAuthenticated(true);
      setSetupRequired(false);
      setUsername('');
      setPassword('');
    } catch {
      setAuthError('Could not connect to server');
    }
  };

  const handleLogout = () => {
    sessionStorage.removeItem('arca_admin_token');
    sessionStorage.removeItem('arca_admin_username');
    setIsAuthenticated(false);
    setToken('');
    setCurrentUsername('');
    setUsername('');
    setPassword('');
  };

  const handleCreateUser = async () => {
    if (!newUserUsername || !newUserPassword) return;
    try {
      await apiCall('/api/admin/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: newUserUsername, password: newUserPassword }),
      });
      setUserMgmtMessage({ type: 'success', text: `User '${newUserUsername}' created` });
      setNewUserUsername('');
      setNewUserPassword('');
      fetchUsers();
    } catch (e) {
      setUserMgmtMessage({ type: 'error', text: `${e}` });
    }
  };

  const handleDeleteUser = async (userId: number, deleteUsername: string) => {
    if (!confirm(`Delete user '${deleteUsername}'? This cannot be undone.`)) return;
    try {
      await apiCall(`/api/admin/users/${userId}`, { method: 'DELETE' });
      setUserMgmtMessage({ type: 'success', text: `User '${deleteUsername}' deleted` });
      fetchUsers();
    } catch (e) {
      setUserMgmtMessage({ type: 'error', text: `${e}` });
    }
  };

  const handleSaveConfig = async () => {
    setLoading(true);
    try {
      const data = await apiCall('/api/admin/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editedConfig),
      });
      const pretty = (data.updated as string[]).map((k: string) => k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())).join(', ');
      setMessage({ type: 'success', text: `Updated: ${pretty}` });
      await fetchConfig();
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to save config: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  const handleResetConfig = async () => {
    setLoading(true);
    try {
      await apiCall('/api/admin/config/reset', { method: 'POST' });
      setMessage({ type: 'success', text: 'Config reset to defaults' });
      await fetchConfig();
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to reset config: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  const handleRecalibrate = async () => {
    setLoading(true);
    try {
      const data = await apiCall('/api/admin/recalibrate', { method: 'POST' });
      setMessage({
        type: 'success',
        text: `Recalibration complete. LLM: ${data.results.llm_warmup_ms}ms, RAG: ${data.results.rag_warmup_ms}ms`,
      });
    } catch (e) {
      setMessage({ type: 'error', text: `Recalibration failed: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  const handleTestExtraction = async () => {
    if (!testFile) return;

    setLoading(true);
    setExtractionResult(null);
    try {
      const formData = new FormData();
      formData.append('file', testFile);

      let url = '/api/admin/test-extraction';
      if (testExtractor) {
        url += `?extractor=${testExtractor}`;
      }

      const data = await apiCall(url, {
        method: 'POST',
        body: formData,
      });
      setExtractionResult(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Extraction test failed: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  // Knowledge actions
  const toggleTopic = async (topic: string) => {
    const newExpanded = new Set(expandedTopics);
    if (newExpanded.has(topic)) {
      newExpanded.delete(topic);
    } else {
      newExpanded.add(topic);
      if (!topicFiles[topic]) {
        await fetchTopicFiles(topic);
      }
    }
    setExpandedTopics(newExpanded);
  };

  const handleAutoIngest = async () => {
    try {
      const data = await apiCall('/api/admin/knowledge/auto-ingest', { method: 'POST' });
      if (!data.job_id) {
        setMessage({ type: 'success', text: data.message || 'Knowledge base up to date' });
        return;
      }
      setMessage({
        type: 'success',
        text: `Auto-ingest started (job ${data.job_id}). Progress will appear below.`,
      });
      setReprocessJobId(data.job_id);
      pollReprocessStatus(data.job_id);
    } catch (e) {
      setMessage({ type: 'error', text: `Auto-ingest failed: ${e}` });
    }
  };

  const handleIngestFile = async () => {
    if (!ingestFile || !ingestTopic) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', ingestFile);
      formData.append('topic', ingestTopic);
      formData.append('chunk_size', ingestChunkSize.toString());
      formData.append('chunk_overlap', ingestChunkOverlap.toString());

      const data = await apiCall('/api/admin/knowledge/ingest', {
        method: 'POST',
        body: formData,
      });

      if (data.success) {
        setMessage({
          type: 'success',
          text: `Ingested ${data.file}: ${data.chunks_created} chunks (${data.processing_ms}ms)`,
        });
        setIngestFile(null);
        await fetchKnowledgeStats();
        if (expandedTopics.has(ingestTopic)) {
          await fetchTopicFiles(ingestTopic);
        }
      } else {
        setMessage({ type: 'error', text: `Ingestion failed: ${data.error}` });
      }
    } catch (e) {
      setMessage({ type: 'error', text: `Ingestion failed: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  const handleCreateTopic = async () => {
    if (!newTopicName.trim()) return;

    setLoading(true);
    try {
      const data = await apiCall('/api/admin/knowledge/topic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newTopicName.trim().toLowerCase() }),
      });

      setMessage({ type: 'success', text: data.message });
      setNewTopicName('');
      await fetchAvailableTopics();
      await fetchKnowledgeStats();
    } catch (e) {
      setMessage({ type: 'error', text: `Create topic failed: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  const handleReindexTopic = async (topic: string) => {
    if (!confirm(`Re-index all files in "${topic}"? This will clear existing chunks and re-process all files.`)) {
      return;
    }

    setLoading(true);
    try {
      const data = await apiCall(`/api/admin/knowledge/reindex/${topic}`, { method: 'POST' });
      setMessage({
        type: 'success',
        text: `Re-indexed ${topic}: ${data.successful} files (${data.processing_ms}ms)`,
      });
      await fetchKnowledgeStats();
      if (expandedTopics.has(topic)) {
        await fetchTopicFiles(topic);
      }
    } catch (e) {
      setMessage({ type: 'error', text: `Re-index failed: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteFile = async (path: string, topic: string) => {
    if (!confirm(`Delete this file from the knowledge base?`)) {
      return;
    }

    setLoading(true);
    try {
      const data = await apiCall(`/api/admin/knowledge/file?path=${encodeURIComponent(path)}`, {
        method: 'DELETE',
      });
      setMessage({
        type: 'success',
        text: `Deleted ${data.file}: ${data.chunks_deleted} chunks removed`,
      });
      await fetchKnowledgeStats();
      await fetchTopicFiles(topic);
    } catch (e) {
      setMessage({ type: 'error', text: `Delete failed: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  const handleIngestUntracked = async (path: string, topic: string) => {
    const extractor = fileExtractorSelection[path] || '';

    setProcessingFiles(prev => ({ ...prev, [path]: 'Ingesting...' }));

    try {
      let url = `/api/admin/knowledge/ingest-existing?path=${encodeURIComponent(path)}&topic=${encodeURIComponent(topic)}`;
      if (extractor) {
        url += `&extractor=${encodeURIComponent(extractor)}`;
      }

      const data = await apiCall(url, { method: 'POST' });

      if (data.success) {
        setMessage({
          type: 'success',
          text: `Ingested ${data.file}: ${data.chunks_created} chunks (${data.processing_ms}ms)`,
        });
        setFileExtractorSelection(prev => {
          const next = { ...prev };
          delete next[path];
          return next;
        });
        await fetchKnowledgeStats();
        await fetchTopicFiles(topic);
      } else {
        setMessage({ type: 'error', text: `Ingestion failed: ${data.error}` });
      }
    } catch (e) {
      setMessage({ type: 'error', text: `Ingestion failed: ${e}` });
    } finally {
      setProcessingFiles(prev => {
        const next = { ...prev };
        delete next[path];
        return next;
      });
    }
  };

  const handleReindexFile = async (path: string, topic: string) => {
    const extractor = fileExtractorSelection[path] || '';

    if (!confirm(`Re-index this file? This will clear existing chunks and re-process the file.`)) {
      return;
    }

    setProcessingFiles(prev => ({ ...prev, [path]: 'Re-indexing...' }));

    try {
      let url = `/api/admin/knowledge/reindex-file?path=${encodeURIComponent(path)}`;
      if (extractor) {
        url += `&extractor=${encodeURIComponent(extractor)}`;
      }

      const data = await apiCall(url, { method: 'POST' });

      if (data.success) {
        setMessage({
          type: 'success',
          text: `Re-indexed ${data.file}: ${data.chunks_created} chunks (${data.processing_ms}ms)`,
        });
        setFileExtractorSelection(prev => {
          const next = { ...prev };
          delete next[path];
          return next;
        });
        await fetchKnowledgeStats();
        await fetchTopicFiles(topic);
      } else {
        setMessage({ type: 'error', text: `Re-index failed: ${data.error}` });
      }
    } catch (e) {
      setMessage({ type: 'error', text: `Re-index failed: ${e}` });
    } finally {
      setProcessingFiles(prev => {
        const next = { ...prev };
        delete next[path];
        return next;
      });
    }
  };

  const handleToggleTopic = async (topic: string, enabled: boolean) => {
    try {
      const data = await apiCall(`/api/admin/knowledge/topics/toggle?topic=${encodeURIComponent(topic)}&enabled=${enabled}`, {
        method: 'PUT',
      });
      if (data.success) {
        setTopicsWithStatus(prev =>
          prev.map(t => t.name === topic ? { ...t, enabled } : t)
        );
        setMessage({
          type: 'success',
          text: `${topic} ${enabled ? 'enabled' : 'disabled'} for search`,
        });
      }
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to toggle topic: ${e}` });
    }
  };

  const handleUpdateRerankerSettings = async (updates: Partial<RerankerSettings>) => {
    setRerankerSettings(prev => prev ? { ...prev, ...updates } : prev);

    try {
      const params = new URLSearchParams();
      Object.entries(updates).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          params.append(key, String(value));
        }
      });

      const data = await apiCall(`/api/admin/knowledge/reranker?${params.toString()}`, {
        method: 'PUT',
      });

      if (data.success) {
        setRerankerSettings(data.settings);
        const pretty = (data.updated as string[]).map((k: string) => k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())).join(', ');
        setMessage({
          type: 'success',
          text: `Updated: ${pretty}`,
        });
      }
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to update settings: ${e}` });
    }
  };

  const handleSelectProfile = async (profileName: string) => {
    try {
      await apiCall(`/api/admin/knowledge/profiles/active?profile=${profileName}`, {
        method: 'PUT',
      });
      fetchRerankerSettings();
    } catch (err) {
      console.error('Failed to select profile:', err);
    }
  };

  const handleClearOverrides = async () => {
    if (rerankerSettings?.retrieval_profile) {
      handleSelectProfile(rerankerSettings.retrieval_profile);
    }
  };

  const handleCleanupStale = async () => {
    setLoading(true);
    try {
      const data = await apiCall('/api/admin/knowledge/cleanup-stale', { method: 'POST' });
      setMessage({
        type: 'success',
        text: `Cleanup: ${data.removed} stale entries removed`,
      });
      await fetchKnowledgeStats();
    } catch (e) {
      setMessage({ type: 'error', text: `Cleanup failed: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  const handleSearchTest = async () => {
    if (!searchQuery.trim()) return;

    setSearchLoading(true);
    setSearchTestResult(null);
    try {
      const data = await apiCall('/api/admin/knowledge/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          topics: searchTopics.length > 0 ? searchTopics : null,
          top_k: searchTopK,
          include_routing: true,
          include_raw_scores: true,
        }),
      });
      setSearchTestResult(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Search test failed: ${e}` });
    } finally {
      setSearchLoading(false);
    }
  };

  const handleGuidelineSearch = async () => {
    if (!guidelineSearch.trim()) return;

    setLoading(true);
    setGuidelineSearchResult(null);
    try {
      const data = await apiCall(`/api/admin/exceedee/search?q=${encodeURIComponent(guidelineSearch)}`);
      setGuidelineSearchResult(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Guideline search failed: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  // Phii/Personality handlers
  const handlePhiiSettingChange = async (key: string, value: boolean) => {
    try {
      const data = await apiCall('/api/admin/phii/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: value }),
      });
      if (data.success) {
        setPhiiStats(prev => prev ? {
          ...prev,
          settings: data.settings,
        } : null);
        setMessage({ type: 'success', text: `Updated ${key.replace('phii_', '').replace(/_/g, ' ')}` });
      }
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to update setting: ${e}` });
    }
  };

  const handleResolveFlag = async (flagId: number, notes: string = '') => {
    try {
      const data = await apiCall(`/api/admin/phii/flags/${flagId}/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes }),
      });
      if (data.success) {
        setPhiiFlags(prev => prev.filter(f => f.id !== flagId));
        setMessage({ type: 'success', text: `Flag ${flagId} resolved` });
        await fetchPhiiStats();
      }
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to resolve flag: ${e}` });
    }
  };

  const handleDeleteCorrection = async (correctionId: number) => {
    try {
      const data = await apiCall(`/api/admin/phii/corrections/${correctionId}`, {
        method: 'DELETE',
      });
      if (data.success) {
        setCorrections(prev => prev.filter(c => c.id !== correctionId));
        setCorrectionsStats(prev => ({
          ...prev,
          active_count: Math.max(0, prev.active_count - 1),
        }));
        setMessage({ type: 'success', text: `Correction #${correctionId} deleted` });
      }
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to delete correction: ${e}` });
    }
  };

  const toggleTable = (tableName: string) => {
    const newExpanded = new Set(expandedTables);
    if (newExpanded.has(tableName)) {
      newExpanded.delete(tableName);
    } else {
      newExpanded.add(tableName);
    }
    setExpandedTables(newExpanded);
  };

  const handlePhiiDebug = async () => {
    if (!phiiDebugMessage.trim()) return;

    setPhiiDebugLoading(true);
    setPhiiDebugResult(null);
    try {
      const data = await apiCall('/api/admin/phii/debug', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: phiiDebugMessage }),
      });
      setPhiiDebugResult(data);
    } catch (e) {
      setMessage({ type: 'error', text: `Phii debug failed: ${e}` });
    } finally {
      setPhiiDebugLoading(false);
    }
  };

  const handleGuidelineComparison = async () => {
    if (!comparisonFile) return;

    setIsComparing(true);
    setComparisonResult(null);
    try {
      const formData = new FormData();
      formData.append('file', comparisonFile);

      const data = await apiCall('/api/admin/exceedee/guidelines/compare', {
        method: 'POST',
        body: formData,
      });

      setComparisonResult(data);
      if (data.success) {
        setMessage({ type: 'success', text: `Comparison complete: ${data.summary || 'See results below'}` });
      }
    } catch (e) {
      setMessage({ type: 'error', text: `Comparison failed: ${e}` });
    } finally {
      setIsComparing(false);
    }
  };

  // =========================================================================
  // EXCEEDEE DB HANDLERS
  // =========================================================================

  const handleApproveRecord = async (recordId: string) => {
    try {
      await apiCall(`/api/admin/exceedee/db/approve/${recordId}`, { method: 'POST' });
      setMessage({ type: 'success', text: 'Record approved' });
      await fetchExceedeeDBStats();
      await fetchPendingRecords();
      await fetchBrowseRecords();
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to approve record: ${e}` });
    }
  };

  const handleRejectRecord = async (recordId: string) => {
    const reason = prompt('Rejection reason (optional):');
    if (reason === null) return;

    try {
      await apiCall(`/api/admin/exceedee/db/reject/${recordId}?reason=${encodeURIComponent(reason)}`, { method: 'POST' });
      setMessage({ type: 'success', text: 'Record rejected' });
      await fetchExceedeeDBStats();
      await fetchPendingRecords();
      await fetchBrowseRecords();
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to reject record: ${e}` });
    }
  };

  const handleDeleteRecord = async (recordId: string) => {
    if (!confirm('Are you sure you want to delete this record?')) return;

    try {
      await apiCall(`/api/admin/exceedee/db/record/${recordId}`, { method: 'DELETE' });
      setMessage({ type: 'success', text: 'Record deleted' });
      await fetchExceedeeDBStats();
      await fetchPendingRecords();
      await fetchBrowseRecords();
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to delete record: ${e}` });
    }
  };

  const handleExportRecords = async (format: 'csv' | 'json') => {
    try {
      const response = await fetch(`${getApiBase()}/api/admin/exceedee/db/export?format=${format}`, {
        headers: { 'Authorization': `Bearer ${token}` },
      });

      if (!response.ok) throw new Error('Export failed');

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `exceedee_export.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (e) {
      setMessage({ type: 'error', text: `Export failed: ${e}` });
    }
  };

  // =========================================================================
  // REPROCESS HANDLERS
  // =========================================================================

  const handleStartReprocess = async () => {
    setShowReprocessConfirm(false);
    setLoading(true);

    try {
      const data = await apiCall('/api/admin/knowledge/reprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: reprocessMode,
          topics: null,
          purge_qdrant: reprocessOptions.purge_qdrant,
          clear_bm25: reprocessOptions.clear_bm25,
          clear_neo4j: reprocessOptions.clear_neo4j,
          build_raptor: reprocessOptions.build_raptor,
          build_graph: reprocessOptions.build_graph,
          skip_vision: reprocessOptions.skip_vision,
        }),
      });

      setReprocessJobId(data.job_id);
      setMessage({
        type: 'success',
        text: `Reprocessing started (Job: ${data.job_id})`,
      });

      pollReprocessStatus(data.job_id);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to start reprocess: ${e}` });
    } finally {
      setLoading(false);
    }
  };

  const pollReprocessStatus = async (jobId: string) => {
    const poll = async () => {
      try {
        const data = await apiCall(`/api/admin/knowledge/reprocess-status/${jobId}`);
        setReprocessStatus(data);

        if (data.status === 'running' || data.status === 'starting') {
          setTimeout(poll, 2000);
        } else {
          await fetchKnowledgeStats();
          if (data.status === 'completed') {
            setMessage({
              type: 'success',
              text: `Reprocessing complete: ${data.successful} files processed, ${data.extraction_stats.vision_pages} pages used vision`,
            });
          }
        }
      } catch (e) {
        console.error('Failed to poll reprocess status:', e);
      }
    };

    poll();
  };

  const handleCancelReprocess = async () => {
    if (!reprocessJobId) return;

    try {
      await apiCall(`/api/admin/knowledge/reprocess-cancel/${reprocessJobId}`, {
        method: 'POST',
      });
      setMessage({ type: 'success', text: 'Reprocess cancellation requested' });
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to cancel: ${e}` });
    }
  };

  return {
    // Context
    domain,

    // Auth
    username, setUsername,
    password, setPassword,
    token,
    currentUsername,
    isAuthenticated,
    authError,
    setupRequired,
    authChecked,
    handleAuth,
    handleLogout,

    // User management
    usersList,
    newUserUsername, setNewUserUsername,
    newUserPassword, setNewUserPassword,
    userMgmtMessage,
    handleCreateUser,
    handleDeleteUser,
    fetchUsers,

    // Data
    status,
    config,
    logs,
    sessions,
    hardwareProfile,

    // UI
    activeTab, setActiveTab,
    configSubTab, setConfigSubTab,
    knowledgeSubTab, setKnowledgeSubTab,
    complianceSubTab, setComplianceSubTab,
    intelligenceSubTab, setIntelligenceSubTab,
    diagnosticsSubTab, setDiagnosticsSubTab,
    loading,
    searchLoading,
    message, setMessage,
    firstBootDismissed, setFirstBootDismissed,

    // API
    apiCall,

    // Fetch helpers
    fetchStatus,
    fetchLogs,
    fetchSessions,
    fetchUsers2: fetchUsers,

    // Config
    editedConfig, setEditedConfig,
    handleSaveConfig,
    handleResetConfig,
    handleRecalibrate,

    // Models
    modelsData,
    pullModelName, setPullModelName,
    pullStatus,
    pullingModel,
    fetchModels,
    handlePullModel,
    handleAssignModel,
    handleDeleteModel,
    fetchTools,
    toolsData,
    handleReloadTools,
    handleScaffoldTool,
    handleUpdateCustomTool,

    // Knowledge
    knowledgeStats,
    expandedTopics,
    topicFiles,
    topicUntracked,
    availableTopics,
    searchTestResult,
    fileExtractorSelection, setFileExtractorSelection,
    topicsWithStatus,
    rerankerSettings,
    profiles,
    processingFiles,
    ingestFile, setIngestFile,
    ingestTopic, setIngestTopic,
    newTopicName, setNewTopicName,
    showAdvancedIngest, setShowAdvancedIngest,
    ingestChunkSize, setIngestChunkSize,
    ingestChunkOverlap, setIngestChunkOverlap,
    searchQuery, setSearchQuery,
    searchTopics, setSearchTopics,
    searchTopK, setSearchTopK,
    toggleTopic,
    handleAutoIngest,
    handleCleanupStale,
    handleIngestFile,
    handleCreateTopic,
    handleReindexTopic,
    handleDeleteFile,
    handleIngestUntracked,
    handleReindexFile,
    handleToggleTopic,
    handleUpdateRerankerSettings,
    handleSelectProfile,
    handleClearOverrides,
    handleSearchTest,

    // Reprocess
    reprocessJobId,
    reprocessStatus,
    showReprocessConfirm, setShowReprocessConfirm,
    reprocessMode, setReprocessMode,
    reprocessOptions, setReprocessOptions,
    handleStartReprocess,
    handleCancelReprocess,

    // Compliance
    guidelineStats,
    fetchGuidelineStats,
    expandedTables,
    toggleTable,
    guidelineSearch, setGuidelineSearch,
    guidelineSearchResult,
    handleGuidelineSearch,
    comparisonFile, setComparisonFile,
    comparisonResult,
    isComparing,
    handleGuidelineComparison,

    // ExceedeeDB
    exceedeeDBStats,
    pendingRecords,
    browseRecords,
    dbProjectFilter, setDbProjectFilter,
    dbStatusFilter, setDbStatusFilter,
    fetchExceedeeDBStats,
    fetchPendingRecords,
    fetchBrowseRecords,
    handleApproveRecord,
    handleRejectRecord,
    handleDeleteRecord,
    handleExportRecords,

    // Personality / Phii
    phiiStats,
    phiiFlags,
    corrections,
    correctionsStats,
    patternStats,
    phiiDebugMessage, setPhiiDebugMessage,
    phiiDebugResult,
    phiiDebugLoading,
    handlePhiiSettingChange,
    handleResolveFlag,
    handleDeleteCorrection,
    fetchPhiiFlags,
    fetchCorrections,
    fetchPatterns,
    handlePhiiDebug,

    // Extraction tester
    testFile, setTestFile,
    testExtractor, setTestExtractor,
    extractionResult,
    handleTestExtraction,

    // Domain
    availableDomains,
    domainSwitching,
    handleDomainSwitch,
  };
}
