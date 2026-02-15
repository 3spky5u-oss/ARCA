'use client';

import React from 'react';
import dynamic from 'next/dynamic';
import { StatusTab } from './components';
import { SubTabLayout } from './components/SubTabLayout';
import { AdminStatusBar } from './components/AdminStatusBar';
import { useAdminState } from './hooks/useAdminState';
import type { MainTab } from './hooks/useAdminState';

// Lazy-load admin tabs for code splitting
const ConfigTab = dynamic(() => import('./components').then(m => ({ default: m.ConfigTab })));
// ModelsTab is now embedded in ConfigTab — no longer needed as standalone import
const LogsTab = dynamic(() => import('./components').then(m => ({ default: m.LogsTab })));
const SessionsTab = dynamic(() => import('./components').then(m => ({ default: m.SessionsTab })));
const KnowledgeDocumentsTab = dynamic(() => import('./components').then(m => ({ default: m.KnowledgeDocumentsTab })));
const KnowledgeRetrievalTab = dynamic(() => import('./components').then(m => ({ default: m.KnowledgeRetrievalTab })));
const GraphTab = dynamic(() => import('./components').then(m => ({ default: m.GraphTab })));
const KnowledgeCollectionsTab = dynamic(() => import('./components').then(m => ({ default: m.KnowledgeCollectionsTab })));
const ComplianceTab = dynamic(() => import('./components').then(m => ({ default: m.ComplianceTab })));
const ExceedeeDBTab = dynamic(() => import('./components').then(m => ({ default: m.ExceedeeDBTab })));
const PersonalityTab = dynamic(() => import('./components').then(m => ({ default: m.PersonalityTab })));
const TesterTab = dynamic(() => import('./components').then(m => ({ default: m.TesterTab })));
const BenchmarkTab = dynamic(() => import('./components').then(m => ({ default: m.BenchmarkTab })));
const LexiconTab = dynamic(() => import('./components').then(m => ({ default: m.LexiconTab })));
const ToolsTab = dynamic(() => import('./components').then(m => ({ default: m.ToolsTab })));

export default function AdminPage() {
  const state = useAdminState();

  const {
    domain,
    username, setUsername,
    password, setPassword,
    currentUsername,
    isAuthenticated,
    authError,
    setupRequired,
    authChecked,
    handleAuth,
    handleLogout,
    usersList,
    newUserUsername, setNewUserUsername,
    newUserPassword, setNewUserPassword,
    userMgmtMessage,
    handleCreateUser,
    handleDeleteUser,
    fetchUsers,
    status,
    config,
    logs,
    sessions,
    hardwareProfile,
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
    apiCall,
    fetchStatus,
    editedConfig, setEditedConfig,
    handleSaveConfig,
    handleResetConfig,
    handleRecalibrate,
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
    reprocessStatus,
    showReprocessConfirm, setShowReprocessConfirm,
    reprocessMode, setReprocessMode,
    reprocessOptions, setReprocessOptions,
    handleStartReprocess,
    handleCancelReprocess,
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
    testFile, setTestFile,
    testExtractor, setTestExtractor,
    extractionResult,
    handleTestExtraction,
    availableDomains,
    domainSwitching,
    handleDomainSwitch,
    fetchLogs,
    fetchSessions,
  } = state;

  // =========================================================================
  // AUTH SCREEN
  // =========================================================================

  if (!authChecked) {
    return (
      <div className="min-h-screen bg-[#1a1a1a] flex items-center justify-center">
        <div className="text-gray-500">Loading...</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-[#1a1a1a] flex items-center justify-center p-4">
        <div className="bg-[#2a2a2a] rounded-lg p-8 w-full max-w-md">
          <div className="flex items-center justify-center mb-6">
            <svg className="w-12 h-12 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          </div>
          <h1 className="text-xl font-semibold text-white text-center mb-2">{domain.app_name} Admin</h1>
          {setupRequired && (
            <p className="text-gray-400 text-sm text-center mb-6">
              No users found. Create your admin account.
            </p>
          )}
          {!setupRequired && (
            <div className="mb-6" />
          )}
          <div className="space-y-4">
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder={setupRequired ? 'Choose a username' : 'Username'}
              autoComplete="username"
              className="w-full px-4 py-3 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAuth()}
              placeholder={setupRequired ? 'Choose a password (min 8 chars)' : 'Password'}
              autoComplete={setupRequired ? 'new-password' : 'current-password'}
              className="w-full px-4 py-3 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {authError && <p className="text-red-400 text-sm">{authError}</p>}
            <button
              onClick={handleAuth}
              disabled={!username || (setupRequired && password.length < 8)}
              className="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-xl font-medium transition-colors"
            >
              {setupRequired ? 'Create Account' : 'Login'}
            </button>
          </div>
          <p className="text-gray-500 text-xs text-center mt-6">
            Ctrl+Shift+D from main app to access
          </p>
        </div>
      </div>
    );
  }

  // =========================================================================
  // MAIN DASHBOARD
  // =========================================================================

  return (
    <div className="min-h-screen bg-[#1a1a1a] text-white">
      {/* Header */}
      <header className="bg-[#2a2a2a] border-b border-[#3a3a3a] px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-semibold">{domain.app_name} Admin</h1>
            {availableDomains.length > 1 && (
              <select
                value={domain.name}
                onChange={(e) => handleDomainSwitch(e.target.value)}
                disabled={domainSwitching}
                className="text-xs bg-[#1a1a1a] border border-[#3a3a3a] text-gray-300 rounded-lg px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
              >
                {availableDomains.map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.display_name} ({d.tools_count} tools)
                  </option>
                ))}
              </select>
            )}
            <span className="text-xs text-gray-500 bg-[#1a1a1a] px-3 py-1 rounded-full">
              {status?.llm?.status === 'connected' ? (
                <span className="text-green-400">Connected</span>
              ) : (
                <span className="text-red-400">Disconnected</span>
              )}
            </span>
          </div>
          <div className="flex items-center gap-3">
            {currentUsername && (
              <span className="text-gray-400 text-sm">{currentUsername}</span>
            )}
            <button
              onClick={handleLogout}
              className="text-gray-400 hover:text-white text-sm"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* Message Toast */}
      {message && (
        <div
          className={`fixed top-4 right-4 px-4 py-3 rounded-2xl shadow-lg z-50 max-w-md ${
            message.type === 'success' ? 'bg-green-600' : 'bg-red-600'
          }`}
        >
          <div className="flex items-center gap-2">
            <span className="text-sm">{message.text}</span>
            <button onClick={() => setMessage(null)} className="text-white/60 hover:text-white ml-2 text-lg leading-none">
              &times;
            </button>
          </div>
        </div>
      )}

      {/* Admin Status Bar */}
      <AdminStatusBar apiCall={apiCall} />

      <div className="flex">
        {/* Sidebar Navigation — 7 consolidated tabs */}
        <nav className="w-48 bg-[#222] border-r border-[#3a3a3a] min-h-[calc(100vh-73px)]">
          <ul className="py-2">
            {([
              { id: 'dashboard', label: 'Dashboard' },
              { id: 'config', label: 'Configuration' },
              { id: 'domain', label: 'Domain' },
              { id: 'knowledge', label: 'Knowledge' },
              { id: 'benchmark', label: 'Benchmark' },
              ...(domain.routes.includes('admin_exceedee') ? [{ id: 'compliance', label: 'Compliance' }] : []),
              { id: 'intelligence', label: 'Intelligence' },
              { id: 'tools', label: 'Tools' },
              { id: 'diagnostics', label: 'Diagnostics' },
            ] as { id: MainTab; label: string }[]).map((tab) => (
              <li key={tab.id}>
                <button
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full px-4 py-2.5 text-left text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-[#2a2a2a]'
                  }`}
                >
                  {tab.label}
                </button>
              </li>
            ))}
          </ul>
        </nav>

        {/* Main Content */}
        <main className="flex-1 p-6 overflow-auto">
          {loading && (
            <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-40">
              <div className="bg-[#2a2a2a] rounded-lg p-4 flex items-center gap-3">
                <svg className="animate-spin h-5 w-5 text-blue-500" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Loading...</span>
              </div>
            </div>
          )}

          {/* First-Boot Welcome Banner */}
          {activeTab === 'dashboard' && status && !firstBootDismissed && (
            status.llm?.status !== 'connected' || (status.rag?.knowledge_chunks ?? 0) === 0
          ) && (
            <div className="bg-gradient-to-r from-blue-900/40 to-indigo-900/40 border border-blue-700/50 rounded-2xl p-6 mb-6">
              <div className="flex items-start justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-white mb-1">Welcome to {domain.app_name}</h2>
                  <p className="text-gray-300 text-sm mb-4">Get started in three steps:</p>
                  <ol className="space-y-2 text-sm">
                    <li className="flex items-center gap-2">
                      <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
                        hardwareProfile ? 'bg-green-600 text-white' : 'bg-[#333] text-gray-400'
                      }`}>1</span>
                      <span className={hardwareProfile ? 'text-green-400' : 'text-gray-300'}>
                        Check hardware profile
                        {hardwareProfile?.profile && <span className="text-gray-500 ml-1">({hardwareProfile.profile})</span>}
                      </span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
                        status.llm?.status === 'connected' ? 'bg-green-600 text-white' : 'bg-[#333] text-gray-400'
                      }`}>2</span>
                      <span className={status.llm?.status === 'connected' ? 'text-green-400' : 'text-gray-300'}>
                        {status.llm?.status === 'connected'
                          ? 'LLM server running'
                          : 'Start LLM server (place GGUF models in ./models/)'}
                      </span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
                        (status.rag?.knowledge_chunks ?? 0) > 0 ? 'bg-green-600 text-white' : 'bg-[#333] text-gray-400'
                      }`}>3</span>
                      <span className={(status.rag?.knowledge_chunks ?? 0) > 0 ? 'text-green-400' : 'text-gray-300'}>
                        {(status.rag?.knowledge_chunks ?? 0) > 0
                          ? `Knowledge base populated (${status.rag?.knowledge_chunks} chunks)`
                          : 'Ingest documents via Knowledge tab'}
                      </span>
                    </li>
                  </ol>
                </div>
                <button
                  onClick={() => {
                    setFirstBootDismissed(true);
                    sessionStorage.setItem('arca_first_boot_dismissed', 'true');
                  }}
                  className="text-gray-400 hover:text-white text-lg leading-none ml-4 mt-1"
                  title="Dismiss"
                >
                  &times;
                </button>
              </div>
            </div>
          )}

          {/* 1. DASHBOARD */}
          {activeTab === 'dashboard' && status && (
            <StatusTab status={status} onRecalibrate={handleRecalibrate} onRefresh={fetchStatus} hardwareProfile={hardwareProfile} />
          )}

          {/* 2. CONFIGURATION (sub-tabs: General, Models, Connections) */}
          {activeTab === 'config' && config && (
            <ConfigTab
              config={config}
              editedConfig={editedConfig}
              serviceStatus={status ? {
                llm: status.llm,
                redis: status.services?.redis ? { status: status.services.redis.status === 'connected' ? 'connected' : 'error' } : (config.redis_enabled ? { status: 'connected' } : undefined),
                postgres: status.services?.postgres ? { status: status.services.postgres.status === 'connected' ? 'connected' : 'error' } : (config.database_enabled ? { status: 'connected' } : undefined),
                neo4j: status.services?.neo4j ? { status: status.services.neo4j.status === 'healthy' || status.services.neo4j.connected ? 'connected' : 'error' } : undefined,
                qdrant: status.services?.qdrant ? { status: status.services.qdrant.status === 'connected' ? 'connected' : 'error' } : { status: 'connected' },
              } : null}
              onEditConfig={setEditedConfig}
              onSave={handleSaveConfig}
              onReset={handleResetConfig}
              modelsData={modelsData}
              pullModelName={pullModelName}
              onPullModelNameChange={setPullModelName}
              pullStatus={pullStatus}
              pullingModel={pullingModel}
              onRefreshModels={fetchModels}
              onPullModel={handlePullModel}
              onAssignModel={handleAssignModel}
              onDeleteModel={handleDeleteModel}
              searxngStatus={status?.services?.searxng ? {
                status: (
                  status.services.searxng.status === 'connected'
                    ? 'connected'
                    : status.services.searxng.status === 'disabled'
                      ? 'disabled'
                      : status.services.searxng.status === 'error'
                        ? 'error'
                        : 'unknown'
                ),
                healthy: status.services.searxng.status === 'connected',
                url: ((status.services.searxng as unknown as Record<string, unknown>).url as string) || 'http://searxng:8080',
                error: status.services.searxng.error,
              } : null}
              vramTotalGb={status?.system?.specs?.gpu_vram_total_mb != null ? status.system.specs.gpu_vram_total_mb / 1024 : null}
              subTab={configSubTab}
              onSubTabChange={setConfigSubTab}
            />
          )}

                    {/* 3. USER ACCOUNTS (within config section) */}
          {activeTab === 'config' && configSubTab === 'general' && (
            <div className="bg-[#2a2a2a] rounded-2xl p-6 mt-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium">User Accounts</h3>
                <button onClick={fetchUsers} className="text-xs text-gray-400 hover:text-white">Refresh</button>
              </div>
              {userMgmtMessage && (
                <div className={`mb-4 px-3 py-2 rounded-lg text-sm ${userMgmtMessage.type === 'success' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>
                  {userMgmtMessage.text}
                </div>
              )}
              {usersList.length > 0 && (
                <div className="space-y-2 mb-4">
                  {usersList.map((u) => (
                    <div key={u.id} className="flex items-center justify-between bg-[#1a1a1a] rounded-xl px-4 py-2">
                      <div className="flex items-center gap-3">
                        <span className="text-sm font-medium">{u.username}</span>
                        <span className="text-xs text-gray-500 bg-[#333] px-2 py-0.5 rounded">{u.role}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-gray-500">
                          {u.created_at ? new Date(u.created_at).toLocaleDateString() : ''}
                        </span>
                        {u.username !== currentUsername && (
                          <button
                            onClick={() => handleDeleteUser(u.id, u.username)}
                            className="text-xs text-red-400 hover:text-red-300"
                          >
                            Delete
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              <div className="border-t border-[#3a3a3a] pt-4">
                <p className="text-sm text-gray-400 mb-3">Create new user</p>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newUserUsername}
                    onChange={(e) => setNewUserUsername(e.target.value)}
                    placeholder="Username"
                    className="flex-1 px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                  <input
                    type="password"
                    value={newUserPassword}
                    onChange={(e) => setNewUserPassword(e.target.value)}
                    placeholder="Password (min 8)"
                    className="flex-1 px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                  <button
                    onClick={handleCreateUser}
                    disabled={!newUserUsername || newUserPassword.length < 8}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-xl text-sm transition-colors"
                  >
                    Create
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* DOMAIN LEXICON */}
          {activeTab === 'domain' && (
            <LexiconTab
              apiCall={apiCall}
              setMessage={setMessage}
            />
          )}

          {/* 4. KNOWLEDGE (sub-tabs: Documents, Retrieval, Graph) */}
          {activeTab === 'knowledge' && (
            <div>
              <h2 className="text-lg font-medium mb-4">Knowledge</h2>
              <SubTabLayout
                tabs={[
                  { id: 'documents', label: 'Documents' },
                  { id: 'retrieval', label: 'Retrieval' },
                  { id: 'collections', label: 'Collections' },
                  { id: 'graph', label: 'Graph' },
                ]}
                activeTab={knowledgeSubTab}
                onTabChange={(t) => setKnowledgeSubTab(t as 'documents' | 'retrieval' | 'collections' | 'graph')}
              >
                {knowledgeSubTab === 'documents' && (
                  <KnowledgeDocumentsTab
                    knowledgeStats={knowledgeStats}
                    expandedTopics={expandedTopics}
                    topicFiles={topicFiles}
                    topicUntracked={topicUntracked}
                    availableTopics={availableTopics}
                    fileExtractorSelection={fileExtractorSelection}
                    processingFiles={processingFiles}
                    ingestFile={ingestFile}
                    ingestTopic={ingestTopic}
                    newTopicName={newTopicName}
                    showAdvancedIngest={showAdvancedIngest}
                    ingestChunkSize={ingestChunkSize}
                    ingestChunkOverlap={ingestChunkOverlap}
                    reprocessStatus={reprocessStatus}
                    showReprocessConfirm={showReprocessConfirm}
                    reprocessMode={reprocessMode}
                    reprocessOptions={reprocessOptions}
                    onToggleTopic={toggleTopic}
                    onAutoIngest={handleAutoIngest}
                    onCleanupStale={handleCleanupStale}
                    onIngestFile={handleIngestFile}
                    onCreateTopic={handleCreateTopic}
                    onReindexTopic={handleReindexTopic}
                    onDeleteFile={handleDeleteFile}
                    onIngestUntracked={handleIngestUntracked}
                    onReindexFile={handleReindexFile}
                    onStartReprocess={handleStartReprocess}
                    onCancelReprocess={handleCancelReprocess}
                    onIngestFileChange={setIngestFile}
                    onIngestTopicChange={setIngestTopic}
                    onNewTopicNameChange={setNewTopicName}
                    onShowAdvancedIngestChange={setShowAdvancedIngest}
                    onIngestChunkSizeChange={setIngestChunkSize}
                    onIngestChunkOverlapChange={setIngestChunkOverlap}
                    onFileExtractorSelectionChange={setFileExtractorSelection}
                    onShowReprocessConfirmChange={setShowReprocessConfirm}
                    onReprocessModeChange={setReprocessMode}
                    onReprocessOptionsChange={setReprocessOptions}
                    loading={loading}
                  />
                )}
                {knowledgeSubTab === 'retrieval' && (
                  <KnowledgeRetrievalTab
                    topicsWithStatus={topicsWithStatus}
                    availableTopics={availableTopics}
                    rerankerSettings={rerankerSettings}
                    searchTestResult={searchTestResult}
                    searchQuery={searchQuery}
                    searchTopics={searchTopics}
                    searchTopK={searchTopK}
                    onToggleTopicEnabled={handleToggleTopic}
                    onUpdateRerankerSettings={handleUpdateRerankerSettings}
                    onSearchTest={handleSearchTest}
                    onSearchQueryChange={setSearchQuery}
                    onSearchTopicsChange={setSearchTopics}
                    onSearchTopKChange={setSearchTopK}
                    loading={searchLoading}
                    profiles={profiles}
                    activeProfile={rerankerSettings?.retrieval_profile}
                    hasManualOverrides={rerankerSettings?.has_manual_overrides}
                    onSelectProfile={handleSelectProfile}
                    onClearOverrides={handleClearOverrides}
                  />
                )}
                {knowledgeSubTab === 'collections' && (
                  <KnowledgeCollectionsTab
                    apiCall={apiCall}
                    setMessage={setMessage}
                  />
                )}
                {knowledgeSubTab === 'graph' && (
                  <GraphTab
                    apiCall={apiCall}
                    setMessage={setMessage}
                  />
                )}
              </SubTabLayout>
            </div>
          )}

          {/* 5. COMPLIANCE (sub-tabs: Guidelines, Records) — domain-gated */}
          {activeTab === 'compliance' && domain.routes.includes('admin_exceedee') && (
            <div>
              <h2 className="text-lg font-medium mb-4">Compliance</h2>
              <SubTabLayout
                tabs={[
                  { id: 'guidelines', label: 'Guidelines' },
                  { id: 'records', label: 'Records' },
                ]}
                activeTab={complianceSubTab}
                onTabChange={(t) => setComplianceSubTab(t as 'guidelines' | 'records')}
              >
                {complianceSubTab === 'guidelines' && (
                  <ComplianceTab
                    stats={guidelineStats}
                    expandedTables={expandedTables}
                    onToggleTable={toggleTable}
                    guidelineSearch={guidelineSearch}
                    onGuidelineSearchChange={setGuidelineSearch}
                    guidelineSearchResult={guidelineSearchResult}
                    onGuidelineSearch={handleGuidelineSearch}
                    comparisonFile={comparisonFile}
                    onComparisonFileChange={setComparisonFile}
                    comparisonResult={comparisonResult}
                    isComparing={isComparing}
                    onGuidelineComparison={handleGuidelineComparison}
                    onRefresh={fetchGuidelineStats}
                  />
                )}
                {complianceSubTab === 'records' && (
                  <ExceedeeDBTab
                    stats={exceedeeDBStats}
                    pendingRecords={pendingRecords}
                    browseRecords={browseRecords}
                    projectFilter={dbProjectFilter}
                    onProjectFilterChange={setDbProjectFilter}
                    statusFilter={dbStatusFilter}
                    onStatusFilterChange={setDbStatusFilter}
                    onRefresh={() => { fetchExceedeeDBStats(); fetchPendingRecords(); fetchBrowseRecords(); }}
                    onExportRecords={handleExportRecords}
                    onApproveRecord={handleApproveRecord}
                    onRejectRecord={handleRejectRecord}
                    onDeleteRecord={handleDeleteRecord}
                    onSearch={fetchBrowseRecords}
                  />
                )}
              </SubTabLayout>
            </div>
          )}

          {/* 6. INTELLIGENCE (sub-tabs: Personality) */}
          {activeTab === 'intelligence' && (
            <div>
              <h2 className="text-lg font-medium mb-4">Intelligence</h2>
              <SubTabLayout
                tabs={[
                  { id: 'personality', label: 'Personality' },
                ]}
                activeTab={intelligenceSubTab}
                onTabChange={(t) => setIntelligenceSubTab(t as 'personality')}
              >
                {intelligenceSubTab === 'personality' && (
                  <PersonalityTab
                    phiiStats={phiiStats}
                    phiiFlags={phiiFlags}
                    corrections={corrections}
                    correctionsStats={correctionsStats}
                    patternStats={patternStats}
                    phiiDebugMessage={phiiDebugMessage}
                    onPhiiDebugMessageChange={setPhiiDebugMessage}
                    phiiDebugResult={phiiDebugResult}
                    phiiDebugLoading={phiiDebugLoading}
                    onPhiiSettingChange={handlePhiiSettingChange}
                    onResolveFlag={handleResolveFlag}
                    onDeleteCorrection={handleDeleteCorrection}
                    onRefreshFlags={fetchPhiiFlags}
                    onRefreshCorrections={fetchCorrections}
                    onRefreshPatterns={fetchPatterns}
                    onPhiiDebug={handlePhiiDebug}
                  />
                )}
              </SubTabLayout>
            </div>
          )}

          {/* BENCHMARK */}
          {activeTab === 'benchmark' && (
            <BenchmarkTab
              apiCall={apiCall}
              setMessage={setMessage}
            />
          )}

          {/* TOOLS */}
          {activeTab === 'tools' && (
            <ToolsTab
              toolsData={toolsData}
              loading={loading}
              onRefresh={fetchTools}
              onReloadTools={handleReloadTools}
              onScaffoldTool={handleScaffoldTool}
              onUpdateCustomTool={handleUpdateCustomTool}
            />
          )}

          {/* 7. DIAGNOSTICS (sub-tabs: Extraction, Tests, Logs, Sessions) */}
          {activeTab === 'diagnostics' && (
            <div>
              <h2 className="text-lg font-medium mb-4">Diagnostics</h2>
              <SubTabLayout
                tabs={[
                  { id: 'extraction', label: 'Extraction' },
                  { id: 'logs', label: 'Logs' },
                  { id: 'sessions', label: 'Sessions' },
                ]}
                activeTab={diagnosticsSubTab}
                onTabChange={(t) => {
                  setDiagnosticsSubTab(t as 'extraction' | 'logs' | 'sessions');
                }}
              >
                {diagnosticsSubTab === 'extraction' && (
                  <TesterTab
                    testFile={testFile}
                    onTestFileChange={setTestFile}
                    testExtractor={testExtractor}
                    onTestExtractorChange={setTestExtractor}
                    extractionResult={extractionResult}
                    onTestExtraction={handleTestExtraction}
                  />
                )}
                {diagnosticsSubTab === 'logs' && (
                  <LogsTab logs={logs} onRefresh={fetchLogs} />
                )}
                {diagnosticsSubTab === 'sessions' && (
                  <SessionsTab sessions={sessions} onRefresh={fetchSessions} />
                )}
              </SubTabLayout>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
