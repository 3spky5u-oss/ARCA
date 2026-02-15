'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { SettingTip } from './shared';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Lexicon = Record<string, any>;

interface LexiconTabProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  apiCall: (endpoint: string, options?: RequestInit) => Promise<any>;
  setMessage: (msg: { type: 'success' | 'error'; text: string } | null) => void;
}

// ---------------------------------------------------------------------------
// Small reusable inputs (same style as ConfigTab)
// ---------------------------------------------------------------------------

function LexTextField({ label, tip, value, onChange, multiline }: {
  label: string; tip?: string; value: string; onChange: (v: string) => void; multiline?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-sm text-gray-300">
        {label}
        {tip && <SettingTip tip={tip} />}
      </label>
      {multiline ? (
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          rows={5}
          className="px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500 resize-y font-mono"
        />
      ) : (
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="px-3 py-2 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Chip list — add/remove strings
// ---------------------------------------------------------------------------

function ChipList({ items, onAdd, onRemove, placeholder }: {
  items: string[]; onAdd: (v: string) => void; onRemove: (i: number) => void; placeholder?: string;
}) {
  const [input, setInput] = useState('');
  const handleAdd = () => {
    const v = input.trim();
    if (v && !items.includes(v)) { onAdd(v); setInput(''); }
  };
  return (
    <div>
      <div className="flex flex-wrap gap-1.5 mb-2">
        {items.map((item, i) => (
          <span key={i} className="inline-flex items-center gap-1 px-2.5 py-1 bg-[#2a2a2a] border border-[#3a3a3a] rounded-lg text-xs text-gray-300">
            {item}
            <button onClick={() => onRemove(i)} className="text-gray-500 hover:text-red-400 ml-0.5">&times;</button>
          </span>
        ))}
        {items.length === 0 && <span className="text-xs text-gray-600 italic">None</span>}
      </div>
      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), handleAdd())}
          placeholder={placeholder || 'Add item...'}
          className="flex-1 px-3 py-1.5 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-xs text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <button onClick={handleAdd} className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs transition-colors">Add</button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Key-value map editor (specialties: key -> {keywords, description})
// ---------------------------------------------------------------------------

function SpecialtiesEditor({ specialties, onChange }: {
  specialties: Record<string, { keywords: string[]; description: string }>;
  onChange: (v: Record<string, { keywords: string[]; description: string }>) => void;
}) {
  const [newKey, setNewKey] = useState('');
  const keys = Object.keys(specialties);

  const addSpecialty = () => {
    const k = newKey.trim().toLowerCase();
    if (k && !specialties[k]) {
      onChange({ ...specialties, [k]: { keywords: [], description: '' } });
      setNewKey('');
    }
  };

  const removeSpecialty = (key: string) => {
    const copy = { ...specialties };
    delete copy[key];
    onChange(copy);
  };

  const updateDescription = (key: string, desc: string) => {
    onChange({ ...specialties, [key]: { ...specialties[key], description: desc } });
  };

  const updateKeywords = (key: string, kws: string[]) => {
    onChange({ ...specialties, [key]: { ...specialties[key], keywords: kws } });
  };

  return (
    <div className="space-y-3">
      {keys.map((key) => (
        <div key={key} className="p-3 bg-[#1e1e1e] border border-[#3a3a3a] rounded-xl">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-blue-400">{key}</span>
            <button onClick={() => removeSpecialty(key)} className="text-xs text-gray-500 hover:text-red-400">Remove</button>
          </div>
          <LexTextField
            label="Description"
            value={specialties[key]?.description || ''}
            onChange={(v) => updateDescription(key, v)}
          />
          <div className="mt-2">
            <label className="text-xs text-gray-400 mb-1 block">Keywords</label>
            <ChipList
              items={specialties[key]?.keywords || []}
              onAdd={(v) => updateKeywords(key, [...(specialties[key]?.keywords || []), v])}
              onRemove={(i) => {
                const kws = [...(specialties[key]?.keywords || [])];
                kws.splice(i, 1);
                updateKeywords(key, kws);
              }}
              placeholder="Add keyword..."
            />
          </div>
        </div>
      ))}
      {keys.length === 0 && <p className="text-xs text-gray-600 italic">No specialties defined</p>}
      <div className="flex gap-2">
        <input
          value={newKey}
          onChange={(e) => setNewKey(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addSpecialty())}
          placeholder="New specialty name..."
          className="flex-1 px-3 py-1.5 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-xs text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <button onClick={addSpecialty} className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs transition-colors">Add Specialty</button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Terminology variants editor
// Each key maps to an array of [pattern, display] pairs
// ---------------------------------------------------------------------------

function TerminologyEditor({ variants, onChange }: {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  variants: Record<string, any>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onChange: (v: Record<string, any>) => void;
}) {
  const [newKey, setNewKey] = useState('');
  const [expandedTerm, setExpandedTerm] = useState<string | null>(null);
  const keys = Object.keys(variants);

  const addTerm = () => {
    const k = newKey.trim();
    if (k && !variants[k]) {
      onChange({ ...variants, [k]: [] });
      setNewKey('');
      setExpandedTerm(k);
    }
  };

  const removeTerm = (key: string) => {
    const copy = { ...variants };
    delete copy[key];
    onChange(copy);
    if (expandedTerm === key) setExpandedTerm(null);
  };

  const addVariant = (key: string) => {
    const current = Array.isArray(variants[key]) ? [...variants[key]] : [];
    current.push(['\\\\b\\\\b', '']);
    onChange({ ...variants, [key]: current });
  };

  const updateVariant = (key: string, idx: number, field: 0 | 1, value: string) => {
    const current = Array.isArray(variants[key]) ? [...variants[key]] : [];
    if (current[idx]) {
      const pair = [...current[idx]];
      pair[field] = value;
      current[idx] = pair;
      onChange({ ...variants, [key]: current });
    }
  };

  const removeVariant = (key: string, idx: number) => {
    const current = Array.isArray(variants[key]) ? [...variants[key]] : [];
    current.splice(idx, 1);
    onChange({ ...variants, [key]: current });
  };

  return (
    <div className="space-y-2">
      {keys.map((key) => {
        const pairs = Array.isArray(variants[key]) ? variants[key] : [];
        const isExpanded = expandedTerm === key;
        return (
          <div key={key} className="bg-[#1e1e1e] border border-[#3a3a3a] rounded-xl overflow-hidden">
            <div
              className="flex items-center justify-between px-3 py-2 cursor-pointer hover:bg-[#252525]"
              onClick={() => setExpandedTerm(isExpanded ? null : key)}
            >
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">{isExpanded ? '\u25BC' : '\u25B6'}</span>
                <span className="text-sm font-medium text-blue-400">{key}</span>
                <span className="text-xs text-gray-500">({pairs.length} variant{pairs.length !== 1 ? 's' : ''})</span>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); removeTerm(key); }}
                className="text-xs text-gray-500 hover:text-red-400"
              >
                Remove
              </button>
            </div>
            {isExpanded && (
              <div className="px-3 pb-3 space-y-2">
                {pairs.map((pair: string[], idx: number) => (
                  <div key={idx} className="flex items-center gap-2">
                    <input
                      value={pair[0] || ''}
                      onChange={(e) => updateVariant(key, idx, 0, e.target.value)}
                      placeholder="Regex pattern"
                      className="flex-1 px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-lg text-xs text-white font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
                    />
                    <input
                      value={pair[1] || ''}
                      onChange={(e) => updateVariant(key, idx, 1, e.target.value)}
                      placeholder="Display text"
                      className="flex-1 px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-lg text-xs text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                    />
                    <button onClick={() => removeVariant(key, idx)} className="text-gray-500 hover:text-red-400 text-xs">&times;</button>
                  </div>
                ))}
                <button
                  onClick={() => addVariant(key)}
                  className="text-xs text-blue-400 hover:text-blue-300"
                >
                  + Add variant pair
                </button>
              </div>
            )}
          </div>
        );
      })}
      {keys.length === 0 && <p className="text-xs text-gray-600 italic">No terminology variants defined</p>}
      <div className="flex gap-2 mt-2">
        <input
          value={newKey}
          onChange={(e) => setNewKey(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addTerm())}
          placeholder="New term key..."
          className="flex-1 px-3 py-1.5 bg-[#1a1a1a] border border-[#3a3a3a] rounded-xl text-xs text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <button onClick={addTerm} className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs transition-colors">Add Term</button>
      </div>
    </div>
  );
}


// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function LexiconTab({ apiCall, setMessage }: LexiconTabProps) {
  const [lexicon, setLexicon] = useState<Lexicon | null>(null);
  const [domainName, setDomainName] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // -------------------------------------------------------------------------
  // Fetch
  // -------------------------------------------------------------------------

  const fetchLexicon = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiCall('/api/admin/lexicon');
      setLexicon(data.lexicon);
      setDomainName(data.domain);
      setDisplayName(data.display_name);
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to load lexicon: ${e}` });
    } finally {
      setLoading(false);
    }
  }, [apiCall, setMessage]);

  useEffect(() => { fetchLexicon(); }, [fetchLexicon]);

  // -------------------------------------------------------------------------
  // Save
  // -------------------------------------------------------------------------

  const handleSave = async () => {
    if (!lexicon) return;
    setSaving(true);
    try {
      await apiCall('/api/admin/lexicon', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lexicon }),
      });
      setMessage({ type: 'success', text: 'Lexicon saved. Domain caches refreshed.' });
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to save lexicon: ${e}` });
    } finally {
      setSaving(false);
    }
  };

  // -------------------------------------------------------------------------
  // Helpers to mutate lexicon state
  // -------------------------------------------------------------------------

  const setField = (path: string[], value: unknown) => {
    if (!lexicon) return;
    const copy = JSON.parse(JSON.stringify(lexicon));
    let target = copy;
    for (let i = 0; i < path.length - 1; i++) {
      if (target[path[i]] === undefined) target[path[i]] = {};
      target = target[path[i]];
    }
    target[path[path.length - 1]] = value;
    setLexicon(copy);
  };

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <svg className="animate-spin h-5 w-5 text-blue-500 mr-3" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
        <span className="text-gray-400">Loading lexicon...</span>
      </div>
    );
  }

  if (!lexicon) {
    return <p className="text-gray-500 py-8 text-center">No lexicon found for the active domain.</p>;
  }

  const identity = lexicon.identity || {};
  const pipeline = lexicon.pipeline || {};
  const topics: string[] = lexicon.topics || [];
  const thinkingMessages: string[] = lexicon.thinking_messages || [];
  const specialties = lexicon.specialties || {};
  const terminologyVariants = lexicon.terminology_variants || {};
  const advancedTriggers: string[] = lexicon.advanced_triggers || [];
  const skipPatterns: string[] = lexicon.skip_patterns || [];
  const technicalPatterns: string[] = lexicon.technical_patterns || [];

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-medium">Domain Lexicon</h2>
          <p className="text-xs text-gray-500 mt-0.5">
            Editing <span className="text-blue-400">{displayName}</span> ({domainName}/lexicon.json)
          </p>
        </div>
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-5 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-xl text-sm font-medium transition-colors"
        >
          {saving ? 'Saving...' : 'Save Lexicon'}
        </button>
      </div>

      <div className="space-y-8">

        {/* ================================================================
            SECTION 1: Identity
        ================================================================ */}
        <section>
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3 border-b border-[#3a3a3a] pb-2">Identity</h3>
          <div className="space-y-4">
            <LexTextField
              label="Personality"
              tip="The system prompt personality injected into every LLM call. Defines how the assistant speaks and behaves."
              value={identity.personality || ''}
              onChange={(v) => setField(['identity', 'personality'], v)}
              multiline
            />
            <LexTextField
              label="Welcome Message"
              tip="Shown in the chat panel when a user starts a new conversation. Supports markdown."
              value={identity.welcome_message || ''}
              onChange={(v) => setField(['identity', 'welcome_message'], v)}
              multiline
            />
          </div>
        </section>

        {/* ================================================================
            SECTION 2: Pipeline Config
        ================================================================ */}
        <section>
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3 border-b border-[#3a3a3a] pb-2">Pipeline Config</h3>
          <p className="text-xs text-gray-500 mb-4">Controls how the RAG pipeline processes and retrieves domain knowledge. These values are injected into prompts and RAPTOR summaries.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <LexTextField
              label="Specialty"
              tip="Domain expertise area, inserted into RAG prompts. E.g. 'scientific and engineering disciplines'"
              value={pipeline.specialty || ''}
              onChange={(v) => setField(['pipeline', 'specialty'], v)}
            />
            <LexTextField
              label="Reference Type"
              tip="How the system describes source documents. E.g. 'a technical reference document'"
              value={pipeline.reference_type || ''}
              onChange={(v) => setField(['pipeline', 'reference_type'], v)}
            />
            <LexTextField
              label="RAPTOR Context"
              tip="Context label for RAPTOR hierarchical summaries. E.g. 'technical documentation'"
              value={pipeline.raptor_context || ''}
              onChange={(v) => setField(['pipeline', 'raptor_context'], v)}
            />
            <LexTextField
              label="RAPTOR Summary Intro"
              tip="Opening instruction for RAPTOR node summaries"
              value={pipeline.raptor_summary_intro || ''}
              onChange={(v) => setField(['pipeline', 'raptor_summary_intro'], v)}
            />
            <LexTextField
              label="Confidence Example"
              tip="Example confidence caveat shown in the response style prompt"
              value={pipeline.confidence_example || ''}
              onChange={(v) => setField(['pipeline', 'confidence_example'], v)}
            />
            <LexTextField
              label="Equation Example"
              tip="Example equation format to teach LaTeX rendering"
              value={pipeline.equation_example || ''}
              onChange={(v) => setField(['pipeline', 'equation_example'], v)}
            />
            <LexTextField
              label="Default Topic"
              tip="Default RAG topic when no specific routing is detected"
              value={pipeline.default_topic || ''}
              onChange={(v) => setField(['pipeline', 'default_topic'], v)}
            />
          </div>
          <div className="mt-4">
            <label className="text-sm text-gray-300">
              RAPTOR Preserve
              <SettingTip tip="List of content types that RAPTOR summaries must preserve when condensing text" />
            </label>
            <div className="mt-2">
              <ChipList
                items={pipeline.raptor_preserve || []}
                onAdd={(v) => setField(['pipeline', 'raptor_preserve'], [...(pipeline.raptor_preserve || []), v])}
                onRemove={(i) => {
                  const copy = [...(pipeline.raptor_preserve || [])];
                  copy.splice(i, 1);
                  setField(['pipeline', 'raptor_preserve'], copy);
                }}
                placeholder="Add preserve rule..."
              />
            </div>
          </div>
        </section>

        {/* ================================================================
            SECTION 3: Topics
        ================================================================ */}
        <section>
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3 border-b border-[#3a3a3a] pb-2">Topics</h3>
          <p className="text-xs text-gray-500 mb-3">Knowledge base topic categories. The tool router uses these for semantic routing.</p>
          <ChipList
            items={topics}
            onAdd={(v) => setField(['topics'], [...topics, v])}
            onRemove={(i) => {
              const copy = [...topics];
              copy.splice(i, 1);
              setField(['topics'], copy);
            }}
            placeholder="Add topic..."
          />
        </section>

        {/* ================================================================
            SECTION 4: Thinking Messages
        ================================================================ */}
        <section>
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3 border-b border-[#3a3a3a] pb-2">Thinking Messages</h3>
          <p className="text-xs text-gray-500 mb-3">Random phrases shown while the LLM is generating a response. Add personality-appropriate messages.</p>
          <ChipList
            items={thinkingMessages}
            onAdd={(v) => setField(['thinking_messages'], [...thinkingMessages, v])}
            onRemove={(i) => {
              const copy = [...thinkingMessages];
              copy.splice(i, 1);
              setField(['thinking_messages'], copy);
            }}
            placeholder="Add thinking message..."
          />
        </section>

        {/* ================================================================
            SECTION 5: Specialties
        ================================================================ */}
        <section>
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3 border-b border-[#3a3a3a] pb-2">Specialties</h3>
          <p className="text-xs text-gray-500 mb-3">Domain expertise areas with keywords for semantic routing. Each specialty has a description and a list of trigger keywords.</p>
          <SpecialtiesEditor
            specialties={specialties}
            onChange={(v) => setField(['specialties'], v)}
          />
        </section>

        {/* ================================================================
            SECTION 6: Terminology Variants
        ================================================================ */}
        <section>
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3 border-b border-[#3a3a3a] pb-2">Terminology Variants</h3>
          <p className="text-xs text-gray-500 mb-3">Maps term keys to regex pattern + display text pairs. Used for query expansion and synonym matching in the RAG pipeline.</p>
          <TerminologyEditor
            variants={terminologyVariants}
            onChange={(v) => setField(['terminology_variants'], v)}
          />
        </section>

        {/* ================================================================
            SECTION 7: Detection Patterns (Advanced / Collapsible)
        ================================================================ */}
        <section>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3 border-b border-[#3a3a3a] pb-2 w-full text-left hover:text-white transition-colors"
          >
            <span className="text-xs text-gray-500">{showAdvanced ? '\u25BC' : '\u25B6'}</span>
            Advanced: Detection Patterns
          </button>
          {showAdvanced && (
            <div className="space-y-6">
              <div>
                <label className="text-sm text-gray-300">
                  Advanced Triggers
                  <SettingTip tip="Keywords that route queries to domain-specific handlers instead of standard RAG search" />
                </label>
                <div className="mt-2">
                  <ChipList
                    items={advancedTriggers}
                    onAdd={(v) => setField(['advanced_triggers'], [...advancedTriggers, v])}
                    onRemove={(i) => {
                      const copy = [...advancedTriggers];
                      copy.splice(i, 1);
                      setField(['advanced_triggers'], copy);
                    }}
                    placeholder="Add trigger..."
                  />
                </div>
              </div>

              <div>
                <label className="text-sm text-gray-300">
                  Skip Patterns
                  <SettingTip tip="Keywords that prevent a query from being treated as a technical question (e.g. 'upload', 'extract')" />
                </label>
                <div className="mt-2">
                  <ChipList
                    items={skipPatterns}
                    onAdd={(v) => setField(['skip_patterns'], [...skipPatterns, v])}
                    onRemove={(i) => {
                      const copy = [...skipPatterns];
                      copy.splice(i, 1);
                      setField(['skip_patterns'], copy);
                    }}
                    placeholder="Add skip pattern..."
                  />
                </div>
              </div>

              <div>
                <label className="text-sm text-gray-300">
                  Technical Patterns
                  <SettingTip tip="Regex patterns that identify domain-specific technical questions. Matched against user queries to trigger RAG search." />
                </label>
                <p className="text-xs text-gray-600 mt-1 mb-2">These are regex patterns. Use \b for word boundaries. Each pattern is tested independently.</p>
                <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
                  {technicalPatterns.map((pat, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <input
                        value={pat}
                        onChange={(e) => {
                          const copy = [...technicalPatterns];
                          copy[i] = e.target.value;
                          setField(['technical_patterns'], copy);
                        }}
                        className="flex-1 px-2 py-1 bg-[#1a1a1a] border border-[#3a3a3a] rounded-lg text-xs text-white font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
                      />
                      <button
                        onClick={() => {
                          const copy = [...technicalPatterns];
                          copy.splice(i, 1);
                          setField(['technical_patterns'], copy);
                        }}
                        className="text-gray-500 hover:text-red-400 text-xs"
                      >
                        &times;
                      </button>
                    </div>
                  ))}
                </div>
                <button
                  onClick={() => setField(['technical_patterns'], [...technicalPatterns, '\\\\b()'])}
                  className="mt-2 text-xs text-blue-400 hover:text-blue-300"
                >
                  + Add pattern
                </button>
              </div>
            </div>
          )}
        </section>
      </div>

      {/* Bottom save bar */}
      <div className="mt-8 pt-4 border-t border-[#3a3a3a] flex justify-end">
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-5 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-xl text-sm font-medium transition-colors"
        >
          {saving ? 'Saving...' : 'Save Lexicon'}
        </button>
      </div>
    </div>
  );
}
