-- PostgreSQL Schema for ARCA Learning Databases
-- Migrated from SQLite (Phii Reinforcement + Logg Learning)
--
-- Run automatically on PostgreSQL container init via docker-entrypoint-initdb.d

-- =============================================================================
-- USER ACCOUNTS
-- =============================================================================

-- Users table - multi-user authentication (replaces single admin key)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'admin',
    settings JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- =============================================================================
-- PHII REINFORCEMENT TABLES
-- =============================================================================

-- Feedback table - stores explicit flags and implicit cues
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    feedback_type TEXT NOT NULL,
    user_message TEXT,
    assistant_response TEXT,
    tools_used JSONB DEFAULT '[]'::jsonb,
    personality TEXT,
    energy_profile JSONB DEFAULT '{}'::jsonb,
    specialty_profile JSONB DEFAULT '{}'::jsonb,
    admin_notes TEXT,
    resolved BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type, resolved);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp DESC);

-- Corrections table - learned corrections from user feedback
CREATE TABLE IF NOT EXISTS corrections (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id TEXT,
    ai_message_excerpt TEXT,
    user_correction TEXT,
    wrong_behavior TEXT NOT NULL,
    right_behavior TEXT NOT NULL,
    context_keywords JSONB DEFAULT '[]'::jsonb,
    confidence REAL DEFAULT 0.8,
    times_applied INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    embedding BYTEA  -- Vector embedding for semantic matching
);

CREATE INDEX IF NOT EXISTS idx_corrections_active ON corrections(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_corrections_timestamp ON corrections(timestamp DESC);

-- Action log - tracks actions per session for pattern learning
CREATE TABLE IF NOT EXISTS action_log (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    action TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_action_log_session ON action_log(session_id, timestamp);

-- Action patterns - learned transitions between actions
CREATE TABLE IF NOT EXISTS action_patterns (
    prev_action TEXT NOT NULL,
    next_action TEXT NOT NULL,
    count INTEGER DEFAULT 1,
    PRIMARY KEY (prev_action, next_action)
);

-- Firm-wide corrections (seeded, applies to all users)
CREATE TABLE IF NOT EXISTS firm_corrections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    wrong_behavior TEXT NOT NULL,
    right_behavior TEXT NOT NULL,
    context_keywords JSONB DEFAULT '[]'::jsonb,
    confidence REAL DEFAULT 0.9,
    category TEXT,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_firm_corrections_active ON firm_corrections(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_firm_corrections_category ON firm_corrections(category);

-- Firm terminology standards
CREATE TABLE IF NOT EXISTS firm_terminology (
    concept TEXT PRIMARY KEY,
    preferred_term TEXT NOT NULL,
    context TEXT DEFAULT 'formal'
);

-- User profiles - persistent cross-session preferences
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    expertise_level TEXT DEFAULT 'intermediate',
    expertise_confidence REAL DEFAULT 0.5,
    preferred_units TEXT DEFAULT 'metric',
    preferred_format TEXT,
    verbosity_preference REAL DEFAULT 0.0,
    technical_depth REAL DEFAULT 0.5,
    specialties JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_count INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_updated ON user_profiles(updated_at);

-- Correction applications - tracks when corrections are applied for feedback loop
CREATE TABLE IF NOT EXISTS correction_applications (
    id SERIAL PRIMARY KEY,
    correction_id INTEGER NOT NULL REFERENCES corrections(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    feedback_received TEXT
);

CREATE INDEX IF NOT EXISTS idx_correction_applications_session ON correction_applications(session_id, applied_at);
CREATE INDEX IF NOT EXISTS idx_correction_applications_correction ON correction_applications(correction_id);

-- =============================================================================
-- LOGG LEARNING TABLES
-- =============================================================================

-- Logg corrections table - vision extraction corrections
CREATE TABLE IF NOT EXISTS logg_corrections (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    field_type TEXT NOT NULL,
    raw_extracted TEXT NOT NULL,
    corrected_value TEXT NOT NULL,
    context_keywords JSONB,
    confidence REAL DEFAULT 0.8,
    times_applied INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_logg_corrections_lookup ON logg_corrections(field_type, raw_extracted);
CREATE INDEX IF NOT EXISTS idx_logg_corrections_active ON logg_corrections(is_active) WHERE is_active = TRUE;

-- Logg abbreviations table - domain abbreviations (seed + learned)
CREATE TABLE IF NOT EXISTS logg_abbreviations (
    abbreviation TEXT PRIMARY KEY,
    expansion TEXT NOT NULL,
    source TEXT DEFAULT 'learned',
    times_seen INTEGER DEFAULT 1,
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_logg_abbreviations_source ON logg_abbreviations(source);

-- Domain-specific seed data is loaded by the active domain pack on first boot

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for user_profiles
DROP TRIGGER IF EXISTS update_user_profiles_updated_at ON user_profiles;
CREATE TRIGGER update_user_profiles_updated_at
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for users
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
