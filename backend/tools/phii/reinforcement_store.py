"""Phii Reinforcement store — PostgreSQL/SQLite storage for feedback and corrections."""

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from .reinforcement_types import FeedbackType, ActionSuggestion, _get_action_suggestions
from .reinforcement_classifier import ActionClassifier
from .reinforcement_cue import CueDetector
from .reinforcement_correction import Correction, CorrectionDetector

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """A feedback signal record."""

    id: Optional[int] = None
    timestamp: str = ""
    session_id: str = ""
    message_id: str = ""
    feedback_type: str = ""
    user_message: str = ""
    assistant_response: str = ""
    tools_used: str = "[]"  # JSON array
    personality: str = ""
    energy_profile: str = "{}"  # JSON object
    specialty_profile: str = "{}"  # JSON object
    admin_notes: str = ""
    resolved: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        d["tools_used"] = json.loads(d["tools_used"]) if d["tools_used"] else []
        d["energy_profile"] = json.loads(d["energy_profile"]) if d["energy_profile"] else {}
        d["specialty_profile"] = json.loads(d["specialty_profile"]) if d["specialty_profile"] else {}
        return d


class ReinforcementStore:
    """PostgreSQL/SQLite storage for feedback signals and learned corrections."""

    # Cache TTL for corrections (5 minutes)
    CORRECTIONS_CACHE_TTL = 300

    # Semantic similarity weight in hybrid scoring (0.0 = keyword only, 1.0 = semantic only)
    SEMANTIC_WEIGHT = 0.6

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize store.

        Args:
            db_path: Path to SQLite database (default: data/phii/reinforcement.sqlite)
        """
        if db_path is None:
            # Default path relative to backend directory
            backend_dir = Path(__file__).parent.parent.parent
            db_path = backend_dir / "data" / "phii" / "reinforcement.sqlite"

        self.db_path = db_path
        self._correction_detector = CorrectionDetector()
        self._action_classifier = ActionClassifier()

        # In-memory cache for corrections
        self._corrections_cache: List[Correction] = []
        self._corrections_cache_time: float = 0
        self._corrections_embeddings_cache: Dict[int, np.ndarray] = {}

        # Firm corrections cache (rarely changes, avoid repeated DB reads)
        self._firm_corrections_cache: List[dict] = []
        self._firm_corrections_cache_valid: bool = False

        # Session action memory for pattern tracking
        self._session_actions: Dict[str, List[str]] = {}

        # Lazy-loaded embedder (only load if needed for semantic matching)
        self._embedder = None

        # Database mode tracking
        self._use_postgres = False
        self._db_manager = None

        self._ensure_db()

    async def _get_db(self):
        """Get database manager for async operations."""
        if self._db_manager is None:
            from services.database import get_database
            self._db_manager = await get_database()
            self._use_postgres = self._db_manager.available
        return self._db_manager

    def _get_embedder(self):
        """Lazily load the embedder for semantic matching."""
        if self._embedder is None:
            try:
                from tools.cohesionn.embeddings import get_embedder

                self._embedder = get_embedder()
                logger.info("Phii semantic matching: embedder loaded")
            except Exception as e:
                logger.warning(f"Could not load embedder for semantic matching: {e}")
                return None
        return self._embedder

    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding for correction text."""
        embedder = self._get_embedder()
        if embedder is None:
            return None
        try:
            embedding = embedder.embed_document(text)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to compute embedding: {e}")
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a.shape != b.shape:
            logger.warning(f"Embedding shape mismatch: {a.shape} vs {b.shape}")
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            logger.warning(f"Near-zero embedding norm: ||a||={norm_a:.4f}, ||b||={norm_b:.4f}")
            return 0.0
        # Embeddings are already normalized by BGE, so dot product = cosine similarity
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _ensure_db(self):
        """Ensure SQLite database and tables exist (fallback)."""
        parent = self.db_path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # Legacy/broken mount case: parent exists but is not a directory.
            if parent.exists() and not parent.is_dir():
                ts = int(time.time())
                legacy_path = parent.with_name(f"{parent.name}.legacy-{ts}")
                try:
                    parent.rename(legacy_path)
                    logger.warning(
                        f"Phii DB parent path was a file, moved to {legacy_path}. Recreating directory."
                    )
                    parent.mkdir(parents=True, exist_ok=True)
                except Exception as migrate_err:
                    fallback_parent = Path("/tmp/arca/phii")
                    fallback_parent.mkdir(parents=True, exist_ok=True)
                    self.db_path = fallback_parent / self.db_path.name
                    logger.warning(
                        "Failed to migrate invalid Phii DB path "
                        f"({migrate_err}). Falling back to {self.db_path}"
                    )
            else:
                fallback_parent = Path("/tmp/arca/phii")
                fallback_parent.mkdir(parents=True, exist_ok=True)
                self.db_path = fallback_parent / self.db_path.name
                logger.warning(f"Could not prepare Phii DB directory. Falling back to {self.db_path}")

        with sqlite3.connect(self.db_path) as conn:
            # Feedback table (existing)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    user_message TEXT,
                    assistant_response TEXT,
                    tools_used TEXT,
                    personality TEXT,
                    energy_profile TEXT,
                    specialty_profile TEXT,
                    admin_notes TEXT,
                    resolved INTEGER DEFAULT 0
                )
            """
            )

            # Corrections table (new)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    ai_message_excerpt TEXT,
                    user_correction TEXT,
                    wrong_behavior TEXT NOT NULL,
                    right_behavior TEXT NOT NULL,
                    context_keywords TEXT,
                    confidence REAL DEFAULT 0.8,
                    times_applied INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    embedding BLOB
                )
            """
            )

            # Add embedding column if missing (migration for existing DBs)
            try:
                conn.execute("ALTER TABLE corrections ADD COLUMN embedding BLOB")
                logger.info("Added embedding column to corrections table")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Create indexes for common queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_type
                ON feedback(feedback_type, resolved)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_session
                ON feedback(session_id)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_recent
                ON feedback(timestamp DESC, resolved)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_corrections_active
                ON corrections(is_active) WHERE is_active = 1
            """
            )

            # Action log - tracks actions per session for pattern learning
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS action_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """
            )

            # Action patterns - learned transitions between actions
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS action_patterns (
                    prev_action TEXT NOT NULL,
                    next_action TEXT NOT NULL,
                    count INTEGER DEFAULT 1,
                    PRIMARY KEY (prev_action, next_action)
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_action_log_session
                ON action_log(session_id, timestamp)
            """
            )

            # Firm-wide corrections (seeded, applies to all users)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS firm_corrections (
                    id TEXT PRIMARY KEY,
                    wrong_behavior TEXT NOT NULL,
                    right_behavior TEXT NOT NULL,
                    context_keywords TEXT,
                    confidence REAL DEFAULT 0.9,
                    category TEXT,
                    is_active INTEGER DEFAULT 1
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_firm_corrections_active
                ON firm_corrections(is_active) WHERE is_active = 1
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_firm_corrections_category
                ON firm_corrections(category)
            """
            )

            # Firm terminology standards
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS firm_terminology (
                    concept TEXT PRIMARY KEY,
                    preferred_term TEXT NOT NULL,
                    context TEXT DEFAULT 'formal'
                )
            """
            )

            # User profiles - persistent cross-session preferences
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    expertise_level TEXT DEFAULT 'intermediate',
                    expertise_confidence REAL DEFAULT 0.5,
                    preferred_units TEXT DEFAULT 'metric',
                    preferred_format TEXT,
                    verbosity_preference REAL DEFAULT 0.0,
                    technical_depth REAL DEFAULT 0.5,
                    specialties TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    session_count INTEGER DEFAULT 1
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_profiles_updated
                ON user_profiles(updated_at)
            """
            )

            # Correction applications - tracks when corrections are applied for feedback loop
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS correction_applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    correction_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    feedback_received TEXT,
                    FOREIGN KEY (correction_id) REFERENCES corrections(id)
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_correction_applications_session
                ON correction_applications(session_id, applied_at)
            """
            )

            conn.commit()

    # =========================================================================
    # ASYNC POSTGRESQL METHODS
    # =========================================================================

    async def add_feedback_async(
        self,
        session_id: str,
        message_id: str,
        feedback_type: FeedbackType,
        user_message: str = "",
        assistant_response: str = "",
        tools_used: List[str] = None,
        personality: str = "",
        energy_profile: Dict[str, Any] = None,
        specialty_profile: Dict[str, Any] = None,
    ) -> int:
        """Add a feedback record (async PostgreSQL)."""
        db = await self._get_db()

        if db.fallback_mode:
            return self.add_feedback(
                session_id, message_id, feedback_type, user_message,
                assistant_response, tools_used, personality, energy_profile, specialty_profile
            )

        result = await db.fetch_with_returning(
            """
            INSERT INTO feedback (
                session_id, message_id, feedback_type,
                user_message, assistant_response, tools_used,
                personality, energy_profile, specialty_profile,
                admin_notes, resolved
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, '', FALSE)
            RETURNING id
            """,
            session_id,
            message_id,
            feedback_type.value,
            user_message,
            assistant_response,
            json.dumps(tools_used or []),
            personality,
            json.dumps(energy_profile or {}),
            json.dumps(specialty_profile or {}),
        )
        return result["id"]

    async def get_flags_async(self, resolved: bool = False, limit: int = 50) -> List[FeedbackRecord]:
        """Get flagged responses (async PostgreSQL)."""
        db = await self._get_db()

        if db.fallback_mode:
            return self.get_flags(resolved=resolved, limit=limit)

        if resolved:
            rows = await db.fetch(
                """
                SELECT * FROM feedback
                WHERE feedback_type = 'flag'
                ORDER BY timestamp DESC
                LIMIT $1
                """,
                limit,
            )
        else:
            rows = await db.fetch(
                """
                SELECT * FROM feedback
                WHERE feedback_type = 'flag' AND resolved = FALSE
                ORDER BY timestamp DESC
                LIMIT $1
                """,
                limit,
            )

        return [self._dict_to_record(row) for row in rows]

    async def resolve_flag_async(self, flag_id: int, notes: str = "") -> bool:
        """Mark a flag as resolved (async PostgreSQL)."""
        db = await self._get_db()

        if db.fallback_mode:
            return self.resolve_flag(flag_id, notes)

        result = await db.execute(
            """
            UPDATE feedback
            SET resolved = TRUE, admin_notes = $1
            WHERE id = $2 AND feedback_type = 'flag'
            """,
            notes, flag_id,
        )
        return "UPDATE 1" in result

    async def get_stats_async(self) -> Dict[str, Any]:
        """Get feedback statistics (async PostgreSQL)."""
        db = await self._get_db()

        if db.fallback_mode:
            return self.get_stats()

        counts = {}
        for ftype in FeedbackType:
            count = await db.fetchval(
                "SELECT COUNT(*) FROM feedback WHERE feedback_type = $1",
                ftype.value,
            )
            counts[ftype.value] = count

        unresolved_flags = await db.fetchval(
            "SELECT COUNT(*) FROM feedback WHERE feedback_type = 'flag' AND resolved = FALSE"
        )

        recent_24h = await db.fetchval(
            "SELECT COUNT(*) FROM feedback WHERE timestamp > NOW() - INTERVAL '1 day'"
        )

        return {
            "counts": counts,
            "unresolved_flags": unresolved_flags,
            "recent_24h": recent_24h,
            "total": sum(counts.values()),
        }

    async def store_correction_async(self, correction: Correction, session_id: str = "") -> int:
        """Store a learned correction (async PostgreSQL)."""
        db = await self._get_db()

        if db.fallback_mode:
            return self.store_correction(correction, session_id)

        # Compute embedding for semantic matching
        embed_text = f"{correction.wrong_behavior} {correction.right_behavior}".strip()
        embedding = self._compute_embedding(embed_text) if embed_text else None
        embedding_bytes = embedding.tobytes() if embedding is not None else None

        result = await db.fetch_with_returning(
            """
            INSERT INTO corrections (
                timestamp, session_id, ai_message_excerpt,
                user_correction, wrong_behavior, right_behavior,
                context_keywords, confidence, times_applied, is_active,
                embedding
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 0, TRUE, $9)
            RETURNING id
            """,
            datetime.fromisoformat(correction.timestamp) if isinstance(correction.timestamp, str) else (correction.timestamp or datetime.now()),
            session_id,
            correction.ai_message_excerpt,
            correction.user_correction,
            correction.wrong_behavior,
            correction.right_behavior,
            json.dumps(correction.context_keywords),
            correction.confidence,
            embedding_bytes,
        )

        # Invalidate cache
        self._corrections_cache_time = 0
        self._corrections_embeddings_cache.clear()

        return result["id"]

    async def get_relevant_corrections_async(self, message: str, top_k: int = 3) -> List[Correction]:
        """Get corrections relevant to the current message (async)."""
        db = await self._get_db()

        if db.fallback_mode:
            return self.get_relevant_corrections(message, top_k)

        # Get firm corrections (baseline)
        firm_corrections = await self._get_firm_corrections_async(message, top_k=2)

        # Get learned corrections
        learned_corrections = await self._get_learned_corrections_async(message, top_k=2)

        return self._merge_corrections(firm_corrections, learned_corrections, max_total=top_k)

    async def _get_learned_corrections_async(self, message: str, top_k: int = 2) -> List[Correction]:
        """Get learned corrections (async PostgreSQL)."""
        db = await self._get_db()

        rows = await db.fetch(
            "SELECT * FROM corrections WHERE is_active = TRUE ORDER BY timestamp DESC LIMIT 100"
        )

        if not rows:
            return []

        corrections = [self._dict_to_correction(row) for row in rows]
        message_words = set(re.findall(r"\b[a-z]{3,}\b", message.lower()))

        # Compute message embedding for semantic scoring
        message_embedding = None
        embedder = self._get_embedder()
        if embedder is not None:
            try:
                message_embedding = np.array(embedder.embed_query(message), dtype=np.float32)
            except Exception as e:
                logger.warning(f"Failed to embed message: {e}")

        scored = []
        for correction in corrections:
            if not correction.is_active:
                continue

            # Keyword scoring
            correction_keywords = set(correction.context_keywords)
            overlap = len(message_words & correction_keywords)

            keyword_score = overlap
            if correction.wrong_behavior:
                wrong_words = set(correction.wrong_behavior.lower().split())
                if wrong_words & message_words:
                    keyword_score += 2.0
            if correction.right_behavior:
                right_words = set(correction.right_behavior.lower().split())
                if right_words & message_words:
                    keyword_score += 1.0

            keyword_score_normalized = min(keyword_score / 5.0, 1.0)

            # Semantic scoring
            semantic_score = 0.0
            if message_embedding is not None and correction.id in self._corrections_embeddings_cache:
                correction_embedding = self._corrections_embeddings_cache[correction.id]
                semantic_score = max(0.0, self._cosine_similarity(message_embedding, correction_embedding))

            # Hybrid score
            if message_embedding is not None and correction.id in self._corrections_embeddings_cache:
                score = (
                    self.SEMANTIC_WEIGHT * semantic_score
                    + (1 - self.SEMANTIC_WEIGHT) * keyword_score_normalized
                )
            else:
                score = keyword_score_normalized

            score += correction.confidence * 0.1

            if score > 0.1:
                scored.append((score, correction))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    async def _get_firm_corrections_async(self, message: str, top_k: int = 2) -> List[Correction]:
        """Get firm-wide corrections (async PostgreSQL)."""
        db = await self._get_db()
        message_words = set(re.findall(r"\b[a-z]{3,}\b", message.lower()))

        rows = await db.fetch("SELECT * FROM firm_corrections WHERE is_active = TRUE")

        if not rows:
            return []

        scored = []
        for row in rows:
            keywords_json = row.get("context_keywords")
            if keywords_json:
                try:
                    keywords = set(json.loads(keywords_json) if isinstance(keywords_json, str) else keywords_json)
                except (json.JSONDecodeError, TypeError):
                    keywords = set()
            else:
                keywords = set()

            overlap = len(message_words & keywords)
            score = overlap + (row["confidence"] * 0.5)

            wrong = row["wrong_behavior"].lower()
            right = row["right_behavior"].lower()

            if set(wrong.split()) & message_words:
                score += 1.0
            if set(right.split()) & message_words:
                score += 0.5

            if score > 0:
                correction = Correction(
                    id=None,
                    timestamp="",
                    session_id="",
                    ai_message_excerpt="",
                    user_correction="",
                    wrong_behavior=row["wrong_behavior"],
                    right_behavior=row["right_behavior"],
                    context_keywords=list(keywords),
                    confidence=row["confidence"],
                    times_applied=0,
                    is_active=True,
                )
                scored.append((score, correction))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    async def get_all_corrections_async(self, active_only: bool = True, limit: int = 50) -> List[Correction]:
        """Get all corrections (async PostgreSQL)."""
        db = await self._get_db()

        if db.fallback_mode:
            return self.get_all_corrections(active_only=active_only, limit=limit)

        if active_only:
            rows = await db.fetch(
                "SELECT * FROM corrections WHERE is_active = TRUE ORDER BY timestamp DESC LIMIT $1",
                limit,
            )
        else:
            rows = await db.fetch(
                "SELECT * FROM corrections ORDER BY timestamp DESC LIMIT $1",
                limit,
            )

        return [self._dict_to_correction(row) for row in rows]

    async def delete_correction_async(self, correction_id: int) -> bool:
        """Delete a correction (async PostgreSQL)."""
        db = await self._get_db()

        if db.fallback_mode:
            return self.delete_correction(correction_id)

        result = await db.execute(
            "UPDATE corrections SET is_active = FALSE WHERE id = $1",
            correction_id,
        )

        self._corrections_cache_time = 0
        self._corrections_embeddings_cache.pop(correction_id, None)

        return "UPDATE 1" in result

    async def get_corrections_stats_async(self) -> Dict[str, Any]:
        """Get statistics about learned corrections (async PostgreSQL)."""
        db = await self._get_db()

        if db.fallback_mode:
            return self.get_corrections_stats()

        active_count = await db.fetchval(
            "SELECT COUNT(*) FROM corrections WHERE is_active = TRUE"
        )
        total_applied = await db.fetchval(
            "SELECT COALESCE(SUM(times_applied), 0) FROM corrections WHERE is_active = TRUE"
        )
        recent_7d = await db.fetchval(
            "SELECT COUNT(*) FROM corrections WHERE is_active = TRUE AND timestamp > NOW() - INTERVAL '7 days'"
        )

        return {
            "active_count": active_count,
            "total_applied": total_applied,
            "recent_7d": recent_7d,
        }

    async def record_action_async(self, session_id: str, action: str) -> None:
        """Record an action and update patterns (async PostgreSQL)."""
        db = await self._get_db()

        if db.fallback_mode:
            self.record_action(session_id, action)
            return

        # Track in session memory
        if session_id not in self._session_actions:
            self._session_actions[session_id] = []

        prev_actions = self._session_actions[session_id]

        await db.execute(
            "INSERT INTO action_log (session_id, action, timestamp) VALUES ($1, $2, $3)",
            session_id, action, datetime.now(),
        )

        if prev_actions:
            prev_action = prev_actions[-1]
            await db.execute(
                """
                INSERT INTO action_patterns (prev_action, next_action, count)
                VALUES ($1, $2, 1)
                ON CONFLICT (prev_action, next_action)
                DO UPDATE SET count = action_patterns.count + 1
                """,
                prev_action, action,
            )

        prev_actions.append(action)
        if len(prev_actions) > 10:
            self._session_actions[session_id] = prev_actions[-10:]

    async def predict_next_action_async(self, session_id: str, min_confidence: float = 0.4) -> Optional[ActionSuggestion]:
        """Predict likely next action (async PostgreSQL)."""
        db = await self._get_db()

        if db.fallback_mode:
            return self.predict_next_action(session_id, min_confidence)

        if session_id not in self._session_actions:
            return None

        actions = self._session_actions[session_id]
        if not actions:
            return None

        last_action = actions[-1]

        rows = await db.fetch(
            """
            SELECT next_action, count FROM action_patterns
            WHERE prev_action = $1
            ORDER BY count DESC
            LIMIT 3
            """,
            last_action,
        )

        if not rows:
            return _get_action_suggestions().get(last_action)

        total = sum(row["count"] for row in rows)
        top_action, top_count = rows[0]["next_action"], rows[0]["count"]
        confidence = top_count / total

        if confidence >= min_confidence:
            suggestions = _get_action_suggestions()
            if top_action in suggestions:
                suggestion = suggestions[top_action]
                return ActionSuggestion(action=top_action, message=suggestion.message, confidence=confidence)
            else:
                readable_action = top_action.replace("_", " ")
                return ActionSuggestion(
                    action=top_action,
                    message=f"Based on your workflow, you might want to proceed with {readable_action}.",
                    confidence=confidence,
                )

        return None

    async def get_pattern_stats_async(self) -> Dict[str, Any]:
        """Get statistics about learned patterns (async PostgreSQL)."""
        db = await self._get_db()

        if db.fallback_mode:
            return self.get_pattern_stats()

        pattern_count = await db.fetchval("SELECT COUNT(*) FROM action_patterns")
        action_count = await db.fetchval("SELECT COUNT(*) FROM action_log")

        rows = await db.fetch(
            """
            SELECT prev_action, next_action, count
            FROM action_patterns
            ORDER BY count DESC
            LIMIT 5
            """
        )
        top_patterns = [{"from": r["prev_action"], "to": r["next_action"], "count": r["count"]} for r in rows]

        return {
            "pattern_count": pattern_count,
            "action_count": action_count,
            "top_patterns": top_patterns,
        }

    async def get_user_profile_async(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get persistent user profile (async PostgreSQL)."""
        db = await self._get_db()

        if db.fallback_mode:
            return self.get_user_profile(user_id)

        row = await db.fetchrow(
            "SELECT * FROM user_profiles WHERE user_id = $1",
            user_id,
        )

        if row is None:
            return None

        specialties = row.get("specialties")
        if specialties:
            if isinstance(specialties, str):
                try:
                    specialties = json.loads(specialties)
                except json.JSONDecodeError:
                    specialties = []

        return {
            "user_id": row["user_id"],
            "expertise_level": row["expertise_level"],
            "expertise_confidence": row["expertise_confidence"],
            "preferred_units": row["preferred_units"],
            "preferred_format": row["preferred_format"],
            "verbosity_preference": row["verbosity_preference"],
            "technical_depth": row["technical_depth"],
            "specialties": specialties or [],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "session_count": row["session_count"],
        }

    async def save_user_profile_async(
        self,
        user_id: str,
        expertise_level: str = None,
        expertise_confidence: float = None,
        preferred_units: str = None,
        preferred_format: str = None,
        verbosity_preference: float = None,
        technical_depth: float = None,
        specialties: List[str] = None,
    ) -> None:
        """Save or update user profile (async PostgreSQL)."""
        db = await self._get_db()

        if db.fallback_mode:
            self.save_user_profile(
                user_id, expertise_level, expertise_confidence,
                preferred_units, preferred_format, verbosity_preference,
                technical_depth, specialties
            )
            return

        existing = await db.fetchrow(
            "SELECT user_id FROM user_profiles WHERE user_id = $1",
            user_id,
        )

        # Defense-in-depth: whitelist of allowed profile columns for dynamic UPDATE
        _PROFILE_COLUMNS = {
            'expertise_level', 'expertise_confidence', 'preferred_units',
            'preferred_format', 'verbosity_preference', 'technical_depth',
            'specialties', 'session_count',
        }

        if existing:
            # Build dynamic update (column names are hardcoded below, not from user input)
            updates = []
            params = []
            param_count = 1

            if expertise_level is not None:
                updates.append(f"expertise_level = ${param_count}")
                params.append(expertise_level)
                param_count += 1
            if expertise_confidence is not None:
                updates.append(f"expertise_confidence = ${param_count}")
                params.append(expertise_confidence)
                param_count += 1
            if preferred_units is not None:
                updates.append(f"preferred_units = ${param_count}")
                params.append(preferred_units)
                param_count += 1
            if preferred_format is not None:
                updates.append(f"preferred_format = ${param_count}")
                params.append(preferred_format)
                param_count += 1
            if verbosity_preference is not None:
                updates.append(f"verbosity_preference = ${param_count}")
                params.append(verbosity_preference)
                param_count += 1
            if technical_depth is not None:
                updates.append(f"technical_depth = ${param_count}")
                params.append(technical_depth)
                param_count += 1
            if specialties is not None:
                updates.append(f"specialties = ${param_count}")
                params.append(json.dumps(specialties))
                param_count += 1

            updates.append("session_count = session_count + 1")
            params.append(user_id)

            # Validate all column names against whitelist before executing
            for u in updates:
                col = u.split("=")[0].strip()
                if col not in _PROFILE_COLUMNS:
                    raise ValueError(f"Invalid profile column: {col}")

            await db.execute(
                f"UPDATE user_profiles SET {', '.join(updates)} WHERE user_id = ${param_count}",
                *params,
            )
        else:
            await db.execute(
                """
                INSERT INTO user_profiles (
                    user_id, expertise_level, expertise_confidence,
                    preferred_units, preferred_format, verbosity_preference,
                    technical_depth, specialties, session_count
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 1)
                """,
                user_id,
                expertise_level or "intermediate",
                expertise_confidence or 0.5,
                preferred_units or "metric",
                preferred_format,
                verbosity_preference or 0.0,
                technical_depth or 0.5,
                json.dumps(specialties) if specialties else None,
            )

    def _dict_to_record(self, row: Dict[str, Any]) -> FeedbackRecord:
        """Convert dict to FeedbackRecord."""
        tools_used = row.get("tools_used", "[]")
        if isinstance(tools_used, list):
            tools_used = json.dumps(tools_used)
        energy_profile = row.get("energy_profile", "{}")
        if isinstance(energy_profile, dict):
            energy_profile = json.dumps(energy_profile)
        specialty_profile = row.get("specialty_profile", "{}")
        if isinstance(specialty_profile, dict):
            specialty_profile = json.dumps(specialty_profile)

        return FeedbackRecord(
            id=row["id"],
            timestamp=str(row["timestamp"]),
            session_id=row["session_id"],
            message_id=row["message_id"],
            feedback_type=row["feedback_type"],
            user_message=row.get("user_message") or "",
            assistant_response=row.get("assistant_response") or "",
            tools_used=tools_used,
            personality=row.get("personality") or "",
            energy_profile=energy_profile,
            specialty_profile=specialty_profile,
            admin_notes=row.get("admin_notes") or "",
            resolved=bool(row.get("resolved")),
        )

    def _dict_to_correction(self, row: Dict[str, Any]) -> Correction:
        """Convert dict to Correction."""
        keywords = row.get("context_keywords")
        if keywords:
            if isinstance(keywords, str):
                try:
                    keywords = json.loads(keywords)
                except json.JSONDecodeError:
                    keywords = []
            elif isinstance(keywords, list):
                pass
            else:
                keywords = []
        else:
            keywords = []

        return Correction(
            id=row["id"],
            timestamp=str(row["timestamp"]),
            session_id=row.get("session_id") or "",
            ai_message_excerpt=row.get("ai_message_excerpt") or "",
            user_correction=row.get("user_correction") or "",
            wrong_behavior=row["wrong_behavior"],
            right_behavior=row["right_behavior"],
            context_keywords=keywords,
            confidence=row["confidence"],
            times_applied=row["times_applied"],
            is_active=bool(row["is_active"]),
        )

    # =========================================================================
    # SYNC SQLITE METHODS (Fallback)
    # =========================================================================

    def add_feedback(
        self,
        session_id: str,
        message_id: str,
        feedback_type: FeedbackType,
        user_message: str = "",
        assistant_response: str = "",
        tools_used: List[str] = None,
        personality: str = "",
        energy_profile: Dict[str, Any] = None,
        specialty_profile: Dict[str, Any] = None,
    ) -> int:
        """Add a feedback record (sync SQLite)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback (
                    timestamp, session_id, message_id, feedback_type,
                    user_message, assistant_response, tools_used,
                    personality, energy_profile, specialty_profile,
                    admin_notes, resolved
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    session_id,
                    message_id,
                    feedback_type.value,
                    user_message,
                    assistant_response,
                    json.dumps(tools_used or []),
                    personality,
                    json.dumps(energy_profile or {}),
                    json.dumps(specialty_profile or {}),
                    "",
                    0,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_flags(self, resolved: bool = False, limit: int = 50) -> List[FeedbackRecord]:
        """Get flagged responses (sync SQLite)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if resolved:
                rows = conn.execute(
                    """
                    SELECT * FROM feedback
                    WHERE feedback_type = 'flag'
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM feedback
                    WHERE feedback_type = 'flag' AND resolved = 0
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (limit,),
                ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def resolve_flag(self, flag_id: int, notes: str = "") -> bool:
        """Mark a flag as resolved (sync SQLite)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE feedback
                SET resolved = 1, admin_notes = ?
                WHERE id = ? AND feedback_type = 'flag'
            """,
                (notes, flag_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics (sync SQLite)."""
        with sqlite3.connect(self.db_path) as conn:
            counts = {}
            for ftype in FeedbackType:
                row = conn.execute(
                    "SELECT COUNT(*) FROM feedback WHERE feedback_type = ?",
                    (ftype.value,),
                ).fetchone()
                counts[ftype.value] = row[0]

            row = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE feedback_type = 'flag' AND resolved = 0"
            ).fetchone()
            unresolved_flags = row[0]

            row = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE datetime(timestamp) > datetime('now', '-1 day')"
            ).fetchone()
            recent_24h = row[0]

            return {
                "counts": counts,
                "unresolved_flags": unresolved_flags,
                "recent_24h": recent_24h,
                "total": sum(counts.values()),
            }

    def _row_to_record(self, row: sqlite3.Row) -> FeedbackRecord:
        """Convert database row to FeedbackRecord."""
        return FeedbackRecord(
            id=row["id"],
            timestamp=row["timestamp"],
            session_id=row["session_id"],
            message_id=row["message_id"],
            feedback_type=row["feedback_type"],
            user_message=row["user_message"] or "",
            assistant_response=row["assistant_response"] or "",
            tools_used=row["tools_used"] or "[]",
            personality=row["personality"] or "",
            energy_profile=row["energy_profile"] or "{}",
            specialty_profile=row["specialty_profile"] or "{}",
            admin_notes=row["admin_notes"] or "",
            resolved=bool(row["resolved"]),
        )

    # =========================================================================
    # CORRECTION METHODS (Sync SQLite)
    # =========================================================================

    def detect_correction(self, user_message: str, ai_message: str = "") -> Optional[Correction]:
        """Detect if a user message contains a correction."""
        return self._correction_detector.detect(user_message, ai_message)

    def store_correction(self, correction: Correction, session_id: str = "") -> int:
        """Store a learned correction (sync SQLite)."""
        embed_text = f"{correction.wrong_behavior} {correction.right_behavior}".strip()
        embedding = self._compute_embedding(embed_text) if embed_text else None
        embedding_blob = embedding.tobytes() if embedding is not None else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO corrections (
                    timestamp, session_id, ai_message_excerpt,
                    user_correction, wrong_behavior, right_behavior,
                    context_keywords, confidence, times_applied, is_active,
                    embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    correction.timestamp or datetime.now().isoformat(),
                    session_id,
                    correction.ai_message_excerpt,
                    correction.user_correction,
                    correction.wrong_behavior,
                    correction.right_behavior,
                    json.dumps(correction.context_keywords),
                    correction.confidence,
                    0,
                    1,
                    embedding_blob,
                ),
            )
            conn.commit()

            self._corrections_cache_time = 0
            self._corrections_embeddings_cache.clear()

            return cursor.lastrowid

    def get_relevant_corrections(self, message: str, top_k: int = 3) -> List[Correction]:
        """Get corrections relevant to the current message."""
        firm_corrections = self._get_firm_corrections(message, top_k=2)
        learned_corrections = self._get_learned_corrections(message, top_k=2)
        return self._merge_corrections(firm_corrections, learned_corrections, max_total=top_k)

    def _get_learned_corrections(self, message: str, top_k: int = 2) -> List[Correction]:
        """Get learned corrections from user feedback."""
        self._refresh_corrections_cache()

        if not self._corrections_cache:
            return []

        message_words = set(re.findall(r"\b[a-z]{3,}\b", message.lower()))

        message_embedding = None
        embedder = self._get_embedder()
        if embedder is not None:
            try:
                message_embedding = np.array(embedder.embed_query(message), dtype=np.float32)
            except Exception as e:
                logger.warning(f"Failed to embed message for semantic matching: {e}")

        self._load_correction_embeddings()

        scored = []
        for correction in self._corrections_cache:
            if not correction.is_active:
                continue

            correction_keywords = set(correction.context_keywords)
            overlap = len(message_words & correction_keywords)

            keyword_score = overlap
            if correction.wrong_behavior:
                wrong_words = set(correction.wrong_behavior.lower().split())
                if wrong_words & message_words:
                    keyword_score += 2.0
            if correction.right_behavior:
                right_words = set(correction.right_behavior.lower().split())
                if right_words & message_words:
                    keyword_score += 1.0

            keyword_score_normalized = min(keyword_score / 5.0, 1.0)

            semantic_score = 0.0
            if message_embedding is not None and correction.id in self._corrections_embeddings_cache:
                correction_embedding = self._corrections_embeddings_cache[correction.id]
                semantic_score = self._cosine_similarity(message_embedding, correction_embedding)
                semantic_score = max(0.0, semantic_score)

            if message_embedding is not None and correction.id in self._corrections_embeddings_cache:
                score = (
                    self.SEMANTIC_WEIGHT * semantic_score
                    + (1 - self.SEMANTIC_WEIGHT) * keyword_score_normalized
                )
            else:
                score = keyword_score_normalized

            score += correction.confidence * 0.1

            if score > 0.1:
                scored.append((score, correction))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def _load_correction_embeddings(self) -> None:
        """Load correction embeddings into cache from DB."""
        if self._corrections_embeddings_cache:
            return

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM corrections WHERE is_active = 1 AND embedding IS NOT NULL"
            ).fetchall()

        for row in rows:
            correction_id = row[0]
            embedding_blob = row[1]
            if embedding_blob:
                try:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    self._corrections_embeddings_cache[correction_id] = embedding
                except Exception as e:
                    logger.warning(f"Failed to load embedding for correction {correction_id}: {e}")

    def _get_firm_corrections(self, message: str, top_k: int = 2) -> List[Correction]:
        """Get firm-wide corrections relevant to the message."""
        message_words = set(re.findall(r"\b[a-z]{3,}\b", message.lower()))

        if not self._firm_corrections_cache_valid:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM firm_corrections WHERE is_active = 1"
                ).fetchall()
                self._firm_corrections_cache = [dict(row) for row in rows]
                self._firm_corrections_cache_valid = True

        rows = self._firm_corrections_cache
        if not rows:
            return []

        scored = []
        for row in rows:
            keywords_json = row["context_keywords"]
            if keywords_json:
                try:
                    keywords = set(json.loads(keywords_json))
                except json.JSONDecodeError:
                    keywords = set()
            else:
                keywords = set()

            overlap = len(message_words & keywords)
            score = overlap + (row["confidence"] * 0.5)

            wrong = row["wrong_behavior"].lower()
            right = row["right_behavior"].lower()

            if set(wrong.split()) & message_words:
                score += 1.0
            if set(right.split()) & message_words:
                score += 0.5

            if score > 0:
                correction = Correction(
                    id=None,
                    timestamp="",
                    session_id="",
                    ai_message_excerpt="",
                    user_correction="",
                    wrong_behavior=row["wrong_behavior"],
                    right_behavior=row["right_behavior"],
                    context_keywords=list(keywords),
                    confidence=row["confidence"],
                    times_applied=0,
                    is_active=True,
                )
                scored.append((score, correction))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def _merge_corrections(
        self, firm: List[Correction], learned: List[Correction], max_total: int = 3
    ) -> List[Correction]:
        """Merge firm and learned corrections."""
        result = []

        for fc in firm:
            if len(result) >= max_total:
                break
            result.append(fc)

        for lc in learned:
            if len(result) >= max_total:
                break

            is_duplicate = False
            for existing in result:
                existing_words = set(existing.right_behavior.lower().split())
                lc_words = set(lc.right_behavior.lower().split())
                overlap = len(existing_words & lc_words)
                if overlap >= 3:
                    is_duplicate = True
                    break

            if not is_duplicate:
                result.append(lc)

        return result

    def get_all_corrections(self, active_only: bool = True, limit: int = 50) -> List[Correction]:
        """Get all corrections (sync SQLite)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if active_only:
                rows = conn.execute(
                    "SELECT * FROM corrections WHERE is_active = 1 ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM corrections ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()

        return [self._row_to_correction(row) for row in rows]

    def delete_correction(self, correction_id: int) -> bool:
        """Delete a correction (sync SQLite)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE corrections SET is_active = 0 WHERE id = ?",
                (correction_id,),
            )
            conn.commit()

            self._corrections_cache_time = 0
            self._corrections_embeddings_cache.pop(correction_id, None)

            return cursor.rowcount > 0

    def backfill_embeddings(self) -> int:
        """Backfill embeddings for corrections that don't have them."""
        embedder = self._get_embedder()
        if embedder is None:
            logger.warning("Cannot backfill embeddings: embedder not available")
            return 0

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, wrong_behavior, right_behavior FROM corrections WHERE is_active = 1 AND embedding IS NULL"
            ).fetchall()

            if not rows:
                logger.info("No corrections need embedding backfill")
                return 0

            updated = 0
            for row in rows:
                correction_id = row[0]
                wrong = row[1] or ""
                right = row[2] or ""
                embed_text = f"{wrong} {right}".strip()

                if not embed_text:
                    continue

                embedding = self._compute_embedding(embed_text)
                if embedding is not None:
                    conn.execute(
                        "UPDATE corrections SET embedding = ? WHERE id = ?",
                        (embedding.tobytes(), correction_id),
                    )
                    updated += 1

            conn.commit()

        logger.info(f"Backfilled embeddings for {updated} corrections")
        self._corrections_embeddings_cache.clear()
        return updated

    def increment_applied(self, correction_id: int, session_id: str = "", message_id: str = "") -> None:
        """Increment the times_applied counter for a correction."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE corrections SET times_applied = times_applied + 1 WHERE id = ?",
                (correction_id,),
            )

            if session_id:
                conn.execute(
                    """
                    INSERT INTO correction_applications (
                        correction_id, session_id, message_id, applied_at
                    ) VALUES (?, ?, ?, ?)
                """,
                    (correction_id, session_id, message_id or "", datetime.now().isoformat()),
                )

            conn.commit()

    def process_correction_feedback(
        self,
        session_id: str,
        feedback_type: FeedbackType,
        lookback_seconds: int = 300,
    ) -> int:
        """Process feedback and adjust confidence for recently applied corrections."""
        POSITIVE_BOOST = 0.05
        NEGATIVE_PENALTY = 0.15
        MIN_CONFIDENCE = 0.3

        cutoff_time = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT correction_id FROM correction_applications
                WHERE session_id = ?
                AND datetime(applied_at) > datetime(?, '-' || ? || ' seconds')
                AND feedback_received IS NULL
            """,
                (session_id, cutoff_time, lookback_seconds),
            ).fetchall()

            if not rows:
                return 0

            correction_ids = [row[0] for row in rows]

            feedback_value = feedback_type.value
            for cid in correction_ids:
                conn.execute(
                    """
                    UPDATE correction_applications
                    SET feedback_received = ?
                    WHERE correction_id = ? AND session_id = ? AND feedback_received IS NULL
                """,
                    (feedback_value, cid, session_id),
                )

            if feedback_type == FeedbackType.POSITIVE:
                conn.execute(
                    f"""
                    UPDATE corrections
                    SET confidence = MIN(1.0, confidence + {POSITIVE_BOOST})
                    WHERE id IN ({','.join('?' * len(correction_ids))})
                """,
                    correction_ids,
                )
            elif feedback_type == FeedbackType.NEGATIVE:
                conn.execute(
                    f"""
                    UPDATE corrections
                    SET confidence = MAX(0.0, confidence - {NEGATIVE_PENALTY})
                    WHERE id IN ({','.join('?' * len(correction_ids))})
                """,
                    correction_ids,
                )

                conn.execute(
                    f"""
                    UPDATE corrections
                    SET is_active = 0
                    WHERE id IN ({','.join('?' * len(correction_ids))})
                    AND confidence < {MIN_CONFIDENCE}
                """,
                    correction_ids,
                )

                deactivated = conn.execute(
                    f"""
                    SELECT id FROM corrections
                    WHERE id IN ({','.join('?' * len(correction_ids))})
                    AND is_active = 0
                """,
                    correction_ids,
                ).fetchall()

                if deactivated:
                    logger.info(
                        f"Deactivated {len(deactivated)} corrections due to low confidence: "
                        f"{[d[0] for d in deactivated]}"
                    )

            conn.commit()

        self._corrections_cache_time = 0
        self._corrections_embeddings_cache.clear()

        logger.debug(f"Adjusted confidence for {len(correction_ids)} corrections ({feedback_type.value})")
        return len(correction_ids)

    async def process_correction_feedback_async(
        self,
        session_id: str,
        feedback_type: FeedbackType,
        lookback_seconds: int = 300,
    ) -> int:
        """Process feedback and adjust confidence for recently applied corrections (async PostgreSQL)."""
        db = await self._get_db()

        if db.fallback_mode:
            return self.process_correction_feedback(session_id, feedback_type, lookback_seconds)

        POSITIVE_BOOST = 0.05
        NEGATIVE_PENALTY = 0.15
        MIN_CONFIDENCE = 0.3

        rows = await db.fetch(
            """
            SELECT DISTINCT correction_id FROM correction_applications
            WHERE session_id = $1
            AND applied_at > NOW() - make_interval(secs => $2)
            AND feedback_received IS NULL
            """,
            session_id, lookback_seconds,
        )

        if not rows:
            return 0

        correction_ids = [row["correction_id"] for row in rows]

        feedback_value = feedback_type.value
        for cid in correction_ids:
            await db.execute(
                """
                UPDATE correction_applications
                SET feedback_received = $1
                WHERE correction_id = $2 AND session_id = $3 AND feedback_received IS NULL
                """,
                feedback_value, cid, session_id,
            )

        if feedback_type == FeedbackType.POSITIVE:
            for cid in correction_ids:
                await db.execute(
                    "UPDATE corrections SET confidence = LEAST(1.0, confidence + $1) WHERE id = $2",
                    POSITIVE_BOOST, cid,
                )
        elif feedback_type == FeedbackType.NEGATIVE:
            for cid in correction_ids:
                await db.execute(
                    "UPDATE corrections SET confidence = confidence - $1 WHERE id = $2",
                    NEGATIVE_PENALTY, cid,
                )
                await db.execute(
                    "UPDATE corrections SET is_active = FALSE WHERE id = $1 AND confidence < $2",
                    cid, MIN_CONFIDENCE,
                )

        self._corrections_cache_time = 0
        self._corrections_embeddings_cache.clear()

        logger.debug(f"Adjusted confidence for {len(correction_ids)} corrections ({feedback_type.value})")
        return len(correction_ids)

    def get_corrections_stats(self) -> Dict[str, Any]:
        """Get statistics about learned corrections (sync SQLite)."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM corrections WHERE is_active = 1"
            ).fetchone()
            active_count = row[0]

            row = conn.execute(
                "SELECT COALESCE(SUM(times_applied), 0) FROM corrections WHERE is_active = 1"
            ).fetchone()
            total_applied = row[0]

            row = conn.execute(
                """
                SELECT COUNT(*) FROM corrections
                WHERE is_active = 1
                AND datetime(timestamp) > datetime('now', '-7 days')
            """
            ).fetchone()
            recent_7d = row[0]

            return {
                "active_count": active_count,
                "total_applied": total_applied,
                "recent_7d": recent_7d,
            }

    def _refresh_corrections_cache(self) -> None:
        """Refresh the corrections cache if expired."""
        now = time.time()
        if now - self._corrections_cache_time < self.CORRECTIONS_CACHE_TTL:
            return

        self._corrections_cache = self.get_all_corrections(active_only=True, limit=100)
        self._corrections_embeddings_cache.clear()
        self._corrections_cache_time = now
        logger.debug(f"Refreshed corrections cache: {len(self._corrections_cache)} corrections")

    def _row_to_correction(self, row: sqlite3.Row) -> Correction:
        """Convert database row to Correction."""
        keywords = row["context_keywords"]
        if keywords:
            try:
                keywords = json.loads(keywords)
            except json.JSONDecodeError:
                keywords = []
        else:
            keywords = []

        return Correction(
            id=row["id"],
            timestamp=row["timestamp"],
            session_id=row["session_id"] or "",
            ai_message_excerpt=row["ai_message_excerpt"] or "",
            user_correction=row["user_correction"] or "",
            wrong_behavior=row["wrong_behavior"],
            right_behavior=row["right_behavior"],
            context_keywords=keywords,
            confidence=row["confidence"],
            times_applied=row["times_applied"],
            is_active=bool(row["is_active"]),
        )

    # =========================================================================
    # ACTION PATTERN METHODS (Sync SQLite)
    # =========================================================================

    def record_action(self, session_id: str, action: str) -> None:
        """Record an action and update patterns (sync SQLite)."""
        if session_id not in self._session_actions:
            self._session_actions[session_id] = []

        prev_actions = self._session_actions[session_id]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO action_log (session_id, action, timestamp) VALUES (?, ?, ?)",
                (session_id, action, datetime.now().isoformat()),
            )

            if prev_actions:
                prev_action = prev_actions[-1]
                conn.execute(
                    """
                    INSERT INTO action_patterns (prev_action, next_action, count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(prev_action, next_action)
                    DO UPDATE SET count = count + 1
                """,
                    (prev_action, action),
                )

            conn.commit()

        prev_actions.append(action)
        if len(prev_actions) > 10:
            self._session_actions[session_id] = prev_actions[-10:]

    def predict_next_action(self, session_id: str, min_confidence: float = 0.4) -> Optional[ActionSuggestion]:
        """Predict likely next action based on patterns (sync SQLite)."""
        if session_id not in self._session_actions:
            return None

        actions = self._session_actions[session_id]
        if not actions:
            return None

        last_action = actions[-1]

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT next_action, count FROM action_patterns
                WHERE prev_action = ?
                ORDER BY count DESC
                LIMIT 3
            """,
                (last_action,),
            ).fetchall()

        if not rows:
            return _get_action_suggestions().get(last_action)

        total = sum(row[1] for row in rows)
        top_action, top_count = rows[0]
        confidence = top_count / total

        if confidence >= min_confidence:
            suggestions = _get_action_suggestions()
            if top_action in suggestions:
                suggestion = suggestions[top_action]
                return ActionSuggestion(action=top_action, message=suggestion.message, confidence=confidence)
            else:
                readable_action = top_action.replace("_", " ")
                return ActionSuggestion(
                    action=top_action,
                    message=f"Based on your workflow, you might want to proceed with {readable_action}.",
                    confidence=confidence,
                )

        return None

    def classify_action(self, message: str, tools_used: List[str]) -> Optional[str]:
        """Classify an exchange into a canonical action."""
        return self._action_classifier.classify(message, tools_used)

    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about learned patterns (sync SQLite)."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM action_patterns").fetchone()
            pattern_count = row[0]

            row = conn.execute("SELECT COUNT(*) FROM action_log").fetchone()
            action_count = row[0]

            rows = conn.execute(
                """
                SELECT prev_action, next_action, count
                FROM action_patterns
                ORDER BY count DESC
                LIMIT 5
            """
            ).fetchall()
            top_patterns = [{"from": r[0], "to": r[1], "count": r[2]} for r in rows]

        return {
            "pattern_count": pattern_count,
            "action_count": action_count,
            "top_patterns": top_patterns,
        }

    def clear_session_actions(self, session_id: str) -> None:
        """Clear session action memory."""
        self._session_actions.pop(session_id, None)

    # =========================================================================
    # USER PROFILE METHODS (Sync SQLite)
    # =========================================================================

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get persistent user profile (sync SQLite)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()

        if row is None:
            return None

        specialties = row["specialties"]
        if specialties:
            try:
                specialties = json.loads(specialties)
            except json.JSONDecodeError:
                specialties = []

        return {
            "user_id": row["user_id"],
            "expertise_level": row["expertise_level"],
            "expertise_confidence": row["expertise_confidence"],
            "preferred_units": row["preferred_units"],
            "preferred_format": row["preferred_format"],
            "verbosity_preference": row["verbosity_preference"],
            "technical_depth": row["technical_depth"],
            "specialties": specialties or [],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "session_count": row["session_count"],
        }

    def save_user_profile(
        self,
        user_id: str,
        expertise_level: str = None,
        expertise_confidence: float = None,
        preferred_units: str = None,
        preferred_format: str = None,
        verbosity_preference: float = None,
        technical_depth: float = None,
        specialties: List[str] = None,
    ) -> None:
        """Save or update user profile (sync SQLite)."""
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT user_id FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()

            # Defense-in-depth: whitelist of allowed profile columns
            _PROFILE_COLS = {
                'expertise_level', 'expertise_confidence', 'preferred_units',
                'preferred_format', 'verbosity_preference', 'technical_depth',
                'specialties', 'updated_at', 'session_count',
            }

            if existing:
                updates = []
                params = []

                if expertise_level is not None:
                    updates.append("expertise_level = ?")
                    params.append(expertise_level)
                if expertise_confidence is not None:
                    updates.append("expertise_confidence = ?")
                    params.append(expertise_confidence)
                if preferred_units is not None:
                    updates.append("preferred_units = ?")
                    params.append(preferred_units)
                if preferred_format is not None:
                    updates.append("preferred_format = ?")
                    params.append(preferred_format)
                if verbosity_preference is not None:
                    updates.append("verbosity_preference = ?")
                    params.append(verbosity_preference)
                if technical_depth is not None:
                    updates.append("technical_depth = ?")
                    params.append(technical_depth)
                if specialties is not None:
                    updates.append("specialties = ?")
                    params.append(json.dumps(specialties))

                updates.append("updated_at = ?")
                params.append(now)
                updates.append("session_count = session_count + 1")

                # Validate column names against whitelist
                for u in updates:
                    col = u.split("=")[0].strip()
                    if col not in _PROFILE_COLS:
                        raise ValueError(f"Invalid profile column: {col}")

                params.append(user_id)

                conn.execute(
                    f"UPDATE user_profiles SET {', '.join(updates)} WHERE user_id = ?",
                    params,
                )
            else:
                conn.execute(
                    """
                    INSERT INTO user_profiles (
                        user_id, expertise_level, expertise_confidence,
                        preferred_units, preferred_format, verbosity_preference,
                        technical_depth, specialties, created_at, updated_at, session_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                    (
                        user_id,
                        expertise_level or "intermediate",
                        expertise_confidence or 0.5,
                        preferred_units or "metric",
                        preferred_format,
                        verbosity_preference or 0.0,
                        technical_depth or 0.5,
                        json.dumps(specialties) if specialties else None,
                        now,
                        now,
                    ),
                )

            conn.commit()

    def merge_session_to_profile(
        self,
        user_id: str,
        expertise_level: str,
        expertise_confidence: float,
        verbosity_preference: float,
        technical_depth: float,
        specialties: List[str],
        decay_factor: float = 0.3,
    ) -> None:
        """Merge session data into persistent profile with exponential smoothing."""
        existing = self.get_user_profile(user_id)

        if existing:
            new_confidence = (
                decay_factor * expertise_confidence + (1 - decay_factor) * existing["expertise_confidence"]
            )
            new_verbosity = (
                decay_factor * verbosity_preference + (1 - decay_factor) * existing["verbosity_preference"]
            )
            new_technical = decay_factor * technical_depth + (1 - decay_factor) * existing["technical_depth"]

            existing_specs = existing.get("specialties") or []
            merged_specs = list(set(existing_specs + specialties))[:5]

            if expertise_confidence > existing["expertise_confidence"]:
                new_expertise_level = expertise_level
            else:
                new_expertise_level = existing["expertise_level"]

            self.save_user_profile(
                user_id=user_id,
                expertise_level=new_expertise_level,
                expertise_confidence=new_confidence,
                verbosity_preference=new_verbosity,
                technical_depth=new_technical,
                specialties=merged_specs,
            )
        else:
            self.save_user_profile(
                user_id=user_id,
                expertise_level=expertise_level,
                expertise_confidence=expertise_confidence,
                verbosity_preference=verbosity_preference,
                technical_depth=technical_depth,
                specialties=specialties,
            )
