"""
Retrieval Profile Manager for ARCA.

Manages named retrieval profiles ("fast", "deep") that control which
pipeline stages are active. Provides override hierarchy:
  per-query param > manual override > active profile > env/config default

Usage:
    from profile_loader import get_profile_manager
    pm = get_profile_manager()
    toggles = pm.resolve_for_query(profile_override="deep", bm25_enabled=True)
"""

import json
import logging
import shutil
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Toggle keys that profiles are allowed to control.
# Prevents profiles from touching non-toggle settings (model names, thresholds, etc.)
PROFILE_TOGGLE_KEYS = frozenset([
    "query_expansion_enabled",
    "hyde_enabled",
    "bm25_enabled",
    "raptor_enabled",
    "graph_rag_enabled",
    "global_search_enabled",
    "reranker_enabled",
    "domain_boost_enabled",
    "rag_diversity_enabled",
    "crag_enabled",
    "vision_ingest_enabled",
])

# Bundled default profile file (ships with source, copied on first run)
_BUNDLED_PATH = Path(__file__).parent / "data" / "config" / "retrieval_profiles.json"

# Runtime location (Docker volume or local data dir)
_RUNTIME_PATH = Path(
    __import__("os").environ.get(
        "RETRIEVAL_PROFILES_PATH",
        "/app/data/config/retrieval_profiles.json",
    )
)


class ProfileManager:
    """
    Singleton manager for retrieval profiles.

    Loads profile definitions from JSON, tracks the active profile,
    manages manual overrides, and resolves toggles for each query.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._default_profile: str = "fast"
        self._active_profile: str = "fast"
        self._manual_overrides: Dict[str, bool] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load profiles from JSON file. Copies bundled default if missing."""
        if self._loaded:
            return

        path = _RUNTIME_PATH

        # Copy bundled file to runtime location if missing
        if not path.exists() and _BUNDLED_PATH.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(_BUNDLED_PATH, path)
                logger.info(f"Copied bundled profiles to {path}")
            except Exception as e:
                logger.warning(f"Could not copy bundled profiles: {e}")
                path = _BUNDLED_PATH  # Fall back to bundled

        if not path.exists():
            logger.info("No retrieval_profiles.json found, using built-in defaults")
            self._profiles = self._hardcoded_defaults()
            self._loaded = True
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._profiles = data.get("profiles", {})
            self._default_profile = data.get("default_profile", "fast")
            self._active_profile = self._default_profile
            self._loaded = True
            logger.info(
                f"Loaded {len(self._profiles)} retrieval profiles "
                f"(default: {self._default_profile})"
            )
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
            self._profiles = self._hardcoded_defaults()
            self._loaded = True

    @staticmethod
    def _hardcoded_defaults() -> Dict[str, Dict[str, Any]]:
        """Fallback profiles when JSON is unavailable."""
        return {
            "fast": {
                "display_name": "Fast",
                "description": "Low-latency retrieval. Dense + reranking core.",
                "toggles": {
                    "query_expansion_enabled": True,
                    "hyde_enabled": False,
                    "bm25_enabled": False,
                    "raptor_enabled": False,
                    "graph_rag_enabled": False,
                    "global_search_enabled": False,
                    "reranker_enabled": True,
                    "domain_boost_enabled": True,
                    "rag_diversity_enabled": True,
                    "crag_enabled": False,
                    "vision_ingest_enabled": False,
                },
            },
            "auto": {
                "display_name": "Auto",
                "description": "Optimized by benchmark. Toggles set by the last benchmark run.",
                "toggles": {
                    "query_expansion_enabled": True,
                    "hyde_enabled": True,
                    "bm25_enabled": False,
                    "raptor_enabled": True,
                    "graph_rag_enabled": True,
                    "global_search_enabled": True,
                    "reranker_enabled": True,
                    "domain_boost_enabled": True,
                    "rag_diversity_enabled": True,
                    "crag_enabled": False,
                    "vision_ingest_enabled": False,
                },
            },
            "deep": {
                "display_name": "Deep",
                "description": "Full pipeline. Maximum coverage at higher latency.",
                "toggles": {
                    "query_expansion_enabled": True,
                    "hyde_enabled": True,
                    "bm25_enabled": True,
                    "raptor_enabled": True,
                    "graph_rag_enabled": True,
                    "global_search_enabled": True,
                    "reranker_enabled": True,
                    "domain_boost_enabled": True,
                    "rag_diversity_enabled": True,
                    "crag_enabled": False,
                    "vision_ingest_enabled": True,
                },
            },
        }

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_profiles(self) -> List[Dict[str, Any]]:
        """Return metadata for all profiles (for API/UI)."""
        self._ensure_loaded()
        result = []
        for name, profile in self._profiles.items():
            result.append({
                "name": name,
                "display_name": profile.get("display_name", name.title()),
                "description": profile.get("description", ""),
                "is_active": name == self._active_profile,
                "is_default": name == self._default_profile,
            })
        return result

    def get_profile(self, name: str) -> Optional[Dict[str, bool]]:
        """Get toggle dict for a named profile. Returns None if not found."""
        self._ensure_loaded()
        profile = self._profiles.get(name)
        if profile is None:
            return None
        return dict(profile.get("toggles", {}))

    @property
    def active_profile(self) -> str:
        self._ensure_loaded()
        return self._active_profile

    @property
    def manual_overrides(self) -> Dict[str, bool]:
        return dict(self._manual_overrides)

    @property
    def has_manual_overrides(self) -> bool:
        return len(self._manual_overrides) > 0

    def set_active(self, name: str) -> Dict[str, Any]:
        """
        Switch to a named profile. Clears manual overrides and applies
        toggle values to RuntimeConfig.

        Returns dict of changes made.
        """
        self._ensure_loaded()
        if name not in self._profiles:
            raise ValueError(f"Unknown profile: {name}. Available: {list(self._profiles.keys())}")

        with self._lock:
            self._active_profile = name
            self._manual_overrides.clear()

        toggles = self.get_profile(name) or {}
        changes = self._apply_toggles_to_config(toggles)
        logger.info(f"Switched to profile '{name}', applied {len(changes)} toggles")
        return {"profile": name, "changes": changes}

    def set_manual_override(self, key: str, value: bool) -> None:
        """Set an individual toggle override on top of the active profile."""
        if key not in PROFILE_TOGGLE_KEYS:
            raise ValueError(f"Key '{key}' is not a profile toggle. Valid: {sorted(PROFILE_TOGGLE_KEYS)}")
        with self._lock:
            self._manual_overrides[key] = value
        # Apply to config immediately
        from config import runtime_config
        if hasattr(runtime_config, key):
            setattr(runtime_config, key, value)
        logger.debug(f"Manual override: {key}={value}")

    def clear_manual_overrides(self) -> None:
        """Clear all manual overrides and reapply active profile."""
        with self._lock:
            self._manual_overrides.clear()
        # Reapply active profile
        toggles = self.get_profile(self._active_profile) or {}
        self._apply_toggles_to_config(toggles)
        logger.info(f"Cleared manual overrides, reapplied profile '{self._active_profile}'")

    def get_active_toggles(self) -> Dict[str, bool]:
        """Get effective toggles: profile + manual overrides merged."""
        self._ensure_loaded()
        toggles = self.get_profile(self._active_profile) or {}
        # Manual overrides win
        toggles.update(self._manual_overrides)
        return toggles

    def resolve_for_query(
        self,
        profile_override: Optional[str] = None,
        **per_query_toggles: Optional[bool],
    ) -> Dict[str, bool]:
        """
        Resolve all toggles for a single query.

        Override hierarchy (most specific wins):
        1. Per-query parameter (passed explicitly to retrieve())
        2. Manual toggle override (user toggled BM25 ON while on "fast")
        3. Active profile ("fast" says BM25=off)
        4. Env/config default (fallback)

        Args:
            profile_override: Use this profile instead of active (e.g. "deep" for deep search)
            **per_query_toggles: Individual toggle overrides for this query.
                Pass None to use profile default, True/False to override.

        Returns:
            Dict of all toggle keys → resolved bool values
        """
        self._ensure_loaded()

        # Start with config defaults (layer 4)
        from config import runtime_config
        result: Dict[str, bool] = {}
        for key in PROFILE_TOGGLE_KEYS:
            if hasattr(runtime_config, key):
                result[key] = getattr(runtime_config, key)

        # Apply profile (layer 3)
        profile_name = profile_override or self._active_profile
        profile_toggles = self.get_profile(profile_name)
        if profile_toggles:
            result.update(profile_toggles)

        # Apply manual overrides (layer 2) — only when using active profile
        if profile_override is None:
            result.update(self._manual_overrides)

        # Apply per-query overrides (layer 1)
        for key, value in per_query_toggles.items():
            if value is not None and key in PROFILE_TOGGLE_KEYS:
                result[key] = value

        return result

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def load_state(self, profile_name: str, overrides: Dict[str, bool]) -> None:
        """Restore state from persisted config (called during startup)."""
        self._ensure_loaded()
        if profile_name in self._profiles:
            self._active_profile = profile_name
        elif profile_name == "custom":
            # Migrated config — keep whatever toggles are in RuntimeConfig
            self._active_profile = "custom"
        self._manual_overrides = {
            k: v for k, v in overrides.items()
            if k in PROFILE_TOGGLE_KEYS
        }

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Get state dict for saving to config_overrides.json."""
        return {
            "retrieval_profile": self._active_profile,
            "manual_overrides": dict(self._manual_overrides),
        }

    def update_profile_toggles(
        self, profile_name: str, toggles: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Update a profile's toggles in memory and persist to JSON file.

        Used by the benchmark Apply Winners flow to write optimal settings
        to the 'auto' profile.

        Returns dict of changes made.
        """
        self._ensure_loaded()
        if profile_name not in self._profiles:
            raise ValueError(
                "Unknown profile: " + profile_name
                + ". Available: " + str(list(self._profiles.keys()))
            )

        profile = self._profiles[profile_name]
        current = profile.get("toggles", {})
        changes = {}

        with self._lock:
            for key, value in toggles.items():
                if key not in PROFILE_TOGGLE_KEYS:
                    continue
                old = current.get(key)
                if old != value:
                    current[key] = value
                    changes[key] = {"old": old, "new": value}
            profile["toggles"] = current

        # Persist to JSON file
        self._save_profiles()

        logger.info(
            "Updated profile '" + profile_name + "' toggles: "
            + str(len(changes)) + " changes"
        )
        return changes

    def _save_profiles(self) -> None:
        """Write current profiles to the JSON file."""
        path = _RUNTIME_PATH
        try:
            data = {
                "_schema_version": 1,
                "profiles": self._profiles,
                "default_profile": self._default_profile,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            logger.info("Persisted profiles to " + str(path))
        except Exception as e:
            logger.error("Failed to save profiles: " + str(e))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_toggles_to_config(self, toggles: Dict[str, bool]) -> Dict[str, Any]:
        """Apply toggle values to RuntimeConfig via setattr. Returns changes dict."""
        from config import runtime_config

        changes = {}
        for key, value in toggles.items():
            if key not in PROFILE_TOGGLE_KEYS:
                continue
            if hasattr(runtime_config, key):
                old = getattr(runtime_config, key)
                if old != value:
                    setattr(runtime_config, key, value)
                    changes[key] = {"old": old, "new": value}
        return changes


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_profile_manager: Optional[ProfileManager] = None


def get_profile_manager() -> ProfileManager:
    """Get the singleton ProfileManager instance."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager()
    return _profile_manager
