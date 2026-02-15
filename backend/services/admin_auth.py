"""
Admin Authentication Service for ARCA — Multi-User Accounts.

Provides bcrypt password hashing and JWT session tokens for the admin panel.
Users are stored in PostgreSQL with file-based fallback for bootstrap.

Features:
- Multi-user accounts stored in PostgreSQL `users` table
- Bcrypt-hashed passwords, JWT HS256 tokens with 8h expiry
- First-run: user creates admin account via /admin (no auto-migration)
- First registered user is always admin, subsequent users are 'user' role
- File-based fallback when PostgreSQL is unavailable
- ADMIN_RESET env var for recovery (deletes all users, forces re-setup)
"""

import json
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# JWT config
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 8

# File-based storage path (fallback + JWT secret persistence)
AUTH_DIR = Path(__file__).parent.parent / "data" / "auth"
AUTH_FILE = AUTH_DIR / "admin_auth.json"

# Optional Bearer token extractor (doesn't auto-raise on missing)
_bearer_scheme = HTTPBearer(auto_error=False)


class AdminAuthManager:
    """
    Multi-user authentication manager.

    Primary storage: PostgreSQL `users` table.
    Fallback: JSON file on disk (for bootstrap when Postgres is down).
    JWT secret persisted in the auth file so tokens survive restarts.
    """

    _instance: Optional["AdminAuthManager"] = None

    def __init__(self):
        self._jwt_secret: Optional[str] = None
        self._setup_required: bool = True
        self._db_available: bool = False
        # Fallback file-based admin (used when Postgres is down)
        self._fallback_hash: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "AdminAuthManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self) -> None:
        """
        Called during app lifespan startup (synchronous).

        1. Load or create JWT secret from auth file
        2. Check for ADMIN_RESET=true
        3. Mark setup_required based on whether users exist
           (actual user count check happens async in initialize_async)
        """
        AUTH_DIR.mkdir(parents=True, exist_ok=True)

        # Handle reset
        admin_reset = os.environ.get("ADMIN_RESET", "false").lower() == "true"
        if admin_reset:
            if AUTH_FILE.exists():
                AUTH_FILE.unlink()
            self._setup_required = True
            logger.warning("ADMIN_RESET=true: auth state cleared, setup required")

        # Load JWT secret (or create new one)
        if AUTH_FILE.exists():
            try:
                data = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
                self._jwt_secret = data.get("jwt_secret")
                self._fallback_hash = data.get("password_hash")
                # fallback_hash is only used for X-Admin-Key legacy auth,
                # NOT for bypassing setup — users always create accounts via UI
            except Exception as e:
                logger.error(f"Failed to load auth file: {e}")

        if not self._jwt_secret:
            self._jwt_secret = secrets.token_hex(32)
            self._save_auth_file()

        # We'll determine setup_required properly in initialize_async
        logger.info("Admin auth manager initialized (async init pending)")

    async def initialize_async(self) -> None:
        """
        Async initialization — checks PostgreSQL for existing users.
        If no users exist, marks setup as required (user creates account via UI).
        """
        try:
            from services.database import get_database
            db = await get_database()

            if not db.available:
                logger.info(
                    "PostgreSQL unavailable for user accounts. "
                    "Using file-based fallback for bootstrap admin."
                )
                self._db_available = False
                # Can't check user count — setup_required stays True (safe default)
                return

            self._db_available = True

            # Ensure users table exists (in case init.sql hasn't run yet)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'admin',
                    settings JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            # Add settings column if missing (existing databases)
            await db.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'users' AND column_name = 'settings'
                    ) THEN
                        ALTER TABLE users ADD COLUMN settings JSONB DEFAULT '{}'::jsonb;
                    END IF;
                END $$;
            """)

            # Check for ADMIN_RESET — clear all users
            admin_reset = os.environ.get("ADMIN_RESET", "false").lower() == "true"
            if admin_reset:
                await db.execute("DELETE FROM users")
                logger.warning("ADMIN_RESET=true: all users deleted")

            # Count existing users
            count = await db.fetchval("SELECT COUNT(*) FROM users")

            if count == 0:
                self._setup_required = True
                logger.info("No users found. Setup required via admin panel.")
            else:
                self._setup_required = False
                logger.info(f"Admin auth ready: {count} user(s) in database")

        except Exception as e:
            logger.warning(f"Async auth init failed: {e}. File-based fallback active.")
            self._db_available = False

    @property
    def setup_required(self) -> bool:
        return self._setup_required

    @property
    def db_available(self) -> bool:
        return self._db_available

    # =========================================================================
    # User Management (PostgreSQL)
    # =========================================================================

    async def register_user(self, username: str, password: str) -> dict:
        """
        Register a new user. First user auto-becomes admin.

        Returns:
            Dict with user_id, username, role.

        Raises:
            ValueError if username taken or password too short.
            RuntimeError if database unavailable.
        """
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        if len(username) < 2 or len(username) > 50:
            raise ValueError("Username must be 2-50 characters")

        # Validate username format (alphanumeric, underscores, hyphens)
        import re
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', username):
            raise ValueError(
                "Username must start with a letter and contain only "
                "letters, numbers, underscores, and hyphens"
            )

        password_hash = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(rounds=12),
        ).decode("utf-8")

        if self._db_available:
            from services.database import get_database
            db = await get_database()
            if not db.available:
                logger.warning("PostgreSQL entered fallback mode during register; using file-based fallback")
                self._db_available = False
                return await self.register_user(username, password)

            # Check if this is the first user
            count = await db.fetchval("SELECT COUNT(*) FROM users")
            role = "admin" if count == 0 else "user"

            try:
                row = await db.fetch_with_returning(
                    "INSERT INTO users (username, password_hash, role) "
                    "VALUES ($1, $2, $3) RETURNING id, username, role",
                    username, password_hash, role,
                )
            except Exception as e:
                if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                    raise ValueError(f"Username '{username}' is already taken")
                raise

            self._setup_required = False

            # Also save to fallback file (first user only, for bootstrap)
            if count == 0:
                self._fallback_hash = password_hash
                self._save_auth_file()

            return row
        else:
            # Fallback: file-based (only for first/bootstrap admin)
            if self._fallback_hash:
                raise ValueError(
                    "File-based fallback only supports one admin account. "
                    "PostgreSQL is required for multi-user accounts."
                )

            self._fallback_hash = password_hash
            self._setup_required = False
            self._save_auth_file()
            return {"id": 0, "username": username, "role": "admin"}

    async def verify_login(self, username: str, password: str) -> Optional[dict]:
        """
        Verify username/password. Returns user dict or None.

        Returns:
            {id, username, role} on success, None on failure.
        """
        if self._db_available:
            from services.database import get_database
            db = await get_database()
            if not db.available:
                self._db_available = False
                return await self.verify_login(username, password)

            user = await db.fetchrow(
                "SELECT id, username, password_hash, role FROM users WHERE username = $1",
                username,
            )
            if not user:
                return None

            try:
                if bcrypt.checkpw(
                    password.encode("utf-8"),
                    user["password_hash"].encode("utf-8"),
                ):
                    return {"id": user["id"], "username": user["username"], "role": user["role"]}
            except Exception:
                pass
            return None
        else:
            # Fallback: file-based single admin
            if not self._fallback_hash:
                return None
            try:
                if bcrypt.checkpw(
                    password.encode("utf-8"),
                    self._fallback_hash.encode("utf-8"),
                ):
                    return {"id": 0, "username": username, "role": "admin"}
            except Exception:
                pass
            return None

    async def change_password(self, user_id: int, old_password: str, new_password: str) -> bool:
        """Change a user's password. Returns True on success."""
        if len(new_password) < 8:
            raise ValueError("New password must be at least 8 characters")

        if self._db_available:
            from services.database import get_database
            db = await get_database()
            if not db.available:
                self._db_available = False
                return await self.change_password(user_id, old_password, new_password)

            user = await db.fetchrow(
                "SELECT password_hash FROM users WHERE id = $1", user_id,
            )
            if not user:
                return False

            try:
                if not bcrypt.checkpw(
                    old_password.encode("utf-8"),
                    user["password_hash"].encode("utf-8"),
                ):
                    return False
            except Exception:
                return False

            new_hash = bcrypt.hashpw(
                new_password.encode("utf-8"),
                bcrypt.gensalt(rounds=12),
            ).decode("utf-8")

            await db.execute(
                "UPDATE users SET password_hash = $1 WHERE id = $2",
                new_hash, user_id,
            )

            # Rotate JWT secret to invalidate all sessions
            self._jwt_secret = secrets.token_hex(32)
            self._save_auth_file()

            return True
        else:
            # Fallback
            if not self._fallback_hash:
                return False
            try:
                if not bcrypt.checkpw(
                    old_password.encode("utf-8"),
                    self._fallback_hash.encode("utf-8"),
                ):
                    return False
            except Exception:
                return False

            self._fallback_hash = bcrypt.hashpw(
                new_password.encode("utf-8"),
                bcrypt.gensalt(rounds=12),
            ).decode("utf-8")
            self._jwt_secret = secrets.token_hex(32)
            self._save_auth_file()
            return True

    async def list_users(self) -> list:
        """List all users (without password hashes)."""
        if not self._db_available:
            return []

        from services.database import get_database
        db = await get_database()
        if not db.available:
            self._db_available = False
            return []

        try:
            rows = await db.fetch(
                "SELECT id, username, role, created_at, updated_at FROM users ORDER BY id"
            )
            return rows
        except RuntimeError:
            self._db_available = False
            return []

    async def delete_user(self, user_id: int, requesting_user_id: int) -> bool:
        """
        Delete a user. Cannot delete yourself.

        Returns True on success.
        """
        if user_id == requesting_user_id:
            raise ValueError("Cannot delete your own account")

        if not self._db_available:
            raise RuntimeError("PostgreSQL required for user management")

        from services.database import get_database
        db = await get_database()
        if not db.available:
            self._db_available = False
            raise RuntimeError("PostgreSQL required for user management")

        # Verify user exists
        user = await db.fetchrow("SELECT id FROM users WHERE id = $1", user_id)
        if not user:
            return False

        await db.execute("DELETE FROM users WHERE id = $1", user_id)
        return True

    async def get_user_by_id(self, user_id: int) -> Optional[dict]:
        """Get a user by ID (without password hash)."""
        if not self._db_available:
            if user_id == 0:
                return {"id": 0, "username": "admin", "role": "admin"}
            return None

        from services.database import get_database
        db = await get_database()
        if not db.available:
            self._db_available = False
            if user_id == 0:
                return {"id": 0, "username": "admin", "role": "admin"}
            return None
        return await db.fetchrow(
            "SELECT id, username, role, created_at, updated_at FROM users WHERE id = $1",
            user_id,
        )

    async def get_user_settings(self, user_id: int) -> dict:
        """Get a user's settings JSONB column."""
        if not self._db_available:
            return {}

        from services.database import get_database
        db = await get_database()
        if not db.available:
            self._db_available = False
            return {}
        row = await db.fetchrow(
            "SELECT settings FROM users WHERE id = $1", user_id,
        )
        if not row or not row.get("settings"):
            return {}

        import json as _json
        settings = row["settings"]
        if isinstance(settings, str):
            return _json.loads(settings)
        return dict(settings) if settings else {}

    async def update_user_settings(self, user_id: int, settings: dict) -> dict:
        """Update (merge) a user's settings JSONB column."""
        if not self._db_available:
            raise RuntimeError("PostgreSQL required for user settings")

        from services.database import get_database
        import json as _json
        db = await get_database()
        if not db.available:
            self._db_available = False
            raise RuntimeError("PostgreSQL required for user settings")

        # Merge with existing settings
        current = await self.get_user_settings(user_id)
        current.update(settings)

        await db.execute(
            "UPDATE users SET settings = $1::jsonb WHERE id = $2",
            _json.dumps(current), user_id,
        )
        return current

    # =========================================================================
    # JWT Token Management
    # =========================================================================

    def create_token(self, user_id: int, username: str, role: str) -> dict:
        """Create a JWT token. Returns {token, expires_at}."""
        if not self._jwt_secret:
            raise ValueError("No JWT secret configured")

        expires_at = time.time() + (JWT_EXPIRY_HOURS * 3600)
        payload = {
            "sub": str(user_id),
            "username": username,
            "role": role,
            "iat": time.time(),
            "exp": expires_at,
        }
        token = jwt.encode(payload, self._jwt_secret, algorithm=JWT_ALGORITHM)
        return {
            "token": token,
            "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        }

    def verify_token(self, token: str) -> Optional[dict]:
        """
        Verify a JWT token. Returns decoded payload or None.

        Payload contains: sub (user_id), username, role.
        """
        if not self._jwt_secret:
            return None
        try:
            payload = jwt.decode(token, self._jwt_secret, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    # =========================================================================
    # Internal
    # =========================================================================

    def _save_auth_file(self) -> None:
        """Persist JWT secret and fallback hash to disk."""
        AUTH_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "jwt_secret": self._jwt_secret,
            "password_hash": self._fallback_hash,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        AUTH_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            AUTH_FILE.chmod(0o600)
        except OSError:
            pass


def get_auth_manager() -> AdminAuthManager:
    """Get the singleton auth manager."""
    return AdminAuthManager.get_instance()


# =============================================================================
# FastAPI Dependencies
# =============================================================================


async def verify_admin(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    x_admin_key: Optional[str] = Header(None),
) -> dict:
    """
    Centralized admin auth dependency.

    Checks in order:
    1. Authorization: Bearer <JWT> header — validates and returns user info
    2. X-Admin-Key header (legacy, verified via bcrypt against fallback)

    Returns:
        Dict with user_id, username, role from JWT payload.

    Raises:
        HTTPException 503 if setup required
        HTTPException 401 if auth fails
    """
    auth = get_auth_manager()

    # If setup hasn't been done yet, block all admin endpoints
    if auth.setup_required:
        raise HTTPException(
            status_code=503,
            detail="No admin account configured. Visit /admin to create your account.",
        )

    # 1. Check Bearer token (preferred)
    if credentials and credentials.credentials:
        payload = auth.verify_token(credentials.credentials)
        if payload:
            return {
                "user_id": int(payload.get("sub", 0)),
                "username": payload.get("username", "admin"),
                "role": payload.get("role", "admin"),
            }
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # 2. Fallback: X-Admin-Key header (legacy, bcrypt-verified against fallback hash)
    if x_admin_key and auth._fallback_hash:
        try:
            if bcrypt.checkpw(
                x_admin_key.encode("utf-8"),
                auth._fallback_hash.encode("utf-8"),
            ):
                return {"user_id": 0, "username": "admin", "role": "admin"}
        except Exception:
            pass
        raise HTTPException(status_code=401, detail="Invalid admin key")

    raise HTTPException(status_code=401, detail="Authentication required")


async def verify_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    """
    Auth dependency for any logged-in user (not admin-only).

    Returns:
        Dict with user_id, username, role from JWT payload.

    Raises:
        HTTPException 401 if auth fails
    """
    auth = get_auth_manager()

    if credentials and credentials.credentials:
        payload = auth.verify_token(credentials.credentials)
        if payload:
            return {
                "user_id": int(payload.get("sub", 0)),
                "username": payload.get("username", "admin"),
                "role": payload.get("role", "user"),
            }
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    raise HTTPException(status_code=401, detail="Authentication required")
