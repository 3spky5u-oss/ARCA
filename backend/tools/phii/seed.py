"""
Phii Seed Data - workspace-level correction and terminology seeds.

Seeds the firm_corrections and firm_terminology tables from the active
lexicon/pipeline configuration. Vanilla ARCA ships with no domain-specific
defaults, so this remains domain-agnostic unless a domain pack provides seeds.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any
import uuid

logger = logging.getLogger(__name__)


def _get_domain_seed_corrections():
    """Load domain-specific seed corrections from lexicon pipeline config."""
    try:
        from domain_loader import get_pipeline_config
        pipeline = get_pipeline_config()
        return pipeline.get("phii_seed_corrections", [])
    except Exception:
        return []


def _get_firm_corrections() -> List[Dict[str, Any]]:
    """Load firm corrections from pipeline config.

    Returns empty list if no domain corrections configured.
    """
    try:
        from domain_loader import get_pipeline_config
        return get_pipeline_config().get("phii_firm_corrections", [])
    except Exception:
        return []


def _get_firm_terminology() -> List[Dict[str, str]]:
    """Load firm terminology from pipeline config.

    Returns empty list if no domain terminology configured.
    """
    try:
        from domain_loader import get_pipeline_config
        return get_pipeline_config().get("phii_seed_terminology", [])
    except Exception:
        return []


def get_db_path() -> Path:
    """Get the path to the reinforcement database."""
    backend_dir = Path(__file__).parent.parent.parent
    return backend_dir / "data" / "phii" / "reinforcement.sqlite"


def _ensure_db_parent(db_path: Path) -> Path:
    """Ensure parent folder exists, with legacy-path and fallback handling."""
    parent = db_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        return db_path
    except FileExistsError:
        if parent.exists() and not parent.is_dir():
            legacy_path = parent.with_name(f"{parent.name}.legacy-{int(time.time())}")
            try:
                parent.rename(legacy_path)
                logger.warning(
                    f"Phii seed path was a file, moved to {legacy_path}. Recreating directory."
                )
                parent.mkdir(parents=True, exist_ok=True)
                return db_path
            except Exception as migrate_err:
                logger.warning(f"Could not migrate invalid Phii seed path: {migrate_err}")

    fallback = Path("/tmp/arca/phii") / db_path.name
    fallback.parent.mkdir(parents=True, exist_ok=True)
    logger.warning(f"Falling back to Phii seed DB path: {fallback}")
    return fallback


def seed_if_empty() -> bool:
    """Seed workspace corrections and terminology if tables are empty.

    Returns:
        True if seeding was performed, False if tables already had data
    """
    db_path = _ensure_db_parent(get_db_path())

    with sqlite3.connect(db_path) as conn:
        # Check if firm_corrections already has data
        try:
            row = conn.execute("SELECT COUNT(*) FROM firm_corrections WHERE is_active = 1").fetchone()
            if row and row[0] > 0:
                logger.debug(f"Workspace corrections already seeded ({row[0]} entries)")
                return False
        except sqlite3.OperationalError:
            # Table doesn't exist yet - will be created by ReinforcementStore
            pass

        # Load all corrections from pipeline config (firm + domain-specific)
        all_corrections = _get_firm_corrections() + _get_domain_seed_corrections()

        # Seed corrections
        seeded_corrections = 0
        for correction in all_corrections:
            try:
                conn.execute(
                    """
                    INSERT INTO firm_corrections (
                        id, wrong_behavior, right_behavior, context_keywords,
                        confidence, category, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        str(uuid.uuid4()),
                        correction["wrong_behavior"],
                        correction["right_behavior"],
                        json.dumps(correction.get("context_keywords", [])),
                        correction.get("confidence", 0.8),
                        correction.get("category", "domain"),
                        1,
                    ),
                )
                seeded_corrections += 1
            except sqlite3.OperationalError as e:
                logger.warning(f"Could not seed firm correction: {e}")
                return False
            except sqlite3.IntegrityError:
                pass

        # Seed firm terminology from pipeline config
        base_terminology = _get_firm_terminology()
        seeded_terminology = 0
        for term in base_terminology:
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO firm_terminology (
                        concept, preferred_term, context
                    ) VALUES (?, ?, ?)
                """,
                    (
                        term["concept"],
                        term["preferred_term"],
                        term["context"],
                    ),
                )
                seeded_terminology += 1
            except sqlite3.OperationalError as e:
                logger.warning(f"Could not seed terminology: {e}")

        conn.commit()

    if seeded_corrections > 0 or seeded_terminology > 0:
        logger.info(f"Seeded Phii workspace defaults: {seeded_corrections} corrections, {seeded_terminology} terminology entries")
        return True

    return False


def get_all_firm_corrections() -> List[Dict[str, Any]]:
    """Get all firm corrections for admin display.

    Returns:
        List of correction dictionaries
    """
    db_path = get_db_path()

    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM firm_corrections
            ORDER BY category, confidence DESC
        """
        ).fetchall()

    return [dict(row) for row in rows]


def get_firm_terminology() -> List[Dict[str, str]]:
    """Get all firm terminology for admin display.

    Returns:
        List of terminology dictionaries
    """
    db_path = get_db_path()

    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM firm_terminology
            ORDER BY concept
        """
        ).fetchall()

    return [dict(row) for row in rows]
