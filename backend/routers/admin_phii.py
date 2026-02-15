"""
Admin Phii Router - Personality/Behavior Management API

Provides endpoints for managing Phii behavior enhancements:
- Feedback statistics
- Flagged response review
- Behavior toggle settings

Protected by ADMIN_KEY header authentication.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from config import runtime_config
from services.admin_auth import verify_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/phii", tags=["admin-phii"])


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class PhiiSettingsUpdate(BaseModel):
    """Phii settings update request."""

    phii_energy_matching: Optional[bool] = None
    phii_specialty_detection: Optional[bool] = None
    phii_implicit_feedback: Optional[bool] = None


class FlagResolveRequest(BaseModel):
    """Request to resolve a flag."""

    notes: Optional[str] = ""


class PhiiDebugRequest(BaseModel):
    """Request body for debug endpoint."""

    message: str
    session_id: str = "debug-session"


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get("/stats")
async def get_phii_stats(_: bool = Depends(verify_admin)):
    """Get Phii feedback statistics and current settings.

    Returns:
        - Feedback counts by type
        - Unresolved flag count
        - Current Phii settings
    """


    try:
        from tools.phii import ReinforcementStore
        from config import runtime_config

        store = ReinforcementStore()
        stats = await store.get_stats_async()

        return {
            "success": True,
            "feedback": stats,
            "settings": {
                "phii_energy_matching": runtime_config.phii_energy_matching,
                "phii_specialty_detection": runtime_config.phii_specialty_detection,
                "phii_implicit_feedback": runtime_config.phii_implicit_feedback,
            },
        }
    except ImportError as e:
        logger.warning(f"Phii module not available: {e}")
        return {
            "success": False,
            "error": "Phii module not available",
            "feedback": {"counts": {}, "unresolved_flags": 0, "recent_24h": 0, "total": 0},
            "settings": {
                "phii_energy_matching": True,
                "phii_specialty_detection": True,
                "phii_implicit_feedback": True,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get Phii stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flags")
async def get_flags(
    resolved: bool = Query(False, description="Include resolved flags"),
    limit: int = Query(50, ge=1, le=200),
    _: bool = Depends(verify_admin),
):
    """Get flagged responses for review.

    Args:
        resolved: Whether to include resolved flags
        limit: Maximum number of flags to return

    Returns:
        List of flagged feedback records
    """


    try:
        from tools.phii import ReinforcementStore

        store = ReinforcementStore()
        flags = await store.get_flags_async(resolved=resolved, limit=limit)

        return {
            "success": True,
            "flags": [f.to_dict() for f in flags],
            "count": len(flags),
        }
    except ImportError as e:
        logger.warning(f"Phii module not available: {e}")
        return {"success": False, "error": "Phii module not available", "flags": [], "count": 0}
    except Exception as e:
        logger.error(f"Failed to get flags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flags/{flag_id}/resolve")
async def resolve_flag(
    flag_id: int,
    request: FlagResolveRequest,
    _: bool = Depends(verify_admin),
):
    """Mark a flagged response as resolved.

    Args:
        flag_id: ID of the flag to resolve
        request: Resolution details including optional notes

    Returns:
        Success status
    """


    try:
        from tools.phii import ReinforcementStore

        store = ReinforcementStore()
        success = await store.resolve_flag_async(flag_id, notes=request.notes or "")

        if success:
            return {"success": True, "message": f"Flag {flag_id} resolved"}
        else:
            raise HTTPException(status_code=404, detail=f"Flag {flag_id} not found")
    except ImportError as e:
        logger.warning(f"Phii module not available: {e}")
        raise HTTPException(status_code=503, detail="Phii module not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve flag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/settings")
async def update_settings(
    request: PhiiSettingsUpdate,
    _: bool = Depends(verify_admin),
):
    """Update Phii behavior settings.

    Args:
        request: Settings to update

    Returns:
        Updated settings and list of changed fields
    """


    try:
        from config import runtime_config

        # Build update dict from non-None values
        updates = {}
        if request.phii_energy_matching is not None:
            updates["phii_energy_matching"] = request.phii_energy_matching
        if request.phii_specialty_detection is not None:
            updates["phii_specialty_detection"] = request.phii_specialty_detection
        if request.phii_implicit_feedback is not None:
            updates["phii_implicit_feedback"] = request.phii_implicit_feedback

        if not updates:
            return {
                "success": True,
                "updated": [],
                "settings": {
                    "phii_energy_matching": runtime_config.phii_energy_matching,
                    "phii_specialty_detection": runtime_config.phii_specialty_detection,
                    "phii_implicit_feedback": runtime_config.phii_implicit_feedback,
                },
            }

        result = runtime_config.update(**updates)

        return {
            "success": True,
            "updated": result["updated"],
            "settings": {
                "phii_energy_matching": runtime_config.phii_energy_matching,
                "phii_specialty_detection": runtime_config.phii_specialty_detection,
                "phii_implicit_feedback": runtime_config.phii_implicit_feedback,
            },
        }
    except Exception as e:
        logger.error(f"Failed to update Phii settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flag")
async def add_flag(
    session_id: str = Query(..., description="Session identifier"),
    message_id: str = Query(..., description="Message identifier"),
    user_message: str = Query("", description="The user's message"),
    assistant_response: str = Query("", description="The assistant's response"),
):
    """Add a flag for a response (called from frontend).

    This endpoint is called when the user clicks the flag button on a message.
    It does not require admin authentication since users can flag responses.

    Args:
        session_id: Session identifier
        message_id: ID of the flagged message
        user_message: The user's message that prompted the response
        assistant_response: The response being flagged

    Returns:
        ID of the created flag
    """
    try:
        from tools.phii import PhiiContextBuilder

        builder = PhiiContextBuilder()
        flag_id = await builder.add_flag_async(
            session_id=session_id,
            message_id=message_id,
            user_message=user_message,
            assistant_response=assistant_response,
        )

        logger.info(f"Flag added: session={session_id}, message={message_id}")

        return {
            "success": True,
            "flag_id": flag_id,
            "message": "Response flagged for review",
        }
    except AttributeError:
        # Fallback to sync method if async not available
        try:
            from tools.phii import PhiiContextBuilder

            builder = PhiiContextBuilder()
            flag_id = builder.add_flag(
                session_id=session_id,
                message_id=message_id,
                user_message=user_message,
                assistant_response=assistant_response,
            )

            logger.info(f"Flag added: session={session_id}, message={message_id}")

            return {
                "success": True,
                "flag_id": flag_id,
                "message": "Response flagged for review",
            }
        except Exception as e:
            logger.error(f"Failed to add flag: {e}")
            return {"success": False, "error": str(e)}
    except ImportError as e:
        logger.warning(f"Phii module not available: {e}")
        return {"success": False, "error": "Phii module not available"}
    except Exception as e:
        logger.error(f"Failed to add flag: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# CORRECTIONS ENDPOINTS
# =============================================================================


@router.get("/corrections")
async def get_corrections(
    active_only: bool = Query(True, description="Only return active corrections"),
    limit: int = Query(50, ge=1, le=200),
    _: bool = Depends(verify_admin),
):
    """Get learned corrections.

    Returns a list of corrections that ARCA has learned from user feedback.
    These corrections influence future responses.

    Args:
        active_only: Only return active (not deleted) corrections
        limit: Maximum number of corrections to return

    Returns:
        List of correction records with metadata
    """


    try:
        from tools.phii import ReinforcementStore

        store = ReinforcementStore()
        corrections = await store.get_all_corrections_async(active_only=active_only, limit=limit)
        stats = await store.get_corrections_stats_async()

        return {
            "success": True,
            "corrections": [c.to_dict() for c in corrections],
            "count": len(corrections),
            "stats": stats,
        }
    except ImportError as e:
        logger.warning(f"Phii module not available: {e}")
        return {
            "success": False,
            "error": "Phii module not available",
            "corrections": [],
            "count": 0,
            "stats": {"active_count": 0, "total_applied": 0, "recent_7d": 0},
        }
    except Exception as e:
        logger.error(f"Failed to get corrections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/corrections/{correction_id}")
async def delete_correction(
    correction_id: int,
    _: bool = Depends(verify_admin),
):
    """Delete a learned correction.

    Soft-deletes a correction so it no longer influences responses.
    The correction record is kept for audit purposes.

    Args:
        correction_id: ID of the correction to delete

    Returns:
        Success status
    """


    try:
        from tools.phii import ReinforcementStore

        store = ReinforcementStore()
        success = await store.delete_correction_async(correction_id)

        if success:
            logger.info(f"Correction {correction_id} deleted")
            return {"success": True, "message": f"Correction {correction_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail=f"Correction {correction_id} not found")
    except ImportError as e:
        logger.warning(f"Phii module not available: {e}")
        raise HTTPException(status_code=503, detail="Phii module not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete correction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PATTERNS ENDPOINTS
# =============================================================================


@router.get("/patterns")
async def get_patterns(_: bool = Depends(verify_admin)):
    """Get learned action patterns.

    Returns pattern statistics including:
    - Total unique patterns
    - Total actions logged
    - Top patterns by frequency

    Returns:
        Pattern statistics dict
    """


    try:
        from tools.phii import ReinforcementStore

        store = ReinforcementStore()
        stats = await store.get_pattern_stats_async()

        return {
            "success": True,
            "patterns": stats,
        }
    except ImportError as e:
        logger.warning(f"Phii module not available: {e}")
        return {
            "success": False,
            "error": "Phii module not available",
            "patterns": {"pattern_count": 0, "action_count": 0, "top_patterns": []},
        }
    except Exception as e:
        logger.error(f"Failed to get patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DEBUG/METRICS ENDPOINTS
# =============================================================================


@router.post("/debug")
async def debug_learning_state(
    request: PhiiDebugRequest,
    _: bool = Depends(verify_admin),
):
    """Debug endpoint to see what Phii would inject for a given message.

    Shows:
    - Corrections that would match the message
    - Expertise level detection
    - Energy profile analysis
    - Specialty detection

    Args:
        request: Debug request with message and optional session_id

    Returns:
        Debug information about Phii's analysis
    """


    try:
        from tools.phii import (
            ReinforcementStore,
            ExpertiseDetector,
            EnergyDetector,
            SpecialtyDetector,
            Specialty,
        )

        message = request.message
        store = ReinforcementStore()

        # Get corrections that would match (use async)
        corrections = await store.get_relevant_corrections_async(message, top_k=5)

        # Analyze energy
        energy_detector = EnergyDetector()
        energy_profile = energy_detector.analyze(message)

        # Analyze expertise
        expertise_detector = ExpertiseDetector()
        expertise_profile = expertise_detector.analyze(message, energy_profile.technical_depth, None)

        # Analyze specialty
        specialty_detector = SpecialtyDetector()
        specialty_profile = specialty_detector.analyze(message)

        return {
            "success": True,
            "message": message,
            "corrections_applied": [
                {
                    "wrong_behavior": c.wrong_behavior,
                    "right_behavior": c.right_behavior,
                    "confidence": c.confidence,
                    "keywords": c.context_keywords,
                }
                for c in corrections
            ],
            "expertise_level": expertise_profile.level.value,
            "energy_analysis": {
                "brevity": energy_profile.brevity.value,
                "formality": energy_profile.formality.value,
                "urgency": energy_profile.urgency.value,
            },
            "specialty": (
                specialty_profile.primary.value if specialty_profile.primary != Specialty.UNKNOWN else "Not detected"
            ),
        }
    except ImportError as e:
        logger.warning(f"Phii module not available: {e}")
        return {
            "success": False,
            "error": "Phii module not available",
        }
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_learning_metrics(_: bool = Depends(verify_admin)):
    """Aggregate metrics on learning system usage.

    Returns:
    - Firm corrections count
    - Learned corrections count
    - Total times corrections applied
    - Top corrections by usage
    - Pattern statistics
    """


    try:
        from tools.phii import ReinforcementStore
        from services.database import get_database

        store = ReinforcementStore()
        db = await get_database()

        if db.available:
            # Use PostgreSQL
            firm = await db.fetchval(
                "SELECT COUNT(*) FROM firm_corrections WHERE is_active = TRUE"
            )
            learned = await db.fetchval(
                "SELECT COUNT(*) FROM corrections WHERE is_active = TRUE"
            )
            applied = await db.fetchval(
                "SELECT COALESCE(SUM(times_applied), 0) FROM corrections"
            )
            top = await db.fetch(
                """
                SELECT wrong_behavior, right_behavior, times_applied
                FROM corrections WHERE is_active = TRUE AND times_applied > 0
                ORDER BY times_applied DESC LIMIT 5
                """
            )
            patterns = await db.fetchval("SELECT COUNT(*) FROM action_patterns")
            actions = await db.fetchval("SELECT COUNT(*) FROM action_log")

            return {
                "success": True,
                "corrections": {
                    "firm_count": firm,
                    "learned_count": learned,
                    "times_applied": applied,
                    "top_used": [
                        {"wrong": r["wrong_behavior"][:60], "right": r["right_behavior"][:60], "times": r["times_applied"]}
                        for r in top
                    ],
                },
                "patterns": {
                    "pattern_count": patterns,
                    "action_count": actions,
                },
            }
        else:
            # Use SQLite fallback
            import sqlite3

            with sqlite3.connect(store.db_path) as conn:
                firm = conn.execute("SELECT COUNT(*) FROM firm_corrections WHERE is_active=1").fetchone()[0]
                learned = conn.execute("SELECT COUNT(*) FROM corrections WHERE is_active=1").fetchone()[0]
                applied = conn.execute("SELECT COALESCE(SUM(times_applied),0) FROM corrections").fetchone()[0]

                top = conn.execute(
                    """
                    SELECT wrong_behavior, right_behavior, times_applied
                    FROM corrections WHERE is_active=1 AND times_applied > 0
                    ORDER BY times_applied DESC LIMIT 5
                """
                ).fetchall()

                patterns = conn.execute("SELECT COUNT(*) FROM action_patterns").fetchone()[0]
                actions = conn.execute("SELECT COUNT(*) FROM action_log").fetchone()[0]

            return {
                "success": True,
                "corrections": {
                    "firm_count": firm,
                    "learned_count": learned,
                    "times_applied": applied,
                    "top_used": [{"wrong": r[0][:60], "right": r[1][:60], "times": r[2]} for r in top],
                },
                "patterns": {
                    "pattern_count": patterns,
                    "action_count": actions,
                },
            }
    except ImportError as e:
        logger.warning(f"Phii module not available: {e}")
        return {
            "success": False,
            "error": "Phii module not available",
            "corrections": {"firm_count": 0, "learned_count": 0, "times_applied": 0, "top_used": []},
            "patterns": {"pattern_count": 0, "action_count": 0},
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
