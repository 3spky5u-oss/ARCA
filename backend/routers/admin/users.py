"""
User management endpoints (admin-only and user-facing).
"""

from typing import Dict, Any

from fastapi import Depends, HTTPException

from . import router
from .models import CreateUserRequest, UserSettingsUpdate
from services.admin_auth import get_auth_manager, verify_admin, verify_user


@router.get("/users")
async def list_users(current_user: dict = Depends(verify_admin)) -> Dict[str, Any]:
    """List all users (admin only)."""
    auth = get_auth_manager()
    users = await auth.list_users()

    serialized = []
    for u in users:
        entry = dict(u)
        for key in ("created_at", "updated_at"):
            if key in entry and hasattr(entry[key], "isoformat"):
                entry[key] = entry[key].isoformat()
        serialized.append(entry)

    return {"users": serialized, "count": len(serialized)}


@router.post("/users")
async def create_user(
    request: CreateUserRequest,
    current_user: dict = Depends(verify_admin),
) -> Dict[str, Any]:
    """Create a new user (admin only)."""
    auth = get_auth_manager()

    try:
        user = await auth.register_user(request.username, request.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        "success": True,
        "message": f"User '{user['username']}' created",
        "user": {"id": user["id"], "username": user["username"], "role": user["role"]},
    }


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: dict = Depends(verify_admin),
) -> Dict[str, Any]:
    """Delete a user (admin only, cannot delete yourself)."""
    auth = get_auth_manager()

    try:
        success = await auth.delete_user(user_id, current_user["user_id"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not success:
        raise HTTPException(status_code=404, detail="User not found")

    return {"success": True, "message": "User deleted"}


# Auth status endpoint -- NO auth required
# Separate from admin endpoints so frontend can detect first-boot state
@router.get("/auth/status")
async def get_auth_status_public() -> Dict[str, Any]:
    """Check if any users exist and if auth is required (no auth needed)."""
    auth = get_auth_manager()
    return {
        "has_users": not auth.setup_required,
        "auth_required": True,
    }


@router.get("/user/settings")
async def get_user_settings(
    current_user: dict = Depends(verify_user),
) -> Dict[str, Any]:
    """Get the current user's settings."""
    auth = get_auth_manager()
    settings = await auth.get_user_settings(current_user["user_id"])
    return {
        "settings": settings,
        "username": current_user["username"],
        "role": current_user["role"],
    }


@router.put("/user/settings")
async def update_user_settings(
    request: UserSettingsUpdate,
    current_user: dict = Depends(verify_user),
) -> Dict[str, Any]:
    """Update the current user's settings."""
    auth = get_auth_manager()

    updates = {}
    if request.theme is not None:
        if request.theme not in ("dark", "light"):
            raise HTTPException(status_code=400, detail="Theme must be 'dark' or 'light'")
        updates["theme"] = request.theme
    if request.display_name is not None:
        if len(request.display_name) > 100:
            raise HTTPException(status_code=400, detail="Display name too long")
        updates["display_name"] = request.display_name
    if request.phii_level_override is not None:
        valid = ("", "beginner", "experienced", "expert")
        if request.phii_level_override not in valid:
            raise HTTPException(status_code=400, detail=f"Invalid phii level. Must be one of: {valid}")
        updates["phii_level_override"] = request.phii_level_override or None

    if not updates:
        raise HTTPException(status_code=400, detail="No settings to update")

    try:
        settings = await auth.update_user_settings(current_user["user_id"], updates)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {"success": True, "settings": settings}
