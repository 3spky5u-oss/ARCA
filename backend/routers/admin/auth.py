"""
Auth endpoints (no auth required for most).
"""

from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from . import router, _bearer_scheme
from .models import LoginRequest, RegisterRequest, ChangePasswordRequest
from services.admin_auth import get_auth_manager, verify_admin


@router.get("/auth-status")
async def get_auth_status() -> Dict[str, Any]:
    """Check if admin setup is required (no auth needed)."""
    auth = get_auth_manager()
    return {"setup_required": auth.setup_required}


@router.post("/auth/register")
async def register(request: RegisterRequest) -> Dict[str, Any]:
    """
    Register a new user.

    First user auto-becomes admin (first-boot setup).
    Subsequent users get 'user' role (open registration).
    """
    auth = get_auth_manager()

    try:
        user = await auth.register_user(request.username, request.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    token_data = auth.create_token(user["id"], user["username"], user["role"])

    is_first = user["role"] == "admin"

    return {
        "success": True,
        "message": "Admin account created" if is_first else "Account created",
        "username": user["username"],
        "role": user["role"],
        "token": token_data["token"],
        "expires_at": token_data["expires_at"],
    }


@router.post("/auth/login")
async def login(request: LoginRequest) -> Dict[str, Any]:
    """Login with username and password, returns JWT."""
    auth = get_auth_manager()

    if auth.setup_required:
        raise HTTPException(
            status_code=503,
            detail="No admin account configured. Create your account first.",
        )

    user = await auth.verify_login(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token_data = auth.create_token(user["id"], user["username"], user["role"])

    return {
        "success": True,
        "username": user["username"],
        "role": user["role"],
        "token": token_data["token"],
        "expires_at": token_data["expires_at"],
    }


@router.get("/auth/verify")
async def verify_token_endpoint(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Dict[str, Any]:
    """Verify if a stored JWT token is still valid (no auth required to call)."""
    if not credentials or not credentials.credentials:
        return {"valid": False}
    auth = get_auth_manager()
    payload = auth.verify_token(credentials.credentials)
    if payload:
        return {
            "valid": True,
            "username": payload.get("username"),
            "role": payload.get("role"),
        }
    return {"valid": False}


@router.post("/auth/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: dict = Depends(verify_admin),
) -> Dict[str, Any]:
    """Change your own password. Invalidates all existing sessions."""
    auth = get_auth_manager()

    try:
        success = await auth.change_password(
            current_user["user_id"], request.old_password, request.new_password,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not success:
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    token_data = auth.create_token(
        current_user["user_id"], current_user["username"], current_user["role"],
    )

    return {
        "success": True,
        "message": "Password changed. All existing sessions have been invalidated.",
        "token": token_data["token"],
        "expires_at": token_data["expires_at"],
    }


# Legacy endpoints -- kept for backward compatibility with old frontend


@router.post("/setup")
async def setup_admin_legacy(request: RegisterRequest) -> Dict[str, Any]:
    """Legacy setup endpoint -- delegates to /auth/register."""
    return await register(request)


@router.post("/login")
async def login_legacy(request: LoginRequest) -> Dict[str, Any]:
    """Legacy login endpoint -- delegates to /auth/login."""
    return await login(request)


@router.post("/verify-token")
async def verify_token_legacy(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Dict[str, Any]:
    """Legacy verify endpoint -- delegates to /auth/verify."""
    return await verify_token_endpoint(credentials)


@router.post("/change-password")
async def change_password_legacy(
    request: ChangePasswordRequest,
    current_user: dict = Depends(verify_admin),
) -> Dict[str, Any]:
    """Legacy change-password endpoint -- delegates to /auth/change-password."""
    return await change_password(request, current_user)
