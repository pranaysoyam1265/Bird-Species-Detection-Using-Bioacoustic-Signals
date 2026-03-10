from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from db_utils import (
    get_user_by_id, get_all_users, get_platform_stats,
    update_user_role, delete_user_by_id
)

router = APIRouter()


def _require_admin(user_id: int):
    """Raise 403 if the requesting user is not an admin."""
    user = get_user_by_id(user_id)
    if not user or user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")


@router.get("/admin/stats")
async def admin_stats(user_id: int = Query(..., description="Requesting user ID")):
    _require_admin(user_id)
    return get_platform_stats()


@router.get("/admin/users")
async def admin_users(user_id: int = Query(..., description="Requesting user ID")):
    _require_admin(user_id)
    users = get_all_users()
    return {
        "users": [
            {
                "id": u["id"],
                "email": u["email"],
                "name": u["name"],
                "role": u["role"],
                "detection_count": u["detection_count"],
                "created_at": u["created_at"],
            }
            for u in users
        ]
    }


class RoleUpdateRequest(BaseModel):
    role: str  # "admin" or "user"


@router.put("/admin/users/{target_id}/role")
async def change_user_role(
    target_id: int,
    body: RoleUpdateRequest,
    user_id: int = Query(..., description="Requesting admin user ID"),
):
    _require_admin(user_id)
    if body.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'user'")
    target = get_user_by_id(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    update_user_role(target_id, body.role)
    return {"success": True, "message": f"User {target_id} role updated to {body.role}"}


@router.delete("/admin/users/{target_id}")
async def remove_user(
    target_id: int,
    user_id: int = Query(..., description="Requesting admin user ID"),
):
    _require_admin(user_id)
    if target_id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own admin account")
    target = get_user_by_id(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    delete_user_by_id(target_id)
    return {"success": True, "message": f"User {target_id} deleted"}
