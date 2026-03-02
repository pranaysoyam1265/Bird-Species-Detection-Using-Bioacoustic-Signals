from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from db_utils import get_user_by_email, create_user, hash_password, verify_password
from typing import Optional

router = APIRouter()

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

@router.post("/auth/login")
async def login(req: LoginRequest):
    user = get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"]
        }
    }

@router.post("/auth/register")
async def register(req: RegisterRequest):
    if get_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed = hash_password(req.password)
    user_id = create_user(req.email, hashed, req.name)
    
    return {
        "user": {
            "id": user_id,
            "email": req.email,
            "name": req.name
        }
    }

class ChangePasswordRequest(BaseModel):
    user_id: int
    current_password: str
    new_password: str

@router.post("/auth/change-password")
async def change_password(req: ChangePasswordRequest):
    from db_utils import get_user_by_id, update_user_password
    user = get_user_by_id(req.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not verify_password(req.current_password, user["password"]):
        raise HTTPException(status_code=403, detail="Current password is incorrect")
    
    new_hash = hash_password(req.new_password)
    update_user_password(req.user_id, new_hash)
    
    return {"success": True, "message": "Password updated successfully"}
