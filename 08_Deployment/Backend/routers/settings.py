from fastapi import APIRouter, HTTPException, Body
from db_utils import get_user_settings, upsert_user_settings
from typing import Dict, Any

router = APIRouter()

@router.get("/settings/{user_id}")
async def fetch_settings(user_id: int):
    try:
        settings = get_user_settings(user_id)
        return {"settings": settings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/settings/{user_id}")
async def update_settings(user_id: int, settings: Dict[str, Any] = Body(...)):
    try:
        upsert_user_settings(user_id, settings)
        return {"settings": settings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
