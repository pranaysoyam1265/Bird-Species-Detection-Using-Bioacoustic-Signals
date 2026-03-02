from fastapi import APIRouter, HTTPException, Query
from db_utils import get_detections_by_user, count_detections_by_user
import json

router = APIRouter()

@router.get("/history")
async def get_history(
    user_id: int = Query(..., description="User ID"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    try:
        rows = get_detections_by_user(user_id, limit, offset)
        total = count_detections_by_user(user_id)
        
        detections = []
        for r in rows:
            detections.append({
                "id": r["id"],
                "filename": r["filename"],
                "date": r["date"],
                "time": r["time"],
                "duration": r["duration"],
                "topSpecies": r["top_species"],
                "topScientific": r["top_scientific"],
                "topConfidence": r["top_confidence"],
                "predictions": json.loads(r["predictions"]),
                "segments": json.loads(r["segments"]),
                "audioUrl": r["audio_url"]
            })
            
        return {
            "detections": detections,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
