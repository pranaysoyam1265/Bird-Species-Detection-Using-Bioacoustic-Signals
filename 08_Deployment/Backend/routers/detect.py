"""
BirdSense Backend — Detection Router
POST /detect  — upload audio → ML inference → return predictions
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from config import AUDIO_CONFIG
from models.schemas import DetectResponse, Prediction, DetectionSegment, ErrorResponse
from services.audio import load_audio, get_audio_duration, process_audio_to_spectrograms, reduce_noise
from services.inference import (
    predict_batch,
    get_english_name,
    get_scientific_name,
    is_model_loaded,
)
from db_utils import insert_detection
import uuid
from datetime import datetime

logger = logging.getLogger("birdsense.detect")

router = APIRouter()

SUPPORTED_EXTENSIONS = AUDIO_CONFIG["supported_formats"]
MAX_SIZE_BYTES = AUDIO_CONFIG["max_file_size_mb"] * 1024 * 1024
MAX_DURATION_S = AUDIO_CONFIG["max_duration_minutes"] * 60


@router.post(
    "/detect",
    response_model=DetectResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Detect bird species in an audio file",
)
async def detect(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG, M4A)"),
    top_k: int = Form(5, ge=1, le=87, description="Number of top predictions"),
    confidence_threshold: float = Form(0.001, ge=0.0, le=1.0, description="Minimum confidence to include"),
    noise_reduction: bool = Form(False, description="Apply noise reduction before inference"),
    user_id: Optional[int] = Form(None, description="User ID for history saving"),
):
    """
    Upload an audio file and receive bird species predictions.

    The audio is split into overlapping 5-second chunks, each converted to a
    mel spectrogram and run through the EfficientNet-B2 model. Results are
    aggregated across chunks to produce per-segment and overall predictions.
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet — server is starting up")

    # ── Validate file extension ──
    suffix = Path(audio_file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # ── Save to temp file ──
    start_time = time.perf_counter()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio_file.read()
        if len(content) > MAX_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({len(content) / 1024 / 1024:.1f} MB). Max: {AUDIO_CONFIG['max_file_size_mb']} MB",
            )
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # ── Load audio ──
        audio, sr = load_audio(tmp_path)
        duration = get_audio_duration(audio, sr)

        if duration > MAX_DURATION_S:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long ({duration / 60:.1f} min). Max: {AUDIO_CONFIG['max_duration_minutes']} min",
            )

        # ── Optional noise reduction ──
        if noise_reduction:
            audio = reduce_noise(audio, sr)

        # ── Generate spectrograms ──
        spectrograms_data = process_audio_to_spectrograms(audio, sr)
        if not spectrograms_data:
            raise HTTPException(status_code=400, detail="Audio too short to produce any chunks")

        specs = [s[0] for s in spectrograms_data]
        time_ranges = [(s[1], s[2]) for s in spectrograms_data]

        # ── Run inference ──
        predictions_raw = predict_batch(specs, top_k=top_k)

        # ── Build segments ──
        segments: list[dict] = []
        for pred, (seg_start, seg_end) in zip(predictions_raw, time_ranges):
            top_idx = pred["predicted_class"]
            conf = float(pred["confidence"]) * 100
            if conf >= confidence_threshold * 100:
                segments.append({
                    "start_time": round(seg_start, 2),
                    "end_time": round(seg_end, 2),
                    "species": get_english_name(top_idx),
                    "scientific": get_scientific_name(top_idx),
                    "confidence": round(conf, 2),
                })

        # ── Aggregate top predictions across all chunks ──
        aggregated: dict[int, list[float]] = {}
        for pred in predictions_raw:
            for idx, prob in zip(pred["top_indices"], pred["top_probs"]):
                idx_int = int(idx)
                aggregated.setdefault(idx_int, []).append(float(prob))

        # Average confidence per species, sort descending
        avg_preds = [
            (idx, sum(confs) / len(confs))
            for idx, confs in aggregated.items()
        ]
        avg_preds.sort(key=lambda x: x[1], reverse=True)
        avg_preds = avg_preds[:top_k]

        # ── Build response ──
        top_idx, top_conf = avg_preds[0] if avg_preds else (0, 0.0)

        result_predictions = [
            Prediction(
                species=get_english_name(idx),
                scientific=get_scientific_name(idx),
                confidence=round(conf * 100, 2),
            )
            for idx, conf in avg_preds
            if conf * 100 >= confidence_threshold * 100
        ]

        result_segments = [
            DetectionSegment(**seg) for seg in segments
        ]

        processing_ms = (time.perf_counter() - start_time) * 1000

        response_obj = DetectResponse(
            status="success",
            duration=round(duration, 2),
            processing_time_ms=round(processing_ms, 1),
            top_species=get_english_name(top_idx),
            top_scientific=get_scientific_name(top_idx),
            top_confidence=round(top_conf * 100, 2),
            predictions=result_predictions,
            segments=result_segments,
        )
        
        # ── Save to DB if user_id provided ──
        if user_id is not None:
            now = datetime.now()
            try:
                insert_detection({
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "filename": audio_file.filename or "upload.wav",
                    "date": now.strftime("%Y-%m-%d"),
                    "time": now.strftime("%H:%M:%S"),
                    "duration": duration,
                    "top_species": get_english_name(top_idx),
                    "top_scientific": get_scientific_name(top_idx),
                    "top_confidence": round(top_conf * 100, 2),
                    "predictions": [p.dict() for p in result_predictions],
                    "segments": [s.dict() for s in result_segments],
                    "audio_url": None
                })
            except Exception as e:
                logger.error(f"Failed to save detection to DB: {e}")

        return response_obj

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
