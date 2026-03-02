"""
🐦 BirdSense Backend — FastAPI Entry Point
==========================================
ML inference service for bird species detection.

Start with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import ALLOWED_ORIGINS, FASTAPI_PORT, MODEL_PATH
from models.schemas import HealthResponse
from services.inference import load_model, is_model_loaded, get_all_species
from db_utils import init_db

# ── Rate Limiter ───────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-24s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("birdsense")


# ── Lifespan (startup / shutdown) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model into memory on startup."""
    logger.info("=" * 60)
    logger.info("  BirdSense Backend — Starting")
    logger.info("=" * 60)

    try:
        init_db()
        logger.info("Database initialized")
        load_model()
        logger.info("Model ready for inference")
    except Exception as exc:
        logger.error("FAILED to load model: %s", exc)
        # Allow server to start even if model fails — /health will report status

    yield  # ← app is running

    logger.info("BirdSense Backend — Shutting down")


# ── FastAPI App ────────────────────────────────────────────────
app = FastAPI(
    title="BirdSense API",
    description="AI-powered bird species detection from audio recordings",
    version="1.0.0",
    lifespan=lifespan,
)

# Attach rate limiter to the app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────
from routers.detect import router as detect_router
from routers.species import router as species_router
from routers.history import router as history_router
from routers.auth import router as auth_router
from routers.settings import router as settings_router

app.include_router(detect_router, tags=["Detection"])
app.include_router(species_router, tags=["Species"])
app.include_router(history_router, tags=["History"])
app.include_router(auth_router, tags=["Authentication"])
app.include_router(settings_router, tags=["Settings"])


# ── Health Check ───────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check server status and whether the model is loaded."""
    species = get_all_species() if is_model_loaded() else []
    return HealthResponse(
        status="ok" if is_model_loaded() else "model_not_loaded",
        model_loaded=is_model_loaded(),
        device=str(MODEL_PATH),
        num_species=len(species),
        model_path=MODEL_PATH.name,
    )


# ── Run directly ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=FASTAPI_PORT, reload=True)
