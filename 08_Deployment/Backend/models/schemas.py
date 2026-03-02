"""
BirdSense Backend — Pydantic Schemas
Request / response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field


# ── Species ────────────────────────────────────────────────────

class SpeciesItem(BaseModel):
    index: int
    scientific_name: str
    english_name: str


class SpeciesListResponse(BaseModel):
    count: int
    species: list[SpeciesItem]


# ── Detection ──────────────────────────────────────────────────

class Prediction(BaseModel):
    species: str
    scientific: str
    confidence: float


class DetectionSegment(BaseModel):
    start_time: float
    end_time: float
    species: str
    scientific: str
    confidence: float


class DetectResponse(BaseModel):
    status: str = "success"
    duration: float = Field(description="Audio duration in seconds")
    processing_time_ms: float
    top_species: str
    top_scientific: str
    top_confidence: float
    predictions: list[Prediction]
    segments: list[DetectionSegment]


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str


# ── Health ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_species: int
    model_path: str
