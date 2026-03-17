"""
BirdSense Backend — Configuration
Paths, model config, and audio processing parameters.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load local .env file if it exists
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent
DEPLOYMENT_DIR = BACKEND_DIR.parent
PROJECT_ROOT = DEPLOYMENT_DIR.parent

MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    r"c:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\05_Model\Saved_Models\best_model_v3.pth",
))

LABEL_MAPPING_PATH = Path(os.getenv(
    "LABEL_MAPPING_PATH",
    str(BACKEND_DIR / "labels" / "label_mapping_290.json"),
))

UPLOAD_DIR = BACKEND_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Device ─────────────────────────────────────────────────────
DEVICE = os.getenv("DEVICE", "cpu")  # "cpu" or "cuda"

# ── Model ──────────────────────────────────────────────────────
MODEL_CONFIG = {
    "model_name": "tf_efficientnet_b2_ns",
    "num_classes": 290,
    "input_size": (128, 216),  # (height, width) for spectrograms
    "dropout_rate": 0.4,
}

# ── Audio ──────────────────────────────────────────────────────
AUDIO_CONFIG = {
    "sample_rate": 22050,
    "chunk_duration": 5.0,     # seconds
    "chunk_overlap": 0.5,      # 50% overlap (matches training pipeline)
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "fmin": 20,
    "fmax": 10000,
    "supported_formats": {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
    "max_file_size_mb": 50,
    "max_duration_minutes": 10,
}

# ── Server ─────────────────────────────────────────────────────
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
