# Save as: 08_Deployment/Streamlit_Version/config.py

"""
Configuration for Bird Detection Application (Streamlit Version)
"""

from pathlib import Path
import json

# ============================================================
# PATHS
# ============================================================

# Project root (go up from Streamlit_Version folder)
STREAMLIT_DIR = Path(__file__).parent
DEPLOYMENT_DIR = STREAMLIT_DIR.parent
PROJECT_ROOT = DEPLOYMENT_DIR.parent

# Model paths
MODEL_PATH = PROJECT_ROOT / "05_Model" / "Saved_Models" / "best_model_v3.pth"
LABEL_MAPPING_PATH = PROJECT_ROOT / "04_Labels" / "Processed_Labels" / "label_mapping_v3.json"
TEST_RESULTS_PATH = PROJECT_ROOT / "05_Model" / "Training_Logs" / "test_results_v3_FINAL.json"

# Assets (within Streamlit version)
ASSETS_DIR = STREAMLIT_DIR / "assets"
TEMP_DIR = STREAMLIT_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    "model_name": "tf_efficientnet_b2_ns",
    "num_classes": 87,
    "input_size": (128, 216),  # (height, width) for spectrograms
    "dropout_rate": 0.4,
}

# ============================================================
# AUDIO CONFIGURATION
# ============================================================

AUDIO_CONFIG = {
    "sample_rate": 22050,
    "chunk_duration": 5.0,  # seconds
    "chunk_overlap": 0.5,  # 50% overlap
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "fmin": 20,
    "fmax": 10000,
    "supported_formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
    "max_file_size_mb": 50,
    "max_duration_minutes": 10,
}

# ============================================================
# UI CONFIGURATION
# ============================================================

UI_CONFIG = {
    "default_top_k": 5,
    "default_confidence_threshold": 0.1,
    "max_batch_size": 50,
    "chart_height": 400,
    "spectrogram_cmap": "magma",
}

# ============================================================
# CONFIDENCE THRESHOLDS
# ============================================================

CONFIDENCE_THRESHOLDS = {
    "high": 0.8,      # Green - confident prediction
    "medium": 0.5,    # Yellow - moderate confidence
    "low": 0.2,       # Orange - low confidence
    # Below low = Red - very uncertain
}

# ============================================================
# LOAD SPECIES DATA
# ============================================================

def load_label_mapping():
    """Load label mapping with species info"""
    if LABEL_MAPPING_PATH.exists():
        with open(LABEL_MAPPING_PATH, 'r') as f:
            return json.load(f)
    return {}

def load_test_results():
    """Load model test results"""
    if TEST_RESULTS_PATH.exists():
        with open(TEST_RESULTS_PATH, 'r') as f:
            return json.load(f)
    return {}

# ============================================================
# SPECIES ADDITIONAL INFO
# ============================================================

# Add common names, descriptions, and links
SPECIES_INFO = {
    "Accipiter cooperii": {
        "common_name": "Cooper's Hawk",
        "family": "Accipitridae",
        "habitat": "Forests, woodlands",
        "description": "Medium-sized hawk with rounded wings and long tail.",
        "similar_species": ["Sharp-shinned Hawk", "Northern Goshawk"],
        "xeno_canto_url": "https://xeno-canto.org/species/Accipiter-cooperii",
    },
    # Add more species info as needed...
}

def get_species_info(scientific_name):
    """Get additional info for a species"""
    default_info = {
        "common_name": scientific_name.split()[1] if len(scientific_name.split()) > 1 else scientific_name,
        "family": "Unknown",
        "habitat": "Various",
        "description": "North American bird species.",
        "similar_species": [],
        "xeno_canto_url": f"https://xeno-canto.org/species/{scientific_name.replace(' ', '-')}",
    }
    return SPECIES_INFO.get(scientific_name, default_info)