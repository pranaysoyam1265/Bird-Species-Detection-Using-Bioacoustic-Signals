"""
BirdSense Backend — ML Inference Service
Loads the model once at startup and exposes prediction functions.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from config import MODEL_PATH, LABEL_MAPPING_PATH, DEVICE, MODEL_CONFIG
from models.classifier import BirdClassifier

logger = logging.getLogger("birdsense.inference")

# ── Module-level singletons (populated on startup) ─────────────
_model: BirdClassifier | None = None
_device: torch.device = torch.device("cpu")
_label_mapping: dict = {}
_idx_to_english: dict[int, str] = {}
_idx_to_scientific: dict[int, str] = {}


# ── Startup ────────────────────────────────────────────────────

def load_model() -> None:
    """Load the model and label mapping into module-level singletons."""
    global _model, _device, _label_mapping, _idx_to_english, _idx_to_scientific

    # Device
    device_str = DEVICE
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        device_str = "cpu"
    _device = torch.device(device_str)
    logger.info("Using device: %s", _device)

    # Model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = BirdClassifier(
        num_classes=MODEL_CONFIG["num_classes"],
        dropout_rate=MODEL_CONFIG["dropout_rate"],
    )

    checkpoint = torch.load(str(MODEL_PATH), map_location=_device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(_device)
    model.eval()
    _model = model

    metadata = {
        "epoch": checkpoint.get("epoch", "N/A"),
        "val_acc": checkpoint.get("val_acc", 0),
        "val_top5": checkpoint.get("val_top5", 0),
    }
    logger.info("Model loaded: %s  (val_acc=%.2f%%)", MODEL_PATH.name, metadata["val_acc"] * 100)

    # Labels
    if not LABEL_MAPPING_PATH.exists():
        raise FileNotFoundError(f"Label mapping not found: {LABEL_MAPPING_PATH}")

    with open(LABEL_MAPPING_PATH, "r") as f:
        _label_mapping = json.load(f)

    for scientific_name, info in _label_mapping.items():
        idx = info["index"]
        _idx_to_scientific[idx] = scientific_name
        _idx_to_english[idx] = info.get("english_name", scientific_name)

    logger.info("Loaded %d species from label mapping", len(_label_mapping))


def is_model_loaded() -> bool:
    return _model is not None


# ── Preprocessing ──────────────────────────────────────────────

def preprocess_spectrogram(spec: np.ndarray) -> torch.Tensor:
    """
    Normalise a mel spectrogram and convert to a 3-channel tensor.
    Must match the training preprocessing exactly.
    """
    spec_min, spec_max = spec.min(), spec.max()
    if spec_max - spec_min > 0:
        spec = (spec - spec_min) / (spec_max - spec_min)
    else:
        spec = np.zeros_like(spec)

    spec = np.clip(spec, 0, 1)

    # (1, H, W) → (3, H, W)
    tensor = torch.FloatTensor(spec).unsqueeze(0).repeat(3, 1, 1)
    return tensor


# ── Single prediction ─────────────────────────────────────────

def predict_single(spectrogram: np.ndarray, top_k: int = 10) -> dict:
    """
    Run inference on one spectrogram.

    Returns:
        dict with keys: top_indices, top_probs, predicted_class, confidence
    """
    assert _model is not None, "Model not loaded — call load_model() first"

    input_tensor = preprocess_spectrogram(spectrogram).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(input_tensor)
        probs = torch.softmax(logits, dim=1)

    probs_np = probs[0].cpu().numpy()
    top_indices = np.argsort(probs_np)[::-1][:top_k]

    return {
        "probabilities": probs_np,
        "top_indices": top_indices,
        "top_probs": probs_np[top_indices],
        "predicted_class": int(top_indices[0]),
        "confidence": float(probs_np[top_indices[0]]),
    }


# ── Batch prediction ──────────────────────────────────────────

def predict_batch(
    spectrograms: list[np.ndarray],
    top_k: int = 10,
    batch_size: int = 16,
) -> list[dict]:
    """Run inference on a list of spectrograms in batches."""
    assert _model is not None, "Model not loaded"

    results: list[dict] = []

    for i in range(0, len(spectrograms), batch_size):
        batch = spectrograms[i : i + batch_size]
        tensors = [preprocess_spectrogram(s) for s in batch]
        batch_tensor = torch.stack(tensors).to(_device)

        with torch.no_grad():
            logits = _model(batch_tensor)
            probs = torch.softmax(logits, dim=1)

        for prob in probs:
            prob_np = prob.cpu().numpy()
            top_indices = np.argsort(prob_np)[::-1][:top_k]
            results.append({
                "probabilities": prob_np,
                "top_indices": top_indices,
                "top_probs": prob_np[top_indices],
                "predicted_class": int(top_indices[0]),
                "confidence": float(prob_np[top_indices[0]]),
            })

    return results


# ── Label helpers ──────────────────────────────────────────────

def get_english_name(idx: int) -> str:
    return _idx_to_english.get(idx, f"Unknown ({idx})")


def get_scientific_name(idx: int) -> str:
    return _idx_to_scientific.get(idx, f"Unknown ({idx})")


def get_all_species() -> list[dict]:
    """Return all species as a list of {index, scientific_name, english_name}."""
    return [
        {
            "index": info["index"],
            "scientific_name": scientific,
            "english_name": info.get("english_name", scientific),
        }
        for scientific, info in _label_mapping.items()
    ]


def get_species_by_index(idx: int) -> dict | None:
    scientific = _idx_to_scientific.get(idx)
    if scientific is None:
        return None
    info = _label_mapping[scientific]
    return {
        "index": idx,
        "scientific_name": scientific,
        "english_name": info.get("english_name", scientific),
    }
