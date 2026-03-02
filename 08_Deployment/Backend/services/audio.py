"""
BirdSense Backend — Audio Processing Service
Load, chunk, and convert audio to mel-spectrograms.
Ported from Streamlit version with all Streamlit dependencies removed.
"""

from __future__ import annotations

import numpy as np
import librosa
from pathlib import Path

from config import AUDIO_CONFIG


# ── Loading ────────────────────────────────────────────────────

def load_audio(file_path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """
    Load an audio file and resample to the target sample rate.

    Returns:
        (audio_array, sample_rate)

    Raises:
        ValueError: If the file cannot be loaded.
    """
    sr = sr or AUDIO_CONFIG["sample_rate"]
    try:
        audio, actual_sr = librosa.load(str(file_path), sr=sr, mono=True)
        return audio, sr
    except Exception as exc:
        raise ValueError(f"Failed to load audio: {exc}") from exc


def get_audio_duration(audio: np.ndarray, sr: int | None = None) -> float:
    """Return the duration in seconds."""
    sr = sr or AUDIO_CONFIG["sample_rate"]
    return len(audio) / sr


# ── Chunking ───────────────────────────────────────────────────

def chunk_audio(
    audio: np.ndarray,
    sr: int | None = None,
    chunk_duration: float | None = None,
    overlap: float | None = None,
) -> list[tuple[np.ndarray, float, float]]:
    """
    Split audio into overlapping chunks.

    Returns:
        list of (chunk_array, start_time_s, end_time_s)
    """
    sr = sr or AUDIO_CONFIG["sample_rate"]
    chunk_duration = chunk_duration or AUDIO_CONFIG["chunk_duration"]
    overlap = overlap or AUDIO_CONFIG["chunk_overlap"]

    chunk_samples = int(chunk_duration * sr)
    hop_samples = int(chunk_samples * (1 - overlap))

    chunks: list[tuple[np.ndarray, float, float]] = []

    for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
        end = start + chunk_samples
        chunk = audio[start:end]
        chunks.append((chunk, start / sr, end / sr))

    # Handle residual tail (pad to chunk_samples)
    if len(audio) > chunk_samples:
        last_start_sample = chunks[-1][0].shape[0] if chunks else 0
        remaining = len(audio) - (start + chunk_samples) if chunks else len(audio)
        if remaining > 0 and remaining < chunk_samples:
            last_start = len(audio) - chunk_samples
            last_chunk = audio[last_start:]
            if len(last_chunk) < chunk_samples:
                last_chunk = np.pad(last_chunk, (0, chunk_samples - len(last_chunk)))
            chunks.append((last_chunk, last_start / sr, len(audio) / sr))

    return chunks


# ── Spectrogram Generation ─────────────────────────────────────

def generate_mel_spectrogram(
    audio: np.ndarray,
    sr: int | None = None,
    n_mels: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
    fmin: int | None = None,
    fmax: int | None = None,
) -> np.ndarray:
    """Generate a mel spectrogram in dB scale."""
    sr = sr or AUDIO_CONFIG["sample_rate"]
    n_mels = n_mels or AUDIO_CONFIG["n_mels"]
    n_fft = n_fft or AUDIO_CONFIG["n_fft"]
    hop_length = hop_length or AUDIO_CONFIG["hop_length"]
    fmin = fmin or AUDIO_CONFIG["fmin"]
    fmax = fmax or AUDIO_CONFIG["fmax"]

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length,
        fmin=fmin, fmax=fmax,
    )
    return librosa.power_to_db(mel_spec, ref=np.max)


def process_audio_to_spectrograms(
    audio: np.ndarray,
    sr: int | None = None,
    target_shape: tuple[int, int] = (128, 216),
) -> list[tuple[np.ndarray, float, float]]:
    """
    Chunk audio and convert each chunk to a mel spectrogram.

    Returns:
        list of (spectrogram, start_time, end_time)
    """
    sr = sr or AUDIO_CONFIG["sample_rate"]
    chunks = chunk_audio(audio, sr)

    spectrograms: list[tuple[np.ndarray, float, float]] = []

    for chunk, start_time, end_time in chunks:
        mel_spec = generate_mel_spectrogram(chunk, sr)

        # Resize time axis to target_shape
        if mel_spec.shape[1] > target_shape[1]:
            mel_spec = mel_spec[:, :target_shape[1]]
        elif mel_spec.shape[1] < target_shape[1]:
            mel_spec = np.pad(
                mel_spec,
                ((0, 0), (0, target_shape[1] - mel_spec.shape[1])),
                mode="constant",
                constant_values=mel_spec.min(),
            )

        spectrograms.append((mel_spec, start_time, end_time))

    return spectrograms


# ── Noise Reduction ────────────────────────────────────────────

def reduce_noise(audio: np.ndarray, sr: int | None = None, prop_decrease: float = 0.8) -> np.ndarray:
    """Apply stationary noise reduction. Falls back to original if noisereduce fails."""
    sr = sr or AUDIO_CONFIG["sample_rate"]
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease, stationary=True)
    except Exception:
        return audio
