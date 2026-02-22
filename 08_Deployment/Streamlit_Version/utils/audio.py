# Save as: 08_Deployment/utils/audio.py

"""
Audio loading and processing utilities
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import tempfile
import os
import streamlit as st

# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(file_path, sr=22050):
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (default: 22050)
    
    Returns:
        tuple: (audio_array, sample_rate)
    """
    try:
        audio, orig_sr = librosa.load(file_path, sr=sr, mono=True)
        return audio, sr
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None, None

def load_uploaded_audio(uploaded_file, sr=22050):
    """
    Load audio from Streamlit uploaded file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        sr: Target sample rate
    
    Returns:
        tuple: (audio_array, sample_rate, temp_path)
    """
    # Save to temporary file
    suffix = Path(uploaded_file.name).suffix
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    
    # Load audio
    audio, sr = load_audio(tmp_path, sr)
    
    return audio, sr, tmp_path

# ============================================================
# AUDIO CHUNKING
# ============================================================

def chunk_audio(audio, sr, chunk_duration=5.0, overlap=0.5):
    """
    Split audio into overlapping chunks.
    
    Args:
        audio: Audio array
        sr: Sample rate
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap ratio (0.5 = 50%)
    
    Returns:
        list of (chunk_array, start_time, end_time)
    """
    chunk_samples = int(chunk_duration * sr)
    hop_samples = int(chunk_samples * (1 - overlap))
    
    chunks = []
    
    for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
        end = start + chunk_samples
        chunk = audio[start:end]
        
        start_time = start / sr
        end_time = end / sr
        
        chunks.append((chunk, start_time, end_time))
    
    # Handle last chunk if audio doesn't divide evenly
    if len(audio) > chunk_samples:
        remaining = len(audio) - (start + chunk_samples)
        if remaining > 0 and remaining < chunk_samples:
            # Pad the last chunk
            last_start = len(audio) - chunk_samples
            last_chunk = audio[last_start:]
            
            if len(last_chunk) < chunk_samples:
                last_chunk = np.pad(last_chunk, (0, chunk_samples - len(last_chunk)))
            
            chunks.append((last_chunk, last_start / sr, len(audio) / sr))
    
    return chunks

# ============================================================
# SPECTROGRAM GENERATION
# ============================================================

def generate_mel_spectrogram(audio, sr=22050, n_mels=128, n_fft=2048, 
                              hop_length=512, fmin=20, fmax=10000):
    """
    Generate mel spectrogram from audio.
    
    Args:
        audio: Audio array
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length
        fmin: Minimum frequency
        fmax: Maximum frequency
    
    Returns:
        numpy array: Mel spectrogram in dB scale
    """
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def process_audio_to_spectrograms(audio, sr=22050, chunk_duration=5.0, 
                                   overlap=0.5, target_shape=(128, 216)):
    """
    Process audio file into mel spectrograms for model input.
    
    Args:
        audio: Audio array
        sr: Sample rate
        chunk_duration: Duration of each chunk
        overlap: Overlap ratio
        target_shape: Target spectrogram shape (n_mels, time_frames)
    
    Returns:
        list of (spectrogram, start_time, end_time)
    """
    # Split into chunks
    chunks = chunk_audio(audio, sr, chunk_duration, overlap)
    
    spectrograms = []
    
    for chunk, start_time, end_time in chunks:
        # Generate mel spectrogram
        mel_spec = generate_mel_spectrogram(chunk, sr)
        
        # Resize if needed
        if mel_spec.shape != target_shape:
            # Resize time axis
            if mel_spec.shape[1] > target_shape[1]:
                mel_spec = mel_spec[:, :target_shape[1]]
            elif mel_spec.shape[1] < target_shape[1]:
                mel_spec = np.pad(
                    mel_spec, 
                    ((0, 0), (0, target_shape[1] - mel_spec.shape[1])),
                    mode='constant',
                    constant_values=mel_spec.min()
                )
        
        spectrograms.append((mel_spec, start_time, end_time))
    
    return spectrograms

# ============================================================
# NOISE REDUCTION
# ============================================================

def reduce_noise(audio, sr=22050, prop_decrease=0.8):
    """
    Apply noise reduction to audio.
    
    Args:
        audio: Audio array
        sr: Sample rate
        prop_decrease: Proportion to decrease noise (0-1)
    
    Returns:
        Denoised audio array
    """
    try:
        import noisereduce as nr
        
        # Apply noise reduction
        reduced = nr.reduce_noise(
            y=audio, 
            sr=sr, 
            prop_decrease=prop_decrease,
            stationary=True
        )
        
        return reduced
    
    except ImportError:
        st.warning("noisereduce not installed. Skipping noise reduction.")
        return audio
    except Exception as e:
        st.warning(f"Noise reduction failed: {e}")
        return audio

# ============================================================
# AUDIO ANALYSIS
# ============================================================

def analyze_audio_quality(audio, sr=22050):
    """
    Analyze audio quality metrics.
    
    Args:
        audio: Audio array
        sr: Sample rate
    
    Returns:
        dict with quality metrics
    """
    # Duration
    duration = len(audio) / sr
    
    # RMS energy (loudness)
    rms = np.sqrt(np.mean(audio**2))
    rms_db = 20 * np.log10(rms + 1e-10)
    
    # Peak amplitude
    peak = np.max(np.abs(audio))
    peak_db = 20 * np.log10(peak + 1e-10)
    
    # Clipping detection
    clipping_threshold = 0.99
    clipping_samples = np.sum(np.abs(audio) > clipping_threshold)
    clipping_percentage = 100 * clipping_samples / len(audio)
    
    # Signal-to-noise ratio estimate (simple)
    # Using top 10% of energy frames vs bottom 10%
    frame_length = int(0.025 * sr)  # 25ms frames
    hop = frame_length // 2
    
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop)
    frame_energy = np.sum(frames**2, axis=0)
    
    sorted_energy = np.sort(frame_energy)
    n = len(sorted_energy)
    
    signal_energy = np.mean(sorted_energy[int(0.9*n):])  # Top 10%
    noise_energy = np.mean(sorted_energy[:int(0.1*n)])   # Bottom 10%
    
    snr_estimate = 10 * np.log10((signal_energy + 1e-10) / (noise_energy + 1e-10))
    
    # Quality score (0-100)
    quality_score = 0
    
    # Penalize clipping
    if clipping_percentage < 0.1:
        quality_score += 25
    elif clipping_percentage < 1:
        quality_score += 15
    
    # Reward good SNR
    if snr_estimate > 20:
        quality_score += 25
    elif snr_estimate > 10:
        quality_score += 15
    elif snr_estimate > 5:
        quality_score += 5
    
    # Reward appropriate loudness
    if -30 < rms_db < -10:
        quality_score += 25
    elif -40 < rms_db < -5:
        quality_score += 15
    
    # Reward appropriate duration
    if 3 < duration < 30:
        quality_score += 25
    elif 1 < duration < 60:
        quality_score += 15
    
    # Quality level
    if quality_score >= 75:
        quality_level = "Excellent"
    elif quality_score >= 50:
        quality_level = "Good"
    elif quality_score >= 25:
        quality_level = "Fair"
    else:
        quality_level = "Poor"
    
    return {
        'duration': duration,
        'sample_rate': sr,
        'rms_db': rms_db,
        'peak_db': peak_db,
        'clipping_percentage': clipping_percentage,
        'snr_estimate': snr_estimate,
        'quality_score': quality_score,
        'quality_level': quality_level,
    }

# ============================================================
# AUDIO EXPORT
# ============================================================

def export_audio_chunk(audio, sr, start_time, end_time, output_path):
    """
    Export a chunk of audio to file.
    
    Args:
        audio: Full audio array
        sr: Sample rate
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Output file path
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    chunk = audio[start_sample:end_sample]
    
    sf.write(output_path, chunk, sr)