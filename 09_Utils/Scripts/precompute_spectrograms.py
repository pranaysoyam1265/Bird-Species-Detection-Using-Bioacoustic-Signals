"""
Script: precompute_spectrograms.py
Purpose: Pre-compute spectrograms (ONE TIME, ~15-20 minutes)
Location: 09_Utils/Scripts/precompute_spectrograms.py

NO QUALITY LOSS - Same spectrograms, just computed once!
"""

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION (Same as training!)
# ============================================================
BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"

AUDIO_DIR = os.path.join(BASE_DIR, "02_Preprocessed", "Audio_Chunks")
TRAIN_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "val.csv")
TEST_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "test.csv")

SPEC_DIR = os.path.join(BASE_DIR, "03_Features", "Spectrograms_Precomputed")

# Audio settings (MUST match training exactly!)
SAMPLE_RATE = 22050
DURATION = 5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 150
FMAX = 15000


def compute_spectrogram(audio_path):
    """
    Compute mel spectrogram - SAME as training pipeline.
    Returns normalized spectrogram (0-1 range).
    """
    try:
        # Load audio
        audio, _ = librosa.load(
            audio_path, 
            sr=SAMPLE_RATE, 
            mono=True, 
            dtype=np.float32
        )
        
        # Ensure exact length (5 seconds)
        target_len = SAMPLE_RATE * DURATION
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        else:
            audio = audio[:target_len]
        
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            fmin=FMIN,
            fmax=FMAX
        )
        
        # Convert to dB scale
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to 0-1 (same as training)
        mel_min = mel_db.min()
        mel_max = mel_db.max()
        
        if mel_max - mel_min > 1e-6:
            mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
        else:
            mel_norm = np.zeros_like(mel_db)
        
        return mel_norm.astype(np.float32)
    
    except Exception as e:
        print(f"\n   ⚠️ Error: {audio_path}: {e}")
        return None


def process_split(csv_path, split_name):
    """Process all files in a data split"""
    
    if not os.path.exists(csv_path):
        print(f"   ❌ CSV not found: {csv_path}")
        return 0, 0
    
    df = pd.read_csv(csv_path)
    print(f"\n📂 Processing {split_name}: {len(df)} files")
    
    # Create output directory
    split_dir = os.path.join(SPEC_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    success = 0
    failed = 0
    skipped = 0
    
    pbar = tqdm(df.iterrows(), total=len(df), desc=f"   {split_name}")
    
    for idx, row in pbar:
        filename = row['chunk_file']
        audio_path = os.path.join(AUDIO_DIR, filename)
        
        # Output filename
        spec_filename = filename.replace('.wav', '.npy').replace('.mp3', '.npy')
        spec_path = os.path.join(split_dir, spec_filename)
        
        # Skip if already exists
        if os.path.exists(spec_path):
            skipped += 1
            success += 1
            continue
        
        # Check audio exists
        if not os.path.exists(audio_path):
            failed += 1
            continue
        
        # Compute spectrogram
        spec = compute_spectrogram(audio_path)
        
        if spec is not None:
            np.save(spec_path, spec)
            success += 1
        else:
            failed += 1
        
        pbar.set_postfix({'done': success, 'fail': failed, 'skip': skipped})
    
    print(f"   ✅ {split_name}: {success} success, {failed} failed, {skipped} skipped")
    return success, failed


def main():
    print("=" * 70)
    print("⚡ PRE-COMPUTING SPECTROGRAMS (One-Time Setup)")
    print("=" * 70)
    print("\nThis will take ~15-20 minutes but makes training 5-10x faster!")
    print("Quality is IDENTICAL - same spectrograms, just pre-computed.\n")
    
    # Create output directory
    os.makedirs(SPEC_DIR, exist_ok=True)
    
    # Save config for verification
    config = {
        'sample_rate': SAMPLE_RATE,
        'duration': DURATION,
        'n_mels': N_MELS,
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'fmin': FMIN,
        'fmax': FMAX,
        'expected_shape': [N_MELS, 216]
    }
    
    config_path = os.path.join(SPEC_DIR, "spectrogram_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Process each split
    total_success = 0
    total_failed = 0
    
    for csv_path, split_name in [
        (TRAIN_CSV, "train"),
        (VAL_CSV, "val"),
        (TEST_CSV, "test")
    ]:
        s, f = process_split(csv_path, split_name)
        total_success += s
        total_failed += f
    
    # Calculate disk usage
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(SPEC_DIR):
        for f in files:
            if f.endswith('.npy'):
                total_size += os.path.getsize(os.path.join(root, f))
                file_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ PRE-COMPUTATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Total spectrograms: {total_success:,}")
    print(f"   Failed:            {total_failed}")
    print(f"   Disk usage:        {total_size / (1024**3):.2f} GB")
    print(f"   Location:          {SPEC_DIR}")
    
    print(f"\n🚀 Next step - Run fast training:")
    print(f"   python 09_Utils/Scripts/train_fast_quality.py")


if __name__ == "__main__":
    main()