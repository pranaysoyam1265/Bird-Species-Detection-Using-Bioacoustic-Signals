"""
generate_spectrograms.py
Fast spectrogram generation for training

Optimized for speed - generates mel-spectrograms from audio chunks
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from pathlib import Path
from tqdm import tqdm
import warnings
import json
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")

# Input paths
CHUNKS_DIR = BASE_DIR / "02_Preprocessed" / "Audio_Chunks"
TRAIN_CSV = BASE_DIR / "04_Labels" / "Train_Val_Test_Split" / "train.csv"
VAL_CSV = BASE_DIR / "04_Labels" / "Train_Val_Test_Split" / "val.csv"
TEST_CSV = BASE_DIR / "04_Labels" / "Train_Val_Test_Split" / "test.csv"

# Output paths
SPECTROGRAMS_DIR = BASE_DIR / "03_Features" / "Spectrograms"
TRAIN_SPEC_DIR = SPECTROGRAMS_DIR / "train"
VAL_SPEC_DIR = SPECTROGRAMS_DIR / "val"
TEST_SPEC_DIR = SPECTROGRAMS_DIR / "test"

# Spectrogram settings (optimized for bird calls)
SAMPLE_RATE = 22050
N_MELS = 128          # Mel frequency bins
N_FFT = 2048          # FFT window size
HOP_LENGTH = 512      # Hop between windows
FMIN = 150            # Min frequency (Hz) - filters out low rumble
FMAX = 15000          # Max frequency (Hz) - bird calls are here
DURATION = 5.0        # Expected chunk duration (seconds)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_directories():
    """Create output directories"""
    for dir_path in [TRAIN_SPEC_DIR, VAL_SPEC_DIR, TEST_SPEC_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✅ Created spectrogram directories")

def audio_to_melspectrogram(audio_path):
    """
    Convert audio file to mel-spectrogram
    Returns numpy array of shape (n_mels, time_steps)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad if too short
        expected_length = int(SAMPLE_RATE * DURATION)
        if len(y) < expected_length:
            y = np.pad(y, (0, expected_length - len(y)), mode='constant')
        
        # Create mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            fmin=FMIN,
            fmax=FMAX
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to 0-1 range
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_norm.astype(np.float32)
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_split(csv_path, chunks_dir, output_dir, split_name):
    """Process all chunks in a split"""
    
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split...")
    print(f"{'='*60}")
    
    # Load split CSV
    df = pd.read_csv(csv_path)
    print(f"Chunks to process: {len(df)}")
    
    successful = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name}"):
        chunk_file = row['chunk_file']
        label_id = row['label_id']
        
        # Input path
        audio_path = chunks_dir / chunk_file
        
        # Output path (save as .npy for fast loading)
        spec_filename = chunk_file.replace('.wav', '.npy')
        spec_path = output_dir / spec_filename
        
        # Skip if already exists
        if spec_path.exists():
            successful += 1
            continue
        
        # Generate spectrogram
        if audio_path.exists():
            mel_spec = audio_to_melspectrogram(audio_path)
            
            if mel_spec is not None:
                # Save as numpy array
                np.save(spec_path, mel_spec)
                successful += 1
            else:
                failed += 1
        else:
            failed += 1
    
    print(f"✅ {split_name}: {successful} successful, {failed} failed")
    return successful, failed

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🎵 SPECTROGRAM GENERATION")
    print("=" * 70)
    
    # Create directories
    create_directories()
    
    # Check if label files exist
    for csv_path in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not csv_path.exists():
            print(f"❌ ERROR: {csv_path} not found!")
            print("Please run label_preparation.py first!")
            return
    
    # Process each split
    train_success, train_fail = process_split(TRAIN_CSV, CHUNKS_DIR, TRAIN_SPEC_DIR, "Train")
    val_success, val_fail = process_split(VAL_CSV, CHUNKS_DIR, VAL_SPEC_DIR, "Val")
    test_success, test_fail = process_split(TEST_CSV, CHUNKS_DIR, TEST_SPEC_DIR, "Test")
    
    # Save config
    config = {
        'sample_rate': SAMPLE_RATE,
        'n_mels': N_MELS,
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'fmin': FMIN,
        'fmax': FMAX,
        'duration': DURATION,
        'train_spectrograms': train_success,
        'val_spectrograms': val_success,
        'test_spectrograms': test_success
    }
    
    config_path = SPECTROGRAMS_DIR / "spectrogram_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Summary
    total = train_success + val_success + test_success
    print("\n" + "=" * 70)
    print("📊 SPECTROGRAM GENERATION COMPLETE")
    print("=" * 70)
    print(f"""
    ✅ Total spectrograms generated: {total}
    
    📂 Output:
    ├── Train: {train_success} spectrograms → {TRAIN_SPEC_DIR}
    ├── Val:   {val_success} spectrograms → {VAL_SPEC_DIR}
    └── Test:  {test_success} spectrograms → {TEST_SPEC_DIR}
    
    📐 Spectrogram shape: ({N_MELS}, ~{int(DURATION * SAMPLE_RATE / HOP_LENGTH)})
    
    🎯 Next: Run training script!
    """)

if __name__ == "__main__":
    main()