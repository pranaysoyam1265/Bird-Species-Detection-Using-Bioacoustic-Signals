"""
COMPLETE PIPELINE: Process New Data + Combine + Prepare for Training
Step 1: Convert MP3 to WAV (22050 Hz, mono)
Step 2: Create 5-second chunks with 50% overlap
Step 3: Generate mel-spectrograms
Step 4: Combine old + new metadata
Step 5: Create label mapping and train/val/test splits
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")

# Input paths
AUDIO_DIR = PROJECT_ROOT / "01_Raw_Data" / "Audio_Recordings"
OLD_METADATA = PROJECT_ROOT / "01_Raw_Data" / "Metadata" / "metadata_merged.csv"
NEW_METADATA = PROJECT_ROOT / "01_Raw_Data" / "Metadata" / "aggressive_scrape_metadata.csv"

# Output paths
STANDARDIZED_DIR = PROJECT_ROOT / "02_Preprocessed" / "Standardized_Audio"
CHUNKS_DIR = PROJECT_ROOT / "02_Preprocessed" / "Audio_Chunks"
SPECTROGRAM_DIR = PROJECT_ROOT / "03_Features" / "Spectrograms"
LABELS_DIR = PROJECT_ROOT / "04_Labels" / "Processed_Labels"
SPLIT_DIR = PROJECT_ROOT / "04_Labels" / "Train_Val_Test_Split"

# Audio settings
TARGET_SR = 22050
CHUNK_DURATION = 5  # seconds
CHUNK_OVERLAP = 0.5  # 50% overlap
MIN_CHUNK_DURATION = 3  # minimum seconds for last chunk

# Spectrogram settings
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# Split settings
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_SAMPLES_PER_SPECIES = 50

# ============================================================
# STEP 1: CONVERT MP3 TO WAV
# ============================================================

def convert_mp3_to_wav():
    """Convert all MP3 files to standardized WAV"""
    print("=" * 70)
    print("STEP 1: CONVERTING MP3 TO WAV (22050 Hz, Mono)")
    print("=" * 70)
    
    STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all MP3 files that don't have a corresponding WAV
    mp3_files = list(AUDIO_DIR.glob("*.mp3"))
    
    # Also find WAV files that haven't been standardized
    wav_files = list(AUDIO_DIR.glob("*.wav"))
    
    # Check which ones already have standardized versions
    existing_standardized = set(f.stem for f in STANDARDIZED_DIR.glob("*.wav"))
    
    to_process = []
    for f in mp3_files + wav_files:
        if f.stem not in existing_standardized:
            to_process.append(f)
    
    print(f"📁 MP3 files found: {len(mp3_files)}")
    print(f"📁 WAV files found: {len(wav_files)}")
    print(f"✅ Already standardized: {len(existing_standardized)}")
    print(f"⏳ Need to process: {len(to_process)}")
    
    if not to_process:
        print("✅ All files already standardized!")
        return len(existing_standardized)
    
    success = 0
    failed = 0
    
    for filepath in tqdm(to_process, desc="Converting"):
        try:
            output_path = STANDARDIZED_DIR / f"{filepath.stem}.wav"
            
            if filepath.suffix.lower() == '.mp3':
                # Try pydub for MP3 first (handles corruption better)
                try:
                    audio = AudioSegment.from_mp3(str(filepath))
                    audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
                    audio.export(str(output_path), format="wav")
                except Exception as pydub_error:
                    # Fallback to librosa if pydub fails (e.g., FFmpeg not found)
                    y, sr = librosa.load(str(filepath), sr=TARGET_SR, mono=True)
                    if np.max(np.abs(y)) > 0:
                        y = y / np.max(np.abs(y))
                    sf.write(str(output_path), y, TARGET_SR)
            else:
                # Use librosa for WAV
                y, sr = librosa.load(str(filepath), sr=TARGET_SR, mono=True)
                if np.max(np.abs(y)) > 0:
                    y = y / np.max(np.abs(y))
                sf.write(str(output_path), y, TARGET_SR)
            
            success += 1
            
        except Exception as e:
            failed += 1
            if failed <= 10:  # Only show first 10 errors
                print(f"\n  ❌ Failed: {filepath.name} - {e}")
    
    total = len(existing_standardized) + success
    print(f"\n✅ Converted: {success}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Total standardized files: {total}")
    
    return total

# ============================================================
# STEP 2: CREATE 5-SECOND CHUNKS
# ============================================================

def create_chunks():
    """Create 5-second chunks with 50% overlap"""
    print("\n" + "=" * 70)
    print("STEP 2: CREATING 5-SECOND CHUNKS (50% OVERLAP)")
    print("=" * 70)
    
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all standardized WAV files
    wav_files = list(STANDARDIZED_DIR.glob("*.wav"))
    
    # Check which files already have chunks
    existing_chunks = set()
    for f in CHUNKS_DIR.glob("*.wav"):
        # Extract base filename from chunk name (e.g., XC12345_chunk_0.wav -> XC12345)
        parts = f.stem.rsplit("_chunk_", 1)
        if len(parts) == 2:
            existing_chunks.add(parts[0])
    
    to_process = [f for f in wav_files if f.stem not in existing_chunks]
    
    print(f"📁 Standardized files: {len(wav_files)}")
    print(f"✅ Already chunked: {len(existing_chunks)}")
    print(f"⏳ Need to chunk: {len(to_process)}")
    
    if not to_process:
        total_chunks = len(list(CHUNKS_DIR.glob("*.wav")))
        print(f"✅ All files already chunked! Total chunks: {total_chunks}")
        return total_chunks
    
    chunk_samples = int(CHUNK_DURATION * TARGET_SR)
    hop_samples = int(chunk_samples * (1 - CHUNK_OVERLAP))
    min_samples = int(MIN_CHUNK_DURATION * TARGET_SR)
    
    new_chunks = 0
    
    for filepath in tqdm(to_process, desc="Chunking"):
        try:
            y, sr = librosa.load(str(filepath), sr=TARGET_SR, mono=True)
            
            # Skip very short files
            if len(y) < min_samples:
                continue
            
            # Create chunks
            start = 0
            chunk_idx = 0
            
            while start + min_samples <= len(y):
                end = min(start + chunk_samples, len(y))
                chunk = y[start:end]
                
                # Pad if necessary
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
                
                # Save chunk
                chunk_name = f"{filepath.stem}_chunk_{chunk_idx}.wav"
                chunk_path = CHUNKS_DIR / chunk_name
                sf.write(str(chunk_path), chunk, TARGET_SR)
                
                new_chunks += 1
                chunk_idx += 1
                start += hop_samples
                
        except Exception as e:
            print(f"\n  ❌ Failed: {filepath.name} - {e}")
    
    total_chunks = len(list(CHUNKS_DIR.glob("*.wav")))
    print(f"\n✅ New chunks created: {new_chunks}")
    print(f"📁 Total chunks: {total_chunks}")
    
    return total_chunks

# ============================================================
# STEP 3: GENERATE MEL-SPECTROGRAMS
# ============================================================

def generate_spectrogram(chunk_path, output_dir):
    """Generate mel-spectrogram for a single chunk"""
    try:
        y, sr = librosa.load(str(chunk_path), sr=TARGET_SR, mono=True)
        
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Save as numpy array
        output_path = output_dir / f"{chunk_path.stem}.npy"
        np.save(str(output_path), mel_spec_db)
        
        return True
    except Exception:
        return False

def generate_spectrograms():
    """Generate mel-spectrograms for all chunks"""
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING MEL-SPECTROGRAMS (128 × 216)")
    print("=" * 70)
    
    SPECTROGRAM_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all chunks
    chunk_files = list(CHUNKS_DIR.glob("*.wav"))
    
    # Check which already have spectrograms
    existing_specs = set(f.stem for f in SPECTROGRAM_DIR.glob("*.npy"))
    to_process = [f for f in chunk_files if f.stem not in existing_specs]
    
    print(f"📁 Total chunks: {len(chunk_files)}")
    print(f"✅ Already have spectrograms: {len(existing_specs)}")
    print(f"⏳ Need spectrograms: {len(to_process)}")
    
    if not to_process:
        print("✅ All spectrograms already generated!")
        return len(existing_specs)
    
    success = 0
    failed = 0
    
    for chunk_path in tqdm(to_process, desc="Spectrograms"):
        if generate_spectrogram(chunk_path, SPECTROGRAM_DIR):
            success += 1
        else:
            failed += 1
    
    total = len(existing_specs) + success
    print(f"\n✅ New spectrograms: {success}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Total spectrograms: {total}")
    
    return total

# ============================================================
# STEP 4: COMBINE OLD + NEW METADATA
# ============================================================

def combine_metadata():
    """Merge old and new metadata"""
    print("\n" + "=" * 70)
    print("STEP 4: COMBINING OLD + NEW METADATA")
    print("=" * 70)
    
    # Load metadata files
    old_df = pd.read_csv(OLD_METADATA)
    new_df = pd.read_csv(NEW_METADATA)
    
    print(f"📁 Old metadata: {len(old_df)} recordings, "
          f"{old_df['species_scientific'].nunique()} species")
    print(f"📁 New metadata: {len(new_df)} recordings, "
          f"{new_df['species_scientific'].nunique()} species")
    
    # Standardize column names
    # Make sure both have: xc_id, species_scientific, species_english
    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)
    
    # Find common columns
    common_cols = list(old_cols & new_cols)
    print(f"📋 Common columns: {common_cols}")
    
    # Ensure xc_id is string type in both
    if 'xc_id' in old_df.columns:
        old_df['xc_id'] = old_df['xc_id'].astype(str)
    if 'xc_id' in new_df.columns:
        new_df['xc_id'] = new_df['xc_id'].astype(str)
    
    # Combine
    combined = pd.concat([old_df, new_df], ignore_index=True)
    
    # Remove duplicates by xc_id
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=['xc_id'], keep='first')
    after_dedup = len(combined)
    
    print(f"\n📊 Before dedup: {before_dedup}")
    print(f"📊 After dedup: {after_dedup}")
    print(f"📊 Duplicates removed: {before_dedup - after_dedup}")
    print(f"🐦 Total unique species: {combined['species_scientific'].nunique()}")
    
    # Save combined metadata
    output_path = PROJECT_ROOT / "01_Raw_Data" / "Metadata" / "metadata_final_combined.csv"
    combined.to_csv(output_path, index=False)
    print(f"💾 Saved: {output_path}")
    
    return combined

# ============================================================
# STEP 5: CREATE LABEL MAPPING AND SPLITS
# ============================================================

def create_labels_and_splits(metadata_df):
    """Match chunks to metadata, create labels and train/val/test splits"""
    print("\n" + "=" * 70)
    print("STEP 5: CREATING LABELS AND TRAIN/VAL/TEST SPLITS")
    print("=" * 70)
    
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all spectrogram files (these are our usable samples)
    spec_files = list(SPECTROGRAM_DIR.glob("*.npy"))
    print(f"📁 Total spectrograms: {len(spec_files)}")
    
    # Parse chunk filenames to extract XC IDs
    # Format: XC12345_chunk_0.npy -> xc_id = 12345
    chunk_data = []
    for spec_file in spec_files:
        filename = spec_file.stem  # e.g., XC12345_chunk_0
        
        # Extract XC ID
        try:
            # Handle format: XC12345_chunk_0
            base_name = filename.rsplit("_chunk_", 1)[0]  # XC12345
            xc_id = base_name.replace("XC", "").replace("xc", "")
            
            chunk_data.append({
                'spectrogram_path': str(spec_file),
                'chunk_filename': filename,
                'xc_id': str(xc_id)
            })
        except Exception:
            continue
    
    chunks_df = pd.DataFrame(chunk_data)
    print(f"📋 Parsed chunks: {len(chunks_df)}")
    
    # Ensure xc_id is string in metadata
    metadata_df['xc_id'] = metadata_df['xc_id'].astype(str)
    
    # Merge chunks with metadata
    merged = chunks_df.merge(
        metadata_df[['xc_id', 'species_scientific', 'species_english']].drop_duplicates(),
        on='xc_id',
        how='inner'
    )
    
    print(f"✅ Matched chunks: {len(merged)}")
    print(f"❌ Unmatched: {len(chunks_df) - len(merged)}")
    print(f"🐦 Species found: {merged['species_scientific'].nunique()}")
    
    # Filter species with minimum samples
    species_counts = merged['species_scientific'].value_counts()
    valid_species = species_counts[species_counts >= MIN_SAMPLES_PER_SPECIES].index.tolist()
    
    filtered = merged[merged['species_scientific'].isin(valid_species)].copy()
    
    print(f"\n🐦 Species with >= {MIN_SAMPLES_PER_SPECIES} samples: {len(valid_species)}")
    print(f"📋 Chunks after filter: {len(filtered)}")
    
    # Show distribution
    print(f"\n📊 Top 10 species:")
    for species, count in species_counts.head(10).items():
        english = merged[merged['species_scientific'] == species]['species_english'].iloc[0]
        print(f"   {english}: {count}")
    
    print(f"\n📊 Bottom 5 species (kept):")
    bottom_species = species_counts[species_counts.index.isin(valid_species)].tail(5)
    for species, count in bottom_species.items():
        english = merged[merged['species_scientific'] == species]['species_english'].iloc[0]
        print(f"   {english}: {count}")
    
    # Create label mapping
    sorted_species = sorted(valid_species)
    label_mapping = {species: idx for idx, species in enumerate(sorted_species)}
    
    # Also create reverse mapping with English names
    label_mapping_full = {}
    for species, idx in label_mapping.items():
        english = merged[merged['species_scientific'] == species]['species_english'].iloc[0]
        label_mapping_full[species] = {
            'index': idx,
            'english_name': english
        }
    
    # Save label mapping
    label_map_path = LABELS_DIR / "label_mapping_v3.json"
    with open(label_map_path, 'w') as f:
        json.dump(label_mapping_full, f, indent=2)
    print(f"\n🏷️ Label mapping saved: {label_map_path}")
    print(f"🏷️ Number of classes: {len(label_mapping)}")
    
    # Add numeric labels
    filtered['label'] = filtered['species_scientific'].map(label_mapping)
    
    # Stratified split by species
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        filtered,
        test_size=TEST_RATIO,
        stratify=filtered['species_scientific'],
        random_state=42
    )
    
    # Second split: train vs val
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        stratify=train_val['species_scientific'],
        random_state=42
    )
    
    print(f"\n📊 Train set: {len(train)} chunks ({len(train)/len(filtered)*100:.1f}%)")
    print(f"📊 Val set: {len(val)} chunks ({len(val)/len(filtered)*100:.1f}%)")
    print(f"📊 Test set: {len(test)} chunks ({len(test)/len(filtered)*100:.1f}%)")
    
    # Save splits
    train.to_csv(SPLIT_DIR / "train_v3.csv", index=False)
    val.to_csv(SPLIT_DIR / "val_v3.csv", index=False)
    test.to_csv(SPLIT_DIR / "test_v3.csv", index=False)
    
    print(f"\n💾 Saved splits to: {SPLIT_DIR}")
    print(f"   - train_v3.csv")
    print(f"   - val_v3.csv")
    print(f"   - test_v3.csv")
    
    return len(label_mapping), len(train), len(val), len(test)

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 COMPLETE DATA PIPELINE: PROCESS + COMBINE + PREPARE")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 70)
    
    # Step 1: Convert MP3 to WAV
    total_standardized = convert_mp3_to_wav()
    
    # Step 2: Create chunks
    total_chunks = create_chunks()
    
    # Step 3: Generate spectrograms
    total_specs = generate_spectrograms()
    
    # Step 4: Combine metadata
    combined_metadata = combine_metadata()
    
    # Step 5: Create labels and splits
    try:
        num_classes, n_train, n_val, n_test = create_labels_and_splits(combined_metadata)
    except ImportError:
        print("\n⚠️ scikit-learn not installed. Installing...")
        os.system(f"{Path(os.sys.executable)} -m pip install scikit-learn")
        from sklearn.model_selection import train_test_split
        num_classes, n_train, n_val, n_test = create_labels_and_splits(combined_metadata)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ COMPLETE PIPELINE FINISHED!")
    print("=" * 70)
    print(f"""
📊 FINAL SUMMARY
{'─' * 50}
Standardized audio files: {total_standardized}
Audio chunks (5-sec):     {total_chunks}
Spectrograms:             {total_specs}
Species (classes):        {num_classes}

TRAIN/VAL/TEST SPLIT
{'─' * 50}
Train: {n_train} chunks
Val:   {n_val} chunks
Test:  {n_test} chunks
Total: {n_train + n_val + n_test} chunks

FILES CREATED
{'─' * 50}
{PROJECT_ROOT / '01_Raw_Data' / 'Metadata' / 'metadata_final_combined.csv'}
{LABELS_DIR / 'label_mapping_v3.json'}
{SPLIT_DIR / 'train_v3.csv'}
{SPLIT_DIR / 'val_v3.csv'}
{SPLIT_DIR / 'test_v3.csv'}

🚀 Ready for training!
""")