"""
Script: debug_dataset.py
Purpose: Diagnose why we're getting 100% accuracy (data leakage/bug)
Location: Save to 09_Utils/Scripts/debug_dataset.py
"""

import os
import json
import pandas as pd
import numpy as np
import librosa
import torch
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"
TRAIN_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split", "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split", "val.csv")
TEST_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split", "test.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "02_Preprocessed", "Audio_Chunks")
LABEL_MAPPING = os.path.join(BASE_DIR, "04_Labels", "Processed_Labels", "label_mapping.json")


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    print_header("🔍 DATASET DEBUGGING TOOL")
    
    # =========================================================
    # CHECK 1: Load and examine CSVs
    # =========================================================
    print_header("CHECK 1: CSV FILES")
    
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    print(f"\n📊 Dataset Sizes:")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    print(f"   Test:  {len(test_df)} samples")
    
    print(f"\n📋 Train CSV Columns: {list(train_df.columns)}")
    print(f"\n📋 First 5 rows of train.csv:")
    print(train_df.head().to_string())
    
    # Detect column names
    file_col = None
    for col in ['filename', 'file', 'filepath', 'path', 'chunk_file', 'audio_file']:
        if col in train_df.columns:
            file_col = col
            break
    
    species_col = None
    for col in ['species', 'label', 'class', 'scientific_name']:
        if col in train_df.columns:
            species_col = col
            break
    
    print(f"\n   Detected file column: '{file_col}'")
    print(f"   Detected species column: '{species_col}'")
    
    if file_col is None or species_col is None:
        print("\n❌ ERROR: Could not detect required columns!")
        return
    
    # =========================================================
    # CHECK 2: Data Leakage - Overlapping files
    # =========================================================
    print_header("CHECK 2: DATA LEAKAGE (Overlapping Files)")
    
    train_files = set(train_df[file_col].tolist())
    val_files = set(val_df[file_col].tolist())
    test_files = set(test_df[file_col].tolist())
    
    train_val_overlap = train_files.intersection(val_files)
    train_test_overlap = train_files.intersection(test_files)
    val_test_overlap = val_files.intersection(test_files)
    
    print(f"\n   Train ∩ Val overlap:  {len(train_val_overlap)} files")
    print(f"   Train ∩ Test overlap: {len(train_test_overlap)} files")
    print(f"   Val ∩ Test overlap:   {len(val_test_overlap)} files")
    
    if len(train_val_overlap) > 0:
        print(f"\n   ⚠️  LEAKAGE DETECTED! Examples of overlapping files:")
        for f in list(train_val_overlap)[:5]:
            print(f"      - {f}")
    
    # Check for same RECORDING (not just chunk) across splits
    print(f"\n   Checking recording-level overlap...")
    
    def extract_recording_id(filename):
        # Extract XC ID or base recording name
        base = str(filename).replace('.wav', '').replace('.mp3', '')
        if '_chunk_' in base:
            return base.split('_chunk_')[0]
        return base
    
    train_recordings = set(train_df[file_col].apply(extract_recording_id))
    val_recordings = set(val_df[file_col].apply(extract_recording_id))
    
    recording_overlap = train_recordings.intersection(val_recordings)
    print(f"   Recording-level overlap: {len(recording_overlap)} recordings")
    
    if len(recording_overlap) > 0:
        print(f"\n   🚨 RECORDING LEAKAGE! Same recordings in train AND val!")
        print(f"   Examples:")
        for r in list(recording_overlap)[:10]:
            print(f"      - {r}")
        print(f"\n   This means chunks from the SAME audio file are in both sets!")
        print(f"   The model is memorizing specific recordings, not learning bird sounds!")
    
    # =========================================================
    # CHECK 3: Label Distribution
    # =========================================================
    print_header("CHECK 3: LABEL DISTRIBUTION")
    
    train_labels = train_df[species_col].value_counts()
    val_labels = val_df[species_col].value_counts()
    
    print(f"\n   Unique species in train: {len(train_labels)}")
    print(f"   Unique species in val:   {len(val_labels)}")
    
    print(f"\n   Train label distribution (top 10):")
    for species, count in train_labels.head(10).items():
        print(f"      {species}: {count}")
    
    print(f"\n   Val label distribution (top 10):")
    for species, count in val_labels.head(10).items():
        print(f"      {species}: {count}")
    
    # Check if all species are in both sets
    train_species = set(train_df[species_col].unique())
    val_species = set(val_df[species_col].unique())
    
    missing_in_val = train_species - val_species
    missing_in_train = val_species - train_species
    
    if missing_in_val:
        print(f"\n   ⚠️  Species in train but not in val: {missing_in_val}")
    if missing_in_train:
        print(f"\n   ⚠️  Species in val but not in train: {missing_in_train}")
    
    # =========================================================
    # CHECK 4: Label Mapping
    # =========================================================
    print_header("CHECK 4: LABEL MAPPING")
    
    with open(LABEL_MAPPING, 'r') as f:
        label_map = json.load(f)
    
    print(f"\n   Label mapping has {len(label_map)} entries")
    print(f"\n   First 10 mappings:")
    for i, (species, idx) in enumerate(list(label_map.items())[:10]):
        print(f"      '{species}' → {idx}")
    
    # Check if all species in train/val are in mapping
    unmapped_train = [s for s in train_species if s not in label_map]
    unmapped_val = [s for s in val_species if s not in label_map]
    
    if unmapped_train:
        print(f"\n   ❌ Unmapped species in train: {unmapped_train}")
    if unmapped_val:
        print(f"\n   ❌ Unmapped species in val: {unmapped_val}")
    
    # =========================================================
    # CHECK 5: Audio Files Existence
    # =========================================================
    print_header("CHECK 5: AUDIO FILE EXISTENCE")
    
    def find_audio_file(filename):
        """Search for audio file"""
        # Direct path
        path = os.path.join(AUDIO_DIR, filename)
        if os.path.exists(path):
            return path
        
        # Search recursively
        for root, dirs, files in os.walk(AUDIO_DIR):
            if filename in files:
                return os.path.join(root, filename)
        
        return None
    
    # Check first 20 files from train
    print(f"\n   Checking first 20 train files...")
    found = 0
    not_found = []
    
    for filename in train_df[file_col].head(20):
        path = find_audio_file(filename)
        if path:
            found += 1
        else:
            not_found.append(filename)
    
    print(f"   Found: {found}/20")
    print(f"   Not found: {len(not_found)}")
    
    if not_found:
        print(f"\n   ❌ Missing files:")
        for f in not_found[:5]:
            print(f"      - {f}")
    
    # Show what's actually in the audio directory
    print(f"\n   Audio directory contents ({AUDIO_DIR}):")
    if os.path.exists(AUDIO_DIR):
        contents = os.listdir(AUDIO_DIR)[:10]
        print(f"   First 10 items: {contents}")
        print(f"   Total items: {len(os.listdir(AUDIO_DIR))}")
        
        # Check if it's organized by species
        first_item = os.path.join(AUDIO_DIR, contents[0]) if contents else None
        if first_item and os.path.isdir(first_item):
            print(f"   ℹ️  Audio is organized in subdirectories")
            subdir_contents = os.listdir(first_item)[:5]
            print(f"   Contents of '{contents[0]}': {subdir_contents}")
    else:
        print(f"   ❌ Directory does not exist!")
    
    # =========================================================
    # CHECK 6: Actual Audio Loading Test
    # =========================================================
    print_header("CHECK 6: AUDIO LOADING TEST")
    
    test_files_to_load = []
    for filename in train_df[file_col].head(10):
        path = find_audio_file(filename)
        if path:
            test_files_to_load.append((filename, path))
    
    if not test_files_to_load:
        print("   ❌ No files found to test!")
    else:
        print(f"\n   Testing {len(test_files_to_load)} files...")
        
        audio_hashes = []
        for filename, path in test_files_to_load:
            try:
                audio, sr = librosa.load(path, sr=22050, mono=True)
                audio_hash = hash(audio.tobytes())
                audio_hashes.append(audio_hash)
                print(f"   ✅ {filename}: {len(audio)} samples, mean={audio.mean():.4f}, std={audio.std():.4f}")
            except Exception as e:
                print(f"   ❌ {filename}: Error - {e}")
        
        # Check if all audio files are identical
        unique_hashes = len(set(audio_hashes))
        if unique_hashes < len(audio_hashes):
            print(f"\n   ⚠️  WARNING: Only {unique_hashes} unique audio files out of {len(audio_hashes)}!")
            print(f"   Some files might be identical or corrupted!")
    
    # =========================================================
    # CHECK 7: Spectrogram Generation Test
    # =========================================================
    print_header("CHECK 7: SPECTROGRAM GENERATION TEST")
    
    if test_files_to_load:
        print(f"\n   Generating spectrograms for {min(3, len(test_files_to_load))} files...")
        
        spec_values = []
        for filename, path in test_files_to_load[:3]:
            try:
                audio, sr = librosa.load(path, sr=22050, mono=True)
                
                # Generate mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=audio, sr=22050, n_mels=128, n_fft=2048, 
                    hop_length=512, fmin=150, fmax=15000
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                spec_values.append({
                    'file': filename,
                    'shape': mel_spec_db.shape,
                    'min': mel_spec_db.min(),
                    'max': mel_spec_db.max(),
                    'mean': mel_spec_db.mean(),
                    'std': mel_spec_db.std()
                })
                
                print(f"   {filename}:")
                print(f"      Shape: {mel_spec_db.shape}")
                print(f"      Range: [{mel_spec_db.min():.2f}, {mel_spec_db.max():.2f}]")
                print(f"      Mean: {mel_spec_db.mean():.2f}, Std: {mel_spec_db.std():.2f}")
                
            except Exception as e:
                print(f"   ❌ {filename}: Error - {e}")
        
        # Check if spectrograms are varied
        if len(spec_values) > 1:
            means = [s['mean'] for s in spec_values]
            if len(set(means)) == 1:
                print(f"\n   ⚠️  WARNING: All spectrograms have identical means!")
    
    # =========================================================
    # SUMMARY
    # =========================================================
    print_header("📋 DIAGNOSIS SUMMARY")
    
    issues = []
    
    if len(train_val_overlap) > 0:
        issues.append("CRITICAL: File-level data leakage between train and val")
    
    if len(recording_overlap) > 0:
        issues.append("CRITICAL: Recording-level leakage - same recordings in train/val")
    
    if not_found:
        issues.append(f"WARNING: {len(not_found)} audio files not found")
    
    if unmapped_train or unmapped_val:
        issues.append("WARNING: Some species not in label mapping")
    
    if not issues:
        issues.append("No obvious issues found - need deeper investigation")
    
    print("\n   Issues Found:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    
    # =========================================================
    # RECOMMENDED FIXES
    # =========================================================
    print_header("🔧 RECOMMENDED FIXES")
    
    if len(recording_overlap) > 0:
        print("""
   🚨 MAIN ISSUE: RECORDING-LEVEL DATA LEAKAGE
   
   Your train/val split was done at the CHUNK level, not RECORDING level.
   This means:
   - Recording XC123456 has chunks: chunk_0, chunk_1, chunk_2, chunk_3
   - chunk_0 and chunk_1 might be in TRAIN
   - chunk_2 and chunk_3 might be in VAL
   
   The model memorizes the recording's acoustic signature (background noise,
   microphone characteristics, etc.) and achieves 100% accuracy.
   
   FIX: Split by RECORDING ID, not by individual chunks.
   
   Run this to create a proper split:
   python 09_Utils/Scripts/fix_data_split.py
        """)
    
    print("\n✅ Debug complete. Review the issues above.")


if __name__ == "__main__":
    main()