"""
label_preparation.py
Phase 3: Prepare labels for ML training

This script:
1. Loads usable metadata
2. Cleans species names (merges subspecies)
3. Removes rare species (< min_samples)
4. Maps chunks to species labels
5. Creates train/val/test splits
6. Saves everything for training

Author: Bird Detection Project
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import json
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")

# Input paths
USABLE_METADATA = BASE_DIR / "01_Raw_Data" / "Metadata" / "metadata_usable_only.csv"
CHUNKS_DIR = BASE_DIR / "02_Preprocessed" / "Audio_Chunks"
CHUNK_MAPPING = BASE_DIR / "02_Preprocessed" / "Quality_Reports" / "chunk_mapping.csv"

# Output paths
LABELS_DIR = BASE_DIR / "04_Labels"
RAW_LABELS_DIR = LABELS_DIR / "Raw_Labels"
PROCESSED_LABELS_DIR = LABELS_DIR / "Processed_Labels"
SPLIT_DIR = LABELS_DIR / "Train_Val_Test_Split"

# Settings
MIN_SAMPLES_PER_SPECIES = 5  # Remove species with fewer samples
MERGE_SUBSPECIES = True       # Merge subspecies into parent species
TEST_SIZE = 0.15              # 15% for test
VAL_SIZE = 0.15               # 15% for validation (of remaining after test)
RANDOM_SEED = 42

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_directories():
    """Create output directories"""
    for dir_path in [RAW_LABELS_DIR, PROCESSED_LABELS_DIR, SPLIT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✅ Created output directories")

def clean_species_name(name):
    """
    Clean species name by:
    1. Removing subspecies (keep only genus + species)
    2. Removing question marks and extra characters
    3. Standardizing format
    """
    if pd.isna(name) or name == '':
        return None
    
    # Remove question marks and parentheses
    name = re.sub(r'\?', '', name)
    name = re.sub(r'\([^)]*\)', '', name)
    name = name.strip()
    
    # Split into parts
    parts = name.split()
    
    if len(parts) >= 2:
        # Keep only genus + species (first two words)
        genus = parts[0].capitalize()
        species = parts[1].lower()
        return f"{genus} {species}"
    elif len(parts) == 1:
        return parts[0].capitalize()
    else:
        return None

def get_numeric_id(recording_id):
    """Extract numeric ID from XC format"""
    id_str = str(recording_id)
    if id_str.upper().startswith('XC'):
        return id_str[2:]
    return id_str

def find_chunks_for_recording(recording_id, chunks_dir):
    """Find all chunk files for a recording"""
    numeric_id = get_numeric_id(recording_id)
    pattern = f"XC{numeric_id}_chunk_*.wav"
    
    chunks = list(chunks_dir.glob(pattern))
    return [chunk.name for chunk in chunks]

# ============================================================
# MAIN PROCESSING
# ============================================================

def main():
    print("=" * 70)
    print("📋 PHASE 3: LABEL PREPARATION")
    print("=" * 70)
    
    # Create directories
    create_directories()
    
    # --------------------------------------------------------
    # STEP 1: Load usable metadata
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1: Loading usable metadata")
    print("-" * 70)
    
    if not USABLE_METADATA.exists():
        print(f"❌ ERROR: File not found: {USABLE_METADATA}")
        print("Please run metadata_diagnostic_v2.py first!")
        return
    
    df = pd.read_csv(USABLE_METADATA)
    print(f"✅ Loaded {len(df)} usable recordings")
    print(f"   Original unique species: {df['scientific_name'].nunique()}")
    
    # --------------------------------------------------------
    # STEP 2: Clean species names
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2: Cleaning species names")
    print("-" * 70)
    
    if MERGE_SUBSPECIES:
        print("   Merging subspecies into parent species...")
        df['species_clean'] = df['scientific_name'].apply(clean_species_name)
    else:
        df['species_clean'] = df['scientific_name']
    
    # Remove rows with invalid species
    df = df[df['species_clean'].notna()].copy()
    
    print(f"✅ After cleaning: {len(df)} recordings")
    print(f"   Unique species after merge: {df['species_clean'].nunique()}")
    
    # Show merge examples
    print("\n   Examples of merged subspecies:")
    original_species = df['scientific_name'].unique()
    cleaned_species = df['species_clean'].unique()
    
    merge_examples = {}
    for orig in original_species:
        clean = clean_species_name(orig)
        if clean != orig and clean is not None:
            if clean not in merge_examples:
                merge_examples[clean] = []
            merge_examples[clean].append(orig)
    
    for clean, originals in list(merge_examples.items())[:5]:
        print(f"     {originals} → {clean}")
    
    # --------------------------------------------------------
    # STEP 3: Remove rare species
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print(f"STEP 3: Removing species with < {MIN_SAMPLES_PER_SPECIES} samples")
    print("-" * 70)
    
    species_counts = df['species_clean'].value_counts()
    
    # Find rare species
    rare_species = species_counts[species_counts < MIN_SAMPLES_PER_SPECIES].index.tolist()
    common_species = species_counts[species_counts >= MIN_SAMPLES_PER_SPECIES].index.tolist()
    
    print(f"   Species with >= {MIN_SAMPLES_PER_SPECIES} samples: {len(common_species)}")
    print(f"   Species with < {MIN_SAMPLES_PER_SPECIES} samples (removing): {len(rare_species)}")
    
    if len(rare_species) > 0:
        print(f"\n   Removing these rare species:")
        for sp in rare_species[:10]:
            print(f"     - {sp}: {species_counts[sp]} samples")
        if len(rare_species) > 10:
            print(f"     ... and {len(rare_species) - 10} more")
    
    # Filter to common species only
    df_filtered = df[df['species_clean'].isin(common_species)].copy()
    
    print(f"\n✅ After filtering: {len(df_filtered)} recordings")
    print(f"   Final species count: {df_filtered['species_clean'].nunique()}")
    
    # --------------------------------------------------------
    # STEP 4: Map chunks to species
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 4: Mapping audio chunks to species labels")
    print("-" * 70)
    
    # Check if chunk mapping exists
    if CHUNK_MAPPING.exists():
        print("   Loading existing chunk mapping...")
        chunk_df = pd.read_csv(CHUNK_MAPPING)
        print(f"   Found {len(chunk_df)} chunk records in mapping file")
    else:
        print("   Chunk mapping file not found. Scanning chunks directory...")
        chunk_df = None
    
    # Build chunk-to-species mapping
    chunk_labels = []
    missing_chunks = []
    
    print("   Scanning for chunks...")
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="   Mapping"):
        recording_id = row['recording_id']
        species = row['species_clean']
        english_name = row.get('english_name', '')
        
        # Find chunks for this recording
        chunks = find_chunks_for_recording(recording_id, CHUNKS_DIR)
        
        if len(chunks) == 0:
            missing_chunks.append(recording_id)
            continue
        
        for chunk_file in chunks:
            chunk_labels.append({
                'chunk_file': chunk_file,
                'recording_id': recording_id,
                'species': species,
                'english_name': english_name
            })
    
    # Create chunk labels dataframe
    df_chunks = pd.DataFrame(chunk_labels)
    
    print(f"\n✅ Chunk mapping complete:")
    print(f"   Total chunks with labels: {len(df_chunks)}")
    print(f"   Recordings with chunks: {df_chunks['recording_id'].nunique()}")
    print(f"   Recordings without chunks: {len(missing_chunks)}")
    
    if len(missing_chunks) > 0:
        print(f"\n   ⚠️ Recordings with no chunks found:")
        for rid in missing_chunks[:10]:
            print(f"      - {rid}")
        if len(missing_chunks) > 10:
            print(f"      ... and {len(missing_chunks) - 10} more")
    
    # --------------------------------------------------------
    # STEP 5: Create label encoding
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 5: Creating label encoding")
    print("-" * 70)
    
    # Get unique species list (sorted for consistency)
    unique_species = sorted(df_chunks['species'].unique())
    num_classes = len(unique_species)
    
    print(f"   Number of classes: {num_classes}")
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_species)
    
    # Encode labels
    df_chunks['label_id'] = label_encoder.transform(df_chunks['species'])
    
    # Create species-to-id mapping
    species_to_id = {species: idx for idx, species in enumerate(unique_species)}
    id_to_species = {idx: species for idx, species in enumerate(unique_species)}
    
    # Save mappings
    mapping_data = {
        'species_to_id': species_to_id,
        'id_to_species': id_to_species,
        'num_classes': num_classes,
        'species_list': unique_species
    }
    
    mapping_path = PROCESSED_LABELS_DIR / "label_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    print(f"✅ Saved label mapping: {mapping_path}")
    
    # --------------------------------------------------------
    # STEP 6: Train/Val/Test Split
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 6: Creating train/val/test splits")
    print("-" * 70)
    
    # Split by RECORDING (not chunk) to avoid data leakage
    unique_recordings = df_chunks['recording_id'].unique()
    recording_species = df_chunks.groupby('recording_id')['species'].first().to_dict()
    
    print(f"   Total unique recordings: {len(unique_recordings)}")
    print(f"   Split ratio: Train {100*(1-TEST_SIZE-VAL_SIZE):.0f}% / Val {100*VAL_SIZE:.0f}% / Test {100*TEST_SIZE:.0f}%")
    
    # First split: separate test set
    train_val_recordings, test_recordings = train_test_split(
        unique_recordings,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=[recording_species[r] for r in unique_recordings]
    )
    
    # Second split: separate validation set from training
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)  # Adjust ratio
    train_recordings, val_recordings = train_test_split(
        train_val_recordings,
        test_size=val_ratio,
        random_state=RANDOM_SEED,
        stratify=[recording_species[r] for r in train_val_recordings]
    )
    
    print(f"\n   Recording split:")
    print(f"     Train: {len(train_recordings)} recordings")
    print(f"     Val:   {len(val_recordings)} recordings")
    print(f"     Test:  {len(test_recordings)} recordings")
    
    # Create chunk splits based on recording splits
    train_chunks = df_chunks[df_chunks['recording_id'].isin(train_recordings)].copy()
    val_chunks = df_chunks[df_chunks['recording_id'].isin(val_recordings)].copy()
    test_chunks = df_chunks[df_chunks['recording_id'].isin(test_recordings)].copy()
    
    print(f"\n   Chunk split:")
    print(f"     Train: {len(train_chunks)} chunks")
    print(f"     Val:   {len(val_chunks)} chunks")
    print(f"     Test:  {len(test_chunks)} chunks")
    
    # Verify class distribution in splits
    print("\n   Class distribution check:")
    train_dist = train_chunks['species'].value_counts()
    val_dist = val_chunks['species'].value_counts()
    test_dist = test_chunks['species'].value_counts()
    
    # Check for missing classes in val/test
    missing_in_val = set(unique_species) - set(val_dist.index)
    missing_in_test = set(unique_species) - set(test_dist.index)
    
    if missing_in_val:
        print(f"   ⚠️ Classes missing in validation: {len(missing_in_val)}")
    if missing_in_test:
        print(f"   ⚠️ Classes missing in test: {len(missing_in_test)}")
    if not missing_in_val and not missing_in_test:
        print("   ✅ All classes present in all splits!")
    
    # --------------------------------------------------------
    # STEP 7: Save everything
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 7: Saving label files")
    print("-" * 70)
    
    # Save full chunk labels
    chunk_labels_path = RAW_LABELS_DIR / "chunk_labels_all.csv"
    df_chunks.to_csv(chunk_labels_path, index=False)
    print(f"✅ All chunk labels: {chunk_labels_path}")
    
    # Save splits
    train_path = SPLIT_DIR / "train.csv"
    val_path = SPLIT_DIR / "val.csv"
    test_path = SPLIT_DIR / "test.csv"
    
    train_chunks.to_csv(train_path, index=False)
    val_chunks.to_csv(val_path, index=False)
    test_chunks.to_csv(test_path, index=False)
    
    print(f"✅ Train split: {train_path}")
    print(f"✅ Val split: {val_path}")
    print(f"✅ Test split: {test_path}")
    
    # Save class weights (for handling imbalance)
    class_counts = train_chunks['species'].value_counts()
    total_samples = len(train_chunks)
    class_weights = {
        species: total_samples / (num_classes * count)
        for species, count in class_counts.items()
    }
    
    weights_path = PROCESSED_LABELS_DIR / "class_weights.json"
    with open(weights_path, 'w') as f:
        json.dump(class_weights, f, indent=2)
    print(f"✅ Class weights: {weights_path}")
    
    # Save species summary
    species_summary = []
    for species in unique_species:
        species_summary.append({
            'species': species,
            'label_id': species_to_id[species],
            'train_chunks': len(train_chunks[train_chunks['species'] == species]),
            'val_chunks': len(val_chunks[val_chunks['species'] == species]),
            'test_chunks': len(test_chunks[test_chunks['species'] == species]),
            'total_chunks': len(df_chunks[df_chunks['species'] == species]),
            'class_weight': class_weights.get(species, 1.0)
        })
    
    summary_df = pd.DataFrame(species_summary)
    summary_df = summary_df.sort_values('total_chunks', ascending=False)
    summary_path = PROCESSED_LABELS_DIR / "species_training_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Species summary: {summary_path}")
    
    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 LABEL PREPARATION COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"""
    📁 Dataset Statistics:
    ├── Total recordings used:     {df_chunks['recording_id'].nunique()}
    ├── Total chunks labeled:      {len(df_chunks)}
    ├── Number of species/classes: {num_classes}
    │
    📂 Train/Val/Test Split:
    ├── Train: {len(train_chunks):,} chunks ({len(train_recordings)} recordings)
    ├── Val:   {len(val_chunks):,} chunks ({len(val_recordings)} recordings)
    └── Test:  {len(test_chunks):,} chunks ({len(test_recordings)} recordings)
    
    📁 Files Created:
    ├── {RAW_LABELS_DIR / 'chunk_labels_all.csv'}
    ├── {SPLIT_DIR / 'train.csv'}
    ├── {SPLIT_DIR / 'val.csv'}
    ├── {SPLIT_DIR / 'test.csv'}
    ├── {PROCESSED_LABELS_DIR / 'label_mapping.json'}
    ├── {PROCESSED_LABELS_DIR / 'class_weights.json'}
    └── {PROCESSED_LABELS_DIR / 'species_training_summary.csv'}
    
    🎯 Next Step: Generate spectrograms for training!
    """)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()