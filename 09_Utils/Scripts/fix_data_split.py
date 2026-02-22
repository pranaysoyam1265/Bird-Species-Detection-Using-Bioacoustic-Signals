"""
Script: fix_data_split.py
Purpose: Create proper train/val/test split by RECORDING (not chunk)
         AND fix the label mapping issue
Location: Save to 09_Utils/Scripts/fix_data_split.py
"""

import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"
TRAIN_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split", "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split", "val.csv")
TEST_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split", "test.csv")
LABEL_MAPPING_OLD = os.path.join(BASE_DIR, "04_Labels", "Processed_Labels", "label_mapping.json")

# Output paths
OUTPUT_DIR = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed")
LABEL_MAPPING_NEW = os.path.join(BASE_DIR, "04_Labels", "Processed_Labels", "label_mapping_fixed.json")

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    print_header("🔧 FIXING DATA SPLIT & LABEL MAPPING")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================
    # STEP 1: Load all data
    # =========================================================
    print("\n📂 Loading original CSVs...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Combine all data
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Remove duplicates (the leaked files)
    original_count = len(all_data)
    all_data = all_data.drop_duplicates(subset=['chunk_file'], keep='first')
    deduped_count = len(all_data)
    
    print(f"   Original total:    {original_count}")
    print(f"   After dedup:       {deduped_count}")
    print(f"   Duplicates removed: {original_count - deduped_count}")
    
    # =========================================================
    # STEP 2: Extract recording IDs properly
    # =========================================================
    print("\n📋 Extracting recording IDs...")
    
    def extract_recording_id(chunk_file):
        """Extract recording ID from chunk filename like XC101288_chunk_001.wav"""
        # Remove extension and chunk suffix
        base = chunk_file.replace('.wav', '').replace('.mp3', '')
        # Split by _chunk_ and take first part
        if '_chunk_' in base:
            return base.split('_chunk_')[0]
        return base
    
    # The CSV already has recording_id column, but let's verify
    all_data['recording_id_extracted'] = all_data['chunk_file'].apply(extract_recording_id)
    
    # Check if it matches the existing recording_id column
    if 'recording_id' in all_data.columns:
        mismatches = (all_data['recording_id'].astype(str) != all_data['recording_id_extracted'].str.replace('XC', ''))
        if mismatches.any():
            print(f"   ⚠️  Found {mismatches.sum()} recording ID mismatches")
            # Use extracted version
            all_data['recording_id'] = all_data['recording_id_extracted']
    else:
        all_data['recording_id'] = all_data['recording_id_extracted']
    
    all_data = all_data.drop('recording_id_extracted', axis=1)
    
    # =========================================================
    # STEP 3: Get unique recordings
    # =========================================================
    print("\n📊 Analyzing recordings...")
    
    recordings = all_data.groupby('recording_id').agg({
        'species': 'first',
        'english_name': 'first',
        'chunk_file': 'count',
        'label_id': 'first'
    }).reset_index()
    recordings.columns = ['recording_id', 'species', 'english_name', 'num_chunks', 'label_id']
    
    print(f"   Unique recordings: {len(recordings)}")
    print(f"   Unique species:    {recordings['species'].nunique()}")
    print(f"   Total chunks:      {all_data['chunk_file'].nunique()}")
    
    # Species distribution
    species_counts = recordings.groupby('species').size().sort_values(ascending=False)
    print(f"\n   Recordings per species (top 10):")
    for species, count in species_counts.head(10).items():
        print(f"      {species}: {count} recordings")
    
    print(f"\n   Species with fewest recordings:")
    for species, count in species_counts.tail(5).items():
        print(f"      {species}: {count} recordings")
    
    # =========================================================
    # STEP 4: Stratified split by RECORDING
    # =========================================================
    print_header("STEP 4: STRATIFIED SPLIT BY RECORDING")
    
    # Handle species with very few recordings
    min_samples = 3  # Need at least 3 recordings per species for stratified split
    
    species_with_few = species_counts[species_counts < min_samples]
    if len(species_with_few) > 0:
        print(f"\n   ⚠️  Species with <{min_samples} recordings (will be split manually):")
        for species, count in species_with_few.items():
            print(f"      {species}: {count}")
    
    # Separate rare species
    rare_recordings = recordings[recordings['species'].isin(species_with_few.index)]
    regular_recordings = recordings[~recordings['species'].isin(species_with_few.index)]
    
    print(f"\n   Regular recordings (stratified): {len(regular_recordings)}")
    print(f"   Rare species recordings (manual): {len(rare_recordings)}")
    
    # Stratified split for regular recordings
    if len(regular_recordings) > 0:
        train_rec, temp_rec = train_test_split(
            regular_recordings,
            test_size=(VAL_RATIO + TEST_RATIO),
            stratify=regular_recordings['species'],
            random_state=RANDOM_SEED
        )
        
        relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
        val_rec, test_rec = train_test_split(
            temp_rec,
            test_size=relative_test_size,
            stratify=temp_rec['species'],
            random_state=RANDOM_SEED
        )
    else:
        train_rec = pd.DataFrame()
        val_rec = pd.DataFrame()
        test_rec = pd.DataFrame()
    
    # Handle rare species: put most in train, 1 in val, 1 in test if possible
    for species in species_with_few.index:
        species_recs = rare_recordings[rare_recordings['species'] == species]
        n = len(species_recs)
        
        if n == 1:
            # Only 1 recording: put in train
            train_rec = pd.concat([train_rec, species_recs])
        elif n == 2:
            # 2 recordings: 1 train, 1 val
            train_rec = pd.concat([train_rec, species_recs.iloc[:1]])
            val_rec = pd.concat([val_rec, species_recs.iloc[1:2]])
        else:
            # 3+ recordings: split manually
            train_rec = pd.concat([train_rec, species_recs.iloc[:n-2]])
            val_rec = pd.concat([val_rec, species_recs.iloc[n-2:n-1]])
            test_rec = pd.concat([test_rec, species_recs.iloc[n-1:]])
    
    print(f"\n📊 Recording Split:")
    print(f"   Train: {len(train_rec)} recordings ({100*len(train_rec)/len(recordings):.1f}%)")
    print(f"   Val:   {len(val_rec)} recordings ({100*len(val_rec)/len(recordings):.1f}%)")
    print(f"   Test:  {len(test_rec)} recordings ({100*len(test_rec)/len(recordings):.1f}%)")
    
    # =========================================================
    # STEP 5: Verify NO overlap
    # =========================================================
    print_header("STEP 5: VERIFY NO OVERLAP")
    
    train_ids = set(train_rec['recording_id'])
    val_ids = set(val_rec['recording_id'])
    test_ids = set(test_rec['recording_id'])
    
    overlap_tv = train_ids.intersection(val_ids)
    overlap_tt = train_ids.intersection(test_ids)
    overlap_vt = val_ids.intersection(test_ids)
    
    print(f"   Train ∩ Val:  {len(overlap_tv)} {'✅' if len(overlap_tv)==0 else '❌'}")
    print(f"   Train ∩ Test: {len(overlap_tt)} {'✅' if len(overlap_tt)==0 else '❌'}")
    print(f"   Val ∩ Test:   {len(overlap_vt)} {'✅' if len(overlap_vt)==0 else '❌'}")
    
    if overlap_tv or overlap_tt or overlap_vt:
        print("\n   ❌ OVERLAP STILL EXISTS!")
        return
    
    print("\n   ✅ NO OVERLAP - Perfect split!")
    
    # =========================================================
    # STEP 6: Create chunk-level DataFrames
    # =========================================================
    print_header("STEP 6: CREATE CHUNK DATAFRAMES")
    
    train_chunks = all_data[all_data['recording_id'].isin(train_ids)].copy()
    val_chunks = all_data[all_data['recording_id'].isin(val_ids)].copy()
    test_chunks = all_data[all_data['recording_id'].isin(test_ids)].copy()
    
    print(f"   Train chunks: {len(train_chunks)} ({100*len(train_chunks)/len(all_data):.1f}%)")
    print(f"   Val chunks:   {len(val_chunks)} ({100*len(val_chunks)/len(all_data):.1f}%)")
    print(f"   Test chunks:  {len(test_chunks)} ({100*len(test_chunks)/len(all_data):.1f}%)")
    
    # Verify species coverage
    train_species = set(train_chunks['species'].unique())
    val_species = set(val_chunks['species'].unique())
    test_species = set(test_chunks['species'].unique())
    
    print(f"\n   Species coverage:")
    print(f"   Train: {len(train_species)} species")
    print(f"   Val:   {len(val_species)} species")
    print(f"   Test:  {len(test_species)} species")
    
    missing_in_val = train_species - val_species
    missing_in_test = train_species - test_species
    
    if missing_in_val:
        print(f"\n   ⚠️  Species in train but not val: {len(missing_in_val)}")
        for s in list(missing_in_val)[:5]:
            print(f"      - {s}")
    
    if missing_in_test:
        print(f"\n   ⚠️  Species in train but not test: {len(missing_in_test)}")
        for s in list(missing_in_test)[:5]:
            print(f"      - {s}")
    
    # =========================================================
    # STEP 7: Save fixed CSVs
    # =========================================================
    print_header("STEP 7: SAVE FIXED CSVs")
    
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    val_path = os.path.join(OUTPUT_DIR, "val.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")
    
    train_chunks.to_csv(train_path, index=False)
    val_chunks.to_csv(val_path, index=False)
    test_chunks.to_csv(test_path, index=False)
    
    print(f"   ✅ {train_path}")
    print(f"   ✅ {val_path}")
    print(f"   ✅ {test_path}")
    
    # =========================================================
    # STEP 8: Fix label mapping
    # =========================================================
    print_header("STEP 8: FIX LABEL MAPPING")
    
    # Load old mapping
    with open(LABEL_MAPPING_OLD, 'r') as f:
        old_mapping = json.load(f)
    
    print(f"   Old mapping structure: {list(old_mapping.keys())}")
    
    # The old mapping is nested - extract species_to_id
    if 'species_to_id' in old_mapping:
        species_to_id = old_mapping['species_to_id']
    else:
        # It's a direct mapping
        species_to_id = old_mapping
    
    # Create FLAT mapping that training script expects
    # The training script does: label_map.get(species, 0)
    # So we need species -> id directly
    
    flat_mapping = {}
    for species, idx in species_to_id.items():
        flat_mapping[species] = idx
    
    print(f"   Flat mapping entries: {len(flat_mapping)}")
    print(f"   Sample: {list(flat_mapping.items())[:3]}")
    
    # Save flat mapping
    with open(LABEL_MAPPING_NEW, 'w') as f:
        json.dump(flat_mapping, f, indent=2)
    
    print(f"\n   ✅ Fixed label mapping saved to: {LABEL_MAPPING_NEW}")
    
    # Also save full mapping for reference
    full_mapping = {
        'species_to_id': species_to_id,
        'id_to_species': {str(v): k for k, v in species_to_id.items()},
        'num_classes': len(species_to_id),
        'species_list': list(species_to_id.keys())
    }
    
    full_mapping_path = os.path.join(OUTPUT_DIR, "label_mapping_full.json")
    with open(full_mapping_path, 'w') as f:
        json.dump(full_mapping, f, indent=2)
    
    # =========================================================
    # STEP 9: Save split summary
    # =========================================================
    summary = {
        'created': pd.Timestamp.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'total_recordings': len(recordings),
        'total_chunks': len(all_data),
        'duplicates_removed': original_count - deduped_count,
        'train': {
            'recordings': len(train_rec),
            'chunks': len(train_chunks),
            'species': len(train_species)
        },
        'val': {
            'recordings': len(val_rec),
            'chunks': len(val_chunks),
            'species': len(val_species)
        },
        'test': {
            'recordings': len(test_rec),
            'chunks': len(test_chunks),
            'species': len(test_species)
        },
        'overlap_check': {
            'train_val': 0,
            'train_test': 0,
            'val_test': 0
        }
    }
    
    summary_path = os.path.join(OUTPUT_DIR, "split_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print_header("✅ FIX COMPLETE")
    
    print(f"""
   📁 Fixed files saved to: {OUTPUT_DIR}
   
   📊 NEW SPLIT (No Leakage):
   ┌─────────────────────────────────────────────────────┐
   │  Set      │ Recordings │ Chunks   │ Species        │
   ├───────────┼────────────┼──────────┼────────────────┤
   │  Train    │ {len(train_rec):>10} │ {len(train_chunks):>8} │ {len(train_species):>14} │
   │  Val      │ {len(val_rec):>10} │ {len(val_chunks):>8} │ {len(val_species):>14} │
   │  Test     │ {len(test_rec):>10} │ {len(test_chunks):>8} │ {len(test_species):>14} │
   └───────────┴────────────┴──────────┴────────────────┘
   
   ✅ Overlap: ZERO (verified)
   
   📝 NEXT STEPS:
   
   1. Update training script to use fixed paths:
   
      TRAIN_CSV = r"{train_path}"
      VAL_CSV = r"{val_path}"
      LABEL_MAPPING = r"{LABEL_MAPPING_NEW}"
   
   2. Run training:
      python 09_Utils/Scripts/train_with_augmentation_v2.py
   
   3. Expected results:
      • Val accuracy: 60-75% (realistic!)
      • Train-Val gap: 10-20% (normal)
      • Test accuracy: similar to val
    """)


if __name__ == "__main__":
    main()