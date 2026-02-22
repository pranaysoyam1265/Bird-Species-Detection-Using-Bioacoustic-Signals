"""
MERGE METADATA AND UPDATE TRAINING DATA
Save as: 09_Utils/Scripts/merge_and_update.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"

# Input files
OLD_METADATA = os.path.join(PROJECT_ROOT, "01_Raw_Data", "Metadata", "bird_metadata_complete.csv")
NEW_METADATA = os.path.join(PROJECT_ROOT, "01_Raw_Data", "Metadata", "scraped_metadata_v3.csv")
CHUNKS_FOLDER = os.path.join(PROJECT_ROOT, "02_Preprocessed", "Audio_Chunks")

# Output files
MERGED_METADATA = os.path.join(PROJECT_ROOT, "01_Raw_Data", "Metadata", "metadata_merged.csv")
LABELS_FOLDER = os.path.join(PROJECT_ROOT, "04_Labels", "Train_Val_Test_Split")
LABEL_MAPPING = os.path.join(PROJECT_ROOT, "04_Labels", "Processed_Labels", "label_mapping_v2.json")

# Minimum samples per species (filter rare species)
MIN_SAMPLES_PER_SPECIES = 50

# ============================================================
# STEP 1: MERGE METADATA
# ============================================================

print("=" * 60)
print("STEP 1: MERGING METADATA")
print("=" * 60)

# Load new metadata
new_df = pd.read_csv(NEW_METADATA)
print(f"📁 New metadata: {len(new_df)} recordings")

# Load old metadata if exists
if os.path.exists(OLD_METADATA):
    old_df = pd.read_csv(OLD_METADATA)
    print(f"📁 Old metadata: {len(old_df)} recordings")
    
    # Standardize column names
    if 'xc_id' not in old_df.columns and 'id' in old_df.columns:
        old_df['xc_id'] = old_df['id'].astype(str)
    
    # Merge (new takes priority)
    new_df['xc_id'] = new_df['xc_id'].astype(str)
    
    # Keep only new data (it's more complete)
    merged_df = new_df.copy()
else:
    merged_df = new_df.copy()

print(f"✅ Merged metadata: {len(merged_df)} recordings")

# Clean species names
merged_df['species_scientific'] = merged_df['species_scientific'].str.strip()
merged_df['species_english'] = merged_df['species_english'].str.strip()

# Remove rows without species
merged_df = merged_df.dropna(subset=['species_scientific'])
merged_df = merged_df[merged_df['species_scientific'] != '']

print(f"✅ After cleaning: {len(merged_df)} recordings")
print(f"🐦 Total species: {merged_df['species_scientific'].nunique()}")

# Save merged metadata
merged_df.to_csv(MERGED_METADATA, index=False)
print(f"💾 Saved: {MERGED_METADATA}")

# ============================================================
# STEP 2: MATCH CHUNKS WITH METADATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: MATCHING CHUNKS WITH METADATA")
print("=" * 60)

# Get all chunks
chunk_files = [f for f in os.listdir(CHUNKS_FOLDER) if f.endswith('.wav')]
print(f"📁 Total chunks: {len(chunk_files)}")

# Extract XC IDs from chunk filenames
# Filename format: XC475302_chunk_0.wav or similar
def extract_xc_id_from_chunk(filename):
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Extract XC ID (everything before _chunk or first underscore)
    if 'XC' in name.upper():
        parts = name.upper().split('XC')
        if len(parts) > 1:
            xc_part = parts[1].split('_')[0]
            return xc_part
    # Try first part before underscore
    parts = name.split('_')
    return parts[0].replace('XC', '').replace('xc', '')

# Create chunk to XC ID mapping
chunk_data = []
for chunk_file in chunk_files:
    xc_id = extract_xc_id_from_chunk(chunk_file)
    chunk_data.append({
        'chunk_file': chunk_file,
        'xc_id': xc_id
    })

chunks_df = pd.DataFrame(chunk_data)
print(f"📋 Parsed chunk files: {len(chunks_df)}")

# Merge with metadata
merged_df['xc_id'] = merged_df['xc_id'].astype(str)
chunks_df['xc_id'] = chunks_df['xc_id'].astype(str)

chunks_with_labels = chunks_df.merge(
    merged_df[['xc_id', 'species_scientific', 'species_english']], 
    on='xc_id', 
    how='left'
)

# Check matches
matched = chunks_with_labels['species_scientific'].notna().sum()
unmatched = chunks_with_labels['species_scientific'].isna().sum()

print(f"✅ Matched chunks: {matched}")
print(f"❌ Unmatched chunks: {unmatched}")

# Keep only matched chunks
labeled_chunks = chunks_with_labels.dropna(subset=['species_scientific'])
print(f"📋 Labeled chunks: {len(labeled_chunks)}")

# ============================================================
# STEP 3: FILTER SPECIES WITH ENOUGH SAMPLES
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: FILTERING SPECIES")
print("=" * 60)

# Count samples per species
species_counts = labeled_chunks['species_scientific'].value_counts()
print(f"🐦 Total species before filter: {len(species_counts)}")

# Filter species with minimum samples
valid_species = species_counts[species_counts >= MIN_SAMPLES_PER_SPECIES].index.tolist()
print(f"🐦 Species with >= {MIN_SAMPLES_PER_SPECIES} samples: {len(valid_species)}")

# Filter dataset
filtered_chunks = labeled_chunks[labeled_chunks['species_scientific'].isin(valid_species)]
print(f"📋 Chunks after filter: {len(filtered_chunks)}")

# Show species distribution
print(f"\n📊 Top 10 species:")
for species, count in species_counts.head(10).items():
    english = merged_df[merged_df['species_scientific'] == species]['species_english'].iloc[0]
    print(f"   {english}: {count}")

print(f"\n📊 Bottom 5 species (kept):")
bottom_species = species_counts[species_counts >= MIN_SAMPLES_PER_SPECIES].tail(5)
for species, count in bottom_species.items():
    english = merged_df[merged_df['species_scientific'] == species]['species_english'].iloc[0]
    print(f"   {english}: {count}")

# ============================================================
# STEP 4: CREATE LABEL MAPPING
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: CREATING LABEL MAPPING")
print("=" * 60)

# Create label to index mapping
unique_species = sorted(filtered_chunks['species_scientific'].unique())
label_to_idx = {species: idx for idx, species in enumerate(unique_species)}
idx_to_label = {idx: species for species, idx in label_to_idx.items()}

# Create English name mapping
species_to_english = {}
for species in unique_species:
    english = merged_df[merged_df['species_scientific'] == species]['species_english'].iloc[0]
    species_to_english[species] = english

print(f"🏷️ Number of classes: {len(label_to_idx)}")

# Save label mapping
import json
os.makedirs(os.path.dirname(LABEL_MAPPING), exist_ok=True)

mapping_data = {
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'species_to_english': species_to_english,
    'num_classes': len(label_to_idx)
}

with open(LABEL_MAPPING, 'w') as f:
    json.dump(mapping_data, f, indent=2)

print(f"💾 Saved: {LABEL_MAPPING}")

# ============================================================
# STEP 5: SPLIT INTO TRAIN/VAL/TEST
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: CREATING TRAIN/VAL/TEST SPLIT")
print("=" * 60)

# Add label index
filtered_chunks['label'] = filtered_chunks['species_scientific'].map(label_to_idx)

# Stratified split: 70% train, 15% val, 15% test
train_df, temp_df = train_test_split(
    filtered_chunks, 
    test_size=0.30, 
    stratify=filtered_chunks['label'],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.50, 
    stratify=temp_df['label'],
    random_state=42
)

print(f"📊 Train set: {len(train_df)} chunks ({len(train_df)/len(filtered_chunks)*100:.1f}%)")
print(f"📊 Val set: {len(val_df)} chunks ({len(val_df)/len(filtered_chunks)*100:.1f}%)")
print(f"📊 Test set: {len(test_df)} chunks ({len(test_df)/len(filtered_chunks)*100:.1f}%)")

# Save splits
os.makedirs(LABELS_FOLDER, exist_ok=True)

train_df.to_csv(os.path.join(LABELS_FOLDER, 'train_v2.csv'), index=False)
val_df.to_csv(os.path.join(LABELS_FOLDER, 'val_v2.csv'), index=False)
test_df.to_csv(os.path.join(LABELS_FOLDER, 'test_v2.csv'), index=False)

print(f"\n💾 Saved to: {LABELS_FOLDER}")
print(f"   - train_v2.csv")
print(f"   - val_v2.csv")
print(f"   - test_v2.csv")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("✅ MERGE AND UPDATE COMPLETE!")
print("=" * 60)

print(f"""
📊 SUMMARY
──────────────────────────────────
Total recordings with metadata: {len(merged_df)}
Total chunks: {len(chunk_files)}
Matched chunks: {len(labeled_chunks)}
Species (filtered): {len(valid_species)}
Minimum samples/species: {MIN_SAMPLES_PER_SPECIES}

SPLIT
──────────────────────────────────
Train: {len(train_df)} chunks
Val: {len(val_df)} chunks  
Test: {len(test_df)} chunks
Total: {len(filtered_chunks)} chunks

FILES CREATED
──────────────────────────────────
{MERGED_METADATA}
{LABEL_MAPPING}
{os.path.join(LABELS_FOLDER, 'train_v2.csv')}
{os.path.join(LABELS_FOLDER, 'val_v2.csv')}
{os.path.join(LABELS_FOLDER, 'test_v2.csv')}

🚀 Ready for retraining!
""")