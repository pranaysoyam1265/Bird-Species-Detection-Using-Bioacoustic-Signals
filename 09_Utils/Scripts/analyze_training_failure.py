# Save as: analyze_training_failure.py

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import glob
import os

# Resolve project root and change to it
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
os.chdir(project_root)

print("="*60)
print("ANALYZING TRAINING FAILURE")
print("="*60)

# ============================================================
# 1. Check Training Log
# ============================================================
print("\nCHECKING TRAINING LOG")
print("-" * 40)

log_files = glob.glob("05_Model/Training_Logs/training_log_v3_*.csv")
if log_files:
    latest_log = max(log_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"Found log: {Path(latest_log).name}")
    
    df_log = pd.read_csv(latest_log)
    print(f"\nTotal epochs logged: {len(df_log)}")
    
    print("\nTraining progression:")
    print(df_log[['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_top5']].to_string(index=False))
    
    # Check for anomalies
    print("\nAnomaly check:")
    
    # Check if accuracy ever increased
    if df_log['val_acc'].max() > 50:
        print(f"  Max validation accuracy reached: {df_log['val_acc'].max():.2f}%")
    else:
        print(f"  Max validation accuracy only: {df_log['val_acc'].max():.2f}%")
    
    # Check for NaN values
    if df_log.isna().any().any():
        print("  NaN values found in training log!")
    else:
        print("  No NaN values in training log")
    
    # Check loss progression
    first_loss = df_log['train_loss'].iloc[0]
    last_loss = df_log['train_loss'].iloc[-1]
    print(f"  Loss reduction: {first_loss:.4f} → {last_loss:.4f} ({100*(first_loss-last_loss)/first_loss:.1f}% reduction)")
    
else:
    print("No training log found!")

# ============================================================
# 2. Check Checkpoint Contents
# ============================================================
print("\n" + "="*60)
print("CHECKING CHECKPOINT CONTENTS")
print("-" * 40)

checkpoint_path = Path('05_Model/Saved_Models/best_model_v3.pth')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print(f"Checkpoint keys: {list(checkpoint.keys())}")
print(f"Epoch saved: {checkpoint.get('epoch', 'N/A')}")
print(f"Val accuracy at save: {checkpoint.get('val_acc', 'N/A')}")
print(f"Val loss at save: {checkpoint.get('val_loss', 'N/A')}")

# Check model state dict
state_dict = checkpoint['model_state_dict']
print(f"\nModel state dict has {len(state_dict)} keys")

# Check classifier weights specifically
print("\nClassifier final layer analysis:")
final_weight = state_dict['classifier.6.weight']  # Shape: [87, 512]
final_bias = state_dict['classifier.6.bias']      # Shape: [87]

print(f"Final layer weight shape: {final_weight.shape}")
print(f"Final layer bias shape: {final_bias.shape}")

# Check weight statistics
print(f"\nWeight statistics:")
print(f"  Mean: {final_weight.mean().item():.6f}")
print(f"  Std:  {final_weight.std().item():.6f}")
print(f"  Min:  {final_weight.min().item():.6f}")
print(f"  Max:  {final_weight.max().item():.6f}")

print(f"\nBias statistics:")
print(f"  Mean: {final_bias.mean().item():.6f}")
print(f"  Std:  {final_bias.std().item():.6f}")
print(f"  Min:  {final_bias.min().item():.6f}")
print(f"  Max:  {final_bias.max().item():.6f}")

# Check which class has highest bias (this would cause constant prediction)
max_bias_idx = final_bias.argmax().item()
min_bias_idx = final_bias.argmin().item()

# Load label mapping
with open('04_Labels/Processed_Labels/label_mapping_v3.json', 'r') as f:
    label_mapping = json.load(f)
    
# Handle nested dict structure
if isinstance(list(label_mapping.values())[0], dict):
    idx_to_species = {v['index']: k for k, v in label_mapping.items()}
else:
    idx_to_species = {v: k for k, v in label_mapping.items()}

print(f"\nClass with highest bias: {idx_to_species[max_bias_idx]} (idx={max_bias_idx}, bias={final_bias[max_bias_idx].item():.4f})")
print(f"Class with lowest bias: {idx_to_species[min_bias_idx]} (idx={min_bias_idx}, bias={final_bias[min_bias_idx].item():.4f})")

# Check if biases are reasonable or collapsed
bias_range = final_bias.max().item() - final_bias.min().item()
print(f"\nBias range: {bias_range:.4f}")
if bias_range < 0.1:
    print("Bias range is very small - might be underfit")
elif bias_range > 10:
    print("Bias range is very large - might be overfit to one class")
else:
    print("Bias range looks reasonable")

# ============================================================
# 3. Check if there's a label mapping issue in training
# ============================================================
print("\n" + "="*60)
print("CHECKING FOR LABEL MAPPING MISMATCH")
print("-" * 40)

# Check what Bucephala albeola's index is
bucephala_idx = label_mapping.get('Bucephala albeola', 'NOT FOUND')
print(f"Bucephala albeola index in current mapping: {bucephala_idx}")

# Check training config
if 'config' in checkpoint:
    config = checkpoint['config']
    print(f"\nTraining config: {config}")

# ============================================================
# 4. Check if old label mapping exists
# ============================================================
print("\n" + "="*60)
print("CHECKING FOR MULTIPLE LABEL MAPPINGS")
print("-" * 40)

label_files = list(Path('04_Labels/Processed_Labels').glob('label_mapping*.json'))
print(f"Found {len(label_files)} label mapping files:")
for f in label_files:
    with open(f) as fp:
        mapping = json.load(fp)
    print(f"  {f.name}: {len(mapping)} classes")
    
    # Check what index 0 is in each mapping
    idx_0_species = [k for k, v in mapping.items() if v == 0]
    if idx_0_species:
        print(f"    Index 0 = {idx_0_species[0]}")

# ============================================================
# 5. Check training script for label handling
# ============================================================
print("\n" + "="*60)
print("HYPOTHESIS")
print("-" * 40)

print("""
Based on the evidence:

1. The model was trained and achieved 95.9% validation accuracy
2. But when we load and evaluate, it always predicts "Bucephala albeola"
3. The model weights loaded successfully with strict=True
4. Training samples also fail (0% accuracy)

POSSIBLE CAUSES:

A) LABEL INDEX MISMATCH:
   - The training script may have used a DIFFERENT label mapping
   - The CSV files have species names, but the training script
     might have created its own label→index mapping
   - Our evaluation uses label_mapping_v3.json which may not match

B) BATCHNORM IN EVAL MODE:
   - BatchNorm layers behave differently in train vs eval mode
   - If running_mean/running_var weren't saved correctly, 
     the model would give wrong outputs

C) CHECKPOINT CORRUPTION:
   - The checkpoint might have been saved at wrong state
   - Or the optimizer state overwrote model weights

Let's check hypothesis A by finding what mapping the training used.
""")

# ============================================================
# 6. Try to reconstruct training label mapping
# ============================================================
print("\n" + "="*60)
print("RECONSTRUCTING TRAINING LABEL MAPPING")
print("-" * 40)

# Load training data and create mapping the same way training script likely did
train_df = pd.read_csv('04_Labels/Train_Val_Test_Split/train_v3.csv')
unique_species = sorted(train_df['species_scientific'].unique())

print(f"Unique species in training CSV: {len(unique_species)}")

# Create mapping (sorted alphabetically - common approach)
reconstructed_mapping = {species: idx for idx, species in enumerate(unique_species)}

print("\nFirst 10 species in RECONSTRUCTED mapping:")
for species, idx in list(reconstructed_mapping.items())[:10]:
    print(f"  {idx:2d}: {species}")

print("\nFirst 10 species in CURRENT label_mapping_v3.json:")
for species, info in sorted(label_mapping.items(), key=lambda x: x[1]['index'] if isinstance(x[1], dict) else x[1])[:10]:
    if isinstance(info, dict):
        print(f"  {info['index']:2d}: {species}")
    else:
        print(f"  {info:2d}: {species}")

# Check if they match
matches = 0
for s in unique_species:
    recon_idx = reconstructed_mapping.get(s)
    # Handle nested dict structure
    current_idx = label_mapping.get(s)
    if isinstance(current_idx, dict):
        current_idx = current_idx.get('index')
    if recon_idx == current_idx:
        matches += 1

print(f"\nMatching indices: {matches}/{len(unique_species)}")

if matches == len(unique_species):
    print("Mappings are IDENTICAL")
else:
    print("Mappings are DIFFERENT!")
    
    # Show mismatches
    print("\nMismatched species:")
    for species in unique_species[:20]:
        recon_idx = reconstructed_mapping.get(species)
        current_info = label_mapping.get(species)
        current_idx = current_info['index'] if isinstance(current_info, dict) else current_info
        if recon_idx != current_idx:
            print(f"  {species}: training={recon_idx}, current={current_idx}")