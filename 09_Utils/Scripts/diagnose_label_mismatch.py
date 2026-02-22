# Save as: diagnose_label_mismatch.py

import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
from PIL import Image
import json
from pathlib import Path
from collections import Counter
import random
import os

print("="*60)
print("DIAGNOSING LABEL MISMATCH")
print("="*60)

# ============================================================
# 1. Load Label Mapping
# ============================================================
print("\nLoading label mapping...")

# Handle path resolution - work from project root
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
os.chdir(project_root)

label_mapping_path = project_root / '04_Labels/Processed_Labels/label_mapping_v3.json'
with open(label_mapping_path, 'r') as f:
    label_mapping = json.load(f)

print(f"Number of classes in mapping: {len(label_mapping)}")
print(f"\nFirst 10 species in label_mapping:")
for i, (species, info) in enumerate(sorted(label_mapping.items(), key=lambda x: x[1]['index'] if isinstance(x[1], dict) else x[1])[:10]):
    if isinstance(info, dict):
        print(f"  {info['index']:2d}: {species}")
    else:
        print(f"  {info:2d}: {species}")

# Create proper reverse mapping
if isinstance(list(label_mapping.values())[0], dict):
    idx_to_species = {v['index']: k for k, v in label_mapping.items()}
else:
    idx_to_species = {v: k for k, v in label_mapping.items()}

# ============================================================
# 2. Check Training vs Test CSV Species
# ============================================================
print("\n" + "="*60)
print("COMPARING TRAIN VS TEST SPECIES")
print("="*60)

train_df = pd.read_csv(project_root / '04_Labels/Train_Val_Test_Split/train_v3.csv')
val_df = pd.read_csv(project_root / '04_Labels/Train_Val_Test_Split/val_v3.csv')
test_df = pd.read_csv(project_root / '04_Labels/Train_Val_Test_Split/test_v3.csv')

# Determine the species column name
species_col = 'species_scientific' if 'species_scientific' in train_df.columns else 'species'

train_species = set(train_df[species_col].unique())
val_species = set(val_df[species_col].unique())
test_species = set(test_df[species_col].unique())
mapping_species = set(label_mapping.keys())

print(f"\nSpecies counts:")
print(f"  Training CSV:     {len(train_species)} species")
print(f"  Validation CSV:   {len(val_species)} species")
print(f"  Test CSV:         {len(test_species)} species")
print(f"  Label mapping:    {len(mapping_species)} species")

# Check for mismatches
train_not_in_mapping = train_species - mapping_species
test_not_in_mapping = test_species - mapping_species
mapping_not_in_train = mapping_species - train_species

if train_not_in_mapping:
    print(f"\n⚠️ Species in TRAIN but NOT in label_mapping ({len(train_not_in_mapping)}):")
    for s in list(train_not_in_mapping)[:5]:
        print(f"    - {s}")

if test_not_in_mapping:
    print(f"\n⚠️ Species in TEST but NOT in label_mapping ({len(test_not_in_mapping)}):")
    for s in list(test_not_in_mapping)[:5]:
        print(f"    - {s}")

if mapping_not_in_train:
    print(f"\n⚠️ Species in label_mapping but NOT in TRAIN ({len(mapping_not_in_train)}):")
    for s in list(mapping_not_in_train)[:5]:
        print(f"    - {s}")

# ============================================================
# 3. Check if Species Names Match Exactly
# ============================================================
print("\n" + "="*60)
print("🔤 CHECKING SPECIES NAME FORMAT")
print("="*60)

# Sample some species from each source
print("\nSample species from training CSV:")
for s in list(train_species)[:5]:
    print(f"  '{s}' (len={len(s)})")

print("\nSample species from label_mapping:")
for s in list(mapping_species)[:5]:
    print(f"  '{s}' (len={len(s)})")

# Check for whitespace issues
def check_whitespace(species_set, name):
    issues = []
    for s in species_set:
        if s != s.strip():
            issues.append(f"Leading/trailing whitespace: '{s}'")
        if '  ' in s:
            issues.append(f"Double space: '{s}'")
    if issues:
        print(f"\n⚠️ Whitespace issues in {name}:")
        for issue in issues[:5]:
            print(f"    {issue}")
    else:
        print(f"No whitespace issues in {name}")

check_whitespace(train_species, "training CSV")
check_whitespace(mapping_species, "label_mapping")
check_whitespace(test_species, "test CSV")

# ============================================================
# 4. Test Model on Training Data (Sanity Check)
# ============================================================
print("\n" + "="*60)
print("TESTING MODEL ON TRAINING SAMPLES")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
class BirdClassifier(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b2_ns', num_classes=87, dropout_rate=0.4):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='avg')
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.75),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

model = BirdClassifier(num_classes=len(label_mapping))
checkpoint = torch.load(project_root / '05_Model/Saved_Models/best_model_v3.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# CRITICAL FIX: Use exact same preprocessing as training!
# Training normalizes spectrograms to [0,1] and does NOT use ImageNet normalization
def normalize_spec(spec):
    """Match training preprocessing exactly"""
    spec_min = spec.min()
    spec_max = spec.max()
    if spec_max - spec_min > 0:
        spec = (spec - spec_min) / (spec_max - spec_min)
    else:
        spec = np.zeros_like(spec)
    spec = np.clip(spec, 0, 1)
    # Convert to 3-channel: (1, 128, 216) -> (3, 128, 216)
    spec_tensor = torch.FloatTensor(spec).unsqueeze(0)
    spec_tensor = spec_tensor.repeat(3, 1, 1)
    return spec_tensor

# Test on random TRAINING samples
print("\nTesting on 20 random TRAINING samples...")
spec_dir = project_root / '03_Features/Spectrograms'

train_samples = train_df.sample(n=min(20, len(train_df))).to_dict('records')
correct = 0
results = []

for sample in train_samples:
    # Handle both chunk_id and chunk_filename column names
    chunk_id = sample.get('chunk_filename', sample.get('chunk_id', ''))
    true_species = sample[species_col]
    true_label = sample.get('label', None)
    
    if true_species not in label_mapping:
        results.append((chunk_id, true_species, "NOT IN MAPPING", "N/A"))
        continue
    
    # Try .npy first, then .png
    spec_path = spec_dir / f"{chunk_id}.npy"
    if not spec_path.exists():
        spec_path = spec_dir / f"{chunk_id}.png"
    
    if not spec_path.exists():
        results.append((chunk_id, true_species, "FILE NOT FOUND", "N/A"))
        continue
    
    # Load spectrogram - MATCH TRAINING EXACTLY
    if spec_path.suffix == '.npy':
        spec = np.load(spec_path)  # (128, 216)
        spec_tensor = normalize_spec(spec)  # Use training normalization
    else:
        # For PNG, convert to numpy first then normalize
        img = Image.open(spec_path).convert('L')  # Grayscale
        spec = np.array(img, dtype=np.float32)
        spec_tensor = normalize_spec(spec)
    
    spec_tensor = spec_tensor.unsqueeze(0).to(device)  # Add batch dim
    
    with torch.no_grad():
        output = model(spec_tensor)
        pred_label = output.argmax(dim=1).item()
        pred_species = idx_to_species[pred_label]
        confidence = torch.softmax(output, dim=1).max().item()
    
    is_correct = (true_label is not None and pred_label == true_label) or (true_label is None and pred_species == true_species)
    if is_correct:
        correct += 1
    
    results.append((chunk_id[:30], true_species[:25], pred_species[:25], f"{confidence:.2%}", "✓" if is_correct else "✗"))

print(f"\nResults on TRAINING samples: {correct}/{len(train_samples)} correct ({100*correct/len(train_samples):.1f}%)")
print("\nSample predictions:")
print(f"{'Chunk ID':<32} {'True Species':<27} {'Predicted':<27} {'Conf':<7} {'OK'}")
print("-" * 100)
for r in results[:15]:
    print(f"{r[0]:<32} {r[1]:<27} {r[2]:<27} {r[3]:<7} {r[4] if len(r) > 4 else ''}")

# ============================================================
# 5. Test on Validation Data
# ============================================================
print("\n" + "="*60)
print("TESTING MODEL ON VALIDATION SAMPLES")
print("="*60)

val_samples = val_df.sample(n=min(20, len(val_df))).to_dict('records')
correct_val = 0

for sample in val_samples:
    chunk_id = sample.get('chunk_filename', sample.get('chunk_id', ''))
    true_species = sample[species_col]
    true_label = sample.get('label', None)
    
    if true_species not in label_mapping:
        continue
    
    # Try .npy first, then .png
    spec_path = spec_dir / f"{chunk_id}.npy"
    if not spec_path.exists():
        spec_path = spec_dir / f"{chunk_id}.png"
    
    if not spec_path.exists():
        continue
    
    # Load spectrogram - MATCH TRAINING EXACTLY
    if spec_path.suffix == '.npy':
        spec = np.load(spec_path)  # (128, 216)
        spec_tensor = normalize_spec(spec)
    else:
        img = Image.open(spec_path).convert('L')
        spec = np.array(img, dtype=np.float32)
        spec_tensor = normalize_spec(spec)
    
    spec_tensor = spec_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(spec_tensor)
        pred_label = output.argmax(dim=1).item()
    
    if true_label is not None and pred_label == true_label:
        correct_val += 1

print(f"Results on VALIDATION samples: {correct_val}/{len(val_samples)} correct ({100*correct_val/len(val_samples):.1f}%)")

# ============================================================
# 6. Test on Test Data
# ============================================================
print("\n" + "="*60)
print("TESTING MODEL ON TEST SAMPLES")
print("="*60)

test_samples = test_df.sample(n=min(20, len(test_df))).to_dict('records')
correct_test = 0
test_results = []

for sample in test_samples:
    chunk_id = sample.get('chunk_filename', sample.get('chunk_id', ''))
    true_species = sample[species_col]
    true_label = sample.get('label', None)
    
    if true_species not in label_mapping:
        test_results.append((chunk_id[:30], true_species[:25], "NOT IN MAPPING", "N/A", "⚠"))
        continue
    
    # Try .npy first, then .png
    spec_path = spec_dir / f"{chunk_id}.npy"
    if not spec_path.exists():
        spec_path = spec_dir / f"{chunk_id}.png"
    
    if not spec_path.exists():
        test_results.append((chunk_id[:30], true_species[:25], "FILE NOT FOUND", "N/A", "warn"))
        continue
    
    # Load spectrogram - MATCH TRAINING EXACTLY
    if spec_path.suffix == '.npy':
        spec = np.load(spec_path)  # (128, 216)
        spec_tensor = normalize_spec(spec)
    else:
        img = Image.open(spec_path).convert('L')
        spec = np.array(img, dtype=np.float32)
        spec_tensor = normalize_spec(spec)
    
    spec_tensor = spec_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(spec_tensor)
        pred_label = output.argmax(dim=1).item()
        pred_species = idx_to_species[pred_label]
        confidence = torch.softmax(output, dim=1).max().item()
    
    is_correct = (true_label is not None and pred_label == true_label) or (true_label is None and pred_species == true_species)
    if is_correct:
        correct_test += 1
    
    test_results.append((chunk_id[:30], true_species[:25], pred_species[:25], f"{confidence:.2%}", "OK" if is_correct else "NO"))

print(f"\nResults on TEST samples: {correct_test}/{len(test_samples)} correct ({100*correct_test/len(test_samples):.1f}%)")
print("\nSample predictions:")
print(f"{'Chunk ID':<32} {'True Species':<27} {'Predicted':<27} {'Conf':<7} {'OK'}")
print("-" * 100)
for r in test_results[:15]:
    print(f"{r[0]:<32} {r[1]:<27} {r[2]:<27} {r[3]:<7} {r[4]}")

# ============================================================
# 7. Check Prediction Distribution
# ============================================================
print("\n" + "="*60)
print("PREDICTION DISTRIBUTION ANALYSIS")
print("="*60)

# Get predictions for more test samples
test_sample_large = test_df.sample(n=min(500, len(test_df))).to_dict('records')
pred_counter = Counter()
true_counter = Counter()

for sample in test_sample_large:
    chunk_id = sample.get('chunk_filename', sample.get('chunk_id', ''))
    true_species = sample[species_col]
    true_counter[true_species] += 1
    
    if true_species not in label_mapping:
        continue
    
    # Try .npy first, then .png
    spec_path = spec_dir / f"{chunk_id}.npy"
    if not spec_path.exists():
        spec_path = spec_dir / f"{chunk_id}.png"
    if not spec_path.exists():
        continue
    
    # Load spectrogram - MATCH TRAINING EXACTLY
    if spec_path.suffix == '.npy':
        spec = np.load(spec_path)  # (128, 216)
        spec_tensor = normalize_spec(spec)
    else:
        img = Image.open(spec_path).convert('L')
        spec = np.array(img, dtype=np.float32)
        spec_tensor = normalize_spec(spec)
    
    spec_tensor = spec_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(spec_tensor)
        pred_label = output.argmax(dim=1).item()
        pred_species = idx_to_species[pred_label]
    
    pred_counter[pred_species] += 1

print("\nTop 10 PREDICTED species:")
for species, count in pred_counter.most_common(10):
    print(f"  {species:<40} {count:4d} predictions")

print("\nTop 10 TRUE species in test sample:")
for species, count in true_counter.most_common(10):
    print(f"  {species:<40} {count:4d} samples")

# ============================================================
# 8. Summary
# ============================================================
print("\n" + "="*60)
print("DIAGNOSIS SUMMARY")
print("="*60)

print(f"""
Training accuracy on sample:    {100*correct/len(train_samples):.1f}%
Validation accuracy on sample:  {100*correct_val/len(val_samples):.1f}%
Test accuracy on sample:        {100*correct_test/len(test_samples):.1f}%

If TRAINING accuracy is high but TEST accuracy is low:
  → Label mapping is likely different between train/test splits
  → Or test spectrograms are from a different batch/processing

If ALL accuracies are low:
  → Model checkpoint may be corrupted
  → Or there's a fundamental data loading issue
""")