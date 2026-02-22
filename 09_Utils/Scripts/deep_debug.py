# Save as: deep_debug_v3.py

import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
from PIL import Image
import json
from pathlib import Path
from torchvision import transforms
import glob
import os

os.chdir(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")

print("="*60)
print("🔬 DEEP DEBUG V3: Complete Analysis")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 1. Load and fix label mapping
# ============================================================
print("\n📋 Loading label mapping...")

with open('04_Labels/Processed_Labels/label_mapping_v3.json', 'r') as f:
    label_mapping_raw = json.load(f)

# Extract correct mapping (it's nested!)
label_mapping = {k: v['index'] for k, v in label_mapping_raw.items()}
idx_to_species = {v: k for k, v in label_mapping.items()}
print(f"✅ Extracted {len(label_mapping)} species from nested structure")

# ============================================================
# 2. Check CSV structure
# ============================================================
print("\n📋 Checking CSV structure...")

train_df = pd.read_csv('04_Labels/Train_Val_Test_Split/train_v3.csv')
print(f"Train CSV columns: {list(train_df.columns)}")
print(f"Train CSV shape: {train_df.shape}")
print(f"\nFirst row:")
print(train_df.iloc[0])

# Find the correct column names
chunk_col = None
species_col = None

for col in train_df.columns:
    col_lower = col.lower()
    if 'chunk' in col_lower or 'file' in col_lower or 'path' in col_lower or 'id' in col_lower:
        chunk_col = col
    if 'species' in col_lower or 'label' in col_lower or 'class' in col_lower:
        species_col = col

# If not found, use first and second columns
if chunk_col is None:
    chunk_col = train_df.columns[0]
if species_col is None:
    species_col = train_df.columns[1] if len(train_df.columns) > 1 else train_df.columns[0]

print(f"\nUsing columns:")
print(f"  Chunk/File column: '{chunk_col}'")
print(f"  Species column: '{species_col}'")

# ============================================================
# 3. Load model
# ============================================================
print("\n" + "="*60)
print("🔧 LOADING MODEL")
print("="*60)

class BirdClassifier(nn.Module):
    def __init__(self, num_classes=87, dropout_rate=0.4):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnet_b2_ns', pretrained=False, 
                                          num_classes=0, global_pool='avg')
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

model = BirdClassifier(num_classes=87)
checkpoint = torch.load('05_Model/Saved_Models/best_model_v3.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model = model.to(device)
model.eval()
print("✅ Model loaded")

# ============================================================
# 4. Test with correct data loading
# ============================================================
print("\n" + "="*60)
print("🔬 TESTING WITH CORRECT DATA")
print("="*60)

transform = transforms.Compose([
    transforms.Resize((128, 216)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

spec_dir = Path('03_Features/Spectrograms')

# Get a sample
sample = train_df.iloc[0]
chunk_id = sample[chunk_col]
true_species = sample[species_col]

print(f"Sample chunk: {chunk_id}")
print(f"Sample species: {true_species}")

# Check if species in mapping
if true_species in label_mapping:
    true_label = label_mapping[true_species]
    print(f"True label index: {true_label}")
else:
    print(f"⚠️ Species '{true_species}' NOT in label mapping!")
    # Check for similar
    for s in label_mapping.keys():
        if true_species.lower() in s.lower() or s.lower() in true_species.lower():
            print(f"  Similar: '{s}'")

# Find spectrogram
spec_path = spec_dir / f"{chunk_id}.png"
if not spec_path.exists():
    # Maybe chunk_id already has extension?
    spec_path = spec_dir / chunk_id
    if not spec_path.exists():
        # Search for it
        matches = list(spec_dir.glob(f"*{chunk_id[:20]}*"))
        print(f"Searching for spectrogram... Found {len(matches)} matches")
        if matches:
            spec_path = matches[0]

print(f"Spectrogram path: {spec_path}")
print(f"Exists: {spec_path.exists()}")

if spec_path.exists():
    image = Image.open(spec_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_label = output.argmax(dim=1).item()
        pred_species = idx_to_species[pred_label]
        pred_prob = probs[0, pred_label].item()
    
    print(f"\n🎯 PREDICTION:")
    print(f"  Predicted: {pred_species} (idx={pred_label}, prob={pred_prob:.4f})")
    print(f"  True:      {true_species} (idx={label_mapping.get(true_species, 'N/A')})")
    
    # Top 5
    top5_probs, top5_indices = torch.topk(probs[0], 5)
    print(f"\n  Top 5:")
    for p, i in zip(top5_probs.cpu().numpy(), top5_indices.cpu().numpy()):
        marker = "👈 TRUE" if i == label_mapping.get(true_species) else ""
        print(f"    {i:2d}: {idx_to_species[i]:40s} {p:.4f} {marker}")

# ============================================================
# 5. Test multiple samples
# ============================================================
print("\n" + "="*60)
print("🧪 TESTING MULTIPLE SAMPLES (with correct mapping)")
print("="*60)

correct = 0
total = 0
predictions = []

for i, (_, row) in enumerate(train_df.sample(n=50, random_state=42).iterrows()):
    chunk_id = row[chunk_col]
    true_species = row[species_col]
    
    if true_species not in label_mapping:
        continue
    
    true_label = label_mapping[true_species]
    
    # Try to find spectrogram
    spec_path = spec_dir / f"{chunk_id}.png"
    if not spec_path.exists():
        spec_path = spec_dir / chunk_id
    if not spec_path.exists():
        continue
    
    image = Image.open(spec_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred_label = output.argmax(dim=1).item()
        pred_prob = torch.softmax(output, dim=1).max().item()
    
    total += 1
    if pred_label == true_label:
        correct += 1
    
    predictions.append({
        'true': true_species,
        'true_idx': true_label,
        'pred': idx_to_species[pred_label],
        'pred_idx': pred_label,
        'correct': pred_label == true_label,
        'conf': pred_prob
    })

print(f"\n✅ Accuracy: {correct}/{total} = {100*correct/total:.1f}%")

# Show distribution
print(f"\nPrediction distribution:")
from collections import Counter
pred_counts = Counter([p['pred'] for p in predictions])
for pred, count in pred_counts.most_common(10):
    print(f"  {pred}: {count}")

# Show some examples
print(f"\nSample predictions:")
print(f"{'True Species':<35} {'Pred Species':<35} {'Conf':>6} {'OK'}")
print("-" * 85)
for p in predictions[:15]:
    status = "✅" if p['correct'] else "❌"
    print(f"{p['true']:<35} {p['pred']:<35} {p['conf']:>6.2%} {status}")

# ============================================================
# 6. Summary
# ============================================================
print("\n" + "="*60)
print("📋 FINAL DIAGNOSIS")
print("="*60)

if correct/total > 0.8:
    print(f"""
🎉 SUCCESS! The model works correctly when using the FIXED label mapping!

The issue was: label_mapping_v3.json has a NESTED structure:
  "Species": {{"index": X, "english_name": "..."}}

But the evaluation code was treating it as:
  "Species": X

SOLUTION: Update all evaluation scripts to extract the 'index' field:
  label_mapping = {{k: v['index'] for k, v in raw_mapping.items()}}

Accuracy with fixed mapping: {100*correct/total:.1f}%
""")
else:
    print(f"""
⚠️ Accuracy still low ({100*correct/total:.1f}%) even with fixed mapping.

There may be additional issues:
1. Check if chunk_id format matches spectrogram filenames
2. Check if training used different preprocessing
3. Check if there's a data split issue

Let me investigate further...
""")

    # Additional diagnostics
    print("\nChecking chunk_id format:")
    sample_chunks = train_df[chunk_col].iloc[:5].tolist()
    for c in sample_chunks:
        exists = (spec_dir / f"{c}.png").exists()
        print(f"  '{c}' -> .png exists: {exists}")