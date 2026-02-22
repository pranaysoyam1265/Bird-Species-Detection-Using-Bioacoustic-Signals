#!/usr/bin/env python
"""
Final test evaluation with correct preprocessing matching training.
Tests model on full test set with proper spectrogram normalization.
"""

import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import json

# Resolve project root
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
os.chdir(project_root)

print("="*60)
print("FINAL TEST EVALUATION")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# 1. Load label mapping from JSON
# ============================================================
print("\nLoading label mapping...")
with open('04_Labels/Processed_Labels/label_mapping_v3.json', 'r') as f:
    label_mapping_nested = json.load(f)

# Handle nested dict structure
if isinstance(list(label_mapping_nested.values())[0], dict):
    idx_to_species = {v['index']: k for k, v in label_mapping_nested.items()}
    label_mapping = {k: v['index'] for k, v in label_mapping_nested.items()}
else:
    idx_to_species = {v: k for k, v in label_mapping_nested.items()}
    label_mapping = label_mapping_nested

num_classes = len(label_mapping)
print(f"Number of classes: {num_classes}")

# ============================================================
# 2. Load test data
# ============================================================
print("\nLoading test data...")
test_df = pd.read_csv('04_Labels/Train_Val_Test_Split/test_v3.csv')
print(f"Test set size: {len(test_df)}")

# ============================================================
# 3. Define Model
# ============================================================
class BirdClassifier(nn.Module):
    def __init__(self, num_classes=87, dropout_rate=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnet_b2_ns',
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )
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

# ============================================================
# 4. Load model checkpoint
# ============================================================
print("\nLoading model...")
model = BirdClassifier(num_classes=num_classes)
checkpoint = torch.load(
    '05_Model/Saved_Models/best_model_v3.pth',
    map_location=device,
    weights_only=False
)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model = model.to(device)
model.eval()
print(f"Model loaded (epoch {checkpoint.get('epoch', 'N/A')}, val_acc {checkpoint.get('val_acc', 0):.2f}%)")

# ============================================================
# 5. Preprocessing function - MUST MATCH TRAINING
# ============================================================
def normalize_spec(spec):
    """Normalize spectrogram to [0,1] - exact match to training"""
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

# ============================================================
# 6. Evaluate on test set
# ============================================================
print("\nEvaluating on test set...")

spec_dir = Path('03_Features/Spectrograms')
all_preds = []
all_labels = []
all_probs = []
valid_count = 0

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    chunk_id = row['chunk_filename']
    true_label = int(row['label'])
    
    # Try .npy first, then .png
    spec_path = spec_dir / f"{chunk_id}.npy"
    if not spec_path.exists():
        spec_path = spec_dir / f"{chunk_id}.png"
    
    if not spec_path.exists():
        continue
    
    # Load and normalize spectrogram
    try:
        if spec_path.suffix == '.npy':
            spec = np.load(spec_path)
        else:
            from PIL import Image
            img = Image.open(spec_path).convert('L')
            spec = np.array(img, dtype=np.float32)
        
        spec_tensor = normalize_spec(spec)
        spec_tensor = spec_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(spec_tensor)
            pred_label = output.argmax(dim=1).item()
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        all_preds.append(pred_label)
        all_labels.append(true_label)
        all_probs.append(probs)
        valid_count += 1
    except Exception as e:
        print(f"Error processing {chunk_id}: {e}")
        continue

# ============================================================
# 7. Calculate metrics
# ============================================================
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

test_accuracy = 100.0 * np.mean(all_preds == all_labels)
print(f"\n{'='*60}")
print(f"TEST RESULTS (n={valid_count})")
print(f"{'='*60}")
print(f"\nAccuracy: {test_accuracy:.2f}%")

# Top-5
top5_correct = 0
for i, probs in enumerate(all_probs):
    top5_indices = np.argsort(probs)[-5:]
    if all_labels[i] in top5_indices:
        top5_correct += 1
top5_accuracy = 100.0 * top5_correct / len(all_labels)
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

# Per-class accuracy
per_class_correct = {}
per_class_total = {}

for pred, label in zip(all_preds, all_labels):
    species = idx_to_species[label]
    if species not in per_class_total:
        per_class_total[species] = 0
        per_class_correct[species] = 0
    per_class_total[species] += 1
    if pred == label:
        per_class_correct[species] += 1

per_class_accuracy = {
    species: 100.0 * per_class_correct[species] / per_class_total[species]
    for species in per_class_total
}

accuracies = list(per_class_accuracy.values())
print(f"\nPER-SPECIES STATISTICS:")
print(f"  Mean: {np.mean(accuracies):.2f}%")
print(f"  Median: {np.median(accuracies):.2f}%")
print(f"  Std: {np.std(accuracies):.2f}%")
print(f"  Min: {np.min(accuracies):.2f}%")
print(f"  Max: {np.max(accuracies):.2f}%")

# Distribution
tier_95 = sum(1 for acc in accuracies if acc >= 95)
tier_90 = sum(1 for acc in accuracies if 90 <= acc < 95)
tier_80 = sum(1 for acc in accuracies if 80 <= acc < 90)
tier_70 = sum(1 for acc in accuracies if 70 <= acc < 80)
tier_below = sum(1 for acc in accuracies if acc < 70)

print(f"\nACCURACY TIERS:")
print(f"  >=95%: {tier_95} species")
print(f"  90-95%: {tier_90} species")
print(f"  80-90%: {tier_80} species")
print(f"  70-80%: {tier_70} species")
print(f"  <70%: {tier_below} species")

# Best and worst
sorted_species = sorted(per_class_accuracy.items(), key=lambda x: x[1])
print(f"\nBEST PERFORMING:")
for species, acc in sorted_species[-5:][::-1]:
    count = per_class_total[species]
    print(f"  {species}: {acc:.1f}% (n={count})")

print(f"\nWORST PERFORMING:")
for species, acc in sorted_species[:5]:
    count = per_class_total[species]
    print(f"  {species}: {acc:.1f}% (n={count})")

print(f"\n{'='*60}")
