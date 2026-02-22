# Save as: final_evaluation_npy.py

import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import json
import os

os.chdir(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")

print("="*60)
print("🎯 FINAL EVALUATION (NPY Spectrograms)")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# 1. Load label mapping
# ============================================================
print("\n📋 Loading label mapping...")

with open('04_Labels/Processed_Labels/label_mapping_v3.json', 'r') as f:
    label_mapping_raw = json.load(f)

species_to_idx = {k: v['index'] for k, v in label_mapping_raw.items()}
idx_to_species = {v: k for k, v in species_to_idx.items()}
idx_to_english = {v['index']: v['english_name'] for k, v in label_mapping_raw.items()}

print(f"✅ Loaded {len(species_to_idx)} species")

# ============================================================
# 2. Load model
# ============================================================
print("\n🔧 Loading model...")

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
checkpoint = torch.load('05_Model/Saved_Models/best_model_v3.pth', 
                        map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model = model.to(device)
model.eval()
print(f"✅ Model loaded (epoch {checkpoint.get('epoch', 'N/A')}, val_acc: {checkpoint.get('val_acc', 0):.2f}%)")

# ============================================================
# 3. Check spectrogram format
# ============================================================
print("\n🔍 Checking spectrogram format...")

spec_dir = Path('03_Features/Spectrograms')
sample_files = list(spec_dir.glob('*.npy'))[:3]

for f in sample_files:
    arr = np.load(f)
    print(f"  {f.name}: shape={arr.shape}, dtype={arr.dtype}, range=[{arr.min():.3f}, {arr.max():.3f}]")

# Determine the format
sample_arr = np.load(sample_files[0])
print(f"\nSpectrogram format: shape={sample_arr.shape}")

# ============================================================
# 4. Create preprocessing function
# ============================================================
def load_and_preprocess_spectrogram(npy_path, target_size=(128, 216)):
    """Load .npy spectrogram and convert to model input format"""
    
    # Load the numpy array
    spec = np.load(npy_path)
    
    # Handle different possible formats
    if spec.ndim == 2:
        # Single channel (H, W) - typical mel spectrogram
        # Convert to 3 channels by repeating
        spec_3ch = np.stack([spec, spec, spec], axis=0)  # (3, H, W)
    elif spec.ndim == 3:
        if spec.shape[0] == 3:
            # Already (3, H, W)
            spec_3ch = spec
        elif spec.shape[2] == 3:
            # (H, W, 3) - transpose to (3, H, W)
            spec_3ch = np.transpose(spec, (2, 0, 1))
        elif spec.shape[0] == 1:
            # (1, H, W) - repeat to 3 channels
            spec_3ch = np.repeat(spec, 3, axis=0)
        else:
            # Unknown format, try to handle
            spec_3ch = np.stack([spec[0], spec[0], spec[0]], axis=0)
    else:
        raise ValueError(f"Unexpected spectrogram shape: {spec.shape}")
    
    # Convert to tensor
    tensor = torch.from_numpy(spec_3ch).float()
    
    # Resize if needed
    if tensor.shape[1:] != target_size:
        tensor = tensor.unsqueeze(0)  # Add batch dim for interpolate
        tensor = torch.nn.functional.interpolate(
            tensor, size=target_size, mode='bilinear', align_corners=False
        )
        tensor = tensor.squeeze(0)  # Remove batch dim
    
    # Normalize (ImageNet stats - same as training)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # First normalize to 0-1 if needed
    if tensor.max() > 1.0:
        tensor = tensor / 255.0 if tensor.max() > 1.0 else tensor
    
    # Handle negative values (dB scale spectrograms)
    if tensor.min() < 0:
        # Normalize to 0-1 range
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    
    # Apply ImageNet normalization
    tensor = (tensor - mean) / std
    
    return tensor

# Test preprocessing
print("\n🧪 Testing preprocessing...")
test_tensor = load_and_preprocess_spectrogram(sample_files[0])
print(f"Preprocessed tensor: shape={test_tensor.shape}, range=[{test_tensor.min():.3f}, {test_tensor.max():.3f}]")

# ============================================================
# 5. Quick sanity check
# ============================================================
print("\n" + "="*60)
print("🧪 SANITY CHECK (20 samples)")
print("="*60)

test_df = pd.read_csv('04_Labels/Train_Val_Test_Split/test_v3.csv')

correct = 0
tested = 0

for _, row in test_df.sample(n=20, random_state=42).iterrows():
    spec_path = Path(row['spectrogram_path'])
    true_label = int(row['label'])
    species_name = row['species_scientific']
    
    if not spec_path.exists():
        # Try relative path
        spec_path = spec_dir / f"{row['chunk_filename']}.npy"
    
    if not spec_path.exists():
        continue
    
    try:
        input_tensor = load_and_preprocess_spectrogram(spec_path)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred_label = output.argmax(dim=1).item()
            pred_prob = torch.softmax(output, dim=1).max().item()
        
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        tested += 1
        
        status = "✅" if is_correct else "❌"
        pred_species = idx_to_species.get(pred_label, f"UNK_{pred_label}")
        print(f"  {status} True: {species_name[:25]:<25} | Pred: {pred_species[:25]:<25} | Conf: {pred_prob:.1%}")
        
    except Exception as e:
        print(f"  ⚠️ Error processing {spec_path.name}: {e}")

print(f"\nSanity check: {correct}/{tested} = {100*correct/tested:.1f}%")

# ============================================================
# 6. Full evaluation
# ============================================================
print("\n" + "="*60)
print("🧪 FULL TEST SET EVALUATION")
print("="*60)

all_preds = []
all_labels = []
all_probs = []
errors = 0

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
    spec_path = Path(row['spectrogram_path'])
    true_label = int(row['label'])
    
    if not spec_path.exists():
        spec_path = spec_dir / f"{row['chunk_filename']}.npy"
    
    if not spec_path.exists():
        errors += 1
        continue
    
    try:
        input_tensor = load_and_preprocess_spectrogram(spec_path)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_label = output.argmax(dim=1).item()
        
        all_preds.append(pred_label)
        all_labels.append(true_label)
        all_probs.append(probs[0].cpu().numpy())
        
    except Exception as e:
        errors += 1

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

print(f"\nProcessed: {len(all_preds)} samples")
print(f"Errors/Missing: {errors}")

# ============================================================
# 7. Calculate metrics
# ============================================================
print("\n" + "="*60)
print("📊 RESULTS")
print("="*60)

# Overall accuracy
accuracy = 100.0 * np.mean(all_preds == all_labels)
print(f"\n🎯 Test Accuracy: {accuracy:.2f}%")

# Top-5 accuracy
top5_correct = 0
for i in range(len(all_labels)):
    top5_indices = np.argsort(all_probs[i])[-5:]
    if all_labels[i] in top5_indices:
        top5_correct += 1
top5_accuracy = 100.0 * top5_correct / len(all_labels)
print(f"🎯 Top-5 Accuracy: {top5_accuracy:.2f}%")

# Per-class accuracy
per_class_correct = {}
per_class_total = {}

for pred, label in zip(all_preds, all_labels):
    if label not in per_class_total:
        per_class_total[label] = 0
        per_class_correct[label] = 0
    per_class_total[label] += 1
    if pred == label:
        per_class_correct[label] += 1

per_class_accuracy = {
    label: 100.0 * per_class_correct[label] / per_class_total[label]
    for label in per_class_total
}

accuracies = list(per_class_accuracy.values())

print(f"\n📈 Per-Species Statistics:")
print(f"  Mean:   {np.mean(accuracies):.2f}%")
print(f"  Median: {np.median(accuracies):.2f}%")
print(f"  Min:    {np.min(accuracies):.2f}%")
print(f"  Max:    {np.max(accuracies):.2f}%")

# Tier breakdown
tier_95 = sum(1 for a in accuracies if a >= 95)
tier_90 = sum(1 for a in accuracies if 90 <= a < 95)
tier_85 = sum(1 for a in accuracies if 85 <= a < 90)
tier_80 = sum(1 for a in accuracies if 80 <= a < 85)
tier_below = sum(1 for a in accuracies if a < 80)

print(f"\n🎯 Accuracy Distribution:")
print(f"  ≥95%:    {tier_95:2d} species")
print(f"  90-95%:  {tier_90:2d} species")
print(f"  85-90%:  {tier_85:2d} species")
print(f"  80-85%:  {tier_80:2d} species")
print(f"  <80%:    {tier_below:2d} species")

# Best and worst performers
sorted_classes = sorted(per_class_accuracy.items(), key=lambda x: x[1])

print(f"\n⚠️ Lowest Performing (Bottom 5):")
for label, acc in sorted_classes[:5]:
    species = idx_to_species.get(label, f"Label_{label}")
    count = per_class_total[label]
    print(f"  {acc:5.1f}% | {species:<35} (n={count})")

print(f"\n✅ Highest Performing (Top 5):")
for label, acc in sorted_classes[-5:][::-1]:
    species = idx_to_species.get(label, f"Label_{label}")
    count = per_class_total[label]
    print(f"  {acc:5.1f}% | {species:<35} (n={count})")

# ============================================================
# 8. Save results
# ============================================================
results = {
    'test_accuracy': float(accuracy),
    'test_top5': float(top5_accuracy),
    'samples_evaluated': len(all_preds),
    'per_class_accuracy': {idx_to_species[k]: float(v) for k, v in per_class_accuracy.items()},
}

with open('05_Model/Training_Logs/test_results_v3_FINAL.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n💾 Results saved to: test_results_v3_FINAL.json")

# ============================================================
# 9. Summary
# ============================================================
print("\n" + "="*60)
print("🏆 FINAL SUMMARY")
print("="*60)

val_acc = checkpoint.get('val_acc', 0)
target_met = accuracy >= 85

print(f"""
┌─────────────────────────────────────────────────────────────┐
│  🐦 MODEL V3 - FINAL TEST RESULTS                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 METRICS                                                 │
│  ├── Test Accuracy:     {accuracy:6.2f}%                         │
│  ├── Top-5 Accuracy:    {top5_accuracy:6.2f}%                         │
│  ├── Val Accuracy:      {val_acc:6.2f}% (training)                │
│  └── Target (85%):      {'✅ ACHIEVED!' if target_met else '❌ Not met'}                         │
│                                                             │
│  📈 SPECIES BREAKDOWN                                       │
│  ├── ≥95%:   {tier_95:2d} species                                   │
│  ├── 90-95%: {tier_90:2d} species                                   │
│  ├── 85-90%: {tier_85:2d} species                                   │
│  ├── 80-85%: {tier_80:2d} species                                   │
│  └── <80%:   {tier_below:2d} species                                   │
│                                                             │
│  Gap (Val - Test): {val_acc - accuracy:+.2f}%                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")

if target_met:
    print("🎉 SUCCESS! Model is ready for deployment!")
    print("   Next step: Build the web application")