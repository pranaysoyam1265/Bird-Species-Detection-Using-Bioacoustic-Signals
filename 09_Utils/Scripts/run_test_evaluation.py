# Save as: run_test_evaluation_fixed.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm
import pandas as pd
import numpy as np
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("="*60)
print("TEST SET EVALUATION (Fixed Architecture)")
print("="*60)

# ============================================================
# 1. Setup
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load label mapping
with open('04_Labels/Processed_Labels/label_mapping_v3.json', 'r') as f:
    label_mapping = json.load(f)

num_classes = len(label_mapping)
# Create reverse mapping from index to species name
idx_to_species = {v['index']: k for k, v in label_mapping.items()}
print(f"Number of classes: {num_classes}")

# ============================================================
# 2. Define Model Architecture (Exact Match to Training)
# ============================================================
class BirdClassifier(nn.Module):
    """EfficientNet-based bird species classifier - EXACT architecture from training"""
    
    def __init__(self, model_name='tf_efficientnet_b2_ns', num_classes=87, dropout_rate=0.4):
        super().__init__()
        
        # Create backbone (feature extractor)
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,  # We'll load weights from checkpoint
            num_classes=0,     # Remove original classifier
            global_pool='avg'
        )
        
        # Get feature dimension
        num_features = self.backbone.num_features  # 1408 for EfficientNet-B2
        
        # Custom classifier with dropout (EXACT structure from checkpoint)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),           # classifier.0
            nn.Dropout(dropout_rate),               # classifier.1
            nn.Linear(num_features, 512),           # classifier.2
            nn.ReLU(inplace=True),                  # classifier.3
            nn.BatchNorm1d(512),                    # classifier.4
            nn.Dropout(dropout_rate * 0.75),        # classifier.5
            nn.Linear(512, num_classes)             # classifier.6
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

# ============================================================
# 3. Load Model
# ============================================================
print("\nLoading model...")

# Create model with exact architecture
model = BirdClassifier(
    model_name='tf_efficientnet_b2_ns',
    num_classes=num_classes,
    dropout_rate=0.4
)

# Load checkpoint
checkpoint = torch.load(
    '05_Model/Saved_Models/best_model_v3.pth',
    map_location=device,
    weights_only=False
)

# Load state dict
state_dict = checkpoint['model_state_dict']

# Load weights with strict=True (should work now!)
try:
    model.load_state_dict(state_dict, strict=True)
    print("Model weights loaded with strict=True!")
except RuntimeError as e:
    print(f"Strict loading failed: {e}")
    # Try with strict=False and report
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

model = model.to(device)
model.eval()

# Print checkpoint info
print(f"\nCheckpoint info:")
print(f"  Best epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"  Val accuracy: {checkpoint.get('val_acc', 0):.2f}%")
print(f"  Val top-5: {checkpoint.get('val_top5', 0):.2f}%")

# ============================================================
# 4. Sanity Check
# ============================================================
print("\nRunning sanity check...")
dummy_input = torch.randn(1, 3, 128, 216).to(device)
with torch.no_grad():
    output = model(dummy_input)
    probs = torch.softmax(output, dim=1)

print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
print(f"Output std: {output.std().item():.4f}")
print(f"Max probability: {probs.max().item():.4f}")

if output.std().item() > 0.1:
    print("Model outputs look reasonable!")
else:
    print("WARNING: Low output variance - check model loading")

# ============================================================
# 5. Dataset
# ============================================================
class SpectrogramDataset(Dataset):
    def __init__(self, csv_path, spectrogram_dir, label_mapping):
        self.df = pd.read_csv(csv_path)
        self.spectrogram_dir = Path(spectrogram_dir)
        self.label_mapping = label_mapping
        
        from torchvision import transforms
        # IMPORTANT: Match the training preprocessing exactly!
        # Training uses raw [0,1] normalized spectrograms with NO ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((128, 216)),
            transforms.ToTensor(),
            # NO ImageNet normalization - training doesn't use it!
        ])
        
        # Filter valid samples
        valid_samples = []
        for idx, row in self.df.iterrows():
            # Handle different CSV formats
            if 'chunk_filename' in self.df.columns:
                base_filename = row['chunk_filename']
            elif 'chunk_id' in self.df.columns:
                base_filename = row['chunk_id']
            else:
                base_filename = Path(row.get('spectrogram_path', '')).stem
            
            # Try .npy first, then .png
            spec_path = self.spectrogram_dir / f"{base_filename}.npy"
            is_npy = True
            if not spec_path.exists():
                spec_path = self.spectrogram_dir / f"{base_filename}.png"
                is_npy = False
            
            # Get label
            if 'label' in self.df.columns:
                label = int(row['label'])
            else:
                species_key = row.get('species_scientific', row.get('species'))
                if species_key in self.label_mapping:
                    label = self.label_mapping[species_key]['index']
                else:
                    continue
            
            # Get species name for reference
            species_key = row.get('species_scientific', row.get('species_english', row.get('species', 'Unknown')))
            
            if spec_path.exists():
                valid_samples.append({
                    'path': spec_path,
                    'label': label,
                    'species': species_key,
                    'is_npy': is_npy
                })
        
        self.samples = valid_samples
        print(f"Loaded {len(self.samples)} valid test samples")
    
    def __len__(self):
        return len(self.samples)
    
    def _normalize(self, spec):
        """Normalize spectrogram to [0, 1] - MUST MATCH TRAINING"""
        spec_min = spec.min()
        spec_max = spec.max()
        if spec_max - spec_min > 0:
            spec = (spec - spec_min) / (spec_max - spec_min)
        else:
            spec = np.zeros_like(spec)
        return spec
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load .npy spectrogram (128, 216) - EXACT MATCH TO TRAINING
        spec = np.load(sample['path'])
        
        # Normalize to [0, 1] - CRITICAL: Training does this!
        spec = self._normalize(spec)
        
        # Clip
        spec = np.clip(spec, 0, 1)
        
        # Convert to 3-channel tensor - EXACT MATCH TO TRAINING
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0)  # (1, 128, 216)
        spec_tensor = spec_tensor.repeat(3, 1, 1)  # (3, 128, 216) - repeat, not convert
        
        return spec_tensor, sample['label'], sample['species']

# ============================================================
# 6. Load Test Data
# ============================================================
print("\n📂 Loading test dataset...")
test_dataset = SpectrogramDataset(
    csv_path='04_Labels/Train_Val_Test_Split/test_v3.csv',
    spectrogram_dir='03_Features/Spectrograms',
    label_mapping=label_mapping
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ============================================================
# 7. Evaluate
# ============================================================
print(f"\nEvaluating on {len(test_dataset)} test samples...")

all_preds = []
all_labels = []
all_probs = []
all_species = []

with torch.no_grad():
    for images, labels, species_names in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_species.extend(species_names)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ============================================================
# 8. Calculate Metrics
# ============================================================
print("\n" + "="*60)
print("TEST SET RESULTS")
print("="*60)

# Overall accuracy
test_accuracy = 100.0 * np.mean(all_preds == all_labels)
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# Top-5 accuracy
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

# Statistics
accuracies = list(per_class_accuracy.values())
print(f"\n📈 PER-SPECIES STATISTICS")
print("-" * 40)
print(f"Mean accuracy:     {np.mean(accuracies):.2f}%")
print(f"Median accuracy:   {np.median(accuracies):.2f}%")
print(f"Std deviation:     {np.std(accuracies):.2f}%")
print(f"Min accuracy:      {np.min(accuracies):.2f}%")
print(f"Max accuracy:      {np.max(accuracies):.2f}%")

# Tier breakdown
tier_95 = sum(1 for acc in accuracies if acc >= 95)
tier_90 = sum(1 for acc in accuracies if 90 <= acc < 95)
tier_85 = sum(1 for acc in accuracies if 85 <= acc < 90)
tier_80 = sum(1 for acc in accuracies if 80 <= acc < 85)
tier_below = sum(1 for acc in accuracies if acc < 80)

print(f"\nACCURACY DISTRIBUTION")
print("-" * 40)
print(f"≥95% accuracy:     {tier_95:3d} species ({100*tier_95/num_classes:.1f}%)")
print(f"90-95% accuracy:   {tier_90:3d} species ({100*tier_90/num_classes:.1f}%)")
print(f"85-90% accuracy:   {tier_85:3d} species ({100*tier_85/num_classes:.1f}%)")
print(f"80-85% accuracy:   {tier_80:3d} species ({100*tier_80/num_classes:.1f}%)")
print(f"<80% accuracy:     {tier_below:3d} species ({100*tier_below/num_classes:.1f}%)")

# Best and worst performers
sorted_species = sorted(per_class_accuracy.items(), key=lambda x: x[1])

print(f"\nLOWEST PERFORMING SPECIES (Bottom 10)")
print("-" * 55)
for species, acc in sorted_species[:10]:
    count = per_class_total[species]
    print(f"  {species:40s} {acc:5.1f}% (n={count:4d})")

print(f"\nHIGHEST PERFORMING SPECIES (Top 10)")
print("-" * 55)
for species, acc in sorted_species[-10:][::-1]:
    count = per_class_total[species]
    print(f"  {species:40s} {acc:5.1f}% (n={count:4d})")

# ============================================================
# 9. Save Results
# ============================================================
results = {
    'test_accuracy': float(test_accuracy),
    'test_top5': float(top5_accuracy),
    'per_class_accuracy': {k: float(v) for k, v in per_class_accuracy.items()},
    'per_class_samples': per_class_total,
    'statistics': {
        'mean': float(np.mean(accuracies)),
        'median': float(np.median(accuracies)),
        'std': float(np.std(accuracies)),
        'min': float(np.min(accuracies)),
        'max': float(np.max(accuracies))
    },
    'tiers': {
        '>=95%': tier_95,
        '90-95%': tier_90,
        '85-90%': tier_85,
        '80-85%': tier_80,
        '<80%': tier_below
    }
}

output_path = Path('05_Model/Training_Logs/test_results_v3.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n💾 Results saved to: {output_path}")

# ============================================================
# 10. Visualizations
# ============================================================
print("\nGenerating visualizations...")

# Per-species accuracy bar chart
fig, ax = plt.subplots(figsize=(14, 20))
sorted_acc = sorted(per_class_accuracy.items(), key=lambda x: x[1])
species_names = [s for s, _ in sorted_acc]
acc_values = [a for _, a in sorted_acc]

colors = ['#d62728' if a < 80 else '#ff7f0e' if a < 85 else '#ffbb78' if a < 90 
          else '#98df8a' if a < 95 else '#2ca02c' for a in acc_values]

y_pos = np.arange(len(species_names))
ax.barh(y_pos, acc_values, color=colors, edgecolor='none')
ax.set_yticks(y_pos)
ax.set_yticklabels(species_names, fontsize=7)
ax.set_xlabel('Accuracy (%)', fontsize=12)
ax.set_title(f'Per-Species Test Accuracy\n(Overall: {test_accuracy:.1f}%, Top-5: {top5_accuracy:.1f}%)', fontsize=14)
ax.axvline(x=85, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Target (85%)')
ax.axvline(x=95, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Excellent (95%)')
ax.legend(loc='lower right')
ax.set_xlim(0, 105)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('05_Model/Training_Logs/per_species_accuracy_v3.png', dpi=150, bbox_inches='tight')
print("  Saved: per_species_accuracy_v3.png")

# Confusion matrix for top 15 species
fig, ax = plt.subplots(figsize=(14, 12))
species_counts = pd.Series(per_class_total).sort_values(ascending=False)
top15_species = species_counts.head(15).index.tolist()
top15_indices = [label_mapping[s]['index'] for s in top15_species]

# Filter for top 15
mask = np.isin(all_labels, top15_indices)
filtered_preds = all_preds[mask]
filtered_labels = all_labels[mask]

# Re-index for confusion matrix
idx_map = {idx: i for i, idx in enumerate(top15_indices)}
mapped_preds = np.array([idx_map.get(p, -1) for p in filtered_preds])
mapped_labels = np.array([idx_map.get(l, -1) for l in filtered_labels])

# Keep only valid predictions (within top 15)
valid_mask = (mapped_preds >= 0) & (mapped_labels >= 0)
if valid_mask.sum() > 0:
    cm = confusion_matrix(mapped_labels[valid_mask], mapped_preds[valid_mask], 
                          labels=list(range(len(top15_species))))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    
    short_names = [s.split()[-1][:12] for s in top15_species]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix (Top 15 Species by Sample Count)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('05_Model/Training_Logs/confusion_matrix_v3.png', dpi=150, bbox_inches='tight')
    print("  Saved: confusion_matrix_v3.png")

# Accuracy distribution histogram
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(accuracies, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(x=np.mean(accuracies), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(accuracies):.1f}%')
ax.axvline(x=np.median(accuracies), color='orange', linestyle='--', linewidth=2,
           label=f'Median: {np.median(accuracies):.1f}%')
ax.axvline(x=85, color='green', linestyle='-', linewidth=2, alpha=0.5,
           label='Target: 85%')
ax.set_xlabel('Accuracy (%)', fontsize=12)
ax.set_ylabel('Number of Species', fontsize=12)
ax.set_title('Distribution of Per-Species Accuracy', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('05_Model/Training_Logs/accuracy_distribution_v3.png', dpi=150)
print("  Saved: accuracy_distribution_v3.png")

plt.show()

# ============================================================
# 11. Final Summary
# ============================================================
print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)

target_met = test_accuracy >= 85

print(f"""
┌─────────────────────────────────────────────────────────────┐
│  🐦 MODEL V3 FINAL TEST RESULTS                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 OVERALL METRICS                                         │
│  ├── Test Accuracy:     {test_accuracy:6.2f}%                         │
│  ├── Top-5 Accuracy:    {top5_accuracy:6.2f}%                         │
│  └── Target (85%):      {'✅ ACHIEVED!' if target_met else '❌ Not met'}                         │
│                                                             │
│  📈 PER-SPECIES BREAKDOWN                                   │
│  ├── ≥95% accuracy:     {tier_95:2d} species ({100*tier_95/num_classes:4.1f}%)              │
│  ├── 90-95% accuracy:   {tier_90:2d} species ({100*tier_90/num_classes:4.1f}%)              │
│  ├── 85-90% accuracy:   {tier_85:2d} species ({100*tier_85/num_classes:4.1f}%)              │
│  ├── 80-85% accuracy:   {tier_80:2d} species ({100*tier_80/num_classes:4.1f}%)              │
│  └── <80% accuracy:     {tier_below:2d} species ({100*tier_below/num_classes:4.1f}%)              │
│                                                             │
│  📁 FILES SAVED                                             │
│  ├── test_results_v3.json                                  │
│  ├── per_species_accuracy_v3.png                           │
│  ├── confusion_matrix_v3.png                               │
│  └── accuracy_distribution_v3.png                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")

if target_met:
    print("""
🎉 CONGRATULATIONS! Your model exceeded the 85% target!

NEXT STEPS:
1. ✅ Proceed to web application development
2. Consider improving species with <85% accuracy in future versions
3. Start building the Streamlit multi-page app

Ready to build the web app? Let me know!
""")
else:
    print(f"""
📋 Model achieved {test_accuracy:.1f}% (target was 85%)

Consider:
- The validation accuracy was 95.9%, so there may be some train/test distribution shift
- Check if the test set spectrograms were generated with the same parameters
- Review the lowest performing species for data quality issues
""")