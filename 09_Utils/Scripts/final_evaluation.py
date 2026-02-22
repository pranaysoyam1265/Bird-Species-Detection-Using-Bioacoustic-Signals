# Save as: final_evaluation_CORRECT.py

import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")

print("="*60)
print("🎯 FINAL EVALUATION (CORRECT PREPROCESSING)")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# 1. Load label mapping
# ============================================================
print("\n📋 Loading label mapping...")

with open('04_Labels/Processed_Labels/label_mapping_v3.json', 'r') as f:
    label_mapping_raw = json.load(f)

idx_to_species = {v['index']: k for k, v in label_mapping_raw.items()}
idx_to_english = {v['index']: v['english_name'] for k, v in label_mapping_raw.items()}
num_classes = len(idx_to_species)

print(f"✅ Loaded {num_classes} species")

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
        return self.classifier(self.backbone(x))

model = BirdClassifier(num_classes=num_classes)
checkpoint = torch.load('05_Model/Saved_Models/best_model_v3.pth', 
                        map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

val_acc = checkpoint.get('val_acc', 0)
print(f"✅ Model loaded (epoch {checkpoint.get('epoch', 'N/A')}, val_acc: {val_acc:.2f}%)")

# ============================================================
# 3. Correct preprocessing (matches training exactly)
# ============================================================
def preprocess_spectrogram(spec):
    """
    Exact preprocessing from training script:
    1. Normalize to [0, 1]
    2. Clip to [0, 1]
    3. Convert to 3-channel tensor
    """
    # Normalize to [0, 1]
    spec_min = spec.min()
    spec_max = spec.max()
    if spec_max - spec_min > 0:
        spec = (spec - spec_min) / (spec_max - spec_min)
    else:
        spec = np.zeros_like(spec)
    
    # Clip values
    spec = np.clip(spec, 0, 1)
    
    # Convert to 3-channel tensor
    spec_tensor = torch.FloatTensor(spec).unsqueeze(0)  # (1, 128, 216)
    spec_tensor = spec_tensor.repeat(3, 1, 1)  # (3, 128, 216)
    
    return spec_tensor

# ============================================================
# 4. Load test data
# ============================================================
print("\n📂 Loading test data...")

test_df = pd.read_csv('04_Labels/Train_Val_Test_Split/test_v3.csv')
print(f"Test samples: {len(test_df)}")

# ============================================================
# 5. Full evaluation
# ============================================================
print("\n🧪 Evaluating on test set...")

all_preds = []
all_labels = []
all_probs = []
errors = 0

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
    spec_path = Path(row['spectrogram_path'])
    true_label = int(row['label'])
    
    if not spec_path.exists():
        errors += 1
        continue
    
    try:
        # Load and preprocess
        spec = np.load(str(spec_path))
        input_tensor = preprocess_spectrogram(spec)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        # Predict
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

print(f"\n✅ Processed: {len(all_preds)} samples")
if errors > 0:
    print(f"⚠️ Errors: {errors}")

# ============================================================
# 6. Calculate metrics
# ============================================================
print("\n" + "="*60)
print("📊 TEST SET RESULTS")
print("="*60)

# Overall accuracy
test_accuracy = 100.0 * np.mean(all_preds == all_labels)
print(f"\n🎯 Test Accuracy: {test_accuracy:.2f}%")

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
print(f"  Std:    {np.std(accuracies):.2f}%")
print(f"  Min:    {np.min(accuracies):.2f}%")
print(f"  Max:    {np.max(accuracies):.2f}%")

# Tier breakdown
tier_95 = sum(1 for a in accuracies if a >= 95)
tier_90 = sum(1 for a in accuracies if 90 <= a < 95)
tier_85 = sum(1 for a in accuracies if 85 <= a < 90)
tier_80 = sum(1 for a in accuracies if 80 <= a < 85)
tier_below = sum(1 for a in accuracies if a < 80)

print(f"\n🎯 Accuracy Distribution:")
print(f"  ≥95%:    {tier_95:2d} species ({100*tier_95/num_classes:.1f}%)")
print(f"  90-95%:  {tier_90:2d} species ({100*tier_90/num_classes:.1f}%)")
print(f"  85-90%:  {tier_85:2d} species ({100*tier_85/num_classes:.1f}%)")
print(f"  80-85%:  {tier_80:2d} species ({100*tier_80/num_classes:.1f}%)")
print(f"  <80%:    {tier_below:2d} species ({100*tier_below/num_classes:.1f}%)")

# Best and worst performers
sorted_classes = sorted(per_class_accuracy.items(), key=lambda x: x[1])

print(f"\n⚠️ Lowest Performing Species (Bottom 10):")
for label, acc in sorted_classes[:10]:
    species = idx_to_species.get(label, f"Label_{label}")
    english = idx_to_english.get(label, "")
    count = per_class_total[label]
    print(f"  {acc:5.1f}% | {species:<32} ({english[:18]}) n={count}")

print(f"\n✅ Highest Performing Species (Top 10):")
for label, acc in sorted_classes[-10:][::-1]:
    species = idx_to_species.get(label, f"Label_{label}")
    english = idx_to_english.get(label, "")
    count = per_class_total[label]
    print(f"  {acc:5.1f}% | {species:<32} ({english[:18]}) n={count}")

# ============================================================
# 7. Save results
# ============================================================
print("\n💾 Saving results...")

results = {
    'test_accuracy': float(test_accuracy),
    'test_top5': float(top5_accuracy),
    'validation_accuracy': float(val_acc),
    'samples_evaluated': len(all_preds),
    'per_class_accuracy': {idx_to_species[k]: float(v) for k, v in per_class_accuracy.items()},
    'per_class_samples': {idx_to_species[k]: int(v) for k, v in per_class_total.items()},
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

with open('05_Model/Training_Logs/test_results_v3_FINAL.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✅ Saved: test_results_v3_FINAL.json")

# ============================================================
# 8. Create visualization
# ============================================================
print("\n📊 Creating visualizations...")

# Per-species accuracy bar chart
fig, ax = plt.subplots(figsize=(14, 20))
sorted_acc = sorted(per_class_accuracy.items(), key=lambda x: x[1])
species_names = [idx_to_species[label] for label, _ in sorted_acc]
acc_values = [acc for _, acc in sorted_acc]

colors = ['#d62728' if a < 80 else '#ff7f0e' if a < 85 else '#ffbb78' if a < 90 
          else '#98df8a' if a < 95 else '#2ca02c' for a in acc_values]

y_pos = np.arange(len(species_names))
ax.barh(y_pos, acc_values, color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels(species_names, fontsize=7)
ax.set_xlabel('Accuracy (%)')
ax.set_title(f'Per-Species Test Accuracy\n(Overall: {test_accuracy:.1f}%, Top-5: {top5_accuracy:.1f}%)')
ax.axvline(x=85, color='red', linestyle='--', alpha=0.7, label='Target (85%)')
ax.axvline(x=test_accuracy, color='blue', linestyle='-', alpha=0.7, label=f'Mean ({test_accuracy:.1f}%)')
ax.legend(loc='lower right')
ax.set_xlim(0, 105)
plt.tight_layout()
plt.savefig('05_Model/Training_Logs/per_species_accuracy_v3_FINAL.png', dpi=150, bbox_inches='tight')
print("✅ Saved: per_species_accuracy_v3_FINAL.png")

# ============================================================
# 9. Final Summary
# ============================================================
print("\n" + "="*60)
print("🏆 FINAL SUMMARY")
print("="*60)

target_met = test_accuracy >= 85
gap = val_acc - test_accuracy

print(f"""
┌─────────────────────────────────────────────────────────────┐
│  🐦 MODEL V3 - FINAL TEST RESULTS                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 OVERALL METRICS                                         │
│  ├── Test Accuracy:     {test_accuracy:6.2f}%                         │
│  ├── Top-5 Accuracy:    {top5_accuracy:6.2f}%                         │
│  ├── Val Accuracy:      {val_acc:6.2f}% (from training)              │
│  ├── Gap (Val - Test):  {gap:+6.2f}%                         │
│  └── Target (85%):      {'✅ ACHIEVED!' if target_met else '❌ Not met'}                         │
│                                                             │
│  📈 PER-SPECIES BREAKDOWN ({num_classes} species)                     │
│  ├── ≥95% accuracy:     {tier_95:2d} species ({100*tier_95/num_classes:4.1f}%)              │
│  ├── 90-95% accuracy:   {tier_90:2d} species ({100*tier_90/num_classes:4.1f}%)              │
│  ├── 85-90% accuracy:   {tier_85:2d} species ({100*tier_85/num_classes:4.1f}%)              │
│  ├── 80-85% accuracy:   {tier_80:2d} species ({100*tier_80/num_classes:4.1f}%)              │
│  └── <80% accuracy:     {tier_below:2d} species ({100*tier_below/num_classes:4.1f}%)              │
│                                                             │
│  📁 FILES SAVED                                             │
│  ├── test_results_v3_FINAL.json                            │
│  └── per_species_accuracy_v3_FINAL.png                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")

if target_met:
    print("""
🎉 CONGRATULATIONS! Your model EXCEEDED the 85% target!

The issue was: ImageNet normalization was applied during evaluation
but NOT during training. The training used simple [0,1] normalization.

✅ Model is validated and ready for deployment!
✅ Ready to build the web application!

Next steps:
1. Build Streamlit multi-page app
2. Implement the 50+ planned features
3. Deploy to production
""")
else:
    print(f"""
Model achieved {test_accuracy:.1f}% (target was 85%).
Gap from validation: {gap:.1f}%

This is still a very good result! Consider:
- Collecting more data for low-performing species
- Fine-tuning with harder examples
""")