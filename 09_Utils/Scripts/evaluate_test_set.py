"""
Script: evaluate_test_set.py
Purpose: Evaluate best model on test set
Location: 09_Utils/Scripts/evaluate_test_set.py
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

import timm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "05_Model", "Saved_Models", "best_model_fast_quality.pth")
SPEC_DIR = os.path.join(BASE_DIR, "03_Features", "Spectrograms_Precomputed")
TEST_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "test.csv")
LABEL_MAPPING = os.path.join(BASE_DIR, "04_Labels", "Processed_Labels", "label_mapping_fixed.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "07_Evaluation", "Test_Results")

BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET (Same as training)
# ============================================================
class TestDataset(Dataset):
    def __init__(self, csv_path, spec_dir, label_mapping_path):
        self.df = pd.read_csv(csv_path)
        self.spec_dir = os.path.join(spec_dir, "test")
        
        with open(label_mapping_path, 'r') as f:
            label_data = json.load(f)
        
        if isinstance(list(label_data.values())[0], dict):
            self.label_map = label_data.get('species_to_id', {})
        else:
            self.label_map = label_data
        
        # Reverse mapping for species names
        self.id_to_species = {v: k for k, v in self.label_map.items()}
        
        print(f"   Test samples: {len(self.df)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['chunk_file']
        species = row['species']
        
        spec_filename = filename.replace('.wav', '.npy').replace('.mp3', '.npy')
        spec_path = os.path.join(self.spec_dir, spec_filename)
        
        try:
            spec = np.load(spec_path)
        except:
            spec = np.zeros((128, 216), dtype=np.float32)
        
        spec_3ch = np.stack([spec, spec, spec], axis=0).astype(np.float32)
        spec_tensor = torch.from_numpy(spec_3ch)
        
        label = self.label_map.get(species, 0)
        
        return spec_tensor, label, species, filename


# ============================================================
# MODEL (Same architecture)
# ============================================================
class BirdClassifier(nn.Module):
    def __init__(self, num_classes=54):
        super().__init__()
        
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0,
            drop_rate=0.3,
            drop_path_rate=0.2,
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 216)
            feat_dim = self.backbone(dummy).shape[1]
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.backbone(x))


# ============================================================
# EVALUATION
# ============================================================
def evaluate():
    print("=" * 70)
    print("📊 EVALUATING MODEL ON TEST SET")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print(f"\n🏗️ Loading model from {MODEL_PATH}")
    
    model = BirdClassifier(num_classes=54).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   Model trained to epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    # Load test data
    print(f"\n📂 Loading test dataset...")
    test_ds = TestDataset(TEST_CSV, SPEC_DIR, LABEL_MAPPING)
    
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Evaluate
    print(f"\n🔍 Running evaluation...")
    
    all_preds = []
    all_targets = []
    all_probs = []
    all_species = []
    all_filenames = []
    
    correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target, species, filenames in tqdm(test_loader, desc="Testing"):
            data = data.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            
            with autocast():
                output = model(data)
                probs = torch.softmax(output, dim=1)
            
            # Top-1
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            
            # Top-3 and Top-5
            _, top3 = output.topk(3, dim=1)
            _, top5 = output.topk(5, dim=1)
            
            for i in range(target.size(0)):
                if target[i] in top3[i]:
                    top3_correct += 1
                if target[i] in top5[i]:
                    top5_correct += 1
            
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_species.extend(species)
            all_filenames.extend(filenames)
    
    # Calculate metrics
    test_acc = 100 * correct / total
    top3_acc = 100 * top3_correct / total
    top5_acc = 100 * top5_correct / total
    
    print(f"\n{'='*70}")
    print(f"📊 TEST SET RESULTS")
    print(f"{'='*70}")
    print(f"\n   Test Samples:      {total}")
    print(f"   Top-1 Accuracy:    {test_acc:.2f}%")
    print(f"   Top-3 Accuracy:    {top3_acc:.2f}%")
    print(f"   Top-5 Accuracy:    {top5_acc:.2f}%")
    
    # Per-class metrics
    print(f"\n📋 Per-Species Performance:")
    
    # Load species names
    with open(LABEL_MAPPING, 'r') as f:
        label_data = json.load(f)
    if isinstance(list(label_data.values())[0], dict):
        species_to_id = label_data.get('species_to_id', {})
    else:
        species_to_id = label_data
    id_to_species = {v: k for k, v in species_to_id.items()}
    
    # Classification report
    target_names = [id_to_species.get(i, f"Class_{i}") for i in range(54)]
    
    report = classification_report(
        all_targets, all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    # Save detailed report
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(OUTPUT_DIR, "classification_report.csv")
    report_df.to_csv(report_path)
    print(f"\n   Classification report saved to: {report_path}")
    
    # Find best and worst performing species
    species_f1 = {name: report[name]['f1-score'] for name in target_names if name in report}
    sorted_species = sorted(species_f1.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n   🏆 Top 5 Best Performing Species:")
    for species, f1 in sorted_species[:5]:
        print(f"      {species}: F1={f1:.3f}")
    
    print(f"\n   ⚠️ Top 5 Worst Performing Species:")
    for species, f1 in sorted_species[-5:]:
        print(f"      {species}: F1={f1:.3f}")
    
    # Confusion matrix
    print(f"\n📊 Generating confusion matrix...")
    
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm, annot=False, fmt='d', cmap='Blues',
        xticklabels=target_names, yticklabels=target_names
    )
    plt.title(f'Confusion Matrix - Test Accuracy: {test_acc:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"   Confusion matrix saved to: {cm_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'filename': all_filenames,
        'true_species': all_species,
        'true_label': all_targets,
        'predicted_label': all_preds,
        'correct': [t == p for t, p in zip(all_targets, all_preds)]
    })
    
    # Add predicted species names
    predictions_df['predicted_species'] = predictions_df['predicted_label'].map(
        lambda x: id_to_species.get(x, f"Unknown_{x}")
    )
    
    # Add confidence
    predictions_df['confidence'] = [probs[pred] for probs, pred in zip(all_probs, all_preds)]
    
    pred_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    predictions_df.to_csv(pred_path, index=False)
    print(f"   Predictions saved to: {pred_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ EVALUATION COMPLETE")
    print(f"{'='*70}")
    
    summary = {
        'test_samples': total,
        'top1_accuracy': test_acc,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
    }
    
    summary_path = os.path.join(OUTPUT_DIR, "test_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Final Summary:")
    print(f"   Test Accuracy:  {test_acc:.2f}%")
    print(f"   Top-3 Accuracy: {top3_acc:.2f}%")
    print(f"   Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"   Macro F1:       {report['macro avg']['f1-score']*100:.2f}%")
    print(f"   Weighted F1:    {report['weighted avg']['f1-score']*100:.2f}%")
    
    # Compare with validation
    val_acc = checkpoint.get('val_acc', 0)
    diff = test_acc - val_acc
    print(f"\n   Val Accuracy:   {val_acc:.2f}%")
    print(f"   Test Accuracy:  {test_acc:.2f}%")
    print(f"   Difference:     {diff:+.2f}%")
    
    if abs(diff) < 3:
        print(f"\n   ✅ Test and Val are similar - model generalizes well!")
    elif diff < -3:
        print(f"\n   ⚠️ Test lower than Val - might need more regularization")
    else:
        print(f"\n   ℹ️ Test higher than Val - test set might be easier")


if __name__ == "__main__":
    evaluate()