"""
evaluate_model.py
Evaluate trained model on test set

Generates:
- Accuracy, Precision, Recall, F1
- Confusion Matrix
- Per-class performance
- Predictions CSV
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

import timm
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    top_k_accuracy_score,
    f1_score
)

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")

# Paths
TEST_CSV = BASE_DIR / "04_Labels" / "Train_Val_Test_Split" / "test.csv"
TEST_SPEC_DIR = BASE_DIR / "03_Features" / "Spectrograms" / "test"
MODEL_PATH = BASE_DIR / "05_Model" / "Saved_Models" / "best_model.pth"
LABEL_MAPPING = BASE_DIR / "04_Labels" / "Processed_Labels" / "label_mapping.json"

# Output
EVAL_DIR = BASE_DIR / "07_Evaluation"
METRICS_DIR = EVAL_DIR / "Metrics"
CONFUSION_DIR = EVAL_DIR / "Confusion_Matrices"
PREDICTIONS_DIR = EVAL_DIR / "Predictions"

# Settings
BATCH_SIZE = 32
NUM_WORKERS = 4

# ============================================================
# DATASET (Same as training)
# ============================================================

class BirdSpectrogramDataset(Dataset):
    def __init__(self, csv_path, spec_dir):
        self.df = pd.read_csv(csv_path)
        self.spec_dir = Path(spec_dir)
        
        valid_rows = []
        for idx, row in self.df.iterrows():
            spec_file = row['chunk_file'].replace('.wav', '.npy')
            if (self.spec_dir / spec_file).exists():
                valid_rows.append(idx)
        
        self.df = self.df.loc[valid_rows].reset_index(drop=True)
        print(f"Dataset: {len(self.df)} valid samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        spec_file = row['chunk_file'].replace('.wav', '.npy')
        spec_path = self.spec_dir / spec_file
        
        spec = np.load(spec_path)
        spec = self._pad_or_crop(spec, target_time=216)
        
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0)
        spec_tensor = spec_tensor.repeat(3, 1, 1)
        
        label = row['label_id']
        chunk_file = row['chunk_file']
        
        return spec_tensor, label, chunk_file
    
    def _pad_or_crop(self, spec, target_time):
        current_time = spec.shape[1]
        if current_time < target_time:
            padding = target_time - current_time
            spec = np.pad(spec, ((0, 0), (0, padding)), mode='constant')
        elif current_time > target_time:
            spec = spec[:, :target_time]
        return spec

# ============================================================
# MODEL (Same as training)
# ============================================================

class BirdClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=54, pretrained=False):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, 128, 216)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        pooled = self.global_pool(features)
        pooled = pooled.flatten(1)
        logits = self.classifier(pooled)
        return logits

# ============================================================
# EVALUATION
# ============================================================

def main():
    print("=" * 70)
    print("🧪 MODEL EVALUATION ON TEST SET")
    print("=" * 70)
    
    # Create directories
    for dir_path in [METRICS_DIR, CONFUSION_DIR, PREDICTIONS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Load label mapping
    with open(LABEL_MAPPING, 'r') as f:
        label_map = json.load(f)
    
    num_classes = label_map['num_classes']
    id_to_species = {int(k): v for k, v in label_map['id_to_species'].items()}
    species_list = label_map['species_list']
    
    print(f"📊 Classes: {num_classes}")
    
    # Load model
    print(f"\n🧠 Loading model from: {MODEL_PATH}")
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    model = BirdClassifier(
        model_name='efficientnet_b0',
        num_classes=num_classes,
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   Loaded from epoch {checkpoint['epoch']} (Val Acc: {checkpoint['val_acc']:.2f}%)")
    
    # Load test dataset
    print(f"\n📂 Loading test dataset...")
    test_dataset = BirdSpectrogramDataset(TEST_CSV, TEST_SPEC_DIR)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Run evaluation
    print(f"\n🔍 Running evaluation on {len(test_dataset)} samples...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_files = []
    
    with torch.no_grad():
        for specs, labels, files in tqdm(test_loader, desc="Evaluating"):
            specs = specs.to(device)
            
            with autocast():
                outputs = model(specs)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_files.extend(files)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    
    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds) * 100
    top3_accuracy = top_k_accuracy_score(all_labels, all_probs, k=3) * 100
    top5_accuracy = top_k_accuracy_score(all_labels, all_probs, k=5) * 100
    
    # F1 scores
    f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
    f1_weighted = f1_score(all_labels, all_preds, average='weighted') * 100
    
    print(f"""
    📈 Overall Metrics:
    ├── Top-1 Accuracy: {accuracy:.2f}%
    ├── Top-3 Accuracy: {top3_accuracy:.2f}%
    ├── Top-5 Accuracy: {top5_accuracy:.2f}%
    ├── F1 Score (Macro): {f1_macro:.2f}%
    └── F1 Score (Weighted): {f1_weighted:.2f}%
    """)
    
    # Classification report
    print("\n📋 Per-Class Performance (Top 15):")
    print("-" * 70)
    
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=species_list,
        output_dict=True,
        zero_division=0
    )
    
    # Convert to DataFrame and sort by F1
    report_df = pd.DataFrame(report).T
    report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    report_df = report_df.sort_values('f1-score', ascending=False)
    
    # Print top 15
    print(f"{'Species':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 75)
    for species in report_df.head(15).index:
        row = report_df.loc[species]
        print(f"{species:<35} {row['precision']:>10.2f} {row['recall']:>10.2f} {row['f1-score']:>10.2f} {int(row['support']):>10}")
    
    # Save full report
    report_path = METRICS_DIR / "classification_report.csv"
    report_df.to_csv(report_path)
    print(f"\n✅ Full report saved: {report_path}")
    
    # Confusion Matrix
    print("\n📊 Generating confusion matrix...")
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix (smaller version for visibility)
    plt.figure(figsize=(20, 16))
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(
        cm_normalized,
        annot=False,
        cmap='Blues',
        xticklabels=species_list,
        yticklabels=species_list
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Test Accuracy: {accuracy:.2f}%)')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    
    cm_path = CONFUSION_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'chunk_file': all_files,
        'true_label': all_labels,
        'true_species': [id_to_species[l] for l in all_labels],
        'pred_label': all_preds,
        'pred_species': [id_to_species[p] for p in all_preds],
        'correct': all_preds == all_labels,
        'confidence': [all_probs[i, p] for i, p in enumerate(all_preds)]
    })
    
    pred_path = PREDICTIONS_DIR / "test_predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    print(f"✅ Predictions saved: {pred_path}")
    
    # Save metrics summary
    metrics_summary = {
        'test_samples': len(test_dataset),
        'num_classes': num_classes,
        'top1_accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'top5_accuracy': top5_accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
    
    metrics_path = METRICS_DIR / "metrics_summary.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"✅ Metrics summary saved: {metrics_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"""
    📊 Test Set Results:
    ├── Samples: {len(test_dataset)}
    ├── Top-1 Accuracy: {accuracy:.2f}%
    ├── Top-3 Accuracy: {top3_accuracy:.2f}%
    └── Top-5 Accuracy: {top5_accuracy:.2f}%
    
    📁 Files Saved:
    ├── {report_path}
    ├── {cm_path}
    ├── {pred_path}
    └── {metrics_path}
    
    🎯 Next: Generate Grad-CAM visualizations!
    """)

if __name__ == "__main__":
    main()