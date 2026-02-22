# Save this as: evaluate_v3_results.py

import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

print("="*60)
print("🐦 MODEL V3 EVALUATION REPORT")
print("="*60)

# ============================================================
# 1. Load Model Checkpoint
# ============================================================
model_path = Path("05_Model/Saved_Models/best_model_v3.pth")

if not model_path.exists():
    print("❌ Model not found! Training may still be in progress.")
    print("   Check: Is the training script still running?")
    exit()

checkpoint = torch.load(model_path, map_location='cpu')

print("\n📊 TRAINING SUMMARY")
print("-" * 40)
print(f"Best Epoch:        {checkpoint.get('epoch', 'N/A')}")
print(f"Validation Acc:    {checkpoint.get('val_acc', 0):.2f}%")
print(f"Validation Top-5:  {checkpoint.get('val_top5', 0):.2f}%")
print(f"Training Acc:      {checkpoint.get('train_acc', 0):.2f}%")
print(f"Validation Loss:   {checkpoint.get('val_loss', 0):.4f}")

# Calculate overfitting gap
train_acc = checkpoint.get('train_acc', 0)
val_acc = checkpoint.get('val_acc', 0)
overfit_gap = train_acc - val_acc
print(f"\n⚠️ Overfitting Gap: {overfit_gap:.2f}%")

# ============================================================
# 2. Load Test Results (if available)
# ============================================================
test_results_path = Path("05_Model/Training_Logs/test_results_v3.json")

if test_results_path.exists():
    with open(test_results_path) as f:
        test_results = json.load(f)
    
    print("\n📊 TEST SET RESULTS")
    print("-" * 40)
    print(f"Test Accuracy:     {test_results.get('test_accuracy', 0):.2f}%")
    print(f"Test Top-5:        {test_results.get('test_top5', 0):.2f}%")
    
    # Per-class analysis
    if 'per_class_accuracy' in test_results:
        per_class = test_results['per_class_accuracy']
        accuracies = list(per_class.values())
        
        print(f"\n📈 PER-SPECIES PERFORMANCE")
        print("-" * 40)
        print(f"Mean accuracy:     {np.mean(accuracies):.2f}%")
        print(f"Median accuracy:   {np.median(accuracies):.2f}%")
        print(f"Std deviation:     {np.std(accuracies):.2f}%")
        print(f"Min accuracy:      {np.min(accuracies):.2f}%")
        print(f"Max accuracy:      {np.max(accuracies):.2f}%")
        
        # Count species by accuracy tier
        tier_90 = sum(1 for acc in accuracies if acc >= 90)
        tier_85 = sum(1 for acc in accuracies if 85 <= acc < 90)
        tier_80 = sum(1 for acc in accuracies if 80 <= acc < 85)
        tier_70 = sum(1 for acc in accuracies if 70 <= acc < 80)
        tier_below = sum(1 for acc in accuracies if acc < 70)
        
        print(f"\n🎯 ACCURACY DISTRIBUTION")
        print("-" * 40)
        print(f"≥90% accuracy:     {tier_90:3d} species ({100*tier_90/87:.1f}%)")
        print(f"85-90% accuracy:   {tier_85:3d} species ({100*tier_85/87:.1f}%)")
        print(f"80-85% accuracy:   {tier_80:3d} species ({100*tier_80/87:.1f}%)")
        print(f"70-80% accuracy:   {tier_70:3d} species ({100*tier_70/87:.1f}%)")
        print(f"<70% accuracy:     {tier_below:3d} species ({100*tier_below/87:.1f}%) ⚠️")
        
        # Worst performing species
        sorted_species = sorted(per_class.items(), key=lambda x: x[1])
        print(f"\n⚠️ LOWEST PERFORMING SPECIES (Bottom 10)")
        print("-" * 40)
        for species, acc in sorted_species[:10]:
            print(f"  {species:30s} {acc:5.1f}%")
        
        # Best performing species
        print(f"\n✅ HIGHEST PERFORMING SPECIES (Top 10)")
        print("-" * 40)
        for species, acc in sorted_species[-10:][::-1]:
            print(f"  {species:30s} {acc:5.1f}%")

else:
    print("\n⚠️ Test results file not found.")
    print("   The training script may not have completed evaluation.")

# ============================================================
# 3. Load Training Log
# ============================================================
import glob
log_files = glob.glob("05_Model/Training_Logs/training_log_v3_*.csv")

if log_files:
    latest_log = max(log_files, key=lambda x: Path(x).stat().st_mtime)
    df_log = pd.read_csv(latest_log)
    
    print(f"\n📋 TRAINING HISTORY ({len(df_log)} epochs logged)")
    print("-" * 40)
    print(df_log.tail(5).to_string(index=False))
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curve
    axes[0].plot(df_log['epoch'], df_log['train_loss'], label='Train')
    axes[0].plot(df_log['epoch'], df_log['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(df_log['epoch'], df_log['train_acc'], label='Train')
    axes[1].plot(df_log['epoch'], df_log['val_acc'], label='Validation')
    axes[1].axhline(y=85, color='r', linestyle='--', alpha=0.5, label='Target (85%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in df_log.columns:
        axes[2].plot(df_log['epoch'], df_log['lr'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('05_Model/Training_Logs/training_curves_v3.png', dpi=150)
    print(f"\n📊 Training curves saved to: training_curves_v3.png")
    plt.show()

# ============================================================
# 4. Decision Recommendation
# ============================================================
print("\n" + "="*60)
print("🎯 RECOMMENDATION")
print("="*60)

test_acc = test_results.get('test_accuracy', val_acc) if test_results_path.exists() else val_acc

if test_acc >= 85:
    print(f"""
✅ EXCELLENT! Test accuracy ({test_acc:.1f}%) meets the 85% target!

RECOMMENDED NEXT STEPS:
1. ✅ Proceed to web app development (Phase 1)
2. Focus on species with <80% accuracy for future improvement
3. Consider deploying with confidence thresholds

Ready to build the Streamlit multi-page app?
    """)
elif test_acc >= 80:
    print(f"""
⚠️ GOOD! Test accuracy ({test_acc:.1f}%) is acceptable but below 85% target.

OPTIONS:
A) Proceed anyway - acceptable for demo/prototype
B) Quick improvements:
   - Add more data for weak species
   - Try stronger augmentation
   - Ensemble with v1/v2 models

C) Larger model (EfficientNet-B3) - requires retraining

Recommendation: Proceed to web app, plan improvements for v4.
    """)
else:
    print(f"""
❌ Below target. Test accuracy ({test_acc:.1f}%) needs improvement.

RECOMMENDED ACTIONS (in priority order):
1. Identify and fix weak species (collect more data)
2. Check for data quality issues
3. Try stronger model (B3) or ensemble
4. Review augmentation settings

Do you want help diagnosing the issue?
    """)

print("\n" + "="*60)